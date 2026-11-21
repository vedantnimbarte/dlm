//! Batched, streaming inference service behind the HTTP API.
//!
//! An [`EngineService`] runs a background thread that owns the generator and a
//! [`BatchScheduler`](crate::batching::BatchScheduler): connection threads
//! `submit` requests and receive a channel of tokens as they are produced.
//! Concurrent requests are therefore **continuously batched** (interleaved a
//! token at a time), and a request can be **streamed** to the client as SSE.
//!
//! The [`router`] wires this into the OpenAI and Anthropic endpoints, supporting
//! `stream: true` for both `/v1/chat/completions` and `/v1/messages`.

use crate::batching::{AcceptanceStats, BatchScheduler};
use crate::forward::ComputeKernel;
use crate::generate::{Decoding, Dry, Generator, Penalties, Sampler, TokenLogprobs};
use crate::grammar::{Constraint, Grammar, TokenBytes};
use crate::server::http::{Handler, Request, Response};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};

/// A token event streamed back for one request. `Done` carries the request's
/// speculative acceptance stats when it was decoded speculatively (`None` on the
/// plain path).
pub enum TokenEvent {
    /// A token, with its log-probabilities when the request asked for them.
    Next(u32, Option<TokenLogprobs>),
    Done(Option<AcceptanceStats>),
    /// Turned away before it started: it waited in the queue past
    /// `--queue-timeout`.
    Rejected(String),
}

/// What the engine thread is asked to do: generate, or embed a text (run
/// between decode steps, on its own KV).
enum Work {
    Generate(Box<Job>),
    Embed {
        tokens: Vec<u32>,
        pooling: crate::generate::Pooling,
        reply: Sender<Result<Vec<f32>, String>>,
    },
    /// Keep the prefix cache in this directory from now on.
    PersistPrefixCache(std::path::PathBuf),
}

struct Job {
    id: u64,
    /// Lower is admitted first.
    priority: i32,
    prompt: Vec<u32>,
    /// Images in the prompt as `pixel_values`, with where each one's tokens
    /// start; encoded on the engine thread, which owns the GPU.
    images: Vec<(usize, Vec<f32>)>,
    vision: Option<Arc<crate::vision::Vision>>,
    max_new: usize,
    eos: Vec<u32>,
    decoding: Decoding,
    sink: Sender<TokenEvent>,
}

/// Cumulative server counters, exposed as Prometheus text at `GET /metrics`.
#[derive(Default)]
struct Metrics {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}

/// Latest layer-streaming cache stats, published by the engine loop from the
/// kernel and read by `/metrics`. `active` is set only when the served kernel
/// streams weights, so resident-model runs omit the streaming gauges.
#[derive(Default)]
struct StreamMetrics {
    active: AtomicBool,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    prefetched: AtomicU64,
    depth: AtomicU64,
    /// Routed-expert VRAM cache (GPU MoE streaming only; stays zero otherwise).
    expert_hits: AtomicU64,
    expert_misses: AtomicU64,
    expert_evictions: AtomicU64,
    /// The scheduler after its latest step, whatever the kernel: requests
    /// decoding, requests queued, and (where the kernel pages KV) free KV
    /// tokens, `u64::MAX` when there is no pool to count.
    running: AtomicU64,
    waiting: AtomicU64,
    kv_free: AtomicU64,
}

impl StreamMetrics {
    fn publish(&self, s: crate::forward::StreamStats) {
        self.active.store(true, Ordering::Relaxed);
        self.hits.store(s.hits, Ordering::Relaxed);
        self.misses.store(s.misses, Ordering::Relaxed);
        self.evictions.store(s.evictions, Ordering::Relaxed);
        self.prefetched.store(s.prefetched, Ordering::Relaxed);
        self.depth.store(s.depth as u64, Ordering::Relaxed);
        self.expert_hits.store(s.expert_hits, Ordering::Relaxed);
        self.expert_misses.store(s.expert_misses, Ordering::Relaxed);
        self.expert_evictions
            .store(s.expert_evictions, Ordering::Relaxed);
    }
}

/// A handle to the background batching engine.
pub struct EngineService {
    job_tx: Mutex<Sender<Work>>,
    next_id: AtomicU64,
    tokenizer: BpeTokenizer,
    vocab_size: usize,
    model_id: String,
    created: u64,
    default_max_tokens: usize,
    chat_template: ChatTemplate,
    embedding_pooling: crate::generate::Pooling,
    /// Stop generation when any of these token ids is produced (the model's
    /// EOS set). Empty means run to `max_tokens`.
    eos_tokens: Vec<u32>,
    /// Maximum context (prompt + generated) the model/KV cache supports.
    /// Requests are clamped to fit; `usize::MAX` disables the guard.
    max_context: usize,
    metrics: Metrics,
    /// Layer-streaming stats, shared with the engine loop that publishes them.
    stream: Arc<StreamMetrics>,
    /// Each token id's bytes, built on first use (see [`token_bytes`](Self::token_bytes)).
    token_bytes: OnceLock<TokenBytes>,
    /// How long a request may wait to be admitted, in milliseconds; 0 for no
    /// limit. Shared with the engine loop, which enforces it.
    queue_timeout: Arc<AtomicU64>,
    /// The vision tower, when the model takes images (Gemma 3 4B+).
    vision: Option<Arc<crate::vision::Vision>>,
}

impl EngineService {
    /// Start the engine: spawns the scheduler thread owning `generator`.
    #[allow(clippy::too_many_arguments)]
    pub fn start<K: ComputeKernel + Send + 'static>(
        generator: Generator<K>,
        tokenizer: BpeTokenizer,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
        max_batch: usize,
        prefix_cache_size: usize,
    ) -> Arc<Self> {
        Self::start_inner(
            generator,
            Drafting::None,
            0,
            tokenizer,
            vocab_size,
            model_id,
            default_max_tokens,
            created,
            max_batch,
            prefix_cache_size,
        )
    }

    /// Start the engine with speculative decoding: `draft` proposes `gamma`
    /// tokens per round for `generator` (the target) to verify. Output is
    /// identical to [`start`](Self::start); `draft` must share the target's
    /// tokenizer/vocabulary.
    #[allow(clippy::too_many_arguments)]
    pub fn start_speculative<K: ComputeKernel + Send + 'static>(
        generator: Generator<K>,
        draft: Generator<K>,
        gamma: usize,
        tokenizer: BpeTokenizer,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
        max_batch: usize,
        prefix_cache_size: usize,
    ) -> Arc<Self> {
        Self::start_inner(
            generator,
            Drafting::Model(Box::new(draft)),
            gamma,
            tokenizer,
            vocab_size,
            model_id,
            default_max_tokens,
            created,
            max_batch,
            prefix_cache_size,
        )
    }

    /// Start the engine speculating with an EAGLE-3 `head`: it proposes `gamma`
    /// tokens per round from `generator`'s own hidden states, and the target
    /// verifies them. Output is identical to [`start`](Self::start).
    #[allow(clippy::too_many_arguments)]
    pub fn start_eagle3<K: ComputeKernel + Send + 'static>(
        generator: Generator<K>,
        head: crate::forward::eagle3::Eagle3Head,
        gamma: usize,
        tokenizer: BpeTokenizer,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
        max_batch: usize,
    ) -> Arc<Self> {
        Self::start_inner(
            generator,
            Drafting::Eagle3(Box::new(head)),
            gamma.max(1),
            tokenizer,
            vocab_size,
            model_id,
            default_max_tokens,
            created,
            max_batch,
            0,
        )
    }

    /// Start the engine speculating with no draft model: each request proposes
    /// up to `gamma` tokens per round by n-gram lookup in its own context.
    /// Output is identical to [`start`](Self::start).
    #[allow(clippy::too_many_arguments)]
    pub fn start_ngram<K: ComputeKernel + Send + 'static>(
        generator: Generator<K>,
        gamma: usize,
        tokenizer: BpeTokenizer,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
        max_batch: usize,
        prefix_cache_size: usize,
    ) -> Arc<Self> {
        Self::start_inner(
            generator,
            Drafting::None,
            gamma.max(1),
            tokenizer,
            vocab_size,
            model_id,
            default_max_tokens,
            created,
            max_batch,
            prefix_cache_size,
        )
    }

    /// `gamma` 0 without a draft is plain decoding; above 0, n-gram lookup.
    #[allow(clippy::too_many_arguments)]
    fn start_inner<K: ComputeKernel + Send + 'static>(
        generator: Generator<K>,
        draft: Drafting<K>,
        gamma: usize,
        tokenizer: BpeTokenizer,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
        max_batch: usize,
        prefix_cache_size: usize,
    ) -> Arc<Self> {
        let (job_tx, job_rx) = channel::<Work>();
        let stream = Arc::new(StreamMetrics::default());
        let stream_loop = Arc::clone(&stream);
        let queue_timeout = Arc::new(AtomicU64::new(0));
        let timeout_loop = Arc::clone(&queue_timeout);
        std::thread::spawn(move || {
            engine_loop(
                generator,
                draft,
                gamma,
                job_rx,
                max_batch,
                prefix_cache_size,
                stream_loop,
                timeout_loop,
            )
        });
        Arc::new(Self {
            job_tx: Mutex::new(job_tx),
            next_id: AtomicU64::new(1),
            tokenizer,
            vocab_size,
            model_id: model_id.into(),
            created,
            default_max_tokens,
            chat_template: ChatTemplate::default(),
            embedding_pooling: crate::generate::Pooling::default(),
            eos_tokens: Vec::new(),
            max_context: usize::MAX,
            metrics: Metrics::default(),
            stream,
            token_bytes: OnceLock::new(),
            queue_timeout,
            vision: None,
        })
    }

    /// Take images in chat requests, encoded by `vision` (called right after
    /// [`start`](Self::start), while the `Arc` is still unique).
    pub fn with_vision(mut self: Arc<Self>, vision: crate::vision::Vision) -> Arc<Self> {
        if let Some(inner) = Arc::get_mut(&mut self) {
            inner.vision = Some(Arc::new(vision));
        }
        self
    }

    /// Each token id's bytes (`None` for special tokens). Requests that look at
    /// token text need them, and decoding the whole vocabulary per request
    /// would cost more than the request, so it happens once.
    fn token_bytes(&self) -> TokenBytes {
        let table = self.token_bytes.get_or_init(|| {
            let ids = 0..self.vocab_size as u32;
            Arc::new(ids.map(|id| self.tokenizer.token_bytes(id)).collect())
        });
        Arc::clone(table)
    }

    /// Record a completed generation into the `/metrics` counters.
    fn record(&self, prompt_tokens: usize, completion_tokens: usize) {
        self.metrics.requests.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.metrics
            .completion_tokens
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
    }

    /// Set the chat template (called right after [`start`](Self::start), while
    /// the `Arc` is still unique). Defaults to [`ChatTemplate::Plain`].
    pub fn with_chat_template(mut self: Arc<Self>, template: ChatTemplate) -> Arc<Self> {
        if let Some(inner) = Arc::get_mut(&mut self) {
            inner.chat_template = template;
        }
        self
    }

    /// Set the EOS token id(s): generation stops (and the token is dropped from
    /// the output) when the model produces any of them. Defaults to empty (run
    /// to length).
    /// Turn away a request that has waited longer than `limit` to be admitted,
    /// with `503` and `Retry-After`, rather than let a backlog grow without
    /// bound. `Duration::ZERO` waits forever, the default.
    pub fn with_queue_timeout(self: Arc<Self>, limit: std::time::Duration) -> Arc<Self> {
        self.queue_timeout.store(
            limit.as_millis().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self
    }

    /// Keep the prefix cache in `dir` too, loading what an earlier run left
    /// there. `dir` must be specific to the model and its settings.
    pub fn with_prefix_cache_dir(self: Arc<Self>, dir: std::path::PathBuf) -> Arc<Self> {
        let _ = self
            .job_tx
            .lock()
            .unwrap()
            .send(Work::PersistPrefixCache(dir));
        self
    }

    /// How `/v1/embeddings` pools the tokens' hidden states.
    pub fn with_embedding_pooling(
        mut self: Arc<Self>,
        pooling: crate::generate::Pooling,
    ) -> Arc<Self> {
        if let Some(inner) = Arc::get_mut(&mut self) {
            inner.embedding_pooling = pooling;
        }
        self
    }

    pub fn with_eos_tokens(mut self: Arc<Self>, eos_tokens: Vec<u32>) -> Arc<Self> {
        if let Some(inner) = Arc::get_mut(&mut self) {
            inner.eos_tokens = eos_tokens;
        }
        self
    }

    /// Set the maximum context (prompt + generated tokens). Requests whose
    /// prompt already fills it are rejected; `max_tokens` is otherwise clamped to
    /// the remaining budget. Defaults to `usize::MAX` (no limit).
    pub fn with_context_window(mut self: Arc<Self>, max_context: usize) -> Arc<Self> {
        if let Some(inner) = Arc::get_mut(&mut self) {
            inner.max_context = max_context;
        }
        self
    }

    /// Submit a request at `priority` (lower is admitted first); returns a
    /// channel that yields its tokens then `Done`.
    fn submit_at(
        &self,
        prompt: Vec<u32>,
        images: Vec<(usize, Vec<f32>)>,
        max_new: usize,
        eos: Vec<u32>,
        decoding: Decoding,
        priority: i32,
    ) -> Receiver<TokenEvent> {
        let (sink, out) = channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .job_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(Work::Generate(Box::new(Job {
                id,
                priority,
                prompt,
                vision: self.vision.clone().filter(|_| !images.is_empty()),
                images,
                max_new,
                eos,
                decoding,
                sink,
            })));
        out
    }

    /// `tokens`' embedding (see [`Generator::embedding`]), computed on the
    /// engine thread between decode steps.
    fn embed(&self, tokens: Vec<u32>) -> Result<Vec<f32>, String> {
        let (reply, out) = channel();
        self.job_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(Work::Embed {
                tokens,
                pooling: self.embedding_pooling,
                reply,
            })
            .map_err(|_| "the engine has stopped".to_string())?;
        out.recv()
            .map_err(|_| "the engine has stopped".to_string())?
    }

    /// Encode a prompt string to token ids, validating against the vocabulary.
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, String> {
        let ids = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| format!("tokenize: {e}"))?;
        if ids.is_empty() {
            return Err("prompt encodes to no tokens".into());
        }
        if let Some(&m) = ids.iter().max() {
            if m as usize >= self.vocab_size {
                return Err("prompt token out of model vocab range".into());
            }
        }
        Ok(ids)
    }
}

/// What proposes tokens for the engine's speculative rounds, if anything.
enum Drafting<K: ComputeKernel> {
    None,
    Model(Box<Generator<K>>),
    Eagle3(Box<crate::forward::eagle3::Eagle3Head>),
}

#[allow(clippy::too_many_arguments)]
fn engine_loop<K: ComputeKernel>(
    generator: Generator<K>,
    draft: Drafting<K>,
    gamma: usize,
    job_rx: Receiver<Work>,
    max_batch: usize,
    prefix_cache_size: usize,
    stream: Arc<StreamMetrics>,
    queue_timeout: Arc<AtomicU64>,
) {
    let mut sched = match &draft {
        // Draft sessions can't resume, so they get no prefix cache.
        Drafting::Model(d) => BatchScheduler::with_speculative(&generator, d, max_batch, gamma),
        Drafting::Eagle3(h) => BatchScheduler::with_eagle3(&generator, h, max_batch, gamma),
        Drafting::None if gamma > 0 => BatchScheduler::with_ngram(&generator, max_batch, gamma)
            .with_prefix_cache(prefix_cache_size),
        Drafting::None => {
            BatchScheduler::new(&generator, max_batch).with_prefix_cache(prefix_cache_size)
        }
    };
    let mut sinks: HashMap<u64, Sender<TokenEvent>> = HashMap::new();
    // The gauges hold from the start, not only after the first request.
    let free = generator.kv_free_tokens().map_or(u64::MAX, |t| t as u64);
    stream.kv_free.store(free, Ordering::Relaxed);

    loop {
        let mut take = |work: Work, sched: &mut BatchScheduler<'_, K>| match work {
            Work::Generate(job) => enqueue(sched, &mut sinks, *job),
            Work::Embed {
                tokens,
                pooling,
                reply,
            } => {
                let _ = reply.send(
                    generator
                        .embedding(&tokens, pooling)
                        .map_err(|e| e.to_string()),
                );
            }
            Work::PersistPrefixCache(dir) => match sched.persist_prefix_cache(dir.clone()) {
                Ok(n) => println!("prefix cache : {n} prefixes loaded from {}", dir.display()),
                Err(e) => eprintln!("prefix cache : cannot use {}: {e}", dir.display()),
            },
        };
        if !sched.has_work() {
            match job_rx.recv() {
                Ok(work) => take(work, &mut sched),
                Err(_) => break, // all handles dropped
            }
        }
        while let Ok(work) = job_rx.try_recv() {
            take(work, &mut sched);
        }
        let limit = queue_timeout.load(Ordering::Relaxed);
        if limit > 0 {
            for id in sched.expire_pending(std::time::Duration::from_millis(limit)) {
                if let Some(s) = sinks.remove(&id) {
                    let _ = s.send(TokenEvent::Rejected(format!(
                        "the server is busy: the request waited over {:.1} s in the queue",
                        limit as f64 / 1000.0
                    )));
                }
            }
        }
        if !sched.has_work() {
            continue;
        }
        match sched.step() {
            Ok(mut tick) => {
                for (id, token) in tick.produced {
                    // A request with log-probabilities makes one token a tick.
                    let lp = tick
                        .logprobs
                        .iter()
                        .position(|(lid, _)| *lid == id)
                        .map(|i| tick.logprobs.swap_remove(i).1);
                    if let Some(s) = sinks.get(&id) {
                        // A failed send means the client dropped the receiver
                        // (disconnected or finished early); stop wasting compute
                        // on it by retiring the slot.
                        if s.send(TokenEvent::Next(token, lp)).is_err() {
                            sinks.remove(&id);
                            sched.abort(id);
                        }
                    }
                }
                for id in tick.finished {
                    if let Some(s) = sinks.remove(&id) {
                        let stats = tick
                            .finished_stats
                            .iter()
                            .find(|(fid, _)| *fid == id)
                            .map(|(_, st)| *st);
                        let _ = s.send(TokenEvent::Done(stats));
                    }
                }
            }
            Err(_) => {
                // Retire the requests along with their streams: left in the
                // scheduler, a session that failed (an exhausted KV pool, say)
                // would fail again on every tick after.
                for (id, s) in sinks.drain() {
                    sched.abort(id);
                    let _ = s.send(TokenEvent::Done(None));
                }
            }
        }
        // Publish the streaming kernel's latest cache stats (no-op / None for
        // resident kernels). Cheap: one stats read per token step.
        if let Some(s) = generator.stream_stats() {
            stream.publish(s);
        }
        stream
            .running
            .store(sched.active_len() as u64, Ordering::Relaxed);
        stream
            .waiting
            .store(sched.pending_len() as u64, Ordering::Relaxed);
        let free = generator.kv_free_tokens().map_or(u64::MAX, |t| t as u64);
        stream.kv_free.store(free, Ordering::Relaxed);
    }
}

fn enqueue<K: ComputeKernel>(
    sched: &mut BatchScheduler<K>,
    sinks: &mut HashMap<u64, Sender<TokenEvent>>,
    job: Job,
) {
    let Job {
        id,
        priority,
        prompt,
        images,
        vision,
        max_new,
        eos,
        decoding,
        sink,
    } = job;
    // Images are encoded here, on the thread that owns the GPU: other requests
    // wait for it, as they wait for a prompt's prefill.
    let images = match (&vision, images.is_empty()) {
        (_, true) => Vec::new(),
        (None, false) => unreachable!("a job with images carries the vision tower"),
        (Some(v), false) => {
            let encoded = images
                .into_iter()
                .map(|(start, pixels)| {
                    v.tower
                        .encode(&pixels, v.gpu)
                        .map(|rows| crate::generate::ImageEmbeds { start, rows })
                })
                .collect::<crate::error::Result<Vec<_>>>();
            match encoded {
                Ok(e) => e,
                Err(e) => {
                    let _ = sink.send(TokenEvent::Rejected(format!(
                        "cannot encode the image: {e}"
                    )));
                    return;
                }
            }
        }
    };
    match sched.submit_with_images(id, prompt, images, max_new, eos, decoding, priority) {
        Ok(()) => {
            sinks.insert(id, sink);
        }
        Err(_) => {
            let _ = sink.send(TokenEvent::Done(None));
        }
    }
}

// ── OpenAI request/response shapes ──────────────────────────────────────────

/// A chat message (`role` + `content`), shared by the local and distributed
/// serving paths. An assistant message may carry the tool calls it made, and a
/// `tool` message is a call's result.
#[derive(Deserialize, Default)]
#[serde(from = "RawChatMessage")]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCallIn>>,
    /// On a `tool` message, the call it answers (Mistral's format names it).
    pub tool_call_id: Option<String>,
    /// The `image_url` parts of `content`: each one's URL, and the byte offset
    /// in `content` (the text parts joined) where it stood.
    pub images: Vec<(usize, String)>,
}

/// A chat message as it arrives, before its `content` parts are split into
/// text and images.
#[derive(Deserialize)]
struct RawChatMessage {
    role: String,
    content: Content,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallIn>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

impl From<RawChatMessage> for ChatMessage {
    fn from(m: RawChatMessage) -> Self {
        let (content, images) = m.content.split();
        Self {
            role: m.role,
            content,
            tool_calls: m.tool_calls,
            tool_call_id: m.tool_call_id,
            images,
        }
    }
}

impl ChatMessage {
    /// The content as Gemma's template writes it: the text trimmed, and with
    /// images, each text part trimmed and each image a `<start_of_image>`.
    fn gemma_content(&self) -> String {
        if self.images.is_empty() {
            return self.content.trim().to_string();
        }
        let mut out = String::new();
        let mut from = 0;
        for (at, _) in &self.images {
            out.push_str(self.content[from..*at].trim());
            out.push_str(crate::vision::IMAGE_MARKER);
            from = *at;
        }
        out.push_str(self.content[from..].trim());
        out
    }
}

/// JSON as Hugging Face chat templates' `tojson` writes it: `", "` and `": "`
/// between items, non-ASCII as is.
fn tojson(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(tojson).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), tojson(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// ChatML, with Qwen's tool calling (its Jinja template, followed to the
/// character): the tools in the system turn, an assistant's calls as
/// `<tool_call>` blocks, and consecutive `tool` results in one user turn as
/// `<tool_response>` blocks. Without tools or tool turns it is plain ChatML.
fn chatml(messages: &[ChatMessage], tools: &[serde_json::Value]) -> String {
    let mut p = String::new();
    let mut rest = messages;
    if !tools.is_empty() {
        p.push_str("<|im_start|>system\n");
        if let Some(system) = messages.first().filter(|m| m.role == "system") {
            p.push_str(&system.content);
            p.push_str("\n\n");
            rest = &messages[1..];
        }
        p.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
             You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for tool in tools {
            p.push('\n');
            p.push_str(&tojson(tool));
        }
        p.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and \
             arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n",
        );
    }
    for (i, m) in rest.iter().enumerate() {
        let calls = m.tool_calls.as_deref().unwrap_or_default();
        match m.role.as_str() {
            "assistant" if !calls.is_empty() => {
                p.push_str("<|im_start|>assistant");
                if !m.content.is_empty() {
                    p.push('\n');
                    p.push_str(&m.content);
                }
                for c in calls {
                    // Arguments arrive as a JSON string; the model saw an object.
                    let args = match &c.function.arguments {
                        serde_json::Value::String(s) => serde_json::from_str(s)
                            .map(|v| tojson(&v))
                            .unwrap_or_else(|_| s.clone()),
                        v => tojson(v),
                    };
                    p.push_str(&format!(
                        "\n<tool_call>\n{{\"name\": \"{}\", \"arguments\": {args}}}\n</tool_call>",
                        c.function.name
                    ));
                }
                p.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i == 0 || rest[i - 1].role != "tool" {
                    p.push_str("<|im_start|>user");
                }
                p.push_str("\n<tool_response>\n");
                p.push_str(&m.content);
                p.push_str("\n</tool_response>");
                if rest.get(i + 1).is_none_or(|n| n.role != "tool") {
                    p.push_str("<|im_end|>\n");
                }
            }
            role => p.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", m.content)),
        }
    }
    p.push_str("<|im_start|>assistant\n");
    p
}

/// Qwen3.5/3.6's chat template without tools, followed to the character: every
/// content trimmed; an assistant turn's `<think>…</think>` block split off its
/// answer, and shown again (as `<think>\n{reasoning}\n</think>\n\n`) only on
/// turns after the last user message; the reply opened with thinking on
/// (`<|im_start|>assistant\n<think>\n`). Tool calls take Qwen3-Coder's XML
/// format there, which dlm does not render or parse yet, so a request with
/// tools is refused ([`ChatTemplate::apply_with_tools`]).
fn qwen35(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    let mut rest = messages;
    if let Some(system) = messages.first().filter(|m| m.role == "system") {
        p.push_str(&format!(
            "<|im_start|>system\n{}<|im_end|>\n",
            system.content.trim()
        ));
        rest = &messages[1..];
    }
    // The template's `last_query_index`: the last user turn that is not just a
    // tool result.
    let is_query = |m: &ChatMessage| {
        let c = m.content.trim();
        m.role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
    };
    let last_query = rest.iter().rposition(is_query);
    for (i, m) in rest.iter().enumerate() {
        let content = m.content.trim();
        match m.role.as_str() {
            "assistant" => {
                let (reasoning, answer) = match content.split_once("</think>") {
                    Some((before, _)) => {
                        let after = content.rsplit("</think>").next().unwrap_or("");
                        let reasoning = before.trim_end_matches('\n');
                        let reasoning = reasoning.rsplit("<think>").next().unwrap_or("");
                        (
                            reasoning.trim_start_matches('\n').trim(),
                            after.trim_start_matches('\n'),
                        )
                    }
                    None => ("", content),
                };
                p.push_str("<|im_start|>assistant\n");
                if last_query.is_some_and(|q| i > q) {
                    p.push_str(&format!("<think>\n{reasoning}\n</think>\n\n{answer}"));
                } else {
                    p.push_str(answer);
                }
                p.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i == 0 || rest[i - 1].role != "tool" {
                    p.push_str("<|im_start|>user");
                }
                p.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
                if rest.get(i + 1).is_none_or(|n| n.role != "tool") {
                    p.push_str("<|im_end|>\n");
                }
            }
            role => p.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n")),
        }
    }
    p.push_str("<|im_start|>assistant\n<think>\n");
    p
}

/// A call's arguments as the model saw them: an object, where OpenAI sends a
/// JSON string.
fn call_arguments(c: &ToolCallIn) -> serde_json::Value {
    match &c.function.arguments {
        serde_json::Value::String(s) => {
            serde_json::from_str(s).unwrap_or_else(|_| serde_json::Value::String(s.clone()))
        }
        v => v.clone(),
    }
}

/// Today as Llama 3's template writes it (`strftime("%d %b %Y")`, UTC).
fn today_dmy() -> String {
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86_400) as i64;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!("{day:02} {} {year}", MONTHS[month as usize - 1])
}

/// JSON as Python's `json.dumps(v, indent=4)` writes it, which Llama 3's
/// template uses for each tool.
fn tojson_indent4(v: &serde_json::Value) -> String {
    use serde::Serialize;
    let mut out = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut out, fmt);
    v.serialize(&mut ser).expect("a JSON value serializes");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

/// Llama 3.1/3.2 with tools, following its Jinja template to the character with
/// its defaults (`tools_in_user_message`): an `ipython` system header, the tools
/// and their instructions in the first user turn, a call as a bare
/// `{"name": …, "parameters": …}` reply, and a result as an `ipython` turn.
fn llama3_tools(
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
    date: &str,
) -> Result<String, String> {
    let (system, mut rest) = match messages.first() {
        Some(m) if m.role == "system" => (m.content.trim(), &messages[1..]),
        _ => ("", messages),
    };
    let mut p = String::from("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
    p.push_str("Environment: ipython\nCutting Knowledge Date: December 2023\n");
    p.push_str(&format!("Today Date: {date}\n\n{system}<|eot_id|>"));
    let Some(first) = rest.first() else {
        return Err(
            "Cannot put tools in the first user message when there's no first user message!".into(),
        );
    };
    rest = &rest[1..];
    p.push_str(
        "<|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please \
         respond with a JSON for a function call with its proper arguments that best answers \
         the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": \
         dictionary of argument name and its value}.Do not use variables.\n\n",
    );
    for t in tools {
        p.push_str(&tojson_indent4(t));
        p.push_str("\n\n");
    }
    p.push_str(first.content.trim());
    p.push_str("<|eot_id|>");
    for m in rest {
        let calls = m.tool_calls.as_deref().unwrap_or_default();
        if !calls.is_empty() {
            let [c] = calls else {
                return Err("This model only supports single tool-calls at once!".into());
            };
            p.push_str(&format!(
                "<|start_header_id|>assistant<|end_header_id|>\n\n{{\"name\": \"{}\", \
                 \"parameters\": {}}}<|eot_id|>",
                c.function.name,
                tojson(&call_arguments(c))
            ));
        } else if m.role == "tool" || m.role == "ipython" {
            // The template JSON-encodes the result: to Jinja a string is iterable.
            p.push_str("<|start_header_id|>ipython<|end_header_id|>\n\n");
            p.push_str(&tojson(&serde_json::Value::String(m.content.clone())));
            p.push_str("<|eot_id|>");
        } else {
            p.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                m.role,
                m.content.trim()
            ));
        }
    }
    p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    Ok(p)
}

/// A tool-call id as Mistral's template requires it: 9 alphanumerics. An id
/// of that shape passes through; any other (OpenAI's `call_…`, or none) maps
/// to one derived from it, the same for the call and the result naming it.
fn mistral_call_id(id: Option<&str>, fallback: usize) -> String {
    match id {
        Some(id) if id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric()) => id.into(),
        _ => {
            let key = id.map_or_else(|| fallback.to_string(), str::to_string);
            let mut h = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
            });
            const ALNUM: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            (0..9)
                .map(|_| {
                    let c = ALNUM[(h % ALNUM.len() as u64) as usize] as char;
                    h /= ALNUM.len() as u64;
                    c
                })
                .collect()
        }
    }
}

/// Mistral v0.3 with tools, following its Jinja template to the character:
/// `[AVAILABLE_TOOLS]` before the last user turn, the system message folded
/// into the last user turn, calls as `[TOOL_CALLS] [...]` and results as
/// `[TOOL_RESULTS]`. BOS is left to the tokenizer, as for plain Mistral.
fn mistral_tools(messages: &[ChatMessage], tools: &[serde_json::Value]) -> Result<String, String> {
    let (system, rest) = match messages.first() {
        Some(m) if m.role == "system" => (Some(m.content.as_str()), &messages[1..]),
        _ => (None, messages),
    };
    let is_call = |m: &ChatMessage| m.tool_calls.as_ref().is_some_and(|c| !c.is_empty());
    let turns = rest.iter().filter(|m| !(m.role == "tool" || is_call(m)));
    for (turn, m) in turns.enumerate() {
        if (m.role == "user") != (turn % 2 == 0) {
            return Err(
                "After the optional system message, conversation roles must alternate \
                        user/assistant/user/assistant/..."
                    .into(),
            );
        }
    }
    let last_user = rest.iter().rposition(|m| m.role == "user");
    let mut p = String::new();
    let mut call_index = 0;
    for (i, m) in rest.iter().enumerate() {
        if m.role == "user" {
            if Some(i) == last_user {
                let listed: Vec<String> = tools
                    .iter()
                    .map(|t| {
                        let fields: Vec<String> = t["function"]
                            .as_object()
                            .into_iter()
                            .flatten()
                            .filter(|(k, _)| k.as_str() != "return")
                            .map(|(k, v)| match v {
                                serde_json::Value::String(s) => format!("\"{k}\": \"{s}\""),
                                v => format!("\"{k}\": {}", tojson(v)),
                            })
                            .collect();
                        format!(
                            "{{\"type\": \"function\", \"function\": {{{}}}}}",
                            fields.join(", ")
                        )
                    })
                    .collect();
                p.push_str(&format!(
                    "[AVAILABLE_TOOLS] [{}][/AVAILABLE_TOOLS]",
                    listed.join(", ")
                ));
            }
            match system {
                Some(sys) if i + 1 == rest.len() => {
                    p.push_str(&format!("[INST] {sys}\n\n{}[/INST]", m.content))
                }
                _ => p.push_str(&format!("[INST] {}[/INST]", m.content)),
            }
        } else if is_call(m) {
            let calls: Vec<String> = m
                .tool_calls
                .iter()
                .flatten()
                .map(|c| {
                    call_index += 1;
                    format!(
                        "{{\"name\": {}, \"arguments\": {}, \"id\": \"{}\"}}",
                        tojson(&serde_json::Value::String(c.function.name.clone())),
                        tojson(&call_arguments(c)),
                        mistral_call_id(c.id.as_deref(), call_index)
                    )
                })
                .collect();
            p.push_str(&format!("[TOOL_CALLS] [{}]</s>", calls.join(", ")));
        } else if m.role == "assistant" {
            p.push_str(&format!(" {}</s>", m.content.trim()));
        } else if m.role == "tool" {
            p.push_str(&format!(
                "[TOOL_RESULTS] {{\"content\": {}, \"call_id\": \"{}\"}}[/TOOL_RESULTS]",
                m.content,
                mistral_call_id(m.tool_call_id.as_deref(), call_index)
            ));
        } else {
            return Err(
                "Only user and assistant roles are supported, with the exception of an \
                        initial optional system message!"
                    .into(),
            );
        }
    }
    Ok(p)
}

/// Where a tool call starts in a ChatML reply.
const TOOL_OPEN: &str = "<tool_call>";

/// The calls in a reply, as `(name, arguments JSON)`: every JSON object (or
/// each object of a JSON array) from `start` on, taking its arguments from
/// `arguments` or Llama's `parameters`. `None` unless all of it parses.
fn json_calls(after: &str) -> Option<Vec<(String, String)>> {
    let mut calls = Vec::new();
    for v in serde_json::Deserializer::from_str(after).into_iter::<serde_json::Value>() {
        let v = v.ok()?;
        let items = match v {
            serde_json::Value::Array(a) => a,
            one => vec![one],
        };
        for c in items {
            let name = c["name"].as_str()?.to_string();
            let args = match c.get("arguments").or_else(|| c.get("parameters")) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => "{}".to_string(),
            };
            calls.push((name, args));
        }
    }
    (!calls.is_empty()).then_some(calls)
}

/// A reply's text before its first tool call, and each call's name and
/// arguments, in `template`'s format: ChatML's `<tool_call>` blocks, Mistral's
/// `[TOOL_CALLS] [...]`, or Llama 3's bare JSON object. A reply whose call does
/// not parse is left as text, with no calls.
fn split_tool_calls_for(template: ChatTemplate, text: &str) -> (String, Vec<(String, String)>) {
    let open = template.tool_open();
    if open == TOOL_OPEN {
        return split_tool_calls(text);
    }
    let Some(start) = text.find(open) else {
        return (text.to_string(), Vec::new());
    };
    // Mistral's marker is a token of its own; Llama's is the call's first bytes.
    let body = if template == ChatTemplate::Mistral {
        &text[start + open.len()..]
    } else {
        &text[start..]
    };
    match json_calls(body) {
        Some(calls) => (text[..start].trim_end().to_string(), calls),
        None => (text.to_string(), Vec::new()),
    }
}

/// ChatML's [`split_tool_calls_for`]: a reply's text before its first
/// `<tool_call>`, and each call's name and arguments (a JSON string). A reply
/// whose blocks do not hold a call's JSON is left as text, with no calls.
fn split_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    const CLOSE: &str = "</tool_call>";
    let Some(start) = text.find(TOOL_OPEN) else {
        return (text.to_string(), Vec::new());
    };
    let mut calls = Vec::new();
    let mut rest = &text[start..];
    while let Some(open) = rest.find(TOOL_OPEN) {
        let body = &rest[open + TOOL_OPEN.len()..];
        let end = body.find(CLOSE);
        let call: Option<serde_json::Value> =
            serde_json::from_str(body[..end.unwrap_or(body.len())].trim()).ok();
        let Some(name) = call.as_ref().and_then(|c| c["name"].as_str()) else {
            return (text.to_string(), Vec::new());
        };
        let arguments = match call.as_ref().and_then(|c| c.get("arguments")) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => "{}".to_string(),
        };
        calls.push((name.to_string(), arguments));
        rest = end.map_or("", |e| &body[e + CLOSE.len()..]);
    }
    (text[..start].trim_end().to_string(), calls)
}

/// How many of `text`'s last bytes could begin `marker`, which then cannot be
/// shown yet.
fn partial_marker(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&k| text.ends_with(&marker[..k]))
        .unwrap_or(0)
}

/// A tool call in an assistant message, as OpenAI sends it back.
#[derive(Deserialize, Default)]
pub struct ToolCallIn {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: FunctionCallIn,
}

#[derive(Deserialize, Default)]
pub struct FunctionCallIn {
    #[serde(default)]
    pub name: String,
    /// A JSON string, as OpenAI sends it, or the object itself.
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// OpenAI's `content`: a string, or a list of typed parts.
///
/// Clients send both. The array form is what anything that can attach an image
/// uses, and several send it for plain text too -- Open WebUI and the OpenAI
/// SDKs among them -- so rejecting it turns a working client into a 400.
#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
    /// `content: null`, which tool-call replies carry.
    Null,
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    /// `{"url": "..."}`, or the URL as a bare string, which some clients send.
    #[serde(default)]
    image_url: Option<serde_json::Value>,
}

impl Content {
    /// The text of a `content` field, whichever form it arrived in, and the
    /// URLs of its images with where each stood in that text. A model without
    /// a vision tower ignores the images: a conversation that mentions a
    /// picture is still worth answering.
    fn split(self) -> (String, Vec<(usize, String)>) {
        match self {
            Content::Text(s) => (s, Vec::new()),
            Content::Null => (String::new(), Vec::new()),
            Content::Parts(parts) => {
                let (mut text, mut images) = (String::new(), Vec::new());
                for p in parts {
                    match p.kind.as_deref().unwrap_or("text") {
                        "text" => text.push_str(p.text.as_deref().unwrap_or_default()),
                        "image_url" => {
                            let url = p
                                .image_url
                                .as_ref()
                                .and_then(|u| u.get("url").and_then(|u| u.as_str()).or(u.as_str()));
                            images.push((text.len(), url.unwrap_or_default().to_string()));
                        }
                        _ => {}
                    }
                }
                (text, images)
            }
        }
    }
}

/// OpenAI `stop`: either a single string or a list of them.
#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    /// Sampling temperature (0 or absent → deterministic greedy).
    #[serde(default)]
    temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    #[serde(default)]
    top_p: Option<f32>,
    /// Top-k truncation (non-standard OpenAI extension; 0/absent → all tokens).
    #[serde(default)]
    top_k: Option<u32>,
    /// Min-p truncation (non-standard extension; 0/absent → disabled).
    #[serde(default)]
    min_p: Option<f32>,
    /// Repetition penalty (non-standard extension; 1.0/absent → disabled).
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// Optional RNG seed for reproducible sampling.
    #[serde(default)]
    seed: Option<u64>,
    /// Stop sequence(s): generation is cut before the first occurrence.
    #[serde(default)]
    stop: Option<Stop>,
    /// Return each output token's log-probability.
    #[serde(default)]
    logprobs: bool,
    /// And this many of the most likely tokens at each position (0-20).
    #[serde(default)]
    top_logprobs: Option<usize>,
    /// How many completions to generate for the prompt (1-16).
    #[serde(default)]
    n: Option<usize>,
    /// Queue priority: lower is admitted first (vLLM's field). 0 by default.
    #[serde(default)]
    priority: i32,
    /// Functions the model may call (`{"type": "function", "function": ...}`).
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// `auto` (the default with tools) or `none`.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(flatten)]
    extra: ExtraSampling,
}

/// The OpenAI sampling fields that layer on the sampler, shared by chat and
/// completions. [`decoding`] validates them before anything is queued.
#[derive(Deserialize, Default)]
struct ExtraSampling {
    /// Token id (as a string, OpenAI's shape) → bias added to its logit.
    #[serde(default)]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    /// DRY (non-standard, llama.cpp's names): off unless `dry_multiplier > 0`.
    #[serde(default)]
    dry_multiplier: Option<f32>,
    #[serde(default)]
    dry_base: Option<f32>,
    #[serde(default)]
    dry_allowed_length: Option<usize>,
    #[serde(default)]
    dry_sequence_breakers: Option<Vec<String>>,
    /// `json_object` or `json_schema` constrains the output to JSON.
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    /// A GBNF grammar the output must match (non-standard, llama.cpp's field).
    #[serde(default)]
    grammar: Option<String>,
}

/// OpenAI's `response_format`.
#[derive(Deserialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    json_schema: Option<JsonSchemaFormat>,
}

/// `response_format.json_schema`. Its `name` and `strict` are accepted and
/// ignored: the output is always held to the schema.
#[derive(Deserialize)]
struct JsonSchemaFormat {
    #[serde(default)]
    schema: Option<serde_json::Value>,
}

/// DRY's sequence breakers when a request names none (llama.cpp's defaults).
const DRY_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// Check the request's extra sampling fields and combine them with `sampler`.
/// An out-of-range value is a client error, not something to clamp quietly: a
/// client that asked for it would otherwise get output it did not ask for and no
/// sign why.
/// Most completions one request may ask for (`n`).
const MAX_N: usize = 16;

/// One [`Decoding`] per choice of a request asking for `n`: a sampled choice
/// draws with its own seed, `seed + i`, so the choices differ.
fn choices(decoding: Decoding, n: Option<usize>) -> Result<Vec<Decoding>, String> {
    let n = n.unwrap_or(1);
    if !(1..=MAX_N).contains(&n) {
        return Err(format!("n must be between 1 and {MAX_N}"));
    }
    Ok((0..n as u64)
        .map(|i| {
            let mut d = decoding.clone();
            if let Sampler::TopPK { seed, .. } = &mut d.sampler {
                *seed = seed.wrapping_add(i);
            }
            d
        })
        .collect())
}

/// Most alternatives a request may ask for per token, as OpenAI allows.
const MAX_TOP_LOGPROBS: usize = 20;

/// `top` alternatives per token when `wanted`, checked against the limit.
fn logprobs_request(wanted: bool, top: Option<usize>) -> Result<Option<usize>, String> {
    match (wanted, top) {
        (_, Some(n)) if n > MAX_TOP_LOGPROBS => Err(format!(
            "top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}"
        )),
        (false, Some(n)) if n > 0 => Err("top_logprobs needs logprobs: true".into()),
        (true, n) => Ok(Some(n.unwrap_or(0))),
        (false, _) => Ok(None),
    }
}

fn decoding(
    engine: &EngineService,
    sampler: Sampler,
    extra: ExtraSampling,
    logprobs: Option<usize>,
) -> Result<Decoding, String> {
    let mut logit_bias = Vec::new();
    for (key, &bias) in extra.logit_bias.iter().flatten() {
        let token: u32 = key
            .parse()
            .map_err(|_| format!("logit_bias: '{key}' is not a token id"))?;
        if token as usize >= engine.vocab_size {
            return Err(format!(
                "logit_bias: token {token} is outside the vocabulary"
            ));
        }
        if !(-100.0..=100.0).contains(&bias) {
            return Err(format!(
                "logit_bias: the bias for token {token} must be between -100 and 100"
            ));
        }
        logit_bias.push((token, bias));
    }
    let penalty = |name: &str, value: Option<f32>| match value.unwrap_or(0.0) {
        v if (-2.0..=2.0).contains(&v) => Ok(v),
        _ => Err(format!("{name} must be between -2 and 2")),
    };
    Ok(Decoding {
        sampler,
        penalties: Penalties {
            logit_bias,
            frequency: penalty("frequency_penalty", extra.frequency_penalty)?,
            presence: penalty("presence_penalty", extra.presence_penalty)?,
            dry: dry(engine, &extra)?,
        },
        grammar: constraint(engine, &extra)?,
        logprobs,
    })
}

/// The grammar the request's output must match, if it asked for one. A
/// grammar or schema that does not parse is the client's error to fix, so it
/// is refused here rather than failing the request mid-generation.
fn constraint(engine: &EngineService, extra: &ExtraSampling) -> Result<Option<Constraint>, String> {
    let format = extra.response_format.as_ref();
    let grammar = match (format.map(|f| f.kind.as_str()), &extra.grammar) {
        (Some("json_object" | "json_schema"), Some(_)) => {
            return Err("set grammar or a JSON response_format, not both".into())
        }
        (None | Some("text"), None) => return Ok(None),
        (None | Some("text"), Some(gbnf)) => Grammar::parse(gbnf)?,
        (Some("json_object"), None) => Grammar::json_object(),
        (Some("json_schema"), None) => {
            let schema = format
                .and_then(|f| f.json_schema.as_ref())
                .and_then(|j| j.schema.as_ref())
                .ok_or("response_format: json_schema needs json_schema.schema")?;
            Grammar::from_json_schema(schema)?
        }
        (Some(other), _) => {
            return Err(format!(
                "response_format: type '{other}' is not text, json_object or json_schema"
            ))
        }
    };
    Ok(Some(Constraint::new(
        Arc::new(grammar),
        engine.token_bytes(),
        engine.eos_tokens.clone(),
    )))
}

/// The request's DRY settings, or `None` when its multiplier leaves it off.
fn dry(engine: &EngineService, extra: &ExtraSampling) -> Result<Option<Dry>, String> {
    let multiplier = extra.dry_multiplier.unwrap_or(0.0);
    let base = extra.dry_base.unwrap_or(1.75);
    let allowed_length = extra.dry_allowed_length.unwrap_or(2);
    if !(0.0..=f32::MAX).contains(&multiplier) {
        return Err("dry_multiplier must be 0 or more".into());
    }
    if !(1.0..=f32::MAX).contains(&base) {
        return Err("dry_base must be at least 1".into());
    }
    if allowed_length == 0 {
        return Err("dry_allowed_length must be at least 1".into());
    }
    let breakers: Vec<&[u8]> = match &extra.dry_sequence_breakers {
        Some(b) => b.iter().map(|s| s.as_bytes()).collect(),
        None => DRY_BREAKERS.iter().map(|s| s.as_bytes()).collect(),
    };
    if breakers.iter().any(|b| b.is_empty()) {
        return Err("dry_sequence_breakers must not contain an empty string".into());
    }
    if multiplier == 0.0 {
        return Ok(None);
    }
    // ponytail: a breaker counts only inside one token; one that a tokenizer
    // splits across two (a multi-character breaker) is not seen. llama.cpp
    // tracks those; the default single-character breakers never need it.
    let tokens = engine.token_bytes();
    let breaks = |t: &[u8]| breakers.iter().any(|b| t.windows(b.len()).any(|w| w == *b));
    let breakers = (0u32..)
        .zip(tokens.iter())
        .filter(|(_, bytes)| bytes.as_deref().is_some_and(breaks))
        .map(|(id, _)| id)
        .collect();
    Ok(Some(Dry {
        multiplier,
        base,
        allowed_length,
        breakers,
    }))
}

/// Flatten the `stop` field into a list of non-empty stop strings.
fn normalize_stop(stop: Option<Stop>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(Stop::One(s)) => vec![s],
        Some(Stop::Many(v)) => v,
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect()
}

/// Byte offset of the earliest stop-sequence occurrence in `text`, if any.
fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|s| text.find(s.as_str())).min()
}

/// Build a [`Sampler`] from the request's sampling fields. Falls back to greedy
/// when no positive temperature is given; seeds from `seed` or the request `id`.
fn sampler_from_request(req: &ChatRequest, id: u64) -> Sampler {
    match req.temperature {
        Some(t) if t > 0.0 => Sampler::TopPK {
            temperature: t,
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(0),
            min_p: req.min_p.unwrap_or(0.0),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            seed: req.seed.unwrap_or(id),
        },
        _ => Sampler::Greedy,
    }
}

#[derive(Serialize)]
struct RespMessage {
    role: &'static str,
    /// `null` when the reply is only tool calls.
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCallOut>,
}

/// A tool call the model made, in OpenAI's shape: `arguments` is a JSON
/// string. In a streamed delta it also carries its `index`.
#[derive(Serialize)]
struct ToolCallOut {
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<usize>,
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionCallOut,
}

#[derive(Serialize)]
struct FunctionCallOut {
    name: String,
    arguments: String,
}

/// `calls` in OpenAI's shape, numbered for request `request`. `index` marks
/// each with its position, for a streamed delta.
fn tool_calls_out(calls: Vec<(String, String)>, request: u64, index: bool) -> Vec<ToolCallOut> {
    calls
        .into_iter()
        .enumerate()
        .map(|(i, (name, arguments))| ToolCallOut {
            index: index.then_some(i),
            id: format!("call_{request}_{i}"),
            kind: "function",
            function: FunctionCallOut { name, arguments },
        })
        .collect()
}
#[derive(Serialize)]
struct Choice {
    index: usize,
    message: RespMessage,
    finish_reason: &'static str,
    logprobs: Option<ChatLogprobs>,
}

/// Chat's `logprobs`: one entry per output token.
#[derive(Serialize)]
struct ChatLogprobs {
    content: Vec<ChatTokenLogprob>,
}
#[derive(Serialize)]
struct ChatTokenLogprob {
    token: String,
    logprob: f32,
    /// The token's bytes: a byte-level token can be part of a character.
    bytes: Vec<u8>,
    top_logprobs: Vec<TopLogprob>,
}
#[derive(Serialize)]
struct TopLogprob {
    token: String,
    logprob: f32,
    bytes: Vec<u8>,
}

/// A token's text (lossy where it is part of a character) and bytes.
fn token_piece(engine: &EngineService, id: u32) -> (String, Vec<u8>) {
    let bytes = engine.tokenizer.token_bytes(id).unwrap_or_default();
    (String::from_utf8_lossy(&bytes).into_owned(), bytes)
}

fn chat_logprob(engine: &EngineService, id: u32, lp: &TokenLogprobs) -> ChatTokenLogprob {
    let (token, bytes) = token_piece(engine, id);
    ChatTokenLogprob {
        token,
        logprob: lp.logprob,
        bytes,
        top_logprobs: lp
            .top
            .iter()
            .map(|&(t, logprob)| {
                let (token, bytes) = token_piece(engine, t);
                TopLogprob {
                    token,
                    logprob,
                    bytes,
                }
            })
            .collect(),
    }
}

/// The legacy completions `logprobs`: parallel lists, one entry per token.
#[derive(Serialize, Default)]
struct CompletionLogprobs {
    tokens: Vec<String>,
    token_logprobs: Vec<f32>,
    top_logprobs: Vec<HashMap<String, f32>>,
    /// Each token's character offset in the returned text.
    text_offset: Vec<usize>,
}

impl CompletionLogprobs {
    fn push(&mut self, engine: &EngineService, id: u32, lp: &TokenLogprobs, offset: usize) {
        self.tokens.push(token_piece(engine, id).0);
        self.token_logprobs.push(lp.logprob);
        self.top_logprobs.push(
            lp.top
                .iter()
                .map(|&(t, l)| (token_piece(engine, t).0, l))
                .collect(),
        );
        self.text_offset.push(offset);
    }
}
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    /// Present only when the request was decoded speculatively.
    #[serde(skip_serializing_if = "Option::is_none")]
    speculative: Option<SpeculativeUsage>,
}

/// Speculative-decoding acceptance, reported inside `usage` when the request
/// was decoded speculatively (by a draft model or n-gram lookup).
#[derive(Serialize)]
struct SpeculativeUsage {
    draft_proposed: usize,
    draft_accepted: usize,
    acceptance_rate: f64,
}

impl From<AcceptanceStats> for SpeculativeUsage {
    fn from(s: AcceptanceStats) -> Self {
        SpeculativeUsage {
            draft_proposed: s.proposed,
            draft_accepted: s.accepted,
            acceptance_rate: s.acceptance_rate(),
        }
    }
}
#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallOut>>,
}
#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<&'static str>,
    /// The log-probabilities of the tokens whose text this chunk carries.
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<ChatLogprobs>,
}
#[derive(Serialize)]
struct Chunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    /// Emitted only on the final speculative chunk (choices empty).
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}
#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

/// How chat messages are rendered into the model's prompt string. Real instruct
/// models are trained on a specific format with special control tokens; using
/// the wrong one degrades output. The control tokens only become single ids when
/// the tokenizer registers them as special (HF `tokenizer.json`); otherwise they
/// tokenize as ordinary text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    /// `role: content` lines, then `assistant:` (default; no special tokens).
    #[default]
    Plain,
    /// ChatML (`<|im_start|>role … <|im_end|>`), used by Qwen and others.
    ChatMl,
    /// Llama-3 (`<|start_header_id|>role<|end_header_id|> … <|eot_id|>`).
    Llama3,
    /// Llama-2 chat (`[INST] <<SYS>>…<</SYS>> … [/INST]`).
    Llama2,
    /// Mistral (`[INST] … [/INST]`); a system prompt is folded into the first
    /// user turn, since the format has no system role.
    Mistral,
    /// Gemma 1/2 (`<start_of_turn>user … <end_of_turn>`); the assistant role is
    /// `model`, and a system prompt is folded into the first user turn.
    Gemma,
    /// Gemma 4 (`<|turn>user … <turn|>`), which has a system turn of its own.
    Gemma4,
    /// Phi-3 (`<|user|>\n … <|end|>`).
    Phi3,
    /// DeepSeek-V2 chat (`User: …\n\nAssistant:`).
    DeepSeek,
    /// Qwen3.5/3.6: ChatML with thinking. Earlier answers lose their
    /// `<think>` block, turns after the last user query keep it, and the reply
    /// opens one (`<|im_start|>assistant\n<think>\n`). See [`qwen35`].
    Qwen35,
}

impl ChatTemplate {
    /// Every name `--chat-template` accepts, besides `auto`.
    pub const NAMES: &'static str =
        "plain, chatml, llama3, llama2, mistral, gemma, gemma4, phi3, deepseek, qwen3.5";

    /// Parse a `--chat-template` value (`auto` is resolved by the caller).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "plain" => Some(ChatTemplate::Plain),
            "chatml" => Some(ChatTemplate::ChatMl),
            "llama3" | "llama-3" => Some(ChatTemplate::Llama3),
            "llama2" | "llama-2" => Some(ChatTemplate::Llama2),
            "mistral" => Some(ChatTemplate::Mistral),
            "gemma" => Some(ChatTemplate::Gemma),
            "gemma4" => Some(ChatTemplate::Gemma4),
            "phi3" | "phi-3" => Some(ChatTemplate::Phi3),
            "deepseek" => Some(ChatTemplate::DeepSeek),
            "qwen3.5" | "qwen35" | "qwen3.6" => Some(ChatTemplate::Qwen35),
            _ => None,
        }
    }

    /// The `--chat-template` name for this template.
    pub fn name(&self) -> &'static str {
        match self {
            ChatTemplate::Plain => "plain",
            ChatTemplate::ChatMl => "chatml",
            ChatTemplate::Llama3 => "llama3",
            ChatTemplate::Llama2 => "llama2",
            ChatTemplate::Mistral => "mistral",
            ChatTemplate::Gemma => "gemma",
            ChatTemplate::Gemma4 => "gemma4",
            ChatTemplate::Phi3 => "phi3",
            ChatTemplate::DeepSeek => "deepseek",
            ChatTemplate::Qwen35 => "qwen3.5",
        }
    }

    /// The checkpoint's Jinja chat template, if it ships one. `chat_template` may be
    /// a string or a list of named templates, of which `default` is the chat one.
    pub fn read_jinja(model_path: &std::path::Path) -> Option<String> {
        // A GGUF file carries it in its metadata.
        if model_path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let store = crate::storage::MmapStore::open_path(model_path).ok()?;
            let meta = store.gguf_metadata()?;
            return meta
                .get("tokenizer.chat_template")?
                .as_str()
                .map(str::to_string);
        }
        if let Ok(s) = std::fs::read_to_string(model_path.join("chat_template.jinja")) {
            return Some(s);
        }
        let bytes = std::fs::read(model_path.join("tokenizer_config.json")).ok()?;
        let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        match cfg.get("chat_template")? {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(list) => list
                .iter()
                .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                .and_then(|t| t.get("template")?.as_str())
                .map(str::to_string),
            _ => None,
        }
    }

    /// Recognize a checkpoint's Jinja `chat_template` by the control markers it
    /// emits. This is a fingerprint, not a Jinja renderer: it maps the formats
    /// dlm's families are trained on onto the built-in renderers, and returns
    /// `None` for anything else, so the caller can say so rather than render a
    /// format the model never saw.
    pub fn detect(jinja: &str) -> Option<Self> {
        let has = |m: &str| jinja.contains(m);
        // Phi-4 puts `<|im_sep|>` between ChatML's role and content; rendering it
        // as plain ChatML would be quietly wrong.
        if has("<|im_sep|>") {
            None
        } else if has("<|im_start|>") && has("enable_thinking") && has("<function=") {
            // Qwen3.5/3.6: thinking, with Qwen3-Coder's XML tool calls.
            Some(ChatTemplate::Qwen35)
        } else if has("<|im_start|>") {
            Some(ChatTemplate::ChatMl)
        } else if has("<|start_header_id|>") {
            Some(ChatTemplate::Llama3)
        } else if has("<|turn>") {
            Some(ChatTemplate::Gemma4)
        } else if has("<start_of_turn>") {
            Some(ChatTemplate::Gemma)
        } else if has("<<SYS>>") {
            Some(ChatTemplate::Llama2)
        } else if has("[INST]") {
            Some(ChatTemplate::Mistral)
        } else if has("<|user|>") && has("<|end|>") {
            Some(ChatTemplate::Phi3)
        } else if has("'User: '") && has("'Assistant:") {
            Some(ChatTemplate::DeepSeek)
        } else {
            None
        }
    }

    /// Where a tool call starts in this format's replies.
    fn tool_open(&self) -> &'static str {
        match self {
            ChatTemplate::Llama3 => "{\"name\"",
            ChatTemplate::Mistral => "[TOOL_CALLS]",
            _ => TOOL_OPEN,
        }
    }

    /// The control token that closes a turn, when the format has one distinct
    /// from EOS. Generation must stop on it even when `config.json` lists only
    /// `<eos>` (gemma-1.1 does), or the model runs on into a fabricated next turn.
    pub fn end_of_turn(&self) -> Option<&'static str> {
        match self {
            ChatTemplate::ChatMl | ChatTemplate::Qwen35 => Some("<|im_end|>"),
            ChatTemplate::Llama3 => Some("<|eot_id|>"),
            ChatTemplate::Gemma => Some("<end_of_turn>"),
            ChatTemplate::Gemma4 => Some("<turn|>"),
            ChatTemplate::Phi3 => Some("<|end|>"),
            _ => None,
        }
    }

    /// Render `messages` into the prompt string for this template. BOS is left
    /// to the tokenizer (`add_bos_token`), which skips it when the text already
    /// starts with it.
    /// [`apply`](Self::apply), with `tools` the model may call, in the format
    /// the model was trained on: ChatML (Qwen's), Llama 3.1+ and Mistral v0.3.
    /// Gemma's template has no tool format, and other templates refuse tools.
    pub fn apply_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> Result<String, String> {
        match self {
            _ if tools.is_empty() => Ok(self.apply(messages)),
            ChatTemplate::ChatMl => Ok(chatml(messages, tools)),
            ChatTemplate::Llama3 => llama3_tools(messages, tools, &today_dmy()),
            ChatTemplate::Mistral => mistral_tools(messages, tools),
            other => Err(format!(
                "tools need the chatml (Qwen), llama3 or mistral chat template; this model's is {}",
                other.name()
            )),
        }
    }

    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut p = String::new();
        match self {
            ChatTemplate::Plain => {
                for m in messages {
                    p.push_str(&m.role);
                    p.push_str(": ");
                    p.push_str(&m.content);
                    p.push('\n');
                }
                p.push_str("assistant:");
            }
            ChatTemplate::ChatMl => p = chatml(messages, &[]),
            ChatTemplate::Qwen35 => p = qwen35(messages),
            ChatTemplate::Llama3 => {
                p.push_str("<|begin_of_text|>");
                for m in messages {
                    p.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role, m.content
                    ));
                }
                p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatTemplate::Llama2 | ChatTemplate::Mistral => {
                let llama2 = *self == ChatTemplate::Llama2;
                let mut system = String::new();
                let mut first_user = true;
                for m in messages {
                    match m.role.as_str() {
                        "system" => system.push_str(&m.content),
                        "assistant" if llama2 => p.push_str(&format!(" {} </s>", m.content)),
                        "assistant" => p.push_str(&format!("{}</s>", m.content)),
                        _ => {
                            // Llama-2 reopens every later user turn with BOS.
                            if llama2 && !first_user {
                                p.push_str("<s>");
                            }
                            p.push_str("[INST] ");
                            if first_user && !system.is_empty() {
                                if llama2 {
                                    p.push_str(&format!("<<SYS>>\n{system}\n<</SYS>>\n\n"));
                                } else {
                                    p.push_str(&format!("{system}\n\n"));
                                }
                            }
                            p.push_str(&m.content);
                            p.push_str(" [/INST]");
                            first_user = false;
                        }
                    }
                }
            }
            ChatTemplate::Gemma => {
                let mut system = String::new();
                for m in messages {
                    let role = match m.role.as_str() {
                        "system" => {
                            system.push_str(&m.content);
                            system.push_str("\n\n");
                            continue;
                        }
                        "assistant" => "model",
                        other => other,
                    };
                    let prefix = if role == "user" {
                        std::mem::take(&mut system)
                    } else {
                        String::new()
                    };
                    p.push_str(&format!(
                        "<start_of_turn>{role}\n{prefix}{}<end_of_turn>\n",
                        m.gemma_content()
                    ));
                }
                p.push_str("<start_of_turn>model\n");
            }
            ChatTemplate::Gemma4 => p = gemma4(messages),
            ChatTemplate::Phi3 => {
                for m in messages {
                    p.push_str(&format!("<|{}|>\n{}<|end|>\n", m.role, m.content));
                }
                p.push_str("<|assistant|>\n");
            }
            ChatTemplate::DeepSeek => {
                for m in messages {
                    match m.role.as_str() {
                        "system" => p.push_str(&format!("{}\n\n", m.content)),
                        "assistant" => {
                            p.push_str(&format!("Assistant: {}<｜end▁of▁sentence｜>", m.content))
                        }
                        _ => p.push_str(&format!("User: {}\n\n", m.content)),
                    }
                }
                p.push_str("Assistant:");
            }
        }
        p
    }
}

/// Gemma 4's chat template (`chat_template.jinja`) for plain text turns, as
/// transformers renders it with `add_generation_prompt`: `<bos>`, a system turn
/// when the conversation opens with a system (or developer) message, then a
/// `<|turn>role\n…<turn|>\n` per message with the assistant as `model`.
/// Consecutive assistant messages share one turn, and an assistant's earlier
/// thinking (`<|channel>…<channel|>`) is dropped from what it said, as the
/// template's `strip_thinking` does. Tools and thinking mode are not rendered.
fn gemma4(messages: &[ChatMessage]) -> String {
    // Jinja's `trim` is Python's `strip`: whitespace both ends.
    fn strip_thinking(text: &str) -> String {
        let mut out = String::new();
        for part in text.split("<channel|>") {
            out.push_str(part.split("<|channel>").next().unwrap_or(""));
        }
        out.trim().to_string()
    }
    let mut p = String::from("<bos>");
    let mut rest = messages;
    if let Some(first) = messages
        .first()
        .filter(|m| matches!(m.role.as_str(), "system" | "developer"))
    {
        p.push_str("<|turn>system\n");
        p.push_str(first.content.trim());
        p.push_str("<turn|>\n");
        rest = &messages[1..];
    }
    for (i, m) in rest.iter().enumerate() {
        let model = m.role == "assistant";
        let continues = i > 0 && model && rest[i - 1].role == "assistant";
        if !continues {
            let role = if model { "model" } else { m.role.as_str() };
            p.push_str(&format!("<|turn>{role}\n"));
        }
        if model {
            p.push_str(&strip_thinking(&m.content));
        } else {
            p.push_str(m.content.trim());
        }
        let into_next = model && rest.get(i + 1).is_some_and(|n| n.role == "assistant");
        if !into_next {
            p.push_str("<turn|>\n");
        }
    }
    p.push_str("<|turn>model\n");
    p
}

fn error_json(message: &str) -> Vec<u8> {
    format!(r#"{{"error":{{"message":{message:?},"type":"invalid_request_error"}}}}"#).into_bytes()
}

/// Constant-time byte comparison, so a wrong key can't be recovered by timing
/// how long the check takes to fail.
fn secret_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// True if the request carries a matching key via `Authorization: Bearer <key>`
/// (OpenAI style) or `x-api-key: <key>` (Anthropic style). The scheme name is
/// matched case-insensitively, as RFC 7235 requires.
pub(crate) fn authorized(req: &Request, key: &str) -> bool {
    let bearer = req.headers.get("authorization").and_then(|h| {
        let (scheme, value) = h.split_once(' ')?;
        scheme.eq_ignore_ascii_case("bearer").then(|| value.trim())
    });
    let api_key = req.headers.get("x-api-key").map(|h| h.trim());
    bearer.is_some_and(|b| secret_eq(b, key)) || api_key.is_some_and(|k| secret_eq(k, key))
}

/// Paths that stay open when an API key is configured: a liveness probe and the
/// root banner. Everything else — including `/metrics`, which leaks request and
/// token counts, and `/v1/telemetry`, which exposes the engine's internal
/// timings — requires the key.
pub(crate) fn is_public_path(path: &str) -> bool {
    matches!(path, "/" | "/health" | "/healthz")
}

/// How often the telemetry stream flushes a batch. 20 Hz is well under what a
/// UI can render and coalesces a streamed layer's events into one frame instead
/// of one write per event.
const TELEMETRY_TICK: std::time::Duration = std::time::Duration::from_millis(50);

/// Re-send the static snapshot on this cadence, so a consumer that reconnects
/// or drops a frame recovers without a separate request.
const TELEMETRY_SNAPSHOT_EVERY: u32 = 100; // ticks → every 5s

/// Turns telemetry off when the subscriber goes away.
///
/// Collection is global and gated on an atomic; leaving it enabled after a
/// disconnect would keep taxing the streaming path for nobody. Drop runs on the
/// early return from a write error (i.e. the client hanging up), which is the
/// only way this stream normally ends.
struct TelemetryGuard;

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        crate::telemetry::disable();
    }
}

/// `GET /v1/telemetry` — per-layer flow events as Server-Sent Events.
///
/// Two frame types: `snapshot` carries the static shape of the run (layer
/// count, layer sizes, whether weights are streaming at all), and `flow`
/// carries batches of measured stage events. The snapshot goes first so a UI
/// attaching mid-generation can render immediately rather than waiting a full
/// pass to infer the model's shape.
///
/// Enabling collection is a side effect of subscribing: there is no point
/// paying for events nobody reads.
fn telemetry_stream() -> Response {
    if !crate::telemetry::is_available() {
        return Response::json(
            404,
            error_json("telemetry is not enabled (pass --telemetry)"),
        );
    }
    Response::stream(200, "text/event-stream", move |w| {
        crate::telemetry::enable();
        let _guard = TelemetryGuard;

        let mut ticks: u32 = 0;
        loop {
            if ticks % TELEMETRY_SNAPSHOT_EVERY == 0 {
                if let Some(snapshot) = crate::telemetry::snapshot() {
                    let json = serde_json::to_string(&snapshot).unwrap_or_default();
                    write!(w, "event: snapshot\ndata: {json}\n\n")?;
                    w.flush()?;
                }
            }
            ticks = ticks.wrapping_add(1);

            let batch = crate::telemetry::drain(1024);
            if batch.events.is_empty() {
                // A comment frame is a valid SSE keep-alive and is how a
                // disconnected peer is detected: the write fails, `?` unwinds,
                // and the guard turns collection back off.
                write!(w, ": keep-alive\n\n")?;
            } else {
                let json = serde_json::to_string(&batch).unwrap_or_default();
                write!(w, "event: flow\ndata: {json}\n\n")?;
            }
            w.flush()?;
            std::thread::sleep(TELEMETRY_TICK);
        }
    })
}

/// Wrap [`router`] with bearer-token auth: when `api_key` is set, every request
/// except [`is_public_path`] must carry a matching key. With `api_key = None`
/// this is exactly [`router`].
pub fn secured_router(engine: Arc<EngineService>, api_key: Option<String>) -> Handler {
    let inner = router(engine);
    match api_key {
        None => inner,
        Some(key) => Arc::new(move |req: &Request| {
            if !is_public_path(&req.path) && !authorized(req, &key) {
                return Response::json(401, error_json("missing or invalid API key"));
            }
            inner(req)
        }),
    }
}

/// Prometheus text-format dump of the cumulative counters.
fn metrics_text(engine: &EngineService) -> String {
    let m = &engine.metrics;
    let mut out = format!(
        "# HELP dlm_requests_total Completed generation requests.\n\
         # TYPE dlm_requests_total counter\n\
         dlm_requests_total {}\n\
         # HELP dlm_prompt_tokens_total Prompt tokens processed.\n\
         # TYPE dlm_prompt_tokens_total counter\n\
         dlm_prompt_tokens_total {}\n\
         # HELP dlm_completion_tokens_total Tokens generated.\n\
         # TYPE dlm_completion_tokens_total counter\n\
         dlm_completion_tokens_total {}\n",
        m.requests.load(Ordering::Relaxed),
        m.prompt_tokens.load(Ordering::Relaxed),
        m.completion_tokens.load(Ordering::Relaxed),
    );
    // Load, as of the scheduler's latest step: what to alert or scale on.
    let s = &engine.stream;
    out.push_str(&format!(
        "# HELP dlm_requests_running Requests decoding now.\n\
         # TYPE dlm_requests_running gauge\n\
         dlm_requests_running {}\n\
         # HELP dlm_requests_waiting Requests queued for a batch slot.\n\
         # TYPE dlm_requests_waiting gauge\n\
         dlm_requests_waiting {}\n",
        s.running.load(Ordering::Relaxed),
        s.waiting.load(Ordering::Relaxed),
    ));
    let kv_free = s.kv_free.load(Ordering::Relaxed);
    if kv_free != u64::MAX {
        out.push_str(&format!(
            "# HELP dlm_kv_free_tokens KV cache tokens free in the pool.\n\
             # TYPE dlm_kv_free_tokens gauge\n\
             dlm_kv_free_tokens {kv_free}\n"
        ));
    }
    // Layer-streaming cache stats — present only when streaming weights, so a
    // scraper can see the hit rate / prefetch effectiveness and tune the window
    // and --prefetch-depth.
    let s = &engine.stream;
    if s.active.load(Ordering::Relaxed) {
        out.push_str(&format!(
            "# HELP dlm_stream_layer_hits_total Layer fetches served from the resident window.\n\
             # TYPE dlm_stream_layer_hits_total counter\n\
             dlm_stream_layer_hits_total {}\n\
             # HELP dlm_stream_layer_misses_total Layer fetches that blocked on a load.\n\
             # TYPE dlm_stream_layer_misses_total counter\n\
             dlm_stream_layer_misses_total {}\n\
             # HELP dlm_stream_layer_evictions_total Layers evicted from the window.\n\
             # TYPE dlm_stream_layer_evictions_total counter\n\
             dlm_stream_layer_evictions_total {}\n\
             # HELP dlm_stream_layer_prefetched_total Layers loaded ahead by the prefetch worker.\n\
             # TYPE dlm_stream_layer_prefetched_total counter\n\
             dlm_stream_layer_prefetched_total {}\n\
             # HELP dlm_stream_prefetch_depth Layers prefetched ahead (live; auto-tuned under --auto-prefetch).\n\
             # TYPE dlm_stream_prefetch_depth gauge\n\
             dlm_stream_prefetch_depth {}\n",
            s.hits.load(Ordering::Relaxed),
            s.misses.load(Ordering::Relaxed),
            s.evictions.load(Ordering::Relaxed),
            s.prefetched.load(Ordering::Relaxed),
            s.depth.load(Ordering::Relaxed),
        ));
        // Routed-expert VRAM cache (GPU MoE streaming). Zero on dense / host runs;
        // on a sparse model the expert hit rate is the throughput signal.
        let (eh, em) = (
            s.expert_hits.load(Ordering::Relaxed),
            s.expert_misses.load(Ordering::Relaxed),
        );
        if eh + em > 0 {
            out.push_str(&format!(
                "# HELP dlm_stream_expert_hits_total Routed-expert fetches served from the VRAM expert cache.\n\
                 # TYPE dlm_stream_expert_hits_total counter\n\
                 dlm_stream_expert_hits_total {}\n\
                 # HELP dlm_stream_expert_misses_total Routed-expert fetches that streamed the expert from the checkpoint.\n\
                 # TYPE dlm_stream_expert_misses_total counter\n\
                 dlm_stream_expert_misses_total {}\n\
                 # HELP dlm_stream_expert_evictions_total Experts evicted from the VRAM expert cache.\n\
                 # TYPE dlm_stream_expert_evictions_total counter\n\
                 dlm_stream_expert_evictions_total {}\n",
                eh,
                em,
                s.expert_evictions.load(Ordering::Relaxed),
            ));
        }
    }
    out
}

/// Build the HTTP router for the batched/streaming engine.
pub fn router(engine: Arc<EngineService>) -> Handler {
    Arc::new(move |req: &Request| -> Response {
        let started = std::time::Instant::now();
        let resp = match (req.method.as_str(), req.path.as_str()) {
            // `/healthz` is routed as well as exempted: `is_public_path` has always
            // listed it, but nothing served it, so a probe pointed there got a 404
            // and restarted a healthy process.
            ("GET", "/") | ("GET", "/health") | ("GET", "/healthz") => {
                Response::text(200, "dlm: ok")
            }
            ("GET", "/metrics") => Response::text(200, metrics_text(&engine)),
            ("GET", "/v1/telemetry") => telemetry_stream(),
            ("GET", "/v1/models") => {
                let body = ModelsResponse {
                    object: "list",
                    data: vec![ModelCard {
                        id: engine.model_id.clone(),
                        object: "model",
                        created: engine.created,
                        owned_by: "dlm",
                    }],
                };
                Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
            }
            ("POST", "/v1/chat/completions") => handle_chat(&engine, req),
            ("POST", "/v1/completions") => handle_completion(&engine, req),
            ("POST", "/v1/embeddings") => handle_embeddings(&engine, req),
            ("POST", "/v1/messages") => handle_messages(&engine, req),
            ("POST", "/v1/messages/count_tokens") => handle_count_tokens(&engine, req),
            ("GET", _) | ("POST", _) => Response::json(404, error_json("no such endpoint")),
            _ => Response::json(405, error_json("method not allowed")),
        };
        // Access log. For streamed responses this times request setup only; the
        // body is written afterwards.
        eprintln!(
            "{} {} → {} ({}ms)",
            req.method,
            req.path,
            resp.status,
            started.elapsed().as_millis()
        );
        resp
    })
}

/// Why generation ended. Mapped to each API's own reason field.
enum Finish {
    /// Model emitted the EOS token → OpenAI `stop` / Anthropic `end_turn`.
    Eos,
    /// A stop sequence was hit; carries the matched sequence when known.
    Stop(Option<String>),
    /// Hit the `max_tokens` cap → OpenAI `length` / Anthropic `max_tokens`.
    Length,
}

/// Clamp a requested `max_tokens` to the context budget left after the prompt,
/// or return an error message if the prompt already fills the window.
fn fit_max_tokens(
    engine: &EngineService,
    prompt_len: usize,
    requested: usize,
) -> Result<usize, String> {
    if prompt_len >= engine.max_context {
        return Err(format!(
            "prompt is {prompt_len} tokens but the context window is {}",
            engine.max_context
        ));
    }
    let budget = engine.max_context - prompt_len;
    Ok(requested.min(budget).max(1))
}

/// The stop sequence with the earliest occurrence in `text`, if any.
fn matched_stop<'a>(text: &str, stops: &'a [String]) -> Option<&'a String> {
    stops
        .iter()
        .filter_map(|s| text.find(s.as_str()).map(|pos| (pos, s)))
        .min_by_key(|(pos, _)| *pos)
        .map(|(_, s)| s)
}

/// A collected non-streaming completion.
struct Generated {
    text: String,
    finish: Finish,
    prompt_tokens: usize,
    completion_tokens: usize,
    spec: Option<AcceptanceStats>,
    /// Each token of `text` with its log-probabilities and its character
    /// offset in `text`, when the request asked for them.
    logprobs: Vec<(u32, TokenLogprobs, usize)>,
}

/// Run one request to completion, ending at the EOS token, the first stop
/// sequence, or `max_tokens`. Shared by the OpenAI and Anthropic non-streaming
/// handlers.
fn collect_completion(
    engine: &Arc<EngineService>,
    ids: Vec<u32>,
    max_tokens: usize,
    decoding: Decoding,
    stops: &[String],
) -> Result<Generated, String> {
    collect_completions(engine, ids, &[], max_tokens, vec![decoding], stops, 0)
        .map(|mut g| g.remove(0))
}

/// Each of a request's choices with its first event, once every one has
/// one. They are queued together, then waited on: a request the queue turns
/// away (`--queue-timeout`) is answered `503` before any response starts,
/// even when it asked to stream. `Err` is why it was turned away.
type Started = Vec<(TokenEvent, Receiver<TokenEvent>)>;

fn start_choices(
    engine: &EngineService,
    ids: &[u32],
    max_tokens: usize,
    decodings: Vec<Decoding>,
    priority: i32,
) -> Result<Started, String> {
    start_choices_with(engine, ids, &[], max_tokens, decodings, priority)
}

/// [`start_choices`] for a prompt with images (`pixel_values`, and where each
/// one's tokens start).
fn start_choices_with(
    engine: &EngineService,
    ids: &[u32],
    images: &[(usize, Vec<f32>)],
    max_tokens: usize,
    decodings: Vec<Decoding>,
    priority: i32,
) -> Result<Started, String> {
    let rxs: Vec<_> = decodings
        .into_iter()
        .map(|d| {
            engine.submit_at(
                ids.to_vec(),
                images.to_vec(),
                max_tokens,
                engine.eos_tokens.clone(),
                d,
                priority,
            )
        })
        .collect();
    let mut started = Vec::with_capacity(rxs.len());
    for rx in rxs {
        match rx.recv().unwrap_or(TokenEvent::Done(None)) {
            TokenEvent::Rejected(why) => return Err(why),
            first => started.push((first, rx)),
        }
    }
    Ok(started)
}

/// `503` for a request the queue turned away, with `Retry-After`.
fn busy(body: Vec<u8>) -> Response {
    Response::json(503, body).with_header("Retry-After", "1")
}

/// [`collect_completion`] for several choices of one prompt: all are queued
/// before any is read, so they decode in the same batch (and share the prompt's
/// cached prefix). The prompt counts once.
fn collect_completions(
    engine: &Arc<EngineService>,
    ids: Vec<u32>,
    images: &[(usize, Vec<f32>)],
    max_tokens: usize,
    decodings: Vec<Decoding>,
    stops: &[String],
    priority: i32,
) -> Result<Vec<Generated>, String> {
    let prompt_tokens = ids.len();
    let started = start_choices_with(engine, &ids, images, max_tokens, decodings, priority)?;
    let out: Vec<Generated> = started
        .into_iter()
        .map(|(first, rx)| collect_from(engine, first, rx, prompt_tokens, stops))
        .collect();
    engine.record(prompt_tokens, out.iter().map(|g| g.completion_tokens).sum());
    Ok(out)
}

/// Read one choice's tokens to its end (see [`collect_completion`]).
fn collect_from(
    engine: &Arc<EngineService>,
    first: TokenEvent,
    rx: Receiver<TokenEvent>,
    prompt_tokens: usize,
    stops: &[String],
) -> Generated {
    let mut tokens = Vec::new();
    let mut lps = Vec::new();
    let mut spec = None;
    let mut hit_eos = false;
    for ev in std::iter::once(first).chain(rx) {
        match ev {
            TokenEvent::Next(t, lp) => {
                // The scheduler emits EOS inclusively as the final token; drop it
                // from the visible output and mark a natural stop.
                if engine.eos_tokens.contains(&t) {
                    hit_eos = true;
                    break;
                }
                tokens.push(t);
                lps.extend(lp);
                if !stops.is_empty() {
                    let so_far = engine.tokenizer.decode(&tokens).unwrap_or_default();
                    if earliest_stop(&so_far, stops).is_some() {
                        break; // dropping the receiver ends generation
                    }
                }
            }
            TokenEvent::Done(stats) => {
                spec = stats;
                break;
            }
            // Only a queued request is turned away, never one that has started.
            TokenEvent::Rejected(_) => break,
        }
    }
    let completion_tokens = tokens.len();
    let mut text = engine.tokenizer.decode(&tokens).unwrap_or_default();
    let finish = if hit_eos {
        Finish::Eos
    } else {
        match matched_stop(&text, stops) {
            Some(s) => {
                let s = s.clone();
                text.truncate(text.find(&s).unwrap_or(text.len()));
                Finish::Stop(Some(s))
            }
            None => Finish::Length,
        }
    };
    // Each token's offset, keeping those that start inside the returned text
    // (a stop sequence cuts the rest).
    let mut logprobs = Vec::new();
    if lps.len() == tokens.len() {
        for (i, (&t, lp)) in tokens.iter().zip(lps).enumerate() {
            let offset = engine
                .tokenizer
                .decode(&tokens[..i])
                .unwrap_or_default()
                .chars()
                .count();
            if offset >= text.chars().count() && i > 0 {
                break;
            }
            logprobs.push((t, lp, offset));
        }
    }
    Generated {
        text,
        finish,
        prompt_tokens,
        completion_tokens,
        spec,
        logprobs,
    }
}

/// One choice of a streamed response: the tokens so far, the text sent, and
/// how it ended. Shared by the chat and completions streams.
struct Streamed {
    /// `None` once the choice is over; dropping it ends its generation.
    rx: Option<Receiver<TokenEvent>>,
    ids: Vec<u32>,
    /// Characters `ids` decode to.
    chars: usize,
    /// Characters sent.
    sent: usize,
    finish: &'static str,
    spec: Option<AcceptanceStats>,
    /// Tokens with log-probabilities whose text has not gone out, with their
    /// character offsets.
    pending: Vec<(u32, TokenLogprobs, usize)>,
    /// Hold back text from the first tool-call marker on: the request has
    /// tools, and calls go out whole at the end.
    hold_tools: Option<&'static str>,
}

impl Streamed {
    fn new(rx: Receiver<TokenEvent>) -> Self {
        Streamed {
            rx: Some(rx),
            ids: Vec::new(),
            chars: 0,
            sent: 0,
            finish: "length",
            spec: None,
            pending: Vec::new(),
            hold_tools: None,
        }
    }

    /// Take one event: the text it completes, if any.
    ///
    /// Decodes the whole run, never a lone token: byte-level BPE splits one
    /// UTF-8 character (emoji, CJK, accents) across several tokens, and a
    /// fragment decoded alone renders as U+FFFD. Emitting the newly completed
    /// characters is correct with or without stop sequences.
    fn take(&mut self, engine: &EngineService, stops: &[String], ev: TokenEvent) -> Option<String> {
        match ev {
            TokenEvent::Next(t, lp) => {
                // EOS is emitted inclusively; stop without showing it.
                if engine.eos_tokens.contains(&t) {
                    self.finish = "stop";
                    self.rx = None;
                    return None;
                }
                if let Some(lp) = lp {
                    self.pending.push((t, lp, self.chars));
                }
                self.ids.push(t);
                let full = engine.tokenizer.decode(&self.ids).unwrap_or_default();
                self.chars = full.chars().count();
                let cut = earliest_stop(&full, stops);
                let mut end = cut.unwrap_or(full.len());
                if let Some(open) = self.hold_tools {
                    end = match full[..end].find(open) {
                        Some(at) => at,
                        None => end - partial_marker(&full[..end], open),
                    };
                }
                let visible = &full[..end];
                let shown = visible.chars().count();
                let delta = (shown > self.sent).then(|| {
                    let d: String = visible.chars().skip(self.sent).collect();
                    self.sent = shown;
                    d
                });
                if cut.is_some() {
                    self.finish = "stop";
                    self.rx = None;
                }
                delta
            }
            TokenEvent::Done(stats) => {
                self.spec = stats;
                self.rx = None;
                None
            }
            TokenEvent::Rejected(_) => {
                self.rx = None;
                None
            }
        }
    }
}

/// Feed each choice its events until all are over, calling `send(index,
/// choice, text)` for every piece of new text. One choice waits on its
/// receiver; several take turns.
fn stream_choices(
    engine: &EngineService,
    started: Started,
    stops: &[String],
    hold_tools: Option<&'static str>,
    mut send: impl FnMut(usize, &mut Streamed, String) -> std::io::Result<()>,
) -> std::io::Result<Vec<Streamed>> {
    use std::sync::mpsc::RecvTimeoutError;
    let mut choices = Vec::with_capacity(started.len());
    for (i, (first, rx)) in started.into_iter().enumerate() {
        let mut c = Streamed {
            hold_tools,
            ..Streamed::new(rx)
        };
        if let Some(text) = c.take(engine, stops, first) {
            send(i, &mut c, text)?;
        }
        choices.push(c);
    }
    let wait = if choices.len() == 1 {
        std::time::Duration::from_secs(3600)
    } else {
        std::time::Duration::from_millis(1)
    };
    while choices.iter().any(|c| c.rx.is_some()) {
        for (i, c) in choices.iter_mut().enumerate() {
            let Some(rx) = &c.rx else { continue };
            match rx.recv_timeout(wait) {
                Ok(ev) => {
                    if let Some(text) = c.take(engine, stops, ev) {
                        send(i, c, text)?;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => c.rx = None,
            }
        }
    }
    Ok(choices)
}

/// A chat prompt with images: its ids, with each image's tokens laid out as
/// Gemma 3's processor lays them out, and each image's `pixel_values` with
/// where its tokens start. Images must arrive inline, as base64 `data:` URLs:
/// the server fetches nothing on a client's behalf.
#[allow(clippy::type_complexity)]
fn image_prompt(
    engine: &EngineService,
    vision: &crate::vision::Vision,
    messages: &[ChatMessage],
    prompt: &str,
) -> Result<(Vec<u32>, Vec<(usize, Vec<f32>)>), String> {
    use crate::vision::image;
    if engine.chat_template != ChatTemplate::Gemma {
        return Err(format!(
            "images need Gemma's chat template; this server uses {} (--chat-template gemma)",
            engine.chat_template.name()
        ));
    }
    let pixels = messages
        .iter()
        .flat_map(|m| &m.images)
        .map(|(_, url)| {
            let bytes = image::data_url_bytes(url)?;
            let img = image::decode(&bytes)?;
            Ok(image::preprocess(&img, vision.tower.cfg.image_size))
        })
        .collect::<crate::error::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    let (ids, starts) = vision
        .tokens
        .encode(&engine.tokenizer, prompt)
        .map_err(|e| format!("tokenize: {e}"))?;
    if starts.len() != pixels.len() {
        return Err(format!(
            "the prompt has {} image markers for {} images (is <start_of_image> in the text?)",
            starts.len(),
            pixels.len()
        ));
    }
    if ids.iter().any(|&id| id as usize >= engine.vocab_size) {
        return Err("prompt token out of model vocab range".into());
    }
    Ok((ids, starts.into_iter().zip(pixels).collect()))
}

fn handle_chat(engine: &Arc<EngineService>, req: &Request) -> Response {
    let mut parsed: ChatRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    if parsed.messages.is_empty() {
        return Response::json(400, error_json("messages must not be empty"));
    }
    // `tool_choice: none` keeps the tools out of the prompt; a forced call
    // would need the output constrained, which is not done.
    let tools = match parsed.tool_choice.as_ref().map(|c| c.as_str()) {
        None | Some(Some("auto")) => parsed.tools.take().unwrap_or_default(),
        Some(Some("none")) => Vec::new(),
        _ => {
            return Response::json(
                400,
                error_json("tool_choice must be auto or none; dlm does not force a tool call"),
            )
        }
    };
    // Images go to a model with a vision tower; any other drops them, as it
    // always has.
    let vision = engine
        .vision
        .as_deref()
        .filter(|_| parsed.messages.iter().any(|m| !m.images.is_empty()));
    if vision.is_none() {
        for m in &mut parsed.messages {
            m.images.clear();
        }
    }
    let prompt = match engine
        .chat_template
        .apply_with_tools(&parsed.messages, &tools)
    {
        Ok(p) => p,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let (ids, images) = match vision {
        None => match engine.encode_prompt(&prompt) {
            Ok(ids) => (ids, Vec::new()),
            Err(e) => return Response::json(400, error_json(&e)),
        },
        Some(v) => match image_prompt(engine, v, &parsed.messages, &prompt) {
            Ok(p) => p,
            Err(e) => return Response::json(400, error_json(&e)),
        },
    };
    let requested = parsed.max_tokens.unwrap_or(engine.default_max_tokens);
    let max_tokens = match fit_max_tokens(engine, ids.len(), requested) {
        Ok(m) => m,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let sampler = sampler_from_request(&parsed, engine.next_id.load(Ordering::Relaxed));
    let logprobs = match logprobs_request(parsed.logprobs, parsed.top_logprobs) {
        Ok(l) => l,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let decodings = match decoding(engine, sampler, parsed.extra, logprobs)
        .and_then(|d| choices(d, parsed.n))
    {
        Ok(d) => d,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let stops = normalize_stop(parsed.stop);
    let model = parsed.model.unwrap_or_else(|| engine.model_id.clone());

    if parsed.stream {
        let tools = !tools.is_empty();
        let started = match start_choices_with(
            engine,
            &ids,
            &images,
            max_tokens,
            decodings,
            parsed.priority,
        ) {
            Ok(s) => s,
            Err(why) => return busy(error_json(&why)),
        };
        let lp = logprobs.is_some();
        stream_chat(
            Arc::clone(engine),
            ids.len(),
            started,
            lp,
            model,
            stops,
            tools,
        )
    } else {
        let request = engine.next_id.load(Ordering::Relaxed);
        let prompt_tokens = ids.len();
        let gs = match collect_completions(
            engine,
            ids,
            &images,
            max_tokens,
            decodings,
            &stops,
            parsed.priority,
        ) {
            Ok(g) => g,
            Err(why) => return busy(error_json(&why)),
        };
        let completion_tokens = gs.iter().map(|g| g.completion_tokens).sum();
        let spec = if gs.len() == 1 { gs[0].spec } else { None };
        let body = ChatResponse {
            id: format!("chatcmpl-{}", engine.next_id.load(Ordering::Relaxed)),
            object: "chat.completion",
            created: engine.created,
            model,
            choices: gs
                .into_iter()
                .enumerate()
                .map(|(index, g)| {
                    let (content, calls) = if tools.is_empty() {
                        (g.text, Vec::new())
                    } else {
                        split_tool_calls_for(engine.chat_template, &g.text)
                    };
                    Choice {
                        index,
                        finish_reason: match g.finish {
                            _ if !calls.is_empty() => "tool_calls",
                            Finish::Length => "length",
                            _ => "stop",
                        },
                        logprobs: logprobs.map(|_| ChatLogprobs {
                            content: g
                                .logprobs
                                .iter()
                                .map(|(t, lp, _)| chat_logprob(engine, *t, lp))
                                .collect(),
                        }),
                        message: RespMessage {
                            role: "assistant",
                            content: (!content.is_empty() || calls.is_empty()).then_some(content),
                            tool_calls: tool_calls_out(calls, request, false),
                        },
                    }
                })
                .collect(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                speculative: spec.map(SpeculativeUsage::from),
            },
        };
        Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
    }
}

// ── OpenAI embeddings (`POST /v1/embeddings`) ───────────────────────────────

/// `input`: one text or several, each as a string or as token ids.
#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
    Ids(Vec<u32>),
    IdLists(Vec<Vec<u32>>),
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    #[serde(default)]
    model: Option<String>,
    input: EmbeddingInput,
    /// `float` (the default) or `base64`, the little-endian f32 bytes, which
    /// the OpenAI Python SDK asks for unless told otherwise.
    #[serde(default)]
    encoding_format: Option<String>,
    /// Keep the first this many dimensions, scaled back to unit length.
    #[serde(default)]
    dimensions: Option<usize>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    index: usize,
    embedding: EmbeddingVector,
}

#[derive(Serialize)]
struct EmbeddingUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: EmbeddingUsage,
}

/// Standard base64 (RFC 4648, padded).
fn base64(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(ABC[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn handle_embeddings(engine: &Arc<EngineService>, req: &Request) -> Response {
    let parsed: EmbeddingRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    let base64_out = match parsed.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Response::json(
                400,
                error_json(&format!(
                    "encoding_format must be float or base64, not {other}"
                )),
            )
        }
    };
    let texts: Result<Vec<Vec<u32>>, String> = match parsed.input {
        EmbeddingInput::Text(t) => engine.encode_prompt(&t).map(|ids| vec![ids]),
        EmbeddingInput::Texts(ts) => ts.iter().map(|t| engine.encode_prompt(t)).collect(),
        EmbeddingInput::Ids(ids) => Ok(vec![ids]),
        EmbeddingInput::IdLists(lists) => Ok(lists),
    };
    let texts = match texts {
        Ok(t) if !t.is_empty() => t,
        Ok(_) => return Response::json(400, error_json("input must not be empty")),
        Err(e) => return Response::json(400, error_json(&e)),
    };
    for ids in &texts {
        let bad = ids.iter().any(|&t| t as usize >= engine.vocab_size);
        if ids.is_empty() || bad || ids.len() > engine.max_context {
            return Response::json(
                400,
                error_json(&format!(
                    "each input must be 1 to {} tokens of the model's vocabulary",
                    engine.max_context
                )),
            );
        }
    }
    let mut data = Vec::with_capacity(texts.len());
    let mut tokens = 0;
    for (index, ids) in texts.into_iter().enumerate() {
        tokens += ids.len();
        let mut v = match engine.embed(ids) {
            Ok(v) => v,
            Err(e) => return Response::json(500, error_json(&format!("embedding failed: {e}"))),
        };
        if let Some(d) = parsed.dimensions {
            if d == 0 || d > v.len() {
                return Response::json(
                    400,
                    error_json(&format!("dimensions must be between 1 and {}", v.len())),
                );
            }
            v.truncate(d);
            let norm = v
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x = (*x as f64 / norm) as f32);
            }
        }
        let embedding = if base64_out {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            EmbeddingVector::Base64(base64(&bytes))
        } else {
            EmbeddingVector::Float(v)
        };
        data.push(EmbeddingData {
            object: "embedding",
            index,
            embedding,
        });
    }
    engine.record(tokens, 0);
    let body = EmbeddingResponse {
        object: "list",
        data,
        model: parsed.model.unwrap_or_else(|| engine.model_id.clone()),
        usage: EmbeddingUsage {
            prompt_tokens: tokens,
            total_tokens: tokens,
        },
    };
    Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
}

// ── OpenAI legacy completions (`POST /v1/completions`) ──────────────────────

/// Request body for `/v1/completions`. Same sampling knobs as chat, but a raw
/// `prompt` string instead of `messages` — no chat template is applied.
#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Stop>,
    /// Return each token's log-probability and this many alternatives (0-20).
    #[serde(default)]
    logprobs: Option<usize>,
    /// How many completions to generate for the prompt (1-16).
    #[serde(default)]
    n: Option<usize>,
    /// Queue priority: lower is admitted first (vLLM's field). 0 by default.
    #[serde(default)]
    priority: i32,
    #[serde(flatten)]
    extra: ExtraSampling,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: &'static str,
    logprobs: Option<CompletionLogprobs>,
}
#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}
#[derive(Serialize)]
struct CompletionChunkChoice {
    index: usize,
    text: String,
    finish_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<CompletionLogprobs>,
}
#[derive(Serialize)]
struct CompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

fn handle_completion(engine: &Arc<EngineService>, req: &Request) -> Response {
    let parsed: CompletionRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    if parsed.prompt.is_empty() {
        return Response::json(400, error_json("prompt must not be empty"));
    }
    let ids = match engine.encode_prompt(&parsed.prompt) {
        Ok(ids) => ids,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let requested = parsed.max_tokens.unwrap_or(engine.default_max_tokens);
    let max_tokens = match fit_max_tokens(engine, ids.len(), requested) {
        Ok(m) => m,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let id = engine.next_id.load(Ordering::Relaxed);
    let sampler = match parsed.temperature {
        Some(t) if t > 0.0 => Sampler::TopPK {
            temperature: t,
            top_p: parsed.top_p.unwrap_or(1.0),
            top_k: parsed.top_k.unwrap_or(0),
            min_p: parsed.min_p.unwrap_or(0.0),
            repetition_penalty: parsed.repetition_penalty.unwrap_or(1.0),
            seed: parsed.seed.unwrap_or(id),
        },
        _ => Sampler::Greedy,
    };
    let logprobs = match logprobs_request(parsed.logprobs.is_some(), parsed.logprobs) {
        Ok(l) => l,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let decodings = match decoding(engine, sampler, parsed.extra, logprobs)
        .and_then(|d| choices(d, parsed.n))
    {
        Ok(d) => d,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let stops = normalize_stop(parsed.stop);
    let model = parsed.model.unwrap_or_else(|| engine.model_id.clone());

    if parsed.stream {
        let started = match start_choices(engine, &ids, max_tokens, decodings, parsed.priority) {
            Ok(s) => s,
            Err(why) => return busy(error_json(&why)),
        };
        let lp = logprobs.is_some();
        stream_completion(Arc::clone(engine), ids.len(), started, lp, model, stops)
    } else {
        let prompt_tokens = ids.len();
        let gs = match collect_completions(
            engine,
            ids,
            &[],
            max_tokens,
            decodings,
            &stops,
            parsed.priority,
        ) {
            Ok(g) => g,
            Err(why) => return busy(error_json(&why)),
        };
        let completion_tokens = gs.iter().map(|g| g.completion_tokens).sum();
        let spec = if gs.len() == 1 { gs[0].spec } else { None };
        let body = CompletionResponse {
            id: format!("cmpl-{id}"),
            object: "text_completion",
            created: engine.created,
            model,
            choices: gs
                .into_iter()
                .enumerate()
                .map(|(index, g)| CompletionChoice {
                    index,
                    finish_reason: match g.finish {
                        Finish::Length => "length",
                        _ => "stop",
                    },
                    logprobs: logprobs.map(|_| {
                        let mut out = CompletionLogprobs::default();
                        for (t, lp, offset) in &g.logprobs {
                            out.push(engine, *t, lp, *offset);
                        }
                        out
                    }),
                    text: g.text,
                })
                .collect(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                speculative: spec.map(SpeculativeUsage::from),
            },
        };
        Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
    }
}

/// Stream a `/v1/completions` response as SSE `text_completion` chunks, each
/// choice's under its `index`. Mirrors [`stream_chat`] but carries the text in
/// `text` and has no role chunk.
fn stream_completion(
    engine: Arc<EngineService>,
    prompt_tokens: usize,
    started: Started,
    logprobs: bool,
    model: String,
    stops: Vec<String>,
) -> Response {
    Response::stream(200, "text/event-stream", move |w| {
        let id = format!("cmpl-{}", engine.next_id.load(Ordering::Relaxed));
        let created = engine.created;
        let send_chunk = |w: &mut dyn std::io::Write,
                          index: usize,
                          text: String,
                          finish: Option<&'static str>,
                          logprobs: Option<CompletionLogprobs>|
         -> std::io::Result<()> {
            let chunk = CompletionChunk {
                id: id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: vec![CompletionChunkChoice {
                    index,
                    text,
                    finish_reason: finish,
                    logprobs,
                }],
                usage: None,
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            write!(w, "data: {json}\n\n")?;
            w.flush()
        };

        let choices = stream_choices(&engine, started, &stops, None, |i, c, text| {
            let lps = logprobs.then(|| {
                let mut out = CompletionLogprobs::default();
                for (t, lp, offset) in std::mem::take(&mut c.pending) {
                    out.push(&engine, t, &lp, offset);
                }
                out
            });
            send_chunk(w, i, text, None, lps)
        })?;
        let completion_tokens = choices.iter().map(|c| c.ids.len()).sum();
        engine.record(prompt_tokens, completion_tokens);
        for (i, c) in choices.iter().enumerate() {
            send_chunk(w, i, String::new(), Some(c.finish), None)?;
        }

        if let [Streamed {
            spec: Some(stats), ..
        }] = choices.as_slice()
        {
            let chunk = CompletionChunk {
                id: id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    speculative: Some((*stats).into()),
                }),
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            write!(w, "data: {json}\n\n")?;
            w.flush()?;
        }

        write!(w, "data: [DONE]\n\n")?;
        w.flush()
    })
}

// ── Anthropic Messages API shapes (`POST /v1/messages`) ─────────────────────

/// Anthropic message/system content: either a plain string or a list of blocks.
/// Only text blocks carry content we render; other block types contribute their
/// `text` field (usually empty) and are otherwise ignored.
#[derive(Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(default)]
    text: String,
}

impl AnthropicContent {
    fn to_text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(bs) => bs
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Deserialize)]
struct MessagesRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<AnthropicMessage>,
    /// Required by Anthropic; we still default it to the server's max if absent.
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    system: Option<AnthropicContent>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Serialize)]
struct TextBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

#[derive(Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Serialize)]
struct MessagesResponse {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    role: &'static str,
    model: String,
    content: Vec<TextBlock>,
    stop_reason: &'static str,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

/// Anthropic error envelope: `{"type":"error","error":{"type","message"}}`.
fn anthropic_error_json(message: &str) -> Vec<u8> {
    format!(
        r#"{{"type":"error","error":{{"type":"invalid_request_error","message":{message:?}}}}}"#
    )
    .into_bytes()
}

/// Fold Anthropic `system` + `messages` into the rendered prompt string. The
/// `system` field becomes a leading system-role message.
fn anthropic_prompt(
    engine: &EngineService,
    system: &Option<AnthropicContent>,
    msgs: &[AnthropicMessage],
) -> String {
    let mut messages: Vec<ChatMessage> = Vec::with_capacity(msgs.len() + 1);
    if let Some(sys) = system {
        let text = sys.to_text();
        if !text.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: text,
                ..Default::default()
            });
        }
    }
    for m in msgs {
        messages.push(ChatMessage {
            role: m.role.clone(),
            content: m.content.to_text(),
            ..Default::default()
        });
    }
    engine.chat_template.apply(&messages)
}

#[derive(Deserialize)]
struct CountTokensRequest {
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<AnthropicContent>,
}

/// `POST /v1/messages/count_tokens`: report how many input tokens the request
/// would consume, without generating.
fn handle_count_tokens(engine: &Arc<EngineService>, req: &Request) -> Response {
    let parsed: CountTokensRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => {
            return Response::json(400, anthropic_error_json(&format!("invalid request: {e}")))
        }
    };
    if parsed.messages.is_empty() {
        return Response::json(400, anthropic_error_json("messages must not be empty"));
    }
    let prompt = anthropic_prompt(engine, &parsed.system, &parsed.messages);
    match engine.encode_prompt(&prompt) {
        Ok(ids) => {
            let body = serde_json::json!({ "input_tokens": ids.len() });
            Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
        }
        Err(e) => Response::json(400, anthropic_error_json(&e)),
    }
}

fn handle_messages(engine: &Arc<EngineService>, req: &Request) -> Response {
    let parsed: MessagesRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => {
            return Response::json(400, anthropic_error_json(&format!("invalid request: {e}")))
        }
    };
    if parsed.messages.is_empty() {
        return Response::json(400, anthropic_error_json("messages must not be empty"));
    }

    let prompt = anthropic_prompt(engine, &parsed.system, &parsed.messages);
    let ids = match engine.encode_prompt(&prompt) {
        Ok(ids) => ids,
        Err(e) => return Response::json(400, anthropic_error_json(&e)),
    };

    let requested = parsed.max_tokens.unwrap_or(engine.default_max_tokens);
    let max_tokens = match fit_max_tokens(engine, ids.len(), requested) {
        Ok(m) => m,
        Err(e) => return Response::json(400, anthropic_error_json(&e)),
    };
    let id = engine.next_id.load(Ordering::Relaxed);
    let sampler = match parsed.temperature {
        Some(t) if t > 0.0 => Sampler::TopPK {
            temperature: t,
            top_p: parsed.top_p.unwrap_or(1.0),
            top_k: parsed.top_k.unwrap_or(0),
            min_p: parsed.min_p.unwrap_or(0.0),
            repetition_penalty: parsed.repetition_penalty.unwrap_or(1.0),
            seed: id,
        },
        _ => Sampler::Greedy,
    };
    let stops: Vec<String> = parsed
        .stop_sequences
        .unwrap_or_default()
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
    let model = parsed.model.unwrap_or_else(|| engine.model_id.clone());

    if parsed.stream {
        // Queue before the stream opens, so a request `--queue-timeout` turns
        // away gets a real `503` rather than a 200 stream that ends empty.
        let started = match start_choices(engine, &ids, max_tokens, vec![sampler.into()], 0) {
            Ok(s) => s,
            Err(why) => return busy(anthropic_error_json(&why)),
        };
        let (first, rx) = started.into_iter().next().expect("one choice");
        return stream_messages(Arc::clone(engine), ids.len(), first, rx, model, stops);
    }

    let g = match collect_completion(engine, ids, max_tokens, sampler.into(), &stops) {
        Ok(g) => g,
        Err(why) => return busy(anthropic_error_json(&why)),
    };
    let body = MessagesResponse {
        id: format!("msg-{id}"),
        kind: "message",
        role: "assistant",
        model,
        content: vec![TextBlock {
            kind: "text",
            text: g.text,
        }],
        stop_reason: match g.finish {
            Finish::Eos => "end_turn",
            Finish::Stop(_) => "stop_sequence",
            Finish::Length => "max_tokens",
        },
        stop_sequence: match g.finish {
            Finish::Stop(s) => s,
            _ => None,
        },
        usage: AnthropicUsage {
            input_tokens: g.prompt_tokens,
            output_tokens: g.completion_tokens,
        },
    };
    Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
}

/// Stream an Anthropic Messages response as Server-Sent Events, following the
/// event sequence Anthropic clients expect: `message_start`,
/// `content_block_start`, a run of `content_block_delta` (`text_delta`),
/// `content_block_stop`, `message_delta` (with `stop_reason`), `message_stop`.
fn stream_messages(
    engine: Arc<EngineService>,
    prompt_tokens: usize,
    first: TokenEvent,
    rx: Receiver<TokenEvent>,
    model: String,
    stops: Vec<String>,
) -> Response {
    Response::stream(200, "text/event-stream", move |w| {
        let id = format!("msg-{}", engine.next_id.load(Ordering::Relaxed));
        // Anthropic frames carry both an `event:` line and a `data:` line.
        let send = |w: &mut dyn std::io::Write,
                    event: &str,
                    data: serde_json::Value|
         -> std::io::Result<()> {
            write!(w, "event: {event}\ndata: {data}\n\n")?;
            w.flush()
        };

        send(
            w,
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                },
            }),
        )?;
        send(
            w,
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
        )?;
        send(w, "ping", serde_json::json!({"type": "ping"}))?;

        let mut completion_tokens = 0usize;
        // Stop-aware emission mirrors stream_chat: accumulate ids, re-decode, and
        // emit only the newly-visible suffix up to any stop sequence.
        let mut acc_ids: Vec<u32> = Vec::new();
        let mut sent_chars = 0usize;
        let mut stop_reason = "max_tokens";
        let mut stop_sequence: Option<String> = None;
        for ev in std::iter::once(first).chain(rx) {
            match ev {
                TokenEvent::Next(t, _) => {
                    // EOS is emitted inclusively; end the turn without showing it.
                    if engine.eos_tokens.contains(&t) {
                        stop_reason = "end_turn";
                        break;
                    }
                    completion_tokens += 1;
                    acc_ids.push(t);
                    let full = engine.tokenizer.decode(&acc_ids).unwrap_or_default();
                    let cut = if stops.is_empty() {
                        None
                    } else {
                        earliest_stop(&full, &stops)
                    };
                    let visible = &full[..cut.unwrap_or(full.len())];
                    let vis_chars = visible.chars().count();
                    if vis_chars > sent_chars {
                        let delta: String = visible.chars().skip(sent_chars).collect();
                        sent_chars = vis_chars;
                        send(
                            w,
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": delta},
                            }),
                        )?;
                    }
                    if cut.is_some() {
                        stop_reason = "stop_sequence";
                        stop_sequence = matched_stop(&full, &stops).cloned();
                        break;
                    }
                }
                TokenEvent::Done(_) | TokenEvent::Rejected(_) => break,
            }
        }
        engine.record(prompt_tokens, completion_tokens);

        send(
            w,
            "content_block_stop",
            serde_json::json!({
                "type": "content_block_stop",
                "index": 0,
            }),
        )?;
        send(
            w,
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": completion_tokens},
            }),
        )?;
        send(
            w,
            "message_stop",
            serde_json::json!({"type": "message_stop"}),
        )
    })
}

/// Stream a chat completion as SSE `chat.completion.chunk`s, each choice's
/// under its `index`: a role chunk, the text as it completes (with the
/// log-probabilities of the tokens behind it when asked for), and a final
/// chunk with the finish reason.
fn stream_chat(
    engine: Arc<EngineService>,
    prompt_tokens: usize,
    started: Started,
    logprobs: bool,
    model: String,
    stops: Vec<String>,
    tools: bool,
) -> Response {
    Response::stream(200, "text/event-stream", move |w| {
        let id = format!("chatcmpl-{}", engine.next_id.load(Ordering::Relaxed));
        let created = engine.created;
        let send_chunk = |w: &mut dyn std::io::Write,
                          index: usize,
                          delta: Delta,
                          finish: Option<&'static str>,
                          logprobs: Option<ChatLogprobs>|
         -> std::io::Result<()> {
            let chunk = Chunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index,
                    delta,
                    finish_reason: finish,
                    logprobs,
                }],
                usage: None,
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            write!(w, "data: {json}\n\n")?;
            w.flush()
        };

        // First, each choice announces the assistant role.
        for i in 0..started.len() {
            let role = Delta {
                role: Some("assistant"),
                content: None,
                tool_calls: None,
            };
            send_chunk(w, i, role, None, None)?;
        }

        let request = engine.next_id.load(Ordering::Relaxed);
        let hold = tools.then(|| engine.chat_template.tool_open());
        let mut choices = stream_choices(&engine, started, &stops, hold, |i, c, text| {
            let lps = logprobs.then(|| ChatLogprobs {
                content: std::mem::take(&mut c.pending)
                    .iter()
                    .map(|(t, lp, _)| chat_logprob(&engine, *t, lp))
                    .collect(),
            });
            let delta = Delta {
                role: None,
                content: Some(text),
                tool_calls: None,
            };
            send_chunk(w, i, delta, None, lps)
        })?;
        let completion_tokens = choices.iter().map(|c| c.ids.len()).sum();
        engine.record(prompt_tokens, completion_tokens);
        // What was held back: the calls, whole, or (when the blocks held no
        // call) the rest of the text after all.
        if tools {
            for (i, c) in choices.iter_mut().enumerate() {
                let full = engine.tokenizer.decode(&c.ids).unwrap_or_default();
                let full = &full[..earliest_stop(&full, &stops).unwrap_or(full.len())];
                let (_, calls) = split_tool_calls_for(engine.chat_template, full);
                let delta = if calls.is_empty() {
                    let rest: String = full.chars().skip(c.sent).collect();
                    if rest.is_empty() {
                        continue;
                    }
                    Delta {
                        role: None,
                        content: Some(rest),
                        tool_calls: None,
                    }
                } else {
                    c.finish = "tool_calls";
                    Delta {
                        role: None,
                        content: None,
                        tool_calls: Some(tool_calls_out(calls, request, true)),
                    }
                };
                send_chunk(w, i, delta, None, None)?;
            }
        }
        for (i, c) in choices.iter().enumerate() {
            let end = Delta {
                role: None,
                content: None,
                tool_calls: None,
            };
            send_chunk(w, i, end, Some(c.finish), None)?;
        }

        // When speculating, a final usage-only chunk carries the acceptance
        // stats (OpenAI-style: empty choices + a `usage` object).
        if let [Streamed {
            spec: Some(stats), ..
        }] = choices.as_slice()
        {
            let chunk = Chunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    speculative: Some((*stats).into()),
                }),
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            write!(w, "data: {json}\n\n")?;
            w.flush()?;
        }

        write!(w, "data: [DONE]\n\n")?;
        w.flush()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEATHER_TOOL: &str = r#"{"type": "function", "function": {
        "description": "Get the weather in a city", "name": "get_weather",
        "parameters": {"properties": {"city": {"description": "City name", "type": "string"}},
                       "required": ["city"], "type": "object"}}}"#;

    /// Llama 3.2's own Jinja template (rendered by `transformers` with
    /// `date_string="17 Nov 2026"`) gave the expected string, from the request
    /// as an OpenAI client sends it.
    #[test]
    fn tools_render_as_llama3s_template_does() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"}]}"#,
        )
        .unwrap();
        let tools = [serde_json::from_str(WEATHER_TOOL).unwrap()];
        let got = llama3_tools(&req.messages, &tools, "17 Nov 2026").unwrap();
        let want = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: 17 Nov 2026\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.\n\n{\n    \"type\": \"function\",\n    \"function\": {\n        \"description\": \"Get the weather in a city\",\n        \"name\": \"get_weather\",\n        \"parameters\": {\n            \"properties\": {\n                \"city\": {\n                    \"description\": \"City name\",\n                    \"type\": \"string\"\n                }\n            },\n            \"required\": [\n                \"city\"\n            ],\n            \"type\": \"object\"\n        }\n    }\n}\n\nWeather in Paris?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n\"18C and sunny\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        assert_eq!(got, want);
    }

    /// Mistral v0.3's own Jinja template (rendered by `transformers`) gave the
    /// expected string. Its 9-character call id passes through unchanged.
    #[test]
    fn tools_render_as_mistrals_template_does() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "abc123XYZ", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "abc123XYZ", "content": "18C and sunny"},
                {"role": "assistant", "content": "It is 18C and sunny."},
                {"role": "user", "content": "And Rome?"}]}"#,
        )
        .unwrap();
        let tools = [serde_json::from_str(WEATHER_TOOL).unwrap()];
        let got = mistral_tools(&req.messages, &tools).unwrap();
        let want = "[INST] Weather in Paris?[/INST][TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}, \"id\": \"abc123XYZ\"}]</s>[TOOL_RESULTS] {\"content\": 18C and sunny, \"call_id\": \"abc123XYZ\"}[/TOOL_RESULTS] It is 18C and sunny.</s>[AVAILABLE_TOOLS] [{\"type\": \"function\", \"function\": {\"description\": \"Get the weather in a city\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"description\": \"City name\", \"type\": \"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}}][/AVAILABLE_TOOLS][INST] Be brief.\n\nAnd Rome?[/INST]";
        assert_eq!(got, want);
        // An OpenAI-style id maps to 9 alphanumerics, the same for the call and
        // the result that names it.
        let id = mistral_call_id(Some("call_17_0"), 1);
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_eq!(id, mistral_call_id(Some("call_17_0"), 5));
    }

    /// Each format's calls come back out of a reply; text before the call is
    /// kept, and a call that does not parse leaves the reply as text.
    #[test]
    fn tool_calls_parse_per_template() {
        let llama = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";
        assert_eq!(
            split_tool_calls_for(ChatTemplate::Llama3, llama),
            (
                String::new(),
                vec![("get_weather".into(), "{\"city\":\"Paris\"}".into())]
            )
        );
        let mistral = "Sure.[TOOL_CALLS] [{\"name\": \"a\", \"arguments\": {\"x\": 1}}, {\"name\": \"b\", \"arguments\": {}}]";
        let (text, calls) = split_tool_calls_for(ChatTemplate::Mistral, mistral);
        assert_eq!(text, "Sure.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].0, "b");
        let broken = "[TOOL_CALLS] [{\"name\": ";
        assert_eq!(
            split_tool_calls_for(ChatTemplate::Mistral, broken),
            (broken.to_string(), vec![])
        );
        assert_eq!(
            split_tool_calls_for(ChatTemplate::Llama3, "Paris is sunny."),
            ("Paris is sunny.".to_string(), vec![])
        );
    }

    /// Tools render exactly as Qwen3's own Jinja template renders them
    /// (`transformers`' `apply_chat_template` on Qwen3-0.6B gave the string
    /// below), from the request as an OpenAI client sends it: arguments as a
    /// JSON string, consecutive tool results.
    #[test]
    fn tools_render_as_qwens_template_does() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
                    {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}}]},
                {"role": "tool", "tool_call_id": "a", "content": "18C"},
                {"role": "tool", "tool_call_id": "b", "content": "24C"}],
              "tools": [{"type": "function", "function": {"name": "get_weather",
                "description": "Weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}]}"#,
        )
        .unwrap();
        let tools = req.tools.unwrap();
        let want: String = serde_json::from_str(r#""<|im_start|>system\nYou are terse.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWeather in Paris and Rome?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C\n</tool_response>\n<tool_response>\n24C\n</tool_response><|im_end|>\n<|im_start|>assistant\n""#).unwrap();
        assert_eq!(
            ChatTemplate::ChatMl
                .apply_with_tools(&req.messages, &tools)
                .unwrap(),
            want
        );
        assert!(ChatTemplate::Llama3
            .apply_with_tools(&req.messages, &tools)
            .is_err());
        // Without tools, ChatML is as it was.
        assert_eq!(
            ChatTemplate::ChatMl
                .apply_with_tools(&req.messages[..2], &[])
                .unwrap(),
            "<|im_start|>system
You are terse.<|im_end|>
<|im_start|>user
Weather in Paris and Rome?<|im_end|>
<|im_start|>assistant
"
        );
    }

    #[test]
    fn tool_calls_are_read_out_of_the_reply() {
        let reply = "Checking.
<tool_call>
{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}
</tool_call>
<tool_call>
{\"name\": \"now\"}
</tool_call>";
        let (text, calls) = split_tool_calls(reply);
        assert_eq!(text, "Checking.");
        assert_eq!(
            calls,
            vec![
                ("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string()),
                ("now".to_string(), "{}".to_string())
            ]
        );
        // No call, or a block that holds none: all text.
        assert_eq!(split_tool_calls("plain"), ("plain".to_string(), vec![]));
        let broken = "x <tool_call>not json</tool_call>";
        assert_eq!(split_tool_calls(broken), (broken.to_string(), vec![]));
        // A call cut off at the length limit is not a call.
        assert!(split_tool_calls(
            "<tool_call>
{\"name\": \"f\", \"argu"
        )
        .1
        .is_empty());
    }

    #[test]
    fn a_stream_holds_back_what_may_start_a_tool_call() {
        assert_eq!(partial_marker("Sure <tool", TOOL_OPEN), 5);
        assert_eq!(partial_marker("Sure <", TOOL_OPEN), 1);
        assert_eq!(partial_marker("Sure", TOOL_OPEN), 0);
        assert_eq!(partial_marker("a <b", TOOL_OPEN), 0);
    }

    /// RFC 4648's own examples, every padding case.
    #[test]
    fn base64_matches_the_rfc() {
        for (raw, enc) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(raw.as_bytes()), enc, "{raw:?}");
        }
    }

    /// OpenAI's `content` arrives as a string from some clients and as a list of
    /// parts from others — the SDKs and anything that can attach an image. Both
    /// have to reach the model as the same text; refusing the array form turns a
    /// working client into a 400.
    #[test]
    fn content_is_read_as_a_string_or_as_parts() {
        let parse = |json: &str| -> String {
            serde_json::from_str::<ChatMessage>(json)
                .unwrap_or_else(|e| panic!("{json}: {e}"))
                .content
        };
        assert_eq!(parse(r#"{"role":"user","content":"hi"}"#), "hi");
        assert_eq!(
            parse(r#"{"role":"user","content":[{"type":"text","text":"hi"}]}"#),
            "hi"
        );
        // Several parts join in order, the way the client laid them out.
        assert_eq!(
            parse(
                r#"{"role":"user","content":[{"type":"text","text":"a"},
                    {"type":"text","text":"b"}]}"#
            ),
            "ab"
        );
        // A part dlm cannot serve is dropped, not refused: the rest of the
        // conversation is still worth answering.
        assert_eq!(
            parse(
                r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}},
                    {"type":"text","text":"what is this"}]}"#
            ),
            "what is this"
        );
        // `content: null` is what a tool-call reply carries.
        assert_eq!(parse(r#"{"role":"assistant","content":null}"#), "");
    }

    /// Qwen3.6's own Jinja template (`transformers`' `apply_chat_template` on
    /// Qwen/Qwen3.6-35B-A3B, `add_generation_prompt=True`) gave these strings:
    /// contents trimmed, an earlier answer's thinking dropped, thinking kept on
    /// turns after the last real user query, and the reply opening `<think>`.
    #[test]
    fn qwen35_renders_as_its_template_does() {
        let msgs = |json: &str| -> Vec<ChatMessage> { serde_json::from_str(json).unwrap() };
        let cases = [
            (
                r#"[{"role":"user","content":"What is 2+2?"}]"#,
                "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n",
            ),
            (
                r#"[{"role":"system","content":"  You are terse.\n"},
                    {"role":"user","content":"Hi there "},
                    {"role":"assistant","content":"<think>\nThe user greets.\n</think>\n\nHello!"},
                    {"role":"user","content":"Name a color."}]"#,
                "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nName a color.<|im_end|>\n<|im_start|>assistant\n<think>\n",
            ),
            (
                r#"[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1"},
                    {"role":"user","content":"<tool_response>\nx\n</tool_response>"},
                    {"role":"assistant","content":"<think>\nhmm\n</think>\n\nA2"}]"#,
                "<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA1<|im_end|>\n<|im_start|>user\n<tool_response>\nx\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\nhmm\n</think>\n\nA2<|im_end|>\n<|im_start|>assistant\n<think>\n",
            ),
        ];
        for (json, want) in cases {
            assert_eq!(ChatTemplate::Qwen35.apply(&msgs(json)), want);
        }
        assert_eq!(ChatTemplate::parse("qwen3.5"), Some(ChatTemplate::Qwen35));
        assert_eq!(ChatTemplate::Qwen35.end_of_turn(), Some("<|im_end|>"));
        // Its tool format (Qwen3-Coder's XML) is not one dlm renders yet.
        let tool: serde_json::Value = serde_json::from_str(WEATHER_TOOL).unwrap();
        assert!(ChatTemplate::Qwen35
            .apply_with_tools(&msgs(cases[0].0), &[tool])
            .is_err());
    }

    #[test]
    fn chat_template_parse_and_render() {
        assert_eq!(ChatTemplate::parse("chatml"), Some(ChatTemplate::ChatMl));
        assert_eq!(ChatTemplate::parse("llama-3"), Some(ChatTemplate::Llama3));
        assert_eq!(ChatTemplate::parse("plain"), Some(ChatTemplate::Plain));
        assert!(ChatTemplate::parse("bogus").is_none());

        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];

        let cm = ChatTemplate::ChatMl.apply(&msgs);
        assert!(cm.contains("<|im_start|>user\nhi<|im_end|>"), "{cm}");
        assert!(cm.ends_with("<|im_start|>assistant\n"), "{cm}");

        let l3 = ChatTemplate::Llama3.apply(&msgs);
        assert!(l3.starts_with("<|begin_of_text|>"), "{l3}");
        assert!(
            l3.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"),
            "{l3}"
        );
        assert!(
            l3.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "{l3}"
        );
    }

    /// Every name round-trips, and each template renders a system + two-turn
    /// conversation the way its model's Jinja template does.
    #[test]
    fn chat_templates_render_system_and_multi_turn() {
        for name in ChatTemplate::NAMES.split(", ") {
            assert_eq!(ChatTemplate::parse(name).unwrap().name(), name);
        }
        let msg = |role: &str, content: &str| ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        };
        let convo = vec![
            msg("system", "Be brief."),
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "bye"),
        ];
        assert_eq!(
            ChatTemplate::Gemma.apply(&convo),
            "<start_of_turn>user\nBe brief.\n\nhi<end_of_turn>\n\
             <start_of_turn>model\nhello<end_of_turn>\n\
             <start_of_turn>user\nbye<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(
            ChatTemplate::Mistral.apply(&convo),
            "[INST] Be brief.\n\nhi [/INST]hello</s>[INST] bye [/INST]"
        );
        assert_eq!(
            ChatTemplate::Llama2.apply(&convo),
            "[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nhi [/INST] hello </s><s>[INST] bye [/INST]"
        );
        assert_eq!(
            ChatTemplate::Phi3.apply(&convo),
            "<|system|>\nBe brief.<|end|>\n<|user|>\nhi<|end|>\n\
             <|assistant|>\nhello<|end|>\n<|user|>\nbye<|end|>\n<|assistant|>\n"
        );
        assert_eq!(
            ChatTemplate::DeepSeek.apply(&convo),
            "Be brief.\n\nUser: hi\n\nAssistant: hello<｜end▁of▁sentence｜>User: bye\n\nAssistant:"
        );
    }

    /// Gemma 4's template as transformers' `apply_chat_template` renders it
    /// (`add_generation_prompt=True`) for google/gemma-4-E4B-it's
    /// `chat_template.jinja`: these strings are its output, not hand-written.
    #[test]
    fn gemma4_template_matches_transformers() {
        let msg = |role: &str, content: &str| ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        };
        let cases = [
            (
                vec![msg("user", "Hi there!")],
                "<bos><|turn>user\nHi there!<turn|>\n<|turn>model\n",
            ),
            (
                vec![msg("system", "  Be brief. "), msg("user", " What is 2+2? ")],
                "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n\
                 <|turn>model\n",
            ),
            (
                vec![
                    msg("system", "S"),
                    msg("user", "a"),
                    msg("assistant", "<|channel>thought\nhmm<channel|> b "),
                    msg("user", "c"),
                ],
                "<bos><|turn>system\nS<turn|>\n<|turn>user\na<turn|>\n<|turn>model\nb<turn|>\n\
                 <|turn>user\nc<turn|>\n<|turn>model\n",
            ),
            (
                vec![
                    msg("user", "a"),
                    msg("assistant", "b"),
                    msg("assistant", "c"),
                    msg("user", "d"),
                ],
                "<bos><|turn>user\na<turn|>\n<|turn>model\nbc<turn|>\n<|turn>user\nd<turn|>\n\
                 <|turn>model\n",
            ),
        ];
        for (convo, want) in cases {
            assert_eq!(ChatTemplate::Gemma4.apply(&convo), want);
        }
        assert_eq!(ChatTemplate::Gemma4.end_of_turn(), Some("<turn|>"));
    }

    #[test]
    fn chat_template_detection_by_marker() {
        let cases = [
            ("{{'<|im_start|>' + role}}", Some(ChatTemplate::ChatMl)),
            ("{{'<|start_header_id|>'}}", Some(ChatTemplate::Llama3)),
            ("{{'<start_of_turn>' + role}}", Some(ChatTemplate::Gemma)),
            (
                "{{- '<|turn>' + role + '\\n' }}",
                Some(ChatTemplate::Gemma4),
            ),
            ("{{'[INST] <<SYS>>\\n'}}", Some(ChatTemplate::Llama2)),
            ("{{'[INST] ' + content}}", Some(ChatTemplate::Mistral)),
            (
                "{{'<|user|>\\n' + c + '<|end|>'}}",
                Some(ChatTemplate::Phi3),
            ),
            (
                "{{ 'User: ' + c }}{{ 'Assistant:' }}",
                Some(ChatTemplate::DeepSeek),
            ),
            ("{{'<|im_start|>user<|im_sep|>'}}", None),
            ("{{ something_else }}", None),
        ];
        for (jinja, want) in cases {
            assert_eq!(ChatTemplate::detect(jinja), want, "{jinja}");
        }
    }

    #[test]
    fn stop_helpers() {
        assert_eq!(normalize_stop(None), Vec::<String>::new());
        assert_eq!(
            normalize_stop(Some(Stop::One("x".into()))),
            vec!["x".to_string()]
        );
        assert_eq!(
            normalize_stop(Some(Stop::Many(vec!["a".into(), "".into()]))),
            vec!["a".to_string()]
        );
        assert_eq!(
            earliest_stop("abcXYdef", &["XY".into(), "de".into()]),
            Some(3)
        );
        assert_eq!(earliest_stop("abc", &["z".into()]), None);
    }

    #[test]
    fn anthropic_content_string_or_blocks() {
        let s: AnthropicContent = serde_json::from_str(r#""hello""#).unwrap();
        assert_eq!(s.to_text(), "hello");

        let b: AnthropicContent =
            serde_json::from_str(r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#)
                .unwrap();
        assert_eq!(b.to_text(), "ab");
    }

    #[test]
    fn anthropic_request_folds_system_and_messages() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":16,"system":"be terse",
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(req.max_tokens, Some(16));

        let mut messages = Vec::new();
        if let Some(sys) = &req.system {
            messages.push(ChatMessage {
                role: "system".into(),
                content: sys.to_text(),
                ..Default::default()
            });
        }
        for m in &req.messages {
            messages.push(ChatMessage {
                role: m.role.clone(),
                content: m.content.to_text(),
                ..Default::default()
            });
        }
        let prompt = ChatTemplate::Plain.apply(&messages);
        assert!(
            prompt.starts_with("system: be terse\nuser: hi\n"),
            "{prompt}"
        );
        assert!(prompt.ends_with("assistant:"), "{prompt}");
    }
}

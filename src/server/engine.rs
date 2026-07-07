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
use crate::generate::{Generator, Sampler};
use crate::server::http::{Handler, Request, Response};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// A token event streamed back for one request. `Done` carries the request's
/// speculative acceptance stats when it was decoded speculatively (`None` on the
/// plain path).
pub enum TokenEvent {
    Next(u32),
    Done(Option<AcceptanceStats>),
}

struct Job {
    id: u64,
    prompt: Vec<u32>,
    max_new: usize,
    eos: Vec<u32>,
    sampler: Sampler,
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
}

impl StreamMetrics {
    fn publish(&self, s: crate::forward::StreamStats) {
        self.active.store(true, Ordering::Relaxed);
        self.hits.store(s.hits, Ordering::Relaxed);
        self.misses.store(s.misses, Ordering::Relaxed);
        self.evictions.store(s.evictions, Ordering::Relaxed);
        self.prefetched.store(s.prefetched, Ordering::Relaxed);
        self.depth.store(s.depth as u64, Ordering::Relaxed);
    }
}

/// A handle to the background batching engine.
pub struct EngineService {
    job_tx: Mutex<Sender<Job>>,
    next_id: AtomicU64,
    tokenizer: BpeTokenizer,
    vocab_size: usize,
    model_id: String,
    created: u64,
    default_max_tokens: usize,
    chat_template: ChatTemplate,
    /// Stop generation when any of these token ids is produced (the model's
    /// EOS set). Empty means run to `max_tokens`.
    eos_tokens: Vec<u32>,
    /// Maximum context (prompt + generated) the model/KV cache supports.
    /// Requests are clamped to fit; `usize::MAX` disables the guard.
    max_context: usize,
    metrics: Metrics,
    /// Layer-streaming stats, shared with the engine loop that publishes them.
    stream: Arc<StreamMetrics>,
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
            None,
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
            Some(draft),
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

    #[allow(clippy::too_many_arguments)]
    fn start_inner<K: ComputeKernel + Send + 'static>(
        generator: Generator<K>,
        draft: Option<Generator<K>>,
        gamma: usize,
        tokenizer: BpeTokenizer,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
        max_batch: usize,
        prefix_cache_size: usize,
    ) -> Arc<Self> {
        let (job_tx, job_rx) = channel::<Job>();
        let stream = Arc::new(StreamMetrics::default());
        let stream_loop = Arc::clone(&stream);
        std::thread::spawn(move || {
            engine_loop(generator, draft, gamma, job_rx, max_batch, prefix_cache_size, stream_loop)
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
            eos_tokens: Vec::new(),
            max_context: usize::MAX,
            metrics: Metrics::default(),
            stream,
        })
    }

    /// Record a completed generation into the `/metrics` counters.
    fn record(&self, prompt_tokens: usize, completion_tokens: usize) {
        self.metrics.requests.fetch_add(1, Ordering::Relaxed);
        self.metrics.prompt_tokens.fetch_add(prompt_tokens as u64, Ordering::Relaxed);
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

    /// Submit a request; returns a channel that yields its tokens then `Done`.
    fn submit(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Vec<u32>,
        sampler: Sampler,
    ) -> Receiver<TokenEvent> {
        let (sink, out) = channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let _ = self.job_tx.lock().unwrap().send(Job {
            id,
            prompt,
            max_new,
            eos,
            sampler,
            sink,
        });
        out
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

#[allow(clippy::too_many_arguments)]
fn engine_loop<K: ComputeKernel>(
    generator: Generator<K>,
    draft: Option<Generator<K>>,
    gamma: usize,
    job_rx: Receiver<Job>,
    max_batch: usize,
    prefix_cache_size: usize,
    stream: Arc<StreamMetrics>,
) {
    let mut sched = match &draft {
        // Speculative sessions can't resume, so the prefix cache is plain-path only.
        Some(d) => BatchScheduler::with_speculative(&generator, d, max_batch, gamma),
        None => BatchScheduler::new(&generator, max_batch).with_prefix_cache(prefix_cache_size),
    };
    let mut sinks: HashMap<u64, Sender<TokenEvent>> = HashMap::new();

    loop {
        if !sched.has_work() {
            match job_rx.recv() {
                Ok(job) => enqueue(&mut sched, &mut sinks, job),
                Err(_) => break, // all handles dropped
            }
        }
        while let Ok(job) = job_rx.try_recv() {
            enqueue(&mut sched, &mut sinks, job);
        }
        if !sched.has_work() {
            continue;
        }
        match sched.step() {
            Ok(tick) => {
                for (id, token) in tick.produced {
                    if let Some(s) = sinks.get(&id) {
                        // A failed send means the client dropped the receiver
                        // (disconnected or finished early); stop wasting compute
                        // on it by retiring the slot.
                        if s.send(TokenEvent::Next(token)).is_err() {
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
                for (_, s) in sinks.drain() {
                    let _ = s.send(TokenEvent::Done(None));
                }
            }
        }
        // Publish the streaming kernel's latest cache stats (no-op / None for
        // resident kernels). Cheap: one stats read per token step.
        if let Some(s) = generator.stream_stats() {
            stream.publish(s);
        }
    }
}

fn enqueue<K: ComputeKernel>(
    sched: &mut BatchScheduler<K>,
    sinks: &mut HashMap<u64, Sender<TokenEvent>>,
    job: Job,
) {
    match sched.submit_sampled(job.id, job.prompt, job.max_new, job.eos, job.sampler) {
        Ok(()) => {
            sinks.insert(job.id, job.sink);
        }
        Err(_) => {
            let _ = job.sink.send(TokenEvent::Done(None));
        }
    }
}

// ── OpenAI request/response shapes ──────────────────────────────────────────

/// A chat message (`role` + `content`), shared by the local and distributed
/// serving paths.
#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
    content: String,
}
#[derive(Serialize)]
struct Choice {
    index: usize,
    message: RespMessage,
    finish_reason: &'static str,
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

/// Speculative-decoding acceptance, reported inside `usage` when a draft model
/// served the request.
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
}
#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<&'static str>,
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
}

impl ChatTemplate {
    /// Parse a `--chat-template` value.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "plain" => Some(ChatTemplate::Plain),
            "chatml" => Some(ChatTemplate::ChatMl),
            "llama3" | "llama-3" => Some(ChatTemplate::Llama3),
            _ => None,
        }
    }

    /// Render `messages` into the prompt string for this template.
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
            ChatTemplate::ChatMl => {
                for m in messages {
                    p.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
                }
                p.push_str("<|im_start|>assistant\n");
            }
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
        }
        p
    }
}

fn error_json(message: &str) -> Vec<u8> {
    format!(r#"{{"error":{{"message":{message:?},"type":"invalid_request_error"}}}}"#).into_bytes()
}

/// True if the request carries a matching key via `Authorization: Bearer <key>`
/// (OpenAI style) or `x-api-key: <key>` (Anthropic style).
fn authorized(req: &Request, key: &str) -> bool {
    let bearer = req
        .headers
        .get("authorization")
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(str::trim);
    let api_key = req.headers.get("x-api-key").map(|h| h.trim());
    bearer == Some(key) || api_key == Some(key)
}

/// Wrap [`router`] with bearer-token auth: when `api_key` is set, `/v1/*`
/// requests must carry a matching `Authorization: Bearer` header (health and
/// root stay open). With `api_key = None` this is exactly [`router`].
pub fn secured_router(engine: Arc<EngineService>, api_key: Option<String>) -> Handler {
    let inner = router(engine);
    match api_key {
        None => inner,
        Some(key) => Arc::new(move |req: &Request| {
            if req.path.starts_with("/v1/") && !authorized(req, &key) {
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
    }
    out
}

/// Build the HTTP router for the batched/streaming engine.
pub fn router(engine: Arc<EngineService>) -> Handler {
    Arc::new(move |req: &Request| -> Response {
        let started = std::time::Instant::now();
        let resp = match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") | ("GET", "/health") => Response::text(200, "dlm: ok"),
            ("GET", "/metrics") => Response::text(200, metrics_text(&engine)),
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
fn fit_max_tokens(engine: &EngineService, prompt_len: usize, requested: usize) -> Result<usize, String> {
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
}

/// Run one request to completion, ending at the EOS token, the first stop
/// sequence, or `max_tokens`. Shared by the OpenAI and Anthropic non-streaming
/// handlers.
fn collect_completion(
    engine: &Arc<EngineService>,
    ids: Vec<u32>,
    max_tokens: usize,
    sampler: Sampler,
    stops: &[String],
) -> Generated {
    let prompt_tokens = ids.len();
    let rx = engine.submit(ids, max_tokens, engine.eos_tokens.clone(), sampler);
    let mut tokens = Vec::new();
    let mut spec = None;
    let mut hit_eos = false;
    for ev in rx {
        match ev {
            TokenEvent::Next(t) => {
                // The scheduler emits EOS inclusively as the final token; drop it
                // from the visible output and mark a natural stop.
                if engine.eos_tokens.contains(&t) {
                    hit_eos = true;
                    break;
                }
                tokens.push(t);
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
    engine.record(prompt_tokens, completion_tokens);
    Generated {
        text,
        finish,
        prompt_tokens,
        completion_tokens,
        spec,
    }
}

fn handle_chat(engine: &Arc<EngineService>, req: &Request) -> Response {
    let parsed: ChatRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    if parsed.messages.is_empty() {
        return Response::json(400, error_json("messages must not be empty"));
    }
    let prompt = engine.chat_template.apply(&parsed.messages);
    let ids = match engine.encode_prompt(&prompt) {
        Ok(ids) => ids,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let requested = parsed.max_tokens.unwrap_or(engine.default_max_tokens);
    let max_tokens = match fit_max_tokens(engine, ids.len(), requested) {
        Ok(m) => m,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let sampler = sampler_from_request(&parsed, engine.next_id.load(Ordering::Relaxed));
    let stops = normalize_stop(parsed.stop);
    let model = parsed.model.unwrap_or_else(|| engine.model_id.clone());

    if parsed.stream {
        stream_chat(Arc::clone(engine), ids, max_tokens, model, sampler, stops)
    } else {
        let g = collect_completion(engine, ids, max_tokens, sampler, &stops);
        let body = ChatResponse {
            id: format!("chatcmpl-{}", engine.next_id.load(Ordering::Relaxed)),
            object: "chat.completion",
            created: engine.created,
            model,
            choices: vec![Choice {
                index: 0,
                message: RespMessage { role: "assistant", content: g.text },
                finish_reason: match g.finish {
                    Finish::Length => "length",
                    _ => "stop",
                },
            }],
            usage: Usage {
                prompt_tokens: g.prompt_tokens,
                completion_tokens: g.completion_tokens,
                total_tokens: g.prompt_tokens + g.completion_tokens,
                speculative: g.spec.map(SpeculativeUsage::from),
            },
        };
        Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
    }
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
            AnthropicContent::Blocks(bs) => {
                bs.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join("")
            }
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
            messages.push(ChatMessage { role: "system".into(), content: text });
        }
    }
    for m in msgs {
        messages.push(ChatMessage { role: m.role.clone(), content: m.content.to_text() });
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
        Err(e) => return Response::json(400, anthropic_error_json(&format!("invalid request: {e}"))),
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
        Err(e) => return Response::json(400, anthropic_error_json(&format!("invalid request: {e}"))),
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
        return stream_messages(Arc::clone(engine), ids, max_tokens, model, sampler, stops);
    }

    let g = collect_completion(engine, ids, max_tokens, sampler, &stops);
    let body = MessagesResponse {
        id: format!("msg-{id}"),
        kind: "message",
        role: "assistant",
        model,
        content: vec![TextBlock { kind: "text", text: g.text }],
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
    ids: Vec<u32>,
    max_tokens: usize,
    model: String,
    sampler: Sampler,
    stops: Vec<String>,
) -> Response {
    Response::stream(200, "text/event-stream", move |w| {
        let id = format!("msg-{}", engine.next_id.load(Ordering::Relaxed));
        let prompt_tokens = ids.len();
        // Anthropic frames carry both an `event:` line and a `data:` line.
        let send = |w: &mut dyn std::io::Write, event: &str, data: serde_json::Value| -> std::io::Result<()> {
            write!(w, "event: {event}\ndata: {data}\n\n")?;
            w.flush()
        };

        send(w, "message_start", serde_json::json!({
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
        }))?;
        send(w, "content_block_start", serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }))?;
        send(w, "ping", serde_json::json!({"type": "ping"}))?;

        let mut completion_tokens = 0usize;
        // Stop-aware emission mirrors stream_chat: accumulate ids, re-decode, and
        // emit only the newly-visible suffix up to any stop sequence.
        let mut acc_ids: Vec<u32> = Vec::new();
        let mut sent_chars = 0usize;
        let mut stop_reason = "max_tokens";
        let mut stop_sequence: Option<String> = None;
        let rx = engine.submit(ids, max_tokens, engine.eos_tokens.clone(), sampler);
        for ev in rx {
            match ev {
                TokenEvent::Next(t) => {
                    // EOS is emitted inclusively; end the turn without showing it.
                    if engine.eos_tokens.contains(&t) {
                        stop_reason = "end_turn";
                        break;
                    }
                    completion_tokens += 1;
                    acc_ids.push(t);
                    let full = engine.tokenizer.decode(&acc_ids).unwrap_or_default();
                    let cut = if stops.is_empty() { None } else { earliest_stop(&full, &stops) };
                    let visible = &full[..cut.unwrap_or(full.len())];
                    let vis_chars = visible.chars().count();
                    if vis_chars > sent_chars {
                        let delta: String = visible.chars().skip(sent_chars).collect();
                        sent_chars = vis_chars;
                        send(w, "content_block_delta", serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": delta},
                        }))?;
                    }
                    if cut.is_some() {
                        stop_reason = "stop_sequence";
                        stop_sequence = matched_stop(&full, &stops).cloned();
                        break;
                    }
                }
                TokenEvent::Done(_) => break,
            }
        }
        engine.record(prompt_tokens, completion_tokens);

        send(w, "content_block_stop", serde_json::json!({
            "type": "content_block_stop",
            "index": 0,
        }))?;
        send(w, "message_delta", serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": completion_tokens},
        }))?;
        send(w, "message_stop", serde_json::json!({"type": "message_stop"}))
    })
}

/// Stream a chat completion as Server-Sent Events.
fn stream_chat(
    engine: Arc<EngineService>,
    ids: Vec<u32>,
    max_tokens: usize,
    model: String,
    sampler: Sampler,
    stops: Vec<String>,
) -> Response {
    Response::stream(200, "text/event-stream", move |w| {
        let id = format!("chatcmpl-{}", engine.next_id.load(Ordering::Relaxed));
        let created = engine.created;
        let send_chunk = |w: &mut dyn std::io::Write, delta: Delta, finish: Option<&'static str>| -> std::io::Result<()> {
            let chunk = Chunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice { index: 0, delta, finish_reason: finish }],
                usage: None,
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            write!(w, "data: {json}\n\n")?;
            w.flush()
        };

        // First chunk announces the assistant role.
        send_chunk(w, Delta { role: Some("assistant"), content: None }, None)?;

        let prompt_tokens = ids.len();
        let mut completion_tokens = 0usize;
        let mut spec = None;
        // Only needed with stop sequences: accumulate ids and re-decode so a stop
        // spanning multiple tokens is detected and the visible text truncated.
        let mut acc_ids: Vec<u32> = Vec::new();
        let mut sent_chars = 0usize;
        let mut finish = "length";
        let rx = engine.submit(ids, max_tokens, engine.eos_tokens.clone(), sampler);
        for ev in rx {
            match ev {
                TokenEvent::Next(t) => {
                    // EOS is emitted inclusively; stop without showing it.
                    if engine.eos_tokens.contains(&t) {
                        finish = "stop";
                        break;
                    }
                    completion_tokens += 1;
                    if stops.is_empty() {
                        // Fast path: emit each token's piece directly.
                        let piece = engine.tokenizer.decode(&[t]).unwrap_or_default();
                        send_chunk(w, Delta { role: None, content: Some(piece) }, None)?;
                    } else {
                        // Stop-aware path: emit the newly-visible suffix up to any
                        // stop sequence, then finish.
                        acc_ids.push(t);
                        let full = engine.tokenizer.decode(&acc_ids).unwrap_or_default();
                        let cut = earliest_stop(&full, &stops);
                        let visible = &full[..cut.unwrap_or(full.len())];
                        let vis_chars = visible.chars().count();
                        if vis_chars > sent_chars {
                            let delta: String = visible.chars().skip(sent_chars).collect();
                            sent_chars = vis_chars;
                            send_chunk(w, Delta { role: None, content: Some(delta) }, None)?;
                        }
                        if cut.is_some() {
                            finish = "stop";
                            break;
                        }
                    }
                }
                TokenEvent::Done(stats) => {
                    spec = stats;
                    break;
                }
            }
        }
        engine.record(prompt_tokens, completion_tokens);

        send_chunk(w, Delta { role: None, content: None }, Some(finish))?;

        // When speculating, a final usage-only chunk carries the acceptance
        // stats (OpenAI-style: empty choices + a `usage` object).
        if let Some(stats) = spec {
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
                    speculative: Some(stats.into()),
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

    #[test]
    fn chat_template_parse_and_render() {
        assert_eq!(ChatTemplate::parse("chatml"), Some(ChatTemplate::ChatMl));
        assert_eq!(ChatTemplate::parse("llama-3"), Some(ChatTemplate::Llama3));
        assert_eq!(ChatTemplate::parse("plain"), Some(ChatTemplate::Plain));
        assert!(ChatTemplate::parse("bogus").is_none());

        let msgs = vec![ChatMessage { role: "user".into(), content: "hi".into() }];

        let cm = ChatTemplate::ChatMl.apply(&msgs);
        assert!(cm.contains("<|im_start|>user\nhi<|im_end|>"), "{cm}");
        assert!(cm.ends_with("<|im_start|>assistant\n"), "{cm}");

        let l3 = ChatTemplate::Llama3.apply(&msgs);
        assert!(l3.starts_with("<|begin_of_text|>"), "{l3}");
        assert!(l3.contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"), "{l3}");
        assert!(l3.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"), "{l3}");
    }

    #[test]
    fn stop_helpers() {
        assert_eq!(normalize_stop(None), Vec::<String>::new());
        assert_eq!(normalize_stop(Some(Stop::One("x".into()))), vec!["x".to_string()]);
        assert_eq!(
            normalize_stop(Some(Stop::Many(vec!["a".into(), "".into()]))),
            vec!["a".to_string()]
        );
        assert_eq!(earliest_stop("abcXYdef", &["XY".into(), "de".into()]), Some(3));
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
            messages.push(ChatMessage { role: "system".into(), content: sys.to_text() });
        }
        for m in &req.messages {
            messages.push(ChatMessage { role: m.role.clone(), content: m.content.to_text() });
        }
        let prompt = ChatTemplate::Plain.apply(&messages);
        assert!(prompt.starts_with("system: be terse\nuser: hi\n"), "{prompt}");
        assert!(prompt.ends_with("assistant:"), "{prompt}");
    }
}

//! Batched, streaming inference service behind the HTTP API.
//!
//! An [`EngineService`] runs a background thread that owns the generator and a
//! [`BatchScheduler`](crate::batching::BatchScheduler): connection threads
//! `submit` requests and receive a channel of tokens as they are produced.
//! Concurrent requests are therefore **continuously batched** (interleaved a
//! token at a time), and a request can be **streamed** to the client as SSE.
//!
//! The [`router`] wires this into the OpenAI endpoints, supporting `stream: true`
//! for `/v1/chat/completions`.

use crate::batching::{AcceptanceStats, BatchScheduler};
use crate::forward::ComputeKernel;
use crate::generate::{Generator, Sampler};
use crate::server::http::{Handler, Request, Response};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
    eos: Option<u32>,
    sampler: Sampler,
    sink: Sender<TokenEvent>,
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
    ) -> Arc<Self> {
        let (job_tx, job_rx) = channel::<Job>();
        std::thread::spawn(move || engine_loop(generator, draft, gamma, job_rx, max_batch));
        Arc::new(Self {
            job_tx: Mutex::new(job_tx),
            next_id: AtomicU64::new(1),
            tokenizer,
            vocab_size,
            model_id: model_id.into(),
            created,
            default_max_tokens,
            chat_template: ChatTemplate::default(),
        })
    }

    /// Set the chat template (called right after [`start`](Self::start), while
    /// the `Arc` is still unique). Defaults to [`ChatTemplate::Plain`].
    pub fn with_chat_template(mut self: Arc<Self>, template: ChatTemplate) -> Arc<Self> {
        if let Some(inner) = Arc::get_mut(&mut self) {
            inner.chat_template = template;
        }
        self
    }

    /// Submit a request; returns a channel that yields its tokens then `Done`.
    fn submit(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        eos: Option<u32>,
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

fn engine_loop<K: ComputeKernel>(
    generator: Generator<K>,
    draft: Option<Generator<K>>,
    gamma: usize,
    job_rx: Receiver<Job>,
    max_batch: usize,
) {
    let mut sched = match &draft {
        Some(d) => BatchScheduler::with_speculative(&generator, d, max_batch, gamma),
        None => BatchScheduler::new(&generator, max_batch),
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
                        let _ = s.send(TokenEvent::Next(token));
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

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
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
    fn apply(&self, messages: &[ChatMessage]) -> String {
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

/// True if the request carries `Authorization: Bearer <key>` matching `key`.
fn authorized(req: &Request, key: &str) -> bool {
    req.headers
        .get("authorization")
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(|t| t.trim() == key)
        .unwrap_or(false)
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

/// Build the HTTP router for the batched/streaming engine.
pub fn router(engine: Arc<EngineService>) -> Handler {
    Arc::new(move |req: &Request| -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") | ("GET", "/health") => Response::text(200, "flip: ok"),
            ("GET", "/v1/models") => {
                let body = ModelsResponse {
                    object: "list",
                    data: vec![ModelCard {
                        id: engine.model_id.clone(),
                        object: "model",
                        created: engine.created,
                        owned_by: "flip",
                    }],
                };
                Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
            }
            ("POST", "/v1/chat/completions") => handle_chat(&engine, req),
            ("GET", _) | ("POST", _) => Response::json(404, error_json("no such endpoint")),
            _ => Response::json(405, error_json("method not allowed")),
        }
    })
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
    let max_tokens = parsed.max_tokens.unwrap_or(engine.default_max_tokens).max(1);
    let sampler = sampler_from_request(&parsed, engine.next_id.load(Ordering::Relaxed));
    let stops = normalize_stop(parsed.stop);
    let model = parsed.model.unwrap_or_else(|| engine.model_id.clone());
    let prompt_tokens = ids.len();

    if parsed.stream {
        stream_chat(Arc::clone(engine), ids, max_tokens, model, sampler, stops)
    } else {
        // Collect the completion, ending early at the first stop sequence.
        let rx = engine.submit(ids, max_tokens, None, sampler);
        let mut tokens = Vec::new();
        let mut spec = None;
        for ev in rx {
            match ev {
                TokenEvent::Next(t) => {
                    tokens.push(t);
                    if !stops.is_empty() {
                        let so_far = engine.tokenizer.decode(&tokens).unwrap_or_default();
                        if earliest_stop(&so_far, &stops).is_some() {
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
        // Truncate the decoded text before the stop sequence, if one hit.
        let mut text = engine.tokenizer.decode(&tokens).unwrap_or_default();
        let finish_reason = match earliest_stop(&text, &stops) {
            Some(cut) => {
                text.truncate(cut);
                "stop"
            }
            None => "length",
        };
        let body = ChatResponse {
            id: format!("chatcmpl-{}", engine.next_id.load(Ordering::Relaxed)),
            object: "chat.completion",
            created: engine.created,
            model,
            choices: vec![Choice {
                index: 0,
                message: RespMessage { role: "assistant", content: text },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: tokens.len(),
                total_tokens: prompt_tokens + tokens.len(),
                speculative: spec.map(SpeculativeUsage::from),
            },
        };
        Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
    }
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
        let rx = engine.submit(ids, max_tokens, None, sampler);
        for ev in rx {
            match ev {
                TokenEvent::Next(t) => {
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
}

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

use crate::batching::BatchScheduler;
use crate::forward::ComputeKernel;
use crate::generate::Generator;
use crate::server::http::{Handler, Request, Response};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// A token event streamed back for one request.
pub enum TokenEvent {
    Next(u32),
    Done,
}

struct Job {
    id: u64,
    prompt: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
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
        let (job_tx, job_rx) = channel::<Job>();
        std::thread::spawn(move || engine_loop(generator, job_rx, max_batch));
        Arc::new(Self {
            job_tx: Mutex::new(job_tx),
            next_id: AtomicU64::new(1),
            tokenizer,
            vocab_size,
            model_id: model_id.into(),
            created,
            default_max_tokens,
        })
    }

    /// Submit a request; returns a channel that yields its tokens then `Done`.
    fn submit(&self, prompt: Vec<u32>, max_new: usize, eos: Option<u32>) -> Receiver<TokenEvent> {
        let (sink, out) = channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let _ = self.job_tx.lock().unwrap().send(Job {
            id,
            prompt,
            max_new,
            eos,
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

fn engine_loop<K: ComputeKernel>(generator: Generator<K>, job_rx: Receiver<Job>, max_batch: usize) {
    let mut sched = BatchScheduler::new(&generator, max_batch);
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
                        let _ = s.send(TokenEvent::Done);
                    }
                }
            }
            Err(_) => {
                for (_, s) in sinks.drain() {
                    let _ = s.send(TokenEvent::Done);
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
    match sched.submit(job.id, job.prompt, job.max_new, job.eos) {
        Ok(()) => {
            sinks.insert(job.id, job.sink);
        }
        Err(_) => {
            let _ = job.sink.send(TokenEvent::Done);
        }
    }
}

// ── OpenAI request/response shapes ──────────────────────────────────────────

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
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

fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut p = String::new();
    for m in messages {
        p.push_str(&m.role);
        p.push_str(": ");
        p.push_str(&m.content);
        p.push('\n');
    }
    p.push_str("assistant:");
    p
}

fn error_json(message: &str) -> Vec<u8> {
    format!(r#"{{"error":{{"message":{message:?},"type":"invalid_request_error"}}}}"#).into_bytes()
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
    let prompt = build_chat_prompt(&parsed.messages);
    let ids = match engine.encode_prompt(&prompt) {
        Ok(ids) => ids,
        Err(e) => return Response::json(400, error_json(&e)),
    };
    let max_tokens = parsed.max_tokens.unwrap_or(engine.default_max_tokens).max(1);
    let model = parsed.model.unwrap_or_else(|| engine.model_id.clone());
    let prompt_tokens = ids.len();

    if parsed.stream {
        stream_chat(Arc::clone(engine), ids, max_tokens, model)
    } else {
        // Collect the whole completion, then return it.
        let rx = engine.submit(ids, max_tokens, None);
        let mut tokens = Vec::new();
        for ev in rx {
            match ev {
                TokenEvent::Next(t) => tokens.push(t),
                TokenEvent::Done => break,
            }
        }
        let text = engine.tokenizer.decode(&tokens).unwrap_or_default();
        let body = ChatResponse {
            id: format!("chatcmpl-{}", engine.next_id.load(Ordering::Relaxed)),
            object: "chat.completion",
            created: engine.created,
            model,
            choices: vec![Choice {
                index: 0,
                message: RespMessage { role: "assistant", content: text },
                finish_reason: "length",
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: tokens.len(),
                total_tokens: prompt_tokens + tokens.len(),
            },
        };
        Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
    }
}

/// Stream a chat completion as Server-Sent Events.
fn stream_chat(engine: Arc<EngineService>, ids: Vec<u32>, max_tokens: usize, model: String) -> Response {
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
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            write!(w, "data: {json}\n\n")?;
            w.flush()
        };

        // First chunk announces the assistant role.
        send_chunk(w, Delta { role: Some("assistant"), content: None }, None)?;

        let rx = engine.submit(ids, max_tokens, None);
        for ev in rx {
            match ev {
                TokenEvent::Next(t) => {
                    let piece = engine.tokenizer.decode(&[t]).unwrap_or_default();
                    send_chunk(w, Delta { role: None, content: Some(piece) }, None)?;
                }
                TokenEvent::Done => break,
            }
        }

        send_chunk(w, Delta { role: None, content: None }, Some("length"))?;
        write!(w, "data: [DONE]\n\n")?;
        w.flush()
    })
}

//! OpenAI-compatible API surface (`PRD.md` §3.1): `/v1/chat/completions`,
//! `/v1/completions`, `/v1/models`.
//!
//! Wraps a [`Generator`] + [`BpeTokenizer`] into an [`Engine`] and exposes a
//! [`router`] that plugs into the [`HttpServer`](super::http::HttpServer). The
//! request/response shapes match the OpenAI schema closely enough for clients
//! like Open WebUI to talk to `dlm` unchanged.

use crate::error::{DlmError, Result};
use crate::forward::ComputeKernel;
use crate::generate::{GenerationConfig, Generator, Sampler};
use crate::server::http::{Handler, Request, Response};
use crate::tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A chat message (`role` + `content`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: usize,
    text: String,
    finish_reason: String,
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

/// A completed generation with token accounting.
pub struct Completion {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// The inference engine behind the API: a generator + tokenizer + defaults.
pub struct Engine<K: ComputeKernel> {
    generator: Generator<K>,
    tokenizer: BpeTokenizer,
    model_id: String,
    default_max_tokens: usize,
    created: u64,
    next_id: AtomicU64,
}

impl<K: ComputeKernel> Engine<K> {
    /// Build an engine. `created` is a fixed unix timestamp reported in
    /// responses (pass the server start time).
    pub fn new(
        generator: Generator<K>,
        tokenizer: BpeTokenizer,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        created: u64,
    ) -> Self {
        Self {
            generator,
            tokenizer,
            model_id: model_id.into(),
            default_max_tokens,
            created,
            next_id: AtomicU64::new(1),
        }
    }

    /// The served model id.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    fn new_id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Tokenize `prompt`, generate up to `max_tokens`, and detokenize.
    pub fn complete(&self, prompt: &str, max_tokens: usize) -> Result<Completion> {
        let ids = self.tokenizer.encode(prompt)?;
        if ids.is_empty() {
            return Err(DlmError::Tokenizer("prompt encodes to no tokens".into()));
        }
        let cfg = GenerationConfig {
            max_new_tokens: max_tokens.max(1),
            eos_token: None,
            sampler: Sampler::Greedy,
        };
        let generated = self.generator.generate(&ids, &cfg)?;
        Ok(Completion {
            text: self.tokenizer.decode(&generated)?,
            prompt_tokens: ids.len(),
            completion_tokens: generated.len(),
        })
    }
}

/// Concatenate chat messages into a single prompt (a simple, model-agnostic
/// template; real chat models use their own).
fn build_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        prompt.push_str(&m.role);
        prompt.push_str(": ");
        prompt.push_str(&m.content);
        prompt.push('\n');
    }
    prompt.push_str("assistant:");
    prompt
}

fn error_json(message: &str) -> Vec<u8> {
    format!(r#"{{"error":{{"message":{:?},"type":"invalid_request_error"}}}}"#, message)
        .into_bytes()
}

/// Build an HTTP [`Handler`] serving the OpenAI endpoints from `engine`.
pub fn router<K>(engine: Arc<Engine<K>>) -> Handler
where
    K: ComputeKernel + Send + Sync + 'static,
{
    Arc::new(move |req: &Request| -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") | ("GET", "/health") => Response::text(200, "dlm: ok"),
            ("GET", "/v1/models") => {
                let body = ModelsResponse {
                    object: "list",
                    data: vec![ModelCard {
                        id: engine.model_id().to_string(),
                        object: "model",
                        created: engine.created,
                        owned_by: "dlm",
                    }],
                };
                Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
            }
            ("POST", "/v1/chat/completions") => handle_chat(&engine, req),
            ("POST", "/v1/completions") => handle_completion(&engine, req),
            ("GET", _) | ("POST", _) => Response::json(404, error_json("no such endpoint")),
            _ => Response::json(405, error_json("method not allowed")),
        }
    })
}

fn handle_chat<K: ComputeKernel>(engine: &Engine<K>, req: &Request) -> Response {
    let parsed: ChatRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    if parsed.messages.is_empty() {
        return Response::json(400, error_json("messages must not be empty"));
    }
    let prompt = build_chat_prompt(&parsed.messages);
    let max_tokens = parsed.max_tokens.unwrap_or(engine.default_max_tokens);

    match engine.complete(&prompt, max_tokens) {
        Ok(c) => {
            let body = ChatResponse {
                id: engine.new_id("chatcmpl"),
                object: "chat.completion",
                created: engine.created,
                model: parsed.model.unwrap_or_else(|| engine.model_id().to_string()),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: c.text,
                    },
                    finish_reason: "length".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: c.prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    total_tokens: c.prompt_tokens + c.completion_tokens,
                },
            };
            Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
        }
        Err(e) => Response::json(500, error_json(&e.to_string())),
    }
}

fn handle_completion<K: ComputeKernel>(engine: &Engine<K>, req: &Request) -> Response {
    let parsed: CompletionRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    let max_tokens = parsed.max_tokens.unwrap_or(engine.default_max_tokens);

    match engine.complete(&parsed.prompt, max_tokens) {
        Ok(c) => {
            let body = CompletionResponse {
                id: engine.new_id("cmpl"),
                object: "text_completion",
                created: engine.created,
                model: parsed.model.unwrap_or_else(|| engine.model_id().to_string()),
                choices: vec![CompletionChoice {
                    index: 0,
                    text: c.text,
                    finish_reason: "length".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: c.prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    total_tokens: c.prompt_tokens + c.completion_tokens,
                },
            };
            Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
        }
        Err(e) => Response::json(500, error_json(&e.to_string())),
    }
}

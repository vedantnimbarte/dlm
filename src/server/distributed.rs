//! Distributed (multi-node) serving: a minimal OpenAI-compatible endpoint
//! backed by a pipeline-parallel [`Coordinator`], which streams the hidden
//! state through layer shards hosted on worker nodes.
//!
//! This path deliberately trades features for reach: it is **non-streaming**,
//! **greedy**, and **serialized** (the coordinator owns one KV history and
//! drives a single sequence at a time). The local batched engine
//! ([`super::engine`]) keeps streaming/sampling/continuous-batching; this exists
//! so a model too big for one node can still be served across several.

use crate::distributed::Coordinator;
use crate::server::engine::{authorized, is_public_path, ChatMessage, ChatTemplate};
use crate::server::http::{Handler, Request, Response};
use crate::tokenizer::BpeTokenizer;
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

/// A distributed inference engine: a [`Coordinator`] plus the tokenizer and
/// chat template used to turn requests into prompts. The coordinator is behind a
/// `Mutex` because it holds a single sequence's KV and must serve one request at
/// a time.
pub struct DistributedEngine {
    coordinator: Mutex<Coordinator>,
    tokenizer: BpeTokenizer,
    template: ChatTemplate,
    vocab_size: usize,
    model_id: String,
    default_max_tokens: usize,
    max_context: usize,
    created: u64,
    next_id: AtomicU64,
}

impl DistributedEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coordinator: Coordinator,
        tokenizer: BpeTokenizer,
        template: ChatTemplate,
        vocab_size: usize,
        model_id: impl Into<String>,
        default_max_tokens: usize,
        max_context: usize,
        created: u64,
    ) -> Self {
        Self {
            coordinator: Mutex::new(coordinator),
            tokenizer,
            template,
            vocab_size,
            model_id: model_id.into(),
            default_max_tokens,
            max_context: max_context.max(1),
            created,
            next_id: AtomicU64::new(1),
        }
    }
}

fn error_json(message: &str) -> Vec<u8> {
    format!(r#"{{"error":{{"message":{message:?},"type":"invalid_request_error"}}}}"#).into_bytes()
}

/// Wrap [`router`] with bearer-token auth, mirroring the batched engine: when
/// `api_key` is set, every request except a liveness probe must carry a matching
/// key. With `api_key = None` this is exactly [`router`] (open — trusted network
/// only). The batched path's `--api-key` used to be silently ignored here.
pub fn secured_router(engine: Arc<DistributedEngine>, api_key: Option<String>) -> Handler {
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

/// Build the HTTP router for the distributed engine (`/health`, `/v1/models`,
/// `POST /v1/chat/completions`).
pub fn router(engine: Arc<DistributedEngine>) -> Handler {
    Arc::new(move |req: &Request| -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            // `/healthz` too — `is_public_path` exempts it on both routers, so both
            // must actually serve it or a probe gets an unauthenticated 404.
            ("GET", "/") | ("GET", "/health") | ("GET", "/healthz") => {
                Response::text(200, "dlm: ok (distributed)")
            }
            ("GET", "/v1/models") => {
                let body = serde_json::json!({
                    "object": "list",
                    "data": [{
                        "id": engine.model_id,
                        "object": "model",
                        "created": engine.created,
                        "owned_by": "dlm",
                    }],
                });
                Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
            }
            ("POST", "/v1/chat/completions") => handle_chat(&engine, req),
            ("GET", _) | ("POST", _) => Response::json(404, error_json("no such endpoint")),
            _ => Response::json(405, error_json("method not allowed")),
        }
    })
}

fn handle_chat(engine: &Arc<DistributedEngine>, req: &Request) -> Response {
    let parsed: ChatRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => return Response::json(400, error_json(&format!("invalid request: {e}"))),
    };
    if parsed.messages.is_empty() {
        return Response::json(400, error_json("messages must not be empty"));
    }
    let prompt = engine.template.apply(&parsed.messages);
    let ids = match engine.tokenizer.encode(&prompt) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => return Response::json(400, error_json("prompt encodes to no tokens")),
        Err(e) => return Response::json(400, error_json(&format!("tokenize: {e}"))),
    };
    if ids.iter().any(|&t| t as usize >= engine.vocab_size) {
        return Response::json(400, error_json("prompt token out of model vocab range"));
    }
    // Clamp to the context window: reject an over-long prompt and cap generation
    // to the remaining budget, so an attacker-supplied `max_tokens` can't force a
    // huge allocation or an unbounded serialized generation loop.
    if ids.len() >= engine.max_context {
        return Response::json(
            400,
            error_json(&format!(
                "prompt is {} tokens but the context window is {}",
                ids.len(),
                engine.max_context
            )),
        );
    }
    let budget = engine.max_context - ids.len();
    let max_tokens = parsed
        .max_tokens
        .unwrap_or(engine.default_max_tokens)
        .min(budget)
        .max(1);

    // The coordinator drives one sequence at a time; serialize on it. Recover a
    // poisoned lock so one panicked request can't wedge the whole server.
    let generated = match engine
        .coordinator
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .generate(&ids, max_tokens)
    {
        Ok(g) => g,
        Err(e) => return Response::json(500, error_json(&e.to_string())),
    };
    let text = engine.tokenizer.decode(&generated).unwrap_or_default();
    let id = engine.next_id.fetch_add(1, Ordering::Relaxed);
    let body = serde_json::json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion",
        "created": engine.created,
        "model": parsed.model.unwrap_or_else(|| engine.model_id.clone()),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": ids.len(),
            "completion_tokens": generated.len(),
            "total_tokens": ids.len() + generated.len(),
        },
    });
    Response::json(200, serde_json::to_vec(&body).unwrap_or_default())
}

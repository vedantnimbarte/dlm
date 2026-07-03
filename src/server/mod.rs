//! Local serving layer: a minimal HTTP server and the OpenAI-compatible API.

pub mod engine;
pub mod http;
pub mod openai;

pub use engine::EngineService;
pub use http::{Body, HttpServer, Request, Response};
pub use openai::{Engine, router, ChatMessage, Completion};

//! Local serving layer: a minimal HTTP server and the OpenAI-compatible API.

pub mod http;
pub mod openai;

pub use http::{HttpServer, Request, Response};
pub use openai::{Engine, router, ChatMessage, Completion};

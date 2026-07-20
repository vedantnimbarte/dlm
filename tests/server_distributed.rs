//! The distributed (Master-mode) HTTP serving path: a Coordinator-backed
//! OpenAI-compatible endpoint. Uses local-fallback routes (no worker_addr) so
//! the test exercises the HTTP layer + coordinator without real workers — the
//! networked pipeline itself is covered by tests/distributed.rs.

use dlm::forward::Weights;
use dlm::distributed::{partition_layers, Coordinator, ShardRoute};
use dlm::forward::{BlockConfig, ExpertFfn, Ffn, LayerTensors};
use dlm::server::engine::ChatTemplate;
use dlm::server::{distributed, DistributedEngine, HttpServer};
use dlm::tokenizer::BpeTokenizer;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * 0.1 - 0.05).collect()
    }
}

/// A small model with vocab 256 so a byte tokenizer's ids are always in range.
fn coordinator() -> Coordinator {
    let (vocab, hidden, num_layers) = (256usize, 16usize, 2usize);
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5, rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(),
    };
    let mut r = Rng::new(9);
    let layers: Vec<LayerTensors> = (0..num_layers)
        .map(|_| LayerTensors {
            q_proj: Weights::from_f32(r.vec(cfg.q_dim() * hidden)),
            k_proj: Weights::from_f32(r.vec(cfg.kv_dim() * hidden)),
            v_proj: Weights::from_f32(r.vec(cfg.kv_dim() * hidden)),
            o_proj: Weights::from_f32(r.vec(hidden * cfg.q_dim())),
            ffn: Ffn::Dense(ExpertFfn { gate: Weights::from_f32(r.vec(cfg.intermediate_size * hidden)), up: Weights::from_f32(r.vec(cfg.intermediate_size * hidden)), down: Weights::from_f32(r.vec(hidden * cfg.intermediate_size)) }),
            input_layernorm: vec![1.0; hidden],
            post_attention_layernorm: vec![1.0; hidden], ..Default::default()
        })
        .collect();

    // Two shards, both local (no worker_addr) — the coordinator runs them itself.
    let routes: Vec<ShardRoute> = partition_layers(num_layers, 2)
        .into_iter()
        .map(|shard| ShardRoute { shard, worker_addr: None })
        .collect();

    Coordinator::new(
        cfg,
        layers,
        r.vec(vocab * hidden),
        vec![1.0; hidden],
        r.vec(vocab * hidden),
        vocab,
        routes,
    )
    .unwrap()
}

/// Small context window (32 tokens) so the max_tokens clamp is easy to trip.
const MAX_CONTEXT: usize = 32;

fn start_server(api_key: Option<&str>) -> SocketAddr {
    let engine = Arc::new(DistributedEngine::new(
        coordinator(),
        BpeTokenizer::bytes_only(),
        ChatTemplate::Plain,
        256,
        "dlm-dist",
        16,
        MAX_CONTEXT,
        0,
    ));
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    let router = distributed::secured_router(engine, api_key.map(String::from));
    std::thread::spawn(move || server.serve(router).unwrap());
    addr
}

fn post(addr: SocketAddr, path: &str, body: &str) -> String {
    post_auth(addr, path, body, None)
}

fn post_auth(addr: SocketAddr, path: &str, body: &str, key: Option<&str>) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let auth = key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    let raw = format!(
        "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\n{auth}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    resp
}

#[test]
fn distributed_chat_completion() {
    let addr = start_server(None);
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}"#;
    let resp = post(addr, "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""object":"chat.completion""#), "{resp}");
    assert!(resp.contains(r#""completion_tokens":4"#), "{resp}");
    assert!(resp.contains(r#""role":"assistant""#), "{resp}");
}

#[test]
fn distributed_rejects_empty_messages() {
    let addr = start_server(None);
    let resp = post(addr, "/v1/chat/completions", r#"{"messages":[]}"#);
    assert!(resp.starts_with("HTTP/1.1 400"), "{resp}");
}

#[test]
fn distributed_clamps_max_tokens_to_context_window() {
    // A hostile max_tokens must not drive an unbounded generation loop: it's
    // capped to the remaining context budget (MAX_CONTEXT − prompt tokens).
    let addr = start_server(None);
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":100000000}"#;
    let resp = post(addr, "/v1/chat/completions", body);
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    // "Hi" under the plain byte tokenizer is a few tokens, so the completion is
    // clamped to well under MAX_CONTEXT, not 100M.
    assert!(
        !resp.contains(r#""completion_tokens":100000000"#),
        "max_tokens was not clamped: {resp}"
    );
}

#[test]
fn distributed_api_key_gates_completions() {
    let addr = start_server(Some("sk-secret"));
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":2}"#;

    // No key → 401.
    let resp = post(addr, "/v1/chat/completions", body);
    assert!(resp.starts_with("HTTP/1.1 401"), "unauth should be 401: {resp}");

    // Correct key → 200.
    let ok = post_auth(addr, "/v1/chat/completions", body, Some("sk-secret"));
    assert!(ok.starts_with("HTTP/1.1 200 OK"), "authed should be 200: {ok}");

    // Health stays public.
    let health = {
        let mut s = TcpStream::connect(addr).unwrap();
        s.write_all(b"GET /health HTTP/1.1\r\n\r\n").unwrap();
        let mut r = String::new();
        s.read_to_string(&mut r).unwrap();
        r
    };
    assert!(health.starts_with("HTTP/1.1 200 OK"), "{health}");
}

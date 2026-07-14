//! The distributed (Master-mode) HTTP serving path: a Coordinator-backed
//! OpenAI-compatible endpoint. Uses local-fallback routes (no worker_addr) so
//! the test exercises the HTTP layer + coordinator without real workers — the
//! networked pipeline itself is covered by tests/distributed.rs.

use dlm::distributed::{partition_layers, Coordinator, ShardRoute};
use dlm::forward::{BlockConfig, LayerTensors};
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
        rms_eps: 1e-5, rope_scaling: None,
    };
    let mut r = Rng::new(9);
    let layers: Vec<LayerTensors> = (0..num_layers)
        .map(|_| LayerTensors {
            q_proj: r.vec(cfg.q_dim() * hidden),
            k_proj: r.vec(cfg.kv_dim() * hidden),
            v_proj: r.vec(cfg.kv_dim() * hidden),
            o_proj: r.vec(hidden * cfg.q_dim()),
            gate_proj: r.vec(cfg.intermediate_size * hidden),
            up_proj: r.vec(cfg.intermediate_size * hidden),
            down_proj: r.vec(hidden * cfg.intermediate_size),
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

fn start_server() -> SocketAddr {
    let engine = Arc::new(DistributedEngine::new(
        coordinator(),
        BpeTokenizer::bytes_only(),
        ChatTemplate::Plain,
        256,
        "dlm-dist",
        16,
        0,
    ));
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(distributed::router(engine)).unwrap());
    addr
}

fn post(addr: SocketAddr, path: &str, body: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!(
        "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    resp
}

#[test]
fn distributed_chat_completion() {
    let addr = start_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}"#;
    let resp = post(addr, "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""object":"chat.completion""#), "{resp}");
    assert!(resp.contains(r#""completion_tokens":4"#), "{resp}");
    assert!(resp.contains(r#""role":"assistant""#), "{resp}");
}

#[test]
fn distributed_rejects_empty_messages() {
    let addr = start_server();
    let resp = post(addr, "/v1/chat/completions", r#"{"messages":[]}"#);
    assert!(resp.starts_with("HTTP/1.1 400"), "{resp}");
}

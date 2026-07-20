//! End-to-end test of the OpenAI-compatible HTTP server over a real socket.

use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, CpuKernel, LayerTensors};
use dlm::generate::Generator;
use dlm::server::{router, Engine, HttpServer};
use dlm::tokenizer::BpeTokenizer;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;

/// Build a tiny synthetic engine (identity block + small embed/head) so the
/// server has something to run. Output is meaningless but the pipeline is real.
fn build_engine() -> Engine<CpuKernel> {
    let vocab = 256usize;
    let hidden = 16usize;
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5, rope_scaling: None, moe: None, sliding_window: None, activation: Default::default(),
    };
    let kernel = CpuKernel::new(cfg, vec![LayerTensors::zeros(&cfg)]).unwrap();
    let fill = |n: usize, off: usize| -> Vec<f32> {
        (0..n).map(|i| (((i + off) % 13) as f32 - 6.0) * 0.02).collect()
    };
    let generator = Generator::new(
        kernel,
        fill(vocab * hidden, 0),  // embedding
        vec![1.0; hidden],        // final norm
        fill(vocab * hidden, 7),  // lm head
        vocab,
        1e-5,
        KvCacheConfig { num_layers: 1, num_kv_heads: 2, head_dim: 4, block_size: 16 },
        64,
    )
    .unwrap();
    Engine::new(generator, BpeTokenizer::bytes_only(), "dlm-test", 8, 0)
}

fn start_server() -> SocketAddr {
    let engine = Arc::new(build_engine());
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    let handler = router(engine);
    std::thread::spawn(move || server.serve(handler).unwrap());
    addr
}

fn request(addr: SocketAddr, method: &str, path: &str, body: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!(
        "{method} {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    resp
}

#[test]
fn chat_completions_endpoint() {
    let addr = start_server();
    let body = r#"{"model":"dlm","messages":[{"role":"user","content":"Hi"}],"max_tokens":4}"#;
    let resp = request(addr, "POST", "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""object":"chat.completion""#), "{resp}");
    assert!(resp.contains(r#""role":"assistant""#), "{resp}");
    assert!(resp.contains(r#""finish_reason":"length""#), "{resp}");
    assert!(resp.contains(r#""completion_tokens":4"#), "{resp}");
    assert!(resp.contains(r#""prompt_tokens":"#), "{resp}");
}

#[test]
fn completions_endpoint() {
    let addr = start_server();
    let body = r#"{"prompt":"hello","max_tokens":3}"#;
    let resp = request(addr, "POST", "/v1/completions", body);
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""object":"text_completion""#), "{resp}");
    assert!(resp.contains(r#""completion_tokens":3"#), "{resp}");
}

#[test]
fn models_endpoint_lists_the_served_model() {
    let addr = start_server();
    let resp = request(addr, "GET", "/v1/models", "");
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""id":"dlm-test""#), "{resp}");
    assert!(resp.contains(r#""object":"list""#), "{resp}");
}

#[test]
fn bad_request_and_not_found() {
    let addr = start_server();
    let bad = request(addr, "POST", "/v1/chat/completions", "{not json");
    assert!(bad.starts_with("HTTP/1.1 400"), "{bad}");

    let missing = request(addr, "GET", "/v1/nope", "");
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");
}

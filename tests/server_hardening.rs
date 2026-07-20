//! Server hardening on the batched OpenAI engine: bearer-token auth, a request
//! body-size cap, and `stop` sequences.

use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, CpuKernel, LayerTensors};
use dlm::generate::Generator;
use dlm::server::{engine::secured_router, EngineService, HttpServer};
use dlm::tokenizer::BpeTokenizer;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

fn start_server(api_key: Option<&str>) -> SocketAddr {
    let (vocab, hidden) = (256usize, 16usize);
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
        fill(vocab * hidden, 0),
        vec![1.0; hidden],
        fill(vocab * hidden, 7),
        vocab,
        1e-5,
        KvCacheConfig { num_layers: 1, num_kv_heads: 2, head_dim: 4, block_size: 16 },
        128,
    )
    .unwrap();
    let engine = EngineService::start(generator, BpeTokenizer::bytes_only(), vocab, "dlm", 8, 0, 4, 0);
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    let router = secured_router(engine, api_key.map(String::from));
    std::thread::spawn(move || server.serve(router).unwrap());
    addr
}

/// Send a raw request with optional extra headers; return the raw response.
fn request(addr: SocketAddr, path: &str, headers: &str, body: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!(
        "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\n{headers}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    resp
}

/// Robustly extract `choices[0].message.content` by parsing the JSON body
/// (the byte model can emit quotes/backslashes, so hand-parsing won't do).
fn content(resp: &str) -> String {
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("");
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    v["choices"][0]["message"]["content"].as_str().unwrap().to_string()
}

const CHAT: &str = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":8}"#;

#[test]
fn api_key_gates_v1_endpoints() {
    let addr = start_server(Some("secret"));

    // No / wrong key → 401.
    assert!(request(addr, "/v1/chat/completions", "", CHAT).starts_with("HTTP/1.1 401"));
    assert!(
        request(addr, "/v1/chat/completions", "Authorization: Bearer nope\r\n", CHAT)
            .starts_with("HTTP/1.1 401")
    );
    // Correct key → 200.
    assert!(
        request(addr, "/v1/chat/completions", "Authorization: Bearer secret\r\n", CHAT)
            .starts_with("HTTP/1.1 200")
    );
    // Health stays open.
    let mut s = TcpStream::connect(addr).unwrap();
    s.write_all(b"GET /health HTTP/1.1\r\n\r\n").unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    assert!(r.starts_with("HTTP/1.1 200"), "{r}");
}

#[test]
fn oversized_body_is_rejected() {
    let addr = start_server(None);
    // A Content-Length past the cap is refused before the body is read.
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n")
        .unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 413"), "{resp}");
}

#[test]
fn stop_sequence_truncates_output() {
    let addr = start_server(None);
    // Deterministic greedy output; pick a 2-char slice from it as the stop.
    let greedy = content(&request(addr, "/v1/chat/completions", "", CHAT));
    let chars: Vec<char> = greedy.chars().collect();
    assert!(chars.len() >= 4, "need some output to slice a stop from: {greedy:?}");
    let stop: String = chars[2..4].iter().collect();

    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"Hi"}}],"max_tokens":8,"stop":{}}}"#,
        serde_json::to_string(&stop).unwrap()
    );
    let resp = request(addr, "/v1/chat/completions", "", &body);
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert!(resp.contains(r#""finish_reason":"stop""#), "{resp}");
    let stopped = content(&resp);
    assert!(!stopped.contains(&stop), "stop sequence {stop:?} leaked into {stopped:?}");
    assert!(stopped.len() < greedy.len(), "output should be truncated: {stopped:?} vs {greedy:?}");
}

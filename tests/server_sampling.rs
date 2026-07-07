//! The batched OpenAI server (`EngineService`, what `dlm serve` runs) must
//! honor per-request sampling params: a request carrying `temperature`/`top_p`/
//! `seed` is accepted and produces output, and `temperature: 0` reduces to the
//! deterministic greedy path. (The sampler math itself is unit-tested in
//! `generate.rs`; this proves the params thread through the HTTP layer.)

use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, CpuKernel, LayerTensors};
use dlm::generate::Generator;
use dlm::server::{engine::router, EngineService, HttpServer};
use dlm::tokenizer::BpeTokenizer;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

fn start_server() -> SocketAddr {
    let vocab = 256usize;
    let hidden = 16usize;
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
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
        64,
    )
    .unwrap();
    let engine = EngineService::start(generator, BpeTokenizer::bytes_only(), vocab, "dlm", 8, 0, 8);
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
    addr
}

fn request(addr: SocketAddr, body: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    resp
}

#[test]
fn temperature_request_is_accepted_and_produces_output() {
    let addr = start_server();
    let body = r#"{"model":"dlm","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":1.5,"top_p":0.9,"seed":42}"#;
    let resp = request(addr, body);
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""completion_tokens":4"#), "{resp}");
    assert!(resp.contains(r#""role":"assistant""#), "{resp}");
}

#[test]
fn zero_temperature_matches_greedy() {
    let addr = start_server();
    // Extract just the assistant content so per-request id/created don't matter.
    let content = |resp: &str| -> String {
        let body = resp.split("\r\n\r\n").nth(1).unwrap_or("");
        let key = r#""content":"#;
        let start = body.find(key).unwrap() + key.len();
        let rest = &body[start..];
        // content is a JSON string; take through its closing quote (test tokens
        // are simple ASCII so no escaped quotes appear).
        let s = rest.strip_prefix('"').unwrap();
        let end = s.find('"').unwrap();
        s[..end].to_string()
    };

    let greedy = request(addr, r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#);
    let zero_temp = request(
        addr,
        r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0.0}"#,
    );
    assert_eq!(content(&greedy), content(&zero_temp), "temp=0 must equal greedy");
}

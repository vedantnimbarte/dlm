//! The batched/streaming engine behind the HTTP server: non-streaming and SSE
//! chat completions, plus concurrent requests.

use flip::cache::KvCacheConfig;
use flip::forward::{BlockConfig, CpuKernel, LayerTensors};
use flip::generate::Generator;
use flip::server::{engine::router, EngineService, HttpServer};
use flip::tokenizer::BpeTokenizer;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

fn build_generator() -> Generator<CpuKernel> {
    let (vocab, hidden) = (256usize, 16usize);
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
    Generator::new(
        kernel,
        fill(vocab * hidden, 0),
        vec![1.0; hidden],
        fill(vocab * hidden, 7),
        vocab,
        1e-5,
        KvCacheConfig { num_layers: 1, num_kv_heads: 2, head_dim: 4, block_size: 16 },
        128,
    )
    .unwrap()
}

fn start_server() -> SocketAddr {
    let engine = EngineService::start(
        build_generator(),
        BpeTokenizer::bytes_only(),
        256,
        "flip-test",
        16,
        0,
        4, // max batch
    );
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
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
fn non_streaming_chat_through_batched_engine() {
    let addr = start_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}"#;
    let resp = post(addr, "/v1/chat/completions", body);
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""object":"chat.completion""#), "{resp}");
    assert!(resp.contains(r#""completion_tokens":4"#), "{resp}");
}

#[test]
fn streaming_chat_emits_sse_chunks() {
    let addr = start_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}"#;
    let resp = post(addr, "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains("Content-Type: text/event-stream"), "{resp}");
    assert!(resp.contains(r#""object":"chat.completion.chunk""#), "{resp}");
    // Role chunk + 4 token chunks + final chunk + [DONE] → several SSE frames.
    let data_frames = resp.matches("data: ").count();
    assert!(data_frames >= 5, "expected several SSE frames, got {data_frames}\n{resp}");
    assert!(resp.trim_end().ends_with("data: [DONE]"), "{resp}");
}

#[test]
fn concurrent_requests_are_served() {
    let addr = start_server();
    let handles: Vec<_> = (0..3)
        .map(|i| {
            std::thread::spawn(move || {
                let body = format!(
                    r#"{{"messages":[{{"role":"user","content":"req {i}"}}],"max_tokens":5}}"#
                );
                post(addr, "/v1/chat/completions", &body)
            })
        })
        .collect();
    for h in handles {
        let resp = h.join().unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
        assert!(resp.contains(r#""completion_tokens":5"#), "{resp}");
    }
}

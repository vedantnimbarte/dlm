//! The batched/streaming engine behind the HTTP server: non-streaming and SSE
//! chat completions, plus concurrent requests.

use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, CpuKernel, LayerTensors};
use dlm::generate::Generator;
use dlm::server::{engine::router, EngineService, HttpServer};
use dlm::tokenizer::BpeTokenizer;
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
        "dlm-test",
        16,
        0,
        4, // max batch
    );
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
    addr
}

/// A server backed by a speculative engine: an identical draft is fully
/// accepted, so acceptance stats are deterministic and non-empty.
fn start_speculative_server() -> SocketAddr {
    let engine = EngineService::start_speculative(
        build_generator(),
        build_generator(), // identical draft → 100% acceptance
        4,                 // gamma
        BpeTokenizer::bytes_only(),
        256,
        "dlm-test",
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
fn speculative_usage_reported_non_streaming() {
    let addr = start_speculative_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":6}"#;
    let resp = post(addr, "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains(r#""completion_tokens":6"#), "{resp}");
    // usage.speculative carries the acceptance breakdown.
    assert!(resp.contains(r#""speculative""#), "{resp}");
    assert!(resp.contains(r#""draft_proposed""#), "{resp}");
    assert!(resp.contains(r#""draft_accepted""#), "{resp}");
    assert!(resp.contains(r#""acceptance_rate""#), "{resp}");
}

#[test]
fn plain_usage_omits_speculative() {
    let addr = start_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}"#;
    let resp = post(addr, "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    // No draft model → no speculative block at all.
    assert!(!resp.contains("speculative"), "plain path must omit speculative usage:\n{resp}");
}

#[test]
fn speculative_stream_emits_usage_chunk() {
    let addr = start_speculative_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}"#;
    let resp = post(addr, "/v1/chat/completions", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    // A final usage-only chunk precedes [DONE] when speculating.
    assert!(resp.contains(r#""usage""#), "expected a usage chunk:\n{resp}");
    assert!(resp.contains(r#""acceptance_rate""#), "{resp}");
    assert!(resp.trim_end().ends_with("data: [DONE]"), "{resp}");
}

#[test]
fn streaming_messages_emits_anthropic_sse_events() {
    let addr = start_server();
    let body = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}"#;
    let resp = post(addr, "/v1/messages", body);

    assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
    assert!(resp.contains("Content-Type: text/event-stream"), "{resp}");
    // The full Anthropic event sequence, in order.
    for ev in [
        "event: message_start",
        "event: content_block_start",
        "event: content_block_delta",
        "event: content_block_stop",
        "event: message_delta",
        "event: message_stop",
    ] {
        assert!(resp.contains(ev), "missing {ev}:\n{resp}");
    }
    assert!(resp.contains(r#""type":"text_delta""#), "{resp}");
    // Ran to max_tokens (no stop sequence), so stop_reason reflects that.
    assert!(resp.contains(r#""stop_reason":"max_tokens""#), "{resp}");
    assert!(resp.contains(r#""output_tokens":4"#), "{resp}");
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

//! Load and soak: many clients at once, every kind of request, some hanging up
//! mid-stream, against the real HTTP server and scheduler.
//!
//! What it holds the server to: a greedy chat answer under load is the same
//! as the one it gives alone (no cross-talk between batched requests' KV),
//! every other request comes back whole, and afterwards nothing is left
//! running or queued, the GPU's KV pool is back to full (nothing leaked by an
//! abandoned stream), and the server still answers.

use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, ComputeKernel, CpuKernel, LayerTensors};
use dlm::generate::Generator;
use dlm::server::{engine::router, EngineService, HttpServer};
use dlm::tokenizer::BpeTokenizer;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

const VOCAB: usize = 256;

fn cfg() -> BlockConfig {
    BlockConfig {
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        intermediate_size: 128,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        ..Default::default()
    }
}

/// Small random layers: real attention over real history, so requests that
/// shared KV by mistake would answer differently.
fn layers() -> Vec<LayerTensors> {
    let c = cfg();
    let mut seed = 0x5EED_u64;
    let mut rand = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    };
    (0..2)
        .map(|_| {
            let mut l = LayerTensors::zeros(&c);
            l.q_proj = dlm::forward::Weights::F32(rand(c.q_dim() * c.hidden_size));
            l.k_proj = dlm::forward::Weights::F32(rand(c.kv_dim() * c.hidden_size));
            l.v_proj = dlm::forward::Weights::F32(rand(c.kv_dim() * c.hidden_size));
            l.o_proj = dlm::forward::Weights::F32(rand(c.hidden_size * c.q_dim()));
            l
        })
        .collect()
}

/// Serve `kernel` with four batch slots and `prefix_cache` cached prompts.
fn serve<K: ComputeKernel + Send + 'static>(kernel: K, prefix_cache: usize) -> SocketAddr {
    let c = cfg();
    let fill = |n: usize, off: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((i as u64 + off as u64).wrapping_mul(2_654_435_761) % 1000) as f32 / 1000.0 - 0.5
            })
            .collect()
    };
    let generator = Generator::new(
        kernel,
        fill(VOCAB * c.hidden_size, 0),
        vec![1.0; c.hidden_size],
        fill(VOCAB * c.hidden_size, 3),
        VOCAB,
        1e-5,
        KvCacheConfig {
            num_layers: 2,
            num_kv_heads: c.num_kv_heads as u32,
            head_dim: c.head_dim as u32,
            block_size: 16,
        },
        64,
    )
    .unwrap();
    // Four slots for sixteen clients: most requests queue, and the queue and
    // admission get worked.
    let engine = EngineService::start(
        generator,
        BpeTokenizer::bytes_only(),
        VOCAB,
        "dlm",
        8,
        0,
        4,
        prefix_cache,
    )
    .with_context_window(256);
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
    addr
}

fn send(addr: SocketAddr, method: &str, path: &str, body: &str) -> TcpStream {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    let raw = format!(
        "{method} {path} HTTP/1.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    stream
}

/// Status line and body.
fn call(addr: SocketAddr, method: &str, path: &str, body: &str) -> (String, String) {
    let mut resp = String::new();
    send(addr, method, path, body)
        .read_to_string(&mut resp)
        .unwrap();
    let (head, body) = resp.split_once("\r\n\r\n").unwrap();
    (head.lines().next().unwrap().to_string(), body.to_string())
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

/// The SSE payloads, `[DONE]` included.
fn events(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(str::to_string)
        .collect()
}

fn chat(prompt: &str, extra: &str) -> String {
    format!(r#"{{"messages":[{{"role":"user","content":"{prompt}"}}],"max_tokens":6{extra}}}"#)
}

fn metric(addr: SocketAddr, name: &str) -> Option<u64> {
    let (_, body) = call(addr, "GET", "/metrics", "");
    body.lines()
        .find_map(|l| l.strip_prefix(&format!("{name} ")))
        .map(|v| v.trim().parse().unwrap())
}

/// Run the load. With `check_kv`, the KV pool must end as full as it started
/// (the prefix cache off, so any block still out is a leak).
fn soak(addr: SocketAddr, check_kv: bool) {
    let kv_pool = check_kv.then(|| metric(addr, "dlm_kv_free_tokens").expect("a KV pool gauge"));
    // What each prompt gets alone, before any load.
    let prompts = ["alpha", "bravo", "charlie", "delta"];
    let alone: Arc<Vec<Value>> = Arc::new(
        prompts
            .iter()
            .map(|p| {
                json(&call(addr, "POST", "/v1/chat/completions", &chat(p, "")).1)["choices"][0]
                    ["message"]
                    .clone()
            })
            .collect(),
    );

    // One round in CI; `DLM_SOAK_ROUNDS=500` makes it a soak worth the name.
    let rounds: usize = std::env::var("DLM_SOAK_ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(1);
    let started = Instant::now();
    for _ in 0..rounds {
        let clients: Vec<_> = (0..16)
            .map(|client| {
                let alone = Arc::clone(&alone);
                std::thread::spawn(move || {
                    for i in 0..12 {
                        let p = (client + i) % prompts.len();
                        match (client * 12 + i) % 6 {
                            // Greedy chat: exactly what it answers alone.
                            0 => {
                                let (status, body) = call(
                                    addr,
                                    "POST",
                                    "/v1/chat/completions",
                                    &chat(prompts[p], ""),
                                );
                                assert!(status.contains("200"), "{status} {body}");
                                assert_eq!(
                                    json(&body)["choices"][0]["message"],
                                    alone[p],
                                    "cross-talk"
                                );
                            }
                            // Streamed chat: whole, ending in a finish and [DONE].
                            1 => {
                                let (status, body) = call(
                                    addr,
                                    "POST",
                                    "/v1/chat/completions",
                                    &chat(prompts[p], r#","stream":true"#),
                                );
                                assert!(status.contains("200"), "{status}");
                                let ev = events(&body);
                                assert_eq!(ev.last().map(String::as_str), Some("[DONE]"), "{body}");
                                let text: String = ev[..ev.len() - 1]
                                    .iter()
                                    .map(|e| {
                                        json(e)["choices"][0]["delta"]["content"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string()
                                    })
                                    .collect();
                                assert_eq!(
                                    Value::String(text),
                                    alone[p]["content"],
                                    "streamed cross-talk"
                                );
                            }
                            // Several choices with logprobs.
                            2 => {
                                let body = format!(
                                    r#"{{"prompt":"{}","max_tokens":5,"n":3,"logprobs":2}}"#,
                                    prompts[p]
                                );
                                let (status, resp) = call(addr, "POST", "/v1/completions", &body);
                                assert!(status.contains("200"), "{status} {resp}");
                                let r = json(&resp);
                                assert_eq!(r["choices"].as_array().unwrap().len(), 3);
                                assert_eq!(r["usage"]["completion_tokens"], 15);
                            }
                            // Embeddings between decode steps.
                            3 => {
                                let body = format!(
                                    r#"{{"input":["{}","{}"]}}"#,
                                    prompts[p],
                                    prompts[(p + 1) % 4]
                                );
                                let (status, resp) = call(addr, "POST", "/v1/embeddings", &body);
                                assert!(status.contains("200"), "{status} {resp}");
                                assert_eq!(json(&resp)["data"].as_array().unwrap().len(), 2);
                            }
                            // A client that hangs up after the first chunk.
                            4 => {
                                let s = send(
                                    addr,
                                    "POST",
                                    "/v1/chat/completions",
                                    &chat(prompts[p], r#","max_tokens":200,"stream":true"#),
                                );
                                let mut lines = BufReader::new(s).lines();
                                while let Some(Ok(l)) = lines.next() {
                                    if l.starts_with("data: ") {
                                        break; // drop the socket mid-stream
                                    }
                                }
                            }
                            // A stop sequence that cuts the reply short.
                            _ => {
                                let (status, body) = call(
                                    addr,
                                    "POST",
                                    "/v1/chat/completions",
                                    &chat(prompts[p], r#","stop":["e"]"#),
                                );
                                assert!(status.contains("200"), "{status} {body}");
                                let content = json(&body)["choices"][0]["message"]["content"]
                                    .as_str()
                                    .unwrap()
                                    .to_string();
                                assert!(!content.contains('e'), "{content}");
                            }
                        }
                    }
                })
            })
            .collect();
        for c in clients {
            c.join().unwrap();
        }
    }
    eprintln!(
        "{} requests from 16 clients in {:?}",
        192 * rounds,
        started.elapsed()
    );

    // Everything drains: nothing running or queued, and the KV pool whole.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let idle = metric(addr, "dlm_requests_running") == Some(0)
            && metric(addr, "dlm_requests_waiting") == Some(0)
            && kv_pool.is_none_or(|full| metric(addr, "dlm_kv_free_tokens") == Some(full));
        if idle {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the server did not drain: {}",
            call(addr, "GET", "/metrics", "").1
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    // And it still answers, the same as before.
    let (_, body) = call(addr, "POST", "/v1/chat/completions", &chat(prompts[0], ""));
    assert_eq!(json(&body)["choices"][0]["message"], alone[0]);
}

#[test]
fn the_server_survives_a_soak_on_the_cpu() {
    // The prefix cache on: cached prompts resume under load too.
    soak(serve(CpuKernel::new(cfg(), layers()).unwrap(), 8), false);
}

/// The same on the GPU, where KV is paged from a pool: a request abandoned
/// mid-stream must give its blocks back.
#[cfg(feature = "cuda-kernels")]
#[test]
fn the_server_survives_a_soak_on_the_gpu() {
    let kernel = dlm::forward::GpuKernel::new(cfg(), layers(), 256).unwrap();
    soak(serve(kernel, 0), true);
}

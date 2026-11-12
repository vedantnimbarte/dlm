//! The OpenAI request fields beyond sampling: `logprobs` (each output token's
//! log-probability and its most likely alternatives, in chat's and the legacy
//! completions' shapes, streamed or not; asking for them must not change what
//! is generated), `n` (several completions of one prompt), and the
//! `/v1/embeddings` endpoint.

use dlm::cache::KvCacheConfig;
use dlm::forward::{BlockConfig, CpuKernel, LayerTensors};
use dlm::generate::Generator;
use dlm::server::{engine::router, EngineService, HttpServer};
use dlm::tokenizer::BpeTokenizer;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

/// A byte model: 256 tokens, one zero layer, varied embedding and head.
fn start_server() -> SocketAddr {
    let (vocab, hidden) = (256usize, 16usize);
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        intermediate_size: 32,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        ..Default::default()
    };
    let kernel = CpuKernel::new(cfg, vec![LayerTensors::zeros(&cfg)]).unwrap();
    // Varied weights, so tokens' logits differ and greedy has one answer.
    let fill = |n: usize, off: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((i as u64 + off as u64).wrapping_mul(2_654_435_761) % 1000) as f32 / 1000.0 - 0.5
            })
            .collect()
    };
    let generator = Generator::new(
        kernel,
        fill(vocab * hidden, 0),
        vec![1.0; hidden],
        fill(vocab * hidden, 5),
        vocab,
        1e-5,
        KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            block_size: 16,
        },
        64,
    )
    .unwrap();
    let engine = EngineService::start(
        generator,
        BpeTokenizer::bytes_only(),
        vocab,
        "dlm",
        8,
        0,
        8,
        0,
    );
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
    addr
}

/// POST `body` to `path`: the status line and the body.
fn post(addr: SocketAddr, path: &str, body: &str) -> (String, String) {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!(
        "POST {path} HTTP/1.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(raw.as_bytes()).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    let (head, body) = resp.split_once("\r\n\r\n").unwrap();
    (head.lines().next().unwrap().to_string(), body.to_string())
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

/// The `data:` payloads of an SSE body, chunked transfer encoding stripped by
/// keeping only the lines that are events.
fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(json)
        .collect()
}

const CHAT: &str = r#""messages":[{"role":"user","content":"Hi there"}],"max_tokens":6"#;

#[test]
fn chat_reports_each_tokens_logprob_and_alternatives() {
    let addr = start_server();
    let (status, body) = post(
        addr,
        "/v1/chat/completions",
        &format!(r#"{{{CHAT},"logprobs":true,"top_logprobs":3}}"#),
    );
    assert!(status.contains("200"), "{status} {body}");
    let r = json(&body);
    let choice = &r["choices"][0];
    let content = choice["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 6, "one entry per token: {body}");
    let mut text = String::new();
    for entry in content {
        let top = entry["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 3, "{entry}");
        let lps: Vec<f64> = top.iter().map(|t| t["logprob"].as_f64().unwrap()).collect();
        assert!(
            lps.windows(2).all(|w| w[0] >= w[1]),
            "most likely first: {lps:?}"
        );
        assert!(lps.iter().map(|l| l.exp()).sum::<f64>() <= 1.0 + 1e-6);
        // Greedy decoding picks the most likely token.
        assert_eq!(entry["token"], top[0]["token"], "{entry}");
        assert_eq!(entry["logprob"], top[0]["logprob"], "{entry}");
        let bytes: Vec<u8> = entry["bytes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect();
        text.push_str(&String::from_utf8_lossy(&bytes));
    }
    assert_eq!(text, choice["message"]["content"].as_str().unwrap());

    // Without them: the same text, and `logprobs: null`.
    let (_, plain) = post(addr, "/v1/chat/completions", &format!("{{{CHAT}}}"));
    let plain = json(&plain);
    assert_eq!(plain["choices"][0]["message"], choice["message"]);
    assert!(plain["choices"][0]["logprobs"].is_null(), "{plain}");
}

#[test]
fn sampled_chat_still_reports_the_sampled_token() {
    let addr = start_server();
    let body =
        format!(r#"{{{CHAT},"temperature":1.5,"seed":7,"logprobs":true,"top_logprobs":20}}"#);
    let (_, with) = post(addr, "/v1/chat/completions", &body);
    let (_, without) = post(
        addr,
        "/v1/chat/completions",
        &format!(r#"{{{CHAT},"temperature":1.5,"seed":7}}"#),
    );
    let (with, without) = (json(&with), json(&without));
    assert_eq!(
        with["choices"][0]["message"], without["choices"][0]["message"],
        "the same seed draws the same tokens"
    );
    let content = with["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap();
    assert_eq!(content.len(), 6);
    // The logprob is the drawn token's own: where the token is among the
    // alternatives, it matches that entry.
    let mut matched = 0;
    for e in content {
        let top = e["top_logprobs"].as_array().unwrap();
        assert!(e["logprob"].as_f64().unwrap() <= top[0]["logprob"].as_f64().unwrap());
        if let Some(t) = top.iter().find(|t| t["bytes"] == e["bytes"]) {
            assert_eq!(t["logprob"], e["logprob"], "{e}");
            matched += 1;
        }
    }
    assert!(matched > 0, "no drawn token among its 20 alternatives");
}

#[test]
fn streamed_chat_carries_the_same_logprobs() {
    let addr = start_server();
    let ask = format!(r#"{CHAT},"logprobs":true,"top_logprobs":2"#);
    let (_, whole) = post(addr, "/v1/chat/completions", &format!("{{{ask}}}"));
    let whole = json(&whole)["choices"][0]["logprobs"]["content"].clone();
    let (_, stream) = post(
        addr,
        "/v1/chat/completions",
        &format!(r#"{{{ask},"stream":true}}"#),
    );
    let streamed: Vec<Value> = events(&stream)
        .iter()
        .filter_map(|c| c["choices"][0]["logprobs"]["content"].as_array().cloned())
        .flatten()
        .collect();
    assert_eq!(Value::Array(streamed), whole);
}

#[test]
fn completions_report_tokens_offsets_and_alternatives() {
    let addr = start_server();
    let (status, body) = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"Hello","max_tokens":5,"logprobs":2}"#,
    );
    assert!(status.contains("200"), "{status} {body}");
    let r = json(&body);
    let choice = &r["choices"][0];
    let lp = &choice["logprobs"];
    let tokens: Vec<&str> = lp["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert_eq!(tokens.len(), 5, "{body}");
    assert_eq!(lp["token_logprobs"].as_array().unwrap().len(), 5);
    assert!(lp["top_logprobs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|m| m.as_object().unwrap().len() <= 2));
    let offsets: Vec<u64> = lp["text_offset"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_u64().unwrap())
        .collect();
    assert_eq!(offsets[0], 0);
    assert!(offsets.windows(2).all(|w| w[0] <= w[1]), "{offsets:?}");

    // Streamed: the same tokens across the chunks.
    let (_, stream) = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"Hello","max_tokens":5,"logprobs":2,"stream":true}"#,
    );
    let streamed: Vec<Value> = events(&stream)
        .iter()
        .filter_map(|c| c["choices"][0]["logprobs"]["tokens"].as_array().cloned())
        .flatten()
        .collect();
    assert_eq!(Value::Array(streamed), lp["tokens"]);
}

#[test]
fn bad_logprob_requests_are_refused() {
    let addr = start_server();
    for (path, body) in [
        (
            "/v1/chat/completions",
            format!(r#"{{{CHAT},"logprobs":true,"top_logprobs":21}}"#),
        ),
        (
            "/v1/chat/completions",
            format!(r#"{{{CHAT},"top_logprobs":2}}"#),
        ),
        (
            "/v1/completions",
            r#"{"prompt":"Hi","logprobs":21}"#.to_string(),
        ),
    ] {
        let (status, resp) = post(addr, path, &body);
        assert!(status.contains("400"), "{body}: {status} {resp}");
    }
}

fn contents(r: &Value) -> Vec<String> {
    r["choices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["message"]["content"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn n_samples_several_choices_each_with_its_own_seed() {
    let addr = start_server();
    let ask = |extra: &str| {
        let (status, body) = post(
            addr,
            "/v1/chat/completions",
            &format!(r#"{{{CHAT},"temperature":1.5,{extra}}}"#),
        );
        assert!(status.contains("200"), "{status} {body}");
        json(&body)
    };
    let r = ask(r#""seed":40,"n":3"#);
    let indexes: Vec<u64> = r["choices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["index"].as_u64().unwrap())
        .collect();
    assert_eq!(indexes, vec![0, 1, 2]);
    // Choice i draws as a lone request seeded `seed + i` does.
    let got = contents(&r);
    for (i, text) in got.iter().enumerate() {
        let alone = ask(&format!(r#""seed":{}"#, 40 + i));
        assert_eq!(&contents(&alone)[0], text, "choice {i}");
    }
    assert!(
        got[0] != got[1] || got[1] != got[2],
        "three seeds, one text: {got:?}"
    );
    // The prompt counts once; the completions add up.
    let usage = &r["usage"];
    let prompt = usage["prompt_tokens"].as_u64().unwrap();
    assert_eq!(usage["completion_tokens"].as_u64().unwrap(), 18);
    assert_eq!(usage["total_tokens"].as_u64().unwrap(), prompt + 18);
}

#[test]
fn n_streams_each_choice_under_its_index() {
    let addr = start_server();
    let ask = format!(r#"{CHAT},"temperature":1.5,"seed":9,"n":2"#);
    let (_, whole) = post(addr, "/v1/chat/completions", &format!("{{{ask}}}"));
    let want = contents(&json(&whole));
    let (_, stream) = post(
        addr,
        "/v1/chat/completions",
        &format!(r#"{{{ask},"stream":true}}"#),
    );
    let mut text = vec![String::new(); 2];
    let mut finished = [0; 2];
    for e in events(&stream) {
        let c = &e["choices"][0];
        let i = c["index"].as_u64().unwrap() as usize;
        text[i] += c["delta"]["content"].as_str().unwrap_or("");
        if !c["finish_reason"].is_null() {
            finished[i] += 1;
        }
    }
    assert_eq!(text, want);
    assert_eq!(finished, [1, 1]);
}

#[test]
fn completions_take_n_too() {
    let addr = start_server();
    let (_, body) = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"Hello","max_tokens":4,"n":2,"logprobs":1}"#,
    );
    let r = json(&body);
    let choices = r["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2, "{body}");
    // Greedy: both choices are the one most likely text.
    assert_eq!(choices[0]["text"], choices[1]["text"]);
    assert_eq!(choices[1]["index"], 1);
    assert_eq!(
        choices[1]["logprobs"]["tokens"].as_array().unwrap().len(),
        4
    );
}

#[test]
fn n_out_of_range_is_refused() {
    let addr = start_server();
    for n in [0, 17] {
        let (status, body) = post(
            addr,
            "/v1/chat/completions",
            &format!(r#"{{{CHAT},"n":{n}}}"#),
        );
        assert!(status.contains("400"), "n {n}: {status} {body}");
    }
}

fn embeddings(addr: SocketAddr, body: &str) -> Vec<Vec<f32>> {
    let (status, resp) = post(addr, "/v1/embeddings", body);
    assert!(status.contains("200"), "{body}: {status} {resp}");
    let r = json(&resp);
    r["data"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, d)| {
            assert_eq!(d["index"], i as u64);
            match &d["embedding"] {
                Value::String(b64) => decode_f32s(b64),
                v => v
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect(),
            }
        })
        .collect()
}

/// Little-endian f32s from standard base64.
fn decode_f32s(b64: &str) -> Vec<f32> {
    let val = |c: u8| {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            .bytes()
            .position(|x| x == c)
            .unwrap() as u32
    };
    let mut bytes = Vec::new();
    for chunk in b64.as_bytes().chunks(4) {
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        let n = chunk
            .iter()
            .fold(0u32, |n, &c| n << 6 | if c == b'=' { 0 } else { val(c) });
        bytes.extend(&n.to_be_bytes()[1..4 - pad]);
    }
    bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn embeddings_are_unit_vectors_of_the_hidden_size() {
    let addr = start_server();
    let vs = embeddings(
        addr,
        r#"{"input":["Hi there","Hi there","Something else!"]}"#,
    );
    assert_eq!(vs.len(), 3);
    assert!(
        vs.iter()
            .all(|v| v.len() == 16 && (norm(v) - 1.0).abs() < 1e-4),
        "{vs:?}"
    );
    assert_eq!(vs[0], vs[1], "the same text, the same vector");
    assert_ne!(vs[0], vs[2]);
    // A string, its token ids (bytes, for this tokenizer), and base64 agree.
    let one = embeddings(addr, r#"{"input":"Hi"}"#);
    let ids = embeddings(addr, r#"{"input":[72,105]}"#);
    let b64 = embeddings(addr, r#"{"input":"Hi","encoding_format":"base64"}"#);
    assert_eq!(one, ids);
    assert_eq!(one, b64);
    // Fewer dimensions: the leading ones, back at unit length.
    let short = embeddings(addr, r#"{"input":"Hi","dimensions":4}"#);
    assert_eq!(short[0].len(), 4);
    assert!((norm(&short[0]) - 1.0).abs() < 1e-4);
    let scale = one[0][0] / short[0][0];
    assert!(one[0][..4]
        .iter()
        .zip(&short[0])
        .all(|(a, b)| (a / b - scale).abs() < 1e-4));
}

#[test]
fn bad_embedding_requests_are_refused() {
    let addr = start_server();
    for body in [
        r#"{"input":[]}"#,
        r#"{"input":"Hi","dimensions":17}"#,
        r#"{"input":"Hi","encoding_format":"hex"}"#,
        r#"{"input":[[300]]}"#,
    ] {
        let (status, resp) = post(addr, "/v1/embeddings", body);
        assert!(status.contains("400"), "{body}: {status} {resp}");
    }
}

/// A server with one batch slot, a model slow enough (4 layers of 256) that
/// a long request holds the slot for a good while, and a `queue_timeout`.
fn start_busy_server(queue_timeout: std::time::Duration) -> SocketAddr {
    let (vocab, hidden) = (256usize, 256usize);
    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        intermediate_size: 1024,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        ..Default::default()
    };
    let layers = (0..4).map(|_| LayerTensors::zeros(&cfg)).collect();
    let kernel = CpuKernel::new(cfg, layers).unwrap();
    let fill = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect() };
    let generator = Generator::new(
        kernel,
        fill(vocab * hidden),
        vec![1.0; hidden],
        fill(vocab * hidden),
        vocab,
        1e-5,
        KvCacheConfig {
            num_layers: 4,
            num_kv_heads: 2,
            head_dim: 64,
            block_size: 16,
        },
        64,
    )
    .unwrap();
    let engine = EngineService::start(
        generator,
        BpeTokenizer::bytes_only(),
        vocab,
        "dlm",
        8,
        0,
        1,
        0,
    )
    .with_queue_timeout(queue_timeout);
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
    addr
}

/// A request that waits past `--queue-timeout` for its slot is answered 503
/// with `Retry-After`, streamed or not; one already decoding runs to its end,
/// and the server takes requests again once the slot is free.
#[test]
fn a_request_that_waits_too_long_is_turned_away() {
    use std::time::Duration;
    let addr = start_busy_server(Duration::from_millis(30));
    let long = std::thread::spawn(move || {
        post(
            addr,
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":600}"#,
        )
    });
    std::thread::sleep(Duration::from_millis(50)); // the long one has the slot
    for (path, body) in [
        ("/v1/completions", r#"{"prompt":"b","max_tokens":2}"#),
        (
            "/v1/completions",
            r#"{"prompt":"b","max_tokens":2,"stream":true}"#,
        ),
        (
            "/v1/messages",
            r#"{"messages":[{"role":"user","content":"b"}],"max_tokens":2,"stream":true}"#,
        ),
    ] {
        let mut stream = TcpStream::connect(addr).unwrap();
        let raw = format!(
            "POST {path} HTTP/1.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(raw.as_bytes()).unwrap();
        let mut resp = String::new();
        stream.read_to_string(&mut resp).unwrap();
        assert!(resp.starts_with("HTTP/1.1 503"), "{body}: {resp}");
        assert!(resp.contains("Retry-After:"), "{resp}");
        assert!(resp.contains("busy"), "{resp}");
    }
    let (status, body) = long.join().unwrap();
    assert!(status.contains("200"), "{status} {body}");
    assert_eq!(json(&body)["usage"]["completion_tokens"], 600);
    let (status, _) = post(addr, "/v1/completions", r#"{"prompt":"c","max_tokens":2}"#);
    assert!(status.contains("200"), "{status}");
}

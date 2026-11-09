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
    start_server_with_eos(Vec::new())
}

/// The byte model, with `eos` as its end-of-sequence ids.
fn start_server_with_eos(eos: Vec<u32>) -> SocketAddr {
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
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: Default::default(),
        mla: None,
        ..Default::default()
    };
    let kernel = CpuKernel::new(cfg, vec![LayerTensors::zeros(&cfg)]).unwrap();
    let fill = |n: usize, off: usize| -> Vec<f32> {
        (0..n)
            .map(|i| (((i + off) % 13) as f32 - 6.0) * 0.02)
            .collect()
    };
    let generator = Generator::new(
        kernel,
        fill(vocab * hidden, 0),
        vec![1.0; hidden],
        fill(vocab * hidden, 7),
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
    )
    .with_eos_tokens(eos);
    let server = HttpServer::bind("127.0.0.1:0").unwrap();
    let addr = server.local_addr().unwrap();
    std::thread::spawn(move || server.serve(router(engine)).unwrap());
    addr
}

fn request(addr: SocketAddr, body: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    let raw = format!(
        // Explicitly single-shot: this helper reads to EOF.
        "POST /v1/chat/completions HTTP/1.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
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

    let greedy = request(
        addr,
        r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#,
    );
    let zero_temp = request(
        addr,
        r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0.0}"#,
    );
    assert_eq!(
        content(&greedy),
        content(&zero_temp),
        "temp=0 must equal greedy"
    );
}

/// The assistant text of a non-streamed chat response.
fn assistant_text(resp: &str) -> String {
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("");
    let json: serde_json::Value =
        serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {resp}"));
    json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("no content: {resp}"))
        .to_string()
}

fn chat(addr: SocketAddr, fields: &str) -> String {
    request(
        addr,
        &format!(r#"{{"messages":[{{"role":"user","content":"Hi"}}],{fields}}}"#),
    )
}

#[test]
fn logit_bias_steers_even_greedy_decoding() {
    let addr = start_server();
    // +100 on byte 'A' outweighs anything this model's logits can do.
    let resp = chat(addr, r#""max_tokens":4,"logit_bias":{"65":100}"#);
    assert_eq!(assistant_text(&resp), "AAAA", "{resp}");
}

#[test]
fn presence_penalty_stops_a_token_repeating() {
    let addr = start_server();
    // A +3 bias makes 'A' the pick; once it has appeared, the -2 presence
    // penalty and -2 per occurrence take back more than the bias gave.
    let biased = r#""max_tokens":3,"logit_bias":{"65":3}"#;
    assert_eq!(assistant_text(&chat(addr, biased)), "AAA");
    let text = assistant_text(&chat(
        addr,
        &format!(r#"{biased},"presence_penalty":2,"frequency_penalty":2"#),
    ));
    assert_eq!(text.matches('A').count(), 1, "{text:?}");
}

#[test]
fn dry_breaks_a_loop_once_it_is_long_enough() {
    let addr = start_server();
    // Biased into "AAAAAA". DRY lets a run reach dry_allowed_length (2), then
    // charges the token that would extend it: after "AAA", A would make the
    // repeat "AA" → "AA" one longer, and costs 5 against the bias's 3.
    let biased = r#""max_tokens":6,"logit_bias":{"65":3}"#;
    assert_eq!(assistant_text(&chat(addr, biased)), "AAAAAA");
    let text = assistant_text(&chat(addr, &format!(r#"{biased},"dry_multiplier":5"#)));
    assert!(
        text.starts_with("AAA") && !text.starts_with("AAAA"),
        "{text:?}"
    );
}

#[test]
fn out_of_range_sampling_fields_are_rejected() {
    let addr = start_server();
    for (field, why) in [
        (r#""logit_bias":{"65":101}"#, "between -100 and 100"),
        (r#""logit_bias":{"abc":1}"#, "not a token id"),
        (r#""logit_bias":{"256":1}"#, "outside the vocabulary"),
        (
            r#""frequency_penalty":2.5"#,
            "frequency_penalty must be between -2 and 2",
        ),
        (
            r#""presence_penalty":-3"#,
            "presence_penalty must be between -2 and 2",
        ),
        (r#""dry_multiplier":-1"#, "dry_multiplier must be 0 or more"),
        (r#""dry_base":0.5"#, "dry_base must be at least 1"),
        (
            r#""dry_allowed_length":0"#,
            "dry_allowed_length must be at least 1",
        ),
        (r#""dry_sequence_breakers":["\n",""]"#, "empty string"),
        (
            r#""grammar":"root ::= missing""#,
            "'missing' is used but never defined",
        ),
        (r#""grammar":"root ::= root \"a\"""#, "left-recursive"),
        (r#""response_format":{"type":"yaml"}"#, "type 'yaml' is not"),
        (
            r#""response_format":{"type":"json_schema"}"#,
            "needs json_schema.schema",
        ),
        (
            r#""response_format":{"type":"json_schema","json_schema":{"schema":{"type":"date"}}}"#,
            "unknown type 'date'",
        ),
        (
            r#""grammar":"root ::= \"a\"","response_format":{"type":"json_object"}"#,
            "not both",
        ),
    ] {
        let resp = chat(addr, field);
        assert!(resp.starts_with("HTTP/1.1 400"), "{field}: {resp}");
        assert!(resp.contains(why), "{field}: {resp}");
    }
}

fn completion(resp: &str) -> serde_json::Value {
    let body = resp.split("\r\n\r\n").nth(1).unwrap_or("");
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {resp}"))
}

/// The model is random weights: left to itself it emits arbitrary bytes. Held
/// to a schema, greedy or sampled, what comes back must parse as JSON of that
/// shape, and end on EOS rather than run out of tokens.
#[test]
fn json_schema_output_parses_and_matches_the_schema() {
    let addr = start_server_with_eos(vec![0]);
    let schema = r#"{"type":"object","properties":{
        "ok":{"type":"boolean"},
        "color":{"enum":["red","green"]},
        "n":{"type":"integer"},
        "maybe":{"type":"null"}
    },"required":["ok","color","n"]}"#
        .replace(char::is_whitespace, "");
    for sampling in [
        r#""temperature":0"#,
        r#""temperature":1.5,"seed":3"#,
        r#""temperature":1,"seed":9"#,
    ] {
        let resp = chat(
            addr,
            &format!(
                r#""max_tokens":400,{sampling},"response_format":{{"type":"json_schema","json_schema":{{"name":"t","schema":{schema}}}}}"#
            ),
        );
        let json = completion(&resp);
        assert_eq!(json["choices"][0]["finish_reason"], "stop", "{resp}");
        let text = json["choices"][0]["message"]["content"].as_str().unwrap();
        let v: serde_json::Value = serde_json::from_str(text)
            .unwrap_or_else(|e| panic!("{sampling}: not JSON ({e}): {text:?}"));
        let obj = v
            .as_object()
            .unwrap_or_else(|| panic!("not an object: {text:?}"));
        assert!(obj["ok"].is_boolean(), "{text:?}");
        assert!(
            ["red", "green"].contains(&obj["color"].as_str().unwrap()),
            "{text:?}"
        );
        assert!(obj["n"].is_i64(), "{text:?}");
        assert!(obj.get("maybe").is_none_or(|m| m.is_null()), "{text:?}");
        assert!(
            obj.keys()
                .all(|k| ["ok", "color", "n", "maybe"].contains(&k.as_str())),
            "{text:?}"
        );
    }
}

#[test]
fn json_object_and_grammar_constrain_the_output() {
    let addr = start_server_with_eos(vec![0]);
    // The bias makes '}' the model's favourite; a JSON object has to open
    // first, then '}' may close it.
    let resp = chat(
        addr,
        r#""max_tokens":50,"logit_bias":{"125":100},"response_format":{"type":"json_object"}"#,
    );
    let text = assistant_text(&resp);
    assert_eq!(text.trim_end(), "{}", "{resp}");

    let resp = chat(
        addr,
        r#""max_tokens":50,"temperature":1,"grammar":"root ::= (\"yes\" | \"no\") \"!\"""#,
    );
    let text = assistant_text(&resp);
    assert!(text == "yes!" || text == "no!", "{text:?}");
}

/// With no EOS token to end on, a request ends once its grammar is complete
/// instead of running on to `max_tokens`.
#[test]
fn a_finished_grammar_ends_the_request_without_eos() {
    let addr = start_server();
    let resp = chat(addr, r#""max_tokens":50,"grammar":"root ::= \"ok\"""#);
    let json = completion(&resp);
    assert_eq!(json["choices"][0]["message"]["content"], "ok", "{resp}");
    assert_eq!(json["usage"]["completion_tokens"], 2, "{resp}");
}

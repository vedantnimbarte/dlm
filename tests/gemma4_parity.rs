//! dlm against Hugging Face `transformers` on a tiny random Gemma 4 checkpoint.
//!
//! `tests/fixtures/gemma4` holds an f16 checkpoint laid out as Google's
//! multimodal one is, and in `reference.json` the log-probability
//! `transformers` gave each token of a fixed sequence, plus its distribution
//! for the token after it. `tests/fixtures/make_gemma4.py` regenerates them. The
//! fixture has windowed and global layers of different head widths, a
//! shared-KV tail with E2B's double-wide MLP, per-layer inputs, layer scalars
//! and a final-logit softcap, over a sequence three times the window.

use dlm::model::QuantScheme;
use std::path::Path;

struct Fixture {
    ids: Vec<u32>,
    token_logprobs: Vec<f32>,
    next_logprobs: Vec<f32>,
    generator: dlm::generate::Generator<dlm::forward::CpuKernel>,
}

fn fixture() -> Fixture {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let floats = |key: &str| -> Vec<f32> {
        reference[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let ids: Vec<u32> = reference["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let (config, store) = dlm::loader::open_model(&dir, QuantScheme::Fp16).unwrap();
    let generator = dlm::loader::load_generator(&store, &config, ids.len() as u32).unwrap();
    Fixture {
        token_logprobs: floats("token_logprobs"),
        next_logprobs: floats("next_logprobs"),
        ids,
        generator,
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "{what}: entry {i}: dlm {g}, transformers {w}"
        );
    }
}

/// Every token's log-probability, and the distribution after the last, as
/// transformers gives them.
#[test]
fn gemma4_matches_transformers() {
    let f = fixture();
    let (scores, next) = f.generator.score_and_next(&f.ids).unwrap();
    assert_close("token log-probs", &scores, &f.token_logprobs);
    assert_close("next-token log-probs", &next, &f.next_logprobs);
}

/// The same through the decode path, one token at a time: the shared-KV
/// layers then read a source cache that holds exactly the rows up to them.
#[test]
fn gemma4_decoding_matches_transformers() {
    let f = fixture();
    let (scores, next) = f.generator.score_and_next_decoding(&f.ids).unwrap();
    assert_close("token log-probs", &scores, &f.token_logprobs);
    assert_close("next-token log-probs", &next, &f.next_logprobs);
}

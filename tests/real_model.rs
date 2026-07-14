//! End-to-end correctness against a **real checkpoint**.
//!
//! Every other test in this repo compares dlm against itself: streaming vs
//! resident, GPU vs the CPU oracle, speculative vs greedy. All of those stayed
//! green while `dlm generate` emitted multilingual token soup on a real model,
//! because nothing asserted that the output means anything. Three bugs hid in
//! that gap at once:
//!
//! * the Q/K/V biases Qwen2 ships were never loaded (wrong attention),
//! * `rope_scaling` was parsed by nobody (wrong rotation on Llama-3.x),
//! * `generate` silently fell back to a raw *byte* tokenizer for any model that
//!   ships `tokenizer.json` rather than the GPT-2 `vocab.json`+`merges.txt` pair.
//!
//! This test closes that gap: it loads the checkpoint in `models/`, greedily
//! decodes a factual prompt, and asserts the model says what it obviously should.
//! It is the one test that fails if the forward pass is wrong in a way that still
//! produces well-formed tokens.
//!
//! Skipped (not failed) when `models/` is absent, so a fresh clone still runs the
//! suite; CI populates it. Point `DLM_TEST_MODEL` elsewhere to use another model.

use dlm::generate::{GenerationConfig, Sampler};
use dlm::loader::load_model_parts;
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use dlm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("DLM_TEST_MODEL").unwrap_or_else(|_| "models".to_string()),
    );
    (dir.join("config.json").exists() && dir.join("model.safetensors").exists()).then_some(dir)
}

/// Greedily continue `prompt` for `max_new` tokens using the real checkpoint.
fn greedy_continue(dir: &PathBuf, prompt: &str, max_new: usize) -> String {
    let config = ModelConfig::from_path(dir, QuantScheme::Fp16).expect("config.json");
    let store = MmapStore::open_dir(dir).expect("safetensors");
    let tokenizer = BpeTokenizer::from_dir(dir).expect("tokenizer");

    let ids = tokenizer.encode(prompt).expect("encode");
    assert!(
        ids.len() < 32,
        "prompt tokenized to {} ids — that is byte-level fallback, not real BPE",
        ids.len()
    );

    let generator = load_model_parts(&store, &config, 256)
        .expect("load model")
        .into_cpu_generator()
        .expect("cpu generator");

    let gen_cfg = GenerationConfig {
        max_new_tokens: max_new,
        eos_token: config.eos_token_ids.first().copied(),
        sampler: Sampler::Greedy,
    };
    let out = generator.generate(&ids, &gen_cfg).expect("generate");
    tokenizer.decode(&out).expect("decode")
}

/// The load-bearing test: a real model, greedily decoded, must produce the
/// obvious factual continuation. Wrong biases or wrong RoPE still yield
/// syntactically valid tokens — they just stop meaning anything, which only an
/// assertion on the *content* can catch.
#[test]
fn real_model_answers_a_factual_prompt() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no model in ./models (set DLM_TEST_MODEL to override)");
        return;
    };

    let text = greedy_continue(&dir, "The capital of France is", 8);
    assert!(
        text.contains("Paris"),
        "expected the model to answer 'Paris', got {text:?}. \
         The forward pass is producing well-formed but meaningless tokens."
    );
}

/// A second prompt, so the first can't pass by luck, and one that needs several
/// coherent tokens in a row rather than a single lucky argmax.
#[test]
fn real_model_completes_a_known_sequence() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no model in ./models (set DLM_TEST_MODEL to override)");
        return;
    };

    let text = greedy_continue(&dir, "One, two, three, four,", 6);
    assert!(
        text.to_lowercase().contains("five"),
        "expected the count to continue with 'five', got {text:?}"
    );
}

/// The EOS set must include every id `generation_config.json` declares, not just
/// the one in `config.json`. Qwen2.5 names `<|im_end|>` in config.json but both
/// `<|im_end|>` and `<|endoftext|>` in the generation config; missing the second
/// means the model never stops — it runs to the token limit and leaks the raw
/// special token into the reply.
#[test]
fn eos_tokens_include_the_generation_config_set() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no model in ./models (set DLM_TEST_MODEL to override)");
        return;
    };
    if !dir.join("generation_config.json").exists() {
        eprintln!("skipping: model ships no generation_config.json");
        return;
    }

    let config = ModelConfig::from_path(&dir, QuantScheme::Fp16).expect("config");
    let gen: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("generation_config.json")).unwrap()).unwrap();

    let declared: Vec<u32> = match &gen["eos_token_id"] {
        serde_json::Value::Number(n) => vec![n.as_u64().unwrap() as u32],
        serde_json::Value::Array(a) => a.iter().map(|v| v.as_u64().unwrap() as u32).collect(),
        _ => vec![],
    };
    for id in declared {
        assert!(
            config.eos_token_ids.contains(&id),
            "eos id {id} is declared in generation_config.json but missing from the \
             loaded config (eos set: {:?}) — generation will not stop on it",
            config.eos_token_ids
        );
    }
}

/// Greedy decoding is deterministic — the same prompt must give the same tokens.
/// Guards against uninitialized state or nondeterministic reduction order.
#[test]
fn real_model_greedy_is_deterministic() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no model in ./models (set DLM_TEST_MODEL to override)");
        return;
    };

    let a = greedy_continue(&dir, "The capital of France is", 8);
    let b = greedy_continue(&dir, "The capital of France is", 8);
    assert_eq!(a, b, "greedy decoding is not deterministic");
}

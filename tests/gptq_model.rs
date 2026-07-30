//! End-to-end correctness against a **real GPTQ export**.
//!
//! GPTQ decoding was refused outright until this test existed, and for a good
//! reason: the packing is only round-trip testable against dlm's own packer,
//! while real exporters differ in ways that corrupt weights *plausibly*. The
//! decisive one is the zero-point convention — AutoGPTQ stores `zero - 1`, so
//! decoding `(q - z) * scale` with the stored `z` shifts every weight by one
//! scale step. That does not crash and does not produce obvious garbage in a unit
//! test; it produces a model that generates fluent nonsense, which is the worst
//! failure there is.
//!
//! Verified against `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (gptq, 4-bit,
//! group_size 128, desc_act=false, sym=true): every packed zero is `7` where a
//! symmetric 4-bit zero-point must be `8`, confirming the `zero - 1` convention.
//! Decoding without the `+1` on that checkpoint emits
//! `"ensisensisensisensis..."`; with it, the model answers "Paris". So this
//! assertion genuinely discriminates — it is not merely observing that *some*
//! text came out.
//!
//! Skipped (not failed) when the fixture is absent, so a fresh clone still runs
//! the suite. Populate it with:
//!
//! ```sh
//! dlm pull Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 --local-dir models/Qwen2.5-0.5B-GPTQ
//! ```
//!
//! Point `DLM_TEST_GPTQ_MODEL` elsewhere to use another GPTQ checkpoint.

use dlm::generate::{GenerationConfig, Sampler};
use dlm::loader::load_model_parts;
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use dlm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn gptq_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("DLM_TEST_GPTQ_MODEL")
            .unwrap_or_else(|_| "models/Qwen2.5-0.5B-GPTQ".to_string()),
    );
    (dir.join("config.json").exists() && dir.join("model.safetensors").exists()).then_some(dir)
}

/// A real GPTQ checkpoint must decode to weights that still mean something.
#[test]
fn gptq_checkpoint_answers_a_factual_prompt() {
    let Some(dir) = gptq_dir() else {
        eprintln!("skipping: no GPTQ fixture (set DLM_TEST_GPTQ_MODEL to override)");
        return;
    };

    // The scheme comes from the checkpoint itself: a packed 4-bit export has no
    // float weights to reinterpret, and its codes are decoded as they are.
    let config = ModelConfig::from_path(&dir, QuantScheme::Int4).expect("config.json");
    assert!(
        config.packed_quant.is_some(),
        "fixture should be recognized as a packed-quantized checkpoint"
    );

    let store = MmapStore::open_dir(&dir).expect("safetensors");
    let tokenizer = BpeTokenizer::from_dir(&dir).expect("tokenizer");
    let ids = tokenizer
        .encode("The capital of France is")
        .expect("encode");

    let generator = load_model_parts(&store, &config, 128)
        .expect("load GPTQ model")
        .into_cpu_generator()
        .expect("cpu generator");

    let out = generator
        .generate(
            &ids,
            &GenerationConfig {
                max_new_tokens: 8,
                eos_token: config.eos_token_ids.first().copied(),
                sampler: Sampler::Greedy,
            },
        )
        .expect("generate");
    let text = tokenizer.decode(&out).expect("decode");

    assert!(
        text.contains("Paris"),
        "expected the GPTQ model to answer 'Paris', got {text:?}. Weights decoded from the \
         packed 4-bit codes are wrong — check the zero-point convention (AutoGPTQ stores \
         zero-1) and the qweight/qzeros/scales layout."
    );
}

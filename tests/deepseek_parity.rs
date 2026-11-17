//! dlm against Hugging Face `transformers` on tiny random DeepSeek checkpoints.
//!
//! Each `tests/fixtures/deepseek_*` directory holds an f16 checkpoint and, in
//! `reference.json`, the log-probability `transformers` gave each token of a
//! fixed sequence. `tests/fixtures/make_deepseek.py` regenerates them. dlm's
//! other DeepSeek tests check it against itself, which passed while its MLA
//! RoPE paired the wrong dimensions; this is the outside reference.

use dlm::model::QuantScheme;
use std::path::Path;

/// Score the fixture's tokens on the CPU and compare with `transformers`.
fn assert_matches_transformers(fixture: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("reference.json")).unwrap()).unwrap();
    let ids: Vec<u32> = reference["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let (config, store) = dlm::loader::open_model(&dir, QuantScheme::Fp16).unwrap();
    let generator = dlm::loader::load_generator(&store, &config, ids.len() as u32).unwrap();
    let got = generator.score(&ids).unwrap();
    let want = reference["token_logprobs"].as_array().unwrap();
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = w.as_f64().unwrap() as f32;
        assert!(
            (g - w).abs() < 1e-4,
            "{fixture}: token {i}: dlm {g}, transformers {w}"
        );
    }
}

/// DeepSeek-V2's routing over an MLA model: the attention path alone.
#[test]
fn deepseek_v2_matches_transformers() {
    assert_matches_transformers("deepseek_v2");
}

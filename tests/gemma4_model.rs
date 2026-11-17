//! Gemma 4 on a **real** checkpoint: the GGUF llama.cpp users download.
//!
//! Runs only when `DLM_GEMMA4_GGUF` names one (e.g.
//! `unsloth/gemma-4-E4B-it-GGUF`'s `gemma-4-E4B-it-Q4_K_M.gguf`); a 5 GB model
//! is not something to load on every `cargo test`. The tiny fixture in
//! `tests/gemma4_parity.rs` holds the math to transformers; this checks the
//! parts only a real file has: its tokenizer, its tensor names and quantized
//! blocks, and that the answer is right.

use dlm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

fn gguf() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("DLM_GEMMA4_GGUF").ok()?);
    path.exists().then_some(path)
}

/// The GGUF's tokenizer gives the ids transformers' Gemma 4 tokenizer
/// (google/gemma-4-E4B-it `tokenizer.json`) gives, after the BOS the file asks
/// for: spaces, a tab, a doubled space, accents, CJK, an emoji falling back to
/// bytes, and a control token.
#[test]
fn gemma4_gguf_tokenizer_matches_transformers() {
    let Some(path) = gguf() else { return };
    let tok = BpeTokenizer::from_gguf_path(&path).unwrap();
    let ids = tok
        .encode("Hello world,\n\tthe  café — naïve 東京 🙂 <|turn>user")
        .unwrap();
    assert_eq!(
        ids,
        [
            2, 9259, 1902, 236764, 107, 255968, 1437, 138, 123125, 236859, 2192, 120362, 86459,
            57235, 236743, 105, 2364
        ]
    );
}

/// A greedy chat answer on the GPU names the capital.
#[cfg(feature = "cuda-kernels")]
#[test]
fn gemma4_gguf_answers_on_the_gpu() {
    use dlm::model::QuantScheme;
    use dlm::server::engine::{ChatMessage, ChatTemplate};
    let Some(path) = gguf() else { return };
    let tok = BpeTokenizer::from_gguf_path(&path).unwrap();
    let prompt = ChatTemplate::Gemma4.apply(&[ChatMessage {
        role: "user".into(),
        content: "What is the capital of France? Answer in one word.".into(),
        ..Default::default()
    }]);
    let ids = tok.encode(&prompt).unwrap();
    let (config, store) = dlm::loader::open_model(&path, QuantScheme::Fp16).unwrap();
    let g = dlm::loader::load_model_parts(&store, &config, 64)
        .unwrap()
        .into_gpu_generator()
        .unwrap();
    let cfg = dlm::generate::GenerationConfig {
        max_new_tokens: 8,
        ..Default::default()
    };
    let out = tok.decode(&g.generate(&ids, &cfg).unwrap()).unwrap();
    assert!(out.contains("Paris"), "{out:?}");
}

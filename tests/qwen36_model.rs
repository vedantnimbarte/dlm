//! Qwen3.6-35B-A3B from a real GGUF, when one is present: the streamed GPU
//! kernel must score text as the CPU does.
//!
//! ```sh
//! DLM_QWEN36_GGUF=models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
//!   cargo test --release --features cuda-kernels --test qwen36_model -- --nocapture
//! ```
#![cfg(feature = "cuda-kernels")]

use dlm::loader::{build_streaming_generator, build_streaming_gpu_generator, open_model};
use dlm::tokenizer::BpeTokenizer;
use dlm::QuantScheme;
use std::path::PathBuf;

const TEXT: &str = "The capital of France is Paris. The capital of Germany is Berlin.";

#[test]
fn streamed_gpu_scores_like_the_cpu() {
    let Some(path) = std::env::var_os("DLM_QWEN36_GGUF").map(PathBuf::from) else {
        eprintln!("skipping: set DLM_QWEN36_GGUF");
        return;
    };
    let ids = BpeTokenizer::from_gguf_path(&path)
        .unwrap()
        .encode(TEXT)
        .unwrap();
    let cpu = {
        let (config, store) = open_model(&path, QuantScheme::Int4).unwrap();
        build_streaming_generator(store, &config, 128, 0, 1, false, 4 << 30)
            .unwrap()
            .score(&ids)
            .unwrap()
    };
    let gpu = {
        let (config, store) = open_model(&path, QuantScheme::Int4).unwrap();
        build_streaming_gpu_generator(store, &config, 128, 8, 4 << 30, 1 << 30, false)
            .unwrap()
            .score(&ids)
            .unwrap()
    };
    eprintln!("cpu {cpu:?}\ngpu {gpu:?}");
    let worst = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        gpu.iter().all(|x| x.is_finite()),
        "GPU log-probs are not finite"
    );
    assert!(worst < 0.15, "GPU log-probs off by up to {worst}");
}

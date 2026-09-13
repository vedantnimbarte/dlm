//! GPT-2 end to end on the **real** checkpoint, on every path that can run it.
//!
//! GPT-2 is the family least like the rest: learned position embeddings instead
//! of RoPE, LayerNorm with biases, an ungated MLP, biases on every projection.
//! Each of those lives in a different place — the generator's head, the host
//! block, the device kernels — and each path assembles its own. All four
//! non-resident paths once produced fluent garbage while the CPU resident path
//! answered correctly, because only that builder attached `wpe` and the
//! LayerNorm head, and the GPU kernels ran GPT-2 as a Llama block.
//!
//! So this pins every path to the same greedy continuation of a factual prompt.
//! Skipped when the checkpoint is absent; fetch it with
//! `dlm pull openai-community/gpt2 --local-dir models/gpt2`, or point
//! `DLM_GPT2_MODEL` at one.

use dlm::forward::ComputeKernel;
use dlm::generate::{GenerationConfig, Generator, Sampler};
use dlm::loader::{build_streaming_generator, load_model_parts};
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use dlm::tokenizer::BpeTokenizer;
use std::path::PathBuf;

const PROMPT: &str = "The capital of France is Paris. The capital of Germany is";

fn model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(
        std::env::var("DLM_GPT2_MODEL").unwrap_or_else(|_| "models/gpt2".to_string()),
    );
    dir.join("model.safetensors").exists().then_some(dir)
}

fn setup(dir: &PathBuf) -> (ModelConfig, BpeTokenizer, Vec<u32>) {
    let config = ModelConfig::from_path(dir, QuantScheme::Fp16).expect("config.json");
    let tokenizer = BpeTokenizer::from_dir(dir).expect("tokenizer");
    let ids = tokenizer.encode(PROMPT).expect("encode");
    (config, tokenizer, ids)
}

fn greedy<K: ComputeKernel>(g: &Generator<K>, ids: &[u32]) -> Vec<u32> {
    let cfg = GenerationConfig {
        max_new_tokens: 8,
        eos_token: None,
        sampler: Sampler::Greedy,
    };
    g.generate(ids, &cfg).expect("generate")
}

/// The reference: the CPU resident path, checked against what the prompt means.
fn reference(dir: &PathBuf) -> Option<Vec<u32>> {
    let (config, tokenizer, ids) = setup(dir);
    let store = MmapStore::open_dir(dir).unwrap();
    let g = load_model_parts(&store, &config, 128)
        .unwrap()
        .into_cpu_generator()
        .unwrap();
    let out = greedy(&g, &ids);
    let text = tokenizer.decode(&out).unwrap();
    assert!(
        text.trim_start().starts_with("Berlin"),
        "GPT-2 on the CPU should continue with Berlin, got {text:?}"
    );
    Some(out)
}

#[test]
fn gpt2_streamed_on_cpu_matches_resident() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no GPT-2 checkpoint");
        return;
    };
    let want = reference(&dir).unwrap();
    let (config, _, ids) = setup(&dir);
    let store = MmapStore::open_dir(&dir).unwrap();
    // 4 of 12 layers resident: most of the model streams.
    let g = build_streaming_generator(store, &config, 128, 4, 0, false, 0).unwrap();
    assert_eq!(
        greedy(&g, &ids),
        want,
        "streamed CPU diverged from resident"
    );
}

#[cfg(feature = "cuda-kernels")]
#[test]
fn gpt2_on_gpu_matches_cpu() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no GPT-2 checkpoint");
        return;
    };
    let want = reference(&dir).unwrap();
    let (config, _, ids) = setup(&dir);

    let store = MmapStore::open_dir(&dir).unwrap();
    let resident = load_model_parts(&store, &config, 128)
        .unwrap()
        .into_gpu_generator()
        .unwrap();
    assert_eq!(
        greedy(&resident, &ids),
        want,
        "resident GPU diverged from CPU"
    );

    let store = MmapStore::open_dir(&dir).unwrap();
    let streamed =
        dlm::loader::build_streaming_gpu_generator(store, &config, 128, 4, 0, 0).unwrap();
    assert_eq!(
        greedy(&streamed, &ids),
        want,
        "streamed GPU diverged from CPU"
    );
}

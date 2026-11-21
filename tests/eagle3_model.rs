//! EAGLE-3 speculation against a real target and head, when both are present.
//!
//! ```sh
//! DLM_EAGLE3_TARGET=models/Qwen3-1.7B \
//! DLM_EAGLE3_HEAD=models/qwen3-1.7b-eagle3 \
//! DLM_EAGLE3_REF=eagle3_ref.json \
//!   cargo test --release --features cuda-kernels --test eagle3_model
//! ```
//!
//! The head is AngelSlim's `Qwen3-1.7B_eagle3` with its `pytorch_model.bin`
//! saved as `model.safetensors`. `DLM_EAGLE3_REF` is optional: the JSON
//! `tools/eagle3_ref.py` writes from transformers and a plain-torch EAGLE-3
//! layer.
#![cfg(feature = "cuda-kernels")]

use dlm::forward::eagle3::{aux_layers, Eagle3Head};
use dlm::forward::ComputeKernel;
use dlm::loader::load_model_parts;
use dlm::speculative::SpeculativeSession;
use dlm::storage::MmapStore;
use dlm::tokenizer::BpeTokenizer;
use dlm::{GenerationConfig, ModelConfig, QuantScheme, Sampler};
use std::path::PathBuf;

fn dirs() -> Option<(PathBuf, PathBuf)> {
    Some((
        std::env::var_os("DLM_EAGLE3_TARGET")?.into(),
        std::env::var_os("DLM_EAGLE3_HEAD")?.into(),
    ))
}

const PROMPT: &str = "<|im_start|>user\nWrite a short Python function that returns the n-th Fibonacci number, then explain how it works in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Speculation must not change greedy output, and a working head must guess
/// well: on this prompt AngelSlim's head keeps 78% of single proposals and
/// about half over three (later guesses ride on earlier ones). A head fed the
/// wrong layers, or features one position off, keeps a few percent.
#[test]
fn eagle3_greedy_matches_plain_and_mostly_accepts() {
    let Some((target, head_dir)) = dirs() else {
        eprintln!("skipping: set DLM_EAGLE3_TARGET and DLM_EAGLE3_HEAD");
        return;
    };
    let config = ModelConfig::from_path(&target, QuantScheme::Int4).unwrap();
    let tokenizer = BpeTokenizer::from_dir(&target).unwrap();
    let prompt = tokenizer.encode(PROMPT).unwrap();
    let store = MmapStore::open_dir(&target).unwrap();
    let g = load_model_parts(&store, &config, 1024)
        .unwrap()
        .into_gpu_generator()
        .unwrap();
    let n = 96;
    let plain = g
        .generate(
            &prompt,
            &GenerationConfig {
                max_new_tokens: n,
                eos_token: None,
                sampler: Sampler::Greedy,
            },
        )
        .unwrap();
    let head = Eagle3Head::load(
        &head_dir,
        QuantScheme::Int4,
        config.hidden_size as usize,
        config.vocab_size as usize,
    )
    .unwrap();
    for gamma in [1, 3, 5] {
        let mut s =
            SpeculativeSession::eagle3(&g, &head, gamma, &prompt, Sampler::Greedy, 1024).unwrap();
        let mut out = Vec::new();
        while out.len() < n {
            out.extend(s.step(n - out.len()).unwrap());
        }
        assert_eq!(
            out, plain,
            "gamma {gamma}: speculation changed greedy output"
        );
        let rate = s.accepted() as f64 / s.proposed() as f64;
        eprintln!(
            "gamma {gamma}: accepted {:.0}% of {}",
            rate * 100.0,
            s.proposed()
        );
        assert!(
            rate > 0.3,
            "gamma {gamma}: accepted only {:.0}%",
            rate * 100.0
        );
    }
}

/// The head's prediction at every position of a passage, fed the target's own
/// features, against a plain-torch EAGLE-3 layer over transformers' hidden
/// states. Positions where the reference's top two logits are within 1 of each
/// other are skipped: bf16 weights and the int8 target can swap those.
#[test]
fn eagle3_head_matches_reference() {
    let (Some((target, head_dir)), Some(reference)) = (dirs(), std::env::var_os("DLM_EAGLE3_REF"))
    else {
        eprintln!("skipping: set DLM_EAGLE3_TARGET, DLM_EAGLE3_HEAD and DLM_EAGLE3_REF");
        return;
    };
    let r: serde_json::Value = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let tokens: Vec<u32> = r["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    let draft: Vec<u32> = r["draft"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    let margin: Vec<f64> = r["margin"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_f64().unwrap())
        .collect();
    let config = ModelConfig::from_path(&target, QuantScheme::Int8).unwrap();
    let store = MmapStore::open_dir(&target).unwrap();
    let g = load_model_parts(&store, &config, 1024)
        .unwrap()
        .into_gpu_generator()
        .unwrap();
    let layers = aux_layers(config.num_layers);
    assert!(g.kernel().capture_next(&layers));
    g.score(&tokens).unwrap();
    let feats = g.kernel().take_capture();
    let h = config.hidden_size as usize;
    assert_eq!(
        feats.len(),
        tokens.len() * 3 * h,
        "one feature row per token"
    );
    let head =
        Eagle3Head::load(&head_dir, QuantScheme::Fp16, h, config.vocab_size as usize).unwrap();
    let mut seq = head.new_seq(tokens.len()).unwrap();
    let (mut checked, mut same) = (0, 0);
    for p in 0..tokens.len() - 1 {
        let embed = g.embed(tokens[p + 1], p + 1).unwrap();
        let out = head
            .step(
                &mut seq,
                &embed,
                Some(&feats[p * 3 * h..(p + 1) * 3 * h]),
                p,
            )
            .unwrap();
        if margin[p] > 1.0 {
            checked += 1;
            same += (head.argmax(&out).unwrap() == draft[p]) as usize;
        }
    }
    eprintln!("draft agrees with the reference at {same}/{checked} confident positions");
    assert!(checked > 20);
    assert!(same * 100 >= checked * 95, "{same}/{checked}");
}

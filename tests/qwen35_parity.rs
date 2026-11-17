//! dlm against Hugging Face `transformers` on a tiny random Qwen3.5/3.6-MoE
//! checkpoint, and the rules a recurrent state imposes on the KV machinery.
//!
//! `tests/fixtures/qwen35_moe` holds an f16 checkpoint laid out like the real
//! Qwen/Qwen3.6-35B-A3B (nested config, `model.language_model.*`, stacked
//! experts, a vision tensor, an MTP head) and, in `reference.json`, the
//! log-probability `transformers` gave each token of an 80-token sequence.
//! `tests/fixtures/make_qwen35.py` regenerates it. The GGUF copy
//! (`qwen35_moe.gguf`, from `make_qwen35_gguf.py`) carries the same weights as
//! llama.cpp's converter writes them.

use dlm::batching::BatchScheduler;
use dlm::generate::{Generator, Sampler};
use dlm::model::QuantScheme;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_moe")
}

fn reference() -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let r: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture().join("reference.json")).unwrap()).unwrap();
    let floats = |k: &str| -> Vec<f32> {
        r[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    };
    let ids = r["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    (ids, floats("token_logprobs"), floats("next_logprobs"))
}

fn cpu_generator(path: &Path) -> Generator<dlm::forward::CpuKernel> {
    let (config, store) = dlm::loader::open_model(path, QuantScheme::Fp16).unwrap();
    dlm::loader::load_generator(&store, &config, 256).unwrap()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < tol,
            "{what}: entry {i}: dlm {g}, transformers {w}"
        );
    }
}

/// The whole sequence as one prefill (the layer-by-layer order a prompt takes),
/// and the next-token distribution after it.
#[test]
fn qwen35_moe_prefill_matches_transformers() {
    let (ids, want, next) = reference();
    let (got, got_next) = cpu_generator(&fixture()).score_and_next(&ids).unwrap();
    assert_close(&got, &want, 1e-4, "prefill");
    assert_close(&got_next, &next, 1e-4, "next-token distribution");
}

/// Every token through the decode path, one at a time, as generation runs it.
#[test]
fn qwen35_moe_decoding_matches_transformers() {
    let (ids, want, _) = reference();
    let (got, _) = cpu_generator(&fixture())
        .score_and_next_decoding(&ids)
        .unwrap();
    assert_close(&got, &want, 1e-4, "decode");
}

/// The fixture's config reads as the real one does: the nested text model,
/// alternating layers, partial RoPE, the gate, (1 + w) norms, the MTP count.
#[test]
fn qwen35_fixture_config_and_mtp_head() {
    let (config, store) = dlm::loader::open_model(&fixture(), QuantScheme::Fp16).unwrap();
    let l = config.linear_attn.expect("hybrid");
    assert_eq!((l.num_k_heads, l.num_v_heads, l.k_head_dim), (2, 4, 8));
    assert!(!l.v_heads_tiled, "transformers' grouped value heads");
    assert_eq!(config.attention_layers(), 2);
    assert_eq!(config.rotary_dim, Some(4));
    assert!(config.attn_output_gate && config.norm_add_one);
    assert_eq!(config.mtp_layers, 1);
    assert!(
        store
            .locate("model.visual.blocks.0.attn.qkv.weight")
            .is_none(),
        "the vision tower is not part of the text model"
    );
    let mtp = dlm::loader::load_mtp(&store, &config)
        .unwrap()
        .expect("MTP head");
    assert_eq!(mtp.layers.len(), 1);
    assert_eq!(mtp.fc.len(), 32 * 64);
    assert!(mtp.layers[0].attn_gate.is_some() && mtp.layers[0].linear.is_none());
}

/// The same weights as llama.cpp's converter writes them (`qwen35moe`: value
/// heads tiled, `-exp(A_log)`, the `+1` baked into the norms, the experts
/// split and stacked) score the same.
#[test]
fn qwen35_moe_gguf_matches_transformers() {
    let gguf = fixture().join("qwen35_moe.gguf");
    let (ids, want, _) = reference();
    let (config, _) = dlm::loader::open_model(&gguf, QuantScheme::Fp16).unwrap();
    assert!(config.linear_attn.unwrap().v_heads_tiled);
    assert!(!config.norm_add_one, "the converter added the +1");
    assert_eq!(config.mtp_layers, 1);
    let got = cpu_generator(&gguf).score(&ids).unwrap();
    assert_close(&got, &want, 1e-4, "gguf");
}

/// A snapshot taken at a position resumes exactly: the recurrent state is part
/// of what it carries.
#[test]
fn snapshot_at_the_position_resumes_exactly() {
    let gen = cpu_generator(&fixture());
    let (ids, _, _) = reference();
    let (prefix, prompt) = (&ids[..30], &ids[..40]);
    let mut full = gen.start_session(prompt, Sampler::Greedy).unwrap();
    let want: Vec<u32> = (0..5).map(|_| full.step().unwrap()).collect();

    let snap = gen
        .start_session(prefix, Sampler::Greedy)
        .unwrap()
        .snapshot();
    assert_eq!(snap.position(), prefix.len());
    let mut resumed = gen.resume_session(snap, prompt, Sampler::Greedy).unwrap();
    let got: Vec<u32> = (0..5).map(|_| resumed.step().unwrap()).collect();
    assert_eq!(got, want);
}

/// A recurrent state cannot be cut back to an earlier position: a snapshot's
/// shorter prefix refuses to run rather than continuing from the wrong state.
#[test]
fn a_shorter_prefix_of_a_recurrent_state_is_refused() {
    let gen = cpu_generator(&fixture());
    let (ids, _, _) = reference();
    let snap = gen
        .start_session(&ids[..30], Sampler::Greedy)
        .unwrap()
        .snapshot();
    let cut = snap.prefix(10);
    let err = gen
        .resume_session(cut, &ids[..20], Sampler::Greedy)
        .err()
        .expect("resuming a cut recurrent state must fail");
    assert!(err.to_string().contains("rolled back"), "{err}");
}

/// Speculation rolls back rejected tokens, which a recurrent model cannot: the
/// session is refused up front.
#[test]
fn speculative_decoding_is_refused() {
    let gen = cpu_generator(&fixture());
    assert!(gen.recurrent());
    let (ids, _, _) = reference();
    let err = dlm::speculative::SpeculativeSession::ngram(&gen, 4, &ids[..10], Sampler::Greedy)
        .err()
        .expect("speculation on a recurrent model must be refused");
    assert!(err.to_string().contains("speculative"), "{err}");
}

/// The prefix cache stays off for a recurrent model (it resumes from prefixes
/// of what it holds), and batching still gives each request its own state.
#[test]
fn batching_keeps_states_apart_and_skips_the_prefix_cache() {
    let gen = cpu_generator(&fixture());
    let (ids, _, _) = reference();
    let prompts = [ids[..12].to_vec(), ids[..20].to_vec(), ids[5..25].to_vec()];
    let alone: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let mut s = gen.start_session(p, Sampler::Greedy).unwrap();
            (0..6).map(|_| s.step().unwrap()).collect()
        })
        .collect();
    let mut sched = BatchScheduler::new(&gen, 3).with_prefix_cache(8);
    for (i, p) in prompts.iter().enumerate() {
        sched.submit(i as u64, p.clone(), 6, vec![]).unwrap();
    }
    let mut done = sched.run().unwrap();
    done.sort_by_key(|f| f.id);
    for (f, want) in done.iter().zip(&alone) {
        assert_eq!(&f.tokens, want, "request {}", f.id);
    }
    assert_eq!(sched.resume_hits(), 0);
}

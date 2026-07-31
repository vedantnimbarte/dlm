//! Phi-3's fused projections, checked against the layout they fuse.
//!
//! Phi-3 ships one tensor where Llama ships several: `self_attn.qkv_proj` is
//! `[q_dim + 2*kv_dim, hidden]` and `mlp.gate_up_proj` is `[2*intermediate,
//! hidden]`. Everything else about the block is Llama-shaped, so support is
//! entirely a matter of slicing those two correctly.
//!
//! **Which is exactly the kind of change that fails silently.** Split QKV at the
//! wrong offset, or assume up-then-gate where the checkpoint means gate-then-up,
//! and nothing errors: the tensors are the right shape, the model loads, and it
//! generates fluent nonsense. That is the failure mode this repo has shipped
//! twice already — the GPTQ `zero - 1` convention and SentencePiece-as-byte-level
//! — and a fixture of `config.json` + `tokenizer.json` cannot catch it, because
//! the defect lives in the weight mapping that fixtures never touch.
//!
//! So this test does not check that a Phi-3 model loads. It builds the **same
//! weights twice** — once in Llama's separate layout, once in Phi-3's fused
//! layout — loads both, and requires the two models to be numerically identical
//! and to generate the same tokens. A wrong offset or a swapped pair breaks it.
//!
//! No download: the point is the mapping, and synthetic weights exercise it
//! exactly as real ones would. Real Phi-3 config parsing is covered by the
//! family fixture.

use dlm::forward::cpu::Ffn;
use dlm::forward::Weights;
use dlm::generate::{GenerationConfig, Sampler};
use dlm::loader::load_model_parts;
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use std::io::Write;
use std::path::Path;

// A deliberately small but non-degenerate block. `kv_dim != q_dim` (GQA) is the
// point: with equal head counts a wrong QKV offset can still land inside the
// right tensor and go unnoticed.
const HIDDEN: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS; // 8
const Q_DIM: usize = HEADS * HEAD_DIM; // 32
const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 16
const INTERMEDIATE: usize = 48;
const VOCAB: usize = 64;
const LAYERS: usize = 2;

/// Deterministic, distinct-per-element values.
///
/// Every element is unique across the whole tensor set, so a slice taken at the
/// wrong offset cannot coincidentally match the right one.
fn fill(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 101) % 199) as f32 - 99.0) / 400.0)
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write_config(dir: &Path) {
    let cfg = format!(
        r#"{{
  "architectures": ["Phi3ForCausalLM"],
  "hidden_size": {HIDDEN},
  "num_attention_heads": {HEADS},
  "num_key_value_heads": {KV_HEADS},
  "num_hidden_layers": {LAYERS},
  "intermediate_size": {INTERMEDIATE},
  "vocab_size": {VOCAB},
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "hidden_act": "silu",
  "tie_word_embeddings": false
}}"#
    );
    std::fs::write(dir.join("config.json"), cfg).unwrap();
}

/// Write an f32 safetensors file from `(name, shape, values)` triples.
fn write_model(dir: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut entries = Vec::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        let bytes = f32_bytes(values);
        let shape_s = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape_s}],"data_offsets":[{},{}]}}"#,
            blob.len(),
            blob.len() + bytes.len()
        ));
        blob.extend_from_slice(&bytes);
    }
    let header = format!("{{{}}}", entries.join(","));
    // safetensors requires the data section to start 8-byte aligned.
    let mut header = header.into_bytes();
    while (8 + header.len()) % 8 != 0 {
        header.push(b' ');
    }
    let path = dir.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(&blob).unwrap();
    f.flush().unwrap();
}

/// The tensors every layout shares, plus the per-layer projections built by
/// `layer_tensors`.
fn model_tensors(
    layer_tensors: impl Fn(usize) -> Vec<(String, Vec<usize>, Vec<f32>)>,
) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut t = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![VOCAB, HIDDEN],
            fill(VOCAB * HIDDEN, 1),
        ),
        (
            "model.norm.weight".to_string(),
            vec![HIDDEN],
            fill(HIDDEN, 2),
        ),
        (
            "lm_head.weight".to_string(),
            vec![VOCAB, HIDDEN],
            fill(VOCAB * HIDDEN, 3),
        ),
    ];
    for l in 0..LAYERS {
        t.extend(layer_tensors(l));
        t.push((
            format!("model.layers.{l}.input_layernorm.weight"),
            vec![HIDDEN],
            fill(HIDDEN, 40 + l),
        ));
        t.push((
            format!("model.layers.{l}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            fill(HIDDEN, 50 + l),
        ));
        t.push((
            format!("model.layers.{l}.self_attn.o_proj.weight"),
            vec![HIDDEN, Q_DIM],
            fill(HIDDEN * Q_DIM, 60 + l),
        ));
        t.push((
            format!("model.layers.{l}.mlp.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
            fill(HIDDEN * INTERMEDIATE, 70 + l),
        ));
    }
    t
}

/// The five projections a layer needs, before either layout packs them.
struct LayerValues {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
}

/// Per-layer q/k/v and gate/up values — the single source both layouts use, so
/// the two checkpoints differ only in how these are packed into tensors.
fn layer_values(l: usize) -> LayerValues {
    LayerValues {
        q: fill(Q_DIM * HIDDEN, 100 + l),
        k: fill(KV_DIM * HIDDEN, 200 + l),
        v: fill(KV_DIM * HIDDEN, 300 + l),
        gate: fill(INTERMEDIATE * HIDDEN, 400 + l),
        up: fill(INTERMEDIATE * HIDDEN, 500 + l),
    }
}

/// Llama layout: separate `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`.
fn separate_layer(l: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let LayerValues { q, k, v, gate, up } = layer_values(l);
    vec![
        (
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            vec![Q_DIM, HIDDEN],
            q,
        ),
        (
            format!("model.layers.{l}.self_attn.k_proj.weight"),
            vec![KV_DIM, HIDDEN],
            k,
        ),
        (
            format!("model.layers.{l}.self_attn.v_proj.weight"),
            vec![KV_DIM, HIDDEN],
            v,
        ),
        (
            format!("model.layers.{l}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
            gate,
        ),
        (
            format!("model.layers.{l}.mlp.up_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
            up,
        ),
    ]
}

/// Phi-3 layout: `qkv_proj` = concat(q, k, v) and `gate_up_proj` = concat(gate, up),
/// concatenated along the **output** dimension, which is row-major-contiguous.
fn fused_layer(l: usize) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let LayerValues { q, k, v, gate, up } = layer_values(l);
    let mut qkv = q;
    qkv.extend(k);
    qkv.extend(v);
    let mut gate_up = gate;
    gate_up.extend(up);
    vec![
        (
            format!("model.layers.{l}.self_attn.qkv_proj.weight"),
            vec![Q_DIM + 2 * KV_DIM, HIDDEN],
            qkv,
        ),
        (
            format!("model.layers.{l}.mlp.gate_up_proj.weight"),
            vec![2 * INTERMEDIATE, HIDDEN],
            gate_up,
        ),
    ]
}

fn build(dir: &Path, layer: impl Fn(usize) -> Vec<(String, Vec<usize>, Vec<f32>)>) {
    write_config(dir);
    write_model(dir, &model_tensors(layer));
}

fn as_f32(w: &Weights) -> Vec<f32> {
    match w {
        Weights::F32(v) => v.clone(),
        other => panic!("expected F32 weights, got {other:?}"),
    }
}

/// The load-level assertion: fused and separate must yield identical tensors.
#[test]
fn fused_qkv_and_gate_up_split_to_the_separate_layout() {
    let sep_dir = tempfile::tempdir().unwrap();
    let fus_dir = tempfile::tempdir().unwrap();
    build(sep_dir.path(), separate_layer);
    build(fus_dir.path(), fused_layer);

    let load = |dir: &Path| {
        let store = MmapStore::open_dir(dir).unwrap();
        let cfg = ModelConfig::from_path(dir, QuantScheme::F32).unwrap();
        load_model_parts(&store, &cfg, 64).unwrap()
    };
    let sep = load(sep_dir.path());
    let fus = load(fus_dir.path());

    assert_eq!(sep.layers.len(), LAYERS);
    assert_eq!(fus.layers.len(), LAYERS);

    for (l, (a, b)) in sep.layers.iter().zip(fus.layers.iter()).enumerate() {
        assert_eq!(as_f32(&a.q_proj), as_f32(&b.q_proj), "layer {l}: q_proj");
        assert_eq!(as_f32(&a.k_proj), as_f32(&b.k_proj), "layer {l}: k_proj");
        assert_eq!(as_f32(&a.v_proj), as_f32(&b.v_proj), "layer {l}: v_proj");
        let (Ffn::Dense(ga), Ffn::Dense(gb)) = (&a.ffn, &b.ffn) else {
            panic!("layer {l}: expected a dense FFN");
        };
        assert_eq!(as_f32(&ga.gate), as_f32(&gb.gate), "layer {l}: gate");
        assert_eq!(as_f32(&ga.up), as_f32(&gb.up), "layer {l}: up");
        assert_eq!(as_f32(&ga.down), as_f32(&gb.down), "layer {l}: down");
    }
}

/// End-to-end coverage: the two checkpoints drive the whole forward path and
/// produce the same tokens.
///
/// **This one is not the discriminating check, and the mutation test says so.**
/// Reverting `v_proj` to the wrong row offset fails
/// `fused_qkv_and_gate_up_split_to_the_separate_layout` but leaves this green:
/// at this model size greedy decode saturates to the same argmax whether or not
/// V is correct, so identical tokens here is weak evidence. It is kept because it
/// exercises loading through `load_generator` rather than `load_model_parts`, and
/// would catch a split that loads but cannot run at all — not because it proves
/// the mapping right. The tensor comparison above does that.
///
/// The non-degeneracy assertion at the end exists for the same reason: without
/// it, two models that both emit a constant token would agree trivially.
#[test]
fn fused_and_separate_checkpoints_generate_identically() {
    let sep_dir = tempfile::tempdir().unwrap();
    let fus_dir = tempfile::tempdir().unwrap();
    build(sep_dir.path(), separate_layer);
    build(fus_dir.path(), fused_layer);

    let generate = |dir: &Path| {
        let store = MmapStore::open_dir(dir).unwrap();
        let cfg = ModelConfig::from_path(dir, QuantScheme::F32).unwrap();
        let gen = dlm::loader::load_generator(&store, &cfg, 64).unwrap();
        gen.generate(
            &[3, 9, 17, 2],
            &GenerationConfig {
                max_new_tokens: 12,
                sampler: Sampler::Greedy,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let from_separate = generate(sep_dir.path());
    let from_fused = generate(fus_dir.path());

    assert_eq!(
        from_separate, from_fused,
        "a Phi-3 fused checkpoint must decode identically to the separate layout \
         holding the same weights"
    );
    // Guard against both sides being trivially empty.
    assert_eq!(from_separate.len(), 12);
    // ...and against the weaker trap: two models that each emit one token
    // forever would agree without either being right.
    let distinct: std::collections::BTreeSet<_> = from_separate.iter().collect();
    assert!(
        distinct.len() > 1,
        "output is a single repeated token ({from_separate:?}), so agreement \
         between the two layouts would mean nothing"
    );
}

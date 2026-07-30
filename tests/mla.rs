//! End-to-end Multi-head Latent Attention (DeepSeek) load + run on the CPU path.
//!
//! Proves a synthetic MLA checkpoint (low-rank Q/KV projections, compressed-latent
//! KV cache, decoupled RoPE) loads through the real loader and generates
//! deterministic tokens — the internal-consistency check for the MLA oracle
//! (the CPU↔GPU parity check lives in `tests/gpu_parity.rs`).

use dlm::generate::{GenerationConfig, Sampler};
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use std::io::Write;

fn write_f32_model(dir: &std::path::Path, tensors: &[(String, Vec<f32>)]) {
    let mut entries = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut offset = 0usize;
    for (name, values) in tensors {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        entries.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[{offset},{}]}}"#,
            values.len(),
            offset + bytes.len()
        ));
        data.extend_from_slice(&bytes);
        offset += bytes.len();
    }
    let header = format!("{{{}}}", entries.join(","));
    let path = dir.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
    f.flush().unwrap();
}

// Small MLA geometry.
const H: usize = 16;
const NH: usize = 2;
const LAYERS: usize = 2;
const VOCAB: usize = 8;
const INTER: usize = 16;
const KV_LORA: usize = 8;
const Q_LORA: usize = 12;
const NOPE: usize = 4;
const ROPE: usize = 4; // even (RoPE)
const VDIM: usize = 4;
const QK: usize = NOPE + ROPE;

fn fill(seed: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + seed) % 17) as f32 - 8.0) * 0.02).collect()
}

fn write_mla_checkpoint(dir: &std::path::Path) -> ModelConfig {
    let mut t: Vec<(String, Vec<f32>)> = Vec::new();
    t.push(("model.embed_tokens.weight".into(), fill(1, VOCAB * H)));
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}.");
        let s = i + 2;
        let a = |name: &str, n: usize, seed: usize| (format!("{p}{name}"), fill(seed, n));
        // MLA projections.
        t.push(a("self_attn.q_a_proj.weight", Q_LORA * H, s));
        t.push((format!("{p}self_attn.q_a_layernorm.weight"), vec![1.0; Q_LORA]));
        t.push(a("self_attn.q_b_proj.weight", NH * QK * Q_LORA, s + 1));
        t.push(a("self_attn.kv_a_proj_with_mqa.weight", (KV_LORA + ROPE) * H, s + 2));
        t.push((format!("{p}self_attn.kv_a_layernorm.weight"), vec![1.0; KV_LORA]));
        t.push(a("self_attn.kv_b_proj.weight", NH * (NOPE + VDIM) * KV_LORA, s + 3));
        t.push(a("self_attn.o_proj.weight", H * NH * VDIM, s + 4));
        t.push((format!("{p}input_layernorm.weight"), vec![1.0; H]));
        t.push((format!("{p}post_attention_layernorm.weight"), vec![1.0; H]));
        // Dense FFN.
        t.push(a("mlp.gate_proj.weight", INTER * H, s + 5));
        t.push(a("mlp.up_proj.weight", INTER * H, s + 6));
        t.push(a("mlp.down_proj.weight", H * INTER, s + 7));
    }
    t.push(("model.norm.weight".into(), vec![1.0; H]));
    t.push(("lm_head.weight".into(), fill(3, VOCAB * H)));
    write_f32_model(dir, &t);

    let config_json = format!(
        r#"{{"hidden_size":{H},"num_attention_heads":{NH},"num_key_value_heads":{NH},
            "num_hidden_layers":{LAYERS},"vocab_size":{VOCAB},"intermediate_size":{INTER},
            "kv_lora_rank":{KV_LORA},"q_lora_rank":{Q_LORA},"qk_nope_head_dim":{NOPE},
            "qk_rope_head_dim":{ROPE},"v_head_dim":{VDIM}}}"#
    );
    std::fs::write(dir.join("config.json"), &config_json).unwrap();
    ModelConfig::from_json_bytes(config_json.as_bytes(), QuantScheme::Fp16).unwrap()
}

#[test]
fn mla_loads_and_runs_deterministically() {
    let tmp = tempfile::tempdir().unwrap();
    let config = write_mla_checkpoint(tmp.path());
    assert!(config.mla.is_some(), "config should be detected as MLA");

    let gen_cfg = GenerationConfig {
        max_new_tokens: 8,
        eos_token: None,
        sampler: Sampler::Greedy,
    };
    let prompt = [1u32, 2, 3];

    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let gen = dlm::loader::load_generator(&store, &config, 32).unwrap();
    let out = gen.generate(&prompt, &gen_cfg).unwrap();
    assert_eq!(out.len(), 8, "MLA model produced no tokens");
    // Greedy decode of the same prompt repeats exactly.
    assert_eq!(out, gen.generate(&prompt, &gen_cfg).unwrap(), "MLA decode not deterministic");
}

/// The compressed-latent KV path with no query low-rank (direct `q_proj`) also
/// loads and runs — exercising the other query branch.
#[test]
fn mla_without_q_lora_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t: Vec<(String, Vec<f32>)> = Vec::new();
    t.push(("model.embed_tokens.weight".into(), fill(1, VOCAB * H)));
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}.");
        let s = i + 2;
        let a = |name: &str, n: usize, seed: usize| (format!("{p}{name}"), fill(seed, n));
        t.push(a("self_attn.q_proj.weight", NH * QK * H, s)); // direct query projection
        t.push(a("self_attn.kv_a_proj_with_mqa.weight", (KV_LORA + ROPE) * H, s + 2));
        t.push((format!("{p}self_attn.kv_a_layernorm.weight"), vec![1.0; KV_LORA]));
        t.push(a("self_attn.kv_b_proj.weight", NH * (NOPE + VDIM) * KV_LORA, s + 3));
        t.push(a("self_attn.o_proj.weight", H * NH * VDIM, s + 4));
        t.push((format!("{p}input_layernorm.weight"), vec![1.0; H]));
        t.push((format!("{p}post_attention_layernorm.weight"), vec![1.0; H]));
        t.push(a("mlp.gate_proj.weight", INTER * H, s + 5));
        t.push(a("mlp.up_proj.weight", INTER * H, s + 6));
        t.push(a("mlp.down_proj.weight", H * INTER, s + 7));
    }
    t.push(("model.norm.weight".into(), vec![1.0; H]));
    t.push(("lm_head.weight".into(), fill(3, VOCAB * H)));
    write_f32_model(tmp.path(), &t);
    let config_json = format!(
        r#"{{"hidden_size":{H},"num_attention_heads":{NH},"num_key_value_heads":{NH},
            "num_hidden_layers":{LAYERS},"vocab_size":{VOCAB},"intermediate_size":{INTER},
            "kv_lora_rank":{KV_LORA},"qk_nope_head_dim":{NOPE},
            "qk_rope_head_dim":{ROPE},"v_head_dim":{VDIM}}}"#
    );
    std::fs::write(tmp.path().join("config.json"), &config_json).unwrap();
    let config = ModelConfig::from_json_bytes(config_json.as_bytes(), QuantScheme::Fp16).unwrap();
    assert!(config.mla.is_some() && config.mla.unwrap().q_lora_rank.is_none());

    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let gen = dlm::loader::load_generator(&store, &config, 32).unwrap();
    let out = gen
        .generate(&[1, 2, 3], &GenerationConfig { max_new_tokens: 5, eos_token: None, sampler: Sampler::Greedy })
        .unwrap();
    assert_eq!(out.len(), 5);
}

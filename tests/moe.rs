//! End-to-end Mixture-of-Experts loading + inference on the CPU reference path.
//!
//! Proves a synthetic Mixtral- and Qwen-MoE checkpoint loads through the real
//! loader (the family-specific `block_sparse_moe.experts.N.w1/w3/w2` vs
//! `mlp.experts.N.{gate,up,down}_proj` naming, plus the Qwen shared expert) and
//! generates deterministic tokens — and that host-streaming a bounded window of
//! layers produces exactly the same tokens as holding them all resident.

use dlm::generate::{GenerationConfig, Sampler};
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use std::io::Write;

/// Write an F32 safetensors checkpoint from named (flattened) tensors.
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

#[derive(Clone, Copy)]
enum Family {
    Mixtral,
    Qwen,
}

// Small but non-trivial MoE geometry, shared by both families.
const H: usize = 8;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 4;
const INTER: usize = 8; // expert (and shared) FFN width
const VOCAB: usize = 6;
const LAYERS: usize = 3;
const EXPERTS: usize = 4;
const TOPK: usize = 2;

/// Deterministic, small-magnitude, seed-varying fill so a wrong-tensor bug
/// (e.g. swapping w1/w2, or reading the wrong expert) changes the output.
fn fill(seed: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + seed) % 17) as f32 - 8.0) * 0.01).collect()
}

fn write_moe_checkpoint(dir: &std::path::Path, family: Family) -> ModelConfig {
    let q_dim = NH * HD;
    let kv_dim = NKV * HD;

    let (router_name, expert_base, proj): (&str, &str, [&str; 3]) = match family {
        Family::Mixtral => ("block_sparse_moe.gate", "block_sparse_moe.experts", ["w1", "w3", "w2"]),
        Family::Qwen => ("mlp.gate", "mlp.experts", ["gate_proj", "up_proj", "down_proj"]),
    };

    let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
    tensors.push(("model.embed_tokens.weight".into(), fill(1, VOCAB * H)));
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}.");
        let s = i + 2;
        tensors.push((format!("{p}self_attn.q_proj.weight"), fill(s, q_dim * H)));
        tensors.push((format!("{p}self_attn.k_proj.weight"), fill(s, kv_dim * H)));
        tensors.push((format!("{p}self_attn.v_proj.weight"), fill(s, kv_dim * H)));
        tensors.push((format!("{p}self_attn.o_proj.weight"), fill(s, H * q_dim)));
        tensors.push((format!("{p}input_layernorm.weight"), vec![1.0; H]));
        tensors.push((format!("{p}post_attention_layernorm.weight"), vec![1.0; H]));
        // Router: [num_experts, hidden].
        tensors.push((format!("{p}{router_name}.weight"), fill(s + 100, EXPERTS * H)));
        // Routed experts, each a distinct SwiGLU triple.
        for e in 0..EXPERTS {
            let eb = format!("{p}{expert_base}.{e}.");
            let es = s + 10 * (e + 1);
            tensors.push((format!("{eb}{}.weight", proj[0]), fill(es, INTER * H)));
            tensors.push((format!("{eb}{}.weight", proj[1]), fill(es + 1, INTER * H)));
            tensors.push((format!("{eb}{}.weight", proj[2]), fill(es + 2, H * INTER)));
        }
        // Qwen shared expert (+ its sigmoid gate).
        if let Family::Qwen = family {
            let sb = format!("{p}mlp.shared_expert.");
            tensors.push((format!("{sb}gate_proj.weight"), fill(s + 200, INTER * H)));
            tensors.push((format!("{sb}up_proj.weight"), fill(s + 201, INTER * H)));
            tensors.push((format!("{sb}down_proj.weight"), fill(s + 202, H * INTER)));
            tensors.push((format!("{p}mlp.shared_expert_gate.weight"), fill(s + 203, H)));
        }
    }
    tensors.push(("model.norm.weight".into(), vec![1.0; H]));
    tensors.push(("lm_head.weight".into(), fill(3, VOCAB * H)));
    write_f32_model(dir, &tensors);

    // config.json: Mixtral uses num_local_experts + intermediate_size for experts;
    // Qwen uses num_experts + moe_intermediate_size + a shared expert.
    let moe_json = match family {
        Family::Mixtral => {
            format!(r#","num_local_experts":{EXPERTS},"num_experts_per_tok":{TOPK}"#)
        }
        Family::Qwen => format!(
            r#","num_experts":{EXPERTS},"num_experts_per_tok":{TOPK},"moe_intermediate_size":{INTER},"shared_expert_intermediate_size":{INTER}"#
        ),
    };
    let config_json = format!(
        r#"{{"hidden_size":{H},"num_attention_heads":{NH},"num_key_value_heads":{NKV},"num_hidden_layers":{LAYERS},"vocab_size":{VOCAB},"intermediate_size":{INTER}{moe_json}}}"#
    );
    std::fs::write(dir.join("config.json"), &config_json).unwrap();
    ModelConfig::from_json_bytes(config_json.as_bytes(), QuantScheme::Fp16).unwrap()
}

fn run_family(family: Family) {
    let tmp = tempfile::tempdir().unwrap();
    let config = write_moe_checkpoint(tmp.path(), family);
    assert!(config.is_moe(), "config should be detected as MoE");

    let gen_cfg = GenerationConfig {
        max_new_tokens: 8,
        eos_token: None,
        sampler: Sampler::Greedy,
    };
    let prompt = [1u32, 2, 3];

    // Resident: every layer (with all experts) held in memory.
    let store_a = MmapStore::open_dir(tmp.path()).unwrap();
    let resident = dlm::loader::load_generator(&store_a, &config, 32).unwrap();
    let out_resident = resident.generate(&prompt, &gen_cfg).unwrap();
    assert_eq!(out_resident.len(), gen_cfg.max_new_tokens, "MoE model produced no tokens");

    // Deterministic: greedy decode of the same prompt repeats exactly.
    let out_again = resident.generate(&prompt, &gen_cfg).unwrap();
    assert_eq!(out_resident, out_again, "MoE greedy decode is not deterministic");

    // Host-streaming a bounded window (2 of 3 layers resident) must emit the same
    // tokens — experts ride along with their layer on the host path.
    let store_b = MmapStore::open_dir(tmp.path()).unwrap();
    let streaming =
        dlm::loader::build_streaming_generator(store_b, &config, 32, 2, 1, false, 64 << 20).unwrap();
    let out_streaming = streaming.generate(&prompt, &gen_cfg).unwrap();
    assert_eq!(out_streaming, out_resident, "streamed MoE diverged from resident");

    // Prove the host path actually *streams* experts (loads the core, pulls the
    // top-k on demand) rather than holding every expert resident — the fix that
    // makes large MoE viable on CPU. A miss means an expert was fetched on demand.
    let stats = streaming.stream_stats().expect("streaming kernel reports stats");
    assert!(
        stats.expert_misses > 0,
        "expected routed experts to be streamed on demand, got {stats:?}"
    );
}

#[test]
fn mixtral_moe_loads_and_runs() {
    run_family(Family::Mixtral);
}

#[test]
fn qwen_moe_with_shared_expert_loads_and_runs() {
    run_family(Family::Qwen);
}

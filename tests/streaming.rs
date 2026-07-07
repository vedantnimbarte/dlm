//! End-to-end layer streaming (`specs.md` §2.2/§3.2): serving a real checkpoint
//! with only a bounded window of layers resident must produce exactly the same
//! tokens as holding every layer in memory. This exercises the full
//! `build_streaming_generator` path (pinned embedding/head + on-demand layer
//! materialization from a memory-mapped store), not just the kernel.

use dlm::generate::{GenerationConfig, Sampler};
use dlm::model::{ModelConfig, QuantScheme};
use dlm::storage::MmapStore;
use std::io::Write;

/// Write an F32 safetensors checkpoint from named 1-D tensors.
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

/// A small but non-trivial multi-layer checkpoint written to `dir`; returns its
/// config.
fn write_checkpoint(dir: &std::path::Path) -> ModelConfig {
    let (h, nh, nkv, hd, inter, vocab, layers) =
        (8usize, 2usize, 1usize, 4usize, 8usize, 6usize, 6u32);
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    // Layer-varying weights so a wrong-layer eviction bug would change output.
    let fill = |seed: usize, n: usize| -> Vec<f32> {
        (0..n).map(|i| (((i + seed) % 13) as f32 - 6.0) * 0.01).collect()
    };

    let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
    tensors.push(("model.embed_tokens.weight".into(), fill(1, vocab * h)));
    for i in 0..layers as usize {
        let p = format!("model.layers.{i}.");
        let s = i + 2;
        tensors.push((format!("{p}self_attn.q_proj.weight"), fill(s, q_dim * h)));
        tensors.push((format!("{p}self_attn.k_proj.weight"), fill(s, kv_dim * h)));
        tensors.push((format!("{p}self_attn.v_proj.weight"), fill(s, kv_dim * h)));
        tensors.push((format!("{p}self_attn.o_proj.weight"), fill(s, h * q_dim)));
        tensors.push((format!("{p}mlp.gate_proj.weight"), fill(s, inter * h)));
        tensors.push((format!("{p}mlp.up_proj.weight"), fill(s, inter * h)));
        tensors.push((format!("{p}mlp.down_proj.weight"), fill(s, h * inter)));
        tensors.push((format!("{p}input_layernorm.weight"), vec![1.0; h]));
        tensors.push((format!("{p}post_attention_layernorm.weight"), vec![1.0; h]));
    }
    tensors.push(("model.norm.weight".into(), vec![1.0; h]));
    tensors.push(("lm_head.weight".into(), fill(3, vocab * h)));
    write_f32_model(dir, &tensors);

    let config_json = format!(
        r#"{{"hidden_size":{h},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"num_hidden_layers":{layers},"vocab_size":{vocab},"intermediate_size":{inter}}}"#
    );
    ModelConfig::from_json_bytes(config_json.as_bytes(), QuantScheme::Fp16).unwrap()
}

#[test]
fn streaming_generation_matches_resident() {
    let tmp = tempfile::tempdir().unwrap();
    let config = write_checkpoint(tmp.path());

    let cfg = GenerationConfig {
        max_new_tokens: 8,
        eos_token: None,
        sampler: Sampler::Greedy,
    };
    let prompt = [1u32, 2, 3];

    // Resident: all layers held in memory.
    let store_a = MmapStore::open_dir(tmp.path()).unwrap();
    let resident = dlm::loader::load_generator(&store_a, &config, 32).unwrap();
    let out_resident = resident.generate(&prompt, &cfg).unwrap();

    // Streaming: only 2 of 6 layers resident at a time (rest streamed from disk).
    let store_b = MmapStore::open_dir(tmp.path()).unwrap();
    let streaming = dlm::loader::build_streaming_generator(store_b, &config, 32, 2, 1).unwrap();
    let out_streaming = streaming.generate(&prompt, &cfg).unwrap();

    assert_eq!(
        out_streaming, out_resident,
        "streamed serving diverged from resident output",
    );
    assert_eq!(out_resident.len(), 8);
}

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
    // Write it out too, so the fixture is a checkpoint directory a caller can
    // re-read at a different `QuantScheme` (as `ModelConfig::from_path` does).
    std::fs::write(dir.join("config.json"), &config_json).unwrap();
    ModelConfig::from_json_bytes(config_json.as_bytes(), QuantScheme::Fp16).unwrap()
}

/// The planned resident window must fit the free VRAM it was planned against,
/// measured at the checkpoint's **real** on-disk layer size.
///
/// Regression: `serve --stream` sized its window from `--quant`'s parameter-count
/// estimate, which defaults to Int4 (0.5 bytes/param). dlm does not quantize
/// weights — it loads them in their native dtype — so for an F32/BF16 checkpoint
/// the estimate claims layers are 4–8x smaller than they are, and the window it
/// derived exceeded free VRAM (on a 3B/4GB card it returned all 28 layers). The
/// engine then thought the whole model was resident and never streamed, leaving
/// the driver to page VRAM behind its back — silent and glacial under Windows
/// WDDM, an OOM where no such paging exists. Planning from the catalog (real
/// tensor sizes) is what keeps this honest.
#[test]
fn planned_window_fits_free_vram_at_real_layer_size() {
    let tmp = tempfile::tempdir().unwrap();
    let config = write_checkpoint(tmp.path());
    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let catalog = dlm::storage::LayerCatalog::build(&store);
    assert!(!catalog.is_empty(), "catalog should see the layers");

    let per_layer = catalog.max_layer_bytes();
    // Zero the safety cushion: at its 1.5 GiB default it would swamp this tiny
    // synthetic budget, saturate `usable` to 0, and pin the window at 1 layer —
    // passing without ever exercising the arithmetic under test.
    let profiler = dlm::profiler::VramProfiler::new(128).with_safety_margin_bytes(0);
    // Room for the KV, the pinned zone, and exactly 3 of the 6 layers.
    let free =
        profiler.kv_total_bytes(&config) + catalog.pinned_bytes() + per_layer * 3;

    // native == config.quant: no quantize-at-load rescale, so the catalog's
    // measured block size is what the window is planned against.
    let plan = profiler.plan_from_catalog(&config, &catalog, config.quant, free);
    assert_eq!(plan.layers_to_load, 3, "should fit exactly 3 layers");
    assert!(
        plan.layers_to_load < config.num_layers,
        "window {} should be < {} layers when the model cannot fit",
        plan.layers_to_load,
        config.num_layers
    );
    let resident = plan.layers_to_load as u64 * per_layer;
    assert!(
        resident <= free,
        "planned {} layers = {resident} B of weights, over the {free} B budget",
        plan.layers_to_load
    );

    // The estimate is not asserted against here: how far it strays from the real
    // layer size is model-dependent (it happens to be *conservative* for this tiny
    // synthetic geometry, and wildly optimistic for a real bf16 checkpoint under
    // the Int4 default), so there is no direction to pin. The invariant above is
    // the one that matters — whatever sizes a plan is built from, the window it
    // returns has to fit.

    // `--quant int4` quantizes at load, so the layer that reaches VRAM is an
    // eighth of this F32 checkpoint's on-disk bytes — the same budget must hold
    // proportionally more of them. Planning from the on-disk size instead would
    // strand most of the VRAM.
    let int4_cfg = ModelConfig::from_path(tmp.path(), QuantScheme::Int4).unwrap();
    let int4 = profiler.plan_from_catalog(&int4_cfg, &catalog, QuantScheme::F32, free);
    assert!(
        int4.per_layer_weight_bytes < plan.per_layer_weight_bytes,
        "int4 layers ({}) must be smaller in VRAM than the F32 originals ({})",
        int4.per_layer_weight_bytes,
        plan.per_layer_weight_bytes
    );
    assert!(
        int4.layers_to_load > plan.layers_to_load,
        "int4 should fit more layers ({}) than F32 ({}) in the same budget",
        int4.layers_to_load,
        plan.layers_to_load
    );
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
    // 64 MiB host-RAM layer cache on, so this also covers the cached source path:
    // caching must not change what the model emits.
    let streaming =
        dlm::loader::build_streaming_generator(store_b, &config, 32, 2, 1, false, 64 << 20).unwrap();
    let out_streaming = streaming.generate(&prompt, &cfg).unwrap();

    assert_eq!(
        out_streaming, out_resident,
        "streamed serving diverged from resident output",
    );
    assert_eq!(out_resident.len(), 8);
}

//! Phase 1 integration tests — exercised with no GPU present.
//!
//! Covers the three deliverables:
//!   1. mmap storage engine (safetensors parse + zero-copy reads)
//!   2. VRAM profiling math
//!   3. page-locked host staging buffers + the linear swap cycle

use flip::memory::{page_size, PinKind, PinnedBuffer};
use flip::model::{ModelConfig, QuantScheme};
use flip::profiler::VramProfiler;
use flip::storage::MmapStore;
use flip::swap::LayerSwapPlan;
use std::io::Write;

/// Serialize a minimal one-tensor safetensors file into `dir` and return its path.
fn write_safetensors(dir: &std::path::Path, tensor: &str, data: &[u8]) -> std::path::PathBuf {
    // shape [len] of U8 → byte_len == data.len()
    let header = format!(
        r#"{{"{tensor}":{{"dtype":"U8","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        data.len(),
        data.len()
    );
    let header_bytes = header.as_bytes();

    let path = dir.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header_bytes).unwrap();
    f.write_all(data).unwrap();
    f.flush().unwrap();
    path
}

#[test]
fn mmap_store_reads_tensor_zero_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let payload: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    write_safetensors(tmp.path(), "block.0.weight", &payload);

    let store = MmapStore::open_dir(tmp.path()).unwrap();
    assert_eq!(store.num_shards(), 1);
    assert_eq!(store.num_tensors(), 1);

    let bytes = store.tensor_bytes("block.0.weight").unwrap();
    assert_eq!(bytes, payload.as_slice());

    let (_, info) = store.locate("block.0.weight").unwrap();
    assert_eq!(info.byte_len(), 4096);
    assert_eq!(info.num_elements(), 4096);
}

#[test]
fn mmap_store_rejects_unknown_tensor() {
    let tmp = tempfile::tempdir().unwrap();
    write_safetensors(tmp.path(), "a", &[1, 2, 3, 4]);
    let store = MmapStore::open_dir(tmp.path()).unwrap();
    assert!(store.tensor_bytes("missing").is_err());
}

#[test]
fn vram_math_matches_hand_computation() {
    // Small, exactly-computable model.
    let json = br#"{
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 10,
        "vocab_size": 32000,
        "intermediate_size": 4096
    }"#;
    let config = ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap();

    // head_dim = 1024/16 = 64
    assert_eq!(config.head_dim(), 64);

    let profiler = VramProfiler::new(2048).with_safety_margin_bytes(0);

    // KV per layer = 2 * kv_heads(4) * head_dim(64) * 2 bytes * ctx(2048)
    //             = 2*4*64*2*2048 = 2,097,152 bytes
    let kv_layer = profiler.kv_bytes_per_layer(&config);
    assert_eq!(kv_layer, 2 * 4 * 64 * 2 * 2048);
    // Total across 10 layers.
    assert_eq!(profiler.kv_total_bytes(&config), kv_layer * 10);

    // Give exactly enough usable memory for 3 layer-weights on top of KV+safety.
    let per_layer = profiler.per_layer_weight_bytes(&config);
    let free = profiler.kv_total_bytes(&config) + per_layer * 3;
    let plan = profiler.plan_with_free(&config, free);
    assert_eq!(plan.layers_to_load, 3);
    assert_eq!(plan.num_layers, 10);
    assert_eq!(plan.stream_passes(), 4); // ceil(10/3)
}

#[test]
fn vram_plan_clamps_between_one_and_num_layers() {
    let config = ModelConfig::from_json_bytes(
        br#"{"hidden_size":512,"num_attention_heads":8,"num_hidden_layers":4,"vocab_size":1000}"#,
        QuantScheme::Int4,
    )
    .unwrap();
    let profiler = VramProfiler::new(128);

    // Starved: 0 free → still clamped up to 1 (streaming needs one slot).
    let starved = profiler.plan_with_free(&config, 0);
    assert_eq!(starved.layers_to_load, 1);

    // Abundant: huge free → clamped down to the model's layer count.
    let abundant = profiler.plan_with_free(&config, u64::MAX / 2);
    assert_eq!(abundant.layers_to_load, 4);
    assert_eq!(abundant.resident_fraction(), 1.0);
}

#[test]
fn pinned_buffer_is_page_aligned_and_sized() {
    let buf = PinnedBuffer::with_len(100).unwrap();
    let page = page_size();

    // Base pointer aligned to a page.
    assert_eq!(buf.as_ptr() as usize % page, 0, "base must be page-aligned");
    // Capacity rounded up to a whole page.
    assert_eq!(buf.capacity() % page, 0, "capacity must be a page multiple");
    assert!(buf.capacity() >= 100);
    assert_eq!(buf.len(), 100);

    // Off-GPU build reports the page-aligned fallback kind.
    #[cfg(not(feature = "cuda"))]
    assert_eq!(buf.kind(), PinKind::PageAligned);
}

#[test]
fn pinned_buffer_round_trips_bytes() {
    let src: Vec<u8> = (0..200u8).collect();
    let mut buf = PinnedBuffer::from_bytes(&src).unwrap();
    assert_eq!(buf.as_slice(), src.as_slice());

    buf.as_mut_slice()[0] = 0xFF;
    assert_eq!(buf.as_slice()[0], 0xFF);
}

#[test]
fn pinned_buffer_rejects_zero_length() {
    assert!(PinnedBuffer::with_len(0).is_err());
}

#[test]
fn swap_plan_tiles_all_layers_without_gaps() {
    let config = ModelConfig::from_json_bytes(
        br#"{"hidden_size":1024,"num_attention_heads":16,"num_hidden_layers":80,"vocab_size":32000}"#,
        QuantScheme::Int4,
    )
    .unwrap();
    let profiler = VramProfiler::new(4096);
    // Force a 7-layer window to get a ragged final pass (80 = 11*7 + 3).
    let free = profiler.kv_total_bytes(&config)
        + profiler.safety_margin_bytes
        + profiler.per_layer_weight_bytes(&config) * 7;
    let plan = profiler.plan_with_free(&config, free);
    assert_eq!(plan.layers_to_load, 7);

    let swap = LayerSwapPlan::from_plan(&plan);

    // Passes must cover [0,80) contiguously with no overlap or gap.
    let mut expected_next = 0u32;
    let mut total = 0u32;
    for pass in &swap.passes {
        assert_eq!(pass.first_layer, expected_next);
        assert!(pass.last_layer >= pass.first_layer);
        total += pass.layer_count();
        expected_next = pass.last_layer + 1;
    }
    assert_eq!(total, 80);
    assert_eq!(expected_next, 80);
    assert_eq!(swap.num_passes(), 12); // ceil(80/7)

    // Staging buffer holds a full window of layer weights.
    let staging = swap.allocate_staging_buffer(plan.per_layer_weight_bytes).unwrap();
    let needed = plan.per_layer_weight_bytes as usize * 7;
    assert!(staging.capacity() >= needed);
}

//! Phase 1 integration tests — exercised with no GPU present.
//!
//! Covers the three deliverables:
//!   1. mmap storage engine (safetensors parse + zero-copy reads)
//!   2. VRAM profiling math
//!   3. page-locked host staging buffers + the linear swap cycle

use flip::memory::{page_size, PinKind, PinnedBuffer};
use flip::model::{ModelConfig, QuantScheme};
use flip::pipeline::{fold_checksum, BufferId, DoubleBufferSchedule, HostPipeline, WeightSource};
use flip::profiler::VramProfiler;
use flip::storage::{LayerCatalog, MmapStore};
use flip::swap::{LayerSwapPlan, StreamPass};
use std::io::Write;

/// Serialize a multi-tensor safetensors file (byte tensors) into `dir`.
fn write_multi_tensor(dir: &std::path::Path, tensors: &[(&str, usize)]) -> std::path::PathBuf {
    // Build the header with sequential data offsets, then a matching data blob.
    let mut entries = Vec::new();
    let mut offset = 0usize;
    for (name, len) in tensors {
        entries.push(format!(
            r#""{name}":{{"dtype":"U8","shape":[{len}],"data_offsets":[{offset},{}]}}"#,
            offset + len
        ));
        offset += len;
    }
    let header = format!("{{{}}}", entries.join(","));
    let header_bytes = header.as_bytes();
    let data: Vec<u8> = vec![0u8; offset];

    let path = dir.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header_bytes).unwrap();
    f.write_all(&data).unwrap();
    f.flush().unwrap();
    path
}

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
fn catalog_groups_layers_and_pinned_overhead() {
    let tmp = tempfile::tempdir().unwrap();
    write_multi_tensor(
        tmp.path(),
        &[
            ("model.embed_tokens.weight", 1000),
            ("model.layers.0.self_attn.q_proj.weight", 400),
            ("model.layers.0.mlp.down_proj.weight", 600), // layer 0 total = 1000
            ("model.layers.1.self_attn.q_proj.weight", 800), // layer 1 total = 800
            ("model.norm.weight", 50),
            ("lm_head.weight", 1000),
        ],
    );

    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let catalog = LayerCatalog::build(&store);

    assert_eq!(catalog.num_layers(), 2);
    assert_eq!(catalog.layer_bytes(0), Some(1000));
    assert_eq!(catalog.layer_bytes(1), Some(800));
    assert_eq!(catalog.max_layer_bytes(), 1000);
    assert_eq!(catalog.mean_layer_bytes(), 900);
    assert_eq!(catalog.total_layer_bytes(), 1800);
    // Pinned = embed(1000) + norm(50) + lm_head(1000) = 2050.
    assert_eq!(catalog.pinned_bytes(), 2050);
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
fn profiler_uses_catalog_sizes_and_pinned_overhead() {
    let tmp = tempfile::tempdir().unwrap();
    // 4 layers of 100 bytes each, plus 500 bytes of pinned tensors.
    write_multi_tensor(
        tmp.path(),
        &[
            ("model.embed_tokens.weight", 250),
            ("lm_head.weight", 250),
            ("model.layers.0.w", 100),
            ("model.layers.1.w", 100),
            ("model.layers.2.w", 100),
            ("model.layers.3.w", 100),
        ],
    );
    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let catalog = LayerCatalog::build(&store);

    let config = ModelConfig::from_json_bytes(
        br#"{"hidden_size":512,"num_attention_heads":8,"num_hidden_layers":4,"vocab_size":1000}"#,
        QuantScheme::Int4,
    )
    .unwrap();
    // Zero out KV and safety to isolate the pinned-overhead effect.
    let profiler = VramProfiler::new(1).with_safety_margin_bytes(0);
    let kv = profiler.kv_total_bytes(&config);

    // Give room for pinned(500) + exactly 2 layers (200) on top of KV.
    let free = kv + 500 + 200;
    let plan = profiler.plan_from_catalog(&config, &catalog, free);

    assert_eq!(plan.pinned_bytes, 500);
    assert_eq!(plan.per_layer_weight_bytes, 100); // measured max block
    assert_eq!(plan.usable_bytes, 200);
    assert_eq!(plan.layers_to_load, 2);
    assert_eq!(plan.num_layers, 4);
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

/// Synthetic weight source: each window is `per_layer_bytes` per layer, filled
/// with a byte pattern unique to the pass so clobbering is detectable.
struct MockSource {
    per_layer_bytes: usize,
}

impl WeightSource for MockSource {
    fn load_window(&self, pass: &StreamPass) -> flip::Result<Vec<u8>> {
        let len = self.per_layer_bytes * pass.layer_count() as usize;
        Ok(vec![(pass.pass_index as u8).wrapping_add(1); len])
    }
}

fn plan_with_window(num_layers: u32, window: u32) -> LayerSwapPlan {
    let config = ModelConfig::from_json_bytes(
        format!(
            r#"{{"hidden_size":1024,"num_attention_heads":16,"num_hidden_layers":{num_layers},"vocab_size":32000}}"#
        )
        .as_bytes(),
        QuantScheme::Int4,
    )
    .unwrap();
    let profiler = VramProfiler::new(4096);
    let free = profiler.kv_total_bytes(&config)
        + profiler.safety_margin_bytes
        + profiler.per_layer_weight_bytes(&config) * window as u64;
    let plan = profiler.plan_with_free(&config, free);
    assert_eq!(plan.layers_to_load, window);
    LayerSwapPlan::from_plan(&plan)
}

#[test]
fn double_buffer_schedule_overlaps_and_alternates() {
    // 3 windows of 4 layers over 12 layers.
    let swap = plan_with_window(12, 4);
    assert_eq!(swap.num_passes(), 3);

    let sched = DoubleBufferSchedule::from_swap_plan(&swap);
    // N windows → N+1 steps (1 prologue + N steady).
    assert_eq!(sched.num_steps(), 4);
    // Prologue: prefetch only, no compute.
    assert!(sched.steps[0].compute.is_none());
    assert_eq!(sched.steps[0].prefetch.unwrap().pass_index, 0);
    assert_eq!(sched.steps[0].prefetch_buffer, BufferId::A);
    // Overlap in every steady step except the last (nothing left to prefetch).
    assert_eq!(sched.overlapping_steps(), 2);
    // Compute buffers ping-pong A, B, A.
    assert_eq!(sched.steps[1].compute_buffer, BufferId::A);
    assert_eq!(sched.steps[2].compute_buffer, BufferId::B);
    assert_eq!(sched.steps[3].compute_buffer, BufferId::A);
    // Each prefetch targets the buffer opposite the concurrent compute.
    assert_eq!(sched.steps[1].prefetch_buffer, BufferId::B);
}

#[test]
fn host_pipeline_computes_each_window_over_intact_data() {
    let swap = plan_with_window(12, 4);
    let sched = DoubleBufferSchedule::from_swap_plan(&swap);

    let per_layer_bytes = 64;
    let source = MockSource { per_layer_bytes };
    let window_bytes = per_layer_bytes * swap.window_size as usize;

    let mut pipeline = HostPipeline::new(window_bytes).unwrap();
    let trace = pipeline.execute(&sched, &source).unwrap();

    // Every window computed exactly once, in order.
    assert_eq!(trace.len(), swap.num_passes());
    for (i, t) in trace.iter().enumerate() {
        assert_eq!(t.pass_index, i as u32);
        // The bytes the compute step saw must equal this pass's source bytes —
        // proving the concurrent prefetch of the next window (into the other
        // buffer) did not corrupt the buffer under compute.
        let expected = source.load_window(&swap.passes[i]).unwrap();
        assert_eq!(t.byte_len, expected.len());
        assert_eq!(t.checksum, fold_checksum(&expected));
    }
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

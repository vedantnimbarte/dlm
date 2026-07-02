//! Phase 1 integration tests — exercised with no GPU present.
//!
//! Covers the three deliverables:
//!   1. mmap storage engine (safetensors parse + zero-copy reads)
//!   2. VRAM profiling math
//!   3. page-locked host staging buffers + the linear swap cycle

use flip::memory::{page_size, PinnedBuffer};
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
use flip::memory::PinKind;
use flip::model::{ModelConfig, QuantScheme};
use flip::pipeline::{
    fold_checksum, BufferId, DoubleBufferSchedule, HostPipeline, MmapWeightSource,
    TieredWeightSource, WeightSource,
};
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

    // Per-layer tensor names are recorded in sorted order.
    assert_eq!(
        catalog.layer_tensor_names(0),
        Some(
            &[
                "model.layers.0.mlp.down_proj.weight".to_string(),
                "model.layers.0.self_attn.q_proj.weight".to_string(),
            ][..]
        )
    );
    assert_eq!(catalog.layer_tensor_names(2), None);
    // Pinned tensor names captured too.
    assert_eq!(catalog.pinned_tensor_names().len(), 3);
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
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
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

/// Like `write_multi_tensor`, but fills each tensor with a byte pattern unique
/// to its position so a mis-copied window is detectable by checksum.
fn write_patterned_model(dir: &std::path::Path, tensors: &[(&str, usize)]) {
    let mut entries = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut offset = 0usize;
    for (i, (name, len)) in tensors.iter().enumerate() {
        entries.push(format!(
            r#""{name}":{{"dtype":"U8","shape":[{len}],"data_offsets":[{offset},{}]}}"#,
            offset + len
        ));
        data.extend(std::iter::repeat((i as u8).wrapping_add(1)).take(*len));
        offset += len;
    }
    let header = format!("{{{}}}", entries.join(","));
    let path = dir.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&data).unwrap();
    f.flush().unwrap();
}

#[test]
fn mmap_source_streams_real_weights_through_pipeline() {
    let tmp = tempfile::tempdir().unwrap();
    write_patterned_model(
        tmp.path(),
        &[
            ("model.embed_tokens.weight", 40),
            ("model.layers.0.attn.weight", 100),
            ("model.layers.0.mlp.weight", 60), // layer 0 = 160 bytes
            ("model.layers.1.attn.weight", 100),
            ("model.layers.1.mlp.weight", 60), // layer 1 = 160 bytes
            ("model.layers.2.attn.weight", 100),
            ("model.layers.2.mlp.weight", 60), // layer 2 = 160 bytes
            ("lm_head.weight", 40),
        ],
    );

    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let catalog = LayerCatalog::build(&store);
    assert_eq!(catalog.num_layers(), 3);
    assert_eq!(catalog.max_layer_bytes(), 160);

    // Window of 2 layers over 3 → 2 passes (layers 0-1, then 2).
    let vram = flip::profiler::VramPlan {
        free_bytes: 0,
        safety_bytes: 0,
        kv_total_bytes: 0,
        pinned_bytes: 0,
        per_layer_weight_bytes: 160,
        usable_bytes: 0,
        layers_to_load: 2,
        num_layers: 3,
    };
    let swap = LayerSwapPlan::from_plan(&vram);
    assert_eq!(swap.num_passes(), 2);

    let sched = DoubleBufferSchedule::from_swap_plan(&swap);
    let source = MmapWeightSource::new(&store, &catalog);

    // Verify the source concatenates the real mapped bytes for pass 0.
    let pass0 = swap.passes[0];
    let mut expected0 = Vec::new();
    expected0.extend_from_slice(store.tensor_bytes("model.layers.0.attn.weight").unwrap());
    expected0.extend_from_slice(store.tensor_bytes("model.layers.0.mlp.weight").unwrap());
    expected0.extend_from_slice(store.tensor_bytes("model.layers.1.attn.weight").unwrap());
    expected0.extend_from_slice(store.tensor_bytes("model.layers.1.mlp.weight").unwrap());
    assert_eq!(source.load_window(&pass0).unwrap(), expected0);
    assert_eq!(source.window_bytes(&pass0), 320);

    // Run the double-buffered pipeline over the real weights.
    let mut pipeline = HostPipeline::new(320).unwrap();
    let trace = pipeline.execute(&sched, &source).unwrap();

    assert_eq!(trace.len(), 2);
    for (i, t) in trace.iter().enumerate() {
        let window = source.load_window(&swap.passes[i]).unwrap();
        assert_eq!(t.byte_len, window.len());
        assert_eq!(t.checksum, fold_checksum(&window));
    }
    // Windows carry different data, so their checksums must differ.
    assert_ne!(trace[0].checksum, trace[1].checksum);
}

#[test]
fn tiered_cache_serves_second_pass_from_ram() {
    let tmp = tempfile::tempdir().unwrap();
    write_patterned_model(
        tmp.path(),
        &[
            ("model.embed_tokens.weight", 40),
            ("model.layers.0.w", 100),
            ("model.layers.1.w", 100),
            ("model.layers.2.w", 100),
            ("lm_head.weight", 40),
        ],
    );
    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let catalog = LayerCatalog::build(&store);
    assert_eq!(catalog.num_layers(), 3);

    // Whole model resident in one window (3 layers).
    let vram = flip::profiler::VramPlan {
        free_bytes: 0,
        safety_bytes: 0,
        kv_total_bytes: 0,
        pinned_bytes: 0,
        per_layer_weight_bytes: 100,
        usable_bytes: 0,
        layers_to_load: 3,
        num_layers: 3,
    };
    let swap = LayerSwapPlan::from_plan(&vram);
    let sched = DoubleBufferSchedule::from_swap_plan(&swap);

    // RAM cache big enough for all 3 layers (300 bytes).
    let inner = MmapWeightSource::new(&store, &catalog);
    let tiered = TieredWeightSource::new(inner, 300);

    let window_bytes = 300;
    let mut pipeline = HostPipeline::new(window_bytes).unwrap();

    // First forward pass: all 3 layers are cold → 3 misses, 0 hits.
    pipeline.execute(&sched, &tiered).unwrap();
    let s1 = tiered.cache_stats();
    assert_eq!(s1.misses, 3);
    assert_eq!(s1.hits, 0);
    assert_eq!(s1.entries, 3);

    // Second forward pass: everything served from RAM → +3 hits, no new misses.
    let trace = pipeline.execute(&sched, &tiered).unwrap();
    let s2 = tiered.cache_stats();
    assert_eq!(s2.misses, 3);
    assert_eq!(s2.hits, 3);
    assert_eq!(s2.evictions, 0);

    // Data is still correct after caching.
    let direct = MmapWeightSource::new(&store, &catalog);
    let expected = direct.load_window(&swap.passes[0]).unwrap();
    assert_eq!(trace[0].checksum, fold_checksum(&expected));
}

#[test]
fn tiered_cache_evicts_under_pressure() {
    let tmp = tempfile::tempdir().unwrap();
    write_patterned_model(
        tmp.path(),
        &[
            ("model.layers.0.w", 100),
            ("model.layers.1.w", 100),
            ("model.layers.2.w", 100),
        ],
    );
    let store = MmapStore::open_dir(tmp.path()).unwrap();
    let catalog = LayerCatalog::build(&store);

    // Budget for only 2 of the 3 layers → the third forces an eviction.
    let tiered = TieredWeightSource::new(MmapWeightSource::new(&store, &catalog), 200);

    // Stream each layer as its own single-layer window, in order.
    for layer in 0..3u32 {
        let pass = flip::swap::StreamPass {
            pass_index: layer,
            first_layer: layer,
            last_layer: layer,
        };
        tiered.load_window(&pass).unwrap();
    }
    let stats = tiered.cache_stats();
    assert_eq!(stats.misses, 3);
    assert_eq!(stats.evictions, 1);
    assert!(stats.resident_bytes <= 200);
}

#[test]
fn orchestrator_drives_stub_kernel_with_kv_growth() {
    use flip::cache::{KvCacheConfig, PagedKvCache};
    use flip::forward::{ForwardOrchestrator, StubKernel};

    let kernel = StubKernel::new(3, 4, 2); // 3 layers, hidden 4, kv_dim 2
    let budget = PagedKvCache::new(
        KvCacheConfig { num_layers: 3, num_kv_heads: 1, head_dim: 2, block_size: 16 },
        8,
    );
    let mut orch = ForwardOrchestrator::new(kernel, budget);

    let mut hidden = vec![0.0f32; 4];
    orch.decode_token(&mut hidden).unwrap();
    // Each layer adds (layer + 1): 1 + 2 + 3 = 6 to every element.
    assert_eq!(hidden, vec![6.0; 4]);
    for l in 0..3 {
        assert_eq!(orch.layer_kv_len(l), 1);
    }
    assert_eq!(orch.position(), 1);

    orch.decode_token(&mut hidden).unwrap();
    assert_eq!(hidden, vec![12.0; 4]);
    for l in 0..3 {
        assert_eq!(orch.layer_kv_len(l), 2);
    }
    assert_eq!(orch.position(), 2);
    assert_eq!(orch.kv_budget().sequence_len(0), 2);
}

#[test]
fn orchestrator_runs_real_cpu_block_autoregressively() {
    use flip::cache::{KvCacheConfig, PagedKvCache};
    use flip::forward::{BlockConfig, CpuKernel, ForwardOrchestrator, LayerTensors};

    let cfg = BlockConfig {
        hidden_size: 4,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        intermediate_size: 6,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };
    // Two zero-weight (identity) layers → hidden passes through unchanged.
    let kernel = CpuKernel::new(cfg, vec![LayerTensors::zeros(&cfg); 2]).unwrap();
    let budget = PagedKvCache::new(
        KvCacheConfig { num_layers: 2, num_kv_heads: 1, head_dim: 2, block_size: 16 },
        8,
    );
    let mut orch = ForwardOrchestrator::new(kernel, budget);

    let original = vec![1.5f32, -2.0, 0.5, 3.0];
    let mut hidden = original.clone();

    // Decode two tokens autoregressively.
    orch.decode_token(&mut hidden).unwrap();
    orch.decode_token(&mut hidden).unwrap();

    // Identity layers leave the residual stream untouched.
    assert_eq!(hidden, original);
    // Each layer accumulated real K/V for both token positions.
    assert_eq!(orch.layer_kv_len(0), 2);
    assert_eq!(orch.layer_kv_len(1), 2);
    assert_eq!(orch.position(), 2);
}

#[test]
fn orchestrator_validates_hidden_length() {
    use flip::cache::{KvCacheConfig, PagedKvCache};
    use flip::forward::{ForwardOrchestrator, StubKernel};

    let kernel = StubKernel::new(2, 4, 1);
    let budget = PagedKvCache::new(
        KvCacheConfig { num_layers: 2, num_kv_heads: 1, head_dim: 1, block_size: 8 },
        4,
    );
    let mut orch = ForwardOrchestrator::new(kernel, budget);

    let mut wrong = vec![0.0f32; 3]; // hidden_size is 4
    assert!(orch.decode_token(&mut wrong).is_err());
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

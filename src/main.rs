//! `flip` binary — Phase 1 demonstration entry point.
//!
//! The full `flip serve` CLI (clap-based argument parsing, the OpenAI-compatible
//! server) lands in Phase 2/3. For now this binary exercises the Phase 1
//! foundation end-to-end with no GPU required: it profiles a representative
//! 70B-class model against a simulated 16 GiB card and prints the resulting
//! layer-streaming schedule. Pass a model directory to profile real weights.

use flip::cache::{KvCacheConfig, PagedKvCache};
use flip::memory::{page_size, PinnedBuffer};
use flip::model::{ModelConfig, QuantScheme};
use flip::profiler::VramProfiler;
use flip::storage::{LayerCatalog, MmapStore};
use flip::pipeline::{DoubleBufferSchedule, HostPipeline, MmapWeightSource};
use flip::swap::LayerSwapPlan;
use flip::{gpu, Result};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<()> {
    println!("flip v{} — Phase 1 (Local Foundation)", env!("CARGO_PKG_VERSION"));
    println!("  gpu backend  : {}", gpu::active_vendor().label());
    println!("  host page    : {} bytes", page_size());
    println!();

    let args: Vec<String> = std::env::args().collect();

    // Either profile a real model directory, or a representative built-in config.
    let (config, source) = match args.get(1) {
        Some(dir) => (
            ModelConfig::from_path(dir, QuantScheme::Int4)?,
            format!("config.json in {dir}"),
        ),
        None => (sample_70b_config(), "built-in Llama-3-70B-class sample".to_string()),
    };

    println!("model source : {source}");
    println!(
        "geometry     : {} layers, hidden {}, {} q-heads / {} kv-heads, head_dim {}",
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim(),
    );
    println!(
        "quantization : {:?} ({} bytes/param), ~{:.1} B params",
        config.quant,
        config.quant.bytes_per_param(),
        config.estimated_total_params() as f64 / 1e9,
    );
    println!();

    // If a model dir was given, map its shards and measure real weight sizes.
    // Keep the store alive alongside the catalog so we can stream from it below.
    let mut mapped: Option<(MmapStore, LayerCatalog)> = None;
    if let Some(dir) = args.get(1) {
        if let Ok(store) = MmapStore::open_dir(dir) {
            let cat = LayerCatalog::build(&store);
            println!(
                "storage      : mapped {} shard(s), {} tensors, {:.2} GiB on disk",
                store.num_shards(),
                store.num_tensors(),
                store.total_mapped_bytes() as f64 / GIB as f64,
            );
            println!(
                "catalog      : {} block(s), max block {:.1} MiB, pinned {:.1} MiB",
                cat.num_layers(),
                cat.max_layer_bytes() as f64 / (1024.0 * 1024.0),
                cat.pinned_bytes() as f64 / (1024.0 * 1024.0),
            );
            println!();
            mapped = Some((store, cat));
        }
    }
    let catalog = mapped.as_ref().map(|(_, cat)| cat);

    // Determine free VRAM: live device when CUDA is present, else simulate 16 GiB.
    let profiler = VramProfiler::new(8192);
    let (free_bytes, live) = match gpu::mem_get_info() {
        Ok(dev) => (dev.free, true),
        Err(_) => (16 * GIB, false),
    };
    if live {
        println!("free VRAM    : live device query ({})", gpu::active_vendor().label());
    } else {
        println!("free VRAM    : simulated {} GiB (no GPU device)", free_bytes / GIB);
    }

    // Prefer measured catalog sizes; fall back to the parameter estimate.
    let plan = match catalog {
        Some(cat) if !cat.is_empty() => profiler.plan_from_catalog(&config, cat, free_bytes),
        _ => profiler.plan_with_free(&config, free_bytes),
    };

    print_plan(&plan);

    // Size the PagedAttention KV pool from the plan's KV budget (Cache Zone).
    let kv_cfg = KvCacheConfig::from_model(&config, 16);
    let kv_cache = PagedKvCache::with_budget(kv_cfg, plan.kv_total_bytes);
    println!();
    println!(
        "kv cache     : {} paged blocks × {} tok, {:.2} MiB/block → {} token capacity",
        kv_cache.total_blocks(),
        kv_cfg.block_size,
        kv_cfg.bytes_per_block() as f64 / (1024.0 * 1024.0),
        kv_cache.capacity_tokens(),
    );

    let swap = LayerSwapPlan::from_plan(&plan);
    println!();
    println!(
        "swap cycle   : {} streaming pass(es), window of {} layer(s)",
        swap.num_passes(),
        swap.window_size,
    );
    for pass in swap.passes.iter().take(4) {
        println!(
            "  pass {:>2} → layers {:>3}..={:<3} ({} layers)",
            pass.pass_index,
            pass.first_layer,
            pass.last_layer,
            pass.layer_count(),
        );
    }
    if swap.num_passes() > 4 {
        println!("  … {} more pass(es)", swap.num_passes() - 4);
    }

    // Allocate the pinned staging buffer the pipeline will DMA from.
    let staging: PinnedBuffer = swap.allocate_staging_buffer(plan.per_layer_weight_bytes)?;
    println!();
    println!(
        "staging buf  : {:.1} MiB pinned ({:?}), page-aligned base {:p}",
        staging.capacity() as f64 / (1024.0 * 1024.0),
        staging.kind(),
        staging.as_ptr(),
    );

    // Build the double-buffered A/B schedule (Phase 2 §3.2).
    let sched = DoubleBufferSchedule::from_swap_plan(&swap);
    println!();
    println!(
        "pipeline     : {} steps, {} overlapped (DMA hidden under compute)",
        sched.num_steps(),
        sched.overlapping_steps(),
    );
    for step in sched.steps.iter().take(4) {
        let compute = step
            .compute
            .map(|p| format!("compute p{} [{:?}]", p.pass_index, step.compute_buffer))
            .unwrap_or_else(|| "compute —".to_string());
        let prefetch = step
            .prefetch
            .map(|p| format!("prefetch p{} → [{:?}]", p.pass_index, step.prefetch_buffer))
            .unwrap_or_else(|| "prefetch —".to_string());
        println!("  A:{compute:<22} | B:{prefetch}");
    }
    if sched.num_steps() > 4 {
        println!("  … {} more step(s)", sched.num_steps() - 4);
    }

    // With a real mapped model, stream its weights end-to-end through the
    // double-buffered host pipeline (disk → pinned → device → compute).
    if let Some((store, catalog)) = mapped.as_ref().filter(|(_, c)| !c.is_empty()) {
        let source = MmapWeightSource::new(store, catalog);
        let max_window = swap
            .passes
            .iter()
            .map(|p| source.window_bytes(p))
            .max()
            .unwrap_or(0)
            .max(1);
        let mut pipeline = HostPipeline::new(max_window as usize)?;
        let trace = pipeline.execute(&sched, &source)?;
        let moved: usize = trace.iter().map(|t| t.byte_len).sum();
        println!();
        println!(
            "streamed     : {} window(s) executed, {:.2} GiB moved through pinned staging",
            trace.len(),
            moved as f64 / GIB as f64,
        );
    }

    Ok(())
}

fn print_plan(plan: &flip::profiler::VramPlan) {
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!();
    println!("── VRAM PLAN ─────────────────────────────────");
    println!("  M_free           : {:>10.1} MiB", mib(plan.free_bytes));
    println!("  M_safety         : {:>10.1} MiB", mib(plan.safety_bytes));
    println!("  M_kv_total       : {:>10.1} MiB", mib(plan.kv_total_bytes));
    println!("  pinned_zone      : {:>10.1} MiB", mib(plan.pinned_bytes));
    println!("  M_layer_weight   : {:>10.1} MiB", mib(plan.per_layer_weight_bytes));
    println!("  usable           : {:>10.1} MiB", mib(plan.usable_bytes));
    println!("  ▶ layers_to_load : {:>10} / {}", plan.layers_to_load, plan.num_layers);
    println!("  ▶ resident       : {:>9.1}%", plan.resident_fraction() * 100.0);
    println!("──────────────────────────────────────────────");
}

/// A representative Llama-3-70B-class configuration for off-GPU demonstration.
fn sample_70b_config() -> ModelConfig {
    let json = br#"{
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 80,
        "vocab_size": 128256,
        "intermediate_size": 28672,
        "max_position_embeddings": 8192
    }"#;
    ModelConfig::from_json_bytes(json, QuantScheme::Int4)
        .expect("built-in sample config is valid")
}

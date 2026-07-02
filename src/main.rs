//! `flip` binary — command-line entry point (`specs.md` §4).
//!
//! Two subcommands:
//! * `flip profile` — map/estimate a model and print the VRAM plan, KV-cache
//!   sizing, and streaming schedule. No GPU required.
//! * `flip serve`   — resolve the full serving configuration and prepare the
//!   engine. The inference/serving loop itself is Phase 3; this validates the
//!   config and runs the planning pipeline so the setup is verifiable today.

use clap::Parser;
use flip::cache::{KvCacheConfig, PagedKvCache};
use flip::cli::{Cli, Command, ProfileArgs, ServeArgs};
use flip::memory::{page_size, PinnedBuffer};
use flip::model::{ModelConfig, QuantScheme};
use flip::pipeline::{DoubleBufferSchedule, HostPipeline, MmapWeightSource, TieredWeightSource};
use flip::profiler::{VramPlan, VramProfiler};
use flip::storage::{LayerCatalog, MmapStore};
use flip::swap::LayerSwapPlan;
use flip::{gpu, Result};
use std::path::Path;

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Profile(args) => run_profile(args),
        Command::Serve(args) => run_serve(args),
    }
}

/// Print the shared startup banner (backend + host page size).
fn banner() {
    println!("flip v{}", env!("CARGO_PKG_VERSION"));
    println!("  gpu backend  : {}", gpu::active_vendor().label());
    println!("  host page    : {} bytes", page_size());
    println!();
}

/// `flip profile` — profile a real or sample model and print the full plan.
fn run_profile(args: ProfileArgs) -> Result<()> {
    banner();

    let quant = args.quant.to_scheme();
    let (config, source) = match &args.model_path {
        Some(dir) => (
            ModelConfig::from_path(dir, quant)?,
            format!("config.json in {}", dir.display()),
        ),
        None => (
            sample_70b_config(quant),
            "built-in Llama-3-70B-class sample".to_string(),
        ),
    };

    println!("model source : {source}");
    print_geometry(&config);

    report_plan(
        &config,
        args.model_path.as_deref(),
        args.context_length,
        args.vram_budget_gb,
        args.ram_cache_gb,
    )
}

/// `flip serve` — resolve the serving config, then run the planning pipeline.
fn run_serve(args: ServeArgs) -> Result<()> {
    banner();

    println!("serve config :");
    println!("  model      : {}", args.model_path.display());
    println!("  api        : http://{}:{}", args.host, args.port);
    println!("  context    : {} tokens", args.context_length);
    println!("  quant      : {:?}", args.quant.to_scheme());
    println!("  mode       : {:?}", args.distributed_mode);
    if let Some(draft) = &args.draft_model_path {
        println!("  draft model: {}", draft.display());
    }
    if !args.multi_gpu_ids.is_empty() {
        println!("  gpu ids    : {:?}", args.multi_gpu_ids);
    }
    if !args.worker_nodes.is_empty() {
        println!("  workers    : {}", args.worker_nodes.join(", "));
    }
    println!();

    let config = ModelConfig::from_path(&args.model_path, args.quant.to_scheme())?;
    println!("model source : config.json in {}", args.model_path.display());
    print_geometry(&config);

    report_plan(
        &config,
        Some(&args.model_path),
        args.context_length,
        args.vram_budget_gb,
        args.ram_cache_gb,
    )?;

    println!();
    println!("note         : model prepared; the inference serving loop is not yet");
    println!("               implemented (Phase 3 — OpenAI-compatible API server).");
    Ok(())
}

/// Shared planner/reporter: map the model (if a dir is given), profile it, size
/// the KV cache, and print the streaming schedule — streaming real weights when
/// a mapped model is available.
fn report_plan(
    config: &ModelConfig,
    model_dir: Option<&Path>,
    context_length: u32,
    vram_budget_gb: Option<f64>,
    ram_cache_gb: Option<f64>,
) -> Result<()> {
    // Map shards + measure real weight sizes when a directory is supplied.
    let mut mapped: Option<(MmapStore, LayerCatalog)> = None;
    if let Some(dir) = model_dir {
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

    // Resolve free VRAM: explicit budget > live device query > simulated 16 GiB.
    let profiler = VramProfiler::new(context_length);
    let (free_bytes, free_source) = resolve_free_bytes(vram_budget_gb);
    println!("free VRAM    : {free_source}");

    let plan = match catalog {
        Some(cat) if !cat.is_empty() => profiler.plan_from_catalog(config, cat, free_bytes),
        _ => profiler.plan_with_free(config, free_bytes),
    };
    print_plan(&plan);

    // Size the PagedAttention KV pool from the plan's KV budget (Cache Zone).
    let kv_cfg = KvCacheConfig::from_model(config, 16);
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

    // Allocate the pinned staging buffer the pipeline will DMA from.
    let staging: PinnedBuffer = swap.allocate_staging_buffer(plan.per_layer_weight_bytes)?;
    println!(
        "staging buf  : {:.1} MiB pinned ({:?})",
        staging.capacity() as f64 / (1024.0 * 1024.0),
        staging.kind(),
    );

    // Build the double-buffered A/B schedule (specs §3.2).
    let sched = DoubleBufferSchedule::from_swap_plan(&swap);
    println!(
        "pipeline     : {} steps, {} overlapped (DMA hidden under compute)",
        sched.num_steps(),
        sched.overlapping_steps(),
    );

    // With a real mapped model, stream its weights end-to-end through the host
    // pipeline (disk → pinned → device → compute).
    if let Some((store, catalog)) = mapped.as_ref().filter(|(_, c)| !c.is_empty()) {
        let inner = MmapWeightSource::new(store, catalog);
        let max_window = swap
            .passes
            .iter()
            .map(|p| inner.window_bytes(p))
            .max()
            .unwrap_or(0)
            .max(1);
        let mut pipeline = HostPipeline::new(max_window as usize)?;

        match ram_cache_gb {
            // Tiered RAM cache: run two forward passes to show cross-step reuse.
            Some(gb) => {
                let cache_bytes = (gb * GIB as f64) as u64;
                let tiered = TieredWeightSource::new(inner, cache_bytes);
                pipeline.execute(&sched, &tiered)?; // token step 1 (cold)
                let trace = pipeline.execute(&sched, &tiered)?; // token step 2 (warm)
                let moved: usize = trace.iter().map(|t| t.byte_len).sum();
                let cs = tiered.cache_stats();
                println!(
                    "streamed     : 2 forward pass(es), {:.2} GiB/pass through pinned staging",
                    moved as f64 / GIB as f64,
                );
                println!(
                    "ram cache    : {:.2} GiB budget → {} hits / {} misses ({:.0}% hit rate), {} evictions",
                    cache_bytes as f64 / GIB as f64,
                    cs.hits,
                    cs.misses,
                    cs.hit_rate() * 100.0,
                    cs.evictions,
                );
            }
            None => {
                let trace = pipeline.execute(&sched, &inner)?;
                let moved: usize = trace.iter().map(|t| t.byte_len).sum();
                println!(
                    "streamed     : {} window(s) executed, {:.2} GiB moved through pinned staging",
                    trace.len(),
                    moved as f64 / GIB as f64,
                );
            }
        }
    }

    Ok(())
}

/// Resolve the free-VRAM figure and a human-readable description of its source.
fn resolve_free_bytes(vram_budget_gb: Option<f64>) -> (u64, String) {
    if let Some(gb) = vram_budget_gb {
        let bytes = (gb * GIB as f64) as u64;
        return (bytes, format!("{gb} GiB manual budget"));
    }
    match gpu::mem_get_info() {
        Ok(dev) => (
            dev.free,
            format!("live device query ({})", gpu::active_vendor().label()),
        ),
        Err(_) => (16 * GIB, "simulated 16 GiB (no GPU device)".to_string()),
    }
}

fn print_geometry(config: &ModelConfig) {
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
}

fn print_plan(plan: &VramPlan) {
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
fn sample_70b_config(quant: QuantScheme) -> ModelConfig {
    let json = br#"{
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 80,
        "vocab_size": 128256,
        "intermediate_size": 28672,
        "max_position_embeddings": 8192
    }"#;
    ModelConfig::from_json_bytes(json, quant).expect("built-in sample config is valid")
}

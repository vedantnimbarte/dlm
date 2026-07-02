//! `flip` binary — Phase 1 demonstration entry point.
//!
//! The full `flip serve` CLI (clap-based argument parsing, the OpenAI-compatible
//! server) lands in Phase 2/3. For now this binary exercises the Phase 1
//! foundation end-to-end with no GPU required: it profiles a representative
//! 70B-class model against a simulated 16 GiB card and prints the resulting
//! layer-streaming schedule. Pass a model directory to profile real weights.

use flip::memory::{page_size, PinnedBuffer};
use flip::model::{ModelConfig, QuantScheme};
use flip::profiler::VramProfiler;
use flip::storage::MmapStore;
use flip::swap::LayerSwapPlan;
use flip::{cuda, Result};

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<()> {
    println!("flip v{} — Phase 1 (Local Foundation)", env!("CARGO_PKG_VERSION"));
    println!("  cuda backend : {}", if cuda::is_available() { "enabled" } else { "disabled (host fallback)" });
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

    // If a model dir was given, map its shards to prove the storage engine works.
    if let Some(dir) = args.get(1) {
        if let Ok(store) = MmapStore::open_dir(dir) {
            println!(
                "storage      : mapped {} shard(s), {} tensors, {:.2} GiB on disk",
                store.num_shards(),
                store.num_tensors(),
                store.total_mapped_bytes() as f64 / GIB as f64,
            );
            println!();
        }
    }

    // Determine free VRAM: live device when CUDA is present, else simulate 16 GiB.
    let profiler = VramProfiler::new(8192);
    let plan = match profiler.profile(&config) {
        Ok(plan) => {
            println!("free VRAM    : live cudaMemGetInfo");
            plan
        }
        Err(_) => {
            let simulated_free = 16 * GIB;
            println!("free VRAM    : simulated {} GiB (no CUDA device)", simulated_free / GIB);
            profiler.plan_with_free(&config, simulated_free)
        }
    };

    print_plan(&plan);

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

    Ok(())
}

fn print_plan(plan: &flip::profiler::VramPlan) {
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!();
    println!("── VRAM PLAN ─────────────────────────────────");
    println!("  M_free           : {:>10.1} MiB", mib(plan.free_bytes));
    println!("  M_safety         : {:>10.1} MiB", mib(plan.safety_bytes));
    println!("  M_kv_total       : {:>10.1} MiB", mib(plan.kv_total_bytes));
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

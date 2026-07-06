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
use flip::cli::{
    Cli, Command, Device, DistributedMode, DoctorArgs, GenerateArgs, ProfileArgs, ServeArgs,
    TokenizeArgs,
};
use flip::forward::{BlockConfig, ComputeKernel, CpuKernel, LayerTensors};
use flip::generate::{GenerationConfig, Generator, Sampler};
use flip::loader::ModelParts;
use flip::tokenizer::BpeTokenizer;
use flip::memory::{page_size, PinnedBuffer};
use flip::model::{ModelConfig, QuantScheme};
use flip::pipeline::{DoubleBufferSchedule, HostPipeline, MmapWeightSource, TieredWeightSource};
use flip::profiler::{VramPlan, VramProfiler};
use flip::storage::{LayerCatalog, MmapStore};
use flip::swap::LayerSwapPlan;
use flip::{gpu, FlipError, Result};
use std::path::Path;

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Profile(args) => run_profile(args),
        Command::Serve(args) => run_serve(args),
        Command::Generate(args) => run_generate(args),
        Command::Tokenize(args) => run_tokenize(args),
        Command::Doctor(args) => run_doctor(args),
    }
}

/// `flip doctor` — environment diagnostics + a self-check. Reports the GPU
/// backend and device memory, runs a tiny CPU inference, probes the GPU at
/// runtime (on a `cuda-kernels` build), and optionally checks a checkpoint loads.
fn run_doctor(args: DoctorArgs) -> Result<()> {
    println!("flip doctor v{}", env!("CARGO_PKG_VERSION"));
    println!("  gpu backend  : {}", gpu::active_vendor().label());
    println!("  host page    : {} bytes", page_size());
    match gpu::mem_get_info() {
        Ok(m) => println!(
            "  gpu memory   : {:.1} / {:.1} GiB free",
            m.free as f64 / GIB as f64,
            m.total as f64 / GIB as f64,
        ),
        Err(_) => println!("  gpu memory   : no device (host fallback)"),
    }
    match cpu_self_check() {
        Ok(n) => println!("  cpu inference: ok ({n} tokens generated)"),
        Err(e) => println!("  cpu inference: FAILED — {e}"),
    }
    gpu_self_check();
    if let Some(dir) = &args.model_path {
        match checkpoint_check(dir) {
            Ok(msg) => println!("  checkpoint   : {msg}"),
            Err(e) => println!("  checkpoint   : FAILED — {e}"),
        }
    }
    Ok(())
}

/// A tiny synthetic block config for the self-checks.
fn tiny_cfg() -> BlockConfig {
    BlockConfig {
        hidden_size: 8,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        intermediate_size: 16,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    }
}

/// Run a few tokens through the real CPU path on a tiny synthetic model.
fn cpu_self_check() -> Result<usize> {
    let (vocab, hidden) = (16usize, 8usize);
    let cfg = tiny_cfg();
    let kernel = CpuKernel::new(cfg, vec![LayerTensors::zeros(&cfg)])?;
    let mut rng = Rng::new(7);
    let generator = Generator::new(
        kernel,
        rng.vec(vocab * hidden, 0.02),
        vec![1.0; hidden],
        rng.vec(vocab * hidden, 0.02),
        vocab,
        1e-5,
        KvCacheConfig { num_layers: 1, num_kv_heads: 1, head_dim: 4, block_size: 16 },
        8,
    )?;
    let out = generator.generate(
        &[1],
        &GenerationConfig { max_new_tokens: 4, eos_token: None, sampler: Sampler::Greedy },
    )?;
    Ok(out.len())
}

/// GPU runtime probe: on a `cuda-kernels` build, run one block on both the CPU
/// and GPU kernels and report the max divergence; otherwise note it was skipped.
#[cfg(feature = "cuda-kernels")]
fn gpu_self_check() {
    match gpu_parity_probe() {
        Ok(diff) => println!("  gpu parity   : ok (max |Δ| {diff:.2e} vs cpu)"),
        Err(e) => println!("  gpu parity   : unavailable at runtime — {e}"),
    }
}

#[cfg(not(feature = "cuda-kernels"))]
fn gpu_self_check() {
    println!("  gpu parity   : skipped (build with --features cuda-kernels)");
}

/// Build one layer, run it on CPU and GPU, return the max absolute difference.
/// Errors if the GPU is unavailable at runtime (`GpuKernel::new` fails).
#[cfg(feature = "cuda-kernels")]
fn gpu_parity_probe() -> Result<f32> {
    use flip::forward::{GpuKernel, KvLayerCache};
    let cfg = tiny_cfg();
    let mut rng = Rng::new(3);
    let s = 0.05;
    let layer = LayerTensors {
        q_proj: rng.vec(cfg.q_dim() * cfg.hidden_size, s),
        k_proj: rng.vec(cfg.kv_dim() * cfg.hidden_size, s),
        v_proj: rng.vec(cfg.kv_dim() * cfg.hidden_size, s),
        o_proj: rng.vec(cfg.hidden_size * cfg.q_dim(), s),
        gate_proj: rng.vec(cfg.intermediate_size * cfg.hidden_size, s),
        up_proj: rng.vec(cfg.intermediate_size * cfg.hidden_size, s),
        down_proj: rng.vec(cfg.hidden_size * cfg.intermediate_size, s),
        input_layernorm: vec![1.0; cfg.hidden_size],
        post_attention_layernorm: vec![1.0; cfg.hidden_size],
    };
    let cpu = CpuKernel::new(cfg, vec![layer.clone()])?;
    let gpu = GpuKernel::new(cfg, vec![layer], 8)?;
    let mut hc = vec![0.1f32; cfg.hidden_size];
    let mut hg = hc.clone();
    let mut kc = KvLayerCache::new(cfg.kv_dim());
    let mut kg = KvLayerCache::new(cfg.kv_dim());
    cpu.run_block(0, &mut hc, &mut kc, 0)?;
    gpu.run_block(0, &mut hg, &mut kg, 0)?;
    Ok(hc.iter().zip(&hg).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max))
}

/// Check that a checkpoint directory loads its config, maps its store, has the
/// pinned embedding tensor, and (if present) a working tokenizer.
fn checkpoint_check(dir: &Path) -> Result<String> {
    let config = ModelConfig::from_path(dir, QuantScheme::Fp16)?;
    let store = MmapStore::open_dir(dir)?;
    if store.locate("model.embed_tokens.weight").is_none() {
        return Err(FlipError::InvalidConfig("missing model.embed_tokens.weight".into()));
    }
    let tok = if dir.join("tokenizer.json").exists()
        || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists())
    {
        format!(", tokenizer {} tokens", BpeTokenizer::from_dir(dir)?.vocab_size())
    } else {
        ", no tokenizer (byte fallback)".to_string()
    };
    Ok(format!("{} layers, vocab {}{tok}", config.num_layers, config.vocab_size))
}

/// `flip tokenize` — encode text and report the round-trip.
fn run_tokenize(args: TokenizeArgs) -> Result<()> {
    let tokenizer = match &args.tokenizer {
        Some(dir) => BpeTokenizer::from_dir(dir)?,
        None => BpeTokenizer::bytes_only(),
    };
    let ids = tokenizer.encode(&args.text)?;
    let decoded = tokenizer.decode(&ids)?;
    println!("text       : {:?}", args.text);
    println!("vocab      : {} tokens", tokenizer.vocab_size());
    println!("encoded    : {} token(s)", ids.len());
    println!("ids        : {ids:?}");
    println!(
        "round-trip : {decoded:?} ({})",
        if decoded == args.text { "ok" } else { "LOSSY" }
    );
    Ok(())
}

/// Deterministic SplitMix64 PRNG for synthetic weights (no external deps).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A small weight in roughly [-scale, scale).
    fn weight(&mut self, scale: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        (unit * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.weight(scale)).collect()
    }
}

/// `flip generate` — end-to-end CPU generation. Loads a real model when
/// `--model-path` is given, otherwise synthesizes a random one. With `--text`
/// the prompt is tokenized (and the output detokenized) via a BPE tokenizer.
fn run_generate(args: GenerateArgs) -> Result<()> {
    println!("flip v{}", env!("CARGO_PKG_VERSION"));
    println!("  gpu backend  : {}", gpu::active_vendor().label());
    println!();

    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        eos_token: args.eos_token,
        sampler: Sampler::Greedy,
    };

    // A tokenizer is needed only for a text prompt.
    let tokenizer = if args.text.is_some() {
        Some(resolve_tokenizer(&args)?)
    } else {
        None
    };

    // Prompt token ids: tokenized text, or raw --prompt ids.
    let prompt_ids: Vec<u32> = match (&args.text, &tokenizer) {
        (Some(text), Some(tok)) => tok.encode(text)?,
        _ => args.prompt.clone(),
    };
    if prompt_ids.is_empty() {
        return Err(FlipError::InvalidConfig("prompt is empty".into()));
    }
    let max_context = (prompt_ids.len() + args.max_new_tokens) as u32;

    // Materialize the model (real checkpoint or synthetic) into host weights.
    let (parts, is_synthetic) = if let Some(dir) = &args.model_path {
        let config = ModelConfig::from_path(dir, QuantScheme::Fp16)?;
        let store = MmapStore::open_dir(dir)?;
        println!("generate     : model {}", dir.display());
        println!(
            "model        : vocab {}, hidden {}, {} layers, {} q-heads / {} kv-heads, head_dim {}",
            config.vocab_size,
            config.hidden_size,
            config.num_layers,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim(),
        );
        (flip::loader::load_model_parts(&store, &config, max_context)?, false)
    } else {
        let parts = build_synthetic_parts(&args, max_context)?;
        println!("generate     : demo with randomly-initialized weights (seed {})", args.seed);
        println!(
            "model        : vocab {}, hidden {}, {} layers, {} q-heads / {} kv-heads, head_dim {}",
            parts.vocab_size,
            parts.cfg.hidden_size,
            parts.layers.len(),
            parts.cfg.num_heads,
            parts.cfg.num_kv_heads,
            parts.cfg.head_dim,
        );
        (parts, true)
    };

    // Prompt ids must fit the model's vocabulary.
    if let Some(&max) = prompt_ids.iter().max() {
        if max as usize >= parts.vocab_size {
            return Err(FlipError::InvalidConfig(format!(
                "prompt token {max} out of vocab range {} (tokenizer/model mismatch?)",
                parts.vocab_size
            )));
        }
    }

    println!("device       : {:?}", args.device);
    if let Some(text) = &args.text {
        println!("prompt text  : {text:?}");
    }
    println!("prompt ids   : {prompt_ids:?}");

    generate_on_device(parts, args.device, &prompt_ids, &gen_cfg, tokenizer.as_ref())?;

    if is_synthetic {
        println!();
        println!("note         : weights are untrained random values — output is not");
        println!("               meaningful text.");
    }
    Ok(())
}

/// Wrap the model parts in the chosen kernel and run generation.
fn generate_on_device(
    parts: ModelParts,
    device: Device,
    prompt_ids: &[u32],
    gen_cfg: &GenerationConfig,
    tokenizer: Option<&BpeTokenizer>,
) -> Result<()> {
    match device {
        Device::Cpu => {
            let generator = parts.into_cpu_generator()?;
            run_generation(&generator, prompt_ids, gen_cfg, tokenizer)
        }
        #[cfg(feature = "cuda-kernels")]
        Device::Gpu => {
            let generator = parts.into_gpu_generator()?;
            run_generation(&generator, prompt_ids, gen_cfg, tokenizer)
        }
        #[cfg(not(feature = "cuda-kernels"))]
        Device::Gpu => Err(FlipError::InvalidConfig(
            "--device gpu requires building with `--features cuda-kernels`".into(),
        )),
    }
}

/// Generate and print ids, plus decoded text when a tokenizer is present.
fn run_generation<K: ComputeKernel>(
    generator: &Generator<K>,
    prompt_ids: &[u32],
    gen_cfg: &GenerationConfig,
    tokenizer: Option<&BpeTokenizer>,
) -> Result<()> {
    let generated = generator.generate(prompt_ids, gen_cfg)?;
    println!("generated ids: {generated:?}");
    if let Some(tok) = tokenizer {
        println!("generated txt: {:?}", tok.decode(&generated)?);
        let mut full = prompt_ids.to_vec();
        full.extend_from_slice(&generated);
        println!("full text    : {:?}", tok.decode(&full)?);
    }
    Ok(())
}

/// Pick a tokenizer for `generate`: explicit `--tokenizer`, else the model
/// directory if it ships one, else a raw byte tokenizer.
fn resolve_tokenizer(args: &GenerateArgs) -> Result<BpeTokenizer> {
    if let Some(dir) = &args.tokenizer {
        return BpeTokenizer::from_dir(dir);
    }
    if let Some(dir) = &args.model_path {
        if dir.join("vocab.json").exists() && dir.join("merges.txt").exists() {
            return BpeTokenizer::from_dir(dir);
        }
    }
    Ok(BpeTokenizer::bytes_only())
}

/// Build synthetic random-weight model parts from the geometry flags.
fn build_synthetic_parts(args: &GenerateArgs, max_context: u32) -> Result<ModelParts> {
    if args.hidden_size == 0 || args.num_heads == 0 || args.num_kv_heads == 0 {
        return Err(FlipError::InvalidConfig("dimensions must be > 0".into()));
    }
    if args.hidden_size % args.num_heads != 0 {
        return Err(FlipError::InvalidConfig(format!(
            "hidden_size ({}) not divisible by num_heads ({})",
            args.hidden_size, args.num_heads
        )));
    }
    if args.num_heads % args.num_kv_heads != 0 {
        return Err(FlipError::InvalidConfig(format!(
            "num_heads ({}) not divisible by num_kv_heads ({})",
            args.num_heads, args.num_kv_heads
        )));
    }

    let head_dim = args.hidden_size / args.num_heads;
    let cfg = BlockConfig {
        hidden_size: args.hidden_size,
        num_heads: args.num_heads,
        num_kv_heads: args.num_kv_heads,
        head_dim,
        intermediate_size: args.intermediate_size,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    };

    // Small random weights (RMSNorm keeps activations bounded).
    let scale = 0.02;
    let mut rng = Rng::new(args.seed);
    let layers: Vec<LayerTensors> = (0..args.num_layers)
        .map(|_| LayerTensors {
            q_proj: rng.vec(cfg.q_dim() * cfg.hidden_size, scale),
            k_proj: rng.vec(cfg.kv_dim() * cfg.hidden_size, scale),
            v_proj: rng.vec(cfg.kv_dim() * cfg.hidden_size, scale),
            o_proj: rng.vec(cfg.hidden_size * cfg.q_dim(), scale),
            gate_proj: rng.vec(cfg.intermediate_size * cfg.hidden_size, scale),
            up_proj: rng.vec(cfg.intermediate_size * cfg.hidden_size, scale),
            down_proj: rng.vec(cfg.hidden_size * cfg.intermediate_size, scale),
            input_layernorm: vec![1.0; cfg.hidden_size],
            post_attention_layernorm: vec![1.0; cfg.hidden_size],
        })
        .collect();

    let embedding = rng.vec(args.vocab_size * args.hidden_size, scale);
    let lm_head = rng.vec(args.vocab_size * args.hidden_size, scale);
    let final_norm = vec![1.0; args.hidden_size];

    let kv_config = KvCacheConfig {
        num_layers: args.num_layers,
        num_kv_heads: args.num_kv_heads as u32,
        head_dim: head_dim as u32,
        block_size: 16,
    };
    let kv_blocks = (max_context as u64).div_ceil(16) as u32 + 2;

    Ok(ModelParts {
        cfg,
        layers,
        embedding,
        final_norm,
        lm_head,
        vocab_size: args.vocab_size,
        rms_eps: 1e-5,
        kv_config,
        kv_blocks,
    })
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
        println!("  draft model: {} (gamma {})", draft.display(), args.draft_gamma);
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

    let store = MmapStore::open_dir(&args.model_path)?;
    let listen = format!("{}:{}", args.host, args.port);

    match args.distributed_mode {
        DistributedMode::Worker => {
            // Serve this node's layer shard (the whole model here) to a master.
            let parts = flip::loader::load_model_parts(&store, &config, args.context_length)?;
            let worker = flip::distributed::Worker::new(parts.cfg, parts.layers)?;
            let listener = flip::distributed::worker::bind(&listen)?;
            println!();
            println!("worker node  : listening on {listen} ({} layers)", config.num_layers);
            worker.serve(listener)?; // blocks
            Ok(())
        }
        DistributedMode::Standalone | DistributedMode::Master => {
            // Multi-GPU is exclusive; --stream and --device may combine
            // (--stream --device gpu = stream a window through VRAM).
            if !args.multi_gpu_ids.is_empty() && (args.stream || args.device == Device::Gpu) {
                return Err(FlipError::InvalidConfig(
                    "--multi-gpu-ids cannot combine with --stream or --device gpu".into(),
                ));
            }

            // Streaming path: keep only a window of layers resident and stream the
            // rest from disk, so a model can exceed the resident budget. Streams to
            // host RAM (CPU) or into VRAM (--device gpu).
            if args.stream {
                if args.draft_model_path.is_some() {
                    return Err(FlipError::InvalidConfig(
                        "--stream does not support speculative decoding (--draft-model-path) yet".into(),
                    ));
                }
                let window = resident_window(&config, &args);
                println!();
                let dest = if args.device == Device::Gpu { "VRAM" } else { "host RAM" };
                println!(
                    "streaming    : {window} / {} layers resident in {dest}, rest streamed from disk",
                    config.num_layers,
                );
                if args.device == Device::Gpu {
                    return serve_streaming_gpu(store, &config, &args, window, &listen);
                }
                let generator =
                    flip::loader::build_streaming_generator(store, &config, args.context_length, window)?;
                return start_batched_server(generator, None, &args, &config, &listen);
            }

            // Non-streaming paths materialize all layers resident.
            let parts = flip::loader::load_model_parts(&store, &config, args.context_length)?;

            // Optional draft model for speculative decoding. It must share the
            // target's vocabulary (same tokenizer) for the accept/reject rule to
            // be exact. Loaded as parts so it can take the same kernel as the target.
            let draft_parts = match &args.draft_model_path {
                Some(dir) => {
                    let dcfg = ModelConfig::from_path(dir, args.quant.to_scheme())?;
                    if dcfg.vocab_size != config.vocab_size {
                        return Err(FlipError::InvalidConfig(format!(
                            "draft vocab {} != target vocab {}",
                            dcfg.vocab_size, config.vocab_size
                        )));
                    }
                    let dstore = MmapStore::open_dir(dir)?;
                    Some(flip::loader::load_model_parts(&dstore, &dcfg, args.context_length)?)
                }
                None => None,
            };

            if !args.multi_gpu_ids.is_empty() {
                // Split the target across local GPUs (specs §3.3); the small,
                // pinned draft stays on the first GPU.
                let ids = args.multi_gpu_ids.clone();
                let split = flip::distributed::partition_layers(config.num_layers as usize, ids.len());
                println!();
                println!("multi-gpu    : pipeline-parallel layer split (specs §3.3)");
                for (stage, shard) in split.iter().enumerate() {
                    println!(
                        "  gpu {:<6}: layers {}..{} ({} layer(s))",
                        ids[stage], shard.start, shard.end, shard.len(),
                    );
                }
                let generator = parts.into_pipeline_parallel_generator(&ids)?;
                let draft = draft_parts
                    .map(|p| p.into_pipeline_parallel_generator(&ids[..1]))
                    .transpose()?;
                start_batched_server(generator, draft, &args, &config, &listen)
            } else if args.device == Device::Gpu {
                serve_on_gpu(parts, draft_parts, &args, &config, &listen)
            } else {
                let generator = parts.into_cpu_generator()?;
                let draft = draft_parts.map(|p| p.into_cpu_generator()).transpose()?;
                start_batched_server(generator, draft, &args, &config, &listen)
            }
        }
    }
}

/// Resolve the streaming resident-layer window: the explicit override, else the
/// VRAM plan's `layers_to_load` for the model + context.
fn resident_window(config: &ModelConfig, args: &ServeArgs) -> usize {
    if let Some(n) = args.resident_layers {
        return n.max(1);
    }
    let (free_bytes, _) = resolve_free_bytes(args.vram_budget_gb);
    let plan = VramProfiler::new(args.context_length).plan_with_free(config, free_bytes);
    (plan.layers_to_load as usize).max(1)
}

/// Serve with the GPU kernel (all layers resident in VRAM). Feature-gated on
/// `cuda-kernels`; the draft model, if any, is also placed on the GPU.
#[cfg(feature = "cuda-kernels")]
fn serve_on_gpu(
    parts: ModelParts,
    draft_parts: Option<ModelParts>,
    args: &ServeArgs,
    config: &ModelConfig,
    listen: &str,
) -> Result<()> {
    println!();
    println!("device       : gpu ({})", gpu::active_vendor().label());
    let generator = parts.into_gpu_generator()?;
    let draft = draft_parts.map(|p| p.into_gpu_generator()).transpose()?;
    start_batched_server(generator, draft, args, config, listen)
}

#[cfg(not(feature = "cuda-kernels"))]
fn serve_on_gpu(
    _parts: ModelParts,
    _draft_parts: Option<ModelParts>,
    _args: &ServeArgs,
    _config: &ModelConfig,
    _listen: &str,
) -> Result<()> {
    Err(FlipError::InvalidConfig(
        "--device gpu requires building with `--features cuda-kernels`".into(),
    ))
}

/// Serve with `--stream --device gpu`: stream a window of layer weights through
/// VRAM (`StreamingGpuKernel`). Experimental — unvalidated on hardware.
#[cfg(feature = "cuda-kernels")]
fn serve_streaming_gpu(
    store: MmapStore,
    config: &ModelConfig,
    args: &ServeArgs,
    window: usize,
    listen: &str,
) -> Result<()> {
    println!("device       : gpu ({}) — VRAM layer streaming [experimental]", gpu::active_vendor().label());
    let generator =
        flip::loader::build_streaming_gpu_generator(store, config, args.context_length, window)?;
    start_batched_server(generator, None, args, config, listen)
}

#[cfg(not(feature = "cuda-kernels"))]
fn serve_streaming_gpu(
    _store: MmapStore,
    _config: &ModelConfig,
    _args: &ServeArgs,
    _window: usize,
    _listen: &str,
) -> Result<()> {
    Err(FlipError::InvalidConfig(
        "--stream --device gpu requires building with `--features cuda-kernels`".into(),
    ))
}

/// Build the batched (optionally speculative) streaming engine over any compute
/// kernel `K` and serve the OpenAI-compatible API. Generic so the CPU and
/// multi-GPU pipeline kernels share one server path.
fn start_batched_server<K: ComputeKernel + Send + 'static>(
    generator: Generator<K>,
    draft: Option<Generator<K>>,
    args: &ServeArgs,
    config: &ModelConfig,
    listen: &str,
) -> Result<()> {
    let has_hf_tokenizer = args.model_path.join("tokenizer.json").exists();
    let has_gpt2_tokenizer = args.model_path.join("vocab.json").exists()
        && args.model_path.join("merges.txt").exists();
    let tokenizer = if has_hf_tokenizer || has_gpt2_tokenizer {
        BpeTokenizer::from_dir(&args.model_path)?
    } else {
        BpeTokenizer::bytes_only()
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let template = flip::server::engine::ChatTemplate::parse(&args.chat_template).ok_or_else(|| {
        FlipError::InvalidConfig(format!(
            "unknown --chat-template {:?} (expected plain, chatml, or llama3)",
            args.chat_template
        ))
    })?;
    // Batched, streaming engine: a background scheduler interleaves concurrent
    // requests a token at a time. With a draft model it decodes speculatively
    // (draft proposes, target verifies).
    let speculative = draft.is_some();
    let engine = match draft {
        Some(d) => flip::server::EngineService::start_speculative(
            generator,
            d,
            args.draft_gamma,
            tokenizer,
            config.vocab_size as usize,
            "flip",
            128,
            created,
            8, // max concurrent batch
        ),
        None => flip::server::EngineService::start(
            generator,
            tokenizer,
            config.vocab_size as usize,
            "flip",
            128,
            created,
            8, // max concurrent batch
        ),
    }
    .with_chat_template(template);
    let server = flip::server::HttpServer::bind(listen)?;
    println!();
    let mode = if speculative { "batched + speculative" } else { "batched" };
    println!("serving      : OpenAI-compatible API on http://{listen} ({mode})");
    println!("  endpoints  : POST /v1/chat/completions (stream supported), GET /v1/models");
    if args.api_key.is_some() {
        println!("  auth       : bearer token required on /v1/*");
    }
    if args.distributed_mode == DistributedMode::Master && !args.worker_nodes.is_empty() {
        println!("  note       : master mode — {} worker(s) configured; the server", args.worker_nodes.len());
        println!("               currently runs the model locally (distributed routing available");
        println!("               via flip::distributed::Coordinator).");
    }
    let router = flip::server::engine::secured_router(engine, args.api_key.clone());
    server.serve(router) // blocks
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

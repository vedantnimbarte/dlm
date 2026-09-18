//! `dlm` binary — command-line entry point.
//!
//! Two subcommands:
//! * `dlm profile` — map/estimate a model and print the VRAM plan, KV-cache
//!   sizing, and streaming schedule. No GPU required.
//! * `dlm serve`   — resolve the full serving configuration and prepare the
//!   engine. The inference/serving loop itself is Phase 3; this validates the
//!   config and runs the planning pipeline so the setup is verifiable today.

use clap::{CommandFactory, Parser};
use dlm::cache::{KvCacheConfig, PagedKvCache};
use dlm::cli::{
    BenchArgs, BenchOpts, Cli, Command, CompletionsArgs, Device, DistributedMode, DoctorArgs,
    GenerateArgs, KvQuantArg, ProfileArgs, PullArgs, QuantArg, ScoreArgs, SearchArgs, ServeArgs,
    TokenizeArgs,
};
use dlm::forward::Weights;
use dlm::forward::{BlockConfig, ComputeKernel, CpuKernel, ExpertFfn, Ffn, LayerTensors};
use dlm::generate::{GenerationConfig, Generator, Sampler};
use dlm::loader::ModelParts;
use dlm::memory::page_size;
use dlm::model::{ModelConfig, QuantScheme};
use dlm::pipeline::{DoubleBufferSchedule, HostPipeline, MmapWeightSource, TieredWeightSource};
use dlm::profiler::{VramPlan, VramProfiler};
use dlm::storage::{LayerCatalog, MmapStore};
use dlm::swap::LayerSwapPlan;
use dlm::tokenizer::BpeTokenizer;
use dlm::{gpu, DlmError, Result};
use std::path::Path;

const GIB: u64 = 1024 * 1024 * 1024;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Profile(args) => run_profile(args),
        Command::Serve(args) => run_serve(args),
        Command::Bench(args) => run_bench(args),
        Command::Generate(args) => run_generate(args),
        Command::Score(args) => run_score(args),
        Command::Tokenize(args) => run_tokenize(args),
        Command::Doctor(args) => run_doctor(args),
        Command::Search(args) => run_search(args),
        Command::Pull(args) => run_pull(args),
        Command::Completions(args) => run_completions(args),
    }
}

/// `dlm completions <shell>` — print a completion script to stdout.
fn run_completions(args: CompletionsArgs) -> Result<()> {
    clap_complete::generate(
        args.shell,
        &mut Cli::command(),
        "dlm",
        &mut std::io::stdout(),
    );
    Ok(())
}

/// `dlm search` — query the Hugging Face hub and print matching models.
fn run_search(args: SearchArgs) -> Result<()> {
    let hits = dlm::hub::search(&args.query, args.limit)?;
    if hits.is_empty() {
        println!("no models found for {:?}", args.query);
        return Ok(());
    }
    for h in &hits {
        let task = h.task.as_deref().unwrap_or("-");
        println!(
            "{:<55} ⭳ {:<10} ♥ {:<6} {}",
            h.id, h.downloads, h.likes, task
        );
    }
    println!("\npull one with:  dlm pull <id>");
    Ok(())
}

/// `dlm pull` — download a model's loadable files from the Hugging Face hub.
fn run_pull(args: PullArgs) -> Result<()> {
    let token = args.token.or_else(|| std::env::var("HF_TOKEN").ok());
    dlm::hub::pull(&args.repo, args.local_dir, token.as_deref())?;
    Ok(())
}

/// `dlm doctor` — environment diagnostics + a self-check. Reports the GPU
/// backend and device memory, runs a tiny CPU inference, probes the GPU at
/// runtime (on a `cuda-kernels` build), and optionally checks a checkpoint loads.
fn run_doctor(args: DoctorArgs) -> Result<()> {
    println!("dlm doctor v{}", env!("CARGO_PKG_VERSION"));
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
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: Default::default(),
        mla: None,
        sliding_window_pattern: None,
        rope_local_theta: None,
        attn_logit_softcap: None,
        query_pre_attn_scalar: None,
        gemma2_norms: false,
        ffn_kind: Default::default(),
        norm_kind: Default::default(),
        parallel_residual: false,
        learned_positions: false,
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
        KvCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            block_size: 16,
        },
        8,
    )?;
    let out = generator.generate(
        &[1],
        &GenerationConfig {
            max_new_tokens: 4,
            eos_token: None,
            sampler: Sampler::Greedy,
        },
    )?;
    Ok(out.len())
}

/// GPU runtime probe: on a `cuda-kernels` build, run one block on both the CPU
/// and GPU kernels and report the max divergence; otherwise note it was skipped.
#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
fn gpu_self_check() {
    match gpu_parity_probe() {
        Ok(diff) => println!("  gpu parity   : ok (max |Δ| {diff:.2e} vs cpu)"),
        Err(e) => println!("  gpu parity   : unavailable at runtime — {e}"),
    }
}

#[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
fn gpu_self_check() {
    println!("  gpu parity   : skipped (build with --features cuda-kernels)");
}

/// Build one layer, run it on CPU and GPU, return the max absolute difference.
/// Errors if the GPU is unavailable at runtime (`GpuKernel::new` fails).
#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
fn gpu_parity_probe() -> Result<f32> {
    use dlm::forward::{GpuKernel, KvLayerCache};
    let cfg = tiny_cfg();
    let mut rng = Rng::new(3);
    let s = 0.05;
    let layer = LayerTensors {
        q_proj: Weights::from_f32(rng.vec(cfg.q_dim() * cfg.hidden_size, s)),
        k_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * cfg.hidden_size, s)),
        v_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * cfg.hidden_size, s)),
        o_proj: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.q_dim(), s)),
        ffn: Ffn::Dense(ExpertFfn {
            gate: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, s)),
            up: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, s)),
            down: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.intermediate_size, s)),
            up_bias: None,
            down_bias: None,
        }),
        input_layernorm: vec![1.0; cfg.hidden_size],
        post_attention_layernorm: vec![1.0; cfg.hidden_size],
        ..Default::default()
    };
    let cpu = CpuKernel::new(cfg, vec![layer.clone()])?;
    let gpu = GpuKernel::new(cfg, vec![layer], 8)?;
    let mut hc = vec![0.1f32; cfg.hidden_size];
    let mut hg = hc.clone();
    let mut kc = KvLayerCache::new(cfg.kv_dim());
    let mut kg = KvLayerCache::new(cfg.kv_dim());
    cpu.run_block(0, &mut hc, &mut kc, 0)?;
    gpu.run_block(0, &mut hg, &mut kg, 0)?;
    Ok(hc
        .iter()
        .zip(&hg)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max))
}

/// Check that a checkpoint directory loads its config, maps its store, has the
/// pinned embedding tensor, and (if present) a working tokenizer.
fn checkpoint_check(dir: &Path) -> Result<String> {
    let config = ModelConfig::from_path(dir, QuantScheme::Fp16)?;
    let store = MmapStore::open_dir(dir)?;
    if store.locate("model.embed_tokens.weight").is_none() {
        return Err(DlmError::InvalidConfig(
            "missing model.embed_tokens.weight".into(),
        ));
    }
    let tok = if dir.join("tokenizer.json").exists()
        || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists())
    {
        format!(
            ", tokenizer {} tokens",
            BpeTokenizer::from_dir(dir)?.vocab_size()
        )
    } else {
        ", no tokenizer (byte fallback)".to_string()
    };
    Ok(format!(
        "{} layers, vocab {}{tok}",
        config.num_layers, config.vocab_size
    ))
}

/// `dlm tokenize` — encode text and report the round-trip.
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

/// `dlm generate` — end-to-end CPU generation. Loads a real model when
/// `--model-path` is given, otherwise synthesizes a random one. With `--text`
/// the prompt is tokenized (and the output detokenized) via a BPE tokenizer.
fn run_generate(args: GenerateArgs) -> Result<()> {
    println!("dlm v{}", env!("CARGO_PKG_VERSION"));
    println!("  gpu backend  : {}", gpu::active_vendor().label());
    println!();

    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        eos_token: args.eos_token,
        sampler: match args.temperature {
            Some(temperature) if temperature > 0.0 => Sampler::TopPK {
                temperature,
                top_p: args.top_p,
                top_k: args.top_k,
                min_p: 0.0,
                repetition_penalty: 1.0,
                seed: args.seed,
            },
            _ => Sampler::Greedy,
        },
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
        return Err(DlmError::InvalidConfig("prompt is empty".into()));
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
        (
            dlm::loader::load_model_parts(&store, &config, max_context)?,
            false,
        )
    } else {
        let parts = build_synthetic_parts(&args, max_context)?;
        println!(
            "generate     : demo with randomly-initialized weights (seed {})",
            args.seed
        );
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
            return Err(DlmError::InvalidConfig(format!(
                "prompt token {max} out of vocab range {} (tokenizer/model mismatch?)",
                parts.vocab_size
            )));
        }
    }

    let device = resolve_device(args.device);
    println!("device       : {device:?}");
    if let Some(text) = &args.text {
        println!("prompt text  : {text:?}");
    }
    println!("prompt ids   : {prompt_ids:?}");

    generate_on_device(parts, device, &prompt_ids, &gen_cfg, tokenizer.as_ref())?;

    if is_synthetic {
        println!();
        println!("note         : weights are untrained random values — output is not");
        println!("               meaningful text.");
    }
    Ok(())
}

/// `dlm score`: the log-probability the model gives each token of a text, and
/// the distribution it would generate from next.
///
/// This is the number a parity check needs. Greedy output can look fine while
/// the forward pass is subtly wrong — a norm applied in the wrong place, a RoPE
/// base off by a factor — and the first place that shows up is the probability
/// mass, not the argmax. `tools/hf_parity.py` compares this JSON against
/// Hugging Face transformers for the same text.
fn run_score(args: ScoreArgs) -> Result<()> {
    let tokenizer = match &args.tokenizer {
        Some(dir) => BpeTokenizer::from_dir(dir)?,
        None => serve_tokenizer(&args.model_path)?,
    };
    let tokens = match &args.text {
        Some(text) => tokenizer.encode(text)?,
        None if !args.prompt.is_empty() => args.prompt.clone(),
        None => {
            return Err(DlmError::InvalidConfig(
                "pass --text to score, or --prompt with token ids".into(),
            ))
        }
    };
    if tokens.len() < 2 {
        return Err(DlmError::InvalidConfig(format!(
            "scoring needs at least 2 tokens (got {}): the first is the context \
             for the second",
            tokens.len()
        )));
    }

    let config = ModelConfig::from_path(&args.model_path, QuantScheme::Fp16)?;
    let store = MmapStore::open_dir(&args.model_path)?;
    let parts = dlm::loader::load_model_parts(&store, &config, tokens.len() as u32)?;
    let device = resolve_device(args.device);
    let (logprobs, next) = score_on_device(parts, device, &tokens)?;

    // The mean of the negated log-probabilities: the text's cross-entropy.
    let mean_nll = -logprobs.iter().sum::<f32>() / logprobs.len() as f32;
    let mut ranked: Vec<u32> = (0..next.len() as u32).collect();
    ranked.sort_unstable_by(|&a, &b| next[b as usize].total_cmp(&next[a as usize]));
    ranked.truncate(args.top);

    if let Some(path) = &args.json {
        let json = serde_json::json!({
            "model": args.model_path.display().to_string(),
            "device": format!("{device:?}").to_lowercase(),
            "tokens": tokens,
            // `logprobs[i]` scores `tokens[i + 1]`, so it is one shorter.
            "logprobs": logprobs,
            "mean_nll": mean_nll,
            "next": ranked.iter().map(|&id| serde_json::json!({
                "id": id,
                "logprob": next[id as usize],
            })).collect::<Vec<_>>(),
        });
        let bytes = serde_json::to_vec_pretty(&json).map_err(|source| DlmError::Json {
            context: "score output".to_string(),
            source,
        })?;
        std::fs::write(path, bytes).map_err(|source| DlmError::Io {
            path: path.clone(),
            source,
        })?;
        println!("wrote {}", path.display());
        return Ok(());
    }

    // A token decoded on its own can be half a character, so a piece that is not
    // valid on its own is shown as the empty string rather than failing.
    let piece = |id: u32| tokenizer.decode(&[id]).unwrap_or_default();
    println!("device       : {device:?}");
    println!("tokens       : {}", tokens.len());
    println!("mean nll     : {mean_nll:.4}");
    println!();
    println!("  {:>8}  {:>10}  piece", "token", "logprob");
    for (i, lp) in logprobs.iter().enumerate() {
        println!(
            "  {:>8}  {lp:>10.4}  {:?}",
            tokens[i + 1],
            piece(tokens[i + 1])
        );
    }
    println!();
    println!("next token, most likely first:");
    for id in ranked {
        println!("  {id:>8}  {:>10.4}  {:?}", next[id as usize], piece(id));
    }
    Ok(())
}

/// Wrap the model parts in the chosen kernel and score `tokens`.
fn score_on_device(
    parts: ModelParts,
    device: Device,
    tokens: &[u32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    match device {
        Device::Cpu => parts.into_cpu_generator()?.score_and_next(tokens),
        #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
        Device::Gpu => parts.into_gpu_generator()?.score_and_next(tokens),
        #[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
        Device::Gpu => Err(gpu_compute_unavailable("--device gpu")),
    }
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
        #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
        Device::Gpu => {
            let generator = parts.into_gpu_generator()?;
            run_generation(&generator, prompt_ids, gen_cfg, tokenizer)
        }
        #[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
        Device::Gpu => Err(gpu_compute_unavailable("--device gpu")),
    }
}

/// Generate and print ids, plus decoded text when a tokenizer is present.
fn run_generation<K: ComputeKernel>(
    generator: &Generator<K>,
    prompt_ids: &[u32],
    gen_cfg: &GenerationConfig,
    tokenizer: Option<&BpeTokenizer>,
) -> Result<()> {
    // Step a session rather than calling `generate`, so prefill and decode can
    // be timed apart: a long prompt would otherwise hide the decode speed.
    let start = std::time::Instant::now();
    let mut session = generator.start_session(prompt_ids, gen_cfg.sampler)?;
    let prefill = start.elapsed().as_secs_f64();
    let decode_start = std::time::Instant::now();
    let mut generated = Vec::with_capacity(gen_cfg.max_new_tokens);
    while generated.len() < gen_cfg.max_new_tokens {
        let next = session.step()?;
        generated.push(next);
        if gen_cfg.eos_token == Some(next) {
            break;
        }
    }
    let decode = decode_start.elapsed().as_secs_f64();
    eprintln!(
        "timing       : prefill {} tokens in {prefill:.2} s ({:.1} tok/s), decode {} tokens in {decode:.2} s ({:.2} tok/s)",
        prompt_ids.len(),
        prompt_ids.len() as f64 / prefill,
        generated.len(),
        generated.len() as f64 / decode,
    );
    println!("generated ids: {generated:?}");
    if let Some(tok) = tokenizer {
        println!("generated txt: {:?}", tok.decode(&generated)?);
        let mut full = prompt_ids.to_vec();
        full.extend_from_slice(&generated);
        println!("full text    : {:?}", tok.decode(&full)?);
    }
    Ok(())
}

/// Load the tokenizer a served model ships (HF `tokenizer.json` or GPT-2
/// `vocab.json`+`merges.txt`), falling back to a raw byte tokenizer.
fn serve_tokenizer(model_path: &Path) -> Result<BpeTokenizer> {
    let has_hf = model_path.join("tokenizer.json").exists();
    let has_gpt2 = model_path.join("vocab.json").exists() && model_path.join("merges.txt").exists();
    if has_hf || has_gpt2 {
        BpeTokenizer::from_dir(model_path)
    } else {
        Ok(BpeTokenizer::bytes_only())
    }
}

/// Resolve `--chat-template`. A named template is used as given; `auto` reads
/// the checkpoint's Jinja template (`chat_template.jinja`, else the
/// `chat_template` field of `tokenizer_config.json`) and fingerprints it. A
/// checkpoint with no template (a base model) gets `plain`; one whose template
/// is not recognized also gets `plain`, with a warning naming the flag to set.
fn resolve_chat_template(args: &ServeArgs) -> Result<dlm::server::engine::ChatTemplate> {
    use dlm::server::engine::ChatTemplate;
    if !args.chat_template.eq_ignore_ascii_case("auto") {
        return ChatTemplate::parse(&args.chat_template).ok_or_else(|| {
            DlmError::InvalidConfig(format!(
                "unknown --chat-template {:?} (expected auto, {})",
                args.chat_template,
                ChatTemplate::NAMES
            ))
        });
    }
    let Some(jinja) = ChatTemplate::read_jinja(&args.model_path) else {
        println!("chat template: plain (checkpoint declares none)");
        return Ok(ChatTemplate::Plain);
    };
    match ChatTemplate::detect(&jinja) {
        Some(t) => {
            println!("chat template: {} (detected from checkpoint)", t.name());
            Ok(t)
        }
        None => {
            eprintln!(
                "warning: the checkpoint's chat template is not one dlm recognizes; using \
                 plain, which the model was not trained on. Pass --chat-template ({}) \
                 if one matches.",
                ChatTemplate::NAMES
            );
            Ok(ChatTemplate::Plain)
        }
    }
}

/// Pick a tokenizer for `generate`: explicit `--tokenizer`, else the model
/// directory if it ships one, else a raw byte tokenizer.
fn resolve_tokenizer(args: &GenerateArgs) -> Result<BpeTokenizer> {
    if let Some(dir) = &args.tokenizer {
        return BpeTokenizer::from_dir(dir);
    }
    if let Some(dir) = &args.model_path {
        // Same rule as `serve`: an HF `tokenizer.json` counts. Checking only for
        // the GPT-2 pair here silently fell back to the byte tokenizer for every
        // modern model, feeding raw bytes in as token ids.
        return serve_tokenizer(dir);
    }
    Ok(BpeTokenizer::bytes_only())
}

/// Build synthetic random-weight model parts from the geometry flags.
fn build_synthetic_parts(args: &GenerateArgs, max_context: u32) -> Result<ModelParts> {
    if args.hidden_size == 0 || args.num_heads == 0 || args.num_kv_heads == 0 {
        return Err(DlmError::InvalidConfig("dimensions must be > 0".into()));
    }
    if args.hidden_size % args.num_heads != 0 {
        return Err(DlmError::InvalidConfig(format!(
            "hidden_size ({}) not divisible by num_heads ({})",
            args.hidden_size, args.num_heads
        )));
    }
    if args.num_heads % args.num_kv_heads != 0 {
        return Err(DlmError::InvalidConfig(format!(
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
        rope_scaling: None,
        moe: None,
        sliding_window: None,
        activation: Default::default(),
        mla: None,
        ..Default::default()
    };

    // Small random weights (RMSNorm keeps activations bounded).
    let scale = 0.02;
    let mut rng = Rng::new(args.seed);
    let layers: Vec<LayerTensors> = (0..args.num_layers)
        .map(|_| LayerTensors {
            q_proj: Weights::from_f32(rng.vec(cfg.q_dim() * cfg.hidden_size, scale)),
            k_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * cfg.hidden_size, scale)),
            v_proj: Weights::from_f32(rng.vec(cfg.kv_dim() * cfg.hidden_size, scale)),
            o_proj: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.q_dim(), scale)),
            ffn: Ffn::Dense(ExpertFfn {
                gate: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, scale)),
                up: Weights::from_f32(rng.vec(cfg.intermediate_size * cfg.hidden_size, scale)),
                down: Weights::from_f32(rng.vec(cfg.hidden_size * cfg.intermediate_size, scale)),
                up_bias: None,
                down_bias: None,
            }),
            input_layernorm: vec![1.0; cfg.hidden_size],
            post_attention_layernorm: vec![1.0; cfg.hidden_size],
            ..Default::default()
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
        position_embedding: None,
        final_norm_bias: None,
        cfg,
        layers,
        embedding,
        final_norm,
        lm_head,
        vocab_size: args.vocab_size,
        rms_eps: 1e-5,
        kv_config,
        kv_blocks,
        embed_scale: None,
        final_logit_softcap: None,
    })
}

/// Print the shared startup banner (backend + host page size).
fn banner() {
    println!("dlm v{}", env!("CARGO_PKG_VERSION"));
    println!("  gpu backend  : {}", gpu::active_vendor().label());
    println!("  host page    : {} bytes", page_size());
    println!();
}

/// `dlm profile` — profile a real or sample model and print the full plan.
fn run_profile(args: ProfileArgs) -> Result<()> {
    banner();

    let (config, source) = match &args.model_path {
        Some(dir) => {
            // Probe the real checkpoint so the plan is sized from the weights it
            // actually has, not an assumed scheme.
            let quant = match MmapStore::open_dir(dir) {
                Ok(store) => resolve_quant(args.quant, &store)?,
                // No mappable store (config-only dir): fall back to the request.
                Err(_) => args.quant.map_or(QuantScheme::Fp16, |q| q.to_scheme()),
            };
            (
                ModelConfig::from_path(dir, quant)?,
                format!("config.json in {}", dir.display()),
            )
        }
        None => (
            // Hypothetical model: nothing to probe, so honour the request.
            sample_70b_config(args.quant.map_or(QuantScheme::Fp16, |q| q.to_scheme())),
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
        args.safety_margin_gb,
        args.ram_cache_gb,
    )
}

/// `dlm serve` — resolve the serving config, then run the planning pipeline.
/// Publish the static context a telemetry subscriber needs before it can make
/// sense of any event: how many layers there are, how big each one is, and
/// whether weights are streaming at all.
///
/// Sent once at startup rather than per request — none of it changes for the
/// life of the process, and a UI attaching mid-generation needs it immediately.
fn publish_telemetry_snapshot(
    config: &ModelConfig,
    store: &MmapStore,
    resident_layers: u32,
    streaming: bool,
    device: Device,
) {
    let catalog = LayerCatalog::build(store);
    let num_layers = if catalog.num_layers() == 0 {
        config.num_layers
    } else {
        catalog.num_layers()
    };
    let layer_bytes = (0..num_layers)
        .map(|i| catalog.layer_bytes(i).unwrap_or(0))
        .collect();
    dlm::telemetry::set_snapshot(dlm::telemetry::Snapshot {
        num_layers,
        resident_layers,
        layer_bytes,
        pinned_bytes: catalog.pinned_bytes(),
        streaming,
        device: match device {
            Device::Gpu => "gpu".into(),
            _ => "cpu".into(),
        },
        // Device-measured H2D timing exists only on a GPU build with the copy
        // stream. Anywhere else a duration would be wall-clock, and the
        // consumer must render it as an estimate rather than a measurement.
        estimated_h2d: !cfg!(any(feature = "cuda", feature = "rocm")) || device != Device::Gpu,
    });
}

fn run_serve(args: ServeArgs) -> Result<()> {
    banner();

    // Decide up front whether `/v1/telemetry` exists. Collection still costs
    // nothing until something actually subscribes.
    dlm::telemetry::set_available(args.telemetry);

    // Map the checkpoint first: the weight precision (and every plan derived from
    // it) comes from the dtype the file actually holds.
    let store = MmapStore::open_dir(&args.model_path)?;
    let quant = resolve_quant(args.quant, &store)?;

    println!("serve config :");
    println!("  model      : {}", args.model_path.display());
    println!("  api        : http://{}:{}", args.host, args.port);
    println!("  context    : {} tokens", args.context_length);
    println!("  quant      : {quant:?}");
    println!("  mode       : {:?}", args.distributed_mode);
    if let Some(draft) = &args.draft_model_path {
        println!(
            "  draft model: {} (gamma {})",
            draft.display(),
            args.draft_gamma
        );
    }
    if !args.multi_gpu_ids.is_empty() {
        println!("  gpu ids    : {:?}", args.multi_gpu_ids);
    }
    if !args.worker_nodes.is_empty() {
        println!("  workers    : {}", args.worker_nodes.join(", "));
    }
    println!();

    let config = ModelConfig::from_path(&args.model_path, quant)?;
    println!(
        "model source : config.json in {}",
        args.model_path.display()
    );
    print_geometry(&config);
    let listen = format!("{}:{}", args.host, args.port);

    match args.distributed_mode {
        DistributedMode::Worker => {
            // Serve this node's layer shard (the whole model here) to a master.
            let parts = dlm::loader::load_model_parts(&store, &config, args.context_length)?;
            let secret = cluster_secret(&args);
            let worker = dlm::distributed::Worker::new(parts.cfg, parts.layers)?
                .with_auth(secret.clone())
                .with_io_timeout(args.worker_timeout_secs.map(std::time::Duration::from_secs));
            let listener = dlm::distributed::worker::bind(&listen)?;
            println!();
            println!(
                "worker node  : listening on {listen} ({} layers)",
                config.num_layers
            );
            println!(
                "  auth       : {}",
                if secret.is_some() {
                    "cluster secret required"
                } else {
                    "OPEN — any peer can drive this worker (set --cluster-secret)"
                }
            );
            worker.serve(listener)?; // blocks
            Ok(())
        }
        DistributedMode::Master if !args.worker_nodes.is_empty() => {
            serve_distributed(store, &config, &args, &listen)
        }
        DistributedMode::Standalone | DistributedMode::Master => {
            // Multi-GPU already implies GPUs and overrides --device, so only
            // --stream conflicts with it.
            if !args.multi_gpu_ids.is_empty() && args.stream {
                return Err(DlmError::InvalidConfig(
                    "--multi-gpu-ids cannot combine with --stream".into(),
                ));
            }

            // Resolve the compute device once: honor an explicit CPU choice, and
            // for GPU fall back to CPU with a warning if no device is usable.
            // Multi-GPU ignores this (it drives the listed GPUs directly).
            let device = if args.multi_gpu_ids.is_empty() {
                resolve_device(args.device)
            } else {
                args.device
            };

            // Prefix caching stores a KV *snapshot* taken from the host-side
            // KvLayerCache. On the GPU compute kernels the real K/V lives in VRAM
            // and the host cache holds only length placeholders, so a resumed
            // prefix would read empty history and silently emit wrong output —
            // refuse it rather than mislead. (Multi-GPU runs the CPU kernel with
            // per-device dispatch, so its host KV is real and prefix caching is
            // fine there.)
            // (Prefix caching used to be refused here: the snapshot is host-side
            // while the GPU keeps K/V in VRAM, so it captured only the length
            // placeholders. The scheduler now takes the snapshot through
            // `snapshot_synced`, which copies the device K/V back to the host
            // first, and `gpu_kv` re-uploads it when a snapshot is resumed — so
            // the GPU path is supported.)

            // Streaming path: keep only a window of layers resident and stream the
            // rest from disk, so a model can exceed the resident budget. Streams to
            // host RAM (CPU) or into VRAM (--device gpu).
            // Baseline snapshot for a fully-resident run: every layer is held,
            // nothing streams. The streaming branch below replaces it with the
            // real window once the plan is known.
            if args.telemetry && !args.stream {
                publish_telemetry_snapshot(&config, &store, config.num_layers, false, device);
            }

            if args.stream {
                if args.draft_model_path.is_some() {
                    return Err(DlmError::InvalidConfig(
                        "--stream does not support speculative decoding (--draft-model-path) yet"
                            .into(),
                    ));
                }
                let plan = stream_plan(&config, &args, &store);
                let window = resident_window(&plan, &args);
                let ram_cache = resolve_ram_cache_bytes(&args, &plan);
                publish_telemetry_snapshot(&config, &store, window as u32, true, device);
                println!();
                let dest = if device == Device::Gpu {
                    "VRAM"
                } else {
                    "host RAM"
                };
                println!(
                    "streaming    : {window} / {} layers resident in {dest}, rest streamed from disk",
                    config.num_layers,
                );
                let depth = args.prefetch_depth.min(window.saturating_sub(1));
                if args.auto_prefetch {
                    println!(
                        "  prefetch   : auto (depth tuned from load vs compute, ≤ {})",
                        window.saturating_sub(1)
                    );
                } else if depth > 0 {
                    println!("  prefetch   : {depth} layer(s) ahead (overlaps load with compute)");
                } else {
                    println!("  prefetch   : off (window too small or --prefetch-depth 0)");
                }
                if ram_cache > 0 {
                    println!(
                        "  ram cache  : {:.1} MiB of materialized layers held in host RAM{}",
                        ram_cache as f64 / (1024.0 * 1024.0),
                        if args.ram_cache_gb.is_some() {
                            ""
                        } else {
                            " (default: every layer fits)"
                        },
                    );
                } else if args.ram_cache_gb.is_none() {
                    let layers = plan.per_layer_weight_bytes * plan.num_layers as u64;
                    let need = layers + layers / 4; // the headroom the default would add
                    println!(
                        "  ram cache  : off: caching every layer needs ~{:.1} GiB, over the default \
                         cap ({:.1} GiB, a quarter of RAM), so every miss re-reads the checkpoint. \
                         Pass --ram-cache-gb to cache them anyway.",
                        need as f64 / GIB as f64,
                        ram_cache_cap() as f64 / GIB as f64,
                    );
                }
                if device == Device::Gpu {
                    // Batch KV is planned into the window; refuse if it left no room
                    // for even one resident layer instead of OOMing mid-request.
                    ensure_batch_kv_fits(
                        plan.free_bytes,
                        plan.safety_bytes,
                        plan.kv_total_bytes,
                        (window as u64).saturating_mul(plan.per_layer_weight_bytes),
                        args.max_batch,
                        args.context_length,
                        "resident window",
                    )?;
                    let expert_cache = resolve_expert_cache_bytes(&args, &config, &plan, window);
                    if config.is_moe() {
                        println!(
                            "  expert$    : {:.1} MiB VRAM for routed experts{}",
                            expert_cache as f64 / (1024.0 * 1024.0),
                            if args.expert_cache_gb.is_some() {
                                ""
                            } else {
                                " (default: VRAM left after the layer window)"
                            },
                        );
                    }
                    return serve_streaming_gpu(
                        store,
                        &config,
                        &args,
                        window,
                        ram_cache,
                        expert_cache,
                        &listen,
                    );
                }
                let generator = dlm::loader::build_streaming_generator(
                    store,
                    &config,
                    args.context_length,
                    window,
                    args.prefetch_depth,
                    args.auto_prefetch,
                    ram_cache,
                )?;
                return start_batched_server(generator, None, &args, &config, &listen, None);
            }

            // Non-streaming paths materialize all layers resident.
            let parts = dlm::loader::load_model_parts(&store, &config, args.context_length)?;

            // Optional draft model for speculative decoding. It must share the
            // target's vocabulary (same tokenizer) for the accept/reject rule to
            // be exact. Loaded as parts so it can take the same kernel as the target.
            let draft_parts = match &args.draft_model_path {
                Some(dir) => {
                    let dcfg = ModelConfig::from_path(dir, quant)?;
                    if dcfg.vocab_size != config.vocab_size {
                        return Err(DlmError::InvalidConfig(format!(
                            "draft vocab {} != target vocab {}",
                            dcfg.vocab_size, config.vocab_size
                        )));
                    }
                    let dstore = MmapStore::open_dir(dir)?;
                    Some(dlm::loader::load_model_parts(
                        &dstore,
                        &dcfg,
                        args.context_length,
                    )?)
                }
                None => None,
            };

            if !args.multi_gpu_ids.is_empty() {
                // Split the target across local GPUs; the small,
                // pinned draft stays on the first GPU.
                let ids = args.multi_gpu_ids.clone();
                let split =
                    dlm::distributed::partition_layers(config.num_layers as usize, ids.len());
                println!();
                println!("multi-gpu    : pipeline-parallel layer split");
                for (stage, shard) in split.iter().enumerate() {
                    println!(
                        "  gpu {:<6}: layers {}..{} ({} layer(s))",
                        ids[stage],
                        shard.start,
                        shard.end,
                        shard.len(),
                    );
                }
                serve_multi_gpu(parts, draft_parts, &ids, &args, &config, &listen)
            } else if device == Device::Gpu {
                serve_on_gpu(parts, draft_parts, &args, &config, &listen)
            } else {
                let generator = parts.into_cpu_generator()?;
                let draft = draft_parts.map(|p| p.into_cpu_generator()).transpose()?;
                start_batched_server(generator, draft, &args, &config, &listen, None)
            }
        }
    }
}

/// Error for a GPU request on a build without any device compute kernels. Each
/// vendor gets the flag that actually works for it: on an AMD (`rocm`) build
/// `--features cuda-kernels` is unsatisfiable — it pulls in `cuda` — so pointing
/// AMD users there is a dead end; they need `rocm-kernels`.
///
/// Gated to match its call sites: a build that *has* device kernels never needs
/// to tell anyone to go build one, so without this the function is dead code on
/// every `cuda-kernels` and `rocm-kernels` build — which is exactly the warning
/// the feature-gated CI jobs now refuse.
#[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
fn gpu_compute_unavailable(what: &str) -> DlmError {
    let feature = match gpu::active_vendor() {
        gpu::GpuVendor::Amd => "rocm-kernels",
        _ => "cuda-kernels",
    };
    DlmError::InvalidConfig(format!(
        "{what} requires building with `--features {feature}`"
    ))
}

/// Resolve the effective compute device. `cpu` is honored as-is. `gpu` is
/// honored only if the binary was built with the device kernels AND a GPU
/// actually responds; otherwise dlm warns and falls back to CPU so a run never
/// dies just because no GPU is present. On a CPU-only build, an explicit
/// `--device gpu` passes through and the downstream `not(cuda-kernels)` arm
/// reports the clearer "requires --features cuda-kernels" error.
fn resolve_device(requested: Device) -> Device {
    if requested == Device::Cpu {
        return Device::Cpu;
    }
    #[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
    {
        if gpu::mem_get_info().is_ok() {
            Device::Gpu
        } else {
            eprintln!(
                "warning: --device gpu requested but no usable GPU found; \
                 running on CPU (pass --device cpu to silence)"
            );
            Device::Cpu
        }
    }
    #[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
    {
        Device::Gpu
    }
}

/// Resolve the streaming resident-layer window: the explicit override, else the
/// VRAM plan's `layers_to_load` for the model + context.
///
/// The plan comes from the **catalog** — the real per-layer tensor sizes in the
/// checkpoint — and not from `--quant`'s parameter-count estimate, which is what
/// `profile` has always done. dlm does not quantize weights; it loads them in
/// their native dtype. So the estimate (defaulting to Int4, 0.5 bytes/param)
/// claims a bf16 layer is a quarter of its real size, and the window it derives
/// overshoots free VRAM — on a 3B/4GB card it returned the full 28 layers. The
/// engine then believes the whole model is resident and never streams at all,
/// leaving the driver to page VRAM behind its back: silently and very slowly
/// under Windows WDDM, and an outright OOM where there is no such paging.
fn stream_plan(config: &ModelConfig, args: &ServeArgs, store: &MmapStore) -> VramPlan {
    let (free_bytes, _) = resolve_free_bytes(args.vram_budget_gb);
    let profiler = build_profiler(
        args.context_length,
        args.safety_margin_gb,
        args.max_batch as u32,
        args.kv_quant != KvQuantArg::F32,
    );
    let catalog = LayerCatalog::build(store);
    let native = dlm::loader::checkpoint_scheme(store).unwrap_or(config.quant);
    if catalog.is_empty() {
        profiler.plan_with_free(config, free_bytes)
    } else {
        profiler.plan_from_catalog(config, &catalog, native, free_bytes)
    }
}

/// The resident-layer window: the explicit `--resident-layers`, else the plan's.
fn resident_window(plan: &VramPlan, args: &ServeArgs) -> usize {
    args.resident_layers
        .unwrap_or(plan.layers_to_load as usize)
        .max(1)
}

/// VRAM budget (bytes) for the routed-expert cache on the GPU MoE streaming path.
///
/// `--expert-cache-gb` overrides it; otherwise it defaults to the VRAM left over
/// after the resident core window — so the expert cache and the layer window
/// together stay inside the plan's `usable` VRAM instead of the cache growing by
/// an unbounded expert *count* (which OOMs a 128-expert card). `0` for dense.
fn resolve_expert_cache_bytes(
    args: &ServeArgs,
    config: &ModelConfig,
    plan: &VramPlan,
    window: usize,
) -> usize {
    if !config.is_moe() {
        return 0;
    }
    if let Some(gb) = args.expert_cache_gb {
        return (gb.max(0.0) * GIB as f64) as usize;
    }
    // Whatever usable VRAM the resident core window doesn't take.
    let window_bytes = (window as u64).saturating_mul(plan.per_layer_weight_bytes);
    plan.usable_bytes.saturating_sub(window_bytes) as usize
}

/// Resolve the weight precision: the explicit `--quant`, else the checkpoint's
/// own dtype.
///
/// Only schemes the engine can actually deliver are accepted. `--quant` used to
/// be planning-only metadata that defaulted to Int4 while the loader read bf16,
/// so it silently described weights that did not exist and mis-sized every plan
/// derived from it. Now it either matches the file, or names a quantization the
/// loader really performs — anything else is an error rather than a no-op.
fn resolve_quant(requested: Option<QuantArg>, store: &MmapStore) -> Result<QuantScheme> {
    let native = dlm::loader::checkpoint_scheme(store)?;
    let Some(requested) = requested else {
        return Ok(native);
    };
    let scheme = requested.to_scheme();
    if scheme == native {
        return Ok(scheme);
    }
    // An already-packed 4-bit checkpoint is decoded as it is; there are no floats
    // left to quantize to something else, and re-quantizing 4-bit codes would only
    // throw away the calibration the export paid for.
    if native == QuantScheme::Int4
        && store
            .locate("model.layers.0.self_attn.q_proj.qweight")
            .is_some()
    {
        return Err(DlmError::UnsupportedQuant(format!(
            "--quant {scheme:?} cannot apply to an already-quantized 4-bit GPTQ checkpoint;              it is loaded at its own int4 precision. Omit --quant."
        )));
    }
    match scheme {
        // Quantized down from the checkpoint's floats at load time.
        QuantScheme::Int4 | QuantScheme::Int8 => Ok(scheme),
        other => Err(DlmError::UnsupportedQuant(format!(
            "--quant {other:?} does not match the checkpoint's {native:?} weights, and dlm \
             does not convert between float widths; omit --quant to use {native:?}, or pass \
             --quant int4 to quantize down"
        ))),
    }
}

/// Build a VRAM profiler, overriding the default safety cushion when the user
/// passed `--safety-margin-gb` (small cards claw back the fixed 1.5 GiB default).
fn build_profiler(
    context_length: u32,
    safety_margin_gb: Option<f64>,
    max_batch: u32,
    half_kv: bool,
) -> VramProfiler {
    let p = VramProfiler::new(context_length)
        .with_max_batch(max_batch)
        .with_half_kv(half_kv);
    let bytes = match safety_margin_gb {
        Some(gb) => (gb.max(0.0) * GIB as f64) as u64,
        // Scaled to the card when there is one; the flat default otherwise.
        None => gpu::mem_get_info().map_or(dlm::profiler::DEFAULT_SAFETY_MARGIN_BYTES, |m| {
            dlm::profiler::default_safety_margin_bytes(m.total)
        }),
    };
    p.with_safety_margin_bytes(bytes)
}

/// Refuse a serve config whose concurrent-batch KV reservation won't fit VRAM,
/// with a message naming the levers to pull — rather than letting the engine OOM
/// on the first token. `resident_bytes` is what sits in VRAM beside the KV caches
/// (the streamed window, or the resident weights); `kv_reservation` already
/// includes the `max_batch` multiplier. A no-op when everything fits.
#[allow(clippy::too_many_arguments)]
fn ensure_batch_kv_fits(
    free: u64,
    safety: u64,
    kv_reservation: u64,
    resident_bytes: u64,
    max_batch: usize,
    context: u32,
    what: &str,
) -> Result<()> {
    let need = safety
        .saturating_add(kv_reservation)
        .saturating_add(resident_bytes);
    if need > free {
        let gib = |b: u64| b as f64 / GIB as f64;
        return Err(DlmError::InvalidConfig(format!(
            "--max-batch {max_batch} needs ~{:.1} GiB VRAM ({what} {:.1} + KV {:.1} for \
             {max_batch}×{context} tokens + safety {:.1}) but only {:.1} GiB is free. \
             Lower --max-batch or --context-length (or use --quant / a smaller model).",
            gib(need),
            gib(resident_bytes),
            gib(kv_reservation),
            gib(safety),
            gib(free),
        )));
    }
    Ok(())
}

/// Tokens of KV to pool for resident GPU serving: everything that fits in `free`
/// after the safety margin and `weights`, at `per_token_bytes` (all layers), up
/// to `max_batch` full contexts. Refuses when not even one full-context request
/// fits, since such a request could never be admitted.
#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels", test))]
fn resident_kv_pool_tokens(
    free: u64,
    safety: u64,
    weights: u64,
    per_token_bytes: u64,
    max_batch: usize,
    context: u32,
) -> Result<usize> {
    let budget = free.saturating_sub(safety).saturating_sub(weights);
    let want = max_batch.max(1) as u64 * context as u64;
    let tokens = want.min(budget / per_token_bytes.max(1));
    if tokens < context as u64 {
        let gib = |b: u64| b as f64 / GIB as f64;
        return Err(DlmError::InvalidConfig(format!(
            "one {context}-token request needs ~{:.2} GiB of KV, but only {:.2} GiB of VRAM is \
             left after weights {:.1} + safety {:.1} (of {:.1} free). Lower --context-length, \
             or use --quant / a smaller model.",
            gib(context as u64 * per_token_bytes),
            gib(budget),
            gib(weights),
            gib(safety),
            gib(free),
        )));
    }
    Ok(tokens as usize)
}

/// Serve a model split across `gpu_ids`. On a `cuda-kernels` build each device
/// computes its own layer shard ([`MultiGpuKernel`]); otherwise it falls back to
/// the CPU-backed pipeline (device-affinity plumbing that no-ops off-GPU), so the
/// split path stays exercisable without hardware.
#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
fn serve_multi_gpu(
    parts: ModelParts,
    draft_parts: Option<ModelParts>,
    ids: &[u32],
    args: &ServeArgs,
    config: &ModelConfig,
    listen: &str,
) -> Result<()> {
    println!("  compute    : gpu (each stage runs its shard on its device)");
    let generator = parts.into_multi_gpu_generator(ids)?;
    let draft = draft_parts
        .map(|p| p.into_multi_gpu_generator(&ids[..1]))
        .transpose()?;
    start_batched_server(generator, draft, args, config, listen, None)
}

#[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
fn serve_multi_gpu(
    parts: ModelParts,
    draft_parts: Option<ModelParts>,
    ids: &[u32],
    args: &ServeArgs,
    config: &ModelConfig,
    listen: &str,
) -> Result<()> {
    println!(
        "  compute    : cpu (device split plumbing; build with --features cuda-kernels for GPU)"
    );
    let generator = parts.into_pipeline_parallel_generator(ids)?;
    let draft = draft_parts
        .map(|p| p.into_pipeline_parallel_generator(&ids[..1]))
        .transpose()?;
    start_batched_server(generator, draft, args, config, listen, None)
}

/// Serve with the GPU kernel (all layers resident in VRAM). Feature-gated on
/// `cuda-kernels`; the draft model, if any, is also placed on the GPU.
#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
fn serve_on_gpu(
    parts: ModelParts,
    draft_parts: Option<ModelParts>,
    args: &ServeArgs,
    config: &ModelConfig,
    listen: &str,
) -> Result<()> {
    println!();
    println!("device       : gpu ({})", gpu::active_vendor().label());
    // All weights sit resident in VRAM; each batched request adds a full-context
    // KV cache. Refuse up front if the batch won't fit rather than OOM mid-request.
    let (free, _) = resolve_free_bytes(args.vram_budget_gb);
    let profiler = build_profiler(
        args.context_length,
        args.safety_margin_gb,
        args.max_batch as u32,
        args.kv_quant != KvQuantArg::F32,
    );
    let weights = (config.estimated_total_params() as f64 * config.quant.bytes_per_param()) as u64;
    // KV is paged from a shared pool, so concurrent requests only need VRAM for
    // the tokens they actually hold. Give the pool whatever fits after the
    // weights, up to --max-batch full contexts.
    let per_token = profiler.kv_bytes_per_layer(config) * config.num_layers as u64
        / args.context_length.max(1) as u64;
    let pool = resident_kv_pool_tokens(
        free,
        profiler.safety_margin_bytes,
        weights,
        per_token,
        args.max_batch,
        args.context_length,
    )?;
    let generator = parts.into_gpu_generator()?;
    let draft = draft_parts.map(|p| p.into_gpu_generator()).transpose()?;
    start_batched_server(generator, draft, args, config, listen, Some(pool))
}

#[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
fn serve_on_gpu(
    _parts: ModelParts,
    _draft_parts: Option<ModelParts>,
    _args: &ServeArgs,
    _config: &ModelConfig,
    _listen: &str,
) -> Result<()> {
    Err(gpu_compute_unavailable("--device gpu"))
}

/// Serve with `--stream --device gpu`: stream a window of layer weights through
/// VRAM (`StreamingGpuKernel`). Experimental — unvalidated on hardware.
#[cfg(any(feature = "cuda-kernels", feature = "rocm-kernels"))]
#[allow(clippy::too_many_arguments)]
fn serve_streaming_gpu(
    store: MmapStore,
    config: &ModelConfig,
    args: &ServeArgs,
    window: usize,
    ram_cache: usize,
    expert_cache: usize,
    listen: &str,
) -> Result<()> {
    println!(
        "device       : gpu ({}) — VRAM layer streaming [experimental]",
        gpu::active_vendor().label()
    );
    let generator = dlm::loader::build_streaming_gpu_generator(
        store,
        config,
        args.context_length,
        window,
        ram_cache,
        expert_cache,
    )?;
    start_batched_server(generator, None, args, config, listen, None)
}

#[cfg(not(any(feature = "cuda-kernels", feature = "rocm-kernels")))]
#[allow(clippy::too_many_arguments)]
fn serve_streaming_gpu(
    _store: MmapStore,
    _config: &ModelConfig,
    _args: &ServeArgs,
    _window: usize,
    _ram_cache: usize,
    _expert_cache: usize,
    _listen: &str,
) -> Result<()> {
    Err(gpu_compute_unavailable("--stream --device gpu"))
}

/// Master mode: partition the model's layers across the configured worker nodes
/// and serve a (non-streaming, greedy) OpenAI-compatible endpoint backed by a
/// pipeline-parallel [`Coordinator`](dlm::distributed::Coordinator). A shard
/// whose worker is unreachable falls back to running locally.
/// Resolve the shared cluster secret: the `--cluster-secret` flag, else the
/// `DLM_CLUSTER_SECRET` env var (which keeps it out of the process argument
/// list). `None` means no auth — trusted-network only.
fn cluster_secret(args: &ServeArgs) -> Option<String> {
    args.cluster_secret
        .clone()
        .or_else(|| std::env::var("DLM_CLUSTER_SECRET").ok())
        .filter(|s| !s.is_empty())
}

fn serve_distributed(
    store: MmapStore,
    config: &ModelConfig,
    args: &ServeArgs,
    listen: &str,
) -> Result<()> {
    let parts = dlm::loader::load_model_parts(&store, config, args.context_length)?;
    let shards =
        dlm::distributed::partition_layers(config.num_layers as usize, args.worker_nodes.len());
    let routes: Vec<dlm::distributed::ShardRoute> = shards
        .iter()
        .zip(&args.worker_nodes)
        .map(|(shard, addr)| dlm::distributed::ShardRoute {
            shard: *shard,
            worker_addr: Some(addr.clone()),
        })
        .collect();

    let secret = cluster_secret(args);
    let coordinator = dlm::distributed::Coordinator::new(
        parts.cfg,
        parts.layers,
        parts.embedding,
        parts.final_norm,
        parts.lm_head,
        config.vocab_size as usize,
        routes,
    )?
    .with_auth(secret.clone());

    let template = resolve_chat_template(args)?;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let engine = std::sync::Arc::new(dlm::server::DistributedEngine::new(
        coordinator,
        serve_tokenizer(&args.model_path)?,
        template,
        config.vocab_size as usize,
        "dlm",
        128,
        args.context_length as usize,
        created,
    ));

    let server = dlm::server::HttpServer::bind(listen)?;
    println!();
    println!("serving      : distributed (pipeline) API on http://{listen}");
    println!(
        "  master     : {} shard(s) across {} worker(s)",
        shards.len(),
        args.worker_nodes.len()
    );
    println!("  endpoints  : POST /v1/chat/completions (non-streaming, greedy), GET /v1/models");
    println!(
        "  auth       : {}",
        if args.api_key.is_some() {
            "API key required"
        } else {
            "OPEN (set --api-key)"
        }
    );
    println!(
        "  cluster    : worker link {}",
        if secret.is_some() {
            "authenticated"
        } else {
            "UNAUTHENTICATED (set --cluster-secret)"
        }
    );
    println!("  note       : distributed mode trades streaming/sampling/batching for");
    println!("               spanning the model across nodes; a dead worker runs locally.");
    // secured_router honors --api-key here; the batched path already did, this
    // path silently ignored it before.
    server.serve(dlm::server::distributed::secured_router(
        engine,
        args.api_key.clone(),
    )) // blocks
}

/// Build the batched (optionally speculative) streaming engine over any compute
/// kernel `K` and serve the OpenAI-compatible API. Generic so the CPU and
/// multi-GPU pipeline kernels share one server path.
/// `kv_pool_tokens`: the KV tokens a GPU kernel's shared pools hold across all
/// concurrent requests; `None` sizes them for `--max-batch` full contexts, which
/// is what the streaming planner reserves. No effect on the CPU kernels.
fn start_batched_server<K: ComputeKernel + Send + 'static>(
    generator: Generator<K>,
    draft: Option<Generator<K>>,
    args: &ServeArgs,
    config: &ModelConfig,
    listen: &str,
    kv_pool_tokens: Option<usize>,
) -> Result<()> {
    let pool = kv_pool_tokens.unwrap_or(args.max_batch.max(1) * args.context_length as usize);
    let generator = generator.with_kv_pool_tokens(pool);
    let draft = draft.map(|d| d.with_kv_pool_tokens(pool));
    let tokenizer = serve_tokenizer(&args.model_path)?;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let template = resolve_chat_template(args)?;
    // EOS: an explicit --eos-token overrides; otherwise auto-detect from the
    // model's config.json (`eos_token_id`, which may list several ids).
    let mut eos_tokens = match args.eos_token {
        Some(id) => vec![id],
        None => config.eos_token_ids.clone(),
    };
    // The template's end-of-turn marker stops generation too, when it is a real
    // token in this vocabulary and --eos-token did not pin the stop set.
    if args.eos_token.is_none() {
        if let Some(id) = template.end_of_turn().and_then(|t| tokenizer.id_of(t)) {
            if !eos_tokens.contains(&id) {
                eos_tokens.push(id);
            }
        }
    }

    // Optional KV-cache quantization (int8/int4) — shrinks the KV memory.
    let kv_quant = args.kv_quant.to_kv_quant();
    let generator = generator.with_kv_quant(kv_quant);
    let draft = draft.map(|d| d.with_kv_quant(kv_quant));
    let kv_pool = generator.kv_free_tokens();
    // Cached prefixes share paged device KV and give it back under pressure, so
    // they are on by default where the kernel pages KV. On the CPU each one is a
    // host copy of its KV, so only when asked for.
    let prefix_cache = args
        .prefix_cache_size
        .unwrap_or(if kv_pool.is_some() { 64 } else { 0 });
    if let Some(opts) = &args.bench {
        return bench_generator(&generator, &tokenizer, opts, args, config);
    }

    // Batched, streaming engine: a background scheduler interleaves concurrent
    // requests a token at a time. With a draft model it decodes speculatively
    // (draft proposes, target verifies).
    let speculative = draft.is_some();
    let engine = match draft {
        Some(d) => dlm::server::EngineService::start_speculative(
            generator,
            d,
            args.draft_gamma,
            tokenizer,
            config.vocab_size as usize,
            "dlm",
            128,
            created,
            args.max_batch.max(1), // max concurrent batch (--max-batch)
            prefix_cache,
        ),
        None => dlm::server::EngineService::start(
            generator,
            tokenizer,
            config.vocab_size as usize,
            "dlm",
            128,
            created,
            args.max_batch.max(1), // max concurrent batch (--max-batch)
            prefix_cache,
        ),
    }
    .with_chat_template(template)
    .with_eos_tokens(eos_tokens.clone())
    .with_context_window(args.context_length as usize);
    let server = dlm::server::HttpServer::bind(listen)?;
    println!();
    let mode = if speculative {
        "batched + speculative"
    } else {
        "batched"
    };
    println!("serving      : OpenAI + Anthropic compatible API on http://{listen} ({mode})");
    println!("  openai     : POST /v1/chat/completions (stream), GET /v1/models");
    println!("  anthropic  : POST /v1/messages (stream), POST /v1/messages/count_tokens");
    println!("  ops        : GET /metrics (Prometheus), GET /health");
    if args.telemetry {
        println!("  telemetry  : GET /v1/telemetry (SSE, per-layer flow events)");
    }
    if eos_tokens.is_empty() {
        println!("  stop       : max_tokens only (no eos_token_id in config; pass --eos-token)");
    } else {
        println!(
            "  stop       : eos {eos_tokens:?} + max_tokens, context {} tokens",
            args.context_length
        );
    }
    if prefix_cache > 0 && !speculative {
        println!(
            "  prefix     : up to {prefix_cache} prompt prefixes cached{}",
            if kv_pool.is_some() {
                " (sharing KV blocks; dropped when requests need the room)"
            } else {
                " (each a host copy of its KV)"
            }
        );
    }
    let on_gpu = args.device == Device::Gpu || !args.multi_gpu_ids.is_empty();
    // The GPU kernels have one approximate KV format, fp16; the CPU kernels have
    // int8/int4 but no fp16.
    let kv_note = match kv_quant {
        dlm::forward::KvQuant::None => "f32 (exact)",
        _ if on_gpu => "fp16 on the GPU (half the memory of f32; --kv-quant f32 for exact)",
        dlm::forward::KvQuant::Int8 => "int8 (≈half memory, approximate)",
        dlm::forward::KvQuant::Int4 => "int4 (≈quarter memory, approximate)",
        dlm::forward::KvQuant::F16 => "f32 (the CPU kernels have no fp16 store)",
    };
    println!("  kv cache   : {kv_note}");
    if let Some(tokens) = kv_pool {
        println!(
            "  kv pool    : {tokens} tokens shared by all requests (≈{:.1} full {}-token contexts; shorter requests fit more)",
            tokens as f64 / args.context_length.max(1) as f64,
            args.context_length,
        );
    }
    if args.api_key.is_some() {
        println!("  auth       : bearer token required on /v1/*");
    }
    let router = dlm::server::engine::secured_router(engine, args.api_key.clone());
    let shutdown = install_signal_handler();
    server.serve_with_shutdown(router, shutdown) // blocks until signalled
}

/// English text the benchmark prompt is tokenized from, so MoE routing and the
/// expert cache see a realistic token mix rather than arbitrary ids.
const BENCH_TEXT: &str = "The river wound through the valley for many miles before it reached the sea. Along its banks, small towns had grown up over the centuries, each with its own market, its own bridge, and its own stories about floods and dry summers. Farmers brought grain and fruit to the markets every week, and traders carried salt, cloth and tools back upstream. When the railway arrived, the river mattered less for moving goods, but people still gathered by the water in the evenings to talk, to fish, and to watch the light change over the hills. ";

/// `dlm bench`: load the model through `serve`'s own path, then measure it
/// instead of serving it (see [`bench_generator`]).
fn run_bench(args: BenchArgs) -> Result<()> {
    let BenchArgs { mut serve, opts } = args;
    if serve.draft_model_path.is_some() || serve.distributed_mode != DistributedMode::Standalone {
        return Err(DlmError::InvalidConfig(
            "bench measures a single standalone model; drop --draft-model-path and --distributed-mode"
                .into(),
        ));
    }
    let needed = opts.prompt_len + opts.gen_len;
    if needed > serve.context_length as usize {
        return Err(DlmError::InvalidConfig(format!(
            "--prompt-len + --gen-len ({needed}) exceeds --context-length ({})",
            serve.context_length
        )));
    }
    // Size the batch-KV fit checks for the largest batch actually measured.
    serve.max_batch = opts.batch.iter().copied().max().unwrap_or(1).max(1);
    serve.bench = Some(opts);
    run_serve(serve)
}

fn bench_generator<K: ComputeKernel>(
    generator: &Generator<K>,
    tokenizer: &BpeTokenizer,
    opts: &BenchOpts,
    args: &ServeArgs,
    config: &ModelConfig,
) -> Result<()> {
    let seed = tokenizer.encode(BENCH_TEXT)?;
    let prompt = dlm::bench::prompt_ids(&seed, opts.prompt_len, config.vocab_size as usize);
    let row = |r: &dlm::bench::RunResult| {
        format!(
            "prefill {:>9.1} tok/s | ttft {:>9.1} ms | decode {:>8.2} tok/s ({:.1} ms/step)",
            r.prefill_tok_s, r.ttft_ms, r.decode_tok_s, r.ms_per_step
        )
    };
    println!();
    println!(
        "bench        : prompt {} tokens, {} generated, batch {:?}, {} run(s) each, greedy",
        opts.prompt_len, opts.gen_len, opts.batch, opts.runs
    );
    let results = dlm::bench::run(generator, &prompt, opts, |r, run| {
        println!("  b{:<3} run {:<2}: {}", r.batch, run + 1, row(r));
    })?;

    let summary = dlm::bench::summarize(&results);
    println!();
    println!("median       :");
    for r in &summary {
        println!("  batch {:<5}: {}", r.batch, row(r));
        for (stage, ms) in &r.stage_ms_per_token {
            println!("    {stage:<9}: {ms:>8.2} ms/token");
        }
    }
    let peak_rss = dlm::bench::peak_rss_bytes();
    if let Some(b) = peak_rss {
        println!("peak rss     : {:.2} GiB", b as f64 / GIB as f64);
    }
    // Device-wide, so another process on the GPU counts too. Taken at the end of
    // decode, while the KV caches are still allocated.
    let gpu_in_use = args.device == Device::Gpu || !args.multi_gpu_ids.is_empty();
    let vram_used = summary
        .iter()
        .filter_map(|r| r.vram_used_bytes)
        .max()
        .filter(|_| gpu_in_use);
    if let Some(b) = vram_used {
        println!(
            "vram in use  : {:.2} GiB at the end of decode (whole device)",
            b as f64 / GIB as f64
        );
    }

    if let Some(path) = &opts.json {
        let report = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "model": args.model_path,
            "device": format!("{:?}", args.device).to_lowercase(),
            "gpu_backend": gpu::active_vendor().label(),
            "quant": format!("{:?}", config.quant),
            "stream": args.stream,
            "kv_quant": format!("{:?}", args.kv_quant).to_lowercase(),
            "context_length": args.context_length,
            "prompt_len": opts.prompt_len,
            "gen_len": opts.gen_len,
            "runs": results,
            "summary": summary,
            "peak_rss_bytes": peak_rss,
            "vram_used_bytes": vram_used,
        });
        let text = serde_json::to_string_pretty(&report)
            .map_err(|e| DlmError::InvalidConfig(format!("bench json: {e}")))?;
        std::fs::write(path, text).map_err(|source| DlmError::Io {
            path: path.clone(),
            source,
        })?;
        println!("json         : {}", path.display());
    }
    Ok(())
}

/// Return a [`Shutdown`] wired to `SIGTERM`/`SIGINT` where the platform allows.
///
/// `SIGTERM` is how supervisors stop a process — `docker stop`, systemd, a
/// Kubernetes eviction — each followed by `SIGKILL` after a grace period. Without
/// a handler the server died mid-generation and cut SSE streams without their
/// terminating chunk, so every deploy dropped live requests.
///
/// **Unix only.** `libc` is already a Unix-only dependency and the standard
/// library exposes no portable signal API. On Windows the handle is returned
/// untriggered, so `serve_with_shutdown` behaves exactly like `serve` did — no
/// regression, just no graceful path. Wiring `SetConsoleCtrlHandler` would mean a
/// new dependency for the one platform least likely to run this under a
/// supervisor.
fn install_signal_handler() -> dlm::server::Shutdown {
    let shutdown = dlm::server::Shutdown::new();

    #[cfg(unix)]
    {
        use std::sync::OnceLock;
        // The handler runs in a signal context, where almost nothing is legal to
        // call. It may only touch an atomic — so it sets a flag, and a watcher
        // thread does the real work (dialing the listener to wake `accept`).
        static FLAG: OnceLock<std::sync::Arc<std::sync::atomic::AtomicBool>> = OnceLock::new();
        let flag = FLAG
            .get_or_init(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
            .clone();

        extern "C" fn on_signal(_sig: libc::c_int) {
            if let Some(f) = FLAG.get() {
                f.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }

        // Cast through `*const ()` rather than straight to the integer
        // `sighandler_t`: a direct function-item-to-integer cast is
        // `clippy::fn_to_numeric_cast_any`, and it is a lint worth heeding —
        // the item type is zero-sized, so which address you get is not
        // something the cast makes obvious.
        let handler_addr = on_signal as *const () as libc::sighandler_t;

        // SAFETY: `signal` with a valid signal number and an `extern "C"` handler
        // is defined behaviour. `on_signal` touches only an `AtomicBool`, which is
        // async-signal-safe; everything else is deferred to the watcher below.
        unsafe {
            libc::signal(libc::SIGTERM, handler_addr);
            libc::signal(libc::SIGINT, handler_addr);
        }

        let watcher = shutdown.clone();
        std::thread::spawn(move || loop {
            if flag.load(std::sync::atomic::Ordering::SeqCst) {
                eprintln!("dlm: shutdown signal received; draining in-flight requests");
                watcher.trigger();
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
    }

    shutdown
}

/// Shared planner/reporter: map the model (if a dir is given), profile it, size
/// the KV cache, and print the streaming schedule — streaming real weights when
/// a mapped model is available.
fn report_plan(
    config: &ModelConfig,
    model_dir: Option<&Path>,
    context_length: u32,
    vram_budget_gb: Option<f64>,
    safety_margin_gb: Option<f64>,
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
    // `profile` plans for a single sequence (batch is a serve-time concern).
    let profiler = build_profiler(context_length, safety_margin_gb, 1, false);
    let (free_bytes, free_source) = resolve_free_bytes(vram_budget_gb);
    println!("free VRAM    : {free_source}");

    let plan = match catalog {
        Some(cat) if !cat.is_empty() => {
            let native = mapped
                .as_ref()
                .and_then(|(store, _)| dlm::loader::checkpoint_scheme(store).ok())
                .unwrap_or(config.quant);
            profiler.plan_from_catalog(config, cat, native, free_bytes)
        }
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

    // Report the pinned staging buffer the pipeline *would* DMA from — computed,
    // not allocated. `profile` is a planning command ("no GPU required"); it has
    // no business page-locking gigabytes of host RAM to print a size, and on a
    // large model it cannot: the 70B sample at a 16 GiB budget wants ~11.8 GiB of
    // pinned memory, so `cudaHostAlloc` failed and the whole plan went with it.
    println!(
        "staging buf  : {:.1} MiB pinned (at serve time)",
        swap.staging_bytes(plan.per_layer_weight_bytes) as f64 / (1024.0 * 1024.0),
    );

    // Build the double-buffered A/B schedule.
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
        // This part *does* stream the real weights, so it really does need the
        // pinned window. If the host won't page-lock that much, say so and skip
        // it — the plan above is what `profile` is for, and it should not be lost
        // to a demo that could not get its buffer.
        let Ok(mut pipeline) = HostPipeline::new(max_window as usize) else {
            println!(
                "streamed     : skipped — could not page-lock {:.1} MiB for the staging window",
                max_window as f64 / (1024.0 * 1024.0),
            );
            return Ok(());
        };

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
/// Host-RAM budget (bytes) for the streaming layer cache. `None` disables it:
/// the cache duplicates layer weights in RAM on top of the page cache, so it is
/// opt-in rather than a surprise allocation on a memory-tight box.
/// Cap on the *defaulted* host-RAM layer cache. An explicit `--ram-cache-gb`
/// overrides it in either direction.
///
/// Floor for the cap when the platform will not report its RAM — sized so the
/// default can still hold a small quantized model outright while never silently
/// claiming a big fraction of a modest box.
const DEFAULT_RAM_CACHE_CAP: u64 = 4 * GIB;

/// Share of physical RAM the *defaulted* cache may claim. The cache duplicates
/// layer weights on top of the page cache, so it stays a minority of the box.
const RAM_CACHE_SHARE: u64 = 4; // one quarter

/// Ceiling on the defaulted host-RAM layer cache, scaled to the machine.
///
/// Asks the OS for physical RAM ([`total_ram`](dlm::memory::page::total_ram) —
/// the same direct platform call [`page_size`] already uses, no dependency) and
/// allows a quarter of it, never less than [`DEFAULT_RAM_CACHE_CAP`]. A fixed
/// ceiling either wasted a large machine or overcommitted a small one; this does
/// neither. An explicit `--ram-cache-gb` still overrides it in both directions.
fn ram_cache_cap() -> u64 {
    match dlm::memory::page::total_ram() {
        Some(total) => (total / RAM_CACHE_SHARE).max(DEFAULT_RAM_CACHE_CAP),
        None => DEFAULT_RAM_CACHE_CAP,
    }
}

/// Host-RAM budget for the streaming layer cache.
///
/// `--ram-cache-gb` wins when given (including `0` to disable). Otherwise a
/// streamed model gets a cache big enough to hold every layer, when that fits
/// under [`ram_cache_cap`], and none when it does not.
///
/// Without it, every window miss re-materializes the layer from the checkpoint:
/// re-reading it from the mmap, and re-quantizing it under `--quant` (measured
/// 12x slower on a 3B). Unquantized streaming pays the read alone, and that
/// alone was the bulk of the time: streamed Gemma 3 1B answering a 1,500-token
/// prompt took 268 s without the cache and 46 s with it. The cache does
/// duplicate weights on top of the OS page cache, which is why it is bounded.
///
/// All or nothing, because a partial cache buys nothing: layers stream in a
/// fixed cycle, so an LRU smaller than the layer set evicts each layer before it
/// comes round again and never records a hit -- it only costs RAM.
fn resolve_ram_cache_bytes(args: &ServeArgs, plan: &VramPlan) -> usize {
    if let Some(gb) = args.ram_cache_gb {
        return (gb.max(0.0) * GIB as f64) as usize;
    }
    if !args.stream {
        return 0;
    }
    default_ram_cache_bytes(
        plan.per_layer_weight_bytes,
        plan.num_layers,
        ram_cache_cap(),
    )
    .unwrap_or(0) as usize
}

/// The default cache for a streamed model: the whole layer set plus headroom,
/// or `None` when that exceeds `cap`.
///
/// Headroom matters more than precision here. `layer_bytes` is what a layer
/// costs in *VRAM* -- bare codes. The host cache holds the whole materialized
/// layer: under `--quant` the per-group scales and zero-points too (+12.5% for
/// int4 at group 128), plus the f32 norms and biases. A budget a few percent
/// short is not a slightly worse cache but no cache at all, for the reason
/// above; overshooting only reserves RAM the cache never fills.
fn default_ram_cache_bytes(layer_bytes: u64, num_layers: u32, cap: u64) -> Option<u64> {
    let whole_model = layer_bytes * num_layers as u64;
    let with_headroom = whole_model + whole_model / 4; // +25%
    (with_headroom <= cap).then_some(with_headroom)
}

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
    println!(
        "  M_kv_total       : {:>10.1} MiB",
        mib(plan.kv_total_bytes)
    );
    println!("  pinned_zone      : {:>10.1} MiB", mib(plan.pinned_bytes));
    println!(
        "  M_layer_weight   : {:>10.1} MiB",
        mib(plan.per_layer_weight_bytes)
    );
    println!("  usable           : {:>10.1} MiB", mib(plan.usable_bytes));
    println!(
        "  ▶ layers_to_load : {:>10} / {}",
        plan.layers_to_load, plan.num_layers
    );
    println!(
        "  ▶ resident       : {:>9.1}%",
        plan.resident_fraction() * 100.0
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The default cache holds the whole streamed layer set or nothing: a
    /// cache that cannot hold every layer never gets a hit on a cyclic scan.
    #[test]
    fn default_ram_cache_is_all_or_nothing() {
        let mib = 1024 * 1024;
        // 28 layers of 50 MiB with 25% headroom = 1750 MiB: fits under 4 GiB.
        assert_eq!(
            default_ram_cache_bytes(50 * mib, 28, 4 * GIB),
            Some(1750 * mib)
        );
        // 34 layers of 222 MiB (Gemma 3 4B in bf16) need ~9.4 GiB: no cache.
        assert_eq!(default_ram_cache_bytes(222 * mib, 34, 8 * GIB), None);
        // Exactly at the cap still fits.
        assert_eq!(
            default_ram_cache_bytes(800 * mib, 1, 1000 * mib),
            Some(1000 * mib)
        );
        assert_eq!(default_ram_cache_bytes(801 * mib, 1, 1000 * mib), None);
    }

    #[test]
    fn resident_kv_pool_takes_what_fits_up_to_the_batch() {
        let mib = 1024 * 1024;
        // 1 MiB per token, 4 GiB free, 1 GiB weights, 0.5 GiB safety: 2.5 GiB left.
        let pool = |batch, ctx| resident_kv_pool_tokens(4 * GIB, GIB / 2, GIB, mib, batch, ctx);
        // The whole batch fits: exactly batch x context.
        assert_eq!(pool(2, 1024).unwrap(), 2048);
        // It does not: everything that fits (2560 tokens), not a refusal.
        assert_eq!(pool(8, 1024).unwrap(), 2560);
        // Not even one full context fits: refuse, naming the lever.
        let err = pool(1, 4096).unwrap_err().to_string();
        assert!(err.contains("--context-length"), "{err}");
    }

    #[test]
    fn batch_kv_fit_check_accepts_and_refuses() {
        // Fits: free 10 GiB, need = safety 0.5 + KV 2 + resident 1 = 3.5.
        assert!(ensure_batch_kv_fits(10 * GIB, GIB / 2, 2 * GIB, GIB, 4, 2048, "weights").is_ok());

        // Doesn't fit: an 8 GiB batch KV reservation blows a 4 GiB card.
        let err = ensure_batch_kv_fits(4 * GIB, GIB, 8 * GIB, GIB, 16, 8192, "weights");
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("--max-batch"),
            "message should name the lever: {msg}"
        );
    }
}

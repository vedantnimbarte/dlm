//! Command-line interface definitions (`specs.md` §4).
//!
//! The `clap` (derive) types live here in the library so argument parsing is
//! unit-testable without spawning the binary. `main.rs` parses a [`Cli`] and
//! dispatches on [`Command`].

use crate::model::QuantScheme;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// `dlm` — dynamic layer-streaming inference engine.
#[derive(Debug, Parser)]
#[command(name = "dlm", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start the OpenAI-compatible inference server for a model.
    Serve(ServeArgs),
    /// Profile a model and print the VRAM plan + streaming schedule (no server).
    Profile(ProfileArgs),
    /// Run the end-to-end CPU generation loop on a synthetic model (demo). To run
    /// a real model — including VRAM-streaming on the GPU — use `serve` instead.
    Generate(GenerateArgs),
    /// Tokenize text with a byte-level BPE tokenizer (round-trip check).
    Tokenize(TokenizeArgs),
    /// Run environment diagnostics + a self-check (GPU availability, CPU
    /// inference, and — on a `cuda-kernels` build — a CPU-vs-GPU parity check).
    Doctor(DoctorArgs),
    /// Search the Hugging Face hub for models (safetensors, most-downloaded).
    Search(SearchArgs),
    /// Download a model from the Hugging Face hub into a local directory.
    Pull(PullArgs),
    /// Print a shell completion script to stdout (e.g. `dlm completions bash`).
    Completions(CompletionsArgs),
}

/// Arguments for `dlm completions`.
#[derive(Debug, Args)]
pub struct CompletionsArgs {
    /// Shell to generate completions for.
    #[arg(value_enum)]
    pub shell: clap_complete::Shell,
}

/// Arguments for `dlm search`.
#[derive(Debug, Args)]
pub struct SearchArgs {
    /// Search terms (e.g. `llama-3.2`). Omit to list the top models overall.
    #[arg(default_value = "")]
    pub query: String,

    /// Maximum results to show.
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
}

/// Arguments for `dlm pull`.
#[derive(Debug, Args)]
pub struct PullArgs {
    /// Repo id (`org/model`); a full HF URL is also accepted.
    pub repo: String,

    /// Destination directory. Defaults to `./models/<model>`.
    #[arg(long, value_name = "DIR")]
    pub local_dir: Option<PathBuf>,

    /// HF access token for gated/private models (or set `$HF_TOKEN`).
    #[arg(long, value_name = "TOK")]
    pub token: Option<String>,
}

/// Arguments for `dlm doctor`.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Optionally check that this model directory loads and tokenizes.
    #[arg(long, value_name = "DIR")]
    pub model_path: Option<PathBuf>,
}

/// Arguments for `dlm tokenize`.
#[derive(Debug, Args)]
pub struct TokenizeArgs {
    /// Text to encode.
    #[arg(long)]
    pub text: String,

    /// Directory with `vocab.json` + `merges.txt`. Defaults to a raw byte
    /// tokenizer (256 tokens, no merges) when omitted.
    #[arg(long, value_name = "DIR")]
    pub tokenizer: Option<std::path::PathBuf>,
}

/// Distributed operating mode (`--distributed-mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DistributedMode {
    /// Single node, no networking.
    Standalone,
    /// Coordinator node in a master-worker cluster.
    Master,
    /// Worker node driven by a master.
    Worker,
}

/// Compute device for generation (`--device`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Device {
    /// Host CPU (always available).
    Cpu,
    /// CUDA GPU (requires building with `--features cuda-kernels`).
    Gpu,
}

impl Device {
    /// Compile-time default device: GPU when the binary is built with the device
    /// compute kernels (`--features cuda-kernels`), else CPU. So a GPU build runs
    /// on the GPU by default and a CPU-only build never surprises the user by
    /// asking for hardware it can't drive.
    pub const DEFAULT: Device = if cfg!(feature = "cuda-kernels") {
        Device::Gpu
    } else {
        Device::Cpu
    };
}

/// On-disk weight precision (`--quant`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QuantArg {
    /// 4-bit AWQ/GPTQ (0.5 bytes/param) — the default.
    Int4,
    /// 8-bit (1 byte/param).
    Int8,
    /// 16-bit FP16/BF16 (2 bytes/param).
    Fp16,
}

impl QuantArg {
    /// Map to the engine's [`QuantScheme`].
    pub fn to_scheme(self) -> QuantScheme {
        match self {
            QuantArg::Int4 => QuantScheme::Int4,
            QuantArg::Int8 => QuantScheme::Int8,
            QuantArg::Fp16 => QuantScheme::Fp16,
        }
    }
}

/// KV cache precision (`--kv-quant`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum KvQuantArg {
    /// Exact f32 (default).
    None,
    /// int8 — about half the KV memory.
    Int8,
    /// int4 — about a quarter of the KV memory, more error.
    Int4,
}

impl KvQuantArg {
    /// Map to the engine's [`KvQuant`](crate::forward::KvQuant).
    pub fn to_kv_quant(self) -> crate::forward::KvQuant {
        match self {
            KvQuantArg::None => crate::forward::KvQuant::None,
            KvQuantArg::Int8 => crate::forward::KvQuant::Int8,
            KvQuantArg::Int4 => crate::forward::KvQuant::Int4,
        }
    }
}

/// Arguments for `dlm serve` (mirrors the `specs.md` §4 schema).
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Model directory containing `config.json` + `*.safetensors` shards.
    #[arg(long, value_name = "DIR")]
    pub model_path: PathBuf,

    /// Manual upper VRAM cap in gigabytes. Overrides the live device query.
    #[arg(long, value_name = "GB")]
    pub vram_budget_gb: Option<f64>,

    /// VRAM safety cushion (GiB) held back for activation spikes. Default 1.5.
    /// Lower it on small cards (e.g. `--safety-margin-gb 0.5` on a 4 GB GPU) to
    /// free room for more resident layers; raise it if you hit OOM.
    #[arg(long, value_name = "GB")]
    pub safety_margin_gb: Option<f64>,

    /// Target conversation context window (tokens).
    #[arg(long, default_value_t = 8192)]
    pub context_length: u32,

    /// On-disk weight quantization.
    #[arg(long, value_enum, default_value_t = QuantArg::Int4)]
    pub quant: QuantArg,

    /// System-RAM budget (GiB) for the tiered layer cache between NVMe and GPU.
    #[arg(long, value_name = "GB")]
    pub ram_cache_gb: Option<f64>,

    /// TCP port for the API server.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Bind address for the API server.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Optional tiny draft model for speculative decoding.
    #[arg(long, value_name = "DIR")]
    pub draft_model_path: Option<PathBuf>,

    /// Draft tokens proposed per speculative round (only used with
    /// `--draft-model-path`).
    #[arg(long, default_value_t = 4)]
    pub draft_gamma: usize,

    /// Comma-separated local GPU indices to split layers across.
    #[arg(long, value_delimiter = ',', value_name = "IDS")]
    pub multi_gpu_ids: Vec<u32>,

    /// Compute device for the serving engine. Defaults to `gpu` on a
    /// `cuda-kernels` build (else `cpu`); pass `--device cpu` to force CPU.
    /// Ignored when `--multi-gpu-ids` is set. If `gpu` is chosen but no device is
    /// usable, dlm warns and falls back to CPU.
    #[arg(long, value_enum, default_value_t = Device::DEFAULT)]
    pub device: Device,

    /// Stream transformer layers from disk instead of holding them all resident,
    /// keeping only a bounded window in memory — lets a model exceed the resident
    /// budget. Streams into VRAM on `--device gpu` (GPU compute) and into host RAM
    /// on `--device cpu` (CPU compute).
    #[arg(long, default_value_t = false)]
    pub stream: bool,

    /// Resident layer-window size for `--stream`. Defaults to the VRAM plan's
    /// `layers_to_load` for the model + context.
    #[arg(long, value_name = "N")]
    pub resident_layers: Option<usize>,

    /// How many layers ahead to prefetch while streaming (`--stream`). Higher
    /// hides more load latency behind compute but needs a bigger
    /// `--resident-layers` window (clamped to window − 1). `0` disables prefetch.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub prefetch_depth: usize,

    /// Auto-tune the streaming prefetch depth from measured load-vs-compute time
    /// instead of a fixed `--prefetch-depth`. Adapts to the disk/model at runtime.
    #[arg(long, default_value_t = false)]
    pub auto_prefetch: bool,

    /// KV cache precision: `none` (exact f32), `int8` (≈half memory), or `int4`
    /// (≈quarter memory, more error). The KV cache can exceed the weights at long
    /// context, so quantizing it lets more context fit in a given budget.
    #[arg(long, value_enum, default_value_t = KvQuantArg::None)]
    pub kv_quant: KvQuantArg,

    /// Cluster role.
    #[arg(long, value_enum, default_value_t = DistributedMode::Standalone)]
    pub distributed_mode: DistributedMode,

    /// Comma-separated worker `host:port` addresses (master mode).
    #[arg(long, value_delimiter = ',', value_name = "ADDRS")]
    pub worker_nodes: Vec<String>,

    /// Require this bearer token on `/v1/*` requests (`Authorization: Bearer …`).
    /// Omit to leave the API open (localhost default).
    #[arg(long, value_name = "KEY")]
    pub api_key: Option<String>,

    /// Chat prompt template applied to `/v1/chat/completions` messages:
    /// `plain` (default), `chatml`, or `llama3`. Match the served model.
    #[arg(long, value_name = "NAME", default_value = "plain")]
    pub chat_template: String,

    /// Stop generation when the model produces this token id (the model's EOS).
    /// Overrides the `eos_token_id` auto-detected from `config.json`; omit to use
    /// that. Sets `finish_reason: stop` / `stop_reason: end_turn` when hit.
    #[arg(long, value_name = "ID")]
    pub eos_token: Option<u32>,

    /// Cache up to this many prompt-prefix KV snapshots so requests sharing a
    /// prefix (e.g. a common system prompt) skip re-prefilling it. `0` disables
    /// it. Each entry holds the prefix's KV in RAM, so size it to your memory.
    /// No effect with a draft model (speculative sessions can't resume).
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub prefix_cache_size: usize,
}

/// Arguments for `dlm profile`.
#[derive(Debug, Args)]
pub struct ProfileArgs {
    /// Model directory to profile. If omitted, a built-in 70B-class sample
    /// config is used for demonstration.
    #[arg(long, value_name = "DIR")]
    pub model_path: Option<PathBuf>,

    /// Target conversation context window (tokens).
    #[arg(long, default_value_t = 8192)]
    pub context_length: u32,

    /// On-disk weight quantization.
    #[arg(long, value_enum, default_value_t = QuantArg::Int4)]
    pub quant: QuantArg,

    /// Manual upper VRAM cap in gigabytes. Overrides the live device query.
    #[arg(long, value_name = "GB")]
    pub vram_budget_gb: Option<f64>,

    /// VRAM safety cushion (GiB) held back for activation spikes. Default 1.5.
    /// Lower it on small cards (e.g. `--safety-margin-gb 0.5` on a 4 GB GPU) to
    /// free room for more resident layers; raise it if you hit OOM.
    #[arg(long, value_name = "GB")]
    pub safety_margin_gb: Option<f64>,

    /// System-RAM budget (GiB) for the tiered layer cache between NVMe and GPU.
    #[arg(long, value_name = "GB")]
    pub ram_cache_gb: Option<f64>,
}

/// Arguments for `dlm generate`.
///
/// Runs the real CPU generation loop over a **randomly-initialized** synthetic
/// model to exercise the full pipeline end-to-end. Weights are not trained, so
/// output token ids are deterministic-but-meaningless — this validates the
/// machinery, not model quality.
#[derive(Debug, Args)]
pub struct GenerateArgs {
    /// Prompt as comma-separated token ids (no tokenizer yet).
    #[arg(long, value_delimiter = ',', default_value = "1")]
    pub prompt: Vec<u32>,

    /// Load a real model from this directory (config.json + safetensors).
    /// When set, the synthetic-model geometry flags below are ignored.
    #[arg(long, value_name = "DIR")]
    pub model_path: Option<std::path::PathBuf>,

    /// Prompt as text, tokenized with the BPE tokenizer. Overrides `--prompt`.
    #[arg(long)]
    pub text: Option<String>,

    /// Tokenizer directory (`vocab.json` + `merges.txt`). Defaults to the model
    /// directory if it has them, else a raw byte tokenizer.
    #[arg(long, value_name = "DIR")]
    pub tokenizer: Option<std::path::PathBuf>,

    /// Number of new tokens to generate.
    #[arg(long, default_value_t = 16)]
    pub max_new_tokens: usize,

    /// Stop when this token id is produced.
    #[arg(long)]
    pub eos_token: Option<u32>,

    /// Vocabulary size.
    #[arg(long, default_value_t = 32)]
    pub vocab_size: usize,

    /// Residual-stream width (`d_model`).
    #[arg(long, default_value_t = 64)]
    pub hidden_size: usize,

    /// Number of transformer layers.
    #[arg(long, default_value_t = 2)]
    pub num_layers: u32,

    /// Query attention heads (must divide `hidden_size`).
    #[arg(long, default_value_t = 4)]
    pub num_heads: usize,

    /// Key/value heads (must divide `num_heads`).
    #[arg(long, default_value_t = 2)]
    pub num_kv_heads: usize,

    /// FFN inner width.
    #[arg(long, default_value_t = 128)]
    pub intermediate_size: usize,

    /// PRNG seed for the synthetic weights.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Compute device. Defaults to `gpu` on a `cuda-kernels` build (else `cpu`);
    /// pass `--device cpu` to force CPU. Falls back to CPU with a warning if no
    /// GPU is usable.
    #[arg(long, value_enum, default_value_t = Device::DEFAULT)]
    pub device: Device,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn serve_parses_full_spec_schema() {
        let cli = Cli::try_parse_from([
            "dlm",
            "serve",
            "--model-path",
            "/models/Llama-3-70B",
            "--vram-budget-gb",
            "13.5",
            "--context-length",
            "8192",
            "--port",
            "8000",
            "--host",
            "127.0.0.1",
            "--draft-model-path",
            "/models/Llama-3-3B",
            "--draft-gamma",
            "6",
            "--multi-gpu-ids",
            "0,1",
            "--distributed-mode",
            "master",
            "--worker-nodes",
            "192.168.1.50:9001,192.168.1.51:9001",
        ])
        .unwrap();

        let Command::Serve(a) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(a.model_path.to_str().unwrap(), "/models/Llama-3-70B");
        assert_eq!(a.vram_budget_gb, Some(13.5));
        assert_eq!(a.context_length, 8192);
        assert_eq!(a.port, 8000);
        assert_eq!(a.host, "127.0.0.1");
        assert_eq!(a.draft_model_path.unwrap().to_str().unwrap(), "/models/Llama-3-3B");
        assert_eq!(a.draft_gamma, 6);
        assert_eq!(a.multi_gpu_ids, vec![0, 1]);
        assert_eq!(a.distributed_mode, DistributedMode::Master);
        assert_eq!(
            a.worker_nodes,
            vec!["192.168.1.50:9001".to_string(), "192.168.1.51:9001".to_string()]
        );
    }

    #[test]
    fn serve_applies_defaults() {
        let cli = Cli::try_parse_from(["dlm", "serve", "--model-path", "/m"]).unwrap();
        let Command::Serve(a) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(a.context_length, 8192);
        assert_eq!(a.port, 8000);
        assert_eq!(a.host, "127.0.0.1");
        assert_eq!(a.quant, QuantArg::Int4);
        assert_eq!(a.distributed_mode, DistributedMode::Standalone);
        assert_eq!(a.draft_gamma, 4);
        assert!(a.multi_gpu_ids.is_empty());
        assert!(a.worker_nodes.is_empty());
    }

    #[test]
    fn serve_requires_model_path() {
        assert!(Cli::try_parse_from(["dlm", "serve"]).is_err());
    }

    #[test]
    fn profile_model_path_is_optional() {
        let cli = Cli::try_parse_from(["dlm", "profile"]).unwrap();
        let Command::Profile(a) = cli.command else {
            panic!("expected profile");
        };
        assert!(a.model_path.is_none());
        assert_eq!(a.context_length, 8192);
    }

    #[test]
    fn quant_arg_maps_to_scheme() {
        assert_eq!(QuantArg::Int4.to_scheme(), QuantScheme::Int4);
        assert_eq!(QuantArg::Int8.to_scheme(), QuantScheme::Int8);
        assert_eq!(QuantArg::Fp16.to_scheme(), QuantScheme::Fp16);
    }

    #[test]
    fn generate_parses_prompt_and_defaults() {
        let cli = Cli::try_parse_from(["dlm", "generate", "--prompt", "1,2,3", "--seed", "42"])
            .unwrap();
        let Command::Generate(a) = cli.command else {
            panic!("expected generate");
        };
        assert_eq!(a.prompt, vec![1, 2, 3]);
        assert_eq!(a.seed, 42);
        assert_eq!(a.max_new_tokens, 16);
        assert_eq!(a.vocab_size, 32);
        assert_eq!(a.num_heads, 4);
        assert!(a.eos_token.is_none());
    }

    #[test]
    fn tokenize_and_generate_text_parse() {
        let cli = Cli::try_parse_from(["dlm", "tokenize", "--text", "hi there"]).unwrap();
        let Command::Tokenize(a) = cli.command else {
            panic!("expected tokenize");
        };
        assert_eq!(a.text, "hi there");
        assert!(a.tokenizer.is_none());

        let cli = Cli::try_parse_from(["dlm", "generate", "--text", "hello"]).unwrap();
        let Command::Generate(a) = cli.command else {
            panic!("expected generate");
        };
        assert_eq!(a.text.as_deref(), Some("hello"));
        // Default flips to Gpu when built with device kernels; assert the same
        // compile-time constant the parser uses, not a hard-coded Cpu.
        assert_eq!(a.device, Device::DEFAULT); // default
    }

    #[test]
    fn generate_device_selection() {
        let cli = Cli::try_parse_from(["dlm", "generate", "--device", "gpu"]).unwrap();
        let Command::Generate(a) = cli.command else {
            panic!("expected generate");
        };
        assert_eq!(a.device, Device::Gpu);
    }

    #[test]
    fn rejects_unknown_distributed_mode() {
        assert!(Cli::try_parse_from([
            "dlm", "serve", "--model-path", "/m", "--distributed-mode", "bogus"
        ])
        .is_err());
    }
}

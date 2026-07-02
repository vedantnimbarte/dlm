//! Command-line interface definitions (`specs.md` §4).
//!
//! The `clap` (derive) types live here in the library so argument parsing is
//! unit-testable without spawning the binary. `main.rs` parses a [`Cli`] and
//! dispatches on [`Command`].

use crate::model::QuantScheme;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// `flip` — dynamic layer-streaming inference engine.
#[derive(Debug, Parser)]
#[command(name = "flip", version, about, long_about = None)]
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

/// Arguments for `flip serve` (mirrors the `specs.md` §4 schema).
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Model directory containing `config.json` + `*.safetensors` shards.
    #[arg(long, value_name = "DIR")]
    pub model_path: PathBuf,

    /// Manual upper VRAM cap in gigabytes. Overrides the live device query.
    #[arg(long, value_name = "GB")]
    pub vram_budget_gb: Option<f64>,

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

    /// Comma-separated local GPU indices to split layers across.
    #[arg(long, value_delimiter = ',', value_name = "IDS")]
    pub multi_gpu_ids: Vec<u32>,

    /// Cluster role.
    #[arg(long, value_enum, default_value_t = DistributedMode::Standalone)]
    pub distributed_mode: DistributedMode,

    /// Comma-separated worker `host:port` addresses (master mode).
    #[arg(long, value_delimiter = ',', value_name = "ADDRS")]
    pub worker_nodes: Vec<String>,
}

/// Arguments for `flip profile`.
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

    /// System-RAM budget (GiB) for the tiered layer cache between NVMe and GPU.
    #[arg(long, value_name = "GB")]
    pub ram_cache_gb: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn serve_parses_full_spec_schema() {
        let cli = Cli::try_parse_from([
            "flip",
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
        assert_eq!(a.multi_gpu_ids, vec![0, 1]);
        assert_eq!(a.distributed_mode, DistributedMode::Master);
        assert_eq!(
            a.worker_nodes,
            vec!["192.168.1.50:9001".to_string(), "192.168.1.51:9001".to_string()]
        );
    }

    #[test]
    fn serve_applies_defaults() {
        let cli = Cli::try_parse_from(["flip", "serve", "--model-path", "/m"]).unwrap();
        let Command::Serve(a) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(a.context_length, 8192);
        assert_eq!(a.port, 8000);
        assert_eq!(a.host, "127.0.0.1");
        assert_eq!(a.quant, QuantArg::Int4);
        assert_eq!(a.distributed_mode, DistributedMode::Standalone);
        assert!(a.multi_gpu_ids.is_empty());
        assert!(a.worker_nodes.is_empty());
    }

    #[test]
    fn serve_requires_model_path() {
        assert!(Cli::try_parse_from(["flip", "serve"]).is_err());
    }

    #[test]
    fn profile_model_path_is_optional() {
        let cli = Cli::try_parse_from(["flip", "profile"]).unwrap();
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
    fn rejects_unknown_distributed_mode() {
        assert!(Cli::try_parse_from([
            "flip", "serve", "--model-path", "/m", "--distributed-mode", "bogus"
        ])
        .is_err());
    }
}

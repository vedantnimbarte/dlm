//! Unified error type for the `dlm` engine.

use std::path::PathBuf;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, DlmError>;

/// All fallible operations in Phase 1 surface through this enum so the
/// orchestration layer can match on failure classes (I/O vs. parse vs. GPU)
/// without stringly-typed comparisons.
#[derive(Debug, thiserror::Error)]
pub enum DlmError {
    #[error("i/o error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to memory-map {path}: {source}")]
    Mmap {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("safetensors header is malformed: {0}")]
    SafetensorsHeader(String),

    #[error("tensor {name:?} out of bounds: range {start}..{end} exceeds data section of {len} bytes")]
    TensorOutOfBounds {
        name: String,
        start: usize,
        end: usize,
        len: usize,
    },

    #[error("unknown tensor {0:?}")]
    UnknownTensor(String),

    #[error("failed to parse JSON in {context}: {source}")]
    Json {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid model config: {0}")]
    InvalidConfig(String),

    #[error("invalid quantized tensor layout: {0}")]
    QuantLayout(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("distributed/network error: {0}")]
    Network(String),

    #[error("model hub error: {0}")]
    Hub(String),

    #[error("host memory allocation failed for {bytes} bytes (align {align})")]
    HostAlloc { bytes: usize, align: usize },

    #[error("KV cache pool exhausted: need {requested} more block(s), {free} free of {total}")]
    KvCacheExhausted {
        requested: u32,
        free: u32,
        total: u32,
    },

    #[error("activation pool exhausted: {in_use} buffer(s) in use, cap {max}")]
    ActivationPoolExhausted { in_use: usize, max: usize },

    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("GPU runtime error ({api}): code {code}")]
    Gpu { api: &'static str, code: i32 },

    #[error("no GPU backend compiled in — rebuild with `--features cuda` or `--features rocm` to use {0}")]
    GpuUnavailable(&'static str),
}

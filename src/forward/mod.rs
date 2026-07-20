//! Forward-pass orchestration skeleton.
//!
//! Ties streaming, dequantization, the compute kernel, the residual activation
//! pool, and the paged KV cache into a per-layer forward pass. The transformer
//! math is abstracted behind [`ComputeKernel`] so a real GPU kernel replaces the
//! [`StubKernel`] without touching the orchestration.

pub mod cpu;
#[cfg(feature = "cuda-kernels")]
pub mod gpu;
pub mod kernel;
pub mod multigpu;
pub mod orchestrator;
pub mod streaming;
#[cfg(feature = "cuda-kernels")]
pub mod streaming_gpu;

pub use cpu::{
    decode_block, rope_inv_freqs, Activation, BlockConfig, CpuKernel, ExpertFfn, Ffn, KvLayerCache,
    KvQuant, LayerTensors, QuantLayout, RopeScaling, Weights, QUANT_GROUP_SIZE,
};
#[cfg(feature = "cuda-kernels")]
pub use gpu::GpuKernel;
pub use kernel::{ComputeKernel, StubKernel};
pub use multigpu::PipelineParallelKernel;
pub use orchestrator::{ForwardOrchestrator, KvSnapshot};
pub use streaming::{CachedLayerSource, LayerSource, StreamStats, StreamingKernel};
#[cfg(feature = "cuda-kernels")]
pub use streaming_gpu::StreamingGpuKernel;

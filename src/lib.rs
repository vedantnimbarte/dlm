//! # flip — dynamic layer-streaming inference engine
//!
//! Phase 1 (Local Foundation) library surface. Modules are added
//! bottom-up as the engine is built; see `PRD.md` §5 for the phase map.

pub mod activation;
pub mod cache;
pub mod cli;
pub mod error;
pub mod forward;
pub mod gpu;
pub mod memory;
pub mod model;
pub mod pipeline;
pub mod profiler;
pub mod quant;
pub mod storage;
pub mod swap;

pub use activation::{ActivationBuffer, ActivationPool};
pub use cache::{KvCacheConfig, LayerRamCache, PagedKvCache};
pub use error::{FlipError, Result};
pub use forward::{
    BlockConfig, ComputeKernel, CpuKernel, ForwardOrchestrator, LayerTensors, StubKernel,
};
pub use memory::{PinKind, PinnedBuffer};
pub use model::{ModelConfig, QuantScheme};
pub use pipeline::{
    DoubleBufferSchedule, HostPipeline, LayerLoader, MmapWeightSource, PipelineStep,
    TieredWeightSource,
};
pub use profiler::{VramPlan, VramProfiler};
pub use quant::Quant4Tensor;
pub use storage::{LayerCatalog, MmapShard, MmapStore};
pub use swap::{LayerSwapPlan, StreamPass};

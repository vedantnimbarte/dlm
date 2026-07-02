//! # flip — dynamic layer-streaming inference engine
//!
//! Phase 1 (Local Foundation) library surface. Modules are added
//! bottom-up as the engine is built; see `PRD.md` §5 for the phase map.

pub mod cache;
pub mod cli;
pub mod error;
pub mod gpu;
pub mod memory;
pub mod model;
pub mod pipeline;
pub mod profiler;
pub mod storage;
pub mod swap;

pub use cache::{KvCacheConfig, LayerRamCache, PagedKvCache};
pub use error::{FlipError, Result};
pub use memory::{PinKind, PinnedBuffer};
pub use model::{ModelConfig, QuantScheme};
pub use pipeline::{
    DoubleBufferSchedule, HostPipeline, LayerLoader, MmapWeightSource, PipelineStep,
    TieredWeightSource,
};
pub use profiler::{VramPlan, VramProfiler};
pub use storage::{LayerCatalog, MmapShard, MmapStore};
pub use swap::{LayerSwapPlan, StreamPass};

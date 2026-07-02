//! # flip — dynamic layer-streaming inference engine
//!
//! Phase 1 (Local Foundation) library surface. Modules are added
//! bottom-up as the engine is built; see `PRD.md` §5 for the phase map.

pub mod cuda;
pub mod error;
pub mod memory;
pub mod model;
pub mod profiler;
pub mod storage;
pub mod swap;

pub use error::{FlipError, Result};
pub use memory::{PinKind, PinnedBuffer};
pub use model::{ModelConfig, QuantScheme};
pub use profiler::{VramPlan, VramProfiler};
pub use storage::{MmapShard, MmapStore};
pub use swap::{LayerSwapPlan, StreamPass};

//! # flip — dynamic layer-streaming inference engine
//!
//! Phase 1 (Local Foundation) library surface. Modules are added
//! bottom-up as the engine is built; see `PRD.md` §5 for the phase map.

pub mod error;
pub mod model;

pub use error::{FlipError, Result};
pub use model::{ModelConfig, QuantScheme};

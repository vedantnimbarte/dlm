//! Model description types (geometry + quantization) shared across the engine.

pub mod config;
pub mod naming;

pub use config::{
    MlaConfig, ModelConfig, MoeConfig, MoeNaming, PackedFormat, PackedQuant, QuantScheme, Routing,
};
pub use naming::{classify, TensorRole};

//! Storage engine: memory-mapped, zero-copy access to safetensors weights.

pub mod catalog;
pub mod mmap_store;
pub mod safetensors;

pub use catalog::LayerCatalog;
pub use mmap_store::{MmapShard, MmapStore};
pub(crate) use safetensors::f16_to_f32;
pub use safetensors::{bytes_to_f32, bytes_to_i32, Dtype, SafetensorsHeader, TensorInfo};

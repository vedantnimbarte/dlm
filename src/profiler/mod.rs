//! Dynamic VRAM profiling: how many transformer blocks fit resident at once.

pub mod vram;

pub use vram::{default_safety_margin_bytes, VramPlan, VramProfiler, DEFAULT_SAFETY_MARGIN_BYTES};

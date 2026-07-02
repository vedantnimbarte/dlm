//! The simple linear layer-swapping execution cycle (`PRD.md` Phase 1).
//!
//! Given a [`VramPlan`] (how many blocks fit) and a model's layer count, this
//! module lays out the deterministic sequence of *streaming passes* a single
//! forward pass decomposes into. Phase 1 is intentionally linear and
//! single-buffered: load a window of layers into a pinned staging buffer, run
//! them, free them, advance. Phase 2 replaces the single window with the
//! double-buffered A/B CUDA-stream pipeline — but the pass schedule computed
//! here is exactly what that pipeline consumes, so the shape carries forward.

use crate::error::Result;
use crate::memory::PinnedBuffer;
use crate::profiler::VramPlan;

/// One contiguous window of transformer blocks to stream in and execute
/// together, as `[first, last]` inclusive layer indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPass {
    /// Zero-based index of this pass within the forward pass.
    pub pass_index: u32,
    /// First layer (inclusive) in this window.
    pub first_layer: u32,
    /// Last layer (inclusive) in this window.
    pub last_layer: u32,
}

impl StreamPass {
    /// Number of layers in this window.
    pub fn layer_count(&self) -> u32 {
        self.last_layer - self.first_layer + 1
    }

    /// Iterate the layer indices this pass covers.
    pub fn layers(&self) -> std::ops::RangeInclusive<u32> {
        self.first_layer..=self.last_layer
    }
}

/// The full linear swap schedule for one forward pass over the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSwapPlan {
    /// Ordered windows covering `[0, num_layers)`.
    pub passes: Vec<StreamPass>,
    /// Layers resident per window (`VramPlan::layers_to_load`).
    pub window_size: u32,
    /// Total model layers.
    pub num_layers: u32,
}

impl LayerSwapPlan {
    /// Build the schedule from a profiling result: tile `[0, num_layers)` into
    /// consecutive windows of `plan.layers_to_load` layers (the final window may
    /// be smaller).
    pub fn from_plan(plan: &VramPlan) -> Self {
        let window = plan.layers_to_load.max(1);
        let num_layers = plan.num_layers;

        let mut passes = Vec::new();
        let mut first = 0u32;
        let mut pass_index = 0u32;
        while first < num_layers {
            let last = (first + window - 1).min(num_layers - 1);
            passes.push(StreamPass {
                pass_index,
                first_layer: first,
                last_layer: last,
            });
            first = last + 1;
            pass_index += 1;
        }

        Self {
            passes,
            window_size: window,
            num_layers,
        }
    }

    /// Number of streaming passes per forward pass.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }

    /// Allocate a single reusable pinned staging buffer big enough for the
    /// largest window, given the per-layer weight size. This buffer is the DMA
    /// source that Phase 2 copies into the streaming-zone VRAM buffers.
    pub fn allocate_staging_buffer(&self, per_layer_weight_bytes: u64) -> Result<PinnedBuffer> {
        let bytes = per_layer_weight_bytes as usize * self.window_size as usize;
        PinnedBuffer::with_len(bytes.max(1))
    }
}

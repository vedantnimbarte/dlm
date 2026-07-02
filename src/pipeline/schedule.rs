//! Double-buffer scheduling (`specs.md` §3.2).
//!
//! Turns a linear [`LayerSwapPlan`] into the overlapped A/B execution timeline:
//!
//! ```text
//! Stream A (Compute):        [ pass 0 ]          [ pass 1 ]          [ pass 2 ]
//! Stream B (Memory):  [load 0]▲ [load 1]         ▲ [load 2]         ▲
//!                            swap               swap               swap
//! ```
//!
//! Each [`PipelineStep`] pairs the compute of the current window (Stream A) with
//! the prefetch of the *next* window into the opposite buffer (Stream B), so the
//! DMA of window `i+1` hides under the compute of window `i`. The schedule is
//! pure data — no GPU needed to build or test it — and is exactly what the CUDA
//! executor will replay with real streams and events.

use crate::pipeline::buffer::BufferId;
use crate::swap::{LayerSwapPlan, StreamPass};

/// One tick of the pipeline: an optional compute and an optional concurrent
/// prefetch. The two run on different streams and are joined at the end of the
/// step by an event before the buffers swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStep {
    /// Window computed this step, read from `compute_buffer` (Stream A).
    /// `None` only on the prologue step, which just primes the first buffer.
    pub compute: Option<StreamPass>,
    /// Buffer the compute reads from.
    pub compute_buffer: BufferId,
    /// Window prefetched this step into `prefetch_buffer` (Stream B).
    /// `None` on the final step, when there is nothing left to load.
    pub prefetch: Option<StreamPass>,
    /// Buffer the prefetch writes into.
    pub prefetch_buffer: BufferId,
}

impl PipelineStep {
    /// True if a DMA load overlaps a compute this step (steady state).
    pub fn overlaps(&self) -> bool {
        self.compute.is_some() && self.prefetch.is_some()
    }
}

/// The full overlapped schedule for one forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoubleBufferSchedule {
    pub steps: Vec<PipelineStep>,
}

impl DoubleBufferSchedule {
    /// Build the overlapped schedule from a linear swap plan.
    ///
    /// For `N` windows the schedule has `N + 1` steps: one prologue prefetch,
    /// then `N` steps each computing window `i` while prefetching window `i+1`.
    pub fn from_swap_plan(plan: &LayerSwapPlan) -> Self {
        let passes = &plan.passes;
        let mut steps = Vec::with_capacity(passes.len() + 1);

        if passes.is_empty() {
            return Self { steps };
        }

        // Prologue: prime buffer for pass 0 with no concurrent compute.
        steps.push(PipelineStep {
            compute: None,
            compute_buffer: BufferId::for_pass(0),
            prefetch: Some(passes[0]),
            prefetch_buffer: BufferId::for_pass(0),
        });

        // Steady state: compute pass i, prefetch pass i+1 into the other buffer.
        for (i, pass) in passes.iter().enumerate() {
            let next = passes.get(i + 1).copied();
            let compute_buffer = BufferId::for_pass(i as u32);
            steps.push(PipelineStep {
                compute: Some(*pass),
                compute_buffer,
                prefetch: next,
                prefetch_buffer: compute_buffer.other(),
            });
        }

        Self { steps }
    }

    /// Number of pipeline steps (`num_windows + 1`).
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Count of steps where a DMA load hides under compute (the overlap the
    /// double buffer buys). Useful as a scheduling-efficiency signal.
    pub fn overlapping_steps(&self) -> usize {
        self.steps.iter().filter(|s| s.overlaps()).count()
    }
}

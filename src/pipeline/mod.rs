//! Asynchronous double-buffered streaming pipeline (`specs.md` §3.2).
//!
//! Phase 2 core. The [`schedule`] module builds the overlapped A/B timeline; the
//! [`HostPipeline`] here *executes* it on the CPU fallback so the buffer-swap
//! logic is fully testable without CUDA. The execution path mirrors the real
//! one exactly:
//!
//! ```text
//!   mmap weights ──► pinned staging buffer ──► device (streaming-zone) buffer ──► compute
//!      (disk)          (page-locked host)         (VRAM / host fallback)
//! ```
//!
//! Swapping the pinned→device `copy_in` for `cudaMemcpyAsync` on the memory
//! stream, and the compute step for a real kernel launch on the compute stream,
//! turns this into the GPU pipeline with no change to the schedule.

pub mod buffer;
pub mod schedule;

pub use buffer::{BufferId, DeviceBuffer};
pub use schedule::{DoubleBufferSchedule, PipelineStep};

use crate::error::Result;
use crate::memory::PinnedBuffer;
use crate::swap::StreamPass;

/// Supplies the concatenated weight bytes for a window of layers.
///
/// A real implementation pulls each layer's tensors from an [`MmapStore`] as a
/// zero-copy slice and stages them; a test can return synthetic bytes.
///
/// [`MmapStore`]: crate::storage::MmapStore
pub trait WeightSource {
    /// Bytes for every layer in `pass`, concatenated in layer order.
    fn load_window(&self, pass: &StreamPass) -> Result<Vec<u8>>;
}

/// Record of one executed compute step, used to verify the pipeline ran the
/// right windows over intact data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeTrace {
    pub pass_index: u32,
    pub first_layer: u32,
    pub last_layer: u32,
    pub byte_len: usize,
    /// Cheap fold of the bytes the compute step actually observed.
    pub checksum: u64,
}

/// Host-side executor of a [`DoubleBufferSchedule`].
///
/// Owns the two streaming-zone buffers and a single page-locked staging buffer,
/// exactly as the GPU engine will. Running the schedule copies each window
/// disk→pinned→device and "computes" it, returning a trace of what each compute
/// step saw.
pub struct HostPipeline {
    device_a: DeviceBuffer,
    device_b: DeviceBuffer,
    stage: PinnedBuffer,
    window_bytes: usize,
}

impl HostPipeline {
    /// Allocate the two device buffers and the pinned stage, each sized to hold
    /// the largest window (`window_bytes`).
    pub fn new(window_bytes: usize) -> Result<Self> {
        let window_bytes = window_bytes.max(1);
        Ok(Self {
            device_a: DeviceBuffer::with_capacity(window_bytes),
            device_b: DeviceBuffer::with_capacity(window_bytes),
            stage: PinnedBuffer::with_len(window_bytes)?,
            window_bytes,
        })
    }

    /// Capacity of one streaming-zone buffer.
    pub fn window_bytes(&self) -> usize {
        self.window_bytes
    }

    fn device_ref(&self, id: BufferId) -> &DeviceBuffer {
        match id {
            BufferId::A => &self.device_a,
            BufferId::B => &self.device_b,
        }
    }

    /// Stage a window's bytes through the pinned buffer into a device buffer —
    /// the host-fallback stand-in for `cudaMemcpyAsync` on the memory stream.
    fn prefetch<S: WeightSource>(
        &mut self,
        source: &S,
        pass: &StreamPass,
        dst: BufferId,
    ) -> Result<()> {
        let bytes = source.load_window(pass)?;
        let n = bytes.len();
        // disk → pinned staging (page-locked host memory).
        self.stage.as_mut_slice()[..n].copy_from_slice(&bytes);
        // pinned → device (async DMA on the real path). Borrow the staging and
        // the target device buffer as disjoint fields to satisfy the borrow
        // checker without an intermediate copy.
        let staged = &self.stage;
        let device = match dst {
            BufferId::A => &mut self.device_a,
            BufferId::B => &mut self.device_b,
        };
        device.copy_in(&staged.as_slice()[..n])?;
        Ok(())
    }

    /// Execute the whole schedule and return the per-compute-step trace.
    ///
    /// Correctness hinges on the A/B alternation: when we compute window `i` from
    /// its buffer, the concurrent prefetch of window `i+1` targets the *other*
    /// buffer, so the bytes under compute are never clobbered. The returned
    /// checksums make that observable.
    pub fn execute<S: WeightSource>(
        &mut self,
        schedule: &DoubleBufferSchedule,
        source: &S,
    ) -> Result<Vec<ComputeTrace>> {
        let mut trace = Vec::new();

        for step in &schedule.steps {
            // Memory stream: prefetch next window into the opposite buffer.
            if let Some(pass) = step.prefetch {
                self.prefetch(source, &pass, step.prefetch_buffer)?;
            }
            // Compute stream: run the current window from its buffer.
            if let Some(pass) = step.compute {
                let observed = self.device_ref(step.compute_buffer).as_slice();
                trace.push(ComputeTrace {
                    pass_index: pass.pass_index,
                    first_layer: pass.first_layer,
                    last_layer: pass.last_layer,
                    byte_len: observed.len(),
                    checksum: fold_checksum(observed),
                });
            }
        }

        Ok(trace)
    }
}

/// A cheap, order-sensitive byte fold used to detect buffer corruption in tests.
pub fn fold_checksum(bytes: &[u8]) -> u64 {
    let mut acc: u64 = 1469598103934665603; // FNV-ish offset basis
    for &b in bytes {
        acc = acc.wrapping_mul(1099511628211).wrapping_add(b as u64);
    }
    acc
}

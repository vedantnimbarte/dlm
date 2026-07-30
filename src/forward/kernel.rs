//! Compute-kernel abstraction for the forward pass.
//!
//! The transformer math is the one piece that must ultimately run on the GPU;
//! everything else in `dlm` is orchestration around it. The math sits behind
//! the block-level [`ComputeKernel`] trait so the orchestration is testable
//! off-GPU: the real CPU block ([`CpuKernel`](crate::forward::cpu::CpuKernel))
//! implements it for correctness, [`StubKernel`] implements it trivially for
//! pure orchestration tests, and a future CUDA/HIP kernel implements it for
//! production — all interchangeable behind [`ForwardOrchestrator`].
//!
//! [`ForwardOrchestrator`]: crate::forward::ForwardOrchestrator

use crate::error::Result;
use crate::forward::cpu::KvLayerCache;

/// A per-layer transformer-block kernel.
///
/// One call runs a whole decoder block for a single token: it updates `hidden`
/// in place (both residual connections included) and appends this token's K/V to
/// the layer's history in `kv`. The kernel reports the model's shape so the
/// orchestrator can size its per-layer KV storage.
pub trait ComputeKernel {
    /// Number of transformer blocks the kernel holds.
    fn num_layers(&self) -> u32;

    /// Residual-stream width the kernel expects for `hidden`.
    fn hidden_size(&self) -> usize;

    /// Key/value width per token (`num_kv_heads × head_dim`).
    fn kv_dim(&self) -> usize;

    /// Run block `layer` for one token at absolute `position`, updating `hidden`
    /// and appending to `kv`.
    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()>;

    /// Run block `layer` for a **batch** of sequences in one call: `hiddens[i]`,
    /// `kvs[i]` and `positions[i]` are sequence `i`'s state. The default loops
    /// [`run_block`](Self::run_block) per sequence — semantically identical, so any
    /// kernel is correct without overriding. The GPU kernel overrides it to fuse
    /// the per-sequence projection GEMVs into batched GEMMs, the throughput win
    /// continuous batching is built for (attention stays per-sequence — each has
    /// its own KV history/length).
    fn run_block_batched(
        &self,
        layer: u32,
        hiddens: &mut [&mut [f32]],
        kvs: &mut [&mut KvLayerCache],
        positions: &[usize],
    ) -> Result<()> {
        for ((hidden, kv), &position) in hiddens.iter_mut().zip(kvs.iter_mut()).zip(positions) {
            self.run_block(layer, hidden, kv, position)?;
        }
        Ok(())
    }

    /// Layer-streaming cache stats, if this kernel streams weights. Resident
    /// kernels (everything held in memory) return `None`; the streaming kernel
    /// overrides it so the server can surface hit rate / prefetch effectiveness.
    fn stream_stats(&self) -> Option<crate::forward::streaming::StreamStats> {
        None
    }
}

/// Borrowing a kernel is itself a kernel — lets a [`ForwardOrchestrator`] hold a
/// reference so an owner (e.g. a text generator) can drive many passes without
/// moving or cloning its weights.
///
/// [`ForwardOrchestrator`]: crate::forward::ForwardOrchestrator
impl<K: ComputeKernel> ComputeKernel for &K {
    fn num_layers(&self) -> u32 {
        (**self).num_layers()
    }

    fn hidden_size(&self) -> usize {
        (**self).hidden_size()
    }

    fn kv_dim(&self) -> usize {
        (**self).kv_dim()
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
        (**self).run_block(layer, hidden, kv, position)
    }

    fn run_block_batched(
        &self,
        layer: u32,
        hiddens: &mut [&mut [f32]],
        kvs: &mut [&mut KvLayerCache],
        positions: &[usize],
    ) -> Result<()> {
        // Forward to the concrete kernel so a borrowed GPU kernel keeps its fusion.
        (**self).run_block_batched(layer, hiddens, kvs, positions)
    }
}

/// A deterministic stand-in for a real kernel.
///
/// It does no transformer math: each layer adds a constant (`layer + 1`) to
/// every hidden element and appends a zero K/V entry, so the orchestration
/// (per-layer iteration, KV growth, position advance) is exactly verifiable
/// while staying numerically trivial. **Not** an inference implementation.
#[derive(Debug, Clone, Copy)]
pub struct StubKernel {
    num_layers: u32,
    hidden_size: usize,
    kv_dim: usize,
}

impl StubKernel {
    /// A stub for a model of the given shape.
    pub fn new(num_layers: u32, hidden_size: usize, kv_dim: usize) -> Self {
        Self {
            num_layers,
            hidden_size,
            kv_dim,
        }
    }
}

impl ComputeKernel for StubKernel {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        _position: usize,
    ) -> Result<()> {
        let zero = vec![0.0f32; self.kv_dim];
        kv.append(&zero, &zero)?;
        let delta = (layer + 1) as f32;
        for h in hidden.iter_mut() {
            *h += delta;
        }
        Ok(())
    }
}

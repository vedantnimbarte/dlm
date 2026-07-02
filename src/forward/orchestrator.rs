//! Forward-pass orchestration.
//!
//! [`ForwardOrchestrator`] runs one token step over the whole model, tying the
//! subsystems together in the order a streaming decoder needs:
//!
//! 1. **KV growth** — append the step's new tokens to the [`PagedKvCache`]
//!    (Cache Zone), which is where attention will read/write history.
//! 2. **Per-layer compute** — for each of the `num_layers` streamed blocks, run
//!    the attention and MLP sublayers through the [`ComputeKernel`], each with a
//!    residual add. The residual deltas use scratch buffers borrowed from an
//!    [`ActivationPool`] and returned immediately, so the whole pass reuses a
//!    single buffer instead of allocating per layer.
//!
//! Swapping [`StubKernel`](crate::forward::StubKernel) for a real GPU kernel
//! turns this skeleton into an actual forward pass with no orchestration change.

use crate::activation::{ActivationPool, ActivationPoolStats};
use crate::cache::PagedKvCache;
use crate::error::{FlipError, Result};
use crate::forward::kernel::{ComputeKernel, LayerWeights};

/// Static shape parameters for the forward pass.
#[derive(Debug, Clone, Copy)]
pub struct ForwardConfig {
    pub num_layers: u32,
    /// Residual-stream width (`d_model`).
    pub hidden_size: usize,
}

/// Orchestrates a streaming forward pass over a compute kernel.
pub struct ForwardOrchestrator<K: ComputeKernel> {
    kernel: K,
    cfg: ForwardConfig,
    pool: ActivationPool,
    kv: PagedKvCache,
}

impl<K: ComputeKernel> ForwardOrchestrator<K> {
    /// Build an orchestrator around a kernel and a KV cache. The activation
    /// pool is sized to the hidden width; only a couple of scratch buffers are
    /// ever live at once, so a small cap suffices.
    pub fn new(kernel: K, cfg: ForwardConfig, kv: PagedKvCache) -> Self {
        let pool = ActivationPool::new(cfg.hidden_size, 4);
        Self {
            kernel,
            cfg,
            pool,
            kv,
        }
    }

    /// Run one token step for sequence `seq_id` covering `n_tokens` new tokens
    /// (`n_tokens > 1` for prefill, `1` for decode), updating `hidden` in place.
    ///
    /// `weights` must hold exactly one [`LayerWeights`] per model layer, in
    /// order. `hidden` must be `hidden_size` long.
    pub fn step(
        &mut self,
        seq_id: u64,
        n_tokens: u64,
        hidden: &mut [f32],
        weights: &[LayerWeights],
    ) -> Result<()> {
        if weights.len() != self.cfg.num_layers as usize {
            return Err(FlipError::ShapeMismatch {
                expected: self.cfg.num_layers as usize,
                got: weights.len(),
            });
        }
        if hidden.len() != self.cfg.hidden_size {
            return Err(FlipError::ShapeMismatch {
                expected: self.cfg.hidden_size,
                got: hidden.len(),
            });
        }

        // 1. Grow the KV history for the new tokens (may fail if the pool is
        //    exhausted — surfaced rather than silently overflowing).
        self.kv.append_tokens(seq_id, n_tokens)?;

        // 2. Stream through the layers, each with two residual sublayers.
        for layer in 0..self.cfg.num_layers {
            let w = &weights[layer as usize];
            self.residual_sublayer(hidden, |k, h, d| k.attention(w, h, d))?;
            self.residual_sublayer(hidden, |k, h, d| k.mlp(w, h, d))?;
        }
        Ok(())
    }

    /// Run one sublayer with a residual add: acquire a scratch buffer, compute
    /// the delta into it, add it back into `hidden`, and return the buffer.
    fn residual_sublayer<F>(&mut self, hidden: &mut [f32], compute: F) -> Result<()>
    where
        F: FnOnce(&K, &[f32], &mut [f32]) -> Result<()>,
    {
        let mut delta = self.pool.acquire()?;
        compute(&self.kernel, hidden, delta.as_mut_slice())?;
        for (h, d) in hidden.iter_mut().zip(delta.as_slice()) {
            *h += *d;
        }
        self.pool.release(delta);
        Ok(())
    }

    /// The KV cache (for inspecting sequence lengths / utilization).
    pub fn kv(&self) -> &PagedKvCache {
        &self.kv
    }

    /// A snapshot of activation-pool reuse.
    pub fn pool_stats(&self) -> ActivationPoolStats {
        self.pool.stats()
    }

    /// Shape configuration.
    pub fn config(&self) -> ForwardConfig {
        self.cfg
    }
}

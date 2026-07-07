//! Forward-pass orchestration.
//!
//! [`ForwardOrchestrator`] drives a single sequence through the model one token
//! at a time, tying the subsystems together:
//!
//! 1. **Budget accounting** — append the new token to the [`PagedKvCache`]
//!    (Cache Zone), which enforces the KV memory budget and returns
//!    [`DlmError::KvCacheExhausted`](crate::error::DlmError::KvCacheExhausted)
//!    rather than overrunning it.
//! 2. **Per-layer compute** — for each block, call the [`ComputeKernel`], which
//!    runs the whole decoder block (attention + MLP, both residuals) over that
//!    layer's real K/V history and updates the hidden state in place.
//!
//! The kernel is interchangeable: [`StubKernel`](crate::forward::StubKernel) for
//! orchestration tests, [`CpuKernel`](crate::forward::cpu::CpuKernel) for a real
//! CPU forward pass, a GPU kernel for production.

use crate::cache::PagedKvCache;
use crate::error::{DlmError, Result};
use crate::forward::cpu::KvLayerCache;
use crate::forward::kernel::ComputeKernel;

/// The single sequence this orchestrator drives (budget accounting id).
const SEQ_ID: u64 = 0;

/// Drives one sequence's autoregressive forward pass over a compute kernel.
pub struct ForwardOrchestrator<K: ComputeKernel> {
    kernel: K,
    /// Real per-layer K/V history the kernel reads and appends to.
    kv_layers: Vec<KvLayerCache>,
    /// Cache Zone budget accounting (fixed-size block pool).
    budget: PagedKvCache,
    /// Absolute position of the next token.
    position: usize,
}

impl<K: ComputeKernel> ForwardOrchestrator<K> {
    /// Build an orchestrator around a kernel and a KV budget pool. One real
    /// [`KvLayerCache`] is allocated per layer, sized to the kernel's KV width.
    pub fn new(kernel: K, budget: PagedKvCache) -> Self {
        let kv_dim = kernel.kv_dim();
        let kv_layers = (0..kernel.num_layers())
            .map(|_| KvLayerCache::new(kv_dim))
            .collect();
        Self {
            kernel,
            kv_layers,
            budget,
            position: 0,
        }
    }

    /// Advance the sequence by one token, updating `hidden` in place through
    /// every layer. `hidden` must be `hidden_size` long.
    pub fn decode_token(&mut self, hidden: &mut [f32]) -> Result<()> {
        if hidden.len() != self.kernel.hidden_size() {
            return Err(DlmError::ShapeMismatch {
                expected: self.kernel.hidden_size(),
                got: hidden.len(),
            });
        }

        // Reserve KV budget for this token first (may reject on exhaustion).
        self.budget.append_tokens(SEQ_ID, 1)?;

        for layer in 0..self.kernel.num_layers() {
            self.kernel.run_block(
                layer,
                hidden,
                &mut self.kv_layers[layer as usize],
                self.position,
            )?;
        }
        self.position += 1;
        Ok(())
    }

    /// Absolute position of the next token (tokens decoded so far).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of layers.
    pub fn num_layers(&self) -> u32 {
        self.kernel.num_layers()
    }

    /// Cached K/V positions for a layer.
    pub fn layer_kv_len(&self, layer: u32) -> usize {
        self.kv_layers[layer as usize].len()
    }

    /// The Cache Zone budget pool (for utilization inspection).
    pub fn kv_budget(&self) -> &PagedKvCache {
        &self.budget
    }
}

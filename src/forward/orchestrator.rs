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

/// A cloneable snapshot of an orchestrator's KV history at a token boundary, so
/// another sequence that shares this prefix can resume from it instead of
/// re-running the prefix through the model. The KV tensors live here (not in the
/// kernel), so a clone fully captures the sequence state.
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    kv_layers: Vec<KvLayerCache>,
    position: usize,
}

impl KvSnapshot {
    /// Number of tokens already decoded into this snapshot.
    pub fn position(&self) -> usize {
        self.position
    }
}

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
    /// With `quantize_kv`, the per-layer KV history is stored int8 (half the
    /// memory, approximate).
    pub fn new(kernel: K, budget: PagedKvCache, quantize_kv: bool) -> Self {
        let kv_dim = kernel.kv_dim();
        let kv_layers = (0..kernel.num_layers())
            .map(|_| {
                if quantize_kv {
                    KvLayerCache::new_quantized(kv_dim)
                } else {
                    KvLayerCache::new(kv_dim)
                }
            })
            .collect();
        Self {
            kernel,
            kv_layers,
            budget,
            position: 0,
        }
    }

    /// Snapshot the current KV history so another sequence sharing this prefix
    /// can [`resume`](Self::resume) from it.
    pub fn snapshot(&self) -> KvSnapshot {
        KvSnapshot {
            kv_layers: self.kv_layers.clone(),
            position: self.position,
        }
    }

    /// Build an orchestrator resuming from `snapshot`'s KV history and position,
    /// reserving matching budget from `budget`. The caller prefills any suffix
    /// tokens that follow the snapshot via [`decode_token`](Self::decode_token).
    /// Errors if `snapshot` was taken on a differently-shaped model.
    pub fn resume(kernel: K, mut budget: PagedKvCache, snapshot: KvSnapshot) -> Result<Self> {
        if snapshot.kv_layers.len() != kernel.num_layers() as usize {
            return Err(DlmError::InvalidConfig(format!(
                "snapshot has {} layers but the model has {}",
                snapshot.kv_layers.len(),
                kernel.num_layers()
            )));
        }
        // Reserve KV budget matching the tokens already in the snapshot so the
        // pool accounting stays consistent with the restored history.
        if snapshot.position > 0 {
            budget.append_tokens(SEQ_ID, snapshot.position as u64)?;
        }
        Ok(Self {
            kernel,
            kv_layers: snapshot.kv_layers,
            budget,
            position: snapshot.position,
        })
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

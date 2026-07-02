//! Layer catalog — the bridge from mapped bytes to the VRAM budget.
//!
//! Walks every tensor in an [`MmapStore`], classifies it by role
//! ([`crate::model::naming`]), and tallies real on-disk byte sizes into:
//!   * per-transformer-block totals (the Streaming Zone working set), and
//!   * a single pinned overhead figure (Embedding + LM Head + norms that live
//!     permanently in the Pinned Zone, §2.1).
//!
//! Feeding these measured sizes to the profiler replaces the parameter-count
//! *estimate* used at bootstrap with the checkpoint's actual geometry.

use crate::model::naming::{classify, TensorRole};
use crate::storage::MmapStore;
use std::collections::BTreeMap;

/// Measured weight sizes for one model, derived from mapped shards.
#[derive(Debug, Clone, Default)]
pub struct LayerCatalog {
    /// Byte size of each transformer block, keyed by layer index.
    layer_bytes: BTreeMap<u32, u64>,
    /// Ordered tensor names making up each transformer block. This is what a
    /// [`WeightSource`](crate::pipeline::WeightSource) walks to stream a layer's
    /// weights out of the map.
    layer_tensors: BTreeMap<u32, Vec<String>>,
    /// Names of the pinned tensors (embedding, LM head, norms, misc).
    pinned_tensors: Vec<String>,
    /// Total bytes of all pinned tensors (embedding, LM head, norms, misc).
    pinned_bytes: u64,
}

impl LayerCatalog {
    /// Build a catalog by scanning every tensor across all shards of a store.
    pub fn build(store: &MmapStore) -> Self {
        let mut layer_bytes: BTreeMap<u32, u64> = BTreeMap::new();
        let mut layer_tensors: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        let mut pinned_tensors: Vec<String> = Vec::new();
        let mut pinned_bytes: u64 = 0;

        for info in store.iter_tensors() {
            let bytes = info.byte_len() as u64;
            match classify(&info.name) {
                TensorRole::Layer(idx) => {
                    *layer_bytes.entry(idx).or_insert(0) += bytes;
                    layer_tensors
                        .entry(idx)
                        .or_default()
                        .push(info.name.clone());
                }
                // Every non-layer tensor is resident overhead.
                _ => {
                    pinned_bytes += bytes;
                    pinned_tensors.push(info.name.clone());
                }
            }
        }

        // Sort each layer's tensor list so streaming order is deterministic
        // regardless of shard iteration order.
        for names in layer_tensors.values_mut() {
            names.sort();
        }
        pinned_tensors.sort();

        Self {
            layer_bytes,
            layer_tensors,
            pinned_tensors,
            pinned_bytes,
        }
    }

    /// Number of distinct transformer blocks found.
    pub fn num_layers(&self) -> u32 {
        self.layer_bytes.len() as u32
    }

    /// Bytes permanently resident in the Pinned Zone.
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned_bytes
    }

    /// Bytes of a specific transformer block, if present.
    pub fn layer_bytes(&self, index: u32) -> Option<u64> {
        self.layer_bytes.get(&index).copied()
    }

    /// Ordered tensor names composing a transformer block, if present.
    pub fn layer_tensor_names(&self, index: u32) -> Option<&[String]> {
        self.layer_tensors.get(&index).map(Vec::as_slice)
    }

    /// Names of every pinned tensor (embedding, LM head, norms, misc).
    pub fn pinned_tensor_names(&self) -> &[String] {
        &self.pinned_tensors
    }

    /// Largest single-block size — the figure the streaming buffers must be
    /// sized for, since any resident window could contain the biggest block.
    pub fn max_layer_bytes(&self) -> u64 {
        self.layer_bytes.values().copied().max().unwrap_or(0)
    }

    /// Mean block size across all layers (0 if empty).
    pub fn mean_layer_bytes(&self) -> u64 {
        if self.layer_bytes.is_empty() {
            0
        } else {
            let total: u64 = self.layer_bytes.values().sum();
            total / self.layer_bytes.len() as u64
        }
    }

    /// Total streamed weight bytes across every block.
    pub fn total_layer_bytes(&self) -> u64 {
        self.layer_bytes.values().sum()
    }

    /// True if no transformer blocks were found (e.g. a non-model directory).
    pub fn is_empty(&self) -> bool {
        self.layer_bytes.is_empty()
    }
}

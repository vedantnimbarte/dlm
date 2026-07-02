//! Weight sources feeding the streaming pipeline.
//!
//! A [`WeightSource`] supplies the concatenated bytes of one window of layers.
//! [`MmapWeightSource`] is the production implementation: it walks the per-layer
//! tensor names recorded in a [`LayerCatalog`] and pulls each tensor's bytes
//! zero-copy from the [`MmapStore`], concatenating them in the catalog's
//! deterministic order. This is the disk→host half of the streaming path; the
//! [`HostPipeline`](crate::pipeline::HostPipeline) then stages the result
//! through page-locked memory into a streaming-zone buffer.

use crate::error::Result;
use crate::pipeline::WeightSource;
use crate::storage::{LayerCatalog, MmapStore};
use crate::swap::StreamPass;

/// Streams layer weights out of a memory-mapped checkpoint.
///
/// Borrows the store and catalog; no weights are copied until a window is
/// requested, and even then only into the returned buffer (the mmap pages
/// themselves are faulted in lazily by the OS).
pub struct MmapWeightSource<'a> {
    store: &'a MmapStore,
    catalog: &'a LayerCatalog,
}

impl<'a> MmapWeightSource<'a> {
    /// Bind a source to a mapped store and its catalog.
    pub fn new(store: &'a MmapStore, catalog: &'a LayerCatalog) -> Self {
        Self { store, catalog }
    }

    /// Total bytes a given window will produce, from cached catalog sizes —
    /// useful for sizing staging buffers without touching the map.
    pub fn window_bytes(&self, pass: &StreamPass) -> u64 {
        pass.layers()
            .filter_map(|layer| self.catalog.layer_bytes(layer))
            .sum()
    }
}

impl WeightSource for MmapWeightSource<'_> {
    fn load_window(&self, pass: &StreamPass) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.window_bytes(pass) as usize);
        for layer in pass.layers() {
            if let Some(names) = self.catalog.layer_tensor_names(layer) {
                for name in names {
                    // Zero-copy slice from the map, copied once into the window.
                    out.extend_from_slice(self.store.tensor_bytes(name)?);
                }
            }
        }
        Ok(out)
    }
}

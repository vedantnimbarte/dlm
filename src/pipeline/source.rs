//! Weight sources feeding the streaming pipeline.
//!
//! A [`WeightSource`] supplies the concatenated bytes of one window of layers.
//! [`MmapWeightSource`] is the production implementation: it walks the per-layer
//! tensor names recorded in a [`LayerCatalog`] and pulls each tensor's bytes
//! zero-copy from the [`MmapStore`], concatenating them in the catalog's
//! deterministic order. This is the disk→host half of the streaming path; the
//! [`HostPipeline`](crate::pipeline::HostPipeline) then stages the result
//! through page-locked memory into a streaming-zone buffer.

use crate::cache::LayerRamCache;
use crate::error::Result;
use crate::pipeline::WeightSource;
use crate::storage::{LayerCatalog, MmapStore};
use crate::swap::StreamPass;
use std::cell::RefCell;

/// Loads the concatenated weight bytes of a single transformer layer.
///
/// This is the granularity the tiered CPU-RAM cache operates at: one layer is
/// the unit that is cached, evicted, and reused across token steps.
pub trait LayerLoader {
    /// Bytes for every tensor in `layer`, concatenated in catalog order.
    fn load_layer(&self, layer: u32) -> Result<Vec<u8>>;
}

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

impl LayerLoader for MmapWeightSource<'_> {
    fn load_layer(&self, layer: u32) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.catalog.layer_bytes(layer).unwrap_or(0) as usize);
        if let Some(names) = self.catalog.layer_tensor_names(layer) {
            for name in names {
                // Zero-copy slice from the map, copied once into the buffer.
                out.extend_from_slice(self.store.tensor_bytes(name)?);
            }
        }
        Ok(out)
    }
}

impl WeightSource for MmapWeightSource<'_> {
    fn load_window(&self, pass: &StreamPass) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.window_bytes(pass) as usize);
        for layer in pass.layers() {
            out.extend_from_slice(&self.load_layer(layer)?);
        }
        Ok(out)
    }
}

/// A [`WeightSource`] that fronts a per-layer [`LayerLoader`] with a bounded
/// CPU-RAM LRU cache ([`LayerRamCache`]).
///
/// On each window it assembles the layers from RAM when cached and falls back to
/// the underlying loader (disk) on a miss, so hot layers survive across forward
/// passes. The cache lives behind a `RefCell` because [`WeightSource::load_window`]
/// takes `&self` (the [`HostPipeline`](crate::pipeline::HostPipeline) borrows the
/// source immutably) while cache bookkeeping needs interior mutability.
pub struct TieredWeightSource<L: LayerLoader> {
    loader: L,
    cache: RefCell<LayerRamCache>,
}

impl<L: LayerLoader> TieredWeightSource<L> {
    /// Wrap `loader` with a RAM cache bounded to `cache_bytes`.
    pub fn new(loader: L, cache_bytes: u64) -> Self {
        Self {
            loader,
            cache: RefCell::new(LayerRamCache::new(cache_bytes)),
        }
    }

    /// A snapshot of cache effectiveness (hits/misses/evictions).
    pub fn cache_stats(&self) -> crate::cache::RamCacheStats {
        self.cache.borrow().stats()
    }

    /// The wrapped loader.
    pub fn loader(&self) -> &L {
        &self.loader
    }
}

impl<L: LayerLoader> WeightSource for TieredWeightSource<L> {
    fn load_window(&self, pass: &StreamPass) -> Result<Vec<u8>> {
        let mut cache = self.cache.borrow_mut();
        let mut out = Vec::new();
        for layer in pass.layers() {
            let bytes = cache.get_or_load(layer, || self.loader.load_layer(layer))?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }
}

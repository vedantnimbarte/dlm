//! Tiered CPU-RAM layer cache (`PRD.md` §3.2).
//!
//! System RAM is far larger than VRAM but far smaller than the model on NVMe, so
//! `dlm` uses it as a middle tier: a bounded, byte-budgeted LRU cache of layer
//! weights sitting between the memory-mapped checkpoint and the pinned staging
//! buffer.
//!
//! ```text
//!   NVMe (mmap)  ──►  CPU-RAM cache (this)  ──►  pinned staging  ──►  VRAM
//!    whole model        hot layers, LRU          page-locked        streaming zone
//! ```
//!
//! Autoregressive generation streams every layer once per token, so without a
//! cache each token re-faults the whole model off disk. With it, as many hot
//! layers as fit in the RAM budget stay resident across token steps and skip the
//! disk read entirely. Eviction is strict LRU by byte budget; a single entry
//! larger than the whole budget is kept rather than thrashed (it has nowhere
//! smaller to go).

use crate::error::Result;
use std::collections::{HashMap, VecDeque};

/// Utilization / effectiveness snapshot for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub resident_bytes: u64,
    pub capacity_bytes: u64,
}

impl RamCacheStats {
    /// Fraction of lookups served from RAM (0.0–1.0). Zero if never queried.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// A byte-budgeted LRU cache of layer weight buffers, keyed by layer index.
#[derive(Debug)]
pub struct LayerRamCache {
    capacity_bytes: u64,
    resident_bytes: u64,
    entries: HashMap<u32, Vec<u8>>,
    /// LRU order: least-recently-used at the front, most-recent at the back.
    order: VecDeque<u32>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl LayerRamCache {
    /// Create a cache bounded to `capacity_bytes` of resident layer weights.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Fetch layer `layer`, loading it via `load` on a miss and caching the
    /// result. On a hit the layer is marked most-recently-used; on a miss the
    /// loaded bytes are inserted and LRU entries are evicted to fit the budget.
    pub fn get_or_load<F>(&mut self, layer: u32, load: F) -> Result<&[u8]>
    where
        F: FnOnce() -> Result<Vec<u8>>,
    {
        if self.entries.contains_key(&layer) {
            self.hits += 1;
            self.touch(layer);
        } else {
            let bytes = load()?;
            self.misses += 1;
            self.insert_new(layer, bytes);
        }
        // Borrow is taken after all mutation completes.
        Ok(self.entries.get(&layer).expect("resident after get_or_load"))
    }

    /// Whether a layer is currently resident (does not affect LRU order).
    pub fn contains(&self, layer: u32) -> bool {
        self.entries.contains_key(&layer)
    }

    /// Number of resident layers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes currently held.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// The byte budget.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Drop all entries (keeps the accumulated hit/miss counters).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.resident_bytes = 0;
    }

    /// A stats snapshot.
    pub fn stats(&self) -> RamCacheStats {
        RamCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.entries.len(),
            resident_bytes: self.resident_bytes,
            capacity_bytes: self.capacity_bytes,
        }
    }

    /// Move an existing layer to the most-recently-used position.
    fn touch(&mut self, layer: u32) {
        if let Some(pos) = self.order.iter().position(|&l| l == layer) {
            self.order.remove(pos);
        }
        self.order.push_back(layer);
    }

    /// Insert a freshly loaded layer and evict LRU entries to fit the budget.
    fn insert_new(&mut self, layer: u32, bytes: Vec<u8>) {
        self.resident_bytes += bytes.len() as u64;
        self.entries.insert(layer, bytes);
        self.order.push_back(layer);
        self.evict_to_fit(layer);
    }

    /// Evict least-recently-used entries until within budget, never evicting the
    /// just-inserted `protected` layer (it sits at the back, so it is only ever
    /// the victim when it is the sole entry — which the length guard prevents).
    fn evict_to_fit(&mut self, protected: u32) {
        while self.resident_bytes > self.capacity_bytes && self.order.len() > 1 {
            let victim = *self.order.front().expect("non-empty");
            if victim == protected {
                // Defensive: rotate it to the back and retry.
                self.order.rotate_left(1);
                continue;
            }
            self.order.pop_front();
            if let Some(bytes) = self.entries.remove(&victim) {
                self.resident_bytes -= bytes.len() as u64;
                self.evictions += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Load helper producing `len` bytes of a per-layer pattern.
    fn bytes_of(layer: u32, len: usize) -> Vec<u8> {
        vec![(layer as u8).wrapping_add(1); len]
    }

    #[test]
    fn miss_then_hit() {
        let mut cache = LayerRamCache::new(1024);
        let mut loads = 0;
        let a = cache
            .get_or_load(0, || {
                loads += 1;
                Ok(bytes_of(0, 100))
            })
            .unwrap()
            .to_vec();
        assert_eq!(a, bytes_of(0, 100));
        // Second fetch is a hit — loader not invoked.
        let b = cache
            .get_or_load(0, || {
                loads += 1;
                Ok(bytes_of(0, 100))
            })
            .unwrap()
            .to_vec();
        assert_eq!(b, a);
        assert_eq!(loads, 1);

        let s = cache.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(s.resident_bytes, 100);
    }

    #[test]
    fn evicts_lru_when_over_budget() {
        // Budget for exactly two 100-byte layers.
        let mut cache = LayerRamCache::new(200);
        cache.get_or_load(0, || Ok(bytes_of(0, 100))).unwrap();
        cache.get_or_load(1, || Ok(bytes_of(1, 100))).unwrap();
        assert_eq!(cache.len(), 2);

        // Touch layer 0 so layer 1 becomes LRU.
        cache.get_or_load(0, || Ok(bytes_of(0, 100))).unwrap();
        // Insert layer 2 → must evict layer 1 (the LRU), keep 0 and 2.
        cache.get_or_load(2, || Ok(bytes_of(2, 100))).unwrap();

        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
        assert_eq!(cache.resident_bytes(), 200);
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn oversized_entry_is_kept_not_thrashed() {
        let mut cache = LayerRamCache::new(50);
        // 100 bytes into a 50-byte budget — kept because there's nothing else.
        cache.get_or_load(0, || Ok(bytes_of(0, 100))).unwrap();
        assert!(cache.contains(0));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resident_bytes(), 100);
    }

    #[test]
    fn hit_rate_accumulates() {
        let mut cache = LayerRamCache::new(1024);
        cache.get_or_load(0, || Ok(bytes_of(0, 10))).unwrap(); // miss
        cache.get_or_load(0, || Ok(bytes_of(0, 10))).unwrap(); // hit
        cache.get_or_load(0, || Ok(bytes_of(0, 10))).unwrap(); // hit
        let s = cache.stats();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }
}

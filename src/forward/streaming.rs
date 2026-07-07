//! Layer-streaming compute kernel (`specs.md` §2.2 / §3.2).
//!
//! This is what makes `dlm` run a model bigger than the resident budget: rather
//! than holding every layer's weights in memory ([`CpuKernel`]), a
//! [`StreamingKernel`] keeps only a **bounded window** of materialized layers
//! and streams the rest in on demand. Per `run_block` it fetches that layer's
//! weights from a [`LayerSource`] (memory-mapped checkpoint in production),
//! caches them in an LRU sized to the window, and evicts the least-recently-used
//! layer when the window is full — so peak memory is `window × per-layer`, not
//! the whole model. Hot layers survive across token steps (the tiered CPU-RAM
//! cache of `specs.md` §2.3).
//!
//! Because the kernel is stateless given (weights, KV, position), the output is
//! **identical for any window size** — the window is purely a memory/throughput
//! knob. That makes it a drop-in [`ComputeKernel`]: the generator, batched
//! server, and speculative stack all drive a streamed model unchanged, and a
//! tiny window is bit-for-bit equal to the fully-resident [`CpuKernel`] (tested).
//!
//! [`CpuKernel`]: crate::forward::CpuKernel

use crate::error::Result;
use crate::forward::cpu::{decode_block, BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// Supplies one transformer layer's materialized weights on demand.
///
/// The production source ([`MmapLayerSource`](crate::loader::MmapLayerSource))
/// reads and dequantizes the layer's tensors from a memory-mapped checkpoint;
/// tests can back it with an in-memory `Vec`.
pub trait LayerSource: Send {
    /// Total transformer layers available.
    fn num_layers(&self) -> u32;
    /// Materialize layer `layer`'s weights.
    fn load_layer(&self, layer: u32) -> Result<LayerTensors>;
}

/// Cache-effectiveness snapshot for a [`StreamingKernel`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// A bounded LRU of materialized layers (front of `order` = least recent).
struct LayerLru {
    capacity: usize,
    map: HashMap<u32, LayerTensors>,
    order: VecDeque<u32>,
    stats: StreamStats,
}

impl LayerLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
            stats: StreamStats::default(),
        }
    }

    fn touch(&mut self, layer: u32) {
        if let Some(pos) = self.order.iter().position(|&l| l == layer) {
            self.order.remove(pos);
        }
        self.order.push_back(layer);
    }

    fn insert(&mut self, layer: u32, tensors: LayerTensors) {
        while self.map.len() >= self.capacity {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
        self.map.insert(layer, tensors);
        self.order.push_back(layer);
    }
}

/// A streaming [`ComputeKernel`]: only a window of layers is resident; the rest
/// are pulled from `source` on demand.
pub struct StreamingKernel<S: LayerSource> {
    cfg: BlockConfig,
    source: S,
    num_layers: u32,
    cache: Mutex<LayerLru>,
}

impl<S: LayerSource> StreamingKernel<S> {
    /// Stream `source`'s layers keeping at most `resident_layers` in memory.
    pub fn new(cfg: BlockConfig, source: S, resident_layers: usize) -> Self {
        let num_layers = source.num_layers();
        Self {
            cfg,
            source,
            num_layers,
            cache: Mutex::new(LayerLru::new(resident_layers)),
        }
    }

    /// Cache stats (hits/misses/evictions) accumulated so far.
    pub fn stats(&self) -> StreamStats {
        self.cache.lock().unwrap().stats
    }

    /// Number of layers currently held resident.
    pub fn resident_len(&self) -> usize {
        self.cache.lock().unwrap().map.len()
    }
}

impl<S: LayerSource> ComputeKernel for StreamingKernel<S> {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    fn kv_dim(&self) -> usize {
        self.cfg.kv_dim()
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
        let mut cache = self.cache.lock().unwrap();
        if cache.map.contains_key(&layer) {
            cache.stats.hits += 1;
            cache.touch(layer);
        } else {
            cache.stats.misses += 1;
            let tensors = self.source.load_layer(layer)?;
            cache.insert(layer, tensors);
        }
        // Run the block from the resident copy (borrow tied to the guard).
        let tensors = cache.map.get(&layer).expect("just inserted/looked up");
        let out = decode_block(&self.cfg, tensors, hidden, kv, position)?;
        hidden.copy_from_slice(&out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::CpuKernel;

    /// In-memory layer source for tests.
    struct VecSource(Vec<LayerTensors>);
    impl LayerSource for VecSource {
        fn num_layers(&self) -> u32 {
            self.0.len() as u32
        }
        fn load_layer(&self, layer: u32) -> Result<LayerTensors> {
            Ok(self.0[layer as usize].clone())
        }
    }

    fn cfg() -> BlockConfig {
        BlockConfig {
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        }
    }

    fn layers(c: &BlockConfig, n: usize) -> Vec<LayerTensors> {
        // Distinct-but-valid weights per layer (index-seeded), so an eviction bug
        // that returned the wrong layer would change the output.
        (0..n)
            .map(|i| {
                let s = 0.01 * (i as f32 + 1.0);
                LayerTensors {
                    q_proj: vec![s; c.q_dim() * c.hidden_size],
                    k_proj: vec![s; c.kv_dim() * c.hidden_size],
                    v_proj: vec![s; c.kv_dim() * c.hidden_size],
                    o_proj: vec![s; c.hidden_size * c.q_dim()],
                    gate_proj: vec![s; c.intermediate_size * c.hidden_size],
                    up_proj: vec![s; c.intermediate_size * c.hidden_size],
                    down_proj: vec![s; c.hidden_size * c.intermediate_size],
                    input_layernorm: vec![1.0; c.hidden_size],
                    post_attention_layernorm: vec![1.0; c.hidden_size],
                }
            })
            .collect()
    }

    /// A tiny window must produce exactly the same forward pass as holding every
    /// layer resident — only memory differs.
    #[test]
    fn streamed_matches_resident() {
        let c = cfg();
        let ls = layers(&c, 6);
        let resident = CpuKernel::new(c, ls.clone()).unwrap();
        let streaming = StreamingKernel::new(c, VecSource(ls), 2); // window of 2/6

        let mut h_res = vec![0.5f32; 8];
        let mut h_str = vec![0.5f32; 8];
        let mut kv_res = KvLayerCache::new(c.kv_dim());
        let mut kv_str = KvLayerCache::new(c.kv_dim());

        for position in 0..4 {
            for layer in 0..6 {
                resident.run_block(layer, &mut h_res, &mut kv_res, position).unwrap();
                streaming.run_block(layer, &mut h_str, &mut kv_str, position).unwrap();
            }
        }
        assert_eq!(h_res, h_str, "streamed forward diverged from resident");
        assert!(streaming.resident_len() <= 2, "window exceeded capacity");
        assert!(streaming.stats().evictions > 0, "expected eviction with a small window");
    }
}

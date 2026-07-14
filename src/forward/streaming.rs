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
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Supplies one transformer layer's materialized weights on demand.
///
/// The production source ([`MmapLayerSource`](crate::loader::MmapLayerSource))
/// reads and dequantizes the layer's tensors from a memory-mapped checkpoint;
/// tests can back it with an in-memory `Vec`. `Sync` is required so a background
/// prefetch thread can pull the next layer while the current one computes.
pub trait LayerSource: Send + Sync {
    /// Total transformer layers available.
    fn num_layers(&self) -> u32;
    /// Materialize layer `layer`'s weights.
    fn load_layer(&self, layer: u32) -> Result<LayerTensors>;
}

/// Cache-effectiveness snapshot for a [`StreamingKernel`]. `prefetched` counts
/// layers the background worker materialized ahead of the compute thread needing
/// them — the overlap that hides load latency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub prefetched: u64,
    /// The prefetch depth in effect now (the live auto-tuned value under
    /// `--auto-prefetch`, else the fixed depth). A gauge, not a counter.
    pub depth: u32,
}

/// A bounded LRU of materialized layers (front of `order` = least recent).
/// Layers are `Arc`ed so the compute thread can hold one for `decode_block`
/// without keeping the cache locked — letting the prefetch worker load in
/// parallel, and surviving eviction while in flight.
struct LayerLru {
    capacity: usize,
    map: HashMap<u32, Arc<LayerTensors>>,
    order: VecDeque<u32>,
    /// Layers currently being loaded (by compute or the worker); de-dupes
    /// concurrent loads of the same layer.
    loading: HashSet<u32>,
    stats: StreamStats,
}

impl LayerLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
            loading: HashSet::new(),
            stats: StreamStats::default(),
        }
    }

    fn touch(&mut self, layer: u32) {
        if let Some(pos) = self.order.iter().position(|&l| l == layer) {
            self.order.remove(pos);
        }
        self.order.push_back(layer);
    }

    fn insert(&mut self, layer: u32, tensors: Arc<LayerTensors>) {
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

/// State shared between the compute thread and the prefetch worker.
struct Shared<S: LayerSource> {
    cfg: BlockConfig,
    source: S,
    cache: Mutex<LayerLru>,
    /// Signaled when a layer finishes loading, so waiters can recheck.
    ready: Condvar,
    /// EWMA of a layer load's duration (ns), for auto prefetch depth.
    load_ns: AtomicU64,
    /// EWMA of a block's compute duration (ns), for auto prefetch depth.
    compute_ns: AtomicU64,
}

/// Fold `sample` into an exponential moving average in `slot` (¼ weight on the
/// new value). Relaxed and racy on purpose — it only feeds a heuristic.
fn ewma(slot: &AtomicU64, sample: u64) {
    let old = slot.load(Ordering::Relaxed);
    let new = if old == 0 { sample } else { (old * 3 + sample) / 4 };
    slot.store(new, Ordering::Relaxed);
}

impl<S: LayerSource> Shared<S> {
    /// Ensure `layer` is resident, loading it (blocking, without holding the
    /// cache lock) if absent. Concurrent callers for the same layer de-dupe: one
    /// loads, the rest wait on `ready`. `by_worker` tags a background prefetch in
    /// the stats. Returns an `Arc` clone of the layer.
    fn ensure(&self, layer: u32, by_worker: bool) -> Result<Arc<LayerTensors>> {
        let mut cache = self.cache.lock().unwrap();
        loop {
            if let Some(t) = cache.map.get(&layer) {
                let t = Arc::clone(t);
                cache.touch(layer);
                return Ok(t);
            }
            if cache.loading.contains(&layer) {
                cache = self.ready.wait(cache).unwrap();
                continue;
            }
            // We own the load. Release the lock so compute/other loads proceed.
            cache.loading.insert(layer);
            drop(cache);
            let t = std::time::Instant::now();
            let loaded = self.source.load_layer(layer);
            ewma(&self.load_ns, t.elapsed().as_nanos() as u64);
            let mut cache = self.cache.lock().unwrap();
            cache.loading.remove(&layer);
            self.ready.notify_all();
            let arc = Arc::new(loaded?);
            cache.insert(layer, Arc::clone(&arc));
            if by_worker {
                cache.stats.prefetched += 1;
            }
            return Ok(arc);
        }
    }

    /// Compute-path fetch: records a hit (resident) or miss (had to load), then
    /// ensures the layer is available.
    fn fetch(&self, layer: u32) -> Result<Arc<LayerTensors>> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(t) = cache.map.get(&layer) {
                let t = Arc::clone(t);
                cache.stats.hits += 1;
                cache.touch(layer);
                return Ok(t);
            }
            cache.stats.misses += 1;
        }
        self.ensure(layer, false)
    }
}

/// A streaming [`ComputeKernel`]: only a window of layers is resident; the rest
/// are pulled from `source` on demand. A background worker **prefetches the next
/// layer while the current one computes**, so a layer's load latency is hidden
/// behind the previous layer's arithmetic (layers are consumed strictly in
/// order). Output is identical to loading synchronously — prefetch only changes
/// *when* the load happens.
pub struct StreamingKernel<S: LayerSource + 'static> {
    shared: Arc<Shared<S>>,
    num_layers: u32,
    /// How many layers ahead to keep loading (1 = just the next). Clamped to
    /// `window - 1` so prefetched layers aren't evicted before they're used.
    /// Ignored when `auto` is set.
    prefetch_depth: u32,
    /// Auto-tune the depth from measured load-vs-compute time each block.
    auto: bool,
    /// Resident window size (LRU capacity), for clamping the depth.
    window: u32,
    /// Requests to prefetch a layer; `None` once the kernel is being dropped.
    prefetch_tx: Option<Sender<u32>>,
    worker: Option<JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
}

impl<S: LayerSource + 'static> StreamingKernel<S> {
    /// Stream `source`'s layers keeping at most `resident_layers` in memory,
    /// prefetching one layer ahead in the background.
    pub fn new(cfg: BlockConfig, source: S, resident_layers: usize) -> Self {
        let num_layers = source.num_layers();
        let shared = Arc::new(Shared {
            cfg,
            source,
            cache: Mutex::new(LayerLru::new(resident_layers)),
            ready: Condvar::new(),
            load_ns: AtomicU64::new(0),
            compute_ns: AtomicU64::new(0),
        });
        let (tx, rx) = channel::<u32>();
        let stopped = Arc::new(AtomicBool::new(false));
        let worker = {
            let shared = Arc::clone(&shared);
            let stopped = Arc::clone(&stopped);
            std::thread::spawn(move || {
                while let Ok(layer) = rx.recv() {
                    if stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    // Best-effort: a failed load just becomes a compute-path miss.
                    let _ = shared.ensure(layer, true);
                }
            })
        };
        let window = resident_layers.max(1) as u32;
        Self {
            shared,
            num_layers,
            // Default: one layer ahead (clamped to the window).
            prefetch_depth: 1.min(window.saturating_sub(1)),
            auto: false,
            window,
            prefetch_tx: Some(tx),
            worker: Some(worker),
            stopped,
        }
    }

    /// Keep `depth` layers loading ahead of the compute thread instead of one.
    /// Useful when a layer's load latency exceeds a single block's compute, so
    /// one lookahead can't fully hide it. Clamped to `window - 1` (can't usefully
    /// prefetch more layers than the resident window holds). `0` disables
    /// prefetch.
    pub fn with_prefetch_depth(mut self, depth: u32) -> Self {
        self.prefetch_depth = depth.min(self.window.saturating_sub(1));
        self.auto = false;
        self
    }

    /// Auto-tune the prefetch depth: each block, size it to
    /// `ceil(load_time / compute_time)` from measured moving averages (clamped to
    /// `window - 1`), so it self-adjusts to whatever the load-vs-compute ratio is
    /// without a manual `--prefetch-depth`. Overrides a fixed depth.
    pub fn with_auto_prefetch(mut self) -> Self {
        self.auto = true;
        self
    }

    /// The prefetch depth this block would use (fixed, or the current auto value).
    pub fn current_prefetch_depth(&self) -> u32 {
        if self.auto {
            self.auto_depth()
        } else {
            self.prefetch_depth
        }
    }

    /// Depth from the measured load/compute ratio, clamped to `[1, window-1]`
    /// (or 0 when the window is too small to prefetch). Falls back to 1 until the
    /// first measurements land.
    fn auto_depth(&self) -> u32 {
        let max = self.window.saturating_sub(1);
        if max == 0 {
            return 0;
        }
        let load = self.shared.load_ns.load(Ordering::Relaxed);
        let compute = self.shared.compute_ns.load(Ordering::Relaxed).max(1);
        if load == 0 {
            return 1; // no load measured yet — one ahead
        }
        let need = load.div_ceil(compute) as u32;
        need.clamp(1, max)
    }

    /// Cache stats (hits/misses/evictions/prefetched) plus the current prefetch
    /// depth. The counters accumulate in the cache; `depth` is filled live.
    pub fn stats(&self) -> StreamStats {
        let mut s = self.shared.cache.lock().unwrap().stats;
        s.depth = self.current_prefetch_depth();
        s
    }

    /// Number of layers currently held resident.
    pub fn resident_len(&self) -> usize {
        self.shared.cache.lock().unwrap().map.len()
    }
}

impl<S: LayerSource + 'static> Drop for StreamingKernel<S> {
    fn drop(&mut self) {
        // Close the request channel and let the worker finish its current load.
        self.stopped.store(true, Ordering::Relaxed);
        self.prefetch_tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl<S: LayerSource + 'static> ComputeKernel for StreamingKernel<S> {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.shared.cfg.hidden_size
    }

    fn kv_dim(&self) -> usize {
        self.shared.cfg.kv_dim()
    }

    fn stream_stats(&self) -> Option<StreamStats> {
        Some(self.stats())
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
        let tensors = self.shared.fetch(layer)?;
        // Kick off the next `depth` layers' loads now, so they overlap this and
        // following blocks' compute (layers run strictly in order, wrapping per
        // token). Already-resident/in-flight layers de-dupe cheaply. `depth` is
        // fixed or, under auto, sized to the current load/compute ratio.
        let depth = if self.auto { self.auto_depth() } else { self.prefetch_depth };
        if let Some(tx) = &self.prefetch_tx {
            for ahead in 1..=depth {
                let _ = tx.send((layer + ahead) % self.num_layers);
            }
        }
        // Compute without holding the cache lock — the worker loads in parallel.
        let t = std::time::Instant::now();
        let out = decode_block(&self.shared.cfg, &tensors, hidden, kv, position)?;
        if self.auto {
            ewma(&self.shared.compute_ns, t.elapsed().as_nanos() as u64);
        }
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
            rms_eps: 1e-5, rope_scaling: None,
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
                    post_attention_layernorm: vec![1.0; c.hidden_size], ..Default::default()
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

    /// A source that sleeps on load, so a prefetch started during one block's
    /// compute has time to finish before the next block is requested.
    struct SlowSource(Vec<LayerTensors>);
    impl LayerSource for SlowSource {
        fn num_layers(&self) -> u32 {
            self.0.len() as u32
        }
        fn load_layer(&self, layer: u32) -> Result<LayerTensors> {
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(self.0[layer as usize].clone())
        }
    }

    #[test]
    fn worker_prefetches_next_layer() {
        let c = cfg();
        let k = StreamingKernel::new(c, SlowSource(layers(&c, 4)), 4);
        let mut h = vec![0.5f32; 8];
        let mut kv = KvLayerCache::new(c.kv_dim());

        // Compute layer 0; this requests a prefetch of layer 1. With no further
        // run_block competing, the worker loads layer 1 uncontended.
        k.run_block(0, &mut h, &mut kv, 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));

        let s = k.stats();
        assert!(s.prefetched >= 1, "background worker should have prefetched a layer: {s:?}");
    }

    #[test]
    fn deeper_prefetch_loads_more_layers_ahead() {
        let c = cfg();
        // Window of 6 with depth 3: one block requests three layers ahead.
        let k = StreamingKernel::new(c, SlowSource(layers(&c, 6)), 6).with_prefetch_depth(3);
        let mut h = vec![0.5f32; 8];
        let mut kv = KvLayerCache::new(c.kv_dim());
        k.run_block(0, &mut h, &mut kv, 0).unwrap(); // requests prefetch of 1,2,3
        std::thread::sleep(std::time::Duration::from_millis(40));
        assert_eq!(k.stats().prefetched, 3, "depth-3 should prefetch three layers: {:?}", k.stats());
    }

    #[test]
    fn auto_prefetch_tunes_to_load_compute_ratio() {
        let c = cfg();
        // Slow load (~1ms) vs tiny compute → huge ratio → depth pins to window-1.
        let k = StreamingKernel::new(c, SlowSource(layers(&c, 6)), 4).with_auto_prefetch();
        let mut h = vec![0.5f32; 8];
        let mut kv = KvLayerCache::new(c.kv_dim());
        for pos in 0..2 {
            for layer in 0..6 {
                k.run_block(layer, &mut h, &mut kv, pos).unwrap();
            }
        }
        assert_eq!(
            k.current_prefetch_depth(),
            3,
            "load >> compute should pin auto depth to window-1: load_ns/compute_ns unbalanced"
        );
    }

    #[test]
    fn fixed_depth_ignores_measurements() {
        let c = cfg();
        let k = StreamingKernel::new(c, SlowSource(layers(&c, 6)), 6).with_prefetch_depth(2);
        let mut h = vec![0.5f32; 8];
        let mut kv = KvLayerCache::new(c.kv_dim());
        for layer in 0..6 {
            k.run_block(layer, &mut h, &mut kv, 0).unwrap();
        }
        assert_eq!(k.current_prefetch_depth(), 2, "fixed depth must not be auto-tuned");
    }

    #[test]
    fn prefetch_depth_clamped_to_window() {
        let c = cfg();
        // Window of 2 only has room for 1 layer ahead; a requested depth of 5
        // must clamp to 1 (else it would thrash the tiny window).
        let k = StreamingKernel::new(c, SlowSource(layers(&c, 6)), 2).with_prefetch_depth(5);
        let mut h = vec![0.5f32; 8];
        let mut kv = KvLayerCache::new(c.kv_dim());
        k.run_block(0, &mut h, &mut kv, 0).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(40));
        assert_eq!(k.stats().prefetched, 1, "depth must clamp to window-1: {:?}", k.stats());
    }

    #[test]
    fn prefetch_makes_later_blocks_hit() {
        // Over several tokens the worker keeps a layer ahead, so most on-demand
        // fetches are hits rather than blocking loads.
        let c = cfg();
        let k = StreamingKernel::new(c, SlowSource(layers(&c, 4)), 4);
        let mut h = vec![0.5f32; 8];
        let mut kv = KvLayerCache::new(c.kv_dim());
        for pos in 0..4 {
            for layer in 0..4 {
                k.run_block(layer, &mut h, &mut kv, pos).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(2)); // let prefetch land
            }
        }
        let s = k.stats();
        assert!(s.hits > s.misses, "prefetch should make hits dominate: {s:?}");
    }
}

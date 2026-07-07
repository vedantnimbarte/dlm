//! GPU layer-streaming compute kernel (`specs.md` §2.2/§3.2) — the combined path
//! that actually delivers the product thesis: **run a model larger than VRAM on
//! the GPU** by keeping only a bounded window of layer *weights* resident in VRAM
//! and streaming the rest in over PCIe on demand, while each layer's KV history
//! stays resident (the cache zone).
//!
//! It mirrors [`GpuKernel`](crate::forward::GpuKernel) op-for-op — same
//! `dlm_decode_block` device call — but instead of uploading every layer's
//! weights up front, it holds an LRU of `resident_layers` device-resident weight
//! sets and uploads a layer's weights from a host [`LayerSource`] on a miss,
//! evicting the least-recently-used set. Peak weight VRAM is
//! `resident_layers × per-layer`, not the whole model. KV buffers for *all*
//! layers are allocated up front and persist across weight evictions, so
//! attention still sees full history.
//!
//! **Status: compiled under `cuda-kernels`, but NOT yet validated on real GPU
//! hardware** (no GPU in CI). The transformer math is the same FFI the CPU-parity
//! test covers ([`tests/gpu_parity.rs`]); the streaming/eviction layer around it
//! is new and unproven on-device. Treat as experimental until run on a GPU.

use crate::error::{DlmError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use crate::forward::streaming::{LayerSource, StreamStats};
use crate::gpu::device::{synchronize, DeviceBuffer};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

extern "C" {
    // Same device entry point as `forward::gpu` (see `src/gpu/kernels.cu`).
    #[allow(clippy::too_many_arguments)]
    fn dlm_decode_block(
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        inter: i32,
        rope_theta: f32,
        rms_eps: f32,
        q_proj: *const f32,
        k_proj: *const f32,
        v_proj: *const f32,
        o_proj: *const f32,
        gate_proj: *const f32,
        up_proj: *const f32,
        down_proj: *const f32,
        in_norm: *const f32,
        post_norm: *const f32,
        x: *mut f32,
        kv_keys: *mut f32,
        kv_values: *mut f32,
        num_positions: i32,
        position: i32,
    ) -> i32;
}

/// One layer's weight buffers, resident in VRAM (no KV — that lives in `GpuKv`).
struct GpuWeights {
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: DeviceBuffer,
    o_proj: DeviceBuffer,
    gate_proj: DeviceBuffer,
    up_proj: DeviceBuffer,
    down_proj: DeviceBuffer,
    input_layernorm: DeviceBuffer,
    post_attention_layernorm: DeviceBuffer,
}

impl GpuWeights {
    fn upload(t: &LayerTensors) -> Result<Self> {
        Ok(Self {
            q_proj: DeviceBuffer::from_slice(&t.q_proj)?,
            k_proj: DeviceBuffer::from_slice(&t.k_proj)?,
            v_proj: DeviceBuffer::from_slice(&t.v_proj)?,
            o_proj: DeviceBuffer::from_slice(&t.o_proj)?,
            gate_proj: DeviceBuffer::from_slice(&t.gate_proj)?,
            up_proj: DeviceBuffer::from_slice(&t.up_proj)?,
            down_proj: DeviceBuffer::from_slice(&t.down_proj)?,
            input_layernorm: DeviceBuffer::from_slice(&t.input_layernorm)?,
            post_attention_layernorm: DeviceBuffer::from_slice(&t.post_attention_layernorm)?,
        })
    }
}

/// One layer's persistent KV history in VRAM.
struct GpuKv {
    keys: DeviceBuffer,
    values: DeviceBuffer,
}

/// A bounded LRU of device-resident weight sets (front of `order` = least
/// recent). Weights are `Arc`ed so the compute thread can hold a layer for the
/// kernel launch without keeping the cache locked, letting the prefetch worker
/// upload the next layer in parallel.
struct WeightLru {
    capacity: usize,
    map: HashMap<u32, Arc<GpuWeights>>,
    order: VecDeque<u32>,
    /// Layers currently being loaded (compute or worker); de-dupes double loads.
    loading: HashSet<u32>,
    stats: StreamStats,
}

impl WeightLru {
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

    fn insert(&mut self, layer: u32, weights: Arc<GpuWeights>) {
        while self.map.len() >= self.capacity {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict); // frees the VRAM buffers via Drop (if last Arc)
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
        self.map.insert(layer, weights);
        self.order.push_back(layer);
    }
}

/// State shared between the compute thread and the prefetch worker.
struct GpuShared<S: LayerSource> {
    cfg: BlockConfig,
    source: S,
    weights: Mutex<WeightLru>,
    ready: Condvar,
}

impl<S: LayerSource> GpuShared<S> {
    /// Ensure `layer`'s weights are VRAM-resident, loading (host) + uploading
    /// (VRAM) without holding the cache lock if absent. Same-layer callers
    /// de-dupe on a `loading` set + condvar. `by_worker` tags a prefetch.
    fn ensure(&self, layer: u32, by_worker: bool) -> Result<Arc<GpuWeights>> {
        let mut cache = self.weights.lock().unwrap();
        loop {
            if let Some(w) = cache.map.get(&layer) {
                let w = Arc::clone(w);
                cache.touch(layer);
                return Ok(w);
            }
            if cache.loading.contains(&layer) {
                cache = self.ready.wait(cache).unwrap();
                continue;
            }
            cache.loading.insert(layer);
            drop(cache);
            // Expensive part — host load (disk + dequant) then VRAM upload — runs
            // with the lock released, so it overlaps the compute thread.
            let uploaded = self
                .source
                .load_layer(layer)
                .and_then(|t| GpuWeights::upload(&t).map(Arc::new));
            let mut cache = self.weights.lock().unwrap();
            cache.loading.remove(&layer);
            self.ready.notify_all();
            let w = uploaded?;
            cache.insert(layer, Arc::clone(&w));
            if by_worker {
                cache.stats.prefetched += 1;
            }
            return Ok(w);
        }
    }

    /// Compute-path fetch: records hit/miss then ensures residency.
    fn fetch(&self, layer: u32) -> Result<Arc<GpuWeights>> {
        {
            let mut cache = self.weights.lock().unwrap();
            if let Some(w) = cache.map.get(&layer) {
                let w = Arc::clone(w);
                cache.stats.hits += 1;
                cache.touch(layer);
                return Ok(w);
            }
            cache.stats.misses += 1;
        }
        self.ensure(layer, false)
    }
}

/// A GPU [`ComputeKernel`] that streams a window of layer weights through VRAM
/// while keeping per-layer KV resident. A background worker **uploads the next
/// layer's weights to VRAM while the current block computes**, so the host load
/// (disk read + dequant) is hidden behind GPU compute.
pub struct StreamingGpuKernel<S: LayerSource + 'static> {
    shared: Arc<GpuShared<S>>,
    num_layers: u32,
    kv_capacity_tokens: usize,
    kv: Vec<GpuKv>,
    prefetch_tx: Option<Sender<u32>>,
    worker: Option<JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
}

impl<S: LayerSource + 'static> StreamingGpuKernel<S> {
    /// Build a streaming GPU kernel: allocate per-layer KV for up to
    /// `max_kv_tokens` positions and stream weights keeping at most
    /// `resident_layers` sets in VRAM, prefetching one layer ahead.
    pub fn new(
        cfg: BlockConfig,
        source: S,
        max_kv_tokens: usize,
        resident_layers: usize,
    ) -> Result<Self> {
        let num_layers = source.num_layers();
        let cap = max_kv_tokens.max(1);
        let kv_buffer_len = cap * cfg.kv_dim();
        let mut kv = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            kv.push(GpuKv {
                keys: DeviceBuffer::new(kv_buffer_len)?,
                values: DeviceBuffer::new(kv_buffer_len)?,
            });
        }
        let shared = Arc::new(GpuShared {
            cfg,
            source,
            weights: Mutex::new(WeightLru::new(resident_layers)),
            ready: Condvar::new(),
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
                    let _ = shared.ensure(layer, true); // best-effort
                }
            })
        };
        Ok(Self {
            shared,
            num_layers,
            kv_capacity_tokens: cap,
            kv,
            prefetch_tx: Some(tx),
            worker: Some(worker),
            stopped,
        })
    }

    /// Streaming cache stats (hits/misses/evictions/prefetched).
    pub fn stats(&self) -> StreamStats {
        let mut s = self.shared.weights.lock().unwrap().stats;
        s.depth = 1;
        s
    }
}

impl<S: LayerSource + 'static> Drop for StreamingGpuKernel<S> {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
        self.prefetch_tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

impl<S: LayerSource + 'static> ComputeKernel for StreamingGpuKernel<S> {
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
        let kv_dim = self.shared.cfg.kv_dim();
        let cfg = &self.shared.cfg;
        let num_positions = kv.len();
        if num_positions >= self.kv_capacity_tokens {
            return Err(DlmError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {position}",
                self.kv_capacity_tokens
            )));
        }

        // Ensure this layer's weights are resident (Arc keeps them alive through
        // the launch even if the worker evicts them from the window).
        let w = self.shared.fetch(layer)?;
        // Kick off the next layer's upload so it overlaps this block's compute.
        if let Some(tx) = &self.prefetch_tx {
            let _ = tx.send((layer + 1) % self.num_layers);
        }
        let dkv = &self.kv[layer as usize];

        // Only the hidden vector crosses the bus per layer; weights are already
        // resident (this window) and KV lives on-device.
        let d_hidden = DeviceBuffer::from_slice(hidden)?;

        // SAFETY: all pointers are live device allocations sized as the kernel
        // expects; the KV buffers have capacity for `num_positions + 1`.
        let code = unsafe {
            dlm_decode_block(
                cfg.hidden_size as i32,
                cfg.q_dim() as i32,
                kv_dim as i32,
                cfg.num_heads as i32,
                cfg.num_kv_heads as i32,
                cfg.head_dim as i32,
                cfg.intermediate_size as i32,
                cfg.rope_theta,
                cfg.rms_eps,
                w.q_proj.as_ptr(),
                w.k_proj.as_ptr(),
                w.v_proj.as_ptr(),
                w.o_proj.as_ptr(),
                w.gate_proj.as_ptr(),
                w.up_proj.as_ptr(),
                w.down_proj.as_ptr(),
                w.input_layernorm.as_ptr(),
                w.post_attention_layernorm.as_ptr(),
                d_hidden.as_mut_ptr(),
                dkv.keys.as_mut_ptr(),
                dkv.values.as_mut_ptr(),
                num_positions as i32,
                position as i32,
            )
        };
        if code != 0 {
            return Err(DlmError::Gpu {
                api: "dlm_decode_block",
                code,
            });
        }
        synchronize()?;
        d_hidden.download(hidden)?;

        // Keep the orchestrator's length bookkeeping in step (real K/V is in VRAM).
        kv.append(&vec![0.0; kv_dim], &vec![0.0; kv_dim])?;
        Ok(())
    }
}

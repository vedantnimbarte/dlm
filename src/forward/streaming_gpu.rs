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
//! Validated on-device against the CPU oracle by [`tests/gpu_parity.rs`]
//! (`streaming_gpu_matches_resident` drives real evictions and asserts the output
//! equals the fully-resident kernel's). CI has no GPU, so that check is manual —
//! run the `cuda-kernels` suite on a real card before tagging a release.

use crate::error::{DlmError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use crate::forward::streaming::{LayerSource, StreamStats};
use crate::gpu::device::{synchronize_default, DeviceBuffer, Stream};
use crate::memory::PinnedBuffer;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

// The device entry point is declared once in `forward::gpu` and imported here.
// Restating the `extern` block let this path keep calling the old ABI after the
// kernel signature changed — a silent mismatch the compiler only warns about.
use crate::forward::gpu::{bias_ptr, dlm_decode_block, upload_bias};

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
    /// Attention biases (Qwen2 et al.). Uploaded synchronously — a few KB each,
    /// so they don't warrant a slot in the async staging layout.
    q_bias: Option<DeviceBuffer>,
    k_bias: Option<DeviceBuffer>,
    v_bias: Option<DeviceBuffer>,
    /// Native dtype of the projection weights (see `Weights::dtype_code`).
    w_dtype: i32,
    /// Group size for int4 weights; 0 for the float dtypes.
    w_group_size: i32,
}

impl GpuWeights {
    /// The layer's nine weight tensors as `(native bytes, element count)`, in a
    /// fixed order (for staging/upload). The seven projections keep their native
    /// checkpoint dtype — they are staged and uploaded verbatim, so a bf16 layer
    /// puts half as many bytes across PCIe as an upsized f32 copy would. The two
    /// norms are f32 (a few KB; not worth a dtype path).
    fn tensors(t: &LayerTensors) -> [(&[u8], usize); 9] {
        use crate::forward::cpu::bytemuck_cast;
        [
            (t.q_proj.as_bytes(), t.q_proj.len()),
            (t.k_proj.as_bytes(), t.k_proj.len()),
            (t.v_proj.as_bytes(), t.v_proj.len()),
            (t.o_proj.as_bytes(), t.o_proj.len()),
            (t.gate_proj.as_bytes(), t.gate_proj.len()),
            (t.up_proj.as_bytes(), t.up_proj.len()),
            (t.down_proj.as_bytes(), t.down_proj.len()),
            (bytemuck_cast(&t.input_layernorm), t.input_layernorm.len()),
            (
                bytemuck_cast(&t.post_attention_layernorm),
                t.post_attention_layernorm.len(),
            ),
        ]
    }

    /// Total `f32` elements across a layer's weights (to size the staging buffer).
    fn f32_count(cfg: &BlockConfig) -> usize {
        let h = cfg.hidden_size;
        cfg.q_dim() * h            // q
            + cfg.kv_dim() * h * 2 // k, v
            + h * cfg.q_dim()      // o
            + cfg.intermediate_size * h * 2 // gate, up
            + h * cfg.intermediate_size // down
            + h * 2 // two norms
    }

    /// Upload a layer's weights to VRAM **asynchronously** via `stream`: stage all
    /// tensors into the page-locked `staging` buffer, enqueue a copy per buffer on
    /// the (non-blocking) copy stream, then wait for it. Because the copy runs on
    /// its own stream, it overlaps kernels on the default stream — the caller's
    /// wait doesn't stop the GPU. On return the uploads are complete and `staging`
    /// is free to reuse.
    fn upload_async(t: &LayerTensors, stream: &Stream, staging: &mut PinnedBuffer) -> Result<Self> {
        let tensors = Self::tensors(t);
        // Phase 1: stage all tensors into pinned memory, recording layout.
        // (byte offset, byte length, element count)
        let mut layout: Vec<(usize, usize, usize)> = Vec::with_capacity(9);
        {
            let dst = staging.as_mut_slice();
            let mut byte_off = 0usize;
            for (src, len) in tensors {
                let bytes = src.len();
                dst[byte_off..byte_off + bytes].copy_from_slice(src);
                layout.push((byte_off, bytes, len));
                byte_off += bytes;
            }
        }
        // Phase 2: allocate device buffers and enqueue async copies from staging.
        let base = staging.as_ptr();
        let mut bufs: Vec<DeviceBuffer> = Vec::with_capacity(9);
        for &(off, bytes, len) in &layout {
            let buf = DeviceBuffer::new_bytes(bytes, len.max(1))?;
            let pinned = unsafe { base.add(off) } as *const std::ffi::c_void;
            buf.upload_async_bytes(pinned, bytes, stream)?;
            bufs.push(buf);
        }
        // Wait for the copies (staging becomes reusable; weights are ready).
        stream.synchronize()?;
        let mut it = bufs.into_iter();
        Ok(Self {
            q_proj: it.next().unwrap(),
            k_proj: it.next().unwrap(),
            v_proj: it.next().unwrap(),
            o_proj: it.next().unwrap(),
            gate_proj: it.next().unwrap(),
            up_proj: it.next().unwrap(),
            down_proj: it.next().unwrap(),
            input_layernorm: it.next().unwrap(),
            post_attention_layernorm: it.next().unwrap(),
            q_bias: upload_bias(t.q_bias.as_ref())?,
            k_bias: upload_bias(t.k_bias.as_ref())?,
            v_bias: upload_bias(t.v_bias.as_ref())?,
            w_dtype: t.q_proj.dtype_code(),
            w_group_size: t.q_proj.group_size() as i32,
        })
    }
}

/// One layer's persistent KV history in VRAM.
struct GpuKv {
    keys: DeviceBuffer,
    values: DeviceBuffer,
}

/// A bounded cache of device-resident weight sets, ordered least-recent first.
///
/// Despite the name it evicts the **most** recently used entry — see
/// [`WeightLru::insert`] for why LRU is the wrong policy for a cyclic layer scan.
/// Weights are `Arc`ed so the compute thread can hold a layer for the kernel
/// launch without keeping the cache locked (and so a layer evicted while still in
/// flight stays alive), letting the prefetch worker upload the next in parallel.
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

    /// Insert `layer`, evicting the **most** recently used entry if full.
    ///
    /// Not a typo, and the reason is the whole ballgame for streaming. A forward
    /// pass walks the layers in a cycle — 0,1,…,N-1,0,1,… — and with a window
    /// smaller than the model, LRU is not merely suboptimal here, it is
    /// *pessimal*: the least-recently-used layer under a cyclic scan is precisely
    /// the one needed soonest, so every eviction throws away the next hit and the
    /// hit rate collapses to zero. Measured on a 3B with 23 of 28 layers
    /// resident: 28 misses per token — every layer, every token, as if the window
    /// did not exist.
    ///
    /// The item used furthest in the future is instead the one just finished (it
    /// comes round again only after a full cycle), which is what Belady's optimal
    /// policy would evict and what MRU picks. Keeping the layers *ahead* of the
    /// cursor turns the window back into a cache: misses drop to roughly
    /// `N - capacity` per token instead of `N`.
    fn insert(&mut self, layer: u32, weights: Arc<GpuWeights>) {
        while self.map.len() >= self.capacity {
            let Some(i) = evict_index(&self.order, layer) else {
                break;
            };
            match self.order.remove(i) {
                Some(evict) => {
                    self.map.remove(&evict); // frees the VRAM buffers via Drop (if last Arc)
                    self.stats.evictions += 1;
                }
                None => break,
            }
        }
        self.map.insert(layer, weights);
        self.order.push_back(layer);
    }
}

/// Index in `order` (least-recent first) of the entry to evict when inserting
/// `inserting`: the most recently used one, skipping `inserting` itself in case a
/// concurrent load already placed it there. `None` when there is nothing safe to
/// evict.
///
/// Split out from [`WeightLru::insert`] so the policy — the subtle part — is
/// testable without a GPU.
fn evict_index(order: &VecDeque<u32>, inserting: u32) -> Option<usize> {
    match order.back().copied() {
        Some(l) if l == inserting => order.len().checked_sub(2),
        Some(_) => order.len().checked_sub(1),
        None => None,
    }
}

/// State shared between the compute thread and the prefetch worker.
struct GpuShared<S: LayerSource> {
    cfg: BlockConfig,
    source: S,
    weights: Mutex<WeightLru>,
    ready: Condvar,
    /// Dedicated non-blocking stream for weight uploads (overlaps compute).
    copy_stream: Stream,
    /// Page-locked staging buffer for async H2D. Guarded so concurrent uploaders
    /// (worker + a compute-path miss) don't clobber it mid-transfer.
    staging: Mutex<PinnedBuffer>,
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
            // Expensive part — host load (disk + dequant) then async VRAM upload —
            // runs with the cache lock released, so it overlaps the compute thread.
            // The upload is enqueued on the non-blocking copy stream, so the PCIe
            // transfer itself overlaps the default-stream kernel.
            let uploaded = self.source.load_layer(layer).and_then(|t| {
                let mut staging = self.staging.lock().unwrap();
                GpuWeights::upload_async(&t, &self.copy_stream, &mut staging).map(Arc::new)
            });
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
    /// RoPE inverse frequencies (see [`GpuKernel`](crate::forward::GpuKernel)):
    /// computed once by the shared host function, resident for the kernel's life.
    inv_freq: DeviceBuffer,
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
        // Page-locked staging sized to one layer's weights (all layers same size).
        let staging_bytes = GpuWeights::f32_count(&cfg) * std::mem::size_of::<f32>();
        let shared = Arc::new(GpuShared {
            cfg,
            source,
            weights: Mutex::new(WeightLru::new(resident_layers)),
            ready: Condvar::new(),
            copy_stream: Stream::new_nonblocking()?,
            staging: Mutex::new(PinnedBuffer::with_len(staging_bytes)?),
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
        let inv_freq = DeviceBuffer::from_slice(&crate::forward::cpu::rope_inv_freqs(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_scaling,
        ))?;
        Ok(Self {
            shared,
            num_layers,
            kv_capacity_tokens: cap,
            inv_freq,
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
                cfg.rms_eps,
                w.w_dtype,
                w.w_group_size,
                w.q_proj.as_ptr() as *const std::ffi::c_void,
                w.k_proj.as_ptr() as *const std::ffi::c_void,
                w.v_proj.as_ptr() as *const std::ffi::c_void,
                w.o_proj.as_ptr() as *const std::ffi::c_void,
                w.gate_proj.as_ptr() as *const std::ffi::c_void,
                w.up_proj.as_ptr() as *const std::ffi::c_void,
                w.down_proj.as_ptr() as *const std::ffi::c_void,
                w.input_layernorm.as_ptr(),
                w.post_attention_layernorm.as_ptr(),
                bias_ptr(&w.q_bias),
                bias_ptr(&w.k_bias),
                bias_ptr(&w.v_bias),
                self.inv_freq.as_ptr(),
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
        // Wait only for the default (compute) stream, not the whole device, so an
        // in-flight weight upload on the copy stream keeps overlapping.
        synchronize_default()?;
        d_hidden.download(hidden)?;

        // Keep the orchestrator's length bookkeeping in step (real K/V is in VRAM).
        kv.append(&vec![0.0; kv_dim], &vec![0.0; kv_dim])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::evict_index;
    use std::collections::{HashSet, VecDeque};

    /// Simulate one forward pass's worth of layer accesses under a given eviction
    /// policy and report the misses.
    fn simulate(num_layers: u32, capacity: usize, cycles: u32, mru: bool) -> u32 {
        let mut order: VecDeque<u32> = VecDeque::new();
        let mut live: HashSet<u32> = HashSet::new();
        let mut misses = 0;
        for _ in 0..cycles {
            for layer in 0..num_layers {
                if live.contains(&layer) {
                    // hit: touch
                    if let Some(p) = order.iter().position(|&l| l == layer) {
                        order.remove(p);
                    }
                    order.push_back(layer);
                    continue;
                }
                misses += 1;
                while live.len() >= capacity {
                    let i = if mru {
                        match evict_index(&order, layer) {
                            Some(i) => i,
                            None => break,
                        }
                    } else {
                        0 // LRU: front of the queue
                    };
                    match order.remove(i) {
                        Some(v) => {
                            live.remove(&v);
                        }
                        None => break,
                    }
                }
                live.insert(layer);
                order.push_back(layer);
            }
        }
        misses
    }

    /// The layers of a forward pass are walked in a cycle, so LRU evicts exactly
    /// the layer needed next and every access misses — the window stops being a
    /// cache at all. MRU keeps the layers ahead of the cursor instead.
    ///
    /// Measured on a real 3B (23 of 28 resident): LRU gave 93 hits / 3463 misses
    /// and 1.47 tok/s; MRU gave 528 / 145 and 3.04.
    #[test]
    fn mru_beats_lru_on_the_cyclic_layer_scan() {
        let (layers, capacity, cycles) = (28u32, 23usize, 6u32);
        let lru = simulate(layers, capacity, cycles, false);
        let mru = simulate(layers, capacity, cycles, true);

        // LRU: every layer misses, every pass — a 0% hit rate by construction.
        assert_eq!(lru, layers * cycles, "LRU should miss everything on a cyclic scan");
        // MRU: only the layers that cannot fit miss.
        let steady = mru - layers; // discount the cold first pass
        let ceiling = (layers as usize - capacity + 1) as u32 * (cycles - 1);
        assert!(
            steady <= ceiling,
            "MRU steady-state misses {steady} should be <= {ceiling} (~N-capacity per pass)"
        );
        assert!(mru * 3 < lru, "MRU ({mru}) should miss far less than LRU ({lru})");
    }

    /// Never pick the entry being inserted, even if a concurrent load put it at
    /// the back — evicting it would drop the layer we are about to use.
    #[test]
    fn evict_index_skips_the_entry_being_inserted() {
        let order: VecDeque<u32> = [1u32, 2, 3].into_iter().collect();
        assert_eq!(evict_index(&order, 9), Some(2)); // back (3) = most recent
        assert_eq!(evict_index(&order, 3), Some(1)); // skip 3, take 2
        assert_eq!(evict_index(&VecDeque::new(), 0), None);
        let single: VecDeque<u32> = [7u32].into_iter().collect();
        assert_eq!(evict_index(&single, 7), None); // nothing else to evict
    }
}

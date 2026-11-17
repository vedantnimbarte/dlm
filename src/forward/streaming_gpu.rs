//! GPU layer-streaming compute kernel — the combined path
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
//! It also covers the two shapes the resident kernel can't: **routed MoE** (only
//! the top-k experts a token selects are materialized, cached per `(layer,
//! expert)`) and **MLA** (DeepSeek's compressed-latent attention), including the
//! combination — MLA + MoE is DeepSeek-V2/V3. Attention branches on the layer's
//! `mla` weights and the FFN on its `ffn`, independently, exactly as the CPU
//! oracle's `decode_block_streaming_moe` does.
//!
//! Validated on-device against the resident kernel by [`tests/gpu_parity.rs`]:
//! `streaming_gpu_matches_resident` drives real evictions, and
//! `streaming_gpu_soak_matches_resident` runs three batched sequences for 300
//! steps through a 3-of-8 window, checking every step (it catches a buffer
//! reused before the kernels reading it finish). `tests/gguf_model.rs`
//! (`gpu_streamed_and_resident_agree`) scores a real GGUF file streamed against
//! resident. CI has no GPU, so these run by hand: run the `cuda-kernels` suite
//! on a real card before tagging a release.

use crate::error::{DlmError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors, Weights};
use crate::forward::kernel::ComputeKernel;
use crate::forward::streaming::{evict_index, ewma, LayerSource, StreamStats};
use crate::gpu::device::{synchronize_default, CopyTimer, DeviceBuffer, Stream};
use crate::memory::PinnedBuffer;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

// The device entry points are declared once in `forward::gpu` and imported here.
// Restating the `extern` block let this path keep calling the old ABI after the
// kernel signature changed — a silent mismatch the compiler only warns about.
use crate::forward::gpu::{
    bias_ptr, dlm_apply_expert_rows, dlm_decode_block, dlm_decode_block_batched, dlm_dense_ffn,
    dlm_mla_attn, dlm_moe_attn, dlm_moe_attn_batched, dlm_moe_load_normed, dlm_moe_norm,
    dlm_moe_save_normed, dlm_moe_take_row, upload_bias, upload_weight, DlmBlockExt, GpuBlockBiases,
    GpuExpert, GpuGate, GpuLinear, GpuMla, GpuMoeCore, SlotKv, DLM_MAX_BATCH, W_O,
};

/// A layer's feed-forward weights in VRAM: a single dense MLP, or the MoE core
/// (router + shared expert) — the routed experts stream separately into the
/// per-`(layer, expert)` cache.
enum GpuFfn {
    Dense(GpuExpert),
    Moe(GpuMoeCore),
}

/// One layer's weight buffers, resident in VRAM (no KV — that lives in `GpuKv`).
struct GpuWeights {
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: DeviceBuffer,
    o_proj: DeviceBuffer,
    ffn: GpuFfn,
    input_layernorm: DeviceBuffer,
    post_attention_layernorm: DeviceBuffer,
    /// Attention biases (Qwen2 et al.). Uploaded synchronously — a few KB each,
    /// so they don't warrant a slot in the async staging layout.
    q_bias: Option<DeviceBuffer>,
    k_bias: Option<DeviceBuffer>,
    v_bias: Option<DeviceBuffer>,
    /// Qwen3 per-head Q/K RMSNorm weights (NULL to the kernel when absent).
    q_norm: Option<DeviceBuffer>,
    k_norm: Option<DeviceBuffer>,
    /// MLA (DeepSeek) attention projections; `None` for standard attention. When
    /// set, `q_proj`/`k_proj`/`v_proj` are unused dummies and attention runs
    /// through `dlm_mla_attn` instead.
    mla: Option<GpuMla>,
    /// Gemma2's extra FFN norm pair; `None` elsewhere.
    pre_ffn_norm: Option<DeviceBuffer>,
    post_ffn_norm: Option<DeviceBuffer>,
    /// LayerNorm, output-projection and MLP biases (GPT-2, Falcon).
    biases: GpuBlockBiases,
    /// Gated DeltaNet weights (a Qwen3.5/3.6 linear-attention layer); when set,
    /// `q/k/v_proj` are dummies and `o_proj` is the layer's `out_proj`.
    linear: Option<GpuLinear>,
    /// A Qwen3.5/3.6 full-attention layer's output gate.
    attn_gate: Option<GpuGate>,
    /// Native dtype of the attention projection weights (see `Weights::dtype_code`).
    w_dtype: i32,
    /// Group size for int4 weights; 0 for the float dtypes.
    w_group_size: i32,
    /// Each projection's own dtype, which a GGUF layer mixes.
    dtypes: crate::forward::gpu::WeightDtypes,
}

impl GpuWeights {
    /// The layer's nine weight tensors as `(native bytes, element count)`, in a
    /// fixed order (for staging/upload). The seven projections keep their native
    /// checkpoint dtype — they are staged and uploaded verbatim, so a bf16 layer
    /// puts half as many bytes across PCIe as an upsized f32 copy would. The two
    /// norms are f32 (a few KB; not worth a dtype path).
    fn tensors(t: &LayerTensors) -> Result<[(&[u8], usize); 9]> {
        use crate::forward::cpu::bytemuck_cast;
        let ffn = t.dense_ffn()?;
        Ok([
            (t.q_proj.as_bytes(), t.q_proj.len()),
            (t.k_proj.as_bytes(), t.k_proj.len()),
            (t.v_proj.as_bytes(), t.v_proj.len()),
            (t.o_proj.as_bytes(), t.o_proj.len()),
            (ffn.gate.as_bytes(), ffn.gate.len()),
            (ffn.up.as_bytes(), ffn.up.len()),
            (ffn.down.as_bytes(), ffn.down.len()),
            (bytemuck_cast(&t.input_layernorm), t.input_layernorm.len()),
            (
                bytemuck_cast(&t.post_attention_layernorm),
                t.post_attention_layernorm.len(),
            ),
        ])
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
    /// is free to reuse. Reports per-stage cost for `layer` when telemetry is on.
    ///
    /// The two phases below are already separate in this function, which is why
    /// they can be reported separately: staging is a host memcpy into pinned
    /// memory, the copy is the PCIe transfer. Conflating them would hide which
    /// of the two is actually the bottleneck.
    ///
    /// `spare` holds an evicted layer's weight buffers. A tensor whose size
    /// matches one is copied into it instead of a fresh allocation, so a
    /// steady-state miss allocates and frees nothing.
    fn upload_async_traced(
        t: &LayerTensors,
        stream: &Stream,
        staging: &mut PinnedBuffer,
        layer: Option<u32>,
        spare: Vec<DeviceBuffer>,
    ) -> Result<Self> {
        let layout = Self::stage_into(t, staging.as_mut_slice(), layer)?;
        Self::upload_staged(t, staging.as_ptr(), &layout, stream, layer, spare)
    }

    /// Copy the layer's nine staged tensors into `dst` back to back, returning
    /// each one's `(byte offset, byte length, element count)`. Reports the copy
    /// as `PinStage` for `layer` when telemetry is on.
    fn stage_into(
        t: &LayerTensors,
        dst: &mut [u8],
        layer: Option<u32>,
    ) -> Result<Vec<(usize, usize, usize)>> {
        let tensors = Self::tensors(t)?;
        let trace = layer.filter(|_| crate::telemetry::is_enabled());

        // Phase 1: stage all tensors into pinned memory, recording layout.
        // (byte offset, byte length, element count)
        let stage_timer = crate::telemetry::Timer::start();
        let mut layout: Vec<(usize, usize, usize)> = Vec::with_capacity(9);
        {
            let dst = &mut *dst;
            let mut byte_off = 0usize;
            for (src, len) in tensors {
                let bytes = src.len();
                dst[byte_off..byte_off + bytes].copy_from_slice(src);
                layout.push((byte_off, bytes, len));
                byte_off += bytes;
            }
        }
        if let Some(layer) = trace {
            let staged_bytes: u64 = layout.iter().map(|&(_, bytes, _)| bytes as u64).sum();
            crate::telemetry::emit(crate::telemetry::FlowEvent::new(
                layer,
                crate::telemetry::Stage::PinStage,
                staged_bytes,
                stage_timer.elapsed_us(),
            ));
        }
        Ok(layout)
    }

    /// Upload a layer whose nine tensors sit in pinned memory at `base`, laid out
    /// as `layout` (from [`stage_into`](Self::stage_into)). Everything else, the
    /// biases, norms and dtype tags, comes from `t`, which may be a
    /// [`LayerTensors::without_dense_weights`] skeleton.
    fn upload_staged(
        t: &LayerTensors,
        base: *const u8,
        layout: &[(usize, usize, usize)],
        stream: &Stream,
        layer: Option<u32>,
        mut spare: Vec<DeviceBuffer>,
    ) -> Result<Self> {
        let trace = layer.filter(|_| crate::telemetry::is_enabled());
        let staged_bytes: u64 = layout.iter().map(|&(_, bytes, _)| bytes as u64).sum();

        // Phase 2: allocate device buffers and enqueue async copies from staging.
        // Timed with device events rather than wall-clock: an async copy has
        // barely started when the enqueue returns.
        let copy_timer = match trace {
            Some(_) => CopyTimer::new().ok(),
            None => None,
        };
        if let Some(timer) = &copy_timer {
            timer.begin(stream)?;
        }
        let mut bufs: Vec<DeviceBuffer> = Vec::with_capacity(9);
        for &(off, bytes, len) in layout {
            let reuse = spare
                .iter()
                .position(|b| b.bytes() == bytes.max(1) && b.len() == len.max(1));
            let buf = match reuse {
                Some(i) => spare.swap_remove(i),
                None => DeviceBuffer::new_bytes(bytes, len.max(1))?,
            };
            let pinned = unsafe { base.add(off) } as *const std::ffi::c_void;
            buf.upload_async_bytes(pinned, bytes, stream)?;
            bufs.push(buf);
        }
        // Wait for the copies (staging becomes reusable; weights are ready).
        if let (Some(layer), Some(timer)) = (trace, &copy_timer) {
            // `end_us` synchronizes the end event, which is the same wait the
            // `stream.synchronize()` below performs.
            match timer.end_us(stream) {
                Ok(dur_us) => crate::telemetry::emit(crate::telemetry::FlowEvent::new(
                    layer,
                    crate::telemetry::Stage::H2d,
                    staged_bytes,
                    dur_us,
                )),
                // Event timing failed; report the transfer without claiming a
                // duration we did not measure.
                Err(_) => crate::telemetry::emit(
                    crate::telemetry::FlowEvent::new(
                        layer,
                        crate::telemetry::Stage::H2d,
                        staged_bytes,
                        0,
                    )
                    .estimated(),
                ),
            }
        }
        stream.synchronize()?;
        let mut it = bufs.into_iter();
        let q_proj = it.next().unwrap();
        let k_proj = it.next().unwrap();
        let v_proj = it.next().unwrap();
        let o_proj = it.next().unwrap();
        let gate = it.next().unwrap();
        let up = it.next().unwrap();
        let down = it.next().unwrap();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            ffn: GpuFfn::Dense(GpuExpert {
                gate,
                up,
                down,
                w_dtype: t.q_proj.dtype_code(),
                w_group_size: t.q_proj.group_size() as i32,
            }),
            input_layernorm: it.next().unwrap(),
            post_attention_layernorm: it.next().unwrap(),
            q_bias: upload_bias(t.q_bias.as_ref())?,
            k_bias: upload_bias(t.k_bias.as_ref())?,
            v_bias: upload_bias(t.v_bias.as_ref())?,
            q_norm: upload_bias(t.q_norm.as_ref())?,
            k_norm: upload_bias(t.k_norm.as_ref())?,
            // The staged fast path is standard-attention only; MLA layers take
            // `upload_sync` (their projection set doesn't fit this layout).
            mla: None,
            pre_ffn_norm: upload_bias(t.pre_feedforward_layernorm.as_ref())?,
            post_ffn_norm: upload_bias(t.post_feedforward_layernorm.as_ref())?,
            biases: GpuBlockBiases::upload(t)?,
            linear: None,
            attn_gate: None,
            w_dtype: t.q_proj.dtype_code(),
            w_group_size: t.q_proj.group_size() as i32,
            dtypes: crate::forward::gpu::WeightDtypes::of(t),
        })
    }

    /// The nine buffers [`upload_async_traced`](Self::upload_async_traced) fills,
    /// for reuse by the next upload. Other buffers (biases, norms that are not
    /// staged) are freed.
    fn into_staged_buffers(self) -> Vec<DeviceBuffer> {
        let mut v = vec![self.q_proj, self.k_proj, self.v_proj, self.o_proj];
        if let GpuFfn::Dense(f) = self.ffn {
            v.extend([f.gate, f.up, f.down]);
        }
        v.extend([self.input_layernorm, self.post_attention_layernorm]);
        v
    }

    /// Upload a layer **synchronously**, with plain per-tensor copies, for every
    /// layer shape the pinned-staging fast path can't express:
    ///
    /// * **MoE cores** — attention, norms, router, shared expert. The routed
    ///   experts are not here; they stream on demand into the per-`(layer,
    ///   expert)` cache.
    /// * **MLA layers** — a different projection set (`q/k/v` are empty and the
    ///   compressed `q_a/q_b/kv_a/kv_b` take their place), so the fixed 9-tensor
    ///   staging layout doesn't fit.
    ///
    /// ponytail: plain `from_bytes` uploads, not the pinned async staging the
    /// dense path uses, so these uploads do not overlap compute. Deliberately
    /// left until measured: the fix is to widen the staging layout to a variable
    /// tensor list, which is only worth it if MoE/MLA core-upload latency shows
    /// up in a profile on real hardware. Trigger: streamed MoE/MLA throughput
    /// materially below the dense path at the same layer size.
    fn upload_sync(t: &LayerTensors) -> Result<Self> {
        let ffn = match &t.ffn {
            crate::forward::Ffn::Moe { experts, .. } => {
                debug_assert!(
                    experts.is_empty(),
                    "GPU core must be loaded without routed experts"
                );
                GpuFfn::Moe(GpuMoeCore::upload(t)?)
            }
            crate::forward::Ffn::Dense(_) => GpuFfn::Dense(GpuExpert::upload(t.dense_ffn()?)?),
        };
        // MLA layers carry their attention dtype on the MLA weights, not q_proj
        // (which is empty for them) — same rule the resident `GpuLayer` uses.
        let (w_dtype, w_group_size) = match &t.mla {
            Some(mw) => (mw.q_b_proj.dtype_code(), mw.q_b_proj.group_size() as i32),
            None => (t.q_proj.dtype_code(), t.q_proj.group_size() as i32),
        };
        Ok(Self {
            q_proj: upload_weight(&t.q_proj)?,
            k_proj: upload_weight(&t.k_proj)?,
            v_proj: upload_weight(&t.v_proj)?,
            o_proj: upload_weight(&t.o_proj)?,
            ffn,
            input_layernorm: DeviceBuffer::from_slice(&t.input_layernorm)?,
            post_attention_layernorm: DeviceBuffer::from_slice(&t.post_attention_layernorm)?,
            q_bias: upload_bias(t.q_bias.as_ref())?,
            k_bias: upload_bias(t.k_bias.as_ref())?,
            v_bias: upload_bias(t.v_bias.as_ref())?,
            q_norm: upload_bias(t.q_norm.as_ref())?,
            k_norm: upload_bias(t.k_norm.as_ref())?,
            mla: t.mla.as_ref().map(GpuMla::upload).transpose()?,
            pre_ffn_norm: upload_bias(t.pre_feedforward_layernorm.as_ref())?,
            post_ffn_norm: upload_bias(t.post_feedforward_layernorm.as_ref())?,
            biases: GpuBlockBiases::upload(t)?,
            linear: t.linear.as_ref().map(GpuLinear::upload).transpose()?,
            attn_gate: GpuGate::upload(t.attn_gate.as_ref())?,
            w_dtype,
            w_group_size,
            dtypes: crate::forward::gpu::WeightDtypes::of(t),
        })
    }
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
    ///
    /// A layer loaded ahead of the pass (`ahead`) goes to the other end, where
    /// it is evicted last: until the pass reaches it, it is needed sooner than
    /// anything the pass has already used.
    fn insert(&mut self, layer: u32, weights: Arc<GpuWeights>, ahead: bool) {
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
        if ahead {
            self.order.push_front(layer);
        } else {
            self.order.push_back(layer);
        }
    }

    /// Evict, ahead of loading `layer`, the entry [`insert`](Self::insert)
    /// would evict for it (most recently used, not `layer`), provided no compute
    /// call is holding it and `keep` does not claim it, and hand back its
    /// weights so their buffers can take the new layer. `None` while the window
    /// still has room.
    ///
    /// Evicting before the upload rather than after also means the window never
    /// holds `capacity + 1` layers mid-load.
    fn take_idle_victim(&mut self, layer: u32, keep: impl Fn(u32) -> bool) -> Option<GpuWeights> {
        if self.map.len() < self.capacity {
            return None;
        }
        // Every clone of a cached `Arc` is taken under this lock, so a count of
        // one here cannot rise before the entry is removed.
        let i = (0..self.order.len()).rev().find(|&i| {
            let l = self.order[i];
            l != layer && !keep(l) && self.map.get(&l).is_some_and(|w| Arc::strong_count(w) == 1)
        })?;
        let evicted = self.order.remove(i)?;
        let w = self.map.remove(&evicted)?;
        self.stats.evictions += 1;
        Arc::try_unwrap(w).ok()
    }
}

/// A dense layer staged once into its own pinned buffer, so later uploads copy
/// to the device straight from it. `skeleton` holds everything else the upload
/// needs.
struct PinnedLayer {
    buf: PinnedBuffer,
    layout: Vec<(usize, usize, usize)>,
    skeleton: LayerTensors,
}

/// Byte-budgeted LRU of [`PinnedLayer`]s: the host tier of GPU streaming for
/// dense models.
///
/// Staging a layer into pinned memory was a full memcpy of its weights on every
/// upload, the larger half of a streamed step. With the layer kept pinned, a hit
/// goes to the device without touching the CPU. It stands in for the host RAM
/// cache on this path rather than sitting beside it, so host memory does not
/// double.
#[derive(Default)]
struct PinnedLru {
    map: HashMap<u32, Arc<PinnedLayer>>,
    /// Least recently used first.
    order: VecDeque<u32>,
    bytes: usize,
    budget: usize,
}

/// A dense layer ready to upload: staged in the pinned cache, or (when it cannot
/// be cached) as host tensors for the shared staging buffer.
enum HostLayer {
    Pinned(Arc<PinnedLayer>),
    Host(Arc<LayerTensors>),
}

/// Passes through a layer over which an expert's routing count halves. inferssd
/// runs DeepSeek decode with a half-life of 2 tokens (`count_decay2`).
const EXPERT_HALF_LIFE: f64 = 2.0;

/// Count-decay routing scores per `(layer, expert)`: each selection adds 1 and
/// the total halves every [`EXPERT_HALF_LIFE`] passes through the layer, so a
/// score tracks what the layer routes to *now*. It drives eviction (lowest score
/// leaves the VRAM cache) and prefetch (highest scores are warmed). LRU would
/// evict an expert picked every other token for one picked once just now.
///
/// `hints` holds each layer's experts as its own router predicts them from the
/// layer before (see `StreamingGpuKernel::hint_next`): exact where `hot` guesses.
#[derive(Default)]
struct ExpertScores {
    /// Passes through each layer so far: the clock its scores decay on.
    clock: HashMap<u32, u64>,
    /// `(score, clock when it was set)`.
    scores: HashMap<(u32, u32), (f64, u64)>,
    hints: HashMap<u32, Vec<u32>>,
}

impl ExpertScores {
    /// One pass through `layer` (a decode token, or a whole prefill).
    fn tick(&mut self, layer: u32) {
        *self.clock.entry(layer).or_insert(0) += 1;
    }

    /// Record that `expert` was routed to in `layer` for one token.
    fn record(&mut self, layer: u32, expert: u32) {
        let now = self.clock.get(&layer).copied().unwrap_or(0);
        let s = self.score((layer, expert));
        self.scores.insert((layer, expert), (s + 1.0, now));
    }

    /// The decayed score of `(layer, expert)`; 0 if never routed to.
    fn score(&self, key: (u32, u32)) -> f64 {
        let now = self.clock.get(&key.0).copied().unwrap_or(0);
        self.scores.get(&key).map_or(0.0, |&(s, at)| {
            s * (-((now - at) as f64) / EXPERT_HALF_LIFE).exp2()
        })
    }

    /// The experts to warm for `layer`: its router's prediction when there is
    /// one, else the `n` highest-scoring (ties on expert id, for determinism).
    fn hot(&self, layer: u32, n: usize) -> Vec<u32> {
        if let Some(h) = self.hints.get(&layer) {
            return h.clone();
        }
        let mut v: Vec<(u32, f64)> = self
            .scores
            .keys()
            .filter(|(l, _)| *l == layer)
            .map(|&(l, e)| (e, self.score((l, e))))
            .collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().take(n).map(|(e, _)| e).collect()
    }
}

/// A bounded VRAM cache of routed MoE experts, keyed `(layer, expert)`.
///
/// Evicts the lowest [`ExpertScores`] score, preferring another layer's expert
/// to one of the layer being filled, so warming a layer's predicted experts
/// never evicts the others just warmed for it. A miss streams the expert from
/// the checkpoint; the prefetch worker warms each layer's predicted experts
/// ahead of compute.
struct GpuExpertCache {
    capacity: usize,
    map: HashMap<(u32, u32), Arc<GpuExpert>>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl GpuExpertCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// The entry to evict for `key`: the lowest score, another layer's expert
    /// before one of `key`'s own layer (so warming a layer's predicted experts
    /// never evicts the others just warmed for it), and with `idle` only one no
    /// compute call holds.
    ///
    /// ponytail: O(capacity) scan per eviction; a heap if caches reach the tens
    /// of thousands.
    fn victim(&self, key: (u32, u32), scores: &ExpertScores, idle: bool) -> Option<(u32, u32)> {
        self.map
            .iter()
            .filter(|(k, e)| **k != key && (!idle || Arc::strong_count(e) == 1))
            .map(|(k, _)| *k)
            .min_by(|a, b| {
                (a.0 == key.0)
                    .cmp(&(b.0 == key.0))
                    .then(scores.score(*a).total_cmp(&scores.score(*b)))
            })
    }

    /// When full, evict the entry `insert(key)` would and hand back its buffers
    /// for the new expert to load into, provided nothing holds it. Evicting
    /// before the load also keeps the cache from holding `capacity + 1`.
    fn take_idle_victim(&mut self, key: (u32, u32), scores: &ExpertScores) -> Option<GpuExpert> {
        if self.map.len() < self.capacity || self.map.contains_key(&key) {
            return None;
        }
        let victim = self.victim(key, scores, true)?;
        self.evictions += 1;
        Arc::try_unwrap(self.map.remove(&victim)?).ok()
    }

    fn insert(&mut self, key: (u32, u32), expert: Arc<GpuExpert>, scores: &ExpertScores) {
        while self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            let Some(victim) = self.victim(key, scores, false) else {
                break;
            };
            self.map.remove(&victim);
            self.evictions += 1;
        }
        self.map.insert(key, expert);
    }
}

/// A MoE layer's router weights, post-attention norm and score bias, on the
/// host.
type HostRouter = (Weights, Vec<f32>, Option<Vec<f32>>);

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
    /// Dense layers kept staged in pinned memory (budget 0 disables it).
    pinned: Mutex<PinnedLru>,
    /// Page-locked staging for routed-expert uploads, grown to the largest.
    expert_staging: Mutex<PinnedBuffer>,
    /// VRAM cache of routed MoE experts (empty/unused for dense models).
    experts: Mutex<GpuExpertCache>,
    /// Count-decay routing scores and router predictions, driving expert
    /// eviction and prefetch (unused for dense models).
    scores: Mutex<ExpertScores>,
    /// Each MoE layer's router and post-attention norm on the host, kept as its
    /// core streams in, so the layer before can predict its experts.
    routers: Mutex<HashMap<u32, Arc<HostRouter>>>,
    /// EWMA ns to serve a decode expert miss by uploading it (GPU) or by
    /// computing it on the CPU; the cheaper one serves the next miss.
    miss_gpu_ns: AtomicU64,
    miss_cpu_ns: AtomicU64,
    misses_seen: AtomicU64,
}

impl<S: LayerSource> GpuShared<S> {
    /// Ensure `layer`'s weights are VRAM-resident, loading (host) + uploading
    /// (VRAM) without holding the cache lock if absent. Same-layer callers
    /// de-dupe on a `loading` set + condvar. `by_worker` tags a prefetch;
    /// `ahead_of: Some(next)` is a dense prefetch for a pass about to run layer
    /// `next`, which only evicts a layer the pass needs after `layer`, and
    /// skips the load (`None`) when the window holds nothing it may evict.
    fn ensure(
        &self,
        layer: u32,
        by_worker: bool,
        ahead_of: Option<u32>,
    ) -> Result<Option<Arc<GpuWeights>>> {
        let mut cache = self.weights.lock().unwrap();
        loop {
            if let Some(w) = cache.map.get(&layer) {
                let w = Arc::clone(w);
                if ahead_of.is_none() {
                    cache.touch(layer);
                }
                return Ok(Some(w));
            }
            if cache.loading.contains(&layer) {
                cache = self.ready.wait(cache).unwrap();
                continue;
            }
            let dense = self.cfg.moe.is_none() && self.cfg.mla.is_none();
            let n = self.source.num_layers();
            let victim = if dense {
                // How far the pass is from needing `l`: a prefetch keeps every
                // layer needed before the one it loads.
                let until = |next: u32, l: u32| (l + n - next) % n;
                let keep =
                    |l: u32| ahead_of.is_some_and(|next| until(next, l) <= until(next, layer));
                cache.take_idle_victim(layer, keep)
            } else {
                None
            };
            if ahead_of.is_some() && victim.is_none() && cache.map.len() >= cache.capacity {
                return Ok(None);
            }
            cache.loading.insert(layer);
            drop(cache);
            // Expensive part — host load (disk + dequant) then VRAM upload — runs
            // with the cache lock released, so it overlaps the compute thread. The
            // dense path stages through pinned memory on the non-blocking copy
            // stream so the PCIe transfer overlaps the default-stream kernel; the
            // MoE path uploads just the layer *core* (attention + router + shared),
            // its routed experts streaming separately per `(layer, expert)`. MLA
            // layers take the sync path either way — their projection set doesn't
            // fit the staged dense layout.
            let uploaded = if self.cfg.moe.is_some() {
                self.source.load_layer_core(layer).and_then(|t| {
                    self.keep_router(layer, &t);
                    GpuWeights::upload_sync(&t).map(Arc::new)
                })
            } else if self.cfg.mla.is_some() {
                self.source
                    .load_layer(layer)
                    .and_then(|t| GpuWeights::upload_sync(&t).map(Arc::new))
            } else {
                self.host_layer(layer).and_then(|host| {
                    // No compute call holds the victim any more, but the kernels
                    // its last one launched may still be queued on the default
                    // stream, reading these buffers. Let them finish before the
                    // copy stream overwrites the weights. (Freeing the buffers
                    // used to wait the same way, inside cudaFree.)
                    if victim.is_some() {
                        synchronize_default()?;
                    }
                    let spare = victim.map_or_else(Vec::new, GpuWeights::into_staged_buffers);
                    match host {
                        HostLayer::Pinned(p) => GpuWeights::upload_staged(
                            &p.skeleton,
                            p.buf.as_ptr(),
                            &p.layout,
                            &self.copy_stream,
                            Some(layer),
                            spare,
                        ),
                        HostLayer::Host(t) => {
                            let mut staging = self.staging.lock().unwrap();
                            GpuWeights::upload_async_traced(
                                &t,
                                &self.copy_stream,
                                &mut staging,
                                Some(layer),
                                spare,
                            )
                        }
                    }
                    .map(Arc::new)
                })
            };
            let mut cache = self.weights.lock().unwrap();
            cache.loading.remove(&layer);
            self.ready.notify_all();
            let w = uploaded?;
            cache.insert(layer, Arc::clone(&w), ahead_of.is_some());
            if by_worker {
                cache.stats.prefetched += 1;
            }
            return Ok(Some(w));
        }
    }

    /// Dense layer `layer` for upload: from the pinned cache, staging it there on
    /// a miss, or as plain host tensors when the cache is off or cannot fit it
    /// (including when pinning the memory fails).
    fn host_layer(&self, layer: u32) -> Result<HostLayer> {
        {
            let mut lru = self.pinned.lock().unwrap();
            if let Some(p) = lru.map.get(&layer).cloned() {
                if let Some(i) = lru.order.iter().position(|&l| l == layer) {
                    lru.order.remove(i);
                }
                lru.order.push_back(layer);
                if crate::telemetry::is_enabled() {
                    crate::telemetry::emit(crate::telemetry::FlowEvent::new(
                        layer,
                        crate::telemetry::Stage::RamHit,
                        p.buf.len() as u64,
                        0,
                    ));
                }
                return Ok(HostLayer::Pinned(p));
            }
        }
        let t = self.source.load_layer(layer)?;
        let (Some(skeleton), Ok(tensors)) = (t.without_dense_weights(), GpuWeights::tensors(&t))
        else {
            return Ok(HostLayer::Host(t));
        };
        let bytes: usize = tensors.iter().map(|(b, _)| b.len()).sum();

        // Make room, keeping an evicted buffer big enough to stage into.
        let mut reuse = None;
        {
            let mut lru = self.pinned.lock().unwrap();
            if bytes > lru.budget {
                return Ok(HostLayer::Host(t));
            }
            // Most recently used first: the pinned tier is read in the same
            // cycle as the window, where LRU never hits (see `evict_index`).
            while lru.bytes + bytes > lru.budget {
                let Some(old) = lru.order.pop_back() else {
                    break;
                };
                if let Some(p) = lru.map.remove(&old) {
                    lru.bytes -= p.buf.len();
                    // Still uploading elsewhere: it frees when that finishes.
                    if let Ok(p) = Arc::try_unwrap(p) {
                        if reuse.is_none() && p.buf.capacity() >= bytes {
                            reuse = Some(p.buf);
                        }
                    }
                }
            }
        }
        let mut buf = match reuse {
            Some(b) => b,
            None => match PinnedBuffer::with_len(bytes) {
                Ok(b) => b,
                Err(_) => return Ok(HostLayer::Host(t)),
            },
        };
        let layout = GpuWeights::stage_into(&t, buf.as_mut_slice(), Some(layer))?;
        let p = Arc::new(PinnedLayer {
            buf,
            layout,
            skeleton,
        });
        let mut lru = self.pinned.lock().unwrap();
        if !lru.map.contains_key(&layer) {
            lru.bytes += p.buf.len();
            lru.map.insert(layer, Arc::clone(&p));
            lru.order.push_back(layer);
        }
        Ok(HostLayer::Pinned(p))
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
        self.ensure(layer, false, None)
            .map(|w| w.expect("a compute fetch always loads"))
    }

    /// Prefetch for a pass about to run layer `next`.
    ///
    /// A dense model loads the first layer from `next` on that is neither
    /// resident nor loading, rather than `next` itself: with part of the model
    /// resident, `next` usually is, and the copy engine sat idle while the pass
    /// computed it. Loading the next missing layer instead keeps the copy
    /// running under the resident layers' compute. MoE and MLA layers keep
    /// one layer ahead.
    fn prefetch(&self, next: u32) {
        if self.cfg.moe.is_some() || self.cfg.mla.is_some() {
            let _ = self.ensure(next, true, None);
            return;
        }
        let n = self.source.num_layers();
        let target = {
            let cache = self.weights.lock().unwrap();
            (0..n)
                .map(|k| (next + k) % n)
                .find(|l| !cache.map.contains_key(l) && !cache.loading.contains(l))
        };
        if let Some(layer) = target {
            let _ = self.ensure(layer, true, Some(next)); // best-effort
        }
    }

    /// Routed expert `(layer, expert)` if VRAM-resident. Records hit/miss.
    fn cached_expert(&self, layer: u32, expert: u32) -> Option<Arc<GpuExpert>> {
        let mut cache = self.experts.lock().unwrap();
        let e = cache.map.get(&(layer, expert)).cloned();
        if e.is_some() {
            cache.hits += 1;
        } else {
            cache.misses += 1;
        }
        e
    }

    /// Stream `(layer, expert)` from the checkpoint into the VRAM cache. Runs
    /// without the cache lock; a rare concurrent double-load just uploads twice.
    fn upload_expert(&self, layer: u32, expert: u32) -> Result<Arc<GpuExpert>> {
        let host = self.source.load_expert(layer, expert)?;
        let spare = {
            let scores = self.scores.lock().unwrap();
            self.experts
                .lock()
                .unwrap()
                .take_idle_victim((layer, expert), &scores)
        };
        if spare.is_some() {
            // No compute call holds the victim, but kernels its last one queued
            // may still be reading it.
            synchronize_default()?;
        }
        let gpu = {
            let mut staging = self.expert_staging.lock().unwrap();
            Arc::new(GpuExpert::upload_staged(
                &host,
                &mut staging,
                &self.copy_stream,
                spare,
            )?)
        };
        let scores = self.scores.lock().unwrap();
        self.experts
            .lock()
            .unwrap()
            .insert((layer, expert), Arc::clone(&gpu), &scores);
        Ok(gpu)
    }

    /// Ensure routed expert `(layer, expert)` is VRAM-resident.
    fn ensure_expert(&self, layer: u32, expert: u32) -> Result<Arc<GpuExpert>> {
        match self.cached_expert(layer, expert) {
            Some(e) => Ok(e),
            None => self.upload_expert(layer, expert),
        }
    }

    /// Whether the next decode expert miss should run on the CPU: whichever of
    /// CPU and upload has measured cheaper, each tried once first, and the
    /// dearer one retried every 32nd miss so a cold first sample can't stick.
    fn miss_on_cpu(&self) -> bool {
        let (cpu, gpu) = (
            self.miss_cpu_ns.load(Ordering::Relaxed),
            self.miss_gpu_ns.load(Ordering::Relaxed),
        );
        let explore = self.misses_seen.fetch_add(1, Ordering::Relaxed) % 32 == 31;
        if cpu == 0 || gpu == 0 {
            return cpu == 0;
        }
        (cpu <= gpu) != explore
    }

    /// Keep `layer`'s router and post-attention norm on the host (MoE only).
    fn keep_router(&self, layer: u32, t: &LayerTensors) {
        if let Ok((router, ..)) = t.moe() {
            self.routers
                .lock()
                .unwrap()
                .entry(layer)
                .or_insert_with(|| {
                    Arc::new((
                        router.clone(),
                        t.post_attention_layernorm.clone(),
                        t.router_bias().map(<[f32]>::to_vec),
                    ))
                });
        }
    }

    /// Warm `layer`'s predicted experts into the VRAM cache, off the compute
    /// critical path: its router's prediction when the layer before made one,
    /// else its `experts_per_tok` highest-scoring. Best-effort (errors ignored)
    /// and a no-op for dense models and unseen layers.
    fn prefetch_experts(&self, layer: u32) {
        let Some(m) = self.cfg.moe else { return };
        let hot = self
            .scores
            .lock()
            .unwrap()
            .hot(layer, m.experts_per_tok as usize);
        for e in hot {
            let _ = self.ensure_expert(layer, e);
        }
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
    /// Paged KV blocks shared by every session, one pool per layer.
    kv_pools: crate::forward::kv_pool::KvPools,
    /// RoPE inverse frequencies (see [`GpuKernel`](crate::forward::GpuKernel)):
    /// computed once by the shared host function, resident for the kernel's life.
    inv_freq: DeviceBuffer,
    /// Gemma3's local-layer frequencies; `None` when every layer shares `inv_freq`.
    local_inv_freq: Option<DeviceBuffer>,
    /// MLA decoupled-RoPE frequencies (over `qk_rope_head_dim`); `None` for
    /// standard attention.
    mla_inv_freq: Option<DeviceBuffer>,
    /// Persistent device hidden buffer, chained across the streamed layer stack:
    /// uploaded once before layer 0 and downloaded once after the last layer, so
    /// only weights (not the hidden vector) cross the bus per layer. Safe because
    /// the scheduler drives one sequence's full stack per step (see `GpuKernel`).
    d_hidden: DeviceBuffer,
    /// `[DLM_MAX_BATCH, hidden]` block `decode_batch` keeps a batch in across
    /// the layer stack.
    d_batch: DeviceBuffer,
    /// One token's MoE FFN input, brought to the host for CPU experts.
    d_normed: DeviceBuffer,
    prefetch_tx: Option<Sender<u32>>,
    worker: Option<JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
}

impl<S: LayerSource + 'static> StreamingGpuKernel<S> {
    /// Size the shared KV pools to hold `tokens` tokens across all sessions (per
    /// layer). Call before the first session runs: the pools are allocated then.
    pub fn with_kv_pool_tokens(mut self, tokens: usize) -> Self {
        self.set_kv_pool_tokens(tokens);
        self
    }

    /// One `dlm_decode_block_batched` call for dense layer `layer` (weights `w`)
    /// over the `[slots, hidden]` device block at `x`.
    fn launch_dense_batched(
        &self,
        layer: u32,
        w: &GpuWeights,
        x: *mut f32,
        slots: &SlotKv,
    ) -> Result<()> {
        let GpuFfn::Dense(f) = &w.ffn else {
            return Err(DlmError::InvalidConfig(
                "batched block on a layer without a dense FFN".into(),
            ));
        };
        let lcfg = self.shared.cfg.for_layer(layer);
        // SAFETY: live device buffers of the sizes the kernel expects; every slot's
        // KV has a block for the row it writes (`device_kv` ensured it).
        let code = unsafe {
            dlm_decode_block_batched(
                lcfg.hidden_size as i32,
                lcfg.q_dim() as i32,
                lcfg.kv_dim() as i32,
                lcfg.num_heads as i32,
                lcfg.num_kv_heads as i32,
                lcfg.head_dim as i32,
                lcfg.intermediate_size as i32,
                lcfg.rms_eps,
                w.w_dtype,
                w.w_group_size,
                w.q_proj.as_ptr() as *const std::ffi::c_void,
                w.k_proj.as_ptr() as *const std::ffi::c_void,
                w.v_proj.as_ptr() as *const std::ffi::c_void,
                w.o_proj.as_ptr() as *const std::ffi::c_void,
                f.gate.as_ptr() as *const std::ffi::c_void,
                f.up.as_ptr() as *const std::ffi::c_void,
                f.down.as_ptr() as *const std::ffi::c_void,
                w.input_layernorm.as_ptr(),
                w.post_attention_layernorm.as_ptr(),
                bias_ptr(&w.q_bias),
                bias_ptr(&w.k_bias),
                bias_ptr(&w.v_bias),
                bias_ptr(&w.q_norm),
                bias_ptr(&w.k_norm),
                self.inv_freq_for(layer),
                x,
                &slots.keys,
                &slots.values,
                &slots.tables,
                &slots.num_positions,
                &slots.positions,
                slots.len() as i32,
                lcfg.sliding_window.unwrap_or(0) as i32,
                lcfg.activation.code(),
                crate::forward::cpu::rope_mscale(lcfg.rope_scaling),
                lcfg.attn_scale(),
                lcfg.attn_logit_softcap.unwrap_or(0.0),
                bias_ptr(&w.pre_ffn_norm),
                bias_ptr(&w.post_ffn_norm),
                &DlmBlockExt::new(&lcfg, &w.biases, slots.elem, std::ptr::null())
                    .with_weight_dtypes(&w.dtypes),
            )
        };
        if code != 0 {
            return Err(DlmError::Gpu {
                api: "dlm_decode_block_batched",
                code,
            });
        }
        Ok(())
    }

    /// Attention for a MoE layer (standard attention) over `slots.len()` rows
    /// at `x`, each projection read once for all of them, then each row's FFN
    /// input saved to the same row of `norm` for
    /// [`moe_ffn_grouped`](Self::moe_ffn_grouped). Token by token, every row
    /// read every attention matrix again.
    fn moe_attn_batched(
        &self,
        layer: u32,
        w: &GpuWeights,
        x: *mut f32,
        slots: &SlotKv,
        norm: *mut f32,
    ) -> Result<()> {
        let cfg = self.shared.cfg.for_layer(layer);
        let h = cfg.hidden_size;
        // SAFETY: live device buffers of the sizes the kernel expects; `x`
        // holds `slots.len()` rows and every slot's KV has a block for the row
        // it writes (`device_kv` ensured it).
        let code = unsafe {
            dlm_moe_attn_batched(
                h as i32,
                cfg.q_dim() as i32,
                cfg.kv_dim() as i32,
                cfg.num_heads as i32,
                cfg.num_kv_heads as i32,
                cfg.head_dim as i32,
                cfg.rms_eps,
                w.w_dtype,
                w.w_group_size,
                w.q_proj.as_ptr() as *const std::ffi::c_void,
                w.k_proj.as_ptr() as *const std::ffi::c_void,
                w.v_proj.as_ptr() as *const std::ffi::c_void,
                w.o_proj.as_ptr() as *const std::ffi::c_void,
                w.input_layernorm.as_ptr(),
                w.post_attention_layernorm.as_ptr(),
                bias_ptr(&w.q_bias),
                bias_ptr(&w.k_bias),
                bias_ptr(&w.v_bias),
                bias_ptr(&w.q_norm),
                bias_ptr(&w.k_norm),
                self.inv_freq_for(layer),
                x,
                &slots.keys,
                &slots.values,
                &slots.tables,
                &slots.num_positions,
                &slots.positions,
                slots.len() as i32,
                cfg.sliding_window.unwrap_or(0) as i32,
                crate::forward::cpu::rope_mscale(cfg.rope_scaling),
                cfg.attn_scale(),
                cfg.attn_logit_softcap.unwrap_or(0.0),
                &DlmBlockExt::new(&cfg, &w.biases, slots.elem, std::ptr::null())
                    .with_weight_dtypes(&w.dtypes)
                    .with_attn_gate(&w.attn_gate),
            )
        };
        gpu_ok("dlm_moe_attn_batched", code)?;
        for b in 0..slots.len() {
            // SAFETY: row `b` of the scratch the batched call filled, and of
            // `norm`'s rows.
            gpu_ok("dlm_moe_take_row", unsafe {
                dlm_moe_take_row(h as i32, b as i32)
            })?;
            gpu_ok("dlm_moe_save_normed", unsafe {
                dlm_moe_save_normed(h as i32, norm.add(b * h))
            })?;
        }
        Ok(())
    }

    /// A linear-attention layer's token mixer for `states.len()` rows at `x`,
    /// row `r` against state `states[r]`, in row order, each row's FFN input
    /// saved to the same row of `norm` for
    /// [`moe_ffn_grouped`](Self::moe_ffn_grouped).
    fn linear_rows(
        &self,
        w: &GpuWeights,
        x: *mut f32,
        states: &[(*mut f32, *mut f32)],
        norm: *mut f32,
    ) -> Result<()> {
        let lin = w.linear.as_ref().expect("a linear-attention layer");
        let h = self.shared.cfg.hidden_size;
        lin.run_rows(
            &self.shared.cfg,
            (&w.input_layernorm, &w.post_attention_layernorm),
            (&w.o_proj, w.dtypes.dtype[W_O], w.dtypes.group[W_O]),
            x,
            states,
        )?;
        for b in 0..states.len() {
            // SAFETY: row `b` of the scratch the call filled, and of `norm`.
            gpu_ok("dlm_moe_take_row", unsafe {
                dlm_moe_take_row(h as i32, b as i32)
            })?;
            gpu_ok("dlm_moe_save_normed", unsafe {
                dlm_moe_save_normed(h as i32, norm.add(b * h))
            })?;
        }
        Ok(())
    }

    /// Keep up to `bytes` of dense layers staged in pinned host memory, so
    /// uploading one copies straight to the device instead of first copying its
    /// weights into the staging buffer. Meant to take the place of a host RAM
    /// cache on the source. MoE and MLA layers never use it.
    pub fn with_pinned_layer_cache(self, bytes: usize) -> Self {
        self.shared.pinned.lock().unwrap().budget = bytes;
        self
    }

    /// The RoPE inverse frequencies `layer` rotates with.
    fn inv_freq_for(&self, layer: u32) -> *const f32 {
        match &self.local_inv_freq {
            Some(local) if self.shared.cfg.uses_local_rope(layer) => local.as_ptr(),
            _ => self.inv_freq.as_ptr(),
        }
    }

    /// Build a streaming GPU kernel: allocate per-layer KV for up to
    /// `max_kv_tokens` positions and stream weights keeping at most
    /// `resident_layers` sets in VRAM, prefetching one layer ahead.
    /// `expert_cache_capacity` bounds how many routed experts stay VRAM-resident.
    /// Pass `Some(n)` from a VRAM budget (`--expert-cache-gb`, see
    /// [`ModelConfig::expert_cache_capacity`](crate::model::ModelConfig::expert_cache_capacity))
    /// so a fine-grained (128-expert) model can't OOM the card; `None` falls back
    /// to a per-window count heuristic for tests and dense models.
    pub fn new(
        cfg: BlockConfig,
        source: S,
        max_kv_tokens: usize,
        resident_layers: usize,
        expert_cache_capacity: Option<usize>,
    ) -> Result<Self> {
        let num_layers = source.num_layers();
        let cap = max_kv_tokens.max(1);
        // KV is per-session (owned by each sequence's KvLayerCache), so the kernel
        // allocates none here — batched sessions keep isolated history.
        // Page-locked staging sized to one layer's weights (all layers same size).
        // MLA layers never use the staged path (`upload_sync` instead), so they
        // need no real staging buffer — and `f32_count`'s dense layout doesn't
        // describe them anyway. One byte (rounded to a page) rather than zero:
        // `PinnedBuffer::with_len(0)` is an error, and a spare page is cheaper
        // than making the field optional for a buffer nothing reads.
        // MoE cores take `upload_sync` as well, and a hybrid MoE config's
        // defaulted `intermediate_size` would size a large buffer for nothing.
        let staging_bytes = if cfg.mla.is_some() || cfg.moe.is_some() {
            1
        } else {
            GpuWeights::f32_count(&cfg) * std::mem::size_of::<f32>()
        };
        // Expert cache capacity: prefer the VRAM-budget-derived count the caller
        // passes; else fall back to a per-window count heuristic (tests, dense).
        let expert_capacity = expert_cache_capacity.unwrap_or_else(|| {
            cfg.moe.map_or(1, |m| {
                let per_tok = m.experts_per_tok as usize;
                (per_tok * resident_layers.max(1) * 4).clamp(
                    per_tok.max(1),
                    m.num_experts as usize * resident_layers.max(1),
                )
            })
        });
        let shared = Arc::new(GpuShared {
            cfg,
            source,
            weights: Mutex::new(WeightLru::new(resident_layers)),
            ready: Condvar::new(),
            copy_stream: Stream::new_nonblocking()?,
            staging: Mutex::new(PinnedBuffer::with_len(staging_bytes)?),
            pinned: Mutex::new(PinnedLru::default()),
            expert_staging: Mutex::new(PinnedBuffer::with_len(1)?),
            experts: Mutex::new(GpuExpertCache::new(expert_capacity)),
            scores: Mutex::new(ExpertScores::default()),
            routers: Mutex::new(HashMap::new()),
            miss_gpu_ns: AtomicU64::new(0),
            miss_cpu_ns: AtomicU64::new(0),
            misses_seen: AtomicU64::new(0),
        });
        let (tx, rx) = channel::<u32>();
        let stopped = Arc::new(AtomicBool::new(false));
        let worker = {
            let shared = Arc::clone(&shared);
            let stopped = Arc::clone(&stopped);
            std::thread::spawn(move || {
                while let Ok(mut layer) = rx.recv() {
                    // Only where the pass is now matters: skip to the latest.
                    while let Ok(later) = rx.try_recv() {
                        layer = later;
                    }
                    if stopped.load(Ordering::Relaxed) {
                        break;
                    }
                    shared.prefetch(layer);
                    // Warm this layer's likely experts while it's still ahead of
                    // compute; no-op for dense models and cold (unseen) layers.
                    shared.prefetch_experts(layer);
                }
            })
        };
        let inv_freq = DeviceBuffer::from_slice(&cfg.rope_freqs(cfg.rope_theta, cfg.rope_scaling))?;
        // Gemma3's windowed layers rotate with their own base, unscaled.
        let local_inv_freq = cfg
            .rope_local_theta
            .map(|theta| DeviceBuffer::from_slice(&cfg.rope_freqs(theta, None)))
            .transpose()?;
        // MLA rotates only its decoupled qk_rope sub-dimension (same rule as the
        // resident `GpuKernel`).
        let mla_inv_freq = match &cfg.mla {
            Some(m) => Some(DeviceBuffer::from_slice(
                &crate::forward::cpu::rope_inv_freqs(
                    m.qk_rope_head_dim as usize,
                    cfg.rope_theta,
                    cfg.rope_scaling,
                ),
            )?),
            None => None,
        };
        let d_hidden = DeviceBuffer::new(cfg.hidden_size)?;
        let d_batch = DeviceBuffer::new(DLM_MAX_BATCH * cfg.hidden_size)?;
        let d_normed = DeviceBuffer::new(cfg.hidden_size)?;
        Ok(Self {
            // One session's worth by default; `with_kv_pool_tokens` sizes it for many.
            kv_pools: crate::forward::kv_pool::KvPools::new(
                num_layers as usize,
                cfg.kv_dim(),
                cfg.head_dim,
                cap,
            ),
            shared,
            num_layers,
            kv_capacity_tokens: cap,
            inv_freq,
            local_inv_freq,
            mla_inv_freq,
            d_hidden,
            d_batch,
            d_normed,
            prefetch_tx: Some(tx),
            worker: Some(worker),
            stopped,
        })
    }

    /// Streaming cache stats: the layer-window counters plus the routed-expert
    /// VRAM cache's hit/miss/eviction counts (the throughput signal for MoE).
    pub fn stats(&self) -> StreamStats {
        let mut s = self.shared.weights.lock().unwrap().stats;
        s.depth = 1;
        let ec = self.shared.experts.lock().unwrap();
        s.expert_hits = ec.hits;
        s.expert_misses = ec.misses;
        s.expert_evictions = ec.evictions;
        s
    }

    /// The MoE FFN for one layer on the device: attention + router on the resident
    /// core, then per-expert application streaming only the top-k experts a token
    /// selects. Mirrors `moe_ffn` in `src/forward/cpu.rs`; the three device calls
    /// share `normed2` via scratch on the in-order default stream.
    #[allow(clippy::too_many_arguments)]
    fn run_moe_block(
        &self,
        layer: u32,
        w: &GpuWeights,
        dkv: &crate::forward::cpu::DeviceKv,
        d_hidden: *mut f32,
        defer: Option<*mut f32>,
        num_positions: usize,
        position: usize,
    ) -> Result<()> {
        use std::ffi::c_void;
        let cfg = &self.shared.cfg;

        // 1. Attention sublayer + post-attn norm; leaves normed2 in device scratch.
        let code = unsafe {
            dlm_moe_attn(
                cfg.hidden_size as i32,
                cfg.q_dim() as i32,
                cfg.kv_dim() as i32,
                cfg.num_heads as i32,
                cfg.num_kv_heads as i32,
                cfg.head_dim as i32,
                cfg.rms_eps,
                w.w_dtype,
                w.w_group_size,
                w.q_proj.as_ptr() as *const c_void,
                w.k_proj.as_ptr() as *const c_void,
                w.v_proj.as_ptr() as *const c_void,
                w.o_proj.as_ptr() as *const c_void,
                w.input_layernorm.as_ptr(),
                w.post_attention_layernorm.as_ptr(),
                bias_ptr(&w.q_bias),
                bias_ptr(&w.k_bias),
                bias_ptr(&w.v_bias),
                bias_ptr(&w.q_norm),
                bias_ptr(&w.k_norm),
                self.inv_freq_for(layer),
                d_hidden,
                dkv.keys,
                dkv.values,
                num_positions as i32,
                position as i32,
                // Per-layer window: Gemma2's global layers must not be clipped.
                cfg.window_for_layer(layer).unwrap_or(0) as i32,
                crate::forward::cpu::rope_mscale(cfg.rope_scaling),
                cfg.attn_scale(),
                cfg.attn_logit_softcap.unwrap_or(0.0),
                dkv.elem.code(),
                dkv.table,
                crate::forward::kv_pool::KV_BLOCK_SHIFT as i32,
                &DlmBlockExt::new(cfg, &w.biases, dkv.elem, dkv.table)
                    .with_weight_dtypes(&w.dtypes)
                    .with_attn_gate(&w.attn_gate),
            )
        };
        if code != 0 {
            return Err(DlmError::Gpu {
                api: "dlm_moe_attn",
                code,
            });
        }
        self.run_moe_ffn(layer, w, d_hidden, defer)
    }

    /// The routed-expert FFN half of a MoE layer: router → top-k → per-expert
    /// application → shared expert. Split out of [`run_moe_block`] because it runs
    /// unchanged after *either* attention path — standard attention (via
    /// `dlm_moe_attn`) or MLA (via `dlm_mla_attn` + `dlm_moe_norm`). Both leave
    /// `normed2` in the same device scratch slot, which is the only state this
    /// consumes, so DeepSeek-V2/V3's MLA + MoE reuses this verbatim.
    /// Run resident layer `layer` (weights `w`) for one token whose hidden state
    /// is already in `d_hidden`, reading and writing `kv`'s device history. The
    /// caller moves the hidden across the bus and keeps `kv`'s host length.
    fn run_layer(
        &self,
        layer: u32,
        w: &GpuWeights,
        kv: &mut KvLayerCache,
        position: usize,
        d_hidden: *mut f32,
        defer: Option<*mut f32>,
    ) -> Result<()> {
        let kv_dim = self.shared.cfg.kv_dim();
        let cfg = &self.shared.cfg;
        // Gated DeltaNet (Qwen3.5/3.6): the token mixer against this sequence's
        // recurrent state, then the post-attention norm and the routed experts,
        // as MLA + MoE runs. No KV for this layer.
        if let Some(lin) = &w.linear {
            lin.run(
                cfg,
                &w.input_layernorm,
                (&w.o_proj, w.dtypes.dtype[W_O], w.dtypes.group[W_O]),
                d_hidden,
                kv,
            )?;
            // SAFETY: `d_hidden` is a live device row of `hidden_size` floats.
            gpu_ok("dlm_moe_norm", unsafe {
                dlm_moe_norm(
                    cfg.hidden_size as i32,
                    cfg.rms_eps,
                    w.post_attention_layernorm.as_ptr(),
                    d_hidden,
                )
            })?;
            return self.run_moe_ffn(layer, w, d_hidden, defer);
        }
        let num_positions = kv.len();
        // Per-session K/V (owned by this sequence's KvLayerCache), so batched
        // requests sharing this kernel keep independent history.
        // MLA reconstructs K and V from one cached latent — no value cache.
        let dkv = kv.device_kv(
            &self.kv_pools,
            layer as usize,
            num_positions + 1,
            self.kv_capacity_tokens,
            self.shared.cfg.mla.is_none(),
        )?;
        let (kv_keys, kv_values) = (dkv.keys, dkv.values);

        // MLA (DeepSeek) runs attention as its own device call, then the FFN half —
        // dense or routed — exactly as the standard-attention paths do. This is the
        // GPU mirror of `decode_block_streaming_moe`, which branches attention on
        // `w.mla` and then runs the same MoE FFN.
        if let (Some(mcfg), Some(mw)) = (&cfg.mla, &w.mla) {
            // SAFETY: all pointers are live device allocations of the sizes the
            // kernel expects; `kv_keys` has capacity for `num_positions + 1` rows
            // of `kv_lora_rank + qk_rope_head_dim` (MLA's `kv_dim`).
            let code = unsafe {
                dlm_mla_attn(
                    cfg.hidden_size as i32,
                    cfg.num_heads as i32,
                    mcfg.q_lora_rank.unwrap_or(0) as i32,
                    mcfg.kv_lora_rank as i32,
                    mcfg.qk_nope_head_dim as i32,
                    mcfg.qk_rope_head_dim as i32,
                    mcfg.v_head_dim as i32,
                    cfg.rms_eps,
                    w.w_dtype,
                    w.w_group_size,
                    mw.q_a_proj
                        .as_ref()
                        .map_or(std::ptr::null(), |b| b.as_ptr())
                        as *const std::ffi::c_void,
                    bias_ptr(&mw.q_a_layernorm),
                    mw.q_b_proj.as_ptr() as *const std::ffi::c_void,
                    mw.kv_a_proj.as_ptr() as *const std::ffi::c_void,
                    mw.kv_a_layernorm.as_ptr(),
                    mw.kv_b_proj.as_ptr() as *const std::ffi::c_void,
                    w.o_proj.as_ptr() as *const std::ffi::c_void,
                    w.input_layernorm.as_ptr(),
                    self.mla_inv_freq
                        .as_ref()
                        .expect("mla_inv_freq present whenever cfg.mla is")
                        .as_ptr(),
                    crate::forward::cpu::rope_mscale(cfg.rope_scaling),
                    d_hidden,
                    kv_keys,
                    num_positions as i32,
                    position as i32,
                )
            };
            if code != 0 {
                return Err(DlmError::Gpu {
                    api: "dlm_mla_attn",
                    code,
                });
            }
            match &w.ffn {
                GpuFfn::Dense(f) => {
                    let code = unsafe {
                        dlm_dense_ffn(
                            cfg.hidden_size as i32,
                            cfg.intermediate_size as i32,
                            cfg.rms_eps,
                            f.w_dtype,
                            f.w_group_size,
                            f.gate.as_ptr() as *const std::ffi::c_void,
                            f.up.as_ptr() as *const std::ffi::c_void,
                            f.down.as_ptr() as *const std::ffi::c_void,
                            w.post_attention_layernorm.as_ptr(),
                            cfg.activation.code(),
                            d_hidden,
                        )
                    };
                    if code != 0 {
                        return Err(DlmError::Gpu {
                            api: "dlm_dense_ffn",
                            code,
                        });
                    }
                }
                GpuFfn::Moe { .. } => {
                    // `dlm_mla_attn` folds the residual but leaves no FFN input;
                    // this produces the `normed2` the router/experts consume.
                    let code = unsafe {
                        dlm_moe_norm(
                            cfg.hidden_size as i32,
                            cfg.rms_eps,
                            w.post_attention_layernorm.as_ptr(),
                            d_hidden,
                        )
                    };
                    if code != 0 {
                        return Err(DlmError::Gpu {
                            api: "dlm_moe_norm",
                            code,
                        });
                    }
                    self.run_moe_ffn(layer, w, d_hidden, defer)?;
                }
            }
        } else {
            match &w.ffn {
                GpuFfn::Dense(f) => {
                    // SAFETY: all pointers are live device allocations sized as the
                    // kernel expects; the KV buffers have capacity for `num_positions + 1`.
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
                            f.gate.as_ptr() as *const std::ffi::c_void,
                            f.up.as_ptr() as *const std::ffi::c_void,
                            f.down.as_ptr() as *const std::ffi::c_void,
                            w.input_layernorm.as_ptr(),
                            w.post_attention_layernorm.as_ptr(),
                            bias_ptr(&w.q_bias),
                            bias_ptr(&w.k_bias),
                            bias_ptr(&w.v_bias),
                            bias_ptr(&w.q_norm),
                            bias_ptr(&w.k_norm),
                            self.inv_freq_for(layer),
                            d_hidden,
                            kv_keys,
                            kv_values,
                            num_positions as i32,
                            position as i32,
                            // Per-layer window: Gemma2's global layers must not be clipped.
                            cfg.window_for_layer(layer).unwrap_or(0) as i32,
                            cfg.activation.code(),
                            crate::forward::cpu::rope_mscale(cfg.for_layer(layer).rope_scaling),
                            cfg.attn_scale(),
                            cfg.attn_logit_softcap.unwrap_or(0.0),
                            bias_ptr(&w.pre_ffn_norm),
                            bias_ptr(&w.post_ffn_norm),
                            &DlmBlockExt::new(cfg, &w.biases, dkv.elem, dkv.table)
                                .with_weight_dtypes(&w.dtypes),
                        )
                    };
                    if code != 0 {
                        return Err(DlmError::Gpu {
                            api: "dlm_decode_block",
                            code,
                        });
                    }
                }
                GpuFfn::Moe { .. } => {
                    self.run_moe_block(layer, w, &dkv, d_hidden, defer, num_positions, position)?;
                }
            }
        }
        Ok(())
    }

    /// The expert half of a MoE layer for one token. Each selected expert runs
    /// on the GPU when VRAM-resident; a miss runs on the CPU or is uploaded,
    /// whichever has measured cheaper. One token per expert is a memory-bound
    /// read, and system RAM can outrun the PCIe link of a small card, which is
    /// why inferssd keeps decode experts on the CPU. With `defer` it only saves
    /// the FFN input there, for [`moe_ffn_grouped`](Self::moe_ffn_grouped).
    fn run_moe_ffn(
        &self,
        layer: u32,
        w: &GpuWeights,
        d_hidden: *mut f32,
        defer: Option<*mut f32>,
    ) -> Result<()> {
        let cfg = &self.shared.cfg;
        let h = cfg.hidden_size;
        if let Some(dst) = defer {
            // SAFETY: `dst` is a live device row of `h` floats.
            return gpu_ok("dlm_moe_save_normed", unsafe {
                dlm_moe_save_normed(h as i32, dst)
            });
        }
        let GpuFfn::Moe(core) = &w.ffn else {
            unreachable!("run_moe_ffn on a dense layer");
        };
        let m = cfg.moe.expect("MoE layer has a MoE config");
        // The router's logits come back to the host, so the device is idle here.
        let selected = core.route(cfg)?;
        {
            let mut sc = self.shared.scores.lock().unwrap();
            sc.tick(layer);
            for &(e, _) in &selected {
                sc.record(layer, e as u32);
            }
        }
        let (mut on_gpu, mut on_cpu) = (Vec::new(), Vec::new());
        for (e, weight) in selected {
            let e = e as u32;
            if let Some(x) = self.shared.cached_expert(layer, e) {
                on_gpu.push((x, weight));
            } else if self.shared.miss_on_cpu() {
                on_cpu.push((e, weight));
            } else {
                let t = std::time::Instant::now();
                on_gpu.push((self.shared.upload_expert(layer, e)?, weight));
                ewma(&self.shared.miss_gpu_ns, t.elapsed().as_nanos() as u64);
            }
        }
        // Single-token decode always chains through `self.d_hidden`.
        debug_assert_eq!(d_hidden, self.d_hidden.as_mut_ptr());
        let mut ffn_inp = vec![0f32; h];
        self.d_hidden.download(&mut ffn_inp)?;
        let mut normed = vec![0f32; h];
        if !on_cpu.is_empty() {
            // SAFETY: `d_normed` holds `h` floats.
            gpu_ok("dlm_moe_save_normed", unsafe {
                dlm_moe_save_normed(h as i32, self.d_normed.as_mut_ptr())
            })?;
            self.d_normed.download(&mut normed)?;
        }
        core.apply_routed(cfg, &on_gpu, d_hidden)?;
        core.apply_shared(cfg, d_hidden)?;
        // Host work from here overlaps the experts running on the device.
        self.hint_next(layer, &ffn_inp);
        if on_cpu.is_empty() {
            return Ok(());
        }
        let inter = m.moe_intermediate_size as usize;
        let mut add = vec![0f32; h];
        for (e, weight) in on_cpu {
            let t = std::time::Instant::now();
            let expert = self.shared.source.load_expert(layer, e)?;
            let y = crate::forward::cpu::swiglu_ffn(&expert, &normed, h, inter, cfg.activation);
            for (a, v) in add.iter_mut().zip(&y) {
                *a += weight * v;
            }
            ewma(&self.shared.miss_cpu_ns, t.elapsed().as_nanos() as u64);
        }
        synchronize_default()?;
        let mut x = vec![0f32; h];
        self.d_hidden.download(&mut x)?;
        for (x, a) in x.iter_mut().zip(&add) {
            *x += a;
        }
        self.d_hidden.upload(&x)
    }

    /// `decode_batch` for a MoE model: layer by layer for the whole batch,
    /// attention batched over up to DLM_MAX_BATCH sequences (one at a time for
    /// MLA), then the experts grouped over the batch
    /// ([`moe_ffn_grouped`](Self::moe_ffn_grouped)). Each layer core streams in
    /// once per step and each expert once per layer, not once per sequence.
    fn decode_batch_moe(
        &self,
        hiddens: &mut [&mut [f32]],
        kv_layers: &mut [&mut [KvLayerCache]],
        positions: &[usize],
    ) -> Result<()> {
        let h = self.shared.cfg.hidden_size;
        let b = hiddens.len();
        let mut flat: Vec<f32> = hiddens.iter().flat_map(|x| x.iter().copied()).collect();
        let d_x = DeviceBuffer::from_slice(&flat)?;
        let d_norm = DeviceBuffer::new(b * h)?;
        // SAFETY (every `row`): both buffers hold `b` rows of `h` floats.
        let row = |d: &DeviceBuffer, i: usize| unsafe { d.as_mut_ptr().add(i * h) };
        for layer in 0..self.num_layers {
            let w = self.shared.fetch(layer)?;
            if let Some(tx) = &self.prefetch_tx {
                let _ = tx.send((layer + 1) % self.num_layers);
            }
            let moe = matches!(w.ffn, GpuFfn::Moe(_));
            for (i, kvs) in kv_layers.iter().enumerate() {
                if kvs[layer as usize].len() >= self.kv_capacity_tokens {
                    return Err(DlmError::InvalidConfig(format!(
                        "GPU KV capacity {} exceeded at position {}",
                        self.kv_capacity_tokens, positions[i]
                    )));
                }
            }
            if let (true, Some(l)) = (moe && w.linear.is_some(), self.shared.cfg.linear) {
                // Each sequence its own recurrent state, up to DLM_MAX_BATCH a call.
                for c in (0..b).step_by(DLM_MAX_BATCH) {
                    let end = (c + DLM_MAX_BATCH).min(b);
                    let states = kv_layers[c..end]
                        .iter_mut()
                        .map(|kvs| {
                            kvs[layer as usize]
                                .linear_device_state(l.conv_state_len(), l.recurrent_state_len())
                        })
                        .collect::<Result<Vec<_>>>()?;
                    self.linear_rows(&w, row(&d_x, c), &states, row(&d_norm, c))?;
                }
                for kvs in kv_layers.iter_mut() {
                    kvs[layer as usize].advance(1);
                }
            } else if moe && self.shared.cfg.mla.is_none() {
                // Up to DLM_MAX_BATCH sequences per call, each with its own KV.
                for c in (0..b).step_by(DLM_MAX_BATCH) {
                    let end = (c + DLM_MAX_BATCH).min(b);
                    let mut slots = SlotKv::new();
                    for (i, kvs) in kv_layers[c..end].iter_mut().enumerate() {
                        let kv = &mut kvs[layer as usize];
                        let n = kv.len();
                        let dkv = kv.device_kv(
                            &self.kv_pools,
                            layer as usize,
                            n + 1,
                            self.kv_capacity_tokens,
                            true,
                        )?;
                        slots.push(&dkv, n, positions[c + i]);
                    }
                    self.moe_attn_batched(layer, &w, row(&d_x, c), &slots, row(&d_norm, c))?;
                }
                for kvs in kv_layers.iter_mut() {
                    kvs[layer as usize].advance(1);
                }
            } else {
                for (i, kvs) in kv_layers.iter_mut().enumerate() {
                    let kv = &mut kvs[layer as usize];
                    let defer = moe.then(|| row(&d_norm, i));
                    self.run_layer(layer, &w, kv, positions[i], row(&d_x, i), defer)?;
                    kv.advance(1);
                }
            }
            if moe {
                self.moe_ffn_grouped(layer, &w, d_x.as_mut_ptr(), d_norm.as_mut_ptr(), b)?;
            }
        }
        synchronize_default()?;
        d_x.download(&mut flat)?;
        for (i, hidden) in hiddens.iter_mut().enumerate() {
            hidden.copy_from_slice(&flat[i * h..(i + 1) * h]);
        }
        Ok(())
    }

    /// Predict layer `layer + 1`'s experts by running its own router on this
    /// layer's FFN input, and send the worker to warm them while this layer's
    /// experts and the next attention run (inferssd's prefetch hint). Exact
    /// more often than the next layer's score history, which only knows what it
    /// routed to before. No-op until the next layer's core has streamed in once.
    fn hint_next(&self, layer: u32, ffn_inp: &[f32]) {
        let next = layer + 1;
        let cfg = &self.shared.cfg;
        let Some(m) = cfg.moe else { return };
        if next >= self.num_layers {
            return;
        }
        let Some(r) = self.shared.routers.lock().unwrap().get(&next).cloned() else {
            return;
        };
        let (router, norm, bias) = &*r;
        let x = crate::forward::cpu::rmsnorm(ffn_inp, norm, cfg.rms_eps);
        let logits =
            crate::forward::cpu::matvec_native(router, &x, m.num_experts as usize, cfg.hidden_size);
        let experts = crate::forward::cpu::route(&logits, &m, bias.as_deref())
            .into_iter()
            .map(|(e, _)| e as u32)
            .collect();
        self.shared
            .scores
            .lock()
            .unwrap()
            .hints
            .insert(next, experts);
        if let Some(tx) = &self.prefetch_tx {
            let _ = tx.send(next);
        }
    }

    /// The expert half of a MoE layer for `n` tokens at once, expert-major:
    /// row `t` of `norm` holds token `t`'s FFN input and row `t` of `x` its
    /// hidden, both on the device. Each selected expert is fetched once and run
    /// once over every token that chose it (`dlm_apply_expert_rows`: a GEMM
    /// for many rows, the batched GEMV for few), and the shared expert once over
    /// all of them. Token by token, a cache smaller than the experts a prompt
    /// touches re-uploaded the same expert over and over, and every token read
    /// each expert's weights again (inferssd's grouped prefill). The pass counts
    /// as one tick of the layer's scores, so a prompt's most-routed experts
    /// start decode warm.
    fn moe_ffn_grouped(
        &self,
        layer: u32,
        w: &GpuWeights,
        x: *mut f32,
        norm: *mut f32,
        n: usize,
    ) -> Result<()> {
        let GpuFfn::Moe(core) = &w.ffn else {
            unreachable!("moe_ffn_grouped on a dense layer");
        };
        let cfg = &self.shared.cfg;
        let h = cfg.hidden_size;
        let inter = cfg.moe.expect("MoE layer").moe_intermediate_size as i32;
        let shared = core.shared(cfg);
        let mut by_expert: std::collections::BTreeMap<u32, (Vec<i32>, Vec<f32>)> =
            std::collections::BTreeMap::new();
        let mut shared_weights = Vec::with_capacity(if shared.is_some() { n } else { 0 });
        for t in 0..n {
            // SAFETY: row `t` of `norm`'s `n` rows of `h` floats.
            gpu_ok("dlm_moe_load_normed", unsafe {
                dlm_moe_load_normed(h as i32, norm.add(t * h))
            })?;
            for (e, weight) in core.route(cfg)? {
                let (rows, ws) = by_expert.entry(e as u32).or_default();
                rows.push(t as i32);
                ws.push(weight);
            }
            if shared.is_some() {
                shared_weights.push(core.shared_weight(cfg)?);
            }
        }
        {
            let mut sc = self.shared.scores.lock().unwrap();
            sc.tick(layer);
            for (&e, (rows, _)) in &by_expert {
                for _ in rows {
                    sc.record(layer, e);
                }
            }
        }
        let apply = |e: &GpuExpert, inter: i32, rows: &[i32], ws: &[f32]| {
            // SAFETY: live device weights; `norm` and `x` hold `n` rows and every
            // index in `rows` is a distinct row below `n`.
            gpu_ok("dlm_apply_expert_rows", unsafe {
                dlm_apply_expert_rows(
                    h as i32,
                    inter,
                    e.w_dtype,
                    e.w_group_size,
                    e.gate.as_ptr() as *const std::ffi::c_void,
                    e.up.as_ptr() as *const std::ffi::c_void,
                    e.down.as_ptr() as *const std::ffi::c_void,
                    norm,
                    rows.as_ptr(),
                    ws.as_ptr(),
                    rows.len() as i32,
                    x,
                    cfg.activation.code(),
                )
            })
        };
        for (e, (rows, ws)) in by_expert {
            let expert = self.shared.ensure_expert(layer, e)?;
            apply(&expert, inter, &rows, &ws)?;
        }
        if let Some((sh, sinter)) = shared {
            let all: Vec<i32> = (0..n as i32).collect();
            apply(sh, sinter, &all, &shared_weights)?;
        }
        Ok(())
    }
}

/// A device call's status as a `Result`.
fn gpu_ok(api: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(DlmError::Gpu { api, code })
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
    fn kv_free_tokens(&self, elem: crate::forward::cpu::KvElem) -> Option<usize> {
        Some(self.kv_pools.free_tokens(elem))
    }

    fn set_kv_pool_tokens(&mut self, tokens: usize) {
        let host_from = self.kv_pools.host_from();
        self.kv_pools = crate::forward::kv_pool::KvPools::new(
            self.num_layers as usize,
            self.shared.cfg.kv_dim(),
            self.shared.cfg.head_dim,
            tokens,
        );
        self.kv_pools.set_host_layers_from(host_from);
    }

    fn set_kv_host_layers(&mut self, from: usize) {
        self.kv_pools.set_host_layers_from(from);
    }

    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.shared.cfg.hidden_size
    }

    fn kv_dim(&self) -> usize {
        self.shared.cfg.kv_dim()
    }

    fn recurrent(&self) -> bool {
        self.shared.cfg.linear.is_some()
    }

    fn stream_stats(&self) -> Option<StreamStats> {
        Some(self.stats())
    }

    /// Prefill layer by layer, so each layer streams into VRAM once per prompt
    /// chunk instead of once per prompt token — on a model bigger than the card,
    /// the difference between re-sending the model per token and sending it once.
    ///
    /// Dense standard-attention layers run up to [`DLM_MAX_BATCH`] tokens per
    /// device call, every slot aliasing the one layer KV (see `GpuKernel::prefill`
    /// for why that is exact causal attention). MoE layers run attention the
    /// same way (token by token for MLA), then each routed expert once for the
    /// whole prompt
    /// ([`moe_ffn_grouped`](Self::moe_ffn_grouped)). MLA layers with a dense FFN
    /// go one token at a time through the body `run_block` uses — still one
    /// stream-in per layer.
    ///
    /// ponytail: the hidden states live on the host between layers and cross the
    /// bus per chunk (dense) or per token (MLA/MoE). Small next to a streamed
    /// layer; keep them on the device if a profile ever shows the round trips.
    /// Streams each layer once for the whole batch: a dense model's sequences all
    /// run on a layer's weights in one fused block call before the next layer is
    /// fetched. The default ran each sequence through the whole stack in turn,
    /// which streamed every non-resident layer once per sequence, so at batch 4 a
    /// streamed model decoded slower in total than at batch 1.
    ///
    /// MoE models take [`decode_batch_moe`](Self::decode_batch_moe); MLA
    /// models have no batched block and keep the default.
    fn decode_batch(
        &self,
        hiddens: &mut [&mut [f32]],
        kv_layers: &mut [&mut [KvLayerCache]],
        positions: &[usize],
    ) -> Result<()> {
        let cfg = &self.shared.cfg;
        if hiddens.len() > 1 && cfg.moe.is_some() {
            return self.decode_batch_moe(hiddens, kv_layers, positions);
        }
        if hiddens.len() <= 1 || cfg.mla.is_some() {
            for ((hidden, kvs), &position) in
                hiddens.iter_mut().zip(kv_layers.iter_mut()).zip(positions)
            {
                for layer in 0..self.num_layers {
                    self.run_block(layer, hidden, &mut kvs[layer as usize], position)?;
                }
            }
            return Ok(());
        }
        let h = cfg.hidden_size;
        let mut staged = vec![0.0f32; DLM_MAX_BATCH * h];
        let chunks = hiddens
            .chunks_mut(DLM_MAX_BATCH)
            .zip(kv_layers.chunks_mut(DLM_MAX_BATCH))
            .zip(positions.chunks(DLM_MAX_BATCH));
        for ((hs, kvs), pos) in chunks {
            let batch = hs.len();
            for (b, hidden) in hs.iter().enumerate() {
                staged[b * h..(b + 1) * h].copy_from_slice(hidden);
            }
            self.d_batch.upload(&staged[..batch * h])?;
            for layer in 0..self.num_layers {
                let w = self.shared.fetch(layer)?;
                if let Some(tx) = &self.prefetch_tx {
                    let _ = tx.send((layer + 1) % self.num_layers);
                }
                let mut slots = SlotKv::new();
                for (b, seq) in kvs.iter_mut().enumerate() {
                    let kv = &mut seq[layer as usize];
                    let n = kv.len();
                    if n >= self.kv_capacity_tokens {
                        return Err(DlmError::InvalidConfig(format!(
                            "GPU KV capacity {} exceeded at position {}",
                            self.kv_capacity_tokens, pos[b]
                        )));
                    }
                    let dkv = kv.device_kv(
                        &self.kv_pools,
                        layer as usize,
                        n + 1,
                        self.kv_capacity_tokens,
                        true,
                    )?;
                    slots.push(&dkv, n, pos[b]);
                }
                self.launch_dense_batched(layer, &w, self.d_batch.as_mut_ptr(), &slots)?;
                for seq in kvs.iter_mut() {
                    seq[layer as usize].advance(1);
                }
            }
            // Drain the stack before the blocking copy, as `run_block` does, so an
            // upload still overlapping on the copy stream finishes first.
            synchronize_default()?;
            self.d_batch.download(&mut staged[..batch * h])?;
            for (b, hidden) in hs.iter_mut().enumerate() {
                hidden.copy_from_slice(&staged[b * h..(b + 1) * h]);
            }
        }
        Ok(())
    }

    fn prefill(
        &self,
        hiddens: &mut [f32],
        kv_layers: &mut [KvLayerCache],
        start: usize,
    ) -> Result<()> {
        let cfg = &self.shared.cfg;
        let hidden_size = cfg.hidden_size;
        let n = hiddens.len() / hidden_size;
        if kv_layers.first().map_or(0, |kv| kv.len()) + n > self.kv_capacity_tokens {
            return Err(DlmError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {}",
                self.kv_capacity_tokens,
                start + n - 1
            )));
        }
        // MoE layers keep the prompt on the device as `[n, hidden]` rows plus
        // each token's FFN input, for `moe_ffn_grouped`.
        let moe_rows = match cfg.moe {
            Some(_) => Some((
                DeviceBuffer::new(n * hidden_size)?,
                DeviceBuffer::new(n * hidden_size)?,
            )),
            None => None,
        };
        for layer in 0..self.num_layers {
            let w = self.shared.fetch(layer)?;
            if let Some(tx) = &self.prefetch_tx {
                let _ = tx.send((layer + 1) % self.num_layers);
            }
            let kv = &mut kv_layers[layer as usize];
            if let (GpuFfn::Moe(_), Some((d_x, d_norm))) = (&w.ffn, &moe_rows) {
                // Attention token by token, then each expert once for the prompt.
                // Reserve the KV up front so no token grows it under queued work.
                // A linear-attention layer keeps no KV.
                if w.linear.is_none() {
                    kv.device_kv(
                        &self.kv_pools,
                        layer as usize,
                        kv.len() + n,
                        self.kv_capacity_tokens,
                        cfg.mla.is_none(),
                    )?;
                }
                d_x.upload(hiddens)?;
                // SAFETY (every `row`): both buffers hold `n` rows of `hidden_size`.
                let row =
                    |d: &DeviceBuffer, i: usize| unsafe { d.as_mut_ptr().add(i * hidden_size) };
                // ponytail: a linear-attention layer prefills DLM_MAX_BATCH
                // tokens a call (batched GEMVs, the state stepped token by
                // token); a chunked delta-rule kernel with GEMMs would read each
                // projection once per prompt.
                if let (Some(_), Some(l)) = (&w.linear, cfg.linear) {
                    for c in (0..n).step_by(DLM_MAX_BATCH) {
                        let batch = (n - c).min(DLM_MAX_BATCH);
                        let state =
                            kv.linear_device_state(l.conv_state_len(), l.recurrent_state_len())?;
                        self.linear_rows(&w, row(d_x, c), &vec![state; batch], row(d_norm, c))?;
                        kv.advance(batch);
                    }
                } else if cfg.mla.is_none() {
                    // Every slot aliases the one layer KV, as the dense chunks do.
                    for c in (0..n).step_by(DLM_MAX_BATCH) {
                        let batch = (n - c).min(DLM_MAX_BATCH);
                        let history = kv.len();
                        let dkv = kv.device_kv(
                            &self.kv_pools,
                            layer as usize,
                            history + batch,
                            self.kv_capacity_tokens,
                            true,
                        )?;
                        let mut slots = SlotKv::new();
                        for b in 0..batch {
                            slots.push(&dkv, history + b, start + c + b);
                        }
                        self.moe_attn_batched(layer, &w, row(d_x, c), &slots, row(d_norm, c))?;
                        kv.advance(batch);
                    }
                } else {
                    for i in 0..n {
                        self.run_layer(
                            layer,
                            &w,
                            kv,
                            start + i,
                            row(d_x, i),
                            Some(row(d_norm, i)),
                        )?;
                        kv.advance(1);
                    }
                }
                self.moe_ffn_grouped(layer, &w, d_x.as_mut_ptr(), d_norm.as_mut_ptr(), n)?;
                synchronize_default()?;
                d_x.download(hiddens)?;
                continue;
            }
            let batched = cfg.mla.is_none() && matches!(w.ffn, GpuFfn::Dense(_));
            if !batched {
                for (i, hidden) in hiddens.chunks_mut(hidden_size).enumerate() {
                    self.d_hidden.upload(hidden)?;
                    self.run_layer(layer, &w, kv, start + i, self.d_hidden.as_mut_ptr(), None)?;
                    synchronize_default()?;
                    self.d_hidden.download(hidden)?;
                    kv.advance(1);
                }
                continue;
            }
            for (c, chunk) in hiddens.chunks_mut(hidden_size * DLM_MAX_BATCH).enumerate() {
                let batch = chunk.len() / hidden_size;
                let history = kv.len();
                let dkv = kv.device_kv(
                    &self.kv_pools,
                    layer as usize,
                    history + batch,
                    self.kv_capacity_tokens,
                    true,
                )?;
                let mut slots = SlotKv::new();
                for b in 0..batch {
                    slots.push(&dkv, history + b, start + c * DLM_MAX_BATCH + b);
                }
                let d_chunk = DeviceBuffer::from_slice(chunk)?;
                self.launch_dense_batched(layer, &w, d_chunk.as_mut_ptr(), &slots)?;
                d_chunk.download(chunk)?;
                kv.advance(batch);
            }
        }
        Ok(())
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
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
        // The hidden stays resident across the streamed stack: upload before the
        // first layer, chain through, download after the last — so only weights
        // cross the bus per layer, not a hidden round-trip each time.
        let is_first = layer == 0;
        let is_last = layer == self.num_layers - 1;
        let d_hidden = &self.d_hidden;
        if is_first {
            d_hidden.upload(hidden)?;
        }
        self.run_layer(layer, &w, kv, position, d_hidden.as_mut_ptr(), None)?;

        // Bring the result back only after the last layer. `download` is a blocking
        // D2H that drains the default stream, so an in-flight weight upload on the
        // copy stream keeps overlapping until then.
        if is_last {
            synchronize_default()?;
            d_hidden.download(hidden)?;
        }

        // Keep the orchestrator's length bookkeeping in step (real K/V is in VRAM).
        kv.advance(1);
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
        assert_eq!(
            lru,
            layers * cycles,
            "LRU should miss everything on a cyclic scan"
        );
        // MRU: only the layers that cannot fit miss.
        let steady = mru - layers; // discount the cold first pass
        let ceiling = (layers as usize - capacity + 1) as u32 * (cycles - 1);
        assert!(
            steady <= ceiling,
            "MRU steady-state misses {steady} should be <= {ceiling} (~N-capacity per pass)"
        );
        assert!(
            mru * 3 < lru,
            "MRU ({mru}) should miss far less than LRU ({lru})"
        );
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

    /// Scores rank a layer's experts by decayed routing count, keep layers
    /// independent, and let a recent expert overtake one that has gone quiet.
    #[test]
    fn expert_scores_rank_and_decay() {
        let mut h = super::ExpertScores::default();
        h.tick(0);
        for _ in 0..3 {
            h.record(0, 5);
        }
        h.record(0, 2);
        h.record(1, 7);
        assert_eq!(h.hot(0, 2), vec![5, 2]);
        assert_eq!(h.hot(1, 2), vec![7], "layer 1 is independent");
        assert!(h.hot(3, 4).is_empty(), "an unseen layer has no hot experts");
        // Six passes later 5's three picks are worth 3/8; 2 picked twice now wins.
        for _ in 0..6 {
            h.tick(0);
        }
        h.record(0, 2);
        assert_eq!(h.hot(0, 2), vec![2, 5]);
        // A router prediction overrides the scores.
        h.hints.insert(0, vec![9]);
        assert_eq!(h.hot(0, 2), vec![9]);
    }

    /// Ties break on expert id so prefetch order is deterministic.
    #[test]
    fn expert_scores_break_ties_by_id() {
        let mut h = super::ExpertScores::default();
        h.record(0, 8);
        h.record(0, 3);
        assert_eq!(h.hot(0, 2), vec![3, 8], "equal scores → ascending id");
    }
}

//! GPU compute kernel — the device `run_block` (feature `cuda-kernels`).
//!
//! Structurally identical to [`CpuKernel`](crate::forward::CpuKernel): it holds a
//! model's per-layer weights in VRAM and implements [`ComputeKernel`] by running
//! one decode block per token. The transformer math lives in
//! `src/gpu/kernels.cu`, invoked through the `dlm_decode_block` FFI entry point.
//!
//! **KV stays on device.** Each layer owns persistent `kv_keys`/`kv_values`
//! buffers in VRAM (sized for the sequence). Per token the kernel writes the new
//! K/V into the next slot in place and attends over the buffer directly, so only
//! the hidden vector (`hidden_size` floats) crosses the PCIe bus each layer —
//! not the whole KV history. The host [`KvLayerCache`] passed by the orchestrator
//! is used only for length bookkeeping on this path (the real K/V is in VRAM); we
//! append zero placeholders to keep its length in step without a device→host copy.
//!
//! Requires nvcc at build time and a GPU at run time; the CPU kernel is the
//! correctness oracle ([`tests/gpu_parity.rs`]).

use crate::error::{DlmError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use crate::gpu::device::DeviceBuffer;
use std::ffi::c_void;

extern "C" {
    /// One decode block on the device (see `src/gpu/kernels.cu`). Returns a
    /// `cudaError_t` (0 == success). All pointers are device pointers; `kv_keys`
    /// and `kv_values` are persistent buffers written in place at slot
    /// `num_positions`. `q_bias`/`k_bias`/`v_bias` may be null.
    ///
    /// Declared **once**, here: the streaming GPU kernel imports this rather than
    /// restating it. Two `extern` declarations of one symbol let the signatures
    /// drift apart silently, which is undefined behavior at the call site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_decode_block(
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        inter: i32,
        rms_eps: f32,
        w_dtype: i32,
        w_group_size: i32,
        q_proj: *const c_void,
        k_proj: *const c_void,
        v_proj: *const c_void,
        o_proj: *const c_void,
        gate_proj: *const c_void,
        up_proj: *const c_void,
        down_proj: *const c_void,
        in_norm: *const f32,
        post_norm: *const f32,
        q_bias: *const f32,
        k_bias: *const f32,
        v_bias: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        inv_freq: *const f32,
        x: *mut f32,
        kv_keys: *mut f32,
        kv_values: *mut f32,
        num_positions: i32,
        position: i32,
        // Sliding-window span (Mistral); 0 = full causal attention.
        sliding_window: i32,
        // Gate activation (DLM_ACT_*): SiLU (0) or GELU (1, Gemma).
        activation: i32,
        // YaRN attention temperature folded into cos/sin; 1.0 otherwise.
        rope_mscale: f32,
        // Attention score scale; <= 0 means the usual 1/sqrt(head_dim).
        attn_scale: f32,
        // Gemma2 attention-logit softcap; 0 = off.
        attn_softcap: f32,
        // Gemma2's extra norm pair; NULL elsewhere. When set they also change
        // what `post_norm` means — see `decode_block` in cpu.rs.
        pre_ffn_norm: *const f32,
        post_ffn_norm: *const f32,
        // Non-Llama block shape (GPT-2, Falcon); null for the Llama default.
        ext: *const DlmBlockExt,
    ) -> i32;

    /// MoE layer, part 1: attention sublayer + post-attn norm. Leaves `normed2`
    /// (the FFN input) in device scratch for [`dlm_moe_matvec`]/[`dlm_apply_expert`]
    /// to consume on the same stream. See `src/gpu/kernels.cu`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_moe_attn(
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rms_eps: f32,
        w_dtype: i32,
        w_group_size: i32,
        q_proj: *const c_void,
        k_proj: *const c_void,
        v_proj: *const c_void,
        o_proj: *const c_void,
        in_norm: *const f32,
        post_norm: *const f32,
        q_bias: *const f32,
        k_bias: *const f32,
        v_bias: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        inv_freq: *const f32,
        x: *mut f32,
        kv_keys: *mut f32,
        kv_values: *mut f32,
        num_positions: i32,
        position: i32,
        // Sliding-window span (Mistral); 0 = full causal attention.
        sliding_window: i32,
        // YaRN attention temperature folded into cos/sin; 1.0 otherwise.
        rope_mscale: f32,
        // Attention score scale; <= 0 means the usual 1/sqrt(head_dim).
        attn_scale: f32,
        // Gemma2 attention-logit softcap; 0 = off.
        attn_softcap: f32,
        // 1: the KV buffers hold fp16.
        kv_half: i32,
        // The sequence's KV block table, and log2 of rows per block.
        kv_table: *const i32,
        kv_block_shift: i32,
    ) -> i32;

    /// Post-attention norm for a MoE layer whose attention ran in a separate call
    /// (the MLA path). Leaves the FFN input in the same scratch slot
    /// [`dlm_moe_attn`] would, so the router/expert calls after it are identical
    /// on either attention path.
    pub(crate) fn dlm_moe_norm(
        hidden_size: i32,
        rms_eps: f32,
        post_norm: *const f32,
        x: *mut f32,
    ) -> i32;

    /// `logits_host[0..vocab] = W · x_host`: the LM head on the device.
    pub(crate) fn dlm_lm_head(
        vocab: i32,
        hidden_size: i32,
        w_dtype: i32,
        w_group_size: i32,
        w: *const c_void,
        x_host: *const f32,
        logits_host: *mut f32,
    ) -> i32;

    /// MoE layer, part 2: `y_host[0..out_dim] = W · normed2`, copied to host. For
    /// the router logits (`out_dim = num_experts`) and the shared gate (`out_dim = 1`).
    pub(crate) fn dlm_moe_matvec(
        out_dim: i32,
        hidden_size: i32,
        w_dtype: i32,
        w_group_size: i32,
        w: *const c_void,
        y_host: *mut f32,
    ) -> i32;

    /// Multi-head Latent Attention sublayer (DeepSeek). All attention-projection
    /// weights share `w_dtype`. `q_a_proj`/`q_a_layernorm` may be null (no query
    /// low-rank). `kv_keys` is the per-session cache (width `kv_lora_rank +
    /// qk_rope`); `x` gets the attention residual folded in. See `src/gpu/kernels.cu`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_mla_attn(
        hidden_size: i32,
        num_heads: i32,
        q_lora_rank: i32,
        kv_lora_rank: i32,
        qk_nope: i32,
        qk_rope: i32,
        v_head_dim: i32,
        rms_eps: f32,
        w_dtype: i32,
        w_group_size: i32,
        q_a_proj: *const c_void,
        q_a_layernorm: *const f32,
        q_b_proj: *const c_void,
        kv_a_proj: *const c_void,
        kv_a_layernorm: *const f32,
        kv_b_proj: *const c_void,
        o_proj: *const c_void,
        in_norm: *const f32,
        inv_freq: *const f32,
        rope_mscale: f32,
        x: *mut f32,
        kv_keys: *mut f32,
        num_positions: i32,
        position: i32,
    ) -> i32;

    /// Dense gated-MLP FFN sublayer on its own (`x += down·(act(gate·norm) ⊙
    /// up·norm)`), for the MLA path whose attention is a separate call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_dense_ffn(
        hidden_size: i32,
        inter: i32,
        rms_eps: f32,
        w_dtype: i32,
        w_group_size: i32,
        gate_proj: *const c_void,
        up_proj: *const c_void,
        down_proj: *const c_void,
        post_norm: *const f32,
        activation: i32,
        x: *mut f32,
    ) -> i32;

    /// One decode block for `batch` sequences at once. `x` is a contiguous
    /// `[batch, hidden_size]` device block; each slot keeps its own KV buffers,
    /// history length and RoPE position. Op-for-op identical to calling
    /// [`dlm_decode_block`] per slot — only the weight traffic changes, since each
    /// weight row is read once for the whole batch instead of once per sequence.
    /// Returns `cudaErrorInvalidValue` if `batch` exceeds [`DLM_MAX_BATCH`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_decode_block_batched(
        hidden_size: i32,
        q_dim: i32,
        kv_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        inter: i32,
        rms_eps: f32,
        w_dtype: i32,
        w_group_size: i32,
        q_proj: *const c_void,
        k_proj: *const c_void,
        v_proj: *const c_void,
        o_proj: *const c_void,
        gate_proj: *const c_void,
        up_proj: *const c_void,
        down_proj: *const c_void,
        in_norm: *const f32,
        post_norm: *const f32,
        q_bias: *const f32,
        k_bias: *const f32,
        v_bias: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        inv_freq: *const f32,
        x: *mut f32,
        kv_keys: *const DlmSlots,
        kv_values: *const DlmSlots,
        kv_tables: *const DlmTables,
        num_positions: *const DlmInts,
        positions: *const DlmInts,
        batch: i32,
        sliding_window: i32,
        activation: i32,
        rope_mscale: f32,
        attn_scale: f32,
        attn_softcap: f32,
        pre_ffn_norm: *const f32,
        post_ffn_norm: *const f32,
        // Non-Llama block shape (GPT-2, Falcon); null for the Llama default.
        ext: *const DlmBlockExt,
    ) -> i32;

    /// MoE layer, part 3 (**grouped**): apply all `n_experts` selected experts in
    /// three launches instead of three per expert. Equivalent to calling
    /// [`dlm_apply_expert`] once per expert; every expert must share
    /// `w_dtype`/`w_group_size`. Returns `cudaErrorInvalidValue` if `n_experts`
    /// exceeds [`DLM_MAX_TOPK`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_apply_experts(
        hidden_size: i32,
        inter: i32,
        n_experts: i32,
        w_dtype: i32,
        w_group_size: i32,
        gate: *const DlmPtrs,
        up: *const DlmPtrs,
        down: *const DlmPtrs,
        weights: *const DlmWeights,
        x: *mut f32,
        activation: i32,
    ) -> i32;

    /// MoE layer, part 3: `x += weight · down·(silu(gate·normed2) ⊙ up·normed2)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dlm_apply_expert(
        hidden_size: i32,
        inter: i32,
        w_dtype: i32,
        w_group_size: i32,
        gate: *const c_void,
        up: *const c_void,
        down: *const c_void,
        weight: f32,
        x: *mut f32,
        // Gate activation (DLM_ACT_*): SiLU (0) or GELU (1, Gemma).
        activation: i32,
    ) -> i32;
}

/// Maximum experts one grouped launch can take — mirrors `DLM_MAX_TOPK` in
/// `src/gpu/kernels.cu`. Real models route 2..8; above this the caller falls back
/// to the per-expert loop.
pub(crate) const DLM_MAX_TOPK: usize = 16;

/// Expert weight pointers passed **by value** to the grouped kernels. CUDA copies
/// kernel parameters to the device, so the pointer table needs no separate
/// upload. Layout must match `DlmPtrs` in `src/gpu/kernels.cu`.
#[repr(C)]
pub(crate) struct DlmPtrs {
    pub(crate) p: [*const c_void; DLM_MAX_TOPK],
}

/// Per-expert gate weights, passed by value alongside [`DlmPtrs`]. Matches
/// `DlmWeights` in `src/gpu/kernels.cu`.
#[repr(C)]
pub(crate) struct DlmWeights {
    pub(crate) w: [f32; DLM_MAX_TOPK],
}

/// Maximum sequences one batched block can take — mirrors `DLM_MAX_BATCH` in
/// `src/gpu/kernels.cu`. Above this the caller falls back to per-slot decoding.
pub(crate) const DLM_MAX_BATCH: usize = 16;

/// Per-slot device KV pointers, passed by value to the batched kernel. Matches
/// `DlmSlots` in `src/gpu/kernels.cu`.
#[repr(C)]
pub(crate) struct DlmSlots {
    pub(crate) p: [*mut f32; DLM_MAX_BATCH],
}

/// Per-slot KV block tables. Matches `DlmTables` in `src/gpu/kernels.cu`.
#[repr(C)]
pub(crate) struct DlmTables {
    pub(crate) p: [*const i32; DLM_MAX_BATCH],
}

/// Everything a `dlm_decode_block_batched` call needs to address each slot's KV:
/// buffers, block table, history length and RoPE position.
pub(crate) struct SlotKv {
    pub(crate) keys: DlmSlots,
    pub(crate) values: DlmSlots,
    pub(crate) tables: DlmTables,
    pub(crate) num_positions: DlmInts,
    pub(crate) positions: DlmInts,
    pub(crate) half: bool,
    len: usize,
}

impl SlotKv {
    pub(crate) fn new() -> Self {
        Self {
            keys: DlmSlots {
                p: [std::ptr::null_mut(); DLM_MAX_BATCH],
            },
            values: DlmSlots {
                p: [std::ptr::null_mut(); DLM_MAX_BATCH],
            },
            tables: DlmTables {
                p: [std::ptr::null(); DLM_MAX_BATCH],
            },
            num_positions: DlmInts {
                v: [0; DLM_MAX_BATCH],
            },
            positions: DlmInts {
                v: [0; DLM_MAX_BATCH],
            },
            half: false,
            len: 0,
        }
    }

    /// Slots added so far.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Add a slot writing its K/V at logical row `row` (its history length) and
    /// rotating at `position`.
    pub(crate) fn push(&mut self, kv: &crate::forward::cpu::DeviceKv, row: usize, position: usize) {
        let b = self.len;
        self.keys.p[b] = kv.keys;
        self.values.p[b] = kv.values;
        self.tables.p[b] = kv.table;
        self.num_positions.v[b] = row as i32;
        self.positions.v[b] = position as i32;
        self.half = kv.half;
        self.len += 1;
    }
}

/// Block shape beyond the Llama default. Matches `DlmBlockExt` in
/// `src/gpu/kernels.cu`, field for field.
#[repr(C)]
pub(crate) struct DlmBlockExt {
    layer_norm: i32,
    gated: i32,
    rope: i32,
    parallel: i32,
    in_norm_bias: *const f32,
    post_norm_bias: *const f32,
    o_bias: *const f32,
    up_bias: *const f32,
    down_bias: *const f32,
    /// 1: the KV buffers hold fp16 (see [`KvLayerCache::device_half`]).
    kv_half: i32,
    /// The sequence's KV block table (single-sequence calls; batched calls pass
    /// one per slot).
    kv_table: *const i32,
    kv_block_shift: i32,
}

impl DlmBlockExt {
    /// The device block options for `cfg`, pointing at the layer's biases. Every
    /// field falls out of the config or a bias's presence, so a Llama-shaped
    /// layer yields exactly the kernel's default.
    pub(crate) fn new(
        cfg: &BlockConfig,
        b: &GpuBlockBiases,
        kv_half: bool,
        kv_table: *const i32,
    ) -> Self {
        use crate::forward::cpu::{FfnKind, NormKind};
        Self {
            layer_norm: (cfg.norm_kind == NormKind::Layer) as i32,
            gated: (cfg.ffn_kind == FfnKind::Gated) as i32,
            rope: (!cfg.learned_positions) as i32,
            parallel: cfg.parallel_residual as i32,
            in_norm_bias: bias_ptr(&b.in_norm),
            post_norm_bias: bias_ptr(&b.post_norm),
            o_bias: bias_ptr(&b.o),
            up_bias: bias_ptr(&b.up),
            down_bias: bias_ptr(&b.down),
            kv_half: kv_half as i32,
            kv_table,
            kv_block_shift: crate::forward::kv_pool::KV_BLOCK_SHIFT as i32,
        }
    }
}

/// The biases a Llama block does not have: LayerNorm's (GPT-2, Falcon), the
/// output projection's, and the MLP's. All `None` elsewhere.
#[derive(Default)]
pub(crate) struct GpuBlockBiases {
    pub(crate) in_norm: Option<DeviceBuffer>,
    pub(crate) post_norm: Option<DeviceBuffer>,
    pub(crate) o: Option<DeviceBuffer>,
    pub(crate) up: Option<DeviceBuffer>,
    pub(crate) down: Option<DeviceBuffer>,
}

impl GpuBlockBiases {
    pub(crate) fn upload(t: &LayerTensors) -> Result<Self> {
        let (up, down) = match &t.ffn {
            crate::forward::Ffn::Dense(f) => (f.up_bias.as_ref(), f.down_bias.as_ref()),
            crate::forward::Ffn::Moe { .. } => (None, None),
        };
        Ok(Self {
            in_norm: upload_bias(t.input_layernorm_bias.as_ref())?,
            post_norm: upload_bias(t.post_attention_layernorm_bias.as_ref())?,
            o: upload_bias(t.o_bias.as_ref())?,
            up: upload_bias(up)?,
            down: upload_bias(down)?,
        })
    }
}

/// Per-slot integers (history length, RoPE position). Matches `DlmInts`.
#[repr(C)]
pub(crate) struct DlmInts {
    pub(crate) v: [i32; DLM_MAX_BATCH],
}

/// Upload an optional bias, or `None` when the checkpoint has none. The kernel
/// takes a NULL pointer to mean "no bias".
///
/// Biases are a few KB against a layer's several MB of matrices, so they go up
/// synchronously rather than through the streaming path's pinned staging buffer.
pub(crate) fn upload_bias(bias: Option<&Vec<f32>>) -> Result<Option<DeviceBuffer>> {
    match bias {
        Some(b) => Ok(Some(DeviceBuffer::from_slice(b)?)),
        None => Ok(None),
    }
}

/// Device pointer for an optional buffer — NULL when absent.
pub(crate) fn bias_ptr(b: &Option<DeviceBuffer>) -> *const f32 {
    b.as_ref().map_or(std::ptr::null(), |d| d.as_ptr())
}

/// The LM head resident in VRAM. Decoding ends every step with a
/// `vocab × hidden` GEMV -- 233M multiply-adds for a 151,936-token vocabulary at
/// hidden 1536 -- which on the host costs more than the whole GPU layer stack
/// of a small model.
pub struct GpuLmHead {
    /// The GPU holding `w`, when that is not simply the current device. A
    /// multi-GPU pipeline leaves whichever stage ran last current, so the head
    /// must select its own device before launching on its own buffer.
    device: Option<u32>,
    w: DeviceBuffer,
    w_dtype: i32,
    w_group_size: i32,
    vocab: usize,
    hidden: usize,
}

impl GpuLmHead {
    /// Upload `head` (row-major `[vocab, hidden]`, already in the precision the
    /// caller chose) to `device`, or to the current device when `None`.
    pub fn new(
        head: &crate::forward::Weights,
        vocab: usize,
        hidden: usize,
        device: Option<u32>,
    ) -> Result<Self> {
        if head.len() != vocab * hidden {
            return Err(DlmError::ShapeMismatch {
                expected: vocab * hidden,
                got: head.len(),
            });
        }
        if let Some(id) = device {
            crate::gpu::set_device(id)?;
        }
        Ok(Self {
            device,
            w: upload_weight(head)?,
            w_dtype: head.dtype_code(),
            w_group_size: head.group_size() as i32,
            vocab,
            hidden,
        })
    }

    /// Logits for the final-normed hidden state `x`.
    pub fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.hidden {
            return Err(DlmError::ShapeMismatch {
                expected: self.hidden,
                got: x.len(),
            });
        }
        if let Some(id) = self.device {
            crate::gpu::set_device(id)?;
        }
        let mut out = vec![0.0f32; self.vocab];
        // SAFETY: `w` is a live device buffer of `vocab × hidden` weights in
        // `w_dtype`; `x` and `out` are host slices of the lengths passed.
        let code = unsafe {
            dlm_lm_head(
                self.vocab as i32,
                self.hidden as i32,
                self.w_dtype,
                self.w_group_size,
                self.w.as_ptr() as *const c_void,
                x.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(DlmError::Gpu {
                api: "dlm_lm_head",
                code,
            });
        }
        Ok(out)
    }
}

/// One layer's weights plus its persistent K/V history, resident in VRAM.
struct GpuLayer {
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: DeviceBuffer,
    o_proj: DeviceBuffer,
    gate_proj: DeviceBuffer,
    up_proj: DeviceBuffer,
    down_proj: DeviceBuffer,
    input_layernorm: DeviceBuffer,
    post_attention_layernorm: DeviceBuffer,
    q_bias: Option<DeviceBuffer>,
    k_bias: Option<DeviceBuffer>,
    v_bias: Option<DeviceBuffer>,
    /// Qwen3 per-head Q/K RMSNorm weights (NULL to the kernel when absent).
    q_norm: Option<DeviceBuffer>,
    k_norm: Option<DeviceBuffer>,
    /// MLA (DeepSeek) attention projections; `None` for standard attention (when
    /// set, `q_proj`/`k_proj`/`v_proj` are unused dummies).
    mla: Option<GpuMla>,
    /// Gemma2's extra FFN norm pair; `None` elsewhere. Their presence switches
    /// the device block to the Gemma2 norm placement.
    pre_ffn_norm: Option<DeviceBuffer>,
    post_ffn_norm: Option<DeviceBuffer>,
    /// LayerNorm, output-projection and MLP biases (GPT-2, Falcon).
    biases: GpuBlockBiases,
    /// Native dtype of this layer's projection weights (see `Weights::dtype_code`).
    w_dtype: i32,
    /// Group size for int4 weights; 0 for the float dtypes.
    w_group_size: i32,
}

/// MLA attention projections resident in VRAM. Shared with the streaming kernel,
/// which runs the same `dlm_mla_attn` call on a streamed layer core.
pub(crate) struct GpuMla {
    pub(crate) q_a_proj: Option<DeviceBuffer>,
    pub(crate) q_a_layernorm: Option<DeviceBuffer>,
    pub(crate) q_b_proj: DeviceBuffer,
    pub(crate) kv_a_proj: DeviceBuffer,
    pub(crate) kv_a_layernorm: DeviceBuffer,
    pub(crate) kv_b_proj: DeviceBuffer,
}

impl GpuMla {
    pub(crate) fn upload(mw: &crate::forward::MlaWeights) -> Result<Self> {
        let up = |w: &crate::forward::Weights| DeviceBuffer::from_bytes(w.as_bytes(), w.len());
        Ok(Self {
            q_a_proj: mw.q_a_proj.as_ref().map(up).transpose()?,
            q_a_layernorm: upload_bias(mw.q_a_layernorm.as_ref())?,
            q_b_proj: up(&mw.q_b_proj)?,
            kv_a_proj: up(&mw.kv_a_proj)?,
            kv_a_layernorm: DeviceBuffer::from_slice(&mw.kv_a_layernorm)?,
            kv_b_proj: up(&mw.kv_b_proj)?,
        })
    }
}

/// Upload a weight matrix, or a 1-element dummy for an intentionally-empty one
/// (MLA layers leave q/k/v empty) — device allocators dislike zero-length buffers.
pub(crate) fn upload_weight(w: &crate::forward::Weights) -> Result<DeviceBuffer> {
    if w.is_empty() {
        DeviceBuffer::from_slice(&[0.0])
    } else {
        DeviceBuffer::from_bytes(w.as_bytes(), w.len())
    }
}

impl GpuLayer {
    fn upload(t: &LayerTensors) -> Result<Self> {
        let ffn = t.dense_ffn()?;
        // MLA layers carry their attention dtype on the MLA weights, not q_proj.
        let (w_dtype, w_group_size) = match &t.mla {
            Some(mw) => (mw.q_b_proj.dtype_code(), mw.q_b_proj.group_size() as i32),
            None => (t.q_proj.dtype_code(), t.q_proj.group_size() as i32),
        };
        Ok(Self {
            q_proj: upload_weight(&t.q_proj)?,
            k_proj: upload_weight(&t.k_proj)?,
            v_proj: upload_weight(&t.v_proj)?,
            o_proj: upload_weight(&t.o_proj)?,
            gate_proj: upload_weight(&ffn.gate)?,
            up_proj: upload_weight(&ffn.up)?,
            down_proj: upload_weight(&ffn.down)?,
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
            w_dtype,
            w_group_size,
        })
    }
}

/// A GPU [`ComputeKernel`] holding a model's weights (and its KV history) in VRAM.
pub struct GpuKernel {
    cfg: BlockConfig,
    layers: Vec<GpuLayer>,
    /// RoPE inverse frequencies, computed once by the same host function the CPU
    /// oracle uses ([`rope_inv_freqs`]) and kept resident so the kernel indexes
    /// it instead of recomputing (and possibly diverging from) the formula.
    inv_freq: DeviceBuffer,
    /// Gemma3's local-layer frequencies; `None` when every layer shares `inv_freq`.
    local_inv_freq: Option<DeviceBuffer>,
    /// Max tokens one session's KV can hold.
    kv_capacity_tokens: usize,
    /// Paged KV blocks shared by every session, one pool per layer.
    kv_pools: crate::forward::kv_pool::KvPools,
    /// Persistent device buffer for the hidden vector, reused across every layer
    /// and token. `run_block` uploads the host hidden into it and downloads the
    /// result back — reusing one allocation instead of cudaMalloc/cudaFree per
    /// layer (a synchronizing driver call in the hot path). Sound because a given
    /// kernel instance is driven by a single inference thread, one layer at a time.
    d_hidden: DeviceBuffer,
    /// `[DLM_MAX_BATCH, hidden]` block that `decode_batch` keeps a batch's
    /// hidden states in across the layer stack.
    d_batch: DeviceBuffer,
    /// MLA decoupled-RoPE frequencies (over `qk_rope_head_dim`); `None` for
    /// standard attention.
    mla_inv_freq: Option<DeviceBuffer>,
}

impl GpuKernel {
    /// Upload a model's weights to the device and allocate per-layer KV history
    /// buffers sized for up to `max_kv_tokens` positions.
    pub fn new(cfg: BlockConfig, layers: Vec<LayerTensors>, max_kv_tokens: usize) -> Result<Self> {
        // The resident kernel has no routed-expert path at all (a `GpuLayer` holds
        // one dense FFN), so MoE — with MLA or not — belongs to the streaming
        // kernel, which caches experts per `(layer, expert)`. That path supports
        // MLA + MoE, so this is a routing error, not a missing capability.
        if cfg.mla.is_some() && cfg.moe.is_some() {
            return Err(DlmError::InvalidConfig(
                "MLA + MoE (DeepSeek-V2/V3) needs the streaming GPU path, which caches routed \
                 experts per layer; the resident GPU kernel holds only a dense FFN per layer. \
                 Run without --no-stream (or on CPU with --device cpu)."
                    .into(),
            ));
        }
        let cap = max_kv_tokens.max(1);
        let mut gpu_layers = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            // Per-layer config, so a MoE model's dense prefix layers (DeepSeek's
            // `first_k_dense_replace`) validate as the dense blocks they are.
            layer.validate(&cfg.for_layer(i as u32))?;
            gpu_layers.push(GpuLayer::upload(layer)?);
        }
        let inv_freq = DeviceBuffer::from_slice(&crate::forward::cpu::rope_inv_freqs(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_scaling,
        ))?;
        // Gemma3's windowed layers rotate with their own base, unscaled.
        let local_inv_freq = cfg
            .rope_local_theta
            .map(|theta| {
                DeviceBuffer::from_slice(&crate::forward::cpu::rope_inv_freqs(
                    cfg.head_dim,
                    theta,
                    None,
                ))
            })
            .transpose()?;
        // MLA rotates only its qk_rope sub-dimension.
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
        // One session's worth by default; `with_kv_pool_tokens` sizes it for many.
        let kv_pools = crate::forward::kv_pool::KvPools::new(gpu_layers.len(), cfg.kv_dim(), cap);
        Ok(Self {
            cfg,
            kv_pools,
            layers: gpu_layers,
            inv_freq,
            local_inv_freq,
            kv_capacity_tokens: cap,
            d_hidden,
            d_batch,
            mla_inv_freq,
        })
    }
}

impl GpuKernel {
    /// Size the shared KV pools to hold `tokens` tokens across all sessions (per
    /// layer). Call before the first session runs: the pools are allocated then.
    pub fn with_kv_pool_tokens(mut self, tokens: usize) -> Self {
        self.set_kv_pool_tokens(tokens);
        self
    }

    /// The RoPE inverse frequencies `layer` rotates with.
    fn inv_freq_for(&self, layer: u32) -> *const f32 {
        match &self.local_inv_freq {
            Some(local) if self.cfg.uses_local_rope(layer) => local.as_ptr(),
            _ => self.inv_freq.as_ptr(),
        }
    }

    /// One `dlm_decode_block_batched` call over the `[batch, hidden]` device block
    /// at `x`, for dense standard-attention layer `layer`. The caller owns the
    /// slot tables and the host KV length bookkeeping.
    #[allow(clippy::too_many_arguments)]
    fn launch_block_batched(&self, layer: u32, x: *mut f32, slots: &SlotKv) -> Result<()> {
        let w = &self.layers[layer as usize];
        let cfg = self.cfg.for_layer(layer);
        let hidden_size = cfg.hidden_size;
        let kv_dim = cfg.kv_dim();
        // SAFETY: all pointers are live device allocations of the sizes the kernel
        // expects; each slot's KV has room for its `num_positions + 1`-th row.
        let code = unsafe {
            dlm_decode_block_batched(
                hidden_size as i32,
                cfg.q_dim() as i32,
                kv_dim as i32,
                cfg.num_heads as i32,
                cfg.num_kv_heads as i32,
                cfg.head_dim as i32,
                cfg.intermediate_size as i32,
                cfg.rms_eps,
                w.w_dtype,
                w.w_group_size,
                w.q_proj.as_ptr() as *const c_void,
                w.k_proj.as_ptr() as *const c_void,
                w.v_proj.as_ptr() as *const c_void,
                w.o_proj.as_ptr() as *const c_void,
                w.gate_proj.as_ptr() as *const c_void,
                w.up_proj.as_ptr() as *const c_void,
                w.down_proj.as_ptr() as *const c_void,
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
                slots.len as i32,
                cfg.sliding_window.unwrap_or(0) as i32,
                cfg.activation.code(),
                crate::forward::cpu::rope_mscale(cfg.rope_scaling),
                cfg.attn_scale(),
                cfg.attn_logit_softcap.unwrap_or(0.0),
                bias_ptr(&w.pre_ffn_norm),
                bias_ptr(&w.post_ffn_norm),
                &DlmBlockExt::new(&cfg, &w.biases, slots.half, std::ptr::null()),
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
}

impl ComputeKernel for GpuKernel {
    fn kv_free_tokens(&self, fp16: bool) -> Option<usize> {
        Some(self.kv_pools.free_tokens(fp16))
    }

    fn set_kv_pool_tokens(&mut self, tokens: usize) {
        self.kv_pools =
            crate::forward::kv_pool::KvPools::new(self.layers.len(), self.cfg.kv_dim(), tokens);
    }

    fn num_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    fn kv_dim(&self) -> usize {
        self.cfg.kv_dim()
    }

    /// Decode `hiddens.len()` sequences through one layer in a single device
    /// call, so each weight row crosses memory once for the whole batch instead
    /// of once per sequence — decode is bound on exactly those reads.
    ///
    /// Falls back to the default per-slot loop whenever the batched kernel does
    /// not apply: MLA layers (a different attention path), batches beyond
    /// [`DLM_MAX_BATCH`], and the degenerate single-sequence case, where batching
    /// only adds a staging copy.
    fn run_block_batched(
        &self,
        layer: u32,
        hiddens: &mut [&mut [f32]],
        kvs: &mut [&mut KvLayerCache],
        positions: &[usize],
    ) -> Result<()> {
        let batch = hiddens.len();
        if batch <= 1 || batch > DLM_MAX_BATCH || self.cfg.mla.is_some() {
            // One layer for each sequence in turn, so every hidden makes its own
            // round trip: `run_block` would leave it on the shared device buffer
            // for its next layer, where the next sequence overwrites it.
            for ((hidden, kv), &position) in hiddens.iter_mut().zip(kvs.iter_mut()).zip(positions) {
                self.d_hidden.upload(hidden)?;
                self.launch_layer(layer, kv, position, &self.d_hidden)?;
                self.d_hidden.download(hidden)?;
            }
            return Ok(());
        }
        let hidden_size = self.cfg.hidden_size;

        // Stage every slot's hidden into one contiguous [batch, hidden] block.
        let mut staged = vec![0.0f32; batch * hidden_size];
        for (b, hidden) in hiddens.iter().enumerate() {
            if hidden.len() != hidden_size {
                return Err(DlmError::ShapeMismatch {
                    expected: hidden_size,
                    got: hidden.len(),
                });
            }
            staged[b * hidden_size..(b + 1) * hidden_size].copy_from_slice(hidden);
        }
        let mut slots = SlotKv::new();
        for (b, kv) in kvs.iter_mut().enumerate() {
            let n = kv.len();
            let dkv = kv.device_kv(
                &self.kv_pools,
                layer as usize,
                n + 1,
                self.kv_capacity_tokens,
                true,
            )?;
            slots.push(&dkv, n, positions[b]);
        }

        let d_batch = DeviceBuffer::from_slice(&staged)?;
        self.launch_block_batched(layer, d_batch.as_mut_ptr(), &slots)?;
        d_batch.download(&mut staged)?;
        for (b, hidden) in hiddens.iter_mut().enumerate() {
            hidden.copy_from_slice(&staged[b * hidden_size..(b + 1) * hidden_size]);
        }
        // Keep each orchestrator's length bookkeeping in step (real K/V is in VRAM).
        for kv in kvs.iter_mut() {
            kv.advance(1);
        }
        Ok(())
    }

    /// Prefill a dense model layer by layer, up to [`DLM_MAX_BATCH`] tokens per
    /// device call. Every slot of a call points at the **same** layer KV, with
    /// `num_positions = history + b`: the batched block writes slot `b`'s K/V
    /// and then attends over rows `[0, history + b]`, in slot order on one
    /// stream, so each token sees exactly the tokens before it — causal prefill
    /// out of the decode kernel, with each projection read once per chunk. The
    /// hidden states stay on the device for the whole stack.
    ///
    /// MLA layers have no batched block and keep the per-token order.
    fn prefill(
        &self,
        hiddens: &mut [f32],
        kv_layers: &mut [KvLayerCache],
        start: usize,
    ) -> Result<()> {
        let hidden_size = self.cfg.hidden_size;
        let n = hiddens.len() / hidden_size;
        if n <= 1 || self.cfg.mla.is_some() {
            return crate::forward::kernel::prefill_token_by_token(self, hiddens, kv_layers, start);
        }
        let d_hiddens = DeviceBuffer::from_slice(hiddens)?;
        for layer in 0..self.num_layers() {
            let kv = &mut kv_layers[layer as usize];
            for chunk in (0..n).step_by(DLM_MAX_BATCH) {
                let batch = (n - chunk).min(DLM_MAX_BATCH);
                let history = kv.len();
                if history + batch > self.kv_capacity_tokens {
                    return Err(DlmError::InvalidConfig(format!(
                        "GPU KV capacity {} exceeded at position {}",
                        self.kv_capacity_tokens,
                        start + chunk + batch - 1
                    )));
                }
                let dkv = kv.device_kv(
                    &self.kv_pools,
                    layer as usize,
                    history + batch,
                    self.kv_capacity_tokens,
                    true,
                )?;
                let mut slots = SlotKv::new();
                for b in 0..batch {
                    slots.push(&dkv, history + b, start + chunk + b);
                }
                // SAFETY: `chunk + batch <= n`, so the offset block lies inside
                // the `n × hidden` device buffer.
                let x = unsafe { d_hiddens.as_mut_ptr().add(chunk * hidden_size) };
                self.launch_block_batched(layer, x, &slots)?;
                kv.advance(batch);
            }
        }
        d_hiddens.download(hiddens)
    }

    fn decode_batch(
        &self,
        hiddens: &mut [&mut [f32]],
        kv_layers: &mut [&mut [KvLayerCache]],
        positions: &[usize],
    ) -> Result<()> {
        // MLA has no batched block; a single sequence gains nothing from one.
        if hiddens.len() <= 1 || self.cfg.mla.is_some() {
            for ((hidden, kvs), &position) in
                hiddens.iter_mut().zip(kv_layers.iter_mut()).zip(positions)
            {
                for layer in 0..self.num_layers() {
                    self.run_block(layer, hidden, &mut kvs[layer as usize], position)?;
                }
            }
            return Ok(());
        }
        let h = self.cfg.hidden_size;
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
            // Upload once; every layer reads and writes the device block in place.
            self.d_batch.upload(&staged[..batch * h])?;
            for layer in 0..self.num_layers() {
                let mut slots = SlotKv::new();
                for (b, seq) in kvs.iter_mut().enumerate() {
                    let kv = &mut seq[layer as usize];
                    let n = kv.len();
                    let dkv = kv.device_kv(
                        &self.kv_pools,
                        layer as usize,
                        n + 1,
                        self.kv_capacity_tokens,
                        true,
                    )?;
                    slots.push(&dkv, n, pos[b]);
                }
                self.launch_block_batched(layer, self.d_batch.as_mut_ptr(), &slots)?;
                for seq in kvs.iter_mut() {
                    seq[layer as usize].advance(1);
                }
            }
            // A blocking copy, so it also waits for the stack's kernels.
            self.d_batch.download(&mut staged[..batch * h])?;
            for (b, hidden) in hs.iter_mut().enumerate() {
                hidden.copy_from_slice(&staged[b * h..(b + 1) * h]);
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
        // The hidden state stays resident on the device for the whole layer stack:
        // upload the host vector once before the first layer, chain every layer
        // through the same persistent buffer on the (in-order) default stream, and
        // download once after the last layer. The orchestrator never reads `hidden`
        // between layers, so the per-layer host round-trip + sync it used to do
        // was pure overhead — ~2/3 of the per-token GPU time on a 16-layer model.
        if layer == 0 {
            self.d_hidden.upload(hidden)?;
        }
        self.launch_layer(layer, kv, position, &self.d_hidden)?;
        // `download` is a blocking cudaMemcpy(D2H), so it already waits for the
        // stack's kernels — no separate synchronize() needed.
        if layer as usize == self.layers.len() - 1 {
            self.d_hidden.download(hidden)?;
        }
        Ok(())
    }
}

impl GpuKernel {
    /// Enqueue layer `layer` for one token on the hidden vector in `d_hidden`,
    /// appending its K/V to `kv`. Nothing is copied to or from the host.
    fn launch_layer(
        &self,
        layer: u32,
        kv: &mut KvLayerCache,
        position: usize,
        d_hidden: &DeviceBuffer,
    ) -> Result<()> {
        let w = &self.layers[layer as usize];
        let kv_dim = self.cfg.kv_dim();

        // Prior sequence length. The device buffers are overwritten from slot 0
        // each sequence (attention only reads slots [0, num_positions+1)), so no
        // per-generation reset is needed — stale slots beyond the window are
        // never read.
        let num_positions = kv.len();
        if num_positions >= self.kv_capacity_tokens {
            return Err(DlmError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {position}",
                self.kv_capacity_tokens
            )));
        }

        // Per-session K/V (owned by this sequence's KvLayerCache, not the shared
        // kernel), so concurrent batched requests keep independent history.
        // MLA reconstructs K and V from one cached latent — no value cache.
        let dkv = kv.device_kv(
            &self.kv_pools,
            layer as usize,
            num_positions + 1,
            self.kv_capacity_tokens,
            self.cfg.mla.is_none(),
        )?;
        let (kv_keys, kv_values) = (dkv.keys, dkv.values);

        // SAFETY: all pointers are live device allocations of the sizes the
        // kernel expects; kv buffers have capacity for `num_positions + 1`.
        match (&self.cfg.mla, &w.mla) {
            (Some(m), Some(mw)) => {
                // MLA attention (compressed-latent) + a separate dense FFN.
                let code = unsafe {
                    dlm_mla_attn(
                        self.cfg.hidden_size as i32,
                        self.cfg.num_heads as i32,
                        m.q_lora_rank.unwrap_or(0) as i32,
                        m.kv_lora_rank as i32,
                        m.qk_nope_head_dim as i32,
                        m.qk_rope_head_dim as i32,
                        m.v_head_dim as i32,
                        self.cfg.rms_eps,
                        w.w_dtype,
                        w.w_group_size,
                        mw.q_a_proj
                            .as_ref()
                            .map_or(std::ptr::null(), |b| b.as_ptr())
                            as *const c_void,
                        bias_ptr(&mw.q_a_layernorm),
                        mw.q_b_proj.as_ptr() as *const c_void,
                        mw.kv_a_proj.as_ptr() as *const c_void,
                        mw.kv_a_layernorm.as_ptr(),
                        mw.kv_b_proj.as_ptr() as *const c_void,
                        w.o_proj.as_ptr() as *const c_void,
                        w.input_layernorm.as_ptr(),
                        self.mla_inv_freq.as_ref().unwrap().as_ptr(),
                        crate::forward::cpu::rope_mscale(self.cfg.rope_scaling),
                        d_hidden.as_mut_ptr(),
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
                let code = unsafe {
                    dlm_dense_ffn(
                        self.cfg.hidden_size as i32,
                        self.cfg.intermediate_size as i32,
                        self.cfg.rms_eps,
                        w.w_dtype,
                        w.w_group_size,
                        w.gate_proj.as_ptr() as *const c_void,
                        w.up_proj.as_ptr() as *const c_void,
                        w.down_proj.as_ptr() as *const c_void,
                        w.post_attention_layernorm.as_ptr(),
                        self.cfg.activation.code(),
                        d_hidden.as_mut_ptr(),
                    )
                };
                if code != 0 {
                    return Err(DlmError::Gpu {
                        api: "dlm_dense_ffn",
                        code,
                    });
                }
            }
            _ => {
                let code = unsafe {
                    dlm_decode_block(
                        self.cfg.hidden_size as i32,
                        self.cfg.q_dim() as i32,
                        kv_dim as i32,
                        self.cfg.num_heads as i32,
                        self.cfg.num_kv_heads as i32,
                        self.cfg.head_dim as i32,
                        self.cfg.intermediate_size as i32,
                        self.cfg.rms_eps,
                        w.w_dtype,
                        w.w_group_size,
                        w.q_proj.as_ptr() as *const c_void,
                        w.k_proj.as_ptr() as *const c_void,
                        w.v_proj.as_ptr() as *const c_void,
                        w.o_proj.as_ptr() as *const c_void,
                        w.gate_proj.as_ptr() as *const c_void,
                        w.up_proj.as_ptr() as *const c_void,
                        w.down_proj.as_ptr() as *const c_void,
                        w.input_layernorm.as_ptr(),
                        w.post_attention_layernorm.as_ptr(),
                        bias_ptr(&w.q_bias),
                        bias_ptr(&w.k_bias),
                        bias_ptr(&w.v_bias),
                        bias_ptr(&w.q_norm),
                        bias_ptr(&w.k_norm),
                        self.inv_freq_for(layer),
                        d_hidden.as_mut_ptr(),
                        kv_keys,
                        kv_values,
                        num_positions as i32,
                        position as i32,
                        // Gemma2 alternates windowed and global layers, so the
                        // window is resolved per layer here exactly as
                        // `run_block_batched` does — passing the unresolved
                        // `sliding_window` would clip the global layers too.
                        self.cfg.window_for_layer(layer).unwrap_or(0) as i32,
                        self.cfg.activation.code(),
                        crate::forward::cpu::rope_mscale(self.cfg.for_layer(layer).rope_scaling),
                        self.cfg.attn_scale(),
                        self.cfg.attn_logit_softcap.unwrap_or(0.0),
                        bias_ptr(&w.pre_ffn_norm),
                        bias_ptr(&w.post_ffn_norm),
                        &DlmBlockExt::new(&self.cfg, &w.biases, dkv.half, dkv.table),
                    )
                };
                if code != 0 {
                    return Err(DlmError::Gpu {
                        api: "dlm_decode_block",
                        code,
                    });
                }
            }
        }
        // Keep the orchestrator's length bookkeeping in step. The real K/V is in
        // VRAM.
        kv.advance(1);
        Ok(())
    }
}

/// GPU pipeline parallelism: one [`GpuKernel`] per device, each holding a
/// contiguous shard of the model's layers on its own GPU. `run_block` selects the
/// owning device (`cudaSetDevice`) and delegates with the shard-local layer index,
/// so a stage's weights and KV live on its device and only the `hidden`-wide
/// residual crosses the host between stages (the ring hand-off).
///
/// This is the real GPU multi-GPU path (the generic
/// [`PipelineParallelKernel`](crate::forward::PipelineParallelKernel) wraps the
/// *CPU* kernel with device-affinity plumbing for off-GPU testing).
pub struct MultiGpuKernel {
    stages: Vec<GpuKernel>,
    devices: Vec<u32>,
    /// Global layer index → (stage index, shard-local layer index).
    layer_map: Vec<(usize, usize)>,
}

impl MultiGpuKernel {
    /// Partition `layers` across `gpu_ids` into contiguous stages, uploading each
    /// stage's shard to its device (earlier stages absorb the remainder).
    pub fn new(
        cfg: BlockConfig,
        layers: Vec<LayerTensors>,
        gpu_ids: &[u32],
        max_kv_tokens: usize,
    ) -> Result<Self> {
        if gpu_ids.is_empty() {
            return Err(DlmError::InvalidConfig(
                "multi-gpu pipeline needs at least one gpu id".into(),
            ));
        }
        let num_layers = layers.len();
        let shards = crate::distributed::partition_layers(num_layers, gpu_ids.len());
        let mut layers = layers;
        let mut stages = Vec::with_capacity(shards.len());
        let mut layer_map = vec![(0usize, 0usize); num_layers];
        let mut offset = 0;
        for (stage_idx, shard) in shards.iter().enumerate() {
            let count = shard.end - shard.start;
            let shard_layers: Vec<LayerTensors> = layers.drain(0..count).collect();
            // Upload this shard onto its device (GpuKernel::new allocates on the
            // current device, so set it first).
            crate::gpu::set_device(gpu_ids[stage_idx])?;
            stages.push(GpuKernel::new(cfg, shard_layers, max_kv_tokens)?);
            for local in 0..count {
                layer_map[offset + local] = (stage_idx, local);
            }
            offset += count;
        }
        Ok(Self {
            stages,
            devices: gpu_ids.to_vec(),
            layer_map,
        })
    }
}

impl ComputeKernel for MultiGpuKernel {
    fn num_layers(&self) -> u32 {
        self.layer_map.len() as u32
    }

    fn kv_free_tokens(&self, fp16: bool) -> Option<usize> {
        self.stages
            .iter()
            .filter_map(|s| s.kv_free_tokens(fp16))
            .min()
    }

    fn set_kv_pool_tokens(&mut self, tokens: usize) {
        for stage in &mut self.stages {
            stage.set_kv_pool_tokens(tokens);
        }
    }

    fn hidden_size(&self) -> usize {
        self.stages[0].hidden_size()
    }

    fn kv_dim(&self) -> usize {
        self.stages[0].kv_dim()
    }

    fn run_block(
        &self,
        layer: u32,
        hidden: &mut [f32],
        kv: &mut KvLayerCache,
        position: usize,
    ) -> Result<()> {
        let (stage, local) = self.layer_map[layer as usize];
        // Make the owning GPU current, then run the layer there. The stage's KV
        // (per-session, allocated on first touch) lands on this device.
        crate::gpu::set_device(self.devices[stage])?;
        self.stages[stage].run_block(local as u32, hidden, kv, position)
    }
}

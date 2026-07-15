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
        q_proj: *const f32,
        k_proj: *const f32,
        v_proj: *const f32,
        o_proj: *const f32,
        gate_proj: *const f32,
        up_proj: *const f32,
        down_proj: *const f32,
        in_norm: *const f32,
        post_norm: *const f32,
        q_bias: *const f32,
        k_bias: *const f32,
        v_bias: *const f32,
        inv_freq: *const f32,
        x: *mut f32,
        kv_keys: *mut f32,
        kv_values: *mut f32,
        num_positions: i32,
        position: i32,
    ) -> i32;
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
    kv_keys: DeviceBuffer,
    kv_values: DeviceBuffer,
}

impl GpuLayer {
    fn upload(t: &LayerTensors, kv_buffer_len: usize) -> Result<Self> {
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
            q_bias: upload_bias(t.q_bias.as_ref())?,
            k_bias: upload_bias(t.k_bias.as_ref())?,
            v_bias: upload_bias(t.v_bias.as_ref())?,
            kv_keys: DeviceBuffer::new(kv_buffer_len)?,
            kv_values: DeviceBuffer::new(kv_buffer_len)?,
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
    /// Max tokens the per-layer KV buffers can hold.
    kv_capacity_tokens: usize,
    /// Persistent device buffer for the hidden vector, reused across every layer
    /// and token. `run_block` uploads the host hidden into it and downloads the
    /// result back — reusing one allocation instead of cudaMalloc/cudaFree per
    /// layer (a synchronizing driver call in the hot path). Sound because a given
    /// kernel instance is driven by a single inference thread, one layer at a time.
    d_hidden: DeviceBuffer,
}

impl GpuKernel {
    /// Upload a model's weights to the device and allocate per-layer KV history
    /// buffers sized for up to `max_kv_tokens` positions.
    pub fn new(cfg: BlockConfig, layers: Vec<LayerTensors>, max_kv_tokens: usize) -> Result<Self> {
        let cap = max_kv_tokens.max(1);
        let kv_buffer_len = cap * cfg.kv_dim();
        let mut gpu_layers = Vec::with_capacity(layers.len());
        for layer in &layers {
            layer.validate(&cfg)?;
            gpu_layers.push(GpuLayer::upload(layer, kv_buffer_len)?);
        }
        let inv_freq = DeviceBuffer::from_slice(&crate::forward::cpu::rope_inv_freqs(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_scaling,
        ))?;
        let d_hidden = DeviceBuffer::new(cfg.hidden_size)?;
        Ok(Self {
            cfg,
            layers: gpu_layers,
            inv_freq,
            kv_capacity_tokens: cap,
            d_hidden,
        })
    }
}

impl ComputeKernel for GpuKernel {
    fn num_layers(&self) -> u32 {
        self.layers.len() as u32
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

        // The hidden state stays resident on the device for the whole layer stack:
        // upload the host vector once before the first layer, chain every layer
        // through the same persistent buffer on the (in-order) default stream, and
        // download once after the last layer. The orchestrator never reads `hidden`
        // between layers, so the per-layer host round-trip + sync it used to do
        // was pure overhead — ~2/3 of the per-token GPU time on a 16-layer model.
        let is_first = layer == 0;
        let is_last = layer as usize == self.layers.len() - 1;
        let d_hidden = &self.d_hidden;
        if is_first {
            d_hidden.upload(hidden)?;
        }

        // SAFETY: all pointers are live device allocations of the sizes the
        // kernel expects; kv buffers have capacity for `num_positions + 1`.
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
                w.q_proj.as_ptr(),
                w.k_proj.as_ptr(),
                w.v_proj.as_ptr(),
                w.o_proj.as_ptr(),
                w.gate_proj.as_ptr(),
                w.up_proj.as_ptr(),
                w.down_proj.as_ptr(),
                w.input_layernorm.as_ptr(),
                w.post_attention_layernorm.as_ptr(),
                bias_ptr(&w.q_bias),
                bias_ptr(&w.k_bias),
                bias_ptr(&w.v_bias),
                self.inv_freq.as_ptr(),
                d_hidden.as_mut_ptr(),
                w.kv_keys.as_mut_ptr(),
                w.kv_values.as_mut_ptr(),
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
        // `download` is a blocking cudaMemcpy(D2H), so it already waits for the
        // stack's kernels — no separate synchronize() needed. Only the last layer
        // brings the result back to the host.
        if is_last {
            d_hidden.download(hidden)?;
        }

        // Keep the orchestrator's length bookkeeping in step. The real K/V is in
        // VRAM; these host placeholders are never read on the GPU path.
        kv.append(&vec![0.0; kv_dim], &vec![0.0; kv_dim])?;
        Ok(())
    }
}

//! GPU compute kernel — the device `run_block` (feature `cuda-kernels`).
//!
//! Structurally identical to [`CpuKernel`](crate::forward::CpuKernel): it holds a
//! model's per-layer weights in VRAM and implements [`ComputeKernel`] by running
//! one decode block per token. The transformer math lives in
//! `src/gpu/kernels.cu`, invoked through the `flip_decode_block` FFI entry point.
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

use crate::error::{FlipError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use crate::gpu::device::{synchronize, DeviceBuffer};

extern "C" {
    /// One decode block on the device (see `src/gpu/kernels.cu`). Returns a
    /// `cudaError_t` (0 == success). All pointers are device pointers; `kv_keys`
    /// and `kv_values` are persistent buffers written in place at slot
    /// `num_positions`.
    #[allow(clippy::too_many_arguments)]
    fn flip_decode_block(
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
            kv_keys: DeviceBuffer::new(kv_buffer_len)?,
            kv_values: DeviceBuffer::new(kv_buffer_len)?,
        })
    }
}

/// A GPU [`ComputeKernel`] holding a model's weights (and its KV history) in VRAM.
pub struct GpuKernel {
    cfg: BlockConfig,
    layers: Vec<GpuLayer>,
    /// Max tokens the per-layer KV buffers can hold.
    kv_capacity_tokens: usize,
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
        Ok(Self {
            cfg,
            layers: gpu_layers,
            kv_capacity_tokens: cap,
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
            return Err(FlipError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {position}",
                self.kv_capacity_tokens
            )));
        }

        // Only the hidden vector crosses the bus; KV lives in VRAM.
        let d_hidden = DeviceBuffer::from_slice(hidden)?;

        // SAFETY: all pointers are live device allocations of the sizes the
        // kernel expects; kv buffers have capacity for `num_positions + 1`.
        let code = unsafe {
            flip_decode_block(
                self.cfg.hidden_size as i32,
                self.cfg.q_dim() as i32,
                kv_dim as i32,
                self.cfg.num_heads as i32,
                self.cfg.num_kv_heads as i32,
                self.cfg.head_dim as i32,
                self.cfg.intermediate_size as i32,
                self.cfg.rope_theta,
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
                d_hidden.as_mut_ptr(),
                w.kv_keys.as_mut_ptr(),
                w.kv_values.as_mut_ptr(),
                num_positions as i32,
                position as i32,
            )
        };
        if code != 0 {
            return Err(FlipError::Gpu {
                api: "flip_decode_block",
                code,
            });
        }
        synchronize()?;
        d_hidden.download(hidden)?;

        // Keep the orchestrator's length bookkeeping in step. The real K/V is in
        // VRAM; these host placeholders are never read on the GPU path.
        kv.append(&vec![0.0; kv_dim], &vec![0.0; kv_dim])?;
        Ok(())
    }
}

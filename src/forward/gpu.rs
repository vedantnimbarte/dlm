//! GPU compute kernel — the device `run_block` (feature `cuda-kernels`).
//!
//! Structurally identical to [`CpuKernel`](crate::forward::CpuKernel): it holds a
//! model's per-layer weights (here resident in VRAM) and implements
//! [`ComputeKernel`] by running one decode block per token. The transformer math
//! lives in `src/gpu/kernels.cu` and is invoked through the `flip_decode_block`
//! FFI entry point.
//!
//! Per call it uploads the hidden state and the current K/V history to the
//! device, launches the block, and reads back the updated hidden state plus the
//! new token's K/V (appended to the host [`KvLayerCache`], which stays the
//! source of truth). This re-uploads KV each layer — correct but not optimal; a
//! production version keeps KV resident on device. Requires nvcc at build time
//! and a GPU at run time; the CPU kernel is the correctness oracle.

use crate::error::{FlipError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use crate::gpu::device::{synchronize, DeviceBuffer};

extern "C" {
    /// One decode block on the device (see `src/gpu/kernels.cu`). Returns a
    /// `cudaError_t` (0 == success). All pointers are device pointers.
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
        kv_keys: *const f32,
        kv_values: *const f32,
        num_positions: i32,
        position: i32,
        new_key: *mut f32,
        new_value: *mut f32,
    ) -> i32;
}

/// One layer's weights resident in device memory.
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
}

impl GpuLayer {
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

/// A GPU [`ComputeKernel`] holding a model's weights in VRAM.
pub struct GpuKernel {
    cfg: BlockConfig,
    layers: Vec<GpuLayer>,
}

impl GpuKernel {
    /// Upload a model's weights to the device. Validates dimensions first.
    pub fn new(cfg: BlockConfig, layers: Vec<LayerTensors>) -> Result<Self> {
        let mut gpu_layers = Vec::with_capacity(layers.len());
        for layer in &layers {
            layer.validate(&cfg)?;
            gpu_layers.push(GpuLayer::upload(layer)?);
        }
        Ok(Self {
            cfg,
            layers: gpu_layers,
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
        let num_positions = kv.len();

        // Stage hidden + current KV history in VRAM.
        let d_hidden = DeviceBuffer::from_slice(hidden)?;
        let d_keys = DeviceBuffer::from_slice(kv.keys())?;
        let d_values = DeviceBuffer::from_slice(kv.values())?;
        let d_new_key = DeviceBuffer::new(kv_dim)?;
        let d_new_value = DeviceBuffer::new(kv_dim)?;

        // SAFETY: all pointers are live device allocations of the sizes the
        // kernel expects (validated weight dims; hidden/kv sized above).
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
                d_keys.as_ptr(),
                d_values.as_ptr(),
                num_positions as i32,
                position as i32,
                d_new_key.as_mut_ptr(),
                d_new_value.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(FlipError::Gpu {
                api: "flip_decode_block",
                code,
            });
        }
        synchronize()?;

        // Read back the updated hidden state and the new token's K/V.
        d_hidden.download(hidden)?;
        let mut new_key = vec![0.0f32; kv_dim];
        let mut new_value = vec![0.0f32; kv_dim];
        d_new_key.download(&mut new_key)?;
        d_new_value.download(&mut new_value)?;
        kv.append(&new_key, &new_value)?;
        Ok(())
    }
}

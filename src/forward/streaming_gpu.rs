//! GPU layer-streaming compute kernel (`specs.md` §2.2/§3.2) — the combined path
//! that actually delivers the product thesis: **run a model larger than VRAM on
//! the GPU** by keeping only a bounded window of layer *weights* resident in VRAM
//! and streaming the rest in over PCIe on demand, while each layer's KV history
//! stays resident (the cache zone).
//!
//! It mirrors [`GpuKernel`](crate::forward::GpuKernel) op-for-op — same
//! `flip_decode_block` device call — but instead of uploading every layer's
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

use crate::error::{FlipError, Result};
use crate::forward::cpu::{BlockConfig, KvLayerCache, LayerTensors};
use crate::forward::kernel::ComputeKernel;
use crate::forward::streaming::LayerSource;
use crate::gpu::device::{synchronize, DeviceBuffer};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

extern "C" {
    // Same device entry point as `forward::gpu` (see `src/gpu/kernels.cu`).
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

/// A bounded LRU of device-resident weight sets (front of `order` = least recent).
struct WeightLru {
    capacity: usize,
    map: HashMap<u32, GpuWeights>,
    order: VecDeque<u32>,
}

impl WeightLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn touch(&mut self, layer: u32) {
        if let Some(pos) = self.order.iter().position(|&l| l == layer) {
            self.order.remove(pos);
        }
        self.order.push_back(layer);
    }

    fn insert(&mut self, layer: u32, weights: GpuWeights) {
        while self.map.len() >= self.capacity {
            if let Some(evict) = self.order.pop_front() {
                self.map.remove(&evict); // frees the VRAM buffers via Drop
            } else {
                break;
            }
        }
        self.map.insert(layer, weights);
        self.order.push_back(layer);
    }
}

/// A GPU [`ComputeKernel`] that streams a window of layer weights through VRAM
/// while keeping per-layer KV resident.
pub struct StreamingGpuKernel<S: LayerSource> {
    cfg: BlockConfig,
    source: S,
    num_layers: u32,
    kv_capacity_tokens: usize,
    kv: Vec<GpuKv>,
    weights: Mutex<WeightLru>,
}

impl<S: LayerSource> StreamingGpuKernel<S> {
    /// Build a streaming GPU kernel: allocate per-layer KV for up to
    /// `max_kv_tokens` positions and stream weights keeping at most
    /// `resident_layers` sets in VRAM.
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
        Ok(Self {
            cfg,
            source,
            num_layers,
            kv_capacity_tokens: cap,
            kv,
            weights: Mutex::new(WeightLru::new(resident_layers)),
        })
    }
}

impl<S: LayerSource> ComputeKernel for StreamingGpuKernel<S> {
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
        let kv_dim = self.cfg.kv_dim();
        let num_positions = kv.len();
        if num_positions >= self.kv_capacity_tokens {
            return Err(FlipError::InvalidConfig(format!(
                "GPU KV capacity {} exceeded at position {position}",
                self.kv_capacity_tokens
            )));
        }

        // Ensure this layer's weights are resident, streaming them in on a miss.
        let mut cache = self.weights.lock().unwrap();
        if cache.map.contains_key(&layer) {
            cache.touch(layer);
        } else {
            let tensors = self.source.load_layer(layer)?;
            cache.insert(layer, GpuWeights::upload(&tensors)?);
        }
        let w = cache.map.get(&layer).expect("just inserted/looked up");
        let dkv = &self.kv[layer as usize];

        // Only the hidden vector crosses the bus per layer; weights are already
        // resident (this window) and KV lives on-device.
        let d_hidden = DeviceBuffer::from_slice(hidden)?;

        // SAFETY: all pointers are live device allocations sized as the kernel
        // expects; the KV buffers have capacity for `num_positions + 1`.
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
                dkv.keys.as_mut_ptr(),
                dkv.values.as_mut_ptr(),
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

        // Keep the orchestrator's length bookkeeping in step (real K/V is in VRAM).
        kv.append(&vec![0.0; kv_dim], &vec![0.0; kv_dim])?;
        Ok(())
    }
}

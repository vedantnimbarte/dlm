//! Load a runnable CPU model from a memory-mapped safetensors checkpoint.
//!
//! Bridges the storage layer to the compute path: it reads the standard
//! HuggingFace-named tensors out of an [`MmapStore`], converts them to `f32`
//! ([`bytes_to_f32`] handles F32/F16/BF16), and assembles a
//! [`Generator`] over a [`CpuKernel`] ready to run [`generate`].
//!
//! This is the connector that lets `dlm generate --model-path <dir>` run a real
//! (small) model on CPU. Quantized checkpoints (AWQ/GPTQ `qweight` triplets)
//! would be materialized through the [`quant`](crate::quant) dequant kernel here
//! — this loader covers the float dtypes small models ship in.
//!
//! [`bytes_to_f32`]: crate::storage::bytes_to_f32
//! [`generate`]: crate::generate::Generator::generate

use crate::cache::KvCacheConfig;
use crate::error::{DlmError, Result};
use crate::forward::{
    BlockConfig, CpuKernel, LayerSource, LayerTensors, PipelineParallelKernel, StreamingKernel,
};
use crate::generate::Generator;
use crate::model::ModelConfig;
use crate::quant::{dequantize_gptq_4bit, PackedQuantConfig};
use crate::storage::{bytes_to_f32, bytes_to_i32, MmapStore};

/// Read a named tensor as `f32`, verifying it has exactly `expected_len` elements.
fn load_tensor(store: &MmapStore, name: &str, expected_len: usize) -> Result<Vec<f32>> {
    let values = load_floats(store, name)?;
    if values.len() != expected_len {
        return Err(DlmError::InvalidConfig(format!(
            "tensor {name:?}: expected {expected_len} elements, got {}",
            values.len()
        )));
    }
    Ok(values)
}

/// Read a float tensor of unknown length.
fn load_floats(store: &MmapStore, name: &str) -> Result<Vec<f32>> {
    let (shard, info) = store
        .locate(name)
        .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;
    bytes_to_f32(shard.tensor_bytes(name)?, info.dtype)
}

/// Read an integer tensor of unknown length (packed `qweight`/`qzeros`).
fn load_ints(store: &MmapStore, name: &str) -> Result<Vec<i32>> {
    let (shard, info) = store
        .locate(name)
        .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;
    bytes_to_i32(shard.tensor_bytes(name)?, info.dtype)
}

/// Load a linear layer's weight as dense row-major `[out, in]`, transparently
/// handling both float (`{base}.weight`) and GPTQ-style quantized
/// (`{base}.qweight` + `.qzeros` + `.scales`) checkpoints.
fn load_linear(
    store: &MmapStore,
    base: &str,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    // Float weight: already row-major [out, in].
    let weight_name = format!("{base}.weight");
    if store.locate(&weight_name).is_some() {
        return load_tensor(store, &weight_name, out_features * in_features);
    }

    // GPTQ-style quantized triplet.
    let qweight_name = format!("{base}.qweight");
    if store.locate(&qweight_name).is_some() {
        let qweight = load_ints(store, &qweight_name)?;
        let qzeros = load_ints(store, &format!("{base}.qzeros"))?;
        let scales = load_floats(store, &format!("{base}.scales"))?;

        // Infer the group size from the scales shape ([in/group_size, out]).
        if scales.is_empty() || scales.len() % out_features != 0 {
            return Err(DlmError::QuantLayout(format!(
                "{base}.scales length {} is not a multiple of out_features {out_features}",
                scales.len()
            )));
        }
        let num_groups = scales.len() / out_features;
        if num_groups == 0 || in_features % num_groups != 0 {
            return Err(DlmError::QuantLayout(format!(
                "{base}: {num_groups} groups do not divide in_features {in_features}"
            )));
        }
        let cfg = PackedQuantConfig {
            in_features,
            out_features,
            group_size: in_features / num_groups,
        };
        return dequantize_gptq_4bit(&qweight, &qzeros, &scales, &cfg);
    }

    Err(DlmError::UnknownTensor(format!(
        "{base}.weight or {base}.qweight"
    )))
}

/// A model materialized to host `f32` weights, ready to wrap in a compute kernel
/// of the caller's choosing (CPU or GPU).
pub struct ModelParts {
    pub cfg: BlockConfig,
    pub layers: Vec<LayerTensors>,
    pub embedding: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub kv_config: KvCacheConfig,
    pub kv_blocks: u32,
}

impl ModelParts {
    /// Wrap the CPU kernel around these weights and build a generator.
    pub fn into_cpu_generator(self) -> Result<Generator<CpuKernel>> {
        let kernel = CpuKernel::new(self.cfg, self.layers)?;
        Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )
    }

    /// Upload every layer to VRAM and build a generator over the GPU kernel.
    #[cfg(feature = "cuda-kernels")]
    pub fn into_gpu_generator(self) -> Result<Generator<crate::forward::GpuKernel>> {
        let max_kv_tokens = self.kv_blocks as usize * self.kv_config.block_size as usize;
        let kernel = crate::forward::GpuKernel::new(self.cfg, self.layers, max_kv_tokens)?;
        Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )
    }

    /// Split the model's layers across `gpu_ids` (multi-GPU pipeline
    /// parallelism, `specs.md` §3.3) and build a generator over the resulting
    /// [`PipelineParallelKernel`]. Off-GPU it runs on the CPU kernel with the
    /// same layer partition, so output equals [`into_cpu_generator`].
    ///
    /// [`into_cpu_generator`]: ModelParts::into_cpu_generator
    pub fn into_pipeline_parallel_generator(
        self,
        gpu_ids: &[u32],
    ) -> Result<Generator<PipelineParallelKernel<CpuKernel>>> {
        let kernel = PipelineParallelKernel::new(CpuKernel::new(self.cfg, self.layers)?, gpu_ids)?;
        Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )
    }
}

/// Build a CPU [`Generator`] from a mapped checkpoint and its config.
///
/// Convenience wrapper over [`load_model_parts`] + [`ModelParts::into_cpu_generator`].
pub fn load_generator(
    store: &MmapStore,
    config: &ModelConfig,
    max_context: u32,
) -> Result<Generator<CpuKernel>> {
    load_model_parts(store, config, max_context)?.into_cpu_generator()
}

/// The per-block geometry a [`ModelConfig`] describes.
fn block_config(config: &ModelConfig) -> BlockConfig {
    BlockConfig {
        hidden_size: config.hidden_size as usize,
        num_heads: config.num_attention_heads as usize,
        num_kv_heads: config.num_kv_heads as usize,
        head_dim: config.head_dim() as usize,
        intermediate_size: config.intermediate_size as usize,
        rope_theta: config.rope_theta,
        rms_eps: config.rms_eps,
    }
}

/// Materialize one transformer layer's weights (`model.layers.{layer}.*`),
/// handling float and GPTQ-quantized projections. This is the unit the streaming
/// kernel pulls on demand and the resident loader pulls for every layer.
pub(crate) fn load_layer_tensors(
    store: &MmapStore,
    cfg: &BlockConfig,
    layer: u32,
) -> Result<LayerTensors> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let intermediate = cfg.intermediate_size;
    let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
    // load_linear takes (in_features, out_features) of the underlying Linear.
    let tensors = LayerTensors {
        q_proj: load_linear(store, &name("self_attn.q_proj"), hidden, q_dim)?,
        k_proj: load_linear(store, &name("self_attn.k_proj"), hidden, kv_dim)?,
        v_proj: load_linear(store, &name("self_attn.v_proj"), hidden, kv_dim)?,
        o_proj: load_linear(store, &name("self_attn.o_proj"), q_dim, hidden)?,
        gate_proj: load_linear(store, &name("mlp.gate_proj"), hidden, intermediate)?,
        up_proj: load_linear(store, &name("mlp.up_proj"), hidden, intermediate)?,
        down_proj: load_linear(store, &name("mlp.down_proj"), intermediate, hidden)?,
        input_layernorm: load_tensor(store, &name("input_layernorm.weight"), hidden)?,
        post_attention_layernorm: load_tensor(
            store,
            &name("post_attention_layernorm.weight"),
            hidden,
        )?,
    };
    tensors.validate(cfg)?;
    Ok(tensors)
}

/// A [`LayerSource`] that streams layer weights out of a memory-mapped
/// checkpoint on demand — the production backend for [`StreamingKernel`].
pub struct MmapLayerSource {
    store: MmapStore,
    cfg: BlockConfig,
    num_layers: u32,
}

impl LayerSource for MmapLayerSource {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }
    fn load_layer(&self, layer: u32) -> Result<LayerTensors> {
        load_layer_tensors(&self.store, &self.cfg, layer)
    }
}

/// The always-resident ("pinned zone") pieces + the on-demand layer source that
/// both streaming kernels (host and GPU) build a generator around.
struct StreamingPieces {
    source: MmapLayerSource,
    embedding: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Vec<f32>,
    vocab: usize,
    rms_eps: f32,
    kv_config: KvCacheConfig,
    kv_blocks: u32,
}

/// Load the pinned pieces (embedding, final norm, LM head, KV sizing) and bind a
/// [`MmapLayerSource`] over `store`'s transformer layers.
fn load_streaming_pieces(
    store: MmapStore,
    config: &ModelConfig,
    max_context: u32,
) -> Result<StreamingPieces> {
    let cfg = block_config(config);
    let hidden = cfg.hidden_size;
    let vocab = config.vocab_size as usize;

    let embedding = load_tensor(&store, "model.embed_tokens.weight", vocab * hidden)?;
    let final_norm = load_tensor(&store, "model.norm.weight", hidden)?;
    let lm_head = if store.locate("lm_head.weight").is_some() {
        load_tensor(&store, "lm_head.weight", vocab * hidden)?
    } else {
        embedding.clone()
    };
    let kv_config = KvCacheConfig {
        num_layers: config.num_layers,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        block_size: 16,
    };
    let kv_blocks = (max_context as u64).div_ceil(16) as u32 + 2;

    Ok(StreamingPieces {
        source: MmapLayerSource { store, cfg, num_layers: config.num_layers },
        embedding,
        final_norm,
        lm_head,
        vocab,
        rms_eps: config.rms_eps,
        kv_config,
        kv_blocks,
    })
}

/// Wrap `store`'s transformer layers in a host [`StreamingKernel`] that keeps at
/// most `resident_layers` layers in memory, with the pinned pieces resident.
///
/// Peak layer-weight memory is `resident_layers × per-layer` instead of the
/// whole model, so this serves models larger than the resident budget. Output is
/// identical to a fully-resident run (the window is a memory knob only).
pub fn build_streaming_generator(
    store: MmapStore,
    config: &ModelConfig,
    max_context: u32,
    resident_layers: usize,
    prefetch_depth: usize,
) -> Result<Generator<StreamingKernel<MmapLayerSource>>> {
    let p = load_streaming_pieces(store, config, max_context)?;
    let cfg = p.source.cfg;
    let kernel = StreamingKernel::new(cfg, p.source, resident_layers)
        .with_prefetch_depth(prefetch_depth as u32);
    Generator::new(
        kernel,
        p.embedding,
        p.final_norm,
        p.lm_head,
        p.vocab,
        p.rms_eps,
        p.kv_config,
        p.kv_blocks,
    )
}

/// GPU counterpart of [`build_streaming_generator`]: stream a window of layer
/// weights through VRAM ([`StreamingGpuKernel`]) while KV stays resident.
/// Experimental — the device path is compile-checked but unvalidated on hardware.
#[cfg(feature = "cuda-kernels")]
pub fn build_streaming_gpu_generator(
    store: MmapStore,
    config: &ModelConfig,
    max_context: u32,
    resident_layers: usize,
) -> Result<Generator<crate::forward::StreamingGpuKernel<MmapLayerSource>>> {
    let p = load_streaming_pieces(store, config, max_context)?;
    let cfg = p.source.cfg;
    let max_kv_tokens = p.kv_blocks as usize * p.kv_config.block_size as usize;
    let kernel =
        crate::forward::StreamingGpuKernel::new(cfg, p.source, max_kv_tokens, resident_layers)?;
    Generator::new(
        kernel,
        p.embedding,
        p.final_norm,
        p.lm_head,
        p.vocab,
        p.rms_eps,
        p.kv_config,
        p.kv_blocks,
    )
}

/// Materialize a checkpoint into [`ModelParts`] (host `f32` weights + shapes).
///
/// `max_context` sizes the KV block pool (tokens the sequence may reach). Uses
/// the standard `model.layers.{i}.*`, `model.embed_tokens.weight`,
/// `model.norm.weight`, and `lm_head.weight` names; a tied LM head (missing
/// `lm_head.weight`) falls back to the embedding matrix.
pub fn load_model_parts(
    store: &MmapStore,
    config: &ModelConfig,
    max_context: u32,
) -> Result<ModelParts> {
    let hidden = config.hidden_size as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = config.head_dim() as usize;
    let vocab = config.vocab_size as usize;

    let cfg = block_config(config);

    let mut layers = Vec::with_capacity(config.num_layers as usize);
    for i in 0..config.num_layers {
        layers.push(load_layer_tensors(store, &cfg, i)?);
    }

    let embedding = load_tensor(store, "model.embed_tokens.weight", vocab * hidden)?;
    let final_norm = load_tensor(store, "model.norm.weight", hidden)?;
    // Weight tying: reuse the embedding when there is no separate LM head.
    let lm_head = if store.locate("lm_head.weight").is_some() {
        load_tensor(store, "lm_head.weight", vocab * hidden)?
    } else {
        embedding.clone()
    };

    let kv_config = KvCacheConfig {
        num_layers: config.num_layers,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        block_size: 16,
    };
    let kv_blocks = (max_context as u64).div_ceil(16) as u32 + 2;

    Ok(ModelParts {
        cfg,
        layers,
        embedding,
        final_norm,
        lm_head,
        vocab_size: vocab,
        rms_eps: config.rms_eps,
        kv_config,
        kv_blocks,
    })
}

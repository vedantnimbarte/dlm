//! Load a runnable CPU model from a memory-mapped safetensors checkpoint.
//!
//! Bridges the storage layer to the compute path: it reads the standard
//! HuggingFace-named tensors out of an [`MmapStore`], converts them to `f32`
//! ([`bytes_to_f32`] handles F32/F16/BF16), and assembles a
//! [`Generator`] over a [`CpuKernel`] ready to run [`generate`].
//!
//! This is the connector that lets `flip generate --model-path <dir>` run a real
//! (small) model on CPU. Quantized checkpoints (AWQ/GPTQ `qweight` triplets)
//! would be materialized through the [`quant`](crate::quant) dequant kernel here
//! — this loader covers the float dtypes small models ship in.
//!
//! [`bytes_to_f32`]: crate::storage::bytes_to_f32
//! [`generate`]: crate::generate::Generator::generate

use crate::cache::KvCacheConfig;
use crate::error::{FlipError, Result};
use crate::forward::{BlockConfig, CpuKernel, LayerTensors};
use crate::generate::Generator;
use crate::model::ModelConfig;
use crate::quant::{dequantize_gptq_4bit, PackedQuantConfig};
use crate::storage::{bytes_to_f32, bytes_to_i32, MmapStore};

/// Read a named tensor as `f32`, verifying it has exactly `expected_len` elements.
fn load_tensor(store: &MmapStore, name: &str, expected_len: usize) -> Result<Vec<f32>> {
    let values = load_floats(store, name)?;
    if values.len() != expected_len {
        return Err(FlipError::InvalidConfig(format!(
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
        .ok_or_else(|| FlipError::UnknownTensor(name.to_string()))?;
    bytes_to_f32(shard.tensor_bytes(name)?, info.dtype)
}

/// Read an integer tensor of unknown length (packed `qweight`/`qzeros`).
fn load_ints(store: &MmapStore, name: &str) -> Result<Vec<i32>> {
    let (shard, info) = store
        .locate(name)
        .ok_or_else(|| FlipError::UnknownTensor(name.to_string()))?;
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
            return Err(FlipError::QuantLayout(format!(
                "{base}.scales length {} is not a multiple of out_features {out_features}",
                scales.len()
            )));
        }
        let num_groups = scales.len() / out_features;
        if num_groups == 0 || in_features % num_groups != 0 {
            return Err(FlipError::QuantLayout(format!(
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

    Err(FlipError::UnknownTensor(format!(
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
    let num_heads = config.num_attention_heads as usize;
    let num_kv_heads = config.num_kv_heads as usize;
    let head_dim = config.head_dim() as usize;
    let intermediate = config.intermediate_size as usize;
    let vocab = config.vocab_size as usize;

    let cfg = BlockConfig {
        hidden_size: hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size: intermediate,
        rope_theta: config.rope_theta,
        rms_eps: config.rms_eps,
    };
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();

    let mut layers = Vec::with_capacity(config.num_layers as usize);
    for i in 0..config.num_layers {
        let name = |suffix: &str| format!("model.layers.{i}.{suffix}");
        // Projections may be float or GPTQ-quantized; norms are always float.
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
        tensors.validate(&cfg)?;
        layers.push(tensors);
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

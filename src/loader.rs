//! Load a runnable CPU model from a memory-mapped safetensors checkpoint.
//!
//! Bridges the storage layer to the compute path: it reads the standard
//! HuggingFace-named tensors out of an [`MmapStore`] and assembles a
//! [`Generator`] over a [`CpuKernel`] ready to run [`generate`].
//!
//! Weight matrices keep their **native checkpoint dtype** (F32/F16/BF16) — see
//! [`Weights`]. Upsizing them to f32 here would be lossless (an f32 exactly
//! represents every f16/bf16 value) and therefore buys no precision, while
//! doubling host RAM, the PCIe traffic per streamed layer, and the bandwidth of
//! the memory-bound GEMV that dominates decode. The small tensors (norms, biases,
//! embedding, LM head) are read as f32.
//!
//! Already-quantized checkpoints (4-bit GPTQ `qweight`/`qzeros`/`scales`) are
//! decoded here too, and are *relabeled* rather than dequantized: their codes and
//! per-group scales go straight into [`Weights::Int4`], so the accuracy GPTQ's
//! calibration bought survives and the layer stays a quarter the size of an fp16
//! one. Only the variant validated against a real export is accepted — see
//! `check_quant_supported`.
//!
//! [`bytes_to_f32`]: crate::storage::bytes_to_f32
//! [`generate`]: crate::generate::Generator::generate

use crate::cache::KvCacheConfig;
use crate::error::{DlmError, Result};
use crate::forward::{
    BlockConfig, CachedLayerSource, CpuKernel, ExpertFfn, Ffn, LayerSource, LayerTensors,
    PipelineParallelKernel, StreamingKernel, Weights,
};
use crate::generate::Generator;
use crate::model::{ModelConfig, PackedQuant, QuantScheme};
use crate::storage::{bytes_to_f32, Dtype, MmapStore};

/// The [`QuantScheme`] matching a checkpoint's own weight dtype.
///
/// This is the honest default for `--quant`: dlm reads the weights the file
/// actually contains. Assuming a scheme the checkpoint does not have (the old
/// Int4 default, against bf16 weights) mis-sizes every plan derived from it by
/// 4x. Probes a layer-0 projection — every weight matrix in a checkpoint shares
/// one dtype — and falls back to the first weight tensor it can find.
pub fn checkpoint_scheme(store: &MmapStore) -> Result<QuantScheme> {
    // An already-packed 4-bit checkpoint (GPTQ) has no float `.weight` to probe —
    // its precision is int4 by construction.
    if store.locate("model.layers.0.self_attn.q_proj.qweight").is_some() {
        return Ok(QuantScheme::Int4);
    }
    let probe = ["model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"];
    let dtype = probe
        .iter()
        .find_map(|n| store.locate(n).map(|(_, info)| info.dtype))
        .ok_or_else(|| {
            DlmError::UnknownTensor(
                "no weight tensor found to probe the checkpoint's dtype".to_string(),
            )
        })?;
    match dtype {
        Dtype::F32 => Ok(QuantScheme::F32),
        Dtype::F16 | Dtype::BF16 => Ok(QuantScheme::Fp16),
        other => Err(DlmError::UnsupportedQuant(format!(
            "checkpoint weights are {other:?}; dlm handles F32/F16/BF16"
        ))),
    }
}

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

/// Reinterpret raw little-endian bytes as 16-bit bit patterns (bf16/f16), with no
/// numeric conversion.
fn bytes_to_u16(bytes: &[u8], name: &str) -> Result<Vec<u16>> {
    if bytes.len() % 2 != 0 {
        return Err(DlmError::SafetensorsHeader(format!(
            "tensor {name:?}: 16-bit tensor byte length {} is not a multiple of 2",
            bytes.len()
        )));
    }
    let len = bytes.len() / 2;
    #[cfg(target_endian = "little")]
    {
        // The checkpoint stores little-endian 16-bit words, so on a little-endian
        // host its bytes already *are* the u16 bit patterns — copy them wholesale
        // into an *uninitialized* buffer. Two things this avoids, both of which
        // land on the streaming hot path (every layer is re-materialized on every
        // VRAM miss): the element-wise `from_le_bytes` decode, which the compiler
        // does not collapse into a memcpy (~2 GB/s vs ~10 GB/s), and `vec![0; n]`
        // zeroing a buffer we immediately overwrite.
        //
        // SAFETY: capacity is `len` u16s = exactly `bytes.len()` bytes, so the
        // copy stays in bounds; u16 is POD with no invalid bit patterns, so every
        // byte pattern is a valid value and `set_len` exposes only initialized
        // memory; source and destination are distinct allocations.
        let mut v: Vec<u16> = Vec::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr() as *mut u8, bytes.len());
            v.set_len(len);
        }
        Ok(v)
    }
    #[cfg(target_endian = "big")]
    {
        let mut v = vec![0u16; len];
        for (slot, c) in v.iter_mut().zip(bytes.chunks_exact(2)) {
            *slot = u16::from_le_bytes([c[0], c[1]]);
        }
        Ok(v)
    }
}

/// Read a weight matrix **in its native dtype, without converting it**.
///
/// The compute path converts to f32 in the register that consumes each element,
/// so materializing an upsized f32 copy here would be lossless (f32 exactly
/// represents every f16/bf16 value) yet cost 2x host RAM, 2x PCIe per streamed
/// layer, and 2x the bandwidth of the memory-bound GEMV. See [`Weights`].
fn load_native(
    store: &MmapStore,
    name: &str,
    expected_len: usize,
    quant: QuantScheme,
) -> Result<Weights> {
    let (shard, info) = store
        .locate(name)
        .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;
    let bytes = shard.tensor_bytes(name)?;
    // `--quant int4`/`int8`: quantize down from the checkpoint's floats at load.
    // Costs one f32 materialization here (transient, per tensor) and buys 4x/2x
    // less VRAM and PCIe per layer for the whole run.
    if matches!(quant, QuantScheme::Int4 | QuantScheme::Int8) {
        let floats = bytes_to_f32(bytes, info.dtype)?;
        if floats.len() != expected_len {
            return Err(DlmError::InvalidConfig(format!(
                "tensor {name:?}: expected {expected_len} elements, got {}",
                floats.len()
            )));
        }
        let group = crate::forward::QUANT_GROUP_SIZE;
        return match quant {
            QuantScheme::Int4 => Weights::quantize_int4(&floats, group),
            _ => Weights::quantize_int8(&floats, group),
        };
    }
    let w = match info.dtype {
        Dtype::F32 => Weights::F32(bytes_to_f32(bytes, info.dtype)?),
        Dtype::BF16 => Weights::Bf16(bytes_to_u16(bytes, name)?),
        Dtype::F16 => Weights::F16(bytes_to_u16(bytes, name)?),
        other => {
            return Err(DlmError::UnsupportedQuant(format!(
                "tensor {name:?} has dtype {other:?}; dlm's compute path handles \
                 F32/F16/BF16 weights"
            )))
        }
    };
    if w.len() != expected_len {
        return Err(DlmError::InvalidConfig(format!(
            "tensor {name:?}: expected {expected_len} elements, got {}",
            w.len()
        )));
    }
    Ok(w)
}


/// Load a linear layer's weight as dense row-major `[out, in]`, transparently
/// handling both float (`{base}.weight`) and GPTQ-style quantized
/// (`{base}.qweight` + `.qzeros` + `.scales`) checkpoints.
fn load_linear(
    store: &MmapStore,
    base: &str,
    in_features: usize,
    out_features: usize,
    quant: QuantScheme,
    packed: Option<PackedQuant>,
) -> Result<Weights> {
    // Float weight: already row-major [out, in].
    let weight_name = format!("{base}.weight");
    if store.locate(&weight_name).is_some() {
        return load_native(store, &weight_name, out_features * in_features, quant);
    }

    // A packed 4-bit checkpoint (GPTQ). `config.json` declares the layout, and
    // `check_quant_supported` has already refused any variant dlm cannot decode
    // correctly. A `qweight` with no `quantization_config` at all is still refused
    // rather than guessed at — the zero-point convention alone decides whether the
    // weights come out right or merely plausible.
    if store.locate(&format!("{base}.qweight")).is_some() {
        let Some(pq) = packed else {
            return Err(DlmError::UnsupportedQuant(format!(
                "{base} is a packed 4-bit tensor but config.json declares no \
                 quantization_config, so its group size and zero-point convention are \
                 unknown. dlm will not guess at them: decoding them wrong yields \
                 plausible-looking but incorrect output."
            )));
        };
        return load_gptq_linear(store, base, in_features, out_features, pq);
    }

    Err(DlmError::UnknownTensor(format!(
        "{base}.weight or {base}.qweight"
    )))
}

/// Load a GPTQ 4-bit linear, keeping its codes.
///
/// The checkpoint is already int4, so it is *relabeled* into dlm's blob order
/// rather than dequantized and re-quantized: the accuracy GPTQ's calibration
/// bought survives intact, and the layer costs a quarter of an fp16 one in VRAM
/// and on the bus. Decoding then runs through the same `load_w<DLM_W_INT4>`
/// kernel as `--quant int4`.
fn load_gptq_linear(
    store: &MmapStore,
    base: &str,
    in_features: usize,
    out_features: usize,
    pq: PackedQuant,
) -> Result<Weights> {
    use crate::model::PackedFormat;
    let qweight = load_i32(store, &format!("{base}.qweight"))?;
    let qzeros = load_i32(store, &format!("{base}.qzeros"))?;
    let scales = load_floats(store, &format!("{base}.scales"))?;
    let cfg = crate::quant::PackedQuantConfig {
        in_features,
        out_features,
        group_size: pq.group_size,
    };
    if pq.kind.is_experimental() {
        warn_experimental_quant(pq.kind);
    }
    match pq.kind {
        // Classic GPTQ: relabel the codes into dlm's int4 layout (keeps int4 VRAM).
        PackedFormat::Gptq { act_order: false } => {
            let (codes, sc, ze) = crate::quant::unpack_gptq_4bit(&qweight, &qzeros, &scales, &cfg)?;
            Weights::from_int4_parts(&codes, &sc, &ze, pq.group_size)
        }
        // Act-order groups are scattered by g_idx; dequantize to f32 (dlm's flat
        // int4 layout can't represent non-contiguous groups).
        PackedFormat::Gptq { act_order: true } => {
            let g_idx = load_i32(store, &format!("{base}.g_idx"))?;
            let dense =
                crate::quant::dequantize_gptq_4bit(&qweight, &qzeros, &scales, &cfg, Some(&g_idx))?;
            Ok(Weights::from_f32(dense))
        }
        // AWQ: different nibble order + zero convention; dequantize to f32.
        PackedFormat::Awq => {
            let dense = crate::quant::dequantize_awq_4bit(&qweight, &qzeros, &scales, &cfg)?;
            Ok(Weights::from_f32(dense))
        }
    }
}

/// Warn once (per process) that an experimental packed-quant path is in use — its
/// decoding is validated only by internal round-trip, not against a real export,
/// so a convention mismatch would produce plausible-but-wrong weights.
fn warn_experimental_quant(kind: crate::model::PackedFormat) {
    use std::sync::Once;
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "warning: decoding an EXPERIMENTAL packed-quant format ({kind:?}). This path is \
             validated only by dlm's internal round-trip, NOT against a reference export — a \
             convention mismatch would yield plausible-but-wrong weights. Verify output with \
             `dlm doctor` and your own prompts before trusting it."
        );
    });
}

/// Read an `i32` tensor (GPTQ packs its codes and zero-points as `int32`).
fn load_i32(store: &MmapStore, name: &str) -> Result<Vec<i32>> {
    let (shard, info) = store
        .locate(name)
        .ok_or_else(|| DlmError::UnknownTensor(name.to_string()))?;
    crate::storage::bytes_to_i32(shard.tensor_bytes(name)?, info.dtype)
}

/// Read an optional f32 tensor by its full name, or `None` when absent (e.g.
/// Qwen3's per-head Q/K norm weights, which most models don't ship).
fn load_optional(store: &MmapStore, name: &str, len: usize) -> Result<Option<Vec<f32>>> {
    if store.locate(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_tensor(store, name, len)?))
}

/// Read a linear layer's bias (`{base}.bias`) when the checkpoint ships one.
///
/// Llama/Mistral have no attention biases; Qwen2 does. Returning `None` for an
/// absent bias is correct, but silently *ignoring* one that exists produces wrong
/// attention and incoherent output, so every caller that has a bias slot uses this.
fn load_bias(store: &MmapStore, base: &str, out_features: usize) -> Result<Option<Vec<f32>>> {
    let name = format!("{base}.bias");
    if store.locate(&name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_tensor(store, &name, out_features)?))
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
    /// Embedding scale (Gemma: `sqrt(hidden)`); `None` leaves embeddings unscaled.
    pub embed_scale: Option<f32>,
}

impl ModelParts {
    /// Wrap the CPU kernel around these weights and build a generator.
    pub fn into_cpu_generator(self) -> Result<Generator<CpuKernel>> {
        let embed_scale = self.embed_scale;
        let kernel = CpuKernel::new(self.cfg, self.layers)?;
        Ok(Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )?
        .with_embed_scale(embed_scale))
    }

    /// Upload every layer to VRAM and build a generator over the GPU kernel.
    #[cfg(feature = "cuda-kernels")]
    pub fn into_gpu_generator(self) -> Result<Generator<crate::forward::GpuKernel>> {
        let max_kv_tokens = self.kv_blocks as usize * self.kv_config.block_size as usize;
        let embed_scale = self.embed_scale;
        let kernel = crate::forward::GpuKernel::new(self.cfg, self.layers, max_kv_tokens)?;
        Ok(Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )?
        .with_embed_scale(embed_scale))
    }

    /// Split the model's layers across `gpu_ids` and build a generator over a
    /// **GPU** pipeline ([`MultiGpuKernel`]) — each device holds and computes its
    /// own layer shard (real multi-GPU compute, not the CPU-backed
    /// [`into_pipeline_parallel_generator`](Self::into_pipeline_parallel_generator)).
    #[cfg(feature = "cuda-kernels")]
    pub fn into_multi_gpu_generator(
        self,
        gpu_ids: &[u32],
    ) -> Result<Generator<crate::forward::MultiGpuKernel>> {
        let max_kv_tokens = self.kv_blocks as usize * self.kv_config.block_size as usize;
        let embed_scale = self.embed_scale;
        let kernel = crate::forward::MultiGpuKernel::new(self.cfg, self.layers, gpu_ids, max_kv_tokens)?;
        Ok(Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )?
        .with_embed_scale(embed_scale))
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
        let embed_scale = self.embed_scale;
        let kernel = PipelineParallelKernel::new(CpuKernel::new(self.cfg, self.layers)?, gpu_ids)?;
        Ok(Generator::new(
            kernel,
            self.embedding,
            self.final_norm,
            self.lm_head,
            self.vocab_size,
            self.rms_eps,
            self.kv_config,
            self.kv_blocks,
        )?
        .with_embed_scale(embed_scale))
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
        rope_scaling: config.rope_scaling,
        moe: config.moe,
        sliding_window: config.sliding_window.map(|w| w as usize),
        activation: config.activation,
        mla: config.mla,
    }
}

/// Materialize one transformer layer's weights (`model.layers.{layer}.*`),
/// handling float and GPTQ-quantized projections. This is the unit the streaming
/// kernel pulls on demand and the resident loader pulls for every layer.
pub(crate) fn load_layer_tensors(
    store: &MmapStore,
    cfg: &BlockConfig,
    layer: u32,
    quant: QuantScheme,
    packed: Option<PackedQuant>,
    norm_add_one: bool,
) -> Result<LayerTensors> {
    load_layer_tensors_opt(store, cfg, layer, quant, packed, true, norm_add_one)
}

/// Load an RMSNorm weight, baking Gemma's `+1` in when `add_one` so the norm
/// kernels can stay a plain `w` multiply.
fn load_norm(store: &MmapStore, name: &str, len: usize, add_one: bool) -> Result<Vec<f32>> {
    let mut v = load_tensor(store, name, len)?;
    if add_one {
        for x in &mut v {
            *x += 1.0;
        }
    }
    Ok(v)
}

/// Like [`load_layer_tensors`], but `include_experts = false` loads only the MoE
/// **core** (attention, norms, router, shared expert) — the routed experts are
/// left out for the GPU streaming path to fetch per `(layer, expert)`. No effect
/// on dense models, which have no separate experts.
pub(crate) fn load_layer_tensors_opt(
    store: &MmapStore,
    cfg: &BlockConfig,
    layer: u32,
    quant: QuantScheme,
    packed: Option<PackedQuant>,
    include_experts: bool,
    norm_add_one: bool,
) -> Result<LayerTensors> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let intermediate = cfg.intermediate_size;
    let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
    // load_linear takes (in_features, out_features) of the underlying Linear.
    let ffn = match cfg.moe {
        None => Ffn::Dense(ExpertFfn {
            gate: load_linear(store, &name("mlp.gate_proj"), hidden, intermediate, quant, packed)?,
            up: load_linear(store, &name("mlp.up_proj"), hidden, intermediate, quant, packed)?,
            down: load_linear(store, &name("mlp.down_proj"), intermediate, hidden, quant, packed)?,
        }),
        Some(_) => load_moe_ffn(store, cfg, layer, quant, packed, include_experts)?,
    };
    let input_layernorm = load_norm(store, &name("input_layernorm.weight"), hidden, norm_add_one)?;
    let post_attention_layernorm =
        load_norm(store, &name("post_attention_layernorm.weight"), hidden, norm_add_one)?;

    // MLA (DeepSeek) replaces q/k/v with latent projections and a wider o_proj.
    let tensors = if let Some(m) = &cfg.mla {
        let vdim = m.v_head_dim as usize;
        LayerTensors {
            o_proj: load_linear(store, &name("self_attn.o_proj"), cfg.num_heads * vdim, hidden, quant, packed)?,
            ffn,
            input_layernorm,
            post_attention_layernorm,
            mla: Some(load_mla(store, cfg, m, layer, quant, packed, norm_add_one)?),
            ..Default::default()
        }
    } else {
        LayerTensors {
            q_proj: load_linear(store, &name("self_attn.q_proj"), hidden, q_dim, quant, packed)?,
            k_proj: load_linear(store, &name("self_attn.k_proj"), hidden, kv_dim, quant, packed)?,
            v_proj: load_linear(store, &name("self_attn.v_proj"), hidden, kv_dim, quant, packed)?,
            o_proj: load_linear(store, &name("self_attn.o_proj"), q_dim, hidden, quant, packed)?,
            ffn,
            input_layernorm,
            post_attention_layernorm,
            // Present on Qwen2 and friends, absent on Llama/Mistral.
            q_bias: load_bias(store, &name("self_attn.q_proj"), q_dim)?,
            k_bias: load_bias(store, &name("self_attn.k_proj"), kv_dim)?,
            v_bias: load_bias(store, &name("self_attn.v_proj"), kv_dim)?,
            // Per-head Q/K RMSNorm (`[head_dim]`), present on Qwen3.
            q_norm: load_optional(store, &name("self_attn.q_norm.weight"), cfg.head_dim)?,
            k_norm: load_optional(store, &name("self_attn.k_norm.weight"), cfg.head_dim)?,
            mla: None,
        }
    };
    tensors.validate(cfg)?;
    Ok(tensors)
}

/// Load a layer's MLA (DeepSeek) projection weights from the
/// `self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,...}` tensors.
#[allow(clippy::too_many_arguments)]
fn load_mla(
    store: &MmapStore,
    cfg: &BlockConfig,
    m: &crate::model::MlaConfig,
    layer: u32,
    quant: QuantScheme,
    packed: Option<PackedQuant>,
    norm_add_one: bool,
) -> Result<crate::forward::MlaWeights> {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let qk = m.qk_head_dim() as usize;
    let latent = m.kv_lora_rank as usize;
    let rope = m.qk_rope_head_dim as usize;
    let vdim = m.v_head_dim as usize;
    let name = |s: &str| format!("model.layers.{layer}.{s}");
    // Query: low-rank (q_a_proj → norm → q_b_proj) when q_lora_rank is set, else
    // a direct projection under the `q_proj` name.
    let (q_a_proj, q_a_layernorm, q_b_proj) = match m.q_lora_rank {
        Some(r) => {
            let r = r as usize;
            (
                Some(load_linear(store, &name("self_attn.q_a_proj"), h, r, quant, packed)?),
                Some(load_norm(store, &name("self_attn.q_a_layernorm.weight"), r, norm_add_one)?),
                load_linear(store, &name("self_attn.q_b_proj"), r, nh * qk, quant, packed)?,
            )
        }
        None => (
            None,
            None,
            load_linear(store, &name("self_attn.q_proj"), h, nh * qk, quant, packed)?,
        ),
    };
    Ok(crate::forward::MlaWeights {
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj: load_linear(store, &name("self_attn.kv_a_proj_with_mqa"), h, latent + rope, quant, packed)?,
        kv_a_layernorm: load_norm(store, &name("self_attn.kv_a_layernorm.weight"), latent, norm_add_one)?,
        kv_b_proj: load_linear(
            store,
            &name("self_attn.kv_b_proj"),
            latent,
            nh * (m.qk_nope_head_dim as usize + vdim),
            quant,
            packed,
        )?,
    })
}

/// Tensor-name prefixes for one layer's MoE block, per family. `router` names the
/// gating Linear; `expert(e)` the base of expert `e`'s three projections;
/// `(gate, up, down)` the per-expert projection suffixes; `shared` the shared
/// expert base and its sigmoid gate (Qwen2-MoE only).
struct MoeNames {
    router: String,
    experts_base: String, // e.g. "…block_sparse_moe.experts" / "…mlp.experts"
    proj: (&'static str, &'static str, &'static str), // (gate, up, down) suffixes
    shared: Option<(String, String)>,                 // (shared_expert base, shared_gate)
}

fn moe_names(naming: crate::model::MoeNaming, layer: u32) -> MoeNames {
    use crate::model::MoeNaming::*;
    let p = format!("model.layers.{layer}");
    match naming {
        // Mixtral: block_sparse_moe.{gate, experts.N.{w1,w3,w2}}, no shared expert.
        Mixtral => MoeNames {
            router: format!("{p}.block_sparse_moe.gate"),
            experts_base: format!("{p}.block_sparse_moe.experts"),
            proj: ("w1", "w3", "w2"),
            shared: None,
        },
        // Qwen: mlp.{gate, experts.N.{gate,up,down}_proj, shared_expert.*, shared_expert_gate}.
        Qwen => MoeNames {
            router: format!("{p}.mlp.gate"),
            experts_base: format!("{p}.mlp.experts"),
            proj: ("gate_proj", "up_proj", "down_proj"),
            shared: Some((format!("{p}.mlp.shared_expert"), format!("{p}.mlp.shared_expert_gate"))),
        },
    }
}

/// Load one MoE expert's SwiGLU triple.
fn load_expert(
    store: &MmapStore,
    base: &str,
    proj: (&str, &str, &str),
    hidden: usize,
    inter: usize,
    quant: QuantScheme,
    packed: Option<PackedQuant>,
) -> Result<ExpertFfn> {
    Ok(ExpertFfn {
        gate: load_linear(store, &format!("{base}.{}", proj.0), hidden, inter, quant, packed)?,
        up: load_linear(store, &format!("{base}.{}", proj.1), hidden, inter, quant, packed)?,
        down: load_linear(store, &format!("{base}.{}", proj.2), inter, hidden, quant, packed)?,
    })
}

/// Build a layer's [`Ffn::Moe`]: the router, the routed experts (skipped when
/// `include_experts` is false — the GPU streaming core defers them to its
/// per-`(layer, expert)` cache), and any shared expert.
///
/// ponytail: the router is quantized like the rest under `--quant`. Routers are
/// small and precision-sensitive; if int4 routing measurably flips top-k on real
/// models, load it native instead. Left uniform until measured.
pub(crate) fn load_moe_ffn(
    store: &MmapStore,
    cfg: &BlockConfig,
    layer: u32,
    quant: QuantScheme,
    packed: Option<PackedQuant>,
    include_experts: bool,
) -> Result<Ffn> {
    let m = cfg.moe.expect("load_moe_ffn requires MoE config");
    let hidden = cfg.hidden_size;
    let inter = m.moe_intermediate_size as usize;
    let names = moe_names(m.naming, layer);

    // The router is precision-sensitive: quantizing it can flip which experts a
    // token routes to, silently degrading a sparse model, and it is tiny
    // (`hidden × num_experts`) so keeping it native costs almost nothing. For a
    // float checkpoint that `--quant` would quantize, load the router in its
    // native dtype instead. (A packed GPTQ checkpoint's router keeps its own
    // calibrated int4 codes — there is no float to fall back to.)
    let router_quant = if packed.is_none() && matches!(quant, QuantScheme::Int4 | QuantScheme::Int8) {
        QuantScheme::Fp16 // sentinel: load_native reads the real dtype, doesn't quantize
    } else {
        quant
    };
    let router =
        load_linear(store, &names.router, hidden, m.num_experts as usize, router_quant, packed)?;
    let experts = if include_experts {
        (0..m.num_experts)
            .map(|e| {
                let base = format!("{}.{e}", names.experts_base);
                load_expert(store, &base, names.proj, hidden, inter, quant, packed)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    let (shared, shared_gate) = match (m.shared_intermediate_size, &names.shared) {
        (Some(si), Some((sbase, sgate))) => {
            let si = si as usize;
            let shared = load_expert(store, sbase, names.proj, hidden, si, quant, packed)?;
            // shared_expert_gate is a Linear(hidden -> 1): weight is [1, hidden].
            let gate = load_linear(store, sgate, hidden, 1, quant, packed)?;
            (Some(shared), Some(gate))
        }
        _ => (None, None),
    };

    Ok(Ffn::Moe { router, experts, shared, shared_gate })
}

/// A [`LayerSource`] that streams layer weights out of a memory-mapped
/// checkpoint on demand — the production backend for [`StreamingKernel`].
pub struct MmapLayerSource {
    store: MmapStore,
    cfg: BlockConfig,
    num_layers: u32,
    /// Weight precision to materialize each layer in (see `--quant`).
    quant: QuantScheme,
    /// Set when the checkpoint is already packed-quantized (GPTQ).
    packed: Option<PackedQuant>,
    /// Bake Gemma's `(1+w)` into RMSNorm weights at load.
    norm_add_one: bool,
}

impl LayerSource for MmapLayerSource {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }
    fn load_layer(&self, layer: u32) -> Result<std::sync::Arc<LayerTensors>> {
        load_layer_tensors(&self.store, &self.cfg, layer, self.quant, self.packed, self.norm_add_one)
            .map(std::sync::Arc::new)
    }
    fn load_layer_core(&self, layer: u32) -> Result<std::sync::Arc<LayerTensors>> {
        // Core only (no routed experts) for the per-expert streaming GPU path.
        load_layer_tensors_opt(
            &self.store,
            &self.cfg,
            layer,
            self.quant,
            self.packed,
            false,
            self.norm_add_one,
        )
        .map(std::sync::Arc::new)
    }
    fn load_expert(&self, layer: u32, expert: u32) -> Result<std::sync::Arc<ExpertFfn>> {
        let m = self.cfg.moe.ok_or_else(|| {
            DlmError::InvalidConfig("load_expert on a dense model".into())
        })?;
        let names = moe_names(m.naming, layer);
        let base = format!("{}.{expert}", names.experts_base);
        load_expert(
            &self.store,
            &base,
            names.proj,
            self.cfg.hidden_size,
            m.moe_intermediate_size as usize,
            self.quant,
            self.packed,
        )
        .map(std::sync::Arc::new)
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
    embed_scale: Option<f32>,
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
    let final_norm = load_norm(&store, "model.norm.weight", hidden, config.norm_add_one)?;
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
        source: MmapLayerSource {
            store,
            cfg,
            norm_add_one: config.norm_add_one,
            num_layers: config.num_layers,
            quant: config.quant,
            packed: config.packed_quant,
        },
        embedding,
        final_norm,
        lm_head,
        vocab,
        rms_eps: config.rms_eps,
        kv_config,
        kv_blocks,
        embed_scale: config.embed_scale,
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
    auto_prefetch: bool,
    ram_cache_bytes: usize,
) -> Result<Generator<StreamingKernel<CachedLayerSource<MmapLayerSource>>>> {
    let p = load_streaming_pieces(store, config, max_context)?;
    let cfg = p.source.cfg;
    let source = CachedLayerSource::new(p.source, ram_cache_bytes);
    let base = StreamingKernel::new(cfg, source, resident_layers);
    let kernel = if auto_prefetch {
        base.with_auto_prefetch()
    } else {
        base.with_prefetch_depth(prefetch_depth as u32)
    };
    Ok(Generator::new(
        kernel,
        p.embedding,
        p.final_norm,
        p.lm_head,
        p.vocab,
        p.rms_eps,
        p.kv_config,
        p.kv_blocks,
    )?
    .with_embed_scale(p.embed_scale))
}

/// GPU counterpart of [`build_streaming_generator`]: stream a window of layer
/// weights through VRAM ([`StreamingGpuKernel`]) while KV stays resident.
/// `ram_cache_bytes` bounds a host-RAM LRU of materialized layers so a VRAM miss
/// doesn't re-read and re-materialize the layer every token (`0` disables it).
#[cfg(feature = "cuda-kernels")]
pub fn build_streaming_gpu_generator(
    store: MmapStore,
    config: &ModelConfig,
    max_context: u32,
    resident_layers: usize,
    ram_cache_bytes: usize,
    expert_cache_bytes: usize,
) -> Result<Generator<crate::forward::StreamingGpuKernel<CachedLayerSource<MmapLayerSource>>>> {
    let p = load_streaming_pieces(store, config, max_context)?;
    let cfg = p.source.cfg;
    let max_kv_tokens = p.kv_blocks as usize * p.kv_config.block_size as usize;
    // Bound the routed-expert VRAM cache by the budget, not an unbounded count —
    // a 128-expert model would otherwise OOM the card (see B2). `None` for dense.
    let expert_cap = config.expert_cache_capacity(expert_cache_bytes as u64);
    let source = CachedLayerSource::new(p.source, ram_cache_bytes);
    let kernel = crate::forward::StreamingGpuKernel::new(
        cfg,
        source,
        max_kv_tokens,
        resident_layers,
        expert_cap,
    )?;
    Ok(Generator::new(
        kernel,
        p.embedding,
        p.final_norm,
        p.lm_head,
        p.vocab,
        p.rms_eps,
        p.kv_config,
        p.kv_blocks,
    )?
    .with_embed_scale(p.embed_scale))
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
        layers.push(load_layer_tensors(
            store,
            &cfg,
            i,
            config.quant,
            config.packed_quant,
            config.norm_add_one,
        )?);
    }

    let embedding = load_tensor(store, "model.embed_tokens.weight", vocab * hidden)?;
    let final_norm = load_norm(store, "model.norm.weight", hidden, config.norm_add_one)?;
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
        embed_scale: config.embed_scale,
    })
}

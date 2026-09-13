//! Model configuration parsed from a HuggingFace-style `config.json`.
//!
//! The VRAM profiler needs the shape parameters (hidden size, head counts,
//! layer count) to size both the streamed weight blocks and the KV cache.
//! Fields mirror the subset of `config.json` that `dlm` consumes in Phase 1.

use crate::error::{DlmError, Result};
use crate::forward::cpu::RopeScaling;
use serde::Deserialize;
use std::path::Path;

/// Numeric precision of the on-disk weights. Drives the bytes-per-parameter
/// term in the VRAM math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Full 32-bit weights (F32): 4 bytes per parameter.
    F32,
    /// Full 16-bit weights (FP16 / BF16): 2 bytes per parameter.
    Fp16,
    /// 8-bit quantization: 1 byte per parameter.
    Int8,
    /// 4-bit AWQ / GPTQ: 0.5 bytes per parameter (the Phase 1 default).
    Int4,
}

impl QuantScheme {
    /// Average bytes occupied by a single weight parameter under this scheme.
    /// Returned as `f64` because 4-bit packing is a fractional 0.5.
    pub fn bytes_per_param(self) -> f64 {
        match self {
            QuantScheme::F32 => 4.0,
            QuantScheme::Fp16 => 2.0,
            QuantScheme::Int8 => 1.0,
            QuantScheme::Int4 => 0.5,
        }
    }
}

/// Raw deserialization target matching HuggingFace `config.json` key names.
/// Kept private; callers get the validated [`ModelConfig`] instead.
#[derive(Debug, Deserialize)]
struct RawConfig {
    /// HF architecture id (e.g. "llama", "gemma", "gemma2"). Drives the Gemma
    /// norm/embed/activation variants.
    #[serde(default)]
    model_type: Option<String>,
    /// Gated-MLP activation name (Gemma ships "gelu_pytorch_tanh"). Older configs
    /// spell it `hidden_act`.
    ///
    /// These are two separate fields rather than one with `alias = "hidden_act"`,
    /// because every Gemma2 config (2b/9b/27b) ships *both* keys — and serde
    /// rejects a field matched by two names in the same object as a duplicate,
    /// which made the whole config unparseable. `hidden_activation` wins when
    /// both are present, matching transformers' Gemma2Config.
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default, alias = "activation_function")]
    hidden_act: Option<String>,
    /// GPT-2 spells the core dimensions `n_embd`/`n_head`/`n_layer`/`n_positions`.
    /// They mean the same things, so they are aliases rather than a second config
    /// path -- a second path is a second place for every later field to be
    /// forgotten.
    #[serde(alias = "n_embd")]
    hidden_size: u32,
    #[serde(alias = "n_head")]
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(alias = "n_layer")]
    num_hidden_layers: u32,
    vocab_size: u32,
    #[serde(default)]
    intermediate_size: Option<u32>,
    /// Aliased to `n_positions` only, **not** `n_ctx`: GPT-2 ships both with the
    /// same value, and serde rejects a doubly-matched field -- the exact shape of
    /// the bug that made every Gemma2 config unparseable.
    #[serde(default, alias = "n_positions")]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default, alias = "layer_norm_epsilon")]
    rms_norm_eps: Option<f32>,
    /// Explicit per-head dimension. Most models omit it (it is then
    /// `hidden_size / num_attention_heads`), but some declare a `head_dim` that
    /// is *not* that quotient, and assuming the quotient loads them mis-shaped.
    #[serde(default)]
    head_dim: Option<u32>,
    /// RoPE frequency scaling. Long-context models are trained with this; it is
    /// not optional decoration, and ignoring it corrupts every position.
    #[serde(default)]
    rope_scaling: Option<RawRopeScaling>,
    /// EOS token id(s). HF configs use either a single int or an array (e.g.
    /// Llama-3 lists `<|eot_id|>` and `<|end_of_text|>`).
    #[serde(default)]
    eos_token_id: Option<EosField>,
    /// Quantization metadata for GPTQ/AWQ checkpoints, when present. Used only
    /// to reject formats the dequantizer would otherwise mis-decode silently.
    #[serde(default)]
    quantization_config: Option<QuantizationConfig>,
    // ── Mixture-of-Experts (MoE) fields; all absent on dense checkpoints. ──
    /// Mixtral's expert count key.
    #[serde(default)]
    num_local_experts: Option<u32>,
    /// Qwen-MoE's expert count key.
    #[serde(default)]
    num_experts: Option<u32>,
    /// DeepSeek-V2/V3's expert count key.
    #[serde(default)]
    n_routed_experts: Option<u32>,
    /// DeepSeek's shared-expert *count* (not a width): the shared FFN is this
    /// many `moe_intermediate_size` experts fused into one wider SwiGLU.
    #[serde(default)]
    n_shared_experts: Option<u32>,
    /// DeepSeek: the first N layers are dense, the rest MoE. Ignoring this loads
    /// a dense FFN for a routed layer and fails on the missing tensor.
    #[serde(default)]
    first_k_dense_replace: Option<u32>,
    /// DeepSeek: place a MoE layer every N layers after the dense prefix. Only
    /// `1` (every layer) is implemented; anything else is refused rather than
    /// silently loading the wrong layers as dense.
    #[serde(default)]
    moe_layer_freq: Option<u32>,
    /// Experts routed per token (top-k). Required when the model is MoE.
    #[serde(default)]
    num_experts_per_tok: Option<u32>,
    /// Per-expert FFN inner width (Qwen). Mixtral reuses `intermediate_size`.
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    /// Shared-expert FFN inner width (Qwen2-MoE); absent on Qwen3-MoE & Mixtral.
    #[serde(default)]
    shared_expert_intermediate_size: Option<u32>,
    /// Renormalize the top-k gate weights so they sum to 1. Mixtral always does;
    /// Qwen exposes it as a flag.
    #[serde(default)]
    norm_topk_prob: Option<bool>,
    // ── Multi-head Latent Attention (MLA, DeepSeek-V2/V3); absent otherwise. ──
    /// Compressed KV latent width (the cache stores this per token, not full K/V).
    #[serde(default)]
    kv_lora_rank: Option<u32>,
    /// Query down-projection rank; absent means Q is projected directly.
    #[serde(default)]
    q_lora_rank: Option<u32>,
    /// Per-head query/key dim NOT carrying RoPE.
    #[serde(default)]
    qk_nope_head_dim: Option<u32>,
    /// Per-head query/key dim carrying the decoupled RoPE.
    #[serde(default)]
    qk_rope_head_dim: Option<u32>,
    /// Per-head value dim (may differ from the QK dim).
    #[serde(default)]
    v_head_dim: Option<u32>,
    /// Sliding-window attention span (Mistral). A query attends only the last
    /// `sliding_window` positions; absent/`null` means full causal attention.
    #[serde(default)]
    sliding_window: Option<u32>,
    /// Gemma2 applies its window to every `n`-th layer rather than all of them.
    /// HF omits this and hard-codes 2 in the model class, so it is defaulted for
    /// `model_type == "gemma2"`.
    #[serde(default)]
    sliding_window_pattern: Option<u32>,
    /// The same, as newer Gemma3 exports spell it. A separate field rather than
    /// an alias, so a config carrying both keys still parses.
    #[serde(default, rename = "_sliding_window_pattern")]
    sliding_window_pattern_private: Option<u32>,
    /// Per-layer attention kind (`"sliding_attention"` / `"full_attention"`),
    /// which newer Gemma3 exports ship in place of a pattern.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    /// Gemma3's RoPE base for its windowed layers.
    #[serde(default)]
    rope_local_base_freq: Option<f32>,
    /// Gemma2 attention-logit softcap (`tanh(score/cap)*cap`); typically 50.0.
    #[serde(default)]
    attn_logit_softcapping: Option<f32>,
    /// Gemma2 output-logit softcap applied to the LM head; typically 30.0.
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    /// Gemma2 decouples the attention scale from `head_dim` (144 on the 27B).
    #[serde(default)]
    query_pre_attn_scalar: Option<f32>,
    /// Falcon: one KV head shared by every query head.
    #[serde(default)]
    multi_query: Option<bool>,
    /// Falcon: attention and the FFN read the same norm and sum into one
    /// residual. False on `falcon-rw-*`, which is sequential.
    #[serde(default)]
    parallel_attn: Option<bool>,
    /// Falcon: ALiBi positional bias instead of RoPE. dlm does not implement it,
    /// so a checkpoint declaring it is refused rather than run without it --
    /// dropping a positional scheme yields fluent nonsense, not an error.
    #[serde(default)]
    alibi: Option<bool>,
    /// Falcon-40B's grouped-KV layout, which interleaves query_key_value by head
    /// group rather than concatenating Q|K|V. Refused: slicing it as a concat
    /// loads plausible, wrong weights.
    #[serde(default)]
    new_decoder_architecture: Option<bool>,
}

/// The subset of HF's `quantization_config` block dlm needs to decide whether it
/// can dequantize a checkpoint correctly. The dequantizer models canonical 4-bit
/// GPTQ (sequential nibble order, no act-order); anything else is refused up front
/// rather than producing plausible-looking garbage.
#[derive(Debug, Deserialize)]
struct QuantizationConfig {
    #[serde(default)]
    quant_method: Option<String>,
    #[serde(default)]
    bits: Option<u32>,
    #[serde(default)]
    group_size: Option<i64>,
    /// GPTQ act-order: weights are permuted by `g_idx` and must be un-permuted.
    #[serde(default)]
    desc_act: Option<bool>,
    /// `gptq_v2` stores the true zero-point; classic `gptq` stores `zero - 1`.
    #[serde(default)]
    checkpoint_format: Option<String>,
}

/// Which packed-4-bit family a checkpoint uses, and its variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedFormat {
    /// GPTQ. `act_order` (desc_act) scatters groups by `g_idx` — decoded to f32.
    Gptq { act_order: bool },
    /// AWQ (interleaved nibble order) — decoded to f32.
    Awq,
}

impl PackedFormat {
    /// True for the paths validated only by internal round-trip, not a real
    /// export — the loader warns for these (act-order GPTQ and AWQ).
    pub fn is_experimental(self) -> bool {
        matches!(
            self,
            PackedFormat::Gptq { act_order: true } | PackedFormat::Awq
        )
    }
}

/// A packed-quantized checkpoint dlm can decode, as declared by `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedQuant {
    /// Weights per quantization group along the input dimension.
    pub group_size: usize,
    /// The packed family/variant (drives which unpacker the loader uses).
    pub kind: PackedFormat,
}

/// Refuse quantized checkpoints the dequantizer can't decode correctly, with a
/// message naming the working alternative. Silent wrong output is worse than a
/// clear error — see [`crate::quant`] for what the canonical path handles.
/// Decide whether a packed-quantized checkpoint is one dlm can decode correctly,
/// returning its layout when it is.
///
/// Only the case that has been **validated against a real export** is accepted:
/// 4-bit GPTQ, `desc_act: false`, classic (`v1`) checkpoint format. Everything
/// else is refused by name rather than guessed at, because wrong-but-plausible
/// weights generate fluent nonsense — the worst failure mode there is.
fn check_quant_supported(q: &QuantizationConfig) -> Result<Option<PackedQuant>> {
    let method = q.quant_method.as_deref().unwrap_or("").to_ascii_lowercase();
    if method.is_empty() {
        return Ok(None); // a plain float checkpoint
    }
    if let Some(bits) = q.bits {
        if bits != 4 {
            return Err(DlmError::UnsupportedQuant(format!(
                "{bits}-bit {method} checkpoint; dlm decodes 4-bit only.                  Use an fp16/bf16 or 4-bit GPTQ (desc_act=false) checkpoint."
            )));
        }
    }
    let group_size = q.group_size.unwrap_or(-1);
    let need_group = || -> Result<usize> {
        if group_size <= 0 {
            return Err(DlmError::UnsupportedQuant(format!(
                "{method} checkpoint declares group_size {group_size}; dlm needs a positive \
                 per-group size (whole-row grouping is not supported)."
            )));
        }
        Ok(group_size as usize)
    };
    match method.as_str() {
        "gptq" => {
            // `gptq_v2` stores the true zero-point; classic `gptq` stores zero-1.
            // dlm's decoder assumes the classic convention (verified against a real
            // export) and has no v2 fixture to check the other against.
            match q
                .checkpoint_format
                .as_deref()
                .map(|f| f.to_ascii_lowercase())
            {
                None => {}
                Some(ref f) if f == "gptq" => {}
                Some(other) => {
                    return Err(DlmError::UnsupportedQuant(format!(
                        "GPTQ checkpoint_format {other:?} is not supported; dlm decodes the \
                         classic `gptq` format, whose zero-point convention it has been \
                         validated against."
                    )))
                }
            }
            // act-order (desc_act) decodes to f32 via `g_idx`; the loader warns
            // because that path is validated only by internal round-trip, not a
            // real export.
            let act_order = q.desc_act == Some(true);
            Ok(Some(PackedQuant {
                group_size: need_group()?,
                kind: PackedFormat::Gptq { act_order },
            }))
        }
        "awq" => Ok(Some(PackedQuant {
            group_size: need_group()?,
            kind: PackedFormat::Awq,
        })),
        other => Err(DlmError::UnsupportedQuant(format!(
            "unrecognized quant_method {other:?}; dlm loads fp16/bf16, GPTQ, and AWQ 4-bit \
             checkpoints."
        ))),
    }
}

/// Raw `rope_scaling` block. HF spells the discriminant `rope_type` on newer
/// configs and `type` on older ones; accept either.
#[derive(Debug, Deserialize)]
struct RawRopeScaling {
    #[serde(default, alias = "type")]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f32>,
    #[serde(default)]
    low_freq_factor: Option<f32>,
    #[serde(default)]
    high_freq_factor: Option<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<u32>,
    // YaRN.
    #[serde(default)]
    beta_fast: Option<f32>,
    #[serde(default)]
    beta_slow: Option<f32>,
    /// Explicit YaRN attention temperature; when absent it's `0.1·ln(factor)+1`.
    #[serde(default)]
    attention_factor: Option<f32>,
}

/// Convert a declared `rope_scaling` block into the [`RopeScaling`] the block
/// kernel applies.
///
/// A scaling type we do not implement is a hard error, never a silent skip: the
/// model was *trained* with that scaling, so ignoring it yields fluent-looking
/// garbage rather than an obvious failure. An explicit refusal is the only safe
/// behavior — see the `dlm` README on supported architectures.
fn parse_rope_scaling(r: &RawRopeScaling) -> Result<Option<RopeScaling>> {
    let kind = r.rope_type.as_deref().unwrap_or("").to_ascii_lowercase();
    let factor = r.factor.unwrap_or(1.0);
    match kind.as_str() {
        // `default`/absent means "no scaling" — plain RoPE.
        "" | "default" => Ok(None),
        "linear" => Ok(Some(RopeScaling::Linear { factor })),
        "llama3" => Ok(Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor: r.low_freq_factor.unwrap_or(1.0),
            high_freq_factor: r.high_freq_factor.unwrap_or(4.0),
            original_max_position: r.original_max_position_embeddings.unwrap_or(8192) as f32,
        })),
        "yarn" => Ok(Some(RopeScaling::Yarn {
            factor,
            original_max_position: r.original_max_position_embeddings.unwrap_or(4096) as f32,
            beta_fast: r.beta_fast.unwrap_or(32.0),
            beta_slow: r.beta_slow.unwrap_or(1.0),
            // HF: default attention factor is `0.1·ln(factor)+1` (1.0 if no scaling).
            mscale: r.attention_factor.unwrap_or(if factor > 1.0 {
                0.1 * factor.ln() + 1.0
            } else {
                1.0
            }),
        })),
        // The remaining HF variants are refused with the specific reason, so a
        // user hitting one knows what is missing rather than just that it is.
        // Running any of them as plain RoPE produces fluent-looking nonsense past
        // the original context length, so none is silently approximated.
        "longrope" | "su" => Err(DlmError::InvalidConfig(format!(
            "rope_scaling type {other:?} (Phi-3 long-context) is not implemented: it needs \
             per-dimension short_factor/long_factor arrays, which dlm's inverse-frequency \
             table does not yet carry. Use a Phi-3 checkpoint at its base context, or run \
             a model with \"linear\", \"llama3\", or \"yarn\" scaling.",
            other = kind
        ))),
        "dynamic" => Err(DlmError::InvalidConfig(
            "rope_scaling type \"dynamic\" (dynamic NTK) is not implemented: its frequencies \
             depend on the *current* sequence length, but dlm precomputes one inverse-frequency \
             table per model and uploads it to the device once. Supporting it means recomputing \
             (and re-uploading) that table as the sequence grows. Use \"linear\", \"llama3\", or \
             \"yarn\"."
                .into(),
        )),
        "mrope" => Err(DlmError::InvalidConfig(
            "rope_scaling type \"mrope\" is multimodal (Qwen2-VL) and has no meaning for a \
             text-only engine; dlm does not run vision checkpoints."
                .into(),
        )),
        other => Err(DlmError::InvalidConfig(format!(
            "rope_scaling type {other:?} is not implemented; dlm supports \"linear\", \"llama3\", \
             and \"yarn\". Running this model without its trained RoPE scaling would produce \
             incoherent output, so it is refused rather than silently mis-run."
        ))),
    }
}

/// Derive validated [`MlaConfig`] from the raw config, or `None` for standard
/// attention. `kv_lora_rank` marks a checkpoint as MLA (DeepSeek); once seen, the
/// remaining latent-attention dims are required — a partial declaration is refused
/// rather than guessed at.
fn build_mla_config(raw: &RawConfig) -> Result<Option<MlaConfig>> {
    let Some(kv_lora_rank) = raw.kv_lora_rank else {
        return Ok(None);
    };
    let need = |v: Option<u32>, name: &str| -> Result<u32> {
        v.ok_or_else(|| {
            DlmError::InvalidConfig(format!(
                "MLA checkpoint (kv_lora_rank set) is missing {name}; dlm will not guess \
                 latent-attention dims."
            ))
        })
    };
    Ok(Some(MlaConfig {
        q_lora_rank: raw.q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim: need(raw.qk_nope_head_dim, "qk_nope_head_dim")?,
        qk_rope_head_dim: need(raw.qk_rope_head_dim, "qk_rope_head_dim")?,
        v_head_dim: need(raw.v_head_dim, "v_head_dim")?,
    }))
}

/// Derive validated [`MoeConfig`] from the raw config, or `None` for a dense
/// model. Mixtral declares experts under `num_local_experts`, Qwen under
/// `num_experts`; the presence of either marks the checkpoint MoE and also picks
/// the tensor naming family. A model that declares experts but omits the top-k
/// count is refused rather than guessed at — routing every token through the
/// wrong number of experts is silent garbage, the worst failure mode.
fn build_moe_config(raw: &RawConfig) -> Result<Option<MoeConfig>> {
    let (num_experts, naming) = match (raw.num_local_experts, raw.num_experts, raw.n_routed_experts)
    {
        (Some(n), _, _) => (n, MoeNaming::Mixtral),
        (None, Some(n), _) => (n, MoeNaming::Qwen),
        (None, None, Some(n)) => (n, MoeNaming::DeepSeek),
        (None, None, None) => return Ok(None),
    };
    if num_experts == 0 {
        return Ok(None); // an expert count of 0 is just a dense model
    }
    let experts_per_tok = raw.num_experts_per_tok.ok_or_else(|| {
        DlmError::InvalidConfig(
            "config declares experts but no num_experts_per_tok; dlm will not guess the \
             routing top-k, as the wrong count produces plausible-looking garbage."
                .into(),
        )
    })?;
    if experts_per_tok == 0 || experts_per_tok > num_experts {
        return Err(DlmError::InvalidConfig(format!(
            "num_experts_per_tok ({experts_per_tok}) must be in 1..={num_experts}"
        )));
    }
    // Mixtral has no separate expert width — its experts use `intermediate_size`.
    let moe_intermediate_size = raw
        .moe_intermediate_size
        .or(raw.intermediate_size)
        .ok_or_else(|| {
            DlmError::InvalidConfig(
                "MoE config declares neither moe_intermediate_size nor intermediate_size".into(),
            )
        })?;
    // DeepSeek states a shared-expert *count*; its shared FFN is that many
    // `moe_intermediate_size` experts fused into one. Qwen states the width
    // directly.
    let shared_intermediate_size = match naming {
        MoeNaming::DeepSeek => raw
            .n_shared_experts
            .filter(|n| *n > 0)
            .map(|n| n * moe_intermediate_size),
        _ => raw.shared_expert_intermediate_size,
    };

    // Only "every layer after the dense prefix" is implemented. A larger period
    // would make some later layers dense too, and guessing wrong loads a routed
    // layer as dense — a missing-tensor error at best, wrong weights at worst.
    if let Some(freq) = raw.moe_layer_freq.filter(|f| *f != 1) {
        return Err(DlmError::InvalidConfig(format!(
            "moe_layer_freq {freq} is not implemented; dlm places a MoE layer at every \
             layer after the first {} dense one(s), and will not guess a sparser pattern.",
            raw.first_k_dense_replace.unwrap_or(0)
        )));
    }

    Ok(Some(MoeConfig {
        num_experts,
        experts_per_tok,
        moe_intermediate_size,
        shared_intermediate_size,
        // Mixtral always renormalizes; Qwen exposes the flag (default on).
        norm_topk_prob: raw.norm_topk_prob.unwrap_or(true),
        naming,
        first_k_dense: raw.first_k_dense_replace.unwrap_or(0),
    }))
}

/// `eos_token_id` as it appears in `config.json`: one id or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosField {
    One(u32),
    Many(Vec<u32>),
}

/// Which family's tensor names an MoE checkpoint uses. Mixtral and Qwen lay the
/// router and per-expert FFN out under different prefixes; the loader keys off
/// this to build the right names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeNaming {
    /// `block_sparse_moe.gate`, `block_sparse_moe.experts.{e}.{w1,w3,w2}`
    /// (w1=gate, w3=up, w2=down). No shared expert.
    Mixtral,
    /// `mlp.gate`, `mlp.experts.{e}.{gate,up,down}_proj`, optional
    /// `mlp.shared_expert.*` gated by `mlp.shared_expert_gate`.
    Qwen,
    /// DeepSeek-V2/V3: like Qwen, but the shared expert is `mlp.shared_experts`
    /// (plural) and is **ungated** — it is added straight to the routed sum, with
    /// no `shared_expert_gate` tensor in the checkpoint.
    DeepSeek,
}

/// Validated MoE geometry, present only on Mixture-of-Experts checkpoints.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    /// Total routed experts per layer.
    pub num_experts: u32,
    /// Experts activated per token (top-k).
    pub experts_per_tok: u32,
    /// Per-expert FFN inner width.
    pub moe_intermediate_size: u32,
    /// Shared-expert FFN inner width, when the model has one (Qwen2-MoE).
    pub shared_intermediate_size: Option<u32>,
    /// Renormalize the top-k gate weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Expert tensor naming family.
    pub naming: MoeNaming,
    /// How many leading layers are **dense** rather than routed (DeepSeek's
    /// `first_k_dense_replace`). `0` — every other MoE family — means every layer
    /// is routed. Resolved per layer by [`BlockConfig::for_layer`].
    pub first_k_dense: u32,
}

/// Validated Multi-head Latent Attention geometry (DeepSeek-V2/V3). Present only
/// on MLA checkpoints; the attention path caches a compressed latent per token
/// (`kv_lora_rank` + `qk_rope_head_dim`) instead of full per-head K/V.
#[derive(Debug, Clone, Copy)]
pub struct MlaConfig {
    /// Query down-projection rank; `None` projects Q directly from the hidden.
    pub q_lora_rank: Option<u32>,
    /// Compressed KV latent width (what the cache stores per token).
    pub kv_lora_rank: u32,
    /// Per-head query/key dim without RoPE.
    pub qk_nope_head_dim: u32,
    /// Per-head query/key dim carrying the decoupled RoPE.
    pub qk_rope_head_dim: u32,
    /// Per-head value dim.
    pub v_head_dim: u32,
}

impl MlaConfig {
    /// Total per-head query/key dim (`nope + rope`).
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// Validated model geometry consumed by the profiler and storage planner.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model embedding / residual stream width (`d_model`).
    pub hidden_size: u32,
    /// Number of query attention heads.
    pub num_attention_heads: u32,
    /// Number of key/value heads. Equals `num_attention_heads` for vanilla MHA;
    /// smaller under Grouped-Query Attention (GQA), which shrinks the KV cache.
    pub num_kv_heads: u32,
    /// Number of transformer blocks — the layers `dlm` streams in and out.
    pub num_layers: u32,
    /// Vocabulary size (drives embedding + LM head parameter counts).
    pub vocab_size: u32,
    /// FFN inner dimension; falls back to `4 * hidden_size` when absent.
    pub intermediate_size: u32,
    /// Model's own maximum context, if declared.
    pub max_position_embeddings: Option<u32>,
    /// EOS token id(s) declared by the model; empty when the config omits them.
    /// Generation stops when any of these is produced.
    pub eos_token_ids: Vec<u32>,
    /// RoPE base frequency (default 10000).
    pub rope_theta: f32,
    /// RMSNorm epsilon (default 1e-5).
    pub rms_eps: f32,
    /// RoPE frequency scaling declared by the model, if any.
    pub rope_scaling: Option<RopeScaling>,
    /// Explicit per-head dim when the config declares one; otherwise `None` and
    /// [`head_dim`](Self::head_dim) falls back to `hidden_size / num_heads`.
    pub explicit_head_dim: Option<u32>,
    /// On-disk weight precision.
    pub quant: QuantScheme,
    /// Set when the checkpoint ships **already** packed-quantized (4-bit GPTQ)
    /// rather than as floats. Its codes are decoded as they are — no
    /// re-quantization — so the calibration the export paid for survives.
    pub packed_quant: Option<PackedQuant>,
    /// Mixture-of-Experts geometry when the checkpoint is sparse; `None` for a
    /// dense model, which keeps the single-FFN path unchanged.
    pub moe: Option<MoeConfig>,
    /// Multi-head Latent Attention geometry (DeepSeek-V2/V3); `None` for standard
    /// GQA/MHA attention.
    pub mla: Option<MlaConfig>,
    /// Sliding-window attention span (Mistral); `None` is full causal attention.
    pub sliding_window: Option<u32>,
    /// Gemma applies RMSNorm as `(1 + weight)` rather than `weight`. When true the
    /// loader bakes the `+1` into the norm weights so the kernels stay unchanged.
    pub norm_add_one: bool,
    /// RMSNorm (Llama-descended) or LayerNorm (GPT-2, Falcon).
    pub norm_kind: crate::forward::cpu::NormKind,
    /// Falcon: attention and FFN read the same normalized input, summed into one
    /// residual.
    pub parallel_residual: bool,
    /// GPT-2: absolute learned position embeddings instead of RoPE.
    pub learned_positions: bool,
    /// Whether the MLP is gated (SwiGLU/GeGLU). `Plain` only for GPT-2.
    pub ffn_kind: crate::forward::cpu::FfnKind,
    /// Scalar applied to token embeddings after lookup (Gemma multiplies by
    /// `sqrt(hidden_size)`); `None` leaves embeddings unscaled.
    pub embed_scale: Option<f32>,
    /// Gated-MLP activation (SiLU for most, GELU for Gemma).
    pub activation: crate::forward::cpu::Activation,
    /// Gemma2/3 make every `n`-th layer global and window the rest; `None`
    /// applies [`sliding_window`](Self::sliding_window) uniformly.
    pub sliding_window_pattern: Option<u32>,
    /// Gemma3's RoPE base for windowed layers; `None` elsewhere.
    pub rope_local_theta: Option<f32>,
    /// Gemma2 attention-logit softcap (`tanh(score/cap)*cap`).
    pub attn_logit_softcap: Option<f32>,
    /// Gemma2 LM-head logit softcap, applied to the final logits before sampling.
    pub final_logit_softcap: Option<f32>,
    /// Attention-scale divisor when decoupled from `head_dim` (Gemma2).
    pub query_pre_attn_scalar: Option<f32>,
    /// Gemma2 carries an extra pre/post-FFN norm pair per layer, which also
    /// changes where the other two norms apply (see
    /// [`LayerTensors::is_gemma2_style`](crate::forward::LayerTensors::is_gemma2_style)).
    pub gemma2_norms: bool,
}

/// The text model of a multimodal Gemma 3 config (`model_type: "gemma3"`, the
/// 4B/12B/27B checkpoints), or `json` unchanged for anything else.
///
/// Those configs nest the language model under `text_config`, next to a vision
/// tower dlm does not run. Google's own exports keep `text_config` sparse and
/// rely on `Gemma3TextConfig`'s defaults for everything they omit, so the
/// defaults are filled in here -- a missing head count or window must not fall
/// through to a generic default that happens to parse. Top-level keys the text
/// model also needs (the EOS/BOS ids, a quantization block) are carried over
/// unless `text_config` sets them itself.
fn gemma3_text_config(json: serde_json::Value) -> serde_json::Value {
    use serde_json::{json, Value};
    let is_multimodal_gemma3 = json.get("model_type").and_then(Value::as_str) == Some("gemma3");
    let Some(Value::Object(mut text)) = json
        .get("text_config")
        .cloned()
        .filter(|_| is_multimodal_gemma3)
    else {
        return json;
    };
    for key in [
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "quantization_config",
    ] {
        if let (false, Some(v)) = (text.contains_key(key), json.get(key)) {
            text.insert(key.to_string(), v.clone());
        }
    }
    // transformers' Gemma3TextConfig defaults, for the keys a sparse export omits.
    let defaults = json!({
        "model_type": "gemma3_text",
        "vocab_size": 262208,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0,
        "query_pre_attn_scalar": 256,
        "sliding_window": 4096,
        "sliding_window_pattern": 6,
    });
    for (key, value) in defaults.as_object().expect("literal object") {
        text.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Value::Object(text)
}

/// Turn Gemma3's `layer_types` list into the `n` of "every `n`-th layer is
/// global". Only that regular shape is accepted: a list dlm cannot express as a
/// pattern is refused rather than approximated, since windowing a global layer
/// (or the reverse) runs quietly wrong past the window length.
fn pattern_from_layer_types(types: &[String]) -> Result<Option<u32>> {
    let global = |t: &String| t == "full_attention";
    let Some(first) = types.iter().position(global) else {
        return Ok(None); // every layer windowed: a uniform window
    };
    let n = first + 1;
    let regular = types.iter().enumerate().all(|(i, t)| {
        let expect_global = (i + 1) % n == 0;
        global(t) == expect_global && (global(t) || t == "sliding_attention")
    });
    if !regular {
        return Err(DlmError::InvalidConfig(format!(
            "layer_types {types:?} is not a regular sliding/full pattern; dlm supports \
             every n-th layer global (Gemma2/Gemma3), and running an irregular layout \
             under that rule would window the wrong layers"
        )));
    }
    Ok(Some(n as u32))
}

impl ModelConfig {
    /// True when this is a Mixture-of-Experts checkpoint.
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }
}

impl ModelConfig {
    /// Load and validate a `config.json` from a model directory or file path.
    /// If `path` is a directory, `config.json` inside it is used — and any
    /// `generation_config.json` beside it is merged in (see [`merge_generation_config`]).
    pub fn from_path(path: impl AsRef<Path>, quant: QuantScheme) -> Result<Self> {
        let path = path.as_ref();
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.to_path_buf()
        };

        let bytes = std::fs::read(&config_path).map_err(|source| DlmError::Io {
            path: config_path.clone(),
            source,
        })?;

        let mut config = Self::from_json_bytes(&bytes, quant)?;

        // HuggingFace treats `generation_config.json` as authoritative for
        // generation parameters, and models routinely declare a *larger* EOS set
        // there than in config.json. Qwen2.5 lists only `<|im_end|>` in
        // config.json but both `<|im_end|>` and `<|endoftext|>` in the generation
        // config — miss the second and the model never stops: it emits the token,
        // generation runs on to the token limit, and the special token itself
        // leaks into the reply.
        if let Some(dir) = config_path.parent() {
            let gen_path = dir.join("generation_config.json");
            if let Ok(gen_bytes) = std::fs::read(&gen_path) {
                config.merge_generation_config(&gen_bytes)?;
            }
        }
        Ok(config)
    }

    /// Merge a `generation_config.json` over this config: union the EOS ids it
    /// declares into [`eos_token_ids`](Self::eos_token_ids). Stopping on any of
    /// them is correct, so a union (rather than a replace) is the safe merge.
    pub fn merge_generation_config(&mut self, bytes: &[u8]) -> Result<()> {
        #[derive(Deserialize)]
        struct GenConfig {
            #[serde(default)]
            eos_token_id: Option<EosField>,
        }
        let gen: GenConfig = serde_json::from_slice(bytes).map_err(|source| DlmError::Json {
            context: "generation_config.json".to_string(),
            source,
        })?;
        let extra = match gen.eos_token_id {
            Some(EosField::One(id)) => vec![id],
            Some(EosField::Many(ids)) => ids,
            None => Vec::new(),
        };
        for id in extra {
            if !self.eos_token_ids.contains(&id) {
                self.eos_token_ids.push(id);
            }
        }
        Ok(())
    }

    /// Parse a config from raw JSON bytes. Separated from [`from_path`] so it
    /// can be unit-tested without touching the filesystem.
    pub fn from_json_bytes(bytes: &[u8], quant: QuantScheme) -> Result<Self> {
        let json: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|source| DlmError::Json {
                context: "config.json".to_string(),
                source,
            })?;
        let raw: RawConfig =
            serde_json::from_value(gemma3_text_config(json)).map_err(|source| DlmError::Json {
                context: "config.json".to_string(),
                source,
            })?;

        // Reject quant formats the decoder would silently mis-decode; keep the
        // layout of the one it can.
        let packed_quant = match &raw.quantization_config {
            Some(qc) => check_quant_supported(qc)?,
            None => None,
        };

        let moe = build_moe_config(&raw)?;
        let mla = build_mla_config(&raw)?;

        // Gemma architecture variants: (1+w) RMSNorm, embedding scaling, GeGLU.
        // Gemma2 adds logit softcapping, alternating window layers, a decoupled
        // attention scale, and a second norm pair per layer.
        let model_type = raw.model_type.as_deref().unwrap_or("").to_ascii_lowercase();
        let is_gemma2 = model_type == "gemma2";
        // Gemma3: Gemma2's norm layout plus per-head Q/K norms (loaded when
        // present), a 5:1 local/global pattern, and a separate RoPE base for the
        // local layers. `gemma3_text` is the text-only export (270M, 1B).
        let is_gemma3 = model_type == "gemma3_text" || model_type == "gemma3";
        let is_gemma = model_type == "gemma" || is_gemma2 || is_gemma3;
        // GPT-2 and Falcon are the two families that are not Llama-descended:
        // both normalize with LayerNorm rather than RMSNorm, and each changes the
        // block in one further way. Keyed on `model_type` rather than
        // `architectures`, since the latter varies between exports of the same
        // model while `model_type` does not.
        let is_gpt2 = model_type == "gpt2";
        // `refinedweb`/`RWForCausalLM` is Falcon's original name; both are live in
        // the wild, so both must map to the same block.
        let is_falcon = matches!(
            model_type.as_str(),
            "falcon" | "refinedweb" | "refinedwebmodel"
        );
        // Two Falcon variants dlm cannot decode correctly, refused up front. Both
        // would otherwise load and run: a dropped positional scheme and a
        // mis-sliced fused tensor each produce fluent, wrong output rather than
        // an error, which is the failure this project refuses to ship.
        if is_falcon {
            if raw.alibi.unwrap_or(false) {
                return Err(DlmError::InvalidConfig(
                    "this Falcon checkpoint uses ALiBi positional bias (`alibi: true`, e.g. \
                     falcon-rw-*), which dlm does not implement. Running it without ALiBi \
                     would not error -- it would silently mis-place every token. Use a \
                     variant with `alibi: false`."
                        .into(),
                ));
            }
            if raw.new_decoder_architecture.unwrap_or(false) {
                return Err(DlmError::InvalidConfig(
                    "this Falcon checkpoint sets `new_decoder_architecture` (Falcon-40B), whose \
                     query_key_value is interleaved by head group rather than concatenated \
                     Q|K|V. dlm splits the concatenated layout; slicing the interleaved one \
                     loads plausible but incorrect weights."
                        .into(),
                ));
            }
        }
        let norm_kind = if is_gpt2 || is_falcon {
            crate::forward::cpu::NormKind::Layer
        } else {
            crate::forward::cpu::NormKind::Rms
        };
        let activation = match raw
            .hidden_activation
            .as_deref()
            .or(raw.hidden_act.as_deref())
        {
            Some(a) if a.to_ascii_lowercase().contains("gelu") => {
                crate::forward::cpu::Activation::GeluTanh
            }
            None if is_gemma => crate::forward::cpu::Activation::GeluTanh,
            _ => crate::forward::cpu::Activation::Silu,
        };

        let config = ModelConfig {
            hidden_size: raw.hidden_size,
            num_attention_heads: raw.num_attention_heads,
            // Default to full multi-head attention when kv-heads is unspecified.
            num_kv_heads: raw.num_key_value_heads.unwrap_or_else(|| {
                // Falcon states this as a flag, not a count: `multi_query` means
                // one KV head shared by every query head.
                if is_falcon && raw.multi_query.unwrap_or(false) {
                    1
                } else {
                    raw.num_attention_heads
                }
            }),
            num_layers: raw.num_hidden_layers,
            vocab_size: raw.vocab_size,
            intermediate_size: raw
                .intermediate_size
                .unwrap_or(raw.hidden_size.saturating_mul(4)),
            max_position_embeddings: raw.max_position_embeddings,
            eos_token_ids: match raw.eos_token_id {
                Some(EosField::One(id)) => vec![id],
                Some(EosField::Many(ids)) => ids,
                None => Vec::new(),
            },
            rope_theta: raw.rope_theta.unwrap_or(10000.0),
            rms_eps: raw.rms_norm_eps.unwrap_or(1e-5),
            rope_scaling: match &raw.rope_scaling {
                Some(r) => parse_rope_scaling(r)?,
                None => None,
            },
            explicit_head_dim: raw.head_dim,
            quant,
            packed_quant,
            moe,
            mla,
            // A window >= the model's own max context is the same as full
            // attention; keep it as declared and let the kernel no-op it.
            sliding_window: raw.sliding_window.filter(|&w| w > 0),
            norm_add_one: is_gemma,
            norm_kind,
            parallel_residual: is_falcon && raw.parallel_attn.unwrap_or(true),
            learned_positions: is_gpt2,
            // Falcon's MLP is `dense_4h_to_h(gelu(dense_h_to_4h(x)))` -- ungated,
            // like GPT-2 and unlike every Llama-descended family.
            ffn_kind: if is_gpt2 || is_falcon {
                crate::forward::cpu::FfnKind::Plain
            } else {
                crate::forward::cpu::FfnKind::Gated
            },
            embed_scale: is_gemma.then(|| (raw.hidden_size as f32).sqrt()),
            activation,
            // HF hard-codes the alternation in the Gemma2 model class rather than
            // the config (`is_sliding = not bool(layer_idx % 2)`), so default it
            // here instead of requiring a key the checkpoints don't ship.
            sliding_window_pattern: match (&raw.layer_types, is_gemma3) {
                (Some(types), true) => pattern_from_layer_types(types)?,
                _ => raw
                    .sliding_window_pattern
                    .or(raw.sliding_window_pattern_private)
                    .or(if is_gemma2 {
                        Some(2)
                    } else if is_gemma3 {
                        Some(6)
                    } else {
                        None
                    }),
            }
            .filter(|&n| n > 1),
            rope_local_theta: is_gemma3.then(|| raw.rope_local_base_freq.unwrap_or(10_000.0)),
            attn_logit_softcap: raw.attn_logit_softcapping.filter(|c| *c > 0.0),
            final_logit_softcap: raw.final_logit_softcapping.filter(|c| *c > 0.0),
            query_pre_attn_scalar: raw.query_pre_attn_scalar.filter(|s| *s > 0.0),
            gemma2_norms: is_gemma2 || is_gemma3,
        };

        config.validate()?;
        Ok(config)
    }

    /// Reject configs that would make the VRAM math divide by zero or produce
    /// nonsense head dimensions.
    fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0 {
            return Err(DlmError::InvalidConfig(
                "num_attention_heads must be > 0".into(),
            ));
        }
        if self.num_layers == 0 {
            return Err(DlmError::InvalidConfig(
                "num_hidden_layers must be > 0".into(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(DlmError::InvalidConfig("hidden_size must be > 0".into()));
        }
        // MLA carries its own per-head dims (qk_nope/qk_rope/v), so the standard
        // hidden÷heads and even-head-dim invariants don't apply; only the RoPE
        // sub-dimension must be even.
        if let Some(m) = &self.mla {
            if m.qk_rope_head_dim % 2 != 0 {
                return Err(DlmError::InvalidConfig(format!(
                    "qk_rope_head_dim ({}) must be even (RoPE rotates dimension pairs)",
                    m.qk_rope_head_dim
                )));
            }
            return Ok(());
        }
        // Only the derived head_dim needs the divisibility guarantee; a config
        // that states head_dim outright is free to break the quotient relation.
        if self.explicit_head_dim.is_none() && self.hidden_size % self.num_attention_heads != 0 {
            return Err(DlmError::InvalidConfig(format!(
                "hidden_size ({}) is not divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(DlmError::InvalidConfig(format!(
                "head_dim ({}) must be even (RoPE rotates dimension pairs)",
                self.head_dim()
            )));
        }
        Ok(())
    }

    /// Per-head dimension: the config's explicit `head_dim` when it declares one,
    /// else `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> u32 {
        self.explicit_head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Attention-projection parameters for one layer: q + o (`h*h` each) and
    /// k + v, scaled by the GQA ratio (`2 * kv_ratio * h*h`).
    fn attn_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let kv_ratio = self.num_kv_heads as f64 / self.num_attention_heads as f64;
        (2.0 * (h * h) as f64 + 2.0 * kv_ratio * (h * h) as f64) as u64
    }

    /// Parameters in one routed MoE expert's SwiGLU triple (gate + up + down),
    /// or `None` for a dense model. This is the unit that streams per
    /// `(layer, expert)` on the GPU MoE path, so it sizes both the expert-cache
    /// budget and the per-expert PCIe cost.
    pub fn expert_params(&self) -> Option<u64> {
        let h = self.hidden_size as u64;
        self.moe
            .as_ref()
            .map(|m| 3 * h * m.moe_intermediate_size as u64)
    }

    /// Parameters that stay **resident per layer** on the streaming path: the
    /// whole layer for a dense model, but for an MoE model only the *core*
    /// (attention + router + optional shared expert + norms) — the routed experts
    /// stream separately into the per-`(layer, expert)` cache and are *not*
    /// resident. The VRAM planner sizes the resident layer window from this, so a
    /// sparse layer isn't mis-planned as if it held all its experts at once.
    pub fn resident_layer_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let norms = 2 * h;
        let ffn = match &self.moe {
            None => 3 * h * self.intermediate_size as u64,
            Some(m) => {
                let router = h * m.num_experts as u64;
                let shared = m
                    .shared_intermediate_size
                    .map_or(0, |s| 3 * h * s as u64 + h);
                // Core only — routed experts are excluded (they stream on demand).
                router + shared
            }
        };
        self.attn_params() + ffn + norms
    }

    /// How many routed experts a VRAM cache of `budget_bytes` can hold, in the
    /// precision the experts land in VRAM (`self.quant`). Clamped so at least a
    /// token's top-k stay resident (progress + intra-token reuse) and never more
    /// than the model has across every layer. `None` for a dense model.
    ///
    /// This is what turns the old count-heuristic into a real VRAM budget: on a
    /// fine-grained (128-expert) checkpoint an unbounded count would OOM the card.
    pub fn expert_cache_capacity(&self, budget_bytes: u64) -> Option<usize> {
        let m = self.moe.as_ref()?;
        let per_expert =
            (self.expert_params()? as f64 * self.quant.bytes_per_param()).ceil() as u64;
        let fit = budget_bytes.checked_div(per_expert).unwrap_or(0) as usize;
        let lo = m.experts_per_tok as usize;
        let hi = m.num_experts as usize * self.num_layers as usize;
        Some(fit.clamp(lo, hi))
    }

    /// Rough total parameter count for the whole model, used to estimate the
    /// average size of one streamed transformer block.
    ///
    /// Approximates each transformer layer as attention projections
    /// (`4 * hidden^2`, folding GQA into a smaller KV share) plus the FFN
    /// (`3 * hidden * intermediate`, covering gate/up/down in SwiGLU MLPs),
    /// and adds the tied embedding + LM head (`2 * vocab * hidden`).
    pub fn estimated_total_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let inter = self.intermediate_size as u64;

        // FFN: one dense SwiGLU (3*h*inter), or per layer the full set of expert
        // FFNs plus an optional shared expert for MoE. The catalog path measures
        // real per-layer bytes; this estimate only backs the fallback planner.
        let ffn = match &self.moe {
            None => 3 * h * inter,
            Some(m) => {
                let moe_inter = m.moe_intermediate_size as u64;
                let experts = 3 * h * moe_inter * m.num_experts as u64;
                let router = h * m.num_experts as u64;
                let shared = m
                    .shared_intermediate_size
                    .map_or(0, |s| 3 * h * s as u64 + h);
                experts + router + shared
            }
        };
        let per_layer = self.attn_params() + ffn;

        let blocks = per_layer * self.num_layers as u64;
        let embed_and_head = 2 * self.vocab_size as u64 * h;
        blocks + embed_and_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 3 states its layer layout two ways; both mean "every 6th layer is
    /// global". An irregular list is refused, not approximated.
    /// A multimodal Gemma 3 config as Google exports the 27B: the text model is
    /// nested and sparse. Everything omitted must come from Gemma3TextConfig's
    /// defaults, and the EOS id from the top level.
    #[test]
    fn multimodal_gemma3_config_reads_its_sparse_text_config() {
        let json = br#"{"architectures":["Gemma3ForConditionalGeneration"],
            "model_type":"gemma3","eos_token_id":[1,106],
            "text_config":{"head_dim":128,"hidden_size":5376,"intermediate_size":21504,
              "model_type":"gemma3_text","num_attention_heads":32,"num_hidden_layers":62,
              "num_key_value_heads":16,"query_pre_attn_scalar":168,
              "rope_scaling":{"factor":8.0,"rope_type":"linear"},"sliding_window":1024},
            "vision_config":{"hidden_size":1152,"num_hidden_layers":27}}"#;
        let c = ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap();
        assert_eq!(
            (c.hidden_size, c.num_layers),
            (5376, 62),
            "not the vision tower's"
        );
        assert_eq!((c.num_attention_heads, c.num_kv_heads), (32, 16));
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.vocab_size, 262208, "default");
        assert_eq!(c.sliding_window, Some(1024));
        assert_eq!(c.sliding_window_pattern, Some(6), "default");
        assert_eq!(c.rope_local_theta, Some(10_000.0));
        assert_eq!(c.rope_theta, 1_000_000.0, "default");
        assert_eq!(c.query_pre_attn_scalar, Some(168.0));
        assert!(matches!(
            c.rope_scaling,
            Some(crate::forward::cpu::RopeScaling::Linear { .. })
        ));
        assert_eq!(c.eos_token_ids, vec![1, 106], "carried from the top level");
        assert!(c.gemma2_norms && c.norm_add_one);
        assert!((c.rms_eps - 1e-6).abs() < 1e-9, "default");
    }

    #[test]
    fn gemma3_layer_pattern_from_either_spelling() {
        let base = r#""model_type":"gemma3_text","hidden_size":16,"num_attention_heads":4,
            "num_key_value_heads":1,"num_hidden_layers":12,"vocab_size":32,
            "intermediate_size":32,"sliding_window":8,"rope_local_base_freq":10000"#;
        let parse = |extra: &str| {
            ModelConfig::from_json_bytes(format!("{{{base}{extra}}}").as_bytes(), QuantScheme::Fp16)
        };
        let kinds = |globals: &[usize]| {
            let v: Vec<String> = (0..12)
                .map(|i| {
                    format!(
                        "{:?}",
                        if globals.contains(&i) {
                            "full_attention"
                        } else {
                            "sliding_attention"
                        }
                    )
                })
                .collect();
            format!(r#","layer_types":[{}]"#, v.join(","))
        };
        for extra in [
            r#","sliding_window_pattern":6"#.to_string(),
            r#","_sliding_window_pattern":6"#.to_string(),
            kinds(&[5, 11]),
            String::new(), // absent: Gemma 3's default
        ] {
            let c = parse(&extra).unwrap();
            assert_eq!(c.sliding_window_pattern, Some(6), "{extra}");
            assert_eq!(c.rope_local_theta, Some(10_000.0));
            assert!(c.gemma2_norms && c.norm_add_one);
        }
        let err = parse(&kinds(&[4, 11])).expect_err("irregular layout must be refused");
        assert!(format!("{err}").contains("layer_types"), "{err}");
    }

    /// DeepSeek-V2/V3 spells its MoE fields differently from Mixtral and Qwen.
    /// Reading only the other two spellings left `moe: None`, so every layer
    /// loaded as dense and the first routed layer failed on a missing
    /// `mlp.gate_proj` — the whole family was unloadable.
    #[test]
    fn deepseek_moe_config_is_recognized() {
        // Shape of deepseek-ai/DeepSeek-V2-Lite-Chat's config.json.
        let json = br#"{"model_type":"deepseek_v2","hidden_size":2048,
            "num_attention_heads":16,"num_key_value_heads":16,"num_hidden_layers":27,
            "vocab_size":102400,"intermediate_size":10944,"moe_intermediate_size":1408,
            "n_routed_experts":64,"n_shared_experts":2,"num_experts_per_tok":6,
            "first_k_dense_replace":1,"moe_layer_freq":1}"#;
        let c = ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap();
        let m = c
            .moe
            .expect("DeepSeek declares experts via n_routed_experts");
        assert_eq!(m.num_experts, 64);
        assert_eq!(m.experts_per_tok, 6);
        assert_eq!(m.naming, MoeNaming::DeepSeek);
        // The shared expert is a *count* of moe_intermediate_size experts fused
        // into one: 2 × 1408. Reading it as a width would size it 1408.
        assert_eq!(m.shared_intermediate_size, Some(2816));
        assert_eq!(
            m.first_k_dense, 1,
            "layer 0 is dense, layers 1.. are routed"
        );
    }

    /// A sparser MoE period is refused rather than guessed: loading a routed
    /// layer as dense is a missing-tensor error at best, wrong weights at worst.
    #[test]
    fn unimplemented_moe_layer_freq_is_refused() {
        let json = br#"{"model_type":"deepseek_v2","hidden_size":16,
            "num_attention_heads":4,"num_hidden_layers":4,"vocab_size":32,
            "intermediate_size":8,"moe_intermediate_size":4,
            "n_routed_experts":8,"num_experts_per_tok":2,"moe_layer_freq":2}"#;
        let err = ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap_err();
        assert!(
            format!("{err}").contains("moe_layer_freq"),
            "expected a moe_layer_freq refusal, got: {err}"
        );
    }

    fn moe_config() -> ModelConfig {
        // 8 experts, top-2, no shared expert (Mixtral-shaped).
        let json = br#"{"hidden_size":16,"num_attention_heads":4,"num_key_value_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":8,
            "num_local_experts":8,"num_experts_per_tok":2}"#;
        ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap()
    }

    #[test]
    fn resident_core_excludes_routed_experts() {
        let c = moe_config();
        let h = 16u64;
        // One expert: 3 * h * moe_inter (= intermediate_size 8 for Mixtral).
        assert_eq!(c.expert_params(), Some(3 * h * 8));
        // The resident core must NOT include the 8 experts — only attn + router +
        // norms. So it is far smaller than a full layer with all experts.
        let core = c.resident_layer_params();
        let one_expert = c.expert_params().unwrap();
        assert!(
            core < one_expert * 8,
            "core {core} should exclude all 8 experts ({}/expert)",
            one_expert
        );
        // Core = attn (q,o = h*h each; k,v full since MHA) + router (h*8) + 2 norms.
        let expected_core = 4 * h * h + h * 8 + 2 * h;
        assert_eq!(core, expected_core);
    }

    /// Every Gemma2 config Google publishes carries `hidden_act` *and*
    /// `hidden_activation`. One field aliasing the other made serde reject the
    /// whole file as a duplicate, so no Gemma2 checkpoint could be loaded at all.
    #[test]
    fn gemma2_config_with_both_activation_spellings_parses() {
        use crate::forward::cpu::Activation;
        // Verbatim shape of google/gemma-2-2b-it's config.json.
        let gemma2 = br#"{"model_type":"gemma2","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64,
            "hidden_act":"gelu_pytorch_tanh","hidden_activation":"gelu_pytorch_tanh",
            "attn_logit_softcapping":50.0,"final_logit_softcapping":30.0,
            "query_pre_attn_scalar":256,"sliding_window":4096}"#;
        let c = ModelConfig::from_json_bytes(gemma2, QuantScheme::Fp16)
            .expect("real Gemma2 config must parse");
        assert_eq!(c.activation, Activation::GeluTanh);
        assert_eq!(c.attn_logit_softcap, Some(50.0));
        assert_eq!(c.final_logit_softcap, Some(30.0));

        // `hidden_act` alone (Gemma v1 / older exports) still resolves.
        let only_act = br#"{"model_type":"llama","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64,
            "hidden_act":"gelu_pytorch_tanh"}"#;
        let c = ModelConfig::from_json_bytes(only_act, QuantScheme::Fp16).unwrap();
        assert_eq!(c.activation, Activation::GeluTanh);
    }

    #[test]
    fn gemma_sets_norm_embed_and_activation() {
        use crate::forward::cpu::Activation;
        let gemma = br#"{"model_type":"gemma","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64,
            "hidden_activation":"gelu_pytorch_tanh"}"#;
        let c = ModelConfig::from_json_bytes(gemma, QuantScheme::Fp16).unwrap();
        assert!(c.norm_add_one, "Gemma uses (1+w) RMSNorm");
        assert_eq!(c.embed_scale, Some(4.0), "sqrt(hidden=16) = 4");
        assert_eq!(c.activation, Activation::GeluTanh);

        // Llama-style: no add-one, no embed scale, SiLU.
        let llama = br#"{"model_type":"llama","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64}"#;
        let c = ModelConfig::from_json_bytes(llama, QuantScheme::Fp16).unwrap();
        assert!(!c.norm_add_one);
        assert_eq!(c.embed_scale, None);
        assert_eq!(c.activation, Activation::Silu);

        // Gemma2 inherits the Gemma norm/embed/activation rules.
        let gemma2 = br#"{"model_type":"gemma2","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64}"#;
        let c = ModelConfig::from_json_bytes(gemma2, QuantScheme::Fp16).unwrap();
        assert!(c.norm_add_one, "Gemma2 also uses (1+w) RMSNorm");
        assert_eq!(c.embed_scale, Some(4.0));
        assert_eq!(c.activation, Activation::GeluTanh);
    }

    /// Gemma2's distinguishing hyperparameters: alternating window layers (HF
    /// hard-codes the period at 2 rather than shipping a config key), attention
    /// and final logit softcaps, a decoupled attention scale, and the extra norm
    /// pair. None of these may leak onto Gemma v1 or Llama.
    #[test]
    fn gemma2_sets_softcaps_window_pattern_and_norms() {
        let gemma2 = br#"{"model_type":"gemma2","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":4,"vocab_size":32,"intermediate_size":64,
            "sliding_window":4096,"attn_logit_softcapping":50.0,
            "final_logit_softcapping":30.0,"query_pre_attn_scalar":144.0}"#;
        let c = ModelConfig::from_json_bytes(gemma2, QuantScheme::Fp16).unwrap();
        assert_eq!(c.sliding_window, Some(4096));
        assert_eq!(
            c.sliding_window_pattern,
            Some(2),
            "HF hard-codes a period of 2"
        );
        assert_eq!(c.attn_logit_softcap, Some(50.0));
        assert_eq!(c.final_logit_softcap, Some(30.0));
        assert_eq!(c.query_pre_attn_scalar, Some(144.0));
        assert!(c.gemma2_norms, "Gemma2 carries the pre/post-FFN norm pair");

        // Gemma v1 gets none of it — same family, different block.
        let gemma = br#"{"model_type":"gemma","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64}"#;
        let c = ModelConfig::from_json_bytes(gemma, QuantScheme::Fp16).unwrap();
        assert_eq!(c.sliding_window_pattern, None);
        assert_eq!(c.attn_logit_softcap, None);
        assert!(!c.gemma2_norms);
    }

    #[test]
    fn parses_yarn_rope_scaling() {
        use crate::forward::cpu::RopeScaling;
        let json = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,
            "vocab_size":32,"intermediate_size":64,
            "rope_scaling":{"rope_type":"yarn","factor":4.0,
                "original_max_position_embeddings":4096,"beta_fast":32,"beta_slow":1}}"#;
        let c = ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap();
        match c.rope_scaling {
            Some(RopeScaling::Yarn {
                factor,
                original_max_position,
                mscale,
                ..
            }) => {
                assert_eq!(factor, 4.0);
                assert_eq!(original_max_position, 4096.0);
                // Default attention factor = 0.1·ln(4)+1.
                assert!((mscale - (0.1 * 4.0f32.ln() + 1.0)).abs() < 1e-6);
            }
            other => panic!("expected YaRN, got {other:?}"),
        }
    }

    #[test]
    fn parses_sliding_window() {
        let with = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,
            "vocab_size":32,"intermediate_size":64,"sliding_window":4096}"#;
        assert_eq!(
            ModelConfig::from_json_bytes(with, QuantScheme::Fp16)
                .unwrap()
                .sliding_window,
            Some(4096)
        );
        // Absent → full attention; a zero window is treated as absent.
        let without = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,
            "vocab_size":32,"intermediate_size":64}"#;
        assert_eq!(
            ModelConfig::from_json_bytes(without, QuantScheme::Fp16)
                .unwrap()
                .sliding_window,
            None
        );
    }

    #[test]
    fn dense_resident_core_is_the_whole_layer_ffn() {
        let json = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,
            "vocab_size":32,"intermediate_size":64}"#;
        let c = ModelConfig::from_json_bytes(json, QuantScheme::Fp16).unwrap();
        assert_eq!(c.expert_params(), None);
        let h = 16u64;
        // Dense: attn + full SwiGLU (3*h*inter) + 2 norms.
        assert_eq!(c.resident_layer_params(), 4 * h * h + 3 * h * 64 + 2 * h);
    }
}

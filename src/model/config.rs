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
    #[serde(default, alias = "hidden_act")]
    hidden_activation: Option<String>,
    hidden_size: u32,
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    num_hidden_layers: u32,
    vocab_size: u32,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
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
    /// Sliding-window attention span (Mistral). A query attends only the last
    /// `sliding_window` positions; absent/`null` means full causal attention.
    #[serde(default)]
    sliding_window: Option<u32>,
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

/// A packed-quantized checkpoint dlm can decode, as declared by `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedQuant {
    /// Weights per quantization group along the input dimension.
    pub group_size: usize,
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
    match method.as_str() {
        "gptq" => {
            // act-order permutes rows by `g_idx`; decoding without un-permuting
            // silently scrambles every weight.
            if q.desc_act == Some(true) {
                return Err(DlmError::UnsupportedQuant(
                    "GPTQ checkpoint uses desc_act (act-order): its rows are permuted by                      g_idx and dlm does not un-permute them. Use a desc_act=false GPTQ                      export, or an fp16/bf16 checkpoint."
                        .into(),
                ));
            }
            // `gptq_v2` stores the true zero-point; classic `gptq` stores zero-1.
            // dlm's decoder assumes the classic convention (verified against a real
            // export) and has no v2 fixture to check the other against.
            match q.checkpoint_format.as_deref().map(|f| f.to_ascii_lowercase()) {
                None => {}
                Some(ref f) if f == "gptq" => {}
                Some(other) => {
                    return Err(DlmError::UnsupportedQuant(format!(
                        "GPTQ checkpoint_format {other:?} is not supported; dlm decodes the                          classic `gptq` format, whose zero-point convention it has been                          validated against."
                    )))
                }
            }
            let group_size = q.group_size.unwrap_or(-1);
            if group_size <= 0 {
                return Err(DlmError::UnsupportedQuant(format!(
                    "GPTQ checkpoint declares group_size {group_size}; dlm needs a positive                      per-group size (act-order/whole-row grouping is not supported)."
                )));
            }
            Ok(Some(PackedQuant { group_size: group_size as usize }))
        }
        "awq" => Err(DlmError::UnsupportedQuant(
            "AWQ packs its nibbles in a permuted order dlm does not unpack, and no real AWQ              fixture has been validated against. Use a 4-bit GPTQ (desc_act=false) or              fp16/bf16 checkpoint."
                .into(),
        )),
        other => Err(DlmError::UnsupportedQuant(format!(
            "unrecognized quant_method {other:?}; dlm loads fp16/bf16 and 4-bit GPTQ              (desc_act=false) checkpoints."
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
        other => Err(DlmError::InvalidConfig(format!(
            "rope_scaling type {other:?} is not implemented; dlm supports \"linear\" and \
             \"llama3\". Running this model without its trained RoPE scaling would produce \
             incoherent output, so it is refused rather than silently mis-run."
        ))),
    }
}

/// Derive validated [`MoeConfig`] from the raw config, or `None` for a dense
/// model. Mixtral declares experts under `num_local_experts`, Qwen under
/// `num_experts`; the presence of either marks the checkpoint MoE and also picks
/// the tensor naming family. A model that declares experts but omits the top-k
/// count is refused rather than guessed at — routing every token through the
/// wrong number of experts is silent garbage, the worst failure mode.
fn build_moe_config(raw: &RawConfig) -> Result<Option<MoeConfig>> {
    let (num_experts, naming) = match (raw.num_local_experts, raw.num_experts) {
        (Some(n), _) => (n, MoeNaming::Mixtral),
        (None, Some(n)) => (n, MoeNaming::Qwen),
        (None, None) => return Ok(None),
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
    Ok(Some(MoeConfig {
        num_experts,
        experts_per_tok,
        moe_intermediate_size,
        shared_intermediate_size: raw.shared_expert_intermediate_size,
        // Mixtral always renormalizes; Qwen exposes the flag (default on).
        norm_topk_prob: raw.norm_topk_prob.unwrap_or(true),
        naming,
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
    /// Sliding-window attention span (Mistral); `None` is full causal attention.
    pub sliding_window: Option<u32>,
    /// Gemma applies RMSNorm as `(1 + weight)` rather than `weight`. When true the
    /// loader bakes the `+1` into the norm weights so the kernels stay unchanged.
    pub norm_add_one: bool,
    /// Scalar applied to token embeddings after lookup (Gemma multiplies by
    /// `sqrt(hidden_size)`); `None` leaves embeddings unscaled.
    pub embed_scale: Option<f32>,
    /// Gated-MLP activation (SiLU for most, GELU for Gemma).
    pub activation: crate::forward::cpu::Activation,
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
        let raw: RawConfig = serde_json::from_slice(bytes).map_err(|source| DlmError::Json {
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

        // Gemma architecture variants: (1+w) RMSNorm, embedding scaling, GeGLU.
        let model_type = raw.model_type.as_deref().unwrap_or("").to_ascii_lowercase();
        if model_type == "gemma2" {
            return Err(DlmError::InvalidConfig(
                "Gemma2 is not yet supported: it needs attention logit softcapping, \
                 alternating sliding-window layers, and pre/post-FFN norms. Gemma (v1) works \
                 and other norm variants are refused rather than silently mis-run."
                    .into(),
            ));
        }
        let is_gemma = model_type == "gemma";
        let activation = match raw.hidden_activation.as_deref() {
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
            num_kv_heads: raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
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
            // A window >= the model's own max context is the same as full
            // attention; keep it as declared and let the kernel no-op it.
            sliding_window: raw.sliding_window.filter(|&w| w > 0),
            norm_add_one: is_gemma,
            embed_scale: is_gemma.then(|| (raw.hidden_size as f32).sqrt()),
            activation,
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
            return Err(DlmError::InvalidConfig("num_hidden_layers must be > 0".into()));
        }
        if self.hidden_size == 0 {
            return Err(DlmError::InvalidConfig("hidden_size must be > 0".into()));
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

    #[test]
    fn gemma_sets_norm_embed_activation_and_refuses_gemma2() {
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

        // Gemma2 is refused (needs softcap + alternating windows + extra norms).
        let gemma2 = br#"{"model_type":"gemma2","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":2,"vocab_size":32,"intermediate_size":64}"#;
        assert!(ModelConfig::from_json_bytes(gemma2, QuantScheme::Fp16).is_err());
    }

    #[test]
    fn parses_sliding_window() {
        let with = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,
            "vocab_size":32,"intermediate_size":64,"sliding_window":4096}"#;
        assert_eq!(
            ModelConfig::from_json_bytes(with, QuantScheme::Fp16).unwrap().sliding_window,
            Some(4096)
        );
        // Absent → full attention; a zero window is treated as absent.
        let without = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":2,
            "vocab_size":32,"intermediate_size":64}"#;
        assert_eq!(
            ModelConfig::from_json_bytes(without, QuantScheme::Fp16).unwrap().sliding_window,
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

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
    // ── DeepSeek routing variants. dlm's router is softmax + plain top-k, which
    // is what DeepSeek-V2 declares; V3/R1 change all four keys at once and are
    // refused rather than routed through the wrong experts. ──
    /// How gate logits become scores: `softmax` (V2) or `sigmoid` (V3).
    #[serde(default)]
    scoring_func: Option<String>,
    /// How the top-k is taken: `greedy` (V2), or the group-limited methods
    /// `group_limited_greedy` / `noaux_tc` (V3).
    #[serde(default)]
    topk_method: Option<String>,
    /// Experts are split into `n_group` groups and only `topk_group` of them are
    /// eligible per token.
    #[serde(default)]
    n_group: Option<u32>,
    #[serde(default)]
    topk_group: Option<u32>,
    /// Scales the routed experts' summed output (V3 uses 2.5).
    #[serde(default)]
    routed_scaling_factor: Option<f32>,
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
    /// Qwen2's switch for the window above. Every Qwen2/2.5 config ships a
    /// `sliding_window` *and* sets this to `false`, so honoring the span on its
    /// own would truncate attention the model expects to be full.
    #[serde(default)]
    use_sliding_window: Option<bool>,
    /// Qwen2: layers from this index on use the window, earlier ones are full.
    /// Refused when it would take effect, since dlm's pattern is "every n-th
    /// layer global", which cannot express a prefix of full layers.
    #[serde(default)]
    max_window_layers: Option<u32>,
    /// Fraction of each head's dimensions that carry RoPE (Phi, GPT-NeoX, GLM,
    /// typically 0.4 or 0.5). dlm rotates the whole head, so a smaller factor is
    /// refused: rotating dimensions the model leaves untouched mis-places every
    /// token without erroring.
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    /// The LM head is the embedding matrix, not a tensor of its own. dlm already
    /// falls back to the embedding when `lm_head.weight` is absent; this makes
    /// the tie explicit, for exports that ship both.
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
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
    // ── Gemma 4 (`gemma4_text`); absent on every other family. ──
    /// Per-layer-type RoPE: `{"full_attention": {...}, "sliding_attention": {...}}`,
    /// each with its `rope_type`, `rope_theta` and (proportional) `partial_rotary_factor`.
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    /// Head width of the full-attention layers.
    #[serde(default)]
    global_head_dim: Option<u32>,
    /// The last this-many layers read the KV of earlier layers.
    #[serde(default)]
    num_kv_shared_layers: Option<u32>,
    /// Width of each layer's per-layer input (0: none).
    #[serde(default)]
    hidden_size_per_layer_input: Option<u32>,
    /// The KV-shared layers' MLP is twice as wide (E2B).
    #[serde(default)]
    use_double_wide_mlp: Option<bool>,
    /// Variants dlm refuses: keys reused as values on global layers, a MoE
    /// block beside the MLP, bidirectional attention.
    #[serde(default)]
    attention_k_eq_v: Option<bool>,
    #[serde(default)]
    enable_moe_block: Option<bool>,
    #[serde(default)]
    /// Gemma 4 spells it `null` or `"all"`; Gemma 3 270M ships `false`.
    use_bidirectional_attention: Option<serde_json::Value>,
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
                "{bits}-bit {method} checkpoint; dlm decodes 4-bit only. Use an fp16/bf16 \
                 or 4-bit GPTQ (desc_act=false) checkpoint."
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
/// The router's rule from DeepSeek's four routing keys: `scoring_func`
/// (softmax or V3's sigmoid), `topk_method` (`greedy`, V2's
/// `group_limited_greedy`, V3's `noaux_tc`, which also reads a per-expert
/// `e_score_correction_bias`), the `n_group`/`topk_group` split, and
/// `routed_scaling_factor`. Mixtral and Qwen set none and get plain softmax
/// top-k. A value dlm does not implement is refused by name: routing to the
/// wrong experts reads as fluent, wrong output rather than failing.
fn routing_from(raw: &RawConfig, num_experts: u32) -> Result<Routing> {
    let refuse = |what: String| -> Result<Routing> {
        Err(DlmError::InvalidConfig(format!(
            "{what} dlm routes with softmax or sigmoid scores and the greedy, \
             group_limited_greedy or noaux_tc top-k (DeepSeek-V2/V3); running this \
             checkpoint under another rule would route tokens through the wrong experts."
        )))
    };
    let sigmoid = match raw.scoring_func.as_deref().map(str::to_ascii_lowercase) {
        None => false,
        Some(ref f) if f == "softmax" => false,
        Some(ref f) if f == "sigmoid" => true,
        Some(other) => return refuse(format!("config declares scoring_func {other:?};")),
    };
    let method = raw.topk_method.as_deref().map(str::to_ascii_lowercase);
    let (grouped, by_top2, score_bias) = match method.as_deref() {
        None | Some("greedy") => (false, false, false),
        Some("group_limited_greedy") => (true, false, false),
        Some("noaux_tc") => (true, true, true),
        Some(other) => return refuse(format!("config declares topk_method {other:?};")),
    };
    let n_group = raw.n_group.unwrap_or(1).max(1);
    let topk_group = raw.topk_group.unwrap_or(n_group);
    if grouped && n_group > 1 {
        let per = num_experts / n_group;
        if num_experts % n_group != 0 || topk_group == 0 || topk_group > n_group {
            return refuse(format!(
                "config groups {num_experts} experts into {n_group} (n_group), keeping \
                 {topk_group} (topk_group), which does not divide them;"
            ));
        }
        if by_top2 && per < 2 {
            return refuse(format!(
                "config's groups hold {per} expert, too few to score by their top two;"
            ));
        }
    }
    Ok(Routing {
        sigmoid,
        n_group: if grouped { n_group } else { 1 },
        topk_group: if grouped { topk_group } else { 1 },
        group_by_top2: by_top2,
        score_bias,
        scale: raw.routed_scaling_factor.unwrap_or(1.0),
    })
}

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
    let routing = routing_from(raw, num_experts)?;
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
        routing,
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
    /// How the router picks experts and weighs them.
    pub routing: Routing,
}

/// How a MoE router turns a token's gate logits into experts and weights. The
/// default is the softmax top-k Mixtral, Qwen and DeepSeek-V2 use; DeepSeek-V2's
/// full model limits the choice to the best groups of experts, and V3 scores
/// with a sigmoid, adds a learned bias for the choice only, and scales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Routing {
    /// Sigmoid scores (V3) rather than softmax.
    pub sigmoid: bool,
    /// Experts split into this many equal groups (1: no grouping), of which the
    /// `topk_group` best are eligible.
    pub n_group: u32,
    pub topk_group: u32,
    /// A group scores the sum of its two best (V3's `noaux_tc`); else its best
    /// (V2's `group_limited_greedy`).
    pub group_by_top2: bool,
    /// The layer's `e_score_correction_bias` is added to the scores for the
    /// choice, not for the weights (V3's `noaux_tc`).
    pub score_bias: bool,
    /// Multiplies the routed experts' weights (`routed_scaling_factor`).
    pub scale: f32,
}

impl Default for Routing {
    fn default() -> Self {
        Routing {
            sigmoid: false,
            n_group: 1,
            topk_group: 1,
            group_by_top2: false,
            score_bias: false,
            scale: 1.0,
        }
    }
}

impl Routing {
    /// The plain softmax top-k: no sigmoid, groups, bias or scale.
    pub fn is_plain(&self) -> bool {
        *self == Routing::default()
    }
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
    /// Sliding-window attention span (Mistral); `None` is full causal attention,
    /// which is also what a declared window switched off by Qwen2's
    /// `use_sliding_window: false` comes to.
    pub sliding_window: Option<u32>,
    /// The LM head is the embedding matrix. The loader then reads the embedding
    /// even when the checkpoint also ships an `lm_head.weight`, which is what
    /// transformers does with a tied config.
    pub tie_word_embeddings: bool,
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
    /// Gemma 4: the full-attention layers' head width (`global_head_dim`), when
    /// it differs from `head_dim`, which is then the windowed layers'.
    pub global_head_dim: Option<u32>,
    /// Gemma 4: the first layer that reads an earlier layer's KV cache instead
    /// of keeping its own (`num_hidden_layers - num_kv_shared_layers`).
    pub kv_shared_from: Option<u32>,
    /// Gemma 4 RMS-normalizes each value head (no learned scale).
    pub v_norm: bool,
    /// Gemma 4: each layer's per-layer input width
    /// (`hidden_size_per_layer_input`); 0 for every other family.
    pub ple_dim: u32,
    /// Gemma 4 E2B: the KV-shared layers' MLP is twice `intermediate_size`.
    pub double_wide_shared_mlp: bool,
}

/// The text model of a multimodal Gemma 4 config (`model_type: "gemma4"`), or
/// `json` unchanged for anything else: its `text_config`, with the top-level
/// keys the text model needs carried over and `Gemma4TextConfig`'s defaults
/// for what it omits. The vision and audio towers beside it are not run.
fn gemma4_text_config(json: serde_json::Value) -> serde_json::Value {
    use serde_json::{json, Value};
    let is_multimodal = json.get("model_type").and_then(Value::as_str) == Some("gemma4");
    let Some(Value::Object(mut text)) = json.get("text_config").cloned().filter(|_| is_multimodal)
    else {
        return json;
    };
    for key in [
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "quantization_config",
        "tie_word_embeddings",
    ] {
        if let (false, Some(v)) = (text.contains_key(key), json.get(key)) {
            text.insert(key.to_string(), v.clone());
        }
    }
    let defaults = json!({
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 2304,
        "intermediate_size": 9216,
        "num_hidden_layers": 30,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "sliding_window": 512,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "num_kv_shared_layers": 0,
        "tie_word_embeddings": true,
    });
    for (key, value) in defaults.as_object().expect("literal object") {
        // A JSON `null` is an omitted key to transformers too.
        if text.get(key).is_none_or(Value::is_null) {
            text.insert(key.clone(), value.clone());
        }
    }
    Value::Object(text)
}

/// Gemma 4's per-layer settings, read from a `gemma4_text` config: the layer
/// pattern, the two RoPE bases (and the global layers' partial rotation), the
/// shared-KV tail and the per-layer inputs. Variants dlm does not implement are
/// refused by name rather than run as the dense E2B/E4B shape.
struct Gemma4 {
    pattern: Option<u32>,
    rope_theta: f32,
    rope_scaling: Option<RopeScaling>,
    rope_local_theta: f32,
    global_head_dim: Option<u32>,
    kv_shared_from: Option<u32>,
    ple_dim: u32,
    double_wide: bool,
}

fn gemma4_settings(raw: &RawConfig) -> Result<Gemma4> {
    let refuse = |what: &str| {
        Err(DlmError::InvalidConfig(format!(
            "this Gemma 4 checkpoint {what}, which dlm does not implement; it runs the \
             dense E2B/E4B shape. Running it as that would load the wrong weights or \
             attend the wrong way."
        )))
    };
    if raw.attention_k_eq_v == Some(true) {
        return refuse("reuses its keys as values (attention_k_eq_v)");
    }
    if raw.enable_moe_block == Some(true) {
        return refuse("has a Mixture-of-Experts block (enable_moe_block)");
    }
    if matches!(
        raw.use_bidirectional_attention,
        Some(serde_json::Value::Bool(true))
    ) || raw
        .use_bidirectional_attention
        .as_ref()
        .and_then(serde_json::Value::as_str)
        == Some("all")
    {
        return refuse("attends bidirectionally (use_bidirectional_attention: \"all\")");
    }
    let layers = raw.num_hidden_layers;
    // transformers' default layout: every 6th layer global, and the last always.
    let types = raw.layer_types.clone().unwrap_or_else(|| {
        (0..layers)
            .map(|i| {
                if (i + 1) % 6 == 0 || i + 1 == layers {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect()
    });
    if types.len() != layers as usize {
        return Err(DlmError::InvalidConfig(format!(
            "layer_types lists {} layers; the model has {layers}",
            types.len()
        )));
    }
    let pattern = pattern_from_layer_types(&types)?;

    // RoPE per layer kind. Absent, these are Gemma4TextConfig's defaults.
    let rope = |kind: &str, theta: f64, rope_type: &str, partial: f64| -> Result<(f32, f32)> {
        let p = raw.rope_parameters.as_ref().and_then(|r| r.get(kind));
        let get = |k: &str| p.and_then(|p| p.get(k));
        let ty = get("rope_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(rope_type);
        let factor = get("factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0);
        if !matches!(ty, "default" | "proportional") || factor != 1.0 {
            return Err(DlmError::InvalidConfig(format!(
                "Gemma 4 {kind} RoPE {ty:?} with factor {factor} is not implemented; dlm runs \
                 the \"default\" and unscaled \"proportional\" RoPE Gemma 4 ships with."
            )));
        }
        let theta = get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(theta);
        let partial = if ty == "proportional" {
            get("partial_rotary_factor")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(partial)
        } else {
            1.0
        };
        Ok((theta as f32, partial as f32))
    };
    let (rope_theta, partial) = rope("full_attention", 1_000_000.0, "proportional", 0.25)?;
    let (rope_local_theta, local_partial) = rope("sliding_attention", 10_000.0, "default", 1.0)?;
    if local_partial < 1.0 {
        return refuse("rotates only part of its windowed layers' heads");
    }

    let kv_shared_from = match raw.num_kv_shared_layers.unwrap_or(0) {
        0 => None,
        n if n >= layers => {
            return Err(DlmError::InvalidConfig(format!(
                "num_kv_shared_layers {n} leaves none of the {layers} layers with KV of its own"
            )))
        }
        n => Some(layers - n),
    };
    // Every shared layer needs an earlier layer of its kind to read.
    if let Some(first) = kv_shared_from {
        let global = |i: usize| types[i] == "full_attention";
        for l in first as usize..types.len() {
            if !(0..first as usize).any(|i| global(i) == global(l)) {
                return Err(DlmError::InvalidConfig(format!(
                    "layer {l} shares KV, but no layer before {first} is of its kind"
                )));
            }
        }
    }
    let head_dim = raw
        .head_dim
        .unwrap_or(raw.hidden_size / raw.num_attention_heads.max(1));
    Ok(Gemma4 {
        pattern,
        rope_theta,
        rope_scaling: (partial < 1.0).then_some(RopeScaling::Partial { fraction: partial }),
        rope_local_theta,
        global_head_dim: raw.global_head_dim.filter(|&g| g != head_dim),
        kv_shared_from,
        ple_dim: raw.hidden_size_per_layer_input.unwrap_or(256),
        double_wide: raw.use_double_wide_mlp == Some(true) && kv_shared_from.is_some(),
    })
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

/// The per-head width a part-built GGUF config implies: its explicit `head_dim`,
/// else hidden size over head count.
fn self_head_dim(cfg: &serde_json::Value) -> Option<u64> {
    let get = |k: &str| cfg.get(k).and_then(serde_json::Value::as_u64);
    get("head_dim").or_else(|| Some(get("hidden_size")? / get("num_attention_heads")?.max(1)))
}

impl ModelConfig {
    /// True when this is a Mixture-of-Experts checkpoint.
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }
}

impl ModelConfig {
    /// The model config a GGUF file carries in its metadata.
    ///
    /// GGUF keeps what `config.json` would hold under `{architecture}.*` keys, in
    /// llama.cpp's own spelling. They are translated into that JSON shape and
    /// parsed by [`from_json_bytes`](Self::from_json_bytes), so a GGUF checkpoint
    /// meets exactly the same validation — and the same refusals — as a
    /// safetensors one, rather than a second, laxer path.
    ///
    /// `has_lm_head` says whether the file ships an output projection; GGUF omits
    /// it when it is the embedding matrix, which is what `tie_word_embeddings`
    /// means in a `config.json`.
    pub fn from_gguf(
        metadata: &std::collections::BTreeMap<String, crate::storage::gguf::Value>,
        has_lm_head: bool,
        quant: QuantScheme,
    ) -> Result<Self> {
        use crate::storage::gguf::Value;
        use serde_json::json;

        let arch = metadata
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                DlmError::InvalidConfig("gguf: no general.architecture in the metadata".into())
            })?;
        // The architectures whose tensor names and block shape dlm knows. An
        // unrecognized one is refused rather than run as a Llama: the names would
        // mostly match, and the parts that did not would load as the wrong thing.
        if arch == "gemma4" {
            return Self::from_gemma4_gguf(metadata, has_lm_head, quant);
        }
        let model_type = match arch {
            "llama" | "mistral" | "qwen2" | "qwen3" | "phi3" | "starcoder2" => arch,
            "gemma" => "gemma",
            "gemma2" => "gemma2",
            "gemma3" | "gemma3_text" => "gemma3_text",
            other => {
                return Err(DlmError::InvalidConfig(format!(
                    "gguf: architecture {other:?} is not one dlm implements. It reads llama,                      mistral, qwen2, qwen3, phi3, starcoder2 and gemma/gemma2/gemma3 files."
                )))
            }
        };
        let num = |key: &str| {
            metadata
                .get(&format!("{arch}.{key}"))
                .and_then(Value::as_f64)
        };
        let int = |key: &str| {
            metadata
                .get(&format!("{arch}.{key}"))
                .and_then(Value::as_u64)
        };
        let need = |key: &str| {
            int(key).ok_or_else(|| {
                DlmError::InvalidConfig(format!("gguf: metadata has no {arch}.{key}"))
            })
        };

        // The vocabulary is the tokenizer's token list; GGUF has no vocab_size key.
        let vocab = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_strings)
            .map(|t| t.len() as u64)
            .ok_or_else(|| {
                DlmError::InvalidConfig("gguf: metadata has no tokenizer.ggml.tokens".into())
            })?;

        let mut cfg = json!({
            "model_type": model_type,
            "hidden_size": need("embedding_length")?,
            "num_hidden_layers": need("block_count")?,
            "num_attention_heads": need("attention.head_count")?,
            "vocab_size": vocab,
            "tie_word_embeddings": !has_lm_head,
        });
        let obj = cfg.as_object_mut().expect("literal object");
        let mut put = |key: &str, value: Option<serde_json::Value>| {
            if let Some(v) = value {
                obj.insert(key.to_string(), v);
            }
        };
        put(
            "num_key_value_heads",
            int("attention.head_count_kv").map(Into::into),
        );
        put(
            "intermediate_size",
            int("feed_forward_length").map(Into::into),
        );
        put(
            "max_position_embeddings",
            int("context_length").map(Into::into),
        );
        put("head_dim", int("attention.key_length").map(Into::into));
        put("rope_theta", num("rope.freq_base").map(Into::into));
        put(
            "rms_norm_eps",
            num("attention.layer_norm_rms_epsilon").map(Into::into),
        );
        put(
            "layer_norm_epsilon",
            num("attention.layer_norm_epsilon").map(Into::into),
        );
        put(
            "sliding_window",
            int("attention.sliding_window").map(Into::into),
        );
        put(
            "attn_logit_softcapping",
            num("attn_logit_softcapping").map(Into::into),
        );
        put(
            "final_logit_softcapping",
            num("final_logit_softcapping").map(Into::into),
        );
        put(
            "query_pre_attn_scalar",
            num("attention.query_pre_attn_scalar").map(Into::into),
        );
        put(
            "rope_local_base_freq",
            num("rope.freq_base_swa").map(Into::into),
        );
        put(
            "eos_token_id",
            metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(Value::as_u64)
                .map(Into::into),
        );
        // llama.cpp spells RoPE scaling as a type plus a factor, the same two
        // fields `rope_scaling` holds.
        if let (Some(kind), Some(factor)) = (
            metadata
                .get(&format!("{arch}.rope.scaling.type"))
                .and_then(Value::as_str),
            num("rope.scaling.factor"),
        ) {
            put(
                "rope_scaling",
                Some(json!({ "rope_type": kind, "factor": factor })),
            );
        }
        // RoPE over fewer dimensions than the head is wide is the partial rotary
        // factor under another name, which `from_json_bytes` refuses.
        if let (Some(rope_dims), Some(head_dim)) =
            (int("rope.dimension_count"), self_head_dim(&cfg))
        {
            if rope_dims < head_dim {
                cfg.as_object_mut().unwrap().insert(
                    "partial_rotary_factor".into(),
                    json!(rope_dims as f64 / head_dim as f64),
                );
            }
        }

        let bytes = serde_json::to_vec(&cfg).expect("config json is serializable");
        let mut config = Self::from_json_bytes(&bytes, quant)?;
        // Gemma's RMSNorm scales by `1 + w`. llama.cpp's converter bakes the 1
        // into every Gemma, Gemma 2 and Gemma 3 norm weight it writes, so adding
        // it again at load would scale every norm by `2 + w`.
        if matches!(arch, "gemma" | "gemma2" | "gemma3" | "gemma3_text") {
            config.norm_add_one = false;
        }
        Ok(config)
    }

    /// A Gemma 4 GGUF's metadata as a `gemma4_text` config. llama.cpp spells
    /// the per-kind settings with a `_swa` suffix for the windowed layers, marks
    /// those layers with a boolean array, and keeps the global layers' partial
    /// RoPE in a `rope_freqs` tensor rather than the metadata: the 0.25 here is
    /// Gemma 4's, which the loader checks that tensor against.
    fn from_gemma4_gguf(
        metadata: &std::collections::BTreeMap<String, crate::storage::gguf::Value>,
        has_lm_head: bool,
        quant: QuantScheme,
    ) -> Result<Self> {
        use crate::storage::gguf::Value;
        use serde_json::json;
        let get = |key: &str| metadata.get(&format!("gemma4.{key}"));
        let num = |key: &str| get(key).and_then(Value::as_f64);
        let need = |key: &str| {
            get(key).and_then(Value::as_u64).ok_or_else(|| {
                DlmError::InvalidConfig(format!("gguf: metadata has no gemma4.{key}"))
            })
        };
        let layers = need("block_count")?;
        // A per-layer list where llama.cpp writes one (the FFN width, the KV
        // heads), else the single value.
        let per_layer = |key: &str| -> Result<Vec<u64>> {
            match get(key) {
                Some(Value::Numbers(v)) if v.len() == layers as usize => {
                    Ok(v.iter().map(|&x| x as u64).collect())
                }
                Some(v) => v.as_u64().map(|x| vec![x; layers as usize]).ok_or_else(|| {
                    DlmError::InvalidConfig(format!("gguf: gemma4.{key} is not a number per layer"))
                }),
                None => Err(DlmError::InvalidConfig(format!(
                    "gguf: metadata has no gemma4.{key}"
                ))),
            }
        };
        let vocab = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_strings)
            .map(|t| t.len() as u64)
            .ok_or_else(|| {
                DlmError::InvalidConfig("gguf: metadata has no tokenizer.ggml.tokens".into())
            })?;
        let shared = get("attention.shared_kv_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // E2B lists a wider FFN for its shared-KV layers: Gemma 4's
        // `use_double_wide_mlp`. Any other variation is not a shape dlm knows.
        let ffn = per_layer("feed_forward_length")?;
        let first_shared = (layers - shared.min(layers)) as usize;
        let double_wide = ffn[first_shared..].iter().all(|&f| f == 2 * ffn[0]) && shared > 0;
        if !ffn[..first_shared].iter().all(|&f| f == ffn[0])
            || !(double_wide || ffn.iter().all(|&f| f == ffn[0]))
        {
            return Err(DlmError::InvalidConfig(format!(
                "gguf: Gemma 4 FFN widths {ffn:?} are neither uniform nor doubled on the shared-KV layers"
            )));
        }
        let kv_heads = per_layer("attention.head_count_kv")?;
        if kv_heads.iter().any(|&k| k != kv_heads[0]) {
            return Err(DlmError::InvalidConfig(format!(
                "gguf: Gemma 4 KV head counts {kv_heads:?} differ by layer, which dlm does not run"
            )));
        }
        let layer_types: Vec<&str> = match get("attention.sliding_window_pattern") {
            Some(Value::Numbers(v)) if v.len() == layers as usize => v
                .iter()
                .map(|&w| {
                    if w != 0.0 {
                        "sliding_attention"
                    } else {
                        "full_attention"
                    }
                })
                .collect(),
            _ => {
                return Err(DlmError::InvalidConfig(
                    "gguf: Gemma 4 metadata has no per-layer sliding_window_pattern".into(),
                ))
            }
        };
        let mut cfg = json!({
            "model_type": "gemma4_text",
            "hidden_size": need("embedding_length")?,
            "num_hidden_layers": layers,
            "num_attention_heads": need("attention.head_count")?,
            "num_key_value_heads": kv_heads[0],
            "intermediate_size": ffn[0],
            "use_double_wide_mlp": double_wide,
            "vocab_size": vocab,
            "tie_word_embeddings": !has_lm_head,
            "head_dim": need("attention.key_length_swa")?,
            "global_head_dim": need("attention.key_length")?,
            "layer_types": layer_types,
            "num_kv_shared_layers": shared,
            "hidden_size_per_layer_input": need("embedding_length_per_layer_input")?,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": num("rope.freq_base").unwrap_or(1_000_000.0),
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": num("rope.freq_base_swa").unwrap_or(10_000.0),
                },
            },
        });
        let obj = cfg.as_object_mut().expect("literal object");
        let int = |key: &str| get(key).and_then(Value::as_u64);
        let optional: [(&str, Option<serde_json::Value>); 5] = [
            (
                "max_position_embeddings",
                int("context_length").map(Into::into),
            ),
            (
                "sliding_window",
                int("attention.sliding_window").map(Into::into),
            ),
            (
                "rms_norm_eps",
                num("attention.layer_norm_rms_epsilon").map(Into::into),
            ),
            (
                "final_logit_softcapping",
                num("final_logit_softcapping").map(Into::into),
            ),
            (
                "eos_token_id",
                metadata
                    .get("tokenizer.ggml.eos_token_id")
                    .and_then(Value::as_u64)
                    .map(Into::into),
            ),
        ];
        for (key, value) in optional {
            if let Some(v) = value {
                obj.insert(key.to_string(), v);
            }
        }
        let bytes = serde_json::to_vec(&cfg).expect("config json is serializable");
        Self::from_json_bytes(&bytes, quant)
    }

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
        let raw: RawConfig = serde_json::from_value(gemma3_text_config(gemma4_text_config(json)))
            .map_err(|source| DlmError::Json {
            context: "config.json".to_string(),
            source,
        })?;
        let model_type = raw.model_type.as_deref().unwrap_or("").to_ascii_lowercase();
        // Gemma 4 first: it is refused or resolved before the generic MoE and
        // RoPE parsing reads keys it spells its own way (a MoE variant's
        // `num_experts` would otherwise be taken for Qwen's).
        let gemma4 = matches!(model_type.as_str(), "gemma4" | "gemma4_text")
            .then(|| gemma4_settings(&raw))
            .transpose()?;

        // Reject quant formats the decoder would silently mis-decode; keep the
        // layout of the one it can.
        let packed_quant = match &raw.quantization_config {
            Some(qc) => check_quant_supported(qc)?,
            None => None,
        };

        let moe = match gemma4 {
            Some(_) => None,
            None => build_moe_config(&raw)?,
        };
        let mla = build_mla_config(&raw)?;

        // Qwen2 declares a window and switches it off; Mistral just declares one.
        // A window kept on when the config turns it off truncates attention the
        // model was trained to have in full, silently, past `sliding_window`.
        let sliding_window = raw
            .sliding_window
            .filter(|&w| w > 0)
            .filter(|_| raw.use_sliding_window.unwrap_or(true));
        // Qwen2 windows only the layers from `max_window_layers` on. dlm's
        // pattern makes every n-th layer global instead, which cannot express a
        // prefix of full layers, so a config that means it is refused.
        if let (Some(_), Some(first)) = (sliding_window, raw.max_window_layers) {
            if first > 0 && first < raw.num_hidden_layers {
                return Err(DlmError::InvalidConfig(format!(
                    "config windows attention from layer {first} of {} on \
                     (max_window_layers), and dlm applies a window either to every layer or \
                     to every n-th (Gemma). Windowing the first {first} layers too would run \
                     quietly wrong past {} tokens.",
                    raw.num_hidden_layers,
                    sliding_window.unwrap_or(0),
                )));
            }
        }
        // RoPE on part of each head (Phi, GPT-NeoX, GLM): dlm rotates all of it.
        if let Some(factor) = raw.partial_rotary_factor.filter(|f| *f < 1.0) {
            return Err(DlmError::InvalidConfig(format!(
                "config sets partial_rotary_factor {factor}, so RoPE covers only that \
                 fraction of each head's dimensions. dlm rotates the whole head; doing that \
                 to a model trained this way mis-places every token without erroring."
            )));
        }

        // Gemma architecture variants: (1+w) RMSNorm, embedding scaling, GeGLU.
        // Gemma2 adds logit softcapping, alternating window layers, a decoupled
        // attention scale, and a second norm pair per layer.
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
            None if is_gemma || gemma4.is_some() => crate::forward::cpu::Activation::GeluTanh,
            _ => crate::forward::cpu::Activation::Silu,
        };

        let mut config = ModelConfig {
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
            sliding_window,
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(false),
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
            global_head_dim: None,
            kv_shared_from: None,
            v_norm: false,
            ple_dim: 0,
            double_wide_shared_mlp: false,
        };
        // Gemma 4: Gemma 3's block with plain-`w` RMSNorm, normed values, an
        // attention scale of 1 (the Q/K norms carry it), wider global heads
        // with partial RoPE, a shared-KV tail and per-layer inputs.
        if let Some(g) = gemma4 {
            config.norm_add_one = false;
            config.embed_scale = Some((raw.hidden_size as f32).sqrt());
            config.gemma2_norms = true;
            config.sliding_window_pattern = g.pattern;
            config.rope_theta = g.rope_theta;
            config.rope_scaling = g.rope_scaling;
            config.rope_local_theta = Some(g.rope_local_theta);
            config.query_pre_attn_scalar = Some(1.0);
            config.global_head_dim = g.global_head_dim;
            config.kv_shared_from = g.kv_shared_from;
            config.v_norm = true;
            config.ple_dim = g.ple_dim;
            config.double_wide_shared_mlp = g.double_wide;
        }

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

    /// True when layers differ in shape (Gemma 4: wider global heads, a
    /// KV-shared tail, per-layer inputs), so per-layer figures are sums over the
    /// layers rather than one layer times the count.
    fn layers_vary(&self) -> bool {
        self.global_head_dim.is_some() || self.kv_shared_from.is_some() || self.ple_dim > 0
    }

    /// Whether `layer` attends globally rather than over the window.
    fn is_global_layer(&self, layer: u32) -> bool {
        match (self.sliding_window, self.sliding_window_pattern) {
            (None, _) => true,
            (Some(_), Some(n)) => n > 0 && (layer + 1) % n == 0,
            (Some(_), None) => false,
        }
    }

    /// `layer`'s head width: Gemma 4's global layers are wider.
    fn layer_head_dim(&self, layer: u32) -> u64 {
        match self.global_head_dim {
            Some(g) if self.is_global_layer(layer) => g as u64,
            _ => self.head_dim() as u64,
        }
    }

    /// Whether `layer` keeps a KV cache of its own (Gemma 4's shared tail does not).
    fn has_own_kv(&self, layer: u32) -> bool {
        self.kv_shared_from.is_none_or(|first| layer < first)
    }

    /// K and V values one token stores across every layer: `2 × kv_heads ×
    /// head_dim` per layer with a cache of its own.
    pub fn kv_values_per_token(&self) -> u64 {
        (0..self.num_layers)
            .filter(|&l| self.has_own_kv(l))
            .map(|l| 2 * self.num_kv_heads as u64 * self.layer_head_dim(l))
            .sum()
    }

    /// Gemma 4's block parameters summed over its layers: attention as wide as
    /// each layer's heads (no K/V on the shared tail), the MLP doubled where
    /// E2B doubles it, each layer's per-layer-input gate and projection, and the
    /// model-level projection that feeds them.
    fn varying_block_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let ple = self.ple_dim as u64;
        let layers: u64 = (0..self.num_layers)
            .map(|l| {
                let d = self.layer_head_dim(l);
                let q_o = 2 * h * self.num_attention_heads as u64 * d;
                let kv = if self.has_own_kv(l) {
                    2 * h * self.num_kv_heads as u64 * d
                } else {
                    0
                };
                let wide = !self.has_own_kv(l) && self.double_wide_shared_mlp;
                let inter = self.intermediate_size as u64 * if wide { 2 } else { 1 };
                q_o + kv + 3 * h * inter + 2 * h * ple
            })
            .sum();
        layers + h * self.num_layers as u64 * ple
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
        if self.layers_vary() {
            return self.varying_block_params() / self.num_layers.max(1) as u64 + norms;
        }
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

        let blocks = if self.layers_vary() {
            self.varying_block_params()
        } else {
            per_layer * self.num_layers as u64
        };
        let embed_and_head = 2 * self.vocab_size as u64 * h;
        blocks + embed_and_head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF's metadata for a tiny model of `arch`: just the keys `from_gguf` needs.
    fn gguf_meta(arch: &str) -> std::collections::BTreeMap<String, crate::storage::gguf::Value> {
        use crate::storage::gguf::Value;
        let mut m = std::collections::BTreeMap::new();
        m.insert("general.architecture".into(), Value::String(arch.into()));
        for (k, v) in [
            ("embedding_length", 16),
            ("block_count", 2),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("feed_forward_length", 32),
            ("attention.key_length", 8),
        ] {
            m.insert(format!("{arch}.{k}"), Value::U64(v));
        }
        m.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Strings((0..64).map(|i| format!("t{i}")).collect()),
        );
        m
    }

    /// google/gemma-4-E4B-it's config.json, the text half verbatim but for
    /// `layer_types` (written out below) and its towers left out.
    fn gemma4_e4b_json(extra: &str) -> String {
        let types: Vec<&str> = (0..42)
            .map(|i| {
                if (i + 1) % 6 == 0 {
                    "\"full_attention\""
                } else {
                    "\"sliding_attention\""
                }
            })
            .collect();
        format!(
            r#"{{"architectures":["Gemma4ForConditionalGeneration"],"model_type":"gemma4",
            "eos_token_id":[1,106],"tie_word_embeddings":true,"vision_config":null,
            "text_config":{{"attention_bias":false,"attention_k_eq_v":false,"bos_token_id":2,
              "dtype":"bfloat16","enable_moe_block":false,"eos_token_id":1,
              "final_logit_softcapping":30.0,"global_head_dim":512,"head_dim":256,
              "hidden_activation":"gelu_pytorch_tanh","hidden_size":2560,
              "hidden_size_per_layer_input":256,"intermediate_size":10240,
              "layer_types":[{}],"max_position_embeddings":131072,"model_type":"gemma4_text",
              "num_attention_heads":8,"num_experts":null,"num_global_key_value_heads":null,
              "num_hidden_layers":42,"num_key_value_heads":2,"num_kv_shared_layers":18,
              "pad_token_id":0,"rms_norm_eps":1e-06,
              "rope_parameters":{{"full_attention":{{"partial_rotary_factor":0.25,
                "rope_theta":1000000.0,"rope_type":"proportional"}},
                "sliding_attention":{{"rope_theta":10000.0,"rope_type":"default"}}}},
              "sliding_window":512,"tie_word_embeddings":true,"top_k_experts":null,
              "use_bidirectional_attention":null,"use_cache":true,"use_double_wide_mlp":false,
              "vocab_size":262144,"vocab_size_per_layer_input":262144{}}}}}"#,
            types.join(","),
            extra
        )
    }

    /// Gemma 4 E4B's nested config resolves to its block: plain-`w` norms, a 5:1
    /// window pattern whose global layers are twice as wide and rotate a
    /// quarter of each head, a shared-KV tail from layer 24, per-layer inputs.
    #[test]
    fn gemma4_e4b_config_resolves() {
        let c = ModelConfig::from_json_bytes(gemma4_e4b_json("").as_bytes(), QuantScheme::Fp16)
            .unwrap();
        assert!(!c.norm_add_one && c.gemma2_norms && c.v_norm);
        assert_eq!(c.sliding_window_pattern, Some(6));
        assert_eq!(c.sliding_window, Some(512));
        assert_eq!((c.head_dim(), c.global_head_dim), (256, Some(512)));
        assert_eq!(c.rope_theta, 1_000_000.0);
        assert_eq!(c.rope_local_theta, Some(10_000.0));
        assert_eq!(
            c.rope_scaling,
            Some(RopeScaling::Partial { fraction: 0.25 })
        );
        assert_eq!(c.query_pre_attn_scalar, Some(1.0));
        assert_eq!(c.kv_shared_from, Some(24));
        assert_eq!(c.ple_dim, 256);
        assert!(!c.double_wide_shared_mlp && c.moe.is_none());
        assert_eq!(c.final_logit_softcap, Some(30.0));
        assert_eq!(c.embed_scale, Some(2560f32.sqrt()));
        // 20 windowed layers of 2×256 and 4 global of 2×512, K and V each; the
        // 18 shared layers keep none.
        assert_eq!(c.kv_values_per_token(), 2 * (20 * 512 + 4 * 1024));
    }

    /// The Gemma 4 variants dlm does not run are refused by name.
    #[test]
    fn gemma4_variants_are_refused() {
        for extra in [
            r#","attention_k_eq_v":true"#,
            r#","enable_moe_block":true"#,
            r#","use_bidirectional_attention":"all""#,
            r#","use_bidirectional_attention":true"#,
        ] {
            let json = gemma4_e4b_json("").replace(r#""attention_k_eq_v":false,"#, "");
            let json = json.replace(r#""enable_moe_block":false,"#, "");
            let json = json.replace(r#""use_bidirectional_attention":null,"#, "");
            let err = ModelConfig::from_json_bytes(
                json.replacen(
                    "\"attention_bias\":false",
                    &format!("\"attention_bias\":false{extra}"),
                    1,
                )
                .as_bytes(),
                QuantScheme::Fp16,
            )
            .unwrap_err();
            assert!(err.to_string().contains("Gemma 4"), "{extra}: {err}");
        }
    }

    /// A Gemma 4 GGUF's metadata, as llama.cpp writes E4B's, reads as the same
    /// config the safetensors one gives.
    #[test]
    fn gemma4_gguf_metadata_matches_the_json_config() {
        use crate::storage::gguf::Value;
        let mut m = std::collections::BTreeMap::new();
        m.insert(
            "general.architecture".to_string(),
            Value::String("gemma4".into()),
        );
        for (k, v) in [
            ("block_count", 42),
            ("context_length", 131072),
            ("embedding_length", 2560),
            ("feed_forward_length", 10240),
            ("attention.head_count", 8),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 512),
            ("attention.key_length_swa", 256),
            ("attention.sliding_window", 512),
            ("attention.shared_kv_layers", 18),
            ("embedding_length_per_layer_input", 256),
        ] {
            m.insert(format!("gemma4.{k}"), Value::U64(v));
        }
        for (k, v) in [
            ("rope.freq_base", 1e6),
            ("rope.freq_base_swa", 1e4),
            ("attention.layer_norm_rms_epsilon", 1e-6),
            ("final_logit_softcapping", 30.0),
        ] {
            m.insert(format!("gemma4.{k}"), Value::F64(v));
        }
        m.insert(
            "gemma4.attention.sliding_window_pattern".to_string(),
            Value::Numbers((0..42).map(|i| ((i + 1) % 6 != 0) as u8 as f64).collect()),
        );
        m.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Strings((0..262144).map(|i| i.to_string()).collect()),
        );
        m.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U64(106));
        let g = ModelConfig::from_gguf(&m, false, QuantScheme::Int4).unwrap();
        let j = ModelConfig::from_json_bytes(gemma4_e4b_json("").as_bytes(), QuantScheme::Int4)
            .unwrap();
        assert_eq!(g.sliding_window_pattern, j.sliding_window_pattern);
        assert_eq!(
            (g.head_dim(), g.global_head_dim),
            (j.head_dim(), j.global_head_dim)
        );
        assert_eq!(
            (g.rope_theta, g.rope_local_theta),
            (j.rope_theta, j.rope_local_theta)
        );
        assert_eq!(g.rope_scaling, j.rope_scaling);
        assert_eq!((g.kv_shared_from, g.ple_dim), (j.kv_shared_from, j.ple_dim));
        assert_eq!(
            (g.intermediate_size, g.double_wide_shared_mlp),
            (10240, false)
        );
        assert!(!g.norm_add_one && g.v_norm && g.tie_word_embeddings);
        assert_eq!(g.eos_token_ids, vec![106]);
        // E2B lists its FFN per layer, doubled on the shared tail.
        let mut e2b = m.clone();
        e2b.insert(
            "gemma4.feed_forward_length".to_string(),
            Value::Numbers(
                (0..42)
                    .map(|i| if i < 24 { 6144.0 } else { 12288.0 })
                    .collect(),
            ),
        );
        let c = ModelConfig::from_gguf(&e2b, false, QuantScheme::Int4).unwrap();
        assert_eq!(
            (c.intermediate_size, c.double_wide_shared_mlp),
            (6144, true)
        );
    }

    #[test]
    fn gguf_gemma_norms_already_carry_the_one() {
        for arch in ["gemma", "gemma2", "gemma3"] {
            let c = ModelConfig::from_gguf(&gguf_meta(arch), false, QuantScheme::Int4).unwrap();
            assert!(!c.norm_add_one, "{arch}: llama.cpp wrote 1 + w already");
        }
        let c = ModelConfig::from_gguf(&gguf_meta("llama"), false, QuantScheme::Int4).unwrap();
        assert!(!c.norm_add_one);
    }

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

    /// DeepSeek-V2's plain softmax top-k keeps loading as the default routing
    /// (V3's is the next test).
    #[test]
    fn deepseek_v2_routing_is_accepted() {
        let json = br#"{"model_type":"deepseek_v2","hidden_size":16,
            "num_attention_heads":4,"num_hidden_layers":4,"vocab_size":32,
            "intermediate_size":8,"moe_intermediate_size":4,"n_routed_experts":64,
            "num_experts_per_tok":6,"scoring_func":"softmax","topk_method":"greedy",
            "n_group":1,"topk_group":1,"routed_scaling_factor":1.0}"#;
        assert!(ModelConfig::from_json_bytes(json, QuantScheme::Fp16)
            .unwrap()
            .is_moe());
    }

    /// DeepSeek-V3's routing keys become its router; a value dlm does not
    /// implement is still refused by name, since routing a token through the
    /// wrong experts reads as fluent, wrong output.
    #[test]
    fn deepseek_v3_routing_parses_and_unknown_routing_is_refused() {
        let base = r#""model_type":"deepseek_v3","hidden_size":16,
            "num_attention_heads":4,"num_hidden_layers":4,"vocab_size":32,
            "intermediate_size":8,"moe_intermediate_size":4,"n_routed_experts":64,
            "num_experts_per_tok":6"#;
        let v3 = format!(
            r#"{{{base},"scoring_func":"sigmoid","topk_method":"noaux_tc","n_group":8,
                "topk_group":4,"routed_scaling_factor":2.5}}"#
        );
        let cfg = ModelConfig::from_json_bytes(v3.as_bytes(), QuantScheme::Fp16).unwrap();
        let r = cfg.moe.unwrap().routing;
        assert_eq!(
            r,
            Routing {
                sigmoid: true,
                n_group: 8,
                topk_group: 4,
                group_by_top2: true,
                score_bias: true,
                scale: 2.5,
            }
        );
        for (key, value) in [
            ("scoring_func", r#""tanh""#),
            ("topk_method", r#""random""#),
            ("n_group", "7"), // 64 experts do not split into 7 groups
        ] {
            let extra = if key == "n_group" {
                r#","topk_method":"noaux_tc""#
            } else {
                ""
            };
            let json = format!(r#"{{{base},"{key}":{value}{extra}}}"#);
            let err = ModelConfig::from_json_bytes(json.as_bytes(), QuantScheme::Fp16)
                .expect_err("unknown routing must be refused");
            assert!(
                format!("{err}").contains(key),
                "expected the refusal to name {key}, got: {err}"
            );
        }
    }

    /// Every Qwen2/2.5 config ships a 32k `sliding_window` *and*
    /// `use_sliding_window: false`. Honoring the span alone windowed attention
    /// the model expects to be full, which only shows up past 32k tokens — and
    /// then as wrong output, not an error.
    #[test]
    fn a_switched_off_sliding_window_is_full_attention() {
        // Shape of Qwen/Qwen2.5-0.5B-Instruct's config.json.
        let qwen = r#""model_type":"qwen2","hidden_size":896,"num_attention_heads":14,
            "num_key_value_heads":2,"num_hidden_layers":24,"vocab_size":151936,
            "intermediate_size":4864,"sliding_window":32768,"max_window_layers":21"#;
        let parse = |extra: &str| {
            ModelConfig::from_json_bytes(
                format!(r#"{{{qwen}{extra}}}"#).as_bytes(),
                QuantScheme::Fp16,
            )
        };
        let off = parse(r#","use_sliding_window":false"#).unwrap();
        assert_eq!(off.sliding_window, None);
        // With the switch on, the window would cover only layers 21.. — a shape
        // dlm cannot express, so it is refused rather than applied to all 24.
        let err = parse(r#","use_sliding_window":true"#)
            .expect_err("a partial window layout must be refused");
        assert!(format!("{err}").contains("max_window_layers"), "{err}");
        // Mistral declares a window and no switch: it applies.
        let mistral = ModelConfig::from_json_bytes(
            br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":4,
                "vocab_size":32,"intermediate_size":8,"sliding_window":4096}"#,
            QuantScheme::Fp16,
        )
        .unwrap();
        assert_eq!(mistral.sliding_window, Some(4096));
    }

    /// Phi/GPT-NeoX/GLM rotate only part of each head. dlm rotates all of it, so
    /// such a checkpoint is refused: the extra rotation mis-places every token
    /// without erroring.
    #[test]
    fn partial_rope_is_refused() {
        let json = br#"{"model_type":"phi","hidden_size":16,"num_attention_heads":4,
            "num_hidden_layers":4,"vocab_size":32,"intermediate_size":8,
            "partial_rotary_factor":0.4}"#;
        let err = ModelConfig::from_json_bytes(json, QuantScheme::Fp16)
            .expect_err("partial RoPE must be refused");
        assert!(format!("{err}").contains("partial_rotary_factor"), "{err}");
        // A factor of 1.0 is the whole head, which is what dlm does.
        let whole = br#"{"hidden_size":16,"num_attention_heads":4,"num_hidden_layers":4,
            "vocab_size":32,"intermediate_size":8,"partial_rotary_factor":1.0}"#;
        assert!(ModelConfig::from_json_bytes(whole, QuantScheme::Fp16).is_ok());
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

//! CPU reference implementation of a transformer decoder block.
//!
//! This is the *real math* that a GPU `ComputeKernel` implements — a
//! Llama-style single-token decode step, in verifiable CPU form:
//!
//! ```text
//!   normed  = rmsnorm(hidden, input_ln)
//!   q,k,v   = Wq·normed, Wk·normed, Wv·normed        (GQA: fewer k/v heads)
//!   q,k     = rope(q,k, position)                    (rotary embedding)
//!   append k,v to this layer's KV history
//!   ctx     = softmax(qᵀK / √d)·V                    (attention over history)
//!   hidden += Wo·ctx                                 (residual)
//!   normed2 = rmsnorm(hidden, post_attn_ln)
//!   hidden += Wdown·(silu(Wgate·normed2) ⊙ Wup·normed2)   (SwiGLU MLP, residual)
//! ```
//!
//! It runs on the host (slow, no batching) and exists as the correctness oracle
//! and porting spec for the GPU kernel — not the production inference path. The
//! streaming `ComputeKernel` trait would widen to this signature (carrying the
//! block config and a KV handle) once the GPU backend lands.

use crate::error::{DlmError, Result};
use crate::forward::kernel::ComputeKernel;

/// RoPE frequency scaling, as declared by `rope_scaling` in `config.json`.
///
/// Modern long-context models are *trained* with a scaled RoPE; ignoring it does
/// not merely truncate context, it makes every position's rotation wrong and the
/// output incoherent. Unsupported scaling types are rejected at config-parse time
/// rather than silently dropped (see [`crate::model::ModelConfig`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// Llama-3 piecewise frequency correction (`rope_type: "llama3"`).
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position: f32,
    },
    /// Linear position interpolation (`rope_type: "linear"`): every frequency is
    /// divided by `factor`.
    Linear { factor: f32 },
}

/// Shape + hyperparameters of one decoder block.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockConfig {
    /// Residual-stream width (`d_model`).
    pub hidden_size: usize,
    /// Query attention heads.
    pub num_heads: usize,
    /// Key/value heads (`< num_heads` under Grouped-Query Attention).
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// FFN inner width.
    pub intermediate_size: usize,
    /// RoPE base frequency (typically 10000).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE frequency scaling, when the model declares one.
    pub rope_scaling: Option<RopeScaling>,
}

impl BlockConfig {
    /// Query projection output width (`num_heads × head_dim`).
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Key/value projection output width (`num_kv_heads × head_dim`).
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Query heads per KV head (the GQA grouping factor).
    pub fn group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }
}

/// A weight matrix in its **native checkpoint dtype**.
///
/// Weights are kept exactly as the checkpoint ships them and converted to f32
/// only in the register that consumes them — never materialized as an upsized
/// f32 copy. Upsizing is *lossless* (an f32 exactly represents every f16/bf16
/// value), so it buys no precision at all, while doubling VRAM, the PCIe traffic
/// per streamed layer, and the bandwidth of the memory-bound GEMV that dominates
/// decode.
///
/// Variants carry the native bit patterns (bf16/f16 as raw `u16`) rather than a
/// byte blob so the F32 path keeps a real `&[f32]` slice for the CPU oracle's
/// autovectorized dot product.
#[derive(Debug, Clone)]
pub enum Weights {
    F32(Vec<f32>),
    /// bf16 bit patterns: the high 16 bits of the equivalent f32.
    Bf16(Vec<u16>),
    /// IEEE half bit patterns.
    F16(Vec<u16>),
    /// 4-bit group-affine codes quantized from the checkpoint at load time
    /// (`--quant int4`): a quarter of bf16's VRAM and PCIe per layer, so more
    /// layers stay resident and a streamed layer costs 4x less to move.
    ///
    /// `blob` holds the whole tensor contiguously — codes, then the per-group
    /// scales, then the per-group zero-points — so it uploads to the device as one
    /// buffer and the kernel still takes a single pointer per matrix. See
    /// [`QuantLayout`] for the offsets; element count and group size are all the
    /// kernel needs to find the scales.
    Int4 { blob: Vec<u8>, group_size: usize, num_elements: usize },
    /// 8-bit group-affine codes quantized from the checkpoint at load
    /// (`--quant int8`): half of bf16's VRAM and PCIe per layer. Coarser than the
    /// original weights but far finer than int4 — 256 levels per group instead of
    /// 16 — so it is the conservative choice when int4 costs too much accuracy.
    /// Same blob layout as [`Weights::Int4`], one code per byte.
    Int8 { blob: Vec<u8>, group_size: usize, num_elements: usize },
}

/// Byte offsets within a quantized [`Weights`] blob.
///
/// Layout: `[codes][pad to 4][scales: g × f32][zeros: g × f32]`, where
/// `g = ceil(n / group_size)`. Only the code width differs between int4 (two
/// codes per byte) and int8 (one). Mirrored by `load_w<DLM_W_INT4>` /
/// `load_w<DLM_W_INT8>` in `src/gpu/kernels.cu` — the two must agree exactly.
#[derive(Debug, Clone, Copy)]
pub struct QuantLayout {
    pub scales_off: usize,
    pub zeros_off: usize,
    pub num_groups: usize,
    pub total_bytes: usize,
}

impl QuantLayout {
    fn new(code_bytes: usize, n: usize, group_size: usize) -> Self {
        let scales_off = code_bytes.div_ceil(4) * 4; // f32 alignment
        let num_groups = n.div_ceil(group_size.max(1));
        let zeros_off = scales_off + num_groups * 4;
        Self { scales_off, zeros_off, num_groups, total_bytes: zeros_off + num_groups * 4 }
    }

    /// Offsets for `n` 4-bit codes (packed two per byte).
    pub fn int4(n: usize, group_size: usize) -> Self {
        Self::new(n.div_ceil(2), n, group_size)
    }

    /// Offsets for `n` 8-bit codes (one per byte).
    pub fn int8(n: usize, group_size: usize) -> Self {
        Self::new(n, n, group_size)
    }
}

/// Weights per quantization group for `--quant int4`/`int8`. 128 is the GPTQ/AWQ
/// convention: small enough that one scale tracks a local range well, large
/// enough that the scale overhead stays a few percent of the codes.
pub const QUANT_GROUP_SIZE: usize = 128;

impl Default for Weights {
    fn default() -> Self {
        Weights::F32(Vec::new())
    }
}

/// bf16 → f32 is exactly a 16-bit shift: bf16 *is* the top half of an f32.
#[inline(always)]
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl Weights {
    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            Weights::F32(v) => v.len(),
            Weights::Bf16(v) | Weights::F16(v) => v.len(),
            Weights::Int4 { num_elements, .. } | Weights::Int8 { num_elements, .. } => {
                *num_elements
            }
        }
    }

    /// True if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build from f32 values (synthetic models, and genuinely-F32 checkpoints).
    pub fn from_f32(v: Vec<f32>) -> Self {
        Weights::F32(v)
    }

    /// An all-zero F32 matrix of `len` elements.
    pub fn zeros(len: usize) -> Self {
        Weights::F32(vec![0.0; len])
    }

    /// Element `i` as f32.
    #[inline(always)]
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Weights::F32(v) => v[i],
            Weights::Bf16(v) => bf16_to_f32(v[i]),
            Weights::F16(v) => crate::storage::f16_to_f32(v[i]),
            Weights::Int4 { blob, group_size, num_elements } => {
                int4_get(blob, *group_size, *num_elements, i)
            }
            Weights::Int8 { blob, group_size, num_elements } => {
                int8_get(blob, *group_size, *num_elements, i)
            }
        }
    }

    /// The raw native bytes, for a verbatim upload to the device.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Weights::F32(v) => bytemuck_cast(v),
            Weights::Bf16(v) | Weights::F16(v) => bytemuck_cast(v),
            Weights::Int4 { blob, .. } | Weights::Int8 { blob, .. } => blob,
        }
    }

    /// Group size for the int4 arm; `0` for the float arms (the kernel ignores it).
    pub fn group_size(&self) -> usize {
        match self {
            Weights::Int4 { group_size, .. } | Weights::Int8 { group_size, .. } => *group_size,
            _ => 0,
        }
    }

    /// Quantize float weights to 4-bit group-affine codes, packed into the blob
    /// layout the kernel reads. Errors only on a zero group size.
    pub fn quantize_int4(values: &[f32], group_size: usize) -> Result<Self> {
        let q = crate::quant::quantize_affine(values, group_size)?;
        let n = values.len();
        let layout = QuantLayout::int4(n, group_size);
        let mut blob = vec![0u8; layout.total_bytes];
        blob[..q.packed().len()].copy_from_slice(q.packed());
        for (g, (&s, &z)) in q.scales().iter().zip(q.zeros()).enumerate() {
            blob[layout.scales_off + g * 4..][..4].copy_from_slice(&s.to_le_bytes());
            blob[layout.zeros_off + g * 4..][..4].copy_from_slice(&z.to_le_bytes());
        }
        Ok(Weights::Int4 { blob, group_size, num_elements: n })
    }

    /// Quantize float weights to 8-bit group-affine codes, in the same blob
    /// layout the kernel reads. Half the shrink of int4, a fraction of the error.
    pub fn quantize_int8(values: &[f32], group_size: usize) -> Result<Self> {
        let (codes, scales, zeros) = crate::quant::quantize_affine_int8(values, group_size)?;
        let n = values.len();
        let layout = QuantLayout::int8(n, group_size);
        let mut blob = vec![0u8; layout.total_bytes];
        blob[..codes.len()].copy_from_slice(&codes);
        for (g, (&s, &z)) in scales.iter().zip(&zeros).enumerate() {
            blob[layout.scales_off + g * 4..][..4].copy_from_slice(&s.to_le_bytes());
            blob[layout.zeros_off + g * 4..][..4].copy_from_slice(&z.to_le_bytes());
        }
        Ok(Weights::Int8 { blob, group_size, num_elements: n })
    }

    /// Dtype tag handed to the CUDA kernel so it can decode elements in-register.
    /// Must match the `DLM_W_*` constants in `src/gpu/kernels.cu`.
    pub fn dtype_code(&self) -> i32 {
        match self {
            Weights::F32(_) => 0,
            Weights::Bf16(_) => 1,
            Weights::F16(_) => 2,
            Weights::Int4 { .. } => 3,
            Weights::Int8 { .. } => 4,
        }
    }
}

/// Decode element `i` of an int4 blob: `(code - zero) * scale` for its group.
/// The device mirror of this is `load_w<DLM_W_INT4>` in `src/gpu/kernels.cu`.
#[inline(always)]
fn int4_get(blob: &[u8], group_size: usize, num_elements: usize, i: usize) -> f32 {
    let layout = QuantLayout::int4(num_elements, group_size);
    let byte = blob[i / 2];
    let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 } as f32;
    let g = i / group_size;
    let at = |off: usize| -> f32 {
        let b = &blob[off + g * 4..][..4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    (code - at(layout.zeros_off)) * at(layout.scales_off)
}

/// Decode element `i` of an int8 blob: `(code - zero) * scale` for its group.
/// The device mirror of this is `load_w<DLM_W_INT8>` in `src/gpu/kernels.cu`.
#[inline(always)]
fn int8_get(blob: &[u8], group_size: usize, num_elements: usize, i: usize) -> f32 {
    let layout = QuantLayout::int8(num_elements, group_size);
    let code = blob[i] as f32;
    let g = i / group_size;
    let at = |off: usize| -> f32 {
        let b = &blob[off + g * 4..][..4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    (code - at(layout.zeros_off)) * at(layout.scales_off)
}

/// Reinterpret a POD slice as bytes. Safe: `T` is a plain numeric type with no
/// padding or invalid bit patterns, and the result borrows the same lifetime.
pub(crate) fn bytemuck_cast<T>(v: &[T]) -> &[u8] {
    // SAFETY: T is f32/u16 (POD, no niches); the byte view has the same lifetime
    // and a length scaled by size_of::<T>().
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Dense weight matrices for one block (row-major `[out, in]`).
///
/// The Q/K/V biases are `Option` because the Llama/Mistral families have none,
/// while Qwen2 ships one per attention projection. Dropping a bias the checkpoint
/// declares yields wrong attention and incoherent text with no error, so the
/// loader reads them whenever they are present.
#[derive(Debug, Clone, Default)]
pub struct LayerTensors {
    pub q_proj: Weights,   // [q_dim, hidden]
    pub k_proj: Weights,   // [kv_dim, hidden]
    pub v_proj: Weights,   // [kv_dim, hidden]
    pub o_proj: Weights,   // [hidden, q_dim]
    pub gate_proj: Weights, // [intermediate, hidden]
    pub up_proj: Weights,   // [intermediate, hidden]
    pub down_proj: Weights, // [hidden, intermediate]
    pub input_layernorm: Vec<f32>,          // [hidden]
    pub post_attention_layernorm: Vec<f32>, // [hidden]
    /// Attention projection biases (Qwen2 et al.); `None` for Llama/Mistral.
    pub q_bias: Option<Vec<f32>>, // [q_dim]
    pub k_bias: Option<Vec<f32>>, // [kv_dim]
    pub v_bias: Option<Vec<f32>>, // [kv_dim]
}

impl LayerTensors {
    /// Approximate host bytes this layer occupies — the projection matrices in
    /// their native dtype plus the norms/biases. Used to bound a byte-budgeted
    /// cache; ignores per-`Vec` overhead, which is negligible against tens of MB
    /// of weights.
    pub fn byte_size(&self) -> usize {
        let bias = |b: &Option<Vec<f32>>| b.as_ref().map_or(0, std::mem::size_of_val);
        self.q_proj.as_bytes().len()
            + self.k_proj.as_bytes().len()
            + self.v_proj.as_bytes().len()
            + self.o_proj.as_bytes().len()
            + self.gate_proj.as_bytes().len()
            + self.up_proj.as_bytes().len()
            + self.down_proj.as_bytes().len()
            + std::mem::size_of_val(&self.input_layernorm[..])
            + std::mem::size_of_val(&self.post_attention_layernorm[..])
            + bias(&self.q_bias)
            + bias(&self.k_bias)
            + bias(&self.v_bias)
    }

    /// Validate every matrix against the config's expected dimensions.
    pub fn validate(&self, cfg: &BlockConfig) -> Result<()> {
        let checks = [
            ("q_proj", self.q_proj.len(), cfg.q_dim() * cfg.hidden_size),
            ("k_proj", self.k_proj.len(), cfg.kv_dim() * cfg.hidden_size),
            ("v_proj", self.v_proj.len(), cfg.kv_dim() * cfg.hidden_size),
            ("o_proj", self.o_proj.len(), cfg.hidden_size * cfg.q_dim()),
            ("gate_proj", self.gate_proj.len(), cfg.intermediate_size * cfg.hidden_size),
            ("up_proj", self.up_proj.len(), cfg.intermediate_size * cfg.hidden_size),
            ("down_proj", self.down_proj.len(), cfg.hidden_size * cfg.intermediate_size),
            ("input_layernorm", self.input_layernorm.len(), cfg.hidden_size),
            ("post_attention_layernorm", self.post_attention_layernorm.len(), cfg.hidden_size),
        ];
        let bias_checks = [
            ("q_bias", self.q_bias.as_ref(), cfg.q_dim()),
            ("k_bias", self.k_bias.as_ref(), cfg.kv_dim()),
            ("v_bias", self.v_bias.as_ref(), cfg.kv_dim()),
        ];
        for (name, bias, expected) in bias_checks {
            if let Some(b) = bias {
                if b.len() != expected {
                    return Err(DlmError::QuantLayout(format!(
                        "{name}: expected {expected} elements, got {}",
                        b.len()
                    )));
                }
            }
        }
        for (name, got, expected) in checks {
            if got != expected {
                return Err(DlmError::QuantLayout(format!(
                    "LayerTensors.{name}: expected {expected} elements, got {got}"
                )));
            }
        }
        Ok(())
    }

    /// All-zero projections with unit norms — a valid block that acts as the
    /// identity on the residual stream (useful as a test baseline).
    pub fn zeros(cfg: &BlockConfig) -> Self {
        Self {
            q_proj: Weights::zeros(cfg.q_dim() * cfg.hidden_size),
            k_proj: Weights::zeros(cfg.kv_dim() * cfg.hidden_size),
            v_proj: Weights::zeros(cfg.kv_dim() * cfg.hidden_size),
            o_proj: Weights::zeros(cfg.hidden_size * cfg.q_dim()),
            gate_proj: Weights::zeros(cfg.intermediate_size * cfg.hidden_size),
            up_proj: Weights::zeros(cfg.intermediate_size * cfg.hidden_size),
            down_proj: Weights::zeros(cfg.hidden_size * cfg.intermediate_size),
            input_layernorm: vec![1.0; cfg.hidden_size],
            post_attention_layernorm: vec![1.0; cfg.hidden_size],
            ..Default::default()
        }
    }
}

/// Real f32 K/V history for one layer (what attention reads). The paged
/// `PagedKvCache` tracks the block bookkeeping; this holds the actual vectors.
#[derive(Debug, Clone, Default)]
pub struct KvLayerCache {
    kv_dim: usize,
    store: KvStore,
}

/// KV cache precision — a memory/quality knob. `None` is exact; the quantized
/// modes symmetric-quantize each token's key and value with a per-token scale,
/// dequantized on the fly during attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// Exact `f32` (default).
    #[default]
    None,
    /// int8, half the memory — small, well-bounded error.
    Int8,
    /// int4 (2 codes/byte), a quarter of the memory — more error.
    Int4,
}

/// How a layer's K/V history is stored: exact `f32`, or int8/int4-quantized per
/// token (each token carries its own scale). See [`KvQuant`].
#[derive(Debug, Clone)]
enum KvStore {
    Full {
        keys: Vec<f32>,
        values: Vec<f32>,
    },
    Int8 {
        keys: Vec<i8>,
        key_scales: Vec<f32>,
        values: Vec<i8>,
        value_scales: Vec<f32>,
    },
    Int4 {
        /// int4 codes packed two-per-byte (`ceil(kv_dim/2)` bytes per token).
        keys: Vec<u8>,
        key_scales: Vec<f32>,
        values: Vec<u8>,
        value_scales: Vec<f32>,
    },
}

impl Default for KvStore {
    fn default() -> Self {
        KvStore::Full { keys: Vec::new(), values: Vec::new() }
    }
}

/// Symmetric per-vector int8 quantization: `scale = max|x| / 127`, `q = round(x /
/// scale)`. Returns all-zero codes with a zero scale for an all-zero input.
fn quantize_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let max = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = max / 127.0;
    if scale <= 0.0 {
        return (vec![0i8; x.len()], 0.0);
    }
    let q = x
        .iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Symmetric per-vector int4 quantization (`scale = max|x| / 7`, codes in
/// `-7..=7`), packed two 4-bit codes per byte (element `i` in byte `i/2`, low
/// nibble for even `i`). Zero scale ⇒ all-zero for an all-zero input.
fn quantize_i4(x: &[f32]) -> (Vec<u8>, f32) {
    let mut packed = vec![0u8; x.len().div_ceil(2)];
    let max = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = max / 7.0;
    if scale <= 0.0 {
        return (packed, 0.0);
    }
    for (i, &v) in x.iter().enumerate() {
        let q = (v / scale).round().clamp(-7.0, 7.0) as i8;
        let nib = (q & 0x0F) as u8; // 4-bit two's complement
        packed[i / 2] |= if i % 2 == 0 { nib } else { nib << 4 };
    }
    (packed, scale)
}

/// Read element `i`'s int4 code from `packed` (starting at byte `base`),
/// sign-extended to `f32`.
#[inline]
fn read_i4(packed: &[u8], base: usize, i: usize) -> f32 {
    let byte = packed[base + i / 2];
    let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    let code = if nib >= 8 { nib as i16 - 16 } else { nib as i16 };
    code as f32
}

impl KvLayerCache {
    /// Empty history at the given precision for a layer of width `kv_dim`.
    pub fn new_quant(kv_dim: usize, quant: KvQuant) -> Self {
        let store = match quant {
            KvQuant::None => KvStore::Full { keys: Vec::new(), values: Vec::new() },
            KvQuant::Int8 => KvStore::Int8 {
                keys: Vec::new(),
                key_scales: Vec::new(),
                values: Vec::new(),
                value_scales: Vec::new(),
            },
            KvQuant::Int4 => KvStore::Int4 {
                keys: Vec::new(),
                key_scales: Vec::new(),
                values: Vec::new(),
                value_scales: Vec::new(),
            },
        };
        Self { kv_dim, store }
    }

    /// Empty (exact `f32`) history for a layer whose K/V width is `kv_dim`.
    pub fn new(kv_dim: usize) -> Self {
        Self::new_quant(kv_dim, KvQuant::None)
    }

    /// Empty int8-quantized history — half the memory of [`new`](Self::new),
    /// approximate. Interchangeable everywhere a `KvLayerCache` is used.
    pub fn new_quantized(kv_dim: usize) -> Self {
        Self::new_quant(kv_dim, KvQuant::Int8)
    }

    /// Key/value width per position.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Cached token positions.
    pub fn len(&self) -> usize {
        match &self.store {
            KvStore::Full { keys, .. } => keys.len() / self.kv_dim.max(1),
            KvStore::Int8 { key_scales, .. } => key_scales.len(),
            KvStore::Int4 { key_scales, .. } => key_scales.len(),
        }
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append one position's key and value vectors.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.kv_dim || value.len() != self.kv_dim {
            return Err(DlmError::ShapeMismatch {
                expected: self.kv_dim,
                got: key.len().min(value.len()),
            });
        }
        match &mut self.store {
            KvStore::Full { keys, values } => {
                keys.extend_from_slice(key);
                values.extend_from_slice(value);
            }
            KvStore::Int8 { keys, key_scales, values, value_scales } => {
                let (qk, ks) = quantize_i8(key);
                let (qv, vs) = quantize_i8(value);
                keys.extend_from_slice(&qk);
                key_scales.push(ks);
                values.extend_from_slice(&qv);
                value_scales.push(vs);
            }
            KvStore::Int4 { keys, key_scales, values, value_scales } => {
                let (qk, ks) = quantize_i4(key);
                let (qv, vs) = quantize_i4(value);
                keys.extend_from_slice(&qk);
                key_scales.push(ks);
                values.extend_from_slice(&qv);
                value_scales.push(vs);
            }
        }
        Ok(())
    }

    /// Dot of `q` with position `pos`'s key head (dequantized inline for int8/4).
    fn key_head_dot(&self, pos: usize, kv_head: usize, d: usize, q: &[f32]) -> f32 {
        let off = kv_head * d;
        match &self.store {
            KvStore::Full { keys, .. } => {
                let base = pos * self.kv_dim + off;
                q.iter().zip(&keys[base..base + d]).map(|(&a, &b)| a * b).sum()
            }
            KvStore::Int8 { keys, key_scales, .. } => {
                let base = pos * self.kv_dim + off;
                let s = key_scales[pos];
                q.iter().zip(&keys[base..base + d]).map(|(&a, &b)| a * (b as f32 * s)).sum()
            }
            KvStore::Int4 { keys, key_scales, .. } => {
                let s = key_scales[pos];
                let byte_base = pos * self.kv_dim.div_ceil(2);
                q.iter()
                    .enumerate()
                    .map(|(j, &a)| a * (read_i4(keys, byte_base, off + j) * s))
                    .sum()
            }
        }
    }

    /// Add `w × value_head(pos)` into `out` (dequantized inline for int8/4).
    fn value_head_accumulate(&self, pos: usize, kv_head: usize, d: usize, w: f32, out: &mut [f32]) {
        let off = kv_head * d;
        match &self.store {
            KvStore::Full { values, .. } => {
                let base = pos * self.kv_dim + off;
                for (o, &vv) in out.iter_mut().zip(&values[base..base + d]) {
                    *o += w * vv;
                }
            }
            KvStore::Int8 { values, value_scales, .. } => {
                let base = pos * self.kv_dim + off;
                let s = value_scales[pos];
                for (o, &vv) in out.iter_mut().zip(&values[base..base + d]) {
                    *o += w * (vv as f32 * s);
                }
            }
            KvStore::Int4 { values, value_scales, .. } => {
                let s = value_scales[pos];
                let byte_base = pos * self.kv_dim.div_ceil(2);
                for (j, o) in out.iter_mut().enumerate() {
                    *o += w * (read_i4(values, byte_base, off + j) * s);
                }
            }
        }
    }
}

// ── numeric primitives ──────────────────────────────────────────────────────

/// Matrix-vector product for a row-major `[out_dim, in_dim]` matrix.
pub(crate) fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    matvec_rows(out_dim, in_dim, x, |o, x| {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        row.iter().zip(x).map(|(&wij, &xj)| wij * xj).sum()
    })
}

/// `matvec` over weights in their native dtype. Dispatches on the dtype **once**
/// so each row's inner loop is monomorphic — the F32 arm keeps the plain
/// `&[f32]` dot product (and its autovectorization) unchanged.
pub(crate) fn matvec_native(w: &Weights, x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    match w {
        Weights::F32(v) => matvec(v, x, out_dim, in_dim),
        Weights::Bf16(v) => matvec_rows(out_dim, in_dim, x, |o, x| {
            let row = &v[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(&b, &xj)| bf16_to_f32(b) * xj).sum()
        }),
        Weights::F16(v) => matvec_rows(out_dim, in_dim, x, |o, x| {
            let row = &v[o * in_dim..(o + 1) * in_dim];
            row.iter()
                .zip(x)
                .map(|(&h, &xj)| crate::storage::f16_to_f32(h) * xj)
                .sum()
        }),
        Weights::Int4 { blob, group_size, num_elements } => {
            matvec_rows(out_dim, in_dim, x, |o, x| {
                let base = o * in_dim;
                (0..in_dim)
                    .map(|j| int4_get(blob, *group_size, *num_elements, base + j) * x[j])
                    .sum()
            })
        }
        Weights::Int8 { blob, group_size, num_elements } => {
            matvec_rows(out_dim, in_dim, x, |o, x| {
                let base = o * in_dim;
                (0..in_dim)
                    .map(|j| int8_get(blob, *group_size, *num_elements, base + j) * x[j])
                    .sum()
            })
        }
    }
}

/// Shared driver: compute `out_dim` independent row dot products via `row_dot`,
/// in parallel for large GEMVs.
///
/// The LM head is [vocab≈128k, hidden] — ~1 GB streamed per token — and
/// single-threaded it dominates decode latency (the GPU path still computes
/// logits on the CPU). Output rows are independent, so split them across cores.
/// ponytail: parallelize only large GEMVs; small per-layer projections aren't
/// worth the thread hop. Threshold in MACs, not a tuned constant.
fn matvec_rows<F>(out_dim: usize, in_dim: usize, x: &[f32], row_dot: F) -> Vec<f32>
where
    F: Fn(usize, &[f32]) -> f32 + Sync,
{
    debug_assert_eq!(x.len(), in_dim);
    let mut y = vec![0.0f32; out_dim];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads > 1 && out_dim.saturating_mul(in_dim) >= (1 << 20) {
        let rows_per = out_dim.div_ceil(threads);
        let row_dot = &row_dot;
        std::thread::scope(|s| {
            for (chunk, yrows) in y.chunks_mut(rows_per).enumerate() {
                let base = chunk * rows_per;
                s.spawn(move || {
                    for (i, slot) in yrows.iter_mut().enumerate() {
                        *slot = row_dot(base + i, x);
                    }
                });
            }
        });
    } else {
        for (o, slot) in y.iter_mut().enumerate() {
            *slot = row_dot(o, x);
        }
    }
    y
}

/// Root-mean-square layer norm with a learned scale.
pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| v * inv * w)
        .collect()
}

/// In-place numerically-stable softmax.
fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// SiLU / swish activation.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The RoPE inverse frequencies for one head: `head_dim / 2` values, with any
/// declared [`RopeScaling`] already folded in.
///
/// This is the **single source of truth** for RoPE frequencies. The CPU block
/// calls it per block; the GPU kernel uploads its output once and indexes the
/// resulting device array — so the two paths cannot drift apart, and a new
/// scaling type is implemented in exactly one place.
pub fn rope_inv_freqs(head_dim: usize, theta: f32, scaling: Option<RopeScaling>) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| {
            let base = theta.powf(-2.0 * i as f32 / head_dim as f32);
            match scaling {
                None => base,
                Some(RopeScaling::Linear { factor }) => base / factor,
                Some(RopeScaling::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position,
                }) => {
                    // HF `_compute_llama3_parameters`: leave high-frequency
                    // (short-wavelength) components alone, divide low-frequency
                    // ones by `factor`, and smoothly blend in between.
                    let wavelen = 2.0 * std::f32::consts::PI / base;
                    let low_wavelen = original_max_position / low_freq_factor;
                    let high_wavelen = original_max_position / high_freq_factor;
                    if wavelen > low_wavelen {
                        base / factor
                    } else if wavelen < high_wavelen {
                        base
                    } else {
                        let smooth = (original_max_position / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * base / factor + smooth * base
                    }
                }
            }
        })
        .collect()
}

/// Apply rotary position embedding in place to a `[num_heads × head_dim]`
/// vector at absolute `position` (NeoX split-half convention), using
/// precomputed [`rope_inv_freqs`].
fn rope_inplace(
    vec: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    position: usize,
    inv_freq: &[f32],
) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let angle = position as f32 * inv_freq[i];
            let (sin, cos) = angle.sin_cos();
            let a = vec[base + i];
            let b = vec[base + i + half];
            vec[base + i] = a * cos - b * sin;
            vec[base + i + half] = a * sin + b * cos;
        }
    }
}

/// Add a projection bias in place, when the checkpoint declares one.
fn add_bias(v: &mut [f32], bias: Option<&Vec<f32>>) {
    if let Some(b) = bias {
        for (x, &bb) in v.iter_mut().zip(b) {
            *x += bb;
        }
    }
}

/// Grouped-query attention over the cached history for a single query vector.
/// Returns the concatenated per-head context (`q_dim` long).
fn attention(cfg: &BlockConfig, q: &[f32], kv: &KvLayerCache) -> Vec<f32> {
    let d = cfg.head_dim;
    let scale = 1.0 / (d as f32).sqrt();
    let group = cfg.group_size();
    let positions = kv.len();
    let mut context = vec![0.0f32; cfg.q_dim()];

    for h in 0..cfg.num_heads {
        let kv_head = h / group;
        let qh = &q[h * d..(h + 1) * d];

        // Scores over every cached position, then softmax.
        let mut scores = vec![0.0f32; positions];
        for (p, score) in scores.iter_mut().enumerate() {
            *score = kv.key_head_dot(p, kv_head, d, qh) * scale;
        }
        softmax_inplace(&mut scores);

        // Weighted sum of values → this head's context.
        let out = &mut context[h * d..(h + 1) * d];
        for (p, &w) in scores.iter().enumerate() {
            kv.value_head_accumulate(p, kv_head, d, w, out);
        }
    }
    context
}

/// Run one decoder block for a single token at absolute `position`, appending to
/// `kv` and returning the updated hidden state. This is the CPU oracle for the
/// GPU `ComputeKernel`.
pub fn decode_block(
    cfg: &BlockConfig,
    w: &LayerTensors,
    hidden: &[f32],
    kv: &mut KvLayerCache,
    position: usize,
) -> Result<Vec<f32>> {
    w.validate(cfg)?;
    if hidden.len() != cfg.hidden_size {
        return Err(DlmError::ShapeMismatch {
            expected: cfg.hidden_size,
            got: hidden.len(),
        });
    }

    // ── attention sublayer ──
    let normed = rmsnorm(hidden, &w.input_layernorm, cfg.rms_eps);
    let mut q = matvec_native(&w.q_proj, &normed, cfg.q_dim(), cfg.hidden_size);
    let mut k = matvec_native(&w.k_proj, &normed, cfg.kv_dim(), cfg.hidden_size);
    let mut v = matvec_native(&w.v_proj, &normed, cfg.kv_dim(), cfg.hidden_size);

    add_bias(&mut q, w.q_bias.as_ref());
    add_bias(&mut k, w.k_bias.as_ref());
    add_bias(&mut v, w.v_bias.as_ref());

    let inv_freq = rope_inv_freqs(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling);
    rope_inplace(&mut q, cfg.num_heads, cfg.head_dim, position, &inv_freq);
    rope_inplace(&mut k, cfg.num_kv_heads, cfg.head_dim, position, &inv_freq);

    kv.append(&k, &v)?;
    let ctx = attention(cfg, &q, kv);
    let attn_out = matvec_native(&w.o_proj, &ctx, cfg.hidden_size, cfg.q_dim());

    let mut h1: Vec<f32> = hidden
        .iter()
        .zip(&attn_out)
        .map(|(&a, &b)| a + b)
        .collect();

    // ── MLP sublayer (SwiGLU) ──
    let normed2 = rmsnorm(&h1, &w.post_attention_layernorm, cfg.rms_eps);
    let gate = matvec_native(&w.gate_proj, &normed2, cfg.intermediate_size, cfg.hidden_size);
    let up = matvec_native(&w.up_proj, &normed2, cfg.intermediate_size, cfg.hidden_size);
    let inter: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    let down = matvec_native(&w.down_proj, &inter, cfg.hidden_size, cfg.intermediate_size);

    for (h, d) in h1.iter_mut().zip(&down) {
        *h += *d;
    }
    Ok(h1)
}

/// A real CPU [`ComputeKernel`] holding a model's per-layer weights.
///
/// Each `run_block` call dispatches to [`decode_block`], giving the
/// [`ForwardOrchestrator`](crate::forward::ForwardOrchestrator) a fully
/// functional (if slow, single-token) CPU forward path. This is the reference
/// implementation the GPU kernel is validated against.
#[derive(Debug, Clone)]
pub struct CpuKernel {
    cfg: BlockConfig,
    layers: Vec<LayerTensors>,
}

impl CpuKernel {
    /// Build a kernel from a shared block config and one [`LayerTensors`] per
    /// layer, validating every layer's matrix dimensions up front.
    pub fn new(cfg: BlockConfig, layers: Vec<LayerTensors>) -> Result<Self> {
        for layer in &layers {
            layer.validate(&cfg)?;
        }
        Ok(Self { cfg, layers })
    }

    /// The block configuration.
    pub fn config(&self) -> &BlockConfig {
        &self.cfg
    }
}

impl ComputeKernel for CpuKernel {
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
        let out = decode_block(&self.cfg, &self.layers[layer as usize], hidden, kv, position)?;
        hidden.copy_from_slice(&out);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < eps, "{x} vs {y}");
        }
    }

    #[test]
    fn matvec_basic() {
        // [[1,2],[3,4]] · [1,1] = [3,7]
        let y = matvec(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, 2);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn rmsnorm_normalizes() {
        // x=[3,4], rms=sqrt(12.5)=3.5355 → [0.8485, 1.1314]
        let y = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        approx(&y, &[0.848528, 1.131371], 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn silu_values() {
        assert!((silu(0.0)).abs() < 1e-6);
        assert!(silu(20.0) > 19.9); // ~x for large x
    }

    #[test]
    fn rope_preserves_norm_and_is_identity_at_zero() {
        let inv = rope_inv_freqs(4, 10000.0, None);
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let before: f32 = v.iter().map(|x| x * x).sum();
        rope_inplace(&mut v, 1, 4, 5, &inv);
        let after: f32 = v.iter().map(|x| x * x).sum();
        assert!((before - after).abs() < 1e-3, "rotation preserves norm");

        // Position 0 → no rotation.
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        rope_inplace(&mut w, 1, 4, 0, &inv);
        approx(&w, &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn unscaled_inv_freqs_are_the_plain_theta_powers() {
        let inv = rope_inv_freqs(8, 10000.0, None);
        // freq_i = theta^(-2i/head_dim)
        approx(
            &inv,
            &[1.0, 10000f32.powf(-0.25), 10000f32.powf(-0.5), 10000f32.powf(-0.75)],
            1e-6,
        );
    }

    #[test]
    fn linear_rope_scaling_divides_every_frequency() {
        let plain = rope_inv_freqs(8, 10000.0, None);
        let scaled = rope_inv_freqs(8, 10000.0, Some(RopeScaling::Linear { factor: 4.0 }));
        let expect: Vec<f32> = plain.iter().map(|f| f / 4.0).collect();
        approx(&scaled, &expect, 1e-9);
    }

    #[test]
    fn llama3_rope_scaling_is_piecewise_in_wavelength() {
        // Llama-3.1/3.2 defaults.
        let s = RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192.0,
        };
        let plain = rope_inv_freqs(128, 500000.0, None);
        let scaled = rope_inv_freqs(128, 500000.0, Some(s));

        for (&p, &sc) in plain.iter().zip(&scaled) {
            let wavelen = 2.0 * std::f32::consts::PI / p;
            if wavelen < 8192.0 / 4.0 {
                // High frequency (short wavelength) → untouched.
                assert!((sc - p).abs() < 1e-9, "high-freq changed: {p} -> {sc}");
            } else if wavelen > 8192.0 / 1.0 {
                // Low frequency (long wavelength) → fully divided by `factor`.
                assert!((sc - p / 8.0).abs() < 1e-9, "low-freq wrong: {p} -> {sc}");
            } else {
                // Mid band → strictly between the two extremes.
                assert!(sc >= p / 8.0 - 1e-9 && sc <= p + 1e-9, "mid-band out of range");
            }
        }
        // Scaling must actually change something, or the test proves nothing.
        assert!(plain.iter().zip(&scaled).any(|(&p, &s)| (p - s).abs() > 1e-9));
    }

    #[test]
    fn qkv_bias_shifts_the_block_output() {
        // A bias the loader drops is silent wrong output — pin that it is applied.
        let cfg = BlockConfig {
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 6,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            rope_scaling: None,
        };
        let hidden = vec![1.5, -2.0, 0.5, 3.0];

        // Zero weights ⇒ identity. Adding a V bias alone feeds a nonzero value
        // through attention → o_proj, but o_proj is zero, so still identity;
        // give o_proj an identity-ish row so the bias can reach the output.
        let mut w = LayerTensors::zeros(&cfg);
        let mut o = vec![0.0f32; cfg.hidden_size * cfg.q_dim()];
        for i in 0..cfg.hidden_size.min(cfg.q_dim()) {
            o[i * cfg.q_dim() + i] = 1.0;
        }
        w.o_proj = Weights::from_f32(o);

        let mut kv = KvLayerCache::new(cfg.kv_dim());
        let base = decode_block(&cfg, &w, &hidden, &mut kv, 0).unwrap();

        w.v_bias = Some(vec![1.0; cfg.kv_dim()]);
        let mut kv2 = KvLayerCache::new(cfg.kv_dim());
        let biased = decode_block(&cfg, &w, &hidden, &mut kv2, 0).unwrap();

        assert!(
            base.iter().zip(&biased).any(|(a, b)| (a - b).abs() > 1e-6),
            "v_bias had no effect on the block output — bias is being dropped"
        );
    }

    #[test]
    fn attention_single_position_returns_its_value() {
        let cfg = BlockConfig {
            hidden_size: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 2,
            rope_theta: 10000.0,
            rms_eps: 1e-5, rope_scaling: None,
        };
        let mut kv = KvLayerCache::new(cfg.kv_dim());
        kv.append(&[1.0, 2.0], &[7.0, 8.0]).unwrap();
        // Softmax over a single score is 1.0, so context == the stored value.
        let ctx = attention(&cfg, &[0.5, -0.5], &kv);
        approx(&ctx, &[7.0, 8.0], 1e-6);
    }

    #[test]
    fn attention_gqa_shares_kv_head_across_query_heads() {
        let cfg = BlockConfig {
            hidden_size: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 1,
            intermediate_size: 2,
            rope_theta: 10000.0,
            rms_eps: 1e-5, rope_scaling: None,
        };
        let mut kv = KvLayerCache::new(cfg.kv_dim());
        kv.append(&[3.0], &[9.0]).unwrap();
        // Both query heads map to the one kv head → both get value 9.0.
        let ctx = attention(&cfg, &[0.1, 0.2], &kv);
        approx(&ctx, &[9.0, 9.0], 1e-6);
    }

    #[test]
    fn decode_block_with_zero_weights_is_identity() {
        let cfg = BlockConfig {
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 6,
            rope_theta: 10000.0,
            rms_eps: 1e-5, rope_scaling: None,
        };
        let w = LayerTensors::zeros(&cfg);
        let mut kv = KvLayerCache::new(cfg.kv_dim());
        let hidden = vec![1.5, -2.0, 0.5, 3.0];
        let out = decode_block(&cfg, &w, &hidden, &mut kv, 0).unwrap();
        // Zero projections ⇒ both residual deltas are zero ⇒ output == input.
        approx(&out, &hidden, 1e-6);
        assert_eq!(kv.len(), 1);
    }

    #[test]
    fn quantize_i8_round_trips_within_a_quantum() {
        let x = [0.0f32, 0.5, -1.0, 3.0, -2.5, 0.1];
        let (q, s) = quantize_i8(&x);
        for (&orig, &code) in x.iter().zip(&q) {
            let deq = code as f32 * s;
            assert!((orig - deq).abs() <= s, "{orig} vs {deq}, scale {s}");
        }
        // All-zero input → zero scale, all-zero codes.
        let (qz, sz) = quantize_i8(&[0.0, 0.0]);
        assert_eq!(sz, 0.0);
        assert!(qz.iter().all(|&c| c == 0));
    }

    #[test]
    fn int8_kv_attention_close_to_full() {
        let cfg = BlockConfig {
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 8,
            rope_theta: 10000.0,
            rms_eps: 1e-5, rope_scaling: None,
        };
        let mut full = KvLayerCache::new(cfg.kv_dim());
        let mut quant = KvLayerCache::new_quantized(cfg.kv_dim());
        for p in 0..5usize {
            let k: Vec<f32> = (0..cfg.kv_dim()).map(|i| ((p + i) % 7) as f32 * 0.3 - 1.0).collect();
            let v: Vec<f32> = (0..cfg.kv_dim()).map(|i| ((p * 2 + i) % 5) as f32 * 0.5).collect();
            full.append(&k, &v).unwrap();
            quant.append(&k, &v).unwrap();
        }
        assert_eq!(full.len(), quant.len());
        let q: Vec<f32> = (0..cfg.q_dim()).map(|i| (i % 3) as f32 * 0.2).collect();
        // int8 KV attention stays well within a small tolerance of exact f32.
        approx(&attention(&cfg, &q, &full), &attention(&cfg, &q, &quant), 0.05);
    }

    #[test]
    fn quantize_i4_round_trips_within_a_quantum() {
        let x = [0.0f32, 0.4, -1.2, 3.0, -2.7, 0.9, -0.1];
        let (packed, s) = quantize_i4(&x);
        for (i, &orig) in x.iter().enumerate() {
            let deq = read_i4(&packed, 0, i) * s;
            assert!((orig - deq).abs() <= s, "{orig} vs {deq}, scale {s}");
        }
        // Two codes per byte: 7 elements → 4 bytes.
        assert_eq!(packed.len(), 4);
        // All-zero input → zero scale, all-zero bytes.
        let (pz, sz) = quantize_i4(&[0.0, 0.0, 0.0]);
        assert_eq!(sz, 0.0);
        assert!(pz.iter().all(|&b| b == 0));
    }

    #[test]
    fn int4_kv_attention_close_to_full() {
        let cfg = BlockConfig {
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 8,
            rope_theta: 10000.0,
            rms_eps: 1e-5, rope_scaling: None,
        };
        let mut full = KvLayerCache::new(cfg.kv_dim());
        let mut q4 = KvLayerCache::new_quant(cfg.kv_dim(), KvQuant::Int4);
        for p in 0..5usize {
            let k: Vec<f32> = (0..cfg.kv_dim()).map(|i| ((p + i) % 7) as f32 * 0.3 - 1.0).collect();
            let v: Vec<f32> = (0..cfg.kv_dim()).map(|i| ((p * 2 + i) % 5) as f32 * 0.5).collect();
            full.append(&k, &v).unwrap();
            q4.append(&k, &v).unwrap();
        }
        assert_eq!(full.len(), q4.len());
        let q: Vec<f32> = (0..cfg.q_dim()).map(|i| (i % 3) as f32 * 0.2).collect();
        // int4 is lossier than int8, but still tracks the exact output closely.
        approx(&attention(&cfg, &q, &full), &attention(&cfg, &q, &q4), 0.3);
    }

    #[test]
    fn decode_block_validates_shapes() {
        let cfg = BlockConfig {
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5, rope_scaling: None,
        };
        let w = LayerTensors::zeros(&cfg);
        let mut kv = KvLayerCache::new(cfg.kv_dim());
        // Wrong hidden length.
        assert!(decode_block(&cfg, &w, &[0.0; 3], &mut kv, 0).is_err());
    }
}

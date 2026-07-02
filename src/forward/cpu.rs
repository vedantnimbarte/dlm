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

use crate::error::{FlipError, Result};
use crate::forward::kernel::ComputeKernel;

/// Shape + hyperparameters of one decoder block.
#[derive(Debug, Clone, Copy)]
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

/// Dense weight matrices for one block (row-major `[out, in]`).
#[derive(Debug, Clone)]
pub struct LayerTensors {
    pub q_proj: Vec<f32>,   // [q_dim, hidden]
    pub k_proj: Vec<f32>,   // [kv_dim, hidden]
    pub v_proj: Vec<f32>,   // [kv_dim, hidden]
    pub o_proj: Vec<f32>,   // [hidden, q_dim]
    pub gate_proj: Vec<f32>, // [intermediate, hidden]
    pub up_proj: Vec<f32>,   // [intermediate, hidden]
    pub down_proj: Vec<f32>, // [hidden, intermediate]
    pub input_layernorm: Vec<f32>,          // [hidden]
    pub post_attention_layernorm: Vec<f32>, // [hidden]
}

impl LayerTensors {
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
        for (name, got, expected) in checks {
            if got != expected {
                return Err(FlipError::QuantLayout(format!(
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
            q_proj: vec![0.0; cfg.q_dim() * cfg.hidden_size],
            k_proj: vec![0.0; cfg.kv_dim() * cfg.hidden_size],
            v_proj: vec![0.0; cfg.kv_dim() * cfg.hidden_size],
            o_proj: vec![0.0; cfg.hidden_size * cfg.q_dim()],
            gate_proj: vec![0.0; cfg.intermediate_size * cfg.hidden_size],
            up_proj: vec![0.0; cfg.intermediate_size * cfg.hidden_size],
            down_proj: vec![0.0; cfg.hidden_size * cfg.intermediate_size],
            input_layernorm: vec![1.0; cfg.hidden_size],
            post_attention_layernorm: vec![1.0; cfg.hidden_size],
        }
    }
}

/// Real f32 K/V history for one layer (what attention reads). The paged
/// `PagedKvCache` tracks the block bookkeeping; this holds the actual vectors.
#[derive(Debug, Clone, Default)]
pub struct KvLayerCache {
    kv_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvLayerCache {
    /// Empty history for a layer whose K/V width is `kv_dim`.
    pub fn new(kv_dim: usize) -> Self {
        Self {
            kv_dim,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Key/value width per position.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Cached token positions.
    pub fn len(&self) -> usize {
        self.keys.len() / self.kv_dim.max(1)
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Append one position's key and value vectors.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.kv_dim || value.len() != self.kv_dim {
            return Err(FlipError::ShapeMismatch {
                expected: self.kv_dim,
                got: key.len().min(value.len()),
            });
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }

    fn key_at(&self, pos: usize) -> &[f32] {
        &self.keys[pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }

    fn value_at(&self, pos: usize) -> &[f32] {
        &self.values[pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }
}

// ── numeric primitives ──────────────────────────────────────────────────────

/// Matrix-vector product for a row-major `[out_dim, in_dim]` matrix.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    let mut y = vec![0.0f32; out_dim];
    for (o, slot) in y.iter_mut().enumerate() {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        *slot = row.iter().zip(x).map(|(&wij, &xj)| wij * xj).sum();
    }
    y
}

/// Root-mean-square layer norm with a learned scale.
fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
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

/// Apply rotary position embedding in place to a `[num_heads × head_dim]`
/// vector at absolute `position` (NeoX split-half convention).
fn rope_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let a = vec[base + i];
            let b = vec[base + i + half];
            vec[base + i] = a * cos - b * sin;
            vec[base + i + half] = a * sin + b * cos;
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
            let kh = &kv.key_at(p)[kv_head * d..(kv_head + 1) * d];
            *score = qh.iter().zip(kh).map(|(&a, &b)| a * b).sum::<f32>() * scale;
        }
        softmax_inplace(&mut scores);

        // Weighted sum of values → this head's context.
        let out = &mut context[h * d..(h + 1) * d];
        for (p, &w) in scores.iter().enumerate() {
            let vh = &kv.value_at(p)[kv_head * d..(kv_head + 1) * d];
            for (o, &vv) in out.iter_mut().zip(vh) {
                *o += w * vv;
            }
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
        return Err(FlipError::ShapeMismatch {
            expected: cfg.hidden_size,
            got: hidden.len(),
        });
    }

    // ── attention sublayer ──
    let normed = rmsnorm(hidden, &w.input_layernorm, cfg.rms_eps);
    let mut q = matvec(&w.q_proj, &normed, cfg.q_dim(), cfg.hidden_size);
    let mut k = matvec(&w.k_proj, &normed, cfg.kv_dim(), cfg.hidden_size);
    let v = matvec(&w.v_proj, &normed, cfg.kv_dim(), cfg.hidden_size);

    rope_inplace(&mut q, cfg.num_heads, cfg.head_dim, position, cfg.rope_theta);
    rope_inplace(&mut k, cfg.num_kv_heads, cfg.head_dim, position, cfg.rope_theta);

    kv.append(&k, &v)?;
    let ctx = attention(cfg, &q, kv);
    let attn_out = matvec(&w.o_proj, &ctx, cfg.hidden_size, cfg.q_dim());

    let mut h1: Vec<f32> = hidden
        .iter()
        .zip(&attn_out)
        .map(|(&a, &b)| a + b)
        .collect();

    // ── MLP sublayer (SwiGLU) ──
    let normed2 = rmsnorm(&h1, &w.post_attention_layernorm, cfg.rms_eps);
    let gate = matvec(&w.gate_proj, &normed2, cfg.intermediate_size, cfg.hidden_size);
    let up = matvec(&w.up_proj, &normed2, cfg.intermediate_size, cfg.hidden_size);
    let inter: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    let down = matvec(&w.down_proj, &inter, cfg.hidden_size, cfg.intermediate_size);

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
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        let before: f32 = v.iter().map(|x| x * x).sum();
        rope_inplace(&mut v, 1, 4, 5, 10000.0);
        let after: f32 = v.iter().map(|x| x * x).sum();
        assert!((before - after).abs() < 1e-3, "rotation preserves norm");

        // Position 0 → no rotation.
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        rope_inplace(&mut w, 1, 4, 0, 10000.0);
        approx(&w, &[1.0, 2.0, 3.0, 4.0], 1e-6);
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
            rms_eps: 1e-5,
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
            rms_eps: 1e-5,
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
            rms_eps: 1e-5,
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
    fn decode_block_validates_shapes() {
        let cfg = BlockConfig {
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 4,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
        };
        let w = LayerTensors::zeros(&cfg);
        let mut kv = KvLayerCache::new(cfg.kv_dim());
        // Wrong hidden length.
        assert!(decode_block(&cfg, &w, &[0.0; 3], &mut kv, 0).is_err());
    }
}

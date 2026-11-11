//! Even out the key channels a quantized KV cache rounds together.
//!
//! An int8 KV cache keeps one scale per 32 values of a key. On Qwen2.5 a few
//! key channels are a hundred times the rest (the `k_proj` bias puts 130-240 on
//! them in layers 0-2 and 8, where the others are 1-6), so each group holding
//! one rounds its neighbours to a step of ~2: mean NLL rose 0.02-0.06 nats.
//!
//! Attention only ever uses `q . k`, which does not change if key channel `c`
//! is divided by `s` and query channel `c` multiplied by it. With `s` shared by
//! the two channels RoPE rotates together, it commutes with the rotation too.
//! So the loader shrinks the large key channels and grows their query channels,
//! in the projection weights and biases (or, where the model normalizes q and
//! k per head, in those norms' weights): no kernel knows.
//!
//! Every `s` is a power of two, applied to exponent bits, so each weight is
//! scaled exactly or not at all, and `(q * 2^e)(k * 2^-e)` is `q * k` to the
//! bit. With an fp16 or f32 KV cache the output is unchanged; only a quantized
//! one sees the difference. The size of each channel comes from the weights
//! alone: `sqrt(bias^2 + |W_row * norm|^2)`, the RMS of a channel whose input
//! the layer norm has made unit-RMS, so no calibration text is needed.

use crate::forward::cpu::{BlockConfig, LayerTensors, QuantLayout, Weights};

/// Largest exponent applied: 2^12 is beyond any channel seen.
const MAX_SHIFT: i32 = 12;

/// `x * 2^e` if that is exact: zero stays zero, a normal value must stay
/// normal. `None` for anything else (subnormal, inf, NaN, out of range).
pub(crate) fn f32_pow2(x: f32, e: i32) -> Option<f32> {
    let bits = x.to_bits();
    if bits & 0x7FFF_FFFF == 0 {
        return Some(x);
    }
    let exp = ((bits >> 23) & 0xFF) as i32;
    let new = exp + e;
    if exp == 0 || exp == 0xFF || !(1..0xFF).contains(&new) {
        return None;
    }
    Some(f32::from_bits(
        (bits & !(0xFF << 23)) | ((new as u32) << 23),
    ))
}

/// [`f32_pow2`] for a 16-bit float with an exponent `exp_bits` wide above
/// `mantissa_bits` (f16: 5 above 10; bf16: 8 above 7).
fn half_pow2(bits: u16, e: i32, mantissa_bits: u32, exp_bits: u32) -> Option<u16> {
    if bits & 0x7FFF == 0 {
        return Some(bits);
    }
    let max = (1i32 << exp_bits) - 1;
    let exp = ((bits >> mantissa_bits) as i32) & max;
    let new = exp + e;
    if exp == 0 || exp == max || !(1..max).contains(&new) {
        return None;
    }
    let mask = (max as u16) << mantissa_bits;
    Some((bits & !mask) | ((new as u16) << mantissa_bits))
}

fn f16_pow2(bits: u16, e: i32) -> Option<u16> {
    half_pow2(bits, e, 10, 5)
}

fn bf16_pow2(bits: u16, e: i32) -> Option<u16> {
    half_pow2(bits, e, 7, 8)
}

/// Scale the f32 scales of the groups covering `span` by `2^e`, all or none.
fn scale_groups(
    blob: &mut [u8],
    layout: QuantLayout,
    group_size: usize,
    span: std::ops::Range<usize>,
    e: i32,
    apply: bool,
) -> bool {
    let gs = group_size.max(1);
    if span.start % gs != 0 || span.len() % gs != 0 {
        return false;
    }
    let at = |g: usize| layout.scales_off + g * 4;
    let read =
        |blob: &[u8], g: usize| f32::from_le_bytes(blob[at(g)..at(g) + 4].try_into().unwrap());
    let groups = span.start / gs..span.end / gs;
    if groups.clone().any(|g| f32_pow2(read(blob, g), e).is_none()) {
        return false;
    }
    if apply {
        for g in groups {
            let v = f32_pow2(read(blob, g), e).unwrap();
            blob[at(g)..at(g) + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    true
}

impl Weights {
    /// Multiply row `row` (`row_len` wide) by `2^e`, exactly: true if every
    /// value that scales it stays a normal float (then, with `apply`, it is
    /// done); false, changing nothing, otherwise. A quantized format scales its
    /// group or block scales, so the row must be whole groups or blocks.
    pub(crate) fn scale_row_pow2(
        &mut self,
        row: usize,
        row_len: usize,
        e: i32,
        apply: bool,
    ) -> bool {
        fn each<T: Copy>(v: &mut [T], f: impl Fn(T) -> Option<T>, apply: bool) -> bool {
            if v.iter().any(|&x| f(x).is_none()) {
                return false;
            }
            if apply {
                for x in v {
                    *x = f(*x).unwrap();
                }
            }
            true
        }
        let span = row * row_len..(row + 1) * row_len;
        match self {
            Weights::F32(v) => each(&mut v[span], |x| f32_pow2(x, e), apply),
            Weights::Bf16(v) => each(&mut v[span], |x| bf16_pow2(x, e), apply),
            Weights::F16(v) => each(&mut v[span], |x| f16_pow2(x, e), apply),
            Weights::Int4 {
                blob,
                group_size,
                num_elements,
            } => {
                let layout = QuantLayout::int4(*num_elements, (*group_size).max(1));
                scale_groups(blob, layout, *group_size, span, e, apply)
            }
            Weights::Int8 {
                blob,
                group_size,
                num_elements,
            } => {
                let layout = QuantLayout::int8(*num_elements, (*group_size).max(1));
                scale_groups(blob, layout, *group_size, span, e, apply)
            }
            Weights::Ggml { blob, kind, .. } => {
                let (be, bb) = (kind.block_elems(), kind.block_bytes());
                if row_len % be != 0 {
                    return false;
                }
                let offsets = kind.f16_scale_offsets();
                let fields = || {
                    (span.start / be..span.end / be)
                        .flat_map(move |b| offsets.iter().map(move |&o| b * bb + o))
                };
                let read = |blob: &[u8], at: usize| u16::from_le_bytes([blob[at], blob[at + 1]]);
                if fields().any(|at| f16_pow2(read(blob, at), e).is_none()) {
                    return false;
                }
                if apply {
                    for at in fields() {
                        let v = f16_pow2(read(blob, at), e).unwrap();
                        blob[at..at + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
                true
            }
        }
    }

    /// Row `row` (`row_len` wide) as f32.
    fn row_f32(&self, row: usize, row_len: usize) -> Vec<f32> {
        if let Weights::Ggml { blob, kind, .. } = self {
            let be = kind.block_elems();
            if row_len % be == 0 {
                let bytes = &blob[row * row_len / be * kind.block_bytes()..];
                let mut out = Vec::with_capacity(row_len);
                for b in 0..row_len / be {
                    let block = crate::storage::ggml_quant::decode_block(
                        *kind,
                        &bytes[b * kind.block_bytes()..],
                    );
                    out.extend((0..be).map(|i| block.value(i)));
                }
                return out;
            }
        }
        (row * row_len..(row + 1) * row_len)
            .map(|i| self.get(i))
            .collect()
    }
}

/// Shrink `layer`'s large key channels and grow the matching query channels
/// by the same power of two (see the module docs). Each RoPE pair of a KV
/// head, `(i, i + head_dim / 2)`, is brought to within a factor of ~1.4 of the
/// layer's median pair, as far as exact scaling of every weight it touches
/// allows. MLA layers, whose keys are not stored this way, are left alone.
pub(crate) fn balance_keys(layer: &mut LayerTensors, cfg: &BlockConfig) {
    if layer.mla.is_some() || cfg.head_dim < 2 || cfg.num_kv_heads == 0 {
        return;
    }
    let (d, kv_heads, h) = (cfg.head_dim, cfg.num_kv_heads, cfg.hidden_size);
    let (half, group) = (d / 2, cfg.num_heads / kv_heads);
    let kv_dim = kv_heads * d;
    if layer.k_proj.len() != kv_dim * h || layer.q_proj.len() != cfg.num_heads * d * h {
        return;
    }

    // Each key channel's RMS when the layer's input is unit-RMS.
    let normed = layer.k_norm.is_some() && layer.q_norm.is_some();
    let size: Vec<f32> = if normed {
        let kn = layer.k_norm.as_ref().unwrap();
        (0..kv_dim).map(|c| kn[c % d].abs()).collect()
    } else {
        let gamma = &layer.input_layernorm;
        (0..kv_dim)
            .map(|c| {
                let w: f32 = layer
                    .k_proj
                    .row_f32(c, h)
                    .iter()
                    .zip(gamma)
                    .map(|(w, g)| (w * g) * (w * g))
                    .sum();
                let b = layer.k_bias.as_ref().map_or(0.0, |b| b[c]);
                (w + b * b).sqrt()
            })
            .collect()
    };
    // A pair's size, per (KV head, i); with q/k norms, shared by every head.
    let pair = |g: usize, i: usize| {
        let (a, b) = (size[g * d + i], size[g * d + i + half]);
        ((a * a + b * b) / 2.0).sqrt()
    };
    let mut all: Vec<f32> = (0..kv_heads)
        .flat_map(|g| (0..half).map(move |i| (g, i)))
        .map(|(g, i)| pair(g, i))
        .filter(|s| s.is_finite())
        .collect();
    if all.is_empty() {
        return;
    }
    all.sort_by(f32::total_cmp);
    let median = all[all.len() / 2].max(f32::MIN_POSITIVE);
    // Only outliers: a pair under 4x the median keeps its scale, and one above
    // comes down into [2x, 4x), where it no longer sets its groups' step alone.
    let shift = |s: f32| ((s / median / 2.0).log2().floor() as i32).clamp(0, MAX_SHIFT);

    if normed {
        for i in 0..half {
            let want = (0..kv_heads).map(|g| shift(pair(g, i))).max().unwrap_or(0);
            for e in (1..=want).rev() {
                let (k, q) = (
                    layer.k_norm.as_mut().unwrap(),
                    layer.q_norm.as_mut().unwrap(),
                );
                let ok = [i, i + half]
                    .iter()
                    .all(|&c| f32_pow2(k[c], -e).is_some() && f32_pow2(q[c], e).is_some());
                if ok {
                    for c in [i, i + half] {
                        k[c] = f32_pow2(k[c], -e).unwrap();
                        q[c] = f32_pow2(q[c], e).unwrap();
                    }
                    break;
                }
            }
        }
        return;
    }

    for g in 0..kv_heads {
        for i in 0..half {
            let k_rows = [g * d + i, g * d + i + half];
            let q_rows: Vec<usize> = (g * group..(g + 1) * group)
                .flat_map(|qh| [qh * d + i, qh * d + i + half])
                .collect();
            for e in (1..=shift(pair(g, i))).rev() {
                let bias_ok = |b: &Option<Vec<f32>>, rows: &[usize], e: i32| {
                    b.as_ref()
                        .is_none_or(|b| rows.iter().all(|&r| f32_pow2(b[r], e).is_some()))
                };
                let ok = k_rows
                    .iter()
                    .all(|&r| layer.k_proj.scale_row_pow2(r, h, -e, false))
                    && q_rows
                        .iter()
                        .all(|&r| layer.q_proj.scale_row_pow2(r, h, e, false))
                    && bias_ok(&layer.k_bias, &k_rows, -e)
                    && bias_ok(&layer.q_bias, &q_rows, e);
                if !ok {
                    continue;
                }
                for &r in &k_rows {
                    layer.k_proj.scale_row_pow2(r, h, -e, true);
                    if let Some(b) = layer.k_bias.as_mut() {
                        b[r] = f32_pow2(b[r], -e).unwrap();
                    }
                }
                for &r in &q_rows {
                    layer.q_proj.scale_row_pow2(r, h, e, true);
                    if let Some(b) = layer.q_bias.as_mut() {
                        b[r] = f32_pow2(b[r], e).unwrap();
                    }
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BlockConfig {
        BlockConfig {
            hidden_size: 16,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            ..Default::default()
        }
    }

    /// A layer whose KV head 0 has a key pair (channels 1 and 5) 300 times the
    /// rest, through its bias, as Qwen2.5's layers 0-2 and 8 do.
    fn layer(bf16: bool) -> LayerTensors {
        let c = cfg();
        let mut seed = 0x9E37_79B9u32;
        let mut rand = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect()
        };
        let mut k_bias = rand(c.num_kv_heads * c.head_dim);
        k_bias[1] = 300.0;
        k_bias[5] = -280.0;
        let w = |v: Vec<f32>| {
            if bf16 {
                Weights::Bf16(v.iter().map(|x| (x.to_bits() >> 16) as u16).collect())
            } else {
                Weights::F32(v)
            }
        };
        LayerTensors {
            q_proj: w(rand(c.num_heads * c.head_dim * c.hidden_size)),
            k_proj: w(rand(c.num_kv_heads * c.head_dim * c.hidden_size)),
            q_bias: Some(rand(c.num_heads * c.head_dim)),
            k_bias: Some(k_bias),
            input_layernorm: vec![1.0; c.hidden_size],
            ..Default::default()
        }
    }

    /// Every head's rotated `q . k` for one input and one position.
    fn scores(l: &LayerTensors) -> Vec<f32> {
        let c = cfg();
        let x: Vec<f32> = (0..c.hidden_size)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let proj = |w: &Weights, b: &[f32], rows: usize| -> Vec<f32> {
            (0..rows)
                .map(|r| {
                    (0..c.hidden_size)
                        .map(|i| w.get(r * c.hidden_size + i) * x[i])
                        .sum::<f32>()
                        + b[r]
                })
                .collect()
        };
        let d = c.head_dim;
        let rope = |v: &mut [f32]| {
            for h in 0..v.len() / d {
                for i in 0..d / 2 {
                    let (s, co) = (0.7 + i as f32).sin_cos();
                    let (a, b) = (v[h * d + i], v[h * d + i + d / 2]);
                    v[h * d + i] = a * co - b * s;
                    v[h * d + i + d / 2] = a * s + b * co;
                }
            }
        };
        let mut q = proj(&l.q_proj, l.q_bias.as_ref().unwrap(), c.num_heads * d);
        let mut k = proj(&l.k_proj, l.k_bias.as_ref().unwrap(), c.num_kv_heads * d);
        rope(&mut q);
        rope(&mut k);
        let group = c.num_heads / c.num_kv_heads;
        (0..c.num_heads)
            .map(|h| (0..d).map(|i| q[h * d + i] * k[(h / group) * d + i]).sum())
            .collect()
    }

    /// The point of it all: attention scores do not move by one bit, while the
    /// outlier pair's keys come down by a power of two.
    #[test]
    fn balancing_keeps_every_score_and_shrinks_the_outlier() {
        for bf16 in [false, true] {
            let mut l = layer(bf16);
            let before = scores(&l);
            balance_keys(&mut l, &cfg());
            assert_eq!(scores(&l), before, "bf16 {bf16}");
            let b = l.k_bias.as_ref().unwrap();
            assert!(
                b[1].abs() <= 300.0 / 16.0 && b[1] != 0.0,
                "bf16 {bf16}: {}",
                b[1]
            );
            assert_eq!(b[1] / 300.0, b[5] / -280.0, "the pair shares its scale");
            assert_eq!(
                b[0],
                layer(bf16).k_bias.unwrap()[0],
                "small channels keep theirs"
            );
        }
    }

    /// With per-head q/k norms the scale goes into those, shared by every head.
    #[test]
    fn normed_models_balance_through_their_norms() {
        let mut l = layer(false);
        let mut k_norm = vec![1.0f32; 8];
        k_norm[2] = 64.0;
        k_norm[6] = 64.0;
        l.k_norm = Some(k_norm.clone());
        l.q_norm = Some(vec![0.5; 8]);
        let (q0, k0) = (l.q_proj.clone(), l.k_bias.clone());
        balance_keys(&mut l, &cfg());
        let (q, k) = (l.q_norm.as_ref().unwrap(), l.k_norm.as_ref().unwrap());
        for c in 0..8 {
            assert_eq!(q[c] * k[c], 0.5 * k_norm[c], "channel {c}");
        }
        assert!(k[2] < 64.0 && k[2] == k[6], "{k:?}");
        assert_eq!(
            l.k_bias, k0,
            "projections untouched when the norms carry it"
        );
        assert_eq!(l.q_proj.get(3), q0.get(3));
    }

    #[test]
    fn powers_of_two_scale_exactly_or_not_at_all() {
        assert_eq!(f32_pow2(3.0, 2), Some(12.0));
        assert_eq!(f32_pow2(-0.75, -3), Some(-0.09375));
        assert_eq!(f32_pow2(0.0, 5), Some(0.0));
        assert_eq!(f32_pow2(f32::MAX, 1), None);
        assert_eq!(f32_pow2(f32::MIN_POSITIVE, -1), None, "would go subnormal");
        let h = |v: f32| crate::storage::safetensors::f32_to_f16(v);
        let back = |b: u16| crate::storage::f16_to_f32(b);
        assert_eq!(f16_pow2(h(1.5), 3).map(back), Some(12.0));
        assert_eq!(f16_pow2(h(6.1035156e-5), -1), None, "f16's smallest normal");
        assert_eq!(f16_pow2(h(40000.0), 1), None, "past f16's range");
        let bf = |v: f32| (v.to_bits() >> 16) as u16;
        assert_eq!(bf16_pow2(bf(2.5), -1), Some(bf(1.25)));
    }
}

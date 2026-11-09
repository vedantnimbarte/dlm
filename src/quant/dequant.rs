//! 4-bit group-affine dequantization kernel (AWQ/GPTQ-style).
//!
//! `dlm` streams 4-bit quantized weights (: 0.5 bytes/param) to
//! keep the model small on disk and on the PCIe bus. Before a layer can be used
//! in a matmul it must be expanded back to a floating form — that expansion is
//! this module.
//!
//! ## Layout
//!
//! We model the canonical **group-wise affine** scheme shared by AWQ and GPTQ:
//! weights are quantized to 4-bit codes `0..=15`, and every `group_size`
//! consecutive weights share one `scale` and one `zero-point`. Dequantization is
//!
//! ```text
//!   w[i] = (code[i] − zero[g]) × scale[g]      where g = i / group_size
//! ```
//!
//! Codes are packed two-per-byte (element `i` uses the low nibble of byte `i/2`
//! when `i` is even, the high nibble when odd). Real AWQ/GPTQ checkpoints add
//! column interleaving and pack into `int32`; that reordering plugs in at the
//! unpack step — the arithmetic here is identical.

use crate::error::{DlmError, Result};

/// Number of distinct 4-bit codes.
const LEVELS: f32 = 15.0; // 2^4 - 1

/// Number of distinct 8-bit codes.
const LEVELS_INT8: f32 = 255.0; // 2^8 - 1

/// A 4-bit group-affine quantized 1-D weight tensor.
#[derive(Debug, Clone)]
pub struct Quant4Tensor {
    /// Packed 4-bit codes, two per byte.
    packed: Vec<u8>,
    /// Per-group scale.
    scales: Vec<f32>,
    /// Per-group zero-point (in code space).
    zeros: Vec<f32>,
    /// Weights per quantization group.
    group_size: usize,
    /// Logical element count (may be < `packed.len() * 2` if odd).
    num_elements: usize,
}

impl Quant4Tensor {
    /// Assemble a tensor from its parts, validating the layout.
    pub fn new(
        packed: Vec<u8>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
        group_size: usize,
        num_elements: usize,
    ) -> Result<Self> {
        if group_size == 0 {
            return Err(DlmError::QuantLayout("group_size must be > 0".into()));
        }
        let expected_bytes = num_elements.div_ceil(2);
        if packed.len() < expected_bytes {
            return Err(DlmError::QuantLayout(format!(
                "packed has {} bytes, need {expected_bytes} for {num_elements} codes",
                packed.len()
            )));
        }
        let num_groups = num_elements.div_ceil(group_size);
        if scales.len() != num_groups {
            return Err(DlmError::QuantLayout(format!(
                "expected {num_groups} scales, got {}",
                scales.len()
            )));
        }
        if zeros.len() != num_groups {
            return Err(DlmError::QuantLayout(format!(
                "expected {num_groups} zero-points, got {}",
                zeros.len()
            )));
        }
        Ok(Self {
            packed,
            scales,
            zeros,
            group_size,
            num_elements,
        })
    }

    /// Logical element count.
    pub fn len(&self) -> usize {
        self.num_elements
    }

    /// True if the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Group count.
    pub fn num_groups(&self) -> usize {
        self.scales.len()
    }

    /// The packed 4-bit codes, two per byte.
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// Per-group scales.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Per-group zero-points, in code space.
    pub fn zeros(&self) -> &[f32] {
        &self.zeros
    }

    /// The 4-bit code at logical index `i` (`0..=15`).
    #[inline]
    pub fn code(&self, i: usize) -> u8 {
        let byte = self.packed[i / 2];
        if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    /// Dequantize a single element to `f32`.
    #[inline]
    pub fn dequantize_element(&self, i: usize) -> f32 {
        let g = i / self.group_size;
        (self.code(i) as f32 - self.zeros[g]) * self.scales[g]
    }

    /// Dequantize the whole tensor into a caller-provided buffer.
    /// `out.len()` must equal [`len`](Self::len).
    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<()> {
        if out.len() != self.num_elements {
            return Err(DlmError::QuantLayout(format!(
                "output buffer holds {} elements, tensor has {}",
                out.len(),
                self.num_elements
            )));
        }
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.dequantize_element(i);
        }
        Ok(())
    }

    /// Dequantize the whole tensor to a fresh `Vec<f32>`.
    pub fn dequantize(&self) -> Vec<f32> {
        (0..self.num_elements)
            .map(|i| self.dequantize_element(i))
            .collect()
    }
}

/// Pack a slice of 4-bit codes (each `0..=15`) two-per-byte.
///
/// The low nibble holds the even index, the high nibble the odd index. Codes
/// above 15 are masked to their low 4 bits.
pub fn pack_codes(codes: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; codes.len().div_ceil(2)];
    for (i, &c) in codes.iter().enumerate() {
        let nib = c & 0x0F;
        if i % 2 == 0 {
            out[i / 2] |= nib;
        } else {
            out[i / 2] |= nib << 4;
        }
    }
    out
}

/// Quantize `values` into a [`Quant4Tensor`] using per-group asymmetric affine
/// quantization: what `--quant int4` does to a float checkpoint at load.
pub fn quantize_affine(values: &[f32], group_size: usize) -> Result<Quant4Tensor> {
    let (codes, scales, zeros) = quantize_groups(values, group_size, LEVELS, true)?;
    Quant4Tensor::new(pack_codes(&codes), scales, zeros, group_size, values.len())
}

/// Per-group asymmetric affine quantization to 8-bit codes: `(codes, scales,
/// zeros)`, one code per element (no packing — a byte already holds one).
///
/// Same layout as [`quantize_affine`], one bit-width up. int8's 256 levels
/// track a group's range ~17x finer than int4's 16, so it costs half the shrink
/// for a fraction of the error.
///
/// Plain min..max rounding, without int4's search: the search cuts int8's
/// squared error by ~10%, but Qwen2.5-0.5B and 1.5B score the same either way
/// (both within 0.002 nats of bf16), and it doubled int8's load time.
pub fn quantize_affine_int8(
    values: &[f32],
    group_size: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    quantize_groups(values, group_size, LEVELS_INT8, false)
}

/// The shared group-affine core: split `values` into groups, and per group pick
/// `scale`/`zero` (see [`fit_group`]) so `dequant(code) = (code - zero) * scale`.
/// `levels` is the top code (15 for 4-bit, 255 for 8-bit).
fn quantize_groups(
    values: &[f32],
    group_size: usize,
    levels: f32,
    search: bool,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    if group_size == 0 {
        return Err(DlmError::QuantLayout("group_size must be > 0".into()));
    }
    let n = values.len();
    let num_groups = n.div_ceil(group_size);
    let mut codes = vec![0u8; n];
    let mut scales = vec![0f32; num_groups];
    let mut zeros = vec![0f32; num_groups];
    // Groups are independent, and the search costs ~20 roundings of each one, so
    // a whole model's worth at load is split across every core.
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = num_groups.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let parts = values
            .chunks(per * group_size)
            .zip(codes.chunks_mut(per * group_size))
            .zip(scales.chunks_mut(per).zip(zeros.chunks_mut(per)));
        for ((values, codes), (scales, zeros)) in parts {
            scope.spawn(move || {
                let mut trial = vec![0u8; group_size];
                let groups = values.chunks(group_size).zip(codes.chunks_mut(group_size));
                for ((group, out), (scale, zero)) in groups.zip(scales.iter_mut().zip(zeros)) {
                    let trial = &mut trial[..group.len()];
                    (*scale, *zero) = fit_group(group, levels, search, out, trial);
                }
            });
        }
    });
    Ok((codes, scales, zeros))
}

/// Choose one group's `(scale, zero)` and write its codes, minimizing the
/// squared reconstruction error rather than just spanning min..max.
///
/// Spanning min..max (round-to-nearest) spends the end codes on the two most
/// extreme weights and rounds everything else on a coarser grid than it needs.
/// This is llama.cpp's `make_qkx2_quants`, the search behind its K-quants, with
/// no calibration data: try grids slightly finer and coarser than min..max, and
/// for each one's codes solve the least-squares `scale`/`min` in closed form,
/// keeping whichever reconstructs the group best. The min..max fit is the first
/// candidate, so the result is never worse than it. Without `search`, the
/// min..max fit is the answer.
fn fit_group(
    group: &[f32],
    levels: f32,
    search: bool,
    codes: &mut [u8],
    trial: &mut [u8],
) -> (f32, f32) {
    let min = group.iter().copied().fold(f32::INFINITY, f32::min);
    let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Degenerate (all equal) → unit scale so code 0 reconstructs the value.
    if (max - min).abs() < f32::EPSILON {
        codes.fill(0);
        return (1.0, -min);
    }
    // The min..max fit: dequant(code=0) = (0 - zero) * scale = min.
    let mut scale = (max - min) / levels;
    let mut zero = -min / scale;
    for (c, &x) in codes.iter_mut().zip(group) {
        *c = ((x - min) / scale).round().clamp(0.0, levels) as u8;
    }
    if !search {
        return (scale, zero);
    }

    // n, Σx and Σx² are the same for every candidate; with each one's Σc, Σc²
    // and Σcx they give its error without a second pass over the group.
    // llama.cpp also weighs each error by `rms + |x|`; on Qwen2.5-0.5B that
    // scored worse than plain squared error, so it is left out.
    let n = group.len() as f64;
    let sx: f64 = group.iter().map(|&x| x as f64).sum();
    let sx2: f64 = group.iter().map(|&x| (x as f64).powi(2)).sum();
    // Squared error of x ≈ s * code + m.
    let error = |s: f64, m: f64, (sl, sl2, sxl): (f64, f64, f64)| {
        s * s * sl2 + 2.0 * s * m * sl - 2.0 * s * sxl + m * m * n - 2.0 * m * sx + sx2
    };
    let sums = |codes: &[u8]| {
        codes
            .iter()
            .zip(group)
            .fold((0.0, 0.0, 0.0), |(sl, sl2, sxl), (&c, &x)| {
                let l = c as f64;
                (sl + l, sl2 + l * l, sxl + l * x as f64)
            })
    };
    let mut best = error(scale as f64, min as f64, sums(codes));

    // Grids from `levels - 1` to `levels + 1` steps across min..max.
    for step in 0..=20 {
        let iscale = (levels - 1.0 + 0.1 * step as f32) / (max - min);
        let (mut sl, mut sl2, mut sxl) = (0u32, 0u32, 0.0f64);
        for (c, &x) in trial.iter_mut().zip(group) {
            // `x - min` is never negative, so +0.5 and truncating rounds it:
            // a plain conversion, where `round()` is a libm call per weight.
            let l = (((x - min) * iscale + 0.5) as u32).min(levels as u32);
            *c = l as u8;
            sl += l;
            sl2 += l * l;
            sxl += l as f64 * x as f64;
        }
        let (sl, sl2) = (sl as f64, sl2 as f64);
        // Least squares of x on these codes: x ≈ s * code + m.
        let d = n * sl2 - sl * sl;
        if d <= 0.0 {
            continue;
        }
        let s = (n * sxl - sx * sl) / d;
        let m = (sl2 * sx - sl * sxl) / d;
        let e = error(s, m, (sl, sl2, sxl));
        if s > 0.0 && e < best {
            best = e;
            scale = s as f32;
            zero = (-m / s) as f32;
            codes.copy_from_slice(trial);
        }
    }
    (scale, zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let codes = [0u8, 15, 1, 8, 7, 3, 9]; // odd length
        let packed = pack_codes(&codes);
        assert_eq!(packed.len(), 4); // ceil(7/2)
        let t = Quant4Tensor::new(packed, vec![1.0], vec![0.0], 8, codes.len()).unwrap();
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(t.code(i), c);
        }
    }

    #[test]
    fn dequantizes_known_example() {
        // codes 0,15 with scale 2.0, zero 0 → 0.0, 30.0
        let packed = pack_codes(&[0, 15]);
        let t = Quant4Tensor::new(packed, vec![2.0], vec![0.0], 2, 2).unwrap();
        let out = t.dequantize();
        assert_eq!(out, vec![0.0, 30.0]);
    }

    #[test]
    fn zero_point_shifts_values() {
        // (code - 8) * 1.0 for codes 8, 0, 15 → 0, -8, 7
        let packed = pack_codes(&[8, 0, 15]);
        let t = Quant4Tensor::new(packed, vec![1.0], vec![8.0], 4, 3).unwrap();
        assert_eq!(t.dequantize(), vec![0.0, -8.0, 7.0]);
    }

    #[test]
    fn quantize_dequantize_round_trip_within_error() {
        // A ramp across two groups of 8.
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 2.0).collect();
        let group_size = 8;
        let q = quantize_affine(&values, group_size).unwrap();
        assert_eq!(q.num_groups(), 2);
        let deq = q.dequantize();

        for (g, chunk) in values.chunks(group_size).enumerate() {
            let (min, max) = chunk
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            let scale = (max - min) / 15.0;
            for (j, &orig) in chunk.iter().enumerate() {
                let got = deq[g * group_size + j];
                // Max affine quant error is half a step.
                assert!(
                    (orig - got).abs() <= scale / 2.0 + 1e-4,
                    "g{g} j{j}: orig {orig} got {got} scale {scale}"
                );
            }
        }
    }

    #[test]
    fn dequantize_into_checks_length() {
        let t = quantize_affine(&[1.0, 2.0, 3.0], 4).unwrap();
        let mut too_small = [0.0f32; 2];
        assert!(t.dequantize_into(&mut too_small).is_err());
        let mut ok = [0.0f32; 3];
        assert!(t.dequantize_into(&mut ok).is_ok());
    }

    #[test]
    fn rejects_mismatched_metadata() {
        let packed = pack_codes(&[1, 2, 3, 4]);
        // 4 elements, group_size 2 → needs 2 scales; give 1.
        assert!(Quant4Tensor::new(packed.clone(), vec![1.0], vec![0.0, 0.0], 2, 4).is_err());
        // too few packed bytes for 4 codes (needs 2, give 1)
        assert!(Quant4Tensor::new(vec![0u8], vec![1.0, 1.0], vec![0.0, 0.0], 2, 4).is_err());
    }

    /// Groups of 128 drawn from a fixed LCG: roughly Gaussian (sum of uniforms),
    /// and with `outliers`, one weight in sixteen scaled up 20x, as in a real
    /// checkpoint's heavy-tailed channels.
    fn groups(seed: u64, outliers: bool) -> Vec<f32> {
        let mut s = seed;
        let mut uniform = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        (0..128 * 64)
            .map(|i| {
                let g = (0..4).map(|_| uniform()).sum::<f32>() * 0.02;
                if outliers && i % 16 == 3 {
                    g * 20.0
                } else {
                    g
                }
            })
            .collect()
    }

    /// The old fit: span min..max, round to nearest. Returns the dequantized group.
    fn min_max_rtn(group: &[f32], levels: f32) -> Vec<f32> {
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / levels;
        group
            .iter()
            .map(|&v| ((v - min) / scale).round().clamp(0.0, levels) * scale + min)
            .collect()
    }

    fn sq_err(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum()
    }

    #[test]
    fn search_never_loses_to_min_max_rounding() {
        for (bits, levels) in [(4, LEVELS), (8, LEVELS_INT8)] {
            for outliers in [false, true] {
                for seed in 1..4 {
                    let values = groups(seed, outliers);
                    let (codes, scales, zeros) =
                        quantize_groups(&values, 128, levels, true).unwrap();
                    let (mut ours, mut rtn) = (0.0, 0.0);
                    for (g, group) in values.chunks(128).enumerate() {
                        let deq: Vec<f32> = codes[g * 128..][..128]
                            .iter()
                            .map(|&c| (c as f32 - zeros[g]) * scales[g])
                            .collect();
                        let (e, r) = (
                            sq_err(group, &deq),
                            sq_err(group, &min_max_rtn(group, levels)),
                        );
                        assert!(e <= r * (1.0 + 1e-5), "int{bits} group {g}: {e} > {r}");
                        ours += e;
                        rtn += r;
                    }
                    // And it is not a no-op: the search finds real gains.
                    assert!(
                        ours < rtn * 0.95,
                        "int{bits} outliers={outliers}: {ours} vs {rtn}"
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_blob_decodes_to_the_chosen_codes() {
        let values = groups(7, true);
        let q = quantize_affine(&values, 128).unwrap();
        let w = crate::forward::Weights::quantize_int4(&values, 128).unwrap();
        for (i, v) in q.dequantize().into_iter().enumerate() {
            assert_eq!(w.get(i), v, "int4 element {i}");
        }
        let (codes, scales, zeros) = quantize_affine_int8(&values, 128).unwrap();
        let w = crate::forward::Weights::quantize_int8(&values, 128).unwrap();
        for (i, &c) in codes.iter().enumerate() {
            assert_eq!(
                w.get(i),
                (c as f32 - zeros[i / 128]) * scales[i / 128],
                "int8 element {i}"
            );
        }
    }

    #[test]
    fn constant_group_reconstructs_exactly() {
        let values = vec![3.5f32; 8];
        let q = quantize_affine(&values, 8).unwrap();
        for v in q.dequantize() {
            assert!((v - 3.5).abs() < 1e-6);
        }
    }
}

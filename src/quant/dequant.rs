//! 4-bit group-affine dequantization kernel (AWQ/GPTQ-style).
//!
//! `dlm` streams 4-bit quantized weights (`specs.md` §3.1: 0.5 bytes/param) to
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
/// quantization. Intended for tests and tooling; the inference path *consumes*
/// already-quantized checkpoints rather than producing them.
pub fn quantize_affine(values: &[f32], group_size: usize) -> Result<Quant4Tensor> {
    if group_size == 0 {
        return Err(DlmError::QuantLayout("group_size must be > 0".into()));
    }
    let n = values.len();
    let mut codes = vec![0u8; n];
    let num_groups = n.div_ceil(group_size);
    let mut scales = Vec::with_capacity(num_groups);
    let mut zeros = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(n);
        let group = &values[start..end];

        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Degenerate (all equal) → unit scale so code 0 reconstructs the value.
        let scale = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            (max - min) / LEVELS
        };
        let zero = -min / scale; // dequant(code=0) = (0 - zero) * scale = min

        for (j, &v) in group.iter().enumerate() {
            let code = ((v - min) / scale).round().clamp(0.0, LEVELS) as u8;
            codes[start + j] = code;
        }
        scales.push(scale);
        zeros.push(zero);
    }

    Quant4Tensor::new(pack_codes(&codes), scales, zeros, group_size, n)
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
            let (min, max) = chunk.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
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

    #[test]
    fn constant_group_reconstructs_exactly() {
        let values = vec![3.5f32; 8];
        let q = quantize_affine(&values, 8).unwrap();
        for v in q.dequantize() {
            assert!((v - 3.5).abs() < 1e-6);
        }
    }
}

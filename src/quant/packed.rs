//! GPTQ-style packed 4-bit group-quantized matrices.
//!
//! Quantized checkpoints (GPTQ / AWQ) don't store one nibble per byte like
//! [`Quant4Tensor`](crate::quant::Quant4Tensor) — they bit-pack eight 4-bit codes
//! into each 32-bit word to match GPU GEMM kernels. This module unpacks that
//! layout back to dense `f32` for the CPU forward path.
//!
//! ## Layout (GPTQ, 4-bit, sequential nibble order)
//!
//! For a `Linear(in_features, out_features)`:
//! * `qweight`: `i32`, shape `[in/8, out]` — 8 consecutive **input** rows packed
//!   per word (row `i`'s code is nibble `i % 8` of word `[i/8, j]`).
//! * `scales`:  `f32`, shape `[in/group_size, out]` — one scale per (group, out).
//! * `qzeros`:  `i32`, shape `[in/group_size, out/8]` — 8 consecutive **output**
//!   columns packed per word (col `j`'s zero is nibble `j % 8`).
//!
//! Dequantization is affine per group: `w[i,j] = (q[i,j] − z[g,j]) × scale[g,j]`
//! with `g = i / group_size`. The result is returned **transposed** to row-major
//! `[out, in]` so it drops straight into [`LayerTensors`](crate::forward::LayerTensors)
//! and the CPU `matvec`.
//!
//! Real exporters vary in two ways this module documents but cannot verify
//! without fixtures: AWQ permutes the nibble order (`[0,2,4,6,1,3,5,7]`) and some
//! GPTQ versions bias the stored zero-point by one. The packing/unpacking,
//! grouping, and transpose logic here is validated by round-trip
//! ([`pack_gptq_4bit`] ↔ [`dequantize_gptq_4bit`]).

use crate::error::{DlmError, Result};

/// Codes per 32-bit word for 4-bit packing.
const NIBBLES_PER_WORD: usize = 8;
/// Max 4-bit code value.
const MAX_CODE: f32 = 15.0;

/// Shape/grouping of a packed quantized matrix.
#[derive(Debug, Clone, Copy)]
pub struct PackedQuantConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: usize,
}

impl PackedQuantConfig {
    fn num_groups(&self) -> usize {
        self.in_features / self.group_size
    }

    fn validate(&self) -> Result<()> {
        if self.in_features % NIBBLES_PER_WORD != 0 {
            return Err(DlmError::QuantLayout(format!(
                "in_features ({}) must be a multiple of 8",
                self.in_features
            )));
        }
        if self.out_features % NIBBLES_PER_WORD != 0 {
            return Err(DlmError::QuantLayout(format!(
                "out_features ({}) must be a multiple of 8",
                self.out_features
            )));
        }
        if self.group_size == 0 || self.in_features % self.group_size != 0 {
            return Err(DlmError::QuantLayout(format!(
                "group_size ({}) must divide in_features ({})",
                self.group_size, self.in_features
            )));
        }
        Ok(())
    }
}

/// Extract nibble `idx` (0..8) from a packed word.
#[inline]
fn nibble(word: i32, idx: usize) -> u32 {
    (word as u32 >> (4 * idx)) & 0xF
}

/// Dequantize a GPTQ-style packed 4-bit matrix to dense row-major `[out, in]`.
///
/// `g_idx` (GPTQ **act-order**, `desc_act: true`) maps each input row to its
/// quantization group; `None` uses the contiguous grouping `g = i / group_size`.
/// Act-order scatters an input row's group non-contiguously, which dlm's flat
/// int4 layout can't represent, so act-order checkpoints dequantize to f32 here.
pub fn dequantize_gptq_4bit(
    qweight: &[i32],
    qzeros: &[i32],
    scales: &[f32],
    cfg: &PackedQuantConfig,
    g_idx: Option<&[i32]>,
) -> Result<Vec<f32>> {
    cfg.validate()?;
    let (inf, out, gs) = (cfg.in_features, cfg.out_features, cfg.group_size);
    let groups = cfg.num_groups();

    let expect = [
        ("qweight", qweight.len(), (inf / NIBBLES_PER_WORD) * out),
        ("scales", scales.len(), groups * out),
        ("qzeros", qzeros.len(), groups * (out / NIBBLES_PER_WORD)),
    ];
    for (name, got, want) in expect {
        if got != want {
            return Err(DlmError::QuantLayout(format!(
                "{name}: expected {want} elements, got {got}"
            )));
        }
    }
    if let Some(gi) = g_idx {
        if gi.len() != inf {
            return Err(DlmError::QuantLayout(format!(
                "g_idx: expected {inf} elements, got {}",
                gi.len()
            )));
        }
    }

    let zeros_per_row = out / NIBBLES_PER_WORD;
    let mut dense = vec![0.0f32; out * inf];
    for i in 0..inf {
        // Act-order: the group of input row `i` is `g_idx[i]`, not `i/group_size`.
        let g = match g_idx {
            Some(gi) => gi[i] as usize,
            None => i / gs,
        };
        if g >= groups {
            return Err(DlmError::QuantLayout(format!("g_idx[{i}] = {g} out of range 0..{groups}")));
        }
        let w_row = i / NIBBLES_PER_WORD;
        let w_nib = i % NIBBLES_PER_WORD;
        for j in 0..out {
            let q = nibble(qweight[w_row * out + j], w_nib) as f32;
            let z = nibble(qzeros[g * zeros_per_row + j / NIBBLES_PER_WORD], j % NIBBLES_PER_WORD)
                as f32;
            let s = scales[g * out + j];
            // Transpose into row-major [out, in].
            dense[j * inf + i] = (q - z) * s;
        }
    }
    Ok(dense)
}

/// AWQ's nibble permutation: output column `p` within a packed word lives at
/// nibble position `AWQ_ORDER[p]`. AWQ interleaves the 8 lanes this way to match
/// its GPU kernel; it is *not* the sequential order GPTQ uses.
const AWQ_ORDER: [usize; NIBBLES_PER_WORD] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Dequantize an **AWQ** 4-bit matrix to dense row-major `[out, in]`.
///
/// AWQ differs from GPTQ in two ways this handles: the output nibbles are packed
/// in [`AWQ_ORDER`], and the stored zero-point is the true zero (no `-1`). Layout
/// (`Linear(in, out)`): `qweight [in, out/8] i32`, `qzeros [in/group, out/8] i32`
/// (both AWQ-ordered along out), `scales [in/group, out] f32`.
pub fn dequantize_awq_4bit(
    qweight: &[i32],
    qzeros: &[i32],
    scales: &[f32],
    cfg: &PackedQuantConfig,
) -> Result<Vec<f32>> {
    cfg.validate()?;
    let (inf, out, gs) = (cfg.in_features, cfg.out_features, cfg.group_size);
    let groups = cfg.num_groups();
    let words = out / NIBBLES_PER_WORD;
    let expect = [
        ("qweight", qweight.len(), inf * words),
        ("scales", scales.len(), groups * out),
        ("qzeros", qzeros.len(), groups * words),
    ];
    for (name, got, want) in expect {
        if got != want {
            return Err(DlmError::QuantLayout(format!(
                "{name}: expected {want} elements, got {got}"
            )));
        }
    }
    let mut dense = vec![0.0f32; out * inf];
    for i in 0..inf {
        let g = i / gs;
        for j in 0..out {
            let lane = AWQ_ORDER[j % NIBBLES_PER_WORD];
            let q = nibble(qweight[i * words + j / NIBBLES_PER_WORD], lane) as f32;
            let z = nibble(qzeros[g * words + j / NIBBLES_PER_WORD], lane) as f32;
            let s = scales[g * out + j];
            dense[j * inf + i] = (q - z) * s;
        }
    }
    Ok(dense)
}

/// Reference AWQ packer (tests/tooling): quantize dense `[in, out]` into AWQ
/// `(qweight, qzeros, scales)` with the [`AWQ_ORDER`] nibble layout, so the
/// round-trip `pack_awq_4bit` ↔ [`dequantize_awq_4bit`] is self-consistent.
pub fn pack_awq_4bit(
    dense_in_by_out: &[f32],
    cfg: &PackedQuantConfig,
) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>)> {
    cfg.validate()?;
    let (inf, out, gs) = (cfg.in_features, cfg.out_features, cfg.group_size);
    if dense_in_by_out.len() != inf * out {
        return Err(DlmError::QuantLayout(format!(
            "dense: expected {} elements, got {}",
            inf * out,
            dense_in_by_out.len()
        )));
    }
    let groups = cfg.num_groups();
    let words = out / NIBBLES_PER_WORD;
    let mut scales = vec![0.0f32; groups * out];
    let mut zeros = vec![0u32; groups * out];
    let mut codes = vec![0u32; inf * out];
    for g in 0..groups {
        for j in 0..out {
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in g * gs..(g + 1) * gs {
                let v = dense_in_by_out[i * out + j];
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let scale = if (mx - mn).abs() < f32::EPSILON { 1.0 } else { (mx - mn) / MAX_CODE };
            let z = (-mn / scale).round().clamp(0.0, MAX_CODE) as u32;
            scales[g * out + j] = scale;
            zeros[g * out + j] = z;
            for i in g * gs..(g + 1) * gs {
                let q = (dense_in_by_out[i * out + j] / scale + z as f32).round().clamp(0.0, MAX_CODE);
                codes[i * out + j] = q as u32;
            }
        }
    }
    let mut qweight = vec![0i32; inf * words];
    for i in 0..inf {
        for j in 0..out {
            let lane = AWQ_ORDER[j % NIBBLES_PER_WORD];
            qweight[i * words + j / NIBBLES_PER_WORD] |=
                ((codes[i * out + j] & 0xF) << (4 * lane)) as i32;
        }
    }
    let mut qzeros = vec![0i32; groups * words];
    for g in 0..groups {
        for j in 0..out {
            let lane = AWQ_ORDER[j % NIBBLES_PER_WORD];
            qzeros[g * words + j / NIBBLES_PER_WORD] |=
                ((zeros[g * out + j] & 0xF) << (4 * lane)) as i32;
        }
    }
    Ok((qweight, qzeros, scales))
}

/// Unpack a GPTQ 4-bit linear into dlm's canonical order: codes in row-major
/// `[out, in]`, plus scales/zeros for *flat* groups of `group_size` over that
/// same order. This is a pure relabeling — no dequantize/requantize round trip —
/// so the checkpoint's own codes and scales survive exactly.
///
/// ## The zero-point convention
///
/// AutoGPTQ stores `zero - 1` in `qzeros`, so the true zero-point is the stored
/// nibble **plus one**. Verified against a real export
/// (`Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4`, `sym: true`): every packed zero is
/// `7`, and a symmetric 4-bit zero-point must be `8`. Using the stored value
/// as-is shifts every weight by one scale step — weights that still look
/// plausible and generate fluent nonsense, which is why this path was refused
/// until a real fixture pinned the convention down.
///
/// `in_features` must be a multiple of `group_size` so a flat group over
/// `[out, in]` never straddles two of GPTQ's per-column groups; the caller checks
/// this (real exports satisfy it — group_size divides the hidden size).
pub fn unpack_gptq_4bit(
    qweight: &[i32],
    qzeros: &[i32],
    scales: &[f32],
    cfg: &PackedQuantConfig,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    cfg.validate()?;
    let (inf, out, gs) = (cfg.in_features, cfg.out_features, cfg.group_size);
    if inf % gs != 0 {
        return Err(DlmError::QuantLayout(format!(
            "in_features {inf} is not a multiple of group_size {gs}; dlm's flat grouping              would straddle two GPTQ groups"
        )));
    }
    let groups = cfg.num_groups();
    let expect = [
        ("qweight", qweight.len(), (inf / NIBBLES_PER_WORD) * out),
        ("scales", scales.len(), groups * out),
        ("qzeros", qzeros.len(), groups * (out / NIBBLES_PER_WORD)),
    ];
    for (name, got, want) in expect {
        if got != want {
            return Err(DlmError::QuantLayout(format!(
                "{name}: expected {want} elements, got {got}"
            )));
        }
    }

    let zeros_per_row = out / NIBBLES_PER_WORD;
    let mut codes = vec![0u8; out * inf];
    // Flat groups over [out, in]: row `j` owns groups `j*groups .. (j+1)*groups`.
    let mut out_scales = vec![0.0f32; out * groups];
    let mut out_zeros = vec![0.0f32; out * groups];

    // Codes: transpose GPTQ's [in/8, out] word grid into row-major [out, in].
    for i in 0..inf {
        let w_row = i / NIBBLES_PER_WORD;
        let w_nib = i % NIBBLES_PER_WORD;
        for j in 0..out {
            codes[j * inf + i] = nibble(qweight[w_row * out + j], w_nib) as u8;
        }
    }
    // Scales/zeros: GPTQ indexes them by (group along in, out); dlm's flat groups
    // over [out, in] map to them 1:1 because in_features % group_size == 0.
    for g in 0..groups {
        for j in 0..out {
            let z = nibble(qzeros[g * zeros_per_row + j / NIBBLES_PER_WORD], j % NIBBLES_PER_WORD);
            // +1: AutoGPTQ stores zero-1 (see above).
            out_zeros[j * groups + g] = z as f32 + 1.0;
            out_scales[j * groups + g] = scales[g * out + j];
        }
    }
    Ok((codes, out_scales, out_zeros))
}

/// Reference packer: quantize a dense `[in, out]` matrix into GPTQ-style
/// `(qweight, qzeros, scales)`. For tests and tooling — inference consumes
/// already-quantized checkpoints.
pub fn pack_gptq_4bit(
    dense_in_by_out: &[f32],
    cfg: &PackedQuantConfig,
) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>)> {
    cfg.validate()?;
    let (inf, out, gs) = (cfg.in_features, cfg.out_features, cfg.group_size);
    if dense_in_by_out.len() != inf * out {
        return Err(DlmError::QuantLayout(format!(
            "dense: expected {} elements, got {}",
            inf * out,
            dense_in_by_out.len()
        )));
    }
    let groups = cfg.num_groups();

    let mut scales = vec![0.0f32; groups * out];
    let mut zeros_int = vec![0u32; groups * out];
    let mut codes = vec![0u32; inf * out];

    for g in 0..groups {
        for j in 0..out {
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in g * gs..(g + 1) * gs {
                let v = dense_in_by_out[i * out + j];
                mn = mn.min(v);
                mx = mx.max(v);
            }
            let scale = if (mx - mn).abs() < f32::EPSILON {
                1.0
            } else {
                (mx - mn) / MAX_CODE
            };
            let z = (-mn / scale).round().clamp(0.0, MAX_CODE) as u32;
            scales[g * out + j] = scale;
            zeros_int[g * out + j] = z;
            for i in g * gs..(g + 1) * gs {
                let v = dense_in_by_out[i * out + j];
                let q = (v / scale + z as f32).round().clamp(0.0, MAX_CODE) as u32;
                codes[i * out + j] = q;
            }
        }
    }

    // Pack qweight: 8 input rows per word.
    let mut qweight = vec![0i32; (inf / NIBBLES_PER_WORD) * out];
    for i in 0..inf {
        for j in 0..out {
            let word = &mut qweight[(i / NIBBLES_PER_WORD) * out + j];
            *word |= ((codes[i * out + j] & 0xF) << (4 * (i % NIBBLES_PER_WORD))) as i32;
        }
    }
    // Pack qzeros: 8 output columns per word.
    let zeros_per_row = out / NIBBLES_PER_WORD;
    let mut qzeros = vec![0i32; groups * zeros_per_row];
    for g in 0..groups {
        for j in 0..out {
            let word = &mut qzeros[g * zeros_per_row + j / NIBBLES_PER_WORD];
            *word |= ((zeros_int[g * out + j] & 0xF) << (4 * (j % NIBBLES_PER_WORD))) as i32;
        }
    }

    Ok((qweight, qzeros, scales))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_extraction() {
        // 0x76543210 → nibbles 0,1,2,3,4,5,6,7
        let w = 0x7654_3210u32 as i32;
        for i in 0..8 {
            assert_eq!(nibble(w, i), i as u32);
        }
    }

    #[test]
    fn pack_dequant_round_trips_within_error() {
        let cfg = PackedQuantConfig {
            in_features: 16,
            out_features: 8,
            group_size: 8,
        };
        // Dense [in, out] ramp.
        let dense: Vec<f32> = (0..cfg.in_features * cfg.out_features)
            .map(|k| (k as f32 % 11.0) * 0.1 - 0.5)
            .collect();

        let (qweight, qzeros, scales) = pack_gptq_4bit(&dense, &cfg).unwrap();
        assert_eq!(qweight.len(), (16 / 8) * 8);
        assert_eq!(qzeros.len(), 16 / 8); // groups × (out/8) = 2 × 1
        assert_eq!(scales.len(), (16 / 8) * 8);

        let deq = dequantize_gptq_4bit(&qweight, &qzeros, &scales, &cfg, None).unwrap();
        assert_eq!(deq.len(), cfg.out_features * cfg.in_features);

        // deq is [out, in]; compare against transposed dense within half a step.
        for i in 0..cfg.in_features {
            let g = i / cfg.group_size;
            for j in 0..cfg.out_features {
                // Recompute this (group, col) scale for the tolerance.
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for ii in g * cfg.group_size..(g + 1) * cfg.group_size {
                    let v = dense[ii * cfg.out_features + j];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let scale = ((mx - mn) / 15.0).max(f32::EPSILON);
                let orig = dense[i * cfg.out_features + j];
                let got = deq[j * cfg.in_features + i];
                assert!(
                    (orig - got).abs() <= scale / 2.0 + 1e-4,
                    "i{i} j{j}: {orig} vs {got} (scale {scale})"
                );
            }
        }
    }

    /// Act-order dequant with an identity `g_idx` equals the contiguous (`None`)
    /// path, and a permuted `g_idx` actually changes the result (it's consulted).
    #[test]
    fn actorder_gidx_is_used() {
        let cfg = PackedQuantConfig { in_features: 16, out_features: 8, group_size: 8 };
        let dense: Vec<f32> =
            (0..cfg.in_features * cfg.out_features).map(|k| (k as f32 % 11.0) * 0.1 - 0.5).collect();
        let (qw, qz, sc) = pack_gptq_4bit(&dense, &cfg).unwrap();

        let none = dequantize_gptq_4bit(&qw, &qz, &sc, &cfg, None).unwrap();
        let identity: Vec<i32> = (0..cfg.in_features).map(|i| (i / cfg.group_size) as i32).collect();
        let with_id = dequantize_gptq_4bit(&qw, &qz, &sc, &cfg, Some(&identity)).unwrap();
        assert_eq!(none, with_id, "identity g_idx must match the contiguous path");

        // Send input row 0 to group 1's scale/zero → a different dequant.
        let mut perm = identity.clone();
        perm[0] = 1;
        let with_perm = dequantize_gptq_4bit(&qw, &qz, &sc, &cfg, Some(&perm)).unwrap();
        assert_ne!(none, with_perm, "a permuted g_idx must change the result");
    }

    /// AWQ pack↔dequant round-trips to within a quantization step, exercising the
    /// interleaved nibble order. (Internal consistency — not a check against a real
    /// AWQ export, which the loader warns about.)
    #[test]
    fn awq_round_trips() {
        let cfg = PackedQuantConfig { in_features: 16, out_features: 8, group_size: 8 };
        let dense: Vec<f32> =
            (0..cfg.in_features * cfg.out_features).map(|k| ((k % 13) as f32 - 6.0) * 0.1).collect();
        let (qw, qz, sc) = pack_awq_4bit(&dense, &cfg).unwrap();
        let deq = dequantize_awq_4bit(&qw, &qz, &sc, &cfg).unwrap();
        for i in 0..cfg.in_features {
            for j in 0..cfg.out_features {
                let scale = sc[(i / cfg.group_size) * cfg.out_features + j];
                let orig = dense[i * cfg.out_features + j];
                let got = deq[j * cfg.in_features + i];
                assert!((orig - got).abs() <= scale / 2.0 + 1e-4, "awq ({i},{j}): {orig} vs {got}");
            }
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        // in_features not a multiple of 8.
        let cfg = PackedQuantConfig {
            in_features: 12,
            out_features: 8,
            group_size: 4,
        };
        assert!(dequantize_gptq_4bit(&[], &[], &[], &cfg, None).is_err());
        // Mismatched lengths.
        let cfg = PackedQuantConfig {
            in_features: 8,
            out_features: 8,
            group_size: 8,
        };
        assert!(dequantize_gptq_4bit(&[0; 3], &[0], &[0.0; 8], &cfg, None).is_err());
    }
}

//! Decoding ggml's block-quantized weights, as they arrive in a GGUF file.
//!
//! Every type here stores weights as `(code - zero) * scale` over a fixed group,
//! which is the same shape as dlm's own int4/int8 weights — the types differ in
//! how many weights share a scale, how the codes are packed, and how the scale
//! itself is stored. So each one is decoded to that common form and handed to
//! the existing kernels, rather than growing a kernel per ggml type.
//!
//! What that costs: dlm's group scale and zero are a pair of f32s, where ggml
//! packs a K-quant's into 6 bits. A Q4_K tensor is 4.5 bits per weight on disk
//! and 6 bits once converted. The types are decoded exactly, so the arithmetic
//! is ggml's; only the storage is dlm's.
//!
//! The layouts mirror `ggml-common.h` and the `dequantize_row_*` functions in
//! `ggml-quants.c`. Each is pinned by a test against values decoded by hand.

use crate::error::{DlmError, Result};
use crate::forward::{QuantLayout, Weights};
use crate::storage::safetensors::{f16_to_f32, Dtype};

/// A decoded block-quantized tensor in dlm's group-affine form: one code per
/// weight, and a scale and zero per group, so a weight is
/// `(code - zero) * scale`.
struct Affine {
    codes: Vec<u8>,
    scales: Vec<f32>,
    zeros: Vec<f32>,
    group_size: usize,
    /// Bits the widest code needs, which decides int4 or int8 storage.
    bits: u32,
}

impl Affine {
    fn new(elements: usize, group_size: usize, bits: u32) -> Self {
        Self {
            codes: Vec::with_capacity(elements),
            scales: Vec::with_capacity(elements / group_size),
            zeros: Vec::with_capacity(elements / group_size),
            group_size,
            bits,
        }
    }

    /// A group's scale and zero from ggml's `scale * code - offset` form: dlm
    /// multiplies the zero by the scale, so it is the offset divided by it. A
    /// zero scale means every weight in the group is zero, whatever the zero is.
    fn push_group(&mut self, scale: f32, offset: f32) {
        self.scales.push(scale);
        self.zeros
            .push(if scale == 0.0 { 0.0 } else { offset / scale });
    }
}

/// Decode `bytes` — `elements` weights in `dtype`'s blocks — into weights the
/// kernels can read. Float dtypes are returned as they are.
pub fn to_weights(dtype: Dtype, bytes: &[u8], elements: usize) -> Result<Weights> {
    let a = affine(dtype, bytes, elements)?;
    let layout = if a.bits <= 4 {
        QuantLayout::int4(elements, a.group_size)
    } else {
        QuantLayout::int8(elements, a.group_size)
    };
    let mut blob = vec![0u8; layout.total_bytes];
    if a.bits <= 4 {
        // dlm packs two codes per byte, the even-indexed weight in the low nibble.
        for (i, code) in a.codes.iter().enumerate() {
            blob[i / 2] |= if i % 2 == 0 { code & 0x0F } else { code << 4 };
        }
    } else {
        blob[..elements].copy_from_slice(&a.codes);
    }
    for (i, v) in a.scales.iter().enumerate() {
        blob[layout.scales_off + i * 4..][..4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in a.zeros.iter().enumerate() {
        blob[layout.zeros_off + i * 4..][..4].copy_from_slice(&v.to_le_bytes());
    }
    Ok(if a.bits <= 4 {
        Weights::Int4 {
            blob,
            group_size: a.group_size,
            num_elements: elements,
        }
    } else {
        Weights::Int8 {
            blob,
            group_size: a.group_size,
            num_elements: elements,
        }
    })
}

/// Decode `bytes` to plain f32, for the tensors dlm keeps unquantized (the
/// embedding matrix, the norms).
pub fn to_f32(dtype: Dtype, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let a = affine(dtype, bytes, elements)?;
    Ok(a.codes
        .iter()
        .enumerate()
        .map(|(i, &code)| {
            let g = i / a.group_size;
            (code as f32 - a.zeros[g]) * a.scales[g]
        })
        .collect())
}

/// The block decoders, one per ggml type.
fn affine(dtype: Dtype, bytes: &[u8], elements: usize) -> Result<Affine> {
    let blocks = elements / dtype.block_elems();
    let need = dtype.byte_len(elements);
    if bytes.len() < need {
        return Err(DlmError::SafetensorsHeader(format!(
            "gguf: a {dtype:?} tensor of {elements} weights needs {need} bytes, got {}",
            bytes.len()
        )));
    }
    let half = |b: &[u8]| f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    Ok(match dtype {
        // 32 weights: an f16 scale, then 16 bytes holding weight j in the low
        // nibble and weight j + 16 in the high one. Codes are 0..15 around 8.
        Dtype::Q4_0 => {
            let mut a = Affine::new(elements, 32, 4);
            for b in bytes.chunks_exact(18).take(blocks) {
                a.codes.resize(a.codes.len() + 32, 0);
                let out = a.codes.len() - 32;
                for (j, &byte) in b[2..18].iter().enumerate() {
                    a.codes[out + j] = byte & 0x0F;
                    a.codes[out + j + 16] = byte >> 4;
                }
                a.scales.push(half(b));
                a.zeros.push(8.0);
            }
            a
        }
        // The "legacy" 32-weight types, which real files still use for the
        // embedding matrix even when every layer is a K-quant. Q4_1 and Q5_1
        // carry a minimum instead of Q4_0/Q5_0's fixed offset, and the 5-bit
        // types keep each weight's top bit in a 32-bit field, one bit per weight.
        Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 => {
            let five = matches!(dtype, Dtype::Q5_0 | Dtype::Q5_1);
            let mut a = Affine::new(elements, 32, if five { 5 } else { 4 });
            for b in bytes.chunks_exact(dtype.block_bytes()).take(blocks) {
                let d = half(b);
                // Q4_1/Q5_1: `d * q + m`, so the zero is -m/d. Q5_0: `d * (q-16)`.
                // `push_group` takes the offset subtracted *after* scaling:
                // Q5_0 is `d * (q - 16)`, so 16 * d; Q4_1/Q5_1 are `d * q + m`.
                let (rest, offset) = match dtype {
                    Dtype::Q5_0 => (&b[2..], 16.0 * d),
                    _ => (&b[4..], -half(&b[2..])),
                };
                a.push_group(d, offset);
                let (qh, qs) = if five {
                    (
                        u32::from_le_bytes(rest[..4].try_into().unwrap()),
                        &rest[4..],
                    )
                } else {
                    (0, rest)
                };
                a.codes.resize(a.codes.len() + 32, 0);
                let out = a.codes.len() - 32;
                for (j, &byte) in qs[..16].iter().enumerate() {
                    let hi_lo = ((qh >> j) as u8 & 1) << 4;
                    let hi_hi = ((qh >> (j + 16)) as u8 & 1) << 4;
                    a.codes[out + j] = (byte & 0x0F) | hi_lo;
                    a.codes[out + j + 16] = (byte >> 4) | hi_hi;
                }
            }
            a
        }
        // 32 weights: an f16 scale, then one signed byte each. dlm's codes are
        // unsigned, so the sign becomes a zero of 128.
        Dtype::Q8_0 => {
            let mut a = Affine::new(elements, 32, 8);
            for b in bytes.chunks_exact(34).take(blocks) {
                a.codes
                    .extend(b[2..34].iter().map(|&q| (q as i8 as i16 + 128) as u8));
                a.scales.push(half(b));
                a.zeros.push(128.0);
            }
            a
        }
        // 256 weights in eight 32-weight sub-blocks. `d` and `dmin` scale the
        // sub-block's 6-bit scale and minimum, packed six to a byte in
        // `scales[12]`; the weights are 4-bit, low nibbles first over each
        // 32-byte run, then high nibbles.
        Dtype::Q4K => {
            let mut a = Affine::new(elements, 32, 4);
            for b in bytes.chunks_exact(144).take(blocks) {
                let (d, dmin) = (half(b), half(&b[2..]));
                let scales = &b[4..16];
                let qs = &b[16..144];
                for sub in 0..8 {
                    let (sc, m) = k_scale_min(sub, scales);
                    a.push_group(d * sc as f32, dmin * m as f32);
                    let run = &qs[(sub / 2) * 32..][..32];
                    let high = sub % 2 == 1;
                    a.codes
                        .extend(run.iter().map(|&q| if high { q >> 4 } else { q & 0x0F }));
                }
            }
            a
        }
        // Q4_K plus a fifth bit per weight, held in `qh[32]` — one bit per weight
        // per sub-block pair, shifting up two bits every 64 weights.
        Dtype::Q5K => {
            let mut a = Affine::new(elements, 32, 5);
            for b in bytes.chunks_exact(176).take(blocks) {
                let (d, dmin) = (half(b), half(&b[2..]));
                let scales = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for sub in 0..8 {
                    let (sc, m) = k_scale_min(sub, scales);
                    a.push_group(d * sc as f32, dmin * m as f32);
                    let run = &qs[(sub / 2) * 32..][..32];
                    let high = sub % 2 == 1;
                    let bit = 1u8 << sub;
                    a.codes.extend(run.iter().zip(qh).map(|(&q, &h)| {
                        let low = if high { q >> 4 } else { q & 0x0F };
                        low + if h & bit != 0 { 16 } else { 0 }
                    }));
                }
            }
            a
        }
        // 256 weights in sixteen 16-weight sub-blocks, each with an int8 scale off
        // a shared f16 `d`. The 6-bit weights are a low nibble in `ql` and two
        // high bits in `qh`, interleaved in runs of 32 across each half-block.
        Dtype::Q6K => {
            let mut a = Affine::new(elements, 16, 6);
            for b in bytes.chunks_exact(210).take(blocks) {
                let (ql, qh) = (&b[0..128], &b[128..192]);
                let scales = &b[192..208];
                let d = half(&b[208..]);
                a.codes.resize(a.codes.len() + 256, 0);
                let out = a.codes.len() - 256;
                for (n, (ql, qh)) in ql.chunks_exact(64).zip(qh.chunks_exact(32)).enumerate() {
                    for l in 0..32 {
                        let (lo, hi) = (ql[l], ql[l + 32]);
                        let h = qh[l];
                        let base = out + n * 128 + l;
                        a.codes[base] = (lo & 0x0F) | ((h & 0x03) << 4);
                        a.codes[base + 32] = (hi & 0x0F) | (((h >> 2) & 0x03) << 4);
                        a.codes[base + 64] = (lo >> 4) | (((h >> 4) & 0x03) << 4);
                        a.codes[base + 96] = (hi >> 4) | (((h >> 6) & 0x03) << 4);
                    }
                }
                // Sub-block scales follow the same interleaving: within each half
                // block, the four runs of 32 weights take scales 0, 2, 4, 6 and
                // then 1, 3, 5, 7 for the second half of each run.
                for &sc in scales {
                    a.scales.push(d * sc as i8 as f32);
                    a.zeros.push(32.0);
                }
            }
            a
        }
        other => {
            return Err(DlmError::UnsupportedQuant(format!(
                "{other:?} is not a block-quantized ggml type"
            )))
        }
    })
}

/// A K-quant sub-block's 6-bit scale and minimum, unpacked from the 12 bytes
/// eight of them share. This is ggml's `get_scale_min_k4`: the first four pairs
/// take six bits from their own byte, the last four take four bits from the
/// upper half of a later byte and two more from the top of an earlier one.
fn k_scale_min(sub: usize, scales: &[u8]) -> (u8, u8) {
    if sub < 4 {
        (scales[sub] & 63, scales[sub + 4] & 63)
    } else {
        (
            (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4),
            (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 bits for a small exact value, so the tests compare exact numbers.
    fn h(v: f32) -> [u8; 2] {
        crate::storage::safetensors::f32_to_f16(v).to_le_bytes()
    }

    /// Q4_0 stores weight j and weight j+16 in one byte, and dlm's codes are
    /// `(code - 8) * d`. Getting the nibble halves backwards would swap the two
    /// halves of every block — a wrong model that still produces fluent text.
    #[test]
    fn q4_0_decodes_both_nibble_halves() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(2.0));
        // Weight j = j % 16 in the low nibble, 15 - j % 16 in the high one.
        for j in 0..16u8 {
            block.push((j & 0x0F) | ((15 - j) << 4));
        }
        let got = to_f32(Dtype::Q4_0, &block, 32).unwrap();
        for j in 0..16 {
            assert_eq!(got[j], (j as f32 - 8.0) * 2.0, "low half at {j}");
            assert_eq!(
                got[j + 16],
                (15.0 - j as f32 - 8.0) * 2.0,
                "high half at {j}"
            );
        }
    }

    /// Q8_0's codes are signed; dlm's are not, so the decode has to put the sign
    /// back through the zero point.
    #[test]
    fn q8_0_decodes_signed_codes() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(0.5));
        for q in [-128i8, -1, 0, 1, 127] {
            block.push(q as u8);
        }
        block.resize(34, 0);
        let got = to_f32(Dtype::Q8_0, &block, 32).unwrap();
        assert_eq!(&got[..5], &[-64.0, -0.5, 0.0, 0.5, 63.5]);
    }

    /// Q5_0 keeps each weight's fifth bit in a 32-bit field, the low 16 bits for
    /// the low nibbles and the high 16 for the high ones. Reading that field the
    /// wrong way round halves the range of every other weight.
    #[test]
    fn q5_0_joins_its_fifth_bit() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(1.0));
        // Bit 0 set (weight 0), bit 16 set (weight 16).
        block.extend_from_slice(&0x0001_0001u32.to_le_bytes());
        block.push(0x21); // weight 0 code 1, weight 16 code 2
        block.resize(22, 0);
        let got = to_f32(Dtype::Q5_0, &block, 32).unwrap();
        assert_eq!(got[0], (1.0 + 16.0) - 16.0);
        assert_eq!(got[16], (2.0 + 16.0) - 16.0);
        // A weight whose bit is clear keeps its nibble.
        assert_eq!(got[1], -16.0);
    }

    /// Q4_1 carries a minimum rather than a fixed offset.
    #[test]
    fn q4_1_uses_its_minimum() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(0.5)); // d
        block.extend_from_slice(&h(-3.0)); // m
        block.push(0x06); // weight 0 code 6
        block.resize(20, 0);
        let got = to_f32(Dtype::Q4_1, &block, 32).unwrap();
        assert!((got[0] - (0.5 * 6.0 - 3.0)).abs() < 1e-5, "{}", got[0]);
        assert!((got[1] - -3.0).abs() < 1e-5, "{}", got[1]);
    }

    /// The 6-bit scale/minimum packing is the part of K-quants most easily got
    /// wrong, and it is shared by Q4_K and Q5_K.
    #[test]
    fn k_quant_scale_packing_matches_ggml() {
        // Sub-blocks 0..3 read their own byte; 4..7 are split across two.
        let mut scales = [0u8; 12];
        scales[0] = 17; // sub 0 scale
        scales[4] = 33; // sub 0 min, and sub 0's low nibble for sub 4's scale
        scales[1] = 0b1100_0000 | 5; // sub 1 scale 5, top bits feed sub 5
        assert_eq!(k_scale_min(0, &scales), (17, 33));
        assert_eq!(k_scale_min(1, &scales), (5, 0));
        // sub 4: low nibble of scales[8], high bits from scales[0] >> 6.
        scales[8] = 0x0A;
        assert_eq!(k_scale_min(4, &scales).0, 0x0A);
        // sub 5's scale takes its top two bits from scales[1] >> 6 = 3.
        scales[9] = 0x02;
        assert_eq!(k_scale_min(5, &scales).0, 0x02 | (3 << 4));
    }

    /// Q4_K's weights are `d * sc * q - dmin * m`, which dlm stores as
    /// `(q - zero) * scale`. The two must agree to float rounding.
    #[test]
    fn q4_k_matches_its_defining_formula() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(0.25)); // d
        block.extend_from_slice(&h(0.5)); // dmin
        let mut scales = [0u8; 12];
        scales[0] = 3; // sub 0 scale
        scales[4] = 7; // sub 0 minimum
        block.extend_from_slice(&scales);
        let mut qs = vec![0u8; 128];
        qs[0] = 0x09; // first weight's code = 9
        block.extend_from_slice(&qs);

        let got = to_f32(Dtype::Q4K, &block, 256).unwrap();
        let want = 0.25 * 3.0 * 9.0 - 0.5 * 7.0;
        assert!((got[0] - want).abs() < 1e-5, "{} vs {want}", got[0]);
    }

    /// Q6_K's 6-bit weights come from a nibble plus two bits held elsewhere, and
    /// its sub-block scales are signed.
    #[test]
    fn q6_k_joins_its_low_and_high_bits() {
        let mut block = vec![0u8; 210];
        block[0] = 0x05; // ql[0]: low nibble of weight 0
        block[128] = 0b0000_0011; // qh[0]: top two bits of weight 0
        block[192] = -2i8 as u8; // scales[0]
        block[208..210].copy_from_slice(&h(0.5));
        let got = to_f32(Dtype::Q6K, &block, 256).unwrap();
        // code = 0x05 | (3 << 4) = 53, value = d * sc * (code - 32)
        let want = 0.5 * -2.0 * (53.0 - 32.0);
        assert!((got[0] - want).abs() < 1e-5, "{} vs {want}", got[0]);
    }

    /// Whatever the ggml type, the weights handed to the kernels must decode to
    /// the same numbers `to_f32` produces — that is the contract the group-affine
    /// conversion rests on.
    #[test]
    fn packed_weights_decode_to_the_same_values() {
        let mut q4k = Vec::new();
        q4k.extend_from_slice(&h(0.25));
        q4k.extend_from_slice(&h(0.5));
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as u8 * 5 + 1) & 63;
        }
        q4k.extend_from_slice(&scales);
        q4k.extend((0..128u8).map(|i| i.wrapping_mul(7)));

        for (dtype, bytes, n) in [(Dtype::Q4K, q4k.as_slice(), 256)] {
            let want = to_f32(dtype, bytes, n).unwrap();
            let w = to_weights(dtype, bytes, n).unwrap();
            for (i, expected) in want.iter().enumerate() {
                let got = w.get(i);
                assert!(
                    (got - expected).abs() < 1e-5,
                    "{dtype:?} weight {i}: {got} vs {expected}"
                );
            }
        }
    }
}

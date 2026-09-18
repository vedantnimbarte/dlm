//! ggml's block-quantized weights, as they arrive in a GGUF file.
//!
//! A block holds a fixed number of weights together with the scale (and often a
//! minimum) that decodes them, packed tightly: a Q4_K block is 256 weights in
//! 144 bytes, 4.5 bits each. dlm keeps the blocks exactly as the file has them
//! and decodes a weight where it is used — in the register that multiplies it —
//! so a quantized model costs the same in VRAM as it does on disk.
//!
//! [`decode_block`] is the one place that knows the layouts. The kernels mirror
//! it: `load_w<DLM_W_Q4_K>` and friends in `src/gpu/kernels.cu` decode the same
//! bytes the same way, and the two must agree exactly.
//!
//! The layouts follow `ggml-common.h` and the `dequantize_row_*` functions in
//! `ggml-quants.c`. Each is pinned by a test against values decoded by hand, and
//! `tests/gguf_model.rs` checks a whole model against llama.cpp's own
//! dequantization of the same file.

use crate::error::{DlmError, Result};
use crate::storage::safetensors::{f16_to_f32, Dtype};

/// A ggml block-quantized weight type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlKind {
    /// 32 weights as `d * (q - 8)`, `d` an f16. 18 bytes.
    Q4_0,
    /// 32 weights as `d * q + m`, both f16. 20 bytes.
    Q4_1,
    /// `Q4_0` with a fifth bit per weight, in a 32-bit field. 22 bytes.
    Q5_0,
    /// `Q4_1` with a fifth bit per weight. 24 bytes.
    Q5_1,
    /// 32 weights as `d * q`, `q` signed. 34 bytes.
    Q8_0,
    /// 256 weights in eight 32-weight sub-blocks, each with a 6-bit scale and
    /// minimum off a shared pair of f16s. 144 bytes.
    Q4K,
    /// `Q4_K` with a fifth bit per weight. 176 bytes.
    Q5K,
    /// 256 weights in sixteen 16-weight sub-blocks, 6-bit weights with an int8
    /// scale each. 210 bytes.
    Q6K,
}

impl GgmlKind {
    /// The kind a [`Dtype`] names, or `None` if it is not a ggml block type.
    pub fn from_dtype(dtype: Dtype) -> Option<Self> {
        Some(match dtype {
            Dtype::Q4_0 => GgmlKind::Q4_0,
            Dtype::Q4_1 => GgmlKind::Q4_1,
            Dtype::Q5_0 => GgmlKind::Q5_0,
            Dtype::Q5_1 => GgmlKind::Q5_1,
            Dtype::Q8_0 => GgmlKind::Q8_0,
            Dtype::Q4K => GgmlKind::Q4K,
            Dtype::Q5K => GgmlKind::Q5K,
            Dtype::Q6K => GgmlKind::Q6K,
            _ => return None,
        })
    }

    /// Weights per block.
    pub fn block_elems(self) -> usize {
        match self {
            GgmlKind::Q4K | GgmlKind::Q5K | GgmlKind::Q6K => 256,
            _ => 32,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            GgmlKind::Q4_0 => 18,
            GgmlKind::Q4_1 => 20,
            GgmlKind::Q5_0 => 22,
            GgmlKind::Q5_1 => 24,
            GgmlKind::Q8_0 => 34,
            GgmlKind::Q4K => 144,
            GgmlKind::Q5K => 176,
            GgmlKind::Q6K => 210,
        }
    }

    /// The tag the CUDA kernels dispatch on. Must match the `DLM_W_*` constants
    /// in `src/gpu/kernels.cu`.
    pub fn dtype_code(self) -> i32 {
        match self {
            GgmlKind::Q4_0 => 5,
            GgmlKind::Q4_1 => 6,
            GgmlKind::Q5_0 => 7,
            GgmlKind::Q5_1 => 8,
            GgmlKind::Q8_0 => 9,
            GgmlKind::Q4K => 10,
            GgmlKind::Q5K => 11,
            GgmlKind::Q6K => 12,
        }
    }

    /// Bytes `elements` weights occupy in this type's blocks.
    pub fn byte_len(self, elements: usize) -> usize {
        elements.div_ceil(self.block_elems()) * self.block_bytes()
    }
}

fn half(b: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// A K-quant sub-block's 6-bit scale and minimum, unpacked from the 12 bytes
/// eight of them share. This is ggml's `get_scale_min_k4`: the first four pairs
/// take six bits from their own byte, the last four take four bits from the
/// upper half of a later byte and two more from the top of an earlier one.
pub(crate) fn k_scale_min(sub: usize, scales: &[u8]) -> (u8, u8) {
    if sub < 4 {
        (scales[sub] & 63, scales[sub + 4] & 63)
    } else {
        (
            (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4),
            (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4),
        )
    }
}

/// One block decoded: a code per weight, and the scale and zero that turn a code
/// into a weight -- `(code - zero) * scale` -- for each group inside the block.
///
/// Both ways dlm reads these weights come from here: the value decode below, and
/// [`expand`], which re-packs them into dlm's own quantized layout. The layouts
/// themselves are written once, in [`decode_block`].
pub struct BlockAffine {
    pub codes: [u8; 256],
    pub scales: [f32; 16],
    pub zeros: [f32; 16],
    /// Weights sharing one scale and zero.
    pub group: usize,
}

impl BlockAffine {
    fn new(group: usize) -> Self {
        Self {
            codes: [0; 256],
            scales: [0.0; 16],
            zeros: [0.0; 16],
            group,
        }
    }

    /// Record a group from ggml's `scale * code - offset` form. dlm multiplies
    /// its zero by the scale, so the zero is the offset divided by it; a zero
    /// scale means the whole group is zero, whatever the zero.
    fn group(&mut self, g: usize, scale: f32, offset: f32) {
        self.scales[g] = scale;
        self.zeros[g] = if scale == 0.0 { 0.0 } else { offset / scale };
    }

    /// Weight `i` of the block.
    pub fn value(&self, i: usize) -> f32 {
        let g = i / self.group;
        (self.codes[i] as f32 - self.zeros[g]) * self.scales[g]
    }
}

/// Decode one block of `kind` from `bytes`.
///
/// This is the definition of each layout; everything else in dlm that reads a
/// ggml weight goes through it or mirrors it in a kernel.
pub fn decode_block(kind: GgmlKind, bytes: &[u8]) -> BlockAffine {
    debug_assert!(bytes.len() >= kind.block_bytes());
    let mut a = BlockAffine::new(if kind == GgmlKind::Q6K { 16 } else { 32 });
    match kind {
        // An f16 scale, then 16 bytes holding weight j in the low nibble and
        // weight j + 16 in the high one. Codes are 0..15 around 8.
        GgmlKind::Q4_0 => {
            for (j, &byte) in bytes[2..18].iter().enumerate() {
                a.codes[j] = byte & 0x0F;
                a.codes[j + 16] = byte >> 4;
            }
            let d = half(bytes);
            a.group(0, d, 8.0 * d);
        }
        // The same, with a minimum in place of the fixed offset.
        GgmlKind::Q4_1 => {
            for (j, &byte) in bytes[4..20].iter().enumerate() {
                a.codes[j] = byte & 0x0F;
                a.codes[j + 16] = byte >> 4;
            }
            a.group(0, half(bytes), -half(&bytes[2..]));
        }
        // A fifth bit per weight, in a 32-bit field: weight j takes bit j.
        GgmlKind::Q5_0 | GgmlKind::Q5_1 => {
            let one = kind == GgmlKind::Q5_1;
            let d = half(bytes);
            let (offset, rest) = if one {
                (-half(&bytes[2..]), &bytes[4..])
            } else {
                (16.0 * d, &bytes[2..])
            };
            let qh = u32::from_le_bytes(rest[..4].try_into().unwrap());
            for (j, &byte) in rest[4..20].iter().enumerate() {
                a.codes[j] = (byte & 0x0F) | (((qh >> j) as u8 & 1) << 4);
                a.codes[j + 16] = (byte >> 4) | (((qh >> (j + 16)) as u8 & 1) << 4);
            }
            a.group(0, d, offset);
        }
        // An f16 scale and one signed byte per weight. dlm's codes are unsigned,
        // so the sign becomes a zero of 128.
        GgmlKind::Q8_0 => {
            for (j, &q) in bytes[2..34].iter().enumerate() {
                a.codes[j] = (q as i8 as i16 + 128) as u8;
            }
            let d = half(bytes);
            a.group(0, d, 128.0 * d);
        }
        // Eight 32-weight sub-blocks. `d` and `dmin` scale the sub-block's 6-bit
        // scale and minimum; the weights are 4-bit, low nibbles over each 32-byte
        // run and then high nibbles. Q5_K adds a fifth bit from `qh`.
        GgmlKind::Q4K | GgmlKind::Q5K => {
            let five = kind == GgmlKind::Q5K;
            let (d, dmin) = (half(bytes), half(&bytes[2..]));
            let scales = &bytes[4..16];
            let (qh, qs) = if five {
                (&bytes[16..48], &bytes[48..176])
            } else {
                (&bytes[0..0], &bytes[16..144])
            };
            for sub in 0..8 {
                let (sc, m) = k_scale_min(sub, scales);
                a.group(sub, d * sc as f32, dmin * m as f32);
                let run = &qs[(sub / 2) * 32..][..32];
                let high = sub % 2 == 1;
                let bit = 1u8 << sub;
                for (l, &q) in run.iter().enumerate() {
                    let mut code = if high { q >> 4 } else { q & 0x0F };
                    if five && qh[l] & bit != 0 {
                        code += 16;
                    }
                    a.codes[sub * 32 + l] = code;
                }
            }
        }
        // Sixteen 16-weight sub-blocks with int8 scales off a shared `d`. The
        // 6-bit weights are a low nibble in `ql` and two high bits in `qh`,
        // interleaved in runs of 32 across each half block.
        GgmlKind::Q6K => {
            let (ql, qh) = (&bytes[0..128], &bytes[128..192]);
            let scales = &bytes[192..208];
            let d = half(&bytes[208..]);
            for (n, (ql, qh)) in ql.chunks_exact(64).zip(qh.chunks_exact(32)).enumerate() {
                for l in 0..32 {
                    let (lo, hi, h) = (ql[l], ql[l + 32], qh[l]);
                    let at = n * 128 + l;
                    let codes = [
                        (lo & 0x0F) | ((h & 0x03) << 4),
                        (hi & 0x0F) | (((h >> 2) & 0x03) << 4),
                        (lo >> 4) | (((h >> 4) & 0x03) << 4),
                        (hi >> 4) | (((h >> 6) & 0x03) << 4),
                    ];
                    for (k, code) in codes.into_iter().enumerate() {
                        a.codes[at + k * 32] = code;
                    }
                }
            }
            for (g, &sc) in scales.iter().enumerate() {
                let scale = d * sc as i8 as f32;
                a.group(g, scale, 32.0 * scale);
            }
        }
    }
    a
}

/// One weight out of a whole tensor's blocks.
///
/// Decodes the block it lands in and takes one value, so reading a run of
/// weights this way re-decodes the block for each. [`row_dot`] is the way to
/// read a run; the kernels do the same on the device.
pub fn get(kind: GgmlKind, blob: &[u8], i: usize) -> f32 {
    let elems = kind.block_elems();
    let b = i / elems;
    decode_block(kind, &blob[b * kind.block_bytes()..]).value(i - b * elems)
}

/// `sum(W[base + j] * x[j])` over a row, decoding each block once.
pub fn row_dot(kind: GgmlKind, blob: &[u8], base: usize, x: &[f32]) -> f32 {
    let elems = kind.block_elems();
    let mut sum = 0.0f32;
    let (mut at, mut done) = (base, 0usize);
    while done < x.len() {
        let b = at / elems;
        let within = at - b * elems;
        let take = (elems - within).min(x.len() - done);
        let block = decode_block(kind, &blob[b * kind.block_bytes()..]);
        for k in 0..take {
            sum += block.value(within + k) * x[done + k];
        }
        at += take;
        done += take;
    }
    sum
}

/// The same weights in dlm's own quantized layout: codes four or eight bits
/// wide, with a scale and zero per group as a pair of f32s.
///
/// Exact -- the codes and the arithmetic stay ggml's -- but larger, because that
/// pair replaces the six bits a K-quant packs its scale into: a Q4_K tensor goes
/// from 4.5 bits per weight to 6. In exchange the kernels read dlm's layout
/// faster than they decode ggml's blocks. Measured on a Qwen2.5 Q4_K_M: a third
/// more decode throughput, for 45% more weight memory. The loader takes that
/// trade only when the model has the VRAM to spare.
pub fn expand(kind: GgmlKind, blob: &[u8], elements: usize) -> crate::forward::Weights {
    use crate::forward::{QuantLayout, Weights};
    let elems = kind.block_elems();
    // Nibbles where every code fits in one, bytes otherwise.
    let four_bit = matches!(kind, GgmlKind::Q4_0 | GgmlKind::Q4_1 | GgmlKind::Q4K);
    let group = if kind == GgmlKind::Q6K { 16 } else { 32 };
    let layout = if four_bit {
        QuantLayout::int4(elements, group)
    } else {
        QuantLayout::int8(elements, group)
    };
    let mut out = vec![0u8; layout.total_bytes];
    for b in 0..elements.div_ceil(elems) {
        let a = decode_block(kind, &blob[b * kind.block_bytes()..]);
        let n = elems.min(elements - b * elems);
        for i in 0..n {
            let at = b * elems + i;
            if four_bit {
                out[at / 2] |= if at % 2 == 0 {
                    a.codes[i] & 0x0F
                } else {
                    a.codes[i] << 4
                };
            } else {
                out[at] = a.codes[i];
            }
        }
        for g in 0..n.div_ceil(group) {
            let at = (b * elems) / group + g;
            out[layout.scales_off + at * 4..][..4].copy_from_slice(&a.scales[g].to_le_bytes());
            out[layout.zeros_off + at * 4..][..4].copy_from_slice(&a.zeros[g].to_le_bytes());
        }
    }
    let (blob, group_size, num_elements) = (out, group, elements);
    if four_bit {
        Weights::Int4 {
            blob,
            group_size,
            num_elements,
        }
    } else {
        Weights::Int8 {
            blob,
            group_size,
            num_elements,
        }
    }
}

/// Decode `elements` weights to plain f32, for the tensors dlm keeps
/// unquantized (the embedding matrix, the norms).
pub fn to_f32(dtype: Dtype, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let kind = GgmlKind::from_dtype(dtype).ok_or_else(|| {
        DlmError::UnsupportedQuant(format!("{dtype:?} is not a block-quantized ggml type"))
    })?;
    let need = kind.byte_len(elements);
    if bytes.len() < need {
        return Err(DlmError::SafetensorsHeader(format!(
            "gguf: a {dtype:?} tensor of {elements} weights needs {need} bytes, got {}",
            bytes.len()
        )));
    }
    let elems = kind.block_elems();
    let mut out = vec![0.0f32; elements.div_ceil(elems) * elems];
    for (b, chunk) in out.chunks_exact_mut(elems).enumerate() {
        let a = decode_block(kind, &bytes[b * kind.block_bytes()..]);
        for (i, slot) in chunk.iter_mut().enumerate() {
            *slot = a.value(i);
        }
    }
    out.truncate(elements);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 bits for a small exact value, so the tests compare exact numbers.
    fn h(v: f32) -> [u8; 2] {
        crate::storage::safetensors::f32_to_f16(v).to_le_bytes()
    }

    /// Q4_0 stores weight j and weight j+16 in one byte, and decodes as
    /// `(code - 8) * d`. Getting the nibble halves backwards would swap the two
    /// halves of every block — a wrong model that still produces fluent text.
    #[test]
    fn q4_0_decodes_both_nibble_halves() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(2.0));
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

    /// Q8_0's codes are signed.
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
        block.extend_from_slice(&0x0001_0001u32.to_le_bytes());
        block.push(0x21); // weight 0 code 1, weight 16 code 2
        block.resize(22, 0);
        let got = to_f32(Dtype::Q5_0, &block, 32).unwrap();
        assert_eq!(got[0], (1.0 + 16.0) - 16.0);
        assert_eq!(got[16], (2.0 + 16.0) - 16.0);
        assert_eq!(got[1], -16.0);
    }

    /// Q4_1 carries a minimum rather than a fixed offset.
    #[test]
    fn q4_1_uses_its_minimum() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(0.5));
        block.extend_from_slice(&h(-3.0));
        block.push(0x06);
        block.resize(20, 0);
        let got = to_f32(Dtype::Q4_1, &block, 32).unwrap();
        assert!((got[0] - (0.5 * 6.0 - 3.0)).abs() < 1e-5, "{}", got[0]);
        assert!((got[1] - -3.0).abs() < 1e-5, "{}", got[1]);
    }

    /// The 6-bit scale/minimum packing is the part of K-quants most easily got
    /// wrong, and it is shared by Q4_K and Q5_K.
    #[test]
    fn k_quant_scale_packing_matches_ggml() {
        let mut scales = [0u8; 12];
        scales[0] = 17;
        scales[4] = 33;
        scales[1] = 0b1100_0000 | 5;
        assert_eq!(k_scale_min(0, &scales), (17, 33));
        assert_eq!(k_scale_min(1, &scales), (5, 0));
        scales[8] = 0x0A;
        assert_eq!(k_scale_min(4, &scales).0, 0x0A);
        scales[9] = 0x02;
        assert_eq!(k_scale_min(5, &scales).0, 0x02 | (3 << 4));
    }

    /// Q4_K's weights are `d * sc * q - dmin * m`.
    #[test]
    fn q4_k_matches_its_defining_formula() {
        let mut block = Vec::new();
        block.extend_from_slice(&h(0.25)); // d
        block.extend_from_slice(&h(0.5)); // dmin
        let mut scales = [0u8; 12];
        scales[0] = 3;
        scales[4] = 7;
        block.extend_from_slice(&scales);
        let mut qs = vec![0u8; 128];
        qs[0] = 0x09;
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
        block[0] = 0x05;
        block[128] = 0b0000_0011;
        block[192] = -2i8 as u8;
        block[208..210].copy_from_slice(&h(0.5));
        let got = to_f32(Dtype::Q6K, &block, 256).unwrap();
        let want = 0.5 * -2.0 * (53.0 - 32.0);
        assert!((got[0] - want).abs() < 1e-5, "{} vs {want}", got[0]);
    }

    /// Expanding into dlm's layout must not change a single weight: it is the
    /// same codes and the same arithmetic, packed differently. A model is loaded
    /// one way or the other depending on what fits, so the two have to agree.
    #[test]
    fn expanding_keeps_every_weight() {
        for (kind, dtype) in [
            (GgmlKind::Q4_0, Dtype::Q4_0),
            (GgmlKind::Q4_1, Dtype::Q4_1),
            (GgmlKind::Q5_0, Dtype::Q5_0),
            (GgmlKind::Q5_1, Dtype::Q5_1),
            (GgmlKind::Q8_0, Dtype::Q8_0),
            (GgmlKind::Q4K, Dtype::Q4K),
            (GgmlKind::Q5K, Dtype::Q5K),
            (GgmlKind::Q6K, Dtype::Q6K),
        ] {
            // Two blocks of pseudo-random bytes: every field of the layout gets
            // some non-zero pattern, which is what catches a misplaced offset.
            let n = kind.block_elems() * 2;
            let blob: Vec<u8> = (0..kind.byte_len(n))
                .map(|i| (i.wrapping_mul(37).wrapping_add(11) % 251) as u8)
                .collect();
            let want = to_f32(dtype, &blob, n).unwrap();
            let got = expand(kind, &blob, n);
            assert_eq!(got.len(), n, "{kind:?}");
            for (i, &b) in want.iter().enumerate() {
                let a = got.get(i);
                let tol = 1e-4 * b.abs().max(1e-3);
                assert!(
                    (a - b).abs() <= tol,
                    "{kind:?} weight {i}: expanded {a} vs native {b}"
                );
            }
        }
    }

    /// The three ways dlm reads these weights — a whole tensor, one weight, and a
    /// row dot product — must agree, since the kernels pick between them.
    #[test]
    fn whole_tensor_single_weight_and_row_dot_agree() {
        let mut blob = Vec::new();
        for b in 0..3u8 {
            blob.extend_from_slice(&h(0.25 * (b + 1) as f32));
            blob.extend_from_slice(&h(0.5));
            let mut scales = [0u8; 12];
            for (i, s) in scales.iter_mut().enumerate() {
                *s = ((i as u8 + b) * 5 + 1) & 63;
            }
            blob.extend_from_slice(&scales);
            blob.extend((0..128u8).map(|i| i.wrapping_mul(7).wrapping_add(b)));
        }
        let all = to_f32(Dtype::Q4K, &blob, 768).unwrap();
        for i in [0, 1, 31, 32, 255, 256, 300, 767] {
            let one = get(GgmlKind::Q4K, &blob, i);
            assert_eq!(one, all[i], "weight {i}");
        }
        // A row that starts mid-block and crosses into the next one.
        let x: Vec<f32> = (0..300).map(|j| (j % 7) as f32 - 3.0).collect();
        let want: f32 = all[200..500].iter().zip(&x).map(|(w, v)| w * v).sum();
        let got = row_dot(GgmlKind::Q4K, &blob, 200, &x);
        assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    }
}

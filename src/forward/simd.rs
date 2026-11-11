//! SIMD row dot products for the CPU matvec, one per weight dtype.
//!
//! The scalar kernels in `cpu.rs` stay the reference. They sum an f32 iterator,
//! which LLVM may not reorder, so they run one multiply-add at a time and decode
//! every quantized weight on its own. These compute the same sums eight
//! (AVX2) or four (NEON) lanes at a time, decoding 32 weights per step, so they
//! agree with the reference to f32 rounding rather than bit for bit.
//!
//! Activations stay f32 for every dtype. llama.cpp quantizes the activation
//! vector to int8 and dots integers, which is several times faster for 4-bit
//! weights; but this path is the oracle the GPU kernels are checked against
//! (`tests/gpu_parity.rs`, 2e-3 on the hidden state), and the GPU keeps f32
//! activations, so an int8 activation's ~0.4% error per dot would be measured as
//! a GPU bug.
//!
//! x86_64 takes AVX2 + FMA + F16C, chosen at run time so the same binary still
//! runs (on the scalar reference) on a CPU without them; aarch64 always has NEON.
//! Each format's decoding is written once, against the [`Simd`] operations, and
//! the two instruction sets implement those.

use super::cpu::{bf16_to_f32, QuantLayout, Weights};
use crate::storage::f16_to_f32;
use crate::storage::ggml_quant::{decode_block, k_scale_min, GgmlKind};

/// The SIMD instruction set this CPU offers, if any.
#[derive(Debug, Clone, Copy)]
enum Isa {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Detected per call: std caches the CPUID result, so this is a load and a test.
fn isa() -> Option<Isa> {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        avx2.then_some(Isa::Avx2)
    }
    #[cfg(target_arch = "aarch64")]
    {
        Some(Isa::Neon)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        None
    }
}

/// Row dot products against one weight matrix, when this CPU and the matrix's
/// layout have a SIMD kernel.
pub(crate) struct RowDot<'a> {
    w: &'a Weights,
    isa: Isa,
}

impl<'a> RowDot<'a> {
    /// `None` sends the caller to the scalar reference: no SIMD on this CPU, or
    /// a layout these kernels do not take (ggml rows that are not whole blocks,
    /// a blob shorter than its element count says).
    pub(crate) fn new(w: &'a Weights, in_dim: usize) -> Option<Self> {
        let ok = match w {
            Weights::F32(_) | Weights::Bf16(_) | Weights::F16(_) => true,
            Weights::Int4 {
                blob,
                group_size,
                num_elements,
            } => {
                *group_size > 0
                    && blob.len() >= QuantLayout::int4(*num_elements, *group_size).total_bytes
            }
            Weights::Int8 {
                blob,
                group_size,
                num_elements,
            } => {
                *group_size > 0
                    && blob.len() >= QuantLayout::int8(*num_elements, *group_size).total_bytes
            }
            Weights::Ggml {
                blob,
                kind,
                num_elements,
            } => in_dim % kind.block_elems() == 0 && blob.len() >= kind.byte_len(*num_elements),
        };
        if !ok {
            return None;
        }
        Some(Self { w, isa: isa()? })
    }

    /// `sum(W[base + j] * x[j])` over `x.len()` weights from element `base`.
    pub(crate) fn row(&self, base: usize, x: &[f32]) -> f32 {
        // The quantized kernels read codes through raw pointers; this and the
        // blob length `new` checked keep those reads inside the blob.
        assert!(base + x.len() <= self.w.len(), "row past the matrix");
        // SAFETY: `isa` was detected on this CPU, and the row is in bounds.
        match self.isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe { avx2::row(self.w, base, x) },
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => unsafe { row(neon::Neon, self.w, base, x) },
        }
    }
}

/// `sum(w[j] * x[j])` over two f32 slices, SIMD where the CPU has it.
pub(crate) fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: AVX2/FMA were detected on this CPU.
        Some(Isa::Avx2) => unsafe { avx2::dot_f32(w, x) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        Some(Isa::Neon) => unsafe { dense_f32(neon::Neon, w, x) },
        #[allow(unreachable_patterns)]
        _ => w.iter().zip(x).map(|(&a, &b)| a * b).sum(),
    }
}

/// The operations the kernels are written in. `F` is a vector of f32 lanes;
/// `B` is 32 bytes of weight codes. Every method is `#[inline(always)]` so it
/// lands inside the `#[target_feature]` entry point that calls it, where the
/// intrinsics it uses are enabled.
trait Simd: Copy {
    type F: Copy;
    type B: Copy;
    const LANES: usize;

    unsafe fn zero(self) -> Self::F;
    unsafe fn load_f32(self, p: *const f32) -> Self::F;
    unsafe fn load_bf16(self, p: *const u16) -> Self::F;
    unsafe fn load_f16(self, p: *const u16) -> Self::F;
    /// `acc + a * b`.
    unsafe fn fma(self, acc: Self::F, a: Self::F, b: Self::F) -> Self::F;
    unsafe fn hsum(self, acc: [Self::F; 4]) -> f32;

    /// 32 bytes as 32 codes.
    unsafe fn bytes(self, p: *const u8) -> Self::B;
    /// 16 bytes as 32 4-bit codes, the low nibbles first, then the high: ggml's
    /// Q4_0/Q4_1/Q5 order.
    unsafe fn nibbles_split(self, p: *const u8) -> Self::B;
    /// 16 bytes as 32 4-bit codes in element order, low nibble first: dlm int4.
    unsafe fn nibbles_interleaved(self, p: *const u8) -> Self::B;
    /// `(b >> n) & mask` in each byte. `mask` must fit in `8 - n` bits.
    unsafe fn shr_and(self, b: Self::B, n: u32, mask: u8) -> Self::B;
    /// `a | (b << 4)` in each byte, every byte of `b` below 16.
    unsafe fn or_shl4(self, a: Self::B, b: Self::B) -> Self::B;
    unsafe fn xor(self, b: Self::B, m: u8) -> Self::B;
    /// Byte `j` is bit `j` of `bits`, as 0 or 1.
    unsafe fn bits(self, bits: u32) -> Self::B;
    /// `acc += (code * s - o) * x` over 32 codes and `x[..32]`: the first 16
    /// with `s[0]`/`o[0]`, the last 16 with `s[1]`/`o[1]`.
    unsafe fn fma_codes(
        self,
        acc: &mut [Self::F; 4],
        codes: Self::B,
        s: [f32; 2],
        o: [f32; 2],
        x: *const f32,
    );
}

/// The dense dot product over any element type `load` reads, with the tail
/// that does not fill a vector done by `scalar`.
macro_rules! dense {
    ($s:expr, $w:expr, $x:expr, $load:ident, $scalar:expr) => {{
        let (s, w, x) = ($s, $w, $x);
        let n = w.len().min(x.len());
        let l = S::LANES;
        let mut acc = [s.zero(); 4];
        let mut j = 0;
        while j + 4 * l <= n {
            for (k, a) in acc.iter_mut().enumerate() {
                let at = j + k * l;
                *a = s.fma(
                    *a,
                    s.$load(w.as_ptr().add(at)),
                    s.load_f32(x.as_ptr().add(at)),
                );
            }
            j += 4 * l;
        }
        while j + l <= n {
            acc[0] = s.fma(
                acc[0],
                s.$load(w.as_ptr().add(j)),
                s.load_f32(x.as_ptr().add(j)),
            );
            j += l;
        }
        let tail: f32 = w[j..n]
            .iter()
            .zip(&x[j..n])
            .map(|(&a, &b)| $scalar(a) * b)
            .sum();
        s.hsum(acc) + tail
    }};
}

#[inline(always)]
unsafe fn dense_f32<S: Simd>(s: S, w: &[f32], x: &[f32]) -> f32 {
    dense!(s, w, x, load_f32, |v: f32| v)
}

/// One row of any dtype. Inlined whole into each instruction set's entry point.
#[inline(always)]
unsafe fn row<S: Simd>(s: S, w: &Weights, base: usize, x: &[f32]) -> f32 {
    let n = x.len();
    match w {
        Weights::F32(v) => dense_f32(s, &v[base..base + n], x),
        Weights::Bf16(v) => dense!(s, &v[base..base + n], x, load_bf16, bf16_to_f32),
        Weights::F16(v) => dense!(s, &v[base..base + n], x, load_f16, f16_to_f32),
        Weights::Int4 {
            blob,
            group_size,
            num_elements,
        } => affine_row(
            s,
            blob,
            QuantLayout::int4(*num_elements, *group_size),
            *group_size,
            true,
            base,
            x,
        ),
        Weights::Int8 {
            blob,
            group_size,
            num_elements,
        } => affine_row(
            s,
            blob,
            QuantLayout::int8(*num_elements, *group_size),
            *group_size,
            false,
            base,
            x,
        ),
        Weights::Ggml { blob, kind, .. } => ggml_row(s, *kind, blob, base / kind.block_elems(), x),
    }
}

fn f32_at(blob: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(blob[off..off + 4].try_into().unwrap())
}

/// A dlm int4/int8 row: `(code - zero) * scale` with a scale and zero per
/// `group` weights, groups running on across rows.
///
/// Every real layout has groups a multiple of 16 that each row starts on a
/// multiple of 16 of (dlm's 128, GPTQ's 32 to 128, expanded ggml's 16 and 32),
/// so every 16 weights share one scale and a 32-weight step needs at most two:
/// that is the fast loop, which divides once per row. Anything else (a group of
/// 100) and the last few weights take the general loop, one group at a time.
#[inline(always)]
unsafe fn affine_row<S: Simd>(
    s: S,
    blob: &[u8],
    layout: QuantLayout,
    group: usize,
    four: bool,
    base: usize,
    x: &[f32],
) -> f32 {
    let at = |g: usize| {
        let scale = f32_at(blob, layout.scales_off + g * 4);
        (scale, f32_at(blob, layout.zeros_off + g * 4) * scale)
    };
    let mut acc = [s.zero(); 4];
    let mut j = 0;
    if group % 16 == 0 && base % 16 == 0 {
        let (mut g, mut left) = (base / group, group - base % group);
        let mut cur = (g, at(g));
        while x.len() - j >= 32 {
            let mut halves = [(0.0f32, 0.0f32); 2];
            for half in &mut halves {
                if cur.0 != g {
                    cur = (g, at(g));
                }
                *half = cur.1;
                left -= 16;
                if left == 0 {
                    (g, left) = (g + 1, group);
                }
            }
            let (sc, o) = ([halves[0].0, halves[1].0], [halves[0].1, halves[1].1]);
            let c = affine_codes(s, blob, four, base + j);
            s.fma_codes(&mut acc, c, sc, o, x.as_ptr().add(j));
            j += 32;
        }
    }
    let mut tail = 0.0f32;
    while j < x.len() {
        let g = (base + j) / group;
        let end = ((g + 1) * group - base).min(x.len());
        let (scale, offset) = at(g);
        let zero = f32_at(blob, layout.zeros_off + g * 4);
        while j < end {
            let i = base + j;
            if end - j >= 32 && !(four && i % 2 == 1) {
                let c = affine_codes(s, blob, four, i);
                s.fma_codes(&mut acc, c, [scale; 2], [offset; 2], x.as_ptr().add(j));
                j += 32;
            } else {
                let code = if four {
                    (blob[i / 2] >> (4 * (i % 2))) & 0x0F
                } else {
                    blob[i]
                };
                tail += (code as f32 - zero) * scale * x[j];
                j += 1;
            }
        }
    }
    s.hsum(acc) + tail
}

/// The 32 codes from element `i` of an int4 (even `i`) or int8 blob.
#[inline(always)]
unsafe fn affine_codes<S: Simd>(s: S, blob: &[u8], four: bool, i: usize) -> S::B {
    if four {
        s.nibbles_interleaved(blob.as_ptr().add(i / 2))
    } else {
        s.bytes(blob.as_ptr().add(i))
    }
}

fn half_at(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// A row of whole ggml blocks from block `first`, each decoded 32 weights at a
/// time as `code * scale - offset`. The layouts are `decode_block`'s.
#[inline(always)]
unsafe fn ggml_row<S: Simd>(s: S, kind: GgmlKind, blob: &[u8], first: usize, x: &[f32]) -> f32 {
    let (bb, be) = (kind.block_bytes(), kind.block_elems());
    let mut acc = [s.zero(); 4];
    for (i, xb) in x.chunks_exact(be).enumerate() {
        let b = &blob[(first + i) * bb..][..bb];
        let (p, xp) = (b.as_ptr(), xb.as_ptr());
        match kind {
            GgmlKind::Q4_0 => {
                let d = half_at(b, 0);
                s.fma_codes(
                    &mut acc,
                    s.nibbles_split(p.add(2)),
                    [d; 2],
                    [8.0 * d; 2],
                    xp,
                );
            }
            GgmlKind::Q4_1 => {
                let (d, m) = (half_at(b, 0), half_at(b, 2));
                s.fma_codes(&mut acc, s.nibbles_split(p.add(4)), [d; 2], [-m; 2], xp);
            }
            GgmlKind::Q5_0 | GgmlKind::Q5_1 => {
                let one = kind == GgmlKind::Q5_1;
                let d = half_at(b, 0);
                let (offset, qh) = if one {
                    (-half_at(b, 2), 4)
                } else {
                    (16.0 * d, 2)
                };
                let high = s.bits(u32::from_le_bytes(b[qh..qh + 4].try_into().unwrap()));
                let codes = s.or_shl4(s.nibbles_split(p.add(qh + 4)), high);
                s.fma_codes(&mut acc, codes, [d; 2], [offset; 2], xp);
            }
            // Signed codes, made unsigned by flipping the sign bit: `q + 128`
            // with an offset of `128 * d`.
            GgmlKind::Q8_0 => {
                let d = half_at(b, 0);
                let codes = s.xor(s.bytes(p.add(2)), 0x80);
                s.fma_codes(&mut acc, codes, [d; 2], [128.0 * d; 2], xp);
            }
            GgmlKind::Q4K | GgmlKind::Q5K => {
                let five = kind == GgmlKind::Q5K;
                let (d, dmin) = (half_at(b, 0), half_at(b, 2));
                let qs = if five { 48 } else { 16 };
                let qh = s.bytes(p.add(16)); // Q4_K: unused, and inside the block
                for pair in 0..4 {
                    let q = s.bytes(p.add(qs + pair * 32));
                    for half in 0..2 {
                        let sub = pair * 2 + half;
                        let mut codes = s.shr_and(q, 4 * half as u32, 0x0F);
                        if five {
                            codes = s.or_shl4(codes, s.shr_and(qh, sub as u32, 1));
                        }
                        let (sc, m) = k_scale_min(sub, &b[4..16]);
                        let (sc, m) = (d * sc as f32, dmin * m as f32);
                        s.fma_codes(&mut acc, codes, [sc; 2], [m; 2], xp.add(sub * 32));
                    }
                }
            }
            // Per half block: 32 bytes of low nibbles twice over, and 32 bytes of
            // high bits, four 2-bit fields each; one scale per 16 weights.
            GgmlKind::Q6K => {
                let d = half_at(b, 208);
                for n in 0..2 {
                    let ql = [s.bytes(p.add(n * 64)), s.bytes(p.add(n * 64 + 32))];
                    let qh = s.bytes(p.add(128 + n * 32));
                    for k in 0..4 {
                        let low = s.shr_and(ql[k % 2], 4 * (k / 2) as u32, 0x0F);
                        let codes = s.or_shl4(low, s.shr_and(qh, 2 * k as u32, 3));
                        let g = n * 8 + k * 2;
                        let sc = [d * b[192 + g] as i8 as f32, d * b[193 + g] as i8 as f32];
                        let o = [32.0 * sc[0], 32.0 * sc[1]];
                        s.fma_codes(&mut acc, codes, sc, o, xp.add(n * 128 + k * 32));
                    }
                }
            }
            // Through the scalar decode: these formats are for VRAM, not the CPU.
            _ => {
                let a = decode_block(kind, b);
                for c in 0..be / 32 {
                    let at = c * 32;
                    let (g0, g1) = (at / a.group, (at + 16) / a.group);
                    let sc = [a.scales[g0], a.scales[g1]];
                    let o = [a.offsets[g0], a.offsets[g1]];
                    let codes = s.bytes(a.codes.as_ptr().add(at));
                    s.fma_codes(&mut acc, codes, sc, o, xp.add(at));
                }
            }
        }
    }
    s.hsum(acc)
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Simd, Weights};
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub(super) struct Avx2;

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn row(w: &Weights, base: usize, x: &[f32]) -> f32 {
        super::row(Avx2, w, base, x)
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn dot_f32(w: &[f32], x: &[f32]) -> f32 {
        super::dense_f32(Avx2, w, x)
    }

    impl Simd for Avx2 {
        type F = __m256;
        type B = __m256i;
        const LANES: usize = 8;

        #[inline(always)]
        unsafe fn zero(self) -> __m256 {
            _mm256_setzero_ps()
        }
        #[inline(always)]
        unsafe fn load_f32(self, p: *const f32) -> __m256 {
            _mm256_loadu_ps(p)
        }
        #[inline(always)]
        unsafe fn load_bf16(self, p: *const u16) -> __m256 {
            let h = _mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i));
            _mm256_castsi256_ps(_mm256_slli_epi32(h, 16))
        }
        #[inline(always)]
        unsafe fn load_f16(self, p: *const u16) -> __m256 {
            _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
        }
        #[inline(always)]
        unsafe fn fma(self, acc: __m256, a: __m256, b: __m256) -> __m256 {
            _mm256_fmadd_ps(a, b, acc)
        }
        #[inline(always)]
        unsafe fn hsum(self, acc: [__m256; 4]) -> f32 {
            let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
            let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
        }

        #[inline(always)]
        unsafe fn bytes(self, p: *const u8) -> __m256i {
            _mm256_loadu_si256(p as *const __m256i)
        }
        #[inline(always)]
        unsafe fn nibbles_split(self, p: *const u8) -> __m256i {
            let q = _mm_loadu_si128(p as *const __m128i);
            let m = _mm_set1_epi8(0x0F);
            let (lo, hi) = (_mm_and_si128(q, m), _mm_and_si128(_mm_srli_epi16(q, 4), m));
            _mm256_set_m128i(hi, lo)
        }
        #[inline(always)]
        unsafe fn nibbles_interleaved(self, p: *const u8) -> __m256i {
            let q = _mm_loadu_si128(p as *const __m128i);
            let m = _mm_set1_epi8(0x0F);
            let (lo, hi) = (_mm_and_si128(q, m), _mm_and_si128(_mm_srli_epi16(q, 4), m));
            _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi))
        }
        #[inline(always)]
        unsafe fn shr_and(self, b: __m256i, n: u32, mask: u8) -> __m256i {
            // A 16-bit shift carries the high byte's low bits into the low
            // byte's top bits; `mask` fitting in `8 - n` bits drops them.
            let shifted = _mm256_srl_epi16(b, _mm_cvtsi32_si128(n as i32));
            _mm256_and_si256(shifted, _mm256_set1_epi8(mask as i8))
        }
        #[inline(always)]
        unsafe fn or_shl4(self, a: __m256i, b: __m256i) -> __m256i {
            _mm256_or_si256(a, _mm256_slli_epi16(b, 4))
        }
        #[inline(always)]
        unsafe fn xor(self, b: __m256i, m: u8) -> __m256i {
            _mm256_xor_si256(b, _mm256_set1_epi8(m as i8))
        }
        #[inline(always)]
        unsafe fn bits(self, bits: u32) -> __m256i {
            // Byte j takes byte j / 8 of `bits`, ORed with every bit but j % 8
            // set: all ones exactly when bit j was. (llama.cpp's
            // `bytes_from_bits_32`.)
            let spread = _mm256_shuffle_epi8(
                _mm256_set1_epi32(bits as i32),
                _mm256_setr_epi64x(
                    0x0000_0000_0000_0000,
                    0x0101_0101_0101_0101,
                    0x0202_0202_0202_0202,
                    0x0303_0303_0303_0303,
                ),
            );
            let others = _mm256_or_si256(spread, _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe));
            let set = _mm256_cmpeq_epi8(others, _mm256_set1_epi64x(-1));
            _mm256_and_si256(set, _mm256_set1_epi8(1))
        }
        #[inline(always)]
        unsafe fn fma_codes(
            self,
            acc: &mut [__m256; 4],
            codes: __m256i,
            s: [f32; 2],
            o: [f32; 2],
            x: *const f32,
        ) {
            let (lo, hi) = (
                _mm256_castsi256_si128(codes),
                _mm256_extracti128_si256(codes, 1),
            );
            let eights = [lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)];
            for (k, (a, e)) in acc.iter_mut().zip(eights).enumerate() {
                let c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(e));
                let w = _mm256_fmsub_ps(c, _mm256_set1_ps(s[k / 2]), _mm256_set1_ps(o[k / 2]));
                *a = _mm256_fmadd_ps(w, _mm256_loadu_ps(x.add(k * 8)), *a);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Simd;
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(super) struct Neon;

    impl Simd for Neon {
        type F = float32x4_t;
        type B = (uint8x16_t, uint8x16_t);
        const LANES: usize = 4;

        #[inline(always)]
        unsafe fn zero(self) -> float32x4_t {
            vdupq_n_f32(0.0)
        }
        #[inline(always)]
        unsafe fn load_f32(self, p: *const f32) -> float32x4_t {
            vld1q_f32(p)
        }
        #[inline(always)]
        unsafe fn load_bf16(self, p: *const u16) -> float32x4_t {
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(p)))
        }
        /// f16 by bit arithmetic (the f16 conversion intrinsics are not stable):
        /// the exponent and mantissa shifted into place and rescaled by 2^112,
        /// which also normalizes subnormals; infinities and NaNs re-flagged.
        #[inline(always)]
        unsafe fn load_f16(self, p: *const u16) -> float32x4_t {
            let h = vmovl_u16(vld1_u16(p));
            let sign = vshlq_n_u32::<16>(vandq_u32(h, vdupq_n_u32(0x8000)));
            let em = vshlq_n_u32::<13>(vandq_u32(h, vdupq_n_u32(0x7FFF)));
            let scaled = vmulq_f32(
                vreinterpretq_f32_u32(em),
                vdupq_n_f32(f32::from_bits(0x7780_0000)),
            );
            let special = vcgeq_u32(em, vdupq_n_u32(0x7C00 << 13));
            let infnan = vorrq_u32(em, vdupq_n_u32(0x7F80_0000));
            let bits = vbslq_u32(special, infnan, vreinterpretq_u32_f32(scaled));
            vreinterpretq_f32_u32(vorrq_u32(bits, sign))
        }
        #[inline(always)]
        unsafe fn fma(self, acc: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vfmaq_f32(acc, a, b)
        }
        #[inline(always)]
        unsafe fn hsum(self, acc: [float32x4_t; 4]) -> f32 {
            vaddvq_f32(vaddq_f32(
                vaddq_f32(acc[0], acc[1]),
                vaddq_f32(acc[2], acc[3]),
            ))
        }

        #[inline(always)]
        unsafe fn bytes(self, p: *const u8) -> Self::B {
            (vld1q_u8(p), vld1q_u8(p.add(16)))
        }
        #[inline(always)]
        unsafe fn nibbles_split(self, p: *const u8) -> Self::B {
            let q = vld1q_u8(p);
            (vandq_u8(q, vdupq_n_u8(0x0F)), vshrq_n_u8::<4>(q))
        }
        #[inline(always)]
        unsafe fn nibbles_interleaved(self, p: *const u8) -> Self::B {
            let (lo, hi) = self.nibbles_split(p);
            (vzip1q_u8(lo, hi), vzip2q_u8(lo, hi))
        }
        #[inline(always)]
        unsafe fn shr_and(self, b: Self::B, n: u32, mask: u8) -> Self::B {
            let (by, m) = (vdupq_n_s8(-(n as i8)), vdupq_n_u8(mask));
            (
                vandq_u8(vshlq_u8(b.0, by), m),
                vandq_u8(vshlq_u8(b.1, by), m),
            )
        }
        #[inline(always)]
        unsafe fn or_shl4(self, a: Self::B, b: Self::B) -> Self::B {
            (
                vorrq_u8(a.0, vshlq_n_u8::<4>(b.0)),
                vorrq_u8(a.1, vshlq_n_u8::<4>(b.1)),
            )
        }
        #[inline(always)]
        unsafe fn xor(self, b: Self::B, m: u8) -> Self::B {
            (veorq_u8(b.0, vdupq_n_u8(m)), veorq_u8(b.1, vdupq_n_u8(m)))
        }
        #[inline(always)]
        unsafe fn bits(self, bits: u32) -> Self::B {
            let b = vreinterpret_u8_u32(vdup_n_u32(bits));
            let which =
                vld1q_u8([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr());
            let one = vdupq_n_u8(1);
            let spread =
                |lo: uint8x8_t, hi: uint8x8_t| vandq_u8(vtstq_u8(vcombine_u8(lo, hi), which), one);
            (
                spread(vdup_lane_u8::<0>(b), vdup_lane_u8::<1>(b)),
                spread(vdup_lane_u8::<2>(b), vdup_lane_u8::<3>(b)),
            )
        }
        #[inline(always)]
        unsafe fn fma_codes(
            self,
            acc: &mut [float32x4_t; 4],
            codes: Self::B,
            s: [f32; 2],
            o: [f32; 2],
            x: *const f32,
        ) {
            for (h, v) in [codes.0, codes.1].into_iter().enumerate() {
                let (lo, hi) = (vmovl_u8(vget_low_u8(v)), vmovl_high_u8(v));
                let fours = [
                    vmovl_u16(vget_low_u16(lo)),
                    vmovl_high_u16(lo),
                    vmovl_u16(vget_low_u16(hi)),
                    vmovl_high_u16(hi),
                ];
                let neg_o = vdupq_n_f32(-o[h]);
                for (k, (a, c)) in acc.iter_mut().zip(fours).enumerate() {
                    let w = vfmaq_n_f32(neg_o, vcvtq_f32_u32(c), s[h]);
                    *a = vfmaq_f32(*a, w, vld1q_f32(x.add(h * 16 + k * 4)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::cpu::{matvec_native, matvec_native_scalar};
    use crate::storage::safetensors::f32_to_f16;

    /// xorshift64*: deterministic, and enough to give every bit of a block a
    /// non-trivial pattern.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform in [-1, 1).
        fn f32(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f32()).collect()
        }
    }

    /// Compare the SIMD matvec with the scalar reference row by row.
    ///
    /// Both sum the same `n` products in different orders, and the quantized
    /// kernels form each weight as `code * scale - offset` where the reference
    /// takes `(code - zero) * scale`: a few roundings per term on numbers no
    /// larger than the row's largest weight times `|x_j|`. Each sum's error is
    /// then within `n * eps` of `max|w| * sum|x|` (the standard bound for
    /// recursive summation), so twice that, with headroom for the per-term
    /// roundings, bounds their difference. A decoding mistake moves the result
    /// by around `max|w| * sum|x| / sqrt(n)` — over thirty times this.
    fn assert_matches_scalar(w: &Weights, out_dim: usize, in_dim: usize, what: &str) {
        if isa().is_some() {
            assert!(RowDot::new(w, in_dim).is_some(), "{what}: no SIMD kernel");
        }
        let x = Rng(0x5EED ^ in_dim as u64).vec(in_dim);
        let got = matvec_native(w, &x, out_dim, in_dim);
        let want = matvec_native_scalar(w, &x, out_dim, in_dim);
        let sum_x: f32 = x.iter().map(|v| v.abs()).sum();
        for o in 0..out_dim {
            let max_w = (0..in_dim)
                .map(|j| w.get(o * in_dim + j).abs())
                .fold(0.0f32, f32::max);
            let tol = 4.0 * in_dim as f32 * f32::EPSILON * max_w * sum_x + 1e-30;
            assert!(
                (got[o] - want[o]).abs() <= tol,
                "{what} row {o}: simd {} vs scalar {} (tol {tol:e})",
                got[o],
                want[o]
            );
        }
    }

    #[test]
    fn float_dtypes_match_the_scalar_reference() {
        let mut rng = Rng(1);
        // 1003 leaves a tail past both the 32- and 8-wide loops.
        for in_dim in [896, 1003, 5] {
            let (out_dim, n) = (6, 6 * in_dim);
            let v = rng.vec(n);
            assert_matches_scalar(&Weights::F32(v.clone()), out_dim, in_dim, "f32");
            let bf16 = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
            assert_matches_scalar(&Weights::Bf16(bf16), out_dim, in_dim, "bf16");
            // Down to f16 subnormals, which the NEON decode rescales.
            let f16 = v
                .iter()
                .enumerate()
                .map(|(i, x)| f32_to_f16(x * 1e-3f32.powi(i as i32 % 3)))
                .collect();
            assert_matches_scalar(&Weights::F16(f16), out_dim, in_dim, "f16");
        }
    }

    #[test]
    fn dlm_int4_and_int8_match_the_scalar_reference() {
        let mut rng = Rng(2);
        // 128: the shipped group, aligned with the rows. 100: groups that
        // straddle rows and are not a multiple of 16. 97 wide: int4 rows that
        // start on an odd element, mid-byte. 16 and 48: two scales in one
        // 32-weight step, and groups that cross a step.
        for (in_dim, group) in [
            (896, 128),
            (300, 100),
            (97, 32),
            (64, 16),
            (96, 16),
            (192, 48),
        ] {
            let (out_dim, n) = (5, 5 * in_dim);
            let v = rng.vec(n);
            let what = format!("in {in_dim} group {group}");
            let int4 = Weights::quantize_int4(&v, group).unwrap();
            assert_matches_scalar(&int4, out_dim, in_dim, &format!("int4 {what}"));
            let int8 = Weights::quantize_int8(&v, group).unwrap();
            assert_matches_scalar(&int8, out_dim, in_dim, &format!("int8 {what}"));
        }
    }

    #[test]
    fn ggml_blocks_match_the_scalar_reference() {
        let mut rng = Rng(3);
        for kind in [
            GgmlKind::Q4_0,
            GgmlKind::Q4_1,
            GgmlKind::Q5_0,
            GgmlKind::Q5_1,
            GgmlKind::Q8_0,
            GgmlKind::Q4K,
            GgmlKind::Q5K,
            GgmlKind::Q6K,
            GgmlKind::Q2K,
            GgmlKind::Q3K,
        ] {
            let (out_dim, in_dim) = (4, kind.block_elems() * 3);
            let n = out_dim * in_dim;
            // Random bytes everywhere, then sane f16s where the scales live so
            // no weight is infinite or NaN.
            let mut blob: Vec<u8> = (0..kind.byte_len(n)).map(|_| rng.next() as u8).collect();
            let halves: &[usize] = match kind {
                GgmlKind::Q4_0 | GgmlKind::Q5_0 | GgmlKind::Q8_0 => &[0],
                GgmlKind::Q6K => &[208],
                GgmlKind::Q2K => &[80, 82],
                GgmlKind::Q3K => &[108],
                _ => &[0, 2],
            };
            for block in blob.chunks_exact_mut(kind.block_bytes()) {
                for &at in halves {
                    block[at..at + 2].copy_from_slice(&f32_to_f16(rng.f32() * 0.1).to_le_bytes());
                }
            }
            let w = Weights::Ggml {
                blob,
                kind,
                num_elements: n,
            };
            assert_matches_scalar(&w, out_dim, in_dim, &format!("{kind:?}"));
        }
    }
}

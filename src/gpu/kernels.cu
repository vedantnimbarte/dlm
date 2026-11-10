// CUDA reference kernels for one transformer decode block.
//
// This mirrors the CPU oracle in `src/forward/cpu.rs` op-for-op so the two can
// be cross-validated on hardware. The kernels are simple (no tiling, no fusion,
// no cuBLAS) — correctness and readability over speed — with one exception: the
// GEMV is coalesced (one block per output row, shared-memory reduction), because
// it is the decode stack's dominant cost. Weights are consumed in their native
// checkpoint dtype and decoded to f32 in-register. A production version would
// fuse the elementwise ops and use cuBLAS / tensor cores.
//
// Entry point: `dlm_decode_block`, called from Rust (see src/forward/gpu.rs)
// via FFI. All pointers are device pointers. Returns a cudaError_t (0 == ok).
//
// NOTE: this file requires nvcc to compile and a GPU to run; it is compiled only
// under the `cuda-kernels` Cargo feature. Validated on device against the CPU
// oracle by tests/gpu_parity.rs.

// Backend-neutral device code: the same kernels compile under NVIDIA CUDA (nvcc)
// and AMD HIP (hipcc). HIP mirrors the CUDA runtime API almost 1:1, so we map the
// handful of `cuda*` runtime symbols this file uses to their `hip*` equivalents
// when compiling for AMD. All the __global__ kernels and device intrinsics
// (rsqrtf/expf/tanhf/__half2float/__syncthreads) are identical on both.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaGetLastError hipGetLastError
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaGetDevice hipGetDevice
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Threads per block for the reduction kernels. Must be <= 1024 (the CUDA
// per-block thread cap) and a power of two for the tree reduction below.
#define RMS_THREADS 256

// out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
//
// Launch <<<1, RMS_THREADS>>> for ANY n. The previous version launched
// <<<1, n>>>, which exceeds the 1024 threads/block limit for every real model
// (hidden_size is 2048..8192), so it failed to launch with cudaErrorInvalidValue
// and never ran on a real checkpoint. Threads now stride over n and cooperate on
// a shared-memory tree reduction for the sum of squares.
__global__ void rmsnorm_kernel(const float* x, const float* w, float* out, int n, float eps) {
    __shared__ float partial[RMS_THREADS];
    __shared__ float inv_rms;

    float ss = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) ss += x[k] * x[k];
    partial[threadIdx.x] = ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_rms = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv_rms * w[i];
}

// out[i] = (x[i] - mean(x)) * rsqrt(var(x) + eps) * w[i] + b[i]  (b may be NULL)
//
// LayerNorm (GPT-2, Falcon): RMSNorm's tree reduction twice over -- the mean
// first, then the sum of squared deviations from it -- which is `layernorm()` in
// src/forward/cpu.rs. Computing var as mean(x^2) - mean^2 in one pass would save
// a reduction but cancels catastrophically on GPT-2's residual stream, whose
// outlier dimensions dwarf the rest. `out` may alias `x`.
__global__ void layernorm_kernel(const float* x, const float* w, const float* b, float* out,
                                 int n, float eps) {
    __shared__ float partial[RMS_THREADS];
    __shared__ float mean;
    __shared__ float inv_std;

    float s = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) s += x[k];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) mean = partial[0] / (float)n;
    __syncthreads();

    float ss = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float d = x[k] - mean;
        ss += d * d;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_std = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = (x[i] - mean) * inv_std * w[i] + (b ? b[i] : 0.0f);
}

// Row-major [out_dim, in_dim] matrix times vector, plus an optional bias.
// Threads per block for the GEMV reduction; power of two for the tree reduction.
//
// Measured on a GTX 1650, decode tok/s by thread count:
//
//   threads          32     64    128    256    512   1024
//   Qwen2.5-0.5B   36.9   56.8   56.5   42.7   29.5   13.8
//   Qwen2.5-1.5B   16.0   23.3   21.6   14.4      -      -   (int4)
//   Gemma 3 4B      6.4    9.5    8.6    6.8      -      -   (int4)
//
// Each doubling adds a barrier to every row's reduction, so wide blocks pay for
// synchronization they don't win back. Narrow blocks give each thread a long
// scalar stride instead. Picking the count per matrix width (in_dim / k) lost to
// a fixed 64 on all three models.
#define MATVEC_THREADS 64

// Weight dtype tags — must match `Weights::dtype_code()` in src/forward/cpu.rs.
// Weights are uploaded in their NATIVE checkpoint dtype and decoded to f32 in the
// register that consumes them. Upsizing to f32 host-side is lossless (an f32
// exactly represents every bf16/f16 value) so it buys no precision, while doubling
// VRAM, PCIe traffic per streamed layer, and the bandwidth of this memory-bound
// GEMV. Accumulation stays f32 regardless — no hardware accumulates in 16-bit.
#define DLM_W_F32  0
#define DLM_W_BF16 1
#define DLM_W_F16  2
#define DLM_W_INT4 3
#define DLM_W_INT8 4

// Byte offsets inside a quantized blob, mirroring `QuantLayout` in
// src/forward/cpu.rs — the two MUST agree.
//   [codes][pad to 4][scales: g x f32][zeros: g x f32],  g = ceil(n/group)
// Only the code width differs: int4 packs two per byte, int8 one.
__device__ __forceinline__ long q_scales_off(long code_bytes) {
    return ((code_bytes + 3) / 4) * 4;   // f32 alignment
}
__device__ __forceinline__ long q_zeros_off(long code_bytes, long n, int group_size) {
    long groups = (n + group_size - 1) / group_size;
    return q_scales_off(code_bytes) + groups * 4;
}

// Decode weight element `i`. Specialized at compile time, so the inner loop has
// no branch. bf16 is literally the high half of an f32 — a 16-bit shift, which
// needs no hardware bf16 support (works on Turing and older).
// `n` (element count) and `group_size` are only read by the INT4 specialization,
// which needs them to find its per-group scales; the float arms ignore them.
template <int DT>
__device__ __forceinline__ float load_w(const void* W, long i, long n, int group_size);

template <>
__device__ __forceinline__ float load_w<DLM_W_F32>(const void* W, long i, long, int) {
    return ((const float*)W)[i];
}
template <>
__device__ __forceinline__ float load_w<DLM_W_BF16>(const void* W, long i, long, int) {
    return __int_as_float(((unsigned int)((const unsigned short*)W)[i]) << 16);
}
template <>
__device__ __forceinline__ float load_w<DLM_W_F16>(const void* W, long i, long, int) {
    return __half2float(((const __half*)W)[i]);
}
// dequant(code) = (code - zero) * scale, per group. Mirrors `int4_get` in
// src/forward/cpu.rs; the CPU oracle and this must decode identically.
template <>
__device__ __forceinline__ float load_w<DLM_W_INT4>(const void* W, long i, long n, int group_size) {
    const unsigned char* bytes = (const unsigned char*)W;
    unsigned char byte = bytes[i >> 1];
    float code = (float)((i & 1) ? (byte >> 4) : (byte & 0x0F));
    long g = i / group_size;
    long code_bytes = (n + 1) / 2;
    const float* scales = (const float*)(bytes + q_scales_off(code_bytes));
    const float* zeros = (const float*)(bytes + q_zeros_off(code_bytes, n, group_size));
    return (code - zeros[g]) * scales[g];
}
// int8: one code per byte. Mirrors `int8_get` in src/forward/cpu.rs.
template <>
__device__ __forceinline__ float load_w<DLM_W_INT8>(const void* W, long i, long n, int group_size) {
    const unsigned char* bytes = (const unsigned char*)W;
    float code = (float)bytes[i];
    long g = i / group_size;
    const float* scales = (const float*)(bytes + q_scales_off(n));
    const float* zeros = (const float*)(bytes + q_zeros_off(n, n, group_size));
    return (code - zeros[g]) * scales[g];
}

// ── ggml's block-quantized weights ────────────────────────────────────────
//
// A GGUF file stores weights in blocks: a fixed number of them behind one scale,
// packed tightly (a Q4_K block is 256 weights in 144 bytes). dlm keeps the blocks
// as the file has them and decodes a weight here, in the register that is about
// to multiply it, so a quantized model costs the same in VRAM as on disk.
//
// These mirror `decode_block` in src/storage/ggml_quant.rs, which is the
// definition; the two must agree exactly. Consecutive threads read consecutive
// weights, so a block's scale bytes are one cache line shared across the warp.
#define DLM_W_Q4_0 5
#define DLM_W_Q4_1 6
#define DLM_W_Q5_0 7
#define DLM_W_Q5_1 8
#define DLM_W_Q8_0 9
#define DLM_W_Q4_K 10
#define DLM_W_Q5_K 11
#define DLM_W_Q6_K 12

__device__ __forceinline__ float dlm_half(const unsigned char* p) {
    unsigned short bits = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(*(const __half*)&bits);
}

// A K-quant sub-block's 6-bit scale and minimum, out of the 12 bytes eight of
// them share (ggml's `get_scale_min_k4`).
__device__ __forceinline__ void k_scale_min(int sub, const unsigned char* s,
                                            unsigned char* sc, unsigned char* m) {
    if (sub < 4) {
        *sc = s[sub] & 63;
        *m = s[sub + 4] & 63;
    } else {
        *sc = (s[sub + 4] & 0x0F) | ((s[sub - 4] >> 6) << 4);
        *m = (s[sub + 4] >> 4) | ((s[sub] >> 6) << 4);
    }
}

// 32 weights: an f16 scale, then weight j in a low nibble and weight j+16 in the
// high one, as `d * (q - 8)`.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q4_0>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 5) * 18;
    int j = (int)(i & 31);
    unsigned char byte = b[2 + (j & 15)];
    float q = (float)(j < 16 ? (byte & 0x0F) : (byte >> 4));
    return dlm_half(b) * (q - 8.0f);
}

// The same with a minimum in place of the offset: `d * q + m`.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q4_1>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 5) * 20;
    int j = (int)(i & 31);
    unsigned char byte = b[4 + (j & 15)];
    float q = (float)(j < 16 ? (byte & 0x0F) : (byte >> 4));
    return dlm_half(b) * q + dlm_half(b + 2);
}

// A fifth bit per weight, held in a 32-bit field: weight j takes bit j.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q5_0>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 5) * 22;
    int j = (int)(i & 31);
    unsigned int qh = (unsigned int)b[2] | ((unsigned int)b[3] << 8) |
                      ((unsigned int)b[4] << 16) | ((unsigned int)b[5] << 24);
    unsigned char byte = b[6 + (j & 15)];
    unsigned char q = (j < 16 ? (byte & 0x0F) : (byte >> 4)) | (((qh >> j) & 1u) << 4);
    return dlm_half(b) * ((float)q - 16.0f);
}

template <>
__device__ __forceinline__ float load_w<DLM_W_Q5_1>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 5) * 24;
    int j = (int)(i & 31);
    unsigned int qh = (unsigned int)b[4] | ((unsigned int)b[5] << 8) |
                      ((unsigned int)b[6] << 16) | ((unsigned int)b[7] << 24);
    unsigned char byte = b[8 + (j & 15)];
    unsigned char q = (j < 16 ? (byte & 0x0F) : (byte >> 4)) | (((qh >> j) & 1u) << 4);
    return dlm_half(b) * (float)q + dlm_half(b + 2);
}

// 32 weights: an f16 scale and one signed byte each.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q8_0>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 5) * 34;
    return dlm_half(b) * (float)(signed char)b[2 + (i & 31)];
}

// 256 weights in eight 32-weight sub-blocks, `d * sc * q - dmin * m` with sc and
// m 6-bit. The 4-bit weights run low nibbles first over each 32-byte run.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q4_K>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 8) * 144;
    int j = (int)(i & 255);
    int sub = j >> 5, l = j & 31;
    unsigned char sc, m;
    k_scale_min(sub, b + 4, &sc, &m);
    unsigned char byte = b[16 + (sub >> 1) * 32 + l];
    float q = (float)((sub & 1) ? (byte >> 4) : (byte & 0x0F));
    return dlm_half(b) * (float)sc * q - dlm_half(b + 2) * (float)m;
}

// Q4_K with a fifth bit per weight, in `qh`: sub-block `sub` takes bit `sub`.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q5_K>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 8) * 176;
    int j = (int)(i & 255);
    int sub = j >> 5, l = j & 31;
    unsigned char sc, m;
    k_scale_min(sub, b + 4, &sc, &m);
    unsigned char byte = b[48 + (sub >> 1) * 32 + l];
    unsigned char q = (sub & 1) ? (byte >> 4) : (byte & 0x0F);
    if (b[16 + l] & (1u << sub)) q += 16;
    return dlm_half(b) * (float)sc * (float)q - dlm_half(b + 2) * (float)m;
}

// 256 weights in sixteen 16-weight sub-blocks with int8 scales off a shared `d`.
// A weight's 6 bits are a nibble in `ql` and two bits in `qh`, interleaved in
// runs of 32 across each half block.
template <>
__device__ __forceinline__ float load_w<DLM_W_Q6_K>(const void* W, long i, long, int) {
    const unsigned char* b = (const unsigned char*)W + (i >> 8) * 210;
    int j = (int)(i & 255);
    int half_block = j >> 7, r = j & 127, l = r & 31, k = r >> 5;
    const unsigned char* ql = b + half_block * 64 + l + ((k & 1) ? 32 : 0);
    unsigned char lo = (k < 2) ? (*ql & 0x0F) : (*ql >> 4);
    unsigned char hi = (b[128 + half_block * 32 + l] >> (2 * k)) & 0x03;
    float q = (float)(lo | (hi << 4));
    return dlm_half(b + 208) * (float)(signed char)b[192 + (j >> 4)] * (q - 32.0f);
}

// Every weight dtype a launcher dispatches over. F32 is the default arm, so it
// is not listed.
#define DLM_FOR_EACH_DTYPE(X) \
    X(DLM_W_BF16)             \
    X(DLM_W_F16)              \
    X(DLM_W_INT4)             \
    X(DLM_W_INT8)             \
    X(DLM_W_Q4_0)             \
    X(DLM_W_Q4_1)             \
    X(DLM_W_Q5_0)             \
    X(DLM_W_Q5_1)             \
    X(DLM_W_Q8_0)             \
    X(DLM_W_Q4_K)             \
    X(DLM_W_Q5_K)             \
    X(DLM_W_Q6_K)

// One thread's share of a row dot product: the sum over `i = start, start +
// step, ... < in_dim` of `W[base + i] * x[i]`.
template <int DT>
__device__ __forceinline__ float row_dot(const void* W, const float* x, long base, int start,
                                         int step, int in_dim, long n, int group_size) {
    float s = 0.0f;
    for (int i = start; i < in_dim; i += step) s += load_w<DT>(W, base + i, n, group_size) * x[i];
    return s;
}

// The quantized rows find their scales and zeros once per call. Going through
// `load_w` recomputed both offsets for every element (two divisions by the
// group size) plus a third division for the group index. A power-of-two group
// (`--quant` uses 128) takes that index by shift instead. Each element still
// decodes as `(code - zero) * scale` and accumulates in the same order as
// `load_w`, so the result is bit-identical.
__device__ __forceinline__ int group_shift(int group_size) {
    int shift = 0;
    while ((1 << shift) < group_size) ++shift;
    return (1 << shift) == group_size ? shift : -1;
}

template <>
__device__ __forceinline__ float row_dot<DLM_W_INT4>(const void* W, const float* x, long base,
                                                     int start, int step, int in_dim, long n,
                                                     int group_size) {
    const unsigned char* bytes = (const unsigned char*)W;
    long code_bytes = (n + 1) / 2;
    const float* scales = (const float*)(bytes + q_scales_off(code_bytes));
    const float* zeros = (const float*)(bytes + q_zeros_off(code_bytes, n, group_size));
    int shift = group_shift(group_size);
    float s = 0.0f;
    for (int i = start; i < in_dim; i += step) {
        long e = base + i;
        unsigned char byte = bytes[e >> 1];
        float code = (float)((e & 1) ? (byte >> 4) : (byte & 0x0F));
        long g = shift >= 0 ? (e >> shift) : e / group_size;
        s += (code - zeros[g]) * scales[g] * x[i];
    }
    return s;
}

template <>
__device__ __forceinline__ float row_dot<DLM_W_INT8>(const void* W, const float* x, long base,
                                                     int start, int step, int in_dim, long n,
                                                     int group_size) {
    const unsigned char* bytes = (const unsigned char*)W;
    const float* scales = (const float*)(bytes + q_scales_off(n));
    const float* zeros = (const float*)(bytes + q_zeros_off(n, n, group_size));
    int shift = group_shift(group_size);
    float s = 0.0f;
    for (int i = start; i < in_dim; i += step) {
        long e = base + i;
        long g = shift >= 0 ? (e >> shift) : e / group_size;
        s += ((float)bytes[e] - zeros[g]) * scales[g] * x[i];
    }
    return s;
}

// out[o] = dot(W[o], x) (+ bias[o]). `bias` may be NULL (Llama/Mistral have no
// attention bias; Qwen2 does — dropping it silently corrupts attention).
//
// One BLOCK per output row. Threads in the block stride over the (contiguous)
// weight row, so consecutive lanes read consecutive addresses — coalesced global
// loads — then tree-reduce the partial dot products in shared memory. Launch
// <<<out_dim, MATVEC_THREADS>>>.
//
// The previous version used one THREAD per output with a sequential row walk:
// adjacent threads touched addresses `in_dim` floats apart, so every 32-lane warp
// load pulled a full cache line to use one float — ~1/32 of memory bandwidth on a
// bandwidth-bound GEMV. This is the dominant cost of the decode stack; coalescing
// it is the single biggest kernel speedup.
// A K-quant weight costs more to decode than a plain one: its scale is six bits
// packed across two bytes, under a pair of f16s, and deriving that per weight
// was a third of decode time. These take a run of consecutive weights at a time,
// so the scale is derived once for the run.
//
// The run is 8 weights, which always lies inside one sub-block (32 weights for
// Q4_K and Q5_K, 16 for Q6_K) as long as the row starts on a multiple of 8 —
// true for every real model, since a row is `in_dim` wide and `in_dim` is a
// multiple of 32. `row_dot` falls back to per-weight decoding when it is not.
//
// Consecutive threads take consecutive runs, so a warp still reads one
// contiguous stretch of the row per step, which is what keeps a GEMV at
// bandwidth.
#define DLM_KRUN 8

template <int DT>
__device__ __forceinline__ float ggml_run_dot(const unsigned char* b, int within, int n,
                                              const float* x);

// Q4_K: `d * sc * q - dmin * m`, the sub-block's 6-bit sc and m shared by 32
// weights, the weights themselves 4-bit over a 32-byte run.
template <>
__device__ __forceinline__ float ggml_run_dot<DLM_W_Q4_K>(const unsigned char* b, int within,
                                                          int n, const float* x) {
    int sub = within >> 5, l = within & 31;
    unsigned char sc, m;
    k_scale_min(sub, b + 4, &sc, &m);
    float scale = dlm_half(b) * (float)sc;
    float min = dlm_half(b + 2) * (float)m;
    const unsigned char* qs = b + 16 + (sub >> 1) * 32 + l;
    float s = 0.0f;
    if (sub & 1) {
        for (int k = 0; k < n; ++k) s += (scale * (float)(qs[k] >> 4) - min) * x[k];
    } else {
        for (int k = 0; k < n; ++k) s += (scale * (float)(qs[k] & 0x0F) - min) * x[k];
    }
    return s;
}

// Q5_K: Q4_K plus a fifth bit per weight, sub-block `sub` taking bit `sub` of
// each `qh` byte.
template <>
__device__ __forceinline__ float ggml_run_dot<DLM_W_Q5_K>(const unsigned char* b, int within,
                                                          int n, const float* x) {
    int sub = within >> 5, l = within & 31;
    unsigned char sc, m;
    k_scale_min(sub, b + 4, &sc, &m);
    float scale = dlm_half(b) * (float)sc;
    float min = dlm_half(b + 2) * (float)m;
    const unsigned char* qs = b + 48 + (sub >> 1) * 32 + l;
    const unsigned char* qh = b + 16 + l;
    unsigned char bit = (unsigned char)(1u << sub);
    float s = 0.0f;
    for (int k = 0; k < n; ++k) {
        unsigned char q = (sub & 1) ? (qs[k] >> 4) : (qs[k] & 0x0F);
        if (qh[k] & bit) q += 16;
        s += (scale * (float)q - min) * x[k];
    }
    return s;
}

// Q6_K: sixteen 16-weight sub-blocks with int8 scales, each weight a nibble in
// `ql` and two bits in `qh`, in runs of 32 across each half block.
template <>
__device__ __forceinline__ float ggml_run_dot<DLM_W_Q6_K>(const unsigned char* b, int within,
                                                          int n, const float* x) {
    int half_block = within >> 7, r = within & 127, l = r & 31, k4 = r >> 5;
    const unsigned char* ql = b + half_block * 64 + l + ((k4 & 1) ? 32 : 0);
    const unsigned char* qh = b + 128 + half_block * 32 + l;
    float scale = dlm_half(b + 208) * (float)(signed char)b[192 + (within >> 4)];
    int shift = 2 * k4;
    float s = 0.0f;
    for (int k = 0; k < n; ++k) {
        unsigned char lo = (k4 < 2) ? (ql[k] & 0x0F) : (ql[k] >> 4);
        unsigned char hi = (qh[k] >> shift) & 0x03;
        s += scale * ((float)(lo | (hi << 4)) - 32.0f) * x[k];
    }
    return s;
}

// One thread's share of a row dot product for a K-quant, a run of weights at a
// time. Mirrors the generic `row_dot`, which the float and int arms use.
template <int DT>
__device__ __forceinline__ float row_dot_kquant(const void* W, const float* x, long base,
                                                int start, int step, int in_dim) {
    const int BLOCK = 256;
    const int BYTES = (DT == DLM_W_Q4_K) ? 144 : ((DT == DLM_W_Q5_K) ? 176 : 210);
    const unsigned char* bytes = (const unsigned char*)W;
    float s = 0.0f;
    for (int i = start * DLM_KRUN; i < in_dim; i += step * DLM_KRUN) {
        long e = base + i;
        int n = min(DLM_KRUN, in_dim - i);
        s += ggml_run_dot<DT>(bytes + (e / BLOCK) * BYTES, (int)(e % BLOCK), n, x + i);
    }
    return s;
}

template <>
__device__ __forceinline__ float row_dot<DLM_W_Q4_K>(const void* W, const float* x, long base,
                                                     int start, int step, int in_dim, long,
                                                     int) {
    if ((base | in_dim) % DLM_KRUN) {
        float s = 0.0f;
        for (int i = start; i < in_dim; i += step) s += load_w<DLM_W_Q4_K>(W, base + i, 0, 0) * x[i];
        return s;
    }
    return row_dot_kquant<DLM_W_Q4_K>(W, x, base, start, step, in_dim);
}

template <>
__device__ __forceinline__ float row_dot<DLM_W_Q5_K>(const void* W, const float* x, long base,
                                                     int start, int step, int in_dim, long,
                                                     int) {
    if ((base | in_dim) % DLM_KRUN) {
        float s = 0.0f;
        for (int i = start; i < in_dim; i += step) s += load_w<DLM_W_Q5_K>(W, base + i, 0, 0) * x[i];
        return s;
    }
    return row_dot_kquant<DLM_W_Q5_K>(W, x, base, start, step, in_dim);
}

template <>
__device__ __forceinline__ float row_dot<DLM_W_Q6_K>(const void* W, const float* x, long base,
                                                     int start, int step, int in_dim, long,
                                                     int) {
    if ((base | in_dim) % DLM_KRUN) {
        float s = 0.0f;
        for (int i = start; i < in_dim; i += step) s += load_w<DLM_W_Q6_K>(W, base + i, 0, 0) * x[i];
        return s;
    }
    return row_dot_kquant<DLM_W_Q6_K>(W, x, base, start, step, in_dim);
}

template <int DT>
__global__ void matvec_kernel(const void* W, const float* x, const float* bias, float* out,
                              int out_dim, int in_dim, int group_size) {
    int o = blockIdx.x;
    if (o >= out_dim) return;
    long base = (long)o * in_dim;
    long n = (long)out_dim * in_dim;   // the tensor's element count (INT4 layout)
    // Reduce within each warp by shuffle, then combine the warps through shared
    // memory: one barrier instead of one per tree level. On a 896-wide projection
    // a block does 14 rows of work, so six barrier rounds were a third of it.
    __shared__ float partial[MATVEC_THREADS / 32];
    float v = row_dot<DT>(W, x, base, threadIdx.x, blockDim.x, in_dim, n, group_size);
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total += partial[w];
        out[o] = total + (bias ? bias[o] : 0.0f);
    }
}

// ── Chunked GEMV: one warp per row, one aligned chunk per lane ─────────────
//
// `matvec_kernel` above decodes one weight per loop step through `load_w`. For
// the formats with more than a cast to decode (every quantized one) that
// re-derives the block header -- the scale, the min, Q5's 32-bit high-bit field
// -- for every single weight, and reads the row a byte or two at a time. A
// one-block-per-row grid also gives a 896-wide row 64 threads with 14 weights
// each, too little in flight to cover memory latency.
//
// Here a lane takes a whole **chunk** -- a run of weights that never straddles a
// scale group or block (8 floats, 16 int8 codes, 32 int4 codes, one 32-weight
// ggml block, one K-quant sub-block) -- reads it with wide loads, decodes the
// header once, and dots it against the matching `x` run. Lanes take consecutive
// chunks, so a warp still reads one contiguous stretch of the row per step, and
// a block holds `MV_ROWS` rows so small matrices still fill the GPU.
//
// The decoded values are the ones `load_w` produces; only the summation order
// changes (per chunk, then across lanes), within float rounding of the old one.
// Shapes the chunk does not fit (a row width or group size that is not a
// multiple of the chunk, or a misaligned pointer) keep the per-weight kernel.
#define MV_ROWS 4

__device__ __forceinline__ unsigned int ld_u16(const unsigned char* p) {
    return (unsigned int)*(const unsigned short*)p;   // 2-byte aligned by the format
}
__device__ __forceinline__ float half_bits(unsigned int bits) {
    unsigned short h = (unsigned short)bits;
    return __half2float(*(const __half*)&h);
}

template <int DT> struct Chunk;

template <> struct Chunk<DLM_W_F32> {
    enum { N = 8 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const float4* p = (const float4*)((const float*)W + e);
        float4 a = p[0], b = p[1];
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
        w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    }
};
template <> struct Chunk<DLM_W_BF16> {
    enum { N = 8 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        uint4 v = *(const uint4*)((const unsigned short*)W + e);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[2 * i] = __int_as_float(u[i] << 16);
            w[2 * i + 1] = __int_as_float(u[i] & 0xFFFF0000u);
        }
    }
};
template <> struct Chunk<DLM_W_F16> {
    enum { N = 8 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        uint4 v = *(const uint4*)((const __half*)W + e);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[2 * i] = half_bits(u[i] & 0xFFFFu);
            w[2 * i + 1] = half_bits(u[i] >> 16);
        }
    }
};
// 32 codes in one 16-byte load; the chunk lies in one group (`group % 32 == 0`).
template <> struct Chunk<DLM_W_INT4> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long n, int gs, float* w) {
        const unsigned char* bytes = (const unsigned char*)W;
        long code_bytes = (n + 1) / 2;
        long g = e / gs;
        float scale = ((const float*)(bytes + q_scales_off(code_bytes)))[g];
        float zero = ((const float*)(bytes + q_zeros_off(code_bytes, n, gs)))[g];
        uint4 v = *(const uint4*)(bytes + (e >> 1));
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned int byte = (u[i >> 2] >> (8 * (i & 3))) & 0xFFu;
            w[2 * i] = ((float)(byte & 0x0F) - zero) * scale;
            w[2 * i + 1] = ((float)(byte >> 4) - zero) * scale;
        }
    }
};
template <> struct Chunk<DLM_W_INT8> {
    enum { N = 16 };
    __device__ static void decode(const void* W, long e, long n, int gs, float* w) {
        const unsigned char* bytes = (const unsigned char*)W;
        long g = e / gs;
        float scale = ((const float*)(bytes + q_scales_off(n)))[g];
        float zero = ((const float*)(bytes + q_zeros_off(n, n, gs)))[g];
        uint4 v = *(const uint4*)(bytes + e);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            w[i] = ((float)((u[i >> 2] >> (8 * (i & 3))) & 0xFFu) - zero) * scale;
    }
};
// The 32-weight ggml blocks: one block per chunk. Every block size is even, so
// the block (and its f16 header) is 2-byte aligned and reads as u16 pairs.
template <> struct Chunk<DLM_W_Q4_0> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 18;
        float d = half_bits(ld_u16(b));
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            unsigned int two = ld_u16(b + 2 + j);
            w[j] = d * ((float)(two & 0x0F) - 8.0f);
            w[j + 16] = d * ((float)((two >> 4) & 0x0F) - 8.0f);
            w[j + 1] = d * ((float)((two >> 8) & 0x0F) - 8.0f);
            w[j + 17] = d * ((float)(two >> 12) - 8.0f);
        }
    }
};
template <> struct Chunk<DLM_W_Q4_1> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 20;
        float d = half_bits(ld_u16(b)), m = half_bits(ld_u16(b + 2));
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            unsigned int two = ld_u16(b + 4 + j);
            w[j] = d * (float)(two & 0x0F) + m;
            w[j + 16] = d * (float)((two >> 4) & 0x0F) + m;
            w[j + 1] = d * (float)((two >> 8) & 0x0F) + m;
            w[j + 17] = d * (float)(two >> 12) + m;
        }
    }
};
template <> struct Chunk<DLM_W_Q5_0> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 22;
        float d = half_bits(ld_u16(b));
        unsigned int qh = ld_u16(b + 2) | (ld_u16(b + 4) << 16);
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            unsigned int two = ld_u16(b + 6 + j);
            w[j] = d * ((float)((two & 0x0F) | (((qh >> j) & 1u) << 4)) - 16.0f);
            w[j + 16] = d * ((float)(((two >> 4) & 0x0F) | (((qh >> (j + 16)) & 1u) << 4)) - 16.0f);
            w[j + 1] = d * ((float)(((two >> 8) & 0x0F) | (((qh >> (j + 1)) & 1u) << 4)) - 16.0f);
            w[j + 17] = d * ((float)((two >> 12) | (((qh >> (j + 17)) & 1u) << 4)) - 16.0f);
        }
    }
};
template <> struct Chunk<DLM_W_Q5_1> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 24;
        float d = half_bits(ld_u16(b)), m = half_bits(ld_u16(b + 2));
        unsigned int qh = ld_u16(b + 4) | (ld_u16(b + 6) << 16);
        #pragma unroll
        for (int j = 0; j < 16; j += 2) {
            unsigned int two = ld_u16(b + 8 + j);
            w[j] = d * (float)((two & 0x0F) | (((qh >> j) & 1u) << 4)) + m;
            w[j + 16] = d * (float)(((two >> 4) & 0x0F) | (((qh >> (j + 16)) & 1u) << 4)) + m;
            w[j + 1] = d * (float)(((two >> 8) & 0x0F) | (((qh >> (j + 1)) & 1u) << 4)) + m;
            w[j + 17] = d * (float)((two >> 12) | (((qh >> (j + 17)) & 1u) << 4)) + m;
        }
    }
};
template <> struct Chunk<DLM_W_Q8_0> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 34;
        float d = half_bits(ld_u16(b));
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
            unsigned int two = ld_u16(b + 2 + j);
            w[j] = d * (float)(signed char)(two & 0xFF);
            w[j + 1] = d * (float)(signed char)(two >> 8);
        }
    }
};
// K-quants: one 32-weight sub-block per chunk (Q6_K's 16-weight scales give a
// chunk two). Q4_K/Q5_K blocks are 16-byte multiples, so their runs load wide.
template <> struct Chunk<DLM_W_Q4_K> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 8) * 144;
        int sub = (int)(e & 255) >> 5;
        unsigned char sc, m;
        k_scale_min(sub, b + 4, &sc, &m);
        float scale = dlm_half(b) * (float)sc, min = dlm_half(b + 2) * (float)m;
        const uint4* qs = (const uint4*)(b + 16 + (sub >> 1) * 32);
        int sh = (sub & 1) * 4;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 v = qs[h];
            unsigned int u[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int i = 0; i < 16; ++i)
                w[h * 16 + i] = scale * (float)((u[i >> 2] >> (8 * (i & 3) + sh)) & 0x0F) - min;
        }
    }
};
template <> struct Chunk<DLM_W_Q5_K> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 8) * 176;
        int sub = (int)(e & 255) >> 5;
        unsigned char sc, m;
        k_scale_min(sub, b + 4, &sc, &m);
        float scale = dlm_half(b) * (float)sc, min = dlm_half(b + 2) * (float)m;
        const uint4* qs = (const uint4*)(b + 48 + (sub >> 1) * 32);
        const uint4* qh = (const uint4*)(b + 16);
        int sh = (sub & 1) * 4;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 v = qs[h], hv = qh[h];
            unsigned int u[4] = {v.x, v.y, v.z, v.w};
            unsigned int hb[4] = {hv.x, hv.y, hv.z, hv.w};
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int s = 8 * (i & 3);
                unsigned int q = ((u[i >> 2] >> (s + sh)) & 0x0F) | (((hb[i >> 2] >> (s + sub)) & 1u) << 4);
                w[h * 16 + i] = scale * (float)q - min;
            }
        }
    }
};
template <> struct Chunk<DLM_W_Q6_K> {
    enum { N = 32 };
    __device__ static void decode(const void* W, long e, long, int, float* w) {
        const unsigned char* b = (const unsigned char*)W + (e >> 8) * 210;
        int within = (int)(e & 255);
        int half_block = within >> 7, k4 = (within & 127) >> 5;
        const unsigned char* ql = b + half_block * 64 + ((k4 & 1) ? 32 : 0);
        const unsigned char* qh = b + 128 + half_block * 32;
        float d = dlm_half(b + 208);
        float s0 = d * (float)(signed char)b[192 + (within >> 4)];
        float s1 = d * (float)(signed char)b[192 + (within >> 4) + 1];
        int lo_sh = (k4 < 2) ? 0 : 4, hi_sh = 2 * k4;
        #pragma unroll
        for (int l = 0; l < 32; l += 2) {
            unsigned int lo2 = ld_u16(ql + l), hi2 = ld_u16(qh + l);
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                unsigned int q = ((lo2 >> (8 * t + lo_sh)) & 0x0F) | (((hi2 >> (8 * t + hi_sh)) & 3u) << 4);
                w[l + t] = (l < 16 ? s0 : s1) * ((float)q - 32.0f);
            }
        }
    }
};

// Can `Chunk<DT>` cover this matrix? Each row must start on a chunk (the width
// a multiple of it), a chunk must sit inside one scale group, and the wide loads
// need 16-byte aligned weights and activations (cudaMalloc gives 256).
template <int DT>
static inline bool chunk_fits(const void* W, const float* x, int in_dim, int group_size) {
    const int N = Chunk<DT>::N;
    if (in_dim % N) return false;
    if ((DT == DLM_W_INT4 || DT == DLM_W_INT8) && (group_size <= 0 || group_size % N)) return false;
    if (DT == DLM_W_Q4_K || DT == DLM_W_Q5_K || DT == DLM_W_Q6_K) {
        if (in_dim % 256) return false;
    }
    return ((unsigned long long)W % 16 == 0) && ((unsigned long long)x % 16 == 0);
}

template <int N>
__device__ __forceinline__ float chunk_dot(const float* w, const float* x) {
    const float4* x4 = (const float4*)x;
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < N / 4; ++i) {
        float4 v = x4[i];
        s += w[4 * i] * v.x + w[4 * i + 1] * v.y + w[4 * i + 2] * v.z + w[4 * i + 3] * v.w;
    }
    return s;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Launch <<<ceil(out_dim / MV_ROWS), 32 * MV_ROWS>>>.
template <int DT>
__global__ void matvec_chunk_kernel(const void* __restrict__ W, const float* __restrict__ x,
                                    const float* __restrict__ bias, float* __restrict__ out,
                                    int out_dim, int in_dim, int group_size) {
    const int N = Chunk<DT>::N;
    int o = blockIdx.x * MV_ROWS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;   // whole warps only: `o` is per warp
    long n = (long)out_dim * in_dim;
    long base = (long)o * in_dim;
    float s = 0.0f;
    for (int i = lane * N; i < in_dim; i += 32 * N) {
        float w[N];
        Chunk<DT>::decode(W, base + i, n, group_size, w);
        s += chunk_dot<N>(w, x + i);
    }
    s = warp_sum(s);
    if (lane == 0) out[o] = s + (bias ? bias[o] : 0.0f);
}

// ── Integer dot products for quantized weights (dp4a) ───────────────────────
//
// Decoding a quantized weight to f32 costs a shift, a mask, an int-to-float
// conversion and a multiply-add per weight, and on a GTX 1650 that -- not
// memory -- bounds every quantized GEMV: 16-40 GB/s against the card's 128.
// llama.cpp's answer, taken here: quantize the activation vector to int8 once
// per GEMV (32 values per f32 scale, plus that block's scaled sum), and dot
// four weight codes with four activation codes in one `dp4a`. The weight's own
// scale, minimum or zero is applied once per 32 weights, through the
// activation block's scale and sum:
//
//     sum_i (scale * code_i - offset) * x_i
//       = scale * dx * sum_i code_i * xq_i  -  offset * dx * sum_i xq_i
//
// Measured on the Q5_0 LM head of Qwen2.5-0.5B: 4.2 ms -> 1.5 ms.
//
// The activation codes round `x` to 8 bits per 32-value block, which the f32
// path does not: the result differs from it by about a part in a thousand.
// `DLM_GEMV_EXACT=1` keeps the f32 path for quantized weights.
#ifdef __HIP_PLATFORM_AMD__
__device__ __forceinline__ int dlm_dp4a(int a, int b, int c) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) c += (int)(signed char)(a >> (8 * i)) * (int)(signed char)(b >> (8 * i));
    return c;
}
#else
__device__ __forceinline__ int dlm_dp4a(int a, int b, int c) { return __dp4a(a, b, c); }
#endif

// `x` as int8 per 32 values: codes, and per block (scale, scale * sum of codes).
// One warp per block: <<<ceil(n / 32 / 8), dim3(32, 8)>>>; `n % 32 == 0`.
__global__ void quantize_q8_kernel(const float* __restrict__ x, int n, signed char* __restrict__ q,
                                   float2* __restrict__ ds) {
    int b = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (b * 32 >= n) return;
    float v = x[b * 32 + lane];
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float d = amax / 127.0f;
    int qi = d == 0.0f ? 0 : __float2int_rn(v / d);
    int sum = qi;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    q[b * 32 + lane] = (signed char)qi;
    if (lane == 0) ds[b] = make_float2(d, d * (float)sum);
}

// Four 4-bit codes, one per byte, from two bytes of dlm's int4 layout (weight
// 2i in byte i's low nibble, 2i + 1 in its high one).
__device__ __forceinline__ int nib4(unsigned int two) {
    return (int)((two & 0xFu) | ((two & 0xF0u) << 4) | ((two & 0xF00u) << 8) | ((two & 0xF000u) << 12));
}
// Spread four bits into the 0x10 bit of four bytes (Q5's fifth bit).
__device__ __forceinline__ unsigned int bits4(unsigned int b) {
    return ((b & 1u) << 4) | ((b & 2u) << 11) | ((b & 4u) << 18) | ((b & 8u) << 25);
}
// Four bytes from a 2-byte aligned address.
__device__ __forceinline__ unsigned int ld_u32_2(const unsigned char* p) {
    return ld_u16(p) | (ld_u16(p + 2) << 16);
}

// One 32-weight chunk for the int8-activation dot product, decoded once and
// dotted with any number of activation blocks: `q[k]` holds four signed codes
// that pair with the block's code word `xi[k]`, and the chunk's value is
//
//     dx.x * (a0 * sum_{k<4} q[k].xi[k] + a1 * sum_{k>=4} q[k].xi[k]) + b * dx.y
//
// where `dx` is the block's (scale, scale * sum of codes). Every format but
// Q6_K has one scale per chunk (a0 = a1, summed as one); its offset or minimum
// goes into `b`. Splitting the decode from the dot lets a batch decode each
// chunk once for all its sequences.
//
// Q6_K (`SPLIT`) has a scale per 16 weights, and unsigned codes with an offset
// of 32 that `b * dx.y` cannot carry, being per half: each half's dot takes off
// 32 times that half's sum of activation codes (`xs0`, `xs1`, from
// `q8_half_sums`). Those depend only on the activations, so a batch finds them
// once per slot for all its rows.
struct Q8Codes {
    int q[8];
    float a0, a1, b;
};

__device__ __forceinline__ void q8_half_sums(const int* xi, int& xs0, int& xs1) {
    xs0 = xs1 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) xs0 = dlm_dp4a(0x01010101, xi[k], xs0);
    #pragma unroll
    for (int k = 4; k < 8; ++k) xs1 = dlm_dp4a(0x01010101, xi[k], xs1);
}

template <bool SPLIT>
__device__ __forceinline__ float q8_apply(const Q8Codes& c, const int* xi, float2 dx, int xs0,
                                          int xs1) {
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) s0 = dlm_dp4a(c.q[k], xi[k], s0);
    #pragma unroll
    for (int k = 4; k < 8; ++k) s1 = dlm_dp4a(c.q[k], xi[k], s1);
    if (SPLIT) return dx.x * (c.a0 * (float)(s0 - 32 * xs0) + c.a1 * (float)(s1 - 32 * xs1));
    return c.a0 * dx.x * (float)(s0 + s1) + c.b * dx.y;
}

// `load` decodes the chunk starting at element `e` (a multiple of 32).
template <int DT> struct Dp4a {
    enum { OK = 0, SPLIT = 0 };
    __device__ static void load(const void*, long, long, int, Q8Codes& c) { c = Q8Codes{}; }
};

template <> struct Dp4a<DLM_W_INT4> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long n, int gs, Q8Codes& c) {
        const unsigned char* bytes = (const unsigned char*)W;
        long code_bytes = (n + 1) / 2, g = e / gs;
        float scale = ((const float*)(bytes + q_scales_off(code_bytes)))[g];
        float zero = ((const float*)(bytes + q_zeros_off(code_bytes, n, gs)))[g];
        uint4 v = *(const uint4*)(bytes + (e >> 1));
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            c.q[2 * j] = nib4(u[j] & 0xFFFFu);
            c.q[2 * j + 1] = nib4(u[j] >> 16);
        }
        c.a0 = c.a1 = scale;
        c.b = -scale * zero;
    }
};
template <> struct Dp4a<DLM_W_INT8> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long n, int gs, Q8Codes& c) {
        const unsigned char* bytes = (const unsigned char*)W;
        long g = e / gs;
        float scale = ((const float*)(bytes + q_scales_off(n)))[g];
        float zero = ((const float*)(bytes + q_zeros_off(n, n, gs)))[g];
        const uint4* p = (const uint4*)(bytes + e);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 v = p[h];
            // Unsigned codes as signed bytes: code - 128.
            c.q[4 * h] = (int)(v.x ^ 0x80808080u);
            c.q[4 * h + 1] = (int)(v.y ^ 0x80808080u);
            c.q[4 * h + 2] = (int)(v.z ^ 0x80808080u);
            c.q[4 * h + 3] = (int)(v.w ^ 0x80808080u);
        }
        c.a0 = c.a1 = scale;
        c.b = scale * (128.0f - zero);
    }
};
template <> struct Dp4a<DLM_W_Q4_0> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 18;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int v = ld_u32_2(b + 2 + 4 * j);
            c.q[j] = (int)(v & 0x0F0F0F0Fu);
            c.q[j + 4] = (int)((v >> 4) & 0x0F0F0F0Fu);
        }
        c.a0 = c.a1 = half_bits(ld_u16(b));
        c.b = -8.0f * c.a0;
    }
};
template <> struct Dp4a<DLM_W_Q4_1> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 20;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int v = ld_u32_2(b + 4 + 4 * j);
            c.q[j] = (int)(v & 0x0F0F0F0Fu);
            c.q[j + 4] = (int)((v >> 4) & 0x0F0F0F0Fu);
        }
        c.a0 = c.a1 = half_bits(ld_u16(b));
        c.b = half_bits(ld_u16(b + 2));
    }
};
template <> struct Dp4a<DLM_W_Q5_0> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 22;
        unsigned int qh = ld_u32_2(b + 2);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int v = ld_u32_2(b + 6 + 4 * j);
            c.q[j] = (int)((v & 0x0F0F0F0Fu) | bits4((qh >> (4 * j)) & 0xFu));
            c.q[j + 4] = (int)(((v >> 4) & 0x0F0F0F0Fu) | bits4((qh >> (4 * j + 16)) & 0xFu));
        }
        c.a0 = c.a1 = half_bits(ld_u16(b));
        c.b = -16.0f * c.a0;
    }
};
template <> struct Dp4a<DLM_W_Q5_1> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 24;
        unsigned int qh = ld_u32_2(b + 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int v = ld_u32_2(b + 8 + 4 * j);
            c.q[j] = (int)((v & 0x0F0F0F0Fu) | bits4((qh >> (4 * j)) & 0xFu));
            c.q[j + 4] = (int)(((v >> 4) & 0x0F0F0F0Fu) | bits4((qh >> (4 * j + 16)) & 0xFu));
        }
        c.a0 = c.a1 = half_bits(ld_u16(b));
        c.b = half_bits(ld_u16(b + 2));
    }
};
template <> struct Dp4a<DLM_W_Q8_0> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 5) * 34;
        #pragma unroll
        for (int j = 0; j < 8; ++j) c.q[j] = (int)ld_u32_2(b + 2 + 4 * j);
        c.a0 = c.a1 = half_bits(ld_u16(b));
        c.b = 0.0f;
    }
};
template <> struct Dp4a<DLM_W_Q4_K> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 8) * 144;
        int sub = (int)(e & 255) >> 5;
        unsigned char sc, m;
        k_scale_min(sub, b + 4, &sc, &m);
        const uint4* qs = (const uint4*)(b + 16 + (sub >> 1) * 32);
        int sh = (sub & 1) * 4;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 v = qs[h];
            c.q[4 * h] = (int)((v.x >> sh) & 0x0F0F0F0Fu);
            c.q[4 * h + 1] = (int)((v.y >> sh) & 0x0F0F0F0Fu);
            c.q[4 * h + 2] = (int)((v.z >> sh) & 0x0F0F0F0Fu);
            c.q[4 * h + 3] = (int)((v.w >> sh) & 0x0F0F0F0Fu);
        }
        c.a0 = c.a1 = dlm_half(b) * (float)sc;
        c.b = -dlm_half(b + 2) * (float)m;
    }
};
template <> struct Dp4a<DLM_W_Q5_K> {
    enum { OK = 1, SPLIT = 0 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 8) * 176;
        int sub = (int)(e & 255) >> 5;
        unsigned char sc, m;
        k_scale_min(sub, b + 4, &sc, &m);
        const uint4* qs = (const uint4*)(b + 48 + (sub >> 1) * 32);
        const uint4* qh = (const uint4*)(b + 16);
        int sh = (sub & 1) * 4;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 v = qs[h], hv = qh[h];
            unsigned int u[4] = {v.x, v.y, v.z, v.w}, hb[4] = {hv.x, hv.y, hv.z, hv.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                c.q[4 * h + j] =
                    (int)(((u[j] >> sh) & 0x0F0F0F0Fu) | (((hb[j] >> sub) & 0x01010101u) << 4));
        }
        c.a0 = c.a1 = dlm_half(b) * (float)sc;
        c.b = -dlm_half(b + 2) * (float)m;
    }
};
// Q6_K's chunk spans two 16-weight scales (a0, a1); its codes stay unsigned,
// the offset of 32 taken off in `q8_apply`.
template <> struct Dp4a<DLM_W_Q6_K> {
    enum { OK = 1, SPLIT = 1 };
    __device__ static void load(const void* W, long e, long, int, Q8Codes& c) {
        const unsigned char* b = (const unsigned char*)W + (e >> 8) * 210;
        int within = (int)(e & 255);
        int half_block = within >> 7, k4 = (within & 127) >> 5;
        const unsigned char* ql = b + half_block * 64 + ((k4 & 1) ? 32 : 0);
        const unsigned char* qh = b + 128 + half_block * 32;
        int lo_sh = (k4 < 2) ? 0 : 4, hi_sh = 2 * k4;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            c.q[j] = (int)(((ld_u32_2(ql + 4 * j) >> lo_sh) & 0x0F0F0F0Fu) |
                           (((ld_u32_2(qh + 4 * j) >> hi_sh) & 0x03030303u) << 4));
        float d = dlm_half(b + 208);
        c.a0 = d * (float)(signed char)b[192 + (within >> 4)];
        c.a1 = d * (float)(signed char)b[192 + (within >> 4) + 1];
        c.b = 0.0f;
    }
};

// One 32-weight chunk dotted with one activation block: `xi` the block's 8
// code words, `dx` its (scale, scaled sum).
template <int DT>
__device__ __forceinline__ float dp4a_dot(const void* W, long e, long n, int gs, const int* xi,
                                          float2 dx) {
    Q8Codes c;
    Dp4a<DT>::load(W, e, n, gs, c);
    int xs0 = 0, xs1 = 0;
    if (Dp4a<DT>::SPLIT) q8_half_sums(xi, xs0, xs1);
    return q8_apply<(bool)Dp4a<DT>::SPLIT>(c, xi, dx, xs0, xs1);
}

// Q6_K's single chunk keeps its decode and dot interleaved: through `Q8Codes`
// its one-sequence decode ran 5-10% slower on Qwen2.5-0.5B Q6_K (98 -> 89-95
// tok/s), where every other format measured the same. Same arithmetic.
template <>
__device__ __forceinline__ float dp4a_dot<DLM_W_Q6_K>(const void* W, long e, long, int,
                                                      const int* xi, float2 dx) {
    const unsigned char* b = (const unsigned char*)W + (e >> 8) * 210;
    int within = (int)(e & 255);
    int half_block = within >> 7, k4 = (within & 127) >> 5;
    const unsigned char* ql = b + half_block * 64 + ((k4 & 1) ? 32 : 0);
    const unsigned char* qh = b + 128 + half_block * 32;
    int lo_sh = (k4 < 2) ? 0 : 4, hi_sh = 2 * k4;
    int s[2] = {0, 0}, xs[2] = {0, 0};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned int q = ((ld_u32_2(ql + 4 * j) >> lo_sh) & 0x0F0F0F0Fu) |
                         (((ld_u32_2(qh + 4 * j) >> hi_sh) & 0x03030303u) << 4);
        s[j >> 2] = dlm_dp4a((int)q, xi[j], s[j >> 2]);
        xs[j >> 2] = dlm_dp4a(0x01010101, xi[j], xs[j >> 2]);
    }
    float d = dlm_half(b + 208);
    float s0 = d * (float)(signed char)b[192 + (within >> 4)];
    float s1 = d * (float)(signed char)b[192 + (within >> 4) + 1];
    return dx.x * (s0 * (float)(s[0] - 32 * xs[0]) + s1 * (float)(s[1] - 32 * xs[1]));
}

// One row against the quantized activations, reduced across the warp.
template <int DT>
__device__ __forceinline__ float q8_row_dot(const void* W, long base, long n, int group_size,
                                            const signed char* xq, const float2* xds, int in_dim,
                                            int lane) {
    float s = 0.0f;
    for (int i = lane * 32; i < in_dim; i += 32 * 32)
        s += dp4a_dot<DT>(W, base + i, n, group_size, (const int*)(xq + i), xds[i >> 5]);
    return warp_sum(s);
}

// R rows per warp, interleaved (see `Q8Rows`).
// <<<ceil(out_dim / (R * MV_ROWS)), 32 * MV_ROWS>>>.
template <int DT, int R>
__global__ void matvec_q8_kernel(const void* __restrict__ W, const signed char* __restrict__ xq,
                                 const float2* __restrict__ xds, const float* __restrict__ bias,
                                 float* __restrict__ out, int out_dim, int in_dim, int group_size) {
    const int o0 = (blockIdx.x * MV_ROWS + (threadIdx.x >> 5)) * R;
    const int lane = threadIdx.x & 31;
    if (o0 >= out_dim) return;
    const long n = (long)out_dim * in_dim;
    float s[R];
    long base[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        s[r] = 0.0f;
        base[r] = (long)min(o0 + r, out_dim - 1) * in_dim;   // a short tail repeats the last row
    }
    for (int i = lane * 32; i < in_dim; i += 32 * 32) {
        const int* xi = (const int*)(xq + i);
        const float2 dx = xds[i >> 5];
        #pragma unroll
        for (int r = 0; r < R; ++r) s[r] += dp4a_dot<DT>(W, base[r] + i, n, group_size, xi, dx);
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_sum(s[r]);
        if (lane == 0 && o0 + r < out_dim) out[o0 + r] = v + (bias ? bias[o0 + r] : 0.0f);
    }
}

// Rows per warp for each format's int8-activation GEMV.
//
// A 32-weight chunk of a quantized format is a few bytes, so on the short rows
// of small models (896 wide on Qwen2.5-0.5B: one chunk per lane) a warp issued
// its loads, waited, and then spent its time on the reduction with nothing in
// flight. With full occupancy that still left int4 latency-bound at 60-65% of
// the bandwidth bf16 reaches. Interleaving R rows per warp keeps R rows' loads
// in flight together. Swept on a GTX 1650 at Qwen2.5 0.5B/1.5B shapes (us for
// the 151936x896 LM head, R = 1 -> chosen): int4 g128 868 -> 500, Q5_0 1092
// -> 624, Q8_0 1607 -> 1044; Q4_K and Q6_K (no 896-wide rows) 1.3-1.5x on
// 8960x1536. int8 reads twice int4's bytes for the same work and was already
// at bandwidth: more rows only cost it registers.
template <int DT> struct Q8Rows { enum { R = 4 }; };   // ggml's 32-weight blocks
template <> struct Q8Rows<DLM_W_INT8> { enum { R = 1 }; };
template <> struct Q8Rows<DLM_W_INT4> { enum { R = 2 }; };
template <> struct Q8Rows<DLM_W_Q4_K> { enum { R = 2 }; };

template <int DT>
static void launch_q8(const void* W, const signed char* xq, const float2* xds, const float* bias,
                      float* out, int out_dim, int in_dim, int group_size) {
    const int R = Q8Rows<DT>::R, per_block = R * MV_ROWS;
    matvec_q8_kernel<DT, R><<<(out_dim + per_block - 1) / per_block, 32 * MV_ROWS>>>(
        W, xq, xds, bias, out, out_dim, in_dim, group_size);
}

// Whether `DLM_GEMV_EXACT` asked for the f32 path everywhere.
static bool gemv_exact() {
    static const bool exact = [] {
        const char* v = getenv("DLM_GEMV_EXACT");
        return v && *v && *v != '0';
    }();
    return exact;
}

// Can this matrix take the dp4a path? The chunked shape rules, 32-weight
// chunks, and a format with integer codes.
template <int DT>
static inline bool q8_fits(const void* W, const float* x, int in_dim, int group_size) {
    if (!Dp4a<DT>::OK || gemv_exact()) return false;
    if (in_dim % 32 || (unsigned long long)W % 16) return false;
    if ((DT == DLM_W_INT4 || DT == DLM_W_INT8) && (group_size <= 0 || group_size % 32)) return false;
    if ((DT == DLM_W_Q4_K || DT == DLM_W_Q5_K || DT == DLM_W_Q6_K) && in_dim % 256) return false;
    return true;
}

// Quantize `x` (`n` floats) into this thread's activation scratch; defined with
// the scratch buffers below. False (nothing launched) if the scratch failed.
static bool quantize_x(const float* x, int n, const signed char** xq, const float2** xds);

// Dispatch the GEMV on the runtime weight dtype: integer dot products where the
// format has codes, else the chunked kernel where the shape allows it, else one
// block per output row.
static void launch_matvec(int dt, const void* W, const float* x, const float* bias, float* out,
                          int out_dim, int in_dim, int group_size) {
    const signed char* xq;
    const float2* xds;
#define DLM_CASE(D) case D: \
        if (q8_fits<D>(W, x, in_dim, group_size) && quantize_x(x, in_dim, &xq, &xds)) \
            launch_q8<D>(W, xq, xds, bias, out, out_dim, in_dim, group_size); \
        else if (chunk_fits<D>(W, x, in_dim, group_size)) \
            matvec_chunk_kernel<D><<<(out_dim + MV_ROWS - 1) / MV_ROWS, 32 * MV_ROWS>>>(W, x, bias, out, out_dim, in_dim, group_size); \
        else \
            matvec_kernel<D><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size); \
        break;
    switch (dt) {
        DLM_FOR_EACH_DTYPE(DLM_CASE)
        default:
            if (chunk_fits<DLM_W_F32>(W, x, in_dim, group_size))
                matvec_chunk_kernel<DLM_W_F32><<<(out_dim + MV_ROWS - 1) / MV_ROWS, 32 * MV_ROWS>>>(
                    W, x, bias, out, out_dim, in_dim, group_size);
            else
                matvec_kernel<DLM_W_F32>
                    <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size);
            break;
    }
#undef DLM_CASE
}

// In-place rotary embedding over [num_heads * head_dim]. One thread per rotated pair.
//
// `inv_freq` is a device array of head_dim/2 precomputed inverse frequencies,
// produced host-side by `rope_inv_freqs` in src/forward/cpu.rs — the same
// function the CPU block uses. Computing the frequency here instead (powf) would
// duplicate the formula and let the GPU silently drift from the CPU oracle the
// moment a RoPE scaling type is added.
// `mscale` (YaRN attention temperature) scales cos/sin; pass 1.0 otherwise.
__global__ void rope_kernel(float* v, int num_heads, int head_dim, int position,
                            const float* inv_freq, float mscale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = num_heads * half;
    if (idx >= total) return;
    int h = idx / half;
    int i = idx % half;
    int base = h * head_dim;
    float ang = (float)position * inv_freq[i];
    float s = sinf(ang) * mscale, c = cosf(ang) * mscale;
    float a = v[base + i];
    float b = v[base + i + half];
    v[base + i] = a * c - b * s;
    v[base + i + half] = a * s + b * c;
}

// Per-head RMSNorm over `head_dim` (Qwen3 Q/K norm), applied after projection and
// before RoPE. One block per head; `w` is `[head_dim]` shared across heads.
// Mirrors `head_rmsnorm` / `rmsnorm` in src/forward/cpu.rs.
__global__ void head_rmsnorm_kernel(float* v, const float* w, int num_heads, int head_dim,
                                    float eps) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    float* head = v + (long)h * head_dim;
    float ss = 0.0f;
    for (int i = 0; i < head_dim; ++i) ss += head[i] * head[i];
    float inv = rsqrtf(ss / (float)head_dim + eps);
    for (int i = 0; i < head_dim; ++i) head[i] = head[i] * inv * w[i];
}

// Grouped-query attention over `positions` cached tokens, in three launches so
// the work spreads over (head, position) rather than one thread per head. The
// one-thread-per-head version walked the whole history in a scalar loop, which
// made every token's cost grow with context: at a few hundred tokens it was the
// dominant cost of a 0.5B model on a GTX 1650.
//
//   1. attn_scores_kernel  — one thread per (head, position): the scaled (and
//                            softcapped) logit.
//   2. attn_softmax_kernel — one block per head: max-subtracted softmax of its
//                            row, in place.
//   3. attn_mix_kernel     — one thread per (head, dim): the weighted sum of the
//                            values.
//
// This is `attention()` in src/forward/cpu.rs step for step: logits, softcap,
// softmax, then accumulate weight × value in position order.
// `sliding_window > 0` bounds attention to the last `sliding_window` positions
// (Mistral); `0` is full causal attention (start = positions - window).
// `scale` is passed in rather than derived from head_dim: Gemma2 decouples it
// (`query_pre_attn_scalar`). `softcap > 0` squashes each logit through
// `tanh(s/cap)*cap` before the softmax (Gemma2); 0 disables it.
// Device KV element type: f32 (exact), or fp16 at half the VRAM (`--kv-quant`
// f16/int8/int4). Attention reads through `load_kv` and appends through
// `kv_append_kernel`, so the choice is one template argument.
#define DLM_KV_F32 0
#define DLM_KV_F16 1
template <int KT>
__device__ __forceinline__ float load_kv(const void* kv, long i);
template <>
__device__ __forceinline__ float load_kv<DLM_KV_F32>(const void* kv, long i) {
    return ((const float*)kv)[i];
}
template <>
__device__ __forceinline__ float load_kv<DLM_KV_F16>(const void* kv, long i) {
    return __half2float(((const __half*)kv)[i]);
}

// The KV cache is paged. A layer's keys (and values) live in one pool of
// fixed-size blocks, `1 << shift` rows each, and a sequence's block table maps
// its logical row `r` to a physical row: block `table[r >> shift]`, offset
// `r & mask`. Sequences take blocks as they grow instead of each reserving a
// full-context buffer. Shifts rather than division: these run once per cached
// position in the attention inner loops.
__device__ __forceinline__ long kv_row(const int* table, int shift, long r) {
    return ((long)table[r >> shift] << shift) | (r & ((1L << shift) - 1));
}

// Write one position's key or value, logical row `row`, into the paged cache.
template <int KT>
__global__ void kv_append_kernel(const float* src, void* kv, const int* table, int shift,
                                 long row, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long at = kv_row(table, shift, row) * n + i;
    if (KT == DLM_KV_F16) ((__half*)kv)[at] = __float2half(src[i]);
    else ((float*)kv)[at] = src[i];
}

template <int KT>
__global__ void attn_scores_kernel(const float* q, const void* keys, const int* table, int shift,
                                   float* scores, int num_heads, int num_kv_heads, int head_dim,
                                   int start, int span, float scale, float softcap) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)num_heads * span) return;
    int h = (int)(idx / span);
    int p = start + (int)(idx % span);
    int kvh = h / (num_heads / num_kv_heads);
    const float* qh = q + h * head_dim;
    long kh = kv_row(table, shift, p) * (num_kv_heads * head_dim) + kvh * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot += qh[d] * load_kv<KT>(keys, kh + d);
    dot *= scale;
    if (softcap > 0.0f) dot = tanhf(dot / softcap) * softcap;
    scores[idx] = dot;
}

// One block per head (launch <<<num_heads, SOFTMAX_THREADS>>>). Threads stride
// the row and tree-reduce the max, then the sum of exponentials, as
// `rmsnorm_kernel` does. The single-thread-per-head version walked every row
// serially three times: at a 2k-token context on Qwen2.5-0.5B that was 10.8 ms
// of a token's 20 ms of attention, and it is 0.7 ms this way. (Scores and mix
// are memory-bound rather than serial-bound. Neither a reduction over positions
// nor scoring each shared GQA key once made them faster.)
#define SOFTMAX_THREADS 64
__global__ void attn_softmax_kernel(float* scores, int num_heads, int span) {
    if (blockIdx.x >= num_heads) return;
    float* row = scores + (long)blockIdx.x * span;
    __shared__ float partial[SOFTMAX_THREADS];
    __shared__ float maxv;
    __shared__ float denom;
    int t = threadIdx.x;

    float m = -1e30f;
    for (int i = t; i < span; i += blockDim.x) if (row[i] > m) m = row[i];
    partial[t] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride && partial[t + stride] > partial[t]) partial[t] = partial[t + stride];
        __syncthreads();
    }
    if (t == 0) maxv = partial[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = t; i < span; i += blockDim.x) {
        row[i] = expf(row[i] - maxv);
        sum += row[i];
    }
    partial[t] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) partial[t] += partial[t + stride];
        __syncthreads();
    }
    if (t == 0) denom = partial[0];
    __syncthreads();

    for (int i = t; i < span; i += blockDim.x) row[i] /= denom;
}

template <int KT>
__global__ void attn_mix_kernel(const float* weights, const void* values, const int* table,
                                int shift, float* ctx,
                                int num_heads, int num_kv_heads, int head_dim,
                                int start, int span) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * head_dim) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    int kvh = h / (num_heads / num_kv_heads);
    const float* row = weights + (long)h * span;
    long off = (long)kvh * head_dim + d;
    long kv_dim = (long)num_kv_heads * head_dim;
    float acc = 0.0f;
    for (int i = 0; i < span; ++i)
        acc += row[i] * load_kv<KT>(values, kv_row(table, shift, start + i) * kv_dim + off);
    ctx[idx] = acc;
}

// x[i] += y[i]  (residual add).
__global__ void add_inplace_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// x[i] += w * y[i] — the residual add for one MoE expert, folding its gate weight
// in so an expert's contribution is scaled without a second pass.
__global__ void scaled_add_kernel(float* x, const float* y, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += w * y[i];
}

// Gated-MLP activation tags — must match `Activation::code()` in src/forward/cpu.rs.
#define DLM_ACT_SILU 0
#define DLM_ACT_GELU_TANH 1

// Gate activation: SiLU (SwiGLU) or tanh-approximate GELU (Gemma's GeGLU).
__device__ __forceinline__ float dlm_activate(float x, int act) {
    if (act == DLM_ACT_GELU_TANH) {
        const float c = 0.7978845608f; // sqrt(2/pi)
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    return x / (1.0f + expf(-x)); // SiLU
}

// out[i] = act(gate[i]) * up[i]
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = dlm_activate(gate[i], act) * up[i];
    }
}

// out[i] = act(x[i]): the ungated MLP's activation (GPT-2, Falcon), where
// SwiGLU would multiply by the up branch.
__global__ void activate_kernel(const float* x, float* out, long n, int act) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dlm_activate(x[i], act);
}

// Copy `n` floats device→device.
__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static inline int grid_for(int n, int block) { return (n + block - 1) / block; }

// Persistent scratch buffers for the decode block. cudaMalloc/cudaFree are
// synchronizing driver calls (each flushes the queue and stalls the pipeline —
// especially under the Windows WDDM driver). The previous code malloc'd and
// freed all 11 scratch buffers on EVERY call — i.e. 11 malloc + 11 free per
// layer per token, ~352 serializing driver calls per token for a 16-layer model
// — which left the GPU idle ~99% of the time and made the GPU path barely beat
// CPU. The scratch sizes are fixed by model geometry, so allocate once and reuse
// across every layer and token; realloc only if a later call needs a bigger
// buffer.
// Keyed by thread *and* device. thread_local because the test harness (and any
// embedder) runs inference on several threads at once, and a single global would
// race. Per device because one thread can drive several GPUs: the multi-GPU
// pipeline runs every stage on the inference thread, calling cudaSetDevice
// before each layer. Scratch keyed by thread alone was allocated on the first
// stage's device and then handed to every later stage's kernels -- device memory
// from another GPU, which a launch cannot address. Each (thread, device) pair
// now allocates its own on that device; the driver reclaims them at exit.
// Slots 0-10 are the dense decode block's scratch. Slots 11-12 are MoE-only:
// 11 holds `normed2` (the FFN input) so it survives from `dlm_moe_attn` across
// the router matvec and every per-expert apply on the same stream; 12 is the
// router/shared-gate matvec output staged for the D2H copy.
// Slots 13-18 are MLA-only: normed, q (nh*qk), c_q (q_lora), kv_a (latent+rope),
// c_kv (latent), and the attention context (nh*v_head_dim).
// Slot 19 holds the attention logits/weights, [num_heads, attended positions].
// Slots 20-21 are the LM head's input hidden and output logits.
// Slots 22-23 are MLA's per-head latent query and latent value mix.
// Slots 24-25 are the int8 activation codes and their block scales (dp4a).
enum { SCRATCH_N = 28 };
enum { Q8_XQ = 24, Q8_XDS = 25, GEMM_SCALE = 26 };   // 26: fp16 tensor-core row scales
enum { ATTN_SCORES = 19, LM_X = 20, LM_LOGITS = 21, MLA_U = 22, MLA_M = 23 };
enum { MOE_NORMED2 = 11, MOE_MATVEC = 12, MOE_NORMED2B = 27 };   // 27: a batch's FFN inputs
enum { MLA_NORMED = 13, MLA_Q = 14, MLA_CQ = 15, MLA_KVA = 16, MLA_CKV = 17, MLA_CTX = 18 };
enum { DLM_MAX_DEVICES = 16 };
static thread_local float* g_scratch_by_device[DLM_MAX_DEVICES][SCRATCH_N] = {{0}};
static thread_local int g_scratch_cap_by_device[DLM_MAX_DEVICES][SCRATCH_N] = {{0}}; // floats

// The device the current entry point runs on, recorded once by DLM_ENTER. Rust
// switches devices only between calls, never during one, so it cannot go stale
// mid-call. Asking the driver on every scratch access instead cost about 30
// `cudaGetDevice` calls per layer per token. -1 if the query failed.
static thread_local int g_cur_dev = 0;
#define DLM_ENTER { int d_ = 0; g_cur_dev = (cudaGetDevice(&d_) == cudaSuccess) ? d_ : -1; }

// The current device's index into the scratch tables. A device id past the table
// is refused by scratch_ensure, which every entry point calls before it reads a
// slot, so the clamp here only keeps an unreachable index in bounds.
static inline int scratch_device() {
    return (g_cur_dev < 0 || g_cur_dev >= DLM_MAX_DEVICES) ? 0 : g_cur_dev;
}
// Every existing `g_scratch[i]` / `g_scratch_cap[i]` reads the current device's row.
#define g_scratch (g_scratch_by_device[scratch_device()])
#define g_scratch_cap (g_scratch_cap_by_device[scratch_device()])

// Ensure scratch slot `i` holds at least `n` floats on the current device;
// (re)allocates only on growth.
static cudaError_t scratch_ensure(int i, int n) {
    if (g_cur_dev < 0 || g_cur_dev >= DLM_MAX_DEVICES) return cudaErrorInvalidDevice;
    if (g_scratch_cap[i] >= n) return cudaSuccess;
    if (g_scratch[i]) cudaFree(g_scratch[i]);
    g_scratch[i] = 0;
    g_scratch_cap[i] = 0;
    cudaError_t e = cudaMalloc(&g_scratch[i], (size_t)n * sizeof(float));
    if (e == cudaSuccess) g_scratch_cap[i] = n;
    return e;
}

static bool quantize_x(const float* x, int n, const signed char** xq, const float2** xds) {
    // Sizes are in floats: n codes take n / 4, n / 32 (scale, sum) pairs n / 16.
    if (scratch_ensure(Q8_XQ, (n + 3) / 4) != cudaSuccess) return false;
    if (scratch_ensure(Q8_XDS, n / 16 + 2) != cudaSuccess) return false;
    *xq = (const signed char*)g_scratch[Q8_XQ];
    *xds = (const float2*)g_scratch[Q8_XDS];
    int blocks = n / 32;
    quantize_q8_kernel<<<(blocks + 7) / 8, dim3(32, 8)>>>(x, n, (signed char*)g_scratch[Q8_XQ],
                                                          (float2*)g_scratch[Q8_XDS]);
    return true;
}

// Block shape beyond the Llama default, for the families that are not
// Llama-descended. Passing NULL means the default: RMSNorm, gated MLP, RoPE,
// sequential residual, no output/MLP/norm biases. Must match `DlmBlockExt` in
// src/forward/gpu.rs.
typedef struct {
    int layer_norm;               // 1: LayerNorm (+ bias) instead of RMSNorm (GPT-2, Falcon)
    int gated;                    // 0: ungated MLP, down(act(up(x))) (GPT-2, Falcon)
    int rope;                     // 0: no rotary (GPT-2's positions are in the embedding)
    int parallel;                 // 1: the FFN reads the block input's norm (Falcon)
    const float* in_norm_bias;    // LayerNorm biases; NULL when absent
    const float* post_norm_bias;
    const float* o_bias;          // output-projection bias
    const float* up_bias;         // MLP biases
    const float* down_bias;
    int kv_half;                  // 1: the KV buffers hold fp16 (DLM_KV_F16)
    const int* kv_table;          // the sequence's block table (single-sequence calls)
    int kv_block_shift;           // rows per KV block = 1 << kv_block_shift
    // Per-projection weight dtype and group size, in the order q, k, v, o, gate,
    // up, down. A safetensors checkpoint stores a whole layer in one dtype, but a
    // GGUF one does not: a Q4_K_M file mixes Q4_K and Q6_K tensors inside the same
    // layer, and decoding a Q6_K matrix as Q4_K reads plausible, wrong weights.
    // `dtype[0] < 0` means "the call's own w_dtype/w_group_size for all of them".
    int w_dtype[7];
    int w_group[7];
} DlmBlockExt;
#define DLM_W_Q 0
#define DLM_W_K 1
#define DLM_W_V 2
#define DLM_W_O 3
#define DLM_W_GATE 4
#define DLM_W_UP 5
#define DLM_W_DOWN 6
static const DlmBlockExt DLM_BLOCK_DEFAULT = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              {-1, -1, -1, -1, -1, -1, -1},
                                              {0, 0, 0, 0, 0, 0, 0}};
// This projection's dtype and group size: the per-tensor pair when the caller
// filled them in, else the call's single dtype.
#define DLM_WDT(i) (o->w_dtype[0] < 0 ? w_dtype : o->w_dtype[(i)])
#define DLM_WGRP(i) (o->w_dtype[0] < 0 ? w_group_size : o->w_group[(i)])

// Launch the three attention kernels for one query over `positions` cached
// tokens. Scores live in scratch slot ATTN_SCORES.
// ── Split-K decode attention (flash-decoding) ───────────────────────────────
//
// The three-kernel attention below reads every key once per *query* head (a GQA
// group of 7 query heads reads each key 7 times), keeps a [heads, span] score
// matrix in global memory, and mixes the values with one thread per output
// element walking the whole history: at a few thousand tokens that walk is a
// long serial loop on a few hundred threads.
//
// Here one block takes one KV head and one chunk of `C` positions: a thread
// scores one position against every query head of the group (the key row read
// once), the chunk's softmax runs in shared memory, and the block mixes its
// values. Chunks are independent, so a long context spreads over many blocks;
// each writes its unnormalized output with its running max and sum, and
// `attn_combine_kernel` merges them with the usual rescaling. A context that
// fits one chunk is normalized in place and needs no second launch.
#define FA_MAX_G 16

// A batched decode's slots: up to DLM_MAX_BATCH sequences, each with its own KV
// cache, block table, history length and position. Passed to kernels by value.
#define DLM_MAX_BATCH 16
typedef struct { float* p[DLM_MAX_BATCH]; } DlmSlots;
typedef struct { const int* p[DLM_MAX_BATCH]; } DlmTables;
typedef struct { int v[DLM_MAX_BATCH]; } DlmInts;

template <int KT>
__device__ __forceinline__ void load_kv8(const void* kv, long i, float* out) {
    if (KT == DLM_KV_F16) {
        uint4 v = *(const uint4*)((const __half*)kv + i);
        unsigned int u[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[2 * j] = half_bits(u[j] & 0xFFFFu);
            out[2 * j + 1] = half_bits(u[j] >> 16);
        }
    } else {
        const float4* p = (const float4*)((const float*)kv + i);
        float4 a = p[0], b = p[1];
        out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
        out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
    }
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Grid (num_kv_heads, splits), block C threads (C >= head_dim, a multiple of 32).
// Shared: G*head_dim (queries) + G*C (scores) + 2G (max, sum) floats.
// `part` is [splits][num_heads][head_dim + 2]: output, then max and sum.
// The body of `attn_split_kernel` for chunk `split` of the span; `alone` when
// it is the only chunk.
template <int KT>
__device__ __forceinline__ void
attn_split_body(const float* __restrict__ q, const void* keys, const void* values, const int* table,
                int shift, int num_heads, int num_kv_heads, int head_dim, int start, int span,
                float scale, float softcap, float* part, float* ctx, int split, bool alone) {
    extern __shared__ float smem[];
    const int G = num_heads / num_kv_heads, kvh = blockIdx.x;
    const int C = blockDim.x, t = threadIdx.x;
    float* qs = smem;
    float* sc = qs + G * head_dim;
    float* ml = sc + G * C;
    for (int i = t; i < G * head_dim; i += C) qs[i] = q[(long)kvh * G * head_dim + i];
    __syncthreads();

    const int p0 = split * C, n = min(C, span - p0);
    const long kv_dim = (long)num_kv_heads * head_dim;
    if (t < n) {
        long kh = kv_row(table, shift, start + p0 + t) * kv_dim + (long)kvh * head_dim;
        float acc[FA_MAX_G];
        #pragma unroll
        for (int g = 0; g < FA_MAX_G; ++g) acc[g] = 0.0f;
        for (int d = 0; d < head_dim; d += 8) {
            float kk[8];
            load_kv8<KT>(keys, kh + d, kk);
            #pragma unroll
            for (int g = 0; g < FA_MAX_G; ++g) {
                if (g < G) {
                    const float* qg = qs + g * head_dim + d;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[g] += qg[j] * kk[j];
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < FA_MAX_G; ++g) {
            if (g < G) {
                float s = acc[g] * scale;
                if (softcap > 0.0f) s = tanhf(s / softcap) * softcap;
                sc[g * C + t] = s;
            }
        }
    }
    __syncthreads();

    // The chunk's softmax statistics, one warp per query head.
    const int warp = t >> 5, lane = t & 31, warps = C >> 5;
    for (int g = warp; g < G; g += warps) {
        float m = -1e30f;
        for (int i = lane; i < n; i += 32) m = fmaxf(m, sc[g * C + i]);
        m = warp_max(m);
        float l = 0.0f;
        for (int i = lane; i < n; i += 32) {
            float e = expf(sc[g * C + i] - m);
            sc[g * C + i] = e;
            l += e;
        }
        l = warp_sum(l);
        if (lane == 0) {
            ml[2 * g] = m;
            ml[2 * g + 1] = l;
        }
    }
    __syncthreads();

    // Values: thread (slice, d) accumulates every head of the group over the
    // positions `slice, slice + P, ...`, so each value element is read once.
    const int P = C / head_dim, d = t % head_dim, slice = t / head_dim;
    float acc[FA_MAX_G];
    #pragma unroll
    for (int g = 0; g < FA_MAX_G; ++g) acc[g] = 0.0f;
    if (slice < P) {
        for (int i = slice; i < n; i += P) {
            float v = load_kv<KT>(values, kv_row(table, shift, start + p0 + i) * kv_dim +
                                              (long)kvh * head_dim + d);
            #pragma unroll
            for (int g = 0; g < FA_MAX_G; ++g)
                if (g < G) acc[g] += sc[g * C + i] * v;
        }
    }
    __syncthreads();   // every read of the weights is done: reuse `sc` for the slices
    if (slice < P) {
        #pragma unroll
        for (int g = 0; g < FA_MAX_G; ++g)
            if (g < G) sc[(slice * G + g) * head_dim + d] = acc[g];
    }
    __syncthreads();
    for (int i = t; i < G * head_dim; i += C) {
        int g = i / head_dim, dd = i % head_dim;
        float o = 0.0f;
        for (int s = 0; s < P; ++s) o += sc[(s * G + g) * head_dim + dd];
        long h = (long)kvh * G + g;
        if (alone) ctx[h * head_dim + dd] = o / ml[2 * g + 1];
        else part[((long)split * num_heads + h) * (head_dim + 2) + dd] = o;
    }
    if (!alone && t < G) {
        long at = ((long)split * num_heads + (long)kvh * G + t) * (head_dim + 2) + head_dim;
        part[at] = ml[2 * t];
        part[at + 1] = ml[2 * t + 1];
    }
}

template <int KT>
__global__ void attn_split_kernel(const float* __restrict__ q, const void* keys, const void* values,
                                  const int* table, int shift, int num_heads, int num_kv_heads,
                                  int head_dim, int start, int span, float scale, float softcap,
                                  float* part, float* ctx) {
    attn_split_body<KT>(q, keys, values, table, shift, num_heads, num_kv_heads, head_dim, start,
                        span, scale, softcap, part, ctx, blockIdx.y, gridDim.y == 1);
}

// The span a decode token at history length `num_positions` attends over.
__host__ __device__ __forceinline__ void attn_span(int num_positions, int sliding_window,
                                                   int* start, int* span) {
    const int positions = num_positions + 1;
    *start = (sliding_window > 0 && positions > sliding_window) ? positions - sliding_window : 0;
    *span = positions - *start;
}

// `attn_split_kernel` for every slot of a batch in one launch: grid
// (num_kv_heads, splits of the longest span, batch). Slot `b` reads its own
// cache, table and history length; chunks past the end of its span exit.
// Its partials go `part_stride` floats into `part`.
template <int KT>
__global__ void attn_split_slots_kernel(const float* __restrict__ q, DlmSlots keys, DlmSlots values,
                                        DlmTables tables, int shift, int num_heads, int num_kv_heads,
                                        int head_dim, DlmInts num_positions, int sliding_window,
                                        float scale, float softcap, float* part, long part_stride,
                                        float* ctx) {
    const int b = blockIdx.z;
    int start, span;
    attn_span(num_positions.v[b], sliding_window, &start, &span);
    const int splits = (span + blockDim.x - 1) / blockDim.x;
    if ((int)blockIdx.y >= splits) return;
    const long q_dim = (long)num_heads * head_dim;
    attn_split_body<KT>(q + b * q_dim, keys.p[b], values.p[b], tables.p[b], shift, num_heads,
                        num_kv_heads, head_dim, start, span, scale, softcap,
                        part + b * part_stride, ctx + b * q_dim, blockIdx.y, splits == 1);
}

__device__ __forceinline__ void attn_combine_body(const float* part, float* ctx, int num_heads,
                                                  int head_dim, int splits) {
    const int h = blockIdx.x;
    const long stride = (long)num_heads * (head_dim + 2);
    const float* p = part + (long)h * (head_dim + 2);
    float m = -1e30f;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, p[s * stride + head_dim]);
    float l = 0.0f;
    for (int s = 0; s < splits; ++s) l += p[s * stride + head_dim + 1] * expf(p[s * stride + head_dim] - m);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < splits; ++s) o += p[s * stride + d] * expf(p[s * stride + head_dim] - m);
        ctx[(long)h * head_dim + d] = o / l;
    }
}

// Merge the chunks of one head: <<<num_heads, 128>>>.
__global__ void attn_combine_kernel(const float* part, float* ctx, int num_heads, int head_dim,
                                    int splits) {
    attn_combine_body(part, ctx, num_heads, head_dim, splits);
}

// `attn_combine_kernel` for every slot: <<<dim3(num_heads, batch), 128>>>. A
// slot whose span fit one chunk was normalized in place and is skipped.
__global__ void attn_combine_slots_kernel(const float* part, long part_stride, float* ctx,
                                          int num_heads, int head_dim, DlmInts num_positions,
                                          int sliding_window, int C) {
    const int b = blockIdx.y;
    int start, span;
    attn_span(num_positions.v[b], sliding_window, &start, &span);
    const int splits = (span + C - 1) / C;
    if (splits > 1)
        attn_combine_body(part + b * part_stride, ctx + (long)b * num_heads * head_dim, num_heads,
                          head_dim, splits);
}

static cudaError_t launch_attention(const float* q, const void* keys, const void* values,
                                    const int* table, int shift, float* ctx, int num_heads,
                                    int num_kv_heads, int head_dim, int positions,
                                    int sliding_window, float scale, float softcap, int kv_half) {
    const int B = 256;
    int start = (sliding_window > 0 && positions > sliding_window) ? positions - sliding_window : 0;
    int span = positions - start;
    // The span grows by one every token; sizing to it exactly would free and
    // re-malloc (two synchronizing driver calls) on every call. Grow in steps.
    // The slot holds the score matrix (old path) or the split partials (new),
    // whichever is larger at the rounded span.
    const int C = head_dim > 128 ? 256 : 128;
    int rounded = (span + 1023) & ~1023;
    long part_need = (long)((rounded + C - 1) / C) * num_heads * (head_dim + 2);
    long need = (long)num_heads * rounded;
    cudaError_t e = scratch_ensure(ATTN_SCORES, (int)(part_need > need ? part_need : need));
    if (e != cudaSuccess) return e;
    float* scores = g_scratch[ATTN_SCORES];

    const int G = num_kv_heads > 0 ? num_heads / num_kv_heads : 0;
    if (G >= 1 && G <= FA_MAX_G && G * num_kv_heads == num_heads && head_dim % 8 == 0 &&
        head_dim <= 256) {
        const int splits = (span + C - 1) / C;
        size_t shmem = (size_t)(G * head_dim + G * C + 2 * G) * sizeof(float);
        dim3 grid(num_kv_heads, splits);
        if (kv_half)
            attn_split_kernel<DLM_KV_F16><<<grid, C, shmem>>>(q, keys, values, table, shift, num_heads,
                                                              num_kv_heads, head_dim, start, span,
                                                              scale, softcap, scores, ctx);
        else
            attn_split_kernel<DLM_KV_F32><<<grid, C, shmem>>>(q, keys, values, table, shift, num_heads,
                                                              num_kv_heads, head_dim, start, span,
                                                              scale, softcap, scores, ctx);
        if (splits > 1) attn_combine_kernel<<<num_heads, 128>>>(scores, ctx, num_heads, head_dim, splits);
        return cudaSuccess;
    }
    if (kv_half)
        attn_scores_kernel<DLM_KV_F16><<<grid_for(num_heads * span, B), B>>>(
            q, keys, table, shift, scores, num_heads, num_kv_heads, head_dim, start, span, scale,
            softcap);
    else
        attn_scores_kernel<DLM_KV_F32><<<grid_for(num_heads * span, B), B>>>(
            q, keys, table, shift, scores, num_heads, num_kv_heads, head_dim, start, span, scale,
            softcap);
    attn_softmax_kernel<<<num_heads, SOFTMAX_THREADS>>>(scores, num_heads, span);
    if (kv_half)
        attn_mix_kernel<DLM_KV_F16><<<grid_for(num_heads * head_dim, B), B>>>(
            scores, values, table, shift, ctx, num_heads, num_kv_heads, head_dim, start, span);
    else
        attn_mix_kernel<DLM_KV_F32><<<grid_for(num_heads * head_dim, B), B>>>(
            scores, values, table, shift, ctx, num_heads, num_kv_heads, head_dim, start, span);
    return cudaSuccess;
}

// Append one position's key and value at row `row` of the cache.
static void launch_kv_append(const float* k, const float* v, void* keys, void* values,
                             const int* table, int shift, long row, int kv_dim, int kv_half) {
    const int B = 256;
    if (kv_half) {
        kv_append_kernel<DLM_KV_F16><<<grid_for(kv_dim, B), B>>>(k, keys, table, shift, row, kv_dim);
        kv_append_kernel<DLM_KV_F16><<<grid_for(kv_dim, B), B>>>(v, values, table, shift, row, kv_dim);
    } else {
        kv_append_kernel<DLM_KV_F32><<<grid_for(kv_dim, B), B>>>(k, keys, table, shift, row, kv_dim);
        kv_append_kernel<DLM_KV_F32><<<grid_for(kv_dim, B), B>>>(v, values, table, shift, row, kv_dim);
    }
}

// ── Fused single-token decode stages ────────────────────────────────────────
//
// A decode step on a small model is launch-bound as much as bandwidth-bound:
// the block above issues ~18 kernels per layer, most of them touching a few KiB.
// These fold the elementwise work into the GEMVs on either side of it:
//
//   * several matrices in one grid (Q, K and V share their input);
//   * an epilogue: `out[o] += v`, the residual add.
//
// Gate and up stay separate GEMVs with the SwiGLU kernel after them. Fused --
// one warp computing both rows -- the int8-activation path lost 40% of batch-1
// decode on Qwen2.5-1.5B int4; with a warp per row and the pair combined in
// shared memory, bf16 merely broke even. Neither beat the three launches.
//
// The norms stay their own (one-block) launch. Folding them into the GEMVs as a
// prologue -- every warp re-deriving the norm and applying the norm weights as
// it read `x` -- cost more than the launch it saved: on Qwen2.5-0.5B the gate/up
// stage then read as many bytes of activations and norm weights from L2 as of
// weights, and batch-1 decode lost 15%.
#define FUSE_PLAIN 0   // out[o] = W·x + b
#define FUSE_RESID 1   // out[o] += W·x + b

typedef struct {
    const void* W[3];
    const float* bias[3];
    float* out[3];
    int rows[3];
    int nseg;
} DlmSegs;

// One row's dot product, reduced across the warp: integer dot products when the
// caller quantized `x` (`xq` set), else the chunked f32 decode.
template <int DT>
__device__ __forceinline__ float fused_row_dot(const void* W, long base, long n, int group_size,
                                               const float* x, const signed char* xq,
                                               const float2* xds, int in_dim, int lane) {
    if (Dp4a<DT>::OK && xq) return q8_row_dot<DT>(W, base, n, group_size, xq, xds, in_dim, lane);
    const int N = Chunk<DT>::N;
    float s = 0.0f;
    for (int i = lane * N; i < in_dim; i += 32 * N) {
        float w[N];
        Chunk<DT>::decode(W, base + i, n, group_size, w);
        s += chunk_dot<N>(w, x + i);
    }
    return warp_sum(s);
}

// Launch <<<ceil(total rows / MV_ROWS), 32 * MV_ROWS>>>; total rows is the sum
// of the segments'.
template <int DT, int MODE>
__global__ void __launch_bounds__(32 * MV_ROWS)
matvec_fused_kernel(DlmSegs s, const float* __restrict__ x, const signed char* __restrict__ xq,
                    const float2* __restrict__ xds, int in_dim, int group_size) {
    int lane = threadIdx.x & 31;
    if (Dp4a<DT>::OK && xq) {
        // Q8Rows<DT>::R rows per warp, as `matvec_q8_kernel`; each row finds
        // its own segment, so a warp's rows may straddle Q, K and V.
        const int R = Q8Rows<DT>::R;
        int total = 0;
        for (int i = 0; i < s.nseg; ++i) total += s.rows[i];
        const int g0 = (blockIdx.x * MV_ROWS + (threadIdx.x >> 5)) * R;
        if (g0 >= total) return;
        int seg[R], row[R];
        long base[R], n[R];
        float acc[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int o = min(g0 + r, total - 1), sg = 0;   // a short tail repeats the last row
            while (sg + 1 < s.nseg && o >= s.rows[sg]) o -= s.rows[sg++];
            seg[r] = sg;
            row[r] = o;
            base[r] = (long)o * in_dim;
            n[r] = (long)s.rows[sg] * in_dim;
            acc[r] = 0.0f;
        }
        for (int i = lane * 32; i < in_dim; i += 32 * 32) {
            const int* xi = (const int*)(xq + i);
            const float2 dx = xds[i >> 5];
            #pragma unroll
            for (int r = 0; r < R; ++r)
                acc[r] += dp4a_dot<DT>(s.W[seg[r]], base[r] + i, n[r], group_size, xi, dx);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = warp_sum(acc[r]);
            if (lane != 0 || g0 + r >= total) continue;
            if (s.bias[seg[r]]) v += s.bias[seg[r]][row[r]];
            if (MODE == FUSE_RESID) s.out[seg[r]][row[r]] += v;
            else s.out[seg[r]][row[r]] = v;
        }
        return;
    }
    int o = blockIdx.x * MV_ROWS + (threadIdx.x >> 5);
    int seg = 0;
    while (seg + 1 < s.nseg && o >= s.rows[seg]) o -= s.rows[seg++];
    if (o >= s.rows[seg]) return;
    long base = (long)o * in_dim;
    long n = (long)s.rows[seg] * in_dim;
    float v = fused_row_dot<DT>(s.W[seg], base, n, group_size, x, xq, xds, in_dim, lane);
    if (s.bias[seg]) v += s.bias[seg][o];
    if (lane == 0) {
        if (MODE == FUSE_RESID) s.out[seg][o] += v;
        else s.out[seg][o] = v;
    }
}

// Launch a fused stage, or return false (launching nothing) when the chunked
// GEMV cannot take this shape: the caller then runs the unfused kernels. Every
// segment must share one dtype and group size.
static bool launch_fused(int mode, int dt, int group_size, const DlmSegs& s, const float* x,
                         int in_dim) {
    int rows = 0;
    for (int i = 0; i < s.nseg; ++i) rows += s.rows[i];
    dim3 block(32 * MV_ROWS);
#define DLM_FUSE(D)                                                                 \
    case D: {                                                                       \
        bool q8 = true;                                                             \
        for (int i = 0; i < s.nseg; ++i) {                                          \
            if (!chunk_fits<D>(s.W[i], x, in_dim, group_size)) return false;        \
            q8 = q8 && q8_fits<D>(s.W[i], x, in_dim, group_size);                   \
        }                                                                           \
        const signed char* xq = 0;                                                  \
        const float2* xds = 0;                                                      \
        if (q8 && !quantize_x(x, in_dim, &xq, &xds)) xq = 0;                        \
        const int per_block = (xq ? (int)Q8Rows<D>::R : 1) * MV_ROWS;               \
        dim3 grid((rows + per_block - 1) / per_block);                              \
        if (mode == FUSE_RESID)                                                     \
            matvec_fused_kernel<D, FUSE_RESID><<<grid, block>>>(s, x, xq, xds, in_dim, group_size); \
        else                                                                        \
            matvec_fused_kernel<D, FUSE_PLAIN><<<grid, block>>>(s, x, xq, xds, in_dim, group_size); \
        return true;                                                                \
    }
    switch (dt) {
        DLM_FUSE(DLM_W_F32)
        DLM_FOR_EACH_DTYPE(DLM_FUSE)
        default:
            return false;
    }
#undef DLM_FUSE
}

// Per-head work between the QKV projection and attention, one warp per head:
// Qwen3's per-head RMSNorm, RoPE, and (for the KV heads) the append of this
// token's K and V into the paged cache. Six launches in the unfused block, two
// of them one thread per head. A lane owns the rotation pairs `(i, i + half)`
// it touches, so the norm and the rotation need no exchange between lanes.
// Launch <<<dim3(num_heads + num_kv_heads, tokens), 32>>>: token `t` (a prefill
// row) is at `position + t`, appends KV row `row + t`, and its q/k/v rows are
// `t` rows into the [tokens, q_dim] / [tokens, kv_dim] blocks.
template <int KT>
__device__ __forceinline__ void
qk_post_body(float* q, float* k, const float* v, const float* q_norm, const float* k_norm,
             int num_heads, int num_kv_heads, int head_dim, int position, const float* inv_freq,
             float mscale, int rope, float eps, void* kv_keys, void* kv_values, const int* table,
             int shift, long row) {
    int h = blockIdx.x, lane = threadIdx.x;
    bool is_k = h >= num_heads;
    int head = is_k ? h - num_heads : h;
    float* vec = (is_k ? k : q) + (long)head * head_dim;
    const float* nw = is_k ? k_norm : q_norm;
    int half = head_dim / 2;
    float inv = 1.0f;
    if (nw) {
        float ss = 0.0f;
        for (int i = lane; i < head_dim; i += 32) ss += vec[i] * vec[i];
        inv = rsqrtf(warp_sum(ss) / (float)head_dim + eps);
    }
    long dst = is_k ? kv_row(table, shift, row) * ((long)num_kv_heads * head_dim) +
                          (long)head * head_dim
                    : 0;
    for (int i = lane; i < half; i += 32) {
        float a = vec[i], b = vec[i + half];
        if (nw) {
            a = a * inv * nw[i];
            b = b * inv * nw[i + half];
        }
        if (rope) {
            float ang = (float)position * inv_freq[i];
            float sn = sinf(ang) * mscale, c = cosf(ang) * mscale;
            float ra = a * c - b * sn, rb = a * sn + b * c;
            a = ra;
            b = rb;
        }
        vec[i] = a;
        vec[i + half] = b;
        if (is_k) {
            if (KT == DLM_KV_F16) {
                ((__half*)kv_keys)[dst + i] = __float2half(a);
                ((__half*)kv_keys)[dst + i + half] = __float2half(b);
            } else {
                ((float*)kv_keys)[dst + i] = a;
                ((float*)kv_keys)[dst + i + half] = b;
            }
        }
    }
    if (is_k) {
        const float* vh = v + (long)head * head_dim;
        for (int i = lane; i < head_dim; i += 32) {
            if (KT == DLM_KV_F16) ((__half*)kv_values)[dst + i] = __float2half(vh[i]);
            else ((float*)kv_values)[dst + i] = vh[i];
        }
    }
}

template <int KT>
__global__ void qk_post_kernel(float* q, float* k, const float* v, const float* q_norm,
                               const float* k_norm, int num_heads, int num_kv_heads,
                               int head_dim, int position, const float* inv_freq, float mscale,
                               int rope, float eps, void* kv_keys, void* kv_values,
                               const int* table, int shift, long row) {
    const int t = blockIdx.y;
    const long kv_dim = (long)num_kv_heads * head_dim;
    qk_post_body<KT>(q + t * (long)num_heads * head_dim, k + t * kv_dim, v + t * kv_dim, q_norm,
                     k_norm, num_heads, num_kv_heads, head_dim, position + t, inv_freq, mscale,
                     rope, eps, kv_keys, kv_values, table, shift, row + t);
}

// `qk_post_kernel` for every slot of a batched decode, slot `b` at its own
// position, appending to its own cache: <<<dim3(num_heads + num_kv_heads, batch), 32>>>.
template <int KT>
__global__ void qk_post_slots_kernel(float* q, float* k, const float* v, const float* q_norm,
                                     const float* k_norm, int num_heads, int num_kv_heads,
                                     int head_dim, DlmInts positions, const float* inv_freq,
                                     float mscale, int rope, float eps, DlmSlots keys,
                                     DlmSlots values, DlmTables tables, int shift,
                                     DlmInts num_positions) {
    const int b = blockIdx.y;
    const long kv_dim = (long)num_kv_heads * head_dim;
    qk_post_body<KT>(q + b * (long)num_heads * head_dim, k + b * kv_dim, v + b * kv_dim, q_norm,
                     k_norm, num_heads, num_kv_heads, head_dim, positions.v[b], inv_freq, mscale,
                     rope, eps, keys.p[b], values.p[b], tables.p[b], shift, num_positions.v[b]);
}

static void launch_qk_post(float* q, float* k, const float* v, const float* q_norm,
                           const float* k_norm, int num_heads, int num_kv_heads, int head_dim,
                           int position, const float* inv_freq, float mscale, int rope, float eps,
                           void* keys, void* values, const int* table, int shift, long row,
                           int kv_half, int tokens = 1) {
    dim3 grid(num_heads + num_kv_heads, tokens);
    if (kv_half)
        qk_post_kernel<DLM_KV_F16><<<grid, 32>>>(q, k, v, q_norm, k_norm, num_heads, num_kv_heads,
                                                 head_dim, position, inv_freq, mscale, rope, eps,
                                                 keys, values, table, shift, row);
    else
        qk_post_kernel<DLM_KV_F32><<<grid, 32>>>(q, k, v, q_norm, k_norm, num_heads, num_kv_heads,
                                                 head_dim, position, inv_freq, mscale, rope, eps,
                                                 keys, values, table, shift, row);
}

// The per-slot half of a batched decode layer's attention, for every slot at
// once: Q/K norm, RoPE and the K/V append (`qk_post`), then split-K attention
// over each slot's own history into `ctx` (`batch * num_heads * head_dim`).
// Three launches whatever the batch. Launching them slot by slot cost about
// 1.5 ms a sequence per step on Qwen2.5-0.5B (24 of a 90 ms step at batch 16),
// nearly all of it launches of grids too small to fill the GPU. Shapes the
// split kernel cannot take still go slot by slot.
static cudaError_t launch_attention_slots(float* q, float* k, const float* v, float* ctx,
                                          const float* q_norm, const float* k_norm, int num_heads,
                                          int num_kv_heads, int head_dim, const float* inv_freq,
                                          float mscale, int rope, float eps, const DlmSlots* keys,
                                          const DlmSlots* values, const DlmTables* tables,
                                          int shift, const DlmInts* num_positions,
                                          const DlmInts* positions, int batch, int sliding_window,
                                          float scale, float softcap, int kv_half) {
    const int q_dim = num_heads * head_dim, kv_dim = num_kv_heads * head_dim;
    const int G = num_kv_heads > 0 ? num_heads / num_kv_heads : 0;
    const bool split_ok = G >= 1 && G <= FA_MAX_G && G * num_kv_heads == num_heads &&
                          head_dim % 8 == 0 && head_dim <= 256;
    if (!split_ok) {
        for (int b = 0; b < batch; ++b) {
            launch_qk_post(q + (long)b * q_dim, k + (long)b * kv_dim, v + (long)b * kv_dim, q_norm,
                           k_norm, num_heads, num_kv_heads, head_dim, positions->v[b], inv_freq,
                           mscale, rope, eps, keys->p[b], values->p[b], tables->p[b], shift,
                           num_positions->v[b], kv_half);
            cudaError_t e = launch_attention(q + (long)b * q_dim, keys->p[b], values->p[b],
                                             tables->p[b], shift, ctx + (long)b * q_dim, num_heads,
                                             num_kv_heads, head_dim, num_positions->v[b] + 1,
                                             sliding_window, scale, softcap, kv_half);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }

    dim3 post(num_heads + num_kv_heads, batch);
    if (kv_half)
        qk_post_slots_kernel<DLM_KV_F16><<<post, 32>>>(q, k, v, q_norm, k_norm, num_heads, num_kv_heads,
                                                      head_dim, *positions, inv_freq, mscale, rope, eps,
                                                      *keys, *values, *tables, shift, *num_positions);
    else
        qk_post_slots_kernel<DLM_KV_F32><<<post, 32>>>(q, k, v, q_norm, k_norm, num_heads, num_kv_heads,
                                                      head_dim, *positions, inv_freq, mscale, rope, eps,
                                                      *keys, *values, *tables, shift, *num_positions);

    // Partials for every slot, each sized to the longest span rounded up as
    // `launch_attention` does, so the slot grows in steps, not every token.
    const int C = head_dim > 128 ? 256 : 128;
    int max_span = 0;
    for (int b = 0; b < batch; ++b) {
        int start, span;
        attn_span(num_positions->v[b], sliding_window, &start, &span);
        max_span = span > max_span ? span : max_span;
    }
    const int splits = (max_span + C - 1) / C;
    const int rounded = (max_span + 1023) & ~1023;
    const long part_stride = (long)((rounded + C - 1) / C) * num_heads * (head_dim + 2);
    cudaError_t e = scratch_ensure(ATTN_SCORES, (int)(part_stride * batch));
    if (e != cudaSuccess) return e;
    float* part = g_scratch[ATTN_SCORES];

    size_t shmem = (size_t)(G * head_dim + G * C + 2 * G) * sizeof(float);
    dim3 grid(num_kv_heads, splits, batch);
    if (kv_half)
        attn_split_slots_kernel<DLM_KV_F16><<<grid, C, shmem>>>(
            q, *keys, *values, *tables, shift, num_heads, num_kv_heads, head_dim, *num_positions,
            sliding_window, scale, softcap, part, part_stride, ctx);
    else
        attn_split_slots_kernel<DLM_KV_F32><<<grid, C, shmem>>>(
            q, *keys, *values, *tables, shift, num_heads, num_kv_heads, head_dim, *num_positions,
            sliding_window, scale, softcap, part, part_stride, ctx);
    if (splits > 1)
        attn_combine_slots_kernel<<<dim3(num_heads, batch), 128>>>(
            part, part_stride, ctx, num_heads, head_dim, *num_positions, sliding_window, C);
    return cudaSuccess;
}

// `kv_keys` / `kv_values` are **persistent** device buffers (capacity
// max_positions * kv_dim) owned by the caller across the whole sequence. This
// call writes the new token's K/V into slot `num_positions` in place and attends
// over the first `num_positions + 1` slots — so the KV history never leaves VRAM
// and only the hidden vector crosses the PCIe bus per token.
extern "C" int dlm_decode_block(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    int inter, float rms_eps,
    int w_dtype,                           // DLM_W_* tag for the projection weights
    int w_group_size,                      // DLM_W_INT4 group size (ignored otherwise)
    const void* q_proj, const void* k_proj, const void* v_proj, const void* o_proj,
    const void* gate_proj, const void* up_proj, const void* down_proj,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias,  // may be NULL
    const float* q_norm, const float* k_norm,  // Qwen3 per-head Q/K RMSNorm; may be NULL
    const float* inv_freq,                 // [head_dim/2], precomputed host-side
    float* x,                              // [hidden] in/out
    float* kv_keys, float* kv_values,      // persistent device KV, mutated in place
    int num_positions, int position,
    int sliding_window,                    // 0 = full causal attention (Mistral SWA otherwise)
    int activation,                        // DLM_ACT_* gate activation (SiLU / GELU)
    float rope_mscale,                     // YaRN attention temperature (1.0 otherwise)
    float attn_scale,                      // <=0: derive 1/sqrt(head_dim) as usual
    float attn_softcap,                    // 0 = off (Gemma2 caps attention logits)
    // Gemma2's extra norm pair; both NULL on every other architecture. When set,
    // `post_norm` normalizes the *attention output* and `pre_ffn_norm` the FFN
    // input, matching `decode_block` in src/forward/cpu.rs.
    const float* pre_ffn_norm, const float* post_ffn_norm,
    const DlmBlockExt* ext)                // NULL: the Llama block shape
{
    DLM_ENTER
    const int B = 256;
    int total_pos = num_positions + 1;
    const DlmBlockExt* o = ext ? ext : &DLM_BLOCK_DEFAULT;
    #define DLM_NORM(src, w, bias, dst) { \
        if (o->layer_norm) layernorm_kernel<<<1, RMS_THREADS>>>((src), (w), (bias), (dst), hidden_size, rms_eps); \
        else rmsnorm_kernel<<<1, RMS_THREADS>>>((src), (w), (dst), hidden_size, rms_eps); }
    if (attn_scale <= 0.0f) attn_scale = rsqrtf((float)head_dim);
    const int gemma2 = (pre_ffn_norm != 0);

    // Persistent scratch (allocated once, reused). Any cudaMalloc can fail (OOM is
    // the common case on a small card); bail out with the real error instead of
    // launching kernels on NULL pointers.
    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(0, hidden_size)
    DLM_ALLOC(1, q_dim)
    DLM_ALLOC(2, kv_dim)
    DLM_ALLOC(3, kv_dim)
    DLM_ALLOC(4, q_dim)
    DLM_ALLOC(5, hidden_size)
    DLM_ALLOC(6, hidden_size)
    DLM_ALLOC(7, inter)
    DLM_ALLOC(8, inter)
    DLM_ALLOC(9, inter)
    DLM_ALLOC(10, hidden_size)
    #undef DLM_ALLOC
    float *normed = g_scratch[0], *q = g_scratch[1], *k = g_scratch[2];
    float *v = g_scratch[3], *ctx = g_scratch[4], *attn_out = g_scratch[5];
    float *normed2 = g_scratch[6], *gate = g_scratch[7], *up = g_scratch[8];
    float *inter_buf = g_scratch[9], *down = g_scratch[10];

    if (e == cudaSuccess) {
        // Each stage below tries its fused form first (see "Fused single-token
        // decode stages") and runs the separate kernels when the layer's shape,
        // dtypes or block family rule it out.
        const bool same_qkv = DLM_WDT(DLM_W_Q) == DLM_WDT(DLM_W_K) && DLM_WDT(DLM_W_Q) == DLM_WDT(DLM_W_V) &&
                              DLM_WGRP(DLM_W_Q) == DLM_WGRP(DLM_W_K) && DLM_WGRP(DLM_W_Q) == DLM_WGRP(DLM_W_V);

        // Attention sublayer: Q/K/V in one grid.
        DLM_NORM(x, in_norm, o->in_norm_bias, normed)
        DlmSegs qkv = {{q_proj, k_proj, v_proj}, {q_bias, k_bias, v_bias}, {q, k, v},
                       {q_dim, kv_dim, kv_dim}, 3};
        bool fused_qkv = same_qkv && launch_fused(FUSE_PLAIN, DLM_WDT(DLM_W_Q), DLM_WGRP(DLM_W_Q),
                                                  qkv, normed, hidden_size);
        if (!fused_qkv) {
            launch_matvec(DLM_WDT(DLM_W_Q), q_proj, normed, q_bias, q, q_dim, hidden_size, DLM_WGRP(DLM_W_Q));
            launch_matvec(DLM_WDT(DLM_W_K), k_proj, normed, k_bias, k, kv_dim, hidden_size, DLM_WGRP(DLM_W_K));
            launch_matvec(DLM_WDT(DLM_W_V), v_proj, normed, v_bias, v, kv_dim, hidden_size, DLM_WGRP(DLM_W_V));
        }
        // Qwen3's per-head Q/K RMSNorm (when present), RoPE, and this token's K/V
        // appended into the persistent history at slot num_positions.
        launch_qk_post(q, k, v, q_norm, k_norm, num_heads, num_kv_heads, head_dim, position, inv_freq,
                       rope_mscale, o->rope, rms_eps, kv_keys, kv_values, o->kv_table,
                       o->kv_block_shift, num_positions, o->kv_half);

        // Attend over history + this token, reading the persistent buffers directly.
        e = launch_attention(q, kv_keys, kv_values, o->kv_table, o->kv_block_shift, ctx, num_heads, num_kv_heads, head_dim, total_pos, sliding_window, attn_scale, attn_softcap, o->kv_half);
        if (e != cudaSuccess) return (int)e;
        // Output projection straight into the residual. Gemma2 norms the
        // attention output first, and Falcon's parallel FFN needs the block input
        // before it changes, so both keep the separate add.
        DlmSegs oseg = {{o_proj, 0, 0}, {o->o_bias, 0, 0}, {x, 0, 0}, {hidden_size, 0, 0}, 1};
        bool fused_o = !gemma2 && !o->parallel &&
                       launch_fused(FUSE_RESID, DLM_WDT(DLM_W_O), DLM_WGRP(DLM_W_O), oseg, ctx,
                                    q_dim);
        if (!fused_o) {
            launch_matvec(DLM_WDT(DLM_W_O), o_proj, ctx, o->o_bias, attn_out, hidden_size, q_dim, DLM_WGRP(DLM_W_O));
            // Gemma2 norms the attention output before the residual add (in place, so
            // the add below is unchanged); elsewhere it goes in raw.
            if (gemma2) rmsnorm_kernel<<<1, RMS_THREADS>>>(attn_out, post_norm, attn_out, hidden_size, rms_eps);
            // Falcon's parallel block: the FFN reads the norm of the block *input*,
            // so take it before the attention output lands in `x`.
            if (o->parallel) DLM_NORM(x, post_norm, o->post_norm_bias, normed2)
            add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);
        }

        // MLP sublayer. Gemma2 uses its dedicated pre-FFN norm here, since
        // `post_norm` was already spent on the attention output.
        if (!o->parallel) {
            if (gemma2) rmsnorm_kernel<<<1, RMS_THREADS>>>(x, pre_ffn_norm, normed2, hidden_size, rms_eps);
            else DLM_NORM(x, post_norm, o->post_norm_bias, normed2)
        }
        launch_matvec(DLM_WDT(DLM_W_UP), up_proj, normed2, o->up_bias, up, inter, hidden_size, DLM_WGRP(DLM_W_UP));
        if (o->gated) {
            launch_matvec(DLM_WDT(DLM_W_GATE), gate_proj, normed2, (const float*)0, gate, inter, hidden_size, DLM_WGRP(DLM_W_GATE));
            swiglu_kernel<<<grid_for(inter, B), B>>>(gate, up, inter_buf, inter, activation);
        } else {
            activate_kernel<<<grid_for(inter, B), B>>>(up, inter_buf, (long)inter, activation);
        }
        // Down projection straight into the residual, except where Gemma2 norms
        // the FFN output first.
        DlmSegs dseg = {{down_proj, 0, 0}, {o->down_bias, 0, 0}, {x, 0, 0}, {hidden_size, 0, 0}, 1};
        bool fused_down = !gemma2 &&
                          launch_fused(FUSE_RESID, DLM_WDT(DLM_W_DOWN), DLM_WGRP(DLM_W_DOWN), dseg,
                                       inter_buf, inter);
        if (!fused_down) {
            launch_matvec(DLM_WDT(DLM_W_DOWN), down_proj, inter_buf, o->down_bias, down, hidden_size, inter, DLM_WGRP(DLM_W_DOWN));
            if (gemma2) rmsnorm_kernel<<<1, RMS_THREADS>>>(down, post_ffn_norm, down, hidden_size, rms_eps);
            add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, down, hidden_size);
        }

        // No blocking cudaDeviceSynchronize here: all kernels run on the in-order
        // default stream, so consecutive decode blocks (and the layers within one)
        // execute in order without a host round-trip. The caller synchronizes once
        // when it needs the result on the host (the D2H copy of the hidden vector
        // after the last layer). A per-block sync here cost ~30ms/layer of pure
        // host-idle stall — the dominant term in per-token latency. We still check
        // cudaGetLastError() to catch launch-time (config) errors synchronously;
        // execution errors surface at the caller's next synchronizing copy.
        e = cudaGetLastError();
    }

    #undef DLM_NORM
    // Scratch is persistent — not freed here. It is reused across every layer and
    // token and reclaimed by the driver at process exit.
    return (int)e;
}

// ── Batched decode block (many sequences, one weight read) ─────────────────
//
// Decoding B sequences by calling `dlm_decode_block` B times reads every weight
// matrix B times. Decode is memory-bound on exactly those reads, so the win here
// is not fusing arithmetic but **streaming each weight row once and using it for
// all B slots**: `matvec_batched_kernel` keeps one accumulator per slot in
// registers and multiplies each weight element into all of them as it is read,
// turning B GEMVs into one GEMM-shaped pass.
//
// Per-slot state that cannot be batched — each sequence has its own KV buffer,
// history length and position — stays per-slot: the norms, RoPE and attention
// read each slot's own cache, in one launch for all slots (`launch_attention_slots`).

// A row's dtype-specific decoding, with everything that is constant across the
// row hoisted out of the element loop: the quantized arms would otherwise redo
// two offset computations and a division by the group size per element (the same
// reason `row_dot` has its own quantized specializations).
template <int DT>
struct RowReader {
    const void* W;
    long n;
    int group_size;
    __device__ __forceinline__ RowReader(const void* w, long n_, int gs)
        : W(w), n(n_), group_size(gs) {}
    __device__ __forceinline__ float at(long e) const { return load_w<DT>(W, e, n, group_size); }
};

template <>
struct RowReader<DLM_W_INT4> {
    const unsigned char* bytes;
    const float* scales;
    const float* zeros;
    int group_size;
    int shift;
    __device__ __forceinline__ RowReader(const void* w, long n, int gs)
        : bytes((const unsigned char*)w), group_size(gs), shift(group_shift(gs)) {
        long code_bytes = (n + 1) / 2;
        scales = (const float*)(bytes + q_scales_off(code_bytes));
        zeros = (const float*)(bytes + q_zeros_off(code_bytes, n, gs));
    }
    __device__ __forceinline__ float at(long e) const {
        unsigned char byte = bytes[e >> 1];
        float code = (float)((e & 1) ? (byte >> 4) : (byte & 0x0F));
        long g = shift >= 0 ? (e >> shift) : e / group_size;
        return (code - zeros[g]) * scales[g];
    }
};

template <>
struct RowReader<DLM_W_INT8> {
    const unsigned char* bytes;
    const float* scales;
    const float* zeros;
    int group_size;
    int shift;
    __device__ __forceinline__ RowReader(const void* w, long n, int gs)
        : bytes((const unsigned char*)w), group_size(gs), shift(group_shift(gs)) {
        scales = (const float*)(bytes + q_scales_off(n));
        zeros = (const float*)(bytes + q_zeros_off(n, n, gs));
    }
    __device__ __forceinline__ float at(long e) const {
        long g = shift >= 0 ? (e >> shift) : e / group_size;
        return ((float)bytes[e] - zeros[g]) * scales[g];
    }
};

// out[b*out_dim + o] = dot(W[o], x + b*in_dim) (+ bias[o]) for all b.
//
// One block per output row, and **each weight element is read from global memory
// once** and multiplied into every slot's accumulator. That is the point of
// batching a memory-bound GEMV: B slots cost one pass over the weights, not B.
//
// The accumulators live in registers, one per slot, which is why the batch loop
// is unrolled over the compile-time maximum with a runtime `b < batch` guard:
// indexing `acc[]` with a runtime value would put it in local memory and give
// back the saving. Each slot's activations are re-read per weight element, but
// `x` is a few KiB and stays in cache — the weight row is the thing too large to
// revisit.
//
// The arithmetic per output is unchanged from the single-slot `row_dot`: the same
// decode, the same values, accumulated over the same strided order.
template <int DT>
__global__ void matvec_batched_kernel(const void* W, const float* x, const float* bias,
                                      float* out, int out_dim, int in_dim, int group_size,
                                      int batch) {
    __shared__ float partial[DLM_MAX_BATCH * (MATVEC_THREADS / 32)];
    int o = blockIdx.x;
    if (o >= out_dim) return;
    long base = (long)o * in_dim;
    long n = (long)out_dim * in_dim;
    RowReader<DT> row(W, n, group_size);

    float acc[DLM_MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < DLM_MAX_BATCH; ++b) acc[b] = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float w = row.at(base + i);
        #pragma unroll
        for (int b = 0; b < DLM_MAX_BATCH; ++b) {
            if (b < batch) acc[b] += w * x[(long)b * in_dim + i];
        }
    }

    // Reduce every slot's accumulator inside its warp with shuffles, which need no
    // barrier, then combine the warps once. A shared-memory tree per slot would
    // cost B barrier rounds per block, and with the weights already read that
    // dominated: on a 896-wide projection each block does 14 rows of work against
    // 48 barrier rounds. Summation order differs slightly from the single-slot
    // kernel's tree, within float rounding.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) / 32;
    #pragma unroll
    for (int b = 0; b < DLM_MAX_BATCH; ++b) {
        if (b < batch) {
            float v = acc[b];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) partial[b * (MATVEC_THREADS / 32) + warp] = v;
        }
    }
    __syncthreads();
    // One thread per slot sums that slot's per-warp partials.
    if ((int)threadIdx.x < batch) {
        float v = 0.0f;
        for (int w = 0; w < warps; ++w) v += partial[threadIdx.x * (MATVEC_THREADS / 32) + w];
        out[(long)threadIdx.x * out_dim + o] = v + (bias ? bias[o] : 0.0f);
    }
}

// The chunked GEMV over a batch: each lane decodes a chunk once and multiplies
// it into every slot's accumulator, so the weights are still read and decoded
// once per batch. Launch as `matvec_chunk_kernel`.
template <int DT>
__global__ void __launch_bounds__(32 * MV_ROWS, 6) matvec_batched_chunk_kernel(const void* __restrict__ W, const float* __restrict__ x,
                                            const float* __restrict__ bias, float* __restrict__ out,
                                            int out_dim, int in_dim, int group_size, int batch) {
    const int N = Chunk<DT>::N;
    int o = blockIdx.x * MV_ROWS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    long n = (long)out_dim * in_dim;
    long base = (long)o * in_dim;
    float acc[DLM_MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < DLM_MAX_BATCH; ++b) acc[b] = 0.0f;
    for (int i = lane * N; i < in_dim; i += 32 * N) {
        float w[N];
        Chunk<DT>::decode(W, base + i, n, group_size, w);
        #pragma unroll
        for (int b = 0; b < DLM_MAX_BATCH; ++b)
            if (b < batch) acc[b] += chunk_dot<N>(w, x + (long)b * in_dim + i);
    }
    float bv = bias ? bias[o] : 0.0f;
    #pragma unroll
    for (int b = 0; b < DLM_MAX_BATCH; ++b) {
        if (b < batch) {
            float v = warp_sum(acc[b]);
            if (lane == 0) out[(long)b * out_dim + o] = v + bv;
        }
    }
}

// The chunked batch GEMV with R weight rows per warp and the slot count fixed
// at compile time (TB, the batch rounded up). The kernel above loads every
// slot's activations once per weight row, eight floats for eight multiply-adds,
// so at batch 16 the load units, not memory, set its speed. Here each loaded
// activation chunk is used for R rows, and a warp's accumulators are R x TB
// registers rather than a fixed sixteen. Launch
// <<<ceil(out_dim / (R * MV_ROWS)), 32 * MV_ROWS>>>.
template <int DT, int R, int TB>
__global__ void __launch_bounds__(32 * MV_ROWS)
matvec_batched_rows_kernel(const void* __restrict__ W, const float* __restrict__ x,
                           const float* __restrict__ bias, float* __restrict__ out, int out_dim,
                           int in_dim, int group_size, int batch) {
    const int N = Chunk<DT>::N;
    const int o0 = (blockIdx.x * MV_ROWS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (o0 >= out_dim) return;
    const long n = (long)out_dim * in_dim;
    float acc[R][TB];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int b = 0; b < TB; ++b) acc[r][b] = 0.0f;
    for (int i = lane * N; i < in_dim; i += 32 * N) {
        float w[R][N];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (o0 + r < out_dim) Chunk<DT>::decode(W, (long)(o0 + r) * in_dim + i, n, group_size, w[r]);
            else
                #pragma unroll
                for (int j = 0; j < N; ++j) w[r][j] = 0.0f;
        }
        #pragma unroll
        for (int b = 0; b < TB; ++b) {
            if (b < batch) {
                const float4* x4 = (const float4*)(x + (long)b * in_dim + i);
                #pragma unroll
                for (int q = 0; q < N / 4; ++q) {
                    const float4 v = x4[q];
                    #pragma unroll
                    for (int r = 0; r < R; ++r)
                        acc[r][b] += w[r][4 * q] * v.x + w[r][4 * q + 1] * v.y +
                                     w[r][4 * q + 2] * v.z + w[r][4 * q + 3] * v.w;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int o = o0 + r;
        const float bv = (bias && o < out_dim) ? bias[o] : 0.0f;
        #pragma unroll
        for (int b = 0; b < TB; ++b) {
            if (b < batch) {
                const float v = warp_sum(acc[r][b]);
                if (lane == 0 && o < out_dim) out[(long)b * out_dim + o] = v + bv;
            }
        }
    }
}

template <int DT, int R, int TB>
static void launch_batched_rows(const void* W, const float* x, const float* bias, float* out,
                                int out_dim, int in_dim, int group_size, int batch) {
    const int per_block = R * MV_ROWS;
    matvec_batched_rows_kernel<DT, R, TB><<<(out_dim + per_block - 1) / per_block, 32 * MV_ROWS>>>(
        W, x, bias, out, out_dim, in_dim, group_size, batch);
}

// Four rows a warp, swept on a GTX 1650 with Qwen2.5-0.5B bf16 (decode tok/s at
// batch 4 / 8 / 16): the one-row kernels 245 / 269 / 290, R = 1 243 / 268 /
// 271, R = 2 276 / 339 / 356, R = 4 287 / 376 / 435-475. R = 8 holds 128
// accumulators at batch 16 and falls back to 359-396 there.
template <int DT>
static void launch_batched_rows_tb(const void* W, const float* x, const float* bias, float* out,
                                   int out_dim, int in_dim, int group_size, int batch) {
    if (batch <= 4) launch_batched_rows<DT, 4, 4>(W, x, bias, out, out_dim, in_dim, group_size, batch);
    else if (batch <= 8) launch_batched_rows<DT, 4, 8>(W, x, bias, out, out_dim, in_dim, group_size, batch);
    else launch_batched_rows<DT, 4, 16>(W, x, bias, out, out_dim, in_dim, group_size, batch);
}

// Up to this many slots the chunked kernel wins; past it the batch is compute
// bound and the one-block-per-row kernel's higher occupancy (about 40 registers
// to the chunked kernel's 80) gives more back. Measured on a GTX 1650, Qwen2.5
// 0.5B Q4_K_M: decode at batch 8 gains a few percent chunked, prefill (16 per
// call) loses 28%. dlm's own int4/int8 always take the per-row kernel: chunked,
// Qwen2.5-1.5B int4 at batch 8 fell from 50 to 30 tok/s.
#define MV_CHUNK_MAX_BATCH 8


// The multi-row kernel, for the formats with 8-weight chunks (f32, bf16, f16);
// false, launching nothing, for any other format or shape.
template <int DT>
static bool try_batched_rows(const void* W, const float* x, const float* bias, float* out,
                             int out_dim, int in_dim, int group_size, int batch) {
    if constexpr (Chunk<DT>::N == 8) {
        if (!chunk_fits<DT>(W, x, in_dim, group_size)) return false;
        launch_batched_rows_tb<DT>(W, x, bias, out, out_dim, in_dim, group_size, batch);
        return true;
    } else {
        return false;
    }
}

// The int8-activation GEMV over a batch. `xq` holds every slot's codes back to
// back (`batch * in_dim`; a slot's row is whole 32-value blocks, `in_dim % 32 ==
// 0`). A warp decodes R rows' chunks into registers once and dots them with
// every slot's activation block, so the weights cross memory and are decoded
// once per batch. The kernel before this one took one row a warp and decoded
// each chunk again for every slot, and at batch 16 that decoding, not memory,
// set its speed: about 250 tok/s on Qwen2.5-0.5B in any quantized format,
// barely above batch 4. The slot count is fixed at compile time (TB, the batch
// rounded up) so the R x TB accumulators stay in registers. A short tail
// repeats the last row, as `matvec_q8_kernel` does.
// <<<ceil(out_dim / (R * MV_ROWS)), 32 * MV_ROWS>>>.
template <int DT, int R, int TB>
__global__ void __launch_bounds__(32 * MV_ROWS)
matvec_q8_batched_rows_kernel(const void* __restrict__ W, const signed char* __restrict__ xq,
                              const float2* __restrict__ xds, const float* __restrict__ bias,
                              float* __restrict__ out, int out_dim, int in_dim, int group_size,
                              int batch) {
    const int o0 = (blockIdx.x * MV_ROWS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (o0 >= out_dim) return;
    const long n = (long)out_dim * in_dim;
    float acc[R][TB];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int b = 0; b < TB; ++b) acc[r][b] = 0.0f;
    for (int i = lane * 32; i < in_dim; i += 32 * 32) {
        Q8Codes c[R];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            Dp4a<DT>::load(W, (long)min(o0 + r, out_dim - 1) * in_dim + i, n, group_size, c[r]);
        #pragma unroll
        for (int b = 0; b < TB; ++b) {
            if (b < batch) {
                const long at = (long)b * in_dim + i;
                const int4* xv = (const int4*)(xq + at);
                const int4 x0 = xv[0], x1 = xv[1];
                const int xi[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                const float2 dx = xds[at >> 5];
                int xs0 = 0, xs1 = 0;
                if (Dp4a<DT>::SPLIT) q8_half_sums(xi, xs0, xs1);
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[r][b] += q8_apply<(bool)Dp4a<DT>::SPLIT>(c[r], xi, dx, xs0, xs1);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int o = o0 + r;
        const float bv = (bias && o < out_dim) ? bias[o] : 0.0f;
        #pragma unroll
        for (int b = 0; b < TB; ++b) {
            if (b < batch) {
                const float v = warp_sum(acc[r][b]);
                if (lane == 0 && o < out_dim) out[(long)b * out_dim + o] = v + bv;
            }
        }
    }
}

template <int DT, int R, int TB>
static void launch_q8_batched_rows(const void* W, const signed char* xq, const float2* xds,
                                   const float* bias, float* out, int out_dim, int in_dim,
                                   int group_size, int batch) {
    const int per_block = R * MV_ROWS;
    matvec_q8_batched_rows_kernel<DT, R, TB><<<(out_dim + per_block - 1) / per_block, 32 * MV_ROWS>>>(
        W, xq, xds, bias, out, out_dim, in_dim, group_size, batch);
}

// Four rows a warp for every format. Swept on a GTX 1650 (decode tok/s at
// batch 16, chunk decoded per slot -> R = 1 / 2 / 4): Qwen2.5-0.5B Q4_K_M
// 233 -> 589 / 669 / 691, Q8_0 249 -> 665 / 815 / 791, Q6_K 258 -> 528 / 596 /
// 609, int8 245 -> 799 / 812 (R = 2 / 4), int4 259 -> 851 / 879; Qwen2.5-1.5B
// Q4_K_M 91 -> 265 / 270, Q8_0 83 -> 310 / 309. At batch 4 and 8, R = 4 led or
// tied in all but Q6_K at batch 8 (436 at R = 2, 417 at 4).
template <int DT>
static void launch_q8_batched(const void* W, const signed char* xq, const float2* xds,
                              const float* bias, float* out, int out_dim, int in_dim,
                              int group_size, int batch) {
    if (batch <= 4) launch_q8_batched_rows<DT, 4, 4>(W, xq, xds, bias, out, out_dim, in_dim, group_size, batch);
    else if (batch <= 8) launch_q8_batched_rows<DT, 4, 8>(W, xq, xds, bias, out, out_dim, in_dim, group_size, batch);
    else launch_q8_batched_rows<DT, 4, 16>(W, xq, xds, bias, out, out_dim, in_dim, group_size, batch);
}

static void launch_matvec_batched(int dt, const void* W, const float* x, const float* bias,
                                  float* out, int out_dim, int in_dim, int group_size, int batch) {
    const signed char* xq;
    const float2* xds;
#define DLM_CASE(D) case D: \
        if (q8_fits<D>(W, x, in_dim, group_size) && quantize_x(x, batch * in_dim, &xq, &xds)) \
            launch_q8_batched<D>(W, xq, xds, bias, out, out_dim, in_dim, group_size, batch); \
        else if (try_batched_rows<D>(W, x, bias, out, out_dim, in_dim, group_size, batch)) {} \
        else if (batch <= MV_CHUNK_MAX_BATCH && D != DLM_W_INT4 && D != DLM_W_INT8 && \
            chunk_fits<D>(W, x, in_dim, group_size)) \
            matvec_batched_chunk_kernel<D><<<(out_dim + MV_ROWS - 1) / MV_ROWS, 32 * MV_ROWS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch); \
        else \
            matvec_batched_kernel<D><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch); \
        break;
    switch (dt) {
        DLM_FOR_EACH_DTYPE(DLM_CASE)
        default:
            if (try_batched_rows<DLM_W_F32>(W, x, bias, out, out_dim, in_dim, group_size, batch)) {}
            else if (batch <= MV_CHUNK_MAX_BATCH && chunk_fits<DLM_W_F32>(W, x, in_dim, group_size))
                matvec_batched_chunk_kernel<DLM_W_F32><<<(out_dim + MV_ROWS - 1) / MV_ROWS, 32 * MV_ROWS>>>(
                    W, x, bias, out, out_dim, in_dim, group_size, batch);
            else
                matvec_batched_kernel<DLM_W_F32>
                    <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch);
            break;
    }
#undef DLM_CASE
}

// rmsnorm over B rows: one block per row (grid.x = batch).
__global__ void rmsnorm_batched_kernel(const float* x, const float* w, float* out, int n,
                                       float eps) {
    __shared__ float partial[RMS_THREADS];
    __shared__ float inv_rms;
    const float* xb = x + (long)blockIdx.x * n;
    float* ob = out + (long)blockIdx.x * n;

    float ss = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) ss += xb[k] * xb[k];
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_rms = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) ob[i] = xb[i] * inv_rms * w[i];
}

// LayerNorm over B rows: one block per row, as rmsnorm_batched_kernel.
__global__ void layernorm_batched_kernel(const float* x, const float* w, const float* b,
                                         float* out, int n, float eps) {
    __shared__ float partial[RMS_THREADS];
    __shared__ float mean;
    __shared__ float inv_std;
    const float* xb = x + (long)blockIdx.x * n;
    float* ob = out + (long)blockIdx.x * n;

    float s = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) s += xb[k];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) mean = partial[0] / (float)n;
    __syncthreads();

    float ss = 0.0f;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        float d = xb[k] - mean;
        ss += d * d;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_std = rsqrtf(partial[0] / (float)n + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ob[i] = (xb[i] - mean) * inv_std * w[i] + (b ? b[i] : 0.0f);
}

// SwiGLU/GeGLU over the whole [batch, inter] plane.
__global__ void swiglu_batched_kernel(const float* gate, const float* up, float* out, long n,
                                      int act) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dlm_activate(gate[i], act) * up[i];
}

// x[i] += y[i] over the whole [batch, n] plane.
__global__ void add_inplace_batched_kernel(float* x, const float* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// One decode block for `batch` sequences at once. `x` is a contiguous
// [batch, hidden_size] device block; each slot keeps its own KV buffers, history
// length and RoPE position. Op-for-op identical to calling `dlm_decode_block`
// per slot — only the weight traffic changes. `batch <= DLM_MAX_BATCH`.
extern "C" int dlm_decode_block_batched(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    int inter, float rms_eps, int w_dtype, int w_group_size,
    const void* q_proj, const void* k_proj, const void* v_proj, const void* o_proj,
    const void* gate_proj, const void* up_proj, const void* down_proj,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* q_norm, const float* k_norm,
    const float* inv_freq,
    float* x,
    const DlmSlots* kv_keys, const DlmSlots* kv_values,
    const DlmTables* kv_tables,            // each slot's block table
    const DlmInts* num_positions, const DlmInts* positions,
    int batch,
    int sliding_window, int activation, float rope_mscale,
    float attn_scale, float attn_softcap,
    const float* pre_ffn_norm, const float* post_ffn_norm,
    const DlmBlockExt* ext)
{
    DLM_ENTER
    const int B = 256;
    if (batch <= 0) return 0;
    const DlmBlockExt* o = ext ? ext : &DLM_BLOCK_DEFAULT;
    #define DLM_NORM_B(src, w, bias, dst) { \
        if (o->layer_norm) layernorm_batched_kernel<<<batch, RMS_THREADS>>>((src), (w), (bias), (dst), hidden_size, rms_eps); \
        else rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>((src), (w), (dst), hidden_size, rms_eps); }
    if (batch > DLM_MAX_BATCH) return (int)cudaErrorInvalidValue;
    if (attn_scale <= 0.0f) attn_scale = rsqrtf((float)head_dim);
    const int gemma2 = (pre_ffn_norm != 0);

    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(0, batch * hidden_size)
    DLM_ALLOC(1, batch * q_dim)
    DLM_ALLOC(2, batch * kv_dim)
    DLM_ALLOC(3, batch * kv_dim)
    DLM_ALLOC(4, batch * q_dim)
    DLM_ALLOC(5, batch * hidden_size)
    DLM_ALLOC(6, batch * hidden_size)
    DLM_ALLOC(7, batch * inter)
    DLM_ALLOC(8, batch * inter)
    DLM_ALLOC(9, batch * inter)
    DLM_ALLOC(10, batch * hidden_size)
    #undef DLM_ALLOC
    if (e != cudaSuccess) return (int)e;
    float *normed = g_scratch[0], *q = g_scratch[1], *k = g_scratch[2];
    float *v = g_scratch[3], *ctx = g_scratch[4], *attn_out = g_scratch[5];
    float *normed2 = g_scratch[6], *gate = g_scratch[7], *up = g_scratch[8];
    float *inter_buf = g_scratch[9], *down = g_scratch[10];

    // Attention sublayer. Projections are batched (one weight read for all
    // slots); everything downstream of them is per-slot state.
    DLM_NORM_B(x, in_norm, o->in_norm_bias, normed)
    launch_matvec_batched(DLM_WDT(DLM_W_Q), q_proj, normed, q_bias, q, q_dim, hidden_size, DLM_WGRP(DLM_W_Q), batch);
    launch_matvec_batched(DLM_WDT(DLM_W_K), k_proj, normed, k_bias, k, kv_dim, hidden_size, DLM_WGRP(DLM_W_K), batch);
    launch_matvec_batched(DLM_WDT(DLM_W_V), v_proj, normed, v_bias, v, kv_dim, hidden_size, DLM_WGRP(DLM_W_V), batch);

    e = launch_attention_slots(q, k, v, ctx, q_norm, k_norm, num_heads, num_kv_heads, head_dim,
                               inv_freq, rope_mscale, o->rope, rms_eps, kv_keys, kv_values,
                               kv_tables, o->kv_block_shift, num_positions, positions, batch,
                               sliding_window, attn_scale, attn_softcap, o->kv_half);
    if (e != cudaSuccess) return (int)e;

    launch_matvec_batched(DLM_WDT(DLM_W_O), o_proj, ctx, o->o_bias, attn_out, hidden_size, q_dim, DLM_WGRP(DLM_W_O), batch);
    if (gemma2) {
        rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(attn_out, post_norm, attn_out, hidden_size, rms_eps);
    }
    if (o->parallel) DLM_NORM_B(x, post_norm, o->post_norm_bias, normed2)
    add_inplace_batched_kernel<<<grid_for(batch * hidden_size, B), B>>>(x, attn_out, (long)batch * hidden_size);

    // MLP sublayer — all batched.
    if (!o->parallel) {
        if (gemma2) rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(x, pre_ffn_norm, normed2, hidden_size, rms_eps);
        else DLM_NORM_B(x, post_norm, o->post_norm_bias, normed2)
    }
    launch_matvec_batched(DLM_WDT(DLM_W_UP), up_proj, normed2, o->up_bias, up, inter, hidden_size, DLM_WGRP(DLM_W_UP), batch);
    if (o->gated) {
        launch_matvec_batched(DLM_WDT(DLM_W_GATE), gate_proj, normed2, (const float*)0, gate, inter, hidden_size, DLM_WGRP(DLM_W_GATE), batch);
        swiglu_batched_kernel<<<grid_for(batch * inter, B), B>>>(gate, up, inter_buf, (long)batch * inter, activation);
    } else {
        activate_kernel<<<grid_for(batch * inter, B), B>>>(up, inter_buf, (long)batch * inter, activation);
    }
    launch_matvec_batched(DLM_WDT(DLM_W_DOWN), down_proj, inter_buf, o->down_bias, down, hidden_size, inter, DLM_WGRP(DLM_W_DOWN), batch);
    if (gemma2) {
        rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(down, post_ffn_norm, down, hidden_size, rms_eps);
    }
    add_inplace_batched_kernel<<<grid_for(batch * hidden_size, B), B>>>(x, down, (long)batch * hidden_size);
    #undef DLM_NORM_B
    return (int)cudaGetLastError();
}

// ── Prefill: a prompt chunk through a layer as GEMMs ────────────────────────
//
// Prefill used to run the decode block 16 tokens at a time, so every weight was
// read and decoded once per 16 prompt tokens, and attention ran one launch per
// token. Here a chunk of up to DLM_PREFILL_T tokens of one sequence goes through
// a layer at once: the projections are tiled GEMMs that decode each weight tile
// into shared memory once for 64 tokens, and attention is one launch for every
// query of the chunk.
//
// Each output is still a sum over the input dimension in a fixed order (one
// 32-wide tile after another), whatever the chunk size, so a prompt prefilled
// in pieces gives exactly the result of prefilling it whole.
#define GEMM_BM 64      // weight rows per block
#define GEMM_BT 64      // tokens per block
#define GEMM_BK 32      // input elements per tile step
#define DLM_PREFILL_T 512

// Y[t, m] (+)= sum_k W[m, k] * X[t, k] (+ bias[m]), for t < T, m < M.
// <<<dim3(ceil(M / 64), ceil(T / 64)), 256>>>; `K % 32 == 0` (see gemm_fits).
template <int DT, int ACC>
__global__ void __launch_bounds__(256)
gemm_kernel(const void* __restrict__ W, const float* __restrict__ X, const float* __restrict__ bias,
            float* __restrict__ Y, int M, int K, int T, int group_size) {
    const int N = Chunk<DT>::N;
    __shared__ float Ws[GEMM_BK][GEMM_BM + 4];
    __shared__ float Xs[GEMM_BK][GEMM_BT + 4];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * GEMM_BM, t0 = blockIdx.y * GEMM_BT;
    const long n = (long)M * K;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        // Weight tile: 64 rows x 32 inputs, decoded a chunk per thread.
        for (int c = tid; c < GEMM_BM * (GEMM_BK / N); c += 256) {
            int r = c / (GEMM_BK / N), kc = (c % (GEMM_BK / N)) * N;
            float w[N];
            if (m0 + r < M) Chunk<DT>::decode(W, (long)(m0 + r) * K + k0 + kc, n, group_size, w);
            else for (int j = 0; j < N; ++j) w[j] = 0.0f;
            #pragma unroll
            for (int j = 0; j < N; ++j) Ws[kc + j][r] = w[j];
        }
        // Activation tile: 64 tokens x 32 inputs, eight floats per thread.
        {
            int t = tid >> 2, kk = (tid & 3) * 8;
            float4 a = make_float4(0, 0, 0, 0), b = a;
            if (t0 + t < T) {
                const float4* p = (const float4*)(X + (long)(t0 + t) * K + k0 + kk);
                a = p[0];
                b = p[1];
            }
            Xs[kk][t] = a.x; Xs[kk + 1][t] = a.y; Xs[kk + 2][t] = a.z; Xs[kk + 3][t] = a.w;
            Xs[kk + 4][t] = b.x; Xs[kk + 5][t] = b.y; Xs[kk + 6][t] = b.z; Xs[kk + 7][t] = b.w;
        }
        __syncthreads();
        #pragma unroll 8
        for (int kk = 0; kk < GEMM_BK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = Ws[kk][ty * 4 + i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Xs[kk][tx * 4 + j];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = m0 + ty * 4 + i;
        if (m >= M) continue;
        float bv = bias ? bias[m] : 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int t = t0 + tx * 4 + j;
            if (t >= T) continue;
            float v = acc[i][j] + bv;
            float* y = Y + (long)t * M + m;
            if (ACC) *y += v;
            else *y = v;
        }
    }
}

template <int DT>
static inline bool gemm_fits(const void* W, const float* X, int K, int group_size) {
    return K % GEMM_BK == 0 && chunk_fits<DT>(W, X, K, group_size);
}

// ── The same GEMM on tensor cores ──────────────────────────────────────────
//
// Volta and every RTX card multiply 16x16x16 tiles of 16-bit inputs into f32
// accumulators in one warp instruction (WMMA). The weight tile is decoded
// exactly as above and rounded to 16 bits in shared memory; the products sum in
// f32. Ampere and later take bf16 inputs, whose range is f32's. Volta and Turing
// have only fp16, whose range ends at 65504 -- below what some models' FFN
// activations reach -- so there each token row is first divided by a power of
// two that brings its largest value under 2^14, and the output multiplied back.
//
// A GTX 16-series card is Turing without tensor cores: there the instructions
// exist but run on the FP16 units, slower than the f32 kernel, so `gemm_mode`
// keeps it on the f32 kernel. `DLM_GEMM=simt|tc` overrides the choice.
#ifndef __HIP_PLATFORM_AMD__
#include <mma.h>
#include <cuda_bf16.h>
#include <type_traits>

// Per token row: the power of two to divide by for fp16 (1 when it fits).
__global__ void row_scale_kernel(const float* X, int K, float* scale) {
    const float* row = X + (long)blockIdx.x * K;
    float m = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(row[i]));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (threadIdx.x == 0) scale[blockIdx.x] = m > 16384.0f ? exp2f(ceilf(log2f(m / 16384.0f))) : 1.0f;
}

template <typename H> __device__ __forceinline__ H to_h(float v);
template <> __device__ __forceinline__ __half to_h<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 to_h<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

#define TC_PAD 8
// <<<dim3(ceil(M / 64), ceil(T / 64)), 128>>>: four warps, each 32 tokens x 32
// rows (2 x 2 fragments). `scale` is null for bf16.
template <int DT, int ACC, typename H>
__device__ __forceinline__ void
gemm_tc_body(const void* __restrict__ W, const float* __restrict__ X, const float* __restrict__ bias,
             float* __restrict__ Y, int M, int K, int T, int group_size, const float* scale) {
    using namespace nvcuda;
    const int N = Chunk<DT>::N;
    __shared__ __align__(32) H Xh[GEMM_BT][GEMM_BK + TC_PAD];
    __shared__ __align__(32) H Wh[GEMM_BM][GEMM_BK + TC_PAD];
    __shared__ __align__(32) float Cs[GEMM_BT][GEMM_BM + 4];
    const int tid = threadIdx.x, warp = tid >> 5;
    const int m0 = blockIdx.x * GEMM_BM, t0 = blockIdx.y * GEMM_BT;
    const int wt = (warp >> 1) * 32, wm = (warp & 1) * 32;   // the warp's corner
    const long n = (long)M * K;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(c[i][j], 0.0f);
    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
        for (int ch = tid; ch < GEMM_BM * (GEMM_BK / N); ch += 128) {
            int r = ch / (GEMM_BK / N), kc = (ch % (GEMM_BK / N)) * N;
            float w[N];
            if (m0 + r < M) Chunk<DT>::decode(W, (long)(m0 + r) * K + k0 + kc, n, group_size, w);
            else for (int j = 0; j < N; ++j) w[j] = 0.0f;
            #pragma unroll
            for (int j = 0; j < N; ++j) Wh[r][kc + j] = to_h<H>(w[j]);
        }
        for (int i = tid; i < GEMM_BT * GEMM_BK; i += 128) {
            int t = i / GEMM_BK, kk = i % GEMM_BK;
            float v = 0.0f;
            if (t0 + t < T) {
                v = X[(long)(t0 + t) * K + k0 + kk];
                if (scale) v /= scale[t0 + t];
            }
            Xh[t][kk] = to_h<H>(v);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GEMM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, H, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, H, wmma::col_major> b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &Xh[wt + 16 * i][kk], GEMM_BK + TC_PAD);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Wh[wm + 16 * j][kk], GEMM_BK + TC_PAD);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[wt + 16 * i][wm + 16 * j], c[i][j], GEMM_BM + 4, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < GEMM_BT * GEMM_BM; i += 128) {
        int t = i / GEMM_BM, r = i % GEMM_BM;
        if (t0 + t >= T || m0 + r >= M) continue;
        float v = Cs[t][r] * (scale ? scale[t0 + t] : 1.0f) + (bias ? bias[m0 + r] : 0.0f);
        float* y = Y + (long)(t0 + t) * M + m0 + r;
        if (ACC) *y += v;
        else *y = v;
    }
}

// bf16 fragments exist from sm_80 and fp16 ones from sm_70; each instantiation
// is compiled only for the architectures that have it (build.rs builds every
// tier). `gemm_mode` picks bf16 only on sm_80+, so the trap is reached only by a
// binary built without that tier -- loudly, instead of returning zeros.
template <int DT, int ACC, typename H>
__global__ void __launch_bounds__(128)
gemm_tc_kernel(const void* __restrict__ W, const float* __restrict__ X, const float* __restrict__ bias,
               float* __restrict__ Y, int M, int K, int T, int group_size, const float* scale) {
#if __CUDA_ARCH__ >= 800
    gemm_tc_body<DT, ACC, H>(W, X, bias, Y, M, K, T, group_size, scale);
#elif __CUDA_ARCH__ >= 700
    if constexpr (std::is_same<H, __half>::value)
        gemm_tc_body<DT, ACC, H>(W, X, bias, Y, M, K, T, group_size, scale);
    else
        __trap();
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}
#endif

// 0: the f32 kernel, 1: tensor cores with bf16, 2: tensor cores with fp16.
static int gemm_mode() {
#ifdef __HIP_PLATFORM_AMD__
    return 0;
#else
    static int mode_by_device[DLM_MAX_DEVICES];
    static bool known[DLM_MAX_DEVICES];
    int dev = scratch_device();
    if (known[dev]) return mode_by_device[dev];
    int mode = 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
        bool has_tc = p.major >= 8 || (p.major == 7 && !strstr(p.name, "GTX 16"));
        if (has_tc) mode = p.major >= 8 ? 1 : 2;
        const char* want = getenv("DLM_GEMM");
        if (want && !strcmp(want, "simt")) mode = 0;
        if (want && !strcmp(want, "tc") && p.major >= 7) mode = p.major >= 8 ? 1 : 2;
    }
    mode_by_device[dev] = mode;
    known[dev] = true;
    return mode;
#endif
}

// Which GEMM prefill runs on the current device: 0 the f32 kernel, 1 bf16 and 2
// fp16 tensor cores. For tests and the startup banner.
extern "C" int dlm_gemm_mode() {
    DLM_ENTER
    return gemm_mode();
}

// Launch the GEMM, or return false (launching nothing) for a shape it cannot
// take; `acc` adds into Y instead of storing.
static bool launch_gemm(int dt, const void* W, const float* X, const float* bias, float* Y, int M,
                        int K, int T, int group_size, bool acc) {
    dim3 grid((M + GEMM_BM - 1) / GEMM_BM, (T + GEMM_BT - 1) / GEMM_BT);
#ifndef __HIP_PLATFORM_AMD__
    const int mode = gemm_mode();
    const float* scale = 0;
    if (mode == 2) {
        if (scratch_ensure(GEMM_SCALE, T) != cudaSuccess) return false;
        scale = g_scratch[GEMM_SCALE];
    }
#define DLM_GEMM_TC(D)                                                                              \
        if (mode == 1) {                                                                            \
            if (acc) gemm_tc_kernel<D, 1, __nv_bfloat16><<<grid, 128>>>(W, X, bias, Y, M, K, T, group_size, 0); \
            else gemm_tc_kernel<D, 0, __nv_bfloat16><<<grid, 128>>>(W, X, bias, Y, M, K, T, group_size, 0);     \
            return true;                                                                            \
        }                                                                                           \
        if (mode == 2) {                                                                            \
            row_scale_kernel<<<T, 32>>>(X, K, (float*)scale);                                       \
            if (acc) gemm_tc_kernel<D, 1, __half><<<grid, 128>>>(W, X, bias, Y, M, K, T, group_size, scale); \
            else gemm_tc_kernel<D, 0, __half><<<grid, 128>>>(W, X, bias, Y, M, K, T, group_size, scale);     \
            return true;                                                                            \
        }
#else
#define DLM_GEMM_TC(D)
#endif
#define DLM_GEMM(D)                                                                     \
    case D:                                                                             \
        if (!gemm_fits<D>(W, X, K, group_size)) return false;                           \
        DLM_GEMM_TC(D)                                                                  \
        if (acc) gemm_kernel<D, 1><<<grid, 256>>>(W, X, bias, Y, M, K, T, group_size);  \
        else gemm_kernel<D, 0><<<grid, 256>>>(W, X, bias, Y, M, K, T, group_size);      \
        return true;
    switch (dt) {
        DLM_GEMM(DLM_W_F32)
        DLM_FOR_EACH_DTYPE(DLM_GEMM)
        default:
            return false;
    }
#undef DLM_GEMM
#undef DLM_GEMM_TC
}

// Causal attention for a prefill chunk: one block per (KV head, query token),
// walking the query's history in chunks of C positions with a running softmax
// (the split kernel's math, merged as it goes instead of across blocks). Query
// `t` is at KV row `history + t` and sees rows up to and including it.
// <<<dim3(num_kv_heads, T), C>>>, shared as attn_split_kernel's.
template <int KT>
__global__ void attn_prefill_kernel(const float* __restrict__ q, const void* keys,
                                    const void* values, const int* table, int shift,
                                    int num_heads, int num_kv_heads, int head_dim, int history,
                                    int sliding_window, float scale, float softcap, float* ctx) {
    extern __shared__ float smem[];
    const int G = num_heads / num_kv_heads, kvh = blockIdx.x, tq = blockIdx.y;
    const int C = blockDim.x, t = threadIdx.x;
    const long q_dim = (long)num_heads * head_dim, kv_dim = (long)num_kv_heads * head_dim;
    float* qs = smem;
    float* sc = qs + G * head_dim;
    float* ml = sc + G * C;            // per head: running max, running sum, rescale
    for (int i = t; i < G * head_dim; i += C) qs[i] = q[tq * q_dim + (long)kvh * G * head_dim + i];
    if (t < G) {
        ml[3 * t] = -1e30f;
        ml[3 * t + 1] = 0.0f;
    }
    const int total = history + tq + 1;
    const int start = (sliding_window > 0 && total > sliding_window) ? total - sliding_window : 0;
    const int P = C / head_dim, d = t % head_dim, slice = t / head_dim;
    const int warp = t >> 5, lane = t & 31, warps = C >> 5;
    float acc[FA_MAX_G];
    #pragma unroll
    for (int g = 0; g < FA_MAX_G; ++g) acc[g] = 0.0f;
    __syncthreads();

    for (int p0 = start; p0 < total; p0 += C) {
        const int n = min(C, total - p0);
        if (t < n) {
            long kh = kv_row(table, shift, p0 + t) * kv_dim + (long)kvh * head_dim;
            float s[FA_MAX_G];
            #pragma unroll
            for (int g = 0; g < FA_MAX_G; ++g) s[g] = 0.0f;
            for (int dd = 0; dd < head_dim; dd += 8) {
                float kk[8];
                load_kv8<KT>(keys, kh + dd, kk);
                #pragma unroll
                for (int g = 0; g < FA_MAX_G; ++g) {
                    if (g < G) {
                        const float* qg = qs + g * head_dim + dd;
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) s[g] += qg[j] * kk[j];
                    }
                }
            }
            #pragma unroll
            for (int g = 0; g < FA_MAX_G; ++g) {
                if (g < G) {
                    float v = s[g] * scale;
                    if (softcap > 0.0f) v = tanhf(v / softcap) * softcap;
                    sc[g * C + t] = v;
                }
            }
        }
        __syncthreads();
        for (int g = warp; g < G; g += warps) {
            float m = -1e30f;
            for (int i = lane; i < n; i += 32) m = fmaxf(m, sc[g * C + i]);
            m = fmaxf(warp_max(m), ml[3 * g]);
            float l = 0.0f;
            for (int i = lane; i < n; i += 32) {
                float e = expf(sc[g * C + i] - m);
                sc[g * C + i] = e;
                l += e;
            }
            l = warp_sum(l);
            if (lane == 0) {
                float r = expf(ml[3 * g] - m);
                ml[3 * g + 1] = ml[3 * g + 1] * r + l;
                ml[3 * g + 2] = r;
                ml[3 * g] = m;
            }
        }
        __syncthreads();
        if (slice < P) {
            #pragma unroll
            for (int g = 0; g < FA_MAX_G; ++g)
                if (g < G) acc[g] *= ml[3 * g + 2];
            for (int i = slice; i < n; i += P) {
                float v = load_kv<KT>(values, kv_row(table, shift, p0 + i) * kv_dim +
                                                  (long)kvh * head_dim + d);
                #pragma unroll
                for (int g = 0; g < FA_MAX_G; ++g)
                    if (g < G) acc[g] += sc[g * C + i] * v;
            }
        }
        __syncthreads();
    }
    if (slice < P) {
        #pragma unroll
        for (int g = 0; g < FA_MAX_G; ++g)
            if (g < G) sc[(slice * G + g) * head_dim + d] = acc[g];
    }
    __syncthreads();
    for (int i = t; i < G * head_dim; i += C) {
        int g = i / head_dim, dd = i % head_dim;
        float o = 0.0f;
        for (int s = 0; s < P; ++s) o += sc[(s * G + g) * head_dim + dd];
        ctx[tq * q_dim + ((long)kvh * G + g) * head_dim + dd] = o / ml[3 * g + 1];
    }
}

// One layer over `T` consecutive tokens of one sequence (rows `x[0..T]`, at
// positions `position, position + 1, ...`, appending KV rows `history ...`).
// The same block as `dlm_decode_block`, step for step. Returns
// cudaErrorNotSupported, having launched nothing, when a projection's shape
// does not fit the GEMM or the heads the attention kernel: the caller then
// prefills through `dlm_decode_block_batched`.
extern "C" int dlm_prefill_block(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    int inter, float rms_eps, int w_dtype, int w_group_size,
    const void* q_proj, const void* k_proj, const void* v_proj, const void* o_proj,
    const void* gate_proj, const void* up_proj, const void* down_proj,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* q_norm, const float* k_norm,
    const float* inv_freq,
    float* x,                               // [T, hidden] in/out
    void* kv_keys, void* kv_values,         // the sequence's paged KV (table in ext)
    int history, int position, int T,
    int sliding_window, int activation, float rope_mscale,
    float attn_scale, float attn_softcap,
    const float* pre_ffn_norm, const float* post_ffn_norm,
    const DlmBlockExt* ext)
{
    DLM_ENTER
    const int B = 256;
    const DlmBlockExt* o = ext ? ext : &DLM_BLOCK_DEFAULT;
    if (T <= 0) return 0;
    if (T > DLM_PREFILL_T) return (int)cudaErrorInvalidValue;
    if (attn_scale <= 0.0f) attn_scale = rsqrtf((float)head_dim);
    const int gemma2 = (pre_ffn_norm != 0);
    const int G = num_kv_heads > 0 ? num_heads / num_kv_heads : 0;
    if (G < 1 || G > FA_MAX_G || G * num_kv_heads != num_heads || head_dim % 8 || head_dim > 256)
        return (int)cudaErrorNotSupported;
    // Every projection must take the GEMM; check them all before launching any.
    {
        const int ins[7] = {hidden_size, hidden_size, hidden_size, q_dim, hidden_size, hidden_size, inter};
        const void* ws[7] = {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj};
        for (int i = 0; i < 7; ++i) {
            if (i == DLM_W_GATE && !o->gated) continue;
            // `x` stands in for every activation block: all come from cudaMalloc.
            bool ok = false;
            switch (DLM_WDT(i)) {
#define DLM_FIT(D) case D: ok = gemm_fits<D>(ws[i], x, ins[i], DLM_WGRP(i)); break;
                DLM_FIT(DLM_W_F32)
                DLM_FOR_EACH_DTYPE(DLM_FIT)
#undef DLM_FIT
                default: ok = false;
            }
            if (!ok) return (int)cudaErrorNotSupported;
        }
    }
    #define DLM_NORM_B(src, w, bias, dst) { \
        if (o->layer_norm) layernorm_batched_kernel<<<T, RMS_THREADS>>>((src), (w), (bias), (dst), hidden_size, rms_eps); \
        else rmsnorm_batched_kernel<<<T, RMS_THREADS>>>((src), (w), (dst), hidden_size, rms_eps); }

    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(0, T * hidden_size)
    DLM_ALLOC(1, T * q_dim)
    DLM_ALLOC(2, T * kv_dim)
    DLM_ALLOC(3, T * kv_dim)
    DLM_ALLOC(4, T * q_dim)
    DLM_ALLOC(5, T * hidden_size)
    DLM_ALLOC(6, T * hidden_size)
    DLM_ALLOC(7, T * inter)
    DLM_ALLOC(8, T * inter)
    DLM_ALLOC(9, T * inter)
    DLM_ALLOC(10, T * hidden_size)
    #undef DLM_ALLOC
    if (e != cudaSuccess) return (int)e;
    float *normed = g_scratch[0], *q = g_scratch[1], *k = g_scratch[2];
    float *v = g_scratch[3], *ctx = g_scratch[4], *attn_out = g_scratch[5];
    float *normed2 = g_scratch[6], *gate = g_scratch[7], *up = g_scratch[8];
    float *inter_buf = g_scratch[9], *down = g_scratch[10];

    DLM_NORM_B(x, in_norm, o->in_norm_bias, normed)
    launch_gemm(DLM_WDT(DLM_W_Q), q_proj, normed, q_bias, q, q_dim, hidden_size, T, DLM_WGRP(DLM_W_Q), false);
    launch_gemm(DLM_WDT(DLM_W_K), k_proj, normed, k_bias, k, kv_dim, hidden_size, T, DLM_WGRP(DLM_W_K), false);
    launch_gemm(DLM_WDT(DLM_W_V), v_proj, normed, v_bias, v, kv_dim, hidden_size, T, DLM_WGRP(DLM_W_V), false);
    launch_qk_post(q, k, v, q_norm, k_norm, num_heads, num_kv_heads, head_dim, position, inv_freq,
                   rope_mscale, o->rope, rms_eps, kv_keys, kv_values, o->kv_table,
                   o->kv_block_shift, history, o->kv_half, T);
    {
        const int C = head_dim > 128 ? 256 : 128;
        size_t shmem = (size_t)(G * head_dim + G * C + 3 * G) * sizeof(float);
        dim3 grid(num_kv_heads, T);
        if (o->kv_half)
            attn_prefill_kernel<DLM_KV_F16><<<grid, C, shmem>>>(q, kv_keys, kv_values, o->kv_table,
                o->kv_block_shift, num_heads, num_kv_heads, head_dim, history, sliding_window,
                attn_scale, attn_softcap, ctx);
        else
            attn_prefill_kernel<DLM_KV_F32><<<grid, C, shmem>>>(q, kv_keys, kv_values, o->kv_table,
                o->kv_block_shift, num_heads, num_kv_heads, head_dim, history, sliding_window,
                attn_scale, attn_softcap, ctx);
    }
    const long TH = (long)T * hidden_size;
    if (gemma2 || o->parallel) {
        launch_gemm(DLM_WDT(DLM_W_O), o_proj, ctx, o->o_bias, attn_out, hidden_size, q_dim, T, DLM_WGRP(DLM_W_O), false);
        if (gemma2) rmsnorm_batched_kernel<<<T, RMS_THREADS>>>(attn_out, post_norm, attn_out, hidden_size, rms_eps);
        if (o->parallel) DLM_NORM_B(x, post_norm, o->post_norm_bias, normed2)
        add_inplace_batched_kernel<<<grid_for((int)TH, B), B>>>(x, attn_out, TH);
    } else {
        // Straight into the residual.
        launch_gemm(DLM_WDT(DLM_W_O), o_proj, ctx, o->o_bias, x, hidden_size, q_dim, T, DLM_WGRP(DLM_W_O), true);
    }
    if (!o->parallel) {
        if (gemma2) rmsnorm_batched_kernel<<<T, RMS_THREADS>>>(x, pre_ffn_norm, normed2, hidden_size, rms_eps);
        else DLM_NORM_B(x, post_norm, o->post_norm_bias, normed2)
    }
    const long TI = (long)T * inter;
    launch_gemm(DLM_WDT(DLM_W_UP), up_proj, normed2, o->up_bias, up, inter, hidden_size, T, DLM_WGRP(DLM_W_UP), false);
    if (o->gated) {
        launch_gemm(DLM_WDT(DLM_W_GATE), gate_proj, normed2, (const float*)0, gate, inter, hidden_size, T, DLM_WGRP(DLM_W_GATE), false);
        swiglu_batched_kernel<<<grid_for((int)TI, B), B>>>(gate, up, inter_buf, TI, activation);
    } else {
        activate_kernel<<<grid_for((int)TI, B), B>>>(up, inter_buf, TI, activation);
    }
    if (gemma2) {
        launch_gemm(DLM_WDT(DLM_W_DOWN), down_proj, inter_buf, o->down_bias, down, hidden_size, inter, T, DLM_WGRP(DLM_W_DOWN), false);
        rmsnorm_batched_kernel<<<T, RMS_THREADS>>>(down, post_ffn_norm, down, hidden_size, rms_eps);
        add_inplace_batched_kernel<<<grid_for((int)TH, B), B>>>(x, down, TH);
    } else {
        launch_gemm(DLM_WDT(DLM_W_DOWN), down_proj, inter_buf, o->down_bias, x, hidden_size, inter, T, DLM_WGRP(DLM_W_DOWN), true);
    }
    #undef DLM_NORM_B
    return (int)cudaGetLastError();
}

// ── Mixture-of-Experts device path ─────────────────────────────────────────
//
// A dense layer runs in one `dlm_decode_block` call. A MoE layer cannot: the
// experts a token uses aren't known until the router runs, so the block is split
// into three host-orchestrated calls on the same (in-order) default stream:
//
//   1. dlm_moe_attn   — attention sublayer (residual into x) + normed2 (FFN
//                        input) left in persistent scratch slot MOE_NORMED2.
//   2. dlm_moe_matvec — W·normed2 → host, for the router logits (and the shared
//                        expert's sigmoid gate). The host does top-k + softmax.
//   3. dlm_apply_expert — for each selected expert, x += weight · SwiGLU(expert).
//
// `normed2` persists in scratch between (1) and every (3) because all three run
// on the same stream on the same thread, and the scratch is thread_local. This
// mirrors `moe_ffn` in `src/forward/cpu.rs` — the CPU oracle these match.

// Attention sublayer + post-attention norm for a MoE layer. Duplicates the
// attention half of `dlm_decode_block` deliberately: refactoring that tested,
// hot dense path to share code here risks a regression it can't easily catch.
extern "C" int dlm_moe_attn(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    float rms_eps,
    int w_dtype,                           // DLM_W_* tag for the core (attn) weights
    int w_group_size,
    const void* q_proj, const void* k_proj, const void* v_proj, const void* o_proj,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias,  // may be NULL
    const float* q_norm, const float* k_norm,  // Qwen3 per-head Q/K RMSNorm; may be NULL
    const float* inv_freq,
    float* x,                              // [hidden] in/out (attn residual folded in)
    float* kv_keys, float* kv_values,      // persistent device KV, mutated in place
    int num_positions, int position,
    int sliding_window,                    // 0 = full causal attention (Mistral SWA otherwise)
    float rope_mscale,                     // YaRN attention temperature (1.0 otherwise)
    float attn_scale,                      // <=0: derive 1/sqrt(head_dim) as usual
    float attn_softcap,                    // 0 = off (Gemma2 caps attention logits)
    int kv_half,                           // 1: the KV buffers hold fp16
    const int* kv_table, int kv_block_shift,  // the sequence's block table
    const DlmBlockExt* ext)                // per-projection dtypes; may be NULL
{
    DLM_ENTER
    const DlmBlockExt* o = ext ? ext : &DLM_BLOCK_DEFAULT;
    const int B = 256;
    int total_pos = num_positions + 1;
    if (attn_scale <= 0.0f) attn_scale = rsqrtf((float)head_dim);

    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(0, hidden_size)
    DLM_ALLOC(1, q_dim)
    DLM_ALLOC(2, kv_dim)
    DLM_ALLOC(3, kv_dim)
    DLM_ALLOC(4, q_dim)
    DLM_ALLOC(5, hidden_size)
    DLM_ALLOC(MOE_NORMED2, hidden_size)
    #undef DLM_ALLOC
    float *normed = g_scratch[0], *q = g_scratch[1], *k = g_scratch[2];
    float *v = g_scratch[3], *ctx = g_scratch[4], *attn_out = g_scratch[5];
    float *normed2 = g_scratch[MOE_NORMED2];

    if (e == cudaSuccess) {
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, in_norm, normed, hidden_size, rms_eps);
        launch_matvec(DLM_WDT(DLM_W_Q), q_proj, normed, q_bias, q, q_dim, hidden_size, DLM_WGRP(DLM_W_Q));
        launch_matvec(DLM_WDT(DLM_W_K), k_proj, normed, k_bias, k, kv_dim, hidden_size, DLM_WGRP(DLM_W_K));
        launch_matvec(DLM_WDT(DLM_W_V), v_proj, normed, v_bias, v, kv_dim, hidden_size, DLM_WGRP(DLM_W_V));
        // Qwen3 per-head Q/K RMSNorm (NULL when absent), before RoPE.
        if (q_norm) head_rmsnorm_kernel<<<num_heads, 1>>>(q, q_norm, num_heads, head_dim, rms_eps);
        if (k_norm) head_rmsnorm_kernel<<<num_kv_heads, 1>>>(k, k_norm, num_kv_heads, head_dim, rms_eps);
        rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, inv_freq, rope_mscale);
        rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, inv_freq, rope_mscale);
        launch_kv_append(k, v, kv_keys, kv_values, kv_table, kv_block_shift, num_positions,
                         kv_dim, kv_half);
        e = launch_attention(q, kv_keys, kv_values, kv_table, kv_block_shift, ctx, num_heads, num_kv_heads, head_dim, total_pos, sliding_window, attn_scale, attn_softcap, kv_half);
        if (e != cudaSuccess) return (int)e;
        launch_matvec(DLM_WDT(DLM_W_O), o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim, DLM_WGRP(DLM_W_O));
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);
        // FFN input, reused by the router matvec and every expert.
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, post_norm, normed2, hidden_size, rms_eps);
        e = cudaGetLastError();
    }
    return (int)e;
}

// `dlm_moe_attn` for a batch of sequences: the attention half of
// `dlm_decode_block_batched` (its projections read once for every slot), then
// each slot's FFN input normed into rows of scratch slot MOE_NORMED2B. The
// routed experts differ per token, so the caller then routes each slot on its
// own, moving its row into MOE_NORMED2 with `dlm_moe_take_row` first.
extern "C" int dlm_moe_attn_batched(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    float rms_eps, int w_dtype, int w_group_size,
    const void* q_proj, const void* k_proj, const void* v_proj, const void* o_proj,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias,
    const float* q_norm, const float* k_norm,
    const float* inv_freq,
    float* x,                              // [batch, hidden] in/out
    const DlmSlots* kv_keys, const DlmSlots* kv_values, const DlmTables* kv_tables,
    const DlmInts* num_positions, const DlmInts* positions,
    int batch, int sliding_window, float rope_mscale, float attn_scale, float attn_softcap,
    const DlmBlockExt* ext)
{
    DLM_ENTER
    const int B = 256;
    if (batch <= 0) return 0;
    if (batch > DLM_MAX_BATCH) return (int)cudaErrorInvalidValue;
    const DlmBlockExt* o = ext ? ext : &DLM_BLOCK_DEFAULT;
    if (attn_scale <= 0.0f) attn_scale = rsqrtf((float)head_dim);
    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(0, batch * hidden_size)
    DLM_ALLOC(1, batch * q_dim)
    DLM_ALLOC(2, batch * kv_dim)
    DLM_ALLOC(3, batch * kv_dim)
    DLM_ALLOC(4, batch * q_dim)
    DLM_ALLOC(5, batch * hidden_size)
    DLM_ALLOC(MOE_NORMED2B, batch * hidden_size)
    #undef DLM_ALLOC
    if (e != cudaSuccess) return (int)e;
    float *normed = g_scratch[0], *q = g_scratch[1], *k = g_scratch[2];
    float *v = g_scratch[3], *ctx = g_scratch[4], *attn_out = g_scratch[5];

    rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(x, in_norm, normed, hidden_size, rms_eps);
    launch_matvec_batched(DLM_WDT(DLM_W_Q), q_proj, normed, q_bias, q, q_dim, hidden_size, DLM_WGRP(DLM_W_Q), batch);
    launch_matvec_batched(DLM_WDT(DLM_W_K), k_proj, normed, k_bias, k, kv_dim, hidden_size, DLM_WGRP(DLM_W_K), batch);
    launch_matvec_batched(DLM_WDT(DLM_W_V), v_proj, normed, v_bias, v, kv_dim, hidden_size, DLM_WGRP(DLM_W_V), batch);
    e = launch_attention_slots(q, k, v, ctx, q_norm, k_norm, num_heads, num_kv_heads, head_dim,
                               inv_freq, rope_mscale, 1, rms_eps, kv_keys, kv_values, kv_tables,
                               o->kv_block_shift, num_positions, positions, batch, sliding_window,
                               attn_scale, attn_softcap, o->kv_half);
    if (e != cudaSuccess) return (int)e;
    launch_matvec_batched(DLM_WDT(DLM_W_O), o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim, DLM_WGRP(DLM_W_O), batch);
    add_inplace_batched_kernel<<<grid_for(batch * hidden_size, B), B>>>(x, attn_out, (long)batch * hidden_size);
    rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(x, post_norm, g_scratch[MOE_NORMED2B], hidden_size, rms_eps);
    return (int)cudaGetLastError();
}

// Make slot `b`'s FFN input (left by `dlm_moe_attn_batched`) the one the router
// and expert calls read.
extern "C" int dlm_moe_take_row(int hidden_size, int b)
{
    DLM_ENTER
    cudaError_t e = scratch_ensure(MOE_NORMED2, hidden_size);
    if (e != cudaSuccess) return (int)e;
    copy_kernel<<<grid_for(hidden_size, 256), 256>>>(g_scratch[MOE_NORMED2B] + (long)b * hidden_size,
                                                      g_scratch[MOE_NORMED2], hidden_size);
    return (int)cudaGetLastError();
}

// Post-attention norm for a MoE layer whose attention ran in a *separate* call —
// the MLA path, where `dlm_mla_attn` folds the attention residual into `x` but
// (unlike `dlm_moe_attn`) leaves no FFN input behind. This produces exactly the
// `normed2` that `dlm_moe_attn` step 1 would have, in the same scratch slot, so
// the router/expert calls that follow are byte-identical on either attention
// path. Mirrors the `rmsnorm(&h1, post_attention_layernorm)` that
// `decode_block_streaming_moe` does between attention and `moe_ffn_streaming`.
extern "C" int dlm_moe_norm(int hidden_size, float rms_eps, const float* post_norm, float* x)
{
    DLM_ENTER
    cudaError_t e = scratch_ensure(MOE_NORMED2, hidden_size);
    if (e != cudaSuccess) return (int)e;
    rmsnorm_kernel<<<1, RMS_THREADS>>>(x, post_norm, g_scratch[MOE_NORMED2], hidden_size, rms_eps);
    return (int)cudaGetLastError();
}

// `y_host[0..out_dim] = W · normed2`, copied to the host. Used for the router
// logits (out_dim = num_experts) and the shared expert's gate (out_dim = 1).
extern "C" int dlm_moe_matvec(int out_dim, int hidden_size, int w_dtype, int w_group_size,
                              const void* w, float* y_host)
{
    DLM_ENTER
    cudaError_t e = cudaSuccess;
    if (e == cudaSuccess) e = scratch_ensure(MOE_MATVEC, out_dim);
    if (e == cudaSuccess) {
        launch_matvec(w_dtype, w, g_scratch[MOE_NORMED2], (const float*)0, g_scratch[MOE_MATVEC],
                      out_dim, hidden_size, w_group_size);
        e = cudaGetLastError();
    }
    // Blocking D2H: this also drains the launches above, so `y_host` is valid on
    // return and the host can route. out_dim is tiny (expert count), so the stall
    // is negligible against the expert GEMVs that follow.
    if (e == cudaSuccess)
        e = cudaMemcpy(y_host, g_scratch[MOE_MATVEC], (size_t)out_dim * sizeof(float),
                       cudaMemcpyDeviceToHost);
    return (int)e;
}

// Index of the largest logit, the lowest index on ties (the host's greedy
// sampler keeps the first maximum too). One block per row of `n` logits:
// <<<rows, 1024>>>, row r's index to out[r].
__global__ void argmax_kernel(const float* v, int n, int* out) {
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    v += (long)blockIdx.x * n;
    out += blockIdx.x;
    float bv = -INFINITY;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float x = v[i];
        if (x > bv) { bv = x; bi = i; }   // strided ascending: first max per thread
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { best_v[warp] = bv; best_i[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x + 31) / 32; ++w)
            if (best_v[w] > bv || (best_v[w] == bv && best_i[w] < bi)) { bv = best_v[w]; bi = best_i[w]; }
        *out = bi == 0x7FFFFFFF ? 0 : bi;   // every logit -inf: index 0, as the host
    }
}

// The LM head over `batch` final-normed rows of `x_host` (`batch * hidden_size`):
// each row's logits to `logits_host` (`batch * vocab`), or, when `token_host` is
// set, only each row's argmax (`batch` ints; `logits_host` is then unused).
//
// The vocabulary-wide GEMV is the largest single matvec in a decode step, so
// the rows go through DLM_MAX_BATCH at a time, each group reading the weights
// once. Before this a batch of B sequences called the head B times: on Qwen2.5
// 0.5B at batch 16 that was 36 of a 90 ms step. One row takes the single-row
// GEMV, faster than the batched one at batch 1.
//
// The argmax saves copying `vocab` floats a row back to the host (600 KB on
// Qwen2.5) for the host to take one maximum of. A final logit softcap is `tanh`,
// which keeps the order, so the caller can skip it there.
extern "C" int dlm_lm_head(int vocab, int hidden_size, int w_dtype, int w_group_size,
                           const void* w, const float* x_host, int batch, float* logits_host,
                           int* token_host)
{
    DLM_ENTER
    if (batch <= 0) return 0;
    const int rows = batch < DLM_MAX_BATCH ? batch : DLM_MAX_BATCH;
    cudaError_t e = scratch_ensure(LM_X, rows * hidden_size);
    if (e == cudaSuccess) e = scratch_ensure(LM_LOGITS, rows * vocab + DLM_MAX_BATCH);
    float* x = g_scratch[LM_X];
    float* logits = g_scratch[LM_LOGITS];
    int* tokens = (int*)(logits + (long)rows * vocab);
    for (int r0 = 0; r0 < batch && e == cudaSuccess; r0 += DLM_MAX_BATCH) {
        const int n = batch - r0 < DLM_MAX_BATCH ? batch - r0 : DLM_MAX_BATCH;
        e = cudaMemcpy(x, x_host + (long)r0 * hidden_size, (size_t)n * hidden_size * sizeof(float),
                       cudaMemcpyHostToDevice);
        if (e != cudaSuccess) break;
        if (n == 1)
            launch_matvec(w_dtype, w, x, (const float*)0, logits, vocab, hidden_size, w_group_size);
        else
            launch_matvec_batched(w_dtype, w, x, (const float*)0, logits, vocab, hidden_size,
                                  w_group_size, n);
        if (token_host) argmax_kernel<<<n, 1024>>>(logits, vocab, tokens);
        e = cudaGetLastError();
        if (e != cudaSuccess) break;
        // Blocking D2H, so it also waits for the launches above.
        e = token_host
            ? cudaMemcpy(token_host + r0, tokens, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost)
            : cudaMemcpy(logits_host + (long)r0 * vocab, logits, (size_t)n * vocab * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    return (int)e;
}

// Apply one expert to the residual: `x += weight · down·(silu(gate·normed2) ⊙ up·normed2)`.
// Reads normed2 from scratch (left by dlm_moe_attn). `w_dtype`/`w_group_size`
// describe the expert's weights, which may differ from the core's.

// ── Grouped expert application (the top-k experts in one launch each) ───────
//
// `dlm_apply_expert` runs 3 kernels per selected expert. With top-k = 2..8 and
// 60 layers that is a few hundred launches per token, each with fixed overhead,
// and each grid is small enough to leave the GPU underutilized. The grouped form
// below does the same arithmetic with **3 launches total**, giving every launch a
// k-times-larger grid. It is op-for-op the same math as the loop — only the
// summation order over experts changes (all k accumulated before the store
// instead of one at a time), which is why the parity tolerance already covers it.
//
// Expert pointers travel by value inside a small struct: CUDA copies kernel
// parameters to the device for us, so no separate pointer-array upload is needed.
#define DLM_MAX_TOPK 16
typedef struct { const void* p[DLM_MAX_TOPK]; } DlmPtrs;
typedef struct { float w[DLM_MAX_TOPK]; } DlmWeights;

// out[e*out_dim + row] = dot(W_e[row], x)  for every (expert, row) pair.
// One block per (e, row); threads stride the row and tree-reduce, exactly like
// `matvec_kernel`.
template <int DT>
__global__ void grouped_matvec_kernel(DlmPtrs W, const float* x, float* out,
                                      int out_dim, int in_dim, int group_size) {
    __shared__ float partial[MATVEC_THREADS];
    int e = blockIdx.x / out_dim;
    int row = blockIdx.x - e * out_dim;
    const void* We = W.p[e];
    long base = (long)row * in_dim;
    long n = (long)out_dim * in_dim;

    partial[threadIdx.x] = row_dot<DT>(We, x, base, threadIdx.x, blockDim.x, in_dim, n, group_size);
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(long)e * out_dim + row] = partial[0];
}

static void launch_grouped_matvec(int dt, DlmPtrs W, const float* x, float* out,
                                  int n_experts, int out_dim, int in_dim, int group_size) {
    int blocks = n_experts * out_dim;
#define DLM_CASE(D) case D: grouped_matvec_kernel<D><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size); break;
    switch (dt) {
        DLM_FOR_EACH_DTYPE(DLM_CASE)
        default:
            grouped_matvec_kernel<DLM_W_F32><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size);
            break;
    }
#undef DLM_CASE
}

// x[h] += sum_e weight[e] * dot(down_e[h], inter_buf + e*inter)
//
// One block per hidden row, reducing over the whole (expert, inter) plane, so the
// k experts' down-projections and their weighted sum collapse into one launch
// with no atomics and no per-expert temporary.
template <int DT>
__global__ void grouped_down_kernel(DlmPtrs W, DlmWeights weights, const float* inter_buf,
                                    float* x, int n_experts, int hidden, int inter,
                                    int group_size) {
    __shared__ float partial[MATVEC_THREADS];
    int h = blockIdx.x;
    long n = (long)hidden * inter;
    long total = (long)n_experts * inter;

    float acc = 0.0f;
    for (long t = threadIdx.x; t < total; t += blockDim.x) {
        int e = (int)(t / inter);
        int i = (int)(t - (long)e * inter);
        acc += weights.w[e] * load_w<DT>(W.p[e], (long)h * inter + i, n, group_size) * inter_buf[t];
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) x[h] += partial[0];
}

static void launch_grouped_down(int dt, DlmPtrs W, DlmWeights weights, const float* inter_buf,
                                float* x, int n_experts, int hidden, int inter, int group_size) {
#define DLM_CASE(D) case D: grouped_down_kernel<D><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size); break;
    switch (dt) {
        DLM_FOR_EACH_DTYPE(DLM_CASE)
        default:
            grouped_down_kernel<DLM_W_F32><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size);
            break;
    }
#undef DLM_CASE
}

// Apply all `n_experts` selected experts to `normed2`, accumulating each one's
// gate-weighted SwiGLU output into `x`. Equivalent to calling `dlm_apply_expert`
// once per expert; every expert must share `w_dtype`/`w_group_size` (they come
// from one checkpoint). `n_experts` must be <= DLM_MAX_TOPK — the caller falls
// back to the per-expert loop otherwise.
extern "C" int dlm_apply_experts(int hidden_size, int inter, int n_experts,
                                 int w_dtype, int w_group_size,
                                 const DlmPtrs* gate, const DlmPtrs* up, const DlmPtrs* down,
                                 const DlmWeights* weights, float* x, int activation)
{
    DLM_ENTER
    const int B = 256;
    if (n_experts <= 0) return 0;
    if (n_experts > DLM_MAX_TOPK) return (int)cudaErrorInvalidValue;

    long span = (long)n_experts * inter;
    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(7, (int)span)
    DLM_ALLOC(8, (int)span)
    DLM_ALLOC(9, (int)span)
    #undef DLM_ALLOC
    if (e != cudaSuccess) return (int)e;

    float *g = g_scratch[7], *u = g_scratch[8], *inter_buf = g_scratch[9];
    float *normed2 = g_scratch[MOE_NORMED2];

    launch_grouped_matvec(w_dtype, *gate, normed2, g, n_experts, inter, hidden_size, w_group_size);
    launch_grouped_matvec(w_dtype, *up, normed2, u, n_experts, inter, hidden_size, w_group_size);
    // Elementwise over the whole (expert, inter) plane in one launch.
    swiglu_kernel<<<grid_for((int)span, B), B>>>(g, u, inter_buf, (int)span, activation);
    launch_grouped_down(w_dtype, *down, *weights, inter_buf, x, n_experts, hidden_size, inter, w_group_size);
    return (int)cudaGetLastError();
}

extern "C" int dlm_apply_expert(int hidden_size, int inter, int w_dtype, int w_group_size,
                                const void* gate, const void* up, const void* down,
                                float weight, float* x, int activation)
{
    DLM_ENTER
    const int B = 256;
    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(7, inter)
    DLM_ALLOC(8, inter)
    DLM_ALLOC(9, inter)
    DLM_ALLOC(10, hidden_size)
    #undef DLM_ALLOC
    float *g = g_scratch[7], *u = g_scratch[8], *inter_buf = g_scratch[9], *down_out = g_scratch[10];
    float *normed2 = g_scratch[MOE_NORMED2];

    if (e == cudaSuccess) {
        launch_matvec(w_dtype, gate, normed2, (const float*)0, g, inter, hidden_size, w_group_size);
        launch_matvec(w_dtype, up, normed2, (const float*)0, u, inter, hidden_size, w_group_size);
        swiglu_kernel<<<grid_for(inter, B), B>>>(g, u, inter_buf, inter, activation);
        launch_matvec(w_dtype, down, inter_buf, (const float*)0, down_out, hidden_size, inter, w_group_size);
        scaled_add_kernel<<<grid_for(hidden_size, B), B>>>(x, down_out, weight, hidden_size);
        e = cudaGetLastError();
    }
    return (int)e;
}

// Dense SwiGLU/GeGLU FFN sublayer on its own: `x += down·(act(gate·norm(x)) ⊙
// up·norm(x))`. Used by the MLA path, whose attention is a separate call (the
// dense block folds attention + FFN into one `dlm_decode_block`).
extern "C" int dlm_dense_ffn(int hidden_size, int inter, float rms_eps, int w_dtype, int w_group_size,
                             const void* gate_proj, const void* up_proj, const void* down_proj,
                             const float* post_norm, int activation, float* x) {
    DLM_ENTER
    const int B = 256;
    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(6, hidden_size)
    DLM_ALLOC(7, inter)
    DLM_ALLOC(8, inter)
    DLM_ALLOC(9, inter)
    DLM_ALLOC(10, hidden_size)
    #undef DLM_ALLOC
    if (e != cudaSuccess) return (int)e;
    float *normed2 = g_scratch[6], *gate = g_scratch[7], *up = g_scratch[8];
    float *inter_buf = g_scratch[9], *down = g_scratch[10];
    rmsnorm_kernel<<<1, RMS_THREADS>>>(x, post_norm, normed2, hidden_size, rms_eps);
    launch_matvec(w_dtype, gate_proj, normed2, (const float*)0, gate, inter, hidden_size, w_group_size);
    launch_matvec(w_dtype, up_proj, normed2, (const float*)0, up, inter, hidden_size, w_group_size);
    swiglu_kernel<<<grid_for(inter, B), B>>>(gate, up, inter_buf, inter, activation);
    launch_matvec(w_dtype, down_proj, inter_buf, (const float*)0, down, hidden_size, inter, w_group_size);
    add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, down, hidden_size);
    e = cudaGetLastError();
    return (int)e;
}

// ── Multi-head Latent Attention device path (DeepSeek) ──────────────────────
//
// Mirrors `mla_attention_sublayer` in src/forward/cpu.rs. The KV cache packs
// `[c_kv (kv_lora_rank) | k_pe (qk_rope)]` per token; per-head K/V are
// reconstructed on the fly from the cached latent inside the attention kernel
// (materializing all of them would need gigabytes of scratch).

// RoPE the rope-part of each query head: q[h*qk + nope .. + qk_rope].
__global__ void mla_rope_q_kernel(float* q, int num_heads, int qk, int nope, int rope,
                                  int position, const float* inv_freq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = rope / 2;
    if (idx >= num_heads * half) return;
    int h = idx / half, i = idx % half;
    int base = h * qk + nope;
    float ang = (float)position * inv_freq[i];
    float s = sinf(ang), c = cosf(ang);
    float a = q[base + i], b = q[base + i + half];
    q[base + i] = a * c - b * s;
    q[base + i + half] = a * s + b * c;
}

// MLA attention over `positions` cached latents, without reconstructing K or V.
//
// Each position caches `[c_p ; k_pe_p]`, and head h's key/value are linear in
// the latent: k_nope = K_h c_p, v = V_h c_p, where K_h and V_h are head h's rows
// of `kv_b`. So both sides of attention factor through latent space:
//
//   score(h, p) = q_nope_h . (K_h c_p) + q_rope_h . k_pe_p
//               = (K_h^T q_nope_h) . c_p + q_rope_h . k_pe_p
//   ctx_h       = sum_p w(h, p) V_h c_p  =  V_h (sum_p w(h, p) c_p)
//
// Five launches, each parallel over a grid:
//
//   1. mla_query_latent_kernel  (head, latent)   u_h = K_h^T q_nope_h
//   2. mla_scores_kernel        (head, position) the scaled logit
//   3. attn_softmax_kernel      (head)           softmax of each row, shared
//                                                with standard attention
//   4. mla_mix_kernel           (head, latent)   m_h = sum_p w(h, p) c_p
//   5. mla_values_kernel        (head, v dim)    ctx_h = V_h m_h
//
// The previous kernel ran one thread per head and rebuilt every K and V element
// of every position with a latent-wide dot product through `kv_b`: heads x
// positions x (2 nope + v) x latent weight reads per token, per layer, serially
// per head. This is heads x (nope + v) x latent once, plus heads x positions x
// (latent + rope). `mla_attention_sublayer` in src/forward/cpu.rs reconstructs
// explicitly; the two agree up to the order of floating-point sums.
template <int DT>
__global__ void mla_query_latent_kernel(const float* q, const void* kv_b, int kv_b_group,
                                        float* u, int num_heads, int nope, int rope, int vdim,
                                        int latent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * latent) return;
    int h = idx / latent, l = idx % latent;
    int qk = nope + rope;
    long per_head = nope + vdim;
    long n = (long)num_heads * per_head * latent;
    const float* q_nope = q + (long)h * qk;
    float acc = 0.0f;
    for (int d = 0; d < nope; ++d)
        acc += q_nope[d] * load_w<DT>(kv_b, ((long)h * per_head + d) * latent + l, n, kv_b_group);
    u[idx] = acc;
}

__global__ void mla_scores_kernel(const float* q, const float* u, const float* kv_keys,
                                  float* scores, int num_heads, int nope, int rope, int latent,
                                  int positions, float scale) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)num_heads * positions) return;
    int h = (int)(idx / positions), p = (int)(idx % positions);
    const float* ck = kv_keys + (long)p * (latent + rope);
    const float* uh = u + (long)h * latent;
    const float* q_rope = q + (long)h * (nope + rope) + nope;
    float sc = 0.0f;
    for (int l = 0; l < latent; ++l) sc += uh[l] * ck[l];
    for (int d = 0; d < rope; ++d) sc += q_rope[d] * ck[latent + d];
    scores[idx] = sc * scale;
}

__global__ void mla_mix_kernel(const float* weights, const float* kv_keys, float* m,
                               int num_heads, int rope, int latent, int positions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * latent) return;
    int h = idx / latent, l = idx % latent;
    const float* row = weights + (long)h * positions;
    long kv_dim = latent + rope;
    float acc = 0.0f;
    for (int p = 0; p < positions; ++p) acc += row[p] * kv_keys[(long)p * kv_dim + l];
    m[idx] = acc;
}

template <int DT>
__global__ void mla_values_kernel(const float* m, const void* kv_b, int kv_b_group, float* ctx,
                                  int num_heads, int nope, int vdim, int latent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_heads * vdim) return;
    int h = idx / vdim, d = idx % vdim;
    long per_head = nope + vdim;
    long n = (long)num_heads * per_head * latent;
    long base = ((long)h * per_head + nope + d) * latent;
    const float* mh = m + (long)h * latent;
    float acc = 0.0f;
    for (int l = 0; l < latent; ++l) acc += load_w<DT>(kv_b, base + l, n, kv_b_group) * mh[l];
    ctx[idx] = acc;
}

// Launch the MLA attention kernels for weights in dtype `dt`.
static cudaError_t launch_mla_attention(int dt, const float* q, const float* kv_keys,
                                        const void* kv_b, int kv_b_group, float* ctx,
                                        int num_heads, int nope, int rope, int vdim, int latent,
                                        int positions, float scale) {
    const int B = 256;
    cudaError_t e = scratch_ensure(MLA_U, num_heads * latent);
    if (e == cudaSuccess) e = scratch_ensure(MLA_M, num_heads * latent);
    // Grown in steps, as in launch_attention: the span grows by one every token.
    if (e == cudaSuccess) e = scratch_ensure(ATTN_SCORES, num_heads * ((positions + 1023) & ~1023));
    if (e != cudaSuccess) return e;
    float* u = g_scratch[MLA_U];
    float* m = g_scratch[MLA_M];
    float* scores = g_scratch[ATTN_SCORES];
    int hl = grid_for(num_heads * latent, B);
    #define DLM_MLA_DT(DT) \
        mla_query_latent_kernel<DT><<<hl, B>>>(q, kv_b, kv_b_group, u, num_heads, nope, rope, vdim, latent); \
        mla_scores_kernel<<<grid_for(num_heads * positions, B), B>>>(q, u, kv_keys, scores, num_heads, nope, rope, latent, positions, scale); \
        attn_softmax_kernel<<<num_heads, SOFTMAX_THREADS>>>(scores, num_heads, positions); \
        mla_mix_kernel<<<hl, B>>>(scores, kv_keys, m, num_heads, rope, latent, positions); \
        mla_values_kernel<DT><<<grid_for(num_heads * vdim, B), B>>>(m, kv_b, kv_b_group, ctx, num_heads, nope, vdim, latent);
#define DLM_CASE(D) case D: DLM_MLA_DT(D) break;
    switch (dt) {
        DLM_FOR_EACH_DTYPE(DLM_CASE)
        default: DLM_MLA_DT(DLM_W_F32) break;
    }
#undef DLM_CASE
    #undef DLM_MLA_DT
    return cudaSuccess;
}

// One MLA attention sublayer. All attention-projection weights share `w_dtype`.
// `kv_keys` is the persistent per-session cache (width kv_lora_rank + qk_rope);
// `kv_values` is unused on this path. `x` gets the attention residual folded in.
extern "C" int dlm_mla_attn(
    int hidden_size, int num_heads,
    int q_lora_rank,          // 0 = direct query projection (no low-rank)
    int kv_lora_rank, int qk_nope, int qk_rope, int v_head_dim,
    float rms_eps, int w_dtype, int w_group_size,
    const void* q_a_proj, const float* q_a_layernorm,  // may be NULL (no q-lora)
    const void* q_b_proj,     // [num_heads*(qk_nope+qk_rope), q_lora_rank | hidden]
    const void* kv_a_proj,    // [kv_lora_rank + qk_rope, hidden]
    const float* kv_a_layernorm,
    const void* kv_b_proj,    // [num_heads*(qk_nope+v_head_dim), kv_lora_rank]
    const void* o_proj,       // [hidden, num_heads*v_head_dim]
    const float* in_norm, const float* inv_freq, float rope_mscale,
    float* x, float* kv_keys, int num_positions, int position) {
    DLM_ENTER
    const int B = 256;
    int qk = qk_nope + qk_rope;
    int nhqk = num_heads * qk;
    int latent = kv_lora_rank;
    int kv_dim = latent + qk_rope;
    int nhv = num_heads * v_head_dim;
    int total_pos = num_positions + 1;

    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(idx, n) if (e == cudaSuccess) { e = scratch_ensure((idx), (n)); }
    DLM_ALLOC(MLA_NORMED, hidden_size)
    DLM_ALLOC(MLA_Q, nhqk)
    DLM_ALLOC(MLA_CQ, q_lora_rank > 0 ? q_lora_rank : 1)
    DLM_ALLOC(MLA_KVA, latent + qk_rope)
    DLM_ALLOC(MLA_CKV, latent)
    DLM_ALLOC(MLA_CTX, nhv)
    #undef DLM_ALLOC
    if (e != cudaSuccess) return (int)e;

    float* normed = g_scratch[MLA_NORMED];
    float* q = g_scratch[MLA_Q];
    float* kva = g_scratch[MLA_KVA];
    float* c_kv = g_scratch[MLA_CKV];
    float* ctx = g_scratch[MLA_CTX];

    rmsnorm_kernel<<<1, RMS_THREADS>>>(x, in_norm, normed, hidden_size, rms_eps);
    // Query: low-rank (down → norm → up) or direct.
    if (q_lora_rank > 0 && q_a_proj) {
        float* c_q = g_scratch[MLA_CQ];
        launch_matvec(w_dtype, q_a_proj, normed, (const float*)0, c_q, q_lora_rank, hidden_size, w_group_size);
        rmsnorm_kernel<<<1, RMS_THREADS>>>(c_q, q_a_layernorm, c_q, q_lora_rank, rms_eps);
        launch_matvec(w_dtype, q_b_proj, c_q, (const float*)0, q, nhqk, q_lora_rank, w_group_size);
    } else {
        launch_matvec(w_dtype, q_b_proj, normed, (const float*)0, q, nhqk, hidden_size, w_group_size);
    }
    // KV down → [latent | k_pe]; norm the latent, RoPE the shared k_pe + q rope-parts.
    launch_matvec(w_dtype, kv_a_proj, normed, (const float*)0, kva, latent + qk_rope, hidden_size, w_group_size);
    rmsnorm_kernel<<<1, RMS_THREADS>>>(kva, kv_a_layernorm, c_kv, latent, rms_eps);
    mla_rope_q_kernel<<<grid_for(num_heads * (qk_rope / 2), B), B>>>(q, num_heads, qk, qk_nope, qk_rope, position, inv_freq);
    rope_kernel<<<grid_for(qk_rope / 2, B), B>>>(kva + latent, 1, qk_rope, position, inv_freq, 1.0f);
    // Append [c_kv ; k_pe] to the cache at slot num_positions.
    copy_kernel<<<grid_for(latent, B), B>>>(c_kv, kv_keys + (long)num_positions * kv_dim, latent);
    copy_kernel<<<grid_for(qk_rope, B), B>>>(kva + latent, kv_keys + (long)num_positions * kv_dim + latent, qk_rope);
    // Attend (reconstructing K/V from each cached latent), then output-project.
    float scale = rope_mscale * rope_mscale / sqrtf((float)qk);
    e = launch_mla_attention(w_dtype, q, kv_keys, kv_b_proj, w_group_size, ctx, num_heads, qk_nope, qk_rope, v_head_dim, latent, total_pos, scale);
    if (e != cudaSuccess) return (int)e;
    // o = o_proj · ctx, added into the residual (scratch slot 0 reused for `o`).
    if (e == cudaSuccess) e = scratch_ensure(0, hidden_size);
    if (e == cudaSuccess) {
        float* o = g_scratch[0];
        launch_matvec(w_dtype, o_proj, ctx, (const float*)0, o, hidden_size, nhv, w_group_size);
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, o, hidden_size);
        e = cudaGetLastError();
    }
    return (int)e;
}

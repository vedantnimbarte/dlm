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

// Dispatch the GEMV on the runtime weight dtype (one block per output row).
static void launch_matvec(int dt, const void* W, const float* x, const float* bias, float* out,
                          int out_dim, int in_dim, int group_size) {
    switch (dt) {
        case DLM_W_BF16:
            matvec_kernel<DLM_W_BF16>
                <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size);
            break;
        case DLM_W_F16:
            matvec_kernel<DLM_W_F16>
                <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size);
            break;
        case DLM_W_INT4:
            matvec_kernel<DLM_W_INT4>
                <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size);
            break;
        case DLM_W_INT8:
            matvec_kernel<DLM_W_INT8>
                <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size);
            break;
        default:
            matvec_kernel<DLM_W_F32>
                <<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size);
            break;
    }
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
enum { SCRATCH_N = 24 };
enum { ATTN_SCORES = 19, LM_X = 20, LM_LOGITS = 21, MLA_U = 22, MLA_M = 23 };
enum { MOE_NORMED2 = 11, MOE_MATVEC = 12 };
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
static cudaError_t launch_attention(const float* q, const void* keys, const void* values,
                                    const int* table, int shift, float* ctx, int num_heads,
                                    int num_kv_heads, int head_dim, int positions,
                                    int sliding_window, float scale, float softcap, int kv_half) {
    const int B = 256;
    int start = (sliding_window > 0 && positions > sliding_window) ? positions - sliding_window : 0;
    int span = positions - start;
    // The span grows by one every token; sizing to it exactly would free and
    // re-malloc (two synchronizing driver calls) on every call. Grow in steps.
    cudaError_t e = scratch_ensure(ATTN_SCORES, num_heads * ((span + 1023) & ~1023));
    if (e != cudaSuccess) return e;
    float* scores = g_scratch[ATTN_SCORES];
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

// `kv_keys` / `kv_values` are **persistent** device buffers// `kv_keys` / `kv_values` are **persistent** device buffers (capacity
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
        // Attention sublayer.
        DLM_NORM(x, in_norm, o->in_norm_bias, normed)
        launch_matvec(DLM_WDT(DLM_W_Q), q_proj, normed, q_bias, q, q_dim, hidden_size, DLM_WGRP(DLM_W_Q));
        launch_matvec(DLM_WDT(DLM_W_K), k_proj, normed, k_bias, k, kv_dim, hidden_size, DLM_WGRP(DLM_W_K));
        launch_matvec(DLM_WDT(DLM_W_V), v_proj, normed, v_bias, v, kv_dim, hidden_size, DLM_WGRP(DLM_W_V));
        // Qwen3 per-head Q/K RMSNorm (NULL when absent), before RoPE.
        if (q_norm) head_rmsnorm_kernel<<<num_heads, 1>>>(q, q_norm, num_heads, head_dim, rms_eps);
        if (k_norm) head_rmsnorm_kernel<<<num_kv_heads, 1>>>(k, k_norm, num_kv_heads, head_dim, rms_eps);
        if (o->rope) {
            rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, inv_freq, rope_mscale);
            rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, inv_freq, rope_mscale);
        }

        // Append this token's K/V into the persistent history at slot num_positions.
        launch_kv_append(k, v, kv_keys, kv_values, o->kv_table, o->kv_block_shift, num_positions,
                         kv_dim, o->kv_half);

        // Attend over history + this token, reading the persistent buffers directly.
        e = launch_attention(q, kv_keys, kv_values, o->kv_table, o->kv_block_shift, ctx, num_heads, num_kv_heads, head_dim, total_pos, sliding_window, attn_scale, attn_softcap, o->kv_half);
        if (e != cudaSuccess) return (int)e;
        launch_matvec(DLM_WDT(DLM_W_O), o_proj, ctx, o->o_bias, attn_out, hidden_size, q_dim, DLM_WGRP(DLM_W_O));
        // Gemma2 norms the attention output before the residual add (in place, so
        // the add below is unchanged); elsewhere it goes in raw.
        if (gemma2) rmsnorm_kernel<<<1, RMS_THREADS>>>(attn_out, post_norm, attn_out, hidden_size, rms_eps);
        // Falcon's parallel block: the FFN reads the norm of the block *input*,
        // so take it before the attention output lands in `x`.
        if (o->parallel) DLM_NORM(x, post_norm, o->post_norm_bias, normed2)
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);

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
        launch_matvec(DLM_WDT(DLM_W_DOWN), down_proj, inter_buf, o->down_bias, down, hidden_size, inter, DLM_WGRP(DLM_W_DOWN));
        if (gemma2) rmsnorm_kernel<<<1, RMS_THREADS>>>(down, post_ffn_norm, down, hidden_size, rms_eps);
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, down, hidden_size);

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
// history length and position — stays per-slot: the norms, RoPE and attention run
// once per sequence, which is cheap (they touch activations, not weights).
#define DLM_MAX_BATCH 16
typedef struct { float* p[DLM_MAX_BATCH]; } DlmSlots;
typedef struct { const int* p[DLM_MAX_BATCH]; } DlmTables;
typedef struct { int v[DLM_MAX_BATCH]; } DlmInts;

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

static void launch_matvec_batched(int dt, const void* W, const float* x, const float* bias,
                                  float* out, int out_dim, int in_dim, int group_size, int batch) {
    switch (dt) {
        case DLM_W_BF16:
            matvec_batched_kernel<DLM_W_BF16><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch);
            break;
        case DLM_W_F16:
            matvec_batched_kernel<DLM_W_F16><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch);
            break;
        case DLM_W_INT4:
            matvec_batched_kernel<DLM_W_INT4><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch);
            break;
        case DLM_W_INT8:
            matvec_batched_kernel<DLM_W_INT8><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch);
            break;
        default:
            matvec_batched_kernel<DLM_W_F32><<<out_dim, MATVEC_THREADS>>>(W, x, bias, out, out_dim, in_dim, group_size, batch);
            break;
    }
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

    for (int b = 0; b < batch; ++b) {
        float* qb = q + (long)b * q_dim;
        float* kb = k + (long)b * kv_dim;
        float* vb = v + (long)b * kv_dim;
        float* ctxb = ctx + (long)b * q_dim;
        int np = num_positions->v[b];
        int pos = positions->v[b];
        int total_pos = np + 1;

        if (q_norm) head_rmsnorm_kernel<<<num_heads, 1>>>(qb, q_norm, num_heads, head_dim, rms_eps);
        if (k_norm) head_rmsnorm_kernel<<<num_kv_heads, 1>>>(kb, k_norm, num_kv_heads, head_dim, rms_eps);
        if (o->rope) {
            rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(qb, num_heads, head_dim, pos, inv_freq, rope_mscale);
            rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(kb, num_kv_heads, head_dim, pos, inv_freq, rope_mscale);
        }
        launch_kv_append(kb, vb, kv_keys->p[b], kv_values->p[b], kv_tables->p[b],
                         o->kv_block_shift, np, kv_dim, o->kv_half);
        e = launch_attention(qb, kv_keys->p[b], kv_values->p[b], kv_tables->p[b],
                             o->kv_block_shift, ctxb, num_heads, num_kv_heads,
                             head_dim, total_pos, sliding_window, attn_scale, attn_softcap, o->kv_half);
        if (e != cudaSuccess) return (int)e;
    }

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

// The LM head: `logits_host[0..vocab] = W · x_host`. The host applies the final
// norm (one hidden-wide vector) and uploads the result; the vocabulary-wide GEMV
// -- by far the largest single matvec in a decode step -- runs here, and the
// logits come back in one blocking copy.
extern "C" int dlm_lm_head(int vocab, int hidden_size, int w_dtype, int w_group_size,
                           const void* w, const float* x_host, float* logits_host)
{
    DLM_ENTER
    cudaError_t e = scratch_ensure(LM_X, hidden_size);
    if (e == cudaSuccess) e = scratch_ensure(LM_LOGITS, vocab);
    if (e == cudaSuccess)
        e = cudaMemcpy(g_scratch[LM_X], x_host, (size_t)hidden_size * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (e == cudaSuccess) {
        launch_matvec(w_dtype, w, g_scratch[LM_X], (const float*)0, g_scratch[LM_LOGITS],
                      vocab, hidden_size, w_group_size);
        e = cudaGetLastError();
    }
    if (e == cudaSuccess)
        e = cudaMemcpy(logits_host, g_scratch[LM_LOGITS], (size_t)vocab * sizeof(float),
                       cudaMemcpyDeviceToHost);
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
    switch (dt) {
        case DLM_W_BF16:
            grouped_matvec_kernel<DLM_W_BF16><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size);
            break;
        case DLM_W_F16:
            grouped_matvec_kernel<DLM_W_F16><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size);
            break;
        case DLM_W_INT4:
            grouped_matvec_kernel<DLM_W_INT4><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size);
            break;
        case DLM_W_INT8:
            grouped_matvec_kernel<DLM_W_INT8><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size);
            break;
        default:
            grouped_matvec_kernel<DLM_W_F32><<<blocks, MATVEC_THREADS>>>(W, x, out, out_dim, in_dim, group_size);
            break;
    }
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
    switch (dt) {
        case DLM_W_BF16:
            grouped_down_kernel<DLM_W_BF16><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size);
            break;
        case DLM_W_F16:
            grouped_down_kernel<DLM_W_F16><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size);
            break;
        case DLM_W_INT4:
            grouped_down_kernel<DLM_W_INT4><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size);
            break;
        case DLM_W_INT8:
            grouped_down_kernel<DLM_W_INT8><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size);
            break;
        default:
            grouped_down_kernel<DLM_W_F32><<<hidden, MATVEC_THREADS>>>(W, weights, inter_buf, x, n_experts, hidden, inter, group_size);
            break;
    }
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
    switch (dt) {
        case DLM_W_BF16: DLM_MLA_DT(DLM_W_BF16) break;
        case DLM_W_F16: DLM_MLA_DT(DLM_W_F16) break;
        case DLM_W_INT4: DLM_MLA_DT(DLM_W_INT4) break;
        case DLM_W_INT8: DLM_MLA_DT(DLM_W_INT8) break;
        default: DLM_MLA_DT(DLM_W_F32) break;
    }
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

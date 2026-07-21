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

// Row-major [out_dim, in_dim] matrix times vector, plus an optional bias.
// Threads per block for the GEMV reduction; power of two for the tree reduction.
#define MATVEC_THREADS 256

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
    __shared__ float partial[MATVEC_THREADS];
    float s = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        s += load_w<DT>(W, base + i, n, group_size) * x[i];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[o] = partial[0] + (bias ? bias[o] : 0.0f);
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

// Grouped-query attention over `positions` cached tokens. One thread per query
// head; online softmax so no per-position scratch is needed.
// `sliding_window > 0` bounds attention to the last `sliding_window` positions
// (Mistral); `0` is full causal attention. Mirrors `attention()` in
// src/forward/cpu.rs (start = positions - window).
// `scale` is passed in rather than derived from head_dim: Gemma2 decouples it
// (`query_pre_attn_scalar`). `softcap > 0` squashes each logit through
// `tanh(s/cap)*cap` before the softmax (Gemma2); 0 disables it. Both mirror
// `attention()` in src/forward/cpu.rs.
__global__ void attention_kernel(const float* q, const float* keys, const float* values,
                                 float* ctx, int num_heads, int num_kv_heads, int head_dim,
                                 int positions, int sliding_window, float scale, float softcap) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    int group = num_heads / num_kv_heads;
    int kvh = h / group;
    int kv_dim = num_kv_heads * head_dim;
    const float* qh = q + h * head_dim;
    float* out = ctx + h * head_dim;
    int start = (sliding_window > 0 && positions > sliding_window)
                    ? positions - sliding_window
                    : 0;

    float maxv = -1e30f;
    for (int p = start; p < positions; ++p) {
        const float* kh = keys + (long)p * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        dot *= scale;
        if (softcap > 0.0f) dot = tanhf(dot / softcap) * softcap;
        if (dot > maxv) maxv = dot;
    }
    for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
    float denom = 0.0f;
    for (int p = start; p < positions; ++p) {
        const float* kh = keys + (long)p * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        dot *= scale;
        if (softcap > 0.0f) dot = tanhf(dot / softcap) * softcap;
        float e = expf(dot - maxv);
        denom += e;
        const float* vh = values + (long)p * kv_dim + kvh * head_dim;
        for (int d = 0; d < head_dim; ++d) out[d] += e * vh[d];
    }
    for (int d = 0; d < head_dim; ++d) out[d] /= denom;
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
// thread_local so each inference thread owns its scratch: the multi-GPU path runs
// one thread per device (each with its own CUDA context), and the test harness
// runs cases in parallel threads — a single global would race and corrupt
// buffers across them. Per-thread scratch is allocated on that thread's current
// device and reclaimed by the driver at process exit.
// Slots 0-10 are the dense decode block's scratch. Slots 11-12 are MoE-only:
// 11 holds `normed2` (the FFN input) so it survives from `dlm_moe_attn` across
// the router matvec and every per-expert apply on the same stream; 12 is the
// router/shared-gate matvec output staged for the D2H copy.
// Slots 13-18 are MLA-only: normed, q (nh*qk), c_q (q_lora), kv_a (latent+rope),
// c_kv (latent), and the attention context (nh*v_head_dim).
enum { SCRATCH_N = 19 };
enum { MOE_NORMED2 = 11, MOE_MATVEC = 12 };
enum { MLA_NORMED = 13, MLA_Q = 14, MLA_CQ = 15, MLA_KVA = 16, MLA_CKV = 17, MLA_CTX = 18 };
static thread_local float* g_scratch[SCRATCH_N] = {0};
static thread_local int g_scratch_cap[SCRATCH_N] = {0}; // capacity in floats

// Ensure scratch slot `i` holds at least `n` floats; (re)allocates only on growth.
static cudaError_t scratch_ensure(int i, int n) {
    if (g_scratch_cap[i] >= n) return cudaSuccess;
    if (g_scratch[i]) cudaFree(g_scratch[i]);
    g_scratch[i] = 0;
    g_scratch_cap[i] = 0;
    cudaError_t e = cudaMalloc(&g_scratch[i], (size_t)n * sizeof(float));
    if (e == cudaSuccess) g_scratch_cap[i] = n;
    return e;
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
    const float* pre_ffn_norm, const float* post_ffn_norm)
{
    const int B = 256;
    int total_pos = num_positions + 1;
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
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, in_norm, normed, hidden_size, rms_eps);
        launch_matvec(w_dtype, q_proj, normed, q_bias, q, q_dim, hidden_size, w_group_size);
        launch_matvec(w_dtype, k_proj, normed, k_bias, k, kv_dim, hidden_size, w_group_size);
        launch_matvec(w_dtype, v_proj, normed, v_bias, v, kv_dim, hidden_size, w_group_size);
        // Qwen3 per-head Q/K RMSNorm (NULL when absent), before RoPE.
        if (q_norm) head_rmsnorm_kernel<<<num_heads, 1>>>(q, q_norm, num_heads, head_dim, rms_eps);
        if (k_norm) head_rmsnorm_kernel<<<num_kv_heads, 1>>>(k, k_norm, num_kv_heads, head_dim, rms_eps);
        rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, inv_freq, rope_mscale);
        rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, inv_freq, rope_mscale);

        // Append this token's K/V into the persistent history at slot num_positions.
        copy_kernel<<<grid_for(kv_dim, B), B>>>(k, kv_keys + (long)num_positions * kv_dim, kv_dim);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(v, kv_values + (long)num_positions * kv_dim, kv_dim);

        // Attend over history + this token, reading the persistent buffers directly.
        attention_kernel<<<grid_for(num_heads, B), B>>>(q, kv_keys, kv_values, ctx, num_heads, num_kv_heads, head_dim, total_pos, sliding_window, attn_scale, attn_softcap);
        launch_matvec(w_dtype, o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim, w_group_size);
        // Gemma2 norms the attention output before the residual add (in place, so
        // the add below is unchanged); elsewhere it goes in raw.
        if (gemma2) rmsnorm_kernel<<<1, RMS_THREADS>>>(attn_out, post_norm, attn_out, hidden_size, rms_eps);
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);

        // MLP sublayer (SwiGLU). Gemma2 uses its dedicated pre-FFN norm here,
        // since `post_norm` was already spent on the attention output.
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, gemma2 ? pre_ffn_norm : post_norm, normed2, hidden_size, rms_eps);
        launch_matvec(w_dtype, gate_proj, normed2, (const float*)0, gate, inter, hidden_size, w_group_size);
        launch_matvec(w_dtype, up_proj, normed2, (const float*)0, up, inter, hidden_size, w_group_size);
        swiglu_kernel<<<grid_for(inter, B), B>>>(gate, up, inter_buf, inter, activation);
        launch_matvec(w_dtype, down_proj, inter_buf, (const float*)0, down, hidden_size, inter, w_group_size);
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

    // Scratch is persistent — not freed here. It is reused across every layer and
    // token and reclaimed by the driver at process exit.
    return (int)e;
}

// ── Batched decode block (many sequences, one weight read) ─────────────────
//
// Decoding B sequences by calling `dlm_decode_block` B times reads every weight
// matrix B times. Decode is memory-bound on exactly those reads, so the win here
// is not fusing arithmetic but **streaming each weight row once and using it for
// all B slots**: `matvec_batched_kernel` holds a row in registers across the
// batch loop, turning B GEMVs into one GEMM-shaped pass.
//
// Per-slot state that cannot be batched — each sequence has its own KV buffer,
// history length and position — stays per-slot: the norms, RoPE and attention run
// once per sequence, which is cheap (they touch activations, not weights).
#define DLM_MAX_BATCH 16
typedef struct { float* p[DLM_MAX_BATCH]; } DlmSlots;
typedef struct { int v[DLM_MAX_BATCH]; } DlmInts;

// out[b*out_dim + o] = dot(W[o], x + b*in_dim) (+ bias[o]) for all b.
// One block per output row; the row is read once and reused across the batch.
template <int DT>
__global__ void matvec_batched_kernel(const void* W, const float* x, const float* bias,
                                      float* out, int out_dim, int in_dim, int group_size,
                                      int batch) {
    __shared__ float partial[MATVEC_THREADS];
    int o = blockIdx.x;
    if (o >= out_dim) return;
    long base = (long)o * in_dim;
    long n = (long)out_dim * in_dim;

    for (int b = 0; b < batch; ++b) {
        const float* xb = x + (long)b * in_dim;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
            acc += load_w<DT>(W, base + i, n, group_size) * xb[i];
        }
        partial[threadIdx.x] = acc;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out[(long)b * out_dim + o] = partial[0] + (bias ? bias[o] : 0.0f);
        }
        __syncthreads();
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
    const DlmInts* num_positions, const DlmInts* positions,
    int batch,
    int sliding_window, int activation, float rope_mscale,
    float attn_scale, float attn_softcap,
    const float* pre_ffn_norm, const float* post_ffn_norm)
{
    const int B = 256;
    if (batch <= 0) return 0;
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
    rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(x, in_norm, normed, hidden_size, rms_eps);
    launch_matvec_batched(w_dtype, q_proj, normed, q_bias, q, q_dim, hidden_size, w_group_size, batch);
    launch_matvec_batched(w_dtype, k_proj, normed, k_bias, k, kv_dim, hidden_size, w_group_size, batch);
    launch_matvec_batched(w_dtype, v_proj, normed, v_bias, v, kv_dim, hidden_size, w_group_size, batch);

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
        rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(qb, num_heads, head_dim, pos, inv_freq, rope_mscale);
        rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(kb, num_kv_heads, head_dim, pos, inv_freq, rope_mscale);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(kb, kv_keys->p[b] + (long)np * kv_dim, kv_dim);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(vb, kv_values->p[b] + (long)np * kv_dim, kv_dim);
        attention_kernel<<<grid_for(num_heads, B), B>>>(qb, kv_keys->p[b], kv_values->p[b], ctxb,
                                                        num_heads, num_kv_heads, head_dim,
                                                        total_pos, sliding_window, attn_scale,
                                                        attn_softcap);
    }

    launch_matvec_batched(w_dtype, o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim, w_group_size, batch);
    if (gemma2) {
        rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(attn_out, post_norm, attn_out, hidden_size, rms_eps);
    }
    add_inplace_batched_kernel<<<grid_for(batch * hidden_size, B), B>>>(x, attn_out, (long)batch * hidden_size);

    // MLP sublayer — all batched.
    rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(x, gemma2 ? pre_ffn_norm : post_norm, normed2, hidden_size, rms_eps);
    launch_matvec_batched(w_dtype, gate_proj, normed2, (const float*)0, gate, inter, hidden_size, w_group_size, batch);
    launch_matvec_batched(w_dtype, up_proj, normed2, (const float*)0, up, inter, hidden_size, w_group_size, batch);
    swiglu_batched_kernel<<<grid_for(batch * inter, B), B>>>(gate, up, inter_buf, (long)batch * inter, activation);
    launch_matvec_batched(w_dtype, down_proj, inter_buf, (const float*)0, down, hidden_size, inter, w_group_size, batch);
    if (gemma2) {
        rmsnorm_batched_kernel<<<batch, RMS_THREADS>>>(down, post_ffn_norm, down, hidden_size, rms_eps);
    }
    add_inplace_batched_kernel<<<grid_for(batch * hidden_size, B), B>>>(x, down, (long)batch * hidden_size);
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
    float attn_softcap)                    // 0 = off (Gemma2 caps attention logits)
{
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
        launch_matvec(w_dtype, q_proj, normed, q_bias, q, q_dim, hidden_size, w_group_size);
        launch_matvec(w_dtype, k_proj, normed, k_bias, k, kv_dim, hidden_size, w_group_size);
        launch_matvec(w_dtype, v_proj, normed, v_bias, v, kv_dim, hidden_size, w_group_size);
        // Qwen3 per-head Q/K RMSNorm (NULL when absent), before RoPE.
        if (q_norm) head_rmsnorm_kernel<<<num_heads, 1>>>(q, q_norm, num_heads, head_dim, rms_eps);
        if (k_norm) head_rmsnorm_kernel<<<num_kv_heads, 1>>>(k, k_norm, num_kv_heads, head_dim, rms_eps);
        rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, inv_freq, rope_mscale);
        rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, inv_freq, rope_mscale);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(k, kv_keys + (long)num_positions * kv_dim, kv_dim);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(v, kv_values + (long)num_positions * kv_dim, kv_dim);
        attention_kernel<<<grid_for(num_heads, B), B>>>(q, kv_keys, kv_values, ctx, num_heads, num_kv_heads, head_dim, total_pos, sliding_window, attn_scale, attn_softcap);
        launch_matvec(w_dtype, o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim, w_group_size);
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

    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        acc += load_w<DT>(We, base + i, n, group_size) * x[i];
    }
    partial[threadIdx.x] = acc;
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

// Reconstruct kv_b row `row` (of the `[., latent]` up-projection) dotted with the
// cached latent `c_kv` — i.e. one element of the on-the-fly K or V reconstruction.
template <int DT>
__device__ __forceinline__ float mla_row_dot(const void* kv_b, long row, int latent,
                                             const float* c_kv, long n, int group) {
    long base = row * (long)latent;
    float s = 0.0f;
    for (int l = 0; l < latent; ++l) s += load_w<DT>(kv_b, base + l, n, group) * c_kv[l];
    return s;
}

// One thread per query head. Reconstructs k_nope/v for each cached position from
// the latent via `kv_b` (dtype DT), scores with the split nope/rope query, and
// accumulates the value context. Slow but a faithful oracle mirror.
template <int DT>
__global__ void mla_attention_kernel(const float* q, const float* kv_keys, const void* kv_b,
                                     int kv_b_group, float* ctx, int num_heads, int nope, int rope,
                                     int vdim, int latent, int positions, float scale) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    int qk = nope + rope;
    int kv_dim = latent + rope;
    int per_head = nope + vdim;
    long kv_b_n = (long)num_heads * per_head * latent;
    const float* q_nope = q + (long)h * qk;
    const float* q_rope = q + (long)h * qk + nope;
    float* out = ctx + (long)h * vdim;

    float maxv = -1e30f;
    for (int p = 0; p < positions; ++p) {
        const float* ck = kv_keys + (long)p * kv_dim;
        const float* kpe = ck + latent;
        float sc = 0.0f;
        for (int d = 0; d < nope; ++d)
            sc += q_nope[d] * mla_row_dot<DT>(kv_b, (long)h * per_head + d, latent, ck, kv_b_n, kv_b_group);
        for (int d = 0; d < rope; ++d) sc += q_rope[d] * kpe[d];
        sc *= scale;
        if (sc > maxv) maxv = sc;
    }
    for (int d = 0; d < vdim; ++d) out[d] = 0.0f;
    float denom = 0.0f;
    for (int p = 0; p < positions; ++p) {
        const float* ck = kv_keys + (long)p * kv_dim;
        const float* kpe = ck + latent;
        float sc = 0.0f;
        for (int d = 0; d < nope; ++d)
            sc += q_nope[d] * mla_row_dot<DT>(kv_b, (long)h * per_head + d, latent, ck, kv_b_n, kv_b_group);
        for (int d = 0; d < rope; ++d) sc += q_rope[d] * kpe[d];
        float ex = expf(sc * scale - maxv);
        denom += ex;
        for (int d = 0; d < vdim; ++d)
            out[d] += ex * mla_row_dot<DT>(kv_b, (long)h * per_head + nope + d, latent, ck, kv_b_n, kv_b_group);
    }
    for (int d = 0; d < vdim; ++d) out[d] /= denom;
}

static void launch_mla_attention(int dt, const float* q, const float* kv_keys, const void* kv_b,
                                 int kv_b_group, float* ctx, int num_heads, int nope, int rope,
                                 int vdim, int latent, int positions, float scale) {
    int grid = grid_for(num_heads, 64);
    switch (dt) {
        case DLM_W_BF16:
            mla_attention_kernel<DLM_W_BF16><<<grid, 64>>>(q, kv_keys, kv_b, kv_b_group, ctx, num_heads, nope, rope, vdim, latent, positions, scale);
            break;
        case DLM_W_F16:
            mla_attention_kernel<DLM_W_F16><<<grid, 64>>>(q, kv_keys, kv_b, kv_b_group, ctx, num_heads, nope, rope, vdim, latent, positions, scale);
            break;
        case DLM_W_INT4:
            mla_attention_kernel<DLM_W_INT4><<<grid, 64>>>(q, kv_keys, kv_b, kv_b_group, ctx, num_heads, nope, rope, vdim, latent, positions, scale);
            break;
        case DLM_W_INT8:
            mla_attention_kernel<DLM_W_INT8><<<grid, 64>>>(q, kv_keys, kv_b, kv_b_group, ctx, num_heads, nope, rope, vdim, latent, positions, scale);
            break;
        default:
            mla_attention_kernel<DLM_W_F32><<<grid, 64>>>(q, kv_keys, kv_b, kv_b_group, ctx, num_heads, nope, rope, vdim, latent, positions, scale);
            break;
    }
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
    launch_mla_attention(w_dtype, q, kv_keys, kv_b_proj, w_group_size, ctx, num_heads, qk_nope, qk_rope, v_head_dim, latent, total_pos, scale);
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

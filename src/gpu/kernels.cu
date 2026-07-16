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

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
__global__ void rope_kernel(float* v, int num_heads, int head_dim, int position,
                            const float* inv_freq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = num_heads * half;
    if (idx >= total) return;
    int h = idx / half;
    int i = idx % half;
    int base = h * head_dim;
    float ang = (float)position * inv_freq[i];
    float s = sinf(ang), c = cosf(ang);
    float a = v[base + i];
    float b = v[base + i + half];
    v[base + i] = a * c - b * s;
    v[base + i + half] = a * s + b * c;
}

// Grouped-query attention over `positions` cached tokens. One thread per query
// head; online softmax so no per-position scratch is needed.
__global__ void attention_kernel(const float* q, const float* keys, const float* values,
                                 float* ctx, int num_heads, int num_kv_heads, int head_dim,
                                 int positions) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    int group = num_heads / num_kv_heads;
    int kvh = h / group;
    int kv_dim = num_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    const float* qh = q + h * head_dim;
    float* out = ctx + h * head_dim;

    float maxv = -1e30f;
    for (int p = 0; p < positions; ++p) {
        const float* kh = keys + (long)p * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }
    for (int d = 0; d < head_dim; ++d) out[d] = 0.0f;
    float denom = 0.0f;
    for (int p = 0; p < positions; ++p) {
        const float* kh = keys + (long)p * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        float e = expf(dot * scale - maxv);
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

// out[i] = silu(gate[i]) * up[i]
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
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
enum { SCRATCH_N = 11 };
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
    const float* inv_freq,                 // [head_dim/2], precomputed host-side
    float* x,                              // [hidden] in/out
    float* kv_keys, float* kv_values,      // persistent device KV, mutated in place
    int num_positions, int position)
{
    const int B = 256;
    int total_pos = num_positions + 1;

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
        rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, inv_freq);
        rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, inv_freq);

        // Append this token's K/V into the persistent history at slot num_positions.
        copy_kernel<<<grid_for(kv_dim, B), B>>>(k, kv_keys + (long)num_positions * kv_dim, kv_dim);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(v, kv_values + (long)num_positions * kv_dim, kv_dim);

        // Attend over history + this token, reading the persistent buffers directly.
        attention_kernel<<<grid_for(num_heads, B), B>>>(q, kv_keys, kv_values, ctx, num_heads, num_kv_heads, head_dim, total_pos);
        launch_matvec(w_dtype, o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim, w_group_size);
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);

        // MLP sublayer (SwiGLU).
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, post_norm, normed2, hidden_size, rms_eps);
        launch_matvec(w_dtype, gate_proj, normed2, (const float*)0, gate, inter, hidden_size, w_group_size);
        launch_matvec(w_dtype, up_proj, normed2, (const float*)0, up, inter, hidden_size, w_group_size);
        swiglu_kernel<<<grid_for(inter, B), B>>>(gate, up, inter_buf, inter);
        launch_matvec(w_dtype, down_proj, inter_buf, (const float*)0, down, hidden_size, inter, w_group_size);
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

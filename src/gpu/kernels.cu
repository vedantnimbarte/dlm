// CUDA reference kernels for one transformer decode block.
//
// This mirrors the CPU oracle in `src/forward/cpu.rs` op-for-op so the two can
// be cross-validated on hardware. The kernels are deliberately naive (one
// thread per output element, no tiling, no fusion, no cuBLAS) — correctness and
// readability over speed. A production version would fuse these and use cuBLAS /
// tensor cores.
//
// Entry point: `dlm_decode_block`, called from Rust (see src/forward/gpu.rs)
// via FFI. All pointers are device pointers. Returns a cudaError_t (0 == ok).
//
// NOTE: this file requires nvcc to compile and a GPU to run; it is compiled only
// under the `cuda-kernels` Cargo feature and has not been executed in the
// environment it was authored in. Treat as reference until validated on device.

#include <cuda_runtime.h>
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
// One thread per output row. `bias` may be NULL (Llama/Mistral have no attention
// bias; Qwen2 does — dropping it silently corrupts attention).
__global__ void matvec_kernel(const float* W, const float* x, const float* bias, float* out,
                              int out_dim, int in_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    float s = 0.0f;
    const float* row = W + (long)o * in_dim;
    for (int i = 0; i < in_dim; ++i) s += row[i] * x[i];
    if (bias) s += bias[o];
    out[o] = s;
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

// `kv_keys` / `kv_values` are **persistent** device buffers (capacity
// max_positions * kv_dim) owned by the caller across the whole sequence. This
// call writes the new token's K/V into slot `num_positions` in place and attends
// over the first `num_positions + 1` slots — so the KV history never leaves VRAM
// and only the hidden vector crosses the PCIe bus per token.
extern "C" int dlm_decode_block(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    int inter, float rms_eps,
    const float* q_proj, const float* k_proj, const float* v_proj, const float* o_proj,
    const float* gate_proj, const float* up_proj, const float* down_proj,
    const float* in_norm, const float* post_norm,
    const float* q_bias, const float* k_bias, const float* v_bias,  // may be NULL
    const float* inv_freq,                 // [head_dim/2], precomputed host-side
    float* x,                              // [hidden] in/out
    float* kv_keys, float* kv_values,      // persistent device KV, mutated in place
    int num_positions, int position)
{
    const int B = 256;
    int total_pos = num_positions + 1;

    float *normed = 0, *q = 0, *k = 0, *v = 0, *ctx = 0, *attn_out = 0, *normed2 = 0;
    float *gate = 0, *up = 0, *inter_buf = 0, *down = 0;

    // Any cudaMalloc can fail (OOM is the common case on a small card); bail out
    // with the real error instead of launching kernels on NULL pointers.
    cudaError_t e = cudaSuccess;
    #define DLM_ALLOC(p, n) if (e == cudaSuccess) { e = cudaMalloc(&(p), (n) * sizeof(float)); }
    DLM_ALLOC(normed, hidden_size)
    DLM_ALLOC(q, q_dim)
    DLM_ALLOC(k, kv_dim)
    DLM_ALLOC(v, kv_dim)
    DLM_ALLOC(ctx, q_dim)
    DLM_ALLOC(attn_out, hidden_size)
    DLM_ALLOC(normed2, hidden_size)
    DLM_ALLOC(gate, inter)
    DLM_ALLOC(up, inter)
    DLM_ALLOC(inter_buf, inter)
    DLM_ALLOC(down, hidden_size)
    #undef DLM_ALLOC

    if (e == cudaSuccess) {
        // Attention sublayer.
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, in_norm, normed, hidden_size, rms_eps);
        matvec_kernel<<<grid_for(q_dim, B), B>>>(q_proj, normed, q_bias, q, q_dim, hidden_size);
        matvec_kernel<<<grid_for(kv_dim, B), B>>>(k_proj, normed, k_bias, k, kv_dim, hidden_size);
        matvec_kernel<<<grid_for(kv_dim, B), B>>>(v_proj, normed, v_bias, v, kv_dim, hidden_size);
        rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, inv_freq);
        rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, inv_freq);

        // Append this token's K/V into the persistent history at slot num_positions.
        copy_kernel<<<grid_for(kv_dim, B), B>>>(k, kv_keys + (long)num_positions * kv_dim, kv_dim);
        copy_kernel<<<grid_for(kv_dim, B), B>>>(v, kv_values + (long)num_positions * kv_dim, kv_dim);

        // Attend over history + this token, reading the persistent buffers directly.
        attention_kernel<<<grid_for(num_heads, B), B>>>(q, kv_keys, kv_values, ctx, num_heads, num_kv_heads, head_dim, total_pos);
        matvec_kernel<<<grid_for(hidden_size, B), B>>>(o_proj, ctx, (const float*)0, attn_out, hidden_size, q_dim);
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);

        // MLP sublayer (SwiGLU).
        rmsnorm_kernel<<<1, RMS_THREADS>>>(x, post_norm, normed2, hidden_size, rms_eps);
        matvec_kernel<<<grid_for(inter, B), B>>>(gate_proj, normed2, (const float*)0, gate, inter, hidden_size);
        matvec_kernel<<<grid_for(inter, B), B>>>(up_proj, normed2, (const float*)0, up, inter, hidden_size);
        swiglu_kernel<<<grid_for(inter, B), B>>>(gate, up, inter_buf, inter);
        matvec_kernel<<<grid_for(hidden_size, B), B>>>(down_proj, inter_buf, (const float*)0, down, hidden_size, inter);
        add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, down, hidden_size);

        e = cudaDeviceSynchronize();
        if (e == cudaSuccess) e = cudaGetLastError();
    }

    cudaFree(normed); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(ctx);
    cudaFree(attn_out); cudaFree(normed2); cudaFree(gate); cudaFree(up);
    cudaFree(inter_buf); cudaFree(down);

    return (int)e;
}

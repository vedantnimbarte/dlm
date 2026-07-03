// CUDA reference kernels for one transformer decode block.
//
// This mirrors the CPU oracle in `src/forward/cpu.rs` op-for-op so the two can
// be cross-validated on hardware. The kernels are deliberately naive (one
// thread per output element, no tiling, no fusion, no cuBLAS) — correctness and
// readability over speed. A production version would fuse these and use cuBLAS /
// tensor cores.
//
// Entry point: `flip_decode_block`, called from Rust (see src/forward/gpu.rs)
// via FFI. All pointers are device pointers. Returns a cudaError_t (0 == ok).
//
// NOTE: this file requires nvcc to compile and a GPU to run; it is compiled only
// under the `cuda-kernels` Cargo feature and has not been executed in the
// environment it was authored in. Treat as reference until validated on device.

#include <cuda_runtime.h>
#include <math.h>

// out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]   (launch: <<<1, n>>>, n <= 1024)
__global__ void rmsnorm_kernel(const float* x, const float* w, float* out, int n, float eps) {
    __shared__ float inv_rms;
    int i = threadIdx.x;
    if (i == 0) {
        float ss = 0.0f;
        for (int k = 0; k < n; ++k) ss += x[k] * x[k];
        inv_rms = rsqrtf(ss / (float)n + eps);
    }
    __syncthreads();
    if (i < n) out[i] = x[i] * inv_rms * w[i];
}

// Row-major [out_dim, in_dim] matrix times vector. One thread per output row.
__global__ void matvec_kernel(const float* W, const float* x, float* out, int out_dim, int in_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    float s = 0.0f;
    const float* row = W + (long)o * in_dim;
    for (int i = 0; i < in_dim; ++i) s += row[i] * x[i];
    out[o] = s;
}

// In-place rotary embedding over [num_heads * head_dim]. One thread per rotated pair.
__global__ void rope_kernel(float* v, int num_heads, int head_dim, int position, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = num_heads * half;
    if (idx >= total) return;
    int h = idx / half;
    int i = idx % half;
    int base = h * head_dim;
    float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
    float ang = (float)position * freq;
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

extern "C" int flip_decode_block(
    int hidden_size, int q_dim, int kv_dim, int num_heads, int num_kv_heads, int head_dim,
    int inter, float rope_theta, float rms_eps,
    const float* q_proj, const float* k_proj, const float* v_proj, const float* o_proj,
    const float* gate_proj, const float* up_proj, const float* down_proj,
    const float* in_norm, const float* post_norm,
    float* x,                                         // [hidden] in/out
    const float* kv_keys, const float* kv_values,     // [num_positions * kv_dim]
    int num_positions, int position,
    float* new_key, float* new_value)                 // [kv_dim] out
{
    const int B = 256;
    int total_pos = num_positions + 1;

    float *normed, *q, *k, *v, *ctx, *attn_out, *normed2, *gate, *up, *inter_buf, *down;
    float *keys_all, *values_all;
    cudaMalloc(&normed, hidden_size * sizeof(float));
    cudaMalloc(&q, q_dim * sizeof(float));
    cudaMalloc(&k, kv_dim * sizeof(float));
    cudaMalloc(&v, kv_dim * sizeof(float));
    cudaMalloc(&ctx, q_dim * sizeof(float));
    cudaMalloc(&attn_out, hidden_size * sizeof(float));
    cudaMalloc(&normed2, hidden_size * sizeof(float));
    cudaMalloc(&gate, inter * sizeof(float));
    cudaMalloc(&up, inter * sizeof(float));
    cudaMalloc(&inter_buf, inter * sizeof(float));
    cudaMalloc(&down, hidden_size * sizeof(float));
    cudaMalloc(&keys_all, (long)total_pos * kv_dim * sizeof(float));
    cudaMalloc(&values_all, (long)total_pos * kv_dim * sizeof(float));

    // Attention sublayer.
    rmsnorm_kernel<<<1, hidden_size>>>(x, in_norm, normed, hidden_size, rms_eps);
    matvec_kernel<<<grid_for(q_dim, B), B>>>(q_proj, normed, q, q_dim, hidden_size);
    matvec_kernel<<<grid_for(kv_dim, B), B>>>(k_proj, normed, k, kv_dim, hidden_size);
    matvec_kernel<<<grid_for(kv_dim, B), B>>>(v_proj, normed, v, kv_dim, hidden_size);
    rope_kernel<<<grid_for(num_heads * (head_dim / 2), B), B>>>(q, num_heads, head_dim, position, rope_theta);
    rope_kernel<<<grid_for(num_kv_heads * (head_dim / 2), B), B>>>(k, num_kv_heads, head_dim, position, rope_theta);

    // Assemble the full K/V history (prior positions + this token) on device.
    if (num_positions > 0) {
        cudaMemcpy(keys_all, kv_keys, (long)num_positions * kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(values_all, kv_values, (long)num_positions * kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    copy_kernel<<<grid_for(kv_dim, B), B>>>(k, keys_all + (long)num_positions * kv_dim, kv_dim);
    copy_kernel<<<grid_for(kv_dim, B), B>>>(v, values_all + (long)num_positions * kv_dim, kv_dim);
    copy_kernel<<<grid_for(kv_dim, B), B>>>(k, new_key, kv_dim);
    copy_kernel<<<grid_for(kv_dim, B), B>>>(v, new_value, kv_dim);

    attention_kernel<<<grid_for(num_heads, B), B>>>(q, keys_all, values_all, ctx, num_heads, num_kv_heads, head_dim, total_pos);
    matvec_kernel<<<grid_for(hidden_size, B), B>>>(o_proj, ctx, attn_out, hidden_size, q_dim);
    add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, attn_out, hidden_size);

    // MLP sublayer (SwiGLU).
    rmsnorm_kernel<<<1, hidden_size>>>(x, post_norm, normed2, hidden_size, rms_eps);
    matvec_kernel<<<grid_for(inter, B), B>>>(gate_proj, normed2, gate, inter, hidden_size);
    matvec_kernel<<<grid_for(inter, B), B>>>(up_proj, normed2, up, inter, hidden_size);
    swiglu_kernel<<<grid_for(inter, B), B>>>(gate, up, inter_buf, inter);
    matvec_kernel<<<grid_for(hidden_size, B), B>>>(down_proj, inter_buf, down, hidden_size, inter);
    add_inplace_kernel<<<grid_for(hidden_size, B), B>>>(x, down, hidden_size);

    cudaError_t err = cudaDeviceSynchronize();

    cudaFree(normed); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(ctx);
    cudaFree(attn_out); cudaFree(normed2); cudaFree(gate); cudaFree(up);
    cudaFree(inter_buf); cudaFree(down); cudaFree(keys_all); cudaFree(values_all);

    if (err != cudaSuccess) return (int)err;
    return (int)cudaGetLastError();
}

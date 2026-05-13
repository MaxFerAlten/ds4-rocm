#pragma once

#include "../ds4_rocm_common.h"

__global__ void ds4_rocm_add_f32(float *out,
                                 const float *a,
                                 const float *b,
                                 uint32_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void ds4_rocm_swiglu_f32(float *out,
                                    const float *gate,
                                    const float *up,
                                    uint32_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float gate_v = gate[i];
    out[i] = up[i] * (gate_v / (1.0f + expf(-gate_v)));
}

__global__ void ds4_rocm_repeat_hc_f32(float *out,
                                       const float *row,
                                       uint32_t n_embd,
                                       uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_embd * n_hc;
    if (i < total) out[i] = row[i % n_embd];
}

__global__ void ds4_rocm_fill_f32(float *out, float value, uint32_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void ds4_rocm_rms_norm_plain_rows_f32(float *out,
                                                 const float *x,
                                                 uint32_t n,
                                                 uint32_t rows,
                                                 float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;

    uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * n;
    float *yr = out + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }

    extern __shared__ float sh[];
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float inv = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i] * inv;
    }
}

__global__ void ds4_rocm_rms_norm_weight_rows_f32(float *out,
                                                  const float *x,
                                                  const float *weight,
                                                  uint32_t n,
                                                  uint32_t rows,
                                                  float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;

    uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * n;
    float *yr = out + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }

    extern __shared__ float sh[];
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float inv = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i] * inv * weight[i];
    }
}

__global__ void ds4_rocm_qkv_rms_norm_rows_f32(float *q_out,
                                               const float *q,
                                               const float *q_weight,
                                               uint32_t q_n,
                                               float *kv_out,
                                               const float *kv,
                                               const float *kv_weight,
                                               uint32_t kv_n,
                                               uint32_t rows,
                                               float eps) {
    uint32_t row = blockIdx.x;
    bool kv_task = blockIdx.y != 0;
    if (row >= rows) return;

    uint32_t n = kv_task ? kv_n : q_n;
    const float *x = kv_task ? kv + (uint64_t)row * kv_n : q + (uint64_t)row * q_n;
    const float *w = kv_task ? kv_weight : q_weight;
    float *out = kv_task ? kv_out + (uint64_t)row * kv_n : q_out + (uint64_t)row * q_n;
    uint32_t tid = threadIdx.x;

    float sum = 0.0f;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }

    extern __shared__ float sh[];
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float inv = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * inv * w[i];
    }
}

__global__ void ds4_rocm_head_rms_norm_f32(float *x,
                                           uint32_t n_rows,
                                           uint32_t head_dim,
                                           float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_rows) return;

    uint32_t tid = threadIdx.x;
    float *head = x + (uint64_t)row * head_dim;

    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        float v = head[i];
        sum += v * v;
    }

    extern __shared__ float sh[];
    sh[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    float inv = rsqrtf(sh[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        head[i] *= inv;
    }
}

__global__ void ds4_rocm_embed_token_hc_f16(float *out,
                                            const uint16_t *table,
                                            uint32_t token,
                                            uint32_t n_embd,
                                            uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_embd * n_hc;
    if (idx >= total) return;

    uint32_t d = (uint32_t)(idx % n_embd);
    out[idx] = ds4_rocm_f16_to_f32(table[(uint64_t)token * n_embd + d]);
}

// ── Directional steering: one block per row ────────────────────
// x[r][c] -= scale * dir[c] * dot(dir, x[r])
// grid = (rows,), block = 256 (or up to width)
__global__ void ds4_rocm_directional_steering_f32(
    float       *x,
    const float *dir,
    float        scale,
    uint32_t     width,
    uint32_t     rows)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t n_threads = blockDim.x;

    float *xr = x + (uint64_t)row * width;

    float dot = 0.0f;
    for (uint32_t i = tid; i < width; i += n_threads)
        dot += xr[i] * dir[i];

    extern __shared__ float sh[];
    sh[tid] = dot;
    __syncthreads();
    for (uint32_t s = n_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    float d = scale * sh[0];
    for (uint32_t i = tid; i < width; i += n_threads)
        xr[i] -= d * dir[i];
}

__global__ void ds4_rocm_embed_tokens_hc_f16(float *out,
                                             const int32_t *tokens,
                                             const uint16_t *table,
                                             uint32_t n_vocab,
                                             uint32_t n_tokens,
                                             uint32_t n_embd,
                                             uint32_t n_hc) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
    if (idx >= total) return;

    uint64_t tmp = idx / n_embd;
    uint32_t d = (uint32_t)(idx % n_embd);
    uint32_t t = (uint32_t)(tmp / n_hc);
    int32_t token = tokens[t];
    if (token < 0 || (uint32_t)token >= n_vocab) return;
    out[idx] = ds4_rocm_f16_to_f32(table[(uint64_t)(uint32_t)token * n_embd + d]);
}

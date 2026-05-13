#pragma once

#include "../ds4_rocm_common.h"

__device__ inline float ds4_rocm_block_sum_256(float v, float *scratch) {
    const uint32_t tid = threadIdx.x;
    scratch[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    return scratch[0];
}

__device__ inline float ds4_rocm_attention_score_512(const float *q,
                                                     const float *kv,
                                                     float scale,
                                                     float *scratch) {
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t d = tid; d < 512u; d += blockDim.x) {
        sum += q[d] * kv[d];
    }
    return ds4_rocm_block_sum_256(sum, scratch) * scale;
}

__global__ void ds4_rocm_attention_decode_heads_f32(float *heads,
                                                    const float *sinks,
                                                    const float *q,
                                                    const float *raw_kv,
                                                    uint32_t n_raw,
                                                    uint32_t raw_cap,
                                                    uint32_t raw_start,
                                                    const float *comp_kv,
                                                    uint32_t n_comp,
                                                    const float *comp_mask,
                                                    uint32_t use_mask,
                                                    uint32_t n_head) {
    extern __shared__ float scratch[];
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (head >= n_head || tid >= 256u) return;

    const float scale = rsqrtf(512.0f);
    const float *qh = q + (uint64_t)head * 512u;
    float max_score = sinks[head];

    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t row = (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)row * 512u;
        const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
        if (tid == 0 && score > max_score) max_score = score;
        __syncthreads();
    }
    for (uint32_t r = 0; r < n_comp; r++) {
        const bool allowed = !use_mask || comp_mask[r] > -1.0e20f;
        float score = -1.0e30f;
        if (allowed) {
            const float *kv = comp_kv + (uint64_t)r * 512u;
            score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
        } else {
            __syncthreads();
        }
        if (tid == 0 && score > max_score) max_score = score;
        __syncthreads();
    }

    __shared__ float s_max;
    if (tid == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    float denom = tid == 0 ? expf(sinks[head] - max_score) : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint32_t d0 = tid;
    const uint32_t d1 = tid + 256u;

    for (uint32_t r = 0; r < n_raw; r++) {
        const uint32_t row = (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)row * 512u;
        const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
        __shared__ float s_weight;
        if (tid == 0) {
            const float w = expf(score - max_score);
            denom += w;
            s_weight = w;
        }
        __syncthreads();
        const float w = s_weight;
        acc0 += kv[d0] * w;
        acc1 += kv[d1] * w;
        __syncthreads();
    }
    for (uint32_t r = 0; r < n_comp; r++) {
        const bool allowed = !use_mask || comp_mask[r] > -1.0e20f;
        __shared__ float s_weight;
        const float *kv = comp_kv + (uint64_t)r * 512u;
        if (allowed) {
            const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
            if (tid == 0) {
                const float w = expf(score - max_score);
                denom += w;
                s_weight = w;
            }
        } else if (tid == 0) {
            s_weight = 0.0f;
        }
        __syncthreads();
        const float w = s_weight;
        if (allowed) {
            acc0 += kv[d0] * w;
            acc1 += kv[d1] * w;
        }
        __syncthreads();
    }

    __shared__ float s_denom;
    if (tid == 0) s_denom = denom;
    __syncthreads();
    const float inv = s_denom > 0.0f ? 1.0f / s_denom : 0.0f;
    float *out = heads + (uint64_t)head * 512u;
    out[d0] = acc0 * inv;
    out[d1] = acc1 * inv;
}

__global__ void ds4_rocm_attention_indexed_mixed_batch_heads_f32(float *heads,
                                                                const float *sinks,
                                                                const float *q,
                                                                const float *raw_kv,
                                                                const float *comp_kv,
                                                                const int32_t *topk,
                                                                uint32_t n_tokens,
                                                                uint32_t pos0,
                                                                uint32_t n_raw,
                                                                uint32_t raw_cap,
                                                                uint32_t raw_start,
                                                                uint32_t n_comp,
                                                                uint32_t top_k,
                                                                uint32_t window,
                                                                uint32_t ratio,
                                                                uint32_t n_head) {
    extern __shared__ float scratch[];
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (token >= n_tokens || head >= n_head || tid >= 256u) return;

    const float scale = rsqrtf(512.0f);
    const float *qh = q + ((uint64_t)token * n_head + head) * 512u;
    const uint32_t qpos = pos0 + token;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
    const uint32_t window_first = (window != 0u && qpos + 1u > window) ?
        qpos + 1u - window : 0u;
    const uint32_t first = first_raw_pos > window_first ? first_raw_pos : window_first;
    const uint32_t last = qpos < raw_last_pos ? qpos : raw_last_pos;
    const uint32_t visible = ((qpos + 1u) / ratio) < n_comp ? ((qpos + 1u) / ratio) : n_comp;
    const int32_t *row_topk = topk + (uint64_t)token * top_k;

    float max_score = sinks[head];
    if (first <= last) {
        for (uint32_t pos = first; pos <= last; pos++) {
            const uint32_t logical = pos - first_raw_pos;
            const uint32_t row = (raw_start + logical) % raw_cap;
            const float *kv = raw_kv + (uint64_t)row * 512u;
            const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
            if (tid == 0 && score > max_score) max_score = score;
            __syncthreads();
        }
    }
    for (uint32_t i = 0; i < top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint32_t)idx >= visible) break;
        const float *kv = comp_kv + (uint64_t)(uint32_t)idx * 512u;
        const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
        if (tid == 0 && score > max_score) max_score = score;
        __syncthreads();
    }

    __shared__ float s_max;
    if (tid == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    float denom = tid == 0 ? expf(sinks[head] - max_score) : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint32_t d0 = tid;
    const uint32_t d1 = tid + 256u;

    if (first <= last) {
        for (uint32_t pos = first; pos <= last; pos++) {
            const uint32_t logical = pos - first_raw_pos;
            const uint32_t row = (raw_start + logical) % raw_cap;
            const float *kv = raw_kv + (uint64_t)row * 512u;
            const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
            __shared__ float s_weight;
            if (tid == 0) {
                const float w = expf(score - max_score);
                denom += w;
                s_weight = w;
            }
            __syncthreads();
            const float w = s_weight;
            acc0 += kv[d0] * w;
            acc1 += kv[d1] * w;
            __syncthreads();
        }
    }
    for (uint32_t i = 0; i < top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint32_t)idx >= visible) break;
        const float *kv = comp_kv + (uint64_t)(uint32_t)idx * 512u;
        const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
        __shared__ float s_weight;
        if (tid == 0) {
            const float w = expf(score - max_score);
            denom += w;
            s_weight = w;
        }
        __syncthreads();
        const float w = s_weight;
        acc0 += kv[d0] * w;
        acc1 += kv[d1] * w;
        __syncthreads();
    }

    __shared__ float s_denom;
    if (tid == 0) s_denom = denom;
    __syncthreads();
    const float inv = s_denom > 0.0f ? 1.0f / s_denom : 0.0f;
    float *out = heads + ((uint64_t)token * n_head + head) * 512u;
    out[d0] = acc0 * inv;
    out[d1] = acc1 * inv;
}

__global__ void ds4_rocm_attention_mixed_batch_heads_f32(float *heads,
                                                         const float *sinks,
                                                         const float *q,
                                                         const float *raw_kv,
                                                         const float *comp_kv,
                                                         const float *comp_mask,
                                                         uint32_t use_comp_mask,
                                                         uint32_t n_tokens,
                                                         uint32_t pos0,
                                                         uint32_t n_raw,
                                                         uint32_t raw_cap,
                                                         uint32_t raw_start,
                                                         uint32_t n_comp,
                                                         uint32_t window,
                                                         uint32_t ratio,
                                                         uint32_t n_head) {
    extern __shared__ float scratch[];
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (token >= n_tokens || head >= n_head || tid >= 256u) return;

    const float scale = rsqrtf(512.0f);
    const float *qh = q + ((uint64_t)token * n_head + head) * 512u;
    const uint32_t qpos = pos0 + token;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
    const uint32_t window_first = (window != 0u && qpos + 1u > window) ?
        qpos + 1u - window : 0u;
    const uint32_t first = first_raw_pos > window_first ? first_raw_pos : window_first;
    const uint32_t last = qpos < raw_last_pos ? qpos : raw_last_pos;
    const uint32_t visible = (ratio != 0u && ((qpos + 1u) / ratio) < n_comp) ?
        ((qpos + 1u) / ratio) : n_comp;
    const float *mask_row = (use_comp_mask && comp_mask) ?
        comp_mask + (uint64_t)token * n_comp : nullptr;

    float max_score = sinks[head];
    if (first <= last) {
        for (uint32_t pos = first; pos <= last; pos++) {
            const uint32_t logical = pos - first_raw_pos;
            const uint32_t row = (raw_start + logical) % raw_cap;
            const float *kv = raw_kv + (uint64_t)row * 512u;
            const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
            if (tid == 0 && score > max_score) max_score = score;
            __syncthreads();
        }
    }
    for (uint32_t c = 0; c < visible; c++) {
        const bool allowed = !mask_row || mask_row[c] > -1.0e20f;
        float score = -1.0e30f;
        if (allowed) {
            const float *kv = comp_kv + (uint64_t)c * 512u;
            score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
        } else {
            __syncthreads();
        }
        if (tid == 0 && score > max_score) max_score = score;
        __syncthreads();
    }

    __shared__ float s_max;
    if (tid == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    float denom = tid == 0 ? expf(sinks[head] - max_score) : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint32_t d0 = tid;
    const uint32_t d1 = tid + 256u;

    if (first <= last) {
        for (uint32_t pos = first; pos <= last; pos++) {
            const uint32_t logical = pos - first_raw_pos;
            const uint32_t row = (raw_start + logical) % raw_cap;
            const float *kv = raw_kv + (uint64_t)row * 512u;
            const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
            __shared__ float s_weight;
            if (tid == 0) {
                const float w = expf(score - max_score);
                denom += w;
                s_weight = w;
            }
            __syncthreads();
            const float w = s_weight;
            acc0 += kv[d0] * w;
            acc1 += kv[d1] * w;
            __syncthreads();
        }
    }
    for (uint32_t c = 0; c < visible; c++) {
        const bool allowed = !mask_row || mask_row[c] > -1.0e20f;
        __shared__ float s_weight;
        const float *kv = comp_kv + (uint64_t)c * 512u;
        if (allowed) {
            const float score = ds4_rocm_attention_score_512(qh, kv, scale, scratch);
            if (tid == 0) {
                const float w = expf(score - max_score);
                denom += w;
                s_weight = w;
            }
        } else if (tid == 0) {
            s_weight = 0.0f;
        }
        __syncthreads();
        const float w = s_weight;
        if (allowed) {
            acc0 += kv[d0] * w;
            acc1 += kv[d1] * w;
        }
        __syncthreads();
    }

    __shared__ float s_denom;
    if (tid == 0) s_denom = denom;
    __syncthreads();
    const float inv = s_denom > 0.0f ? 1.0f / s_denom : 0.0f;
    float *out = heads + ((uint64_t)token * n_head + head) * 512u;
    out[d0] = acc0 * inv;
    out[d1] = acc1 * inv;
}

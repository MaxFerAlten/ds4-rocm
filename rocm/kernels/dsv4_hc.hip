#pragma once

#include "../ds4_rocm_common.h"

__global__ void ds4_rocm_hc_split_sinkhorn_f32(float *out,
                                               const float *mixes,
                                               const float *scale,
                                               const float *base,
                                               uint32_t n_hc,
                                               uint32_t sinkhorn_iters,
                                               uint64_t n_rows,
                                               float eps) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || n_hc == 0 || n_hc > 16) return;

    uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const float *mix = mixes + row * mix_hc;
    float *dst = out + row * mix_hc;

    float pre_scale = scale[0];
    float post_scale = scale[1];
    float comb_scale = scale[2];

    for (uint32_t i = 0; i < n_hc; i++) {
        float z = mix[i] * pre_scale + base[i];
        dst[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    for (uint32_t i = 0; i < n_hc; i++) {
        uint32_t off = n_hc + i;
        float z = mix[off] * post_scale + base[off];
        dst[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];
    for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
        float row_max = -3.4028234663852886e38f;
        for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
            uint32_t idx = src_hc + dst_hc * n_hc;
            uint32_t off = 2 * n_hc + idx;
            float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
            uint32_t idx = src_hc + dst_hc * n_hc;
            float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        float inv = 1.0f / row_sum;
        for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
            uint32_t idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float sum = 0.0f;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            sum += c[src_hc + dst_hc * n_hc];
        }
        float inv = 1.0f / (sum + eps);
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            c[src_hc + dst_hc * n_hc] *= inv;
        }
    }

    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float sum = 0.0f;
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            float inv = 1.0f / (sum + eps);
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                c[src_hc + dst_hc * n_hc] *= inv;
            }
        }
        for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
            float sum = 0.0f;
            for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            float inv = 1.0f / (sum + eps);
            for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
                c[src_hc + dst_hc * n_hc] *= inv;
            }
        }
    }

    for (uint32_t i = 0; i < n_hc * n_hc; i++) {
        dst[2 * n_hc + i] = c[i];
    }
}

__global__ void ds4_rocm_hc_weighted_sum_f32(float *out,
                                             const float *residual_hc,
                                             const float *weights,
                                             uint32_t n_embd,
                                             uint32_t n_hc,
                                             uint32_t n_tokens,
                                             uint32_t weight_stride) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_embd;
    if (idx >= total) return;

    uint32_t d = (uint32_t)(idx % n_embd);
    uint32_t t = (uint32_t)(idx / n_embd);
    const float *x = residual_hc + (uint64_t)t * n_hc * n_embd;
    const float *w = weights + (uint64_t)t * weight_stride;

    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)h * n_embd + d] * w[h];
    }
    out[idx] = acc;
}

__global__ void ds4_rocm_output_hc_weights_f32(float *out,
                                               const float *pre,
                                               const float *scale,
                                               const float *base,
                                               uint32_t n_hc,
                                               uint32_t n_tokens,
                                               float eps) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_hc;
    if (idx >= total) return;

    uint32_t h = (uint32_t)(idx % n_hc);
    float z = pre[idx] * scale[0] + base[h];
    out[idx] = 1.0f / (1.0f + expf(-z)) + eps;
}

__global__ void ds4_rocm_hc_expand_f32(float *out,
                                       const float *block_out,
                                       const float *block_add,
                                       const float *residual_hc,
                                       const float *post,
                                       const float *comb,
                                       uint32_t n_embd,
                                       uint32_t n_hc,
                                       uint32_t n_tokens,
                                       uint32_t post_stride,
                                       uint32_t comb_stride,
                                       int has_add) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_hc * n_embd;
    if (idx >= total) return;

    uint32_t d = (uint32_t)(idx % n_embd);
    uint64_t tmp = idx / n_embd;
    uint32_t dst_hc = (uint32_t)(tmp % n_hc);
    uint32_t t = (uint32_t)(tmp / n_hc);

    const float *block = block_out + (uint64_t)t * n_embd;
    const float *add = block_add + (uint64_t)t * n_embd;
    const float *res = residual_hc + (uint64_t)t * n_hc * n_embd;
    const float *post_t = post + (uint64_t)t * post_stride;
    const float *comb_t = comb + (uint64_t)t * comb_stride;

    float block_v = block[d];
    if (has_add) block_v += add[d];
    float acc = block_v * post_t[dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        acc += comb_t[dst_hc + src_hc * n_hc] *
               res[(uint64_t)src_hc * n_embd + d];
    }
    out[idx] = acc;
}

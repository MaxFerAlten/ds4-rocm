#pragma once

#include "../ds4_rocm_common.h"

__device__ inline float ds4_rocm_softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ void ds4_rocm_indexer_score_one_f32(float *scores,
                                               const float *q,
                                               const float *weights,
                                               const float *index_comp,
                                               uint32_t n_comp,
                                               uint32_t n_head,
                                               uint32_t head_dim,
                                               float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= n_comp) return;

    const float *kv = index_comp + (uint64_t)row * head_dim;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += qh[d] * kv[d];
        }

        dot = ds4_rocm_warp_sum(dot);
        __shared__ float smem[32];
        if ((threadIdx.x & 31u) == 0) smem[threadIdx.x >> 5] = dot;
        __syncthreads();

        float head_dot = 0.0f;
        if (threadIdx.x == 0) {
            const uint32_t nwarps = (blockDim.x + 31u) >> 5;
            head_dot = smem[0];
            for (uint32_t i = 1; i < nwarps; i++) head_dot += smem[i];
        }
        if (threadIdx.x == 0) acc += fmaxf(head_dot, 0.0f) * weights[h] * scale;
        __syncthreads();
    }

    if (threadIdx.x == 0) scores[row] = acc;
}

__global__ void ds4_rocm_indexer_scores_batch_f32(float *scores,
                                                  const float *q,
                                                  const float *weights,
                                                  const float *index_comp,
                                                  uint32_t n_comp,
                                                  uint32_t n_tokens,
                                                  uint32_t pos0,
                                                  uint32_t n_head,
                                                  uint32_t head_dim,
                                                  uint32_t ratio,
                                                  float scale) {
    const uint32_t comp = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (comp >= n_comp || token >= n_tokens || ratio == 0u) return;

    const uint32_t visible = min((pos0 + token + 1u) / ratio, n_comp);
    float *dst = scores + (uint64_t)token * n_comp + comp;
    if (comp >= visible) {
        if (threadIdx.x == 0) *dst = -1.0e30f;
        return;
    }

    const float *kv = index_comp + (uint64_t)comp * head_dim;
    const float *qt = q + (uint64_t)token * n_head * head_dim;
    const float *wt = weights + (uint64_t)token * n_head;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = qt + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += qh[d] * kv[d];
        }

        dot = ds4_rocm_warp_sum(dot);
        __shared__ float smem[32];
        if ((threadIdx.x & 31u) == 0) smem[threadIdx.x >> 5] = dot;
        __syncthreads();

        float head_dot = 0.0f;
        if (threadIdx.x == 0) {
            const uint32_t nwarps = (blockDim.x + 31u) >> 5;
            head_dot = smem[0];
            for (uint32_t i = 1; i < nwarps; i++) head_dot += smem[i];
        }
        if (threadIdx.x == 0) acc += fmaxf(head_dot, 0.0f) * wt[h] * scale;
        __syncthreads();
    }

    if (threadIdx.x == 0) *dst = acc;
}

__global__ void ds4_rocm_indexer_topk_i32(int32_t *selected,
                                          const float *scores,
                                          uint32_t n_comp,
                                          uint32_t n_tokens,
                                          uint32_t top_k) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens || threadIdx.x != 0) return;

    const float *row = scores + (uint64_t)token * n_comp;
    int32_t *dst = selected + (uint64_t)token * top_k;
    for (uint32_t k = 0; k < top_k; k++) dst[k] = -1;

    for (uint32_t i = 0; i < n_comp; i++) {
        for (uint32_t j = 0; j < top_k; j++) {
            if (dst[j] < 0 || row[i] > row[(uint32_t)dst[j]]) {
                for (uint32_t m = top_k - 1u; m > j; m--) dst[m] = dst[m - 1u];
                dst[j] = (int32_t)i;
                break;
            }
        }
    }
}

__global__ void ds4_rocm_topk_mask_f32(float *mask,
                                       const int32_t *topk,
                                       uint32_t n_comp,
                                       uint32_t n_tokens,
                                       uint32_t top_k) {
    const uint64_t total = (uint64_t)n_tokens * n_comp;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    const uint32_t token = (uint32_t)(gid / n_comp);
    const uint32_t col = (uint32_t)(gid % n_comp);
    float v = -1.0e30f;
    const int32_t *row_topk = topk + (uint64_t)token * top_k;
    for (uint32_t k = 0; k < top_k; k++) {
        if (row_topk[k] == (int32_t)col) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

__global__ void ds4_rocm_router_select_one(float *probs,
                                           int32_t *selected,
                                           float *weights,
                                           const float *logits,
                                           const float *bias,
                                           const int32_t *hash,
                                           uint32_t hash_rows,
                                           uint32_t token,
                                           uint32_t has_bias,
                                           uint32_t hash_mode) {
    const uint32_t tid = threadIdx.x;
    __shared__ float p[256];
    __shared__ float sel_scores[256];

    if (tid < 256u) {
        const float prob = sqrtf(ds4_rocm_softplus_stable(logits[tid]));
        p[tid] = prob;
        probs[tid] = prob;
        sel_scores[tid] = prob + (has_bias ? bias[tid] : 0.0f);
    }
    __syncthreads();

    if (tid != 0) return;

    if (hash_mode) {
        const uint32_t row = token < hash_rows ? token : hash_rows - 1u;
        const int32_t *src = hash + (uint64_t)row * 6u;
        for (uint32_t i = 0; i < 6u; i++) selected[i] = src[i];
    } else {
        for (uint32_t i = 0; i < 6u; i++) selected[i] = -1;
        for (uint32_t i = 0; i < 256u; i++) {
            for (uint32_t j = 0; j < 6u; j++) {
                if (selected[j] < 0 || sel_scores[i] > sel_scores[(uint32_t)selected[j]]) {
                    for (uint32_t m = 5u; m > j; m--) selected[m] = selected[m - 1u];
                    selected[j] = (int32_t)i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < 6u; i++) {
        const int32_t idx = selected[i];
        const float v = (idx >= 0 && idx < 256) ? p[(uint32_t)idx] : 0.0f;
        weights[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t i = 0; i < 6u; i++) weights[i] = weights[i] / sum * 1.5f;
}

__global__ void ds4_rocm_router_select_batch(float *probs,
                                             int32_t *selected,
                                             float *weights,
                                             const float *logits,
                                             const int32_t *tokens,
                                             const float *bias,
                                             const int32_t *hash,
                                             uint32_t hash_rows,
                                             uint32_t n_tokens,
                                             uint32_t has_bias,
                                             uint32_t hash_mode) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tokens) return;

    const uint32_t tid = threadIdx.x;
    __shared__ float p[256];
    __shared__ float sel_scores[256];

    const float *logits_row = logits + (uint64_t)tok * 256u;
    float *probs_row = probs + (uint64_t)tok * 256u;
    int32_t *selected_row = selected + (uint64_t)tok * 6u;
    float *weights_row = weights + (uint64_t)tok * 6u;

    if (tid < 256u) {
        const float prob = sqrtf(ds4_rocm_softplus_stable(logits_row[tid]));
        p[tid] = prob;
        probs_row[tid] = prob;
        sel_scores[tid] = prob + ((has_bias && !hash_mode) ? bias[tid] : 0.0f);
    }
    __syncthreads();

    if (tid != 0) return;

    if (hash_mode) {
        const uint32_t token_id = tokens ? (uint32_t)tokens[tok] : tok;
        const uint32_t row = token_id < hash_rows ? token_id : hash_rows - 1u;
        const int32_t *src = hash + (uint64_t)row * 6u;
        for (uint32_t i = 0; i < 6u; i++) selected_row[i] = src[i];
    } else {
        for (uint32_t i = 0; i < 6u; i++) selected_row[i] = -1;
        for (uint32_t i = 0; i < 256u; i++) {
            for (uint32_t j = 0; j < 6u; j++) {
                if (selected_row[j] < 0 || sel_scores[i] > sel_scores[(uint32_t)selected_row[j]]) {
                    for (uint32_t m = 5u; m > j; m--) selected_row[m] = selected_row[m - 1u];
                    selected_row[j] = (int32_t)i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < 6u; i++) {
        const int32_t idx = selected_row[i];
        const float v = (idx >= 0 && idx < 256) ? p[(uint32_t)idx] : 0.0f;
        weights_row[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t i = 0; i < 6u; i++) weights_row[i] = weights_row[i] / sum * 1.5f;
}

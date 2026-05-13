#pragma once

#include "../ds4_rocm_common.h"

__device__ inline float ds4_rocm_q8_0_scale(const uint8_t *block) {
    uint16_t h = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
    return ds4_rocm_f16_to_f32(h);
}

__device__ inline float ds4_rocm_dot_q8_0_f32_row(const uint8_t *row,
                                                  const float *x,
                                                  uint32_t in_dim,
                                                  uint32_t tid,
                                                  uint32_t stride) {
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += stride) {
        const uint32_t block_idx = i >> 5;
        const uint32_t q_idx = i & 31u;
        const uint8_t *block = row + (uint64_t)block_idx * 34u;
        const int8_t q = (int8_t)block[2u + q_idx];
        acc += ds4_rocm_q8_0_scale(block) * (float)q * x[i];
    }
    return acc;
}

__global__ void ds4_rocm_mul_mv_f16_f32_1tok(const uint16_t *w,
                                             const float *x,
                                             float *out,
                                             uint32_t in_dim,
                                             uint32_t out_dim) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    uint32_t tid = threadIdx.x;
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) {
        acc += ds4_rocm_f16_to_f32(w[(uint64_t)row * in_dim + i]) * x[i];
    }

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if ((tid & 31u) == 0) smem[tid >> 5] = acc;
    __syncthreads();

    if (tid == 0) {
        uint32_t nwarps = (blockDim.x + 31u) >> 5;
        float v = smem[0];
        for (uint32_t i = 1; i < nwarps; i++) v += smem[i];
        out[row] = v;
    }
}

__global__ void ds4_rocm_mul_mv_f32_f32_1tok(const float *w,
                                             const float *x,
                                             float *out,
                                             uint32_t in_dim,
                                             uint32_t out_dim) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    uint32_t tid = threadIdx.x;
    const float *wrow = w + (uint64_t)row * in_dim;
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) {
        acc += wrow[i] * x[i];
    }

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if ((tid & 31u) == 0) smem[tid >> 5] = acc;
    __syncthreads();

    if (tid == 0) {
        uint32_t nwarps = (blockDim.x + 31u) >> 5;
        float v = smem[0];
        for (uint32_t i = 1; i < nwarps; i++) v += smem[i];
        out[row] = v;
    }
}

__global__ void ds4_rocm_mul_mm_f16_f32(const uint16_t *w,
                                        const float *x,
                                        float *out,
                                        uint32_t in_dim,
                                        uint32_t out_dim,
                                        uint32_t n_tokens) {
    uint32_t row = blockIdx.x;
    uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    uint32_t tid = threadIdx.x;
    const uint16_t *wrow = w + (uint64_t)row * in_dim;
    const float *xt = x + (uint64_t)tok * in_dim;
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) {
        acc += ds4_rocm_f16_to_f32(wrow[i]) * xt[i];
    }

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if ((tid & 31u) == 0) smem[tid >> 5] = acc;
    __syncthreads();

    if (tid == 0) {
        uint32_t nwarps = (blockDim.x + 31u) >> 5;
        float v = smem[0];
        for (uint32_t i = 1; i < nwarps; i++) v += smem[i];
        out[(uint64_t)tok * out_dim + row] = v;
    }
}

__global__ void ds4_rocm_mul_mm_f32_f32(const float *w,
                                        const float *x,
                                        float *out,
                                        uint32_t in_dim,
                                        uint32_t out_dim,
                                        uint32_t n_tokens) {
    uint32_t row = blockIdx.x;
    uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    uint32_t tid = threadIdx.x;
    const float *wrow = w + (uint64_t)row * in_dim;
    const float *xt = x + (uint64_t)tok * in_dim;
    float acc = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) {
        acc += wrow[i] * xt[i];
    }

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if ((tid & 31u) == 0) smem[tid >> 5] = acc;
    __syncthreads();

    if (tid == 0) {
        uint32_t nwarps = (blockDim.x + 31u) >> 5;
        float v = smem[0];
        for (uint32_t i = 1; i < nwarps; i++) v += smem[i];
        out[(uint64_t)tok * out_dim + row] = v;
    }
}

__global__ void ds4_rocm_mul_mv_f16_pair_f32_1tok(const uint16_t *wa,
                                                  const uint16_t *wb,
                                                  const float *x,
                                                  float *out_a,
                                                  float *out_b,
                                                  uint32_t in_dim,
                                                  uint32_t out_dim) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    uint32_t tid = threadIdx.x;
    const uint16_t *row_a = wa + (uint64_t)row * in_dim;
    const uint16_t *row_b = wb + (uint64_t)row * in_dim;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint32_t i = tid; i < in_dim; i += blockDim.x) {
        const float xv = x[i];
        acc_a += ds4_rocm_f16_to_f32(row_a[i]) * xv;
        acc_b += ds4_rocm_f16_to_f32(row_b[i]) * xv;
    }

    acc_a = ds4_rocm_warp_sum(acc_a);
    acc_b = ds4_rocm_warp_sum(acc_b);
    __shared__ float smem_a[32];
    __shared__ float smem_b[32];
    if ((tid & 31u) == 0) {
        smem_a[tid >> 5] = acc_a;
        smem_b[tid >> 5] = acc_b;
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t nwarps = (blockDim.x + 31u) >> 5;
        float va = smem_a[0];
        float vb = smem_b[0];
        for (uint32_t i = 1; i < nwarps; i++) {
            va += smem_a[i];
            vb += smem_b[i];
        }
        out_a[row] = va;
        out_b[row] = vb;
    }
}

// Warp-cooperative Q8_0 dot: each thread in a warp processes one quant
// within a Q8_0 block (32 quants/block, 32 threads/warp on gfx1151).
// Scale is broadcast from lane 0 via __shfl_sync (64-bit mask for ROCm 7.2).
__device__ inline float ds4_rocm_dot_q8_0_f32_warp(
    const uint8_t *row, const float *x, uint32_t in_dim,
    uint32_t wid, uint32_t lane, uint32_t n_warps)
{
    uint32_t n_blocks = in_dim / 32;
    float acc = 0.0f;
    for (uint32_t b = wid; b < n_blocks; b += n_warps) {
        const uint8_t *block = row + (uint64_t)b * 34u;
        float scale = __shfl_sync(0xffffffffffffffffull,
                                  ds4_rocm_q8_0_scale(block), 0);
        acc += scale * (float)(int8_t)block[2u + lane] * x[(uint64_t)b * 32 + lane];
    }
    return acc;
}

__global__ void ds4_rocm_mul_mv_q8_0_f32_1tok(const uint8_t *w,
                                              const float *x,
                                              float *out,
                                              uint32_t in_dim,
                                              uint32_t out_dim) {
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    uint32_t tid = threadIdx.x;
    uint32_t wid = tid / 32;
    uint32_t lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    const uint64_t row_bytes = ((uint64_t)in_dim / 32u) * 34u;
    const uint8_t *wrow = w + (uint64_t)row * row_bytes;
    float acc = ds4_rocm_dot_q8_0_f32_warp(wrow, x, in_dim, wid, lane, n_warps);

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float v = lane < n_warps ? smem[lane] : 0.0f;
        v = ds4_rocm_warp_sum(v);
        if (lane == 0) out[row] = v;
    }
}

__global__ void ds4_rocm_mul_mm_q8_0_f32(const uint8_t *w,
                                         const float *x,
                                         float *out,
                                         uint32_t in_dim,
                                         uint32_t out_dim,
                                         uint32_t n_tokens) {
    uint32_t row = blockIdx.x;
    uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    uint32_t tid = threadIdx.x;
    uint32_t wid = tid / 32;
    uint32_t lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    const uint64_t row_bytes = ((uint64_t)in_dim / 32u) * 34u;
    const uint8_t *wrow = w + (uint64_t)row * row_bytes;
    const float *xt = x + (uint64_t)tok * in_dim;
    float acc = ds4_rocm_dot_q8_0_f32_warp(wrow, xt, in_dim, wid, lane, n_warps);

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float v = lane < n_warps ? smem[lane] : 0.0f;
        v = ds4_rocm_warp_sum(v);
        if (lane == 0) out[(uint64_t)tok * out_dim + row] = v;
    }
}

__global__ void ds4_rocm_mul_mm_q8_0_grouped_f32(const uint8_t *w,
                                                 const float *heads,
                                                 float *low,
                                                 uint32_t group_dim,
                                                 uint32_t rank,
                                                 uint32_t n_groups,
                                                 uint32_t n_tokens) {
    uint32_t row = blockIdx.x;
    uint32_t tok = blockIdx.y;
    const uint32_t low_dim = rank * n_groups;
    if (row >= low_dim || tok >= n_tokens) return;

    uint32_t group = row / rank;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid / 32;
    uint32_t lane = tid % 32;
    uint32_t n_warps = blockDim.x / 32;
    const uint64_t row_bytes = ((uint64_t)group_dim / 32u) * 34u;
    const uint8_t *wrow = w + (uint64_t)row * row_bytes;
    const float *x = heads + ((uint64_t)tok * n_groups + group) * group_dim;
    float acc = ds4_rocm_dot_q8_0_f32_warp(wrow, x, group_dim, wid, lane, n_warps);

    acc = ds4_rocm_warp_sum(acc);
    __shared__ float smem[32];
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float v = lane < n_warps ? smem[lane] : 0.0f;
        v = ds4_rocm_warp_sum(v);
        if (lane == 0) low[(uint64_t)tok * low_dim + row] = v;
    }
}

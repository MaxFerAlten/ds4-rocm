#pragma once

#include "../ds4_rocm_common.h"

// Structs, constant arrays, and f16 conversion are in dsv4_moe.hip (included first).
// This file adds __device__ vec_dot and __global__ kernels for GPU MoE.

// ── Device helpers ──────────────────────────────────────────────

static inline __device__ float ds4_rocm_dev_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t u; float f; } z = { sign };
            return z.f;
        }
        int32_t e = -14;
        while ((mant & 0x0400u) == 0) { mant <<= 1; e--; }
        mant &= 0x03ffu;
        union { uint32_t u; float f; } v = {
            sign | ((uint32_t)(e + 127) << 23) | (mant << 13),
        };
        return v.f;
    }
    if (exp == 31) {
        union { uint32_t u; float f; } v = { sign | 0x7f800000u | (mant << 13) };
        return v.f;
    }
    exp = exp + (127u - 15u);
    union { uint32_t u; float f; } v = { sign | (exp << 23) | (mant << 13) };
    return v.f;
}

static inline __device__ float ds4_rocm_dev_silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline __device__ int32_t ds4_rocm_dev_dot_q2_16(
    const uint8_t *q2, const int8_t *q8, int shift)
{
    int32_t s = 0;
    for (int i = 0; i < 16; i++)
        s += (int32_t)((q2[i] >> shift) & 3u) * (int32_t)q8[i];
    return s;
}

static inline __device__ int32_t ds4_rocm_dev_dot_iq2_pair_16(
    const uint8_t *grid0, uint8_t sign0,
    const uint8_t *grid1, uint8_t sign1,
    const int8_t *q8)
{
    int32_t s = 0;
    for (int i = 0; i < 8; i++) {
        int32_t v0 = (sign0 & ds4_rocm_kmask_iq2xs[i]) ? -(int32_t)grid0[i] : (int32_t)grid0[i];
        int32_t v1 = (sign1 & ds4_rocm_kmask_iq2xs[i]) ? -(int32_t)grid1[i] : (int32_t)grid1[i];
        s += v0 * (int32_t)q8[i];
        s += v1 * (int32_t)q8[8 + i];
    }
    return s;
}

// ── Device vec_dot: Q2_K weight × Q8_K input ────────────────────

static inline __device__ float ds4_rocm_dev_vec_dot_q2_K_q8_K(
    uint32_t n, const ds4_rocm_block_q2_K *x, const ds4_rocm_block_q8_K *y)
{
    uint32_t nb = n / DS4_ROCM_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t  *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);
        float dall = y[i].d * ds4_rocm_dev_f16_to_f32(x[i].d);
        float dmin = y[i].d * ds4_rocm_dev_f16_to_f32(x[i].dmin);
        int isum = 0, is = 0;
        for (int k = 0; k < DS4_ROCM_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum += d * ds4_rocm_dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum += d * ds4_rocm_dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

// ── Device vec_dot: Q4_K weight × Q8_K input ────────────────────

static inline __device__ uint8_t ds4_rocm_dev_q4_scale(const uint8_t *s, int j, int k) {
    return j < 4 ? (uint8_t)(s[j + k] & 63u)
                 : (uint8_t)((s[j + 4 + k] & 0x0fu) | ((s[j - 4 + k] & 0xc0u) >> 2));
}
static inline __device__ uint8_t ds4_rocm_dev_q4_min(const uint8_t *s, int j, int k) {
    return j < 4 ? (uint8_t)(s[j + 4 + k] & 63u)
                 : (uint8_t)((s[j + 4 + k] >> 4) | ((s[j + k] & 0xc0u) >> 2));
}

static inline __device__ float ds4_rocm_dev_vec_dot_q4_K_q8_K(
    uint32_t n, const ds4_rocm_block_q4_K *x, const ds4_rocm_block_q8_K *y)
{
    uint32_t nb = n / DS4_ROCM_QK_K;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        float d_all  = ds4_rocm_dev_f16_to_f32(x[b].d);
        float min_all = ds4_rocm_dev_f16_to_f32(x[b].dmin);
        for (uint32_t il = 0; il < 16; il++) {
            uint32_t is   = (il / 4u) * 2u;
            uint32_t qoff = (il / 4u) * 32u + 16u * (il & 1u);
            uint32_t sub  = il & 3u;
            uint8_t sc = ds4_rocm_dev_q4_scale(x[b].scales, (int)is, (int)(sub / 2u));
            uint8_t mn = ds4_rocm_dev_q4_min(x[b].scales, (int)is, (int)(sub / 2u));
            float d = sub < 2u ? d_all : d_all / 16.0f;
            uint8_t mask = sub < 2u ? 0x0fu : 0xf0u;
            for (uint32_t i = 0; i < 16; i++) {
                uint32_t e = il * 16u + i;
                float xv = d * (float)(x[b].qs[qoff + i] & mask) * (float)sc -
                    min_all * (float)mn;
                sum += xv * (y[b].d * (float)y[b].qs[e]);
            }
        }
    }
    return sum;
}

// ── Device vec_dot: IQ2_XXS weight × Q8_K input ─────────────────

static inline __device__ float ds4_rocm_dev_vec_dot_iq2_xxs_q8_K(
    uint32_t n, const ds4_rocm_block_iq2_xxs *x, const ds4_rocm_block_q8_K *y)
{
    uint32_t nb = n / DS4_ROCM_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        float d = ds4_rocm_dev_f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t   *q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < DS4_ROCM_QK_K / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint8_t *aux8 = (const uint8_t *)aux32;
            uint32_t ls = 2u * (aux32[1] >> 28) + 1u;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127u;
                uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127u;
                const uint8_t *g0 = (const uint8_t *)(ds4_rocm_iq2xxs_grid + aux8[l]);
                const uint8_t *g1 = (const uint8_t *)(ds4_rocm_iq2xxs_grid + aux8[l + 1]);
                sumi += ds4_rocm_dev_dot_iq2_pair_16(g0,
                    ds4_rocm_ksigns_iq2xs[sign_idx0],
                    g1,
                    ds4_rocm_ksigns_iq2xs[sign_idx1],
                    q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
}

// ── Device dispatch: vec_dot for any supported quant type ────────

static inline __device__ float ds4_rocm_dev_vec_dot_quant(
    uint32_t type, const uint8_t *row, const ds4_rocm_block_q8_K *q8, uint32_t n)
{
    if (type == DS4_ROCM_TENSOR_IQ2_XXS)
        return ds4_rocm_dev_vec_dot_iq2_xxs_q8_K(n, (const ds4_rocm_block_iq2_xxs *)row, q8);
    if (type == DS4_ROCM_TENSOR_Q2_K)
        return ds4_rocm_dev_vec_dot_q2_K_q8_K(n, (const ds4_rocm_block_q2_K *)row, q8);
    if (type == DS4_ROCM_TENSOR_Q4_K)
        return ds4_rocm_dev_vec_dot_q4_K_q8_K(n, (const ds4_rocm_block_q4_K *)row, q8);
    return 0.0f;
}

// ── Device: quantize one Q8_K block (cooperative, all threads) ──

static inline __device__ void ds4_rocm_dev_quantize_one_q8_K_block(
    const float *x, ds4_rocm_block_q8_K *y, int tid, int nThreads)
{
    __shared__ float s_maxval[256];
    float ax = fabsf(x[tid]);
    s_maxval[tid] = ax;
    __syncthreads();
    for (int s = nThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_maxval[tid + s] > s_maxval[tid])
                s_maxval[tid] = s_maxval[tid + s];
        }
        __syncthreads();
    }
    float amax = s_maxval[0];
    __syncthreads();

    if (amax == 0.0f) {
        if (tid == 0) {
            y->d = 0.0f;
            for (int j = 0; j < DS4_ROCM_QK_K; j++) y->qs[j] = 0;
            for (int j = 0; j < DS4_ROCM_QK_K / 16; j++) y->bsums[j] = 0;
        }
        return;
    }

    float iscale = -127.0f / amax;
    int v = (int)lrintf(iscale * x[tid]);
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    y->qs[tid] = (int8_t)v;
    __syncthreads();

    __shared__ int s_partial[16];
    if (tid < DS4_ROCM_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++)
            sum += y->qs[tid * 16 + i];
        s_partial[tid] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        for (int j = 0; j < DS4_ROCM_QK_K / 16; j++)
            y->bsums[j] = (int16_t)s_partial[j];
        y->d = 1.0f / iscale;
    }
}

// ── Global kernel: quantize float vector to Q8_K blocks ─────────
// grid = (nb,), block = 256
__global__ void ds4_rocm_quantize_row_q8_K_f32(
    const float *x, uint32_t n, ds4_rocm_block_q8_K *y)
{
    uint32_t nb = n / DS4_ROCM_QK_K;
    uint32_t b  = blockIdx.x;
    if (b >= nb) return;
    uint32_t tid = threadIdx.x;

    __shared__ float s_x[DS4_ROCM_QK_K];
    s_x[tid] = x[(uint64_t)b * DS4_ROCM_QK_K + tid];
    __syncthreads();

    ds4_rocm_dev_quantize_one_q8_K_block(s_x, &y[b], tid, blockDim.x);
}

// ── Cooperative vec_dot: IQ2_XXS weight × Q8_K input ──────────
// 32 threads cooperate on one 256-element block.
// 8 ib32 groups, each with 2 pairs → 2 active lanes per group.
// Lanes with sub_idx >= 2 contribute 0.
// Reduction across 32 lanes via __shfl_xor gives the full dot product.
static inline __device__ float ds4_rocm_dev_vec_dot_iq2_xxs_block_coop(
    const ds4_rocm_block_iq2_xxs *w_block,
    const ds4_rocm_block_q8_K    *q8_block,
    uint32_t                       lane)
{
    // Each ib32 group has only 2 valid pairs; lanes 2,3 return 0
    uint32_t group_id = lane / 4;
    uint32_t sub_idx  = lane % 4;
    if (sub_idx >= 2) return 0.0f;

    const uint16_t *q2 = w_block->qs;
    const int8_t   *q8 = q8_block->qs;
    uint32_t aux32[2];
    memcpy(aux32, q2 + (uint64_t)group_id * 4, 2 * sizeof(uint32_t));
    const uint8_t *aux8 = (const uint8_t *)aux32;

    uint32_t l = sub_idx * 2;
    uint32_t sign_idx0 = (aux32[1] >> (7 * l))       & 127u;
    uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127u;
    const uint8_t *g0 = (const uint8_t *)(ds4_rocm_iq2xxs_grid + aux8[l]);
    const uint8_t *g1 = (const uint8_t *)(ds4_rocm_iq2xxs_grid + aux8[l + 1]);

    int32_t sumi = 0;
    uint32_t q8_off = group_id * 32u + sub_idx * 16u;
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t mask_bit = 1u << i;
        int32_t v0 = (sign_idx0 & mask_bit) ? -(int32_t)g0[i] : (int32_t)g0[i];
        int32_t v1 = (sign_idx1 & mask_bit) ? -(int32_t)g1[i] : (int32_t)g1[i];
        sumi += v0 * (int32_t)q8[q8_off + i];
        sumi += v1 * (int32_t)q8[q8_off + 8 + i];
    }
    uint32_t ls = 2u * (aux32[1] >> 28) + 1u;
    float d = ds4_rocm_dev_f16_to_f32(w_block->d) * q8_block->d;
    return d * 0.125f * (float)(sumi * (int32_t)ls);
}

// ── Cooperative vec_dot: Q2_K weight × Q8_K input ──────────
// Q2_K block: 256 elements, 2-bit, 16 sub-blocks of 16 elements each.
// Each sub-block j has scale d_j = sc[j] & 0x0f and min m_j = sc[j] >> 4.
// Full formula: dall * sum_e(d_{e/16} * q2_val(e) * q8[e]) - dmin * sum_j(m_j * bsums[j]).
// Each lane handles 8 elements. Returns dall*partial - dmin*summs/32 so
// reduction across 32 lanes yields the correct
//   dall * isum - dmin * summs.
static inline __device__ float ds4_rocm_dev_vec_dot_q2_K_block_coop(
    const ds4_rocm_block_q2_K *w_block,
    const ds4_rocm_block_q8_K *q8_block,
    uint32_t                    lane)
{
    const uint8_t *q2 = w_block->qs;
    const int8_t  *q8 = q8_block->qs;
    const uint8_t *sc = w_block->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++)
        summs += q8_block->bsums[j] * (int)(sc[j] >> 4);
    float dall = q8_block->d * ds4_rocm_dev_f16_to_f32(w_block->d);
    float dmin = q8_block->d * ds4_rocm_dev_f16_to_f32(w_block->dmin);

    int32_t partial = 0;
    uint32_t e0 = lane * 8;
    for (uint32_t e = e0; e < e0 + 8 && e < DS4_ROCM_QK_K; e++) {
        uint32_t sidx = e / 16;              // sub-block index (0..15)
        uint32_t byte_idx = e / 4;
        uint32_t shift   = (e % 4) * 2;
        int q2_val = (q2[byte_idx] >> shift) & 3;
        partial += (int)(sc[sidx] & 0x0fu) * q2_val * (int32_t)q8[e];
    }
    return dall * (float)partial - dmin * (float)summs / 32.0f;
}

// ── Cooperative vec_dot: Q4_K weight × Q8_K input ──────────
static inline __device__ float ds4_rocm_dev_vec_dot_q4_K_block_coop(
    const ds4_rocm_block_q4_K *w_block,
    const ds4_rocm_block_q8_K *q8_block,
    uint32_t                    lane)
{
    float d_all  = ds4_rocm_dev_f16_to_f32(w_block->d);
    float min_all = ds4_rocm_dev_f16_to_f32(w_block->dmin);
    float sum = 0.0f;
    // Each lane handles 8 elements
    uint32_t e0 = lane * 8;
    for (uint32_t e = e0; e < e0 + 8 && e < DS4_ROCM_QK_K; e++) {
        uint32_t il   = e / 16;
        uint32_t ioff = e % 16;
        uint32_t is   = (il / 4u) * 2u;
        uint32_t qoff = (il / 4u) * 32u + 16u * (il & 1u);
        uint32_t sub  = il & 3u;
        uint8_t sc = ds4_rocm_dev_q4_scale(w_block->scales, (int)is, (int)(sub / 2u));
        uint8_t mn = ds4_rocm_dev_q4_min(w_block->scales, (int)is, (int)(sub / 2u));
        float d = sub < 2u ? d_all : d_all / 16.0f;
        uint8_t mask = sub < 2u ? 0x0fu : 0xf0u;
        float xv = d * (float)(w_block->qs[qoff + ioff] & mask) * (float)sc -
            min_all * (float)mn;
        sum += xv * (q8_block->d * (float)q8_block->qs[e]);
    }
    return sum;
}

// ── Cooperative vec_dot dispatch ──────────────────────────
static inline __device__ float ds4_rocm_dev_vec_dot_block_coop(
    uint32_t type, const uint8_t *row, uint32_t block_idx,
    const ds4_rocm_block_q8_K *q8, uint32_t lane)
{
    if (type == DS4_ROCM_TENSOR_IQ2_XXS)
        return ds4_rocm_dev_vec_dot_iq2_xxs_block_coop(
            (const ds4_rocm_block_iq2_xxs *)row + block_idx,
            q8 + block_idx, lane);
    if (type == DS4_ROCM_TENSOR_Q2_K)
        return ds4_rocm_dev_vec_dot_q2_K_block_coop(
            (const ds4_rocm_block_q2_K *)row + block_idx,
            q8 + block_idx, lane);
    if (type == DS4_ROCM_TENSOR_Q4_K)
        return ds4_rocm_dev_vec_dot_q4_K_block_coop(
            (const ds4_rocm_block_q4_K *)row + block_idx,
            q8 + block_idx, lane);
    return 0.0f;
}

// ── Fused gate/up kernel (multi-block tiled) ───────────────
// grid = (n_expert, n_tiles_gate), block = 256
// Each (expert, tile) block handles a tile of the output rows.
// When preq_x is non-null, it points to a pre-quantized Q8_K version of x
// in device memory — this avoids redundant per-tile quantize work and
// eliminates shared-memory usage, allowing many more tiles for better CU
// occupancy without penalty.
__global__ void ds4_rocm_moe_gate_up_f32(
    float                    *scratch_gate,
    float                    *scratch_up,
    float                    *scratch_mid,
    const uint8_t            *const *gate_weight_ptrs,
    const uint8_t            *const *up_weight_ptrs,
    uint64_t                  gate_row_bytes,
    const float               *x,
    uint32_t                  expert_in_dim,
    uint32_t                  expert_mid_dim,
    const float              *weights,
    uint32_t                  n_expert,
    uint32_t                  gate_type,
    float                     clamp_val,
    const ds4_rocm_block_q8_K *preq_x)
{
    uint32_t slot = blockIdx.x;
    if (slot >= n_expert) return;
    uint32_t tid = threadIdx.x;
    uint32_t n_threads = blockDim.x;
    uint32_t tile = blockIdx.y;
    uint32_t n_tiles = gridDim.y;

    const uint8_t *gate_base = gate_weight_ptrs[slot];
    const uint8_t *up_base   = up_weight_ptrs[slot];
    float weight = weights[slot];
    uint32_t n_blocks_in = expert_in_dim / DS4_ROCM_QK_K;

    // When pre-quantized x is provided, read from device memory (L2-cached)
    // instead of quantizing inline to shared memory.
    const ds4_rocm_block_q8_K *q8_x;
    if (preq_x) {
        q8_x = preq_x;
    } else {
        // Fallback: inline quantize via shared memory (backward compat)
        extern __shared__ char s_raw[];
        ds4_rocm_block_q8_K *s_q8_x = (ds4_rocm_block_q8_K *)s_raw;
        float *s_x = (float *)(s_raw + (size_t)n_blocks_in * sizeof(ds4_rocm_block_q8_K));
        for (uint32_t b = 0; b < n_blocks_in; b++) {
            s_x[tid] = x[(uint64_t)b * DS4_ROCM_QK_K + tid];
            __syncthreads();
            ds4_rocm_dev_quantize_one_q8_K_block(s_x, &s_q8_x[b], tid, n_threads);
            __syncthreads();
        }
        q8_x = s_q8_x;
    }

    // Process rows for this tile: each thread handles a different row
    uint32_t rows_per_tile = (expert_mid_dim + n_tiles - 1) / n_tiles;
    uint32_t row_base = tile * rows_per_tile;
    uint32_t row_end = min(row_base + rows_per_tile, expert_mid_dim);
    for (uint32_t row = row_base + tid; row < row_end; row += n_threads) {
        const uint8_t *gate_row = gate_base + (uint64_t)row * gate_row_bytes;
        const uint8_t *up_row   = up_base   + (uint64_t)row * gate_row_bytes;
        float g = ds4_rocm_dev_vec_dot_quant(gate_type, gate_row, q8_x, expert_in_dim);
        float u = ds4_rocm_dev_vec_dot_quant(gate_type, up_row,   q8_x, expert_in_dim);
        if (clamp_val > 1.0e-6f) {
            if (g > clamp_val) g = clamp_val;
            if (u > clamp_val) u = clamp_val;
            if (u < -clamp_val) u = -clamp_val;
        }
        float m = ds4_rocm_dev_silu(g) * u * weight;
        uint64_t idx = (uint64_t)slot * expert_mid_dim + row;
        scratch_gate[idx] = g;
        scratch_up[idx]   = u;
        scratch_mid[idx]  = m;
    }
}

// ── Down matmul + accumulate (multi-block tiled) ───────────
// grid = (n_expert, n_tiles_down), block = 256
// When preq_mid is non-null, it points to a pre-quantized Q8_K version of the
// per-expert mid slice in device memory — skips redundant per-tile quantize.
// preq_mid layout: slot * n_mid_blocks blocks, contiguous per expert.
__global__ void ds4_rocm_moe_down_f32(
    float                    *out,
    const float              *scratch_mid,
    const uint8_t            *const *down_weight_ptrs,
    uint64_t                  down_row_bytes,
    uint32_t                  expert_mid_dim,
    uint32_t                  out_dim,
    uint32_t                  n_expert,
    uint32_t                  down_type,
    const ds4_rocm_block_q8_K *preq_mid)
{
    uint32_t slot = blockIdx.x;
    if (slot >= n_expert) return;
    uint32_t tid = threadIdx.x;
    uint32_t n_threads = blockDim.x;
    uint32_t tile = blockIdx.y;
    uint32_t n_tiles = gridDim.y;

    const uint8_t *down_base = down_weight_ptrs[slot];
    uint32_t n_mid_blocks = expert_mid_dim / DS4_ROCM_QK_K;
    uint64_t mid_base = (uint64_t)slot * expert_mid_dim;

    // When pre-quantized mid is provided, read from device memory (L2-cached).
    const ds4_rocm_block_q8_K *q8_mid;
    if (preq_mid) {
        q8_mid = preq_mid + (uint64_t)slot * n_mid_blocks;
    } else {
        // Fallback: inline quantize via shared memory
        extern __shared__ char smem[];
        ds4_rocm_block_q8_K *s_q8_mid = (ds4_rocm_block_q8_K *)smem;
        float *s_mid = (float *)(smem + (size_t)n_mid_blocks * sizeof(ds4_rocm_block_q8_K));
        for (uint32_t b = 0; b < n_mid_blocks; b++) {
            s_mid[tid] = scratch_mid[mid_base + (uint64_t)b * DS4_ROCM_QK_K + tid];
            __syncthreads();
            ds4_rocm_dev_quantize_one_q8_K_block(s_mid, &s_q8_mid[b], tid, n_threads);
            __syncthreads();
        }
        q8_mid = s_q8_mid;
    }

    // Process output rows for this tile: per-thread vec_dot
    uint32_t rows_per_tile = (out_dim + n_tiles - 1) / n_tiles;
    uint32_t row_base = tile * rows_per_tile;
    uint32_t row_end = min(row_base + rows_per_tile, out_dim);
    for (uint32_t row = row_base + tid; row < row_end; row += n_threads) {
        const uint8_t *down_row = down_base + (uint64_t)row * down_row_bytes;
        float v = ds4_rocm_dev_vec_dot_quant(down_type, down_row, q8_mid, expert_mid_dim);
        atomicAdd(&out[row], v);
    }
}

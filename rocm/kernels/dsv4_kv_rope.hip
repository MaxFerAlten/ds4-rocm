#pragma once

#include "../ds4_rocm_common.h"

__device__ inline uint16_t ds4_rocm_f32_to_f16_bits(float f) {
    const uint32_t bits = __float_as_uint(f);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

__device__ inline float ds4_rocm_f16_round_f32(float f) {
    return ds4_rocm_f16_to_f32(ds4_rocm_f32_to_f16_bits(f));
}

__device__ inline float ds4_rocm_e4m3fn_value(int i) {
    const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0 ? (float)mant * 0.001953125f :
        (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

__device__ inline float ds4_rocm_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ds4_rocm_e4m3fn_value(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - ds4_rocm_e4m3fn_value(best));
        const float next_diff = fabsf(ax - ds4_rocm_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }
    return sign * ds4_rocm_e4m3fn_value(best);
}

__device__ inline float ds4_rocm_rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ inline float ds4_rocm_rope_yarn_corr_dim(int n_dims,
                                                   uint32_t n_ctx_orig,
                                                   float n_rot,
                                                   float base) {
    const float pi = 3.14159265358979323846f;
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * pi)) /
           (2.0f * logf(base));
}

__device__ inline void ds4_rocm_rope_yarn_corr_dims(int n_dims,
                                                   uint32_t n_ctx_orig,
                                                   float freq_base,
                                                   float beta_fast,
                                                   float beta_slow,
                                                   float dims[2]) {
    const float start = floorf(ds4_rocm_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(ds4_rocm_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

__global__ void ds4_rocm_rope_tail_f32(float *x,
                                       uint32_t n_tok,
                                       uint32_t n_head,
                                       uint32_t head_dim,
                                       uint32_t n_rot,
                                       uint32_t pos0,
                                       uint32_t n_ctx_orig,
                                       uint32_t inverse,
                                       float freq_base,
                                       float freq_scale,
                                       float ext_factor,
                                       float attn_factor,
                                       float beta_fast,
                                       float beta_slow) {
    const uint64_t pairs_per_head = n_rot / 2u;
    const uint64_t total = (uint64_t)n_tok * n_head * pairs_per_head;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || n_rot > head_dim) return;

    const uint32_t pair = (uint32_t)(gid % pairs_per_head);
    const uint64_t head_linear = gid / pairs_per_head;
    const uint32_t head = (uint32_t)(head_linear % n_head);
    const uint32_t tok = (uint32_t)(head_linear / n_head);
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t i = pair * 2u;
    float theta_extrap = (float)(pos0 + tok);
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    for (uint32_t p = 0; p < pair; p++) theta_extrap *= theta_scale;

    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        ds4_rocm_rope_yarn_corr_dims((int)n_rot, n_ctx_orig, freq_base,
                                     beta_fast, beta_slow, corr_dims);
    }

    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix = ds4_rocm_rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    const float s_sign = inverse ? -1.0f : 1.0f;
    const float c = cosf(theta) * mscale;
    const float s = s_sign * sinf(theta) * mscale;
    float *tail = x + ((uint64_t)tok * n_head + head) * head_dim + n_nope;
    const float x0 = tail[i + 0];
    const float x1 = tail[i + 1];
    tail[i + 0] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__global__ void ds4_rocm_fp8_kv_quantize_f32(float *x,
                                             uint32_t n_tok,
                                             uint32_t head_dim,
                                             uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= n_tok || tid >= 64u || n_rot > head_dim) return;

    const uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];

    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float v = 0.0f;
        if (off + tid < n_nope) {
            v = xr[off + tid];
            scratch[tid] = fabsf(v);
        } else {
            scratch[tid] = 0.0f;
        }
        __syncthreads();

        for (uint32_t stride = 32u; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (off + tid < n_nope) {
            const float clamped = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[off + tid] = ds4_rocm_e4m3fn_dequant(clamped) * scale;
        }
        __syncthreads();
    }
}

__global__ void ds4_rocm_store_raw_kv_f32(float *raw_cache,
                                          const float *kv,
                                          uint32_t raw_cap,
                                          uint32_t pos0,
                                          uint32_t n_tokens,
                                          uint32_t head_dim) {
    const uint64_t total = (uint64_t)n_tokens * head_dim;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || raw_cap == 0) return;

    const uint32_t tok = (uint32_t)(gid / head_dim);
    const uint32_t i = (uint32_t)(gid % head_dim);
    const uint32_t row = (pos0 + tok) % raw_cap;
    raw_cache[(uint64_t)row * head_dim + i] =
        ds4_rocm_f16_round_f32(kv[(uint64_t)tok * head_dim + i]);
}

__global__ void ds4_rocm_kv_fp8_store_raw_f32(float *kv,
                                              float *raw_cache,
                                              uint32_t raw_row,
                                              uint32_t head_dim,
                                              uint32_t n_rot) {
    const uint32_t tid = threadIdx.x;
    if (tid >= 64u || n_rot > head_dim) return;

    const uint32_t n_nope = head_dim - n_rot;
    float *raw = raw_cache + (uint64_t)raw_row * head_dim;
    __shared__ float scratch[64];

    for (uint32_t off = 0; off < n_nope; off += 64u) {
        float v = 0.0f;
        if (off + tid < n_nope) {
            v = kv[off + tid];
            scratch[tid] = fabsf(v);
        } else {
            scratch[tid] = 0.0f;
        }
        __syncthreads();

        for (uint32_t stride = 32u; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (off + tid < n_nope) {
            const float clamped = fminf(448.0f, fmaxf(-448.0f, v / scale));
            const float q = ds4_rocm_e4m3fn_dequant(clamped) * scale;
            kv[off + tid] = q;
            raw[off + tid] = ds4_rocm_f16_round_f32(q);
        }
        __syncthreads();
    }

    for (uint32_t i = n_nope + tid; i < head_dim; i += 64u) {
        raw[i] = ds4_rocm_f16_round_f32(kv[i]);
    }
}

__global__ void ds4_rocm_compressor_store_one_f32(const float *kv,
                                                  const float *score,
                                                  const uint8_t *ape,
                                                  float *state_kv,
                                                  float *state_score,
                                                  uint32_t width,
                                                  uint32_t ratio,
                                                  uint32_t pos,
                                                  uint32_t ape_type) {
    const uint32_t gid = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= width || width == 0 || ratio == 0) return;

    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    const uint64_t dst = (uint64_t)dst_row * width + gid;
    const uint64_t ape_i = (uint64_t)pos_mod * width + gid;
    const float ape_v = ape_type == 1u
        ? ds4_rocm_f16_to_f32(((const uint16_t *)ape)[ape_i])
        : ((const float *)ape)[ape_i];

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}

__global__ void ds4_rocm_compressor_store_batch_f32(const float *kv,
                                                    const float *score,
                                                    const uint8_t *ape,
                                                    float *state_kv,
                                                    float *state_score,
                                                    uint32_t width,
                                                    uint32_t ratio,
                                                    uint32_t pos0,
                                                    uint32_t n_tokens,
                                                    uint32_t ape_type) {
    const uint64_t total = (uint64_t)n_tokens * width;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || width == 0 || ratio == 0) return;

    const uint32_t token = (uint32_t)(gid / width);
    const uint32_t j = (uint32_t)(gid % width);
    const uint32_t pos_mod = (pos0 + token) % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    const uint64_t dst = (uint64_t)dst_row * width + j;
    const uint64_t ape_i = (uint64_t)pos_mod * width + j;
    const float ape_v = ape_type == 1u
        ? ds4_rocm_f16_to_f32(((const uint16_t *)ape)[ape_i])
        : ((const float *)ape)[ape_i];

    state_kv[dst] = kv[gid];
    state_score[dst] = score[gid] + ape_v;
}

__global__ void ds4_rocm_compressor_store_ratio4_state_f32(const float *kv,
                                                           const float *score,
                                                           const uint8_t *ape,
                                                           float *state_kv,
                                                           float *state_score,
                                                           uint32_t width,
                                                           uint32_t pos0,
                                                           uint32_t ape_type) {
    const uint32_t ratio = 4u;
    const uint64_t total = (uint64_t)ratio * width;
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total || width == 0) return;

    const uint32_t token = (uint32_t)(gid / width);
    const uint32_t j = (uint32_t)(gid % width);
    const uint32_t pos_mod = (pos0 + token) % ratio;
    const uint64_t ape_i = (uint64_t)pos_mod * width + j;
    const float ape_v = ape_type == 1u
        ? ds4_rocm_f16_to_f32(((const uint16_t *)ape)[ape_i])
        : ((const float *)ape)[ape_i];

    state_kv[(uint64_t)token * width + j] = kv[gid];
    state_score[(uint64_t)token * width + j] = score[gid] + ape_v;
}

__global__ void ds4_rocm_compressor_pool_f32(float *out,
                                             const float *state_kv,
                                             const float *state_score,
                                             uint32_t head_dim,
                                             uint32_t ratio) {
    const uint32_t j = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= head_dim || head_dim == 0 || ratio == 0) return;

    const float neg_inf = -1.0e30f;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    float max_score = neg_inf;

    if (ratio == 4u) {
        for (uint32_t r = 0; r < ratio; r++) {
            const float sp = state_score[(uint64_t)r * width + j];
            const float sc = state_score[(uint64_t)(ratio + r) * width + head_dim + j];
            max_score = fmaxf(max_score, sp);
            max_score = fmaxf(max_score, sc);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            max_score = fmaxf(max_score, state_score[(uint64_t)r * width + j]);
        }
    }

    if (max_score <= neg_inf * 0.5f) {
        out[j] = 0.0f;
        return;
    }

    float denom = 0.0f;
    float sum = 0.0f;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < ratio; r++) {
            const float wp = expf(state_score[(uint64_t)r * width + j] - max_score);
            const float wc = expf(state_score[(uint64_t)(ratio + r) * width + head_dim + j] - max_score);
            denom += wp + wc;
            sum += wp * state_kv[(uint64_t)r * width + j];
            sum += wc * state_kv[(uint64_t)(ratio + r) * width + head_dim + j];
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            const float w = expf(state_score[(uint64_t)r * width + j] - max_score);
            denom += w;
            sum += w * state_kv[(uint64_t)r * width + j];
        }
    }
    out[j] = denom > 0.0f ? sum / denom : 0.0f;
}

__global__ void ds4_rocm_compressor_rms_norm_f32(float *x,
                                                 const float *weight,
                                                 uint32_t head_dim,
                                                 float eps) {
    extern __shared__ float scratch[];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        const float v = x[i];
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        x[i] = x[i] * inv * weight[i];
    }
}

__global__ void ds4_rocm_compressor_ratio4_shift_f32(float *state_kv,
                                                     float *state_score,
                                                     uint32_t width) {
    const uint32_t gid = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t n = 4u * width;
    if (gid >= n) return;

    state_kv[gid] = state_kv[(uint64_t)n + gid];
    state_score[gid] = state_score[(uint64_t)n + gid];
}

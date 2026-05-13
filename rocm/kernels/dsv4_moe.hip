#pragma once

#include "../ds4_rocm_common.h"

#include <math.h>
#include <string.h>

#define DS4_ROCM_QK_K 256
#define DS4_ROCM_TENSOR_Q2_K 10u
#define DS4_ROCM_TENSOR_Q4_K 12u
#define DS4_ROCM_TENSOR_IQ2_XXS 16u

typedef struct {
    uint8_t  scales[DS4_ROCM_QK_K / 16];
    uint8_t  qs[DS4_ROCM_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} ds4_rocm_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[DS4_ROCM_QK_K / 2];
} ds4_rocm_block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[DS4_ROCM_QK_K];
    int16_t bsums[DS4_ROCM_QK_K / 16];
} ds4_rocm_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[DS4_ROCM_QK_K / 8];
} ds4_rocm_block_iq2_xxs;

static const uint8_t ds4_rocm_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ds4_rocm_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t ds4_rocm_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static inline float ds4_rocm_host_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t u; float f; } z = { sign };
            return z.f;
        }
        int32_t e = -14;
        while ((mant & 0x0400u) == 0) {
            mant <<= 1;
            e--;
        }
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

static inline int32_t ds4_rocm_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t s = 0;
    for (int i = 0; i < 16; i++) {
        s += (int32_t)((q2[i] >> shift) & 3u) * (int32_t)q8[i];
    }
    return s;
}

static inline int32_t ds4_rocm_dot_iq2_pair_16(const uint8_t *grid0,
                                               uint8_t sign0,
                                               const uint8_t *grid1,
                                               uint8_t sign1,
                                               const int8_t *q8) {
    int32_t s = 0;
    for (int i = 0; i < 8; i++) {
        const int32_t v0 = (sign0 & ds4_rocm_kmask_iq2xs[i]) ? -(int32_t)grid0[i] : (int32_t)grid0[i];
        const int32_t v1 = (sign1 & ds4_rocm_kmask_iq2xs[i]) ? -(int32_t)grid1[i] : (int32_t)grid1[i];
        s += v0 * (int32_t)q8[i];
        s += v1 * (int32_t)q8[8 + i];
    }
    return s;
}

static inline void ds4_rocm_quantize_row_q8_K_host(const float *x,
                                                   ds4_rocm_block_q8_K *y,
                                                   uint32_t k) {
    const uint32_t nb = k / DS4_ROCM_QK_K;
    for (uint32_t b = 0; b < nb; b++) {
        float max_v = 0.0f;
        float amax = 0.0f;
        for (uint32_t j = 0; j < DS4_ROCM_QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max_v = x[j];
            }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += DS4_ROCM_QK_K;
            continue;
        }
        const float iscale = -127.0f / max_v;
        for (uint32_t j = 0; j < DS4_ROCM_QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (uint32_t j = 0; j < DS4_ROCM_QK_K / 16; j++) {
            int sum = 0;
            for (uint32_t i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += DS4_ROCM_QK_K;
    }
}

static inline float ds4_rocm_vec_dot_q2_K_q8_K_host(uint32_t n,
                                                    const ds4_rocm_block_q2_K *x,
                                                    const ds4_rocm_block_q8_K *y) {
    const uint32_t nb = n / DS4_ROCM_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * ds4_rocm_host_f16_to_f32(x[i].d);
        const float dmin = y[i].d * ds4_rocm_host_f16_to_f32(x[i].dmin);
        int isum = 0;
        int is = 0;
        for (int k = 0; k < DS4_ROCM_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum += d * ds4_rocm_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum += d * ds4_rocm_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

static inline uint8_t ds4_rocm_q4_scale(const uint8_t *s, int j, int k) {
    return j < 4 ? (uint8_t)(s[j + k] & 63u)
                 : (uint8_t)((s[j + 4 + k] & 0x0fu) | ((s[j - 4 + k] & 0xc0u) >> 2));
}

static inline uint8_t ds4_rocm_q4_min(const uint8_t *s, int j, int k) {
    return j < 4 ? (uint8_t)(s[j + 4 + k] & 63u)
                 : (uint8_t)((s[j + 4 + k] >> 4) | ((s[j + k] & 0xc0u) >> 2));
}

static inline float ds4_rocm_vec_dot_q4_K_q8_K_host(uint32_t n,
                                                    const ds4_rocm_block_q4_K *x,
                                                    const ds4_rocm_block_q8_K *y) {
    const uint32_t nb = n / DS4_ROCM_QK_K;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const float d_all = ds4_rocm_host_f16_to_f32(x[b].d);
        const float min_all = ds4_rocm_host_f16_to_f32(x[b].dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t is = (il / 4u) * 2u;
            const uint32_t qoff = (il / 4u) * 32u + 16u * (il & 1u);
            const uint32_t sub = il & 3u;
            const uint8_t sc = ds4_rocm_q4_scale(x[b].scales, (int)is, (int)(sub / 2u));
            const uint8_t mn = ds4_rocm_q4_min(x[b].scales, (int)is, (int)(sub / 2u));
            const float d = sub < 2u ? d_all : d_all / 16.0f;
            const uint8_t mask = sub < 2u ? 0x0fu : 0xf0u;
            for (uint32_t i = 0; i < 16; i++) {
                const uint32_t e = il * 16u + i;
                const float xv = d * (float)(x[b].qs[qoff + i] & mask) * (float)sc -
                    min_all * (float)mn;
                sum += xv * (y[b].d * (float)y[b].qs[e]);
            }
        }
    }
    return sum;
}

static inline float ds4_rocm_vec_dot_iq2_xxs_q8_K_host(uint32_t n,
                                                       const ds4_rocm_block_iq2_xxs *x,
                                                       const ds4_rocm_block_q8_K *y) {
    const uint32_t nb = n / DS4_ROCM_QK_K;
    float sumf = 0.0f;
    for (uint32_t i = 0; i < nb; i++) {
        const float d = ds4_rocm_host_f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < DS4_ROCM_QK_K / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const uint32_t ls = 2u * (aux32[1] >> 28) + 1u;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127u;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127u;
                const uint8_t *g0 = (const uint8_t *)(ds4_rocm_iq2xxs_grid + aux8[l]);
                const uint8_t *g1 = (const uint8_t *)(ds4_rocm_iq2xxs_grid + aux8[l + 1]);
                sumi += ds4_rocm_dot_iq2_pair_16(g0,
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

static inline float ds4_rocm_vec_dot_quant_q8_host(uint32_t type,
                                                   const uint8_t *row,
                                                   const ds4_rocm_block_q8_K *q8,
                                                   uint32_t n) {
    if (type == DS4_ROCM_TENSOR_IQ2_XXS) {
        return ds4_rocm_vec_dot_iq2_xxs_q8_K_host(n, (const ds4_rocm_block_iq2_xxs *)row, q8);
    }
    if (type == DS4_ROCM_TENSOR_Q2_K) {
        return ds4_rocm_vec_dot_q2_K_q8_K_host(n, (const ds4_rocm_block_q2_K *)row, q8);
    }
    if (type == DS4_ROCM_TENSOR_Q4_K) {
        return ds4_rocm_vec_dot_q4_K_q8_K_host(n, (const ds4_rocm_block_q4_K *)row, q8);
    }
    return 0.0f;
}

static inline uint64_t ds4_rocm_routed_block_bytes(uint32_t type) {
    if (type == DS4_ROCM_TENSOR_IQ2_XXS) return sizeof(ds4_rocm_block_iq2_xxs);
    if (type == DS4_ROCM_TENSOR_Q2_K) return sizeof(ds4_rocm_block_q2_K);
    if (type == DS4_ROCM_TENSOR_Q4_K) return sizeof(ds4_rocm_block_q4_K);
    return 0;
}

static inline float ds4_rocm_silu_host(float x) {
    return x / (1.0f + expf(-x));
}

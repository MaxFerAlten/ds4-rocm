#ifndef DS4_ROCM_COMMON_H
#define DS4_ROCM_COMMON_H

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_ROCM_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

#define DS4_ROCM_CHECK(expr)                                                   \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "ds4: ROCm error %s:%d: %s\n",                    \
                    __FILE__, __LINE__, hipGetErrorString(_e));               \
            return 0;                                                          \
        }                                                                      \
    } while (0)

static inline int ds4_rocm_not_implemented(const char *func) {
    fprintf(stderr, "ds4: ROCm primitive not implemented: %s\n", func);
    return 0;
}

__device__ inline float ds4_rocm_warp_sum(float v) {
    // Use sync variant with width=32 for portability
    // (AMD wavefront=64, CUDA warp=32; width=32 gives 32-thread logical warps)
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffffffffffull, v, off, 32);
    return v;
}

__device__ inline float ds4_rocm_warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffffffffffffull, v, off, 32);
        v = fmaxf(v, other);
    }
    return v;
}

__device__ inline float ds4_rocm_f16_to_f32(uint16_t h) {
    _Float16 v;
    __builtin_memcpy(&v, &h, sizeof(h));
    return (float)v;
}

#endif

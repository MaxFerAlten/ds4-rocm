extern "C" {
#include "../ds4.h"
#include "../ds4_gpu.h"
}

#include "ds4_rocm_common.h"
#include "kernels/dense.hip"
#include "kernels/elementwise.hip"
#include "kernels/dsv4_hc.hip"
#include "kernels/dsv4_kv_rope.hip"
#include "kernels/dsv4_indexer.hip"
#include "kernels/dsv4_attention.hip"
#include "kernels/dsv4_moe.hip"
#include "kernels/moe.hip"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <map>

struct ds4_gpu_tensor {
    uint8_t *ptr;
    uint64_t bytes;
    bool owner;
    ds4_gpu_tensor *base;
    uint64_t offset;
};

static hipStream_t g_stream;
static int g_initialized;
static const uint8_t *g_model_map;
static uint64_t g_model_size;
static uint64_t g_model_offset;
static uint64_t g_model_bytes;
static bool g_quality;
static int g_model_fd = -1;

// Pinned staging buffer for weight uploads.  Allocated at init via hipHostMalloc,
// used by rocm_weight_hip_upload as a bounce buffer: CPU memcpy from mmap (pageable)
// → staging (pinned) → hipMemcpyAsync to device (DMA at full PCIe bandwidth).
// This avoids the driver's slow staged copy from pageable memory (~400 MB/s)
// without needing to hipHostRegister the entire multi-GB model map.
// g_staging_pos is reset to 0 before each MoE batch (after stream sync ensures
// all prior DMAs from the staging buffer have completed).
static uint8_t *g_weight_staging;
static uint64_t g_weight_staging_size;
static uint64_t g_staging_pos;

// Device-side buffers for pre-quantized MoE inputs.  Pre-quantizing x and mid
// once per layer eliminates the redundant per-tile quantize work when using
// multi-block tiling, allowing many more tiles without penalty.
static ds4_rocm_block_q8_K *g_preq_x;
static uint64_t g_preq_x_cap;       // allocated capacity in blocks
static ds4_rocm_block_q8_K *g_preq_mid;
static uint64_t g_preq_mid_cap;     // allocated capacity in blocks (per-expert * n_expert max)

// ── Device-side expert weight pointer table ──────────────────────
//
// Flat 3D array on device: [layer][expert_id][matrix] → uint8_t *GPU addr.
// matrix: 0=gate, 1=up, 2=down.
// CPU-side shadow has the same layout so we can read pointers without sync.
// Pre-allocated GPU pointer arrays avoid hipMalloc/hipFree on the hot path.
#define ROCM_EXPERT_GATE 0
#define ROCM_EXPERT_UP   1
#define ROCM_EXPERT_DOWN 2
#define ROCM_MAX_EXPERTS 256
#define ROCM_MAX_MOE_LAYERS 64
#define ROCM_EXPERT_TABLE_STEP (ROCM_MAX_EXPERTS * 3)

static uint8_t *g_expert_table_cpu[ROCM_MAX_MOE_LAYERS * ROCM_EXPERT_TABLE_STEP];
static uint32_t g_expert_table_cache_idx[ROCM_MAX_MOE_LAYERS * ROCM_EXPERT_TABLE_STEP];
static uint64_t g_expert_table_cache_gen[ROCM_MAX_MOE_LAYERS * ROCM_EXPERT_TABLE_STEP];
static uint8_t *g_expert_table_dev;   // device allocation, same layout

// Pre-allocated device arrays for 6 expert pointers + weights
static uint8_t **g_gate_ptrs_dev;
static uint8_t **g_up_ptrs_dev;
static uint8_t **g_down_ptrs_dev;
static float    *g_weights_dev;
static bool      g_hot_path_arrays_allocated;
static bool      g_moe_prewarm_legacy_warned;

static inline uint64_t rocm_expert_table_idx(int layer, int expert, int matrix) {
    return (uint64_t)layer * ROCM_EXPERT_TABLE_STEP + (uint64_t)expert * 3 + matrix;
}

typedef struct {
    const void *ops;
    const uint8_t *model_map;
    uint64_t model_size;
    uint64_t offset;
    uint64_t bytes;
    uint8_t *device;
    uint64_t last_use;
    uint64_t generation;
    void (*free_fn)(uint8_t *device, void *ctx);
    void *free_ctx;
} rocm_weight_entry;

typedef struct {
    uint8_t *(*alloc)(uint64_t bytes, void *ctx);
    void (*free)(uint8_t *device, void *ctx);
    int (*upload)(uint8_t *device, const uint8_t *host, uint64_t bytes, void *ctx);
    void *ctx;
    bool needs_init;
} rocm_weight_ops;

#define DS4_ROCM_WEIGHT_CACHE_DEFAULT_ENTRIES 16384u
#define DS4_ROCM_WEIGHT_CACHE_MAX_ENTRIES     49152u

static rocm_weight_entry g_weight_cache[DS4_ROCM_WEIGHT_CACHE_MAX_ENTRIES];
static uint32_t g_weight_cache_len;
static uint32_t g_weight_cache_capacity = DS4_ROCM_WEIGHT_CACHE_DEFAULT_ENTRIES;
static uint64_t g_weight_use_clock;
static uint64_t g_weight_cache_bytes;
static uint64_t g_weight_cache_hits;
static uint64_t g_weight_cache_misses;
static uint64_t g_weight_cache_generation;
static uint64_t g_weight_cache_min_free_bytes;

// Open-addressing hash table overlay on g_weight_cache for O(1) lookup.
// Each slot holds an index into g_weight_cache, or -1 for empty.
// Size is the next power of two above the max entry count to keep load < 50%.
#define DS4_ROCM_CACHE_HASH_SIZE 131072u
static int32_t g_cache_hash[DS4_ROCM_CACHE_HASH_SIZE];

static uint32_t rocm_cache_hash_slot(const rocm_weight_ops *ops,
                                      const uint8_t *map,
                                      uint64_t offset, uint64_t bytes) {
    uint64_t h = (uint64_t)(uintptr_t)ops
               ^ (uint64_t)(uintptr_t)map
               ^ offset
               ^ (bytes << 16);
    // 64-bit mix
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return (uint32_t)(h & (DS4_ROCM_CACHE_HASH_SIZE - 1u));
}

static void rocm_cache_hash_clear(void) {
    memset(g_cache_hash, -1, sizeof(g_cache_hash));
}

static void rocm_cache_hash_insert(uint32_t idx) {
    const rocm_weight_entry *e = &g_weight_cache[idx];
    uint32_t slot = rocm_cache_hash_slot((const rocm_weight_ops *)e->ops,
                                          e->model_map, e->offset, e->bytes);
    while (g_cache_hash[slot] != -1) {
        slot = (slot + 1u) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
    }
    g_cache_hash[slot] = (int32_t)idx;
}

static void rocm_cache_hash_remove(uint32_t idx) {
    const rocm_weight_entry *e = &g_weight_cache[idx];
    uint32_t slot = rocm_cache_hash_slot((const rocm_weight_ops *)e->ops,
                                          e->model_map, e->offset, e->bytes);
    for (;;) {
        if (g_cache_hash[slot] == (int32_t)idx) {
            g_cache_hash[slot] = -1;
            // Back-shift subsequent entries in the same probe cluster
            uint32_t next = (slot + 1u) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
            while (g_cache_hash[next] != -1) {
                int32_t moved = g_cache_hash[next];
                uint32_t home = rocm_cache_hash_slot(
                    (const rocm_weight_ops *)g_weight_cache[moved].ops,
                    g_weight_cache[moved].model_map,
                    g_weight_cache[moved].offset,
                    g_weight_cache[moved].bytes);
                // If this entry belongs at or before 'slot', move it back
                uint32_t dist = (next - home) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
                uint32_t gap  = (next - slot) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
                if (dist < gap) break;  // can't move, belongs after
                g_cache_hash[slot] = moved;
                g_cache_hash[next] = -1;
                slot = next;
                next = (next + 1u) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
            }
            return;
        }
        if (g_cache_hash[slot] == -1) return;  // not in table (shouldn't happen)
        slot = (slot + 1u) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
    }
}

static uint64_t rocm_parse_u64_env(const char *name, uint64_t fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = nullptr;
    unsigned long long value = strtoull(s, &end, 10);
    if (end == s) return fallback;
    return (uint64_t)value;
}

static void rocm_weight_cache_configure(void) {
    uint32_t entries = DS4_ROCM_WEIGHT_CACHE_DEFAULT_ENTRIES;
    size_t free_size = 0, total_size = 0;
    const bool have_mem_info = hipMemGetInfo(&free_size, &total_size) == hipSuccess;
    uint64_t free_b = (uint64_t)free_size;
    uint64_t total_b = (uint64_t)total_size;
    if (have_mem_info) {
        if (total_b >= 64ull * 1024ull * 1024ull * 1024ull) {
            entries = 40960u;
        } else if (total_b >= 48ull * 1024ull * 1024ull * 1024ull) {
            entries = 32768u;
        }
    }

    uint64_t env_entries = rocm_parse_u64_env("DS4_ROCM_WEIGHT_CACHE_ENTRIES", entries);
    if (env_entries < 1) env_entries = 1;
    if (env_entries > DS4_ROCM_WEIGHT_CACHE_MAX_ENTRIES) env_entries = DS4_ROCM_WEIGHT_CACHE_MAX_ENTRIES;
    g_weight_cache_capacity = (uint32_t)env_entries;

    uint64_t reserve_mib = 0;
    if (have_mem_info && total_b > 0) {
        reserve_mib = total_b / (20ull * 1024ull * 1024ull); // Keep roughly 5% VRAM free.
        if (reserve_mib < 2048ull) reserve_mib = 2048ull;
    }
    reserve_mib = rocm_parse_u64_env("DS4_ROCM_WEIGHT_CACHE_MIN_FREE_MIB", reserve_mib);
    g_weight_cache_min_free_bytes = reserve_mib * 1024ull * 1024ull;
}

static void rocm_weight_cache_evict_index(uint32_t idx) {
    if (idx >= g_weight_cache_len) return;
    rocm_weight_entry *evict = &g_weight_cache[idx];
    rocm_cache_hash_remove(idx);
    if (evict->device) {
        if (evict->free_fn) {
            evict->free_fn(evict->device, evict->free_ctx);
        } else {
            (void)hipFree(evict->device);
        }
    }
    if (g_weight_cache_bytes >= evict->bytes) {
        g_weight_cache_bytes -= evict->bytes;
    }
    memset(evict, 0, sizeof(*evict));
}

static bool rocm_weight_cache_find_empty_slot(uint32_t *idx_out) {
    for (uint32_t i = 0; i < g_weight_cache_len; i++) {
        if (!g_weight_cache[i].device && !g_weight_cache[i].ops) {
            *idx_out = i;
            return true;
        }
    }
    return false;
}

static bool rocm_weight_cache_oldest_index(uint32_t *idx_out) {
    uint64_t oldest = UINT64_MAX;
    bool found = false;
    for (uint32_t i = 0; i < g_weight_cache_len; i++) {
        if ((g_weight_cache[i].device || g_weight_cache[i].ops) &&
            g_weight_cache[i].last_use < oldest) {
            oldest = g_weight_cache[i].last_use;
            *idx_out = i;
            found = true;
        }
    }
    return found;
}

static void rocm_weight_cache_make_room(uint64_t bytes, bool check_device_free) {
    uint32_t idx = 0;
    while (g_weight_cache_len > 0 &&
           g_weight_cache_len >= g_weight_cache_capacity &&
           !rocm_weight_cache_find_empty_slot(&idx)) {
        if (!rocm_weight_cache_oldest_index(&idx)) return;
        rocm_weight_cache_evict_index(idx);
    }
    if (!check_device_free || g_weight_cache_min_free_bytes == 0) return;
    for (;;) {
        size_t free_b = 0, total_b = 0;
        if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return;
        (void)total_b;
        if ((uint64_t)free_b >= bytes + g_weight_cache_min_free_bytes) return;
        if (g_weight_cache_len == 0) return;
        if (!rocm_weight_cache_oldest_index(&idx)) return;
        rocm_weight_cache_evict_index(idx);
    }
}

static bool rocm_expert_table_entry_valid(uint64_t table_idx) {
    uint8_t *ptr = g_expert_table_cpu[table_idx];
    uint64_t gen = g_expert_table_cache_gen[table_idx];
    uint32_t cache_idx = g_expert_table_cache_idx[table_idx];
    if (!ptr || gen == 0 || cache_idx >= g_weight_cache_len) return false;
    const rocm_weight_entry *entry = &g_weight_cache[cache_idx];
    return entry->device == ptr && entry->generation == gen;
}

static void rocm_expert_table_touch(uint64_t table_idx) {
    uint32_t cache_idx = g_expert_table_cache_idx[table_idx];
    g_weight_cache[cache_idx].last_use = ++g_weight_use_clock;
}

static void rocm_expert_table_store(int layer, int expert, int matrix,
                                    uint8_t *dev_ptr,
                                    uint32_t cache_idx,
                                    uint64_t cache_gen) {
    if (layer < 0 || layer >= ROCM_MAX_MOE_LAYERS ||
        expert < 0 || expert >= ROCM_MAX_EXPERTS ||
        matrix < 0 || matrix > 2 ||
        cache_gen == 0) return;
    uint64_t idx = rocm_expert_table_idx(layer, expert, matrix);
    g_expert_table_cpu[idx] = dev_ptr;
    g_expert_table_cache_idx[idx] = cache_idx;
    g_expert_table_cache_gen[idx] = cache_gen;
    if (g_expert_table_dev) {
        (void)hipMemcpyAsync(&g_expert_table_dev[idx],
                             &dev_ptr, sizeof(uint8_t *),
                             hipMemcpyHostToDevice, g_stream);
    }
}

static int rocm_make_1d_launch(uint64_t n, dim3 *grid, dim3 *block) {
    if (n == 0) return 0;
    const uint32_t threads = 256;
    uint64_t blocks = (n + threads - 1u) / threads;
    if (blocks == 0 || blocks > UINT32_MAX) {
        fprintf(stderr, "ds4: ROCm 1D launch too large: %llu elements\n",
                (unsigned long long)n);
        return 0;
    }
    *grid = dim3((uint32_t)blocks);
    *block = dim3(threads);
    return 1;
}

static int rocm_require_tensor_bytes(const char *name,
                                     const ds4_gpu_tensor *tensor,
                                     uint64_t bytes) {
    if (!tensor || ds4_gpu_tensor_bytes(tensor) < bytes) {
        fprintf(stderr, "ds4: ROCm %s received undersized buffers\n", name);
        return 0;
    }
    return 1;
}

static int rocm_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int rocm_bytes_for_elems(uint64_t elems, uint64_t elem_size, uint64_t *bytes) {
    if (elem_size != 0 && elems > UINT64_MAX / elem_size) return 0;
    *bytes = elems * elem_size;
    return 1;
}

static int rocm_q8_0_weight_bytes(uint64_t in_dim, uint64_t out_dim, uint64_t *bytes) {
    if (in_dim == 0 || out_dim == 0 || (in_dim & 31u) != 0) return 0;
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    return rocm_mul_u64(row_bytes, out_dim, bytes);
}

static void rocm_weight_cache_reset(void) {
    for (uint32_t i = 0; i < g_weight_cache_len; i++) {
        if (g_weight_cache[i].device) {
            if (g_weight_cache[i].free_fn) {
                g_weight_cache[i].free_fn(g_weight_cache[i].device, g_weight_cache[i].free_ctx);
            } else {
                (void)hipFree(g_weight_cache[i].device);
            }
        }
    }
    memset(g_weight_cache, 0, sizeof(g_weight_cache));
    g_weight_cache_len = 0;
    g_weight_use_clock = 0;
    g_weight_cache_bytes = 0;
    g_weight_cache_hits = 0;
    g_weight_cache_misses = 0;
    g_weight_cache_generation = 0;
    g_staging_pos = 0;
    memset(g_expert_table_cpu, 0, sizeof(g_expert_table_cpu));
    memset(g_expert_table_cache_idx, 0, sizeof(g_expert_table_cache_idx));
    memset(g_expert_table_cache_gen, 0, sizeof(g_expert_table_cache_gen));
    rocm_cache_hash_clear();
    rocm_weight_cache_configure();
}

static uint8_t *rocm_weight_hip_alloc(uint64_t bytes, void *ctx) {
    (void)ctx;
    uint8_t *device = nullptr;
    hipError_t err = hipMalloc((void **)&device, bytes);
    if (err != hipSuccess) {
        fprintf(stderr,
                "ds4: ROCm failed to allocate %.2f MiB for cached weight: %s\n",
                (double)bytes / (1024.0 * 1024.0),
                hipGetErrorString(err));
        return nullptr;
    }
    return device;
}

static void rocm_weight_hip_free(uint8_t *device, void *ctx) {
    (void)ctx;
    (void)hipFree(device);
}

static int rocm_weight_hip_upload(uint8_t *device, const uint8_t *host, uint64_t bytes, void *ctx) {
    (void)ctx;
    if (g_weight_staging && bytes <= g_weight_staging_size) {
        // Use a non-overlapping slice of the pinned staging buffer for this
        // upload.  The caller resets g_staging_pos to 0 before each MoE batch
        // (after a stream sync guarantees all prior staging DMAs are done),
        // so slices never overlap across batches.
        uint64_t pos = g_staging_pos;
        if (pos + bytes > g_weight_staging_size) {
            hipError_t sync_err = hipStreamSynchronize(g_stream);
            if (sync_err != hipSuccess) {
                fprintf(stderr, "ds4: ROCm weight staging sync failed: %s\n",
                        hipGetErrorString(sync_err));
                return 0;
            }
            g_staging_pos = 0;
            pos = 0;
        }
        if (pos + bytes <= g_weight_staging_size) {
            g_staging_pos = pos + bytes;
            memcpy(g_weight_staging + pos, host, (size_t)bytes);
            hipError_t err = hipMemcpyAsync(device, g_weight_staging + pos, bytes,
                                            hipMemcpyHostToDevice, g_stream);
            if (err == hipSuccess) return 1;
        } else {
            // Staging buffer full — fall through to fallback path.
            // (Caller should reset position before the next batch to avoid this.)
        }
    }
    // Fallback: direct async copy via driver bounce buffer (slow for pageable host)
    hipError_t err = hipMemcpyAsync(device, host, bytes, hipMemcpyHostToDevice, g_stream);
    if (err != hipSuccess) {
        err = hipMemcpy(device, host, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "ds4: ROCm weight upload failed: %s\n", hipGetErrorString(err));
            return 0;
        }
    }
    return 1;
}

static rocm_weight_ops g_rocm_weight_hip_ops = {
    rocm_weight_hip_alloc,
    rocm_weight_hip_free,
    rocm_weight_hip_upload,
    nullptr,
    true,
};

static uint8_t *rocm_weight_ptr_with_ops(const rocm_weight_ops *ops,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t offset,
                                         uint64_t bytes,
                                         uint32_t *cache_idx_out = nullptr,
                                         uint64_t *cache_gen_out = nullptr) {
    if (!ops || !ops->alloc || !ops->free || !ops->upload) return nullptr;
    if (ops->needs_init && !g_initialized && !ds4_gpu_init()) return nullptr;
    if (!model_map || offset > model_size || bytes > model_size - offset) return nullptr;
    if (bytes == 0) return nullptr;
    const uint8_t *map = (const uint8_t *)model_map;
    uint32_t slot = rocm_cache_hash_slot(ops, map, offset, bytes);
    for (uint32_t probe = 0; probe < DS4_ROCM_CACHE_HASH_SIZE; probe++) {
        int32_t idx = g_cache_hash[slot];
        if (idx == -1) break;  // empty slot → not in cache
        rocm_weight_entry *e = &g_weight_cache[idx];
        if (e->ops == ops &&
            e->model_map == map &&
            e->model_size == model_size &&
            e->offset == offset &&
            e->bytes == bytes)
        {
            e->last_use = ++g_weight_use_clock;
            g_weight_cache_hits++;
            if (cache_idx_out) *cache_idx_out = (uint32_t)idx;
            if (cache_gen_out) *cache_gen_out = e->generation;
            return e->device;
        }
        slot = (slot + 1u) & (DS4_ROCM_CACHE_HASH_SIZE - 1u);
    }

    rocm_weight_cache_make_room(bytes, ops->needs_init);

    uint32_t idx = g_weight_cache_len;
    if (idx >= g_weight_cache_capacity) {
        idx = UINT32_MAX;
        for (uint32_t i = 0; i < g_weight_cache_len; i++) {
            if (!g_weight_cache[i].device && !g_weight_cache[i].ops) {
                idx = i;
                break;
            }
        }
        if (idx == UINT32_MAX) {
            if (!rocm_weight_cache_oldest_index(&idx)) return nullptr;
            rocm_weight_cache_evict_index(idx);
        }
    }

    uint8_t *device = ops->alloc(bytes, ops->ctx);
    if (!device) return nullptr;
    if (!ops->upload(device, map + offset, bytes, ops->ctx)) {
        ops->free(device, ops->ctx);
        return nullptr;
    }

    rocm_weight_entry *e = &g_weight_cache[idx];
    e->ops = ops;
    e->model_map = map;
    e->model_size = model_size;
    e->offset = offset;
    e->bytes = bytes;
    e->device = device;
    e->last_use = ++g_weight_use_clock;
    e->generation = ++g_weight_cache_generation;
    if (e->generation == 0) e->generation = ++g_weight_cache_generation;
    e->free_fn = ops->free;
    e->free_ctx = ops->ctx;
    g_weight_cache_bytes += bytes;
    g_weight_cache_misses++;
    rocm_cache_hash_insert(idx);
    if (idx == g_weight_cache_len) g_weight_cache_len = idx + 1;
    if (cache_idx_out) *cache_idx_out = idx;
    if (cache_gen_out) *cache_gen_out = e->generation;
    return e->device;
}

static uint8_t *rocm_weight_ptr(const void *model_map,
                                uint64_t model_size,
                                uint64_t offset,
                                uint64_t bytes) {
    return rocm_weight_ptr_with_ops(&g_rocm_weight_hip_ops, model_map, model_size, offset, bytes);
}

static uint8_t *rocm_weight_ptr_info(const void *model_map,
                                     uint64_t model_size,
                                     uint64_t offset,
                                     uint64_t bytes,
                                     uint32_t *cache_idx_out,
                                     uint64_t *cache_gen_out) {
    return rocm_weight_ptr_with_ops(&g_rocm_weight_hip_ops, model_map, model_size,
                                    offset, bytes, cache_idx_out, cache_gen_out);
}

static bool rocm_moe_selected_entries_ready(int layer_id,
                                            const int32_t *selected_host,
                                            uint32_t n_expert) {
    if (g_expert_table_dev == nullptr || !g_hot_path_arrays_allocated) return false;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        uint32_t e = (uint32_t)selected_host[slot];
        uint64_t idx_g = rocm_expert_table_idx(layer_id, e, ROCM_EXPERT_GATE);
        uint64_t idx_u = rocm_expert_table_idx(layer_id, e, ROCM_EXPERT_UP);
        uint64_t idx_d = rocm_expert_table_idx(layer_id, e, ROCM_EXPERT_DOWN);
        if (!rocm_expert_table_entry_valid(idx_g) ||
            !rocm_expert_table_entry_valid(idx_u) ||
            !rocm_expert_table_entry_valid(idx_d)) {
            return false;
        }
    }
    return true;
}

static int rocm_moe_prewarm_layer_with_ops(const rocm_weight_ops *ops,
                                           int layer_id,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t gate_offset,
                                           uint64_t up_offset,
                                           uint64_t down_offset,
                                           uint64_t gate_expert_bytes,
                                           uint64_t down_expert_bytes,
                                           uint64_t gate_bytes,
                                           uint64_t down_bytes) {
    if (!ops || layer_id < 0 || layer_id >= ROCM_MAX_MOE_LAYERS) return 0;
    for (uint32_t e = 0; e < ROCM_MAX_EXPERTS; e++) {
        uint32_t gate_cache_idx = 0, up_cache_idx = 0, down_cache_idx = 0;
        uint64_t gate_cache_gen = 0, up_cache_gen = 0, down_cache_gen = 0;
        uint8_t *gate_dev = rocm_weight_ptr_with_ops(ops, model_map, model_size,
                                                     gate_offset + (uint64_t)e * gate_expert_bytes,
                                                     gate_bytes,
                                                     &gate_cache_idx,
                                                     &gate_cache_gen);
        uint8_t *up_dev = rocm_weight_ptr_with_ops(ops, model_map, model_size,
                                                   up_offset + (uint64_t)e * gate_expert_bytes,
                                                   gate_bytes,
                                                   &up_cache_idx,
                                                   &up_cache_gen);
        uint8_t *down_dev = rocm_weight_ptr_with_ops(ops, model_map, model_size,
                                                     down_offset + (uint64_t)e * down_expert_bytes,
                                                     down_bytes,
                                                     &down_cache_idx,
                                                     &down_cache_gen);
        if (!gate_dev || !up_dev || !down_dev) return 0;
        rocm_expert_table_store(layer_id, (int)e, ROCM_EXPERT_GATE,
                                gate_dev, gate_cache_idx, gate_cache_gen);
        rocm_expert_table_store(layer_id, (int)e, ROCM_EXPERT_UP,
                                up_dev, up_cache_idx, up_cache_gen);
        rocm_expert_table_store(layer_id, (int)e, ROCM_EXPERT_DOWN,
                                down_dev, down_cache_idx, down_cache_gen);
    }
    return 1;
}

static int rocm_moe_prewarm_layer(int layer_id,
                                  const void *model_map,
                                  uint64_t model_size,
                                  uint64_t gate_offset,
                                  uint64_t up_offset,
                                  uint64_t down_offset,
                                  uint64_t gate_expert_bytes,
                                  uint64_t down_expert_bytes,
                                  uint64_t gate_bytes,
                                  uint64_t down_bytes) {
    return rocm_moe_prewarm_layer_with_ops(&g_rocm_weight_hip_ops, layer_id,
                                           model_map, model_size,
                                           gate_offset, up_offset, down_offset,
                                           gate_expert_bytes, down_expert_bytes,
                                           gate_bytes, down_bytes);
}

static bool rocm_moe_prewarm_enabled(void) {
    const char *s = getenv("DS4_ROCM_MOE_PREWARM");
    if (!s || !s[0] || strcmp(s, "0") == 0) return false;
    if (strcmp(s, "full") == 0) return true;
    if (!g_moe_prewarm_legacy_warned) {
        fprintf(stderr,
                "ds4: ROCm DS4_ROCM_MOE_PREWARM=%s ignored; full MoE prewarm "
                "is memory-heavy. Use DS4_ROCM_MOE_PREWARM=full only for diagnostics.\n",
                s);
        g_moe_prewarm_legacy_warned = true;
    }
    return false;
}

typedef struct {
    uint32_t allocs;
    uint32_t frees;
    uint32_t uploads;
} rocm_weight_test_ctx;

static uint8_t *rocm_weight_test_alloc(uint64_t bytes, void *opaque) {
    rocm_weight_test_ctx *ctx = (rocm_weight_test_ctx *)opaque;
    ctx->allocs++;
    return (uint8_t *)malloc((size_t)bytes);
}

static void rocm_weight_test_free(uint8_t *device, void *opaque) {
    rocm_weight_test_ctx *ctx = (rocm_weight_test_ctx *)opaque;
    ctx->frees++;
    free(device);
}

static int rocm_weight_test_upload(uint8_t *device,
                                   const uint8_t *host,
                                   uint64_t bytes,
                                   void *opaque) {
    rocm_weight_test_ctx *ctx = (rocm_weight_test_ctx *)opaque;
    ctx->uploads++;
    memcpy(device, host, (size_t)bytes);
    return 1;
}

#define DS4_ROCM_STUB() return ds4_rocm_not_implemented(__func__)

/* =========================================================================
 * GPU-side profiler using hipEvent timestamps.
 *
 * Call ds4_gpu_profile_mark(label) at stage boundaries within a command
 * buffer.  Each call records a hipEvent in the stream.  After
 * ds4_gpu_synchronize(), time differences between consecutive marks are
 * aggregated by label pair and printed.
 *
 * Enable with the DS4_GPU_PROFILE environment variable.
 * ========================================================================= */

struct RocmProfileMark {
    const char *label;
    hipEvent_t event;
};

static std::vector<RocmProfileMark> g_profile_marks;
static bool g_profile_enabled = false;

/* Record a GPU timestamp with the given label in the current stream. */
void ds4_gpu_profile_mark(const char *label) {
    if (!g_profile_enabled) return;
    if (!g_initialized && !ds4_gpu_init()) return;
    RocmProfileMark pm;
    pm.label = label;
    hipEventCreate(&pm.event);
    hipEventRecord(pm.event, g_stream);
    g_profile_marks.push_back(pm);
}

static void rocm_profile_print_and_clear(void) {
    if (g_profile_marks.empty()) return;
    // Callers reach this after synchronizing g_stream.  Synchronizing every
    // individual event again makes DS4_GPU_PROFILE distort decode timing.
    // Aggregate time between consecutive same-label pairs into named buckets
    typedef struct { float total_ms; int count; } ProfileBucket;
    std::map<std::string, ProfileBucket> buckets;
    for (size_t i = 1; i < g_profile_marks.size(); i++) {
        float ms = 0.0f;
        hipEventElapsedTime(&ms, g_profile_marks[i - 1].event, g_profile_marks[i].event);
        char key[256];
        snprintf(key, sizeof(key), "%s -> %s", g_profile_marks[i - 1].label, g_profile_marks[i].label);
        auto it = buckets.find(key);
        if (it != buckets.end()) {
            it->second.total_ms += ms;
            it->second.count++;
        } else {
            buckets[key] = {ms, 1};
        }
    }
    for (auto &pm : g_profile_marks) {
        hipEventDestroy(pm.event);
    }
    g_profile_marks.clear();

    fprintf(stderr, "\nds4: ROCm GPU profile (cumulative across layers):\n");
    double grand_total = 0.0;
    for (auto &kv : buckets) {
        fprintf(stderr, "  %-50s %8.2f ms  (%d calls)\n",
                kv.first.c_str(), (double)kv.second.total_ms, kv.second.count);
        grand_total += kv.second.total_ms;
    }
    fprintf(stderr, "  %-50s %8.2f ms\n", "total", grand_total);
}

extern "C" {

int ds4_gpu_init(void) {
    if (g_initialized) return 1;
    if (getenv("DS4_GPU_PROFILE") != NULL) g_profile_enabled = true;
    DS4_ROCM_CHECK(hipSetDevice(0));
    DS4_ROCM_CHECK(hipStreamCreateWithFlags(&g_stream, hipStreamNonBlocking));
    // Allocate pinned staging buffer for weight uploads (256 MB).
    // Used by rocm_weight_hip_upload as a CPU→pinned bounce buffer to avoid
    // the driver's slow staged copy from pageable mmap memory.
    g_weight_staging_size = 256ull * 1048576ull;
    hipError_t stage_err = hipHostMalloc((void **)&g_weight_staging,
                                          g_weight_staging_size, 0);
    if (stage_err != hipSuccess) {
        fprintf(stderr, "ds4: ROCm hipHostMalloc(%zu) for weight staging failed (%s); "
                        "uploads will be slower (staged copies)\n",
                        (size_t)g_weight_staging_size, hipGetErrorString(stage_err));
        g_weight_staging = nullptr;
        g_weight_staging_size = 0;
    }
    rocm_cache_hash_clear();
    rocm_weight_cache_configure();

    // Allocate device-side expert pointer table
    if (!g_expert_table_dev) {
        hipError_t et = hipMalloc(&g_expert_table_dev,
                                   (size_t)ROCM_MAX_MOE_LAYERS * ROCM_EXPERT_TABLE_STEP * sizeof(uint8_t *));
        if (et == hipSuccess) {
            DS4_ROCM_CHECK(hipMemsetAsync(g_expert_table_dev, 0,
                                           (size_t)ROCM_MAX_MOE_LAYERS * ROCM_EXPERT_TABLE_STEP * sizeof(uint8_t *),
                                           g_stream));
        } else {
            fprintf(stderr, "ds4: ROCm expert table allocation failed: %s\n", hipGetErrorString(et));
            g_expert_table_dev = nullptr;
        }
    }

    // Pre-allocate hot-path pointer arrays (max 6 experts)
    if (!g_hot_path_arrays_allocated) {
        hipError_t e1 = hipMalloc(&g_gate_ptrs_dev, 6 * sizeof(uint8_t *));
        hipError_t e2 = hipMalloc(&g_up_ptrs_dev,   6 * sizeof(uint8_t *));
        hipError_t e3 = hipMalloc(&g_down_ptrs_dev, 6 * sizeof(uint8_t *));
        hipError_t e4 = hipMalloc(&g_weights_dev,   6 * sizeof(float));
        if (e1 == hipSuccess && e2 == hipSuccess && e3 == hipSuccess && e4 == hipSuccess) {
            g_hot_path_arrays_allocated = true;
        } else {
            hipFree(g_gate_ptrs_dev); g_gate_ptrs_dev = nullptr;
            hipFree(g_up_ptrs_dev);   g_up_ptrs_dev   = nullptr;
            hipFree(g_down_ptrs_dev); g_down_ptrs_dev = nullptr;
            hipFree(g_weights_dev);   g_weights_dev   = nullptr;
            g_hot_path_arrays_allocated = false;
        }
    }

    g_initialized = 1;
    return 1;
}

void ds4_gpu_cleanup(void) {
    if (!g_initialized) return;
    fprintf(stderr, "ROCm weight cache: %u/%u entries, %llu hits, %llu misses, %llu bytes\n",
            g_weight_cache_len,
            g_weight_cache_capacity,
            (unsigned long long)g_weight_cache_hits,
            (unsigned long long)g_weight_cache_misses,
            (unsigned long long)g_weight_cache_bytes);
    (void)hipStreamSynchronize(g_stream);
    rocm_weight_cache_reset();
    if (g_weight_staging) {
        (void)hipHostFree(g_weight_staging);
        g_weight_staging = nullptr;
        g_weight_staging_size = 0;
    }
    if (g_preq_x)   { (void)hipFree(g_preq_x);   g_preq_x = nullptr;   g_preq_x_cap = 0; }
    if (g_preq_mid) { (void)hipFree(g_preq_mid); g_preq_mid = nullptr; g_preq_mid_cap = 0; }
    if (g_expert_table_dev) { (void)hipFree(g_expert_table_dev); g_expert_table_dev = nullptr; }
    memset(g_expert_table_cpu, 0, sizeof(g_expert_table_cpu));
    memset(g_expert_table_cache_idx, 0, sizeof(g_expert_table_cache_idx));
    memset(g_expert_table_cache_gen, 0, sizeof(g_expert_table_cache_gen));
    if (g_hot_path_arrays_allocated) {
        (void)hipFree(g_gate_ptrs_dev);  g_gate_ptrs_dev  = nullptr;
        (void)hipFree(g_up_ptrs_dev);    g_up_ptrs_dev    = nullptr;
        (void)hipFree(g_down_ptrs_dev);  g_down_ptrs_dev  = nullptr;
        (void)hipFree(g_weights_dev);    g_weights_dev    = nullptr;
        g_hot_path_arrays_allocated = false;
    }
    (void)hipStreamDestroy(g_stream);
    g_stream = nullptr;
    g_initialized = 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_initialized && !ds4_gpu_init()) return nullptr;
    if (bytes == 0) return nullptr;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return nullptr;
    if (hipMalloc((void **)&t->ptr, bytes) != hipSuccess) {
        free(t);
        return nullptr;
    }
    t->bytes = bytes;
    t->owner = true;
    return t;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return nullptr;
    ds4_gpu_tensor *v = (ds4_gpu_tensor *)calloc(1, sizeof(*v));
    if (!v) return nullptr;
    v->ptr = base->ptr + offset;
    v->bytes = bytes;
    v->owner = false;
    v->base = (ds4_gpu_tensor *)base;
    v->offset = offset;
    return v;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)hipFree(tensor->ptr);
    free(tensor);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    (void)tensor;
    return nullptr;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > UINT32_MAX) return 0;
    uint64_t bytes;
    if (!rocm_bytes_for_elems(count, sizeof(float), &bytes) ||
        !rocm_require_tensor_bytes("tensor fill f32", tensor, bytes)) {
        return 0;
    }
    if (count == 0) return 1;
    if (value == 0.0f) {
        DS4_ROCM_CHECK(hipMemsetAsync(tensor->ptr, 0, bytes, g_stream));
        return 1;
    }
    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(count, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_fill_f32, grid, block, 0, g_stream,
                       (float *)tensor->ptr,
                       value,
                       (uint32_t)count);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0) || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    DS4_ROCM_CHECK(hipMemcpyAsync(tensor->ptr + offset, data, bytes, hipMemcpyHostToDevice, g_stream));
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes != 0) || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    if (bytes == 0) return 1;
    DS4_ROCM_CHECK(hipMemcpyAsync(data, tensor->ptr + offset, bytes, hipMemcpyDeviceToHost, g_stream));
    DS4_ROCM_CHECK(hipStreamSynchronize(g_stream));
    if (g_profile_enabled) rocm_profile_print_and_clear();
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || bytes > dst->bytes - dst_offset ||
        src_offset > src->bytes || bytes > src->bytes - src_offset)
    {
        return 0;
    }
    if (bytes == 0) return 1;
    DS4_ROCM_CHECK(hipMemcpyAsync(dst->ptr + dst_offset,
                                  src->ptr + src_offset,
                                  bytes,
                                  hipMemcpyDeviceToDevice,
                                  g_stream));
    return 1;
}

int ds4_gpu_begin_commands(void) {
    return (!g_initialized && !ds4_gpu_init()) ? 0 : 1;
}

int ds4_gpu_flush_commands(void) {
    return (!g_initialized && !ds4_gpu_init()) ? 0 : 1;
}

int ds4_gpu_end_commands(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    return 1;
}

int ds4_gpu_synchronize(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    DS4_ROCM_CHECK(hipStreamSynchronize(g_stream));
    if (g_profile_enabled) rocm_profile_print_and_clear();
    return 1;
}

int ds4_gpu_set_model_map_range(const void *model_map,
                                  uint64_t model_size,
                                  uint64_t map_offset,
                                  uint64_t map_size) {
    if (!model_map || map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;
    rocm_weight_cache_reset();
    g_model_map = (const uint8_t *)model_map;
    g_model_size = model_size;
    g_model_offset = map_offset;
    g_model_bytes = map_size;
    return 1;
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map_range(model_map, model_size, 0, model_size);
}

void ds4_gpu_set_quality(bool quality) {
    g_quality = quality;
}

void ds4_gpu_print_memory_report(const char *label) {
    fprintf(stderr,
            "ds4: ROCm memory report: %s model_bytes=%llu mapped_bytes=%llu quality=%d weight_cache_entries=%u weight_cache_capacity=%u weight_cache_bytes=%llu weight_cache_min_free=%llu weight_cache_hits=%llu weight_cache_misses=%llu\n",
            label ? label : "",
            (unsigned long long)g_model_size,
            (unsigned long long)g_model_bytes,
            g_quality ? 1 : 0,
            g_weight_cache_len,
            g_weight_cache_capacity,
            (unsigned long long)g_weight_cache_bytes,
            (unsigned long long)g_weight_cache_min_free_bytes,
            (unsigned long long)g_weight_cache_hits,
            (unsigned long long)g_weight_cache_misses);
}

int ds4_rocm_test_weight_cache(void) {
    rocm_weight_cache_reset();
    rocm_weight_test_ctx ctx = {0};
    rocm_weight_ops ops = {
        rocm_weight_test_alloc,
        rocm_weight_test_free,
        rocm_weight_test_upload,
        &ctx,
        false,
    };
    int ok = 0;

    uint8_t model_a[512];
    uint8_t model_b[512];
    for (uint32_t i = 0; i < 512; i++) {
        model_a[i] = (uint8_t)(i ^ 0x5au);
        model_b[i] = (uint8_t)(255u - (i & 255u));
    }

    uint8_t *first = rocm_weight_ptr_with_ops(&ops, model_a, sizeof(model_a), 64, 128);
    uint8_t *again = rocm_weight_ptr_with_ops(&ops, model_a, sizeof(model_a), 64, 128);
    uint8_t *second_map = rocm_weight_ptr_with_ops(&ops, model_b, sizeof(model_b), 64, 128);
    uint8_t *bad_range = rocm_weight_ptr_with_ops(&ops, model_a, sizeof(model_a), 500, 16);
    if (!first || !again || !second_map || bad_range) goto done;
    if (first != again) goto done;
    if (second_map == first) goto done;
    if (memcmp(first, model_a + 64, 128) != 0) goto done;
    if (memcmp(second_map, model_b + 64, 128) != 0) goto done;
    if (g_weight_cache_len != 2) goto done;
    if (g_weight_cache_bytes != 256) goto done;
    if (g_weight_cache_hits != 1 || g_weight_cache_misses != 2) goto done;
    if (ctx.allocs != 2 || ctx.uploads != 2 || ctx.frees != 0) goto done;
    ok = 1;

done:
    rocm_weight_cache_reset();
    return ok && ctx.frees == ctx.allocs;
}

int ds4_rocm_test_weight_cache_dynamic_capacity(void) {
    const char *old_env = getenv("DS4_ROCM_WEIGHT_CACHE_ENTRIES");
    char *old_copy = old_env ? strdup(old_env) : nullptr;
    setenv("DS4_ROCM_WEIGHT_CACHE_ENTRIES", "40000", 1);

    rocm_weight_cache_reset();
    rocm_weight_test_ctx ctx = {0};
    rocm_weight_ops ops = {
        rocm_weight_test_alloc,
        rocm_weight_test_free,
        rocm_weight_test_upload,
        &ctx,
        false,
    };
    const uint32_t target_entries = 33025;
    uint8_t *model = (uint8_t *)malloc(target_entries);
    int ok = 0;
    if (!model) goto done;
    for (uint32_t i = 0; i < target_entries; i++) model[i] = (uint8_t)(i & 255u);
    for (uint32_t i = 0; i < target_entries; i++) {
        uint8_t *ptr = rocm_weight_ptr_with_ops(&ops, model, target_entries, i, 1);
        if (!ptr || *ptr != (uint8_t)(i & 255u)) goto done;
    }
    ok = g_weight_cache_len == target_entries &&
         g_weight_cache_misses == target_entries &&
         ctx.frees == 0;

done:
    rocm_weight_cache_reset();
    free(model);
    if (old_copy) {
        setenv("DS4_ROCM_WEIGHT_CACHE_ENTRIES", old_copy, 1);
        free(old_copy);
    } else {
        unsetenv("DS4_ROCM_WEIGHT_CACHE_ENTRIES");
    }
    return ok && ctx.frees == ctx.allocs;
}

int ds4_rocm_test_moe_prewarm_layer(void) {
    rocm_weight_cache_reset();
    rocm_weight_test_ctx ctx = {0};
    rocm_weight_ops ops = {
        rocm_weight_test_alloc,
        rocm_weight_test_free,
        rocm_weight_test_upload,
        &ctx,
        false,
    };
    const uint64_t gate_bytes = 2;
    const uint64_t down_bytes = 3;
    const uint64_t gate_expert_bytes = gate_bytes;
    const uint64_t down_expert_bytes = down_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_offset + 256ull * gate_expert_bytes;
    const uint64_t down_offset = up_offset + 256ull * gate_expert_bytes;
    const uint64_t model_size = down_offset + 256ull * down_expert_bytes;
    uint8_t *model = (uint8_t *)malloc((size_t)model_size);
    int ok = 0;
    if (!model) goto done;
    for (uint64_t i = 0; i < model_size; i++) model[i] = (uint8_t)(i & 255u);

    if (!rocm_moe_prewarm_layer_with_ops(&ops, 7, model, model_size,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, down_expert_bytes,
                                         gate_bytes, down_bytes)) {
        goto done;
    }
    {
        const uint32_t expert = 123;
        uint64_t idx_g = rocm_expert_table_idx(7, expert, ROCM_EXPERT_GATE);
        uint64_t idx_u = rocm_expert_table_idx(7, expert, ROCM_EXPERT_UP);
        uint64_t idx_d = rocm_expert_table_idx(7, expert, ROCM_EXPERT_DOWN);
        ok = g_weight_cache_len == 256u * 3u &&
             g_weight_cache_misses == 256u * 3u &&
             rocm_expert_table_entry_valid(idx_g) &&
             rocm_expert_table_entry_valid(idx_u) &&
             rocm_expert_table_entry_valid(idx_d) &&
             g_expert_table_cpu[idx_g][0] == model[gate_offset + (uint64_t)expert * gate_expert_bytes] &&
             g_expert_table_cpu[idx_u][0] == model[up_offset + (uint64_t)expert * gate_expert_bytes] &&
             g_expert_table_cpu[idx_d][0] == model[down_offset + (uint64_t)expert * down_expert_bytes];
    }

done:
    rocm_weight_cache_reset();
    free(model);
    return ok && ctx.frees == ctx.allocs;
}

int ds4_rocm_test_moe_prewarm_env(void) {
    const char *old_env = getenv("DS4_ROCM_MOE_PREWARM");
    char *old_copy = old_env ? strdup(old_env) : nullptr;
    int ok = 0;

    unsetenv("DS4_ROCM_MOE_PREWARM");
    if (rocm_moe_prewarm_enabled()) goto done;
    setenv("DS4_ROCM_MOE_PREWARM", "0", 1);
    if (rocm_moe_prewarm_enabled()) goto done;
    setenv("DS4_ROCM_MOE_PREWARM", "1", 1);
    if (rocm_moe_prewarm_enabled()) goto done;
    setenv("DS4_ROCM_MOE_PREWARM", "full", 1);
    if (!rocm_moe_prewarm_enabled()) goto done;
    ok = 1;

done:
    if (old_copy) {
        setenv("DS4_ROCM_MOE_PREWARM", old_copy, 1);
        free(old_copy);
    } else {
        unsetenv("DS4_ROCM_MOE_PREWARM");
    }
    return ok;
}

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !model_map || n_vocab == 0 || token >= n_vocab || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    uint64_t out_elems;
    uint64_t weight_elems;
    if (!rocm_mul_u64(n_embd, n_hc, &out_elems) ||
        !rocm_mul_u64(n_vocab, n_embd, &weight_elems) ||
        out_elems > UINT64_MAX / sizeof(float) ||
        weight_elems > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    uint64_t out_bytes = out_elems * sizeof(float);
    uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (!rocm_require_tensor_bytes("graph embedding", out_hc, out_bytes)) return 0;

    uint8_t *w = rocm_weight_ptr(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(out_elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_embed_token_hc_f16, grid, block, 0, g_stream,
                       (float *)out_hc->ptr,
                       (const uint16_t *)w,
                       token,
                       n_embd,
                       n_hc);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !tokens || !model_map || n_vocab == 0 || n_tokens == 0 || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    uint64_t token_hc;
    uint64_t out_elems;
    uint64_t weight_elems;
    if (!rocm_mul_u64(n_tokens, n_hc, &token_hc) ||
        !rocm_mul_u64(token_hc, n_embd, &out_elems) ||
        !rocm_mul_u64(n_vocab, n_embd, &weight_elems) ||
        out_elems > UINT64_MAX / sizeof(float) ||
        weight_elems > UINT64_MAX / sizeof(uint16_t)) {
        return 0;
    }
    uint64_t out_bytes = out_elems * sizeof(float);
    uint64_t token_bytes = (uint64_t)n_tokens * sizeof(int32_t);
    uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (!rocm_require_tensor_bytes("batched embedding", out_hc, out_bytes) ||
        !rocm_require_tensor_bytes("batched embedding", tokens, token_bytes)) {
        return 0;
    }

    uint8_t *w = rocm_weight_ptr(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(out_elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_embed_tokens_hc_f16, grid, block, 0, g_stream,
                       (float *)out_hc->ptr,
                       (const int32_t *)tokens->ptr,
                       (const uint16_t *)w,
                       n_vocab,
                       n_tokens,
                       n_embd,
                       n_hc);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_indexer_score_one_tensor(ds4_gpu_tensor *scores,
                                       const ds4_gpu_tensor *q,
                                       const ds4_gpu_tensor *weights,
                                       const ds4_gpu_tensor *index_comp,
                                       uint32_t n_comp,
                                       uint32_t n_head,
                                       uint32_t head_dim,
                                       float scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_head == 0 || head_dim == 0) {
        return 0;
    }

    uint64_t q_elems;
    uint64_t comp_elems;
    if (!rocm_mul_u64(n_head, head_dim, &q_elems) ||
        !rocm_mul_u64(n_comp, head_dim, &comp_elems)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("indexer score q", q, q_elems * sizeof(float)) ||
        !rocm_require_tensor_bytes("indexer score weights", weights, (uint64_t)n_head * sizeof(float)) ||
        !rocm_require_tensor_bytes("indexer score comp", index_comp, comp_elems * sizeof(float)) ||
        !rocm_require_tensor_bytes("indexer score out", scores, (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_indexer_score_one_f32, dim3(n_comp), dim3(256), 0, g_stream,
                       (float *)scores->ptr,
                       (const float *)q->ptr,
                       (const float *)weights->ptr,
                       (const float *)index_comp->ptr,
                       n_comp,
                       n_head,
                       head_dim,
                       scale);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

static int rocm_indexer_scores_batch_tensor(ds4_gpu_tensor *scores,
                                            const ds4_gpu_tensor *q,
                                            const ds4_gpu_tensor *weights,
                                            const ds4_gpu_tensor *index_comp,
                                            uint32_t n_comp,
                                            uint32_t n_tokens,
                                            uint32_t pos0,
                                            uint32_t n_head,
                                            uint32_t head_dim,
                                            uint32_t ratio,
                                            float scale) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0) {
        return 0;
    }

    uint64_t q_rows;
    uint64_t q_elems;
    uint64_t weight_elems;
    uint64_t comp_elems;
    uint64_t score_elems;
    uint64_t q_bytes;
    uint64_t weight_bytes;
    uint64_t comp_bytes;
    uint64_t score_bytes;
    if (!rocm_mul_u64(n_tokens, n_head, &q_rows) ||
        !rocm_mul_u64(q_rows, head_dim, &q_elems) ||
        !rocm_mul_u64(n_tokens, n_head, &weight_elems) ||
        !rocm_mul_u64(n_comp, head_dim, &comp_elems) ||
        !rocm_mul_u64(n_tokens, n_comp, &score_elems) ||
        !rocm_bytes_for_elems(q_elems, sizeof(float), &q_bytes) ||
        !rocm_bytes_for_elems(weight_elems, sizeof(float), &weight_bytes) ||
        !rocm_bytes_for_elems(comp_elems, sizeof(float), &comp_bytes) ||
        !rocm_bytes_for_elems(score_elems, sizeof(float), &score_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("indexer scores q", q, q_bytes) ||
        !rocm_require_tensor_bytes("indexer scores weights", weights, weight_bytes) ||
        !rocm_require_tensor_bytes("indexer scores comp", index_comp, comp_bytes) ||
        !rocm_require_tensor_bytes("indexer scores out", scores, score_bytes)) {
        return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_indexer_scores_batch_f32,
                       dim3(n_comp, n_tokens), dim3(256), 0, g_stream,
                       (float *)scores->ptr,
                       (const float *)q->ptr,
                       (const float *)weights->ptr,
                       (const float *)index_comp->ptr,
                       n_comp,
                       n_tokens,
                       pos0,
                       n_head,
                       head_dim,
                       ratio,
                       scale);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_indexer_scores_prefill_tensor(ds4_gpu_tensor *scores,
                                            const ds4_gpu_tensor *q,
                                            const ds4_gpu_tensor *weights,
                                            const ds4_gpu_tensor *index_comp,
                                            uint32_t n_comp,
                                            uint32_t n_tokens,
                                            uint32_t n_head,
                                            uint32_t head_dim,
                                            uint32_t ratio,
                                            float scale) {
    return rocm_indexer_scores_batch_tensor(scores, q, weights, index_comp,
                                            n_comp, n_tokens, 0, n_head,
                                            head_dim, ratio, scale);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(ds4_gpu_tensor *scores,
                                                 const ds4_gpu_tensor *q,
                                                 const ds4_gpu_tensor *weights,
                                                 const ds4_gpu_tensor *index_comp,
                                                 uint32_t n_comp,
                                                 uint32_t n_tokens,
                                                 uint32_t pos0,
                                                 uint32_t n_head,
                                                 uint32_t head_dim,
                                                 uint32_t ratio,
                                                 float scale) {
    return rocm_indexer_scores_batch_tensor(scores, q, weights, index_comp,
                                            n_comp, n_tokens, pos0, n_head,
                                            head_dim, ratio, scale);
}

int ds4_gpu_indexer_topk_tensor(ds4_gpu_tensor *selected,
                                  const ds4_gpu_tensor *scores,
                                  uint32_t n_comp,
                                  uint32_t n_tokens,
                                  uint32_t top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;

    uint64_t score_elems;
    uint64_t selected_elems;
    if (!rocm_mul_u64(n_comp, n_tokens, &score_elems) ||
        !rocm_mul_u64(top_k, n_tokens, &selected_elems)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("indexer topk scores", scores, score_elems * sizeof(float)) ||
        !rocm_require_tensor_bytes("indexer topk selected", selected, selected_elems * sizeof(int32_t))) {
        return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_indexer_topk_i32, dim3(n_tokens), dim3(1), 0, g_stream,
                       (int32_t *)selected->ptr,
                       (const float *)scores->ptr,
                       n_comp,
                       n_tokens,
                       top_k);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_dsv4_topk_mask_tensor(ds4_gpu_tensor *mask,
                                    const ds4_gpu_tensor *topk,
                                    uint32_t n_comp,
                                    uint32_t n_tokens,
                                    uint32_t top_k) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0) return 0;

    uint64_t mask_elems;
    uint64_t topk_elems;
    if (!rocm_mul_u64(n_comp, n_tokens, &mask_elems) ||
        !rocm_mul_u64(top_k, n_tokens, &topk_elems)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("topk mask dst", mask, mask_elems * sizeof(float)) ||
        !rocm_require_tensor_bytes("topk mask src", topk, topk_elems * sizeof(int32_t))) {
        return 0;
    }

    const uint64_t launch_elems = mask_elems > topk_elems ? mask_elems : topk_elems;
    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(launch_elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_topk_mask_f32, grid, block, 0, g_stream,
                       (float *)mask->ptr,
                       (const int32_t *)topk->ptr,
                       n_comp,
                       n_tokens,
                       top_k);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t weight_offset,
                                 uint64_t in_dim,
                                 uint64_t out_dim,
                                 const ds4_gpu_tensor *x,
                                 uint64_t n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    if ((in_dim & 31u) != 0) return 0;

    uint64_t x_elems;
    uint64_t out_elems;
    uint64_t x_bytes;
    uint64_t out_bytes;
    uint64_t weight_bytes;
    if (!rocm_mul_u64(in_dim, n_tok, &x_elems) ||
        !rocm_mul_u64(out_dim, n_tok, &out_elems) ||
        !rocm_bytes_for_elems(x_elems, sizeof(float), &x_bytes) ||
        !rocm_bytes_for_elems(out_elems, sizeof(float), &out_bytes) ||
        !rocm_q8_0_weight_bytes(in_dim, out_dim, &weight_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("Q8_0 matmul x", x, x_bytes) ||
        !rocm_require_tensor_bytes("Q8_0 matmul out", out, out_bytes)) {
        return 0;
    }

    uint8_t *w = rocm_weight_ptr(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    dim3 block(256);
    if (n_tok == 1) {
        dim3 grid((uint32_t)out_dim);
        hipLaunchKernelGGL(ds4_rocm_mul_mv_q8_0_f32_1tok, grid, block, 0, g_stream,
                           (const uint8_t *)w,
                           (const float *)x->ptr,
                           (float *)out->ptr,
                           (uint32_t)in_dim,
                           (uint32_t)out_dim);
    } else {
        dim3 grid((uint32_t)out_dim, (uint32_t)n_tok);
        hipLaunchKernelGGL(ds4_rocm_mul_mm_q8_0_f32, grid, block, 0, g_stream,
                           (const uint8_t *)w,
                           (const float *)x->ptr,
                           (float *)out->ptr,
                           (uint32_t)in_dim,
                           (uint32_t)out_dim,
                           (uint32_t)n_tok);
    }
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(ds4_gpu_tensor *gate,
                                                ds4_gpu_tensor *up,
                                                ds4_gpu_tensor *mid,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t gate_offset,
                                                uint64_t up_offset,
                                                uint64_t in_dim,
                                                uint64_t out_dim,
                                                const ds4_gpu_tensor *x) {
    if (!ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                      gate_offset, in_dim, out_dim, x, 1)) {
        return 0;
    }
    if (!ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                      up_offset, in_dim, out_dim, x, 1)) {
        return 0;
    }
    if (out_dim > UINT32_MAX) return 0;
    return ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, 0.0f, 1.0f);
}

int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint64_t in_dim,
                                uint64_t out_dim,
                                const ds4_gpu_tensor *x,
                                uint64_t n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    if (in_dim == 0 || out_dim == 0) return 0;

    uint64_t weight_elems;
    uint64_t x_elems;
    uint64_t out_elems;
    uint64_t weight_bytes;
    uint64_t x_bytes;
    uint64_t out_bytes;
    if (!rocm_mul_u64(in_dim, out_dim, &weight_elems) ||
        !rocm_mul_u64(in_dim, n_tok, &x_elems) ||
        !rocm_mul_u64(out_dim, n_tok, &out_elems) ||
        !rocm_bytes_for_elems(weight_elems, sizeof(uint16_t), &weight_bytes) ||
        !rocm_bytes_for_elems(x_elems, sizeof(float), &x_bytes) ||
        !rocm_bytes_for_elems(out_elems, sizeof(float), &out_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("F16 matmul x", x, x_bytes) ||
        !rocm_require_tensor_bytes("F16 matmul out", out, out_bytes)) {
        return 0;
    }

    uint8_t *w = rocm_weight_ptr(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    dim3 block(256);
    if (n_tok == 1) {
        dim3 grid((uint32_t)out_dim);
        hipLaunchKernelGGL(ds4_rocm_mul_mv_f16_f32_1tok, grid, block, 0, g_stream,
                           (const uint16_t *)w,
                           (const float *)x->ptr,
                           (float *)out->ptr,
                           (uint32_t)in_dim,
                           (uint32_t)out_dim);
    } else {
        dim3 grid((uint32_t)out_dim, (uint32_t)n_tok);
        hipLaunchKernelGGL(ds4_rocm_mul_mm_f16_f32, grid, block, 0, g_stream,
                           (const uint16_t *)w,
                           (const float *)x->ptr,
                           (float *)out->ptr,
                           (uint32_t)in_dim,
                           (uint32_t)out_dim,
                           (uint32_t)n_tok);
    }
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_matmul_f16_pair_tensor(ds4_gpu_tensor *out_a,
                                     ds4_gpu_tensor *out_b,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_a_offset,
                                     uint64_t weight_b_offset,
                                     uint64_t in_dim,
                                     uint64_t out_dim,
                                     const ds4_gpu_tensor *x,
                                     uint64_t n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (n_tok != 1) {
        fprintf(stderr,
                "ds4: ROCm primitive not implemented: %s n_tok=%llu\n",
                __func__,
                (unsigned long long)n_tok);
        return 0;
    }
    if (!out_a || !out_b || !x || in_dim == 0 || out_dim == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX) return 0;

    uint64_t elems;
    uint64_t weight_bytes;
    uint64_t out_bytes;
    if (!rocm_mul_u64(in_dim, out_dim, &elems) ||
        !rocm_bytes_for_elems(elems, sizeof(uint16_t), &weight_bytes) ||
        !rocm_bytes_for_elems(out_dim, sizeof(float), &out_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("F16 pair matvec x", x, in_dim * sizeof(float)) ||
        !rocm_require_tensor_bytes("F16 pair matvec out_a", out_a, out_bytes) ||
        !rocm_require_tensor_bytes("F16 pair matvec out_b", out_b, out_bytes)) {
        return 0;
    }

    uint8_t *wa = rocm_weight_ptr(model_map, model_size, weight_a_offset, weight_bytes);
    uint8_t *wb = rocm_weight_ptr(model_map, model_size, weight_b_offset, weight_bytes);
    if (!wa || !wb) return 0;

    dim3 block(256);
    dim3 grid((uint32_t)out_dim);
    hipLaunchKernelGGL(ds4_rocm_mul_mv_f16_pair_f32_1tok, grid, block, 0, g_stream,
                       (const uint16_t *)wa,
                       (const uint16_t *)wb,
                       (const float *)x->ptr,
                       (float *)out_a->ptr,
                       (float *)out_b->ptr,
                       (uint32_t)in_dim,
                       (uint32_t)out_dim);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint64_t in_dim,
                                uint64_t out_dim,
                                const ds4_gpu_tensor *x,
                                uint64_t n_tok) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    uint64_t elems;
    uint64_t x_elems;
    uint64_t out_elems;
    uint64_t weight_bytes;
    uint64_t x_bytes;
    uint64_t out_bytes;
    if (!rocm_mul_u64(in_dim, out_dim, &elems) ||
        !rocm_mul_u64(in_dim, n_tok, &x_elems) ||
        !rocm_mul_u64(out_dim, n_tok, &out_elems) ||
        !rocm_bytes_for_elems(elems, sizeof(float), &weight_bytes) ||
        !rocm_bytes_for_elems(x_elems, sizeof(float), &x_bytes) ||
        !rocm_bytes_for_elems(out_elems, sizeof(float), &out_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("F32 matmul x", x, x_bytes) ||
        !rocm_require_tensor_bytes("F32 matmul out", out, out_bytes)) {
        return 0;
    }

    uint8_t *w = rocm_weight_ptr(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    dim3 block(256);
    if (n_tok == 1) {
        dim3 grid((uint32_t)out_dim);
        hipLaunchKernelGGL(ds4_rocm_mul_mv_f32_f32_1tok, grid, block, 0, g_stream,
                           (const float *)w,
                           (const float *)x->ptr,
                           (float *)out->ptr,
                           (uint32_t)in_dim,
                           (uint32_t)out_dim);
    } else {
        dim3 grid((uint32_t)out_dim, (uint32_t)n_tok);
        hipLaunchKernelGGL(ds4_rocm_mul_mm_f32_f32, grid, block, 0, g_stream,
                           (const float *)w,
                           (const float *)x->ptr,
                           (float *)out->ptr,
                           (uint32_t)in_dim,
                           (uint32_t)out_dim,
                           (uint32_t)n_tok);
    }
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !row || n_embd == 0 || n_hc == 0) return 0;

    uint64_t total = (uint64_t)n_embd * n_hc;
    if (total > UINT64_MAX / sizeof(float)) return 0;
    uint64_t row_bytes = (uint64_t)n_embd * sizeof(float);
    uint64_t out_bytes = total * sizeof(float);
    if (!rocm_require_tensor_bytes("HC repeat", row, row_bytes) ||
        !rocm_require_tensor_bytes("HC repeat", out, out_bytes)) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(total, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_repeat_hc_f32, grid, block, 0, g_stream,
                       (float *)out->ptr,
                       (const float *)row->ptr,
                       n_embd,
                       n_hc);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    uint64_t elems;
    if (!rocm_mul_u64(n, rows, &elems) || elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t bytes = elems * sizeof(float);
    if (!rocm_require_tensor_bytes("plain RMS norm", x, bytes) ||
        !rocm_require_tensor_bytes("plain RMS norm", out, bytes)) {
        return 0;
    }

    dim3 grid(rows);
    dim3 block(256);
    hipLaunchKernelGGL(ds4_rocm_rms_norm_plain_rows_f32, grid, block,
                       block.x * sizeof(float), g_stream,
                       (float *)out->ptr,
                       (const float *)x->ptr,
                       n,
                       rows,
                       eps);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, model_map, model_size, weight_offset, n, 1, eps);
}

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !x || !model_map || n == 0 || rows == 0 || (n & 3u) != 0) return 0;

    uint64_t elems;
    if (!rocm_mul_u64(n, rows, &elems) || elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t bytes = elems * sizeof(float);
    uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (!rocm_require_tensor_bytes("weighted RMS norm", x, bytes) ||
        !rocm_require_tensor_bytes("weighted RMS norm", out, bytes)) {
        return 0;
    }

    uint8_t *w = rocm_weight_ptr(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    dim3 grid(rows);
    dim3 block(256);
    hipLaunchKernelGGL(ds4_rocm_rms_norm_weight_rows_f32, grid, block,
                       block.x * sizeof(float), g_stream,
                       (float *)out->ptr,
                       (const float *)x->ptr,
                       (const float *)w,
                       n,
                       rows,
                       eps);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!q_out || !q || !kv_out || !kv || !model_map ||
        q_n == 0 || kv_n == 0 || rows == 0 || (q_n & 3u) != 0 || (kv_n & 3u) != 0) {
        return 0;
    }

    uint64_t q_elems;
    uint64_t kv_elems;
    if (!rocm_mul_u64(q_n, rows, &q_elems) ||
        !rocm_mul_u64(kv_n, rows, &kv_elems) ||
        q_elems > UINT64_MAX / sizeof(float) ||
        kv_elems > UINT64_MAX / sizeof(float)) {
        return 0;
    }
    uint64_t q_bytes = q_elems * sizeof(float);
    uint64_t kv_bytes = kv_elems * sizeof(float);
    uint64_t q_weight_bytes = (uint64_t)q_n * sizeof(float);
    uint64_t kv_weight_bytes = (uint64_t)kv_n * sizeof(float);
    if (!rocm_require_tensor_bytes("fused q/kv RMS norm", q, q_bytes) ||
        !rocm_require_tensor_bytes("fused q/kv RMS norm", q_out, q_bytes) ||
        !rocm_require_tensor_bytes("fused q/kv RMS norm", kv, kv_bytes) ||
        !rocm_require_tensor_bytes("fused q/kv RMS norm", kv_out, kv_bytes)) {
        return 0;
    }

    uint8_t *qw = rocm_weight_ptr(model_map, model_size, q_weight_offset, q_weight_bytes);
    uint8_t *kvw = rocm_weight_ptr(model_map, model_size, kv_weight_offset, kv_weight_bytes);
    if (!qw || !kvw) return 0;

    dim3 grid(rows, 2);
    dim3 block(256);
    hipLaunchKernelGGL(ds4_rocm_qkv_rms_norm_rows_f32, grid, block,
                       block.x * sizeof(float), g_stream,
                       (float *)q_out->ptr,
                       (const float *)q->ptr,
                       (const float *)qw,
                       q_n,
                       (float *)kv_out->ptr,
                       (const float *)kv->ptr,
                       (const float *)kvw,
                       kv_n,
                       rows,
                       eps);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || (head_dim & 3u) != 0) return 0;

    uint64_t rows;
    uint64_t elems;
    if (!rocm_mul_u64(n_tok, n_head, &rows) ||
        !rocm_mul_u64(rows, head_dim, &elems) ||
        rows > UINT32_MAX ||
        elems > UINT64_MAX / sizeof(float)) {
        return 0;
    }
    uint64_t bytes = elems * sizeof(float);
    if (!rocm_require_tensor_bytes("head RMS norm", x, bytes)) return 0;

    dim3 grid((uint32_t)rows);
    dim3 block(256);
    hipLaunchKernelGGL(ds4_rocm_head_rms_norm_f32, grid, block,
                       block.x * sizeof(float), g_stream,
                       (float *)x->ptr,
                       (uint32_t)rows,
                       head_dim,
                       eps);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x,
                                          uint32_t n_tok,
                                          uint32_t head_dim,
                                          uint32_t n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    if (((head_dim - n_rot) & 63u) != 0) return 0;

    uint64_t elems;
    uint64_t bytes;
    if (!rocm_mul_u64(n_tok, head_dim, &elems) ||
        !rocm_bytes_for_elems(elems, sizeof(float), &bytes) ||
        !rocm_require_tensor_bytes("FP8 KV quantize", x, bytes)) {
        return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_fp8_kv_quantize_f32, dim3(n_tok), dim3(64), 0, g_stream,
                       (float *)x->ptr,
                       n_tok,
                       head_dim,
                       n_rot);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x,
                               uint32_t n_tok,
                               uint32_t n_head,
                               uint32_t head_dim,
                               uint32_t n_rot,
                               uint32_t pos0,
                               uint32_t n_ctx_orig,
                               bool inverse,
                               float freq_base,
                               float freq_scale,
                               float ext_factor,
                               float attn_factor,
                               float beta_fast,
                               float beta_slow) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 ||
        n_rot == 0 || n_rot > head_dim || (n_rot & 1u) != 0) {
        return 0;
    }

    uint64_t rows;
    uint64_t elems;
    uint64_t bytes;
    uint64_t pairs;
    if (!rocm_mul_u64(n_tok, n_head, &rows) ||
        !rocm_mul_u64(rows, head_dim, &elems) ||
        !rocm_bytes_for_elems(elems, sizeof(float), &bytes) ||
        !rocm_mul_u64(rows, n_rot / 2u, &pairs) ||
        !rocm_require_tensor_bytes("RoPE tail", x, bytes)) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(pairs, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_rope_tail_f32, grid, block, 0, g_stream,
                       (float *)x->ptr,
                       n_tok,
                       n_head,
                       head_dim,
                       n_rot,
                       pos0,
                       n_ctx_orig,
                       inverse ? 1u : 0u,
                       freq_base,
                       freq_scale,
                       ext_factor,
                       attn_factor,
                       beta_fast,
                       beta_slow);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_kv_fp8_store_raw_tensor(ds4_gpu_tensor *kv,
                                      ds4_gpu_tensor *raw_cache,
                                      uint32_t raw_cap,
                                      uint32_t row,
                                      uint32_t head_dim,
                                      uint32_t n_rot) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !raw_cache || raw_cap == 0 || row >= raw_cap ||
        head_dim == 0 || n_rot > head_dim) {
        return 0;
    }
    if (((head_dim - n_rot) & 63u) != 0) return 0;

    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    uint64_t raw_elems;
    uint64_t raw_bytes;
    if (!rocm_mul_u64(raw_cap, head_dim, &raw_elems) ||
        !rocm_bytes_for_elems(raw_elems, sizeof(float), &raw_bytes) ||
        !rocm_require_tensor_bytes("KV FP8/raw kv", kv, kv_bytes) ||
        !rocm_require_tensor_bytes("KV FP8/raw cache", raw_cache, raw_bytes)) {
        return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_kv_fp8_store_raw_f32, dim3(1), dim3(64), 0, g_stream,
                       (float *)kv->ptr,
                       (float *)raw_cache->ptr,
                       row,
                       head_dim,
                       n_rot);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache,
                                  const ds4_gpu_tensor *kv,
                                  uint32_t raw_cap,
                                  uint32_t row,
                                  uint32_t head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || row >= raw_cap || head_dim == 0) return 0;

    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    uint64_t raw_elems;
    uint64_t raw_bytes;
    if (!rocm_mul_u64(raw_cap, head_dim, &raw_elems) ||
        !rocm_bytes_for_elems(raw_elems, sizeof(float), &raw_bytes) ||
        !rocm_require_tensor_bytes("raw KV store src", kv, kv_bytes) ||
        !rocm_require_tensor_bytes("raw KV store dst", raw_cache, raw_bytes)) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(head_dim, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_store_raw_kv_f32, grid, block, 0, g_stream,
                       (float *)raw_cache->ptr,
                       (const float *)kv->ptr,
                       raw_cap,
                       row,
                       1,
                       head_dim);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache,
                                        const ds4_gpu_tensor *kv,
                                        uint32_t raw_cap,
                                        uint32_t pos0,
                                        uint32_t n_tokens,
                                        uint32_t head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0) return 0;

    uint64_t kv_elems;
    uint64_t raw_elems;
    uint64_t kv_bytes;
    uint64_t raw_bytes;
    if (!rocm_mul_u64(n_tokens, head_dim, &kv_elems) ||
        !rocm_mul_u64(raw_cap, head_dim, &raw_elems) ||
        !rocm_bytes_for_elems(kv_elems, sizeof(float), &kv_bytes) ||
        !rocm_bytes_for_elems(raw_elems, sizeof(float), &raw_bytes) ||
        !rocm_require_tensor_bytes("raw KV batch src", kv, kv_bytes) ||
        !rocm_require_tensor_bytes("raw KV batch dst", raw_cache, raw_bytes)) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(kv_elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_store_raw_kv_f32, grid, block, 0, g_stream,
                       (float *)raw_cache->ptr,
                       (const float *)kv->ptr,
                       raw_cap,
                       pos0,
                       n_tokens,
                       head_dim);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    if (head_dim > UINT32_MAX / coff || ratio > UINT32_MAX / coff) return 0;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    if (emit && comp_row == UINT32_MAX) return 0;
    if (head_dim == 512u && ((head_dim - n_rot) & 63u) != 0) return 0;

    const uint64_t elem_ape = ape_type == 1u ? sizeof(uint16_t) : sizeof(float);
    uint64_t state_elems;
    uint64_t state_bytes;
    uint64_t kv_bytes;
    uint64_t ape_elems;
    uint64_t ape_bytes;
    uint64_t norm_bytes;
    if (!rocm_bytes_for_elems(width, sizeof(float), &kv_bytes) ||
        !rocm_mul_u64(state_rows, width, &state_elems) ||
        !rocm_bytes_for_elems(state_elems, sizeof(float), &state_bytes) ||
        !rocm_mul_u64(ratio, width, &ape_elems) ||
        !rocm_bytes_for_elems(ape_elems, elem_ape, &ape_bytes) ||
        !rocm_bytes_for_elems(head_dim, sizeof(float), &norm_bytes)) {
        return 0;
    }

    uint64_t comp_bytes = 0;
    if (emit) {
        uint64_t comp_rows = (uint64_t)comp_row + 1u;
        uint64_t comp_elems;
        if (!rocm_mul_u64(comp_rows, head_dim, &comp_elems) ||
            !rocm_bytes_for_elems(comp_elems, sizeof(float), &comp_bytes)) {
            return 0;
        }
    }

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset) {
        fprintf(stderr, "ds4: ROCm compressor tensor range is outside the mapped model\n");
        return 0;
    }
    if (!rocm_require_tensor_bytes("compressor kv", kv_cur, kv_bytes) ||
        !rocm_require_tensor_bytes("compressor score", sc_cur, kv_bytes) ||
        !rocm_require_tensor_bytes("compressor state kv", state_kv, state_bytes) ||
        !rocm_require_tensor_bytes("compressor state score", state_score, state_bytes) ||
        (emit && !rocm_require_tensor_bytes("compressor cache", comp_cache, comp_bytes))) {
        return 0;
    }

    uint8_t *ape = rocm_weight_ptr(model_map, model_size, ape_offset, ape_bytes);
    if (!ape) return 0;

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(width, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_compressor_store_one_f32, grid, block, 0, g_stream,
                       (const float *)kv_cur->ptr,
                       (const float *)sc_cur->ptr,
                       (const uint8_t *)ape,
                       (float *)state_kv->ptr,
                       (float *)state_score->ptr,
                       width,
                       ratio,
                       pos,
                       ape_type);
    DS4_ROCM_CHECK(hipGetLastError());
    if (!emit) return 1;

    uint8_t *norm = rocm_weight_ptr(model_map, model_size, norm_offset, norm_bytes);
    if (!norm) return 0;
    float *comp = (float *)(comp_cache->ptr + (uint64_t)comp_row * head_dim * sizeof(float));

    if (!rocm_make_1d_launch(head_dim, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_compressor_pool_f32, grid, block, 0, g_stream,
                       comp,
                       (const float *)state_kv->ptr,
                       (const float *)state_score->ptr,
                       head_dim,
                       ratio);
    DS4_ROCM_CHECK(hipGetLastError());

    hipLaunchKernelGGL(ds4_rocm_compressor_rms_norm_f32, dim3(1), dim3(256),
                       256u * sizeof(float), g_stream,
                       comp,
                       (const float *)norm,
                       head_dim,
                       rms_eps);
    DS4_ROCM_CHECK(hipGetLastError());

    if (n_rot != 0) {
        if (!rocm_make_1d_launch(n_rot / 2u, &grid, &block)) return 0;
        hipLaunchKernelGGL(ds4_rocm_rope_tail_f32, grid, block, 0, g_stream,
                           comp,
                           1,
                           1,
                           head_dim,
                           n_rot,
                           pos + 1u - ratio,
                           n_ctx_orig,
                           0,
                           freq_base,
                           freq_scale,
                           ext_factor,
                           attn_factor,
                           beta_fast,
                           beta_slow);
        DS4_ROCM_CHECK(hipGetLastError());
    }

    if (head_dim == 512u) {
        hipLaunchKernelGGL(ds4_rocm_fp8_kv_quantize_f32, dim3(1), dim3(64), 0, g_stream,
                           comp,
                           1,
                           head_dim,
                           n_rot);
        DS4_ROCM_CHECK(hipGetLastError());
    }

    if (ratio == 4u) {
        const uint64_t shift_elems = 4ull * width;
        if (!rocm_make_1d_launch(shift_elems, &grid, &block)) return 0;
        hipLaunchKernelGGL(ds4_rocm_compressor_ratio4_shift_f32, grid, block, 0, g_stream,
                           (float *)state_kv->ptr,
                           (float *)state_score->ptr,
                           width);
        DS4_ROCM_CHECK(hipGetLastError());
    }

    return 1;
}

static int rocm_fill_f32_tensor(ds4_gpu_tensor *tensor, uint64_t elems, float value) {
    if (!tensor || elems > UINT32_MAX) return 0;
    uint64_t bytes;
    if (!rocm_bytes_for_elems(elems, sizeof(float), &bytes) ||
        !rocm_require_tensor_bytes("fill f32", tensor, bytes)) {
        return 0;
    }
    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_fill_f32, grid, block, 0, g_stream,
                       (float *)tensor->ptr,
                       value,
                       (uint32_t)elems);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

static int rocm_clear_compressor_state(ds4_gpu_tensor *state_kv,
                                       ds4_gpu_tensor *state_score,
                                       uint32_t head_dim,
                                       uint32_t ratio) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elems = (uint64_t)state_rows * width;
    return rocm_fill_f32_tensor(state_kv, elems, 0.0f) &&
           rocm_fill_f32_tensor(state_score, elems, -1.0e30f);
}

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    if (head_dim > UINT32_MAX / coff || ratio > UINT32_MAX / coff) return 0;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? sizeof(uint16_t) : sizeof(float);
    uint64_t kv_elems;
    uint64_t state_elems;
    uint64_t ape_elems;
    uint64_t kv_bytes;
    uint64_t state_bytes;
    uint64_t ape_bytes;
    if (!rocm_mul_u64(n_tokens, width, &kv_elems) ||
        !rocm_mul_u64(state_rows, width, &state_elems) ||
        !rocm_mul_u64(ratio, width, &ape_elems) ||
        !rocm_bytes_for_elems(kv_elems, sizeof(float), &kv_bytes) ||
        !rocm_bytes_for_elems(state_elems, sizeof(float), &state_bytes) ||
        !rocm_bytes_for_elems(ape_elems, elem_ape, &ape_bytes)) {
        return 0;
    }
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset) {
        fprintf(stderr, "ds4: ROCm compressor batch APE range is outside the mapped model\n");
        return 0;
    }
    if (!rocm_require_tensor_bytes("compressor batch kv", kv, kv_bytes) ||
        !rocm_require_tensor_bytes("compressor batch score", sc, kv_bytes) ||
        !rocm_require_tensor_bytes("compressor batch state kv", state_kv, state_bytes) ||
        !rocm_require_tensor_bytes("compressor batch state score", state_score, state_bytes)) {
        return 0;
    }
    uint8_t *ape = rocm_weight_ptr(model_map, model_size, ape_offset, ape_bytes);
    if (!ape) return 0;

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(kv_elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_compressor_store_batch_f32, grid, block, 0, g_stream,
                       (const float *)kv->ptr,
                       (const float *)sc->ptr,
                       (const uint8_t *)ape,
                       (float *)state_kv->ptr,
                       (float *)state_score->ptr,
                       width,
                       ratio,
                       pos0,
                       n_tokens,
                       ape_type);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

static int rocm_compressor_prefill_loop(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        bool                    clear_state,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    (void)quantize_fp8;
    if (!comp_cache || !state_kv || !state_score || !kv || !sc ||
        head_dim == 0 || ratio == 0 || n_tokens == 0) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    if (head_dim > UINT32_MAX / coff) return 0;
    const uint32_t width = coff * head_dim;
    const uint64_t row_bytes = (uint64_t)width * sizeof(float);
    uint64_t kv_bytes;
    if (!rocm_bytes_for_elems((uint64_t)n_tokens * width, sizeof(float), &kv_bytes) ||
        !rocm_require_tensor_bytes("compressor prefill kv", kv, kv_bytes) ||
        !rocm_require_tensor_bytes("compressor prefill score", sc, kv_bytes)) {
        return 0;
    }
    if (clear_state && !rocm_clear_compressor_state(state_kv, state_score, head_dim, ratio)) {
        return 0;
    }

    uint32_t emitted = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *kv_view = ds4_gpu_tensor_view(kv, (uint64_t)t * row_bytes, row_bytes);
        ds4_gpu_tensor *sc_view = ds4_gpu_tensor_view(sc, (uint64_t)t * row_bytes, row_bytes);
        if (!kv_view || !sc_view) {
            ds4_gpu_tensor_free(kv_view);
            ds4_gpu_tensor_free(sc_view);
            return 0;
        }
        const uint32_t pos = pos0 + t;
        const uint32_t comp_row = emitted;
        const int ok = ds4_gpu_compressor_update_tensor(kv_view,
                                                          sc_view,
                                                          state_kv,
                                                          state_score,
                                                          comp_cache,
                                                          model_map,
                                                          model_size,
                                                          ape_offset,
                                                          ape_type,
                                                          norm_offset,
                                                          norm_type,
                                                          head_dim,
                                                          ratio,
                                                          pos,
                                                          comp_row,
                                                          n_rot,
                                                          n_ctx_orig,
                                                          freq_base,
                                                          freq_scale,
                                                          ext_factor,
                                                          attn_factor,
                                                          beta_fast,
                                                          beta_slow,
                                                          rms_eps);
        ds4_gpu_tensor_free(kv_view);
        ds4_gpu_tensor_free(sc_view);
        if (!ok) return 0;
        if (((pos + 1u) % ratio) == 0u) emitted++;
    }
    return 1;
}

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    return rocm_compressor_prefill_loop(comp_cache, state_kv, state_score,
                                        kv, sc, model_map, model_size,
                                        ape_offset, ape_type, norm_offset,
                                        norm_type, head_dim, ratio, pos0,
                                        n_tokens, n_rot, n_ctx_orig,
                                        quantize_fp8, true, freq_base,
                                        freq_scale, ext_factor, attn_factor,
                                        beta_fast, beta_slow, rms_eps);
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if ((pos0 & 3u) != 0u || (n_tokens & 3u) != 0u) return 0;
    return rocm_compressor_prefill_loop(comp_cache, state_kv, state_score,
                                        kv, sc, model_map, model_size,
                                        ape_offset, ape_type, norm_offset,
                                        norm_type, head_dim, 4, pos0,
                                        n_tokens, n_rot, n_ctx_orig,
                                        quantize_fp8, false, freq_base,
                                        freq_scale, ext_factor, attn_factor,
                                        beta_fast, beta_slow, rms_eps);
}

int ds4_gpu_compressor_prefill_state_ratio4_tensor(ds4_gpu_tensor *state_kv,
                                                     ds4_gpu_tensor *state_score,
                                                     const ds4_gpu_tensor *kv_tail,
                                                     const ds4_gpu_tensor *sc_tail,
                                                     const void *model_map,
                                                     uint64_t model_size,
                                                     uint64_t ape_offset,
                                                     uint32_t ape_type,
                                                     uint32_t head_dim,
                                                     uint32_t pos0) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? sizeof(uint16_t) : sizeof(float);
    uint64_t tail_elems;
    uint64_t state_elems;
    uint64_t ape_elems;
    uint64_t tail_bytes;
    uint64_t state_bytes;
    uint64_t ape_bytes;
    if (!rocm_mul_u64(ratio, width, &tail_elems) ||
        !rocm_mul_u64(state_rows, width, &state_elems) ||
        !rocm_mul_u64(ratio, width, &ape_elems) ||
        !rocm_bytes_for_elems(tail_elems, sizeof(float), &tail_bytes) ||
        !rocm_bytes_for_elems(state_elems, sizeof(float), &state_bytes) ||
        !rocm_bytes_for_elems(ape_elems, elem_ape, &ape_bytes)) {
        return 0;
    }
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset) return 0;
    if (!rocm_require_tensor_bytes("compressor ratio4 state kv tail", kv_tail, tail_bytes) ||
        !rocm_require_tensor_bytes("compressor ratio4 state score tail", sc_tail, tail_bytes) ||
        !rocm_require_tensor_bytes("compressor ratio4 state kv", state_kv, state_bytes) ||
        !rocm_require_tensor_bytes("compressor ratio4 state score", state_score, state_bytes)) {
        return 0;
    }
    if (!rocm_clear_compressor_state(state_kv, state_score, head_dim, ratio)) return 0;
    uint8_t *ape = rocm_weight_ptr(model_map, model_size, ape_offset, ape_bytes);
    if (!ape) return 0;
    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(tail_elems, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_compressor_store_ratio4_state_f32, grid, block, 0, g_stream,
                       (const float *)kv_tail->ptr,
                       (const float *)sc_tail->ptr,
                       (const uint8_t *)ape,
                       (float *)state_kv->ptr,
                       (float *)state_score->ptr,
                       width,
                       pos0,
                       ape_type);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_attention_decode_heads_tensor(ds4_gpu_tensor *heads,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t sinks_offset,
                                            const ds4_gpu_tensor *q,
                                            const ds4_gpu_tensor *raw_kv,
                                            uint32_t n_raw,
                                            uint32_t raw_cap,
                                            uint32_t raw_start,
                                            const ds4_gpu_tensor *comp_kv,
                                            uint32_t n_comp,
                                            const ds4_gpu_tensor *comp_mask,
                                            uint32_t use_mask,
                                            uint32_t n_head,
                                            uint32_t head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_head == 0 || head_dim != 512u ||
        (n_comp != 0 && !comp_kv) ||
        (use_mask != 0 && !comp_mask)) {
        return 0;
    }

    uint64_t q_elems;
    uint64_t raw_elems;
    uint64_t comp_elems;
    uint64_t q_bytes;
    uint64_t raw_bytes;
    uint64_t comp_bytes;
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    const uint64_t mask_bytes = use_mask ? (uint64_t)n_comp * sizeof(float) : 0u;
    if (!rocm_mul_u64(n_head, head_dim, &q_elems) ||
        !rocm_mul_u64(raw_cap, head_dim, &raw_elems) ||
        !rocm_mul_u64(n_comp, head_dim, &comp_elems) ||
        !rocm_bytes_for_elems(q_elems, sizeof(float), &q_bytes) ||
        !rocm_bytes_for_elems(raw_elems, sizeof(float), &raw_bytes) ||
        !rocm_bytes_for_elems(comp_elems, sizeof(float), &comp_bytes)) {
        return 0;
    }
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
        fprintf(stderr, "ds4: ROCm attention sinks range is outside the mapped model\n");
        return 0;
    }
    if (!rocm_require_tensor_bytes("attention heads q", q, q_bytes) ||
        !rocm_require_tensor_bytes("attention heads raw", raw_kv, raw_bytes) ||
        !rocm_require_tensor_bytes("attention heads out", heads, q_bytes) ||
        (n_comp != 0 && !rocm_require_tensor_bytes("attention heads comp", comp_kv, comp_bytes)) ||
        (use_mask != 0 && !rocm_require_tensor_bytes("attention heads mask", comp_mask, mask_bytes))) {
        return 0;
    }

    uint8_t *sinks = rocm_weight_ptr(model_map, model_size, sinks_offset, sink_bytes);
    if (!sinks) return 0;

    hipLaunchKernelGGL(ds4_rocm_attention_decode_heads_f32,
                       dim3(n_head), dim3(256), 256u * sizeof(float), g_stream,
                       (float *)heads->ptr,
                       (const float *)sinks,
                       (const float *)q->ptr,
                       (const float *)raw_kv->ptr,
                       n_raw,
                       raw_cap,
                       raw_start,
                       n_comp ? (const float *)comp_kv->ptr : nullptr,
                       n_comp,
                       use_mask ? (const float *)comp_mask->ptr : nullptr,
                       use_mask ? 1u : 0u,
                       n_head);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

static int rocm_attention_mixed_batch_heads(ds4_gpu_tensor *heads,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t sinks_offset,
                                            const ds4_gpu_tensor *q,
                                            const ds4_gpu_tensor *raw_kv,
                                            const ds4_gpu_tensor *comp_kv,
                                            const ds4_gpu_tensor *comp_mask,
                                            uint32_t use_comp_mask,
                                            uint32_t n_tokens,
                                            uint32_t pos0,
                                            uint32_t n_raw,
                                            uint32_t raw_cap,
                                            uint32_t raw_start,
                                            uint32_t n_comp,
                                            uint32_t window,
                                            uint32_t ratio,
                                            uint32_t n_head,
                                            uint32_t head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_head == 0 || head_dim != 512u || (n_comp != 0 && !comp_kv) ||
        (use_comp_mask != 0 && !comp_mask)) {
        return 0;
    }

    uint64_t q_rows;
    uint64_t q_elems;
    uint64_t raw_elems;
    uint64_t comp_elems;
    uint64_t mask_elems;
    uint64_t q_bytes;
    uint64_t raw_bytes;
    uint64_t comp_bytes;
    uint64_t mask_bytes;
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (!rocm_mul_u64(n_tokens, n_head, &q_rows) ||
        !rocm_mul_u64(q_rows, head_dim, &q_elems) ||
        !rocm_mul_u64(raw_cap, head_dim, &raw_elems) ||
        !rocm_mul_u64(n_comp, head_dim, &comp_elems) ||
        !rocm_mul_u64(n_tokens, n_comp, &mask_elems) ||
        !rocm_bytes_for_elems(q_elems, sizeof(float), &q_bytes) ||
        !rocm_bytes_for_elems(raw_elems, sizeof(float), &raw_bytes) ||
        !rocm_bytes_for_elems(comp_elems, sizeof(float), &comp_bytes) ||
        !rocm_bytes_for_elems(mask_elems, sizeof(float), &mask_bytes)) {
        return 0;
    }
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
        fprintf(stderr, "ds4: ROCm batch attention sinks range is outside the mapped model\n");
        return 0;
    }
    if (!rocm_require_tensor_bytes("batch attention q", q, q_bytes) ||
        !rocm_require_tensor_bytes("batch attention raw", raw_kv, raw_bytes) ||
        !rocm_require_tensor_bytes("batch attention out", heads, q_bytes) ||
        (n_comp != 0 && !rocm_require_tensor_bytes("batch attention comp", comp_kv, comp_bytes)) ||
        (use_comp_mask != 0 && !rocm_require_tensor_bytes("batch attention mask", comp_mask, mask_bytes))) {
        return 0;
    }

    uint8_t *sinks = rocm_weight_ptr(model_map, model_size, sinks_offset, sink_bytes);
    if (!sinks) return 0;

    hipLaunchKernelGGL(ds4_rocm_attention_mixed_batch_heads_f32,
                       dim3(n_tokens, n_head), dim3(256),
                       256u * sizeof(float), g_stream,
                       (float *)heads->ptr,
                       (const float *)sinks,
                       (const float *)q->ptr,
                       (const float *)raw_kv->ptr,
                       n_comp ? (const float *)comp_kv->ptr : nullptr,
                       use_comp_mask ? (const float *)comp_mask->ptr : nullptr,
                       use_comp_mask ? 1u : 0u,
                       n_tokens,
                       pos0,
                       n_raw,
                       raw_cap,
                       raw_start,
                       n_comp,
                       window,
                       ratio,
                       n_head);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads,
                                                 const void *model_map,
                                                 uint64_t model_size,
                                                 uint64_t sinks_offset,
                                                 const ds4_gpu_tensor *q,
                                                 const ds4_gpu_tensor *raw_kv,
                                                 uint32_t n_tokens,
                                                 uint32_t window,
                                                 uint32_t n_head,
                                                 uint32_t head_dim) {
    return rocm_attention_mixed_batch_heads(heads, model_map, model_size,
                                            sinks_offset, q, raw_kv,
                                            nullptr, nullptr, 0, n_tokens,
                                            0, n_tokens, n_tokens, 0,
                                            0, window, 1, n_head, head_dim);
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t window,
        uint32_t n_head, uint32_t head_dim) {
    return rocm_attention_mixed_batch_heads(heads, model_map, model_size,
                                            sinks_offset, q, raw_kv,
                                            nullptr, nullptr, 0, n_tokens,
                                            pos0, n_raw, raw_cap, raw_start,
                                            0, window, 1, n_head, head_dim);
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    return rocm_attention_mixed_batch_heads(heads, model_map, model_size,
                                            sinks_offset, q, raw_kv,
                                            comp_kv, comp_mask, use_comp_mask,
                                            n_tokens, pos0, n_raw, raw_cap,
                                            raw_start, n_comp, window, ratio,
                                            n_head, head_dim);
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads,
        const void *model_map,
        uint64_t model_size,
        uint64_t sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!heads || !model_map || !q || !raw_kv || !comp_kv || !topk ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 || top_k > n_comp ||
        ratio == 0 || n_head == 0 || head_dim != 512u) {
        return 0;
    }

    uint64_t q_rows;
    uint64_t q_elems;
    uint64_t raw_elems;
    uint64_t comp_elems;
    uint64_t topk_elems;
    uint64_t q_bytes;
    uint64_t raw_bytes;
    uint64_t comp_bytes;
    uint64_t topk_bytes;
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (!rocm_mul_u64(n_tokens, n_head, &q_rows) ||
        !rocm_mul_u64(q_rows, head_dim, &q_elems) ||
        !rocm_mul_u64(raw_cap, head_dim, &raw_elems) ||
        !rocm_mul_u64(n_comp, head_dim, &comp_elems) ||
        !rocm_mul_u64(n_tokens, top_k, &topk_elems) ||
        !rocm_bytes_for_elems(q_elems, sizeof(float), &q_bytes) ||
        !rocm_bytes_for_elems(raw_elems, sizeof(float), &raw_bytes) ||
        !rocm_bytes_for_elems(comp_elems, sizeof(float), &comp_bytes) ||
        !rocm_bytes_for_elems(topk_elems, sizeof(int32_t), &topk_bytes)) {
        return 0;
    }
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset) {
        fprintf(stderr, "ds4: ROCm indexed attention sinks range is outside the mapped model\n");
        return 0;
    }
    if (!rocm_require_tensor_bytes("indexed attention q", q, q_bytes) ||
        !rocm_require_tensor_bytes("indexed attention raw", raw_kv, raw_bytes) ||
        !rocm_require_tensor_bytes("indexed attention comp", comp_kv, comp_bytes) ||
        !rocm_require_tensor_bytes("indexed attention topk", topk, topk_bytes) ||
        !rocm_require_tensor_bytes("indexed attention out", heads, q_bytes)) {
        return 0;
    }

    uint8_t *sinks = rocm_weight_ptr(model_map, model_size, sinks_offset, sink_bytes);
    if (!sinks) return 0;

    hipLaunchKernelGGL(ds4_rocm_attention_indexed_mixed_batch_heads_f32,
                       dim3(n_tokens, n_head), dim3(256), 256u * sizeof(float), g_stream,
                       (float *)heads->ptr,
                       (const float *)sinks,
                       (const float *)q->ptr,
                       (const float *)raw_kv->ptr,
                       (const float *)comp_kv->ptr,
                       (const int32_t *)topk->ptr,
                       n_tokens,
                       pos0,
                       n_raw,
                       raw_cap,
                       raw_start,
                       n_comp,
                       top_k,
                       window,
                       ratio,
                       n_head);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(ds4_gpu_tensor *heads,
                                                          const void *model_map,
                                                          uint64_t model_size,
                                                          uint64_t sinks_offset,
                                                          const ds4_gpu_tensor *q,
                                                          const ds4_gpu_tensor *raw_kv,
                                                          const ds4_gpu_tensor *comp_kv,
                                                          uint32_t n_tokens,
                                                          uint32_t n_comp,
                                                          uint32_t window,
                                                          uint32_t ratio,
                                                          uint32_t n_head,
                                                          uint32_t head_dim) {
    return rocm_attention_mixed_batch_heads(heads, model_map, model_size,
                                            sinks_offset, q, raw_kv,
                                            comp_kv, nullptr, 0, n_tokens,
                                            0, n_tokens, n_tokens, 0,
                                            n_comp, window, ratio,
                                            n_head, head_dim);
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(ds4_gpu_tensor *heads,
                                                          const void *model_map,
                                                          uint64_t model_size,
                                                          uint64_t sinks_offset,
                                                          const ds4_gpu_tensor *q,
                                                          const ds4_gpu_tensor *raw_kv,
                                                          const ds4_gpu_tensor *comp_kv,
                                                          const ds4_gpu_tensor *comp_mask,
                                                          uint32_t n_tokens,
                                                          uint32_t n_comp,
                                                          uint32_t window,
                                                          uint32_t ratio,
                                                          uint32_t n_head,
                                                          uint32_t head_dim) {
    return rocm_attention_mixed_batch_heads(heads, model_map, model_size,
                                            sinks_offset, q, raw_kv,
                                            comp_kv, comp_mask, 1,
                                            n_tokens, 0, n_tokens,
                                            n_tokens, 0, n_comp, window,
                                            ratio, n_head, head_dim);
}

int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor *out,
                                               ds4_gpu_tensor *low,
                                               ds4_gpu_tensor *group_tmp,
                                               ds4_gpu_tensor *low_tmp,
                                               const void *model_map,
                                               uint64_t model_size,
                                               uint64_t out_a_offset,
                                               uint64_t out_b_offset,
                                               uint64_t group_dim,
                                               uint64_t rank,
                                               uint32_t n_groups,
                                               uint64_t out_dim,
                                               const ds4_gpu_tensor *heads,
                                               uint32_t n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !low || !heads || group_dim == 0 || rank == 0 ||
        n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    if (group_dim > UINT32_MAX || rank > UINT32_MAX || out_dim > UINT32_MAX) return 0;
    if ((group_dim & 31u) != 0) return 0;

    uint64_t low_dim;
    uint64_t heads_per_token;
    uint64_t heads_elems;
    uint64_t low_elems;
    uint64_t out_elems;
    if (!rocm_mul_u64(rank, n_groups, &low_dim) ||
        !rocm_mul_u64(group_dim, n_groups, &heads_per_token) ||
        !rocm_mul_u64(heads_per_token, n_tokens, &heads_elems) ||
        !rocm_mul_u64(low_dim, n_tokens, &low_elems) ||
        !rocm_mul_u64(out_dim, n_tokens, &out_elems)) {
        return 0;
    }
    if (low_dim == 0 || low_dim > UINT32_MAX || (low_dim & 31u) != 0) return 0;

    uint64_t heads_bytes;
    uint64_t low_bytes;
    uint64_t out_bytes;
    uint64_t out_a_bytes;
    uint64_t out_b_bytes;
    if (!rocm_bytes_for_elems(heads_elems, sizeof(float), &heads_bytes) ||
        !rocm_bytes_for_elems(low_elems, sizeof(float), &low_bytes) ||
        !rocm_bytes_for_elems(out_elems, sizeof(float), &out_bytes) ||
        !rocm_q8_0_weight_bytes(group_dim, low_dim, &out_a_bytes) ||
        !rocm_q8_0_weight_bytes(low_dim, out_dim, &out_b_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("attention output heads", heads, heads_bytes) ||
        !rocm_require_tensor_bytes("attention output low", low, low_bytes) ||
        !rocm_require_tensor_bytes("attention output out", out, out_bytes)) {
        return 0;
    }

    uint8_t *out_a = rocm_weight_ptr(model_map, model_size, out_a_offset, out_a_bytes);
    uint8_t *out_b = rocm_weight_ptr(model_map, model_size, out_b_offset, out_b_bytes);
    if (!out_a || !out_b) return 0;

    dim3 block(256);
    dim3 low_grid((uint32_t)low_dim, n_tokens);
    hipLaunchKernelGGL(ds4_rocm_mul_mm_q8_0_grouped_f32, low_grid, block, 0, g_stream,
                       (const uint8_t *)out_a,
                       (const float *)heads->ptr,
                       (float *)low->ptr,
                       (uint32_t)group_dim,
                       (uint32_t)rank,
                       n_groups,
                       n_tokens);
    DS4_ROCM_CHECK(hipGetLastError());

    dim3 out_grid((uint32_t)out_dim, n_tokens);
    hipLaunchKernelGGL(ds4_rocm_mul_mm_q8_0_f32, out_grid, block, 0, g_stream,
                       (const uint8_t *)out_b,
                       (const float *)low->ptr,
                       (float *)out->ptr,
                       (uint32_t)low_dim,
                       (uint32_t)out_dim,
                       n_tokens);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor *low,
                                             const void *model_map,
                                             uint64_t model_size,
                                             uint64_t out_a_offset,
                                             uint64_t group_dim,
                                             uint64_t rank,
                                             uint32_t n_groups,
                                             const ds4_gpu_tensor *heads) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!low || !heads || group_dim == 0 || rank == 0 || n_groups == 0) return 0;
    if (group_dim > UINT32_MAX || rank > UINT32_MAX || (group_dim & 31u) != 0) return 0;

    uint64_t low_dim;
    uint64_t heads_elems;
    if (!rocm_mul_u64(rank, n_groups, &low_dim) ||
        !rocm_mul_u64(group_dim, n_groups, &heads_elems)) {
        return 0;
    }
    if (low_dim == 0 || low_dim > UINT32_MAX) return 0;

    uint64_t low_bytes;
    uint64_t heads_bytes;
    uint64_t weight_bytes;
    if (!rocm_bytes_for_elems(low_dim, sizeof(float), &low_bytes) ||
        !rocm_bytes_for_elems(heads_elems, sizeof(float), &heads_bytes) ||
        !rocm_q8_0_weight_bytes(group_dim, low_dim, &weight_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("attention low heads", heads, heads_bytes) ||
        !rocm_require_tensor_bytes("attention low out", low, low_bytes)) {
        return 0;
    }

    uint8_t *w = rocm_weight_ptr(model_map, model_size, out_a_offset, weight_bytes);
    if (!w) return 0;

    dim3 block(256);
    dim3 grid((uint32_t)low_dim, 1);
    hipLaunchKernelGGL(ds4_rocm_mul_mm_q8_0_grouped_f32, grid, block, 0, g_stream,
                       (const uint8_t *)w,
                       (const float *)heads->ptr,
                       (float *)low->ptr,
                       (uint32_t)group_dim,
                       (uint32_t)rank,
                       n_groups,
                       1);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || n == 0) return 0;
    if (fabsf(clamp) > 1.0e-12f || fabsf(weight - 1.0f) > 1.0e-12f) {
        fprintf(stderr, "ds4: ROCm SwiGLU kernel does not support clamp/weight\n");
        return 0;
    }

    uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!rocm_require_tensor_bytes("SwiGLU", gate, bytes) ||
        !rocm_require_tensor_bytes("SwiGLU", up, bytes) ||
        !rocm_require_tensor_bytes("SwiGLU", out, bytes)) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(n, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_swiglu_f32, grid, block, 0, g_stream,
                       (float *)out->ptr,
                       (const float *)gate->ptr,
                       (const float *)up->ptr,
                       n);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !a || !b || n == 0) return 0;

    uint64_t bytes = (uint64_t)n * sizeof(float);
    if (!rocm_require_tensor_bytes("tensor add", a, bytes) ||
        !rocm_require_tensor_bytes("tensor add", b, bytes) ||
        !rocm_require_tensor_bytes("tensor add", out, bytes)) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(n, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_add_f32, grid, block, 0, g_stream,
                       (float *)out->ptr,
                       (const float *)a->ptr,
                       (const float *)b->ptr,
                       n);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected,
                                   ds4_gpu_tensor *weights,
                                   ds4_gpu_tensor *probs,
                                   const void *model_map,
                                   uint64_t model_size,
                                   uint64_t bias_offset,
                                   uint64_t hash_offset,
                                   uint32_t hash_rows,
                                   uint32_t token,
                                   uint32_t n_expert_groups,
                                   uint32_t n_group_used,
                                   bool has_bias,
                                   bool hash_mode,
                                   const ds4_gpu_tensor *logits) {
    (void)n_expert_groups;
    (void)n_group_used;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits) return 0;
    if (hash_mode && hash_rows == 0) return 0;

    const uint64_t expert_bytes = 256u * sizeof(float);
    const uint64_t selected_bytes = 6u * sizeof(int32_t);
    const uint64_t weights_bytes = 6u * sizeof(float);
    if (!rocm_require_tensor_bytes("router logits", logits, expert_bytes) ||
        !rocm_require_tensor_bytes("router probs", probs, expert_bytes) ||
        !rocm_require_tensor_bytes("router selected", selected, selected_bytes) ||
        !rocm_require_tensor_bytes("router weights", weights, weights_bytes)) {
        return 0;
    }

    const float *bias = nullptr;
    const int32_t *hash = nullptr;
    if (has_bias) {
        bias = (const float *)rocm_weight_ptr(model_map, model_size, bias_offset, expert_bytes);
        if (!bias) return 0;
    }
    if (hash_mode) {
        uint64_t hash_elems;
        uint64_t hash_bytes;
        if (!rocm_mul_u64(hash_rows, 6u, &hash_elems) ||
            !rocm_bytes_for_elems(hash_elems, sizeof(int32_t), &hash_bytes)) {
            return 0;
        }
        hash = (const int32_t *)rocm_weight_ptr(model_map, model_size, hash_offset, hash_bytes);
        if (!hash) return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_router_select_one, dim3(1), dim3(256), 0, g_stream,
                       (float *)probs->ptr,
                       (int32_t *)selected->ptr,
                       (float *)weights->ptr,
                       (const float *)logits->ptr,
                       bias,
                       hash,
                       hash_rows,
                       token,
                       has_bias ? 1u : 0u,
                       hash_mode ? 1u : 0u);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected,
                                         ds4_gpu_tensor *weights,
                                         ds4_gpu_tensor *probs,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t bias_offset,
                                         uint64_t hash_offset,
                                         uint32_t hash_rows,
                                         uint32_t n_expert_groups,
                                         uint32_t n_group_used,
                                         bool has_bias,
                                         bool hash_mode,
                                         const ds4_gpu_tensor *logits,
                                         const ds4_gpu_tensor *tokens,
                                         uint32_t n_tokens) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!selected || !weights || !probs || !logits || !tokens ||
        !model_map || n_tokens == 0) {
        return 0;
    }
    if (n_expert_groups > 1u || n_group_used > 0u) {
        fprintf(stderr, "ds4: ROCm router group gating is not part of this DeepSeek V4 Flash path\n");
        return 0;
    }
    if (hash_mode && hash_rows == 0) return 0;

    uint64_t expert_elems;
    uint64_t selected_elems;
    uint64_t expert_bytes;
    uint64_t selected_bytes;
    uint64_t weights_bytes;
    uint64_t token_bytes;
    if (!rocm_mul_u64(n_tokens, 256u, &expert_elems) ||
        !rocm_mul_u64(n_tokens, 6u, &selected_elems) ||
        !rocm_bytes_for_elems(expert_elems, sizeof(float), &expert_bytes) ||
        !rocm_bytes_for_elems(selected_elems, sizeof(int32_t), &selected_bytes) ||
        !rocm_bytes_for_elems(selected_elems, sizeof(float), &weights_bytes) ||
        !rocm_bytes_for_elems(n_tokens, sizeof(int32_t), &token_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("router batch logits", logits, expert_bytes) ||
        !rocm_require_tensor_bytes("router batch probs", probs, expert_bytes) ||
        !rocm_require_tensor_bytes("router batch selected", selected, selected_bytes) ||
        !rocm_require_tensor_bytes("router batch weights", weights, weights_bytes) ||
        !rocm_require_tensor_bytes("router batch tokens", tokens, token_bytes)) {
        return 0;
    }

    const float *bias = nullptr;
    const int32_t *hash = nullptr;
    if (has_bias && !hash_mode) {
        bias = (const float *)rocm_weight_ptr(model_map, model_size, bias_offset,
                                             256u * sizeof(float));
        if (!bias) return 0;
    }
    if (hash_mode) {
        uint64_t hash_elems;
        uint64_t hash_bytes;
        if (!rocm_mul_u64(hash_rows, 6u, &hash_elems) ||
            !rocm_bytes_for_elems(hash_elems, sizeof(int32_t), &hash_bytes)) {
            return 0;
        }
        hash = (const int32_t *)rocm_weight_ptr(model_map, model_size, hash_offset, hash_bytes);
        if (!hash) return 0;
    }

    hipLaunchKernelGGL(ds4_rocm_router_select_batch, dim3(n_tokens), dim3(256), 0, g_stream,
                       (float *)probs->ptr,
                       (int32_t *)selected->ptr,
                       (float *)weights->ptr,
                       (const float *)logits->ptr,
                       (const int32_t *)tokens->ptr,
                       bias,
                       hash,
                       hash_rows,
                       n_tokens,
                       (has_bias && !hash_mode) ? 1u : 0u,
                       hash_mode ? 1u : 0u);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out,
                                    ds4_gpu_tensor *gate,
                                    ds4_gpu_tensor *up,
                                    ds4_gpu_tensor *mid,
                                    ds4_gpu_tensor *experts,
                                    const void *model_map,
                                    uint64_t model_size,
                                    uint64_t gate_offset,
                                    uint64_t up_offset,
                                    uint64_t down_offset,
                                    uint32_t gate_type,
                                    uint32_t down_type,
                                    uint64_t gate_expert_bytes,
                                    uint64_t gate_row_bytes,
                                    uint64_t down_expert_bytes,
                                    uint64_t down_row_bytes,
                                    uint32_t expert_in_dim,
                                    uint32_t expert_mid_dim,
                                    uint32_t out_dim,
                                    const ds4_gpu_tensor *selected,
                                    const ds4_gpu_tensor *weights,
                                    uint32_t n_expert,
                                    float clamp,
                                    const ds4_gpu_tensor *x,
                                    int layer_id) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_expert == 0 || n_expert > 6 || expert_in_dim == 0 ||
        expert_mid_dim == 0 || out_dim == 0 ||
        (expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) {
        return 0;
    }
    if (layer_id < 0 || layer_id >= ROCM_MAX_MOE_LAYERS) return 0;

    const uint64_t gate_block_bytes = ds4_rocm_routed_block_bytes(gate_type);
    const uint64_t down_block_bytes = ds4_rocm_routed_block_bytes(down_type);
    if (gate_block_bytes == 0 || down_block_bytes == 0) return 0;
    const uint64_t expected_gate_row = ((uint64_t)expert_in_dim / 256u) * gate_block_bytes;
    const uint64_t expected_down_row = ((uint64_t)expert_mid_dim / 256u) * down_block_bytes;
    if (gate_row_bytes < expected_gate_row || down_row_bytes < expected_down_row ||
        gate_expert_bytes < (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes < (uint64_t)out_dim * down_row_bytes) {
        return 0;
    }

    const uint64_t x_bytes = (uint64_t)expert_in_dim * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_expert * expert_mid_dim;
    const uint64_t mid_bytes = mid_elems * sizeof(float);
    const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_bytes = (uint64_t)n_expert * out_dim * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_expert * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t gate_tensor_bytes = 256ull * gate_expert_bytes;
    const uint64_t down_tensor_bytes = 256ull * down_expert_bytes;
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: ROCm routed MoE tensor range is outside the mapped model\n");
        return 0;
    }
    if (!rocm_require_tensor_bytes("routed MoE x", x, x_bytes) ||
        !rocm_require_tensor_bytes("routed MoE gate", gate, mid_bytes) ||
        !rocm_require_tensor_bytes("routed MoE up", up, mid_bytes) ||
        !rocm_require_tensor_bytes("routed MoE mid", mid, mid_bytes) ||
        !rocm_require_tensor_bytes("routed MoE out", out, out_bytes) ||
        !rocm_require_tensor_bytes("routed MoE selected", selected, selected_bytes) ||
        !rocm_require_tensor_bytes("routed MoE weights", weights, weights_bytes) ||
        (n_expert > 1 && (!experts || !rocm_require_tensor_bytes("routed MoE experts", experts, expert_bytes)))) {
        return 0;
    }

    // Read selected and weights to host (small, needed to index the table)
    int32_t selected_host[6];
    float weights_host[6];
    DS4_ROCM_CHECK(hipMemcpyAsync(selected_host, selected->ptr, selected_bytes, hipMemcpyDeviceToHost, g_stream));
    DS4_ROCM_CHECK(hipMemcpyAsync(weights_host, weights->ptr, weights_bytes, hipMemcpyDeviceToHost, g_stream));
    DS4_ROCM_CHECK(hipStreamSynchronize(g_stream));
    ds4_gpu_profile_mark("moe_select_sync");
    for (uint32_t i = 0; i < n_expert; i++) {
        if (selected_host[i] < 0 || selected_host[i] >= 256) return 0;
    }

    // ── Resolve per-expert weight pointers ──────────────────────────
    uint8_t *gate_dev[6], *up_dev[6], *down_dev[6];
    float *weights_dev;
    uint8_t **gate_ptrs_dev, **up_ptrs_dev, **down_ptrs_dev;
    const uint64_t gate_bytes = (uint64_t)expert_mid_dim * gate_row_bytes;
    const uint64_t down_bytes = (uint64_t)out_dim * down_row_bytes;

    // Check whether all 6 experts are already in the device table
    bool all_have_ptrs = rocm_moe_selected_entries_ready(layer_id, selected_host, n_expert);

    // Reset staging pos — safe because the sync above drained all prior DMAs.
    g_staging_pos = 0;
    if (!all_have_ptrs && rocm_moe_prewarm_enabled()) {
        ds4_gpu_profile_mark("moe_prewarm_start");
        if (!rocm_moe_prewarm_layer(layer_id, model_map, model_size,
                                    gate_offset, up_offset, down_offset,
                                    gate_expert_bytes, down_expert_bytes,
                                    gate_bytes, down_bytes)) {
            return 0;
        }
        ds4_gpu_profile_mark("moe_prewarm_done");
        all_have_ptrs = rocm_moe_selected_entries_ready(layer_id, selected_host, n_expert);
    }

    if (all_have_ptrs) {
        // ── Fast path: use device table, no hipMalloc  ──────────────
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            uint32_t e = (uint32_t)selected_host[slot];
            uint64_t idx_g = rocm_expert_table_idx(layer_id, e, ROCM_EXPERT_GATE);
            uint64_t idx_u = rocm_expert_table_idx(layer_id, e, ROCM_EXPERT_UP);
            uint64_t idx_d = rocm_expert_table_idx(layer_id, e, ROCM_EXPERT_DOWN);
            gate_dev[slot]  = g_expert_table_cpu[idx_g];
            up_dev[slot]    = g_expert_table_cpu[idx_u];
            down_dev[slot]  = g_expert_table_cpu[idx_d];
            rocm_expert_table_touch(idx_g);
            rocm_expert_table_touch(idx_u);
            rocm_expert_table_touch(idx_d);
        }
        weights_dev = g_weights_dev;
        gate_ptrs_dev = g_gate_ptrs_dev;
        up_ptrs_dev   = g_up_ptrs_dev;
        down_ptrs_dev = g_down_ptrs_dev;

        DS4_ROCM_CHECK(hipMemcpyAsync(weights_dev, weights_host, weights_bytes, hipMemcpyHostToDevice, g_stream));
        DS4_ROCM_CHECK(hipMemcpyAsync(gate_ptrs_dev, gate_dev, (size_t)n_expert * sizeof(uint8_t *), hipMemcpyHostToDevice, g_stream));
        DS4_ROCM_CHECK(hipMemcpyAsync(up_ptrs_dev,   up_dev,   (size_t)n_expert * sizeof(uint8_t *), hipMemcpyHostToDevice, g_stream));
        DS4_ROCM_CHECK(hipMemcpyAsync(down_ptrs_dev, down_dev, (size_t)n_expert * sizeof(uint8_t *), hipMemcpyHostToDevice, g_stream));
    } else {
        // ── Slow path: cache missing experts and populate table  ────
        // Build pointer arrays and update table for THIS batch
        for (uint32_t slot = 0; slot < n_expert; slot++) {
            uint32_t e = (uint32_t)selected_host[slot];
            uint32_t gate_cache_idx = 0, up_cache_idx = 0, down_cache_idx = 0;
            uint64_t gate_cache_gen = 0, up_cache_gen = 0, down_cache_gen = 0;
            gate_dev[slot] = rocm_weight_ptr_info(model_map, model_size,
                                                  gate_offset + (uint64_t)e * gate_expert_bytes,
                                                  gate_bytes,
                                                  &gate_cache_idx,
                                                  &gate_cache_gen);
            up_dev[slot]   = rocm_weight_ptr_info(model_map, model_size,
                                                  up_offset   + (uint64_t)e * gate_expert_bytes,
                                                  gate_bytes,
                                                  &up_cache_idx,
                                                  &up_cache_gen);
            down_dev[slot] = rocm_weight_ptr_info(model_map, model_size,
                                                  down_offset + (uint64_t)e * down_expert_bytes,
                                                  down_bytes,
                                                  &down_cache_idx,
                                                  &down_cache_gen);
            if (!gate_dev[slot] || !up_dev[slot] || !down_dev[slot]) return 0;

            rocm_expert_table_store(layer_id, e, ROCM_EXPERT_GATE,
                                    gate_dev[slot], gate_cache_idx, gate_cache_gen);
            rocm_expert_table_store(layer_id, e, ROCM_EXPERT_UP,
                                    up_dev[slot], up_cache_idx, up_cache_gen);
            rocm_expert_table_store(layer_id, e, ROCM_EXPERT_DOWN,
                                    down_dev[slot], down_cache_idx, down_cache_gen);
        }
        ds4_gpu_profile_mark("moe_ptr_cache");

        // Allocate GPU arrays (needed for kernel launch)
        DS4_ROCM_CHECK(hipMalloc((void **)&weights_dev, weights_bytes));
        DS4_ROCM_CHECK(hipMalloc((void **)&gate_ptrs_dev, (size_t)n_expert * sizeof(uint8_t *)));
        DS4_ROCM_CHECK(hipMalloc((void **)&up_ptrs_dev,   (size_t)n_expert * sizeof(uint8_t *)));
        DS4_ROCM_CHECK(hipMalloc((void **)&down_ptrs_dev, (size_t)n_expert * sizeof(uint8_t *)));
        if (!weights_dev || !gate_ptrs_dev || !up_ptrs_dev || !down_ptrs_dev) {
            hipFree(weights_dev); hipFree(gate_ptrs_dev); hipFree(up_ptrs_dev); hipFree(down_ptrs_dev);
            return 0;
        }
        DS4_ROCM_CHECK(hipMemcpyAsync(weights_dev, weights_host, weights_bytes, hipMemcpyHostToDevice, g_stream));
        DS4_ROCM_CHECK(hipMemcpyAsync(gate_ptrs_dev, gate_dev, (size_t)n_expert * sizeof(uint8_t *), hipMemcpyHostToDevice, g_stream));
        DS4_ROCM_CHECK(hipMemcpyAsync(up_ptrs_dev,   up_dev,   (size_t)n_expert * sizeof(uint8_t *), hipMemcpyHostToDevice, g_stream));
        DS4_ROCM_CHECK(hipMemcpyAsync(down_ptrs_dev, down_dev, (size_t)n_expert * sizeof(uint8_t *), hipMemcpyHostToDevice, g_stream));
    }
    ds4_gpu_profile_mark("moe_ptr_arrays");

    // ── Pre-quantize x and launch gate/up kernel ───────────────────
    {
        uint32_t xq_blocks = expert_in_dim / DS4_ROCM_QK_K;
        if (g_preq_x_cap < (uint64_t)xq_blocks) {
            if (g_preq_x) (void)hipFree(g_preq_x);
            DS4_ROCM_CHECK(hipMalloc((void **)&g_preq_x,
                                      (size_t)xq_blocks * sizeof(ds4_rocm_block_q8_K)));
            g_preq_x_cap = xq_blocks;
        }
        dim3 preq_grid(xq_blocks);
        dim3 preq_block(256);
        hipLaunchKernelGGL(ds4_rocm_quantize_row_q8_K_f32, preq_grid, preq_block, 0, g_stream,
                           (const float *)x->ptr, expert_in_dim, g_preq_x);
        DS4_ROCM_CHECK(hipGetLastError());
        ds4_gpu_profile_mark("moe_quant_x");

        uint32_t n_tiles_gate = max(1u, (192u + n_expert - 1u) / n_expert);
        if (n_tiles_gate > 64u) n_tiles_gate = 64u;
        dim3 grid(n_expert, n_tiles_gate);
        dim3 block(256);
        hipLaunchKernelGGL(ds4_rocm_moe_gate_up_f32, grid, block, 0, g_stream,
                           (float *)gate->ptr,
                           (float *)up->ptr,
                           (float *)mid->ptr,
                           (const uint8_t *const *)gate_ptrs_dev,
                           (const uint8_t *const *)up_ptrs_dev,
                           gate_row_bytes,
                           (const float *)x->ptr,
                           expert_in_dim,
                           expert_mid_dim,
                           weights_dev,
                           n_expert,
                           gate_type,
                           clamp,
                           g_preq_x);
        DS4_ROCM_CHECK(hipGetLastError());
        ds4_gpu_profile_mark("moe_gate_up");
    }

    // Zero output tensor
    {
        uint32_t midq_blocks = expert_mid_dim / DS4_ROCM_QK_K;
        uint64_t preq_mid_blocks = (uint64_t)n_expert * midq_blocks;
        if (g_preq_mid_cap < preq_mid_blocks) {
            if (g_preq_mid) (void)hipFree(g_preq_mid);
            DS4_ROCM_CHECK(hipMalloc((void **)&g_preq_mid,
                                      (size_t)preq_mid_blocks * sizeof(ds4_rocm_block_q8_K)));
            g_preq_mid_cap = preq_mid_blocks;
        }
        dim3 preq_mid_grid((uint32_t)preq_mid_blocks);
        dim3 preq_mid_block(256);
        hipLaunchKernelGGL(ds4_rocm_quantize_row_q8_K_f32, preq_mid_grid, preq_mid_block, 0, g_stream,
                           (const float *)mid->ptr,
                           (uint32_t)((uint64_t)n_expert * expert_mid_dim),
                           g_preq_mid);
        DS4_ROCM_CHECK(hipGetLastError());
        ds4_gpu_profile_mark("moe_quant_mid");
    }

    DS4_ROCM_CHECK(hipMemsetAsync(out->ptr, 0, out_bytes, g_stream));
    ds4_gpu_profile_mark("moe_zero");

    // ── Fused down kernel ──────────────────────────────────────────
    {
        uint32_t midq_blocks = expert_mid_dim / DS4_ROCM_QK_K;
        uint32_t n_tiles_down = max(1u, (96u + n_expert - 1u) / n_expert);
        if (n_tiles_down > 16u) n_tiles_down = 16u;
        dim3 grid(n_expert, n_tiles_down);
        dim3 block(256);
        hipLaunchKernelGGL(ds4_rocm_moe_down_f32, grid, block, 0, g_stream,
                           (float *)out->ptr,
                           (const float *)mid->ptr,
                           (const uint8_t *const *)down_ptrs_dev,
                           down_row_bytes,
                           expert_mid_dim,
                           out_dim,
                           n_expert,
                           down_type,
                           g_preq_mid);
        DS4_ROCM_CHECK(hipGetLastError());
        ds4_gpu_profile_mark("moe_down");
    }

    (void)experts;
    (void)expert_bytes;

    // Only free in slow path (fast path uses pre-allocated arrays)
    if (!all_have_ptrs) {
        DS4_ROCM_CHECK(hipFree(gate_ptrs_dev));
        DS4_ROCM_CHECK(hipFree(up_ptrs_dev));
        DS4_ROCM_CHECK(hipFree(down_ptrs_dev));
        DS4_ROCM_CHECK(hipFree(weights_dev));
    }
    return 1;
}

int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out,
                                      ds4_gpu_tensor *gate,
                                      ds4_gpu_tensor *up,
                                      ds4_gpu_tensor *mid,
                                      ds4_gpu_tensor *experts,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t gate_offset,
                                      uint64_t up_offset,
                                      uint64_t down_offset,
                                      uint32_t gate_type,
                                      uint32_t down_type,
                                      uint64_t gate_expert_bytes,
                                      uint64_t gate_row_bytes,
                                      uint64_t down_expert_bytes,
                                      uint64_t down_row_bytes,
                                      uint32_t expert_in_dim,
                                      uint32_t expert_mid_dim,
                                      uint32_t out_dim,
                                      const ds4_gpu_tensor *selected,
                                      const ds4_gpu_tensor *weights,
                                      uint32_t n_expert,
                                      float clamp,
                                      const ds4_gpu_tensor *x,
                                      uint32_t n_tokens,
                                      int layer_id,
                                      bool *mid_is_f16) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (mid_is_f16) *mid_is_f16 = false;
    if (!out || !gate || !up || !mid || !x || !model_map || !selected || !weights ||
        n_tokens == 0 || n_expert == 0 || n_expert > 6 ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        (expert_in_dim % 256u) != 0 || (expert_mid_dim % 256u) != 0) {
        return 0;
    }

    const uint64_t gate_block_bytes = ds4_rocm_routed_block_bytes(gate_type);
    const uint64_t down_block_bytes = ds4_rocm_routed_block_bytes(down_type);
    if (gate_block_bytes == 0 || down_block_bytes == 0) return 0;
    const uint64_t expected_gate_row = ((uint64_t)expert_in_dim / 256u) * gate_block_bytes;
    const uint64_t expected_down_row = ((uint64_t)expert_mid_dim / 256u) * down_block_bytes;
    if (gate_row_bytes < expected_gate_row || down_row_bytes < expected_down_row ||
        gate_expert_bytes < (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes < (uint64_t)out_dim * down_row_bytes) {
        return 0;
    }
    if (gate_expert_bytes > UINT64_MAX / 256ull ||
        down_expert_bytes > UINT64_MAX / 256ull) {
        return 0;
    }

    const uint64_t gate_tensor_bytes = 256ull * gate_expert_bytes;
    const uint64_t down_tensor_bytes = 256ull * down_expert_bytes;
    if (gate_offset > model_size || gate_tensor_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_tensor_bytes > model_size - up_offset ||
        down_offset > model_size || down_tensor_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: ROCm routed batch MoE tensor range is outside the mapped model\n");
        return 0;
    }

    uint64_t x_elems;
    uint64_t mid_pairs;
    uint64_t mid_elems;
    uint64_t out_elems;
    uint64_t expert_elems;
    uint64_t selected_elems;
    uint64_t x_bytes;
    uint64_t mid_bytes;
    uint64_t out_bytes;
    uint64_t expert_bytes;
    uint64_t selected_bytes;
    uint64_t weights_bytes;
    if (!rocm_mul_u64(n_tokens, expert_in_dim, &x_elems) ||
        !rocm_mul_u64(n_tokens, n_expert, &mid_pairs) ||
        !rocm_mul_u64(mid_pairs, expert_mid_dim, &mid_elems) ||
        !rocm_mul_u64(n_tokens, out_dim, &out_elems) ||
        !rocm_mul_u64(mid_pairs, out_dim, &expert_elems) ||
        !rocm_mul_u64(n_tokens, n_expert, &selected_elems) ||
        !rocm_bytes_for_elems(x_elems, sizeof(float), &x_bytes) ||
        !rocm_bytes_for_elems(mid_elems, sizeof(float), &mid_bytes) ||
        !rocm_bytes_for_elems(out_elems, sizeof(float), &out_bytes) ||
        !rocm_bytes_for_elems(expert_elems, sizeof(float), &expert_bytes) ||
        !rocm_bytes_for_elems(selected_elems, sizeof(int32_t), &selected_bytes) ||
        !rocm_bytes_for_elems(selected_elems, sizeof(float), &weights_bytes)) {
        return 0;
    }
    if (!rocm_require_tensor_bytes("routed batch MoE x", x, x_bytes) ||
        !rocm_require_tensor_bytes("routed batch MoE gate", gate, mid_bytes) ||
        !rocm_require_tensor_bytes("routed batch MoE up", up, mid_bytes) ||
        !rocm_require_tensor_bytes("routed batch MoE mid", mid, mid_bytes) ||
        !rocm_require_tensor_bytes("routed batch MoE out", out, out_bytes) ||
        !rocm_require_tensor_bytes("routed batch MoE selected", selected, selected_bytes) ||
        !rocm_require_tensor_bytes("routed batch MoE weights", weights, weights_bytes) ||
        (n_expert > 1 && (!experts || !rocm_require_tensor_bytes("routed batch MoE experts", experts, expert_bytes)))) {
        return 0;
    }

    const uint64_t x_stride = (uint64_t)expert_in_dim * sizeof(float);
    const uint64_t mid_stride = (uint64_t)n_expert * expert_mid_dim * sizeof(float);
    const uint64_t out_stride = (uint64_t)out_dim * sizeof(float);
    const uint64_t expert_stride = (uint64_t)n_expert * out_dim * sizeof(float);
    const uint64_t selected_stride = (uint64_t)n_expert * sizeof(int32_t);
    const uint64_t weights_stride = (uint64_t)n_expert * sizeof(float);

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(x, (uint64_t)t * x_stride, x_stride);
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(out, (uint64_t)t * out_stride, out_stride);
        ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(gate, (uint64_t)t * mid_stride, mid_stride);
        ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(up, (uint64_t)t * mid_stride, mid_stride);
        ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(mid, (uint64_t)t * mid_stride, mid_stride);
        ds4_gpu_tensor *selected_view = ds4_gpu_tensor_view(selected, (uint64_t)t * selected_stride, selected_stride);
        ds4_gpu_tensor *weights_view = ds4_gpu_tensor_view(weights, (uint64_t)t * weights_stride, weights_stride);
        ds4_gpu_tensor *experts_view = experts
            ? ds4_gpu_tensor_view(experts, (uint64_t)t * expert_stride, expert_stride)
            : nullptr;
        const int ok = x_view && out_view && gate_view && up_view && mid_view &&
                       selected_view && weights_view && (!experts || experts_view) &&
                       ds4_gpu_routed_moe_one_tensor(out_view, gate_view, up_view,
                                                       mid_view, experts_view,
                                                       model_map, model_size,
                                                       gate_offset, up_offset,
                                                       down_offset, gate_type,
                                                       down_type,
                                                       gate_expert_bytes,
                                                       gate_row_bytes,
                                                       down_expert_bytes,
                                                       down_row_bytes,
                                                       expert_in_dim,
                                                       expert_mid_dim,
                                                       out_dim, selected_view,
                                                       weights_view, n_expert,
                                                       clamp, x_view,
                                                       layer_id);
        ds4_gpu_tensor_free(x_view);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(experts_view);
        if (!ok) return 0;
    }
    return 1;
}

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !mix || !model_map || n_hc == 0 || n_hc > 16) return 0;

    uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (mix_hc > UINT64_MAX / sizeof(float)) return 0;
    uint64_t mix_bytes = mix_hc * sizeof(float);
    uint64_t mix_tensor_bytes = ds4_gpu_tensor_bytes(mix);
    uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
    if (mix_tensor_bytes < mix_bytes || out_tensor_bytes < mix_bytes) {
        fprintf(stderr, "ds4: ROCm HC split received undersized activation buffers\n");
        return 0;
    }

    uint64_t n_rows = mix_tensor_bytes / mix_bytes;
    uint64_t out_rows = out_tensor_bytes / mix_bytes;
    if (out_rows < n_rows) n_rows = out_rows;
    if (n_rows == 0) return 0;

    uint8_t *scale = rocm_weight_ptr(model_map, model_size, scale_offset, 3ull * sizeof(float));
    uint8_t *base = rocm_weight_ptr(model_map, model_size, base_offset, mix_bytes);
    if (!scale || !base) return 0;

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(n_rows, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_hc_split_sinkhorn_f32, grid, block, 0, g_stream,
                       (float *)out->ptr,
                       (const float *)mix->ptr,
                       (const float *)scale,
                       (const float *)base,
                       n_hc,
                       sinkhorn_iters,
                       n_rows,
                       eps);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

static int rocm_hc_weighted_sum_strided(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint64_t                weight_offset,
        uint64_t                weight_row_stride,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0 ||
        weight_row_stride < (uint64_t)n_hc * sizeof(float) ||
        (weight_row_stride % sizeof(float)) != 0) {
        return 0;
    }

    uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
    if (out_row_bytes == 0 || out_tensor_bytes < out_row_bytes ||
        (out_tensor_bytes % out_row_bytes) != 0) {
        fprintf(stderr, "ds4: ROCm %s output size is not a whole token row\n", label);
        return 0;
    }
    uint64_t n_tokens64 = out_tensor_bytes / out_row_bytes;
    if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) return 0;

    uint64_t hc_values;
    uint64_t residual_bytes;
    if (!rocm_mul_u64(n_hc, n_embd, &hc_values) ||
        !rocm_mul_u64(hc_values, n_tokens64, &residual_bytes) ||
        residual_bytes > UINT64_MAX / sizeof(float)) {
        return 0;
    }
    residual_bytes *= sizeof(float);
    uint64_t weights_need = weight_offset +
                            (n_tokens64 - 1u) * weight_row_stride +
                            (uint64_t)n_hc * sizeof(float);
    if (!rocm_require_tensor_bytes(label, residual_hc, residual_bytes) ||
        !rocm_require_tensor_bytes(label, weights, weights_need)) {
        return 0;
    }

    uint64_t n_elem;
    if (!rocm_mul_u64(n_tokens64, n_embd, &n_elem)) return 0;
    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(n_elem, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_hc_weighted_sum_f32, grid, block, 0, g_stream,
                       (float *)out->ptr,
                       (const float *)residual_hc->ptr,
                       (const float *)(weights->ptr + weight_offset),
                       n_embd,
                       n_hc,
                       (uint32_t)n_tokens64,
                       (uint32_t)(weight_row_stride / sizeof(float)));
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return rocm_hc_weighted_sum_strided(out,
                                        residual_hc,
                                        weights,
                                        0,
                                        (uint64_t)n_hc * sizeof(float),
                                        n_embd,
                                        n_hc,
                                        "HC weighted sum");
}

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return rocm_hc_weighted_sum_strided(out,
                                        residual_hc,
                                        split,
                                        0,
                                        mix_hc * sizeof(float),
                                        n_embd,
                                        n_hc,
                                        "HC weighted sum split");
}

int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!ds4_gpu_hc_split_sinkhorn_tensor(split,
                                            mix,
                                            model_map,
                                            model_size,
                                            scale_offset,
                                            base_offset,
                                            n_hc,
                                            sinkhorn_iters,
                                            eps)) {
        return 0;
    }
    return ds4_gpu_hc_weighted_sum_split_tensor(out, residual_hc, split, n_embd, n_hc);
}

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (!ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                split,
                                                mix,
                                                residual_hc,
                                                model_map,
                                                model_size,
                                                scale_offset,
                                                base_offset,
                                                n_embd,
                                                n_hc,
                                                sinkhorn_iters,
                                                eps)) {
        return 0;
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(norm_out,
                                                 out,
                                                 model_map,
                                                 model_size,
                                                 norm_weight_offset,
                                                 n_embd,
                                                 ds4_gpu_tensor_bytes(out) / ((uint64_t)n_embd * sizeof(float)),
                                                 norm_eps);
}

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out || !pre || !model_map || n_hc == 0) return 0;

    uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out);
    if (row_bytes == 0 || out_tensor_bytes < row_bytes ||
        (out_tensor_bytes % row_bytes) != 0) {
        fprintf(stderr, "ds4: ROCm output HC weights size is not a whole token row\n");
        return 0;
    }
    uint64_t n_tokens64 = out_tensor_bytes / row_bytes;
    if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) return 0;
    if (!rocm_require_tensor_bytes("output HC weights", pre, out_tensor_bytes)) return 0;

    uint8_t *scale = rocm_weight_ptr(model_map, model_size, scale_offset, sizeof(float));
    uint8_t *base = rocm_weight_ptr(model_map, model_size, base_offset, row_bytes);
    if (!scale || !base) return 0;

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch((uint64_t)n_tokens64 * n_hc, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_output_hc_weights_f32, grid, block, 0, g_stream,
                       (float *)out->ptr,
                       (const float *)pre->ptr,
                       (const float *)scale,
                       (const float *)base,
                       n_hc,
                       (uint32_t)n_tokens64,
                       eps);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

static int rocm_hc_expand_launch(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post_src,
        uint64_t                post_offset,
        uint32_t                post_stride,
        const ds4_gpu_tensor *comb_src,
        uint64_t                comb_offset,
        uint32_t                comb_stride,
        uint32_t                n_embd,
        uint32_t                n_hc,
        int                     has_add,
        const char             *label) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!out_hc || !block_out || !residual_hc || !post_src || !comb_src ||
        (has_add && !block_add) || n_embd == 0 || n_hc == 0) {
        return 0;
    }

    uint64_t hc_values;
    uint64_t hc_row_bytes;
    if (!rocm_mul_u64(n_hc, n_embd, &hc_values) ||
        hc_values > UINT64_MAX / sizeof(float)) {
        return 0;
    }
    hc_row_bytes = hc_values * sizeof(float);
    uint64_t out_tensor_bytes = ds4_gpu_tensor_bytes(out_hc);
    if (out_tensor_bytes < hc_row_bytes || (out_tensor_bytes % hc_row_bytes) != 0) {
        fprintf(stderr, "ds4: ROCm %s output size is not a whole HC token row\n", label);
        return 0;
    }
    uint64_t n_tokens64 = out_tensor_bytes / hc_row_bytes;
    if (n_tokens64 == 0 || n_tokens64 > UINT32_MAX) return 0;

    uint64_t block_bytes = n_tokens64 * (uint64_t)n_embd * sizeof(float);
    uint64_t hc_bytes = n_tokens64 * hc_row_bytes;
    uint64_t post_need = post_offset +
                         ((n_tokens64 - 1u) * post_stride + (uint64_t)n_hc) * sizeof(float);
    uint64_t comb_need = comb_offset +
                         ((n_tokens64 - 1u) * comb_stride + (uint64_t)n_hc * n_hc) * sizeof(float);
    if (!rocm_require_tensor_bytes(label, block_out, block_bytes) ||
        !rocm_require_tensor_bytes(label, residual_hc, hc_bytes) ||
        !rocm_require_tensor_bytes(label, post_src, post_need) ||
        !rocm_require_tensor_bytes(label, comb_src, comb_need) ||
        (has_add && !rocm_require_tensor_bytes(label, block_add, block_bytes))) {
        return 0;
    }

    dim3 grid;
    dim3 block;
    if (!rocm_make_1d_launch(n_tokens64 * hc_values, &grid, &block)) return 0;
    hipLaunchKernelGGL(ds4_rocm_hc_expand_f32, grid, block, 0, g_stream,
                       (float *)out_hc->ptr,
                       (const float *)block_out->ptr,
                       has_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
                       (const float *)residual_hc->ptr,
                       (const float *)(post_src->ptr + post_offset),
                       (const float *)(comb_src->ptr + comb_offset),
                       n_embd,
                       n_hc,
                       (uint32_t)n_tokens64,
                       post_stride,
                       comb_stride,
                       has_add);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    return rocm_hc_expand_launch(out_hc,
                                 block_out,
                                 nullptr,
                                 residual_hc,
                                 post,
                                 0,
                                 n_hc,
                                 comb,
                                 0,
                                 n_hc * n_hc,
                                 n_embd,
                                 n_hc,
                                 0,
                                 "HC expand");
}

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return rocm_hc_expand_launch(out_hc,
                                 block_out,
                                 nullptr,
                                 residual_hc,
                                 split,
                                 (uint64_t)n_hc * sizeof(float),
                                 (uint32_t)mix_hc,
                                 split,
                                 (uint64_t)(2u * n_hc) * sizeof(float),
                                 (uint32_t)mix_hc,
                                 n_embd,
                                 n_hc,
                                 0,
                                 "HC expand split");
}

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    return rocm_hc_expand_launch(out_hc,
                                 block_out,
                                 block_add,
                                 residual_hc,
                                 split,
                                 (uint64_t)n_hc * sizeof(float),
                                 (uint32_t)mix_hc,
                                 split,
                                 (uint64_t)(2u * n_hc) * sizeof(float),
                                 (uint32_t)mix_hc,
                                 n_embd,
                                 n_hc,
                                 1,
                                 "HC expand add split");
}

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(ds4_gpu_tensor *out_hc,
                                                ds4_gpu_tensor *shared_out,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint64_t in_dim,
                                                uint64_t out_dim,
                                                const ds4_gpu_tensor *shared_mid,
                                                const ds4_gpu_tensor *routed_out,
                                                const ds4_gpu_tensor *residual_hc,
                                                const ds4_gpu_tensor *split,
                                                uint32_t n_embd,
                                                uint32_t n_hc) {
    if (out_dim != n_embd) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                      weight_offset, in_dim, out_dim,
                                      shared_mid, 1)) {
        return 0;
    }
    return ds4_gpu_hc_expand_add_split_tensor(out_hc, routed_out, shared_out,
                                                residual_hc, split, n_embd, n_hc);
}

int ds4_gpu_matmul_q8_0_hc_expand_tensor(ds4_gpu_tensor *out_hc,
                                           ds4_gpu_tensor *block_out,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint64_t in_dim,
                                           uint64_t out_dim,
                                           const ds4_gpu_tensor *x,
                                           const ds4_gpu_tensor *residual_hc,
                                           const ds4_gpu_tensor *split,
                                           uint32_t n_embd,
                                           uint32_t n_hc) {
    if (out_dim != n_embd) return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x, 1)) {
        return 0;
    }
    return ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}

/* =========================================================================
 * Missing symbols: stubs and thin wrappers for ds4_gpu.h completeness.
 * =========================================================================
 */

int ds4_gpu_set_model_fd(int fd) {
    /* ROCm uses mmap-backed model access; fd is stored for reference only. */
    g_model_fd = fd;
    return 1;
}

int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                               uint64_t offset, uint64_t bytes,
                               const char *label) {
    if (!model_map || offset > model_size || bytes == 0 || bytes > model_size - offset) return 0;
    uint8_t *ptr = rocm_weight_ptr(model_map, model_size, offset, bytes);
    if (!ptr) {
        fprintf(stderr, "ds4: ROCm cache_model_range failed for %s (off=%lu bytes=%lu)\n", label ? label : "?", (unsigned long)offset, (unsigned long)bytes);
        return 0;
    }
    return 1;
}

int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size,
                                uint64_t offset, uint64_t bytes,
                                uint64_t in_dim, uint64_t out_dim,
                                const char *label) {
    /* ROCm uses Q8_0 weights directly; no separate F16 dequant cache needed.
     * Warm the raw weight cache so the range is resident on device. */
    (void)in_dim; (void)out_dim;
    return ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, label);
}

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || rows == 0 || width == 0) return 0;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    uint64_t x_bytes = (uint64_t)rows * width * sizeof(float);
    uint64_t dir_bytes = (uint64_t)(layer + 1) * width * sizeof(float);
    if (ds4_gpu_tensor_bytes(x) < x_bytes ||
        ds4_gpu_tensor_bytes(directions) < dir_bytes)
        return 0;
    const float *dir = (const float *)(((const ds4_gpu_tensor *)directions)->ptr + (uint64_t)layer * width * sizeof(float));
    uint32_t n_threads = width < 256 ? width : 256;
    dim3 grid(rows);
    dim3 block(n_threads);
    size_t sm_bytes = (size_t)n_threads * sizeof(float);
    hipLaunchKernelGGL(ds4_rocm_directional_steering_f32, grid, block, sm_bytes, g_stream,
                       (float *)x->ptr, dir, scale, width, rows);
    DS4_ROCM_CHECK(hipGetLastError());
    return 1;
}

}

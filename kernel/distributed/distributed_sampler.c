#include "../../include/neural_core/architecture/distributed.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

/*
 * SNEPPX - Distributed Sampler
 *
 * WHAT
 *   Distributed Sampler.
 *
 * CONCEPT
 *   Provides distributed training.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// Distributed Sampler (sharded data loading)
// ============================================================================

typedef struct {
    int64_t* indices;
    int64_t num_samples;
    int64_t num_samples_per_rank;
    int64_t epoch;
    int world_size;
    int rank;
    int shuffle;
    int64_t seed;
} SNEPPX_DistributedSampler;

int sneppx_distributed_sampler_init(SNEPPX_DistributedSampler** sampler,
                                     int64_t num_samples, int world_size,
                                     int rank, int shuffle, int64_t seed) {
    if (!sampler || world_size <= 0) return -1;
    SNEPPX_DistributedSampler* s = (SNEPPX_DistributedSampler*)calloc(1, sizeof(SNEPPX_DistributedSampler));
    if (!s) return -1;
    s->num_samples = num_samples;
    s->world_size = world_size;
    s->rank = rank;
    s->shuffle = shuffle;
    s->seed = seed;
    s->epoch = 0;
    
    // Compute partition
    int64_t total = num_samples;
    int64_t per_rank = (total + world_size - 1) / world_size;
    s->num_samples_per_rank = per_rank;
    s->indices = (int64_t*)malloc(per_rank * sizeof(int64_t));
    
    for (int64_t i = 0; i < per_rank; i++) {
        int64_t idx = rank * per_rank + i;
        if (idx >= total) idx = total - 1;
        s->indices[i] = idx;
    }
    
    *sampler = s;
    return 0;
}

void sneppx_distributed_sampler_destroy(SNEPPX_DistributedSampler* sampler) {
    if (!sampler) return;
    free(sampler->indices);
    free(sampler);
}

void sneppx_distributed_sampler_set_epoch(SNEPPX_DistributedSampler* sampler,
                                            int64_t epoch) {
    if (!sampler) return;
    sampler->epoch = epoch;
    if (sampler->shuffle) {
        // Fisher-Yates shuffle with epoch-based seed
        uint64_t sm_state = (uint64_t)(sampler->seed + epoch) * 0x9E3779B97F4A7C15ULL + 1;
        for (int64_t i = sampler->num_samples_per_rank - 1; i > 0; i--) {
            sm_state += 0x9E3779B97F4A7C15ULL;
            uint64_t z = sm_state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            int64_t j = (int64_t)(z % (uint64_t)(i + 1));
            int64_t tmp = sampler->indices[i];
            sampler->indices[i] = sampler->indices[j];
            sampler->indices[j] = tmp;
        }
    }
}

int64_t sneppx_distributed_sampler_get_indices(SNEPPX_DistributedSampler* sampler,
                                                 int64_t** out_indices) {
    if (!sampler || !out_indices) return -1;
    *out_indices = sampler->indices;
    return sampler->num_samples_per_rank;
}

#ifdef SNEPPX_HAS_CUDA

// ============================================================================
// Gradient Accumulation Manager
// ============================================================================

typedef struct {
    float* accum_buffer;
    int accum_steps;
    int current_step;
    int numel;
} SNEPPX_GradAccumulator;

int sneppx_grad_accumulator_init(SNEPPX_GradAccumulator** ga,
                                  int numel, int accum_steps) {
    if (!ga || numel <= 0) return -1;
    SNEPPX_GradAccumulator* g = (SNEPPX_GradAccumulator*)calloc(1, sizeof(SNEPPX_GradAccumulator));
    if (!g) return -1;
    g->numel = numel;
    g->accum_steps = accum_steps;
    g->current_step = 0;
    cudaMalloc(&g->accum_buffer, numel * sizeof(float));
    cudaMemset(g->accum_buffer, 0, numel * sizeof(float));
    *ga = g;
    return 0;
}

void sneppx_grad_accumulator_destroy(SNEPPX_GradAccumulator* ga) {
    if (!ga) return;
    if (ga->accum_buffer) cudaFree(ga->accum_buffer);
    free(ga);
}

int sneppx_grad_accumulate(SNEPPX_GradAccumulator* ga,
                            const float* grads, cudaStream_t stream) {
    if (!ga || !grads) return -1;
    auto accum = [] __global__ (float* buf, const float* g, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n) buf[i] += g[i];
    };
    accum<<<(ga->numel + 255) / 256, 256, 0, stream>>>(ga->accum_buffer, grads, ga->numel);
    ga->current_step++;
    return 0;
}

int sneppx_grad_accumulator_step(SNEPPX_GradAccumulator* ga,
                                  float* output_grads, cudaStream_t stream) {
    if (!ga || !output_grads) return -1;
    if (ga->current_step < ga->accum_steps) return 0;
    float scale = 1.0f / ga->accum_steps;
    auto finalize = [] __global__ (float* out, const float* buf, float s, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n) out[i] = buf[i] * s;
    };
    finalize<<<(ga->numel + 255) / 256, 256, 0, stream>>>(output_grads, ga->accum_buffer, scale, ga->numel);
    cudaMemsetAsync(ga->accum_buffer, 0, ga->numel * sizeof(float), stream);
    ga->current_step = 0;
    return 1;
}
#endif
/*
 * SNEPPX - Sharded Data Sampling
 *
 * WHAT
 *   Sharded Data Sampling.
 *
 * CONCEPT
 *   Sharded Data Sampling implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


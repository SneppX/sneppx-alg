#ifndef SNEPPX_RNG_CUDA_H
#define SNEPPX_RNG_CUDA_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "common.cuh"

#ifdef __cplusplus
/*
 * SNEPPX - Rng Cuda
 *
 * WHAT
 *   Rng Cuda.
 *
 * CONCEPT
 *   Provides random number generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ============================================================================
// RNG States
// ============================================================================

typedef struct {
    curandStatePhilox4_32_10_t* states;
    int num_states;
    unsigned long long seed;
    unsigned long long offset;
} SNEPPX_CudaRNG;

// ============================================================================
// RNG Lifecycle
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rng_create(
    SNEPPX_CudaRNG** rng,
    int num_states,
    unsigned long long seed,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_rng_destroy(
    SNEPPX_CudaRNG* rng
);

SNEPPX_CudaError sneppx_cuda_rng_seed(
    SNEPPX_CudaRNG* rng,
    unsigned long long seed,
    unsigned long long offset,
    SNEPPX_CudaStream_t stream
);

// ============================================================================
// Uniform Random
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rand_uniform_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int numel,
    float low,
    float high
);

SNEPPX_CudaError sneppx_cuda_rand_uniform_f16(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    half* output,
    int numel,
    float low,
    float high
);

// ============================================================================
// Normal / Gaussian Random
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rand_normal_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int numel,
    float mean,
    float std
);

SNEPPX_CudaError sneppx_cuda_rand_normal_f16(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    half* output,
    int numel,
    float mean,
    float std
);

// ============================================================================
// Truncated Normal (for weight initialization)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rand_truncated_normal_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int numel,
    float mean,
    float std,
    float a,
    float b
);

SNEPPX_CudaError sneppx_cuda_rand_truncated_normal_f16(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    half* output,
    int numel,
    float mean,
    float std,
    float a,
    float b
);

// ============================================================================
// Bernoulli (for dropout mask generation)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rand_bernoulli_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int numel,
    float p
);

SNEPPX_CudaError sneppx_cuda_rand_bernoulli_mask(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    half* output,
    int numel,
    float p
);

// ============================================================================
// Integer Random
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rand_int(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    int* output,
    int numel,
    int low,
    int high
);

// ============================================================================
// Xavier / Kaiming Initialization
// ============================================================================

SNEPPX_CudaError sneppx_cuda_xavier_uniform_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int rows,
    int cols,
    SNEPPX_ActivationType act
);

SNEPPX_CudaError sneppx_cuda_xavier_normal_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int rows,
    int cols,
    SNEPPX_ActivationType act
);

SNEPPX_CudaError sneppx_cuda_kaiming_uniform_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int rows,
    int cols,
    SNEPPX_ActivationType act
);

SNEPPX_CudaError sneppx_cuda_kaiming_normal_f32(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    float* output,
    int rows,
    int cols,
    SNEPPX_ActivationType act
);

// ============================================================================
// Dropout (fused generate mask + apply)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_dropout_fwd(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    const half* input,
    half* output,
    half* mask,
    int numel,
    float p
);

// ============================================================================
// Batch RNG (multiple parallel streams)
// ============================================================================

typedef struct {
    SNEPPX_CudaRNG* rng;
    int batch_size;
    unsigned long long base_seed;
} SNEPPX_BatchRNG;

SNEPPX_CudaError sneppx_cuda_batch_rng_create(
    SNEPPX_BatchRNG** brng,
    int batch_size,
    unsigned long long base_seed,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_batch_rng_destroy(
    SNEPPX_BatchRNG* brng
);

SNEPPX_CudaError sneppx_cuda_batch_rand_normal(
    SNEPPX_CudaStream_t stream,
    SNEPPX_BatchRNG* brng,
    half** outputs,
    int batch_size,
    int numel_per_batch,
    float mean,
    float std
);

// ============================================================================
// Permutation (for shuffling)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rand_permutation(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaRNG* rng,
    int* output,
    int n
);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_RNG_CUDA_H

#ifndef SNEPPX_ARC_CUDA_CUH
#define SNEPPX_ARC_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../kernel/cuda/common.cuh"

/*
 * SNEPPX - Arc Cuda
 *
 * WHAT
 *   Arc Cuda.
 *
 * CONCEPT
 *   Provides the Arc Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ARC Adversarial Robustness CUDA Kernels
// - Gradient obfuscation (input gradient masking)
// - Adversarial attack simulation (PGD, FGSM, CW)
// - Certified defense forward pass

#ifdef __cplusplus
extern "C" {
#endif

// PGD (Projected Gradient Descent) Attack
SNEPPX_CudaError sneppx_cuda_pgd_attack(
    SNEPPX_CudaStream_t stream,
    const half* input,
    const int* labels,
    half* adversarial,
    int batch_size,
    int numel,
    float epsilon,
    float alpha,
    int num_steps,
    float* (*get_grad_fn)(const half*, const int*, int, int)
);

// Fast Gradient Sign Method (FGSM)
SNEPPX_CudaError sneppx_cuda_fgsm_attack(
    SNEPPX_CudaStream_t stream,
    const half* input,
    const half* gradients,
    half* adversarial,
    int numel,
    float epsilon
);

// Gradient obfuscation (input perturbation)
SNEPPX_CudaError sneppx_cuda_gradient_obfuscate(
    SNEPPX_CudaStream_t stream,
    const half* input,
    half* output,
    const half* noise_scale,
    int numel
);

// Certified defense forward pass with randomized smoothing
SNEPPX_CudaError sneppx_cuda_randomized_smoothing(
    SNEPPX_CudaStream_t stream,
    const half* input,
    half* output,
    int batch_size,
    int num_classes,
    int num_samples,
    float noise_std
);

#ifdef __cplusplus
}
#endif

#endif

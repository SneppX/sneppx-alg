#ifndef SNEPPX_OPTIM_CUDA_H
#define SNEPPX_OPTIM_CUDA_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.cuh"

#ifdef __cplusplus
/*
 * SNEPPX - Optim Cuda
 *
 * WHAT
 *   Optim Cuda.
 *
 * CONCEPT
 *   Provides the Optim Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ============================================================================
// Optimizer Types
// ============================================================================

typedef enum {
    SNEPPX_OPTIM_SGD = 0,
    SNEPPX_OPTIM_SGD_MOMENTUM = 1,
    SNEPPX_OPTIM_ADAM = 2,
    SNEPPX_OPTIM_ADAMW = 3,
    SNEPPX_OPTIM_LION = 4,
    SNEPPX_OPTIM_ADAFACTOR = 5,
    SNEPPX_OPTIM_RMS_PROP = 6,
    SNEPPX_OPTIM_SOFSIGN = 7,
    SNEPPX_OPTIM_LAMB = 8,
    SNEPPX_OPTIM_LARS = 9,
} SNEPPX_OptimizerType;

// ============================================================================
// Optimizer State on Device
// ============================================================================

typedef struct {
    SNEPPX_OptimizerType type;
    
    // Step counter
    int step;
    
    // Learning rate
    float lr;
    float lr_min;
    float lr_decay;
    float lr_warmup_steps;
    float weight_decay;
    
    // Adam/AdamW params
    float beta1;
    float beta2;
    float epsilon;
    
    // AdamW-specific
    bool decoupled_wd;  // true: decoupled weight decay (AdamW)
    
    // Lion params
    float beta1_lion;
    float beta2_lion;
    
    // RMSProp params
    float rms_alpha;
    float rms_momentum;
    
    // Gradient clipping
    float grad_clip_max_norm;
    bool grad_clip_enabled;
    
    // FP16 master weights (for mixed precision)
    half* master_weights;
    half** param_groups;
    int* param_sizes;
    int num_params;
    
    // State buffers (allocated on device)
    void* state_buf1;  // m (momentum / first moment)
    void* state_buf2;  // v (velocity / second moment)
    void* state_buf3;  // extra (e.g., variance for AdaFactor)
} SNEPPX_OptimizerState;

// ============================================================================
// Fused Optimizer Steps (all-in-one kernel)
// ============================================================================

// SGD
SNEPPX_CudaError sneppx_cuda_sgd_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float lr,
    float weight_decay,
    int numel
);

// SGD with Momentum
SNEPPX_CudaError sneppx_cuda_sgd_momentum_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* momentum,
    float lr,
    float momentum_factor,
    float dampening,
    float weight_decay,
    bool nesterov,
    int numel
);

// AdamW (fused: bias correction + weight decay + update in one kernel)
SNEPPX_CudaError sneppx_cuda_adamw_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* exp_avg,
    float* exp_avg_sq,
    int step,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int numel
);

// Lion (fused: symbolic discovery optimizer)
SNEPPX_CudaError sneppx_cuda_lion_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* exp_avg,
    int step,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    int numel
);

// LAMB (Layer-wise Adaptive Moments)
SNEPPX_CudaError sneppx_cuda_lamb_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* exp_avg,
    float* exp_avg_sq,
    int step,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int numel
);

// LARS (Layer-wise Adaptive Rate Scaling)
SNEPPX_CudaError sneppx_cuda_lars_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* momentum,
    int step,
    float lr,
    float momentum_factor,
    float weight_decay,
    float epsilon,
    float trust_coefficient,
    int numel
);

// AdaFactor
SNEPPX_CudaError sneppx_cuda_adafactor_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* exp_avg_sq,
    float* exp_avg_sq_row,
    float* exp_avg_sq_col,
    int step,
    float lr,
    float beta2,
    float epsilon,
    float weight_decay,
    int numel,
    int rows,
    int cols
);

// ============================================================================
// Distributed Optimizer (ZeRO-1 style)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_zero1_adamw_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float* exp_avg,
    float* exp_avg_sq,
    int step,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float loss_scale,
    int numel,
    int world_size,
    int rank
);

// ============================================================================
// Learning Rate Scheduling (in-kernel)
// ============================================================================

// Cosine decay with warmup
__device__ float sneppx_lr_cosine(
    int step,
    int warmup_steps,
    int total_steps,
    float lr_init,
    float lr_min
);

// Linear decay with warmup
__device__ float sneppx_lr_linear(
    int step,
    int warmup_steps,
    int total_steps,
    float lr_init,
    float lr_min
);

// Constant with warmup
__device__ float sneppx_lr_constant_warmup(
    int step,
    int warmup_steps,
    float lr_init
);

// ============================================================================
// Optimizer Lifecycle
// ============================================================================

SNEPPX_CudaError sneppx_cuda_optimizer_init(
    SNEPPX_OptimizerState* state,
    SNEPPX_OptimizerType type,
    float lr,
    int num_params,
    const int* param_sizes,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_optimizer_destroy(
    SNEPPX_OptimizerState* state,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_optimizer_zero_grad(
    SNEPPX_CudaStream_t stream,
    half** gradients,
    int num_params,
    const int* sizes
);

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    bool decoupled_wd;
    float beta1_lion;
    float beta2_lion;
    float grad_clip_max_norm;
    bool grad_clip_enabled;
    float lr_min;
    float lr_decay;
    float lr_warmup_steps;
    int total_steps;
} SNEPPX_OptimizerConfig;

SNEPPX_CudaError sneppx_cuda_optimizer_step(
    SNEPPX_CudaStream_t stream,
    SNEPPX_OptimizerState* state,
    half** params,
    half** grads,
    int num_params,
    const int* sizes,
    int current_step,
    const SNEPPX_OptimizerConfig* config
);

// ============================================================================
// Mixed Precision Optimizer
// ============================================================================

SNEPPX_CudaError sneppx_cuda_optimizer_grad_scaler(
    SNEPPX_CudaStream_t stream,
    half** grads,
    const int* sizes,
    int num_params,
    float* scale_ptr,
    float growth_factor,
    float backoff_factor,
    int growth_interval,
    float* scale,
    int* num_overflow_buckets
);

// Check for overflow in gradients
SNEPPX_CudaError sneppx_cuda_check_overflow(
    SNEPPX_CudaStream_t stream,
    const half* grads,
    int numel,
    int* overflow_flag
);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_OPTIM_CUDA_H

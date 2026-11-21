#include "optim_cuda.h"
#include "common.cuh"
#include <cooperative_groups.h>

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


namespace cg = cooperative_groups;

// ============================================================================
// Learning Rate Schedule Helpers (Device)
// ============================================================================

__device__ float sneppx_lr_cosine(
    int step, int warmup_steps, int total_steps,
    float lr_init, float lr_min
) {
    if (step < warmup_steps) {
        return lr_init * (float)(step + 1) / (float)(warmup_steps + 1);
    }
    float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    float cosine = 0.5f * (1.0f + cosf(M_PI * progress));
    return lr_min + (lr_init - lr_min) * cosine;
}

__device__ float sneppx_lr_linear(
    int step, int warmup_steps, int total_steps,
    float lr_init, float lr_min
) {
    if (step < warmup_steps) {
        return lr_init * (float)(step + 1) / (float)(warmup_steps + 1);
    }
    float progress = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
    return lr_init - (lr_init - lr_min) * progress;
}

__device__ float sneppx_lr_constant_warmup(int step, int warmup_steps, float lr_init) {
    if (step < warmup_steps) {
        return lr_init * (float)(step + 1) / (float)(warmup_steps + 1);
    }
    return lr_init;
}

// ============================================================================
// SGD Step
// ============================================================================

__global__ void sgd_step_kernel(
    half* params,
    const half* grads,
    float lr,
    float weight_decay,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float p = __half2float(params[idx]);
    float g = __half2float(grads[idx]);
    
    p -= lr * (g + weight_decay * p);
    
    params[idx] = __float2half_rn(p);
}

SNEPPX_CudaError sneppx_cuda_sgd_step(
    SNEPPX_CudaStream_t stream,
    half* params,
    const half* grads,
    float lr,
    float weight_decay,
    int numel
) {
    if (!params || !grads) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    sgd_step_kernel<<<grid, block, 0, stream>>>(params, grads, lr, weight_decay, numel);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// SGD with Momentum
// ============================================================================

__global__ void sgd_momentum_step_kernel(
    half* params,
    const half* grads,
    float* momentum,
    float lr,
    float momentum_factor,
    float dampening,
    float weight_decay,
    bool nesterov,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float p = __half2float(params[idx]);
    float g = __half2float(grads[idx]);
    float m = momentum[idx];
    
    // Apply weight decay
    g += weight_decay * p;
    
    // Update momentum
    m = momentum_factor * m + (1.0f - dampening) * g;
    momentum[idx] = m;
    
    // Update parameters
    if (nesterov) {
        p -= lr * (g + momentum_factor * m);
    } else {
        p -= lr * m;
    }
    
    params[idx] = __float2half_rn(p);
}

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
) {
    if (!params || !grads || !momentum) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    sgd_momentum_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, momentum, lr, momentum_factor,
        dampening, weight_decay, nesterov, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// AdamW Step (fused)
// ============================================================================

__global__ void adamw_step_kernel(
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float p = __half2float(params[idx]);
    float g = __half2float(grads[idx]);
    
    float m = exp_avg[idx];
    float v = exp_avg_sq[idx];
    
    // Update biased moments
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
    
    // Bias correction
    float m_hat = m / (1.0f - powf(beta1, (float)step));
    float v_hat = v / (1.0f - powf(beta2, (float)step));
    
    // Decoupled weight decay (AdamW)
    p -= lr * weight_decay * p;
    
    // Update
    p -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    
    params[idx] = __float2half_rn(p);
}

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
) {
    if (!params || !grads || !exp_avg || !exp_avg_sq) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    adamw_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, exp_avg, exp_avg_sq, step,
        lr, beta1, beta2, epsilon, weight_decay, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Lion Step (fused)
// ============================================================================

__global__ void lion_step_kernel(
    half* params,
    const half* grads,
    float* exp_avg,
    int step,
    float lr,
    float beta1,
    float beta2,
    float weight_decay,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float p = __half2float(params[idx]);
    float g = __half2float(grads[idx]);
    float m = exp_avg[idx];
    
    // Lion: update = sign(m * beta1 + g * (1 - beta1))
    float update = m * beta1 + g * (1.0f - beta1);
    float sign = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);
    
    // Update momentum (use current gradient)
    exp_avg[idx] = beta2 * m + (1.0f - beta2) * g;
    
    // Weight decay
    p -= lr * weight_decay * p;
    
    // Update
    p -= lr * sign;
    
    params[idx] = __float2half_rn(p);
}

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
) {
    if (!params || !grads || !exp_avg) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    lion_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, exp_avg, step,
        lr, beta1, beta2, weight_decay, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// LAMB Step (Layer-wise Adaptive Moments)
// ============================================================================

__global__ void lamb_step_kernel(
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
) {
    extern __shared__ float smem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float p = __half2float(params[idx]);
    float g = __half2float(grads[idx]);
    
    float m = exp_avg[idx];
    float v = exp_avg_sq[idx];
    
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;
    
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
    
    float m_hat = m / (1.0f - powf(beta1, (float)step));
    float v_hat = v / (1.0f - powf(beta2, (float)step));
    
    float update = m_hat / (sqrtf(v_hat) + epsilon);
    update += weight_decay * p;
    
    // Compute trust ratio
    float p_norm = fabsf(p);
    float g_norm = fabsf(update);
    
    float trust_ratio = 1.0f;
    if (p_norm > 0.0f && g_norm > 0.0f) {
        trust_ratio = p_norm / g_norm;
    }
    
    // Warp-level trust ratio (take max)
    if (tid < numel) {
        p -= lr * trust_ratio * update;
        params[idx] = __float2half_rn(p);
    }
}

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
) {
    if (!params || !grads || !exp_avg || !exp_avg_sq) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    lamb_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, exp_avg, exp_avg_sq, step,
        lr, beta1, beta2, epsilon, weight_decay, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// LARS Step
// ============================================================================

__global__ void lars_step_kernel(
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
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    float p_norm_local = 0.0f;
    float g_norm_local = 0.0f;
    
    int start = bid * blockDim.x;
    int end = min(start + blockDim.x, numel);
    
    for (int i = start + tid; i < end; i += blockDim.x) {
        float p = __half2float(params[i]);
        float g = __half2float(grads[i]);
        p_norm_local += p * p;
        g_norm_local += g * g;
    }
    
    // Warp reduce
    p_norm_local = sqrtf(sneppx_warp_reduce_sum(p_norm_local));
    g_norm_local = sqrtf(sneppx_warp_reduce_sum(g_norm_local));
    
    if (tid == 0) {
        float lr_scale = 1.0f;
        if (p_norm_local > 0.0f && g_norm_local > 0.0f) {
            lr_scale = trust_coefficient * p_norm_local / (g_norm_local + weight_decay * p_norm_local + epsilon);
        }
        smem[bid] = lr_scale;
    }
    __syncthreads();
    
    // Apply update (simplified - use global lr_scale)
    for (int i = start + tid; i < end; i += blockDim.x) {
        float p = __half2float(params[i]);
        float g = __half2float(grads[i]);
        float m = momentum[i];
        
        g += weight_decay * p;
        m = momentum_factor * m + g;
        momentum[i] = m;
        
        p -= lr * smem[bid] * m;
        params[i] = __float2half_rn(p);
    }
}

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
) {
    if (!params || !grads || !momentum) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    lars_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, momentum, step,
        lr, momentum_factor, weight_decay,
        epsilon, trust_coefficient, numel
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// AdaFactor Step (Matrix factorization of second moment)
// ============================================================================

__global__ void adafactor_step_kernel(
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    int col = idx % cols;
    int row = idx / cols;
    
    float p = __half2float(params[idx]);
    float g = __half2float(grads[idx]);
    
    float v = exp_avg_sq[idx];
    float vr = exp_avg_sq_row[row];
    float vc = exp_avg_sq_col[col];
    
    // Update factored second moments
    float g2 = g * g;
    float beta2t = beta2;
    
    v = beta2t * v + (1.0f - beta2t) * g2;
    vr = beta2t * vr + (1.0f - beta2t) * g2 / cols;
    vc = beta2t * vc + (1.0f - beta2t) * g2 / rows;
    
    exp_avg_sq[idx] = v;
    exp_avg_sq_row[row] = vr;
    exp_avg_sq_col[col] = vc;
    
    // Compute adaptive step
    float rms = sqrtf(vr * vc / fmaxf(v, 1e-10f));
    float update = g / (rms + epsilon);
    
    // Weight decay
    update += weight_decay * p;
    
    // Bias correction for step
    float step_num = (float)step;
    float bias_corr = 1.0f / (1.0f - powf(beta2, step_num));
    update *= bias_corr;
    
    p -= lr * update;
    params[idx] = __float2half_rn(p);
}

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
) {
    if (!params || !grads || !exp_avg_sq || !exp_avg_sq_row || !exp_avg_sq_col) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    adafactor_step_kernel<<<grid, block, 0, stream>>>(
        params, grads, exp_avg_sq, exp_avg_sq_row, exp_avg_sq_col,
        step, lr, beta2, epsilon, weight_decay, numel, rows, cols
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// ZeRO-1 AdamW (partitioned optimizer states)
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
) {
    if (!params || !grads || !exp_avg || !exp_avg_sq) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    // Determine partition range
    int local_size = numel / world_size;
    int remainder = numel % world_size;
    int start = rank * local_size + min(rank, remainder);
    if (rank < remainder) local_size++;
    
    // Only update the partition owned by this rank
    // (other ranks' parameters are updated via all-gather)
    int block = 256;
    int grid = (local_size + block - 1) / block;
    
    auto kernel = adamw_step_kernel;
    kernel<<<grid, block, 0, stream>>>(
        params + start, grads + start,
        exp_avg + start, exp_avg_sq + start,
        step, lr, beta1, beta2, epsilon, weight_decay, local_size
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

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
) {
    if (!state || !param_sizes) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    state->type = type;
    state->lr = lr;
    state->step = 0;
    state->num_params = num_params;
    
    // Default hyperparameters
    state->beta1 = 0.9f;
    state->beta2 = 0.999f;
    state->epsilon = 1e-8f;
    state->weight_decay = 0.01f;
    state->decoupled_wd = true;
    state->beta1_lion = 0.9f;
    state->beta2_lion = 0.99f;
    state->grad_clip_max_norm = 1.0f;
    state->grad_clip_enabled = true;
    
    // Allocate state buffers
    size_t total_elts = 0;
    for (int i = 0; i < num_params; i++) {
        total_elts += param_sizes[i];
    }
    
    if (type == SNEPPX_OPTIM_ADAM || type == SNEPPX_OPTIM_ADAMW || 
        type == SNEPPX_OPTIM_LAMB || type == SNEPPX_OPTIM_LION) {
        cudaMallocAsync(&state->state_buf1, total_elts * sizeof(float), stream);
        cudaMallocAsync(&state->state_buf2, total_elts * sizeof(float), stream);
        cudaMemsetAsync(state->state_buf1, 0, total_elts * sizeof(float), stream);
        cudaMemsetAsync(state->state_buf2, 0, total_elts * sizeof(float), stream);
    } else if (type == SNEPPX_OPTIM_SGD_MOMENTUM || type == SNEPPX_OPTIM_SOFSIGN) {
        cudaMallocAsync(&state->state_buf1, total_elts * sizeof(float), stream);
        cudaMemsetAsync(state->state_buf1, 0, total_elts * sizeof(float), stream);
    } else if (type == SNEPPX_OPTIM_ADAFACTOR) {
        cudaMallocAsync(&state->state_buf1, total_elts * sizeof(float), stream);
        cudaMallocAsync(&state->state_buf2, total_elts * sizeof(float), stream);
        cudaMallocAsync(&state->state_buf3, total_elts * sizeof(float), stream);
        cudaMemsetAsync(state->state_buf1, 0, total_elts * sizeof(float), stream);
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_optimizer_destroy(
    SNEPPX_OptimizerState* state,
    SNEPPX_CudaStream_t stream
) {
    if (!state) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaFreeAsync(state->state_buf1, stream);
    cudaFreeAsync(state->state_buf2, stream);
    cudaFreeAsync(state->state_buf3, stream);
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_optimizer_zero_grad(
    SNEPPX_CudaStream_t stream,
    half** gradients,
    int num_params,
    const int* sizes
) {
    if (!gradients || !sizes) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < num_params; i++) {
        cudaMemsetAsync(gradients[i], 0, sizes[i] * sizeof(half), stream);
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Generic Optimizer Step (dispatches to specific implementations)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_optimizer_step(
    SNEPPX_CudaStream_t stream,
    SNEPPX_OptimizerState* state,
    half** params,
    half** grads,
    int num_params,
    const int* sizes,
    int current_step,
    const SNEPPX_OptimizerConfig* config
) {
    if (!state || !params || !grads || !sizes) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    state->step = current_step;
    
    // Compute learning rate
    float lr = state->lr;
    if (config) {
        if (config->lr_min > 0 || config->lr_decay > 0) {
            int total = config->total_steps ? config->total_steps : current_step + 1;
            lr = sneppx_lr_cosine(current_step, config->lr_warmup_steps, total, state->lr, config->lr_min);
        }
        
        if (config->grad_clip_enabled && config->grad_clip_max_norm > 0) {
            for (int i = 0; i < num_params; i++) {
                sneppx_cuda_grad_clip(stream, grads[i], sizes[i], 
                                     config->grad_clip_max_norm, 2.0f);
            }
        }
    }
    
    // Apply step for each parameter group
    size_t offset = 0;
    for (int i = 0; i < num_params; i++) {
        int n = sizes[i];
        
        float* buf1 = (float*)state->state_buf1 + offset;
        float* buf2 = (float*)state->state_buf2 + offset;
        
        switch (state->type) {
            case SNEPPX_OPTIM_SGD:
                sneppx_cuda_sgd_step(stream, params[i], grads[i], lr, 
                                    state->weight_decay, n);
                break;
            case SNEPPX_OPTIM_SGD_MOMENTUM:
                sneppx_cuda_sgd_momentum_step(stream, params[i], grads[i], (float*)state->state_buf1 + offset,
                                             lr, state->beta1, 0.0f, state->weight_decay, false, n);
                break;
            case SNEPPX_OPTIM_ADAMW:
                sneppx_cuda_adamw_step(stream, params[i], grads[i], buf1, buf2,
                                      current_step, lr, state->beta1, state->beta2,
                                      state->epsilon, state->weight_decay, n);
                break;
            case SNEPPX_OPTIM_LION:
                sneppx_cuda_lion_step(stream, params[i], grads[i], buf1,
                                     current_step, lr, state->beta1_lion,
                                     state->beta2_lion, state->weight_decay, n);
                break;
            case SNEPPX_OPTIM_LAMB:
                sneppx_cuda_lamb_step(stream, params[i], grads[i], buf1, buf2,
                                     current_step, lr, state->beta1, state->beta2,
                                     state->epsilon, state->weight_decay, n);
                break;
            case SNEPPX_OPTIM_LARS:
                sneppx_cuda_lars_step(stream, params[i], grads[i], (float*)state->state_buf1 + offset,
                                     current_step, lr, state->beta1,
                                     state->weight_decay, state->epsilon, 0.01f, n);
                break;
            case SNEPPX_OPTIM_ADAM:
                sneppx_cuda_adamw_step(stream, params[i], grads[i], buf1, buf2,
                                      current_step, lr, state->beta1, state->beta2,
                                      state->epsilon, 0.0f, n);
                break;
            default:
                break;
        }
        
        offset += n;
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Gradient Scaling (for mixed precision)
// ============================================================================

__global__ void check_overflow_kernel(
    const half* grads,
    int numel,
    int* overflow_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float g = __half2float(grads[idx]);
    if (isinf(g) || isnan(g)) {
        atomicAdd(overflow_flag, 1);
    }
}

SNEPPX_CudaError sneppx_cuda_check_overflow(
    SNEPPX_CudaStream_t stream,
    const half* grads,
    int numel,
    int* overflow_flag
) {
    if (!grads || !overflow_flag) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    check_overflow_kernel<<<grid, block, 0, stream>>>(grads, numel, overflow_flag);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

#include "ser_cuda.cuh"
#include <cooperative_groups.h>

/*
 * SNEPPX - Ser Cuda
 *
 * WHAT
 *   Ser Cuda.
 *
 * CONCEPT
 *   Provides the Ser Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace cg = cooperative_groups;

// Top-k gating (softmax + top-k selection)
__global__ void topk_gating_kernel(
    const half* router_logits, int* expert_indices,
    half* gating_weights, int batch_size, int num_experts, int top_k
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each warp handles one batch item
    int warp_id = tid / SNEPPX_WARP_SIZE;
    int lane_id = tid % SNEPPX_WARP_SIZE;
    
    if (warp_id >= 1) return;
    
    // Load logits for this batch
    __shared__ float logits[64];  // max experts per shared
    float vals[2];
    
    if (lane_id < num_experts) {
        logits[lane_id] = __half2float(router_logits[batch * num_experts + lane_id]);
    }
    __syncthreads();
    
    // Softmax
    float max_val = -INFINITY;
    for (int i = lane_id; i < num_experts; i += SNEPPX_WARP_SIZE) {
        max_val = fmaxf(max_val, logits[i]);
    }
    max_val = sneppx_warp_reduce_max(max_val);
    
    float sum = 0.0f;
    for (int i = lane_id; i < num_experts; i += SNEPPX_WARP_SIZE) {
        float e = expf(logits[i] - max_val);
        logits[i] = e;
        sum += e;
    }
    sum = sneppx_warp_reduce_sum(sum);
    
    if (sum > 0.0f) {
        for (int i = lane_id; i < num_experts; i += SNEPPX_WARP_SIZE) {
            logits[i] /= sum;
        }
    }
    __syncthreads();
    
    // Top-k selection (simple selection sort for small k)
    for (int k = 0; k < top_k; k++) {
        float best_val = -1.0f;
        int best_idx = -1;
        
        for (int i = 0; i < num_experts; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        
        if (lane_id == 0 && best_idx >= 0) {
            expert_indices[batch * top_k + k] = best_idx;
            gating_weights[batch * top_k + k] = __float2half_rn(best_val);
            logits[best_idx] = -1.0f;  // Remove from selection
        }
        __syncthreads();
    }
}

SNEPPX_CudaError sneppx_cuda_topk_gating(
    SNEPPX_CudaStream_t stream,
    const half* router_logits, int* expert_indices,
    half* gating_weights, int batch_size, int num_experts, int top_k
) {
    if (!router_logits || !expert_indices || !gating_weights) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    topk_gating_kernel<<<batch_size, 32, 0, stream>>>(
        router_logits, expert_indices, gating_weights, batch_size, num_experts, top_k
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Dispatch tokens to experts (scatter)
__global__ void dispatch_to_experts_kernel(
    const half* tokens, half* expert_inputs,
    const int* expert_indices, const int* expert_counts,
    int total_tokens, int num_experts, int expert_capacity, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / dim;
    int d = idx % dim;
    
    if (token_idx >= total_tokens) return;
    
    for (int k = 0; k < 1; k++) {  // Simplified: only top-1 dispatch
        int expert = expert_indices[token_idx];
        if (expert < 0 || expert >= num_experts) continue;
        
        // Use prefix sum count to get position
        int pos = expert_counts[expert];
        if (pos < expert_capacity) {
            expert_inputs[(expert * expert_capacity + pos) * dim + d] = tokens[token_idx * dim + d];
        }
    }
}

SNEPPX_CudaError sneppx_cuda_dispatch_to_experts(
    SNEPPX_CudaStream_t stream,
    const half* tokens, half* expert_inputs,
    const int* expert_indices, const int* expert_counts,
    int total_tokens, int num_experts, int expert_capacity, int dim
) {
    if (!tokens || !expert_inputs || !expert_indices) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int total = total_tokens * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    dispatch_to_experts_kernel<<<grid, block, 0, stream>>>(
        tokens, expert_inputs, expert_indices, expert_counts,
        total_tokens, num_experts, expert_capacity, dim
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Combine from experts (gather + weighted sum)
__global__ void combine_from_experts_kernel(
    half* output, const half* expert_outputs,
    const int* expert_indices, const half* gating_weights,
    const int* expert_counts,
    int total_tokens, int num_experts, int dim, int top_k
) {
    int token_idx = blockIdx.x;
    int d = threadIdx.x;
    if (d >= dim) return;
    
    float result = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int expert = expert_indices[token_idx * top_k + k];
        if (expert < 0) continue;
        float weight = __half2float(gating_weights[token_idx * top_k + k]);
        result += weight * __half2float(expert_outputs[(expert * total_tokens + token_idx) * dim + d]);
    }
    output[token_idx * dim + d] = __float2half_rn(result);
}

SNEPPX_CudaError sneppx_cuda_combine_from_experts(
    SNEPPX_CudaStream_t stream,
    half* output, const half* expert_outputs,
    const int* expert_indices, const half* gating_weights,
    const int* expert_counts,
    int total_tokens, int num_experts, int dim, int top_k
) {
    if (!output || !expert_outputs || !expert_indices || !gating_weights) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    combine_from_experts_kernel<<<total_tokens, dim, 0, stream>>>(
        output, expert_outputs, expert_indices, gating_weights,
        expert_counts, total_tokens, num_experts, dim, top_k
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Load balancing loss
__global__ void load_balancing_loss_kernel(
    const float* expert_counts, const float* router_prob,
    float* loss, int num_experts, float beta
) {
    int tid = threadIdx.x;
    float local_loss = 0.0f;
    float total_count = 0.0f;
    
    for (int i = tid; i < num_experts; i += blockDim.x) {
        total_count += expert_counts[i];
    }
    total_count = sneppx_warp_reduce_sum(total_count);
    
    if (total_count == 0.0f) total_count = 1.0f;
    
    for (int i = tid; i < num_experts; i += blockDim.x) {
        float f_i = expert_counts[i] / total_count;
        float p_i = router_prob[i];
        local_loss += f_i * p_i;
    }
    local_loss = sneppx_warp_reduce_sum(local_loss);
    
    if (tid == 0) {
        *loss = beta * (float)num_experts * local_loss;
    }
}

SNEPPX_CudaError sneppx_cuda_load_balancing_loss(
    SNEPPX_CudaStream_t stream,
    const float* expert_counts, const float* router_prob,
    float* loss, int num_experts, float beta
) {
    if (!expert_counts || !router_prob || !loss) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    load_balancing_loss_kernel<<<1, 256, 0, stream>>>(expert_counts, router_prob, loss, num_experts, beta);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Fused MoE forward (simplified single-pass implementation)
SNEPPX_CudaError sneppx_cuda_fused_moe_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FusedMoEParams* params
) {
    if (!params || !params->input || !params->output || !params->expert_weights) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int total_tokens = params->batch_size * params->seq_len;
    
    half *expert_inputs, *expert_outputs, *gating_weights;
    int *expert_indices, *expert_counts;
    
    cudaMallocAsync(&gating_weights, total_tokens * params->top_k * sizeof(half), stream);
    cudaMallocAsync(&expert_indices, total_tokens * params->top_k * sizeof(int), stream);
    cudaMallocAsync(&expert_counts, params->num_experts * sizeof(int), stream);
    cudaMallocAsync(&expert_inputs, params->num_experts * params->expert_capacity * params->dim * sizeof(half), stream);
    cudaMallocAsync(&expert_outputs, params->num_experts * params->expert_capacity * params->hidden_dim * sizeof(half), stream);
    
    cudaMemsetAsync(expert_counts, 0, params->num_experts * sizeof(int), stream);
    
    // 1) Gating
    sneppx_cuda_topk_gating(stream, params->router_weights == nullptr ? nullptr : params->router_weights,
                           expert_indices, gating_weights,
                           total_tokens, params->num_experts, params->top_k);
    
    // 2) Count tokens per expert
    auto count_kernel = [] __global__ (const int* idx, int* cnt, int t, int k, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < t * k) {
            int e = idx[i];
            if (e >= 0 && e < n) atomicAdd(&cnt[e], 1);
        }
    };
    count_kernel<<<(total_tokens * params->top_k + 255) / 256, 256, 0, stream>>>(
        expert_indices, expert_counts, total_tokens, params->top_k, params->num_experts
    );
    
    // 3) Dispatch
    sneppx_cuda_dispatch_to_experts(stream, params->input, expert_inputs,
                                   expert_indices, expert_counts,
                                   total_tokens, params->num_experts,
                                   params->expert_capacity, params->dim);
    
    // 4) Expert compute (GEMM per expert)
    for (int e = 0; e < params->num_experts; e++) {
        cublasHandle_t handle = sneppx_cublas_get_handle();
        cublasSetStream(handle, stream);
        
        float one = 1.0f, zero = 0.0f;
        cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            params->hidden_dim, params->expert_capacity, params->dim,
            &one,
            &params->expert_weights[e * params->hidden_dim * params->dim], CUDA_R_16F, params->hidden_dim,
            &expert_inputs[e * params->expert_capacity * params->dim], CUDA_R_16F, params->expert_capacity,
            &zero,
            &expert_outputs[e * params->expert_capacity * params->hidden_dim], CUDA_R_16F, params->expert_capacity,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    
    // 5) Combine
    sneppx_cuda_combine_from_experts(stream, params->output, expert_outputs,
                                    expert_indices, gating_weights,
                                    expert_counts, total_tokens, params->num_experts,
                                    params->hidden_dim, params->top_k);
    
    cudaFreeAsync(gating_weights, stream);
    cudaFreeAsync(expert_indices, stream);
    cudaFreeAsync(expert_counts, stream);
    cudaFreeAsync(expert_inputs, stream);
    cudaFreeAsync(expert_outputs, stream);
    
    return SNEPPX_CUDA_SUCCESS;
}

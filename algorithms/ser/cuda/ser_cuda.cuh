#ifndef SNEPPX_SER_CUDA_CUH
#define SNEPPX_SER_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../kernel/cuda/common.cuh"

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


// SER (Sparse Experts Router) CUDA Kernels
// - Expert routing (top-k gating)
// - Expert parallel dispatch
// - Load balancing loss
// - Expert weight updates

#ifdef __cplusplus
extern "C" {
#endif

// Top-k gating (router)
SNEPPX_CudaError sneppx_cuda_topk_gating(
    SNEPPX_CudaStream_t stream,
    const half* router_logits,     // [batch, num_experts]
    int* expert_indices,           // [batch, top_k]
    half* gating_weights,          // [batch, top_k]
    int batch_size,
    int num_experts,
    int top_k
);

// Expert dispatch (scatter tokens to experts)
SNEPPX_CudaError sneppx_cuda_dispatch_to_experts(
    SNEPPX_CudaStream_t stream,
    const half* tokens,            // [batch, seq, dim]
    half* expert_inputs,           // [num_experts, capacity, dim]
    const int* expert_indices,     // [batch * seq, top_k]
    const int* expert_counts,      // [num_experts] (prefix sum)
    int total_tokens,
    int num_experts,
    int expert_capacity,
    int dim
);

// Expert combine (gather from experts back)
SNEPPX_CudaError sneppx_cuda_combine_from_experts(
    SNEPPX_CudaStream_t stream,
    half* output,                    // [batch, seq, dim]
    const half* expert_outputs,      // [num_experts, capacity, dim]
    const int* expert_indices,
    const half* gating_weights,
    const int* expert_counts,
    int total_tokens,
    int num_experts,
    int dim,
    int top_k
);

// Load balancing auxiliary loss
SNEPPX_CudaError sneppx_cuda_load_balancing_loss(
    SNEPPX_CudaStream_t stream,
    const float* expert_counts,
    const float* router_prob,
    float* loss,
    int num_experts,
    float beta
);

// Expert parallelism: all-to-all dispatch (for distributed)
SNEPPX_CudaError sneppx_cuda_expert_alltoall(
    SNEPPX_CudaStream_t stream,
    const half* local_input,
    half* local_output,
    const int* send_counts,
    const int* recv_counts,
    int num_experts_per_rank,
    int dim,
    int world_size,
    int rank
);

// Sparse MoE forward (fused: gating + dispatch + compute + combine)
typedef struct {
    const half* input;
    half* output;
    const half* expert_weights;    // [num_experts, hidden_dim, dim]
    const half* router_weights;    // [dim, num_experts]
    int batch_size;
    int seq_len;
    int dim;
    int hidden_dim;
    int num_experts;
    int top_k;
    int expert_capacity;
    bool use_shared_expert;
} SNEPPX_FusedMoEParams;

SNEPPX_CudaError sneppx_cuda_fused_moe_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FusedMoEParams* params
);

#ifdef __cplusplus
}
#endif

#endif

#ifndef SNEPPX_DISTRIBUTED_H
#define SNEPPX_DISTRIBUTED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SNEPPX_HAS_CUDA
/*
 * SNEPPX - Distributed Training Configuration
 *
 * WHAT
 *   Distributed Training Configuration.
 *
 * CONCEPT
 *   Unified configuration for DP/TP/PP/EP sizes, ZeRO stage, NCCL dtype, pipeline microbatches.
 *
 * ROLE
 *   Declares the distributed training configuration API used by the kernel/distributed module.
 *
 * REFERENCES
 *   None (internal module).
 */


typedef void* cudaStream_t;
#endif

// ============================================================================
// Distributed Training Configuration
// ============================================================================

typedef struct {
    int world_size;
    int rank;
    int local_rank;
    int num_nodes;
    int node_id;
    
    // Parallelism strategies
    int data_parallel_size;
    int tensor_parallel_size;
    int pipeline_parallel_size;
    int expert_parallel_size;
    
    // ZeRO stage
    int zero_stage;  // 0=disabled, 1=optimizer, 2=gradient, 3=parameter
    
    // Communication
    int nccl_dtype;
    int overlap_comm;       // overlap communication with computation
    int gradient_sync_mode; // 0=all-reduce, 1=reduce-scatter, 2=hierarchical
    int bucket_size_mb;
    
    // Mixed precision
    int fp16_enabled;
    int bf16_enabled;
    int loss_scale;
    int dynamic_loss_scale;
    
    // Pipeline
    int pipeline_num_microbatches;
    int pipeline_chunks;
    
    // Expert parallel
    int num_experts;
    int num_local_experts;
    int expert_capacity_factor;
    
    // Misc
    int seed;
    int verbose;
    char master_addr[64];
    int master_port;
} SNEPPX_DistributedConfig;

// ============================================================================
// ZeRO Optimizer (Stage 1-3)
// ============================================================================

typedef struct {
    float* params_fp32;         // FP32 master weights (ZeRO-2/3)
    float* exp_avg;             // First moment
    float* exp_avg_sq;          // Second moment
    float* grad_buffer;         // Gradient buffer
    int* param_owners;          // Which rank owns each param
    int num_params;
    int world_size;
    int rank;
    int stage;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int step;
} SNEPPX_ZeroOptimizer;

int sneppx_zero_init(SNEPPX_ZeroOptimizer** opt, int num_params,
                     int world_size, int rank, int stage,
                     const SNEPPX_DistributedConfig* config);
int sneppx_zero_step(SNEPPX_ZeroOptimizer* opt, float* grads,
                     cudaStream_t stream);
int sneppx_zero_destroy(SNEPPX_ZeroOptimizer* opt);

// ============================================================================
// Pipeline Parallelism
// ============================================================================

typedef struct {
    int num_stages;
    int stage_id;
    int num_microbatches;
    int chunks;
    int fwd_bwd_overlap;
    void* (*forward_fn)(void* ctx, void* input, int microbatch);
    void* (*backward_fn)(void* ctx, void* grad_output, int microbatch);
    void* context;
} SNEPPX_PipelineParallel;

int sneppx_pipeline_init(SNEPPX_PipelineParallel** pp,
                         const SNEPPX_DistributedConfig* config);
int sneppx_pipeline_forward(SNEPPX_PipelineParallel* pp,
                             void* input, void** output,
                             cudaStream_t stream);
int sneppx_pipeline_backward(SNEPPX_PipelineParallel* pp,
                              void* grad_output,
                              cudaStream_t stream);
int sneppx_pipeline_destroy(SNEPPX_PipelineParallel* pp);

// ============================================================================
// Tensor Parallelism (row/column split)
// ============================================================================

typedef struct {
    int tp_size;
    int tp_rank;
    int use_nccl;
    void* nccl_comm;
} SNEPPX_TensorParallel;

int sneppx_tp_init(SNEPPX_TensorParallel** tp,
                   const SNEPPX_DistributedConfig* config);
int sneppx_tp_linear_forward(SNEPPX_TensorParallel* tp,
                              const float* input, const float* weight,
                              float* output, int m, int n, int k,
                              cudaStream_t stream);
int sneppx_tp_all_reduce(SNEPPX_TensorParallel* tp,
                          float* data, int size,
                          cudaStream_t stream);
int sneppx_tp_destroy(SNEPPX_TensorParallel* tp);

// ============================================================================
// Expert Parallelism (distributed MoE)
// ============================================================================

typedef struct {
    int num_experts;
    int num_local_experts;
    int ep_size;
    int ep_rank;
    int* expert_to_rank;  // [num_experts] mapping
    void* nccl_comm;
} SNEPPX_ExpertParallel;

int sneppx_ep_init(SNEPPX_ExpertParallel** ep,
                   const SNEPPX_DistributedConfig* config);
int sneppx_ep_all_to_all(SNEPPX_ExpertParallel* ep,
                          const float* sendbuf, float* recvbuf,
                          int send_size, int recv_size,
                          cudaStream_t stream);
int sneppx_ep_destroy(SNEPPX_ExpertParallel* ep);

// ============================================================================
// FM Distributed Communication
// ============================================================================

int sneppx_fm_distributed_all_reduce(float* data, int size,
                                      const SNEPPX_DistributedConfig* config,
                                      cudaStream_t stream);
int sneppx_fm_distributed_broadcast(float* data, int size, int root,
                                     const SNEPPX_DistributedConfig* config,
                                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_DISTRIBUTED_H
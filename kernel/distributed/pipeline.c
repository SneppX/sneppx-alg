#ifdef SNEPPX_HAS_CUDA
#include "../../include/neural_core/architecture/distributed.h"
#include "../../net/distributed/nccl.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Pipeline
 *
 * WHAT
 *   Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// Pipeline Parallelism (1F1B Schedule)
// ============================================================================

struct SNEPPX_PipelineParallel {
    int num_stages;
    int stage_id;
    int num_microbatches;
    int chunks;
    int fwd_bwd_overlap;
    void* context;
    
    // Forward/backward function pointers
    void* (*forward_fn)(void*, void*, int);
    void* (*backward_fn)(void*, void*, int);
    
    // Communication buffers
    float* recv_buffer;
    float* send_buffer;
    size_t buffer_size;
    SNEPPX_ProcessGroup* pg;
};

int sneppx_pipeline_init(SNEPPX_PipelineParallel** pp,
                         const SNEPPX_DistributedConfig* config) {
    if (!pp || !config) return -1;
    
    SNEPPX_PipelineParallel* p = (SNEPPX_PipelineParallel*)calloc(1, sizeof(SNEPPX_PipelineParallel));
    if (!p) return -1;
    
    p->num_stages = config->pipeline_parallel_size;
    p->stage_id = config->rank % config->pipeline_parallel_size;
    p->num_microbatches = config->pipeline_num_microbatches;
    p->chunks = config->pipeline_chunks;
    p->fwd_bwd_overlap = 1;
    p->buffer_size = 1024 * 1024;  // 1MB default

    cudaMalloc(&p->recv_buffer, p->buffer_size);
    cudaMalloc(&p->send_buffer, p->buffer_size);

    sneppx_nccl_initialize();
    sneppx_pg_create(&p->pg, p->num_stages, p->stage_id);

    *pp = p;
    return 0;
}

int sneppx_pipeline_forward(SNEPPX_PipelineParallel* pp,
                             void* input, void** output,
                             cudaStream_t stream) {
    if (!pp || !input) return -1;
    
    // 1F1B (One-Forward-One-Backward) schedule
    // For simplicity, sequential forward through microbatches
    
    int prev_stage = pp->stage_id - 1;
    int next_stage = pp->stage_id + 1;

    void* hidden = input;

    for (int mb = 0; mb < pp->num_microbatches; mb++) {
        // Receive activation from previous stage (if not first)
        if (pp->stage_id > 0) {
            if (pp->pg && prev_stage >= 0) {
                int ret = sneppx_nccl_recv(pp->recv_buffer,
                                          pp->buffer_size / sizeof(float),
                                          SNEPPX_NCCL_FLOAT, prev_stage,
                                          pp->pg->comms[0], stream);
                if (ret != 0) return ret;
            } else {
                cudaMemcpyAsync(pp->recv_buffer, hidden,
                                pp->buffer_size, cudaMemcpyDeviceToDevice, stream);
            }
            hidden = pp->recv_buffer;
        }

        // Forward through this stage
        if (pp->forward_fn) {
            hidden = pp->forward_fn(pp->context, hidden, mb);
        }

        // Send activation to next stage (if not last)
        if (pp->stage_id < pp->num_stages - 1) {
            if (pp->pg && next_stage < pp->num_stages && hidden) {
                int ret = sneppx_nccl_send(hidden,
                                           pp->buffer_size / sizeof(float),
                                           SNEPPX_NCCL_FLOAT, next_stage,
                                           pp->pg->comms[0], stream);
                if (ret != 0) return ret;
            } else if (hidden) {
                cudaMemcpyAsync(pp->send_buffer, hidden,
                                pp->buffer_size, cudaMemcpyDeviceToDevice, stream);
                hidden = pp->send_buffer;
            }
        }
    }

    *output = hidden;
    return 0;
}

int sneppx_pipeline_backward(SNEPPX_PipelineParallel* pp,
                              void* grad_output,
                              cudaStream_t stream) {
    if (!pp || !grad_output) return -1;
    
    void* grad = grad_output;
    int prev_stage = pp->stage_id - 1;
    int next_stage = pp->stage_id + 1;

    for (int mb = pp->num_microbatches - 1; mb >= 0; mb--) {
        // Receive gradient from next stage (if not last)
        if (pp->stage_id < pp->num_stages - 1) {
            if (pp->pg && next_stage < pp->num_stages) {
                int ret = sneppx_nccl_recv(pp->recv_buffer,
                                          pp->buffer_size / sizeof(float),
                                          SNEPPX_NCCL_FLOAT, next_stage,
                                          pp->pg->comms[0], stream);
                if (ret != 0) return ret;
            } else {
                cudaMemcpyAsync(pp->recv_buffer, grad,
                                pp->buffer_size, cudaMemcpyDeviceToDevice, stream);
            }
            grad = pp->recv_buffer;
        }

        if (pp->backward_fn) {
            grad = pp->backward_fn(pp->context, grad, mb);
        }

        // Send gradient to previous stage (if not first)
        if (pp->stage_id > 0) {
            if (pp->pg && prev_stage >= 0) {
                int ret = sneppx_nccl_send(grad,
                                           pp->buffer_size / sizeof(float),
                                           SNEPPX_NCCL_FLOAT, prev_stage,
                                           pp->pg->comms[0], stream);
                if (ret != 0) return ret;
            }
        }
    }

    return 0;
}

int sneppx_pipeline_destroy(SNEPPX_PipelineParallel* pp) {
    if (!pp) return -1;
    if (pp->pg) sneppx_pg_destroy(pp->pg);
    if (pp->recv_buffer) cudaFree(pp->recv_buffer);
    if (pp->send_buffer) cudaFree(pp->send_buffer);
    free(pp);
    return 0;
}
#endif
/*
 * SNEPPX - Pipeline Parallelism (1F1B)
 *
 * WHAT
 *   Pipeline Parallelism (1F1B).
 *
 * CONCEPT
 *   Pipeline Parallelism (1F1B) implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


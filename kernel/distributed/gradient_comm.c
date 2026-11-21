#ifdef SNEPPX_HAS_CUDA
#include "../../include/neural_core/architecture/distributed.h"
#include "../../net/distributed/nccl.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Gradient Comm
 *
 * WHAT
 *   Gradient Comm.
 *
 * CONCEPT
 *   Provides the Gradient Comm.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// Hierarchical All-Reduce (NVLink intra-node + RDMA inter-node)
// ============================================================================

typedef struct {
    int local_rank;
    int local_size;
    int global_rank;
    int global_size;
    int node_id;
    int num_nodes;
    
    // Intra-node NVLink communicator
    void* intra_comm;
    // Inter-node RDMA communicator
    void* inter_comm;
    
    cudaStream_t intra_stream;
    cudaStream_t inter_stream;
    
    float* intra_buffer;
    float* inter_buffer;
    size_t buffer_size;
    SNEPPX_ProcessGroup* pg;
} SNEPPX_HierarchicalAR;

int sneppx_hierarchical_ar_init(SNEPPX_HierarchicalAR** har,
                                 int local_rank, int local_size,
                                 int global_rank, int global_size,
                                 int node_id, int num_nodes) {
    if (!har) return -1;
    SNEPPX_HierarchicalAR* h = (SNEPPX_HierarchicalAR*)calloc(1, sizeof(SNEPPX_HierarchicalAR));
    if (!h) return -1;
    h->local_rank = local_rank;
    h->local_size = local_size;
    h->global_rank = global_rank;
    h->global_size = global_size;
    h->node_id = node_id;
    h->num_nodes = num_nodes;
    h->buffer_size = 256 * 1024 * 1024;
    cudaStreamCreate(&h->intra_stream);
    cudaStreamCreate(&h->inter_stream);
    cudaMalloc(&h->intra_buffer, h->buffer_size);
    cudaMalloc(&h->inter_buffer, h->buffer_size);
    sneppx_nccl_initialize();
    sneppx_pg_create(&h->pg, h->global_size, h->global_rank);
    *har = h;
    return 0;
}

void sneppx_hierarchical_ar_destroy(SNEPPX_HierarchicalAR* har) {
    if (!har) return;
    if (har->pg) sneppx_pg_destroy(har->pg);
    if (har->intra_buffer) cudaFree(har->intra_buffer);
    if (har->inter_buffer) cudaFree(har->inter_buffer);
    cudaStreamDestroy(har->intra_stream);
    cudaStreamDestroy(har->inter_stream);
    free(har);
}

// Two-step hierarchical all-reduce:
// Step 1: Reduce-scatter within node (NVLink)
// Step 2: All-reduce across nodes (RDMA)
// Step 3: All-gather within node (NVLink)
int sneppx_hierarchical_ar_all_reduce(SNEPPX_HierarchicalAR* har,
                                        float* data, size_t numel,
                                        cudaStream_t stream) {
    if (!har || !data) return -1;

    // Real all-reduce across all ranks via NCCL (CPU fallback if NCCL absent).
    // The previous implementation only copied buffers device-to-device within a
    // single process, which is a no-op for distributed training.
    if (!har->pg) return -1;
    return sneppx_pg_all_reduce(har->pg, data, numel,
                                SNEPPX_NCCL_FLOAT, SNEPPX_NCCL_SUM, stream);
}

// ============================================================================
// Gradient Compression (top-k sparsification + error feedback)
// ============================================================================

typedef struct {
    float* error_feedback;
    float* compressed_grads;
    int* compressed_indices;
    float compression_ratio;
    int numel;
    int k;
} SNEPPX_GradCompressor;

int sneppx_grad_compressor_init(SNEPPX_GradCompressor** gc,
                                 int numel, float compression_ratio) {
    if (!gc || numel <= 0) return -1;
    SNEPPX_GradCompressor* g = (SNEPPX_GradCompressor*)calloc(1, sizeof(SNEPPX_GradCompressor));
    if (!g) return -1;
    g->numel = numel;
    g->compression_ratio = compression_ratio;
    g->k = (int)(numel * compression_ratio);
    if (g->k < 1) g->k = 1;
    cudaMalloc(&g->error_feedback, numel * sizeof(float));
    cudaMemset(g->error_feedback, 0, numel * sizeof(float));
    cudaMalloc(&g->compressed_grads, g->k * sizeof(float));
    cudaMalloc(&g->compressed_indices, g->k * sizeof(int));
    *gc = g;
    return 0;
}

void sneppx_grad_compressor_destroy(SNEPPX_GradCompressor* gc) {
    if (!gc) return;
    if (gc->error_feedback) cudaFree(gc->error_feedback);
    if (gc->compressed_grads) cudaFree(gc->compressed_grads);
    if (gc->compressed_indices) cudaFree(gc->compressed_indices);
    free(gc);
}

__global__ void topk_compress_kernel(const float* grads, const float* error,
                                      float* values, int* indices,
                                      int numel, int k) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = bid * blockDim.x;
    int end = min(start + blockDim.x, numel);
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float best_val = -1e30f;
    int best_idx = -1;
    for (int i = start + tid; i < end; i += blockDim.x) {
        float v = fabsf(grads[i] + error[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    // Simple warp-level top-1 (block-level in practice uses shared memory merge)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float nv = __shfl_xor_sync(0xFFFFFFFF, best_val, offset);
        int ni = __shfl_xor_sync(0xFFFFFFFF, best_idx, offset);
        if (nv > best_val) { best_val = nv; best_idx = ni; }
    }
    if (tid == 0 && bid < k) {
        values[bid] = best_val;
        indices[bid] = best_idx;
    }
}

int sneppx_grad_compress(SNEPPX_GradCompressor* gc,
                          const float* grads, cudaStream_t stream) {
    if (!gc || !grads) return -1;
    // Add error feedback
    float* tmp;
    cudaMalloc(&tmp, gc->numel * sizeof(float));
    auto add_err = [] __global__ (const float* g, const float* e, float* o, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n) o[i] = g[i] + e[i];
    };
    add_err<<<(gc->numel + 255) / 256, 256, 0, stream>>>(grads, gc->error_feedback, tmp, gc->numel);
    topk_compress_kernel<<<gc->k, 256, 0, stream>>>(tmp, gc->error_feedback,
                                                      gc->compressed_grads, gc->compressed_indices,
                                                      gc->numel, gc->k);
    // Update error feedback
    auto update_err = [] __global__ (float* e, const float* g, const float* v, const int* idx, int k, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n) return;
        e[i] += g[i];
        for (int j = 0; j < k; j++) {
            if (idx[j] == i) { e[i] -= v[j]; break; }
        }
    };
    update_err<<<(gc->numel + 255) / 256, 256, 0, stream>>>(gc->error_feedback, grads,
                                                              gc->compressed_grads, gc->compressed_indices,
                                                              gc->k, gc->numel);
    cudaFree(tmp);
    return 0;
}

int sneppx_grad_decompress(SNEPPX_GradCompressor* gc,
                            float* output, cudaStream_t stream) {
    if (!gc || !output) return -1;
    cudaMemset(output, 0, gc->numel * sizeof(float));
    auto decompress = [] __global__ (float* o, const float* v, const int* idx, int k) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < k) o[idx[i]] = v[i];
    };
    decompress<<<(gc->k + 255) / 256, 256, 0, stream>>>(output, gc->compressed_grads,
                                                          gc->compressed_indices, gc->k);
    return 0;
}
#endif
/*
 * SNEPPX - Hierarchical Gradient Communication
 *
 * WHAT
 *   Hierarchical Gradient Communication.
 *
 * CONCEPT
 *   Hierarchical Gradient Communication implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


#ifdef SNEPPX_HAS_CUDA
#include "../../include/neural_core/architecture/distributed.h"
#include "../../net/distributed/nccl.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Ddp
 *
 * WHAT
 *   Ddp.
 *
 * CONCEPT
 *   Provides distributed data parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// Distributed Data Parallel (DDP)
// ============================================================================

typedef struct {
    float* grad_buffer;
    void* nccl_comm;
    int world_size;
    int rank;
    int bucket_size;
    int num_buckets;
    int overlap_comm;
    cudaStream_t comm_stream;
    cudaEvent_t* sync_events;
    SNEPPX_ProcessGroup* pg;
} SNEPPX_DDPState;

int sneppx_ddp_init(SNEPPX_DDPState** ddp, int world_size, int rank,
                    int bucket_size_mb, int overlap_comm) {
    if (!ddp) return -1;
    SNEPPX_DDPState* d = (SNEPPX_DDPState*)calloc(1, sizeof(SNEPPX_DDPState));
    if (!d) return -1;
    d->world_size = world_size;
    d->rank = rank;
    d->bucket_size = bucket_size_mb * 1024 * 1024;
    d->overlap_comm = overlap_comm;
    sneppx_nccl_initialize();
    sneppx_pg_create(&d->pg, d->world_size, d->rank);
    if (overlap_comm) {
        cudaStreamCreate(&d->comm_stream);
        d->num_buckets = 4;
        d->sync_events = (cudaEvent_t*)calloc(d->num_buckets, sizeof(cudaEvent_t));
        for (int i = 0; i < d->num_buckets; i++) cudaEventCreate(&d->sync_events[i]);
    }
    *ddp = d;
    return 0;
}

void sneppx_ddp_destroy(SNEPPX_DDPState* ddp) {
    if (!ddp) return;
    if (ddp->pg) sneppx_pg_destroy(ddp->pg);
    if (ddp->grad_buffer) cudaFree(ddp->grad_buffer);
    if (ddp->overlap_comm && ddp->sync_events) {
        for (int i = 0; i < ddp->num_buckets; i++) cudaEventDestroy(ddp->sync_events[i]);
        free(ddp->sync_events);
        cudaStreamDestroy(ddp->comm_stream);
    }
    free(ddp);
}

// Bucket-based gradient all-reduce with optional compute overlap
int sneppx_ddp_all_reduce_grads(SNEPPX_DDPState* ddp, float** grads,
                                 size_t* sizes, int num_grads,
                                 cudaStream_t compute_stream) {
    if (!ddp || !grads || !sizes) return -1;
    size_t total_bytes = 0;
    for (int i = 0; i < num_grads; i++) total_bytes += sizes[i];
    if (!ddp->grad_buffer) cudaMalloc(&ddp->grad_buffer, total_bytes);
    size_t offset = 0;
    for (int i = 0; i < num_grads; i++) {
        cudaMemcpyAsync((char*)ddp->grad_buffer + offset, grads[i],
                        sizes[i], cudaMemcpyDeviceToDevice, compute_stream);
        offset += sizes[i];
    }
    cudaStreamSynchronize(compute_stream);
    // Real all-reduce across data-parallel ranks via NCCL (CPU fallback if NCCL absent)
    if (ddp->pg) {
        int ret = sneppx_pg_all_reduce(ddp->pg, ddp->grad_buffer,
                                       total_bytes / sizeof(float),
                                       SNEPPX_NCCL_FLOAT, SNEPPX_NCCL_SUM,
                                       compute_stream);
        if (ret != 0) return ret;
    }
    offset = 0;
    for (int i = 0; i < num_grads; i++) {
        cudaMemcpy(grads[i], (float*)ddp->grad_buffer + offset,
                   sizes[i], cudaMemcpyDeviceToDevice);
        offset += sizes[i];
    }
    return 0;
}

// Bucket-level all-reduce with async compute overlap
int sneppx_ddp_bucket_all_reduce(SNEPPX_DDPState* ddp, int bucket_id,
                                  float* data, size_t size,
                                  cudaStream_t compute_stream) {
    if (!ddp || !data) return -1;
    if (ddp->overlap_comm && bucket_id > 0) {
        cudaEventSynchronize(ddp->sync_events[bucket_id - 1]);
    }
    cudaMemcpyAsync(data, data, size, cudaMemcpyDeviceToDevice, ddp->comm_stream);
    if (ddp->pg) {
        int ret = sneppx_pg_all_reduce(ddp->pg, data, size / sizeof(float),
                                       SNEPPX_NCCL_FLOAT, SNEPPX_NCCL_SUM,
                                       ddp->comm_stream);
        if (ret != 0) return ret;
    }
    if (ddp->overlap_comm) {
        cudaEventRecord(ddp->sync_events[bucket_id], ddp->comm_stream);
    }
    return 0;
}
#endif
/*
 * SNEPPX - Distributed Data Parallel
 *
 * WHAT
 *   Distributed Data Parallel.
 *
 * CONCEPT
 *   Distributed Data Parallel implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


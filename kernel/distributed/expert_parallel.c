#ifdef SNEPPX_HAS_CUDA
#include "../../include/neural_core/architecture/distributed.h"
#include "../../net/distributed/nccl.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Expert Parallel
 *
 * WHAT
 *   Expert Parallel.
 *
 * CONCEPT
 *   Provides tensor/expert parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// Expert Parallelism (Distributed MoE via All-to-All)
// ============================================================================

struct SNEPPX_ExpertParallel {
    int num_experts;
    int num_local_experts;
    int ep_size;
    int ep_rank;
    int* expert_to_rank;
    int* local_expert_ids;
    SNEPPX_ProcessGroup* pg;
};

int sneppx_ep_init(SNEPPX_ExpertParallel** ep,
                   const SNEPPX_DistributedConfig* config) {
    if (!ep || !config) return -1;
    
    SNEPPX_ExpertParallel* e = (SNEPPX_ExpertParallel*)calloc(1, sizeof(SNEPPX_ExpertParallel));
    if (!e) return -1;
    
    e->num_experts = config->num_experts;
    e->num_local_experts = config->num_experts / config->expert_parallel_size;
    e->ep_size = config->expert_parallel_size;
    e->ep_rank = config->rank % config->expert_parallel_size;

    sneppx_nccl_initialize();
    sneppx_pg_create(&e->pg, e->ep_size, e->ep_rank);
    
    // Build expert-to-rank mapping
    e->expert_to_rank = (int*)calloc(config->num_experts, sizeof(int));
    e->local_expert_ids = (int*)calloc(e->num_local_experts, sizeof(int));
    
    for (int i = 0; i < config->num_experts; i++) {
        e->expert_to_rank[i] = i % config->expert_parallel_size;
    }
    
    int local_idx = 0;
    for (int i = e->ep_rank; i < config->num_experts; i += config->expert_parallel_size) {
        if (local_idx < e->num_local_experts) {
            e->local_expert_ids[local_idx++] = i;
        }
    }
    
    *ep = e;
    return 0;
}

// All-to-All communication for expert dispatch
// sendbuf: [ep_size, send_size] per rank, recvbuf: [ep_size, recv_size]
int sneppx_ep_all_to_all(SNEPPX_ExpertParallel* ep,
                          const float* sendbuf, float* recvbuf,
                          int send_size, int recv_size,
                          cudaStream_t stream) {
    if (!ep || !sendbuf || !recvbuf || !ep->pg) return -1;

    // Real expert-parallel all-to-all: each rank sends its slice to every peer
    // and receives every peer's slice, using NCCL send/recv (enqueued on the
    // stream and matched by the NCCL scheduler).
    SNEPPX_NCCLComm* comm = ep->pg->comms[0];
    for (int p = 0; p < ep->ep_size; p++) {
        if (p == ep->ep_rank) continue;
        int ret = sneppx_nccl_send(sendbuf + (size_t)p * send_size, send_size,
                                   SNEPPX_NCCL_FLOAT, p, comm, stream);
        if (ret != 0) return ret;
    }
    for (int p = 0; p < ep->ep_size; p++) {
        if (p == ep->ep_rank) continue;
        int ret = sneppx_nccl_recv(recvbuf + (size_t)p * recv_size, recv_size,
                                   SNEPPX_NCCL_FLOAT, p, comm, stream);
        if (ret != 0) return ret;
    }
    // Local self-copy
    if (ep->ep_size > 0) {
        cudaMemcpyAsync(recvbuf + (size_t)ep->ep_rank * recv_size,
                        sendbuf + (size_t)ep->ep_rank * send_size,
                        (size_t)recv_size * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    return 0;
}

int sneppx_ep_destroy(SNEPPX_ExpertParallel* ep) {
    if (!ep) return -1;
    if (ep->pg) sneppx_pg_destroy(ep->pg);
    free(ep->expert_to_rank);
    free(ep->local_expert_ids);
    free(ep);
    return 0;
}

// ============================================================================
// FM Distributed Communication
// ============================================================================

// Lazily-created, cached process group for FM distributed communication.
static SNEPPX_ProcessGroup* g_fm_pg = NULL;
static int g_fm_world = -1;
static int g_fm_rank = -1;

static int fm_get_pg(const SNEPPX_DistributedConfig* config, SNEPPX_ProcessGroup** out) {
    if (!config) return -1;
    if (!g_fm_pg || g_fm_world != config->world_size || g_fm_rank != config->rank) {
        if (g_fm_pg) sneppx_pg_destroy(g_fm_pg);
        sneppx_nccl_initialize();
        if (sneppx_pg_create(&g_fm_pg, config->world_size, config->rank) != 0) return -1;
        g_fm_world = config->world_size;
        g_fm_rank = config->rank;
    }
    *out = g_fm_pg;
    return 0;
}

int sneppx_fm_distributed_all_reduce(float* data, int size,
                                       const SNEPPX_DistributedConfig* config,
                                       cudaStream_t stream) {
    if (!data || !config) return -1;
    SNEPPX_ProcessGroup* pg;
    if (fm_get_pg(config, &pg) != 0 || !pg) return -1;
    return sneppx_pg_all_reduce(pg, data, (size_t)size,
                                SNEPPX_NCCL_FLOAT, SNEPPX_NCCL_SUM, stream);
}

int sneppx_fm_distributed_broadcast(float* data, int size, int root,
                                     const SNEPPX_DistributedConfig* config,
                                     cudaStream_t stream) {
    if (!data || !config) return -1;
    SNEPPX_ProcessGroup* pg;
    if (fm_get_pg(config, &pg) != 0 || !pg) return -1;
    return sneppx_nccl_broadcast(data, data, (size_t)size,
                                SNEPPX_NCCL_FLOAT, root,
                                pg->comms[0], stream);
}
#endif
/*
 * SNEPPX - Expert Parallelism
 *
 * WHAT
 *   Expert Parallelism.
 *
 * CONCEPT
 *   Expert Parallelism implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


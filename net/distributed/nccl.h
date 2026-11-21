#ifndef SNEPPX_NCCL_H
#define SNEPPX_NCCL_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Nccl
 *
 * WHAT
 *   Nccl.
 *
 * CONCEPT
 *   Provides NCCL communication layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ============================================================================
// NCCL Communication Primitives
// ============================================================================

typedef enum {
    SNEPPX_NCCL_SUM = 0,
    SNEPPX_NCCL_PROD = 1,
    SNEPPX_NCCL_MAX = 2,
    SNEPPX_NCCL_MIN = 3,
    SNEPPX_NCCL_AVG = 4,
} SNEPPX_NCCL_RedOp;

typedef enum {
    SNEPPX_NCCL_FLOAT = 0,
    SNEPPX_NCCL_HALF = 1,
    SNEPPX_NCCL_INT = 2,
    SNEPPX_NCCL_INT64 = 3,
    SNEPPX_NCCL_FLOAT16 = 4,
    SNEPPX_NCCL_BFLOAT16 = 5,
} SNEPPX_NCCL_DataType;

typedef struct SNEPPX_NCCLComm SNEPPX_NCCLComm;

// Initialize NCCL
int sneppx_nccl_initialize(void);

// Finalize NCCL
int sneppx_nccl_finalize(void);

// Create communicator
int sneppx_nccl_comm_init_rank(
    SNEPPX_NCCLComm** comm,
    int ndev,
    int rank,
    int* devs
);

// Destroy communicator
int sneppx_nccl_comm_destroy(SNEPPX_NCCLComm* comm);

// Get rank and world size
int sneppx_nccl_comm_rank(const SNEPPX_NCCLComm* comm, int* rank);
int sneppx_nccl_comm_size(const SNEPPX_NCCLComm* comm, int* size);

// All-reduce
int sneppx_nccl_all_reduce(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    SNEPPX_NCCL_DataType datatype,
    SNEPPX_NCCL_RedOp op,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

// All-gather
int sneppx_nccl_all_gather(
    const void* sendbuf,
    void* recvbuf,
    size_t sendcount,
    SNEPPX_NCCL_DataType datatype,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

// Reduce-scatter
int sneppx_nccl_reduce_scatter(
    const void* sendbuf,
    void* recvbuf,
    size_t recvcount,
    SNEPPX_NCCL_DataType datatype,
    SNEPPX_NCCL_RedOp op,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

// Broadcast
int sneppx_nccl_broadcast(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    SNEPPX_NCCL_DataType datatype,
    int root,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

// Reduce
int sneppx_nccl_reduce(
    const void* sendbuf,
    void* recvbuf,
    size_t count,
    SNEPPX_NCCL_DataType datatype,
    SNEPPX_NCCL_RedOp op,
    int root,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

// Send / Recv
int sneppx_nccl_send(
    const void* buf,
    size_t count,
    SNEPPX_NCCL_DataType datatype,
    int peer,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

int sneppx_nccl_recv(
    void* buf,
    size_t count,
    SNEPPX_NCCL_DataType datatype,
    int peer,
    SNEPPX_NCCLComm* comm,
    cudaStream_t stream
);

// ============================================================================
// Distributed Optimizer (ZeRO) helpers
// ============================================================================

// All-reduce gradients (fused for multiple tensors)
int sneppx_nccl_all_reduce_grads(
    SNEPPX_NCCLComm* comm,
    void** grads,
    size_t* sizes,
    int num_grads,
    cudaStream_t stream
);

// ============================================================================
// Error handling
// ============================================================================

const char* sneppx_nccl_get_error_string(int error);

// ============================================================================
// Process Group Management (higher-level)
// ============================================================================

typedef struct {
    SNEPPX_NCCLComm** comms;
    int num_comms;
    int world_size;
    int rank;
} SNEPPX_ProcessGroup;

int sneppx_pg_create(
    SNEPPX_ProcessGroup** pg,
    int world_size,
    int rank
);

int sneppx_pg_destroy(SNEPPX_ProcessGroup* pg);

int sneppx_pg_barrier(
    SNEPPX_ProcessGroup* pg,
    cudaStream_t stream
);

// Barrier across all ranks
int sneppx_pg_all_reduce(
    SNEPPX_ProcessGroup* pg,
    void* data,
    size_t count,
    SNEPPX_NCCL_DataType datatype,
    SNEPPX_NCCL_RedOp op,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_NCCL_H

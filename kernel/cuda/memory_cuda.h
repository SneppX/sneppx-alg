#ifndef SNEPPX_MEMORY_CUDA_H
#define SNEPPX_MEMORY_CUDA_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.cuh"

#ifdef __cplusplus
/*
 * SNEPPX - Memory Cuda
 *
 * WHAT
 *   Memory Cuda.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ============================================================================
// Memory Pool (reduces cudaMalloc/cudaFree overhead)
// ============================================================================

typedef struct {
    // Pre-allocated blocks
    void** blocks;
    size_t* block_sizes;
    int* block_used;
    int num_blocks;
    int max_blocks;
    
    // Pool configuration
    size_t block_size;
    size_t total_capacity;
    size_t total_used;
    
    // Device
    int device_id;
} SNEPPX_MemoryPool;

// Create a memory pool
SNEPPX_CudaError sneppx_mempool_create(
    SNEPPX_MemoryPool** pool,
    int device_id,
    size_t block_size,
    int initial_blocks,
    int max_blocks
);

// Destroy memory pool
SNEPPX_CudaError sneppx_mempool_destroy(
    SNEPPX_MemoryPool* pool
);

// Allocate from pool (returns nullptr if full)
SNEPPX_CudaError sneppx_mempool_alloc(
    SNEPPX_MemoryPool* pool,
    void** ptr,
    size_t size
);

// Free back to pool
SNEPPX_CudaError sneppx_mempool_free(
    SNEPPX_MemoryPool* pool,
    void* ptr
);

// Reset all allocations
SNEPPX_CudaError sneppx_mempool_reset(
    SNEPPX_MemoryPool* pool
);

// Get pool statistics
typedef struct {
    size_t total_capacity;
    size_t total_used;
    int num_blocks;
    int num_free_blocks;
    int device_id;
} SNEPPX_MemoryPoolStats;

SNEPPX_CudaError sneppx_mempool_get_stats(
    const SNEPPX_MemoryPool* pool,
    SNEPPX_MemoryPoolStats* stats
);

// ============================================================================
// CUDA Stream Pool (for multi-stream concurrency)
// ============================================================================

typedef struct {
    cudaStream_t* streams;
    int* stream_available;
    int num_streams;
    int max_streams;
} SNEPPX_StreamPool;

// Create stream pool
SNEPPX_CudaError sneppx_stream_pool_create(
    SNEPPX_StreamPool** pool,
    int num_streams,
    unsigned int stream_flags
);

// Destroy stream pool
SNEPPX_CudaError sneppx_stream_pool_destroy(
    SNEPPX_StreamPool* pool
);

// Acquire a stream from pool (round-robin)
SNEPPX_CudaError sneppx_stream_pool_acquire(
    SNEPPX_StreamPool* pool,
    cudaStream_t* stream
);

// Release stream back to pool
SNEPPX_CudaError sneppx_stream_pool_release(
    SNEPPX_StreamPool* pool,
    cudaStream_t stream
);

// Synchronize all streams in pool
SNEPPX_CudaError sneppx_stream_pool_sync_all(
    SNEPPX_StreamPool* pool
);

// ============================================================================
// CUDA Event Pool (for timing/synchronization)
// ============================================================================

typedef struct {
    cudaEvent_t* events;
    int* event_available;
    int num_events;
    int max_events;
} SNEPPX_EventPool;

SNEPPX_CudaError sneppx_event_pool_create(
    SNEPPX_EventPool** pool,
    int num_events,
    unsigned int event_flags
);

SNEPPX_CudaError sneppx_event_pool_destroy(
    SNEPPX_EventPool* pool
);

SNEPPX_CudaError sneppx_event_pool_acquire(
    SNEPPX_EventPool* pool,
    cudaEvent_t* event
);

SNEPPX_CudaError sneppx_event_pool_release(
    SNEPPX_EventPool* pool,
    cudaEvent_t event
);

// ============================================================================
// Asynchronous Memory Operations
// ============================================================================

// Async memcpy with stream
SNEPPX_CudaError sneppx_cuda_memcpy_2d_async(
    SNEPPX_CudaStream_t stream,
    void* dst,
    size_t dpitch,
    const void* src,
    size_t spitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind
);

// Batched async memcpy (e.g., for gradient communication)
typedef struct {
    void* dst;
    const void* src;
    size_t size;
    cudaMemcpyKind kind;
} SNEPPX_MemcpyBatchEntry;

SNEPPX_CudaError sneppx_cuda_memcpy_batched_async(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_MemcpyBatchEntry* entries,
    int num_entries
);

// ============================================================================
// Pinned Memory (Host-side)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_alloc_pinned(
    void** ptr,
    size_t size
);

SNEPPX_CudaError sneppx_cuda_free_pinned(
    void* ptr
);

// ============================================================================
// Unified Memory
// ============================================================================

SNEPPX_CudaError sneppx_cuda_alloc_managed(
    void** ptr,
    size_t size,
    unsigned int flags
);

// Prefetch managed memory to device
SNEPPX_CudaError sneppx_cuda_mem_prefetch(
    void* ptr,
    size_t size,
    int device_id,
    SNEPPX_CudaStream_t stream
);

// ============================================================================
// Memory Statistics
// ============================================================================

typedef struct {
    size_t total_global;
    size_t free_global;
    size_t used_global;
    size_t total_managed;
    size_t free_managed;
    size_t total_pinned;
    size_t free_pinned;
} SNEPPX_MemoryStats;

SNEPPX_CudaError sneppx_cuda_get_memory_stats(
    SNEPPX_MemoryStats* stats
);

// ============================================================================
// Kernel Launch Utilities
// ============================================================================

// Auto-tune block size for a given kernel
typedef int (*SNEPPX_KernelBlockFunc)(int numel);

int sneppx_kernel_auto_block_size(int numel, int max_block_size);

// ============================================================================
// CUBLAS Handle (TLS singleton)
// ============================================================================

// These are declared in common.cuh, implemented here
cublasHandle_t sneppx_cublas_get_handle(void);
void sneppx_cublas_destroy_handle(void);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_MEMORY_CUDA_H

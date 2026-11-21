#include "memory_cuda.h"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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


// ============================================================================
// CUBLAS Handle (TLS singleton)
// ============================================================================

#include <cublas_v2.h>

// Thread-local CUBLAS handle
__thread cublasHandle_t tls_cublas_handle = nullptr;

cublasHandle_t sneppx_cublas_get_handle(void) {
    if (!tls_cublas_handle) {
        cublasCreate(&tls_cublas_handle);
        cublasSetMathMode(tls_cublas_handle, CUBLAS_TENSOR_OP_MATH);
        
        // Enable TF32 for tensor cores (Ampere+)
        #if CUBLAS_VERSION >= 11000
        cublasSetMathMode(tls_cublas_handle, CUBLAS_DEFAULT_MATH);
        #endif
    }
    return tls_cublas_handle;
}

void sneppx_cublas_destroy_handle(void) {
    if (tls_cublas_handle) {
        cublasDestroy(tls_cublas_handle);
        tls_cublas_handle = nullptr;
    }
}

// ============================================================================
// Device Properties
// ============================================================================

SNEPPX_CudaError sneppx_cuda_get_device_props(int device_id, SNEPPX_DeviceProps* props) {
    if (!props) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return SNEPPX_CUDA_ERROR_INVALID_DEVICE;
    
    props->device_id = device_id;
    snprintf(props->name, sizeof(props->name), "%s", prop.name);
    props->compute_capability_major = prop.major;
    props->compute_capability_minor = prop.minor;
    props->global_mem_bytes = prop.totalGlobalMem;
    props->shared_mem_per_block = prop.sharedMemPerBlock;
    props->shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    props->max_threads_per_block = prop.maxThreadsPerBlock;
    props->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    props->max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    props->warp_size = prop.warpSize;
    props->num_sms = prop.multiProcessorCount;
    props->clock_rate_khz = prop.clockRate;
    props->memory_clock_rate_khz = prop.memoryClockRate;
    props->memory_bus_width = prop.memoryBusWidth;
    props->l2_cache_size = prop.l2CacheSize;
    props->max_shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_INVALID_DEVICE;
}

SNEPPX_CudaError sneppx_cuda_get_device(int* device_id) {
    cudaError_t err = cudaGetDevice(device_id);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_INVALID_DEVICE;
}

SNEPPX_CudaError sneppx_cuda_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Memory Pool Implementation
// ============================================================================

SNEPPX_CudaError sneppx_mempool_create(
    SNEPPX_MemoryPool** pool,
    int device_id,
    size_t block_size,
    int initial_blocks,
    int max_blocks
) {
    if (!pool || initial_blocks <= 0 || max_blocks <= 0) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    SNEPPX_MemoryPool* p = (SNEPPX_MemoryPool*)malloc(sizeof(SNEPPX_MemoryPool));
    if (!p) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    
    p->device_id = device_id;
    p->block_size = block_size;
    p->num_blocks = 0;
    p->max_blocks = max_blocks;
    p->total_capacity = (size_t)max_blocks * block_size;
    p->total_used = 0;
    
    p->blocks = (void**)calloc(max_blocks, sizeof(void*));
    p->block_sizes = (size_t*)calloc(max_blocks, sizeof(size_t));
    p->block_used = (int*)calloc(max_blocks, sizeof(int));
    
    if (!p->blocks || !p->block_sizes || !p->block_used) {
        free(p->blocks);
        free(p->block_sizes);
        free(p->block_used);
        free(p);
        return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    cudaSetDevice(device_id);
    
    // Pre-allocate initial blocks
    for (int i = 0; i < initial_blocks; i++) {
        cudaError_t err = cudaMalloc(&p->blocks[i], block_size);
        if (err != cudaSuccess) break;
        p->block_sizes[i] = block_size;
        p->block_used[i] = 0;
        p->num_blocks++;
    }
    
    *pool = p;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mempool_destroy(SNEPPX_MemoryPool* pool) {
    if (!pool) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaSetDevice(pool->device_id);
    
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i]) {
            cudaFree(pool->blocks[i]);
        }
    }
    
    free(pool->blocks);
    free(pool->block_sizes);
    free(pool->block_used);
    free(pool);
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mempool_alloc(
    SNEPPX_MemoryPool* pool,
    void** ptr,
    size_t size
) {
    if (!pool || !ptr || size == 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Find a free block large enough
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->block_used[i] && pool->block_sizes[i] >= size) {
            pool->block_used[i] = 1;
            pool->total_used += pool->block_sizes[i];
            *ptr = pool->blocks[i];
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    // Allocate new block if possible
    if (pool->num_blocks < pool->max_blocks) {
        int i = pool->num_blocks;
        cudaError_t err = cudaMalloc(&pool->blocks[i], size);
        if (err != cudaSuccess) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
        
        pool->block_sizes[i] = size;
        pool->block_used[i] = 1;
        pool->total_used += size;
        pool->num_blocks++;
        *ptr = pool->blocks[i];
        return SNEPPX_CUDA_SUCCESS;
    }
    
    // Pool exhausted, fallback to direct allocation
    return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
}

SNEPPX_CudaError sneppx_mempool_free(
    SNEPPX_MemoryPool* pool,
    void* ptr
) {
    if (!pool || !ptr) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i] == ptr) {
            pool->block_used[i] = 0;
            pool->total_used -= pool->block_sizes[i];
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    // Not in pool, free directly
    cudaFree(ptr);
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mempool_reset(SNEPPX_MemoryPool* pool) {
    if (!pool) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_blocks; i++) {
        pool->block_used[i] = 0;
    }
    pool->total_used = 0;
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mempool_get_stats(
    const SNEPPX_MemoryPool* pool,
    SNEPPX_MemoryPoolStats* stats
) {
    if (!pool || !stats) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    stats->total_capacity = pool->total_capacity;
    stats->total_used = pool->total_used;
    stats->num_blocks = pool->num_blocks;
    
    int free_count = 0;
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->block_used[i]) free_count++;
    }
    stats->num_free_blocks = free_count;
    stats->device_id = pool->device_id;
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Stream Pool Implementation
// ============================================================================

SNEPPX_CudaError sneppx_stream_pool_create(
    SNEPPX_StreamPool** pool,
    int num_streams,
    unsigned int stream_flags
) {
    if (!pool || num_streams <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    SNEPPX_StreamPool* p = (SNEPPX_StreamPool*)malloc(sizeof(SNEPPX_StreamPool));
    if (!p) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    
    p->num_streams = 0;
    p->max_streams = num_streams;
    p->streams = (cudaStream_t*)calloc(num_streams, sizeof(cudaStream_t));
    p->stream_available = (int*)calloc(num_streams, sizeof(int));
    
    if (!p->streams || !p->stream_available) {
        free(p->streams);
        free(p->stream_available);
        free(p);
        return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    for (int i = 0; i < num_streams; i++) {
        cudaError_t err = cudaStreamCreateWithFlags(&p->streams[i], stream_flags);
        if (err != cudaSuccess) break;
        p->stream_available[i] = 1;
        p->num_streams++;
    }
    
    *pool = p;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_stream_pool_destroy(SNEPPX_StreamPool* pool) {
    if (!pool) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_streams; i++) {
        cudaStreamDestroy(pool->streams[i]);
    }
    
    free(pool->streams);
    free(pool->stream_available);
    free(pool);
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_stream_pool_acquire(
    SNEPPX_StreamPool* pool,
    cudaStream_t* stream
) {
    if (!pool || !stream) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Round-robin: find first available stream
    for (int i = 0; i < pool->num_streams; i++) {
        if (pool->stream_available[i]) {
            pool->stream_available[i] = 0;
            *stream = pool->streams[i];
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    // All busy, create temporary stream
    cudaError_t err = cudaStreamCreate(stream);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_stream_pool_release(
    SNEPPX_StreamPool* pool,
    cudaStream_t stream
) {
    if (!pool || !stream) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_streams; i++) {
        if (pool->streams[i] == stream) {
            pool->stream_available[i] = 1;
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    // Not in pool, destroy directly
    cudaStreamDestroy(stream);
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_stream_pool_sync_all(SNEPPX_StreamPool* pool) {
    if (!pool) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_streams; i++) {
        cudaStreamSynchronize(pool->streams[i]);
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Event Pool Implementation
// ============================================================================

SNEPPX_CudaError sneppx_event_pool_create(
    SNEPPX_EventPool** pool,
    int num_events,
    unsigned int event_flags
) {
    if (!pool || num_events <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    SNEPPX_EventPool* p = (SNEPPX_EventPool*)malloc(sizeof(SNEPPX_EventPool));
    if (!p) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    
    p->num_events = 0;
    p->max_events = num_events;
    p->events = (cudaEvent_t*)calloc(num_events, sizeof(cudaEvent_t));
    p->event_available = (int*)calloc(num_events, sizeof(int));
    
    if (!p->events || !p->event_available) {
        free(p->events);
        free(p->event_available);
        free(p);
        return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    for (int i = 0; i < num_events; i++) {
        cudaError_t err = cudaEventCreateWithFlags(&p->events[i], event_flags);
        if (err != cudaSuccess) break;
        p->event_available[i] = 1;
        p->num_events++;
    }
    
    *pool = p;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_event_pool_destroy(SNEPPX_EventPool* pool) {
    if (!pool) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_events; i++) {
        cudaEventDestroy(pool->events[i]);
    }
    
    free(pool->events);
    free(pool->event_available);
    free(pool);
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_event_pool_acquire(
    SNEPPX_EventPool* pool,
    cudaEvent_t* event
) {
    if (!pool || !event) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_events; i++) {
        if (pool->event_available[i]) {
            pool->event_available[i] = 0;
            *event = pool->events[i];
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    cudaError_t err = cudaEventCreate(event);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_event_pool_release(
    SNEPPX_EventPool* pool,
    cudaEvent_t event
) {
    if (!pool || !event) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    for (int i = 0; i < pool->num_events; i++) {
        if (pool->events[i] == event) {
            pool->event_available[i] = 1;
            return SNEPPX_CUDA_SUCCESS;
        }
    }
    
    cudaEventDestroy(event);
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Pinned Memory
// ============================================================================

SNEPPX_CudaError sneppx_cuda_alloc_pinned(void** ptr, size_t size) {
    if (!ptr) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
}

SNEPPX_CudaError sneppx_cuda_free_pinned(void* ptr) {
    if (!ptr) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaError_t err = cudaFreeHost(ptr);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Unified Memory
// ============================================================================

SNEPPX_CudaError sneppx_cuda_alloc_managed(void** ptr, size_t size, unsigned int flags) {
    if (!ptr) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaError_t err = cudaMallocManaged(ptr, size, flags);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
}

SNEPPX_CudaError sneppx_cuda_mem_prefetch(
    void* ptr, size_t size, int device_id, SNEPPX_CudaStream_t stream
) {
    if (!ptr) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaError_t err = cudaMemPrefetchAsync(ptr, size, device_id, stream);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Memory Statistics
// ============================================================================

SNEPPX_CudaError sneppx_cuda_get_memory_stats(SNEPPX_MemoryStats* stats) {
    if (!stats) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    size_t free_byte, total_byte;
    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
    if (err != cudaSuccess) return SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
    
    stats->total_global = total_byte;
    stats->free_global = free_byte;
    stats->used_global = total_byte - free_byte;
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Kernel Auto Block Size
// ============================================================================

int sneppx_kernel_auto_block_size(int numel, int max_block_size) {
    if (numel <= 0) return 32;
    
    // Heuristic: scale block size based on problem size
    if (numel >= 1024 * 1024) return 256;
    if (numel >= 256 * 256) return 128;
    if (numel >= 64 * 64) return 64;
    return 32;
}

// ============================================================================
// Async 2D Memcpy
// ============================================================================

SNEPPX_CudaError sneppx_cuda_memcpy_2d_async(
    SNEPPX_CudaStream_t stream,
    void* dst, size_t dpitch,
    const void* src, size_t spitch,
    size_t width, size_t height,
    cudaMemcpyKind kind
) {
    cudaError_t err = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Batched Async Memcpy
// ============================================================================

__global__ void batched_memcpy_kernel(
    SNEPPX_MemcpyBatchEntry* entries,
    int num_entries
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int entry_idx = idx;
    
    if (entry_idx < num_entries) {
        auto& entry = entries[entry_idx];
        // On-device memcpy (for device-to-device entries)
        for (size_t i = 0; i < entry.size; i += sizeof(uint4)) {
            size_t remaining = min(entry.size - i, sizeof(uint4));
            if (remaining == sizeof(uint4)) {
                uint4 val = *(const uint4*)((const char*)entry.src + i);
                *(uint4*)((char*)entry.dst + i) = val;
            }
        }
    }
}

SNEPPX_CudaError sneppx_cuda_memcpy_batched_async(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_MemcpyBatchEntry* entries,
    int num_entries
) {
    if (!entries || num_entries <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Use host-side loop for simplicity (could be optimized with on-device list)
    for (int i = 0; i < num_entries; i++) {
        const auto& e = entries[i];
        if (e.kind == cudaMemcpyDeviceToDevice) {
            cudaMemcpyAsync(e.dst, e.src, e.size, cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyAsync(e.dst, e.src, e.size, e.kind, stream);
        }
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

#ifndef SNEPPX_CUDA_COMMON_CUH
#define SNEPPX_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

/*
 * SNEPPX - Common
 *
 * WHAT
 *   Common.
 *
 * CONCEPT
 *   Provides the Common.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// Architecture Detection & Feature Flags
// ============================================================================

#if __CUDA_ARCH__ >= 900
    #define SNEPPX_HOPPER 1
    #define SNEPPX_TENSOR_CORE_MMA_M 16
    #define SNEPPX_TENSOR_CORE_MMA_N 8
    #define SNEPPX_TENSOR_CORE_MMA_K 16
    #define SNEPPX_HAS_TMA 1
    #define SNEPPX_HAS_CLUSTER 1
    #define SNEPPX_HAS_FP8 1
    #define SNEPPX_HAS_WGMMA 1
#elif __CUDA_ARCH__ >= 800
    #define SNEPPX_AMPERE 1
    #define SNEPPX_TENSOR_CORE_MMA_M 16
    #define SNEPPX_TENSOR_CORE_MMA_N 8
    #define SNEPPX_TENSOR_CORE_MMA_K 16
    #define SNEPPX_HAS_TMA 0
    #define SNEPPX_HAS_CLUSTER 0
    #define SNEPPX_HAS_FP8 0
    #define SNEPPX_HAS_WGMMA 0
#elif __CUDA_ARCH__ >= 700
    #define SNEPPX_VOLTA 1
    #define SNEPPX_TENSOR_CORE_MMA_M 16
    #define SNEPPX_TENSOR_CORE_MMA_N 8
    #define SNEPPX_TENSOR_CORE_MMA_K 16
    #define SNEPPX_HAS_TMA 0
    #define SNEPPX_HAS_CLUSTER 0
    #define SNEPPX_HAS_FP8 0
    #define SNEPPX_HAS_WGMMA 0
#else
    #define SNEPPX_PASCAL 1
    #define SNEPPX_TENSOR_CORE_MMA_M 0
    #define SNEPPX_TENSOR_CORE_MMA_N 0
    #define SNEPPX_TENSOR_CORE_MMA_K 0
    #define SNEPPX_HAS_TMA 0
    #define SNEPPX_HAS_CLUSTER 0
    #define SNEPPX_HAS_FP8 0
    #define SNEPPX_HAS_WGMMA 0
#endif

// ============================================================================
// Tile & Block Configuration
// ============================================================================

// GEMM tile dimensions (optimized for tensor cores)
#define SNEPPX_GEMM_BLOCK_ROWS 128
#define SNEPPX_GEMM_BLOCK_COLS 128
#define SNEPPX_GEMM_BLOCK_K 32

// Warp tile dimensions
#define SNEPPX_GEMM_WARP_ROWS 64
#define SNEPPX_GEMM_WARP_COLS 64
#define SNEPPX_GEMM_WARP_K 32

// Warps per block
#define SNEPPX_GEMM_WARPS_PER_BLOCK 4

// Threads per warp
#define SNEPPX_WARP_SIZE 32

// Threads per block
#define SNEPPX_GEMM_THREADS_PER_BLOCK (SNEPPX_GEMM_WARPS_PER_BLOCK * SNEPPX_WARP_SIZE)

// MMA tile dimensions
#define SNEPPX_MMA_M SNEPPX_TENSOR_CORE_MMA_M
#define SNEPPX_MMA_N SNEPPX_TENSOR_CORE_MMA_N
#define SNEPPX_MMA_K SNEPPX_TENSOR_CORE_MMA_K

// Warp MMA tiles
#define SNEPPX_WARP_MMA_ROWS (SNEPPX_GEMM_WARP_ROWS / SNEPPX_MMA_M)
#define SNEPPX_WARP_MMA_COLS (SNEPPX_GEMM_WARP_COLS / SNEPPX_MMA_N)

// Shared memory banks
#define SNEPPX_SHARED_MEM_BANKS 32

// ============================================================================
// Data Types & Utilities
// ============================================================================

// FP16 accumulation in FP32
typedef float sneppx_accum_t;

// Activation types (match CPU enum)
typedef enum {
    SNEPPX_ACT_NONE = 0,
    SNEPPX_ACT_RELU = 1,
    SNEPPX_ACT_GELU = 2,
    SNEPPX_ACT_SILU = 3,
    SNEPPX_ACT_TANH = 4,
    SNEPPX_ACT_SIGMOID = 5,
} SNEPPX_ActivationType;

// Error codes
typedef enum {
    SNEPPX_CUDA_SUCCESS = 0,
    SNEPPX_CUDA_ERROR_INVALID_ARG = -1,
    SNEPPX_CUDA_ERROR_OUT_OF_MEMORY = -2,
    SNEPPX_CUDA_ERROR_LAUNCH_FAILED = -3,
    SNEPPX_CUDA_ERROR_INVALID_DEVICE = -4,
    SNEPPX_CUDA_ERROR_UNSUPPORTED = -5,
} SNEPPX_CudaError;

// Stream handle
typedef cudaStream_t SNEPPX_CudaStream_t;

// ============================================================================
// Error Checking Macros
// ============================================================================

#define SNEPPX_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[SNEPPX CUDA ERROR] %s:%d %s failed: %s\n", \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
            return SNEPPX_CUDA_ERROR_LAUNCH_FAILED; \
        } \
    } while(0)

#define SNEPPX_CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "[SNEPPX CUBLAS ERROR] %s:%d %s failed: %d\n", \
                    __FILE__, __LINE__, #call, err); \
            return SNEPPX_CUDA_ERROR_LAUNCH_FAILED; \
        } \
    } while(0)

#define SNEPPX_CURAND_CHECK(call) \
    do { \
        curandStatus_t err = call; \
        if (err != CURAND_STATUS_SUCCESS) { \
            fprintf(stderr, "[SNEPPX CURAND ERROR] %s:%d %s failed: %d\n", \
                    __FILE__, __LINE__, #call, err); \
            return SNEPPX_CUDA_ERROR_LAUNCH_FAILED; \
        } \
    } while(0)

// Device-side assert
#ifdef __CUDA_ARCH__
#define SNEPPX_DEVICE_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            asm("trap;"); \
        } \
    } while(0)
#else
#define SNEPPX_DEVICE_ASSERT(cond)
#endif

// ============================================================================
// Math Utilities (Device)
// ============================================================================

__device__ __forceinline__ float sneppx_fast_gelu(float x) {
    // GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float kBeta = 0.7978845608028654f; // sqrt(2/pi)
    const float kKappa = 0.044715f;
    float x3 = x * x * x;
    float tanh_arg = kBeta * (x + kKappa * x3);
    float tanh_val = tanhf(tanh_arg);
    return 0.5f * x * (1.0f + tanh_val);
}

__device__ __forceinline__ float sneppx_fast_silu(float x) {
    // SiLU(x) = x * sigmoid(x)
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return x * sigmoid;
}

__device__ __forceinline__ float sneppx_fast_relu(float x) {
    return fmaxf(x, 0.0f);
}

__device__ __forceinline__ float sneppx_apply_activation(float x, SNEPPX_ActivationType act) {
    switch (act) {
        case SNEPPX_ACT_RELU:   return sneppx_fast_relu(x);
        case SNEPPX_ACT_GELU:   return sneppx_fast_gelu(x);
        case SNEPPX_ACT_SILU:   return sneppx_fast_silu(x);
        case SNEPPX_ACT_TANH:   return tanhf(x);
        case SNEPPX_ACT_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case SNEPPX_ACT_NONE:
        default:                return x;
    }
}

// ============================================================================
// FP16/BF16 Conversion Utilities
// ============================================================================

__device__ __forceinline__ float sneppx_half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half sneppx_float_to_half(float f) {
    return __float2half_rn(f);
}

#if defined(__CUDA_BF16_H__)
__device__ __forceinline__ float sneppx_bf16_to_float(__nv_bfloat16 bf) {
    return __bfloat162float(bf);
}

__device__ __forceinline__ __nv_bfloat16 sneppx_float_to_bf16(float f) {
    return __float2bfloat16_rn(f);
}
#endif

// ============================================================================
// Tensor Core MMA Wrappers (Device)
// ============================================================================

#if defined(SNEPPX_HOPPER) && SNEPPX_HOPPER
// Hopper WGMMA (Warp Group MMA) - 16x8x16 FP16/BF16
__device__ __forceinline__ void sneppx_wgmma_f16(
    uint32_t& d0, uint32_t& d1,
    const uint32_t& a0, const uint32_t& a1,
    const uint32_t& b0, const uint32_t& b1,
    const uint32_t& c0, const uint32_t& c1
) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3}, {%4, %5}, {%6, %7};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1)
    );
}

#elif defined(SNEPPX_AMPERE) && SNEPPX_AMPERE
// Ampere MMA (mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16)
__device__ __forceinline__ void sneppx_mma_f16(
    uint32_t& d0, uint32_t& d1,
    const uint32_t& a0, const uint32_t& a1,
    const uint32_t& b0, const uint32_t& b1,
    const uint32_t& c0, const uint32_t& c1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3}, {%4, %5}, {%6, %7};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1)
    );
}
#endif

// ============================================================================
// Async Copy (TMA / cp.async)
// ============================================================================

#if defined(SNEPPX_HOPPER) && SNEPPX_HOPPER && SNEPPX_HAS_TMA
// TMA descriptor setup and launch would go here
// For now, use cp.async as fallback
#define SNEPPX_USE_TMA 0
#else
#define SNEPPX_USE_TMA 0
#endif

// cp.async for shared memory loads (Ampere+)
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void sneppx_cp_async_cg(void* dst, const void* src, size_t bytes) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        : : "l"(dst), "l"(src), "n"(bytes)
    );
}

__device__ __forceinline__ void sneppx_cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void sneppx_cp_async_wait_group(int group) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(group) : "memory");
}
#else
#define sneppx_cp_async_cg(dst, src, bytes) \
    do { memcpy(dst, src, bytes); } while(0)
#define sneppx_cp_async_wait_all()
#define sneppx_cp_async_wait_group(group)
#endif

// ============================================================================
// Warp/Block Synchronization
// ============================================================================

__device__ __forceinline__ void sneppx_warp_sync(unsigned mask = 0xFFFFFFFF) {
    __syncwarp(mask);
}

__device__ __forceinline__ void sneppx_block_sync() {
    __syncthreads();
}

// Warp shuffle reductions
__device__ __forceinline__ float sneppx_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float sneppx_warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, tmp);
    }
    return val;
}

__device__ __forceinline__ float sneppx_warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fminf(val, tmp);
    }
    return val;
}

// Broadcast from lane 0
__device__ __forceinline__ float sneppx_warp_broadcast(float val, int src_lane = 0) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

// ============================================================================
// Indexing Helpers
// ============================================================================

__device__ __forceinline__ int sneppx_global_idx_1d() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int sneppx_global_idx_2d(int& row, int& col, int stride) {
    row = blockIdx.y * blockDim.y + threadIdx.y;
    col = blockIdx.x * blockDim.x + threadIdx.x;
    return row * stride + col;
}

// ============================================================================
// Random Number Generation (Philox)
// ============================================================================

__device__ __forceinline__ uint32_t sneppx_philox_round(uint32_t x, uint32_t key) {
    uint64_t mul = 0xD2511F53ULL * (uint64_t)x + key;
    return (uint32_t)(mul ^ (mul >> 32));
}

__device__ __forceinline__ uint32_t sneppx_philox_rand(uint64_t seed, uint32_t seq, uint32_t idx) {
    uint32_t k0 = (uint32_t)seed;
    uint32_t k1 = (uint32_t)(seed >> 32);
    uint32_t x0 = idx;
    uint32_t x1 = seq;

    #pragma unroll
    for (int i = 0; i < 10; i++) {
        x0 = sneppx_philox_round(x0, k0);
        x1 = sneppx_philox_round(x1, k1);
    }
    return x0 ^ x1;
}

// ============================================================================
// Kernel Launch Configuration Helpers
// ============================================================================

inline dim3 sneppx_gemm_grid_2d(int M, int N, int block_rows = SNEPPX_GEMM_BLOCK_ROWS, int block_cols = SNEPPX_GEMM_BLOCK_COLS) {
    return dim3((N + block_cols - 1) / block_cols, (M + block_rows - 1) / block_rows);
}

inline dim3 sneppx_gemm_block_2d(int threads_per_block = SNEPPX_GEMM_THREADS_PER_BLOCK) {
    return dim3(32, threads_per_block / 32);
}

inline dim3 sneppx_elementwise_grid(int numel, int block_size = 256) {
    return dim3((numel + block_size - 1) / block_size);
}

inline dim3 sneppx_reduction_grid(int numel, int block_size = 256) {
    return dim3((numel + block_size - 1) / block_size);
}

// ============================================================================
// CUBLAS Handle Management
// ============================================================================

extern "C" {
    // These are implemented in memory_cuda.cu
    cublasHandle_t sneppx_cublas_get_handle();
    void sneppx_cublas_destroy_handle();
}

// ============================================================================
// Device Properties Query
// ============================================================================

struct SNEPPX_DeviceProps {
    int device_id;
    char name[256];
    int compute_capability_major;
    int compute_capability_minor;
    size_t global_mem_bytes;
    size_t shared_mem_per_block;
    size_t shared_mem_per_sm;
    int max_threads_per_block;
    int max_threads_per_sm;
    int max_blocks_per_sm;
    int warp_size;
    int num_sms;
    int clock_rate_khz;
    int memory_clock_rate_khz;
    int memory_bus_width;
    int l2_cache_size;
    int max_shared_mem_per_block_optin;
};

extern "C" {
    SNEPPX_CudaError sneppx_cuda_get_device_props(int device_id, SNEPPX_DeviceProps* props);
    SNEPPX_CudaError sneppx_cuda_set_device(int device_id);
    SNEPPX_CudaError sneppx_cuda_get_device(int* device_id);
    SNEPPX_CudaError sneppx_cuda_device_synchronize();
}

#endif // SNEPPX_CUDA_COMMON_CUH

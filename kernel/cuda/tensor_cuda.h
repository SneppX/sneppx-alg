#ifndef SNEPPX_TENSOR_CUDA_H
#define SNEPPX_TENSOR_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include "common.cuh"

#ifdef __cplusplus
/*
 * SNEPPX - Tensor Cuda
 *
 * WHAT
 *   Tensor Cuda.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ============================================================================
// Tensor Descriptor (Device-Compatible)
// ============================================================================

typedef struct {
    void* data;                    // Device pointer to data
    int64_t* shape;                // Device pointer to shape array
    int64_t* strides;              // Device pointer to strides array
    int ndim;                      // Number of dimensions
    int dtype;                     // SNEPPX_Dtype enum
    int64_t numel;                 // Total elements
    size_t element_size;           // Bytes per element
} SNEPPX_CudaTensor;

// Data type enum (must match CPU enum)
typedef enum {
    SNEPPX_DTYPE_BOOL = 0,
    SNEPPX_DTYPE_INT8 = 1,
    SNEPPX_DTYPE_INT16 = 2,
    SNEPPX_DTYPE_INT32 = 3,
    SNEPPX_DTYPE_INT64 = 4,
    SNEPPX_DTYPE_UINT8 = 5,
    SNEPPX_DTYPE_FLOAT16 = 6,
    SNEPPX_DTYPE_BFLOAT16 = 7,
    SNEPPX_DTYPE_FLOAT32 = 8,
    SNEPPX_DTYPE_FLOAT64 = 9,
    SNEPPX_DTYPE_COMPLEX64 = 10,
    SNEPPX_DTYPE_COMPLEX128 = 11,
} SNEPPX_CudaDtype;

// ============================================================================
// Memory Management
// ============================================================================

// Allocate device tensor
SNEPPX_CudaError sneppx_cuda_tensor_alloc(
    SNEPPX_CudaTensor** tensor,
    const int64_t* shape,
    int ndim,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

// Free device tensor
SNEPPX_CudaError sneppx_cuda_tensor_free(
    SNEPPX_CudaTensor* tensor,
    SNEPPX_CudaStream_t stream
);

// Copy tensor (device to device, host to device, device to host)
SNEPPX_CudaError sneppx_cuda_tensor_copy(
    SNEPPX_CudaTensor* dst,
    const SNEPPX_CudaTensor* src,
    SNEPPX_CudaStream_t stream
);

// Create tensor view (no copy, shared data)
SNEPPX_CudaError sneppx_cuda_tensor_view(
    SNEPPX_CudaTensor** view,
    const SNEPPX_CudaTensor* src,
    const int64_t* shape,
    int ndim,
    const int64_t* strides
);

// ============================================================================
// GEMM Operations (C = alpha * A * B + beta * C + bias, then activation)
// ============================================================================

// Main GEMM: C = alpha * A * B + beta * C + bias, then activation
// Supports: FP16, BF16, FP32, FP64
// Layout: Row-major (C-contiguous)
SNEPPX_CudaError sneppx_cuda_gemm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* A,      // [M, K]
    const SNEPPX_CudaTensor* B,      // [K, N]
    SNEPPX_CudaTensor* C,            // [M, N] - output
    const SNEPPX_CudaTensor* bias,   // [N] or nullptr
    float alpha,
    float beta,
    SNEPPX_ActivationType activation
);

// Batched GEMM
SNEPPX_CudaError sneppx_cuda_batch_gemm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* A,      // [batch, M, K]
    const SNEPPX_CudaTensor* B,      // [batch, K, N]
    SNEPPX_CudaTensor* C,            // [batch, M, N]
    const SNEPPX_CudaTensor* bias,   // [batch, N] or nullptr
    float alpha,
    float beta,
    SNEPPX_ActivationType activation
);

// Strided GEMM (for non-contiguous tensors)
SNEPPX_CudaError sneppx_cuda_strided_gemm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* A,
    const SNEPPX_CudaTensor* B,
    SNEPPX_CudaTensor* C,
    const SNEPPX_CudaTensor* bias,
    float alpha,
    float beta,
    SNEPPX_ActivationType activation
);

// ============================================================================
// Element-wise Operations
// ============================================================================

// Binary ops: out = op(A, B) with broadcasting
typedef enum {
    SNEPPX_EW_ADD = 0,
    SNEPPX_EW_SUB = 1,
    SNEPPX_EW_MUL = 2,
    SNEPPX_EW_DIV = 3,
    SNEPPX_EW_POW = 4,
    SNEPPX_EW_MAX = 5,
    SNEPPX_EW_MIN = 6,
    SNEPPX_EW_EQ = 7,
    SNEPPX_EW_NE = 8,
    SNEPPX_EW_LT = 8,
    SNEPPX_EW_LE = 9,
    SNEPPX_EW_GT = 10,
    SNEPPX_EW_GE = 11,
} SNEPPX_EwOp;

SNEPPX_CudaError sneppx_cuda_elementwise_binary(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* A,
    const SNEPPX_CudaTensor* B,
    SNEPPX_CudaTensor* out,
    SNEPPX_EwOp op
);

// Unary ops: out = op(A)
typedef enum {
    SNEPPX_UNARY_NEG = 0,
    SNEPPX_UNARY_ABS = 1,
    SNEPPX_UNARY_SQRT = 2,
    SNEPPX_UNARY_RSQRT = 3,
    SNEPPX_UNARY_EXP = 3,
    SNEPPX_UNARY_LOG = 4,
    SNEPPX_UNARY_SIN = 4,
    SNEPPX_UNARY_COS = 5,
    SNEPPX_UNARY_TAN = 6,
    SNEPPX_UNARY_SIGMOID = 6,
    SNEPPX_UNARY_TANH = 7,
    SNEPPX_UNARY_GELU = 7,
    SNEPPX_UNARY_SILU = 8,
    SNEPPX_UNARY_RELU = 8,
    SNEPPX_UNARY_FLOOR = 9,
    SNEPPX_UNARY_CEIL = 10,
    SNEPPX_UNARY_ROUND = 10,
    SNEPPX_UNARY_TRUNC = 11,
    SNEPPX_UNARY_SIGN = 11,
} SNEPPX_UnaryOp;

SNEPPX_CudaError sneppx_cuda_elementwise_unary(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* A,
    SNEPPX_CudaTensor* out,
    SNEPPX_UnaryOp op
);

// In-place element-wise (out = A op B, A is modified)
SNEPPX_CudaError sneppx_cuda_elementwise_binary_inplace(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaTensor* A,
    const SNEPPX_CudaTensor* B,
    SNEPPX_EwOp op
);

// Activation functions (in-place and out-of-place)
SNEPPX_CudaError sneppx_cuda_activation(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    SNEPPX_ActivationType act
);

SNEPPX_CudaError sneppx_cuda_activation_inplace(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaTensor* tensor,
    SNEPPX_ActivationType act
);

// ============================================================================
// Reduction Operations
// ============================================================================

typedef enum {
    SNEPPX_REDUCE_SUM = 0,
    SNEPPX_REDUCE_MEAN = 1,
    SNEPPX_REDUCE_MAX = 2,
    SNEPPX_REDUCE_MIN = 3,
    SNEPPX_REDUCE_PROD = 3,
    SNEPPX_REDUCE_AMAX = 4,  // argmax
    SNEPPX_REDUCE_AMIN = 4,  // argmin
    SNEPPX_REDUCE_NORM_L1 = 5,
    SNEPPX_REDUCE_NORM_L2 = 5,
} SNEPPX_ReduceOp;

SNEPPX_CudaError sneppx_cuda_reduce(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    const int* dims,
    int num_dims,
    SNEPPX_ReduceOp op,
    bool keepdim
);

// All-reduce (single tensor, all elements)
SNEPPX_CudaError sneppx_cuda_reduce_all(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    SNEPPX_ReduceOp op
);

// ============================================================================
// Normalization Operations
// ============================================================================

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
SNEPPX_CudaError sneppx_cuda_layer_norm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    const SNEPPX_CudaTensor* weight,   // [normalized_shape] or nullptr
    const SNEPPX_CudaTensor* bias,     // [normalized_shape] or nullptr
    SNEPPX_CudaTensor* output,
    const int* normalized_shape,
    int normalized_ndim,
    float eps
);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
SNEPPX_CudaError sneppx_cuda_rms_norm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    const SNEPPX_CudaTensor* weight,
    SNEPPX_CudaTensor* output,
    int normalized_dim,
    float eps
);

// GroupNorm
SNEPPX_CudaError sneppx_cuda_group_norm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    const SNEPPX_CudaTensor* weight,
    const SNEPPX_CudaTensor* bias,
    SNEPPX_CudaTensor* output,
    int num_groups,
    int num_channels,
    float eps
);

// ============================================================================
// Convolution Operations
// ============================================================================

typedef enum {
    SNEPPX_CONV_FWD = 0,
    SNEPPX_CONV_BWD_DATA = 1,
    SNEPPX_CONV_BWD_FILTER = 2,
} SNEPPX_ConvMode;

SNEPPX_CudaError sneppx_cuda_conv2d(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,      // [N, C, H, W] or [N, H, W, C]
    const SNEPPX_CudaTensor* weight,     // [OC, IC, KH, KW]
    const SNEPPX_CudaTensor* bias,       // [OC] or nullptr
    SNEPPX_CudaTensor* output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    bool channels_last
);

// Depthwise separable convolution
SNEPPX_CudaError sneppx_cuda_depthwise_conv2d(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    const SNEPPX_CudaTensor* weight,
    const SNEPPX_CudaTensor* bias,
    SNEPPX_CudaTensor* output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w
);

// ============================================================================
// Pooling Operations
// ============================================================================

typedef enum {
    SNEPPX_POOL_MAX = 0,
    SNEPPX_POOL_AVG = 1,
    SNEPPX_POOL_MAX_UNPOOL = 2,
} SNEPPX_PoolType;

SNEPPX_CudaError sneppx_cuda_pool2d(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    SNEPPX_PoolType pool_type,
    bool channels_last
);

// Adaptive pooling (output size specified)
SNEPPX_CudaError sneppx_cuda_adaptive_pool2d(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    int output_h, int output_w,
    SNEPPX_PoolType pool_type,
    bool channels_last
);

// ============================================================================
// Tensor Manipulation
// ============================================================================

// Reshape (view, no copy)
SNEPPX_CudaError sneppx_cuda_reshape(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    const int64_t* shape,
    int ndim
);

// Transpose/permute
SNEPPX_CudaError sneppx_cuda_transpose(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    const int* perm,
    int ndim
);

// Contiguous (make contiguous if not)
SNEPPX_CudaError sneppx_cuda_contiguous(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output
);

// Slice/strided slice
SNEPPX_CudaError sneppx_cuda_slice(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    const int64_t* starts,
    const int64_t* ends,
    const int64_t* steps,
    int ndim
);

// Concat along dimension
SNEPPX_CudaError sneppx_cuda_concat(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor** inputs,
    int num_inputs,
    SNEPPX_CudaTensor* output,
    int dim
);

// Split
SNEPPX_CudaError sneppx_cuda_split(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor** outputs,
    const int64_t* split_sizes,
    int num_splits,
    int dim
);

// ============================================================================
// Random Number Generation
// ============================================================================

typedef enum {
    SNEPPX_RAND_UNIFORM = 0,
    SNEPPX_RAND_NORMAL = 1,
    SNEPPX_RAND_BERNOULLI = 2,
} SNEPPX_RandDist;

SNEPPX_CudaError sneppx_cuda_random(
    SNEPPX_CudaStream_t stream,
    SNEPPX_CudaTensor* output,
    SNEPPX_RandDist dist,
    float param1,  // mean for normal, low for uniform, p for bernoulli
    float param2,  // stddev for normal, high for uniform
    uint64_t seed,
    uint64_t offset
);

// Philox RNG state
typedef struct {
    uint64_t seed;
    uint64_t offset;
    uint64_t counter;
} SNEPPX_PhiloxState;

SNEPPX_CudaError sneppx_cuda_philox_init(SNEPPX_PhiloxState* state, uint64_t seed);
SNEPPX_CudaError sneppx_cuda_philox_random_uniform(
    SNEPPX_CudaStream_t stream,
    SNEPPX_PhiloxState* state,
    SNEPPX_CudaTensor* output,
    float low, float high
);
SNEPPX_CudaError sneppx_cuda_philox_random_normal(
    SNEPPX_CudaStream_t stream,
    SNEPPX_PhiloxState* state,
    SNEPPX_CudaTensor* output,
    float mean, float stddev
);

// ============================================================================
// Tensor Creation Helpers
// ============================================================================

SNEPPX_CudaError sneppx_cuda_tensor_zeros(
    SNEPPX_CudaTensor** tensor,
    const int64_t* shape,
    int ndim,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_tensor_ones(
    SNEPPX_CudaTensor** tensor,
    const int64_t* shape,
    int ndim,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_tensor_eye(
    SNEPPX_CudaTensor** tensor,
    int n,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_tensor_arange(
    SNEPPX_CudaTensor** tensor,
    float start, float stop, float step,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_tensor_linspace(
    SNEPPX_CudaTensor** tensor,
    float start, float stop, int steps,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_cuda_tensor_full(
    SNEPPX_CudaTensor** tensor,
    const int64_t* shape,
    int ndim,
    float value,
    SNEPPX_CudaDtype dtype,
    SNEPPX_CudaStream_t stream
);

// ============================================================================
// Device Synchronization & Stream Management
// ============================================================================

SNEPPX_CudaError sneppx_cuda_stream_create(SNEPPX_CudaStream_t* stream);
SNEPPX_CudaError sneppx_cuda_stream_destroy(SNEPPX_CudaStream_t stream);
SNEPPX_CudaError sneppx_cuda_stream_synchronize(SNEPPX_CudaStream_t stream);
SNEPPX_CudaError sneppx_cuda_device_synchronize();

// ============================================================================
// Device Query & Properties
// ============================================================================

typedef struct {
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
} SNEPPX_DeviceProps;

SNEPPX_CudaError sneppx_cuda_get_device_props(int device_id, struct SNEPPX_DeviceProps* props);
SNEPPX_CudaError sneppx_cuda_set_device(int device_id);
SNEPPX_CudaError sneppx_cuda_get_device(int* device_id);

// ============================================================================
// CUBLAS Handle Management
// ============================================================================

// Get per-thread CUBLAS handle (thread-safe)
cublasHandle_t sneppx_cublas_get_handle();

// Destroy handle at thread exit
void sneppx_cublas_destroy_handle();

// ============================================================================
// Kernel Launch Configuration Helpers
// ============================================================================

// Optimal block/grid for GEMM
typedef struct {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
} SNEPPX_GemmLaunchConfig;

SNEPPX_GemmLaunchConfig sneppx_gemm_get_launch_config(
    int M, int N, int K,
    SNEPPX_CudaDtype dtype
);

// ============================================================================
// Kernel Auto-tuning
// ============================================================================

typedef struct {
    int block_rows;
    int block_cols;
    int warp_rows;
    int warp_cols;
    int stages;
    int chunks_k;
} SNEPPX_GemmTuneParams;

// Auto-tune GEMM for given problem size
SNEPPX_CudaError sneppx_cuda_gemm_autotune(
    int M, int N, int K,
    SNEPPX_CudaDtype dtype,
    SNEPPX_GemmTuneParams* best_params
);

// ============================================================================
// Profiling & Debugging
// ============================================================================

// NVTX range markers
void sneppx_nvtx_range_push(const char* name);
void sneppx_nvtx_range_pop();

// Kernel duration timing
typedef struct {
    cudaEvent_t start;
    cudaEvent_t end;
    float elapsed_ms;
} SNEPPX_CudaTimer;

SNEPPX_CudaError sneppx_cuda_timer_create(SNEPPX_CudaTimer* timer);
SNEPPX_CudaError sneppx_cuda_timer_start(SNEPPX_CudaTimer* timer, SNEPPX_CudaStream_t stream);
SNEPPX_CudaError sneppx_cuda_timer_stop(SNEPPX_CudaTimer* timer, SNEPPX_CudaStream_t stream);
SNEPPX_CudaError sneppx_cuda_timer_elapsed(const SNEPPX_CudaTimer* timer, float* ms);
SNEPPX_CudaError sneppx_cuda_timer_destroy(SNEPPX_CudaTimer* timer);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_TENSOR_CUDA_H

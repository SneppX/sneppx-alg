#ifndef SNEPPX_FM_CUDA_CUH
#define SNEPPX_FM_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../kernel/cuda/common.cuh"

/*
 * SNEPPX - Fm Cuda
 *
 * WHAT
 *   Fm Cuda.
 *
 * CONCEPT
 *   Provides the Fm Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// FM (Federated Memory) CUDA Kernels
// - All-reduce gradient aggregation (ring/butterfly/tree)
// - Memory bank synchronization
// - Federated averaging
// - Gradient compression (quantization, sparsification, Top-K)

#ifdef __cplusplus
extern "C" {
#endif

// All-reduce using ring algorithm (simulated)
typedef struct {
    half* buffer;
    size_t size;
    int rank;
    int world_size;
} SNEPPX_AllReduceRingState;

SNEPPX_CudaError sneppx_cuda_all_reduce_ring(
    SNEPPX_CudaStream_t stream,
    half* data,
    size_t numel,
    int rank,
    int world_size
);

// Butterfly all-reduce
SNEPPX_CudaError sneppx_cuda_all_reduce_butterfly(
    SNEPPX_CudaStream_t stream,
    half* data,
    size_t numel,
    int rank,
    int world_size
);

// Gradient quantization (FP16 -> INT8 with scaling)
SNEPPX_CudaError sneppx_cuda_quantize_gradients(
    SNEPPX_CudaStream_t stream,
    const half* gradients,
    int8_t* quantized,
    float* scale,
    int numel,
    int bits
);

// Gradient dequantization
SNEPPX_CudaError sneppx_cuda_dequantize_gradients(
    SNEPPX_CudaStream_t stream,
    const int8_t* quantized,
    half* gradients,
    const float* scale,
    int numel,
    int bits
);

// Top-K gradient sparsification
SNEPPX_CudaError sneppx_cuda_topk_sparsify(
    SNEPPX_CudaStream_t stream,
    const half* gradients,
    half* sparse_values,
    int* sparse_indices,
    int numel,
    int k
);

// Federated averaging (weighted)
SNEPPX_CudaError sneppx_cuda_federated_average(
    SNEPPX_CudaStream_t stream,
    half** model_chunks,
    const float* weights,
    half* output,
    int num_chunks,
    int chunk_size
);

// Memory bank sync (push/pull)
SNEPPX_CudaError sneppx_cuda_memory_bank_sync(
    SNEPPX_CudaStream_t stream,
    half* local_bank,
    const half* remote_bank,
    size_t bank_size,
    int sync_direction  // 0=pull, 1=push, 2=sync
);

#ifdef __cplusplus
}
#endif

#endif

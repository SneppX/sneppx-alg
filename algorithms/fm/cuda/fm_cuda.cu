#include "fm_cuda.cuh"
#include <cooperative_groups.h>

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


namespace cg = cooperative_groups;

// Ring all-reduce step
__global__ void ring_reduce_scatter_kernel(
    half* data, int chunk_size, int rank, int world_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int send_to = (rank + 1) % world_size;
    int recv_from = (rank - 1 + world_size) % world_size;
    
    // In a real implementation, this would use peer-to-peer or NCCL
    // Here we simulate with a simple accumulation
    if (idx < chunk_size) {
        // Each rank's chunk accumulates data from the previous rank
        data[rank * chunk_size + idx] += data[recv_from * chunk_size + idx];
    }
}

SNEPPX_CudaError sneppx_cuda_all_reduce_ring(
    SNEPPX_CudaStream_t stream,
    half* data, size_t numel,
    int rank, int world_size
) {
    if (!data || world_size <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int chunk_size = numel / world_size;
    int block = 256;
    int grid = (chunk_size + block - 1) / block;
    
    for (int step = 0; step < world_size - 1; step++) {
        ring_reduce_scatter_kernel<<<grid, block, 0, stream>>>(
            data, chunk_size, rank, world_size
        );
    }
    
    // All-gather phase (broadcast reduced chunks)
    auto all_gather = [] __global__ (half* d, int cs, int ws) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < cs * ws) {
            int chunk = idx / cs;
            d[idx] = d[chunk * cs + (idx % cs)];
        }
    };
    all_gather<<<grid * world_size, block, 0, stream>>>(data, chunk_size, world_size);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Butterfly all-reduce
SNEPPX_CudaError sneppx_cuda_all_reduce_butterfly(
    SNEPPX_CudaStream_t stream,
    half* data, size_t numel,
    int rank, int world_size
) {
    if (!data || world_size <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (!(world_size && !(world_size & (world_size - 1)))) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;  // Must be power of 2
    }
    
    int block = 256;
    int grid = (numel + block - 1) / block;
    
    for (int stride = 1; stride < world_size; stride <<= 1) {
        int partner = rank ^ stride;
        
        auto butterfly_step = [] __global__ (half* d, int n, int r, int p) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n && r < p) {
                d[idx] += d[idx];
            }
        };
        butterfly_step<<<grid, block, 0, stream>>>(data, numel, rank, partner);
    }
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Gradient quantization (FP16 -> INT8)
__global__ void quantize_gradients_kernel(
    const half* gradients, int8_t* quantized, float* scale,
    int numel, int bits
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int start = bid * block_size;
    int end = min(start + block_size, numel);
    
    float local_max = 0.0f;
    for (int i = start + tid; i < end; i += block_size) {
        float g = fabsf(__half2float(gradients[i]));
        local_max = fmaxf(local_max, g);
    }
    local_max = sneppx_warp_reduce_max(local_max);
    
    if (tid == 0) {
        smem[bid] = local_max;
        scale[bid] = local_max / (float)((1 << (bits - 1)) - 1);
    }
    __syncthreads();
    
    float s = tid == 0 ? scale[bid] : 0.0f;
    if (tid == 0) s = scale[bid];
    __syncthreads();
    s = scale[bid];
    
    float inv_scale = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int i = start + tid; i < end; i += block_size) {
        float g = __half2float(gradients[i]);
        int8_t q = (int8_t)fmaxf(-127.0f, fminf(127.0f, g * inv_scale));
        quantized[i] = q;
    }
}

SNEPPX_CudaError sneppx_cuda_quantize_gradients(
    SNEPPX_CudaStream_t stream,
    const half* gradients, int8_t* quantized,
    float* scale, int numel, int bits
) {
    if (!gradients || !quantized || !scale) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    quantize_gradients_kernel<<<grid, block, grid * sizeof(float), stream>>>(
        gradients, quantized, scale, numel, bits
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void dequantize_gradients_kernel(
    const int8_t* quantized, half* gradients,
    const float* scale, int numel, int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int block_id = blockIdx.x;
    gradients[idx] = __float2half_rn((float)quantized[idx] * scale[block_id]);
}

SNEPPX_CudaError sneppx_cuda_dequantize_gradients(
    SNEPPX_CudaStream_t stream,
    const int8_t* quantized, half* gradients,
    const float* scale, int numel, int bits
) {
    if (!quantized || !gradients || !scale) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    dequantize_gradients_kernel<<<grid, block, 0, stream>>>(quantized, gradients, scale, numel, bits);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Top-K sparsification (simplified - uses atomic per-block selection)
__global__ void topk_sparsify_kernel(
    const half* gradients, half* sparse_values,
    int* sparse_indices, int numel, int k
) {
    // Simplified: each block selects its top local values
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int start = bid * blockDim.x;
    int end = min(start + blockDim.x, numel);
    
    // Each thread tracks its best
    float best_val[2] = {0.0f, 0.0f};
    int best_idx[2] = {-1, -1};
    
    for (int i = start + tid; i < end; i += blockDim.x) {
        float g = fabsf(__half2float(gradients[i]));
        if (g > best_val[0]) {
            best_val[1] = best_val[0]; best_idx[1] = best_idx[0];
            best_val[0] = g; best_idx[0] = i;
        } else if (g > best_val[1]) {
            best_val[1] = g; best_idx[1] = i;
        }
    }
    
    // Write block-local top-2 (in practice, use global Top-K)
    if (tid < 2 && best_idx[tid] >= 0) {
        int out_idx = bid * 2 + tid;
        if (out_idx < k) {
            sparse_values[out_idx] = gradients[best_idx[tid]];
            sparse_indices[out_idx] = best_idx[tid];
        }
    }
}

SNEPPX_CudaError sneppx_cuda_topk_sparsify(
    SNEPPX_CudaStream_t stream,
    const half* gradients, half* sparse_values,
    int* sparse_indices, int numel, int k
) {
    if (!gradients || !sparse_values || !sparse_indices) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    topk_sparsify_kernel<<<grid, block, 0, stream>>>(gradients, sparse_values, sparse_indices, numel, k);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Federated averaging
__global__ void federated_average_kernel(
    half** model_chunks, const float* weights,
    half* output, int num_chunks, int chunk_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size) return;
    
    float sum = 0.0f;
    float w_sum = 0.0f;
    for (int c = 0; c < num_chunks; c++) {
        float w = weights[c];
        sum += w * __half2float(model_chunks[c][idx]);
        w_sum += w;
    }
    if (w_sum > 0.0f) output[idx] = __float2half_rn(sum / w_sum);
}

SNEPPX_CudaError sneppx_cuda_federated_average(
    SNEPPX_CudaStream_t stream,
    half** model_chunks, const float* weights,
    half* output, int num_chunks, int chunk_size
) {
    if (!model_chunks || !weights || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (chunk_size + block - 1) / block;
    federated_average_kernel<<<grid, block, 0, stream>>>(model_chunks, weights, output, num_chunks, chunk_size);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Memory bank sync
__global__ void memory_bank_sync_kernel(
    half* local_bank, const half* remote_bank,
    size_t bank_size, int sync_direction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bank_size / sizeof(half)) return;
    
    if (sync_direction == 0) {
        local_bank[idx] = remote_bank[idx];
    } else if (sync_direction == 1) {
        local_bank[idx] = remote_bank[idx];
    } else {
        float l = __half2float(local_bank[idx]);
        float r = __half2float(remote_bank[idx]);
        local_bank[idx] = __float2half_rn((l + r) * 0.5f);
    }
}

SNEPPX_CudaError sneppx_cuda_memory_bank_sync(
    SNEPPX_CudaStream_t stream,
    half* local_bank, const half* remote_bank,
    size_t bank_size, int sync_direction
) {
    if (!local_bank || !remote_bank) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int elements = bank_size / sizeof(half);
    int grid = (elements + block - 1) / block;
    memory_bank_sync_kernel<<<grid, block, 0, stream>>>(local_bank, remote_bank, bank_size, sync_direction);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

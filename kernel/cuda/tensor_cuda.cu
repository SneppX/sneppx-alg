#include "tensor_cuda.h"
#include "common.cuh"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>

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


// ============================================================================
// GEMM Kernel: C = alpha * A * B + beta * C + bias, then activation
// ============================================================================

// FP16 GEMM using Tensor Cores (Ampere/Hopper)
template <int BLOCK_ROWS, int BLOCK_COLS, int WARP_ROWS, int WARP_COLS, int MMA_M, int MMA_N, int MMA_K>
__global__ void sneppx_gemm_f16_tensor_core_kernel(
    const half* __restrict__ A,     // [M, K]
    const half* __restrict__ B,     // [K, N]
    half* __restrict__ C,           // [M, N]
    const half* __restrict__ bias,  // [N] or nullptr
    const float alpha,
    const float beta,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldc,
    SNEPPX_ActivationType act_type
) {
    // Shared memory for tiles
    __shared__ half As[2][BLOCK_ROWS][BLOCK_COLS];  // Double buffering
    __shared__ half Bs[2][BLOCK_ROWS][BLOCK_COLS];

    // Thread indexing
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_row = warp_id / (BLOCK_COLS / WARP_COLS);
    const int warp_col = warp_id % (BLOCK_COLS / WARP_COLS);

    // Block tile coordinates
    const int block_tile_m = blockIdx.y * BLOCK_ROWS;
    const int block_tile_n = blockIdx.x * BLOCK_COLS;

    // Warp tile coordinates within block
    const int warp_tile_m = warp_row * WARP_ROWS;
    const int warp_tile_n = warp_col * WARP_COLS;

    // Accumulators in registers (FP32 for numerical stability)
    float accum[WARP_ROWS / MMA_M][WARP_COLS / MMA_N][2] = {0.0f};

    // Global memory pointers
    const half* A_ptr = A + block_tile_m * lda;
    const half* B_ptr = B + block_tile_n;
    half* C_ptr = C + block_tile_m * ldc + block_tile_n;

    // K-dimension loop with double buffering
    const int K_tiles = (K + BLOCK_COLS - 1) / BLOCK_COLS;
    int k_base = 0;

    // Stage 0: Load first tile
    #pragma unroll
    for (int k_inner = 0; k_inner < BLOCK_COLS; k_inner += 16) {
        int k = k_base + k_inner;
        if (k < K) {
            // Coalesced load from global to shared memory
            int row = threadIdx.y * 16 + threadIdx.x / 2;
            int col = threadIdx.x % 2 * 8;
            
            if (block_tile_m + row < M && k + col < K) {
                As[0][row][col] = A[(block_tile_m + row) * lda + k + col];
                As[0][row][col + 1] = A[(block_tile_m + row) * lda + k + col + 1];
            } else {
                As[0][row][col] = __float2half(0.0f);
                As[0][row][col + 1] = __float2half(0.0f);
            }

            if (k + row < K && block_tile_n + col < N) {
                Bs[0][row][col] = B[(k + row) * ldb + block_tile_n + col];
                Bs[0][row][col + 1] = B[(k + row) * ldb + block_tile_n + col + 1];
            } else {
                Bs[0][row][col] = __float2half(0.0f);
                Bs[0][row][col + 1] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // Main K-loop with double buffering
    for (int tile_idx = 0; tile_idx < K_tiles; ++tile_idx) {
        int next_tile = (tile_idx + 1) % 2;
        int curr_tile = tile_idx % 2;
        k_base = tile_idx * BLOCK_COLS;

        // Prefetch next tile
        if (tile_idx + 1 < K_tiles) {
            int next_k = k_base + BLOCK_COLS;
            #pragma unroll
            for (int k_inner = 0; k_inner < BLOCK_COLS; k_inner += 16) {
                int k = next_k + k_inner;
                int row = threadIdx.y * 16 + threadIdx.x / 2;
                int col = threadIdx.x % 2 * 8;
                
                if (block_tile_m + row < M && k + col < K) {
                    As[next_tile][row][col] = A[(block_tile_m + row) * lda + k + col];
                    As[next_tile][row][col + 1] = A[(block_tile_m + row) * lda + k + col + 1];
                } else {
                    As[next_tile][row][col] = __float2half(0.0f);
                    As[next_tile][row][col + 1] = __float2half(0.0f);
                }

                if (k + row < K && block_tile_n + col < N) {
                    Bs[next_tile][row][col] = B[(k + row) * ldb + block_tile_n + col];
                    Bs[next_tile][row][col + 1] = B[(k + row) * ldb + block_tile_n + col + 1];
                } else {
                    Bs[next_tile][row][col] = __float2half(0.0f);
                    Bs[next_tile][row][col + 1] = __float2half(0.0f);
                }
            }
        }

        // Compute on current tile using tensor cores
        // Each warp computes a 64x64 tile using 4x8 MMA operations
        const int warp_mma_m = WARP_ROWS / MMA_M;  // 4
        const int warp_mma_n = WARP_COLS / MMA_N;  // 8

        for (int kk = 0; kk < BLOCK_COLS; kk += MMA_K) {
            // Load A fragment (WARP_ROWS x MMA_K)
            uint32_t A_frag[WARP_ROWS / MMA_M][2];
            #pragma unroll
            for (int i = 0; i < WARP_ROWS / MMA_M; ++i) {
                int row = warp_tile_m + i * MMA_M + lane_id / 4;
                int col = kk + lane_id % 4;
                if (row < BLOCK_ROWS && k_base + kk + col < BLOCK_COLS) {
                    half2 val = *reinterpret_cast<const half2*>(&As[curr_tile][row][kk + (lane_id % 4) * 2]);
                    A_frag[i][lane_id / 4] = *reinterpret_cast<const uint32_t*>(&val);
                } else {
                    A_frag[i][lane_id / 4] = 0;
                }
            }

            // Load B fragment (MMA_K x WARP_COLS)
            uint32_t B_frag[WARP_COLS / MMA_N][2];
            #pragma unroll
            for (int j = 0; j < WARP_COLS / MMA_N; ++j) {
                int row = kk + lane_id / 8;
                int col = warp_tile_n + j * MMA_N + lane_id % 8;
                if (k_base + kk + row < BLOCK_COLS && col < BLOCK_COLS) {
                    half2 val = *reinterpret_cast<const half2*>(&Bs[curr_tile][kk + lane_id / 8][col]);
                    B_frag[j][lane_id / 8] = *reinterpret_cast<const uint32_t*>(&val);
                } else {
                    B_frag[j][lane_id / 8] = 0;
                }
            }

            // Tensor core MMA
            #pragma unroll
            for (int i = 0; i < WARP_ROWS / MMA_M; ++i) {
                #pragma unroll
                for (int j = 0; j < WARP_COLS / MMA_N; ++j) {
                    uint32_t c0 = *reinterpret_cast<uint32_t*>(&accum[i][j][0]);
                    uint32_t c1 = *reinterpret_cast<uint32_t*>(&accum[i][j][1]);
                    
                    #if defined(SNEPPX_HOPPER) && SNEPPX_HOPPER
                    sneppx_wgmma_f16(
                        *reinterpret_cast<uint32_t*>(&accum[i][j][0]),
                        *reinterpret_cast<uint32_t*>(&accum[i][j][1]),
                        A_frag[i][0], A_frag[i][1],
                        B_frag[j][0], B_frag[j][1],
                        c0, c1
                    );
                    #elif defined(SNEPPX_AMPERE) && SNEPPX_AMPERE
                    sneppx_mma_f16(
                        *reinterpret_cast<uint32_t*>(&accum[i][j][0]),
                        *reinterpret_cast<uint32_t*>(&accum[i][j][1]),
                        A_frag[i][0], A_frag[i][1],
                        B_frag[j][0], B_frag[j][1],
                        c0, c1
                    );
                    #endif
                }
            }
        }

        __syncthreads();  // Wait before overwriting shared memory
    }

    // Apply alpha, beta, bias, activation and store
    for (int i = 0; i < WARP_ROWS / MMA_M; ++i) {
        for (int j = 0; j < WARP_COLS / MMA_N; ++j) {
            int row = block_tile_m + warp_tile_m + i * MMA_M;
            int col = block_tile_n + warp_tile_n + j * MMA_N;
            
            if (row < M && col < N) {
                // Apply alpha
                float val = accum[i][j][0] * alpha;
                
                // Apply beta * C
                if (beta != 0.0f) {
                    val += __half2float(C[row * ldc + col]) * beta;
                }
                
                // Apply bias
                if (bias) {
                    val += __half2float(bias[col]);
                }
                
                // Apply activation
                switch (act_type) {
                    case SNEPPX_ACT_RELU:
                        val = fmaxf(val, 0.0f);
                        break;
                    case SNEPPX_ACT_GELU:
                        val = sneppx_fast_gelu(val);
                        break;
                    case SNEPPX_ACT_SILU:
                        val = sneppx_fast_silu(val);
                        break;
                    case SNEPPX_ACT_TANH:
                        val = tanhf(val);
                        break;
                    case SNEPPX_ACT_SIGMOID:
                        val = 1.0f / (1.0f + expf(-val));
                        break;
                    default:
                        break;
                }
                
                C[row * ldc + col] = __float2half_rn(val);
            }
        }
    }
}

// Simplified kernel for non-tensor-core fallback
__global__ void sneppx_gemm_fallback_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const half* __restrict__ bias,
    const float alpha,
    const float beta,
    const int M, const int N, const int K,
    const int lda, const int ldb, const int ldc,
    SNEPPX_ActivationType act_type
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[row * lda + k]) * __half2float(B[k * ldb + col]);
    }
    
    float val = sum * alpha;
    if (beta != 0.0f) {
        val += __half2float(C[row * ldc + col]) * beta;
    }
    if (bias) {
        val += __half2float(bias[col]);
    }
    
    // Activation
    switch (act_type) {
        case SNEPPX_ACT_RELU: val = fmaxf(val, 0.0f); break;
        case SNEPPX_ACT_GELU: val = sneppx_fast_gelu(val); break;
        case SNEPPX_ACT_SILU: val = sneppx_fast_silu(val); break;
        case SNEPPX_ACT_TANH: val = tanhf(val); break;
        case SNEPPX_ACT_SIGMOID: val = 1.0f / (1.0f + expf(-val)); break;
        default: break;
    }
    
    C[row * ldc + col] = __float2half_rn(val);
}

// ============================================================================
// Element-wise Operations
// ============================================================================

__global__ void sneppx_add_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ out,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = __hadd(a[idx], b[idx]);
}

__global__ void sneppx_mul_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ out,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    out[idx] = __hmul(a[idx], b[idx]);
}

__global__ void sneppx_relu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float val = __half2float(input[idx]);
    output[idx] = __float2half_rn(fmaxf(val, 0.0f));
}

__global__ void sneppx_gelu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = __half2float(input[idx]);
    output[idx] = __float2half_rn(sneppx_fast_gelu(x));
}

__global__ void sneppx_silu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = __half2float(input[idx]);
    output[idx] = __float2half_rn(sneppx_fast_silu(x));
}

// ============================================================================
// Reduction Operations
// ============================================================================

__global__ void sneppx_sum_reduce_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t outer, int64_t inner
) {
    extern __shared__ half smem[];
    int tid = threadIdx.x;
    int64_t outer_idx = blockIdx.x;
    
    if (outer_idx >= outer) return;
    
    float sum = 0.0f;
    for (int64_t i = tid; i < inner; i += blockDim.x) {
        sum += __half2float(input[outer_idx * inner + i]);
    }
    
    smem[tid] = sum;
    __syncthreads();
    
    // Warp-level reduction
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp shuffle reduction
    if (tid < 32) {
        float val = smem[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            output[outer_idx] = __float2half_rn(smem[0] + val);
        }
    }
}

// ============================================================================
// LayerNorm Kernel
// ============================================================================

__global__ void sneppx_layernorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int64_t num_rows, int64_t row_stride,
    float eps
) {
    int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;
    
    int tid = threadIdx.x;
    const half* x = input + row * row_stride;
    half* y = output + row * row_stride;
    int64_t D = row_stride;
    
    // Compute mean (warp-level)
    float sum = 0.0f;
    for (int64_t i = tid; i < D; i += blockDim.x) {
        sum += __half2float(x[i]);
    }
    
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    float mean = __shfl_sync(0xFFFFFFFF, sum, 0) / D;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int64_t i = tid; i < D; i += blockDim.x) {
        float diff = __half2float(x[i]) - mean;
        var_sum += diff * diff;
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    float var = __shfl_sync(0xFFFFFFFF, var_sum, 0) / D;
    
    // Normalize and scale
    float inv_std = rsqrtf(var + eps);
    for (int64_t i = tid; i < D; i += blockDim.x) {
        float x_norm = (__half2float(x[i]) - mean) * inv_std;
        float w = weight ? __half2float(weight[i]) : 1.0f;
        float b = bias ? __half2float(bias[i]) : 0.0f;
        y[i] = __float2half_rn(x_norm * w + b);
    }
}

// ============================================================================
// Softmax Kernel
// ============================================================================

__global__ void sneppx_softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int64_t num_rows, int64_t row_stride,
    int dim
) {
    int64_t row = blockIdx.x;
    if (row >= num_rows) return;
    
    int tid = threadIdx.x;
    const half* x = input + row * row_stride;
    half* y = output + row * row_stride;
    int64_t D = row_stride;
    
    // Find max (warp reduce)
    float max_val = -INFINITY;
    for (int64_t i = tid; i < D; i += blockDim.x) {
        max_val = fmaxf(max_val, __half2float(x[i]));
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        max_val = fmaxf(max_val, tmp);
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
    
    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int64_t i = tid; i < D; i += blockDim.x) {
        float val = expf(__half2float(x[i]) - max_val);
        y[i] = __float2half_rn(val);
        exp_sum += val;
    }
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        exp_sum += __shfl_down_sync(0xFFFFFFFF, exp_sum, offset);
    }
    exp_sum = __shfl_sync(0xFFFFFFFF, exp_sum, 0);
    
    // Normalize
    for (int64_t i = tid; i < D; i += blockDim.x) {
        float val = __half2float(y[i]) / exp_sum;
        y[i] = __float2half_rn(val);
    }
}

// ============================================================================
// Host API Wrappers
// ============================================================================

SNEPPX_CudaError sneppx_gemm_fused(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* A,
    const SNEPPX_CudaTensor* B,
    SNEPPX_CudaTensor* C,
    const SNEPPX_CudaTensor* bias,
    float alpha,
    float beta,
    SNEPPX_ActivationType act
) {
    // Validate inputs
    if (!A || !B || !C) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];
    
    if (B->shape[0] != K) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (C->shape[0] != M || C->shape[1] != N) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (bias && (bias->shape[0] != N || bias->ndim != 1)) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Choose kernel based on architecture and problem size
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    int major = prop.major;
    
    dim3 block(32, 8);  // 256 threads
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    
    if (major >= 8) {
        // Tensor core kernel
        sneppx_gemm_f16_tensor_core_kernel<128, 128, 64, 64, 16, 8, 16>
            <<<grid, block, 0, stream>>>(
                static_cast<const half*>(A->data),
                static_cast<const half*>(B->data),
                static_cast<half*>(C->data),
                bias ? static_cast<const half*>(bias->data) : nullptr,
                alpha, beta,
                M, N, A->shape[1],
                A->strides[0], B->strides[0], C->strides[0],
                act
            );
    } else {
        // Fallback kernel
        sneppx_gemm_fallback_kernel<<<grid, block, 0, stream>>>(
            static_cast<const half*>(A->data),
            static_cast<const half*>(B->data),
            static_cast<half*>(C->data),
            bias ? static_cast<const half*>(bias->data) : nullptr,
            alpha, beta,
            M, N, A->shape[1],
            A->strides[0], B->strides[0], C->strides[0],
            act
        );
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

// Element-wise operations
SNEPPX_CudaError sneppx_add(SNEPPX_CudaStream_t stream, const SNEPPX_CudaTensor* a, const SNEPPX_CudaTensor* b, SNEPPX_CudaTensor* out) {
    int64_t numel = a->numel;
    dim3 block(256);
    dim3 grid((numel + 255) / 256);
    sneppx_add_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(a->data),
        static_cast<const half*>(b->data),
        static_cast<half*>(out->data),
        numel
    );
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_mul(SNEPPX_CudaStream_t stream, const SNEPPX_CudaTensor* a, const SNEPPX_CudaTensor* b, SNEPPX_CudaTensor* out) {
    int64_t numel = a->numel;
    dim3 block(256);
    dim3 grid((numel + 255) / 256);
    sneppx_mul_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(a->data),
        static_cast<const half*>(b->data),
        static_cast<half*>(out->data),
        numel
    );
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_relu(SNEPPX_CudaStream_t stream, const SNEPPX_CudaTensor* input, SNEPPX_CudaTensor* output) {
    int64_t numel = input->numel;
    dim3 block(256);
    dim3 grid((numel + 255) / 256);
    sneppx_relu_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(input->data),
        static_cast<half*>(output->data),
        numel
    );
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_gelu(SNEPPX_CudaStream_t stream, const SNEPPX_CudaTensor* input, SNEPPX_CudaTensor* output) {
    int64_t numel = input->numel;
    dim3 block(256);
    dim3 grid((numel + 255) / 256);
    sneppx_gelu_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(input->data),
        static_cast<half*>(output->data),
        numel
    );
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_silu(SNEPPX_CudaStream_t stream, const SNEPPX_CudaTensor* input, SNEPPX_CudaTensor* output) {
    int64_t numel = input->numel;
    dim3 block(256);
    dim3 grid((numel + 255) / 256);
    sneppx_silu_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(input->data),
        static_cast<half*>(output->data),
        numel
    );
    return SNEPPX_CUDA_SUCCESS;
}

// LayerNorm
SNEPPX_CudaError sneppx_layernorm(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    const SNEPPX_CudaTensor* weight,
    const SNEPPX_CudaTensor* bias,
    SNEPPX_CudaTensor* output,
    float eps
) {
    int64_t num_rows = 1;
    for (int i = 0; i < input->ndim - 1; ++i) {
        num_rows *= input->shape[i];
    }
    int64_t row_stride = input->shape[input->ndim - 1];
    
    dim3 block(32, 8);  // 256 threads
    dim3 grid((num_rows + 7) / 8);
    
    sneppx_layernorm_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(input->data),
        weight ? static_cast<const half*>(weight->data) : nullptr,
        bias ? static_cast<const half*>(bias->data) : nullptr,
        static_cast<half*>(output->data),
        num_rows, row_stride, eps
    );
    return SNEPPX_CUDA_SUCCESS;
}

// Softmax
SNEPPX_CudaError sneppx_softmax(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_CudaTensor* input,
    SNEPPX_CudaTensor* output,
    int dim
) {
    // Simplified: assume dim = -1 (last dimension)
    int64_t num_rows = 1;
    for (int i = 0; i < input->ndim - 1; ++i) {
        num_rows *= input->shape[i];
    }
    int64_t row_stride = input->shape[input->ndim - 1];
    
    dim3 block(256);
    dim3 grid(num_rows);
    
    sneppx_softmax_kernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(input->data),
        static_cast<half*>(output->data),
        num_rows, row_stride, dim
    );
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// CUBLAS Handle Management
// ============================================================================

static __thread cublasHandle_t tls_cublas_handle = nullptr;
static __thread bool tls_cublas_initialized = false;

cublasHandle_t sneppx_cublas_get_handle() {
    if (!tls_cublas_initialized) {
        cublasCreate(&tls_cublas_handle);
        cublasSetMathMode(tls_cublas_handle, CUBLAS_TENSOR_OP_MATH);
        tls_cublas_initialized = true;
    }
    return tls_cublas_handle;
}

void sneppx_cublas_destroy_handle() {
    if (tls_cublas_initialized) {
        cublasDestroy(tls_cublas_handle);
        tls_cublas_handle = nullptr;
        tls_cublas_initialized = false;
    }
}

// ============================================================================
// Timer Implementation
// ============================================================================

SNEPPX_CudaError sneppx_cuda_timer_create(SNEPPX_CudaTimer* timer) {
    cudaEventCreate(&timer->start);
    cudaEventCreate(&timer->end);
    timer->elapsed_ms = 0.0f;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_timer_start(SNEPPX_CudaTimer* timer, SNEPPX_CudaStream_t stream) {
    cudaEventRecord(timer->start, stream);
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_timer_stop(SNEPPX_CudaTimer* timer, SNEPPX_CudaStream_t stream) {
    cudaEventRecord(timer->end, stream);
    cudaEventSynchronize(timer->end);
    cudaEventElapsedTime(&timer->elapsed_ms, timer->start, timer->end);
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_timer_elapsed(const SNEPPX_CudaTimer* timer, float* ms) {
    *ms = timer->elapsed_ms;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_timer_destroy(SNEPPX_CudaTimer* timer) {
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->end);
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// Device Properties
// ============================================================================

SNEPPX_CudaError sneppx_cuda_get_device_props(int device_id, SNEPPX_DeviceProps* props) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return SNEPPX_CUDA_ERROR_INVALID_DEVICE;
    
    props->device_id = device_id;
    strncpy(props->name, prop.name, 255);
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
    props->max_shared_mem_per_block_optin = prop.maxSharedMemoryPerBlockOptin;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_set_device(int device_id) {
    return cudaSetDevice(device_id) == cudaSuccess ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_INVALID_DEVICE;
}

SNEPPX_CudaError sneppx_cuda_get_device(int* device_id) {
    return cudaGetDevice(device_id) == cudaSuccess ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_INVALID_DEVICE;
}

SNEPPX_CudaError sneppx_cuda_device_synchronize() {
    return cudaDeviceSynchronize() == cudaSuccess ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// NVTX Profiling
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#if defined(SNEPPX_USE_NVTX) && defined(__CUDACC__)
#include <nvtx3/nvToolsExt.h>

void sneppx_nvtx_range_push(const char* name) {
    nvtxRangePushA(name);
}

void sneppx_nvtx_range_pop() {
    nvtxRangePop();
}

#else
/* Software fallback: use the profiler range stack */
#include "../../include/neural_core/kernel/profiler.h"

void sneppx_nvtx_range_push(const char* name) {
    SNEPPX_range_push(name);
}

void sneppx_nvtx_range_pop() {
    SNEPPX_range_pop();
}
#endif

#ifdef __cplusplus
}
#endif

void sneppx_nvtx_range_push(const char* name) {
    // NVTX would be used here if available
    (void)name;
}

void sneppx_nvtx_range_pop() {
    // NVTX would be used here if available
}

#ifdefcplusplus
}
#endif

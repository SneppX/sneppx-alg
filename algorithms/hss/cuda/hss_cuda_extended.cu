#include "hss_cuda_extended.cuh"
#include <cooperative_groups.h>
#include <cufft.h>

/*
 * SNEPPX - Hss Cuda Extended
 *
 * WHAT
 *   Hss Cuda Extended.
 *
 * CONCEPT
 *   Provides the Hss Cuda Extended.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace cg = cooperative_groups;

// ============================================================================
// Selective Scan (Mamba/S6)
// ============================================================================

__global__ void selective_scan_kernel(
    const float* x, const float* delta,
    const float* A, const float* B, const float* C,
    float* y, float* h_final,
    int seq_len, int dim, int d_state
) {
    int batch = blockIdx.y;
    int d = blockIdx.x;
    int tid = threadIdx.x;
    
    if (d >= dim) return;
    
    extern __shared__ float h_smem[];
    float* h = h_smem;
    
    for (int s = 0; s < d_state; s++) h[s] = 0.0f;
    
    int batch_offset = batch * seq_len * dim;
    
    for (int t = 0; t < seq_len; t++) {
        float dt_val = delta[batch_offset + t * dim + d];
        float x_t = x[batch_offset + t * dim + d];
        float dt_exp = expf(dt_val);
        
        // Selective scan
        for (int s = tid; s < d_state; s += blockDim.x) {
            float a_val = A[d * d_state + s];
            float b_val = B[batch * seq_len * d_state + t * d_state + s];
            
            float a_bar = expf(dt_val * a_val);
            float b_bar = dt_val * b_val;
            
            h[s] = a_bar * h[s] + b_bar * x_t;
        }
        __syncthreads();
        
        // Output
        if (tid == 0) {
            float y_t = 0.0f;
            for (int s = 0; s < d_state; s++) {
                y_t += C[batch * seq_len * d_state + t * d_state + s] * h[s];
            }
            y[batch_offset + t * dim + d] = y_t;
        }
    }
    
    // Save final state
    if (tid < d_state) {
        h_final[batch * dim * d_state + d * d_state + tid] = h[tid];
    }
}

SNEPPX_CudaError sneppx_cuda_selective_scan_fwd(
    SNEPPX_CudaStream_t stream,
    const float* x, const float* delta,
    const float* A, const float* B, const float* C,
    float* y, float* h_final,
    int batch_size, int seq_len, int dim, int d_state
) {
    if (!x || !delta || !A || !B || !C || !y) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    dim3 grid(dim, batch_size);
    dim3 block(min(d_state, 256));
    size_t smem = d_state * sizeof(float);
    
    selective_scan_kernel<<<grid, block, smem, stream>>>(
        x, delta, A, B, C, y, h_final, seq_len, dim, d_state
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// S4 Forward
// ============================================================================

__global__ void s4_fwd_kernel(
    const float* u, float* y,
    const float* A, const float* B, const float* C,
    float* h_state,
    int seq_len, int channels, int d_state
) {
    int batch = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    
    if (c >= channels) return;
    
    // Each channel has its own state
    float* h = &h_state[(batch * channels + c) * d_state];
    const float* A_c = &A[c * d_state * d_state];
    const float* B_c = &B[c * d_state];
    const float* C_c = &C[c * d_state];
    
    for (int t = 0; t < seq_len; t++) {
        float u_t = u[(batch * seq_len + t) * channels + c];
        
        // h = A @ h + B @ u (using discretized matrices)
        float h_new[16];  // up to d_state=16
        for (int s = 0; s < d_state; s++) {
            h_new[s] = 0.0f;
            for (int s2 = 0; s2 < d_state; s2++) {
                h_new[s] += A_c[s * d_state + s2] * h[s2];
            }
            h_new[s] += B_c[s] * u_t;
        }
        
        // Copy back
        for (int s = 0; s < d_state; s++) h[s] = h_new[s];
        
        // y = C @ h
        float y_t = 0.0f;
        for (int s = 0; s < d_state; s++) y_t += C_c[s] * h[s];
        y[(batch * seq_len + t) * channels + c] = y_t;
    }
}

SNEPPX_CudaError sneppx_cuda_s4_fwd(
    SNEPPX_CudaStream_t stream,
    const float* u, float* y,
    const float* A, const float* B, const float* C,
    float* h_state,
    int batch_size, int seq_len, int channels, int d_state
) {
    if (!u || !y || !A || !B || !C) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (d_state > 16) return SNEPPX_CUDA_ERROR_UNSUPPORTED;
    
    dim3 grid(channels, batch_size);
    dim3 block(32);
    
    s4_fwd_kernel<<<grid, block, 0, stream>>>(
        u, y, A, B, C, h_state, seq_len, channels, d_state
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// HiPPO Matrix Initialization
// ============================================================================

__global__ void hippo_matrix_kernel(float* A, float* B, int N, float dt) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    
    if (i >= N || j >= N) return;
    
    // HiPPO-LegS matrix: A_{ij} = -(2i+1)^{1/2}(2j+1)^{1/2} * 1_{i>j}
    // Continuous-time: A_{ij} = - (2i+1) * 1_{i==j} - sqrt((2i+1)(2j+1)) * 1_{i>j}
    if (i == j) {
        A[i * N + j] = -0.5f * (float)(2 * i + 1) * dt;
    } else if (i > j) {
        float val = -sqrtf((float)((2 * i + 1) * (2 * j + 1))) * dt;
        A[i * N + j] = val;
    } else {
        A[i * N + j] = 0.0f;
    }
    
    if (threadIdx.x == 0) {
        B[i] = sqrtf((float)(2 * i + 1)) * dt;
    }
}

SNEPPX_CudaError sneppx_cuda_hippo_matrix(
    SNEPPX_CudaStream_t stream,
    float* A, float* B, int N, float dt
) {
    if (!A || !B) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    hippo_matrix_kernel<<<N, N, 0, stream>>>(A, B, N, dt);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// SSM Convolution Mode
// ============================================================================

__global__ void ssm_conv_kernel(
    const float* kernel, const float* input,
    float* output, int seq_len, int dim
) {
    int batch = blockIdx.y;
    int d = blockIdx.x;
    int t = threadIdx.x;
    
    if (d >= dim || t >= seq_len) return;
    
    float sum = 0.0f;
    int batch_offset = batch * seq_len * dim;
    
    for (int k = 0; k <= t; k++) {
        sum += kernel[k] * input[batch_offset + (t - k) * dim + d];
    }
    
    output[batch_offset + t * dim + d] = sum;
}

SNEPPX_CudaError sneppx_cuda_ssm_conv_fwd(
    SNEPPX_CudaStream_t stream,
    const float* kernel, const float* input,
    float* output, int batch_size, int seq_len, int dim
) {
    if (!kernel || !input || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    dim3 grid(dim, batch_size);
    dim3 block(seq_len);
    
    ssm_conv_kernel<<<grid, block, 0, stream>>>(kernel, input, output, seq_len, dim);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

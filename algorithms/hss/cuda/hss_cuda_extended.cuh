#ifndef SNEPPX_HSS_CUDA_EXTENDED_H
#define SNEPPX_HSS_CUDA_EXTENDED_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../../kernel/cuda/common.cuh"

#ifdef __cplusplus
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


extern "C" {
#endif

// ============================================================================
// Selective Scan (Mamba/S6 style)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_selective_scan_fwd(
    SNEPPX_CudaStream_t stream,
    const float* x,           // [batch, seq_len, dim]
    const float* delta,       // [batch, seq_len, dim]
    const float* A,           // [dim, d_state]
    const float* B,           // [batch, seq_len, d_state]
    const float* C,           // [batch, seq_len, d_state]
    float* y,                 // [batch, seq_len, dim]
    float* h_final,           // [batch, dim, d_state]
    int batch_size,
    int seq_len,
    int dim,
    int d_state
);

// ============================================================================
// Structured State Space Sequence Model (S4)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_s4_fwd(
    SNEPPX_CudaStream_t stream,
    const float* u,           // [batch, seq_len, channels]
    float* y,                 // [batch, seq_len, channels]
    const float* A,           // [channels * d_state, channels * d_state]
    const float* B,           // [channels, d_state]
    const float* C,           // [channels, d_state]
    float* h_state,           // [batch, channels, d_state]
    int batch_size,
    int seq_len,
    int channels,
    int d_state
);

// ============================================================================
// HiPPO (Legendre State Space) Initialization
// ============================================================================

SNEPPX_CudaError sneppx_cuda_hippo_matrix(
    SNEPPX_CudaStream_t stream,
    float* A,                 // [N, N]
    float* B,                 // [N, 1]
    int N,
    float dt
);

// ============================================================================
// HSS SSM Convolution Mode (global convolution via FFT)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_ssm_conv_fwd(
    SNEPPX_CudaStream_t stream,
    const float* kernel,      // [seq_len]
    const float* input,       // [batch, seq_len, dim]
    float* output,            // [batch, seq_len, dim]
    int batch_size,
    int seq_len,
    int dim
);

#ifdef __cplusplus
}
#endif

#endif

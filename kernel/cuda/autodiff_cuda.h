#ifndef SNEPPX_AUTODIFF_CUDA_H
#define SNEPPX_AUTODIFF_CUDA_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.cuh"

#ifdef __cplusplus
/*
 * SNEPPX - Autodiff Cuda
 *
 * WHAT
 *   Autodiff Cuda.
 *
 * CONCEPT
 *   Provides the Autodiff Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// ============================================================================
// GEMM Backward (dA, dB from dC)
// ============================================================================

SNEPPX_CudaError sneppx_cuda_gemm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_a,              // [M, K]
    half* d_b,              // [K, N]
    const half* d_c,        // [M, N]
    const half* a,          // [M, K]
    const half* b,          // [K, N]
    const half* c,          // [M, N] (optional, for activation grad)
    SNEPPX_ActivationType act,
    int M, int N, int K,
    float alpha, float beta
);

// ============================================================================
// Batched GEMM Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_batched_gemm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_a,
    half* d_b,
    const half* d_c,
    const half* a,
    const half* b,
    SNEPPX_ActivationType act,
    int batch_size,
    int M, int N, int K,
    float alpha, float beta
);

// ============================================================================
// Element-wise Backward
// ============================================================================

typedef enum {
    SNEPPX_EW_BWD_ADD,
    SNEPPX_EW_BWD_MUL,
    SNEPPX_EW_BWD_SUB,
    SNEPPX_EW_BWD_DIV,
    SNEPPX_EW_BWD_POW,
} SNEPPX_EwBwdOp;

SNEPPX_CudaError sneppx_cuda_elementwise_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_a,
    half* d_b,
    const half* d_output,
    const half* a,
    const half* b,
    const half* output,
    SNEPPX_EwBwdOp op,
    int numel
);

// ============================================================================
// Activation Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_activation_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* input,
    const half* output,
    SNEPPX_ActivationType act,
    int numel
);

// ============================================================================
// LayerNorm Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_layernorm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,          // [rows, cols]
    half* d_gamma,          // [cols]
    half* d_beta,           // [cols]
    const half* d_output,   // [rows, cols]
    const half* input,      // [rows, cols]
    const half* gamma,      // [cols]
    const half* mu,         // [rows]
    const half* rsigma,     // [rows]
    int rows, int cols,
    float epsilon
);

// ============================================================================
// RMSNorm Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rmsnorm_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    half* d_weight,
    const half* d_output,
    const half* input,
    const half* weight,
    const half* rms,        // [rows]
    int rows, int cols,
    float epsilon
);

// ============================================================================
// Softmax Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_softmax_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* output,     // softmax output
    int rows, int cols
);

// ============================================================================
// Cross-Entropy Loss Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_cross_entropy_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,   // scalar gradient (typically 1.0)
    const half* probs,      // softmax output [batch, classes]
    const int* targets,     // [batch]
    int batch_size,
    int num_classes
);

// ============================================================================
// Convolution Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_conv2d_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,          // [N, C, H, W]
    half* d_weight,         // [F, C, KH, KW]
    const half* d_output,   // [N, F, H_out, W_out]
    const half* input,      // [N, C, H, W]
    const half* weight,     // [F, C, KH, KW]
    int N, int C, int H, int W,
    int F, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
);

// ============================================================================
// Attention Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_attention_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_q,
    half* d_k,
    half* d_v,
    const half* d_output,
    const half* q,
    const half* k,
    const half* v,
    const half* output,
    int batch_size,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    int head_dim,
    float scale
);

// ============================================================================
// Dropout Backward
// ============================================================================

SNEPPX_CudaError sneppx_cuda_dropout_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* d_output,
    const half* mask,
    int numel,
    float p
);

// ============================================================================
// Loss Functions + Gradients
// ============================================================================

SNEPPX_CudaError sneppx_cuda_mse_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* prediction,
    const half* target,
    int numel
);

SNEPPX_CudaError sneppx_cuda_bce_bwd(
    SNEPPX_CudaStream_t stream,
    half* d_input,
    const half* prediction,
    const half* target,
    int numel,
    float epsilon
);

// ============================================================================
// Gradient Clipping
// ============================================================================

SNEPPX_CudaError sneppx_cuda_grad_clip(
    SNEPPX_CudaStream_t stream,
    half* gradients,
    int numel,
    float max_norm,
    float norm_type
);

SNEPPX_CudaError sneppx_cuda_grad_clip_global(
    SNEPPX_CudaStream_t stream,
    half** gradients,
    const int* sizes,
    int num_tensors,
    float max_norm,
    float norm_type
);

// ============================================================================
// Gradient Accumulation / Normalization
// ============================================================================

SNEPPX_CudaError sneppx_cuda_grad_scale(
    SNEPPX_CudaStream_t stream,
    half* gradients,
    float scale_factor,
    int numel
);

SNEPPX_CudaError sneppx_cuda_grad_accumulate(
    SNEPPX_CudaStream_t stream,
    half* grad_accum,
    const half* grad_new,
    float beta,
    int numel
);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_AUTODIFF_CUDA_H

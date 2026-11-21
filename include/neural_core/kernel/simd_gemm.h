#ifndef SNEPPX_SIMD_GEMM_H
#define SNEPPX_SIMD_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Simd Gemm
 *
 * WHAT
 *   Simd Gemm.
 *
 * CONCEPT
 *   Provides the Simd Gemm.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* =========================================================================
 * SIMD Dispatch Layer for Tensor Engine
 *
 * Provides CPU GEMM / elementwise / reduction kernels with runtime
 * dispatch across instruction-set levels (Scalar, SSE4.2, AVX2, AVX-512).
 * All kernels operate on contiguous float32 row-major matrices unless
 * noted. Blocking + prefetch keep L1/L2 hot for very large matmuls.
 * ========================================================================= */

typedef enum {
    SNEPPX_SIMD_SCALAR = 0,
    SNEPPX_SIMD_SSE42  = 1,
    SNEPPX_SIMD_AVX2   = 2,
    SNEPPX_SIMD_AVX512 = 3
} SNEPPXSimdLevel;

/* Detect the highest SIMD level available at runtime */
/**
 * @brief Perform Simd Detect.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSimdLevel SNEPPX_simd_detect(void);

/* Human-readable string for a SIMD level */
/**
 * @brief Perform Simd Level Name.
 *
 * @param lvl [in] Lvl value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_simd_level_name(SNEPPXSimdLevel lvl);

/* -------------------------------------------------------------------------
 * GEMM: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
 * ------------------------------------------------------------------------- */

/* Generic blocked scalar GEMM (always available, used as fallback) */
/**
 * @brief Perform Gemm Scalar.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm_scalar(const float* A, const float* B, float* C,
                         int M, int N, int K,
                         float alpha, float beta);

/* SSE4.2 GEMM (4-wide) */
/**
 * @brief Perform Gemm Sse42.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm_sse42(const float* A, const float* B, float* C,
                        int M, int N, int K, float alpha, float beta);

/* AVX2 GEMM (8-wide FMA) */
/**
 * @brief Perform Gemm Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, float alpha, float beta);

/* AVX-512 GEMM (16-wide FMA) */
/**
 * @brief Perform Gemm Avx512.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm_avx512(const float* A, const float* B, float* C,
                        int M, int N, int K, float alpha, float beta);

/* Dispatch to the best available kernel */
/**
 * @brief Perform Gemm.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, float alpha, float beta);

/* Batched GEMM: C[b] = A[b] @ B[b]; arrays of [M,K] and [K,N] */
/**
 * @brief Perform Gemm Batched.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param batch [in] Batch value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm_batched(const float* A, const float* B, float* C,
                         int batch, int M, int N, int K,
                         float alpha, float beta);

/* Strided batched GEMM (for attention heads) */
/**
 * @brief Perform Gemm Batched Strided.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param batch [in] Batch value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param stride_a [in] Stride A value.
 * @param stride_b [in] Stride B value.
 * @param stride_c [in] Stride C value.
 * @param alpha [in] Alpha value.
 * @param beta [in] Beta value.
 */
void SNEPPX_gemm_batched_strided(const float* A, const float* B, float* C,
                                 int batch, int M, int N, int K,
                                 int stride_a, int stride_b, int stride_c,
                                 float alpha, float beta);

/* -------------------------------------------------------------------------
 * Elementwise kernels (C = op(A, B) or C = f(A))
 * ------------------------------------------------------------------------- */

/**
 * @brief Perform Elem Add Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param n [in] N value.
 */
void SNEPPX_elem_add_avx2(const float* A, const float* B, float* C, size_t n);
/**
 * @brief Perform Elem Mul Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 * @param n [in] N value.
 */
void SNEPPX_elem_mul_avx2(const float* A, const float* B, float* C, size_t n);
/**
 * @brief Perform Elem Relu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 * @param n [in] N value.
 */
void SNEPPX_elem_relu_avx2(const float* A, float* C, size_t n);
/**
 * @brief Perform Elem Gelu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 * @param n [in] N value.
 */
void SNEPPX_elem_gelu_avx2(const float* A, float* C, size_t n);
/**
 * @brief Perform Elem Silu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 * @param n [in] N value.
 */
void SNEPPX_elem_silu_avx2(const float* A, float* C, size_t n);
/**
 * @brief Perform Elem Tanh Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 * @param n [in] N value.
 */
void SNEPPX_elem_tanh_avx2(const float* A, float* C, size_t n);

/* Fused bias + activation: C = act(A @ W + bias) */
/**
 * @brief Perform Fused Linear Bias Avx2.
 *
 * @param A [in] A value.
 * @param W [in] W value.
 * @param bias [in] Bias value.
 * @param C [out] C value.
 * @param M [in] M value.
 * @param N [in] N value.
 * @param K [in] K value.
 * @param none [out] None value.
 * @param relu [in] Relu value.
 * @param gelu [in] Gelu value.
 */
void SNEPPX_fused_linear_bias_avx2(const float* A, const float* W,
                                   const float* bias, float* C,
                                   int M, int N, int K,
                                   int act /*0=none,1=relu,2=gelu,3=silu*/);

/* -------------------------------------------------------------------------
 * Reductions
 * ------------------------------------------------------------------------- */

/**
 * @brief Perform Reduce Sum Avx2.
 *
 * @param A [in] A value.
 * @param n [in] N value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_reduce_sum_avx2(const float* A, size_t n);
/**
 * @brief Perform Reduce Max Avx2.
 *
 * @param A [in] A value.
 * @param n [in] N value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_reduce_max_avx2(const float* A, size_t n);
/**
 * @brief Perform Reduce Rowmax Avx2.
 *
 * @param A [in] A value.
 * @param out [out] Out value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 */
void  SNEPPX_reduce_rowmax_avx2(const float* A, float* out, int rows, int cols);
/**
 * @brief Perform Reduce Rowsum Exp Avx2.
 *
 * @param A [in] A value.
 * @param out [out] Out value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 */
void  SNEPPX_reduce_rowsum_exp_avx2(const float* A, float* out,
                                    int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_SIMD_GEMM_H */

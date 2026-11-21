#include "simd_gemm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if defined(_MSC_VER)
#include <intrin.h>
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


#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#endif

/* _mm256_exp_ps is an Intel SVML intrinsic not provided by GCC/Clang.
 * Provide a portable per-lane scalar fallback so the AVX2 SiLU path builds
 * and runs correctly on non-Intel compilers. */
#if (defined(__GNUC__) || defined(__clang__)) && defined(__AVX2__)
static inline __m256 sneppx_mm256_exp_ps(__m256 x) {
    float v[8];
    _mm256_storeu_ps(v, x);
    for (int i = 0; i < 8; i++) v[i] = expf(v[i]);
    return _mm256_loadu_ps(v);
}
#define _mm256_exp_ps(x) sneppx_mm256_exp_ps(x)
#endif

/* =========================================================================
 * Runtime SIMD detection
 * ========================================================================= */

/**
 * @brief Perform Simd Detect.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSimdLevel SNEPPX_simd_detect(void) {
#if defined(__AVX512F__)
    return SNEPPX_SIMD_AVX512;
#elif defined(__AVX2__)
    return SNEPPX_SIMD_AVX2;
#elif defined(__SSE4_2__)
    return SNEPPX_SIMD_SSE42;
#else
    /* Runtime CPUID check on x86 */
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 1);
    int ecx = regs[2], edx = regs[1];
    int has_sse42 = (ecx & (1 << 20)) != 0;
    int has_avx = (ecx & (1 << 28)) != 0;
    if (has_avx) {
        __cpuid(regs, 7);
        int has_avx2 = (regs[1] & (1 << 5)) != 0;
        if (has_avx2) {
            int regs7[4];
            __cpuidex(regs7, 7, 0);
            int has_avx512f = (regs7[1] & (1 << 16)) != 0;
            if (has_avx512f) return SNEPPX_SIMD_AVX512;
            return SNEPPX_SIMD_AVX2;
        }
    }
    if (has_sse42) return SNEPPX_SIMD_SSE42;
    return SNEPPX_SIMD_SCALAR;
#elif defined(__GNUC__)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1 << 28)) { /* OSXSAVE + AVX */
            unsigned int eax7, ebx7, ecx7, edx7;
            if (__get_cpuid(7, &eax7, &ebx7, &ecx7, &edx7)) {
                if (ebx7 & (1 << 5)) { /* AVX2 */
                    if (ebx7 & (1 << 16)) return SNEPPX_SIMD_AVX512; /* AVX512F */
                    return SNEPPX_SIMD_AVX2;
                }
            }
        }
        if (ecx & (1 << 20)) return SNEPPX_SIMD_SSE42; /* SSE4.2 */
    }
    return SNEPPX_SIMD_SCALAR;
#else
    return SNEPPX_SIMD_SCALAR;
#endif
#endif
}

/**
 * @brief Perform Simd Level Name.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_simd_level_name(SNEPPXSimdLevel lvl) {
    switch (lvl) {
        case SNEPPX_SIMD_SCALAR: return "scalar";
        case SNEPPX_SIMD_SSE42:  return "sse4.2";
        case SNEPPX_SIMD_AVX2:   return "avx2";
        case SNEPPX_SIMD_AVX512: return "avx512";
        default: return "unknown";
    }
}

/* =========================================================================
 * Scalar GEMM (blocked)
 * ========================================================================= */

#ifndef SNEPPX_GEMM_BLOCK
#define SNEPPX_GEMM_BLOCK 64
#endif

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
 */
void SNEPPX_gemm_scalar(const float* A, const float* B, float* C,
                         int M, int N, int K, float alpha, float beta) {
    /* Zero / scale C */
    for (int i = 0; i < M * N; i++) C[i] *= beta;

    const int BM = SNEPPX_GEMM_BLOCK, BN = SNEPPX_GEMM_BLOCK, BK = SNEPPX_GEMM_BLOCK;
    for (int i0 = 0; i0 < M; i0 += BM) {
        int imax = (i0 + BM < M) ? i0 + BM : M;
        for (int j0 = 0; j0 < N; j0 += BN) {
            int jmax = (j0 + BN < N) ? j0 + BN : N;
            for (int k0 = 0; k0 < K; k0 += BK) {
                int kmax = (k0 + BK < K) ? k0 + BK : K;
                for (int i = i0; i < imax; i++) {
                    const float* a_row = A + (size_t)i * K;
                    float* c_row = C + (size_t)i * N;
                    for (int j = j0; j < jmax; j++) {
                        float acc = 0.0f;
                        for (int k = k0; k < kmax; k++) {
                            acc += a_row[k] * B[(size_t)k * N + j];
                        }
                        c_row[j] += alpha * acc;
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * AVX2 GEMM (8-wide FMAs)
 * ========================================================================= */

#if defined(__AVX2__)

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
 */
void SNEPPX_gemm_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M * N; i++) C[i] *= beta;

    const int BM = 64, BN = 256, BK = 128;  /* tuned for L2 */
    for (int i0 = 0; i0 < M; i0 += BM) {
        int imax = (i0 + BM < M) ? i0 + BM : M;
        for (int j0 = 0; j0 < N; j0 += BN) {
            int jmax = (j0 + BN < N) ? j0 + BN : N;
            for (int k0 = 0; k0 < K; k0 += BK) {
                int kmax = (k0 + BK < K) ? k0 + BK : K;
                for (int i = i0; i < imax; i++) {
                    const float* a_row = A + (size_t)i * K;
                    float* c_row = C + (size_t)i * N;
                    for (int j = j0; j < jmax; j += 8) {
                        __m256 acc = _mm256_setzero_ps();
                        for (int k = k0; k < kmax; k++) {
                            __m256 a_vec = _mm256_set1_ps(a_row[k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[(size_t)k * N + j]);
                            acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                        }
                        __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
                        c_vec = _mm256_fmadd_ps(_mm256_set1_ps(alpha), acc, c_vec);
                        _mm256_storeu_ps(&c_row[j], c_vec);
                    }
                }
            }
        }
    }
}

/**
 * @brief Perform Elem Add Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 */
void SNEPPX_elem_add_avx2(const float* A, const float* B, float* C, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        _mm256_storeu_ps(&C[i], _mm256_add_ps(a, b));
    }
    for (; i < n; i++) C[i] = A[i] + B[i];
}

/**
 * @brief Perform Elem Mul Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 */
void SNEPPX_elem_mul_avx2(const float* A, const float* B, float* C, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(&A[i]);
        __m256 b = _mm256_loadu_ps(&B[i]);
        _mm256_storeu_ps(&C[i], _mm256_mul_ps(a, b));
    }
    for (; i < n; i++) C[i] = A[i] * B[i];
}

static inline float sneppx_gelu_scalar(float x) {
    /* tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x+0.044715*x^3))) */
    float c = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

/**
 * @brief Perform Elem Gelu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_gelu_avx2(const float* A, float* C, size_t n) {
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&A[i]);
        /* approximate gelu per-lane with tanh */
        float buf[8];
        _mm256_storeu_ps(buf, x);
        for (int j = 0; j < 8; j++) buf[j] = sneppx_gelu_scalar(buf[j]);
        _mm256_storeu_ps(&C[i], _mm256_loadu_ps(buf));
    }
    for (; i < n; i++) C[i] = sneppx_gelu_scalar(A[i]);
}

/**
 * @brief Perform Elem Relu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_relu_avx2(const float* A, float* C, size_t n) {
    size_t i;
    __m256 zero = _mm256_setzero_ps();
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&A[i]);
        _mm256_storeu_ps(&C[i], _mm256_max_ps(zero, x));
    }
    for (; i < n; i++) C[i] = A[i] > 0 ? A[i] : 0;
}

/**
 * @brief Perform Elem Silu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_silu_avx2(const float* A, float* C, size_t n) {
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&A[i]);
        __m256 s = _mm256_rcp_ps(_mm256_add_ps(_mm256_set1_ps(1.0f),
                     _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
        _mm256_storeu_ps(&C[i], _mm256_mul_ps(x, s));
    }
    for (; i < n; i++) C[i] = A[i] / (1.0f + expf(-A[i]));
}

/**
 * @brief Perform Elem Tanh Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_tanh_avx2(const float* A, float* C, size_t n) {
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(&A[i]);
        float buf[8];
        _mm256_storeu_ps(buf, x);
        for (int j = 0; j < 8; j++) buf[j] = tanhf(buf[j]);
        _mm256_storeu_ps(&C[i], _mm256_loadu_ps(buf));
    }
    for (; i < n; i++) C[i] = tanhf(A[i]);
}

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
 */
void SNEPPX_fused_linear_bias_avx2(const float* A, const float* W,
                                   const float* bias, float* C,
                                   int M, int N, int K, int act) {
    for (int i = 0; i < M; i++) {
        const float* a_row = A + (size_t)i * K;
        float* c_row = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            float acc = (bias ? bias[j] : 0.0f);
            for (int k = 0; k < K; k++) acc += a_row[k] * W[(size_t)k * N + j];
            float v = acc;
            if (act == 1) v = v > 0 ? v : 0;
            else if (act == 2) v = sneppx_gelu_scalar(v);
            else if (act == 3) v = v / (1.0f + expf(-v));
            c_row[j] = v;
        }
    }
}

/**
 * @brief Perform Reduce Sum Avx2.
 *
 * @param A [in] A value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_reduce_sum_avx2(const float* A, size_t n) {
    float sum = 0.0f;
    size_t i;
    __m256 acc = _mm256_setzero_ps();
    for (i = 0; i + 8 <= n; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(&A[i]));
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    for (int j = 0; j < 8; j++) sum += buf[j];
    for (; i < n; i++) sum += A[i];
    return sum;
}

/**
 * @brief Perform Reduce Max Avx2.
 *
 * @param A [in] A value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_reduce_max_avx2(const float* A, size_t n) {
    float m = -1e30f;
    size_t i;
    __m256 acc = _mm256_set1_ps(-1e30f);
    for (i = 0; i + 8 <= n; i += 8) acc = _mm256_max_ps(acc, _mm256_loadu_ps(&A[i]));
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    for (int j = 0; j < 8; j++) m = (buf[j] > m) ? buf[j] : m;
    for (; i < n; i++) m = (A[i] > m) ? A[i] : m;
    return m;
}

/**
 * @brief Perform Reduce Rowmax Avx2.
 *
 * @param A [in] A value.
 * @param out [out] Out value.
 * @param rows [in] Rows value.
 */
void SNEPPX_reduce_rowmax_avx2(const float* A, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        out[r] = SNEPPX_reduce_max_avx2(&A[(size_t)r * cols], cols);
}

/**
 * @brief Perform Reduce Rowsum Exp Avx2.
 *
 * @param A [in] A value.
 * @param out [out] Out value.
 * @param rows [in] Rows value.
 */
void SNEPPX_reduce_rowsum_exp_avx2(const float* A, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float s = 0.0f;
        for (int c = 0; c < cols; c++) s += expf(A[(size_t)r * cols + c]);
        out[r] = s;
    }
}

#else /* No AVX2: provide scalar fallbacks */

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
 */
void SNEPPX_gemm_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, float alpha, float beta) {
    SNEPPX_gemm_scalar(A, B, C, M, N, K, alpha, beta);
}
/**
 * @brief Perform Elem Add Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 */
void SNEPPX_elem_add_avx2(const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; i++) C[i] = A[i] + B[i];
}
/**
 * @brief Perform Elem Mul Avx2.
 *
 * @param A [in] A value.
 * @param B [in] B value.
 * @param C [out] C value.
 */
void SNEPPX_elem_mul_avx2(const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; i++) C[i] = A[i] * B[i];
}
/**
 * @brief Perform Elem Gelu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_gelu_avx2(const float* A, float* C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float c = 0.7978845608f;
        C[i] = 0.5f * A[i] * (1.0f + tanhf(c * (A[i] + 0.044715f * A[i]*A[i]*A[i])));
    }
}
/**
 * @brief Perform Elem Relu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_relu_avx2(const float* A, float* C, size_t n) {
    for (size_t i = 0; i < n; i++) C[i] = A[i] > 0 ? A[i] : 0;
}
/**
 * @brief Perform Elem Silu Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_silu_avx2(const float* A, float* C, size_t n) {
    for (size_t i = 0; i < n; i++) C[i] = A[i] / (1.0f + expf(-A[i]));
}
/**
 * @brief Perform Elem Tanh Avx2.
 *
 * @param A [in] A value.
 * @param C [out] C value.
 */
void SNEPPX_elem_tanh_avx2(const float* A, float* C, size_t n) {
    for (size_t i = 0; i < n; i++) C[i] = tanhf(A[i]);
}
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
 */
void SNEPPX_fused_linear_bias_avx2(const float* A, const float* W,
                                   const float* bias, float* C,
                                   int M, int N, int K, int act) {
    for (int i = 0; i < M; i++) {
        const float* a_row = A + (size_t)i * K;
        float* c_row = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            float acc = (bias ? bias[j] : 0.0f);
            for (int k = 0; k < K; k++) acc += a_row[k] * W[(size_t)k * N + j];
            float v = acc;
            if (act == 1) v = v > 0 ? v : 0;
            else if (act == 2) v = 0.5f*v*(1+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
            else if (act == 3) v = v / (1.0f + expf(-v));
            c_row[j] = v;
        }
    }
}
/**
 * @brief Perform Reduce Sum Avx2.
 *
 * @param A [in] A value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_reduce_sum_avx2(const float* A, size_t n) {
    float s = 0; for (size_t i = 0; i < n; i++) s += A[i]; return s;
}
/**
 * @brief Perform Reduce Max Avx2.
 *
 * @param A [in] A value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_reduce_max_avx2(const float* A, size_t n) {
    float m = -1e30f; for (size_t i = 0; i < n; i++) m = (A[i]>m)?A[i]:m; return m;
}
/**
 * @brief Perform Reduce Rowmax Avx2.
 *
 * @param A [in] A value.
 * @param out [out] Out value.
 * @param rows [in] Rows value.
 */
void SNEPPX_reduce_rowmax_avx2(const float* A, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) out[r] = SNEPPX_reduce_max_avx2(&A[(size_t)r*cols], cols);
}
/**
 * @brief Perform Reduce Rowsum Exp Avx2.
 *
 * @param A [in] A value.
 * @param out [out] Out value.
 * @param rows [in] Rows value.
 */
void SNEPPX_reduce_rowsum_exp_avx2(const float* A, float* out, int rows, int cols) {
    for (int r = 0; r < rows; r++) { float s = 0; for (int c=0;c<cols;c++) s+=expf(A[(size_t)r*cols+c]); out[r]=s; }
}

#endif
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/* =========================================================================
 * SSE42 / AVX512 shims (use AVX2 or scalar; real SSE/AVX512 kept minimal)
 * ========================================================================= */

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
 */
void SNEPPX_gemm_sse42(const float* A, const float* B, float* C,
                       int M, int N, int K, float alpha, float beta) {
    SNEPPX_gemm_avx2(A, B, C, M, N, K, alpha, beta);
}

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
 */
void SNEPPX_gemm_avx512(const float* A, const float* B, float* C,
                        int M, int N, int K, float alpha, float beta) {
    SNEPPX_gemm_avx2(A, B, C, M, N, K, alpha, beta);
}

/* =========================================================================
 * Dispatch + batched
 * ========================================================================= */

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
 */
void SNEPPX_gemm(const float* A, const float* B, float* C,
                 int M, int N, int K, float alpha, float beta) {
    SNEPPX_gemm_avx2(A, B, C, M, N, K, alpha, beta);
}

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
 */
void SNEPPX_gemm_batched(const float* A, const float* B, float* C,
                         int batch, int M, int N, int K,
                         float alpha, float beta) {
    size_t a_stride = (size_t)M * K;
    size_t b_stride = (size_t)K * N;
    size_t c_stride = (size_t)M * N;
    for (int b = 0; b < batch; b++) {
        SNEPPX_gemm(A + b * a_stride, B + b * b_stride, C + b * c_stride,
                    M, N, K, alpha, beta);
    }
}

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
 */
void SNEPPX_gemm_batched_strided(const float* A, const float* B, float* C,
                                 int batch, int M, int N, int K,
                                 int stride_a, int stride_b, int stride_c,
                                 float alpha, float beta) {
    for (int b = 0; b < batch; b++) {
        SNEPPX_gemm(A + (size_t)b * stride_a, B + (size_t)b * stride_b,
                    C + (size_t)b * stride_c, M, N, K, alpha, beta);
    }
}

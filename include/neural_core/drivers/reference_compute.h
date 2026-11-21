#ifndef SNEPPX_REFERENCE_COMPUTE_H
#define SNEPPX_REFERENCE_COMPUTE_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
/*
 * SNEPPX - Reference Compute
 *
 * WHAT
 *   Reference Compute.
 *
 * CONCEPT
 *   Provides the Reference Compute.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/* Portable CPU reference implementations of the operations the Metal and
 * oneAPI backends expose. When the vendor GPU runtime is unavailable (or the
 * backend is built on a platform without the SDK) these provide a real,
 * functional compute path so the backend is not a silent no-op. */

#ifdef _WIN32
  #include <string.h>
  #define sneppx_ref_stricmp _stricmp
#else
  #include <strings.h>
  #define sneppx_ref_stricmp strcasecmp
#endif

static inline void sneppx_ref_gemm(int M, int N, int K,
                                   const float* A, const float* B, float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
    }
}

static inline void sneppx_ref_elementwise(const char* op,
                                          float* dst, const float* src,
                                          size_t n, float alpha) {
    if (op && (sneppx_ref_stricmp(op, "relu") == 0 || sneppx_ref_stricmp(op, "gelu") == 0)) {
        for (size_t i = 0; i < n; i++) {
            float x = src[i] * alpha;
            dst[i] = x > 0.0f ? x : 0.0f;
        }
    } else if (op && sneppx_ref_stricmp(op, "sigmoid") == 0) {
        for (size_t i = 0; i < n; i++)
            dst[i] = 1.0f / (1.0f + expf(-src[i] * alpha));
    } else if (op && sneppx_ref_stricmp(op, "tanh") == 0) {
        for (size_t i = 0; i < n; i++)
            dst[i] = tanhf(src[i] * alpha);
    } else { /* default: scaled add */
        for (size_t i = 0; i < n; i++)
            dst[i] = src[i] * alpha;
    }
}

static inline void sneppx_ref_layernorm(float* out, const float* in,
                                        const float* gamma, const float* beta,
                                        size_t rows, size_t cols, float eps) {
    for (size_t r = 0; r < rows; r++) {
        const float* x = in + r * cols;
        float* y = out + r * cols;
        float mean = 0.0f, var = 0.0f;
        for (size_t c = 0; c < cols; c++) mean += x[c];
        mean /= (float)cols;
        for (size_t c = 0; c < cols; c++) {
            float d = x[c] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float inv = 1.0f / sqrtf(var + eps);
        for (size_t c = 0; c < cols; c++) {
            float v = (x[c] - mean) * inv;
            if (gamma) v *= gamma[c];
            if (beta) v += beta[c];
            y[c] = v;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_REFERENCE_COMPUTE_H */

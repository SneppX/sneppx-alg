#include "position_encoding.h"
#include <math.h>
#include <stdlib.h>
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



/**
 * @brief Perform Rope Precompute Freqs.
 *
 * @param cos [out] Cos value.
 * @param sin [out] Sin value.
 * @param max_seq [in] Max Seq value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rope_precompute_freqs(float* cos, float* sin, int max_seq, int dim, float base) {
    if (!cos || !sin || max_seq <= 0 || dim <= 0) return -1;
    int half_dim = dim / 2;
    for (int pos = 0; pos < max_seq; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float theta = (float)pos / powf(base, 2.0f * d / (float)dim);
            cos[pos * half_dim + d] = cosf(theta);
            sin[pos * half_dim + d] = sinf(theta);
        }
    }
    return 0;
}

/**
 * @brief Perform Rope Apply Freqs.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param cos [in] Cos value.
 * @param sin [in] Sin value.
 * @param batch [in] Batch value.
 * @param seq [in] Seq value.
 * @param heads [in] Heads value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rope_apply_freqs(const float* x, float* output, const float* cos, const float* sin, int batch, int seq, int heads, int dim) {
    if (!x || !output || !cos || !sin || batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return -1;
    int half_dim = dim / 2;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < heads; h++) {
                int base = ((b * heads + h) * seq + s) * dim;
                for (int d = 0; d < half_dim; d++) {
                    float x0 = x[base + 2 * d];
                    float x1 = x[base + 2 * d + 1];
                    float c = cos[s * half_dim + d];
                    float si = sin[s * half_dim + d];
                    output[base + 2 * d] = x0 * c - x1 * si;
                    output[base + 2 * d + 1] = x0 * si + x1 * c;
                }
            }
        }
    }
    return 0;
}

/**
 * @brief Perform Rope Apply Freqs Inplace.
 *
 * @param x [out] X value.
 * @param cos [in] Cos value.
 * @param sin [in] Sin value.
 * @param batch [in] Batch value.
 * @param seq [in] Seq value.
 * @param heads [in] Heads value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rope_apply_freqs_inplace(float* x, const float* cos, const float* sin, int batch, int seq, int heads, int dim) {
    return SNEPPX_rope_apply_freqs(x, x, cos, sin, batch, seq, heads, dim);
}

/**
 * @brief Perform Rope Precompute Tensor.
 *
 * @param max_seq [in] Max Seq value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
float* SNEPPX_rope_precompute_tensor(int max_seq, int dim, float base) {
    int half_dim = dim / 2;
    float* cache = (float*)malloc((size_t)max_seq * half_dim * 2 * sizeof(float));
    if (!cache) return NULL;
    float* cos = cache;
    float* sin = cache + (size_t)max_seq * half_dim;
    if (SNEPPX_rope_precompute_freqs(cos, sin, max_seq, dim, base) != 0) {
        free(cache);
        return NULL;
    }
    return cache;
}

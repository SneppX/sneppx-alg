#include "position_encoding.h"
#include <math.h>
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
 * @brief Compute Yarn Pre.
 *
 * @param cos [out] Cos value.
 * @param sin [out] Sin value.
 * @param max_seq [in] Max Seq value.
 * @param dim [in] Dim value.
 * @param base [in] Base value.
 * @param scale [in] Scale value.
 * @param alpha [in] Alpha value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_yarn_precompute(float* cos, float* sin, int max_seq, int dim, float base, float scale, float alpha, float beta) {
    if (!cos || !sin || max_seq <= 0 || dim <= 0) return -1;
    (void)scale;
    int half_dim = dim / 2;
    for (int pos = 0; pos < max_seq; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float theta = (float)pos / powf(base, 2.0f * d / (float)dim);
            float t = (float)pos / (float)max_seq;
            float ram = (1.0f - t) * alpha + t * beta;
            cos[pos * half_dim + d] = cosf(theta * ram);
            sin[pos * half_dim + d] = sinf(theta * ram);
        }
    }
    return 0;
}

/**
 * @brief Apply Yarn.
 *
 * @param x [in] X value.
 * @param output [out] Output value.
 * @param cos [in] Cos value.
 * @param sin [in] Sin value.
 * @param batch [in] Batch value.
 * @param seq [in] Seq value.
 * @param heads [in] Heads value.
 * @param dim [in] Dim value.
 * @param alpha [in] Alpha value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_yarn_apply(const float* x, float* output, const float* cos, const float* sin, int batch, int seq, int heads, int dim, float alpha, float beta) {
    if (!x || !output || !cos || !sin || batch <= 0 || seq <= 0 || heads <= 0 || dim <= 0) return -1;
    int half_dim = dim / 2;
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < heads; h++) {
                int base = ((b * heads + h) * seq + s) * dim;
                float t = (float)s / (float)seq;
                float ram = (1.0f - t) * alpha + t * beta;
                for (int d = 0; d < half_dim; d++) {
                    float x0 = x[base + 2 * d];
                    float x1 = x[base + 2 * d + 1];
                    float c = cos[s * half_dim + d];
                    float si = sin[s * half_dim + d];
                    float cr = cosf(ram * acosf(c));
                    float sr = sinf(ram * asinf(si));
                    output[base + 2 * d] = x0 * cr - x1 * sr;
                    output[base + 2 * d + 1] = x0 * sr + x1 * cr;
                }
            }
        }
    }
    return 0;
}

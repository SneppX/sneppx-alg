#include "quantization.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#ifndef SNEPPX_MIN
#define SNEPPX_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef SNEPPX_MAX
#define SNEPPX_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef SNEPPX_CLAMP
#define SNEPPX_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif
/*
 * SNEPPX - GPTQ Weight Quantization
 *
 * WHAT
 *   GPTQ Weight Quantization.
 *
 * CONCEPT
 *   GPTQ Weight Quantization implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/*
 * GPTQ (Post-Training Quantization):
 *
 * One-shot weight quantization using an approximate second-order method.
 * Processes columns left-to-right, quantizing each and updating the
 * remaining weights to compensate for quantization error.
 *
 * Reference: "GPTQ: Accurate Post-Training Quantization for Generative
 *             Pre-trained Transformers" (Frantar et al., 2023)
 */

/**
 * @brief Perform Gptq Compute Hessian.
 *
 * @param hessian [out] Hessian value.
 * @param activation [in] Activation value.
 * @param n [in] N value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gptq_compute_hessian(float* hessian, const float* activation,
                                 size_t n, size_t dim)
{
    if (!hessian || !activation || n == 0 || dim == 0) return -1;
    memset(hessian, 0, dim * dim * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        const float* row = activation + i * dim;
        for (size_t r = 0; r < dim; r++) {
            for (size_t c = r; c < dim; c++) {
                hessian[r * dim + c] += row[r] * row[c];
            }
        }
    }
    for (size_t r = 0; r < dim; r++) {
        for (size_t c = 0; c < r; c++) {
            hessian[r * dim + c] = hessian[c * dim + r];
        }
    }
    float reg = 1e-5f * hessian[0];
    if (reg < 1e-8f) reg = 1e-5f;
    for (size_t i = 0; i < dim; i++) {
        hessian[i * dim + i] += reg;
    }
    return 0;
}

static int _cholesky_inv(float* h, size_t dim)
{
    float* L = (float*)calloc(dim * dim, sizeof(float));
    if (!L) return -1;
    for (size_t j = 0; j < dim; j++) {
        float s = 0.0f;
        for (size_t k = 0; k < j; k++) {
            s += L[j * dim + k] * L[j * dim + k];
        }
        float d = h[j * dim + j] - s;
        if (d < 1e-10f) { free(L); return -1; }
        L[j * dim + j] = sqrtf(d);
        for (size_t i = j + 1; i < dim; i++) {
            s = 0.0f;
            for (size_t k = 0; k < j; k++) {
                s += L[i * dim + k] * L[j * dim + k];
            }
            L[i * dim + j] = (h[i * dim + j] - s) / L[j * dim + j];
        }
    }
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            if (j > i) {
                h[i * dim + j] = 0.0f;
            } else if (j == i) {
                h[i * dim + j] = 1.0f / L[i * dim + i];
            } else {
                float s = 0.0f;
                for (size_t k = j; k < dim; k++) {
                    s += L[i * dim + k] * h[k * dim + j];
                }
                h[i * dim + j] = -s / L[i * dim + i];
            }
        }
    }
    free(L);
    return 0;
}

/**
 * @brief Perform Gptq Quantize Block.
 *
 * @param w [out] W value.
 * @param h_inv [out] H Inv value.
 * @param qw [out] Qw value.
 * @param scale [out] Scale value.
 * @param zero [out] Zero value.
 * @param block_size [in] Block Size value.
 * @param bits [in] Bits value.
 * @param sym [in] Sym value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gptq_quantize_block(float* w, float* h_inv,
                                float* qw, float* scale, int32_t* zero,
                                size_t block_size, int bits, int sym)
{
    if (!w || !h_inv || !qw || !scale) return -1;
    int qmax = (1 << (bits - 1)) - 1;
    int qmin = -qmax;
    if (!sym) { qmax = (1 << bits) - 1; qmin = 0; }
    size_t cols = block_size;
    float max_abs = 0.0f;
    for (size_t j = 0; j < cols; j++) {
        float a = fabsf(w[j]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < FLT_MIN) {
        *scale = 1.0f;
        if (zero) *zero = 0;
        memset(qw, 0, cols * sizeof(float));
        return 0;
    }
    *scale = max_abs / (float)qmax;
    if (zero) *zero = 0;
    float inv_scale = 1.0f / (*scale);
    for (size_t j = 0; j < cols; j++) {
        int qi = (int)roundf(w[j] * inv_scale);
        qi = SNEPPX_CLAMP(qi, qmin, qmax);
        float q = (float)qi * (*scale);
        float err = q - w[j];
        qw[j] = q;
        float h_col = h_inv[j * cols + j];
        if (fabsf(h_col) > 1e-10f) {
            float coef = -err / h_col;
            for (size_t k = j + 1; k < cols; k++) {
                w[k] += coef * h_inv[k * cols + j];
            }
        }
    }
    return 0;
}

/**
 * @brief Quantize Gptq.
 *
 * @param weights [out] Weights value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 * @param group_size [in] Group Size value.
 * @param bits [in] Bits value.
 * @param qweight [out] Qweight value.
 * @param scales [out] Scales value.
 * @param zeros [out] Zeros value.
 * @param sym [in] Sym value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gptq_quantize(float* weights, size_t rows, size_t cols,
                          int group_size, int bits,
                          SNEPPXDtype* qweight, float* scales,
                          int32_t* zeros, int sym)
{
    if (!weights || !qweight || !scales || rows == 0 || cols == 0) return -1;
    if (group_size <= 0) group_size = 128;
    if (bits <= 0 || bits > 16) bits = 8;
    size_t num_groups = (cols + group_size - 1) / group_size;
    for (size_t r = 0; r < rows; r++) {
        float* row = weights + r * cols;
        for (size_t g = 0; g < num_groups; g++) {
            size_t g_start = g * group_size;
            size_t g_end = SNEPPX_MIN(g_start + group_size, cols);
            size_t gs = g_end - g_start;
            float* block = (float*)malloc(gs * sizeof(float));
            float* h_inv = (float*)malloc(gs * gs * sizeof(float));
            float* qblock = (float*)malloc(gs * sizeof(float));
            if (!block || !h_inv || !qblock) {
                free(block); free(h_inv); free(qblock);
                return -1;
            }
            memcpy(block, row + g_start, gs * sizeof(float));
            memset(h_inv, 0, gs * gs * sizeof(float));
            for (size_t i = 0; i < gs; i++) h_inv[i * gs + i] = 1.0f;
            if (_cholesky_inv(h_inv, gs) != 0) {
                for (size_t i = 0; i < gs; i++) {
                    for (size_t j = 0; j < gs; j++) {
                        h_inv[i * gs + j] = (i == j) ? 1.0f : 0.0f;
                    }
                }
            }
            float scale;
            int32_t zero = 0;
            SNEPPX_gptq_quantize_block(block, h_inv, qblock,
                                        &scale, &zero, gs, bits, sym);
            scales[r * num_groups + g] = scale;
            if (zeros) zeros[r * num_groups + g] = zero;
            size_t elem_size = (bits <= 8) ? 1 : 2;
            if (elem_size == 1) {
                for (size_t j = 0; j < gs; j++) {
                    int qi = (int)roundf(block[j] / scale);
                    ((int8_t*)qweight)[r * cols + g_start + j] =
                        (int8_t)SNEPPX_CLAMP(qi, -128, 127);
                }
            }
            free(block); free(h_inv); free(qblock);
        }
    }
    return 0;
}

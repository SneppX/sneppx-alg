#include "sparse_attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/*
 * SNEPPX - Sparse Attention
 *
 * WHAT
 *   Sparse Attention.
 *
 * CONCEPT
 *   Provides attention mechanisms.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Perform Sparse Attn Build Mask.
 *
 * @param mask [out] Mask value.
 * @param cfg [in] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sparse_attn_build_mask(int* mask, const SNEPPXSparseAttnConfig* cfg, int seq) {
    if (!mask || !cfg || seq <= 0) return -1;
    memset(mask, 0, (size_t)seq * seq * sizeof(int));
    switch (cfg->pattern) {
        case SNEPPX_SPARSE_TOP_K:
            return 0;
        case SNEPPX_SPARSE_STRIDED: {
            int s = cfg->stride > 0 ? cfg->stride : 1;
            int w = cfg->window_size > 0 ? cfg->window_size : seq;
            for (int i = 0; i < seq; i++)
                for (int j = 0; j < seq; j++)
                    if ((i - j) <= w && (i - j) >= -w && (j % s == 0 || i % s == 0))
                        mask[i * seq + j] = 1;
            break;
        }
        case SNEPPX_SPARSE_RANDOM: {
            unsigned int rng = cfg->seed ? cfg->seed : 42;
            float rate = 0.1f;
            int w = cfg->window_size > 0 ? cfg->window_size : 64;
            for (int i = 0; i < seq; i++) {
                for (int j = 0; j < seq; j++) {
                    if (abs(i - j) <= w) { mask[i * seq + j] = 1; continue; }
                    rng = rng * 1103515245U + 12345U;
                    if ((rng & 0xFFFF) < (unsigned int)(rate * 65535.0f))
                        mask[i * seq + j] = 1;
                }
            }
            break;
        }
        case SNEPPX_SPARSE_BLOCK_LOCAL: {
            int bs = cfg->block_size > 0 ? cfg->block_size : 32;
            int w = cfg->window_size > 0 ? cfg->window_size : bs;
            for (int qi = 0; qi < seq; qi += bs)
                for (int kj = 0; kj < seq; kj += bs)
                    if (abs(qi - kj) <= w)
                        for (int i = 0; i < bs && qi + i < seq; i++)
                            for (int j = 0; j < bs && kj + j < seq; j++)
                                mask[(qi + i) * seq + (kj + j)] = 1;
            break;
        }
    }
    return 0;
}

/**
 * @brief Run the forward pass for Sparse Attn.
 *
 * @param q [in] Q value.
 * @param k [in] K value.
 * @param v [in] V value.
 * @param output [out] Output value.
 * @param cfg [in] Cfg value.
 * @param batch [in] Batch value.
 * @param heads [in] Heads value.
 * @param seq [in] Seq value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sparse_attn_forward(const float* q, const float* k, const float* v, float* output, const SNEPPXSparseAttnConfig* cfg, int batch, int heads, int seq, int dim, float scale) {
    if (!q || !k || !v || !output || !cfg || batch <= 0 || heads <= 0 || seq <= 0 || dim <= 0) return -1;
    int* mask = (int*)malloc((size_t)seq * seq * sizeof(int));
    if (!mask) return -1;
    if (SNEPPX_sparse_attn_build_mask(mask, cfg, seq) != 0) { free(mask); return -1; }
    memset(output, 0, (size_t)batch * heads * seq * dim * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            float* out_head = output + ((size_t)b * heads + h) * seq * dim;
            for (int qi = 0; qi < seq; qi++) {
                float maxv = -FLT_MAX;
                float* row_scores = (float*)malloc(seq * sizeof(float));
                if (!row_scores) { free(mask); return -1; }
                for (int kj = 0; kj < seq; kj++) {
                    if (!mask[qi * seq + kj]) { row_scores[kj] = -FLT_MAX; continue; }
                    float s = 0.0f;
                    int q_off = ((size_t)b * heads + h) * seq * dim + qi * dim;
                    int k_off = ((size_t)b * heads + h) * seq * dim + kj * dim;
                    for (int d = 0; d < dim; d++)
                        s += q[q_off + d] * k[k_off + d];
                    row_scores[kj] = s * scale;
                    if (row_scores[kj] > maxv) maxv = row_scores[kj];
                }
                float sum = 0.0f;
                for (int kj = 0; kj < seq; kj++) {
                    if (row_scores[kj] == -FLT_MAX) continue;
                    row_scores[kj] = expf(row_scores[kj] - maxv);
                    sum += row_scores[kj];
                }
                if (sum > 0.0f)
                    for (int kj = 0; kj < seq; kj++)
                        if (row_scores[kj] != -FLT_MAX) row_scores[kj] /= sum;
                for (int d = 0; d < dim; d++) {
                    float acc = 0.0f;
                    for (int kj = 0; kj < seq; kj++) {
                        if (row_scores[kj] == -FLT_MAX || row_scores[kj] == 0.0f) continue;
                        int v_off = ((size_t)b * heads + h) * seq * dim + kj * dim;
                        acc += row_scores[kj] * v[v_off + d];
                    }
                    out_head[qi * dim + d] = acc;
                }
                free(row_scores);
            }
        }
    }
    free(mask);
    return 0;
}

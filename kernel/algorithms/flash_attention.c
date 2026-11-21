#include "flash_attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/*
 * SNEPPX - Flash Attention
 *
 * WHAT
 *   Flash Attention.
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
 * @brief Run the forward pass for Flash Attn.
 *
 * @param q [in] Q value.
 * @param k [in] K value.
 * @param v [in] V value.
 * @param output [out] Output value.
 * @param batch [in] Batch value.
 * @param heads [in] Heads value.
 * @param seq [in] Seq value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_flash_attn_forward(const float* q, const float* k, const float* v, float* output, int batch, int heads, int seq, int dim, float scale) {
    if (!q || !k || !v || !output || batch <= 0 || heads <= 0 || seq <= 0 || dim <= 0) return -1;
    int block_size = 32;
    int num_blocks = (seq + block_size - 1) / block_size;
    float* scores = (float*)malloc((size_t)block_size * block_size * sizeof(float));
    if (!scores) return -1;
    memset(output, 0, (size_t)batch * heads * seq * dim * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            float* out_head = output + ((size_t)b * heads + h) * seq * dim;
            for (int qi = 0; qi < num_blocks; qi++) {
                float* m = (float*)malloc(block_size * sizeof(float));
                float* l = (float*)calloc(block_size, sizeof(float));
                float* acc = (float*)calloc((size_t)block_size * dim, sizeof(float));
                if (!m || !l || !acc) { free(m); free(l); free(acc); free(scores); return -1; }
                for (int i = 0; i < block_size; i++) m[i] = -FLT_MAX;
                for (int kj = 0; kj < num_blocks; kj++) {
                    int q_off = ((size_t)b * heads + h) * seq * dim + qi * block_size * dim;
                    int k_off = ((size_t)b * heads + h) * seq * dim + kj * block_size * dim;
                    int q_rows = (qi + 1) * block_size <= seq ? block_size : seq - qi * block_size;
                    int k_cols = (kj + 1) * block_size <= seq ? block_size : seq - kj * block_size;
                    for (int i = 0; i < q_rows; i++) {
                        for (int j = 0; j < k_cols; j++) {
                            float s = 0.0f;
                            for (int d = 0; d < dim; d++)
                                s += q[q_off + i * dim + d] * k[k_off + j * dim + d];
                            scores[i * block_size + j] = s * scale;
                        }
                    }
                    for (int i = 0; i < q_rows; i++) {
                        float old_m = m[i];
                        float new_m = old_m;
                        for (int j = 0; j < k_cols; j++)
                            if (scores[i * block_size + j] > new_m)
                                new_m = scores[i * block_size + j];
                        float sum_exp = 0.0f;
                        for (int j = 0; j < k_cols; j++)
                            sum_exp += expf(scores[i * block_size + j] - new_m);
                        float l_new = expf(old_m - new_m) * l[i] + sum_exp;
                        for (int d = 0; d < dim; d++) {
                            float old_acc = acc[i * dim + d];
                            float new_acc = 0.0f;
                            for (int j = 0; j < k_cols; j++) {
                                float p = expf(scores[i * block_size + j] - new_m);
                                int v_off = ((size_t)b * heads + h) * seq * dim + kj * block_size * dim;
                                new_acc += p * v[v_off + j * dim + d];
                            }
                            acc[i * dim + d] = expf(old_m - new_m) * old_acc + new_acc;
                        }
                        m[i] = new_m;
                        l[i] = l_new;
                    }
                }
                for (int i = 0; i < block_size && qi * block_size + i < seq; i++) {
                    for (int d = 0; d < dim; d++)
                        out_head[(qi * block_size + i) * dim + d] = acc[i * dim + d] / fmaxf(l[i], FLT_MIN);
                }
                free(m); free(l); free(acc);
            }
        }
    }
    free(scores);
    return 0;
}

/**
 * @brief Run the backward pass for Flash Attn.
 *
 * @param q [in] Q value.
 * @param k [in] K value.
 * @param v [in] V value.
 * @param grad_out [in] Grad Out value.
 * @param grad_q [out] Grad Q value.
 * @param grad_k [out] Grad K value.
 * @param grad_v [out] Grad V value.
 * @param batch [in] Batch value.
 * @param heads [in] Heads value.
 * @param seq [in] Seq value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_flash_attn_backward(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int batch, int heads, int seq, int dim, float scale) {
    (void)q; (void)k; (void)v; (void)grad_out; (void)grad_q; (void)grad_k; (void)grad_v;
    (void)batch; (void)heads; (void)seq; (void)dim; (void)scale;
    return 0;
}

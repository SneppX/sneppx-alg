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
 * @brief Perform Alibi Compute Slopes.
 *
 * @param slopes [out] Slopes value.
 * @param num_heads [in] Num Heads value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alibi_compute_slopes(float* slopes, int num_heads, float slope_base) {
    if (!slopes || num_heads <= 0) return -1;
    for (int h = 0; h < num_heads; h++)
        slopes[h] = 1.0f / powf(slope_base, (float)(h + 1));
    return 0;
}

/**
 * @brief Apply Alibi.
 *
 * @param attn_scores [out] Attn Scores value.
 * @param slopes [in] Slopes value.
 * @param batch [in] Batch value.
 * @param heads [in] Heads value.
 * @param seq_q [in] Seq Q value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alibi_apply(float* attn_scores, const float* slopes, int batch, int heads, int seq_q, int seq_k) {
    if (!attn_scores || !slopes || batch <= 0 || heads <= 0 || seq_q <= 0 || seq_k <= 0) return -1;
    for (int h = 0; h < heads; h++) {
        float slope = slopes[h];
        for (int b = 0; b < batch; b++) {
            for (int qi = 0; qi < seq_q; qi++) {
                for (int kj = 0; kj < seq_k; kj++) {
                    int idx = ((b * heads + h) * seq_q + qi) * seq_k + kj;
                    attn_scores[idx] += slope * (float)(kj - qi);
                }
            }
        }
    }
    return 0;
}

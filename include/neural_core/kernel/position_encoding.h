#ifndef SNEPPX_POSITION_ENCODING_H
#define SNEPPX_POSITION_ENCODING_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Position Encoding
 *
 * WHAT
 *   Position Encoding.
 *
 * CONCEPT
 *   Provides position encodings.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/**
 * @brief Perform Rope Precompute Freqs.
 *
 * @param cos [out] Cos value.
 * @param sin [out] Sin value.
 * @param max_seq [in] Max Seq value.
 * @param dim [in] Dim value.
 * @param base [in] Base value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rope_precompute_freqs(float* cos, float* sin, int max_seq, int dim, float base);
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
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rope_apply_freqs(const float* x, float* output, const float* cos, const float* sin, int batch, int seq, int heads, int dim);
/**
 * @brief Perform Rope Apply Freqs Inplace.
 *
 * @param x [out] X value.
 * @param cos [in] Cos value.
 * @param sin [in] Sin value.
 * @param batch [in] Batch value.
 * @param seq [in] Seq value.
 * @param heads [in] Heads value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rope_apply_freqs_inplace(float* x, const float* cos, const float* sin, int batch, int seq, int heads, int dim);

/**
 * @brief Perform Rope Precompute Tensor.
 *
 * @param max_seq [in] Max Seq value.
 * @param dim [in] Dim value.
 * @param base [in] Base value.
 *
 * @return Pointer on success, NULL on error.
 */
float* SNEPPX_rope_precompute_tensor(int max_seq, int dim, float base);

/**
 * @brief Perform Alibi Compute Slopes.
 *
 * @param slopes [out] Slopes value.
 * @param num_heads [in] Num Heads value.
 * @param slope_base [in] Slope Base value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alibi_compute_slopes(float* slopes, int num_heads, float slope_base);
/**
 * @brief Apply Alibi.
 *
 * @param attn_scores [out] Attn Scores value.
 * @param slopes [in] Slopes value.
 * @param batch [in] Batch value.
 * @param heads [in] Heads value.
 * @param seq_q [in] Seq Q value.
 * @param seq_k [in] Seq K value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_alibi_apply(float* attn_scores, const float* slopes, int batch, int heads, int seq_q, int seq_k);

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
 * @param beta [in] Beta value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_yarn_precompute(float* cos, float* sin, int max_seq, int dim, float base, float scale, float alpha, float beta);
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
 * @param beta [in] Beta value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_yarn_apply(const float* x, float* output, const float* cos, const float* sin, int batch, int seq, int heads, int dim, float alpha, float beta);

#ifdef __cplusplus
}
#endif

#endif

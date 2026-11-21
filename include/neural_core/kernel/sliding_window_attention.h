#ifndef SNEPPX_SLIDING_WINDOW_ATTENTION_H
#define SNEPPX_SLIDING_WINDOW_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Sliding Window Attention
 *
 * WHAT
 *   Sliding Window Attention.
 *
 * CONCEPT
 *   Provides attention mechanisms.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/**
 * @brief Run the forward pass for Swa.
 *
 * @param q [in] Q value.
 * @param k [in] K value.
 * @param v [in] V value.
 * @param output [out] Output value.
 * @param batch [in] Batch value.
 * @param heads [in] Heads value.
 * @param seq [in] Seq value.
 * @param dim [in] Dim value.
 * @param window_size [in] Window Size value.
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_swa_forward(const float* q, const float* k, const float* v, float* output, int batch, int heads, int seq, int dim, int window_size, float scale);
/**
 * @brief Run the backward pass for Swa.
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
 * @param window_size [in] Window Size value.
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_swa_backward(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int batch, int heads, int seq, int dim, int window_size, float scale);

#ifdef __cplusplus
}
#endif

#endif

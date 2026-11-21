#ifndef SNEPPX_FLASH_ATTENTION_H
#define SNEPPX_FLASH_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
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


extern "C" {
#endif

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
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_flash_attn_forward(const float* q, const float* k, const float* v, float* output, int batch, int heads, int seq, int dim, float scale);
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
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_flash_attn_backward(const float* q, const float* k, const float* v, const float* grad_out, float* grad_q, float* grad_k, float* grad_v, int batch, int heads, int seq, int dim, float scale);

#ifdef __cplusplus
}
#endif

#endif

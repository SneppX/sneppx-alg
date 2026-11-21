#ifndef SNEPPX_SPARSE_ATTENTION_H
#define SNEPPX_SPARSE_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
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


extern "C" {
#endif

typedef enum {
    SNEPPX_SPARSE_TOP_K,
    SNEPPX_SPARSE_STRIDED,
    SNEPPX_SPARSE_RANDOM,
    SNEPPX_SPARSE_BLOCK_LOCAL,
} SNEPPXSparseAttnPattern;

typedef struct {
    SNEPPXSparseAttnPattern pattern;
    int block_size;
    int top_k;
    int stride;
    int window_size;
    unsigned int seed;
} SNEPPXSparseAttnConfig;

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
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sparse_attn_forward(const float* q, const float* k, const float* v, float* output, const SNEPPXSparseAttnConfig* cfg, int batch, int heads, int seq, int dim, float scale);
/**
 * @brief Perform Sparse Attn Build Mask.
 *
 * @param mask [out] Mask value.
 * @param cfg [in] Cfg value.
 * @param seq [in] Seq value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sparse_attn_build_mask(int* mask, const SNEPPXSparseAttnConfig* cfg, int seq);

#ifdef __cplusplus
}
#endif

#endif

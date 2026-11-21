#ifndef SNEPPX_ATTENTION_H
#define SNEPPX_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"

/*
 * SNEPPX - Multi Head Attention Module
 *
 * WHAT
 *   Multi Head Attention Module.
 *
 * CONCEPT
 *   Provides attention mechanisms.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    size_t num_heads;
    size_t head_dim;
    size_t d_model;
    float dropout;
    int use_causal_mask;
    int use_rope;
    float rope_base;
} SNEPPXAttentionConfig;

typedef struct {
    SNEPPXAttentionConfig config;
    SNEPPXTensor* w_q; SNEPPXTensor* b_q;
    SNEPPXTensor* w_k; SNEPPXTensor* b_k;
    SNEPPXTensor* w_v; SNEPPXTensor* b_v;
    SNEPPXTensor* w_o; SNEPPXTensor* b_o;
} SNEPPXAttentionWeights;

typedef struct {
    SNEPPXTensor* k_cache;
    SNEPPXTensor* v_cache;
    size_t seq_len;
} SNEPPXKVCache;

/**
 * @brief Perform Attn Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXAttentionConfig SNEPPX_attn_config_default(void);
/**
 * @brief Create Attn Weights.
 *
 * @param cfg [in] Cfg value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXAttentionWeights* SNEPPX_attn_weights_create(SNEPPXAttentionConfig cfg, unsigned int seed);
/**
 * @brief Destroy Attn Weights.
 *
 * @param w [out] W value.
 */
void SNEPPX_attn_weights_destroy(SNEPPXAttentionWeights* w);
/**
 * @brief Perform Attn Num Params.
 *
 * @param w [in] W value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_attn_num_params(const SNEPPXAttentionWeights* w);
/**
 * @brief Perform Attn Get Params.
 *
 * @param w [in] W value.
 * @param out [out] Out value.
 * @param max [in] Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_attn_get_params(const SNEPPXAttentionWeights* w, SNEPPXTensor** out, size_t max);

/**
 * @brief Run the forward pass for Attn.
 *
 * @param w [in] W value.
 * @param x [in] X value.
 * @param cos_t [out] Cos T value.
 * @param sin_t [out] Sin T value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_attn_forward(const SNEPPXAttentionWeights* w, const SNEPPXTensor* x,
                               SNEPPXTensor* cos_t, SNEPPXTensor* sin_t);
/**
 * @brief Perform Attn Forward Cached.
 *
 * @param w [in] W value.
 * @param x [in] X value.
 * @param cache [out] Cache value.
 * @param cos_t [out] Cos T value.
 * @param sin_t [out] Sin T value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_attn_forward_cached(const SNEPPXAttentionWeights* w, const SNEPPXTensor* x,
                                       SNEPPXKVCache* cache, SNEPPXTensor* cos_t, SNEPPXTensor* sin_t);

/**
 * @brief Create Kv Cache.
 *
 * @param max_batch [in] Max Batch value.
 * @param max_seq [in] Max Seq value.
 * @param num_heads [in] Num Heads value.
 * @param head_dim [in] Head Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXKVCache* SNEPPX_kv_cache_create(size_t max_batch, size_t max_seq, size_t num_heads, size_t head_dim);
/**
 * @brief Destroy Kv Cache.
 *
 * @param cache [out] Cache value.
 */
void SNEPPX_kv_cache_destroy(SNEPPXKVCache* cache);
/**
 * @brief Clear Kv Cache.
 *
 * @param cache [out] Cache value.
 */
void SNEPPX_kv_cache_clear(SNEPPXKVCache* cache);

/**
 * @brief Compute Rope Pre.
 *
 * @param seq_len [in] Seq Len value.
 * @param head_dim [in] Head Dim value.
 * @param base [in] Base value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_rope_precompute(size_t seq_len, size_t head_dim, float base);
/**
 * @brief Apply Rope.
 *
 * @param q [out] Q value.
 * @param k [out] K value.
 * @param cos [in] Cos value.
 * @param sin [in] Sin value.
 * @param offset [in] Offset value.
 */
void SNEPPX_rope_apply(SNEPPXTensor* q, SNEPPXTensor* k, const SNEPPXTensor* cos, const SNEPPXTensor* sin,
                     size_t offset);
/**
 * @brief Perform Tensor Rope.
 *
 * @param x [in] X value.
 * @param cos_table [in] Cos Table value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_rope(const SNEPPXTensor* x, const SNEPPXTensor* cos_table);

/**
 * @brief Perform Batched Matmul.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param transpose_b [in] Transpose B value.
 * @param transpose_a [in] Transpose A value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_batched_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b,
                                 int transpose_b, int transpose_a);

/**
 * @brief Perform Attn Build Train Graph.
 *
 * @param w [out] W value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 * @param cos [out] Cos value.
 * @param sin [out] Sin value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_attn_build_train_graph(SNEPPXAttentionWeights* w, SNEPPXTape* tape,
                                 SNEPPXVariable* input_var,
                                 SNEPPXVariable** weight_vars, size_t num_weights,
                                 SNEPPXVariable** output_var,
                                 SNEPPXTensor* cos, SNEPPXTensor* sin);


#ifdef __cplusplus
}
#endif
#endif

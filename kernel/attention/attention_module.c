#include "multi_head_attention_module.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475f
#endif
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



typedef struct SNEPPXMultiHeadAttention {
    int num_heads;
    int head_dim;
    int hidden_dim;
    int dropout;
    int is_causal;
} SNEPPXMultiHeadAttention;

/**
 * @brief Create Mha.
 *
 * @param num_heads [in] Num Heads value.
 * @param head_dim [in] Head Dim value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param dropout [in] Dropout value.
 * @param is_causal [in] Is Causal value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXMultiHeadAttention* SNEPPX_mha_create(int num_heads, int head_dim, int hidden_dim, int dropout, int is_causal, int use_flash) {
    (void)use_flash;
    SNEPPXMultiHeadAttention* mha = (SNEPPXMultiHeadAttention*)calloc(1, sizeof(SNEPPXMultiHeadAttention));
    if (!mha) return NULL;
    mha->num_heads = num_heads;
    mha->head_dim = head_dim;
    mha->hidden_dim = hidden_dim;
    mha->dropout = dropout;
    mha->is_causal = is_causal;
    return mha;
}

/**
 * @brief Destroy Mha.
 */
void SNEPPX_mha_destroy(SNEPPXMultiHeadAttention* mha) {
    free(mha);
}

/**
 * @brief Run the forward pass for Mha.
 *
 * @param mha [out] Mha value.
 * @param query [in] Query value.
 * @param key [in] Key value.
 * @param value [in] Value value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_mha_forward(SNEPPXMultiHeadAttention* mha, const SNEPPXTensor* query, const SNEPPXTensor* key, const SNEPPXTensor* value, const SNEPPXTensor* mask) {
    if (!mha || !query || !key || !value) return NULL;
    int num_heads = mha->num_heads;
    int head_dim = mha->head_dim;
    int batch = (int)query->shape[0];
    int seq_q = (int)query->shape[1];
    int seq_k = (int)key->shape[1];
    int hidden_dim = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t out_shape[3] = {(size_t)batch, (size_t)seq_q, (size_t)hidden_dim};
    SNEPPXTensor* output = SNEPPX_tensor_create(out_shape, 3, SNEPPX_FLOAT32);
    if (!output) return NULL;
    size_t attn_shape[3] = {(size_t)batch, (size_t)seq_q, (size_t)seq_k};
    SNEPPXTensor* attn_scores = SNEPPX_tensor_create(attn_shape, 3, SNEPPX_FLOAT32);
    if (!attn_scores) { SNEPPX_tensor_destroy(output); return NULL; }
    float* qd = (float*)query->data;
    float* kd = (float*)key->data;
    float* vd = (float*)value->data;
    float* od = (float*)output->data;
    float* ad = (float*)attn_scores->data;
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int qi = 0; qi < seq_q; qi++) {
                for (int ki = 0; ki < seq_k; ki++) {
                    float sum = 0.0f;
                    if (mha->is_causal && ki > qi) { sum = -INFINITY; }
                    else {
                        for (int d = 0; d < head_dim; d++) {
                            float qv = qd[((b * num_heads + h) * seq_q + qi) * head_dim + d];
                            float kv = kd[((b * num_heads + h) * seq_k + ki) * head_dim + d];
                            sum += qv * kv;
                        }
                        sum *= scale;
                    }
                    if (mask) {
                        float* md = (float*)mask->data;
                        sum += md[qi * seq_k + ki];
                    }
                    ad[(b * seq_q + qi) * seq_k + ki] = sum;
                }
                float max_val = -INFINITY;
                for (int ki = 0; ki < seq_k; ki++) {
                    float v = ad[(b * seq_q + qi) * seq_k + ki];
                    if (v > max_val) max_val = v;
                }
                float sum_exp = 0.0f;
                for (int ki = 0; ki < seq_k; ki++) {
                    float v = ad[(b * seq_q + qi) * seq_k + ki];
                    float e = expf(v - max_val);
                    ad[(b * seq_q + qi) * seq_k + ki] = e;
                    sum_exp += e;
                }
                float inv_sum = 1.0f / (sum_exp + 1e-10f);
                for (int ki = 0; ki < seq_k; ki++) {
                    ad[(b * seq_q + qi) * seq_k + ki] *= inv_sum;
                }
                for (int d = 0; d < head_dim; d++) {
                    float val = 0.0f;
                    for (int ki = 0; ki < seq_k; ki++) {
                        float attn_w = ad[(b * seq_q + qi) * seq_k + ki];
                        float vv = vd[((b * num_heads + h) * seq_k + ki) * head_dim + d];
                        val += attn_w * vv;
                    }
                    od[((b * num_heads + h) * seq_q + qi) * head_dim + d] = val;
                }
            }
        }
    }
    SNEPPX_tensor_destroy(attn_scores);
    return output;
}

/**
 * @brief Perform Mha Get Output Dim.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_mha_get_output_dim(const SNEPPXMultiHeadAttention* mha) {
    return mha ? mha->num_heads * mha->head_dim : 0;
}

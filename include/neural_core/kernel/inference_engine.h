#ifndef SNEPPX_INFERENCE_ENGINE_H
#define SNEPPX_INFERENCE_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multi_head_attention_module.h"
#include "subword_tokenization_pipeline.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>

/*
 * SNEPPX - Inference Engine
 *
 * WHAT
 *   Inference Engine.
 *
 * CONCEPT
 *   Provides the Inference Engine.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct SNEPPXInferenceEngine SNEPPXInferenceEngine;

/**
 * @brief Create Inference Engine.
 *
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXInferenceEngine* SNEPPX_inference_engine_create(unsigned int seed);
/**
 * @brief Destroy Inference Engine.
 *
 * @param engine [out] Engine value.
 */
void SNEPPX_inference_engine_destroy(SNEPPXInferenceEngine* engine);
/**
 * @brief Run Inference Engine.
 *
 * @param engine [out] Engine value.
 * @param input [in] Input value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_inference_engine_run(SNEPPXInferenceEngine* engine, const SNEPPXTensor* input);
/**
 * @brief Reset Inference Engine.
 *
 * @param engine [out] Engine value.
 */
void SNEPPX_inference_engine_reset(SNEPPXInferenceEngine* engine);

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    int max_new_tokens;
    int eos_token_id;
} SNEPPXGenerationConfig;

/**
 * @brief Perform Generation Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXGenerationConfig SNEPPX_generation_config_default(void);

/**
 * @brief Perform Sample From Logits.
 *
 * @param logits [in] Logits value.
 * @param vocab_size [in] Vocab Size value.
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sample_from_logits(const float* logits, size_t vocab_size, SNEPPXGenerationConfig* cfg);

/**
 * @brief Perform Argmax.
 *
 * @param logits [in] Logits value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_argmax(const float* logits, size_t n);

/**
 * @brief Perform Generate.
 *
 * @param embed_weight [out] Embed Weight value.
 * @param unembed_weight [out] Unembed Weight value.
 * @param attn [out] Attn value.
 * @param tok [out] Tok value.
 * @param prompt [in] Prompt value.
 * @param cfg [out] Cfg value.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_generate(SNEPPXTensor* embed_weight, SNEPPXTensor* unembed_weight,
                    SNEPPXAttentionWeights* attn, SNEPPXTokenizer* tok,
                    const char* prompt, SNEPPXGenerationConfig* cfg);

/**
 * @brief Perform Generate Tokens.
 *
 * @param embed_weight [out] Embed Weight value.
 * @param unembed_weight [out] Unembed Weight value.
 * @param attn [out] Attn value.
 * @param input_ids [in] Input Ids value.
 * @param input_len [in] Input Len value.
 * @param output_ids [out] Output Ids value.
 * @param max_output_len [in] Max Output Len value.
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_generate_tokens(SNEPPXTensor* embed_weight, SNEPPXTensor* unembed_weight,
                         SNEPPXAttentionWeights* attn,
                         const int* input_ids, size_t input_len,
                         int* output_ids, size_t max_output_len,
                         SNEPPXGenerationConfig* cfg);


#ifdef __cplusplus
}
#endif
#endif

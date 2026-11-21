#ifndef SNEPPX_TRANSFORMER_MODEL_H
#define SNEPPX_TRANSFORMER_MODEL_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Transformer Model
 *
 * WHAT
 *   Transformer Model.
 *
 * CONCEPT
 *   Transformer configuration and forward API for the SNEPPX-Algo system.
 *
 * ROLE
 *   Declares the transformer API for standard transformer operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTransformer SNEPPXTransformer;

/**
 * @brief Create Transformer.
 *
 * @param vocab_size [in] Vocab Size value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param num_heads [in] Num Heads value.
 * @param num_layers [in] Num Layers value.
 * @param ffn_dim [in] Ffn Dim value.
 * @param dropout [in] Dropout value.
 * @param use_rope [in] Use Rope value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTransformer* SNEPPX_transformer_create(size_t vocab_size, size_t hidden_dim,
    size_t num_heads, size_t num_layers, size_t ffn_dim, float dropout, int use_rope);
/**
 * @brief Destroy Transformer.
 *
 * @param model [out] Model value.
 */
void SNEPPX_transformer_destroy(void* model);
/**
 * @brief Run the forward pass for Transformer.
 *
 * @param model [out] Model value.
 * @param input_ids [in] Input Ids value.
 * @param seq_len [in] Seq Len value.
 * @param logits [out] Logits value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transformer_forward(void* model, const int* input_ids, size_t seq_len, float* logits);
/**
 * @brief Perform Transformer Generate.
 *
 * @param model [out] Model value.
 * @param prompt [in] Prompt value.
 * @param prompt_len [in] Prompt Len value.
 * @param output [out] Output value.
 * @param max_len [in] Max Len value.
 * @param temperature [in] Temperature value.
 * @param top_k [in] Top K value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transformer_generate(void* model, const int* prompt, size_t prompt_len,
    int* output, size_t max_len, int temperature, int top_k);

#ifdef __cplusplus
}
#endif
#endif

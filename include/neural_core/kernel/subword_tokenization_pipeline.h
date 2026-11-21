#ifndef SNEPPX_TOKENIZER_H
#define SNEPPX_TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include <stddef.h>

/*
 * SNEPPX - Subword Tokenization Pipeline
 *
 * WHAT
 *   Subword Tokenization Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TOK_BPE,
} SNEPPXTokenizerType;

typedef struct {
    int pad_id;
    int bos_id;
    int eos_id;
    int unk_id;
} SNEPPXSpecialTokens;

typedef struct SNEPPXTokenizer SNEPPXTokenizer;

/**
 * @brief Create Tokenizer.
 *
 * @param vocab_size [in] Vocab Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTokenizer*  SNEPPX_tokenizer_create(int vocab_size);
/**
 * @brief Destroy Tokenizer.
 *
 * @param tok [out] Tok value.
 */
void            SNEPPX_tokenizer_destroy(SNEPPXTokenizer* tok);

/**
 * @brief Perform Tokenizer Vocab Size.
 *
 * @param tok [in] Tok value.
 *
 * @return 0 on success, -1 on error.
 */
int             SNEPPX_tokenizer_vocab_size(const SNEPPXTokenizer* tok);
/**
 * @brief Perform Tokenizer Special.
 *
 * @param tok [in] Tok value.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSpecialTokens SNEPPX_tokenizer_special(const SNEPPXTokenizer* tok);
/**
 * @brief Perform Tokenizer Set Special.
 *
 * @param tok [out] Tok value.
 * @param sp [in] Sp value.
 */
void            SNEPPX_tokenizer_set_special(SNEPPXTokenizer* tok, SNEPPXSpecialTokens sp);

/**
 * @brief Perform Tokenizer Add Token.
 *
 * @param tok [out] Tok value.
 * @param token [in] Token value.
 * @param id [in] Id value.
 *
 * @return 0 on success, -1 on error.
 */
int             SNEPPX_tokenizer_add_token(SNEPPXTokenizer* tok, const char* token, int id);

/**
 * @brief Perform Tokenizer Encode.
 *
 * @param tok [in] Tok value.
 * @param text [in] Text value.
 * @param out_ids [out] Out Ids value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int             SNEPPX_tokenizer_encode(const SNEPPXTokenizer* tok, const char* text, int* out_ids, size_t max_len);
/**
 * @brief Perform Tokenizer Decode.
 *
 * @param tok [in] Tok value.
 * @param ids [in] Ids value.
 * @param len [in] Len value.
 *
 * @return Pointer on success, NULL on error.
 */
char*           SNEPPX_tokenizer_decode(const SNEPPXTokenizer* tok, const int* ids, size_t len);

/**
 * @brief Perform Tokenizer Train Bpe.
 *
 * @param texts [in] Texts value.
 * @param num_texts [in] Num Texts value.
 * @param vocab_size [in] Vocab Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTokenizer*  SNEPPX_tokenizer_train_bpe(const char** texts, size_t num_texts, size_t vocab_size);

/**
 * @brief Save Tokenizer.
 *
 * @param tok [in] Tok value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int             SNEPPX_tokenizer_save(const SNEPPXTokenizer* tok, const char* path);
/**
 * @brief Load Tokenizer.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTokenizer*  SNEPPX_tokenizer_load(const char* path);

/**
 * @brief Perform Tokenizer Ids To Tensor.
 *
 * @param tok [in] Tok value.
 * @param ids [in] Ids value.
 * @param len [in] Len value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor*     SNEPPX_tokenizer_ids_to_tensor(const SNEPPXTokenizer* tok, const int* ids, size_t len);
/**
 * @brief Perform Tokenizer Tensor To Ids.
 *
 * @param t [in] T value.
 * @param out_len [out] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int*            SNEPPX_tokenizer_tensor_to_ids(const SNEPPXTensor* t, size_t* out_len);


#ifdef __cplusplus
}
#endif
#endif

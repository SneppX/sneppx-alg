#include "test_gtest.h"
#include "data_pipeline.h"
#include "inference_engine.h"
#include "polymorphic_memory_allocator.h"
#include "multi_head_attention_module.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Inference
 *
 * WHAT
 *   Test Inference.
 *
 * CONCEPT
 *   Provides the Test Inference.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ===== Data Pipeline Tests ===== */

static void test_dataset_create(void) {
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(256);
    SX_ASSERT_NOT_NULL(tok, "tokenizer created");
    SNEPPX_tokenizer_add_token(tok, "hello", 0);
    SNEPPX_tokenizer_add_token(tok, "world", 1);
    SNEPPX_tokenizer_add_token(tok, "test", 2);
    SNEPPX_tokenizer_add_token(tok, "data", 3);

    const char* texts[] = {"hello world", "test data", "hello test"};
    SNEPPXTokenizer* bpe = SNEPPX_tokenizer_train_bpe(texts, 3, 270);
    SX_ASSERT_NOT_NULL(bpe, "bpe trained");

    const char* tmpfile = "test_corpus.txt";
    FILE* f = fopen(tmpfile, "w");
    SX_ASSERT_NOT_NULL(f, "file opened");
    fprintf(f, "hello world\ntest data\nhello test\n");
    fclose(f);

    SNEPPXTextDataset* ds = SNEPPX_text_dataset_create(tmpfile, bpe, 8, 1);
    SX_ASSERT_NOT_NULL(ds, "dataset created");
    SX_ASSERT_EQ(SNEPPX_text_dataset_size(ds), 3, "3 samples");
    SNEPPX_text_dataset_destroy(ds);
    remove(tmpfile);
    SNEPPX_tokenizer_destroy(bpe);
    SNEPPX_tokenizer_destroy(tok);
}

static void test_dataset_batch(void) {
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(256);
    SX_ASSERT_NOT_NULL(tok, "tokenizer created");
    const char* texts[] = {"a b c d e f g h i j k l m n o p q r s t u v w x y z"};
    SNEPPXTokenizer* bpe = SNEPPX_tokenizer_train_bpe(texts, 1, 270);
    SX_ASSERT_NOT_NULL(bpe, "bpe trained");

    const char* tmpfile = "test_batch.txt";
    FILE* f = fopen(tmpfile, "w");
    fprintf(f, "hello world this is a test of the data pipeline system\n");
    fclose(f);

    SNEPPXTextDataset* ds = SNEPPX_text_dataset_create(tmpfile, bpe, 4, 0);
    SX_ASSERT_NOT_NULL(ds, "dataset created");

    SNEPPXTensor* in = NULL;
    SNEPPXTensor* tgt = NULL;
    int ret = SNEPPX_text_dataset_get_batch(ds, 0, 2, &in, &tgt);
    SX_ASSERT_EQ(ret, 0, "batch retrieved");
    SX_ASSERT_NOT_NULL(in, "input tensor");
    SX_ASSERT_NOT_NULL(tgt, "target tensor");
    SX_ASSERT_EQ(in->ndim, 2, "input is 2D");
    SX_ASSERT_EQ(in->shape[0], 2, "batch size 2");
    SX_ASSERT_EQ(in->shape[1], 4, "seq len 4");

    SNEPPX_tensor_destroy(in);
    SNEPPX_tensor_destroy(tgt);
    SNEPPX_text_dataset_destroy(ds);
    remove(tmpfile);
    SNEPPX_tokenizer_destroy(bpe);
    SNEPPX_tokenizer_destroy(tok);
}

/* ===== Inference Engine Tests ===== */

static void test_argmax(void) {
    float logits[] = {-2.0f, 5.0f, 1.0f, 3.0f};
    int result = SNEPPX_argmax(logits, 4);
    SX_ASSERT_EQ(result, 1, "argmax at index 1");
}

static void test_sample_logits(void) {
    float logits[] = {0.0f, 0.0f, 100.0f, 0.0f};
    SNEPPXGenerationConfig cfg = SNEPPX_generation_config_default();
    cfg.temperature = 1.0f;
    cfg.top_k = 0;
    cfg.top_p = 0.0f;
    int result = SNEPPX_sample_from_logits(logits, 4, &cfg);
    SX_ASSERT_EQ(result, 2, "sample picks highest prob");
}

static void test_gen_config_default(void) {
    SNEPPXGenerationConfig cfg = SNEPPX_generation_config_default();
    SX_ASSERT_EQ(cfg.max_new_tokens, 256, "max tokens default");
    SX_ASSERT(cfg.temperature > 0, "temperature positive");
    SX_ASSERT(cfg.top_p > 0, "top-p positive");
}

static void test_generate_tokens_small(void) {
    SNEPPXAttentionConfig attn_cfg = SNEPPX_attn_config_default();
    attn_cfg.d_model = 16;
    attn_cfg.num_heads = 2;
    attn_cfg.head_dim = 8;
    attn_cfg.use_rope = 1;
    attn_cfg.use_causal_mask = 1;

    SNEPPXAttentionWeights* attn = SNEPPX_attn_weights_create(attn_cfg, 42);
    SX_ASSERT_NOT_NULL(attn, "attn weights created");

    size_t vocab = 16, d = 16;
    size_t emb_shape[] = {vocab, d};
    SNEPPXTensor* embed = SNEPPX_tensor_empty(emb_shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(embed, "embed created");
    for (size_t i = 0; i < vocab * d; i++) ((float*)embed->data)[i] = ((float)(i % 7) - 3.0f) * 0.1f;

    size_t unemb_shape[] = {d, vocab};
    SNEPPXTensor* unembed = SNEPPX_tensor_create(unemb_shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(unembed, "unembed created");
    for (size_t i = 0; i < d * vocab; i++) ((float*)unembed->data)[i] = ((float)(i * 3 % 11) - 5.0f) * 0.1f;

    int input[] = {0, 1, 2, 3};
    int output[32];
    SNEPPXGenerationConfig cfg = SNEPPX_generation_config_default();
    cfg.max_new_tokens = 5;
    cfg.top_k = 1;

    int n = SNEPPX_generate_tokens(embed, unembed, attn, input, 4, output, 32, &cfg);
    SX_ASSERT(n > 0, "tokens generated");
    SX_ASSERT_EQ(output[0], 0, "first output token matches input");

    SNEPPX_tensor_destroy(embed);
    SNEPPX_tensor_destroy(unembed);
    SNEPPX_attn_weights_destroy(attn);
}

/* ===== Main ===== */


TEST(test_inference, dataset_create) { test_dataset_create(); }
TEST(test_inference, dataset_batch) { test_dataset_batch(); }
TEST(test_inference, argmax) { test_argmax(); }
TEST(test_inference, sample_logits) { test_sample_logits(); }
TEST(test_inference, gen_config_default) { test_gen_config_default(); }
TEST(test_inference, generate_tokens_small) { test_generate_tokens_small(); }

#include "subword_tokenization_pipeline.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Tokenizer
 *
 * WHAT
 *   Test Tokenizer.
 *
 * CONCEPT
 *   Provides tokenization.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_create_destroy(void) {
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(1000);
    SX_ASSERT(tok != NULL, "create returned non-NULL");
    SX_ASSERT_EQ(SNEPPX_tokenizer_vocab_size(tok), 256, "starts with 256 bytes");
    SNEPPX_tokenizer_destroy(tok);
}

static void test_add_token(void) {
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(500);
    int r = SNEPPX_tokenizer_add_token(tok, "hello", 256);
    SX_ASSERT_EQ(r, 0, "add hello token");
    SX_ASSERT_EQ(SNEPPX_tokenizer_vocab_size(tok), 257, "vocab now 257");
    SNEPPX_tokenizer_destroy(tok);
}

static void test_encode_decode_basic(void) {
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(1000);
    const char* text = "hello";
    int ids[256];
    int n = SNEPPX_tokenizer_encode(tok, text, ids, 256);
    SX_ASSERT(n > 0, "encoded non-empty");
    char* decoded = SNEPPX_tokenizer_decode(tok, ids, (size_t)n);
    SX_ASSERT(decoded != NULL, "decoded non-NULL");
    SX_ASSERT_STR_EQ(decoded, text, "roundtrip match");
    SNEPPX_free(decoded, strlen(decoded) + 1);
    SNEPPX_tokenizer_destroy(tok);
}

static void test_train_bpe(void) {
    const char* texts[] = {"low low low low low", "lowest lowest", "newer newer"};
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_train_bpe(texts, 3, 270);
    SX_ASSERT(tok != NULL, "trained non-NULL");
    SX_ASSERT(SNEPPX_tokenizer_vocab_size(tok) > 256, "vocab grew");
    int ids[256];
    int n = SNEPPX_tokenizer_encode(tok, "low", ids, 256);
    SX_ASSERT(n < 3, "low encoded with fewer tokens than letters (bpe works)");
    char* decoded = SNEPPX_tokenizer_decode(tok, ids, (size_t)n);
    SX_ASSERT(decoded != NULL, "decoded non-NULL");
    SX_ASSERT_STR_EQ(decoded, "low", "roundtrip low");
    SNEPPX_free(decoded, strlen(decoded) + 1);
    SNEPPX_tokenizer_destroy(tok);
}

static void test_save_load(void) {
    SNEPPXTokenizer* tok1 = SNEPPX_tokenizer_create(500);
    SNEPPX_tokenizer_add_token(tok1, "test", 256);
    SNEPPX_tokenizer_add_token(tok1, "save", 257);
    int r = SNEPPX_tokenizer_save(tok1, "test_tokenizer.bin");
    SX_ASSERT_EQ(r, 0, "save succeeded");
    SNEPPXTokenizer* tok2 = SNEPPX_tokenizer_load("test_tokenizer.bin");
    SX_ASSERT(tok2 != NULL, "load succeeded");
    SX_ASSERT_EQ(SNEPPX_tokenizer_vocab_size(tok2), 258, "vocab size matches");
    const char* text = "test";
    int ids1[256], ids2[256];
    int n1 = SNEPPX_tokenizer_encode(tok1, text, ids1, 256);
    int n2 = SNEPPX_tokenizer_encode(tok2, text, ids2, 256);
    SX_ASSERT_EQ(n1, n2, "encode lengths match");
    SNEPPX_tokenizer_destroy(tok1);
    SNEPPX_tokenizer_destroy(tok2);
    remove("test_tokenizer.bin");
}

static void test_special_tokens(void) {
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(500);
    SNEPPXSpecialTokens sp = {0, 1, 2, 3};
    SNEPPX_tokenizer_set_special(tok, sp);
    SNEPPXSpecialTokens got = SNEPPX_tokenizer_special(tok);
    SX_ASSERT_EQ(got.pad_id, 0, "pad=0");
    SX_ASSERT_EQ(got.bos_id, 1, "bos=1");
    SX_ASSERT_EQ(got.eos_id, 2, "eos=2");
    SX_ASSERT_EQ(got.unk_id, 3, "unk=3");
    SNEPPX_tokenizer_destroy(tok);
}


TEST(test_tokenizer, create_destroy) { test_create_destroy(); }
TEST(test_tokenizer, add_token) { test_add_token(); }
TEST(test_tokenizer, encode_decode_basic) { test_encode_decode_basic(); }
TEST(test_tokenizer, train_bpe) { test_train_bpe(); }
TEST(test_tokenizer, save_load) { test_save_load(); }
TEST(test_tokenizer, special_tokens) { test_special_tokens(); }

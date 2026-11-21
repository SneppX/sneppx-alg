#include "multi_head_attention_module.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Rope
 *
 * WHAT
 *   Test Rope.
 *
 * CONCEPT
 *   Provides the Test Rope.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_rope_precompute(void) {
    SNEPPXTensor* cos = SNEPPX_rope_precompute(8, 4, 10000.0f);
    SX_ASSERT(cos != NULL, "precomputed cos table");
    SX_ASSERT(cos->shape[0] == 8, "seq_len dim");
    SX_ASSERT(cos->shape[1] == 4, "head_dim dim");
    SNEPPX_tensor_destroy(cos);
}

static void test_rope_apply_changes_values(void) {
    size_t shape[] = {2, 4};
    SNEPPXTensor* q = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* k = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* qd = (float*)q->data;
    float* kd = (float*)k->data;
    for (size_t i = 0; i < 8; i++) { qd[i] = 1.0f; kd[i] = 0.5f; }

    SNEPPXTensor* cos = SNEPPX_rope_precompute(2, 4, 10000.0f);
    SNEPPXTensor* sin = NULL;
    SNEPPX_rope_apply(q, k, cos, sin, 0);

    float* q_out = (float*)q->data;
    int changed = 0;
    for (size_t i = 0; i < 8; i++) if (q_out[i] != 1.0f) { changed = 1; break; }
    SX_ASSERT(changed, "rope rotates q values");

    SNEPPX_tensor_destroy(sin);
    SNEPPX_tensor_destroy(cos);
    SNEPPX_tensor_destroy(k);
    SNEPPX_tensor_destroy(q);
}

static void test_attention_self_attention(void) {
    SNEPPXAttentionConfig cfg;
    cfg.num_heads = 2;
    cfg.head_dim = 4;
    cfg.d_model = cfg.num_heads * cfg.head_dim;
    cfg.dropout = 0.0f;
    cfg.use_causal_mask = 0;
    cfg.use_rope = 0;
    cfg.rope_base = 10000.0f;
    SNEPPXAttentionWeights* attn = SNEPPX_attn_weights_create(cfg, 42);
    SX_ASSERT(attn != NULL, "attention layer created");

    size_t seq_len = 3;
    size_t shape[] = {seq_len, cfg.num_heads * cfg.head_dim};
    SNEPPXTensor* x = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = SNEPPX_attn_forward(attn, x, NULL, NULL);
    SX_ASSERT(output != NULL, "attention forward output");
    SX_ASSERT(output->shape[0] == seq_len, "output seq_len");
    SX_ASSERT(output->shape[1] == shape[1], "output feat dim");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(x);
    SNEPPX_attn_weights_destroy(attn);
}

static void test_attention_causal_mask(void) {
    SNEPPXAttentionConfig cfg;
    cfg.num_heads = 1;
    cfg.head_dim = 2;
    cfg.d_model = cfg.num_heads * cfg.head_dim;
    cfg.dropout = 0.0f;
    cfg.use_causal_mask = 1;
    cfg.use_rope = 0;
    cfg.rope_base = 10000.0f;
    SNEPPXAttentionWeights* attn = SNEPPX_attn_weights_create(cfg, 42);

    size_t seq_len = 3;
    size_t shape[] = {seq_len, cfg.num_heads * cfg.head_dim};
    SNEPPXTensor* x = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = SNEPPX_attn_forward(attn, x, NULL, NULL);
    SX_ASSERT(output != NULL, "causal masked forward");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(x);
    SNEPPX_attn_weights_destroy(attn);
}


TEST(test_rope, rope_precompute) { test_rope_precompute(); }
TEST(test_rope, rope_apply_changes_values) { test_rope_apply_changes_values(); }
TEST(test_rope, attention_self_attention) { test_attention_self_attention(); }
TEST(test_rope, attention_causal_mask) { test_attention_causal_mask(); }

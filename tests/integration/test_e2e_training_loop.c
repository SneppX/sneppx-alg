#include "system_architecture_definitions.h"
#include "test_gtest.h"
#include "differentiable_training_pipeline.h"
#include "hierarchical_state_space.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test E2e Training Loop
 *
 * WHAT
 *   Test E2e Training Loop.
 *
 * CONCEPT
 *   Provides the Test E2e Training Loop.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_attn_training_step(void) {
    SNEPPXArchConfig arch_cfg = SNEPPX_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.enable_hss = 0; arch_cfg.enable_ser = 0;
    arch_cfg.enable_attention = 1;
    arch_cfg.attention_config.d_model = 8;
    arch_cfg.attention_config.num_heads = 2;
    arch_cfg.attention_config.head_dim = 4;
    arch_cfg.attention_config.use_causal_mask = 0;
    arch_cfg.vocab_size = 16;

    SNEPPXModel* model = SNEPPX_model_create(&arch_cfg);
    SX_ASSERT(model != NULL, "attn model");

    SNEPPXTrainConfig train_cfg = SNEPPX_train_config_default();
    train_cfg.learning_rate = 0.001f;
    SNEPPXTrainer* trainer = SNEPPX_trainer_create(model, &train_cfg);
    SX_ASSERT(trainer != NULL, "trainer");

    size_t B = 1, S = 1;
    size_t in_shape[] = {B, S};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(in_shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input");
    ((float*)input->data)[0] = 0.0f;

    size_t tgt_shape[] = {B * S, arch_cfg.vocab_size};
    SNEPPXTensor* target = SNEPPX_tensor_zeros(tgt_shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(target != NULL, "target");
    ((float*)target->data)[0] = 1.0f;

    float val0 = SNEPPX_trainer_evaluate(trainer, input, target);
    SX_ASSERT(isfinite(val0) && val0 >= 0.0f, "evaluate returns valid loss");

    float loss1 = SNEPPX_trainer_train_step(trainer, input, target);
    SX_ASSERT(isfinite(loss1) && loss1 >= 0.0f, "first train step valid");

    float loss2 = SNEPPX_trainer_train_step(trainer, input, target);
    SX_ASSERT(isfinite(loss2) && loss2 >= 0.0f, "second train step valid");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(target);
    SNEPPX_trainer_destroy(trainer);
    SNEPPX_model_destroy(model);
}

static void test_model_forward_hss(void) {
    SNEPPXArchConfig cfg = SNEPPX_arch_config_default();
    cfg.input_dim = 8; cfg.output_dim = 8;
    cfg.enable_attention = 0;
    cfg.enable_hss = 1;
    cfg.hss_config.state_dim = 4;
    cfg.hss_config.input_dim = 8;
    cfg.hss_config.output_dim = 8;
    cfg.hss_config.num_layers = 1;
    cfg.hss_config.seq_len = 4;
    cfg.hss_config.use_hierarchical = 0;

    SNEPPXModel* model = SNEPPX_model_create(&cfg);
    SX_ASSERT(model != NULL, "hss model");

    size_t sh_in[] = {1, 4, 8};
    SNEPPXTensor* input = SNEPPX_tensor_randn(sh_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input");

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_model_forward(model, input, &output);
    SX_ASSERT(ret == 0, "forward");
    SX_ASSERT(output != NULL, "output");
    SX_ASSERT(output->size > 0, "output non-empty");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_model_destroy(model);
}

static void test_model_get_params(void) {
    SNEPPXArchConfig cfg = SNEPPX_arch_config_default();
    cfg.input_dim = 8; cfg.output_dim = 8;
    cfg.enable_attention = 0;
    cfg.enable_hss = 1;
    cfg.hss_config.state_dim = 4;
    cfg.hss_config.input_dim = 8;
    cfg.hss_config.output_dim = 8;
    cfg.hss_config.num_layers = 1;
    cfg.hss_config.seq_len = 1;

    SNEPPXModel* model = SNEPPX_model_create(&cfg);
    SX_ASSERT(model != NULL, "model");

    size_t nw = SNEPPX_model_get_params(model, NULL, 0);
    SX_ASSERT(nw > 0, "model has params");

    SNEPPXTensor** params = (SNEPPXTensor**)malloc(nw * sizeof(SNEPPXTensor*));
    SNEPPX_model_get_params(model, params, nw);
    for (size_t i = 0; i < nw; i++) {
        SX_ASSERT(params[i] != NULL, "param non-null");
        SX_ASSERT(params[i]->size > 0, "param non-empty");
    }
    free(params);
    SNEPPX_model_destroy(model);
}


TEST(test_e2e_training_loop, attn_training_step) { test_attn_training_step(); }
TEST(test_e2e_training_loop, model_forward_hss) { test_model_forward_hss(); }
TEST(test_e2e_training_loop, model_get_params) { test_model_get_params(); }

#include "system_architecture_definitions.h"
#include "test_gtest.h"
#include "differentiable_training_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * SNEPPX - Test Multi Module Stack Train
 *
 * WHAT
 *   Test Multi Module Stack Train.
 *
 * CONCEPT
 *   Provides the Test Multi Module Stack Train.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




static void test_attn_hss_stack_train(void) {
    SNEPPXArchConfig arch_cfg = SNEPPX_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.enable_hss = 1;
    arch_cfg.enable_ser = 0;
    arch_cfg.enable_attention = 1;
    arch_cfg.hss_config.input_dim = 8; arch_cfg.hss_config.output_dim = 8;
    arch_cfg.hss_config.state_dim = 4;
    arch_cfg.attention_config.d_model = 8;
    arch_cfg.attention_config.num_heads = 2;
    arch_cfg.attention_config.head_dim = 4;
    arch_cfg.attention_config.use_causal_mask = 0;
    arch_cfg.attention_config.use_rope = 0;
    arch_cfg.vocab_size = 16;

    SNEPPXModel* model = SNEPPX_model_create(&arch_cfg);
    SX_ASSERT(model != NULL, "model created");

    SNEPPXTrainConfig train_cfg = SNEPPX_train_config_default();
    train_cfg.learning_rate = 0.001f;
    SNEPPXTrainer* trainer = SNEPPX_trainer_create(model, &train_cfg);

    /* Use S=1 for HSS compatibility (HSS outputs per-token predictions) */
    size_t B = 1, S = 1;
    size_t in_shape[] = {B, S};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(in_shape, 2, SNEPPX_FLOAT32);
    float* id = (float*)input->data;
    for (size_t i = 0; i < B * S; i++) id[i] = (float)(i % 16);

    size_t tgt_shape[] = {B * S, arch_cfg.vocab_size};
    SNEPPXTensor* target = SNEPPX_tensor_zeros(tgt_shape, 2, SNEPPX_FLOAT32);
    if (target) {
        float* td = (float*)target->data;
        for (size_t i = 0; i < B * S; i++) td[i * arch_cfg.vocab_size + (int)id[i]] = 1.0f;
    }

    float val0 = SNEPPX_trainer_evaluate(trainer, input, target);
    SX_ASSERT(isfinite(val0) && val0 >= 0.0f, "eval ok");

    int steps = 50;
    float last_loss = -1.0f;
    for (int s = 0; s < steps; s++) {
        float loss = SNEPPX_trainer_train_step(trainer, input, target);
        if (isfinite(loss) && loss >= 0.0f) last_loss = loss;
    }
    SX_ASSERT(last_loss >= 0.0f, "last loss valid");
    printf("  init=%.6f final=%.6f ratio=%.4f\n", (double)val0, (double)last_loss,
           (double)(last_loss / (val0 + 1e-10f)));
    SX_ASSERT(last_loss < val0 * 0.9f, "loss decreased >10%%");

    SNEPPX_tensor_destroy(input); SNEPPX_tensor_destroy(target);
    SNEPPX_trainer_destroy(trainer);
    SNEPPX_model_destroy(model);
}

static void test_attn_hss_ser_stack_train(void) {
    SNEPPXArchConfig arch_cfg = SNEPPX_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.enable_hss = 1;
    arch_cfg.enable_ser = 1;
    arch_cfg.enable_attention = 1;
    arch_cfg.hss_config.input_dim = 8; arch_cfg.hss_config.output_dim = 8;
    arch_cfg.hss_config.state_dim = 4;
    arch_cfg.ser_config.input_dim = 8; arch_cfg.ser_config.output_dim = 8;
    arch_cfg.ser_config.expert_dim = 4; arch_cfg.ser_config.num_experts = 2;
    arch_cfg.ser_config.num_active = 1;
    arch_cfg.attention_config.d_model = 8;
    arch_cfg.attention_config.num_heads = 2;
    arch_cfg.attention_config.head_dim = 4;
    arch_cfg.attention_config.use_causal_mask = 0;
    arch_cfg.attention_config.use_rope = 0;
    arch_cfg.vocab_size = 16;

    SNEPPXModel* model = SNEPPX_model_create(&arch_cfg);
    SX_ASSERT(model != NULL, "model created");

    SNEPPXTrainConfig train_cfg = SNEPPX_train_config_default();
    train_cfg.learning_rate = 0.001f;
    SNEPPXTrainer* trainer = SNEPPX_trainer_create(model, &train_cfg);

    size_t B = 1, S = 1;
    size_t in_shape[] = {B, S};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(in_shape, 2, SNEPPX_FLOAT32);
    float* id = (float*)input->data;
    for (size_t i = 0; i < B * S; i++) id[i] = (float)(i % 16);

    size_t tgt_shape[] = {B * S, arch_cfg.vocab_size};
    SNEPPXTensor* target = SNEPPX_tensor_zeros(tgt_shape, 2, SNEPPX_FLOAT32);
    if (target) {
        float* td = (float*)target->data;
        for (size_t i = 0; i < B * S; i++) td[i * arch_cfg.vocab_size + (int)id[i]] = 1.0f;
    }

    float val0 = SNEPPX_trainer_evaluate(trainer, input, target);
    SX_ASSERT(isfinite(val0) && val0 >= 0.0f, "eval ok");

    int steps = 50;
    float last_loss = -1.0f;
    for (int s = 0; s < steps; s++) {
        float loss = SNEPPX_trainer_train_step(trainer, input, target);
        if (isfinite(loss) && loss >= 0.0f) last_loss = loss;
    }
    SX_ASSERT(last_loss >= 0.0f, "last loss valid");
    printf("  init=%.6f final=%.6f ratio=%.4f\n", (double)val0, (double)last_loss,
           (double)(last_loss / (val0 + 1e-10f)));
    SX_ASSERT(last_loss < val0 * 0.9f, "loss decreased >10%%");

    SNEPPX_tensor_destroy(input); SNEPPX_tensor_destroy(target);
    SNEPPX_trainer_destroy(trainer);
    SNEPPX_model_destroy(model);
}


TEST(test_multi_module_stack_train, test_attn_hss_stack_train) { test_attn_hss_stack_train(); }
TEST(test_multi_module_stack_train, test_attn_hss_ser_stack_train) { test_attn_hss_ser_stack_train(); }

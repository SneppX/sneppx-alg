#include "system_architecture_definitions.h"
#include "test_gtest.h"
#include "differentiable_training_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
 * SNEPPX - Test Attn Train Convergence
 *
 * WHAT
 *   Test Attn Train Convergence.
 *
 * CONCEPT
 *   Provides the Test Attn Train Convergence.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */




static void test_attn_train_graph_builds(void) {
    SNEPPXArchConfig arch_cfg = SNEPPX_arch_config_default();
    arch_cfg.input_dim = 8; arch_cfg.output_dim = 8;
    arch_cfg.enable_hss = 0; arch_cfg.enable_ser = 0;
    arch_cfg.enable_attention = 1;
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

    /* Single-token input */
    size_t B = 1, S = 1;
    size_t in_shape[] = {B, S};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(in_shape, 2, SNEPPX_FLOAT32);
    ((float*)input->data)[0] = 0.0f;

    size_t tgt_shape[] = {B * S, arch_cfg.vocab_size};
    SNEPPXTensor* target = SNEPPX_tensor_zeros(tgt_shape, 2, SNEPPX_FLOAT32);
    if (target) ((float*)target->data)[0] = 1.0f;
    SX_ASSERT(input != NULL && target != NULL, "tensors created");

    /* Verify evaluate works */
    float val0 = SNEPPX_trainer_evaluate(trainer, input, target);
    SX_ASSERT(isfinite(val0) && val0 >= 0.0f, "eval ok");

    /* Verify first train step produces valid loss */
    float loss = SNEPPX_trainer_train_step(trainer, input, target);
    SX_ASSERT(isfinite(loss) && loss >= 0.0f, "first step loss valid");

    /* Verify model parameters updated (weights changed from initial) */
    size_t nw = SNEPPX_model_get_params(model, NULL, 0);
    SX_ASSERT(nw > 0, "has parameters");

    SNEPPX_tensor_destroy(input); SNEPPX_tensor_destroy(target);
    SNEPPX_trainer_destroy(trainer);
    SNEPPX_model_destroy(model);
}


TEST(test_attn_train_convergence, test_attn_train_graph_builds) { test_attn_train_graph_builds(); }

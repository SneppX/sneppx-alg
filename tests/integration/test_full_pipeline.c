#include "system_architecture_definitions.h"
#include "test_gtest.h"
#include "differentiable_training_pipeline.h"
#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Full Pipeline
 *
 * WHAT
 *   Test Full Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_arch_config_default(void) {
    SNEPPXArchConfig cfg = SNEPPX_arch_config_default();
    SX_ASSERT(cfg.input_dim == 512, "default input dim");
    SX_ASSERT(cfg.output_dim == 512, "default output dim");
}

static void test_model_create(void) {
    SNEPPXArchConfig cfg = SNEPPX_arch_config_default();
    cfg.input_dim = 16;
    cfg.output_dim = 16;
    cfg.hss_config.state_dim = 8;
    cfg.hss_config.input_dim = 16;
    cfg.hss_config.output_dim = 16;
    cfg.ser_config.num_experts = 4;
    cfg.ser_config.num_active = 2;
    cfg.ser_config.input_dim = 16;
    cfg.ser_config.expert_dim = 32;
    cfg.ser_config.output_dim = 16;

    SNEPPXModel* model = SNEPPX_model_create(&cfg);
    SX_ASSERT(model != NULL, "model created");

    size_t n = SNEPPX_model_get_params(model, NULL, 0);
    SX_ASSERT(n > 0, "model has parameters");

    SNEPPX_model_destroy(model);
}

static void test_hss_ser_arc_stack(void) {
    SNEPPXHSSConfig hss_cfg = SNEPPX_hss_config_default();
    hss_cfg.state_dim = 4;
    hss_cfg.input_dim = 8;
    hss_cfg.output_dim = 8;
    hss_cfg.num_layers = 1;
    hss_cfg.seq_len = 8;
    hss_cfg.use_hierarchical = 0;
    hss_cfg.use_parallel_scan = 1;

    SNEPPXHSSModel* hss = SNEPPX_hss_model_create(&hss_cfg, 42);
    SX_ASSERT(hss != NULL, "hss model created");

    SNEPPXSERConfig ser_cfg = SNEPPX_ser_config_default();
    ser_cfg.num_experts = 4;
    ser_cfg.num_active = 2;
    ser_cfg.input_dim = 8;
    ser_cfg.expert_dim = 16;
    ser_cfg.output_dim = 8;

    SNEPPXSERModel* ser = SNEPPX_ser_model_create(&ser_cfg, 42, 1);
    SX_ASSERT(ser != NULL, "ser model created");

    SNEPPXARCConfig arc_cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* arc = SNEPPX_arc_layer_create(&arc_cfg, 8, 8, 42);
    SX_ASSERT(arc != NULL, "arc layer created");

    size_t shape_in[] = {1, hss_cfg.seq_len, hss_cfg.input_dim};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 3, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < input->size; i++) d[i] = sinf((float)i * 0.5f);

    SNEPPXTensor* hss_out = NULL;
    int ret = SNEPPX_hss_forward(hss, input, &hss_out);
    SX_ASSERT(ret == 0, "hss forward ok");
    SX_ASSERT(hss_out != NULL, "hss output");

    SNEPPX_tensor_destroy(hss_out);
    SNEPPX_tensor_destroy(input);
    SNEPPX_arc_layer_destroy(arc);
    SNEPPX_ser_model_destroy(ser);
    SNEPPX_hss_model_destroy(hss);
}


TEST(test_full_pipeline, arch_config_default) { test_arch_config_default(); }
TEST(test_full_pipeline, model_create) { test_model_create(); }
TEST(test_full_pipeline, hss_ser_arc_stack) { test_hss_ser_arc_stack(); }

#include "adversarial_robustness_certification.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Arc Forward
 *
 * WHAT
 *   Test Arc Forward.
 *
 * CONCEPT
 *   Provides the Test Arc Forward.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_arc_config_default(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SX_ASSERT(cfg.input_guard_strength > 0, "input guard strength > 0");
    SX_ASSERT(cfg.gradient_obfuscation_method >= SNEPPX_OBF_NONE, "obf method valid");
}

static void test_arc_layer_create_destroy(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 8, 8, 42);
    SX_ASSERT(layer != NULL, "arc layer created");
    SX_ASSERT(layer->input_guard != NULL, "input guard created");
    SX_ASSERT(layer->output_verifier != NULL, "output verifier created");
    SX_ASSERT(layer->gradient_obfuscator != NULL, "gradient obfuscator created");
    SNEPPX_arc_layer_destroy(layer);
}

static void test_arc_forward_pass(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    cfg.input_guard_strength = 3.0f;
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 4, 4, 42);
    SX_ASSERT(layer != NULL, "layer created");

    size_t shape_in[] = {1, 4};
    SNEPPXTensor* input = SNEPPX_tensor_create(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4; i++) d[i] = 1.0f;

    SNEPPXTensor* output = NULL;
    float metrics[4];
    SNEPPX_arc_forward(layer, input, &output, metrics);
    SX_ASSERT(output != NULL, "forward output created");
    SX_ASSERT(metrics[0] >= 0.0f, "anomaly score >= 0");
    SX_ASSERT(metrics[1] >= 0.0f, "confidence >= 0");

    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_arc_layer_destroy(layer);
}

static void test_arc_input_guard(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 4, 4, 42);

    SNEPPXTensor* input = SNEPPX_tensor_create(SX_ARR_C(size_t, 2, 1,4), 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4; i++) d[i] = 0.5f;

    SNEPPXTensor* sanitized = NULL;
    float score = 0.0f;
    SNEPPX_arc_input_guard_forward(layer->input_guard, input, &sanitized, &score);
    SX_ASSERT(sanitized != NULL, "sanitized output");

    SNEPPX_tensor_destroy(sanitized);
    SNEPPX_tensor_destroy(input);
    SNEPPX_arc_layer_destroy(layer);
}

static void test_arc_simulate_attack(void) {
    size_t shape[] = {1, 4};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4; i++) d[i] = 1.0f;

    SNEPPXTensor* adv = NULL;
    SNEPPX_arc_simulate_attack(input, SNEPPX_ATTACK_FGSM, 0.1f, &adv);
    SX_ASSERT(adv != NULL, "adversarial sample created");

    float* ad = (float*)adv->data;
    int changed = 0;
    for (size_t i = 0; i < 4; i++) if (ad[i] != 1.0f) changed = 1;
    SX_ASSERT(changed, "adversarial perturbation applied");

    SNEPPX_tensor_destroy(adv);
    SNEPPX_tensor_destroy(input);
}


TEST(test_arc_forward, arc_config_default) { test_arc_config_default(); }
TEST(test_arc_forward, arc_layer_create_destroy) { test_arc_layer_create_destroy(); }
TEST(test_arc_forward, arc_forward_pass) { test_arc_forward_pass(); }
TEST(test_arc_forward, arc_input_guard) { test_arc_input_guard(); }
TEST(test_arc_forward, arc_simulate_attack) { test_arc_simulate_attack(); }

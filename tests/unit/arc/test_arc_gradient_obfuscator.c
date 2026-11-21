#include "adversarial_robustness_certification.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Arc Gradient Obfuscator
 *
 * WHAT
 *   Test Arc Gradient Obfuscator.
 *
 * CONCEPT
 *   Provides code obfuscation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_obf_create(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(1000, 42);
    SX_ASSERT(obf != NULL, "obf not null");
    SX_ASSERT(obf->noise_buffer->shape[0] == 1000, "noise buf size 1000");
    SX_ASSERT(obf->clamp_mask->shape[0] == 1000, "clamp mask size 1000");
    SNEPPX_gradient_obfuscator_destroy(obf);
}

static void test_obf_noise(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(100, 42);
    SX_ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(shape_g, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = 0.5f;

    SNEPPX_arc_obfuscate_gradients(obf, grad, SNEPPX_OBF_NOISE);
    int changed = 0;
    for (size_t i = 0; i < 100; i++) { if (gd[i] != 0.5f) { changed = 1; break; } }
    SX_ASSERT(changed, "values changed by noise");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_gradient_obfuscator_destroy(obf);
}

static void test_obf_clamp(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(100, 42);
    SX_ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(shape_g, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (i % 2 == 0) ? 5.0f : 0.1f;

    SNEPPX_arc_obfuscate_gradients(obf, grad, SNEPPX_OBF_CLAMP);
    float max_abs = 0.0f;
    for (size_t i = 0; i < 100; i++) {
        float a = fabsf(gd[i]);
        if (a > max_abs) max_abs = a;
    }
    SX_ASSERT(max_abs <= 1.0f + 1e-6f, "max abs <= 1.0");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_gradient_obfuscator_destroy(obf);
}

static void test_obf_mixed(void) {
    SNEPPXGradientObfuscator* obf = SNEPPX_gradient_obfuscator_create(100, 42);
    SX_ASSERT(obf != NULL, "obf not null");

    size_t shape_g[] = {100};
    SNEPPXTensor* grad = SNEPPX_tensor_zeros(shape_g, 1, SNEPPX_FLOAT32);
    float* gd = (float*)grad->data;
    for (size_t i = 0; i < 100; i++) gd[i] = (i % 2 == 0) ? 5.0f : 0.1f;

    SNEPPX_arc_obfuscate_gradients(obf, grad, SNEPPX_OBF_MIXED);
    float max_abs = 0.0f;
    for (size_t i = 0; i < 100; i++) {
        float a = fabsf(gd[i]);
        if (a > max_abs) max_abs = a;
    }
    SX_ASSERT(max_abs <= 1.0f + 1e-4f, "clamped after mixed");

    SNEPPX_tensor_destroy(grad);
    SNEPPX_gradient_obfuscator_destroy(obf);
}


TEST(test_arc_gradient_obfuscator, test_obf_create) { test_obf_create(); }
TEST(test_arc_gradient_obfuscator, test_obf_noise) { test_obf_noise(); }
TEST(test_arc_gradient_obfuscator, test_obf_clamp) { test_obf_clamp(); }
TEST(test_arc_gradient_obfuscator, test_obf_mixed) { test_obf_mixed(); }

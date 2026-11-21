#include "adversarial_robustness_certification.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Arc Attack
 *
 * WHAT
 *   Test Arc Attack.
 *
 * CONCEPT
 *   Provides the Test Arc Attack.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_fgsm(void) {
    size_t shape_in[] = {8, 16};
    SNEPPXTensor* clean = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)clean->data;
    for (size_t i = 0; i < 8 * 16; i++) d[i] = 0.1f;

    SNEPPXTensor* adv = NULL;
    SNEPPX_arc_simulate_attack(clean, SNEPPX_ATTACK_FGSM, 0.1f, &adv);
    SX_ASSERT(adv != NULL, "adv not null");
    SX_ASSERT(adv->shape[0] == 8, "batch ok");
    SX_ASSERT(adv->shape[1] == 16, "dim ok");

    float* cd = (float*)clean->data;
    float* ad = (float*)adv->data;
    int different = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < 8 * 16; i++) {
        float diff = fabsf(ad[i] - cd[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) different = 1;
    }
    SX_ASSERT(different, "adversarial != clean");
    SX_ASSERT(max_diff <= 0.1f + 1e-4f, "L_inf <= epsilon");

    SNEPPX_tensor_destroy(clean);
    SNEPPX_tensor_destroy(adv);
}

static void test_pgd(void) {
    size_t shape_in[] = {8, 16};
    SNEPPXTensor* clean = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)clean->data;
    for (size_t i = 0; i < 8 * 16; i++) d[i] = 0.1f;

    SNEPPXTensor* adv = NULL;
    SNEPPX_arc_simulate_attack(clean, SNEPPX_ATTACK_PGD, 0.1f, &adv);
    SX_ASSERT(adv != NULL, "adv not null");

    float* cd = (float*)clean->data;
    float* ad = (float*)adv->data;
    float max_diff = 0.0f;
    for (size_t i = 0; i < 8 * 16; i++) {
        float diff = fabsf(ad[i] - cd[i]);
        if (diff > max_diff) max_diff = diff;
    }
    SX_ASSERT(max_diff <= 0.1f + 1e-4f, "L_inf <= epsilon");

    SNEPPX_tensor_destroy(clean);
    SNEPPX_tensor_destroy(adv);
}

static void test_cw(void) {
    size_t shape_in[] = {8, 16};
    SNEPPXTensor* clean = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)clean->data;
    for (size_t i = 0; i < 8 * 16; i++) d[i] = 0.1f;

    SNEPPXTensor* adv = NULL;
    SNEPPX_arc_simulate_attack(clean, SNEPPX_ATTACK_CW, 0.1f, &adv);
    SX_ASSERT(adv != NULL, "adv not null");

    float* cd = (float*)clean->data;
    float* ad = (float*)adv->data;
    float l2 = 0.0f;
    int different = 0;
    for (size_t i = 0; i < 8 * 16; i++) {
        float diff = ad[i] - cd[i];
        l2 += diff * diff;
        if (fabsf(diff) > 1e-6f) different = 1;
    }
    SX_ASSERT(different, "adversarial != clean");

    SNEPPX_tensor_destroy(clean);
    SNEPPX_tensor_destroy(adv);
}


TEST(test_arc_attack, test_fgsm) { test_fgsm(); }
TEST(test_arc_attack, test_pgd) { test_pgd(); }
TEST(test_arc_attack, test_cw) { test_cw(); }

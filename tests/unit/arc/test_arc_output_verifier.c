#include "adversarial_robustness_certification.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Arc Output Verifier
 *
 * WHAT
 *   Test Arc Output Verifier.
 *
 * CONCEPT
 *   Provides the Test Arc Output Verifier.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_verifier_create(void) {
    SNEPPXOutputVerifier* v = SNEPPX_arc_output_verifier_create(32, 2, 42);
    SX_ASSERT(v != NULL, "verifier not null");
    SX_ASSERT(v->verification_weights[0]->shape[0] == 32, "w0 rows 32");
    SX_ASSERT(v->verification_weights[0]->shape[1] == 32, "w0 cols 32");
    SX_ASSERT(v->verification_weights[1] != NULL, "w1 not null");
    SX_ASSERT(v->num_layers == 2, "num_layers == 2");
    SNEPPX_arc_output_verifier_destroy(v);
}

static void test_verify_normal(void) {
    SNEPPXOutputVerifier* v = SNEPPX_arc_output_verifier_create(16, 1, 42);
    SX_ASSERT(v != NULL, "verifier not null");

    size_t shape_in[] = {4, 16};
    SNEPPXTensor* out = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)out->data;
    for (size_t i = 0; i < 4 * 16; i++) d[i] = 0.1f;

    SNEPPXTensor* verified = NULL;
    float conf = 0.0f;
    SNEPPX_arc_verify_output(v, out, &verified, &conf);
    SX_ASSERT(verified != NULL, "verified not null");
    SX_ASSERT(conf == 1.0f, "confidence 1.0 on first call (no history)");

    SNEPPX_arc_verify_output(v, out, &verified, &conf);
    SX_ASSERT(conf > 0.8f, "confidence > 0.8 on second call (consistent)");

    SNEPPX_tensor_destroy(out);
    if (verified) SNEPPX_tensor_destroy(verified);
    SNEPPX_arc_output_verifier_destroy(v);
}

static void test_verify_inconsistent(void) {
    SNEPPXOutputVerifier* v = SNEPPX_arc_output_verifier_create(16, 1, 42);
    SX_ASSERT(v != NULL, "verifier not null");

    size_t shape_in[] = {4, 16};
    SNEPPXTensor* out1 = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d1 = (float*)out1->data;
    for (size_t i = 0; i < 4 * 16; i++) d1[i] = 0.1f;

    SNEPPXTensor* verified1 = NULL;
    float conf1 = 0.0f;
    SNEPPX_arc_verify_output(v, out1, &verified1, &conf1);
    SNEPPX_tensor_destroy(verified1);

    SNEPPXTensor* out2 = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d2 = (float*)out2->data;
    unsigned long s2 = 999;
    for (size_t i = 0; i < 4 * 16; i++) {
        s2 = s2 * 1103515245UL + 12345UL;
        d2[i] = ((float)((s2 >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 10.0f;
    }

    SNEPPXTensor* verified2 = NULL;
    float conf2 = 0.0f;
    SNEPPX_arc_verify_output(v, out2, &verified2, &conf2);
    SX_ASSERT(verified2 != NULL, "verified2 not null");
    SX_ASSERT(conf2 < 0.8f, "confidence < 0.8 for random vs constant");

    SNEPPX_tensor_destroy(out1);
    SNEPPX_tensor_destroy(out2);
    if (verified2) SNEPPX_tensor_destroy(verified2);
    SNEPPX_arc_output_verifier_destroy(v);
}


TEST(test_arc_output_verifier, test_verifier_create) { test_verifier_create(); }
TEST(test_arc_output_verifier, test_verify_normal) { test_verify_normal(); }
TEST(test_arc_output_verifier, test_verify_inconsistent) { test_verify_inconsistent(); }

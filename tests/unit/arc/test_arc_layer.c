#include "adversarial_robustness_certification.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Arc Layer
 *
 * WHAT
 *   Test Arc Layer.
 *
 * CONCEPT
 *   Provides the Test Arc Layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_arc_forward(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 32, 32, 42);
    SX_ASSERT(layer != NULL, "layer not null");

    size_t shape_in[] = {4, 32};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* d = (float*)input->data;
    for (size_t i = 0; i < 4 * 32; i++) d[i] = 0.1f;

    SNEPPXTensor* output = NULL;
    float metrics[4];
    SNEPPX_arc_forward(layer, input, &output, metrics);

    SX_ASSERT(output != NULL, "output not null");
    SX_ASSERT(output->shape[0] == 4, "batch == 4");
    SX_ASSERT(output->shape[1] == 32, "dim == 32");

    int all_finite = 1;
    float* od = (float*)output->data;
    for (size_t i = 0; i < output->size; i++) {
        if (!isfinite(od[i])) { all_finite = 0; break; }
    }
    SX_ASSERT(all_finite, "all finite in output");

    for (int i = 0; i < 4; i++) {
        SX_ASSERT(isfinite(metrics[i]), "all metrics finite");
    }

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_arc_layer_destroy(layer);
}


TEST(test_arc_layer, test_arc_forward) { test_arc_forward(); }

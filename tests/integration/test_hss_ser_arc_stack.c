#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Hss Ser Arc Stack
 *
 * WHAT
 *   Test Hss Ser Arc Stack.
 *
 * CONCEPT
 *   Provides the Test Hss Ser Arc Stack.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_hss_ser_arc_stack(void) {
    SNEPPXHSSConfig hss_cfg = SNEPPX_hss_config_default();
    hss_cfg.state_dim = 8; hss_cfg.input_dim = 16; hss_cfg.output_dim = 16;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;
    SNEPPXHSSModel* hss = SNEPPX_hss_model_create(&hss_cfg, 42);
    SX_ASSERT(hss != NULL, "hss model");

    SNEPPXSERConfig ser_cfg = SNEPPX_ser_config_default();
    ser_cfg.num_experts = 4; ser_cfg.num_active = 2; ser_cfg.input_dim = 16;
    ser_cfg.expert_dim = 32; ser_cfg.output_dim = 16;
    SNEPPXSERModel* ser = SNEPPX_ser_model_create(&ser_cfg, 99, 1);
    SX_ASSERT(ser != NULL, "ser model");

    SNEPPXARCConfig arc_cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* arc = SNEPPX_arc_layer_create(&arc_cfg, 16, 16, 200);
    SX_ASSERT(arc != NULL, "arc layer");

    size_t shape_in[] = {2, 8, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input");

    SNEPPXTensor* hss_out = NULL;
    int ret = SNEPPX_hss_forward(hss, input, &hss_out);
    SX_ASSERT(ret == 0 && hss_out != NULL, "hss forward");
    SX_ASSERT(hss_out->shape[0] == 2 && hss_out->shape[1] == 8 && hss_out->shape[2] == 16, "hss shape");

    size_t merged = 2 * 8;
    SNEPPXTensor flat;
    size_t flat_shape[] = {merged, 16};
    flat.data = hss_out->data;
    flat.shape = flat_shape; flat.ndim = 2;
    flat.size = merged * 16; flat.item_size = sizeof(float);
    flat.dtype = SNEPPX_FLOAT32; flat.strides = NULL;

    SNEPPXTensor* ser_out = NULL;
    SNEPPX_ser_forward(ser->layers[0], &flat, &ser_out);
    SX_ASSERT(ser_out != NULL, "ser output");
    SX_ASSERT(ser_out->shape[0] == merged && ser_out->shape[1] == 16, "ser shape");

    SNEPPXTensor* arc_out = NULL;
    float metrics[4];
    SNEPPX_arc_forward(arc, &flat, &arc_out, metrics);
    SX_ASSERT(arc_out != NULL, "arc output");
    SX_ASSERT(arc_out->shape[0] == merged && arc_out->shape[1] == 16, "arc shape");

    float* od = (float*)arc_out->data;
    int ok = 1;
    for (size_t i = 0; i < arc_out->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    SX_ASSERT(ok, "all finite");
    for (int i = 0; i < 4; i++) SX_ASSERT(isfinite(metrics[i]), "all metrics finite");

    printf("  Security metrics: anomaly=%.4f conf=%.4f clamp=%.4f norm=%.4f\n",
           metrics[0], metrics[1], metrics[2], metrics[3]);

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(hss_out);
    SNEPPX_tensor_destroy(ser_out);
    if (arc_out) SNEPPX_tensor_destroy(arc_out);
    SNEPPX_hss_model_destroy(hss);
    SNEPPX_ser_model_destroy(ser);
    SNEPPX_arc_layer_destroy(arc);
}


TEST(test_hss_ser_arc_stack, test_hss_ser_arc_stack) { test_hss_ser_arc_stack(); }

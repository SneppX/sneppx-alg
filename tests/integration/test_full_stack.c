#include "hierarchical_state_space.h"
#include "test_gtest.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Full Stack
 *
 * WHAT
 *   Test Full Stack.
 *
 * CONCEPT
 *   Provides the Test Full Stack.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_full_stack(void) {
    SNEPPXHSSConfig hss_cfg = SNEPPX_hss_config_default();
    hss_cfg.state_dim = 4; hss_cfg.input_dim = 8; hss_cfg.output_dim = 8;
    hss_cfg.num_layers = 1; hss_cfg.use_hierarchical = 0;
    SNEPPXHSSModel* hss = SNEPPX_hss_model_create(&hss_cfg, 42);

    SNEPPXSERConfig ser_cfg = SNEPPX_ser_config_default();
    ser_cfg.num_experts = 2; ser_cfg.num_active = 1; ser_cfg.input_dim = 8;
    ser_cfg.expert_dim = 16; ser_cfg.output_dim = 8;
    SNEPPXSERModel* ser = SNEPPX_ser_model_create(&ser_cfg, 99, 1);
    SX_ASSERT(ser != NULL, "ser model");

    SNEPPXARCConfig arc_cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* arc = SNEPPX_arc_layer_create(&arc_cfg, 8, 8, 200);
    SX_ASSERT(arc != NULL, "arc layer");

    SNEPPXNPEProgram* npe_prog = SNEPPX_npe_compile_mlp(8, 16);
    SX_ASSERT(npe_prog != NULL, "npe program");
    unsigned long s = 42;
    for (size_t i = 0; i < 16 * 8 + 16 + 8 * 16 + 8; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)npe_prog->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
    }

    SNEPPXNPEConfig npe_cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&npe_cfg);
    SNEPPX_npe_vm_load(vm, npe_prog);

    size_t shape_in[] = {1, 4, 8};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 3, SNEPPX_FLOAT32);
    SX_ASSERT(input != NULL, "input");

    SNEPPXTensor* hss_out = NULL;
    int ret = SNEPPX_hss_forward(hss, input, &hss_out);
    SX_ASSERT(ret == 0 && hss_out != NULL && hss_out->shape[0] == 1 && hss_out->shape[1] == 4 && hss_out->shape[2] == 8, "hss ok");

    size_t merged = 4;
    SNEPPXTensor flat;
    size_t flat_shape[] = {merged, 8};
    flat.data = hss_out->data; flat.shape = flat_shape; flat.ndim = 2;
    flat.size = merged * 8; flat.item_size = sizeof(float);
    flat.dtype = SNEPPX_FLOAT32; flat.strides = NULL;

    SNEPPXTensor* ser_out = NULL;
    SNEPPX_ser_forward(ser->layers[0], &flat, &ser_out);
    SX_ASSERT(ser_out != NULL && ser_out->shape[0] == merged && ser_out->shape[1] == 8, "ser ok");

    SNEPPXTensor* arc_out = NULL;
    float metrics[4];
    SNEPPX_arc_forward(arc, ser_out, &arc_out, metrics);
    SX_ASSERT(arc_out != NULL && arc_out->shape[0] == merged && arc_out->shape[1] == 8, "arc ok");

    SNEPPXTensor* npe_out = NULL;
    int r = SNEPPX_npe_vm_run(vm, arc_out, &npe_out);
    SX_ASSERT(npe_out != NULL && (r == 0 || r == 2), "npe ok");

    float* od = (float*)npe_out->data;
    int ok = 1;
    for (size_t i = 0; i < npe_out->size; i++) {
        if (!isfinite(od[i])) { ok = 0; break; }
    }
    SX_ASSERT(ok, "all finite");
    SX_ASSERT(vm->trace_length > 0, "execution trace non-empty");

    printf("  Trace length: %zu\n", vm->trace_length);

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(hss_out);
    SNEPPX_tensor_destroy(ser_out);
    SNEPPX_tensor_destroy(arc_out);
    SNEPPX_tensor_destroy(npe_out);
    SNEPPX_hss_model_destroy(hss);
    SNEPPX_ser_model_destroy(ser);
    SNEPPX_arc_layer_destroy(arc);
    SNEPPX_npe_vm_destroy(vm);
    SNEPPX_npe_program_destroy(npe_prog);
}


TEST(test_full_stack, test_full_stack) { test_full_stack(); }

#include "system_architecture_definitions.h"
#include "test_gtest.h"
#include "differentiable_training_pipeline.h"
#include "automatic_differentiation_framework.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Npe Train
 *
 * WHAT
 *   Test Npe Train.
 *
 * CONCEPT
 *   Provides the Test Npe Train.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_npe_train_step(void) {
    SNEPPXArchConfig cfg = SNEPPX_arch_config_default();
    cfg.enable_attention = 0;
    cfg.enable_hss = 0;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.enable_npe = 1;
    cfg.npe_config = SNEPPX_npe_config_default();

    SNEPPXModel* model = SNEPPX_model_create(&cfg);
    SX_ASSERT(model != NULL, "model created");
    SX_ASSERT(model->npe_program != NULL, "npe program created");

    size_t nw = SNEPPX_model_get_params(model, NULL, 0);
    SX_ASSERT(nw == 4, "npe has 4 params");

    SNEPPXTensor** params = (SNEPPXTensor**)malloc(nw * sizeof(SNEPPXTensor*));
    SNEPPX_model_get_params(model, params, nw);

    size_t sh[] = {4, 4};
    SNEPPXTensor* input = SNEPPX_tensor_randn(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* target = SNEPPX_tensor_randn(sh, 2, SNEPPX_FLOAT32);

    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable** wv = (SNEPPXVariable**)malloc(nw * sizeof(SNEPPXVariable*));
    for (size_t i = 0; i < nw; i++) wv[i] = SNEPPX_variable_create(params[i], 1);

    SNEPPXVariable* inv = SNEPPX_variable_create(input, 0);
    SNEPPXVariable* outv = NULL;
    int ret = SNEPPX_model_build_train_graph(model, tape, inv, wv, nw, &outv);
    SX_ASSERT(ret == 0, "build graph ok");
    SX_ASSERT(outv != NULL, "output non-null");

    SNEPPXVariable* tgv = SNEPPX_variable_create(target, 0);
    SNEPPXVariable* loss_v = SNEPPX_mse_loss(tape, outv, tgv);
    SX_ASSERT(loss_v != NULL, "loss non-null");
    float loss = ((float*)loss_v->data->data)[0];
    SX_ASSERT(isfinite(loss), "loss finite");
    SX_ASSERT(loss >= 0.0f, "loss non-negative");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(target);
    for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; SNEPPX_variable_destroy(wv[i]); }
    inv->data = NULL; SNEPPX_variable_destroy(inv);
    tgv->data = NULL; SNEPPX_variable_destroy(tgv);
    SNEPPX_tape_destroy(tape);
    free(wv);
    free(params);
    SNEPPX_model_destroy(model);
}

static void test_npe_train_convergence(void) {
    SNEPPXArchConfig cfg = SNEPPX_arch_config_default();
    cfg.enable_attention = 0;
    cfg.enable_hss = 0;
    cfg.input_dim = 4;
    cfg.output_dim = 4;
    cfg.enable_npe = 1;
    cfg.npe_config = SNEPPX_npe_config_default();

    SNEPPXModel* model = SNEPPX_model_create(&cfg);
    SX_ASSERT(model != NULL, "model created");

    SNEPPXOptimizerConfig opt_cfg = SNEPPX_optimizer_config_default();
    opt_cfg.learning_rate = 0.01f;
    opt_cfg.type = SNEPPX_OPTIMIZER_ADAM;
    opt_cfg.weight_decay = 0.0f;
    opt_cfg.grad_clip = 1.0f;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&opt_cfg);

    size_t nw = SNEPPX_model_get_params(model, NULL, 0);
    SX_ASSERT(nw == 4, "npe has 4 params");

    SNEPPXTensor** params = (SNEPPXTensor**)malloc(nw * sizeof(SNEPPXTensor*));
    SNEPPX_model_get_params(model, params, nw);

    size_t sh[] = {4, 4};
    SNEPPXTensor* input = SNEPPX_tensor_randn(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* target = SNEPPX_tensor_randn(sh, 2, SNEPPX_FLOAT32);

    float init_loss = 0.0f;
    {
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable** wv = (SNEPPXVariable**)malloc(nw * sizeof(SNEPPXVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = SNEPPX_variable_create(params[i], 1);
        SNEPPXVariable* inv = SNEPPX_variable_create(input, 0);
        SNEPPXVariable* outv = NULL;
        SNEPPX_model_build_train_graph(model, tape, inv, wv, nw, &outv);
        SNEPPXVariable* tgv = SNEPPX_variable_create(target, 0);
        SNEPPXVariable* loss_v = SNEPPX_mse_loss(tape, outv, tgv);
        if (loss_v) init_loss = ((float*)loss_v->data->data)[0];
        printf("  init loss=%.6f\n", (double)init_loss);
        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; SNEPPX_variable_destroy(wv[i]); }
        inv->data = NULL; SNEPPX_variable_destroy(inv);
        tgv->data = NULL; SNEPPX_variable_destroy(tgv);
        SNEPPX_tape_destroy(tape);
        free(wv);
    }

    int steps = 50;
    for (int s = 0; s < steps; s++) {
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable** wv = (SNEPPXVariable**)malloc(nw * sizeof(SNEPPXVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = SNEPPX_variable_create(params[i], 1);
        SNEPPXVariable* inv = SNEPPX_variable_create(input, 0);
        SNEPPXVariable* outv = NULL;
        SNEPPX_model_build_train_graph(model, tape, inv, wv, nw, &outv);
        SNEPPXVariable* tgv = SNEPPX_variable_create(target, 0);
        SNEPPXVariable* loss_v = SNEPPX_mse_loss(tape, outv, tgv);
        if (loss_v) {
            SNEPPX_tape_backward(tape, loss_v);
            SNEPPXTensor** grads = (SNEPPXTensor**)malloc(nw * sizeof(SNEPPXTensor*));
            for (size_t i = 0; i < nw; i++) grads[i] = wv[i]->grad;
            SNEPPX_optimizer_step(opt, params, grads, nw);
            free(grads);
        }
        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; SNEPPX_variable_destroy(wv[i]); }
        inv->data = NULL; SNEPPX_variable_destroy(inv);
        tgv->data = NULL; SNEPPX_variable_destroy(tgv);
        SNEPPX_tape_destroy(tape);
        free(wv);
    }

    float final_loss = 0.0f;
    {
        SNEPPXTape* tape = SNEPPX_tape_create();
        SNEPPXVariable** wv = (SNEPPXVariable**)malloc(nw * sizeof(SNEPPXVariable*));
        for (size_t i = 0; i < nw; i++) wv[i] = SNEPPX_variable_create(params[i], 1);
        SNEPPXVariable* inv = SNEPPX_variable_create(input, 0);
        SNEPPXVariable* outv = NULL;
        SNEPPX_model_build_train_graph(model, tape, inv, wv, nw, &outv);
        SNEPPXVariable* tgv = SNEPPX_variable_create(target, 0);
        SNEPPXVariable* loss_v = SNEPPX_mse_loss(tape, outv, tgv);
        if (loss_v) final_loss = ((float*)loss_v->data->data)[0];
        printf("  final loss=%.6f ratio=%.4f\n", (double)final_loss, (double)(final_loss / (init_loss + 1e-10f)));
        for (size_t i = 0; i < nw; i++) { wv[i]->data = NULL; wv[i]->grad = NULL; SNEPPX_variable_destroy(wv[i]); }
        inv->data = NULL; SNEPPX_variable_destroy(inv);
        tgv->data = NULL; SNEPPX_variable_destroy(tgv);
        SNEPPX_tape_destroy(tape);
        free(wv);
    }

    SX_ASSERT(final_loss < init_loss * 0.9f, "loss decreased >10%%");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(target);
    SNEPPX_optimizer_destroy(opt);
    SNEPPX_model_destroy(model);
    free(params);
}


TEST(test_npe_train, test_npe_train_step) { test_npe_train_step(); }
TEST(test_npe_train, test_npe_train_convergence) { test_npe_train_convergence(); }

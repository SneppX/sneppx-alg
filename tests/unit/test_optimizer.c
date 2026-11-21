#include "gradient_optimization_suite.h"
#include "test_gtest.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Optimizer
 *
 * WHAT
 *   Test Optimizer.
 *
 * CONCEPT
 *   Provides optimizer implementations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_optimizer_config_default(void) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.type = SNEPPX_OPTIMIZER_ADAM;
    SX_ASSERT(cfg.type == SNEPPX_OPTIMIZER_ADAM, "adam default type");
    SX_ASSERT(cfg.learning_rate > 0, "learning rate > 0");
}

static void test_sgd_create_step(void) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.type = SNEPPX_OPTIMIZER_SGD;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&cfg);
    SX_ASSERT(opt != NULL, "sgd optimizer created");

    size_t shape[] = {4, 4};
    SNEPPXTensor* params = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* grads = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* gd = (float*)grads->data;
    for (size_t i = 0; i < grads->size; i++) gd[i] = 0.1f;

    SNEPPX_optimizer_step(opt, &params, &grads, 1);
    float* pd = (float*)params->data;
    for (size_t i = 0; i < params->size; i++) {
        SX_ASSERT(pd[i] < 1.0f, "parameter decreased after SGD step");
    }

    SNEPPX_tensor_destroy(grads);
    SNEPPX_tensor_destroy(params);
    SNEPPX_optimizer_destroy(opt);
}

static void test_adam_create_step(void) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.type = SNEPPX_OPTIMIZER_ADAM;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&cfg);
    SX_ASSERT(opt != NULL, "adam optimizer created");

    size_t shape[] = {2, 2};
    SNEPPXTensor* params = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* grads = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* gd = (float*)grads->data;
    for (size_t i = 0; i < grads->size; i++) gd[i] = 0.05f;

    SNEPPX_optimizer_step(opt, &params, &grads, 1);
    float* pd = (float*)params->data;
    for (size_t i = 0; i < params->size; i++) {
        SX_ASSERT(pd[i] <= 1.0f, "parameter updated after Adam step");
    }

    SNEPPX_tensor_destroy(grads);
    SNEPPX_tensor_destroy(params);
    SNEPPX_optimizer_destroy(opt);
}

static void test_adamw_create_step(void) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.type = SNEPPX_OPTIMIZER_ADAMW;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&cfg);
    SX_ASSERT(opt != NULL, "adamw optimizer created");

    size_t shape[] = {3, 3};
    SNEPPXTensor* params = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* grads = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    float* gd = (float*)grads->data;
    for (size_t i = 0; i < grads->size; i++) gd[i] = 0.1f;

    SNEPPX_optimizer_step(opt, &params, &grads, 1);

    SNEPPX_tensor_destroy(grads);
    SNEPPX_tensor_destroy(params);
    SNEPPX_optimizer_destroy(opt);
}

static void test_lr_scheduler_step(void) {
    SNEPPXOptimizerConfig cfg = SNEPPX_optimizer_config_default();
    cfg.type = SNEPPX_OPTIMIZER_SGD;
    SNEPPXOptimizer* opt = SNEPPX_optimizer_create(&cfg);

    SNEPPXLRScheduler* sched = SNEPPX_lr_scheduler_step_lr(&cfg.learning_rate, 0.1f, 10);
    SX_ASSERT(sched != NULL, "lr scheduler created");

    SNEPPX_lr_scheduler_step(sched, 0.0f);

    SNEPPX_lr_scheduler_destroy(sched);
    SNEPPX_optimizer_destroy(opt);
}


TEST(test_optimizer, optimizer_config_default) { test_optimizer_config_default(); }
TEST(test_optimizer, sgd_create_step) { test_sgd_create_step(); }
TEST(test_optimizer, adam_create_step) { test_adam_create_step(); }
TEST(test_optimizer, adamw_create_step) { test_adamw_create_step(); }
TEST(test_optimizer, lr_scheduler_step) { test_lr_scheduler_step(); }

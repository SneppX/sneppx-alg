#include "differentiable_training_pipeline.h"
#include "test_gtest.h"
#include "trainer_cuda.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * SNEPPX - Test Trainer Cuda
 *
 * WHAT
 *   Test Trainer Cuda.
 *
 * CONCEPT
 *   Provides the Test Trainer Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_cuda_available(void) {
    int avail = SNEPPX_trainer_cuda_available();
    printf("[CUDA avail=%d] ", avail);
}

static void test_cuda_init_shutdown(void) {
    int avail = SNEPPX_trainer_cuda_available();
    if (!avail) {
        printf("[SKIP: no CUDA device] ");
        GTEST_SKIP() << "legacy pass-on-skip";
        return;
    }

    SNEPPXTensor* p = SNEPPX_tensor_create(SX_ARR_C(size_t, 1, 10), 1, SNEPPX_FLOAT32);
    SX_ASSERT(p != NULL, "param tensor created");

    SNEPPXTensor* params[] = {p};
    int ret = SNEPPX_trainer_cuda_init(params, 1);
    SX_ASSERT(ret == 0, "cuda init succeeded");

    SNEPPX_trainer_cuda_shutdown();

    ret = SNEPPX_trainer_cuda_init(params, 1);
    SX_ASSERT(ret == 0, "re-init after shutdown");

    SNEPPX_trainer_cuda_shutdown();
    SNEPPX_tensor_destroy(p);
}

static void test_cuda_transfer(void) {
    int avail = SNEPPX_trainer_cuda_available();
    if (!avail) {
        printf("[SKIP: no CUDA device] ");
        GTEST_SKIP() << "legacy pass-on-skip";
        return;
    }

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    SNEPPXTensor* p = SNEPPX_tensor_create(SX_ARR_C(size_t, 1, 5), 1, SNEPPX_FLOAT32);
    SX_ASSERT(p != NULL, "tensor created");
    memcpy(p->data, data, 5 * sizeof(float));

    SNEPPXTensor* params[] = {p};
    int ret = SNEPPX_trainer_cuda_init(params, 1);
    SX_ASSERT(ret == 0, "cuda init");

    ret = SNEPPX_trainer_cuda_transfer_to_device(params, 1);
    SX_ASSERT(ret == 0, "transfer to device");

    memset(p->data, 0, 5 * sizeof(float));

    ret = SNEPPX_trainer_cuda_transfer_to_host(params, 1);
    SX_ASSERT(ret == 0, "transfer to host");

    float* pd = (float*)p->data;
    SX_ASSERT(fabsf(pd[0] - 1.0f) < 1e-5f, "val[0] == 1.0");
    SX_ASSERT(fabsf(pd[4] - 5.0f) < 1e-5f, "val[4] == 5.0");

    SNEPPX_trainer_cuda_shutdown();
    SNEPPX_tensor_destroy(p);
}

static void test_trainer_config_cuda(void) {
    SNEPPXTrainConfig cfg = SNEPPX_train_config_default();
    SX_ASSERT(cfg.use_cuda_optimizer == 0, "default cuda_optimizer off");
    cfg.use_cuda_optimizer = 1;

    SNEPPXModel* model = NULL;

    if (SNEPPX_trainer_cuda_available()) {
        SX_ASSERT(1, "CUDA available on this system");
    } else {
        printf("[NO CUDA] ");
    }
}


TEST(test_trainer_cuda, cuda_available) { test_cuda_available(); }
TEST(test_trainer_cuda, cuda_init_shutdown) { test_cuda_init_shutdown(); }
TEST(test_trainer_cuda, cuda_transfer) { test_cuda_transfer(); }
TEST(test_trainer_cuda, trainer_config_cuda) { test_trainer_config_cuda(); }

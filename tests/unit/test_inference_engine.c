#include "inference_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Test Inference Engine
 *
 * WHAT
 *   Test Inference Engine.
 *
 * CONCEPT
 *   Provides the Test Inference Engine.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_inference_engine_create(void) {
    SNEPPXInferenceEngine* engine = SNEPPX_inference_engine_create(42);
    SX_ASSERT(engine != NULL, "engine created");
    SNEPPX_inference_engine_destroy(engine);
}

static void test_inference_engine_run(void) {
    SNEPPXInferenceEngine* engine = SNEPPX_inference_engine_create(42);
    size_t shape[] = {1, 8};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = SNEPPX_inference_engine_run(engine, input);
    SX_ASSERT(output != NULL, "inference output");
    SX_ASSERT(output->size > 0, "output has data");
    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_inference_engine_destroy(engine);
}

static void test_inference_engine_reset(void) {
    SNEPPXInferenceEngine* engine = SNEPPX_inference_engine_create(42);
    SNEPPX_inference_engine_reset(engine);
    SX_ASSERT(engine != NULL, "reset does not crash");
    SNEPPX_inference_engine_destroy(engine);
}


TEST(test_inference_engine, inference_engine_create) { test_inference_engine_create(); }
TEST(test_inference_engine, inference_engine_run) { test_inference_engine_run(); }
TEST(test_inference_engine, inference_engine_reset) { test_inference_engine_reset(); }

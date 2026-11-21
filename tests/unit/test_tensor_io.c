#include "multidimensional_tensor_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Tensor Io
 *
 * WHAT
 *   Test Tensor Io.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






/* ---------- Save/Load ---------- */

static void test_save_load_basic(void) {
    size_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 6 * sizeof(float));
    int ret = SNEPPX_tensor_save(t, "test_tensor.bin");
    SX_ASSERT(ret == 0, "save returns 0");
    SNEPPXTensor* loaded = SNEPPX_tensor_load("test_tensor.bin");
    SX_ASSERT(loaded != NULL, "load returns tensor");
    SX_ASSERT(loaded->ndim == 2, "loaded ndim == 2");
    SX_ASSERT(loaded->shape[0] == 2 && loaded->shape[1] == 3, "loaded shape 2x3");
    SX_ASSERT(loaded->dtype == SNEPPX_FLOAT32, "loaded dtype F32");
    float* ld = (float*)loaded->data;
    for (size_t i = 0; i < 6; i++) SX_ASSERT_NEAR(ld[i], data[i], 1e-6f, "loaded data matches");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(loaded);
    remove("test_tensor.bin");
}

static void test_save_load_f64(void) {
    size_t shape[] = {2, 2};
    double data[] = {1.5, 2.5, 3.5, 4.5};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT64);
    memcpy(t->data, data, 4 * sizeof(double));
    SX_ASSERT(SNEPPX_tensor_save(t, "test_f64.bin") == 0, "save f64");
    SNEPPXTensor* loaded = SNEPPX_tensor_load("test_f64.bin");
    SX_ASSERT(loaded != NULL, "load f64");
    SX_ASSERT(loaded->dtype == SNEPPX_FLOAT64, "loaded dtype F64");
    double* ld = (double*)loaded->data;
    for (size_t i = 0; i < 4; i++) SX_ASSERT_NEAR((float)ld[i], (float)data[i], 1e-5f, "f64 data matches");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(loaded);
    remove("test_f64.bin");
}

static void test_save_load_null(void) {
    SX_ASSERT(SNEPPX_tensor_save(NULL, "test.bin") == -1, "save null returns -1");
    SX_ASSERT(SNEPPX_tensor_load(NULL) == NULL, "load null returns NULL");
    SX_ASSERT(SNEPPX_tensor_load("nonexistent.bin") == NULL, "load nonexistent returns NULL");
}

/* ---------- Cast ---------- */

static void test_cast_f32_to_f64(void) {
    size_t shape[] = {3};
    float data[] = {1.0f, 2.5f, 3.14f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 3 * sizeof(float));
    SNEPPXTensor* c = SNEPPX_tensor_cast(t, SNEPPX_FLOAT64);
    SX_ASSERT(c != NULL, "cast result not null");
    SX_ASSERT(c->dtype == SNEPPX_FLOAT64, "cast to F64");
    double* d = (double*)c->data;
    SX_ASSERT_NEAR((float)d[0], 1.0f, 1e-6f, "cast f32->f64[0]");
    SX_ASSERT_NEAR((float)d[1], 2.5f, 1e-6f, "cast f32->f64[1]");
    SX_ASSERT_NEAR((float)d[2], 3.14f, 1e-5f, "cast f32->f64[2]");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(c);
}

static void test_cast_f64_to_f32(void) {
    size_t shape[] = {3};
    double data[] = {1.0, 2.5, 3.14};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT64);
    memcpy(t->data, data, 3 * sizeof(double));
    SNEPPXTensor* c = SNEPPX_tensor_cast(t, SNEPPX_FLOAT32);
    SX_ASSERT(c != NULL, "cast result not null");
    SX_ASSERT(c->dtype == SNEPPX_FLOAT32, "cast to F32");
    float* d = (float*)c->data;
    SX_ASSERT_NEAR(d[0], 1.0f, 1e-6f, "cast f64->f32[0]");
    SX_ASSERT_NEAR(d[1], 2.5f, 1e-6f, "cast f64->f32[1]");
    SX_ASSERT_NEAR(d[2], 3.14f, 1e-5f, "cast f64->f32[2]");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(c);
}

static void test_cast_f32_to_i32(void) {
    size_t shape[] = {3};
    float data[] = {1.7f, 2.4f, -3.9f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 3 * sizeof(float));
    SNEPPXTensor* c = SNEPPX_tensor_cast(t, SNEPPX_INT32);
    SX_ASSERT(c != NULL, "cast result not null");
    SX_ASSERT(c->dtype == SNEPPX_INT32, "cast to INT32");
    int32_t* d = (int32_t*)c->data;
    SX_ASSERT(d[0] == 1, "cast f32->i32 1.7->1");
    SX_ASSERT(d[1] == 2, "cast f32->i32 2.4->2");
    SX_ASSERT(d[2] == -3, "cast f32->i32 -3.9->-3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(c);
}

static void test_cast_identity(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* c = SNEPPX_tensor_cast(t, SNEPPX_FLOAT32);
    SX_ASSERT(c != NULL, "identity cast");
    SX_ASSERT(c->dtype == SNEPPX_FLOAT32, "identity dtype");
    float* d = (float*)c->data;
    SX_ASSERT_NEAR(d[0], 1.0f, 1e-6f, "identity data");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(c);
}

/* ---------- To Device / To Layout ---------- */

static void test_to_device_cpu(void) {
    size_t shape[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* d = SNEPPX_tensor_to_device(t, SNEPPX_DEVICE_CPU);
    SX_ASSERT(d != NULL, "to_device result not null");
    SX_ASSERT(d->device == SNEPPX_DEVICE_CPU, "device == CPU");
    SX_ASSERT(d != t, "to_device returns copy");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(d);
}

static void test_to_device_null(void) {
    SX_ASSERT(SNEPPX_tensor_to_device(NULL, SNEPPX_DEVICE_CPU) == NULL, "to_device null");
}


TEST(test_tensor_io, test_save_load_basic) { test_save_load_basic(); }
TEST(test_tensor_io, test_save_load_f64) { test_save_load_f64(); }
TEST(test_tensor_io, test_save_load_null) { test_save_load_null(); }
TEST(test_tensor_io, test_cast_f32_to_f64) { test_cast_f32_to_f64(); }
TEST(test_tensor_io, test_cast_f64_to_f32) { test_cast_f64_to_f32(); }
TEST(test_tensor_io, test_cast_f32_to_i32) { test_cast_f32_to_i32(); }
TEST(test_tensor_io, test_cast_identity) { test_cast_identity(); }
TEST(test_tensor_io, test_to_device_cpu) { test_to_device_cpu(); }
TEST(test_tensor_io, test_to_device_null) { test_to_device_null(); }

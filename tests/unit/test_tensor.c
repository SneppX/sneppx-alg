#include "multidimensional_tensor_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/*
 * SNEPPX - Test Tensor
 *
 * WHAT
 *   Test Tensor.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






static void test_create_2d(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor not null");
    SX_ASSERT(t->ndim == 2, "ndim == 2");
    SX_ASSERT(t->shape[0] == 2, "shape[0] == 2");
    SX_ASSERT(t->shape[1] == 3, "shape[1] == 3");
    SX_ASSERT(t->size == 6, "size == 6");
    SNEPPX_tensor_destroy(t);
}

static void test_zeros(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor not null");
    float* data = (float*)t->data;
    for (size_t i = 0; i < 4; i++) {
        SX_ASSERT(data[i] == 0.0f, "all zeros");
    }
    SNEPPX_tensor_destroy(t);
}

static void test_ones(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor not null");
    float* data = (float*)t->data;
    for (size_t i = 0; i < 4; i++) {
        SX_ASSERT(data[i] == 1.0f, "all ones");
    }
    SNEPPX_tensor_destroy(t);
}

static void test_set_get(void) {
    size_t shape[] = {3, 3};
    SNEPPXTensor* t = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor not null");
    size_t indices[] = {1, 2};
    SNEPPX_tensor_set_f32(t, indices, 3.14f);
    float val = SNEPPX_tensor_get_f32(t, indices);
    SX_ASSERT_NEAR(val, 3.14f, 1e-4f, "value at [1,2] == 3.14");
    SNEPPX_tensor_destroy(t);
}

static void test_randn_range(void) {
    size_t shape[] = {100};
    SNEPPXTensor* t = SNEPPX_tensor_randn(shape, 1, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor not null");
    float* data = (float*)t->data;
    float min_val = data[0], max_val = data[0];
    for (size_t i = 0; i < 100; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    SX_ASSERT(min_val > -5.0f, "min > -5.0");
    SX_ASSERT(max_val < 5.0f, "max < 5.0");
    SNEPPX_tensor_destroy(t);
}

static void test_print(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT(t != NULL, "tensor not null");
    SNEPPX_tensor_print(t);
    SNEPPX_tensor_destroy(t);
}


TEST(test_tensor, test_create_2d) { test_create_2d(); }
TEST(test_tensor, test_zeros) { test_zeros(); }
TEST(test_tensor, test_ones) { test_ones(); }
TEST(test_tensor, test_set_get) { test_set_get(); }
TEST(test_tensor, test_randn_range) { test_randn_range(); }
TEST(test_tensor, test_print) { test_print(); }
static void test_element_access_bounds_and_null_safety(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* tensor = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(tensor, "bounds tensor");
    size_t valid[] = {1, 1};
    size_t invalid[] = {2, 0};
    SNEPPX_tensor_set_f32(tensor, valid, 7.0f);
    SNEPPX_tensor_set_f32(tensor, invalid, 99.0f);
    SX_ASSERT_NEAR(SNEPPX_tensor_get_f32(tensor, valid), 7.0f, 1e-6f, "valid access preserved");
    SX_ASSERT_NEAR(SNEPPX_tensor_get_f32(tensor, invalid), 0.0f, 1e-6f, "invalid read returns zero");
    SX_ASSERT_NEAR(SNEPPX_tensor_get_f32(tensor, NULL), 0.0f, 1e-6f, "null indices return zero");
    SX_ASSERT_NEAR(SNEPPX_tensor_get_f32(NULL, valid), 0.0f, 1e-6f, "null tensor returns zero");
    SNEPPX_tensor_set_f32(NULL, valid, 1.0f);
    SNEPPX_tensor_set_f32(tensor, NULL, 1.0f);
    SNEPPX_tensor_destroy(tensor);
}

TEST(test_tensor, element_access_bounds_and_null_safety) { test_element_access_bounds_and_null_safety(); }

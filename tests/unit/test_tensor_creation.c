#include "test_gtest.h"
#include "multidimensional_tensor_engine.h"

/*
 * SNEPPX - Test Tensor Creation
 *
 * WHAT
 *   Test Tensor Creation.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_empty_2d(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_empty(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "empty 2d not null");
    SX_ASSERT_EQ(t->ndim, 2, "ndim == 2");
    SX_ASSERT_EQ(t->shape[0], 2, "shape[0] == 2");
    SX_ASSERT_EQ(t->shape[1], 3, "shape[1] == 3");
    SX_ASSERT_EQ(t->size, 6, "size == 6");
    SX_ASSERT_EQ(t->dtype, SNEPPX_FLOAT32, "dtype == float32");
    SX_ASSERT_NOT_NULL(t->data, "data not null");
    SNEPPX_tensor_destroy(t);
}

static void test_empty_3d(void) {
    size_t shape[] = {2, 3, 4};
    SNEPPXTensor* t = SNEPPX_tensor_empty(shape, 3, SNEPPX_FLOAT64);
    SX_ASSERT_NOT_NULL(t, "empty 3d not null");
    SX_ASSERT_EQ(t->ndim, 3, "ndim == 3");
    SX_ASSERT_EQ(t->size, 24, "size == 24");
    SX_ASSERT_EQ(t->dtype, SNEPPX_FLOAT64, "dtype == float64");
    SNEPPX_tensor_destroy(t);
}

static void test_empty_null_shape(void) {
    SNEPPXTensor* t = SNEPPX_tensor_empty(NULL, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(t, "empty NULL shape -> NULL");
}

static void test_empty_zero_ndim(void) {
    SNEPPXTensor* t = SNEPPX_tensor_empty(NULL, 0, SNEPPX_FLOAT32);
    if (t) { SX_ASSERT_EQ(t->ndim, 0, "ndim == 0"); SX_ASSERT_EQ(t->size, 1, "size == 1"); SNEPPX_tensor_destroy(t); }
}

static void test_full_f32(void) {
    size_t shape[] = {2, 3};
    float val = 3.14f;
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, &val);
    SX_ASSERT_NOT_NULL(t, "full f32 not null");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 6; i++) SX_ASSERT_NEAR(d[i], 3.14f, 1e-4f, "full f32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_full_f64(void) {
    size_t shape[] = {3};
    double val = 2.71;
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 1, SNEPPX_FLOAT64, &val);
    SX_ASSERT_NOT_NULL(t, "full f64 not null");
    double* d = (double*)t->data;
    for (size_t i = 0; i < 3; i++) SX_ASSERT_NEAR(d[i], 2.71, 1e-4, "full f64 value");
    SNEPPX_tensor_destroy(t);
}

static void test_full_null_value(void) {
    size_t shape[] = {2, 2};
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, NULL);
    SX_ASSERT_NOT_NULL(t, "full NULL value returns tensor");
    SNEPPX_tensor_destroy(t);
}

static void test_full_int32(void) {
    size_t shape[] = {4};
    int32_t val = 42;
    SNEPPXTensor* t = SNEPPX_tensor_full(shape, 1, SNEPPX_INT32, &val);
    SX_ASSERT_NOT_NULL(t, "full int32 not null");
    int32_t* d = (int32_t*)t->data;
    for (size_t i = 0; i < 4; i++) SX_ASSERT_EQ(d[i], 42, "full int32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_f32(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "arange f32 not null");
    SX_ASSERT_EQ(t->size, 5, "arange size == 5");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 5; i++) SX_ASSERT_NEAR(d[i], (float)i, 1e-4f, "arange f32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_f64(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(1.0f, 10.0f, 3.0f, SNEPPX_FLOAT64);
    SX_ASSERT_NOT_NULL(t, "arange f64 not null");
    double* d = (double*)t->data;
    SX_ASSERT_NEAR(d[0], 1.0, 1e-4, "arange f64 [0]");
    SX_ASSERT_NEAR(d[1], 4.0, 1e-4, "arange f64 [1]");
    SX_ASSERT_NEAR(d[2], 7.0, 1e-4, "arange f64 [2]");
    SX_ASSERT_EQ(t->size, 3, "arange f64 size == 3");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_int32(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_INT32);
    SX_ASSERT_NOT_NULL(t, "arange int32 not null");
    SX_ASSERT_EQ(t->size, 5, "arange int32 size == 5");
    int32_t* d = (int32_t*)t->data;
    for (size_t i = 0; i < 5; i++) SX_ASSERT_EQ(d[i], (int32_t)i, "arange int32 value");
    SNEPPX_tensor_destroy(t);
}

static void test_arange_zero_step(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 0.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(t, "arange zero step -> NULL");
}

static void test_arange_negative_step(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(5.0f, 0.0f, -1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "arange negative step not null");
    SX_ASSERT_EQ(t->size, 5, "arange negative step size == 5");
    float* d = (float*)t->data;
    SX_ASSERT_NEAR(d[0], 5.0f, 1e-4f, "arange negative [0]");
    SX_ASSERT_NEAR(d[4], 1.0f, 1e-4f, "arange negative [4]");
    SNEPPX_tensor_destroy(t);
}

static void test_linspace_f32(void) {
    SNEPPXTensor* t = SNEPPX_tensor_linspace(0.0f, 1.0f, 5, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "linspace f32 not null");
    SX_ASSERT_EQ(t->size, 5, "linspace size == 5");
    float* d = (float*)t->data;
    SX_ASSERT_NEAR(d[0], 0.0f, 1e-4f, "linspace f32 start");
    SX_ASSERT_NEAR(d[4], 1.0f, 1e-4f, "linspace f32 end");
    SX_ASSERT_NEAR(d[2], 0.5f, 1e-4f, "linspace f32 mid");
    SNEPPX_tensor_destroy(t);
}

static void test_linspace_f64(void) {
    SNEPPXTensor* t = SNEPPX_tensor_linspace(1.0f, 3.0f, 3, SNEPPX_FLOAT64);
    SX_ASSERT_NOT_NULL(t, "linspace f64 not null");
    double* d = (double*)t->data;
    SX_ASSERT_NEAR(d[0], 1.0, 1e-4, "linspace f64 [0]");
    SX_ASSERT_NEAR(d[1], 2.0, 1e-4, "linspace f64 [1]");
    SX_ASSERT_NEAR(d[2], 3.0, 1e-4, "linspace f64 [2]");
    SNEPPX_tensor_destroy(t);
}

static void test_linspace_zero_steps(void) {
    SNEPPXTensor* t = SNEPPX_tensor_linspace(0.0f, 1.0f, 0, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(t, "linspace zero steps -> NULL");
}

static void test_eye_2d(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(3, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "eye 3 not null");
    SX_ASSERT_EQ(t->ndim, 2, "eye ndim == 2");
    SX_ASSERT_EQ(t->shape[0], 3, "eye shape[0] == 3");
    SX_ASSERT_EQ(t->shape[1], 3, "eye shape[1] == 3");
    float* d = (float*)t->data;
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            SX_ASSERT_NEAR(d[i * 3 + j], (i == j) ? 1.0f : 0.0f, 1e-6f, "eye value");
    SNEPPX_tensor_destroy(t);
}

static void test_eye_1(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(1, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "eye 1 not null");
    float* d = (float*)t->data;
    SX_ASSERT_NEAR(d[0], 1.0f, 1e-6f, "eye 1x1 = 1");
    SNEPPX_tensor_destroy(t);
}

static void test_eye_f64(void) {
    SNEPPXTensor* t = SNEPPX_tensor_eye(2, SNEPPX_FLOAT64);
    SX_ASSERT_NOT_NULL(t, "eye f64 not null");
    double* d = (double*)t->data;
    SX_ASSERT_NEAR(d[0], 1.0, 1e-6, "eye f64 [0,0]");
    SX_ASSERT_NEAR(d[1], 0.0, 1e-6, "eye f64 [0,1]");
    SX_ASSERT_NEAR(d[2], 0.0, 1e-6, "eye f64 [1,0]");
    SX_ASSERT_NEAR(d[3], 1.0, 1e-6, "eye f64 [1,1]");
    SNEPPX_tensor_destroy(t);
}


static void test_ones_dtype_correctness(void) {
    size_t shape[] = {3};

    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_INT32);
    SX_ASSERT_NOT_NULL(t, "int32 ones not null");
    for (size_t i = 0; i < 3; ++i) SX_ASSERT_EQ(((int32_t*)t->data)[i], 1, "int32 ones value");
    SNEPPX_tensor_destroy(t);

    t = SNEPPX_tensor_ones(shape, 1, SNEPPX_INT16);
    SX_ASSERT_NOT_NULL(t, "int16 ones not null");
    for (size_t i = 0; i < 3; ++i) SX_ASSERT_EQ(((int16_t*)t->data)[i], 1, "int16 ones value");
    SNEPPX_tensor_destroy(t);

    t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT16);
    SX_ASSERT_NOT_NULL(t, "float16 ones not null");
    for (size_t i = 0; i < 3; ++i) SX_ASSERT_EQ(((uint16_t*)t->data)[i], UINT16_C(0x3c00), "float16 one encoding");
    SNEPPX_tensor_destroy(t);

    t = SNEPPX_tensor_ones(shape, 1, SNEPPX_BFLOAT16);
    SX_ASSERT_NOT_NULL(t, "bfloat16 ones not null");
    for (size_t i = 0; i < 3; ++i) SX_ASSERT_EQ(((uint16_t*)t->data)[i], UINT16_C(0x3f80), "bfloat16 one encoding");
    SNEPPX_tensor_destroy(t);

    t = SNEPPX_tensor_ones(shape, 1, SNEPPX_COMPLEX64);
    SX_ASSERT_NOT_NULL(t, "complex64 ones not null");
    for (size_t i = 0; i < 3; ++i) {
        SX_ASSERT_NEAR(((float*)t->data)[2 * i], 1.0f, 1e-6f, "complex64 real part");
        SX_ASSERT_NEAR(((float*)t->data)[2 * i + 1], 0.0f, 1e-6f, "complex64 imaginary part");
    }
    SNEPPX_tensor_destroy(t);

    t = SNEPPX_tensor_ones(shape, 1, SNEPPX_COMPLEX128);
    SX_ASSERT_NOT_NULL(t, "complex128 ones not null");
    for (size_t i = 0; i < 3; ++i) {
        SX_ASSERT_NEAR(((double*)t->data)[2 * i], 1.0, 1e-12, "complex128 real part");
        SX_ASSERT_NEAR(((double*)t->data)[2 * i + 1], 0.0, 1e-12, "complex128 imaginary part");
    }
    SNEPPX_tensor_destroy(t);

    t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT8);
    SX_ASSERT_NULL(t, "float8 ones rejected until encoding is specified");
}

TEST(test_tensor_creation, empty_2d) { test_empty_2d(); }
TEST(test_tensor_creation, ones_dtype_correctness) { test_ones_dtype_correctness(); }
static void test_create_rejects_invalid_dtype_and_size_overflow(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, (SNEPPXDtype)999);
    SX_ASSERT_NULL(t, "unknown dtype rejected");

    size_t overflowing_shape[] = {SIZE_MAX, 2};
    t = SNEPPX_tensor_create(overflowing_shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(t, "shape element count overflow rejected");

    size_t byte_overflow_shape[] = {SIZE_MAX / sizeof(double) + 1};
    t = SNEPPX_tensor_create(byte_overflow_shape, 1, SNEPPX_FLOAT64);
    SX_ASSERT_NULL(t, "tensor byte size overflow rejected");
}

TEST(test_tensor_creation, reject_invalid_dtype_and_overflow) { test_create_rejects_invalid_dtype_and_size_overflow(); }
TEST(test_tensor_creation, empty_3d) { test_empty_3d(); }
TEST(test_tensor_creation, empty_NULL_shape) { test_empty_null_shape(); }
TEST(test_tensor_creation, empty_zero_ndim) { test_empty_zero_ndim(); }
TEST(test_tensor_creation, full_f32) { test_full_f32(); }
TEST(test_tensor_creation, full_f64) { test_full_f64(); }
TEST(test_tensor_creation, full_NULL_value) { test_full_null_value(); }
TEST(test_tensor_creation, full_int32) { test_full_int32(); }
TEST(test_tensor_creation, arange_f32) { test_arange_f32(); }
TEST(test_tensor_creation, arange_f64) { test_arange_f64(); }
TEST(test_tensor_creation, arange_int32) { test_arange_int32(); }
TEST(test_tensor_creation, arange_zero_step) { test_arange_zero_step(); }
TEST(test_tensor_creation, arange_negative_step) { test_arange_negative_step(); }
TEST(test_tensor_creation, linspace_f32) { test_linspace_f32(); }
TEST(test_tensor_creation, linspace_f64) { test_linspace_f64(); }
TEST(test_tensor_creation, linspace_zero_steps) { test_linspace_zero_steps(); }
TEST(test_tensor_creation, eye_3x3) { test_eye_2d(); }
TEST(test_tensor_creation, eye_1x1) { test_eye_1(); }
TEST(test_tensor_creation, eye_f64) { test_eye_f64(); }

#include "test_gtest.h"
#include "multidimensional_tensor_engine.h"

/*
 * SNEPPX - Test Tensor Edge
 *
 * WHAT
 *   Test Tensor Edge.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_create_0d(void) {
    size_t shape[] = {1};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "0d tensor");
    SX_ASSERT_EQ(t->size, 1, "size == 1");
    SX_ASSERT_EQ(t->ndim, 1, "ndim == 1");
    SNEPPX_tensor_destroy(t);
}

static void test_create_zero_dim(void) {
    SNEPPXTensor* t = SNEPPX_tensor_create(NULL, 0, SNEPPX_FLOAT32);
    if (t) { SX_ASSERT_EQ(t->ndim, 0, "ndim == 0"); SX_ASSERT_EQ(t->size, 1, "size == 1"); SNEPPX_tensor_destroy(t); }
}

static void test_create_large_ndim(void) {
    size_t shape[16];
    for (size_t i = 0; i < 16; i++) shape[i] = 1;
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 16, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "16-dim tensor");
    SX_ASSERT_EQ(t->ndim, 16, "ndim == 16");
    SNEPPX_tensor_destroy(t);
}

static void test_destroy_null(void) {
    SNEPPX_tensor_destroy(NULL);
}

static void test_add_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_add(a, NULL);
    SX_ASSERT_NULL(r, "add NULL b -> NULL");
    r = SNEPPX_tensor_add(NULL, a);
    SX_ASSERT_NULL(r, "add NULL a -> NULL");
    r = SNEPPX_tensor_add(NULL, NULL);
    SX_ASSERT_NULL(r, "add NULL NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_sub_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_sub(a, NULL);
    SX_ASSERT_NULL(r, "sub NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_mul_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_mul(a, NULL);
    SX_ASSERT_NULL(r, "mul NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_div_null(void) {
    size_t sh[] = {2, 2};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_div(a, NULL);
    SX_ASSERT_NULL(r, "div NULL -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_matmul_null(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_matmul(a, NULL);
    SX_ASSERT_NULL(r, "matmul NULL -> NULL");
    r = SNEPPX_tensor_matmul(NULL, a);
    SX_ASSERT_NULL(r, "matmul NULL a -> NULL");
    SNEPPX_tensor_destroy(a);
}

static void test_matmul_mismatched_inner(void) {
    size_t sha[] = {2, 3};
    size_t shb[] = {4, 5};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sha, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(shb, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_matmul(a, b);
    SX_ASSERT_NULL(r, "matmul mismatched inner -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_matmul_1d_inputs(void) {
    size_t sh[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_matmul(a, b);
    SX_ASSERT_NULL(r, "matmul 1d -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_slice_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_slice(NULL, 0, 0, 1);
    SX_ASSERT_NULL(r, "slice NULL -> NULL");
}

static void test_slice_out_of_bounds_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 5, 0, 1);
    SX_ASSERT_NULL(r, "slice dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_slice_invalid_start_end(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 0, 2, 1);
    SX_ASSERT_NULL(r, "slice start>=end -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_slice_end_exceeds_shape(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 0, 0, 5);
    SX_ASSERT_NULL(r, "slice end>shape -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_reshape_null(void) {
    size_t ns[] = {6};
    SNEPPXTensor* r = SNEPPX_tensor_reshape(NULL, ns, 1);
    SX_ASSERT_NULL(r, "reshape NULL -> NULL");
}

static void test_reshape_invalid_total(void) {
    size_t sh[] = {2, 3};
    size_t ns[] = {5};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_reshape(t, ns, 1);
    SX_ASSERT_NOT_NULL(r, "reshape mismatched still returns data");
    if (r) {
        SX_ASSERT_EQ(r->size, 5, "size is 5 despite mismatch");
        SNEPPX_tensor_destroy(r);
    }
    SNEPPX_tensor_destroy(t);
}

static void test_permute_null(void) {
    size_t axes[] = {1, 0};
    SNEPPXTensor* r = SNEPPX_tensor_permute(NULL, axes);
    SX_ASSERT_NULL(r, "permute NULL -> NULL");
}

static void test_transpose_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_transpose(NULL, 0, 1);
    SX_ASSERT_NULL(r, "transpose NULL -> NULL");
}

static void test_transpose_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_transpose(t, 0, 5);
    SX_ASSERT_NULL(r, "transpose dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_expand_null(void) {
    size_t ns[] = {2, 3};
    SNEPPXTensor* r = SNEPPX_tensor_expand(NULL, ns, 2);
    SX_ASSERT_NULL(r, "expand NULL -> NULL");
}

static void test_expand_reduce_ndim(void) {
    size_t sh[] = {2, 3};
    size_t ns[] = {6};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_expand(t, ns, 1);
    SX_ASSERT_NULL(r, "expand fewer dims -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_unsqueeze_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_unsqueeze(NULL, 0);
    SX_ASSERT_NULL(r, "unsqueeze NULL -> NULL");
}

static void test_unsqueeze_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_unsqueeze(t, 5);
    SX_ASSERT_NULL(r, "unsqueeze dim>ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_squeeze_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(NULL, 0);
    SX_ASSERT_NULL(r, "squeeze NULL -> NULL");
}

static void test_squeeze_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(t, 5);
    SX_ASSERT_NOT_NULL(r, "squeeze dim>=ndim returns copy");
    SNEPPX_tensor_destroy(t);
    if (r) SNEPPX_tensor_destroy(r);
}

static void test_concat_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_concat(NULL, 0, 0);
    SX_ASSERT_NULL(r, "concat NULL -> NULL");
}

static void test_concat_single(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    const SNEPPXTensor* arr[] = {t};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 1, 0);
    SX_ASSERT_NOT_NULL(r, "concat single tensor");
    SX_ASSERT_EQ(r->shape[0], 2, "dim0 unchanged");
    SX_ASSERT_EQ(r->shape[1], 3, "dim1 unchanged");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_concat_mismatched_ndim(void) {
    size_t sh1[] = {2, 3};
    size_t sh2[] = {6};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 1, SNEPPX_FLOAT32);
    const SNEPPXTensor* arr[] = {a, b};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 2, 0);
    SX_ASSERT_NULL(r, "concat mismatched ndim -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_concat_mismatched_shape(void) {
    size_t sh1[] = {2, 3};
    size_t sh2[] = {2, 4};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);
    const SNEPPXTensor* arr[] = {a, b};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 2, 0);
    SX_ASSERT_NULL(r, "concat mismatched non-concat dim -> NULL");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_sum_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_sum(NULL, 0);
    SX_ASSERT_NULL(r, "sum NULL -> NULL");
}

static void test_sum_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_sum(t, 5);
    SX_ASSERT_NULL(r, "sum dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_mean_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_mean(NULL, 0);
    SX_ASSERT_NULL(r, "mean NULL -> NULL");
}

static void test_mean_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_mean(t, 5);
    SX_ASSERT_NULL(r, "mean dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_min_max_empty(void) {
    size_t sh[] = {0};
    SNEPPXTensor* t = SNEPPX_tensor_create(sh, 1, SNEPPX_FLOAT32);
    if (t) {
        float v = SNEPPX_tensor_min(t);
        SX_ASSERT_EQ(v, 0.0f, "min of empty -> 0");
        v = SNEPPX_tensor_max(t);
        SX_ASSERT_EQ(v, 0.0f, "max of empty -> 0");
        size_t idx = SNEPPX_tensor_argmin(t);
        SX_ASSERT_EQ(idx, 0, "argmin of empty -> 0");
        idx = SNEPPX_tensor_argmax(t);
        SX_ASSERT_EQ(idx, 0, "argmax of empty -> 0");
        SNEPPX_tensor_destroy(t);
    }
}

static void test_min_max_null(void) {
    float v = SNEPPX_tensor_min(NULL);
    SX_ASSERT_EQ(v, 0.0f, "min NULL -> 0");
    v = SNEPPX_tensor_max(NULL);
    SX_ASSERT_EQ(v, 0.0f, "max NULL -> 0");
    size_t idx = SNEPPX_tensor_argmin(NULL);
    SX_ASSERT_EQ(idx, 0, "argmin NULL -> 0");
    idx = SNEPPX_tensor_argmax(NULL);
    SX_ASSERT_EQ(idx, 0, "argmax NULL -> 0");
}

static void test_dot_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    float v = SNEPPX_tensor_dot(a, NULL);
    SX_ASSERT_EQ(v, 0.0f, "dot with NULL -> 0");
    v = SNEPPX_tensor_dot(NULL, a);
    SX_ASSERT_EQ(v, 0.0f, "dot NULL -> 0");
    SNEPPX_tensor_destroy(a);
}

static void test_dot_mismatched(void) {
    size_t sh1[] = {3};
    size_t sh2[] = {4};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 1, SNEPPX_FLOAT32);
    float v = SNEPPX_tensor_dot(a, b);
    SX_ASSERT_EQ(v, 0.0f, "dot mismatched -> 0");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_cast_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_cast(NULL, SNEPPX_FLOAT64);
    SX_ASSERT_NULL(r, "cast NULL -> NULL");
}

static void test_copy_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_copy(NULL);
    SX_ASSERT_NULL(r, "copy NULL -> NULL");
}

static void test_clone_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_clone(NULL);
    SX_ASSERT_NULL(r, "clone NULL -> NULL");
}

static void test_split_null(void) {
    SNEPPXTensor** r = SNEPPX_tensor_split(NULL, 2, 0);
    SX_ASSERT_NULL(r, "split NULL -> NULL");
}

static void test_split_zero_splits(void) {
    size_t sh[] = {4, 4};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor** r = SNEPPX_tensor_split(t, 0, 0);
    SX_ASSERT_NULL(r, "split 0 -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_split_uneven(void) {
    size_t sh[] = {4, 4};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor** r = SNEPPX_tensor_split(t, 3, 0);
    SX_ASSERT_NULL(r, "split uneven -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_relu_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_relu(NULL);
    SX_ASSERT_NULL(r, "relu NULL -> NULL");
}

static void test_sigmoid_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_sigmoid(NULL);
    SX_ASSERT_NULL(r, "sigmoid NULL -> NULL");
}

static void test_softmax_null(void) {
    SNEPPXTensor* r = SNEPPX_tensor_softmax(NULL, 0);
    SX_ASSERT_NULL(r, "softmax NULL -> NULL");
}

static void test_layer_norm_null(void) {
    size_t sh[] = {4};
    SNEPPXTensor* g = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_layer_norm(NULL, g, b, 1e-5f);
    SX_ASSERT_NULL(r, "layer_norm NULL input -> NULL");
    SNEPPX_tensor_destroy(g); SNEPPX_tensor_destroy(b);
}

static void test_where_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_where(NULL, t, t);
    SX_ASSERT_NULL(r, "where NULL condition -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_masked_select_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_masked_select(NULL, t);
    SX_ASSERT_NULL(r, "masked_select NULL src -> NULL");
    r = SNEPPX_tensor_masked_select(t, NULL);
    SX_ASSERT_NULL(r, "masked_select NULL mask -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_gather_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    size_t ish[] = {2};
    SNEPPXTensor* idx = SNEPPX_tensor_zeros(ish, 1, SNEPPX_INT32);
    SNEPPXTensor* r = SNEPPX_tensor_gather(NULL, 0, idx);
    SX_ASSERT_NULL(r, "gather NULL src -> NULL");
    r = SNEPPX_tensor_gather(t, 0, NULL);
    SX_ASSERT_NULL(r, "gather NULL idx -> NULL");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(idx);
}

static void test_tile_null(void) {
    size_t reps[] = {2};
    SNEPPXTensor* r = SNEPPX_tensor_tile(NULL, reps, 1);
    SX_ASSERT_NULL(r, "tile NULL -> NULL");
}

static void test_repeat_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_repeat(NULL, 2, 0);
    SX_ASSERT_NULL(r, "repeat NULL -> NULL");
    r = SNEPPX_tensor_repeat(t, 2, 5);
    SX_ASSERT_NULL(r, "repeat dim>=ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_nan_inf_ops(void) {
    size_t sh[] = {4};
    SNEPPXTensor* t = SNEPPX_tensor_create(sh, 1, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "created for nan/inf");
    float* d = (float*)t->data;
    d[0] = NAN; d[1] = INFINITY;
    d[2] = -INFINITY; d[3] = 3.14f;
    SNEPPXTensor* r = SNEPPX_tensor_add(t, t);
    SX_ASSERT_NOT_NULL(r, "add with nan/inf");
    float* rd = (float*)r->data;
    SX_ASSERT(isnan(rd[0]), "nan + nan = nan");
    SX_ASSERT(isinf(rd[1]) && rd[1] > 0, "inf + inf = inf");
    SX_ASSERT(isinf(rd[2]) && rd[2] < 0, "-inf + -inf = -inf");
    SX_ASSERT_NEAR(rd[3], 6.28f, 1e-4f, "normal value preserved");
    SNEPPX_tensor_destroy(r);
    r = SNEPPX_tensor_mul(t, t);
    SX_ASSERT_NOT_NULL(r, "mul with nan/inf");
    rd = (float*)r->data;
    SX_ASSERT(isnan(rd[0]), "nan * nan = nan");
    SX_ASSERT(isinf(rd[1]) && rd[1] > 0, "inf * inf = inf");
    SNEPPX_tensor_destroy(r);
    SNEPPX_tensor_destroy(t);
}

static void test_unary_op_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_neg(NULL), "neg NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_abs(NULL), "abs NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_sign(NULL), "sign NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_floor(NULL), "floor NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_ceil(NULL), "ceil NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_round(NULL), "round NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_trunc(NULL), "trunc NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_exp(NULL), "exp NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_log(NULL), "log NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_sqrt(NULL), "sqrt NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_sin(NULL), "sin NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_cos(NULL), "cos NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_tan(NULL), "tan NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_asin(NULL), "asin NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_acos(NULL), "acos NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_atan(NULL), "atan NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_sinh(NULL), "sinh NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_cosh(NULL), "cosh NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_tanh(NULL), "tanh NULL");
}

static void test_compare_op_null(void) {
    size_t sh[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 1, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(SNEPPX_tensor_eq(NULL, t), "eq NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_eq(t, NULL), "eq NULL b");
    SX_ASSERT_NULL(SNEPPX_tensor_ne(NULL, t), "ne NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_lt(NULL, t), "lt NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_le(NULL, t), "le NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_gt(NULL, t), "gt NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_ge(NULL, t), "ge NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_cumsum_cumprod_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_cumsum(NULL, 0), "cumsum NULL");
    SX_ASSERT_NULL(SNEPPX_tensor_cumprod(NULL, 0), "cumprod NULL");
}

static void test_cumsum_invalid_dim(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(SNEPPX_tensor_cumsum(t, 5), "cumsum bad dim");
    SX_ASSERT_NULL(SNEPPX_tensor_cumprod(t, 5), "cumprod bad dim");
    SNEPPX_tensor_destroy(t);
}


TEST(test_tensor_edge, create_0d) { test_create_0d(); }
TEST(test_tensor_edge, create_zero_dim) { test_create_zero_dim(); }
TEST(test_tensor_edge, create_large_ndim) { test_create_large_ndim(); }
TEST(test_tensor_edge, destroy_null) { test_destroy_null(); }
TEST(test_tensor_edge, add_null) { test_add_null(); }
TEST(test_tensor_edge, sub_null) { test_sub_null(); }
TEST(test_tensor_edge, mul_null) { test_mul_null(); }
TEST(test_tensor_edge, div_null) { test_div_null(); }
TEST(test_tensor_edge, matmul_null) { test_matmul_null(); }
TEST(test_tensor_edge, matmul_mismatched_inner) { test_matmul_mismatched_inner(); }
TEST(test_tensor_edge, matmul_1d_inputs) { test_matmul_1d_inputs(); }
TEST(test_tensor_edge, slice_null) { test_slice_null(); }
TEST(test_tensor_edge, slice_out_of_bounds_dim) { test_slice_out_of_bounds_dim(); }
TEST(test_tensor_edge, slice_invalid_start_end) { test_slice_invalid_start_end(); }
TEST(test_tensor_edge, slice_end_exceeds_shape) { test_slice_end_exceeds_shape(); }
TEST(test_tensor_edge, reshape_null) { test_reshape_null(); }
TEST(test_tensor_edge, permute_null) { test_permute_null(); }
TEST(test_tensor_edge, transpose_null) { test_transpose_null(); }
TEST(test_tensor_edge, transpose_invalid_dim) { test_transpose_invalid_dim(); }
TEST(test_tensor_edge, expand_null) { test_expand_null(); }
TEST(test_tensor_edge, expand_reduce_ndim) { test_expand_reduce_ndim(); }
TEST(test_tensor_edge, unsqueeze_null) { test_unsqueeze_null(); }
TEST(test_tensor_edge, unsqueeze_invalid_dim) { test_unsqueeze_invalid_dim(); }
TEST(test_tensor_edge, squeeze_null) { test_squeeze_null(); }
TEST(test_tensor_edge, squeeze_invalid_dim) { test_squeeze_invalid_dim(); }
TEST(test_tensor_edge, concat_null) { test_concat_null(); }
TEST(test_tensor_edge, concat_single) { test_concat_single(); }
TEST(test_tensor_edge, concat_mismatched_ndim) { test_concat_mismatched_ndim(); }
TEST(test_tensor_edge, concat_mismatched_shape) { test_concat_mismatched_shape(); }
TEST(test_tensor_edge, sum_null) { test_sum_null(); }
TEST(test_tensor_edge, sum_invalid_dim) { test_sum_invalid_dim(); }
TEST(test_tensor_edge, mean_null) { test_mean_null(); }
TEST(test_tensor_edge, mean_invalid_dim) { test_mean_invalid_dim(); }
TEST(test_tensor_edge, min_max_empty) { test_min_max_empty(); }
TEST(test_tensor_edge, min_max_null) { test_min_max_null(); }
TEST(test_tensor_edge, dot_null) { test_dot_null(); }
TEST(test_tensor_edge, dot_mismatched) { test_dot_mismatched(); }
TEST(test_tensor_edge, cast_null) { test_cast_null(); }
TEST(test_tensor_edge, copy_null) { test_copy_null(); }
TEST(test_tensor_edge, clone_null) { test_clone_null(); }
TEST(test_tensor_edge, split_null) { test_split_null(); }
TEST(test_tensor_edge, split_zero) { test_split_zero_splits(); }
TEST(test_tensor_edge, split_uneven) { test_split_uneven(); }
TEST(test_tensor_edge, relu_null) { test_relu_null(); }
TEST(test_tensor_edge, sigmoid_null) { test_sigmoid_null(); }
TEST(test_tensor_edge, softmax_null) { test_softmax_null(); }
TEST(test_tensor_edge, layer_norm_null) { test_layer_norm_null(); }
TEST(test_tensor_edge, where_null) { test_where_null(); }
TEST(test_tensor_edge, masked_select_null) { test_masked_select_null(); }
TEST(test_tensor_edge, gather_null) { test_gather_null(); }
TEST(test_tensor_edge, tile_null) { test_tile_null(); }
TEST(test_tensor_edge, repeat_null_bad_dim) { test_repeat_null(); }
TEST(test_tensor_edge, nan_inf_ops) { test_nan_inf_ops(); }
TEST(test_tensor_edge, unary_op_null) { test_unary_op_null(); }
TEST(test_tensor_edge, compare_op_null) { test_compare_op_null(); }
TEST(test_tensor_edge, cumsum_cumprod_null) { test_cumsum_cumprod_null(); }
TEST(test_tensor_edge, cumsum_cumprod_bad_dim) { test_cumsum_invalid_dim(); }

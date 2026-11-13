#include "test_gtest.h"
#include "multidimensional_tensor_engine.h"
#include "polymorphic_memory_allocator.h"

/*
 * SNEPPX - Test Tensor Shape
 *
 * WHAT
 *   Test Tensor Shape.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_copy_independent(void) {
    SNEPPXTensor* a = SNEPPX_tensor_arange(0.0f, 6.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(a, "copy src not null");
    SNEPPXTensor* b = SNEPPX_tensor_copy(a);
    SX_ASSERT_NOT_NULL(b, "copy dst not null");
    ((float*)a->data)[0] = 999.0f;
    SX_ASSERT_NEAR(((float*)b->data)[0], 0.0f, 1e-6f, "copy independent");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void test_copy_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_copy(NULL), "copy NULL -> NULL");
}

static void test_clone_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_clone(NULL), "clone NULL -> NULL");
}

static void test_reshape_2d_to_1d(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 6.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "reshape src not null");
    size_t ns[] = {6};
    SNEPPXTensor* r = SNEPPX_tensor_reshape(t, ns, 1);
    SX_ASSERT_NOT_NULL(r, "reshape not null");
    SX_ASSERT_EQ(r->ndim, 1, "reshape ndim == 1");
    SX_ASSERT_EQ(r->size, 6, "reshape size == 6");
    float* d = (float*)r->data;
    for (size_t i = 0; i < 6; i++) SX_ASSERT_NEAR(d[i], (float)i, 1e-6f, "reshape value");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_reshape_auto_dim(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 24.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "reshape auto src not null");
    size_t ns[] = {2, (size_t)-1};
    SNEPPXTensor* r = SNEPPX_tensor_reshape(t, ns, 2);
    SX_ASSERT_NOT_NULL(r, "reshape auto not null");
    SX_ASSERT_EQ(r->shape[1], 12, "reshape auto dim == 12");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_reshape_null(void) {
    size_t ns[] = {6};
    SX_ASSERT_NULL(SNEPPX_tensor_reshape(NULL, ns, 1), "reshape NULL -> NULL");
}

static void test_permute_2d(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "permute src not null");
    size_t axes[] = {1, 0};
    SNEPPXTensor* r = SNEPPX_tensor_permute(t, axes);
    SX_ASSERT_NOT_NULL(r, "permute not null");
    SX_ASSERT_EQ(r->shape[0], 3, "permute shape[0] == 3");
    SX_ASSERT_EQ(r->shape[1], 2, "permute shape[1] == 2");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_permute_null(void) {
    size_t axes[] = {1, 0};
    SX_ASSERT_NULL(SNEPPX_tensor_permute(NULL, axes), "permute NULL -> NULL");
}

static void test_expand_broadcast(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 3.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "expand src not null");
    size_t ns[] = {2, 3};
    SNEPPXTensor* r = SNEPPX_tensor_expand(t, ns, 2);
    SX_ASSERT_NOT_NULL(r, "expand not null");
    SX_ASSERT_EQ(r->shape[0], 2, "expand shape[0] == 2");
    SX_ASSERT_EQ(r->shape[1], 3, "expand shape[1] == 3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_expand_null(void) {
    size_t ns[] = {2, 3};
    SX_ASSERT_NULL(SNEPPX_tensor_expand(NULL, ns, 2), "expand NULL -> NULL");
}

static void test_expand_fewer_dims(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    size_t ns[] = {6};
    SX_ASSERT_NULL(SNEPPX_tensor_expand(t, ns, 1), "expand fewer dims -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_squeeze_specific(void) {
    size_t sh_[] = {2, 1, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh_, 3, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "squeeze src not null");
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(t, 1);
    SX_ASSERT_NOT_NULL(r, "squeeze not null");
    SX_ASSERT_EQ(r->ndim, 2, "squeeze ndim == 2");
    SX_ASSERT_EQ(r->shape[0], 2, "squeeze shape[0] == 2");
    SX_ASSERT_EQ(r->shape[1], 3, "squeeze shape[1] == 3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_squeeze_not_one(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(t, 0);
    SX_ASSERT_NOT_NULL(r, "squeeze non-1 dim returns copy");
    SX_ASSERT_EQ(r->ndim, 2, "squeeze non-1 ndim unchanged");
    SNEPPX_tensor_destroy(t); if (r) SNEPPX_tensor_destroy(r);
}

static void test_squeeze_invalid_dim(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* r = SNEPPX_tensor_squeeze(t, 5);
    SX_ASSERT_NOT_NULL(r, "squeeze invalid dim returns copy");
    SNEPPX_tensor_destroy(t); if (r) SNEPPX_tensor_destroy(r);
}

static void test_unsqueeze_front(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 3.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "unsqueeze src not null");
    SNEPPXTensor* r = SNEPPX_tensor_unsqueeze(t, 0);
    SX_ASSERT_NOT_NULL(r, "unsqueeze not null");
    SX_ASSERT_EQ(r->ndim, 2, "unsqueeze ndim == 2");
    SX_ASSERT_EQ(r->shape[0], 1, "unsqueeze shape[0] == 1");
    SX_ASSERT_EQ(r->shape[1], 3, "unsqueeze shape[1] == 3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_unsqueeze_invalid_dim(void) {
    size_t shape[] = {3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SX_ASSERT_NULL(SNEPPX_tensor_unsqueeze(t, 5), "unsqueeze dim>ndim -> NULL");
    SNEPPX_tensor_destroy(t);
}

static void test_concat_dim0(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    const SNEPPXTensor* arr[] = {a, b};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 2, 0);
    SX_ASSERT_NOT_NULL(r, "concat dim0 not null");
    SX_ASSERT_EQ(r->shape[0], 4, "concat dim0 shape[0] == 4");
    SX_ASSERT_EQ(r->shape[1], 3, "concat dim0 shape[1] == 3");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(r);
}

static void test_concat_dim1(void) {
    size_t sh[] = {2, 3};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    const SNEPPXTensor* arr[] = {a, b};
    SNEPPXTensor* r = SNEPPX_tensor_concat(arr, 2, 1);
    SX_ASSERT_NOT_NULL(r, "concat dim1 not null");
    SX_ASSERT_EQ(r->shape[0], 2, "concat dim1 shape[0] == 2");
    SX_ASSERT_EQ(r->shape[1], 6, "concat dim1 shape[1] == 6");
    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b); SNEPPX_tensor_destroy(r);
}

static void test_concat_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_concat(NULL, 0, 0), "concat NULL -> NULL");
}

static void test_split_equal(void) {
    size_t sh[] = {4, 2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor** parts = SNEPPX_tensor_split(t, 2, 0);
    SX_ASSERT_NOT_NULL(parts, "split not null");
    SX_ASSERT_EQ(parts[0]->shape[0], 2, "split part0 shape[0] == 2");
    SX_ASSERT_EQ(parts[1]->shape[0], 2, "split part1 shape[0] == 2");
    SNEPPX_tensor_destroy(t);
    SNEPPX_tensor_destroy(parts[0]); SNEPPX_tensor_destroy(parts[1]);
    SNEPPX_free(parts, 2 * sizeof(SNEPPXTensor*));
}

static void test_split_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_split(NULL, 2, 0), "split NULL -> NULL");
}

static void test_slice_basic(void) {
    size_t sh[] = {4, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "slice src not null");
    SNEPPXTensor* r = SNEPPX_tensor_slice(t, 0, 1, 3);
    SX_ASSERT_NOT_NULL(r, "slice not null");
    SX_ASSERT_EQ(r->shape[0], 2, "slice shape[0] == 2");
    SX_ASSERT_EQ(r->shape[1], 3, "slice shape[1] == 3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_slice_null(void) {
    SX_ASSERT_NULL(SNEPPX_tensor_slice(NULL, 0, 0, 1), "slice NULL -> NULL");
}

static void test_masked_select_basic(void) {
    size_t sh_[] = {5};
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "masked select src not null");
    uint8_t mask_data[] = {1, 0, 1, 0, 1};
    SNEPPXTensor* mask = SNEPPX_tensor_empty(sh_, 1, SNEPPX_BOOL);
    SX_ASSERT_NOT_NULL(mask, "mask not null");
    memcpy(mask->data, mask_data, 5);
    SNEPPXTensor* r = SNEPPX_tensor_masked_select(t, mask);
    SX_ASSERT_NOT_NULL(r, "masked select result not null");
    SX_ASSERT_EQ(r->size, 3, "masked select size == 3");
    float* d = (float*)r->data;
    SX_ASSERT_NEAR(d[0], 0.0f, 1e-6f, "masked select [0]");
    SX_ASSERT_NEAR(d[1], 2.0f, 1e-6f, "masked select [1]");
    SX_ASSERT_NEAR(d[2], 4.0f, 1e-6f, "masked select [2]");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(mask); SNEPPX_tensor_destroy(r);
}

static void test_masked_fill_basic(void) {
    size_t sh_[] = {5};
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 5.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "masked fill src not null");
    uint8_t mask_data[] = {1, 0, 0, 0, 1};
    SNEPPXTensor* mask = SNEPPX_tensor_empty(sh_, 1, SNEPPX_BOOL);
    SX_ASSERT_NOT_NULL(mask, "mask not null");
    memcpy(mask->data, mask_data, 5);
    float fill_val = -1.0f;
    SNEPPXTensor* r = SNEPPX_tensor_masked_fill(t, mask, &fill_val);
    SX_ASSERT_NOT_NULL(r, "masked fill result not null");
    float* d = (float*)r->data;
    SX_ASSERT_NEAR(d[0], -1.0f, 1e-6f, "masked fill [0]");
    SX_ASSERT_NEAR(d[1], 1.0f, 1e-6f, "masked fill [1]");
    SX_ASSERT_NEAR(d[4], -1.0f, 1e-6f, "masked fill [4]");
    SX_ASSERT_EQ(r, t, "masked fill returns src");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(mask);
}

static void test_where_basic(void) {
    size_t sh_[] = {4};
    SNEPPXTensor* cond = SNEPPX_tensor_empty(sh_, 1, SNEPPX_BOOL);
    uint8_t cond_data[] = {1, 0, 1, 0};
    memcpy(cond->data, cond_data, 4);
    float xv[] = {10, 20, 30, 40};
    float yv[] = {-1, -2, -3, -4};
    SNEPPXTensor* x = SNEPPX_tensor_empty(sh_, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* y = SNEPPX_tensor_empty(sh_, 1, SNEPPX_FLOAT32);
    memcpy(x->data, xv, 4 * sizeof(float));
    memcpy(y->data, yv, 4 * sizeof(float));
    SNEPPXTensor* r = SNEPPX_tensor_where(cond, x, y);
    SX_ASSERT_NOT_NULL(r, "where not null");
    float* d = (float*)r->data;
    SX_ASSERT_NEAR(d[0], 10.0f, 1e-6f, "where [0]");
    SX_ASSERT_NEAR(d[1], -2.0f, 1e-6f, "where [1]");
    SX_ASSERT_NEAR(d[2], 30.0f, 1e-6f, "where [2]");
    SX_ASSERT_NEAR(d[3], -4.0f, 1e-6f, "where [3]");
    SNEPPX_tensor_destroy(cond); SNEPPX_tensor_destroy(x); SNEPPX_tensor_destroy(y); SNEPPX_tensor_destroy(r);
}

static void test_gather_basic(void) {
    size_t sh_[] = {3, 3};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh_, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "gather src not null");
    size_t idx_sh[] = {2, 3};
    int32_t idx_data[] = {0, 1, 2, 2, 1, 0};
    SNEPPXTensor* idx = SNEPPX_tensor_empty(idx_sh, 2, SNEPPX_INT32);
    memcpy(idx->data, idx_data, 6 * sizeof(int32_t));
    SNEPPXTensor* r = SNEPPX_tensor_gather(t, 0, idx);
    SX_ASSERT_NOT_NULL(r, "gather not null");
    SX_ASSERT_EQ(r->shape[0], 2, "gather shape[0] == 2");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(idx); SNEPPX_tensor_destroy(r);
}

static void test_tile_basic(void) {
    size_t sh_[] = {1, 2};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh_, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "tile src not null");
    size_t reps[] = {2, 2};
    SNEPPXTensor* r = SNEPPX_tensor_tile(t, reps, 2);
    SX_ASSERT_NOT_NULL(r, "tile not null");
    SX_ASSERT_EQ(r->shape[0], 2, "tile shape[0] == 2");
    SX_ASSERT_EQ(r->shape[1], 4, "tile shape[1] == 4");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_repeat_basic(void) {
    SNEPPXTensor* t = SNEPPX_tensor_arange(0.0f, 3.0f, 1.0f, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(t, "repeat src not null");
    SNEPPXTensor* r = SNEPPX_tensor_repeat(t, 2, 0);
    SX_ASSERT_NOT_NULL(r, "repeat not null");
    SX_ASSERT_EQ(r->shape[0], 6, "repeat shape[0] == 6");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}


TEST(test_tensor_shape, copy_independent) { test_copy_independent(); }
TEST(test_tensor_shape, copy_NULL) { test_copy_null(); }
TEST(test_tensor_shape, clone_NULL) { test_clone_null(); }
TEST(test_tensor_shape, reshape_2d_1d) { test_reshape_2d_to_1d(); }
TEST(test_tensor_shape, reshape_auto_dim) { test_reshape_auto_dim(); }
TEST(test_tensor_shape, reshape_NULL) { test_reshape_null(); }
TEST(test_tensor_shape, permute_2d) { test_permute_2d(); }
TEST(test_tensor_shape, permute_NULL) { test_permute_null(); }
TEST(test_tensor_shape, expand_broadcast) { test_expand_broadcast(); }
TEST(test_tensor_shape, expand_NULL) { test_expand_null(); }
TEST(test_tensor_shape, expand_fewer_dims) { test_expand_fewer_dims(); }
TEST(test_tensor_shape, squeeze_specific) { test_squeeze_specific(); }
TEST(test_tensor_shape, squeeze_not_one) { test_squeeze_not_one(); }
TEST(test_tensor_shape, squeeze_invalid_dim) { test_squeeze_invalid_dim(); }
TEST(test_tensor_shape, unsqueeze_front) { test_unsqueeze_front(); }
TEST(test_tensor_shape, unsqueeze_invalid_dim) { test_unsqueeze_invalid_dim(); }
TEST(test_tensor_shape, concat_dim0) { test_concat_dim0(); }
TEST(test_tensor_shape, concat_dim1) { test_concat_dim1(); }
TEST(test_tensor_shape, concat_NULL) { test_concat_null(); }
TEST(test_tensor_shape, split_equal) { test_split_equal(); }
TEST(test_tensor_shape, split_NULL) { test_split_null(); }
TEST(test_tensor_shape, slice_basic) { test_slice_basic(); }
TEST(test_tensor_shape, slice_NULL) { test_slice_null(); }
TEST(test_tensor_shape, masked_select_basic) { test_masked_select_basic(); }
TEST(test_tensor_shape, masked_fill_basic) { test_masked_fill_basic(); }
TEST(test_tensor_shape, where_basic) { test_where_basic(); }
TEST(test_tensor_shape, gather_basic) { test_gather_basic(); }
TEST(test_tensor_shape, tile_basic) { test_tile_basic(); }
TEST(test_tensor_shape, repeat_basic) { test_repeat_basic(); }
static void test_as_strided_bounds_and_aliasing(void) {
    size_t shape[] = {2, 3};
    SNEPPXTensor* source = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(source, "as_strided source");
    size_t view_shape[] = {2, 2};
    size_t view_strides[] = {3, 1};
    SNEPPXTensor* view = SNEPPX_tensor_as_strided(source, 0, view_shape, 2, view_strides);
    SX_ASSERT_NOT_NULL(view, "valid strided view");
    size_t index[] = {1, 1};
    SNEPPX_tensor_set_f32(view, index, 9.0f);
    SX_ASSERT_NEAR(((float*)source->data)[4], 9.0f, 1e-6f, "view aliases source storage");
    SNEPPX_tensor_destroy(view);

    size_t invalid_strides[] = {3, 3};
    view = SNEPPX_tensor_as_strided(source, 0, view_shape, 2, invalid_strides);
    SX_ASSERT_NULL(view, "out-of-bounds strided view rejected");

    size_t empty_shape[] = {2, 0};
    size_t empty_strides[] = {3, 1};
    view = SNEPPX_tensor_as_strided(source, 0, empty_shape, 2, empty_strides);
    SX_ASSERT_NOT_NULL(view, "empty strided view is valid");
    SX_ASSERT_EQ(view->size, 0, "empty view has zero elements");
    SNEPPX_tensor_destroy(view);
    SNEPPX_tensor_destroy(source);
}

static void test_narrow_rank_above_sixteen(void) {
    size_t shape[17];
    for (size_t i = 0; i < 17; ++i) shape[i] = 1;
    shape[16] = 3;
    SNEPPXTensor* source = SNEPPX_tensor_ones(shape, 17, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(source, "17D source");
    SNEPPXTensor* view = SNEPPX_tensor_narrow(source, 16, 1, 2);
    SX_ASSERT_NOT_NULL(view, "17D narrow view");
    SX_ASSERT_EQ(view->ndim, 17, "narrow preserves rank");
    SX_ASSERT_EQ(view->shape[16], 2, "narrow updates selected dimension");
    SX_ASSERT_EQ(view->size, 2, "narrow computes view size");
    SNEPPX_tensor_destroy(view);
    SNEPPX_tensor_destroy(source);
}

TEST(test_tensor_shape, as_strided_bounds_and_aliasing) { test_as_strided_bounds_and_aliasing(); }
TEST(test_tensor_shape, narrow_rank_above_sixteen) { test_narrow_rank_above_sixteen(); }

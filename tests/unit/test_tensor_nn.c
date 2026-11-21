#include "multidimensional_tensor_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

/*
 * SNEPPX - Test Tensor Nn
 *
 * WHAT
 *   Test Tensor Nn.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */






/* ---------- Activation Functions ---------- */

static void test_softmax_basic(void) {
    size_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 6 * sizeof(float));
    SNEPPXTensor* sm = SNEPPX_tensor_softmax(t, 1);
    SX_ASSERT(sm != NULL, "softmax result not null");
    float* d = (float*)sm->data;
    float sum_row0 = expf(1.0f) + expf(2.0f) + expf(3.0f);
    SX_ASSERT_NEAR(d[0], expf(1.0f) / sum_row0, 1e-5f, "softmax[0]");
    SX_ASSERT_NEAR(d[1], expf(2.0f) / sum_row0, 1e-5f, "softmax[1]");
    SX_ASSERT_NEAR(d[2], expf(3.0f) / sum_row0, 1e-5f, "softmax[2]");
    SX_ASSERT_NEAR(d[0] + d[1] + d[2], 1.0f, 1e-5f, "softmax row sums to 1");
    SX_ASSERT_NEAR(d[3] + d[4] + d[5], 1.0f, 1e-5f, "softmax row1 sums to 1");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(sm);
}

static void test_relu_basic(void) {
    size_t shape[] = {3};
    float data[] = {-2.0f, 0.0f, 3.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 3 * sizeof(float));
    SNEPPXTensor* r = SNEPPX_tensor_relu(t);
    SX_ASSERT(r != NULL, "relu result not null");
    float* d = (float*)r->data;
    SX_ASSERT_NEAR(d[0], 0.0f, 1e-6f, "relu(-2)==0");
    SX_ASSERT_NEAR(d[1], 0.0f, 1e-6f, "relu(0)==0");
    SX_ASSERT_NEAR(d[2], 3.0f, 1e-6f, "relu(3)==3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(r);
}

static void test_sigmoid_basic(void) {
    size_t shape[] = {3};
    float data[] = {-1.0f, 0.0f, 1.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 3 * sizeof(float));
    SNEPPXTensor* s = SNEPPX_tensor_sigmoid(t);
    SX_ASSERT(s != NULL, "sigmoid result not null");
    float* d = (float*)s->data;
    SX_ASSERT_NEAR(d[0], 1.0f / (1.0f + expf(1.0f)), 1e-5f, "sigmoid(-1)");
    SX_ASSERT_NEAR(d[1], 0.5f, 1e-5f, "sigmoid(0)==0.5");
    SX_ASSERT_NEAR(d[2], 1.0f / (1.0f + expf(-1.0f)), 1e-5f, "sigmoid(1)");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(s);
}

static void test_gelu_basic(void) {
    size_t shape[] = {2};
    float data[] = {0.0f, 1.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(t->data, data, 2 * sizeof(float));
    SNEPPXTensor* g = SNEPPX_tensor_gelu(t);
    SX_ASSERT(g != NULL, "gelu result not null");
    SX_ASSERT_NEAR(((float*)g->data)[0], 0.0f, 1e-5f, "gelu(0)==0");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(g);
}

static void test_dropout_basic(void) {
    size_t shape[] = {1000};
    SNEPPXTensor* t = SNEPPX_tensor_ones(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* d = SNEPPX_tensor_dropout(t, 0.5f, 42);
    SX_ASSERT(d != NULL, "dropout result not null");
    float* dd = (float*)d->data;
    size_t zeros = 0;
    for (size_t i = 0; i < 1000; i++) if (dd[i] == 0.0f) zeros++;
    SX_ASSERT(zeros > 300 && zeros < 700, "dropout ~50% zeros");
    float scale = 1.0f / 0.5f;
    for (size_t i = 0; i < 1000; i++)
        if (dd[i] != 0.0f) SX_ASSERT_NEAR(dd[i], scale, 1e-5f, "dropout scales up");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(d);
}

/* ---------- Loss Functions ---------- */

static void test_mse_loss(void) {
    size_t shape[] = {3};
    float pred_data[] = {2.0f, 4.0f, 6.0f};
    float target_data[] = {1.0f, 3.0f, 5.0f};
    SNEPPXTensor* p = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(p->data, pred_data, 3 * sizeof(float));
    memcpy(t->data, target_data, 3 * sizeof(float));
    SNEPPXTensor* l = SNEPPX_tensor_mse_loss(p, t);
    SX_ASSERT(l != NULL, "mse loss not null");
    float mse = ((1.0f + 1.0f + 1.0f) / 3.0f);
    SX_ASSERT_NEAR(((float*)l->data)[0], mse, 1e-5f, "mse loss");
    SNEPPX_tensor_destroy(p); SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(l);
}

static void test_mae_loss(void) {
    size_t shape[] = {3};
    float pred_data[] = {2.0f, 4.0f, 6.0f};
    float target_data[] = {1.0f, 3.0f, 5.0f};
    SNEPPXTensor* p = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 1, SNEPPX_FLOAT32);
    memcpy(p->data, pred_data, 3 * sizeof(float));
    memcpy(t->data, target_data, 3 * sizeof(float));
    SNEPPXTensor* l = SNEPPX_tensor_mae_loss(p, t);
    SX_ASSERT(l != NULL, "mae loss not null");
    SX_ASSERT_NEAR(((float*)l->data)[0], 1.0f, 1e-5f, "mae loss");
    SNEPPX_tensor_destroy(p); SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(l);
}

/* ---------- Convolution ---------- */

static void test_conv1d_basic(void) {
    size_t input_shape[] = {1, 1, 5};
    float idata[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    SNEPPXTensor* input = SNEPPX_tensor_create(input_shape, 3, SNEPPX_FLOAT32);
    memcpy(input->data, idata, 5 * sizeof(float));
    size_t kernel_shape[] = {1, 1, 3};
    float kdata[] = {1.0f, 0.0f, -1.0f};
    SNEPPXTensor* kernel = SNEPPX_tensor_create(kernel_shape, 3, SNEPPX_FLOAT32);
    memcpy(kernel->data, kdata, 3 * sizeof(float));
    SNEPPXTensor* out = SNEPPX_tensor_conv1d(input, kernel, 1, 0);
    SX_ASSERT(out != NULL, "conv1d result not null");
    SX_ASSERT(out->shape[2] == 3, "conv1d output length 5-3+1==3");
    float* d = (float*)out->data;
    SX_ASSERT_NEAR(d[0], -2.0f, 1e-5f, "conv1d[0]: 1*1+2*0+3*(-1)==-2");
    SX_ASSERT_NEAR(d[1], -2.0f, 1e-5f, "conv1d[1]: 2*1+3*0+4*(-1)==-2");
    SX_ASSERT_NEAR(d[2], -2.0f, 1e-5f, "conv1d[2]: 3*1+4*0+5*(-1)==-2");
    SNEPPX_tensor_destroy(input); SNEPPX_tensor_destroy(kernel); SNEPPX_tensor_destroy(out);
}

static void test_conv2d_basic(void) {
    size_t input_shape[] = {1, 1, 4, 4};
    float idata[16];
    for (size_t i = 0; i < 16; i++) idata[i] = (float)i;
    SNEPPXTensor* input = SNEPPX_tensor_create(input_shape, 4, SNEPPX_FLOAT32);
    memcpy(input->data, idata, 16 * sizeof(float));
    size_t kernel_shape[] = {1, 1, 2, 2};
    float kdata[] = {1.0f, 0.0f, 0.0f, -1.0f};
    SNEPPXTensor* kernel = SNEPPX_tensor_create(kernel_shape, 4, SNEPPX_FLOAT32);
    memcpy(kernel->data, kdata, 4 * sizeof(float));
    SNEPPXTensor* out = SNEPPX_tensor_conv2d(input, kernel, 1, 1, 0, 0);
    SX_ASSERT(out != NULL, "conv2d result not null");
    SX_ASSERT(out->shape[2] == 3 && out->shape[3] == 3, "conv2d output 3x3");
    float* d = (float*)out->data;
    SX_ASSERT_NEAR(d[0], -5.0f, 1e-5f, "conv2d[0,0]: 0-5==-5");
    SNEPPX_tensor_destroy(input); SNEPPX_tensor_destroy(kernel); SNEPPX_tensor_destroy(out);
}

static void test_pool1d_basic(void) {
    size_t shape[] = {1, 1, 6};
    float data[] = {1.0f, 5.0f, 2.0f, 8.0f, 3.0f, 0.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 3, SNEPPX_FLOAT32);
    memcpy(t->data, data, 6 * sizeof(float));
    SNEPPXTensor* p = SNEPPX_tensor_pool1d(t, 2, 2);
    SX_ASSERT(p != NULL, "pool1d result not null");
    SX_ASSERT(p->shape[2] == 3, "pool1d output length 3");
    float* d = (float*)p->data;
    SX_ASSERT_NEAR(d[0], 5.0f, 1e-6f, "pool1d max(1,5)==5");
    SX_ASSERT_NEAR(d[1], 8.0f, 1e-6f, "pool1d max(2,8)==8");
    SX_ASSERT_NEAR(d[2], 3.0f, 1e-6f, "pool1d max(3,0)==3");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(p);
}

static void test_pool2d_basic(void) {
    size_t shape[] = {1, 1, 4, 4};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 9.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 4, SNEPPX_FLOAT32);
    memcpy(t->data, data, 16 * sizeof(float));
    SNEPPXTensor* p = SNEPPX_tensor_pool2d(t, 2, 2, 2, 2);
    SX_ASSERT(p != NULL, "pool2d result not null");
    SX_ASSERT(p->shape[2] == 2 && p->shape[3] == 2, "pool2d output 2x2");
    float* d = (float*)p->data;
    SX_ASSERT_NEAR(d[0], 9.0f, 1e-6f, "pool2d max([1,2,5,9])==9");
    SX_ASSERT_NEAR(d[1], 8.0f, 1e-6f, "pool2d max([3,4,7,8])==8");
    SX_ASSERT_NEAR(d[2], 14.0f, 1e-6f, "pool2d max([9,10,13,14])==14");
    SX_ASSERT_NEAR(d[3], 16.0f, 1e-6f, "pool2d max([11,12,15,16])==16");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(p);
}

/* ---------- Normalization ---------- */

static void test_layer_norm_basic(void) {
    size_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    memcpy(t->data, data, 6 * sizeof(float));
    SNEPPXTensor* ln = SNEPPX_tensor_layer_norm(t, NULL, NULL, 1e-5f);
    SX_ASSERT(ln != NULL, "layer_norm result not null");
    float* d = (float*)ln->data;
    float mean0 = (1.0f + 2.0f + 3.0f) / 3.0f;
    float var0 = ((1.0f-mean0)*(1.0f-mean0) + (2.0f-mean0)*(2.0f-mean0) + (3.0f-mean0)*(3.0f-mean0)) / 3.0f;
    float inv_std0 = 1.0f / sqrtf(var0 + 1e-5f);
    SX_ASSERT_NEAR(d[0], (1.0f - mean0) * inv_std0, 1e-4f, "layer_norm[0]");
    SNEPPX_tensor_destroy(t); SNEPPX_tensor_destroy(ln);
}

static void test_embedding_basic(void) {
    size_t wshape[] = {4, 3};
    float wdata[] = {1.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f,
                     0.0f, 0.0f, 1.0f,
                     1.0f, 1.0f, 1.0f};
    SNEPPXTensor* w = SNEPPX_tensor_create(wshape, 2, SNEPPX_FLOAT32);
    memcpy(w->data, wdata, 12 * sizeof(float));
    size_t idx_shape[] = {2};
    int32_t idx_data[] = {1, 3};
    SNEPPXTensor* idx = SNEPPX_tensor_create(idx_shape, 1, SNEPPX_INT32);
    memcpy(idx->data, idx_data, 2 * sizeof(int32_t));
    SNEPPXTensor* e = SNEPPX_tensor_embedding(w, idx);
    SX_ASSERT(e != NULL, "embedding result not null");
    SX_ASSERT(e->shape[0] == 2 && e->shape[1] == 3, "embedding shape 2x3");
    float* d = (float*)e->data;
    SX_ASSERT_NEAR(d[0], 0.0f, 1e-6f, "embedding[1,0]==0");
    SX_ASSERT_NEAR(d[1], 1.0f, 1e-6f, "embedding[1,1]==1");
    SX_ASSERT_NEAR(d[3], 1.0f, 1e-6f, "embedding[3,0]==1");
    SX_ASSERT_NEAR(d[4], 1.0f, 1e-6f, "embedding[3,1]==1");
    SNEPPX_tensor_destroy(w); SNEPPX_tensor_destroy(idx); SNEPPX_tensor_destroy(e);
}

static void test_null_inputs_nn(void) {
    SX_ASSERT(SNEPPX_tensor_softmax(NULL, 0) == NULL, "softmax null");
    SX_ASSERT(SNEPPX_tensor_relu(NULL) == NULL, "relu null");
    SX_ASSERT(SNEPPX_tensor_sigmoid(NULL) == NULL, "sigmoid null");
    SX_ASSERT(SNEPPX_tensor_conv1d(NULL, NULL, 1, 0) == NULL, "conv1d null");
    SX_ASSERT(SNEPPX_tensor_conv2d(NULL, NULL, 1, 1, 0, 0) == NULL, "conv2d null");
    SX_ASSERT(SNEPPX_tensor_pool1d(NULL, 2, 2) == NULL, "pool1d null");
    SX_ASSERT(SNEPPX_tensor_pool2d(NULL, 2, 2, 2, 2) == NULL, "pool2d null");
    SX_ASSERT(SNEPPX_tensor_mse_loss(NULL, NULL) == NULL, "mse null");
    SX_ASSERT(SNEPPX_tensor_layer_norm(NULL, NULL, NULL, 1e-5f) == NULL, "layer_norm null");
    SX_ASSERT(SNEPPX_tensor_embedding(NULL, NULL) == NULL, "embedding null");
}


TEST(test_tensor_nn, test_softmax_basic) { test_softmax_basic(); }
TEST(test_tensor_nn, test_relu_basic) { test_relu_basic(); }
TEST(test_tensor_nn, test_sigmoid_basic) { test_sigmoid_basic(); }
TEST(test_tensor_nn, test_gelu_basic) { test_gelu_basic(); }
TEST(test_tensor_nn, test_dropout_basic) { test_dropout_basic(); }
TEST(test_tensor_nn, test_mse_loss) { test_mse_loss(); }
TEST(test_tensor_nn, test_mae_loss) { test_mae_loss(); }
TEST(test_tensor_nn, test_conv1d_basic) { test_conv1d_basic(); }
TEST(test_tensor_nn, test_conv2d_basic) { test_conv2d_basic(); }
TEST(test_tensor_nn, test_pool1d_basic) { test_pool1d_basic(); }
TEST(test_tensor_nn, test_pool2d_basic) { test_pool2d_basic(); }
TEST(test_tensor_nn, test_layer_norm_basic) { test_layer_norm_basic(); }
TEST(test_tensor_nn, test_embedding_basic) { test_embedding_basic(); }
TEST(test_tensor_nn, test_null_inputs_nn) { test_null_inputs_nn(); }

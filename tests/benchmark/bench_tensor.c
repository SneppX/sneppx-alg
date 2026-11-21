#include "bench_common.h"
#include "multidimensional_tensor_engine.h"

/*
 * SNEPPX - Bench Tensor
 *
 * WHAT
 *   Bench Tensor.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void bench_create_ops(void) {
    BENCH_INIT(bs);
    printf("  Tensor create:\n");

    size_t sh_small[] = {16, 16};
    BENCH_START(bs, 10000, 100, {
        SNEPPXTensor* t = SNEPPX_tensor_create(sh_small, 2, SNEPPX_FLOAT32);
        SNEPPX_tensor_destroy(t);
    });
    bench_print("create [16,16]", &bs);

    size_t sh_med[] = {256, 256};
    BENCH_START(bs, 1000, 100, {
        SNEPPXTensor* t = SNEPPX_tensor_create(sh_med, 2, SNEPPX_FLOAT32);
        SNEPPX_tensor_destroy(t);
    });
    bench_print("create [256,256]", &bs);

    size_t sh_large[] = {1024, 1024};
    BENCH_START(bs, 100, 10, {
        SNEPPXTensor* t = SNEPPX_tensor_create(sh_large, 2, SNEPPX_FLOAT32);
        SNEPPX_tensor_destroy(t);
    });
    bench_print("create [1024,1024]", &bs);
}

static void bench_fill_ops(void) {
    BENCH_INIT(bs);

    size_t sh_small[] = {256, 256};
    SNEPPXTensor* t = SNEPPX_tensor_create(sh_small, 2, SNEPPX_FLOAT32);
    printf("  Fill [256,256]:\n");

    BENCH_START(bs, 1000, 100, {
        SNEPPX_tensor_fill_f32(t, 1.0f);
    });
    bench_print("fill_f32", &bs);

    SNEPPX_tensor_destroy(t);

    size_t sh_large[] = {1024, 1024};
    t = SNEPPX_tensor_create(sh_large, 2, SNEPPX_FLOAT32);
    printf("  Fill [1024,1024]:\n");

    BENCH_START(bs, 100, 10, {
        SNEPPX_tensor_fill_f32(t, 1.0f);
    });
    bench_print("fill_f32", &bs);

    SNEPPX_tensor_destroy(t);
}

static void bench_elemwise_ops(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Element-wise [256,256]:\n");

    BENCH_START(bs, 1000, 100, {
        SNEPPXTensor* r = SNEPPX_tensor_add(a, b);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("add", &bs);

    BENCH_START(bs, 1000, 100, {
        SNEPPXTensor* r = SNEPPX_tensor_mul(a, b);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("mul", &bs);

    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void bench_matmul_ops(void) {
    BENCH_INIT(bs);
    printf("  Matmul:\n");

    size_t sh1[] = {128, 128};
    size_t sh2[] = {128, 128};
    SNEPPXTensor* a = SNEPPX_tensor_ones(sh1, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* b = SNEPPX_tensor_ones(sh2, 2, SNEPPX_FLOAT32);

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_matmul(a, b);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("matmul [128,128]x[128,128]", &bs);

    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);

    size_t sh3[] = {64, 256};
    size_t sh4[] = {256, 64};
    a = SNEPPX_tensor_ones(sh3, 2, SNEPPX_FLOAT32);
    b = SNEPPX_tensor_ones(sh4, 2, SNEPPX_FLOAT32);

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_matmul(a, b);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("matmul [64,256]x[256,64]", &bs);

    SNEPPX_tensor_destroy(a); SNEPPX_tensor_destroy(b);
}

static void bench_slice_reshape_ops(void) {
    BENCH_INIT(bs);
    size_t sh[] = {128, 128};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Slice/Reshape [128,128]:\n");

    BENCH_START(bs, 1000, 100, {
        SNEPPXTensor* r = SNEPPX_tensor_slice(t, 0, 0, 64);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("slice dim0 0:64", &bs);

    size_t ns[] = {16384};
    BENCH_START(bs, 1000, 100, {
        SNEPPXTensor* r = SNEPPX_tensor_reshape(t, ns, 1);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("reshape [16384]", &bs);

    BENCH_START(bs, 1000, 100, {
        SNEPPXTensor* r = SNEPPX_tensor_transpose(t, 0, 1);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("transpose 0,1", &bs);

    SNEPPX_tensor_destroy(t);
}

static void bench_unary_ops(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Unary [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_relu(t);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("relu", &bs);

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_sigmoid(t);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("sigmoid", &bs);

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_softmax(t, 1);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("softmax dim=1", &bs);

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_sqrt(t);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("sqrt", &bs);

    SNEPPX_tensor_destroy(t);
}

static void bench_reduction_ops(void) {
    BENCH_INIT(bs);
    size_t sh[] = {256, 256};
    SNEPPXTensor* t = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    printf("  Reduction [256,256]:\n");

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_sum(t, 0);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("sum dim=0", &bs);

    BENCH_START(bs, 500, 50, {
        SNEPPXTensor* r = SNEPPX_tensor_mean(t, 1);
        SNEPPX_tensor_destroy(r);
    });
    bench_print("mean dim=1", &bs);

    SNEPPX_tensor_destroy(t);
}

int main(void) {
    printf("=== Tensor Benchmarks ===\n");
    BENCH_RUN("Create ops", bench_create_ops);
    BENCH_RUN("Fill ops", bench_fill_ops);
    BENCH_RUN("Element-wise ops", bench_elemwise_ops);
    BENCH_RUN("Matmul ops", bench_matmul_ops);
    BENCH_RUN("Slice/Reshape ops", bench_slice_reshape_ops);
    BENCH_RUN("Unary ops", bench_unary_ops);
    BENCH_RUN("Reduction ops", bench_reduction_ops);
    BENCH_MAIN();
}

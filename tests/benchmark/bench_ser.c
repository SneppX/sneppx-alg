#include "bench_common.h"
#include "sparse_expert_routing.h"

/*
 * SNEPPX - Bench Ser
 *
 * WHAT
 *   Bench Ser.
 *
 * CONCEPT
 *   Provides the Bench Ser.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void bench_ser_create(void) {
    BENCH_INIT(bs);
    printf("  SER create:\n");

    BENCH_START(bs, 10000, 100, {
        SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
        cfg.num_experts = 8; cfg.input_dim = 16; cfg.expert_dim = 32;
        SNEPPXSERLayer* l = SNEPPX_ser_layer_create(&cfg, 42);
        SNEPPX_ser_layer_destroy(l);
    });
    bench_print("layer_create(8 experts, dim=16)", &bs);

    BENCH_START(bs, 5000, 50, {
        SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
        cfg.num_experts = 32; cfg.input_dim = 64; cfg.expert_dim = 128;
        SNEPPXSERLayer* l = SNEPPX_ser_layer_create(&cfg, 42);
        SNEPPX_ser_layer_destroy(l);
    });
    bench_print("layer_create(32 experts, dim=64)", &bs);
}

static void bench_ser_forward(void) {
    BENCH_INIT(bs);
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 8; cfg.num_active = 2;
    cfg.input_dim = 32; cfg.expert_dim = 64; cfg.output_dim = 32;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);

    size_t sh[] = {64, 32};
    SNEPPXTensor* input = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;

    printf("  SER forward:\n");
    BENCH_START(bs, 500, 50, {
        SNEPPX_ser_forward(layer, input, &output);
        if (output) SNEPPX_tensor_destroy(output);
    });
    bench_print("forward [64,32] 8 experts", &bs);

    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

static void bench_ser_forward_large(void) {
    BENCH_INIT(bs);
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 32; cfg.num_active = 4;
    cfg.input_dim = 128; cfg.expert_dim = 256; cfg.output_dim = 128;
    SNEPPXSERLayer* layer = SNEPPX_ser_layer_create(&cfg, 42);

    size_t sh[] = {128, 128};
    SNEPPXTensor* input = SNEPPX_tensor_ones(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;

    printf("  SER forward large:\n");
    BENCH_START(bs, 100, 10, {
        SNEPPX_ser_forward(layer, input, &output);
        if (output) SNEPPX_tensor_destroy(output);
    });
    bench_print("forward [128,128] 32 experts", &bs);

    SNEPPX_tensor_destroy(input);
    SNEPPX_ser_layer_destroy(layer);
}

int main(void) {
    printf("=== SER Benchmarks ===\n");
    BENCH_RUN("Create", bench_ser_create);
    BENCH_RUN("Forward", bench_ser_forward);
    BENCH_RUN("Forward large", bench_ser_forward_large);
    BENCH_MAIN();
}

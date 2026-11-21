#include "bench_common.h"
#include "neural_programming_engine.h"

/*
 * SNEPPX - Bench Npe
 *
 * WHAT
 *   Bench Npe.
 *
 * CONCEPT
 *   Provides the Bench Npe.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void bench_npe_create(void) {
    BENCH_INIT(bs);
    printf("  NPE create:\n");
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    BENCH_START(bs, 10000, 100, {
        SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
        SNEPPX_npe_vm_destroy(vm);
    });
    bench_print("vm_create", &bs);

    BENCH_START(bs, 10000, 100, {
        SNEPPXNPEProgram* p = SNEPPX_npe_program_create(32);
        SNEPPX_npe_program_destroy(p);
    });
    bench_print("program_create", &bs);
}

static void bench_npe_basic_ops(void) {
    BENCH_INIT(bs);
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction i1 = {SNEPPX_LOAD, 1, 0, 0, 0, {64,64}, {0,0}};
    SNEPPXNPEInstruction i2 = {SNEPPX_RELU, 2, 1, 0, 0, {0,0}, {0,0}};
    SNEPPXNPEInstruction i3 = {SNEPPX_SOFTMAX, 3, 2, 0, 0, {0,0}, {0,0}};
    SNEPPXNPEInstruction i4 = {SNEPPX_HALT, 0, 0, 0, 0, {0,0}, {0,0}};
    SNEPPX_npe_program_append(prog, i1);
    SNEPPX_npe_program_append(prog, i2);
    SNEPPX_npe_program_append(prog, i3);
    SNEPPX_npe_program_append(prog, i4);

    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SNEPPX_npe_vm_load(vm, prog);
    size_t sh_in[] = {64, 64};
    SNEPPXTensor* input = SNEPPX_tensor_ones(sh_in, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;

    printf("  NPE forward [64,64]:\n");
    BENCH_START(bs, 1000, 50, {
        SNEPPX_npe_vm_run(vm, input, &output);
        if (output) SNEPPX_tensor_destroy(output);
    });
    bench_print("relu+softmax", &bs);

    SNEPPX_tensor_destroy(input);
    SNEPPX_npe_vm_destroy(vm);
    SNEPPX_npe_program_destroy(prog);
}

static void bench_npe_matmul(void) {
    BENCH_INIT(bs);
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction i1 = {SNEPPX_LOAD, 1, 0, 0, 0, {128,64}, {0,0}};
    SNEPPXNPEInstruction i2 = {SNEPPX_LOAD, 2, 0, 0, (int)(64*64), {64,128}, {0,0}};
    SNEPPXNPEInstruction i3 = {SNEPPX_MATMUL, 3, 1, 2, 0, {0,0}, {0,0}};
    SNEPPXNPEInstruction i4 = {SNEPPX_HALT, 0, 0, 0, 0, {0,0}, {0,0}};
    SNEPPX_npe_program_append(prog, i1);
    SNEPPX_npe_program_append(prog, i2);
    SNEPPX_npe_program_append(prog, i3);
    SNEPPX_npe_program_append(prog, i4);

    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SNEPPX_npe_vm_load(vm, prog);
    size_t sh_in[] = {128, 64};
    SNEPPXTensor* input = SNEPPX_tensor_ones(sh_in, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;

    printf("  NPE matmul [128,64]x[64,128]:\n");
    BENCH_START(bs, 500, 50, {
        SNEPPX_npe_vm_run(vm, input, &output);
        if (output) SNEPPX_tensor_destroy(output);
    });
    bench_print("matmul", &bs);

    SNEPPX_tensor_destroy(input);
    SNEPPX_npe_vm_destroy(vm);
    SNEPPX_npe_program_destroy(prog);
}

int main(void) {
    printf("=== NPE Benchmarks ===\n");
    BENCH_RUN("Create", bench_npe_create);
    BENCH_RUN("Basic ops", bench_npe_basic_ops);
    BENCH_RUN("Matmul", bench_npe_matmul);
    BENCH_MAIN();
}

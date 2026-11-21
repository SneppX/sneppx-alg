#include "neural_programming_engine.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Npe Demo
 *
 * WHAT
 *   Npe Demo.
 *
 * CONCEPT
 *   Provides the Npe Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


int main(void) {
    printf("=== NPE Demo: MLP Execution ===\n");
    size_t dim = 8, hidden = 16;

    SNEPPXNPEProgram* prog = SNEPPX_npe_compile_mlp(dim, hidden);
    if (!prog) { printf("ERROR: compile mlp\n"); return 1; }
    printf("Program length: %zu instructions\n", prog->num_instructions);

    unsigned long s = 42;
    for (size_t i = 0; i < hidden * dim; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)prog->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
    }
    for (size_t i = hidden * dim; i < hidden * dim + hidden + dim * hidden + dim; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)prog->memory->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.1f;
    }

    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SNEPPX_npe_vm_load(vm, prog);

    size_t shape_in[] = {4, 8};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    if (!input) { printf("ERROR: input\n"); return 1; }

    SNEPPXTensor* output = NULL;
    int r = SNEPPX_npe_vm_run(vm, input, &output);
    printf("Run result: %d (0=ok)\n", r);

    if (output) {
        printf("Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);
        printf("First 8 values: ");
        float* od = (float*)output->data;
        for (size_t i = 0; i < 8 && i < output->size; i++) printf("%f ", od[i]);
        printf("\n");

        int has_nan = 0, has_inf = 0;
        for (size_t i = 0; i < output->size; i++) {
            if (isnan(od[i])) has_nan = 1;
            if (isinf(od[i])) has_inf = 1;
        }
        printf("NaN: %s, Inf: %s\n", has_nan ? "YES" : "NO", has_inf ? "YES" : "NO");
    }

    printf("Execution trace opcodes: ");
    for (size_t i = 0; i < vm->trace_length && i < 15; i++) {
        printf("%d ", vm->execution_trace[i].opcode);
    }
    printf("\n");

    SNEPPX_tensor_destroy(input);
    if (output) SNEPPX_tensor_destroy(output);
    SNEPPX_npe_vm_destroy(vm);
    SNEPPX_npe_program_destroy(prog);
    printf("Demo complete.\n");
    return 0;
}

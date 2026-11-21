#include "fractal_memory_orchestrator.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Fm Demo
 *
 * WHAT
 *   Fm Demo.
 *
 * CONCEPT
 *   Provides the Fm Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void print_bank_summary(const char* label, SNEPPXFMMemoryBank* bank) {
    printf("  %s: %zu/%zu entries", label, bank->num_entries, bank->max_entries);
    if (bank->num_entries > 0) {
        float* vd = (float*)bank->values->data;
        printf(", first val=%.4f", vd[0]);
    }
    printf("\n");
}

int main(void) {
    printf("=== FM Demo: Federated Memory Sync ===\n");

    SNEPPXFMConfig cfg = SNEPPX_fm_config_default();
    cfg.num_nodes = 4;
    cfg.memory_dim = 16;
    cfg.memory_capacity = 32;
    cfg.sync_interval = 1000;
    cfg.privacy_epsilon = 100.0f;

    SNEPPXFMController* ctrl = SNEPPX_fm_controller_create(&cfg);
    if (!ctrl) { printf("ERROR: create controller\n"); return 1; }

    size_t sh[] = {16};
    unsigned long s = 42;
    for (size_t n = 0; n < 4; n++) {
        for (int k = 0; k < 5; k++) {
            SNEPPXTensor* key = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
            SNEPPXTensor* val = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
            for (size_t j = 0; j < 16; j++) {
                s = s * 1103515245UL + 12345UL;
                float v = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f;
                ((float*)key->data)[j] = v + (float)(k + n);
                s = s * 1103515245UL + 12345UL;
                v = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 2.0f;
                ((float*)val->data)[j] = v + (float)(n * 10);
            }
            SNEPPX_fm_memory_bank_write(ctrl->nodes[n]->memory_bank, key, val);
            SNEPPX_tensor_destroy(key);
            SNEPPX_tensor_destroy(val);
        }
    }

    printf("\nBefore sync:\n");
    for (size_t n = 0; n < 4; n++) {
        char label[32]; snprintf(label, sizeof(label), "Node %zu", n);
        print_bank_summary(label, ctrl->nodes[n]->memory_bank);
    }

    int r = SNEPPX_fm_sync_all_reduce(ctrl);
    printf("\nSync result: %d\n", r);
    printf("Sync round: %zu\n", ctrl->sync_state.sync_round);

    printf("\nAfter sync:\n");
    for (size_t n = 0; n < 4; n++) {
        char label[32]; snprintf(label, sizeof(label), "Node %zu", n);
        print_bank_summary(label, ctrl->nodes[n]->memory_bank);
    }

    printf("\n=== FM Forward ===\n");
    size_t in_sh[] = {4, 16};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(in_sh, 2, SNEPPX_FLOAT32);
    for (size_t i = 0; i < input->size; i++) {
        s = s * 1103515245UL + 12345UL;
        ((float*)input->data)[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 0.5f;
    }

    SNEPPXTensor* output = NULL;
    r = SNEPPX_fm_forward(ctrl, 0, input, &output);
    printf("Forward result: %d\n", r);
    if (output) {
        printf("Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);
        printf("First 4 values: ");
        float* od = (float*)output->data;
        for (size_t i = 0; i < 4 && i < output->size; i++) printf("%.6f ", od[i]);
        printf("\n");

        int has_nan = 0, has_inf = 0;
        for (size_t i = 0; i < output->size; i++) {
            if (isnan(od[i])) has_nan = 1;
            if (isinf(od[i])) has_inf = 1;
        }
        printf("NaN: %s, Inf: %s\n", has_nan ? "YES" : "NO", has_inf ? "YES" : "NO");
        if (!has_nan && !has_inf) printf("Output: CLEAN\n");
    }

    print_bank_summary("After forward", ctrl->nodes[0]->memory_bank);

    SNEPPX_tensor_destroy(input);
    if (output) SNEPPX_tensor_destroy(output);
    SNEPPX_fm_controller_destroy(ctrl);
    printf("Demo complete.\n");
    return 0;
}

#include "adversarial_robustness_certification.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Arc Demo
 *
 * WHAT
 *   Arc Demo.
 *
 * CONCEPT
 *   Provides the Arc Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


int main(void) {
    SNEPPXARCConfig cfg = SNEPPX_arc_config_default();
    SNEPPXARCLayer* layer = SNEPPX_arc_layer_create(&cfg, 16, 16, 42);
    if (!layer) { printf("ERROR: create layer\n"); return 1; }

    size_t shape_in[] = {8, 16};

    SNEPPXTensor* normal_input = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* nd = (float*)normal_input->data;
    for (size_t i = 0; i < 8 * 16; i++) nd[i] = 0.1f;

    printf("=== Normal Input ===\n");
    SNEPPXTensor* out_normal = NULL;
    float metrics_n[4];
    SNEPPX_arc_forward(layer, normal_input, &out_normal, metrics_n);
    printf("Anomaly score: %f, Confidence: %f, Clamp ratio: %f, Output norm: %f\n",
           metrics_n[0], metrics_n[1], metrics_n[2], metrics_n[3]);

    SNEPPXTensor* extreme_input = SNEPPX_tensor_zeros(shape_in, 2, SNEPPX_FLOAT32);
    float* ed = (float*)extreme_input->data;
    unsigned long s = 123;
    for (size_t i = 0; i < 8 * 16; i++) {
        s = s * 1103515245UL + 12345UL;
        ed[i] = ((float)((s >> 16) & 0x7FFF) / 32767.0f - 0.5f) * 10.0f;
    }

    printf("\n=== Extreme Input ===\n");
    SNEPPXTensor* out_extreme = NULL;
    float metrics_e[4];
    SNEPPX_arc_forward(layer, extreme_input, &out_extreme, metrics_e);
    printf("Anomaly score: %f, Confidence: %f, Clamp ratio: %f, Output norm: %f\n",
           metrics_e[0], metrics_e[1], metrics_e[2], metrics_e[3]);

    printf("\n=== FGSM Attack Simulation ===\n");
    size_t shape_small[] = {4, 16};
    SNEPPXTensor* clean = SNEPPX_tensor_zeros(shape_small, 2, SNEPPX_FLOAT32);
    float* cd = (float*)clean->data;
    for (size_t i = 0; i < 4 * 16; i++) cd[i] = 0.1f;

    SNEPPXTensor* adv = NULL;
    SNEPPX_arc_simulate_attack(clean, SNEPPX_ATTACK_FGSM, 0.1f, &adv);
    float max_pert = 0.0f;
    float* ad = (float*)adv->data;
    for (size_t i = 0; i < 4 * 16; i++) {
        float d = fabsf(ad[i] - cd[i]);
        if (d > max_pert) max_pert = d;
    }
    printf("Max perturbation: %f\n", max_pert);

    printf("\nNaN/Inf checks:\n");
    int any = 0;
    if (out_normal) {
        float* od = (float*)out_normal->data;
        for (size_t i = 0; i < out_normal->size; i++) {
            if (isnan(od[i]) || isinf(od[i])) any = 1;
        }
    }
    printf("Normal output: %s\n", any ? "ISSUE" : "CLEAN");

    SNEPPX_tensor_destroy(normal_input);
    SNEPPX_tensor_destroy(out_normal);
    SNEPPX_tensor_destroy(extreme_input);
    SNEPPX_tensor_destroy(out_extreme);
    SNEPPX_tensor_destroy(clean);
    SNEPPX_tensor_destroy(adv);
    SNEPPX_arc_layer_destroy(layer);
    printf("Demo complete.\n");
    return any ? 1 : 0;
}

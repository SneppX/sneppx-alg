#include "hierarchical_state_space.h"
#include <stdio.h>
#include <math.h>

/*
 * SNEPPX - Hss Demo
 *
 * WHAT
 *   Hss Demo.
 *
 * CONCEPT
 *   Provides the Hss Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


int main(void) {
    SNEPPXHSSConfig cfg = SNEPPX_hss_config_default();
    cfg.state_dim = 4;
    cfg.input_dim = 8;
    cfg.output_dim = 8;
    cfg.num_layers = 1;
    cfg.seq_len = 16;
    cfg.use_hierarchical = 0;

    SNEPPXHSSModel* model = SNEPPX_hss_model_create(&cfg, 42);
    if (!model) {
        printf("ERROR: failed to create model\n");
        return 1;
    }

    size_t shape_in[] = {16, 8};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    if (!input) {
        printf("ERROR: failed to create input\n");
        SNEPPX_hss_model_destroy(model);
        return 1;
    }

    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_hss_forward(model, input, &output);
    if (ret != 0 || !output) {
        printf("ERROR: forward pass failed (ret=%d)\n", ret);
        SNEPPX_tensor_destroy(input);
        SNEPPX_hss_model_destroy(model);
        return 1;
    }

    printf("Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);

    float* data = (float*)output->data;
    printf("First 8 values: ");
    for (size_t i = 0; i < 8 && i < output->size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    int has_nan = 0, has_inf = 0;
    for (size_t i = 0; i < output->size; i++) {
        if (isnan(data[i])) has_nan = 1;
        if (isinf(data[i])) has_inf = 1;
    }
    printf("NaN: %s, Inf: %s\n", has_nan ? "YES" : "NO", has_inf ? "YES" : "NO");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_hss_model_destroy(model);

    printf("Demo complete.\n");
    return (has_nan || has_inf) ? 1 : 0;
}

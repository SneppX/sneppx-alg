#include "neural_core/kernel/polymorphic_memory_allocator.h"
#include "sparse_expert_routing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * SNEPPX - Ser Demo
 *
 * WHAT
 *   Ser Demo.
 *
 * CONCEPT
 *   Provides the Ser Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */

int main(void)
{
    SNEPPXSERConfig cfg = SNEPPX_ser_config_default();
    cfg.num_experts = 4;
    cfg.num_active = 2;
    cfg.input_dim = 16;
    cfg.expert_dim = 32;
    cfg.output_dim = 16;

    SNEPPXSERModel* model = SNEPPX_ser_model_create(&cfg, 42, 1);
    if (!model)
    {
        printf("ERROR: create model\n");
        return 1;
    }

    size_t shape_in[] = {8, 16};
    SNEPPXTensor* input = SNEPPX_tensor_randn(shape_in, 2, SNEPPX_FLOAT32);
    if (!input)
    {
        printf("ERROR: create input\n");
        return 1;
    }

    SNEPPXTensor* output = NULL;
    SNEPPX_ser_forward(model->layers[0], input, &output);
    if (!output)
    {
        printf("ERROR: forward\n");
        return 1;
    }

    printf("Output shape: [%zu, %zu]\n", output->shape[0], output->shape[1]);

    SNEPPXTensor* gw = NULL;
    int* ei = NULL;
    SNEPPX_ser_route(model->layers[0], input, &gw, &ei);
    if (gw && ei)
    {
        int counts[4] = {0};
        for (size_t t = 0; t < 8; t++)
        {
            for (size_t k = 0; k < 2; k++)
            {
                counts[ei[t * 2 + k]]++;
            }
        }
        printf("Routing distribution (tokens per expert): ");
        for (int i = 0; i < 4; i++)
            printf("%d ", counts[i]);
        printf("\n");

        float loss = SNEPPX_ser_load_balance_loss(gw, ei, 8);
        printf("Load balance loss: %f\n", loss);

        SNEPPX_tensor_destroy(gw);
        SNEPPX_free(ei, 8 * 2 * sizeof(int));
    }

    float* od = (float*) output->data;
    int has_nan = 0, has_inf = 0;
    for (size_t i = 0; i < output->size; i++)
    {
        if (isnan(od[i]))
            has_nan = 1;
        if (isinf(od[i]))
            has_inf = 1;
    }
    printf("NaN: %s, Inf: %s\n", has_nan ? "YES" : "NO", has_inf ? "YES" : "NO");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_ser_model_destroy(model);
    printf("Demo complete.\n");
    return (has_nan || has_inf) ? 1 : 0;
}

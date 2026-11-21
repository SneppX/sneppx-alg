#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
/*
 * SNEPPX - FM Forward Pass
 *
 * WHAT
 *   FM Forward Pass.
 *
 * CONCEPT
 *   Fractal memory forward pass: read from memory bank, combine with input, write back.
 *
 * ROLE
 *   Core FM forward that retrieves relevant memories, combines them with the current input, and produces the output.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Run the forward pass for Fm.
 *
 * @param ctrl [out] Ctrl value.
 * @param node_id [in] Node Id value.
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_fm_forward(SNEPPXFMController* ctrl, size_t node_id, const SNEPPXTensor* input, SNEPPXTensor** output) {
    if (!ctrl || !input || !output) return 1;
    if (node_id >= ctrl->config.num_nodes) return 1;
    if (!ctrl->nodes[node_id]->is_online) return 1;

    SNEPPXFMNode* node = ctrl->nodes[node_id];
    SNEPPXFMMemoryBank* bank = node->memory_bank;

    SNEPPXTensor* result = NULL;

    SNEPPXTensor* retrieved = SNEPPX_fm_memory_bank_read(bank, input);
    if (retrieved) {
        float alpha = ctrl->config.ewm_alpha;
        size_t sz = input->size < retrieved->size ? input->size : retrieved->size;
        result = SNEPPX_tensor_create(input->shape, input->ndim, SNEPPX_FLOAT32);
        if (result) {
            float* rd = (float*)result->data;
            float* id = (float*)input->data;
            float* vd = (float*)retrieved->data;
            for (size_t i = 0; i < sz; i++) {
                rd[i] = alpha * id[i] + (1.0f - alpha) * vd[i];
            }
        }
        SNEPPX_tensor_destroy(retrieved);
    } else {
        result = SNEPPX_tensor_create(input->shape, input->ndim, SNEPPX_FLOAT32);
        if (result) {
            memcpy((float*)result->data, (float*)input->data, input->size * sizeof(float));
        }
        SNEPPX_fm_memory_bank_write(bank, input, input);
    }

    if (!result) return 1;

    if (node->gradient_accumulator && result->size == node->gradient_accumulator->size) {
        float* ga = (float*)node->gradient_accumulator->data;
        float* rd = (float*)result->data;
        for (size_t i = 0; i < result->size; i++) {
            ga[i] += rd[i] * 0.01f;
        }
    }

    *output = result;

    ctrl->step_counter++;

    if (ctrl->config.sync_interval > 0 && ctrl->step_counter % ctrl->config.sync_interval == 0) {
        switch (ctrl->config.sync_method) {
            case SNEPPX_SYNC_ALL_REDUCE:
                SNEPPX_fm_sync_all_reduce(ctrl);
                break;
            case SNEPPX_SYNC_GOSSIP:
                SNEPPX_fm_sync_gossip(ctrl, ctrl->config.num_nodes);
                break;
            case SNEPPX_SYNC_TOPOLOGY:
                SNEPPX_fm_sync_topology(ctrl);
                break;
        }
    }

    return 0;
}

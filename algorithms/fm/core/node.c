#include "fractal_memory_orchestrator.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
/*
 * SNEPPX - FM Memory Node
 *
 * WHAT
 *   FM Memory Node.
 *
 * CONCEPT
 *   Individual memory node operations: read, write, and update in the fractal memory tree.
 *
 * ROLE
 *   Implements the basic read/write/merge operations on a single memory node in the fractal hierarchy.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Create Fm Node.
 *
 * @param node_id [in] Node Id value.
 * @param memory_dim [in] Memory Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXFMNode* SNEPPX_fm_node_create(size_t node_id, size_t memory_dim, size_t capacity) {
    SNEPPXFMNode* node = (SNEPPXFMNode*)SNEPPX_malloc(sizeof(SNEPPXFMNode), 64);
    if (!node) return NULL;
    memset(node, 0, sizeof(SNEPPXFMNode));

    node->node_id = node_id;
    node->memory_bank = SNEPPX_fm_memory_bank_create(memory_dim, capacity);
    if (!node->memory_bank) {
        SNEPPX_free(node, sizeof(SNEPPXFMNode));
        return NULL;
    }

    size_t grad_shape[] = {memory_dim, memory_dim};
    node->gradient_accumulator = SNEPPX_tensor_zeros(grad_shape, 2, SNEPPX_FLOAT32);
    if (!node->gradient_accumulator) {
        SNEPPX_fm_memory_bank_destroy(node->memory_bank);
        SNEPPX_free(node, sizeof(SNEPPXFMNode));
        return NULL;
    }

    node->last_sync_time = 0;
    node->is_online = 1;
    node->trust_score = 1.0f;
    return node;
}

/**
 * @brief Destroy Fm Node.
 */
void SNEPPX_fm_node_destroy(SNEPPXFMNode* node) {
    if (!node) return;
    if (node->memory_bank) SNEPPX_fm_memory_bank_destroy(node->memory_bank);
    if (node->gradient_accumulator) SNEPPX_tensor_destroy(node->gradient_accumulator);
    SNEPPX_free(node, sizeof(SNEPPXFMNode));
}

#include "cognitive_memory.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Working
 *
 * WHAT
 *   Working.
 *
 * CONCEPT
 *   Provides the Working.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    SNEPPXTensor* content;
    float attention_weight;
    uint64_t last_modified;
    uint64_t created_at;
    int is_free;
} SNEPPXWorkingSlot;

typedef struct SNEPPXWorkingMemory SNEPPXWorkingMemory;

struct SNEPPXWorkingMemory {
    SNEPPXWorkingSlot* slots;
    size_t num_slots;
    size_t slot_dim;
    uint64_t clock;
};

SNEPPXWorkingMemory* sneppx_working_create(const SNEPPXWorkingConfig* cfg) {
    SNEPPXWorkingMemory* wm = SNEPPX_malloc(sizeof(SNEPPXWorkingMemory), 64);
    if (!wm) return NULL;
    memset(wm, 0, sizeof(SNEPPXWorkingMemory));
    wm->num_slots = cfg->num_slots;
    if (wm->num_slots < SNEPPX_WORKING_MEMORY_MIN_SLOTS) wm->num_slots = SNEPPX_WORKING_MEMORY_MIN_SLOTS;
    if (wm->num_slots > SNEPPX_WORKING_MEMORY_MAX_SLOTS) wm->num_slots = SNEPPX_WORKING_MEMORY_MAX_SLOTS;
    wm->slot_dim = cfg->slot_dim > 0 ? cfg->slot_dim : 64;
    wm->slots = SNEPPX_malloc(wm->num_slots * sizeof(SNEPPXWorkingSlot), 64);
    if (!wm->slots) {
        SNEPPX_free(wm, sizeof(SNEPPXWorkingMemory));
        return NULL;
    }
    memset(wm->slots, 0, wm->num_slots * sizeof(SNEPPXWorkingSlot));
    for (size_t i = 0; i < wm->num_slots; i++) {
        wm->slots[i].is_free = 1;
    }
    return wm;
}

void sneppx_working_destroy(SNEPPXWorkingMemory* wm) {
    if (!wm) return;
    for (size_t i = 0; i < wm->num_slots; i++) {
        if (wm->slots[i].content) SNEPPX_tensor_destroy(wm->slots[i].content);
    }
    SNEPPX_free(wm->slots, wm->num_slots * sizeof(SNEPPXWorkingSlot));
    SNEPPX_free(wm, sizeof(SNEPPXWorkingMemory));
}

int sneppx_working_write(SNEPPXWorkingMemory* wm, size_t slot_index,
                         const SNEPPXTensor* content) {
    if (!wm || !content) return -1;
    if (slot_index >= wm->num_slots) return -1;
    SNEPPXWorkingSlot* slot = &wm->slots[slot_index];
    if (slot->content) SNEPPX_tensor_destroy(slot->content);
    slot->content = SNEPPX_tensor_copy(content);
    if (!slot->content) return -1;
    slot->attention_weight = 1.0f;
    slot->last_modified = ++wm->clock;
    if (slot->is_free) {
        slot->created_at = wm->clock;
        slot->is_free = 0;
    }
    return 0;
}

int sneppx_working_read(SNEPPXWorkingMemory* wm, size_t slot_index,
                        SNEPPXTensor** output) {
    if (!wm || !output) return -1;
    if (slot_index >= wm->num_slots) return -1;
    SNEPPXWorkingSlot* slot = &wm->slots[slot_index];
    if (slot->is_free || !slot->content) { *output = NULL; return 0; }
    *output = SNEPPX_tensor_copy(slot->content);
    if (!*output) return -1;
    slot->last_modified = ++wm->clock;
    return 0;
}

int sneppx_working_attend(SNEPPXWorkingMemory* wm, const SNEPPXTensor* query,
                          SNEPPXTensor** output) {
    if (!wm || !query || !output) return -1;
    float* qd = (float*)query->data;
    size_t qlen = query->size;
    float query_norm = 0.0f;
    for (size_t i = 0; i < qlen; i++) query_norm += qd[i] * qd[i];
    query_norm = sqrtf(query_norm + 1e-10f);
    float total_weight = 0.0f;
    for (size_t i = 0; i < wm->num_slots; i++) {
        if (wm->slots[i].is_free || !wm->slots[i].content) {
            wm->slots[i].attention_weight = 0.0f;
            continue;
        }
        float* sd = (float*)wm->slots[i].content->data;
        size_t slen = wm->slots[i].content->size;
        size_t min_dim = qlen < slen ? qlen : slen;
        float dot = 0.0f, s_norm = 0.0f;
        for (size_t j = 0; j < min_dim; j++) dot += qd[j] * sd[j];
        for (size_t j = 0; j < slen; j++) s_norm += sd[j] * sd[j];
        s_norm = sqrtf(s_norm + 1e-10f);
        float sim = dot / (query_norm * s_norm + 1e-10f);
        float recency = 1.0f / (1.0f + (float)(wm->clock - wm->slots[i].last_modified));
        wm->slots[i].attention_weight = sim * 0.7f + recency * 0.3f;
        if (wm->slots[i].attention_weight < 0.0f) wm->slots[i].attention_weight = 0.0f;
        total_weight += wm->slots[i].attention_weight;
    }
    size_t shape[2] = {1, wm->slot_dim};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    if (!result) return -1;
    float* rd = (float*)result->data;
    for (size_t i = 0; i < wm->num_slots; i++) {
        if (wm->slots[i].is_free || !wm->slots[i].content ||
            wm->slots[i].attention_weight <= 0.0f) continue;
        float w = wm->slots[i].attention_weight / (total_weight + 1e-10f);
        float* sd = (float*)wm->slots[i].content->data;
        size_t copy_dim = wm->slot_dim < wm->slots[i].content->size ?
                          wm->slot_dim : wm->slots[i].content->size;
        for (size_t j = 0; j < copy_dim; j++) rd[j] += w * sd[j];
        wm->slots[i].last_modified = ++wm->clock;
    }
    *output = result;
    return 0;
}

int sneppx_working_clear(SNEPPXWorkingMemory* wm) {
    if (!wm) return -1;
    for (size_t i = 0; i < wm->num_slots; i++) {
        if (wm->slots[i].content) {
            SNEPPX_tensor_destroy(wm->slots[i].content);
            wm->slots[i].content = NULL;
        }
        wm->slots[i].is_free = 1;
        wm->slots[i].attention_weight = 0.0f;
    }
    return 0;
}

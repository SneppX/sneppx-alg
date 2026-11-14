#include "cognitive_memory.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Episodic
 *
 * WHAT
 *   Episodic.
 *
 * CONCEPT
 *   Provides the Episodic.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    SNEPPXTensor* state;
    SNEPPXTensor* action;
    float reward;
    SNEPPXTensor* next_state;
    uint64_t timestamp;
    float priority;
    int valid;
} SNEPPXEpisodicExperience;

typedef struct SNEPPXEpisodicBuffer SNEPPXEpisodicBuffer;

struct SNEPPXEpisodicBuffer {
    SNEPPXEpisodicExperience* experiences;
    size_t capacity;
    size_t count;
    size_t head;
    uint64_t next_timestamp;
    uint64_t consolidation_interval;
    uint64_t last_consolidation;
    unsigned int* seed_ptr;
};

/* Internal helper used by record() before its definition below. */
static int sneppx_episodic_consolidate(SNEPPXEpisodicBuffer* buf);

static int compare_priority_desc(const void* a, const void* b) {
    const SNEPPXEpisodicExperience* ea = (const SNEPPXEpisodicExperience*)a;
    const SNEPPXEpisodicExperience* eb = (const SNEPPXEpisodicExperience*)b;
    if (eb->priority > ea->priority) return 1;
    if (eb->priority < ea->priority) return -1;
    return 0;
}

static float frand(unsigned int* seed) {
    *seed = *seed * 1103515245U + 12345U;
    return (float)((*seed >> 16) & 0x7FFF) / 32767.0f;
}

SNEPPXEpisodicBuffer* sneppx_episodic_create(const SNEPPXEpisodicConfig* cfg, unsigned int* seed_ptr) {
    SNEPPXEpisodicBuffer* buf = SNEPPX_malloc(sizeof(SNEPPXEpisodicBuffer), 64);
    if (!buf) return NULL;
    memset(buf, 0, sizeof(SNEPPXEpisodicBuffer));
    buf->capacity = cfg->capacity > 0 ? cfg->capacity : 1000;
    buf->consolidation_interval = cfg->consolidation_interval > 0 ? cfg->consolidation_interval : 100;
    buf->seed_ptr = seed_ptr;
    buf->experiences = SNEPPX_malloc(buf->capacity * sizeof(SNEPPXEpisodicExperience), 64);
    if (!buf->experiences) {
        SNEPPX_free(buf, sizeof(SNEPPXEpisodicBuffer));
        return NULL;
    }
    memset(buf->experiences, 0, buf->capacity * sizeof(SNEPPXEpisodicExperience));
    return buf;
}

void sneppx_episodic_destroy(SNEPPXEpisodicBuffer* buf) {
    if (!buf) return;
    for (size_t i = 0; i < buf->capacity; i++) {
        SNEPPXTensor* exp = buf->experiences[i].state;
        if (exp) SNEPPX_tensor_destroy(exp);
        if (buf->experiences[i].action) SNEPPX_tensor_destroy(buf->experiences[i].action);
        if (buf->experiences[i].next_state) SNEPPX_tensor_destroy(buf->experiences[i].next_state);
    }
    SNEPPX_free(buf->experiences, buf->capacity * sizeof(SNEPPXEpisodicExperience));
    SNEPPX_free(buf, sizeof(SNEPPXEpisodicBuffer));
}

int sneppx_episodic_record(SNEPPXEpisodicBuffer* buf, const SNEPPXTensor* state,
                           const SNEPPXTensor* action, float reward,
                           const SNEPPXTensor* next_state) {
    if (!buf || !state) return -1;
    size_t idx = buf->count < buf->capacity ? buf->count : buf->head;
    SNEPPXEpisodicExperience* exp = &buf->experiences[idx];
    if (exp->valid) {
        if (exp->state) SNEPPX_tensor_destroy(exp->state);
        if (exp->action) SNEPPX_tensor_destroy(exp->action);
        if (exp->next_state) SNEPPX_tensor_destroy(exp->next_state);
    }
    exp->state = SNEPPX_tensor_copy(state);
    if (!exp->state) return -1;
    if (action) {
        exp->action = SNEPPX_tensor_copy(action);
        if (!exp->action) { SNEPPX_tensor_destroy(exp->state); exp->state = NULL; return -1; }
    } else {
        exp->action = NULL;
    }
    exp->reward = reward;
    if (next_state) {
        exp->next_state = SNEPPX_tensor_copy(next_state);
        if (!exp->next_state) {
            SNEPPX_tensor_destroy(exp->state); exp->state = NULL;
            if (exp->action) { SNEPPX_tensor_destroy(exp->action); exp->action = NULL; }
            return -1;
        }
    } else {
        exp->next_state = NULL;
    }
    exp->timestamp = buf->next_timestamp++;
    exp->priority = reward > 0 ? reward : 0.01f;
    exp->valid = 1;
    if (buf->count < buf->capacity) {
        buf->count++;
    } else {
        buf->head = (buf->head + 1) % buf->capacity;
    }
    if (exp->timestamp - buf->last_consolidation >= buf->consolidation_interval) {
        sneppx_episodic_consolidate(buf);
    }
    return 0;
}

int sneppx_episodic_retrieve(SNEPPXEpisodicBuffer* buf, uint64_t time_start,
                             uint64_t time_end, SNEPPXTensor** output) {
    if (!buf || !output) return -1;
    size_t match_count = 0;
    for (size_t i = 0; i < buf->capacity; i++) {
        if (buf->experiences[i].valid &&
            buf->experiences[i].timestamp >= time_start &&
            buf->experiences[i].timestamp <= time_end) {
            match_count++;
        }
    }
    if (match_count == 0) {
        *output = NULL;
        return 0;
    }
    size_t state_dim = buf->experiences[0].state ? buf->experiences[0].state->size : 1;
    size_t shape[2] = {match_count, state_dim};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    if (!result) return -1;
    size_t row = 0;
    float* data = (float*)result->data;
    for (size_t i = 0; i < buf->capacity; i++) {
        if (buf->experiences[i].valid &&
            buf->experiences[i].timestamp >= time_start &&
            buf->experiences[i].timestamp <= time_end) {
            if (buf->experiences[i].state) {
                memcpy(data + row * state_dim, buf->experiences[i].state->data,
                       state_dim * sizeof(float));
            }
            row++;
        }
    }
    *output = result;
    return 0;
}

int sneppx_episodic_sample(SNEPPXEpisodicBuffer* buf, size_t batch_size,
                           SNEPPXTensor** states, SNEPPXTensor** actions,
                           float* rewards, SNEPPXTensor** next_states) {
    if (!buf || !states || !rewards || batch_size == 0) return -1;
    size_t available = 0;
    for (size_t i = 0; i < buf->capacity; i++) {
        if (buf->experiences[i].valid) available++;
    }
    if (available == 0) return -1;
    size_t actual_batch = batch_size < available ? batch_size : available;
    SNEPPXEpisodicExperience* sorted = SNEPPX_malloc(available * sizeof(SNEPPXEpisodicExperience), 64);
    if (!sorted) return -1;
    size_t idx = 0;
    for (size_t i = 0; i < buf->capacity; i++) {
        if (buf->experiences[i].valid) {
            sorted[idx++] = buf->experiences[i];
        }
    }
    qsort(sorted, available, sizeof(SNEPPXEpisodicExperience), compare_priority_desc);
    float total_priority = 0.0f;
    for (size_t i = 0; i < available; i++) total_priority += sorted[i].priority;
    if (total_priority < 1e-10f) total_priority = 1.0f;
    for (size_t i = 0; i < actual_batch; i++) {
        float r = frand(buf->seed_ptr) * total_priority;
        float cum = 0.0f;
        size_t chosen = 0;
        for (size_t j = 0; j < available; j++) {
            cum += sorted[j].priority;
            if (r <= cum) { chosen = j; break; }
        }
        SNEPPXEpisodicExperience* exp = &sorted[chosen];
        if (exp->state) {
            states[i] = SNEPPX_tensor_copy(exp->state);
            if (!states[i]) {
                for (size_t k = 0; k < i; k++) SNEPPX_tensor_destroy(states[k]);
                SNEPPX_free(sorted, available * sizeof(SNEPPXEpisodicExperience));
                return -1;
            }
        } else {
            states[i] = NULL;
        }
        if (actions && exp->action) {
            actions[i] = SNEPPX_tensor_copy(exp->action);
        } else if (actions) {
            actions[i] = NULL;
        }
        rewards[i] = exp->reward;
        if (next_states && exp->next_state) {
            next_states[i] = SNEPPX_tensor_copy(exp->next_state);
        } else if (next_states) {
            next_states[i] = NULL;
        }
        sorted[chosen].priority *= 0.9f;
    }
    SNEPPX_free(sorted, available * sizeof(SNEPPXEpisodicExperience));
    return (int)actual_batch;
}

static int sneppx_episodic_consolidate(SNEPPXEpisodicBuffer* buf) {
    if (!buf) return -1;
    buf->last_consolidation = buf->next_timestamp;
    if (buf->count < 2) return 0;
    float mean_reward = 0.0f;
    float max_reward = -1e10f;
    float min_reward = 1e10f;
    size_t valid_count = 0;
    for (size_t i = 0; i < buf->capacity; i++) {
        if (buf->experiences[i].valid) {
            mean_reward += buf->experiences[i].reward;
            if (buf->experiences[i].reward > max_reward) max_reward = buf->experiences[i].reward;
            if (buf->experiences[i].reward < min_reward) min_reward = buf->experiences[i].reward;
            valid_count++;
        }
    }
    if (valid_count == 0) return 0;
    mean_reward /= (float)valid_count;
    for (size_t i = 0; i < buf->capacity; i++) {
        if (buf->experiences[i].valid) {
            float age = (float)(buf->next_timestamp - buf->experiences[i].timestamp);
            if (age > 1000.0f && buf->experiences[i].reward < mean_reward * 0.5f) {
                buf->experiences[i].priority *= 0.5f;
            }
        }
    }
    return 0;
}

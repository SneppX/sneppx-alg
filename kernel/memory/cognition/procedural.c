#include "cognitive_memory.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Procedural
 *
 * WHAT
 *   Procedural.
 *
 * CONCEPT
 *   Provides the Procedural.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    SNEPPXTensor* state_pattern;
    SNEPPXTensor* skill_output;
    size_t frequency;
    uint64_t last_used;
    size_t sequence_length;
    int valid;
} SNEPPXProceduralSkill;

typedef struct SNEPPXProceduralCache SNEPPXProceduralCache;

struct SNEPPXProceduralCache {
    SNEPPXProceduralSkill* skills;
    size_t num_skills;
    size_t max_skills;
    size_t state_dim;
    size_t skill_dim;
    size_t compilation_threshold;
    uint64_t access_counter;
    unsigned int* seed_ptr;
};

SNEPPXProceduralCache* sneppx_procedural_create(const SNEPPXProceduralConfig* cfg,
                                                 unsigned int* seed_ptr) {
    SNEPPXProceduralCache* cache = SNEPPX_malloc(sizeof(SNEPPXProceduralCache), 64);
    if (!cache) return NULL;
    memset(cache, 0, sizeof(SNEPPXProceduralCache));
    cache->max_skills = cfg->cache_size > 0 ? cfg->cache_size : 50;
    cache->state_dim = cfg->state_dim > 0 ? cfg->state_dim : 64;
    cache->skill_dim = cfg->skill_dim > 0 ? cfg->skill_dim : 64;
    cache->compilation_threshold = cfg->compilation_threshold > 0 ?
                                   cfg->compilation_threshold : 5;
    cache->seed_ptr = seed_ptr;
    cache->skills = SNEPPX_malloc(cache->max_skills * sizeof(SNEPPXProceduralSkill), 64);
    if (!cache->skills) {
        SNEPPX_free(cache, sizeof(SNEPPXProceduralCache));
        return NULL;
    }
    memset(cache->skills, 0, cache->max_skills * sizeof(SNEPPXProceduralSkill));
    return cache;
}

void sneppx_procedural_destroy(SNEPPXProceduralCache* cache) {
    if (!cache) return;
    for (size_t i = 0; i < cache->num_skills; i++) {
        if (cache->skills[i].state_pattern) SNEPPX_tensor_destroy(cache->skills[i].state_pattern);
        if (cache->skills[i].skill_output) SNEPPX_tensor_destroy(cache->skills[i].skill_output);
    }
    SNEPPX_free(cache->skills, cache->max_skills * sizeof(SNEPPXProceduralSkill));
    SNEPPX_free(cache, sizeof(SNEPPXProceduralCache));
}

static int find_matching_skill(SNEPPXProceduralCache* cache, const SNEPPXTensor* state) {
    float* sd = (float*)state->data;
    size_t slen = state->size;
    float s_norm = 0.0f;
    for (size_t i = 0; i < slen; i++) s_norm += sd[i] * sd[i];
    s_norm = sqrtf(s_norm + 1e-10f);
    int best_idx = -1;
    float best_sim = -1.0f;
    for (size_t i = 0; i < cache->num_skills; i++) {
        if (!cache->skills[i].valid || !cache->skills[i].state_pattern) continue;
        float* pd = (float*)cache->skills[i].state_pattern->data;
        size_t plen = cache->skills[i].state_pattern->size;
        size_t min_dim = slen < plen ? slen : plen;
        float dot = 0.0f, p_norm = 0.0f;
        for (size_t j = 0; j < min_dim; j++) dot += sd[j] * pd[j];
        for (size_t j = 0; j < plen; j++) p_norm += pd[j] * pd[j];
        p_norm = sqrtf(p_norm + 1e-10f);
        float sim = dot / (s_norm * p_norm + 1e-10f);
        if (sim > best_sim) { best_sim = sim; best_idx = (int)i; }
    }
    if (best_sim > 0.85f) return best_idx;
    return -1;
}

static int evict_lru(SNEPPXProceduralCache* cache) {
    if (cache->num_skills < cache->max_skills) {
        int idx = (int)cache->num_skills;
        cache->num_skills++;
        return idx;
    }
    uint64_t oldest = cache->access_counter;
    int lru_idx = -1;
    for (size_t i = 0; i < cache->num_skills; i++) {
        if (cache->skills[i].valid && cache->skills[i].last_used < oldest) {
            oldest = cache->skills[i].last_used;
            lru_idx = (int)i;
        }
    }
    if (lru_idx >= 0) {
        if (cache->skills[lru_idx].state_pattern) {
            SNEPPX_tensor_destroy(cache->skills[lru_idx].state_pattern);
            cache->skills[lru_idx].state_pattern = NULL;
        }
        if (cache->skills[lru_idx].skill_output) {
            SNEPPX_tensor_destroy(cache->skills[lru_idx].skill_output);
            cache->skills[lru_idx].skill_output = NULL;
        }
        memset(&cache->skills[lru_idx], 0, sizeof(SNEPPXProceduralSkill));
        return lru_idx;
    }
    return -1;
}

int sneppx_procedural_learn(SNEPPXProceduralCache* cache, const SNEPPXTensor* state,
                            const SNEPPXTensor* skill) {
    if (!cache || !state || !skill) return -1;
    int idx = find_matching_skill(cache, state);
    if (idx >= 0) {
        SNEPPXProceduralSkill* sk = &cache->skills[idx];
        if (sk->skill_output) SNEPPX_tensor_destroy(sk->skill_output);
        sk->skill_output = SNEPPX_tensor_copy(skill);
        if (!sk->skill_output) return -1;
        sk->frequency++;
        sk->last_used = ++cache->access_counter;
        return 0;
    }
    idx = evict_lru(cache);
    if (idx < 0) return -1;
    SNEPPXProceduralSkill* sk = &cache->skills[idx];
    sk->state_pattern = SNEPPX_tensor_copy(state);
    if (!sk->state_pattern) return -1;
    sk->skill_output = SNEPPX_tensor_copy(skill);
    if (!sk->skill_output) {
        SNEPPX_tensor_destroy(sk->state_pattern);
        sk->state_pattern = NULL;
        return -1;
    }
    sk->frequency = 1;
    sk->last_used = ++cache->access_counter;
    sk->sequence_length = 1;
    sk->valid = 1;
    return 0;
}

int sneppx_procedural_recall(SNEPPXProceduralCache* cache, const SNEPPXTensor* state,
                             SNEPPXTensor** output) {
    if (!cache || !state || !output) return -1;
    int idx = find_matching_skill(cache, state);
    if (idx < 0) { *output = NULL; return 0; }
    SNEPPXProceduralSkill* sk = &cache->skills[idx];
    sk->last_used = ++cache->access_counter;
    *output = SNEPPX_tensor_copy(sk->skill_output);
    if (!*output) return -1;
    return 0;
}

int sneppx_procedural_compile(SNEPPXProceduralCache* cache) {
    if (!cache) return -1;
    int compiled = 0;
    for (size_t i = 0; i < cache->num_skills; i++) {
        if (!cache->skills[i].valid) continue;
        if (cache->skills[i].frequency >= cache->compilation_threshold) {
            float boost = (float)(cache->skills[i].frequency) /
                          (float)(cache->compilation_threshold);
            cache->skills[i].sequence_length += 1;
            cache->skills[i].frequency = cache->compilation_threshold;
            compiled++;
        }
    }
    return compiled;
}

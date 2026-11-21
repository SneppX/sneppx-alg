#include "cognitive_memory.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Cognitive Manager
 *
 * WHAT
 *   Cognitive Manager.
 *
 * CONCEPT
 *   Provides the Cognitive Manager.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct SNEPPXEpisodicBuffer SNEPPXEpisodicBuffer;
typedef struct SNEPPXSemanticGraph SNEPPXSemanticGraph;
typedef struct SNEPPXWorkingMemory SNEPPXWorkingMemory;
typedef struct SNEPPXProceduralCache SNEPPXProceduralCache;

SNEPPXEpisodicBuffer* sneppx_episodic_create(const SNEPPXEpisodicConfig* cfg, unsigned int* seed_ptr);
void sneppx_episodic_destroy(SNEPPXEpisodicBuffer* buf);
int sneppx_episodic_record(SNEPPXEpisodicBuffer* buf, const SNEPPXTensor* state,
                           const SNEPPXTensor* action, float reward,
                           const SNEPPXTensor* next_state);
int sneppx_episodic_retrieve(SNEPPXEpisodicBuffer* buf, uint64_t time_start,
                             uint64_t time_end, SNEPPXTensor** output);
int sneppx_episodic_sample(SNEPPXEpisodicBuffer* buf, size_t batch_size,
                           SNEPPXTensor** states, SNEPPXTensor** actions,
                           float* rewards, SNEPPXTensor** next_states);
int sneppx_episodic_consolidate(SNEPPXEpisodicBuffer* buf);

SNEPPXSemanticGraph* sneppx_semantic_create(const SNEPPXSemanticConfig* cfg, unsigned int* seed_ptr);
void sneppx_semantic_destroy(SNEPPXSemanticGraph* graph);
int sneppx_semantic_store(SNEPPXSemanticGraph* graph, const char* key,
                          const SNEPPXTensor* embedding);
int sneppx_semantic_retrieve(SNEPPXSemanticGraph* graph, const SNEPPXTensor* query,
                             size_t top_k, SNEPPXTensor** output, char*** out_keys);
int sneppx_semantic_relate(SNEPPXSemanticGraph* graph, const char* from_key,
                           const char* to_key, float weight);
int sneppx_semantic_forget(SNEPPXSemanticGraph* graph, const char* key);

SNEPPXWorkingMemory* sneppx_working_create(const SNEPPXWorkingConfig* cfg);
void sneppx_working_destroy(SNEPPXWorkingMemory* wm);
int sneppx_working_write(SNEPPXWorkingMemory* wm, size_t slot_index,
                         const SNEPPXTensor* content);
int sneppx_working_read(SNEPPXWorkingMemory* wm, size_t slot_index,
                        SNEPPXTensor** output);
int sneppx_working_attend(SNEPPXWorkingMemory* wm, const SNEPPXTensor* query,
                          SNEPPXTensor** output);
int sneppx_working_clear(SNEPPXWorkingMemory* wm);

SNEPPXProceduralCache* sneppx_procedural_create(const SNEPPXProceduralConfig* cfg,
                                                  unsigned int* seed_ptr);
void sneppx_procedural_destroy(SNEPPXProceduralCache* cache);
int sneppx_procedural_learn(SNEPPXProceduralCache* cache, const SNEPPXTensor* state,
                            const SNEPPXTensor* skill);
int sneppx_procedural_recall(SNEPPXProceduralCache* cache, const SNEPPXTensor* state,
                             SNEPPXTensor** output);
int sneppx_procedural_compile(SNEPPXProceduralCache* cache);

struct SNEPPXCognitiveMemory {
    SNEPPXEpisodicBuffer* episodic;
    SNEPPXSemanticGraph* semantic;
    SNEPPXWorkingMemory* working;
    SNEPPXProceduralCache* procedural;
    SNEPPXCognitiveMemoryConfig config;
    unsigned int seed;
};

static float frand(unsigned int* seed) {
    *seed = *seed * 1103515245U + 12345U;
    return (float)((*seed >> 16) & 0x7FFF) / 32767.0f;
}

/**
 * @brief Perform Cognitive Memory Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXCognitiveMemoryConfig SNEPPX_cognitive_memory_config_default(void) {
    SNEPPXCognitiveMemoryConfig cfg;
    cfg.episodic.capacity = 1000;
    cfg.episodic.max_priority_samples = 64;
    cfg.episodic.consolidation_interval = 100;
    cfg.semantic.max_concepts = 100;
    cfg.semantic.max_edges = 500;
    cfg.semantic.embedding_dim = 64;
    cfg.semantic.lru_size = 20;
    cfg.working.num_slots = SNEPPX_WORKING_MEMORY_DEFAULT_SLOTS;
    cfg.working.slot_dim = 64;
    cfg.procedural.cache_size = 50;
    cfg.procedural.state_dim = 64;
    cfg.procedural.skill_dim = 64;
    cfg.procedural.compilation_threshold = 5;
    return cfg;
}

/**
 * @brief Create Cognitive Memory.
 *
 * @param config [in] Config value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXCognitiveMemory* SNEPPX_cognitive_memory_create(
    const SNEPPXCognitiveMemoryConfig* config, unsigned int seed) {
    if (!config) return NULL;
    SNEPPXCognitiveMemory* cmem = SNEPPX_malloc(sizeof(SNEPPXCognitiveMemory), 64);
    if (!cmem) return NULL;
    memset(cmem, 0, sizeof(SNEPPXCognitiveMemory));
    cmem->config = *config;
    cmem->seed = seed;
    cmem->episodic = sneppx_episodic_create(&config->episodic, &cmem->seed);
    if (!cmem->episodic) {
        SNEPPX_free(cmem, sizeof(SNEPPXCognitiveMemory));
        return NULL;
    }
    cmem->semantic = sneppx_semantic_create(&config->semantic, &cmem->seed);
    if (!cmem->semantic) {
        sneppx_episodic_destroy(cmem->episodic);
        SNEPPX_free(cmem, sizeof(SNEPPXCognitiveMemory));
        return NULL;
    }
    cmem->working = sneppx_working_create(&config->working);
    if (!cmem->working) {
        sneppx_episodic_destroy(cmem->episodic);
        sneppx_semantic_destroy(cmem->semantic);
        SNEPPX_free(cmem, sizeof(SNEPPXCognitiveMemory));
        return NULL;
    }
    cmem->procedural = sneppx_procedural_create(&config->procedural, &cmem->seed);
    if (!cmem->procedural) {
        sneppx_episodic_destroy(cmem->episodic);
        sneppx_semantic_destroy(cmem->semantic);
        sneppx_working_destroy(cmem->working);
        SNEPPX_free(cmem, sizeof(SNEPPXCognitiveMemory));
        return NULL;
    }
    return cmem;
}

/**
 * @brief Destroy Cognitive Memory.
 */
void SNEPPX_cognitive_memory_destroy(SNEPPXCognitiveMemory* cmem) {
    if (!cmem) return;
    if (cmem->episodic) sneppx_episodic_destroy(cmem->episodic);
    if (cmem->semantic) sneppx_semantic_destroy(cmem->semantic);
    if (cmem->working) sneppx_working_destroy(cmem->working);
    if (cmem->procedural) sneppx_procedural_destroy(cmem->procedural);
    SNEPPX_free(cmem, sizeof(SNEPPXCognitiveMemory));
}

/**
 * @brief Run the forward pass for Cognitive Memory.
 *
 * @param cmem [out] Cmem value.
 * @param input [in] Input value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cognitive_memory_forward(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* input,
    SNEPPXTensor** output) {
    if (!cmem || !input || !output) return -1;
    sneppx_working_write(cmem->working, 0, input);
    SNEPPXTensor* attended = NULL;
    sneppx_working_attend(cmem->working, input, &attended);
    SNEPPXTensor* semantic_result = NULL;
    sneppx_semantic_retrieve(cmem->semantic, input, 3, &semantic_result, NULL);
    SNEPPXTensor* procedural_result = NULL;
    sneppx_procedural_recall(cmem->procedural, input, &procedural_result);
    size_t out_dim = input->size + cmem->config.working.slot_dim +
                     cmem->config.semantic.embedding_dim +
                     cmem->config.procedural.skill_dim;
    size_t shape[2] = {1, out_dim};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    if (!result) {
        if (attended) SNEPPX_tensor_destroy(attended);
        if (semantic_result) SNEPPX_tensor_destroy(semantic_result);
        if (procedural_result) SNEPPX_tensor_destroy(procedural_result);
        return -1;
    }
    float* rd = (float*)result->data;
    size_t offset = 0;
    memcpy(rd + offset, input->data, input->size * sizeof(float));
    offset += input->size;
    if (attended) {
        size_t copy_dim = cmem->config.working.slot_dim < attended->size ?
                          cmem->config.working.slot_dim : attended->size;
        memcpy(rd + offset, attended->data, copy_dim * sizeof(float));
        SNEPPX_tensor_destroy(attended);
    }
    offset += cmem->config.working.slot_dim;
    if (semantic_result) {
        float* sd = (float*)semantic_result->data;
        size_t num_rows = semantic_result->shape[0];
        size_t emb_dim = semantic_result->shape[1];
        size_t copy_dim = cmem->config.semantic.embedding_dim < emb_dim ?
                          cmem->config.semantic.embedding_dim : emb_dim;
        for (size_t r = 0; r < num_rows && r < 3; r++) {
            for (size_t c = 0; c < copy_dim; c++) {
                rd[offset + c] += sd[r * emb_dim + c] / (float)num_rows;
            }
        }
        SNEPPX_tensor_destroy(semantic_result);
    }
    offset += cmem->config.semantic.embedding_dim;
    if (procedural_result) {
        size_t copy_dim = cmem->config.procedural.skill_dim < procedural_result->size ?
                          cmem->config.procedural.skill_dim : procedural_result->size;
        memcpy(rd + offset, procedural_result->data, copy_dim * sizeof(float));
        SNEPPX_tensor_destroy(procedural_result);
    }
    *output = result;
    return 0;
}

/**
 * @brief Perform Episodic Record.
 *
 * @param cmem [out] Cmem value.
 * @param state [in] State value.
 * @param action [in] Action value.
 * @param reward [in] Reward value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_record(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    const SNEPPXTensor* action,
    float reward,
    const SNEPPXTensor* next_state) {
    if (!cmem || !cmem->episodic) return -1;
    return sneppx_episodic_record(cmem->episodic, state, action, reward, next_state);
}

/**
 * @brief Perform Episodic Retrieve.
 *
 * @param cmem [out] Cmem value.
 * @param time_start [in] Time Start value.
 * @param time_end [in] Time End value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_retrieve(
    SNEPPXCognitiveMemory* cmem,
    uint64_t time_start,
    uint64_t time_end,
    SNEPPXTensor** output) {
    if (!cmem || !cmem->episodic || !output) return -1;
    return sneppx_episodic_retrieve(cmem->episodic, time_start, time_end, output);
}

/**
 * @brief Perform Episodic Sample.
 *
 * @param cmem [out] Cmem value.
 * @param batch_size [in] Batch Size value.
 * @param states [out] States value.
 * @param actions [out] Actions value.
 * @param rewards [out] Rewards value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_sample(
    SNEPPXCognitiveMemory* cmem,
    size_t batch_size,
    SNEPPXTensor** states,
    SNEPPXTensor** actions,
    float* rewards,
    SNEPPXTensor** next_states) {
    if (!cmem || !cmem->episodic) return -1;
    return sneppx_episodic_sample(cmem->episodic, batch_size, states,
                                  actions, rewards, next_states);
}

/**
 * @brief Perform Episodic Consolidate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_consolidate(SNEPPXCognitiveMemory* cmem) {
    if (!cmem || !cmem->episodic) return -1;
    return sneppx_episodic_consolidate(cmem->episodic);
}

/**
 * @brief Perform Semantic Store.
 *
 * @param cmem [out] Cmem value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_store(
    SNEPPXCognitiveMemory* cmem,
    const char* key,
    const SNEPPXTensor* embedding) {
    if (!cmem || !cmem->semantic) return -1;
    return sneppx_semantic_store(cmem->semantic, key, embedding);
}

/**
 * @brief Perform Semantic Retrieve.
 *
 * @param cmem [out] Cmem value.
 * @param query [in] Query value.
 * @param top_k [in] Top K value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_retrieve(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* query,
    size_t top_k,
    SNEPPXTensor** output,
    char*** out_keys) {
    if (!cmem || !cmem->semantic || !output) return -1;
    return sneppx_semantic_retrieve(cmem->semantic, query, top_k, output, out_keys);
}

/**
 * @brief Perform Semantic Relate.
 *
 * @param cmem [out] Cmem value.
 * @param from_key [in] From Key value.
 * @param to_key [in] To Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_relate(
    SNEPPXCognitiveMemory* cmem,
    const char* from_key,
    const char* to_key,
    float weight) {
    if (!cmem || !cmem->semantic) return -1;
    return sneppx_semantic_relate(cmem->semantic, from_key, to_key, weight);
}

/**
 * @brief Get Semantic For.
 *
 * @param cmem [out] Cmem value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_forget(
    SNEPPXCognitiveMemory* cmem,
    const char* key) {
    if (!cmem || !cmem->semantic) return -1;
    return sneppx_semantic_forget(cmem->semantic, key);
}

/**
 * @brief Write Working.
 *
 * @param cmem [out] Cmem value.
 * @param slot_index [in] Slot Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_write(
    SNEPPXCognitiveMemory* cmem,
    size_t slot_index,
    const SNEPPXTensor* content) {
    if (!cmem || !cmem->working) return -1;
    return sneppx_working_write(cmem->working, slot_index, content);
}

/**
 * @brief Read Working.
 *
 * @param cmem [out] Cmem value.
 * @param slot_index [in] Slot Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_read(
    SNEPPXCognitiveMemory* cmem,
    size_t slot_index,
    SNEPPXTensor** output) {
    if (!cmem || !cmem->working || !output) return -1;
    return sneppx_working_read(cmem->working, slot_index, output);
}

/**
 * @brief Perform Working Attend.
 *
 * @param cmem [out] Cmem value.
 * @param query [in] Query value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_attend(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* query,
    SNEPPXTensor** output) {
    if (!cmem || !cmem->working || !query || !output) return -1;
    return sneppx_working_attend(cmem->working, query, output);
}

/**
 * @brief Clear Working.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_clear(SNEPPXCognitiveMemory* cmem) {
    if (!cmem || !cmem->working) return -1;
    return sneppx_working_clear(cmem->working);
}

/**
 * @brief Perform Procedural Learn.
 *
 * @param cmem [out] Cmem value.
 * @param state [in] State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_procedural_learn(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    const SNEPPXTensor* skill) {
    if (!cmem || !cmem->procedural) return -1;
    return sneppx_procedural_learn(cmem->procedural, state, skill);
}

/**
 * @brief Perform Procedural Recall.
 *
 * @param cmem [out] Cmem value.
 * @param state [in] State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_procedural_recall(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    SNEPPXTensor** output) {
    if (!cmem || !cmem->procedural || !state || !output) return -1;
    return sneppx_procedural_recall(cmem->procedural, state, output);
}

/**
 * @brief Perform Procedural Compile.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_procedural_compile(SNEPPXCognitiveMemory* cmem) {
    if (!cmem || !cmem->procedural) return -1;
    return sneppx_procedural_compile(cmem->procedural);
}

/**
 * @brief Perform Cognitive Memory Get Params.
 *
 * @param cmem [in] Cmem value.
 * @param out [out] Out value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_cognitive_memory_get_params(
    const SNEPPXCognitiveMemory* cmem,
    SNEPPXTensor** out,
    size_t max) {
    (void)cmem;
    if (out && max > 0) {
        out[0] = NULL;
    }
    return 0;
}

/**
 * @brief Perform Cognitive Memory Build Train Graph.
 *
 * @param cmem [out] Cmem value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cognitive_memory_build_train_graph(
    SNEPPXCognitiveMemory* cmem,
    SNEPPXTape* tape,
    SNEPPXVariable* input_var,
    SNEPPXVariable** weight_vars,
    size_t num_weights,
    SNEPPXVariable** output_var) {
    if (!cmem || !input_var || !output_var) return -1;
    (void)tape;
    (void)weight_vars;
    (void)num_weights;
    SNEPPXTensor* raw = NULL;
    int ret = SNEPPX_cognitive_memory_forward(cmem, input_var->data, &raw);
    if (ret != 0 || !raw) return -1;
    SNEPPXVariable* var = SNEPPX_variable_create(raw, 0);
    if (!var) { SNEPPX_tensor_destroy(raw); return -1; }
    *output_var = var;
    return 0;
}

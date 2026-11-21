#ifndef SNEPPX_COGNITIVE_MEMORY_H
#define SNEPPX_COGNITIVE_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "multidimensional_tensor_engine.h"
#include "automatic_differentiation_framework.h"
#include <stddef.h>
#include <stdint.h>

#define SNEPPX_WORKING_MEMORY_MIN_SLOTS 5
#define SNEPPX_WORKING_MEMORY_MAX_SLOTS 9
#define SNEPPX_WORKING_MEMORY_DEFAULT_SLOTS 7
#define SNEPPX_MAX_CONCEPT_KEY_LEN 128
/*
 * SNEPPX - Cognitive Memory
 *
 * WHAT
 *   Cognitive Memory.
 *
 * CONCEPT
 *   Cognitive memory configuration and API for the SNEPPX-Algo system.
 *
 * ROLE
 *   Declares the cognitive memory API for persistent memory across inference sessions.
 *
 * REFERENCES
 *   None (internal module).
 */



typedef struct {
    size_t capacity;
    size_t max_priority_samples;
    uint64_t consolidation_interval;
} SNEPPXEpisodicConfig;

typedef struct {
    size_t max_concepts;
    size_t max_edges;
    size_t embedding_dim;
    size_t lru_size;
} SNEPPXSemanticConfig;

typedef struct {
    size_t num_slots;
    size_t slot_dim;
} SNEPPXWorkingConfig;

typedef struct {
    size_t cache_size;
    size_t state_dim;
    size_t skill_dim;
    size_t compilation_threshold;
} SNEPPXProceduralConfig;

typedef struct {
    SNEPPXEpisodicConfig episodic;
    SNEPPXSemanticConfig semantic;
    SNEPPXWorkingConfig working;
    SNEPPXProceduralConfig procedural;
} SNEPPXCognitiveMemoryConfig;

typedef struct SNEPPXCognitiveMemory SNEPPXCognitiveMemory;

/**
 * @brief Perform Cognitive Memory Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXCognitiveMemoryConfig SNEPPX_cognitive_memory_config_default(void);

/**
 * @brief Create Cognitive Memory.
 *
 * @param config [in] Config value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXCognitiveMemory* SNEPPX_cognitive_memory_create(
    const SNEPPXCognitiveMemoryConfig* config, unsigned int seed);
/**
 * @brief Destroy Cognitive Memory.
 *
 * @param cmem [out] Cmem value.
 */
void SNEPPX_cognitive_memory_destroy(SNEPPXCognitiveMemory* cmem);

/**
 * @brief Run the forward pass for Cognitive Memory.
 *
 * @param cmem [out] Cmem value.
 * @param input [in] Input value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cognitive_memory_forward(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* input,
    SNEPPXTensor** output);

/**
 * @brief Perform Episodic Record.
 *
 * @param cmem [out] Cmem value.
 * @param state [in] State value.
 * @param action [in] Action value.
 * @param reward [in] Reward value.
 * @param next_state [in] Next State value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_record(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    const SNEPPXTensor* action,
    float reward,
    const SNEPPXTensor* next_state);

/**
 * @brief Perform Episodic Retrieve.
 *
 * @param cmem [out] Cmem value.
 * @param time_start [in] Time Start value.
 * @param time_end [in] Time End value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_retrieve(
    SNEPPXCognitiveMemory* cmem,
    uint64_t time_start,
    uint64_t time_end,
    SNEPPXTensor** output);

/**
 * @brief Perform Episodic Sample.
 *
 * @param cmem [out] Cmem value.
 * @param batch_size [in] Batch Size value.
 * @param states [out] States value.
 * @param actions [out] Actions value.
 * @param rewards [out] Rewards value.
 * @param next_states [out] Next States value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_sample(
    SNEPPXCognitiveMemory* cmem,
    size_t batch_size,
    SNEPPXTensor** states,
    SNEPPXTensor** actions,
    float* rewards,
    SNEPPXTensor** next_states);

/**
 * @brief Perform Episodic Consolidate.
 *
 * @param cmem [out] Cmem value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_episodic_consolidate(SNEPPXCognitiveMemory* cmem);

/**
 * @brief Perform Semantic Store.
 *
 * @param cmem [out] Cmem value.
 * @param key [in] Key value.
 * @param embedding [in] Embedding value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_store(
    SNEPPXCognitiveMemory* cmem,
    const char* key,
    const SNEPPXTensor* embedding);

/**
 * @brief Perform Semantic Retrieve.
 *
 * @param cmem [out] Cmem value.
 * @param query [in] Query value.
 * @param top_k [in] Top K value.
 * @param output [out] Output value.
 * @param out_keys [out] Out Keys value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_retrieve(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* query,
    size_t top_k,
    SNEPPXTensor** output,
    char*** out_keys);

/**
 * @brief Perform Semantic Relate.
 *
 * @param cmem [out] Cmem value.
 * @param from_key [in] From Key value.
 * @param to_key [in] To Key value.
 * @param weight [in] Weight value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_relate(
    SNEPPXCognitiveMemory* cmem,
    const char* from_key,
    const char* to_key,
    float weight);

/**
 * @brief Get Semantic For.
 *
 * @param cmem [out] Cmem value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_semantic_forget(
    SNEPPXCognitiveMemory* cmem,
    const char* key);

/**
 * @brief Write Working.
 *
 * @param cmem [out] Cmem value.
 * @param slot_index [in] Slot Index value.
 * @param content [in] Content value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_write(
    SNEPPXCognitiveMemory* cmem,
    size_t slot_index,
    const SNEPPXTensor* content);

/**
 * @brief Read Working.
 *
 * @param cmem [out] Cmem value.
 * @param slot_index [in] Slot Index value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_read(
    SNEPPXCognitiveMemory* cmem,
    size_t slot_index,
    SNEPPXTensor** output);

/**
 * @brief Perform Working Attend.
 *
 * @param cmem [out] Cmem value.
 * @param query [in] Query value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_attend(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* query,
    SNEPPXTensor** output);

/**
 * @brief Clear Working.
 *
 * @param cmem [out] Cmem value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_working_clear(SNEPPXCognitiveMemory* cmem);

/**
 * @brief Perform Procedural Learn.
 *
 * @param cmem [out] Cmem value.
 * @param state [in] State value.
 * @param skill [in] Skill value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_procedural_learn(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    const SNEPPXTensor* skill);

/**
 * @brief Perform Procedural Recall.
 *
 * @param cmem [out] Cmem value.
 * @param state [in] State value.
 * @param output [out] Output value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_procedural_recall(
    SNEPPXCognitiveMemory* cmem,
    const SNEPPXTensor* state,
    SNEPPXTensor** output);

/**
 * @brief Perform Procedural Compile.
 *
 * @param cmem [out] Cmem value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_procedural_compile(SNEPPXCognitiveMemory* cmem);

/**
 * @brief Perform Cognitive Memory Get Params.
 *
 * @param cmem [in] Cmem value.
 * @param out [out] Out value.
 * @param max [in] Max value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_cognitive_memory_get_params(
    const SNEPPXCognitiveMemory* cmem,
    SNEPPXTensor** out,
    size_t max);

/**
 * @brief Perform Cognitive Memory Build Train Graph.
 *
 * @param cmem [out] Cmem value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cognitive_memory_build_train_graph(
    SNEPPXCognitiveMemory* cmem,
    SNEPPXTape* tape,
    SNEPPXVariable* input_var,
    SNEPPXVariable** weight_vars,
    size_t num_weights,
    SNEPPXVariable** output_var);


#ifdef __cplusplus
}
#endif
#endif

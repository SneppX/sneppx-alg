#include "cognitive_memory.h"
#include "polymorphic_memory_allocator.h"
#include "test_gtest.h"
#include <string.h>
#include <math.h>

/*
 * SNEPPX - Test Cognitive Memory
 *
 * WHAT
 *   Test Cognitive Memory.
 *
 * CONCEPT
 *   Provides memory management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_config_defaults(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    SX_ASSERT_EQ(cfg.episodic.capacity, 1000, "episodic capacity");
    SX_ASSERT_EQ(cfg.semantic.max_concepts, 100, "semantic max concepts");
    SX_ASSERT_EQ(cfg.semantic.embedding_dim, 64, "semantic embedding dim");
    SX_ASSERT_EQ(cfg.working.num_slots, 7, "working num slots");
    SX_ASSERT_EQ(cfg.working.slot_dim, 64, "working slot dim");
    SX_ASSERT_EQ(cfg.procedural.cache_size, 50, "procedural cache size");
    SX_ASSERT_EQ(cfg.procedural.compilation_threshold, 5, "compilation threshold");
}

static void test_create_destroy(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create cognitive memory");
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_forward_pass(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.semantic.embedding_dim = 8;
    cfg.working.slot_dim = 8;
    cfg.procedural.skill_dim = 8;
    cfg.procedural.state_dim = 8;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for forward test");
    size_t shape[2] = {1, 8};
    SNEPPXTensor* input = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(input, "create input tensor");
    SNEPPXTensor* output = NULL;
    int ret = SNEPPX_cognitive_memory_forward(cmem, input, &output);
    SX_ASSERT_EQ(ret, 0, "forward pass success");
    SX_ASSERT_NOT_NULL(output, "forward output not null");
    SX_ASSERT_EQ(output->size, 32, "output size (8+8+8+8)");
    SNEPPX_tensor_destroy(output);
    SNEPPX_tensor_destroy(input);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_episodic_record_retrieve(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.episodic.capacity = 10;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for episodic test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* s1 = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* s2 = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(s1, "create state1");
    SX_ASSERT_NOT_NULL(s2, "create state2");
    int ret = SNEPPX_episodic_record(cmem, s1, NULL, 1.0f, s2);
    SX_ASSERT_EQ(ret, 0, "record experience");
    ret = SNEPPX_episodic_record(cmem, s2, NULL, 0.5f, s1);
    SX_ASSERT_EQ(ret, 0, "record second experience");
    SNEPPXTensor* retrieved = NULL;
    ret = SNEPPX_episodic_retrieve(cmem, 0, 10, &retrieved);
    SX_ASSERT_EQ(ret, 0, "retrieve success");
    SX_ASSERT_NOT_NULL(retrieved, "retrieved tensor not null");
    SX_ASSERT_EQ(retrieved->shape[0], 2, "retrieved 2 experiences");
    SNEPPX_tensor_destroy(retrieved);
    SNEPPX_tensor_destroy(s1);
    SNEPPX_tensor_destroy(s2);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_episodic_sample(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.episodic.capacity = 20;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for sample test");
    size_t shape[2] = {1, 4};
    for (int i = 0; i < 10; i++) {
        SNEPPXTensor* s = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, SX_ARR_C(float, 1, (float)i));
        SNEPPXTensor* ns = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, SX_ARR_C(float, 1, (float)(i + 1)));
        SNEPPX_episodic_record(cmem, s, NULL, (float)i * 0.1f, ns);
        SNEPPX_tensor_destroy(s);
        SNEPPX_tensor_destroy(ns);
    }
    SNEPPXTensor* states[5];
    float rewards[5];
    int ret = SNEPPX_episodic_sample(cmem, 5, states, NULL, rewards, NULL);
    SX_ASSERT(ret > 0, "sampled at least some experiences");
    for (int i = 0; i < ret; i++) {
        SX_ASSERT_NOT_NULL(states[i], "sampled state not null");
        SX_ASSERT(rewards[i] >= 0.0f, "reward non-negative");
        SNEPPX_tensor_destroy(states[i]);
    }
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_semantic_store_retrieve(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.semantic.max_concepts = 10;
    cfg.semantic.embedding_dim = 4;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for semantic test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* emb1 = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* emb2 = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SX_ASSERT_NOT_NULL(emb1, "create emb1");
    SX_ASSERT_NOT_NULL(emb2, "create emb2");
    int ret = SNEPPX_semantic_store(cmem, "cat", emb1);
    SX_ASSERT_EQ(ret, 0, "store cat");
    ret = SNEPPX_semantic_store(cmem, "dog", emb2);
    SX_ASSERT_EQ(ret, 0, "store dog");
    SNEPPXTensor* query = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* results = NULL;
    char** keys = NULL;
    ret = SNEPPX_semantic_retrieve(cmem, query, 3, &results, &keys);
    SX_ASSERT(ret > 0, "retrieved at least one concept");
    SX_ASSERT_NOT_NULL(results, "results tensor not null");
    SX_ASSERT_STREQ(keys[0], "cat", "closest concept is cat");
    SNEPPX_tensor_destroy(results);
    for (int i = 0; i < ret; i++) SNEPPX_free(keys[i], SNEPPX_MAX_CONCEPT_KEY_LEN);
    SNEPPX_free(keys, (size_t)ret * sizeof(char*));
    SNEPPX_tensor_destroy(query);
    SNEPPX_tensor_destroy(emb1);
    SNEPPX_tensor_destroy(emb2);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_semantic_relate_forget(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.semantic.max_concepts = 10;
    cfg.semantic.embedding_dim = 4;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for relate test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* emb = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPX_semantic_store(cmem, "a", emb);
    SNEPPX_semantic_store(cmem, "b", emb);
    int ret = SNEPPX_semantic_relate(cmem, "a", "b", 0.5f);
    SX_ASSERT_EQ(ret, 0, "relate a->b");
    ret = SNEPPX_semantic_forget(cmem, "a");
    SX_ASSERT_EQ(ret, 0, "forget a");
    SNEPPXTensor* results = NULL;
    ret = SNEPPX_semantic_retrieve(cmem, emb, 5, &results, NULL);
    SX_ASSERT(ret > 0, "still has concepts after forget");
    SNEPPX_tensor_destroy(results);
    SNEPPX_tensor_destroy(emb);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_working_write_read(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.working.num_slots = 5;
    cfg.working.slot_dim = 4;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for working test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* content = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    int ret = SNEPPX_working_write(cmem, 0, content);
    SX_ASSERT_EQ(ret, 0, "write slot 0");
    SNEPPXTensor* readback = NULL;
    ret = SNEPPX_working_read(cmem, 0, &readback);
    SX_ASSERT_EQ(ret, 0, "read slot 0");
    SX_ASSERT_NOT_NULL(readback, "readback not null");
    SX_ASSERT_NEAR(SNEPPX_tensor_get_f32(readback, SX_ARR_C(size_t, 2, 0,0)), 1.0f, 0.001f, "slot content");
    SNEPPX_tensor_destroy(readback);
    SNEPPX_tensor_destroy(content);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_working_attend(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.working.num_slots = 3;
    cfg.working.slot_dim = 4;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for attend test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* slot0 = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* slot1 = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    SNEPPX_working_write(cmem, 0, slot0);
    SNEPPX_working_write(cmem, 1, slot1);
    SNEPPXTensor* query = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* attended = NULL;
    int ret = SNEPPX_working_attend(cmem, query, &attended);
    SX_ASSERT_EQ(ret, 0, "attend success");
    SX_ASSERT_NOT_NULL(attended, "attended output not null");
    float val = SNEPPX_tensor_get_f32(attended, SX_ARR_C(size_t, 2, 0,0));
    SX_ASSERT(val > 0.0f, "attended value positive");
    SNEPPX_tensor_destroy(attended);
    SNEPPX_tensor_destroy(query);
    SNEPPX_tensor_destroy(slot0);
    SNEPPX_tensor_destroy(slot1);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_working_clear(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.working.num_slots = 3;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for clear test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* content = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPX_working_write(cmem, 0, content);
    int ret = SNEPPX_working_clear(cmem);
    SX_ASSERT_EQ(ret, 0, "clear success");
    SNEPPXTensor* readback = NULL;
    SNEPPX_working_read(cmem, 0, &readback);
    SX_ASSERT_NULL(readback, "read after clear returns null");
    SNEPPX_tensor_destroy(content);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_procedural_learn_recall(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.procedural.cache_size = 5;
    cfg.procedural.state_dim = 4;
    cfg.procedural.skill_dim = 4;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for procedural test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* state = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* skill = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, SX_ARR_C(float, 1, 2.0f));
    int ret = SNEPPX_procedural_learn(cmem, state, skill);
    SX_ASSERT_EQ(ret, 0, "learn skill");
    SNEPPXTensor* recalled = NULL;
    ret = SNEPPX_procedural_recall(cmem, state, &recalled);
    SX_ASSERT_EQ(ret, 0, "recall skill");
    SX_ASSERT_NOT_NULL(recalled, "recalled skill not null");
    SX_ASSERT_NEAR(SNEPPX_tensor_get_f32(recalled, SX_ARR_C(size_t, 2, 0,0)), 2.0f, 0.001f, "skill output");
    SNEPPX_tensor_destroy(recalled);
    SNEPPXTensor* unknown = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    ret = SNEPPX_procedural_recall(cmem, unknown, &recalled);
    SX_ASSERT_EQ(ret, 0, "recall unknown state returns success");
    SX_ASSERT_NULL(recalled, "unknown state returns null");
    SNEPPX_tensor_destroy(unknown);
    SNEPPX_tensor_destroy(state);
    SNEPPX_tensor_destroy(skill);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_procedural_compile(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.procedural.cache_size = 5;
    cfg.procedural.state_dim = 4;
    cfg.procedural.skill_dim = 4;
    cfg.procedural.compilation_threshold = 3;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for compile test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* state = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* skill = SNEPPX_tensor_zeros(shape, 2, SNEPPX_FLOAT32);
    for (int i = 0; i < 5; i++) {
        SNEPPX_procedural_learn(cmem, state, skill);
    }
    int compiled = SNEPPX_procedural_compile(cmem);
    SX_ASSERT(compiled > 0, "compiled at least one skill");
    SNEPPX_tensor_destroy(state);
    SNEPPX_tensor_destroy(skill);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_consolidation(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.episodic.capacity = 5;
    cfg.episodic.consolidation_interval = 2;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for consolidation test");
    size_t shape[2] = {1, 2};
    for (int i = 0; i < 5; i++) {
        SNEPPXTensor* s = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, SX_ARR_C(float, 1, (float)i));
        SNEPPXTensor* ns = SNEPPX_tensor_full(shape, 2, SNEPPX_FLOAT32, SX_ARR_C(float, 1, (float)(i + 1)));
        SNEPPX_episodic_record(cmem, s, NULL, (float)i * 0.2f, ns);
        SNEPPX_tensor_destroy(s);
        SNEPPX_tensor_destroy(ns);
    }
    int ret = SNEPPX_episodic_consolidate(cmem);
    SX_ASSERT_EQ(ret, 0, "consolidate success");
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_build_train_graph(void) {
    SNEPPXCognitiveMemoryConfig cfg = SNEPPX_cognitive_memory_config_default();
    cfg.semantic.embedding_dim = 4;
    cfg.working.slot_dim = 4;
    cfg.procedural.skill_dim = 4;
    cfg.procedural.state_dim = 4;
    SNEPPXCognitiveMemory* cmem = SNEPPX_cognitive_memory_create(&cfg, 42);
    SX_ASSERT_NOT_NULL(cmem, "create for train graph test");
    size_t shape[2] = {1, 4};
    SNEPPXTensor* data = SNEPPX_tensor_ones(shape, 2, SNEPPX_FLOAT32);
    SNEPPXVariable* input_var = SNEPPX_variable_create(data, 0);
    SX_ASSERT_NOT_NULL(input_var, "input variable");
    SNEPPXTape* tape = SNEPPX_tape_create();
    SNEPPXVariable* output_var = NULL;
    int ret = SNEPPX_cognitive_memory_build_train_graph(cmem, tape, input_var, NULL, 0, &output_var);
    SX_ASSERT_EQ(ret, 0, "build train graph success");
    SX_ASSERT_NOT_NULL(output_var, "output variable not null");
    SX_ASSERT_EQ(output_var->data->size, 16, "output size (4+4+4+4)");
    SNEPPX_variable_destroy(output_var);
    SNEPPX_tape_destroy(tape);
    SNEPPX_variable_destroy(input_var);
    SNEPPX_cognitive_memory_destroy(cmem);
}

static void test_edge_cases(void) {
    SX_ASSERT_NULL(SNEPPX_cognitive_memory_create(NULL, 0), "null config returns null");
    int ret = SNEPPX_cognitive_memory_forward(NULL, NULL, NULL);
    SX_ASSERT_EQ(ret, -1, "null forward returns -1");
    ret = SNEPPX_episodic_record(NULL, NULL, NULL, 0, NULL);
    SX_ASSERT_EQ(ret, -1, "null record returns -1");
    ret = SNEPPX_semantic_store(NULL, NULL, NULL);
    SX_ASSERT_EQ(ret, -1, "null semantic store returns -1");
    ret = SNEPPX_working_write(NULL, 0, NULL);
    SX_ASSERT_EQ(ret, -1, "null working write returns -1");
    ret = SNEPPX_procedural_learn(NULL, NULL, NULL);
    SX_ASSERT_EQ(ret, -1, "null procedural learn returns -1");
}


TEST(test_cognitive_memory, config_defaults) { test_config_defaults(); }
TEST(test_cognitive_memory, create_destroy) { test_create_destroy(); }
TEST(test_cognitive_memory, forward_pass) { test_forward_pass(); }
TEST(test_cognitive_memory, episodic_record_retrieve) { test_episodic_record_retrieve(); }
TEST(test_cognitive_memory, episodic_sample) { test_episodic_sample(); }
TEST(test_cognitive_memory, semantic_store_retrieve) { test_semantic_store_retrieve(); }
TEST(test_cognitive_memory, semantic_relate_forget) { test_semantic_relate_forget(); }
TEST(test_cognitive_memory, working_write_read) { test_working_write_read(); }
TEST(test_cognitive_memory, working_attend) { test_working_attend(); }
TEST(test_cognitive_memory, working_clear) { test_working_clear(); }
TEST(test_cognitive_memory, procedural_learn_recall) { test_procedural_learn_recall(); }
TEST(test_cognitive_memory, procedural_compile) { test_procedural_compile(); }
TEST(test_cognitive_memory, consolidation) { test_consolidation(); }
TEST(test_cognitive_memory, build_train_graph) { test_build_train_graph(); }
TEST(test_cognitive_memory, edge_cases) { test_edge_cases(); }

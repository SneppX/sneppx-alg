#include <neural_core/kernel/model_config.h>
#include <neural_core/model_zoo/registry.h>
#include <neural_core/model_zoo/weights.h>
#include <neural_core/model_zoo/model_card.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Integration
 *
 * WHAT
 *   Test Integration.
 *
 * CONCEPT
 *   Provides the Test Integration.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0;
static int tests_total = 0;

#define SNEPPX_TEST(name) do { printf("test_%s... ", name); tests_total++; } while(0)
#define SNEPPX_PASS() do { printf("PASS\n"); tests_passed++; } while(0)

/* ── 1. Config → JSON → Registry roundtrip ─────────────────────────────── */

void test_config_registry_integration(void) {
    SNEPPX_TEST("config_registry_integration");

    ModelConfig *cfg = model_config_llama2_7b();
    assert(cfg != NULL);

    /* Serialize to JSON */
    char *json = model_config_to_json(cfg, 1);
    assert(json != NULL);
    assert(strstr(json, "4096") != NULL);

    /* Parse back */
    ModelConfig *cfg2 = model_config_from_json(json);
    assert(cfg2 != NULL);
    assert(cfg2->hidden_size == 4096);
    assert(cfg2->num_layers == 32);
    assert(cfg2->num_heads == 32);
    assert(cfg2->vocab_size == 32000);

    /* Register in registry */
    ModelRegistry *reg = model_registry_create();
    int rc = model_registry_register(reg, "llama2-7b", "1.0.0", "transformer",
                                     "LLaMA 2 7B", "Meta", "MIT",
                                     "", "", "", 1);
    assert(rc == 0);

    /* Look up by name */
    ModelRegistryEntry *entry = model_registry_get(reg, "llama2-7b", "1.0.0");
    assert(entry != NULL);
    assert(strcmp(entry->name, "llama2-7b") == 0);
    assert(strcmp(entry->architecture, "transformer") == 0);

    /* Search */
    int count = 0;
    ModelRegistryEntry **results = model_registry_search(reg, "llama", &count);
    assert(count > 0);
    free(results);

    /* Deprecate */
    model_registry_deprecate(reg, "llama2-7b", "1.0.0", "superseded by llama3");
    assert(model_registry_is_deprecated(entry) != 0);

    /* Cleanup */
    model_registry_destroy(reg);
    model_config_destroy(cfg);
    model_config_destroy(cfg2);
    free(json);

    SNEPPX_PASS();
}

/* ── 2. Weights + Config integration ───────────────────────────────────── */

void test_weights_config_integration(void) {
    SNEPPX_TEST("weights_config_integration");

    WeightCollection *wc = weight_collection_create();
    assert(wc != NULL);

    int64_t shape[] = {4, 4};
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int rc = weight_collection_add(wc, "test.weight", shape, 2, "f32", data, sizeof(data), 0);
    assert(rc == 0);
    assert(wc->count == 1);

    WeightTensor *t = weight_collection_get(wc, "test.weight");
    assert(t != NULL);
    assert(t->ndim == 2);

    int count = 0;
    char **names = weight_collection_names(wc, &count);
    assert(count == 1);
    free(names[0]);
    free(names);

    weight_collection_destroy(wc);
    SNEPPX_PASS();
}

/* ── 3. Model card + Config integration ────────────────────────────────── */

void test_model_card_config_integration(void) {
    SNEPPX_TEST("model_card_config_integration");

    ModelCard *card = model_card_create();
    assert(card != NULL);

    model_card_set_name(card, "integration-test");
    model_card_set_version(card, "1.0");
    model_card_set_architecture(card, "transformer");
    model_card_add_tag(card, "integration");
    model_card_set_num_parameters(card, 1000000);

    assert(strcmp(card->name, "integration-test") == 0);
    assert(card->num_tags == 1);

    model_card_destroy(card);
    SNEPPX_PASS();
}

/* ── 4. Global registry singleton ──────────────────────────────────────── */

void test_global_registry_integration(void) {
    SNEPPX_TEST("global_registry_integration");

    ModelRegistry *global = model_registry_global();
    assert(global != NULL);

    /* Register several models */
    model_registry_register(global, "llama2-7b", "1.0", "transformer",
                            "", "", "", "", "", "", 1);
    model_registry_register(global, "mistral-7b", "1.0", "transformer",
                            "", "", "", "", "", "", 1);

    /* List all */
    int count = 0;
    ModelRegistryEntry **all = model_registry_list(global, NULL, &count);
    assert(count >= 2);
    free(all);

    /* List by architecture */
    ModelRegistryEntry **archs = model_registry_list(global, "transformer", &count);
    assert(count >= 2);
    free(archs);

    SNEPPX_PASS();
}

/* ── 5. Quantization + Weights integration ─────────────────────────────── */

void test_quantize_weights_integration(void) {
    SNEPPX_TEST("quantize_weights_integration");

    WeightCollection *wc = weight_collection_create();
    assert(wc != NULL);

    int64_t shape[] = {64, 64};
    float *data = (float *)calloc(64 * 64, sizeof(float));
    for (int i = 0; i < 64 * 64; i++) data[i] = (float)i;

    weight_collection_add(wc, "weight", shape, 2, "f32", data, 64 * 64 * sizeof(float), 1);

    /* INT8 quantize */
    int rc = weight_tensor_quantize_int8(weight_collection_get(wc, "weight"));
    assert(rc == 0);

    /* Dequantize back to f32 */
    rc = weight_tensor_convert_dtype(weight_collection_get(wc, "weight"), "f32");
    assert(rc == 0);

    weight_collection_destroy(wc);

    SNEPPX_PASS();
}

/* ── 6. Full pipeline: config → card → weights → registry ─────────────── */

void test_full_pipeline(void) {
    SNEPPX_TEST("full_pipeline");

    /* 1. Create config */
    ModelConfig *cfg = model_config_qwen2_7b();
    assert(cfg != NULL);

    /* 2. Create model card */
    ModelCard *card = model_card_create();
    model_card_set_name(card, cfg->name);
    model_card_set_version(card, cfg->version);
    model_card_set_architecture(card, "transformer");
    model_card_set_num_parameters(card, cfg->hidden_size * cfg->num_layers * cfg->intermediate_size * 4);

    /* 3. Create weights (stub with 1 tensor) */
    WeightCollection *wc = weight_collection_create();
    int64_t shape[] = {cfg->hidden_size, cfg->hidden_size};
    float data[1] = {0.0f};
    weight_collection_add(wc, "model.embed_tokens.weight", shape, 2, "f32", data, sizeof(data), 0);

    /* 4. Register model */
    ModelRegistry *reg = model_registry_create();
    model_registry_register(reg, cfg->name, cfg->version, "transformer",
                            cfg->description ? cfg->description : "",
                            cfg->author ? cfg->author : "",
                            cfg->license ? cfg->license : "",
                            "", "", "", 1);

    ModelRegistryEntry *entry = model_registry_get(reg, cfg->name, cfg->version);
    assert(entry != NULL);
    assert(strcmp(entry->name, cfg->name) == 0);

    /* 5. Verify everything still valid */
    assert(wc->count == 1);
    assert(card->num_parameters > 0);
    assert(cfg->hidden_size == 4096);

    /* 6. Save/load config roundtrip */
    char *json = model_config_to_json(cfg, 0);
    ModelConfig *cfg2 = model_config_from_json(json);
    assert(cfg2->hidden_size == cfg->hidden_size);
    assert(cfg2->num_layers == cfg->num_layers);

    model_config_destroy(cfg2);
    free(json);
    model_registry_destroy(reg);
    weight_collection_destroy(wc);
    model_card_destroy(card);
    model_config_destroy(cfg);

    SNEPPX_PASS();
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n=== Phase 1.8 C Integration Tests ===\n\n");

    test_config_registry_integration();
    test_weights_config_integration();
    test_model_card_config_integration();
    test_global_registry_integration();
    test_full_pipeline();

    printf("\n=== Results: %d/%d tests passed! ===\n\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}

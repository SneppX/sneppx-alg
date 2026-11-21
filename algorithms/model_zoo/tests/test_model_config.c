#include "model_config.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

/*
 * SNEPPX - Test Model Config
 *
 * WHAT
 *   Test Model Config.
 *
 * CONCEPT
 *   Provides the Test Model Config.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


void test_create_destroy(void) {
    printf("test_create_destroy... ");
    ModelConfig *cfg = model_config_create();
    assert(cfg != NULL);
    model_config_destroy(cfg);
    printf("PASS\n");
}

void test_llama2_7b(void) {
    printf("test_llama2_7b... ");
    ModelConfig *cfg = model_config_llama2_7b();
    assert(cfg != NULL);
    assert(strcmp(cfg->name, "llama2-7b") == 0);
    assert(cfg->vocab_size == 32000);
    assert(cfg->hidden_size == 4096);
    assert(cfg->num_layers == 32);
    assert(cfg->num_heads == 32);
    assert(cfg->intermediate_size == 11008);
    assert(cfg->max_position_embeddings == 2048);
    assert(cfg->use_rms_norm == 1);
    assert(cfg->pos_encoding == POS_ENCODING_ROPE);
    model_config_destroy(cfg);
    printf("PASS\n");
}

void test_mistral_7b(void) {
    printf("test_mistral_7b... ");
    ModelConfig *cfg = model_config_mistral_7b();
    assert(cfg != NULL);
    assert(strcmp(cfg->name, "mistral-7b") == 0);
    assert(cfg->sliding_window == 4096);
    assert(cfg->num_kv_heads == 8);
    model_config_destroy(cfg);
    printf("PASS\n");
}

void test_json_roundtrip(void) {
    printf("test_json_roundtrip... ");
    ModelConfig *cfg = model_config_llama2_7b();
    char *json = model_config_to_json(cfg, 1);
    assert(json != NULL);
    assert(strstr(json, "llama2-7b") != NULL);
    assert(strstr(json, "vocab_size") != NULL);

    ModelConfig *cfg2 = model_config_from_json(json);
    assert(cfg2 != NULL);
    assert(strcmp(cfg2->name, "llama2-7b") == 0);
    assert(cfg2->vocab_size == 32000);
    assert(cfg2->hidden_size == 4096);
    assert(cfg2->num_layers == 32);

    free(json);
    model_config_destroy(cfg);
    model_config_destroy(cfg2);
    printf("PASS\n");
}

void test_file_io(void) {
    printf("test_file_io... ");
    ModelConfig *cfg = model_config_llama2_7b();
#ifdef _WIN32
    const char *path = "C:\\Temp\\test_model_config.json";
#else
    const char *path = "/tmp/test_model_config.json";
#endif

    int rc = model_config_save(cfg, path);
    assert(rc == 0);

    ModelConfig *cfg2 = model_config_load(path);
    assert(cfg2 != NULL);
    assert(strcmp(cfg2->name, "llama2-7b") == 0);
    assert(cfg2->vocab_size == 32000);

    remove(path);
    model_config_destroy(cfg);
    model_config_destroy(cfg2);
    printf("PASS\n");
}

void test_validation(void) {
    printf("test_validation... ");
    ModelConfig *cfg = model_config_create();
    char *error = NULL;

    // Valid config
    cfg->name = strdup("test");
    cfg->vocab_size = 32000;
    cfg->hidden_size = 4096;
    cfg->num_layers = 32;
    cfg->num_heads = 32;
    cfg->intermediate_size = 11008;
    cfg->max_position_embeddings = 2048;
    cfg->learning_rate = 2e-4f;
    assert(model_config_validate(cfg, &error) == 0);
    free(error);

    // Invalid: missing name
    free(cfg->name);
    cfg->name = NULL;
    assert(model_config_validate(cfg, &error) != 0);
    assert(error != NULL);
    free(error);

    // Invalid: hidden_size not divisible by num_heads
    cfg->name = strdup("test");
    cfg->num_heads = 33;
    assert(model_config_validate(cfg, &error) != 0);
    assert(error != NULL);
    free(error);

    model_config_destroy(cfg);
    printf("PASS\n");
}

void test_copy(void) {
    printf("test_copy... ");
    ModelConfig *cfg = model_config_llama2_7b();
    ModelConfig *copy = model_config_copy(cfg);
    assert(copy != NULL);
    assert(strcmp(copy->name, "llama2-7b") == 0);
    assert(copy->vocab_size == 32000);
    assert(copy->hidden_size == 4096);
    assert(copy->num_layers == 32);
    // Ensure deep copy
    free(copy->name);
    copy->name = strdup("copy");
    assert(strcmp(cfg->name, "llama2-7b") == 0);
    assert(strcmp(copy->name, "copy") == 0);
    model_config_destroy(cfg);
    model_config_destroy(copy);
    printf("PASS\n");
}

void test_merge(void) {
    printf("test_merge... ");
    ModelConfig *base = model_config_llama2_7b();
    ModelConfig *override = model_config_create();
    override->hidden_size = 5120;
    override->num_layers = 40;

    model_config_merge(base, override);
    assert(base->hidden_size == 5120);
    assert(base->num_layers == 40);
    // Original should be unchanged
    assert(strcmp(base->name, "llama2-7b") == 0);

    model_config_destroy(base);
    model_config_destroy(override);
    printf("PASS\n");
}

int main(void) {
    test_create_destroy();
    test_llama2_7b();
    test_mistral_7b();
    test_json_roundtrip();
    test_file_io();
    test_validation();
    test_copy();
    test_merge();
    printf("\nAll tests passed!\n");
    return 0;
}

#include "model_config.h"
#include <neural_core/model_zoo/registry.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/*
 * SNEPPX - Test Model Registry
 *
 * WHAT
 *   Test Model Registry.
 *
 * CONCEPT
 *   Provides the Test Model Registry.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


void test_create_destroy(void) {
    printf("test_create_destroy... ");
    ModelRegistry *r = model_registry_create();
    assert(r != NULL);
    model_registry_destroy(r);
    printf("PASS\n");
}

void test_register_unregister(void) {
    printf("test_register_unregister... ");
    ModelRegistry *r = model_registry_create();

    int rc = model_registry_register(r, "llama2", "7b", "transformer",
                                    "LLaMA 2 7B", "Meta AI", "llama2-community",
                                    "https://hf.co/meta-llama/Llama-2-7b",
                                    "/config/llama2-7b.json", "/weights/llama2-7b.safetensors", 0);
    assert(rc == 0);

    ModelRegistryEntry *e = model_registry_get(r, "llama2", "7b");
    assert(e != NULL);
    assert(strcmp(e->name, "llama2") == 0);
    assert(strcmp(e->version, "7b") == 0);
    assert(strcmp(e->architecture, "transformer") == 0);

    // Duplicate registration should fail
    rc = model_registry_register(r, "llama2", "7b", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    assert(rc != 0);

    // Unregister
    rc = model_registry_unregister(r, "llama2", "7b");
    assert(rc == 0);
    e = model_registry_get(r, "llama2", "7b");
    assert(e == NULL);

    model_registry_destroy(r);
    printf("PASS\n");
}

void test_get_latest(void) {
    printf("test_get_latest... ");
    ModelRegistry *r = model_registry_create();

    model_registry_register(r, "mistral", "7b-v0.1", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "mistral", "7b-v0.2", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "mistral", "7b-v0.3", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);

    ModelRegistryEntry *latest = model_registry_get_latest(r, "mistral");
    assert(latest != NULL);
    assert(strcmp(latest->version, "7b-v0.3") == 0);

    model_registry_destroy(r);
    printf("PASS\n");
}

void test_list_by_architecture(void) {
    printf("test_list_by_architecture... ");
    ModelRegistry *r = model_registry_create();

    model_registry_register(r, "llama2", "7b", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "llama2", "13b", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "vit", "base", "vit", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "bert", "base", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);

    int count = 0;
    ModelRegistryEntry **entries = model_registry_list(r, "transformer", &count);
    assert(count == 3);
    for (int i = 0; i < count; i++) {
        assert(strcmp(entries[i]->architecture, "transformer") == 0);
    }
    free(entries);

    entries = model_registry_list(r, "vit", &count);
    assert(count == 1);
    assert(strcmp(entries[0]->name, "vit") == 0);
    free(entries);

    model_registry_destroy(r);
    printf("PASS\n");
}

void test_search(void) {
    printf("test_search... ");
    ModelRegistry *r = model_registry_create();

    model_registry_register(r, "llama2-7b", "7b", "transformer", "LLaMA 2 7B model", "Meta", NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "mistral-7b", "7b", "transformer", "Mistral 7B model", "Mistral AI", NULL, NULL, NULL, NULL, 0);
    model_registry_register(r, "vit-base", "base", "vit", "Vision Transformer Base", "Google", NULL, NULL, NULL, NULL, 0);

    int count = 0;
    ModelRegistryEntry **entries = model_registry_search(r, "7b", &count);
    assert(count == 2);
    free(entries);

    entries = model_registry_search(r, "Meta", &count);
    assert(count == 1);
    free(entries);

    entries = model_registry_search(r, "nonexistent", &count);
    assert(count == 0);
    free(entries);

    model_registry_destroy(r);
    printf("PASS\n");
}

void test_save_load(void) {
    printf("test_save_load... ");
    ModelRegistry *r = model_registry_create();

    model_registry_register(r, "test-model", "1.0", "transformer", "Test model", "Test", "MIT",
                           "https://example.com", "/config.json", "/weights.safetensors", 1);

#ifdef _WIN32
    const char *path = "C:\\Temp\\test_registry.txt";
#else
    const char *path = "/tmp/test_registry.txt";
#endif
    int rc = model_registry_save(r, path);
    assert(rc == 0);

    ModelRegistry *r2 = model_registry_load(path);
    assert(r2 != NULL);
    assert(r2->count == 1);

    ModelRegistryEntry *e = model_registry_get(r2, "test-model", "1.0");
    assert(e != NULL);
    assert(strcmp(e->name, "test-model") == 0);
    assert(e->is_local == 1);

    remove(path);
    model_registry_destroy(r);
    model_registry_destroy(r2);
    printf("PASS\n");
}

void test_deprecate(void) {
    printf("test_deprecate... ");
    ModelRegistry *r = model_registry_create();

    model_registry_register(r, "old-model", "1.0", "transformer", NULL, NULL, NULL, NULL, NULL, NULL, 0);
    model_registry_deprecate(r, "old-model", "1.0", "Use new-model v2.0 instead");

    ModelRegistryEntry *e = model_registry_get(r, "old-model", "1.0");
    assert(e != NULL);
    assert(e->deprecated == 1);
    assert(strcmp(e->deprecated_reason, "Use new-model v2.0 instead") == 0);

    model_registry_destroy(r);
    printf("PASS\n");
}

void test_global_registry(void) {
    printf("test_global_registry... ");
    ModelRegistry *r1 = model_registry_global();
    ModelRegistry *r2 = model_registry_global();
    assert(r1 == r2);
    printf("PASS\n");
}

int main(void) {
    printf("\n=== Model Registry Tests ===\n\n");

    test_create_destroy();
    test_register_unregister();
    test_get_latest();
    test_list_by_architecture();
    test_search();
    test_save_load();
    test_deprecate();
    test_global_registry();

    printf("\n=== All tests passed ===\n\n");
    return 0;
}

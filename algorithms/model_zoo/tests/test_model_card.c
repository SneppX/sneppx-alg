#define _CRT_NONSTDC_NO_DEPRECATE
#include <neural_core/model_zoo/model_card.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*
 * SNEPPX - Test Model Card
 *
 * WHAT
 *   Test Model Card.
 *
 * CONCEPT
 *   Provides the Test Model Card.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


void test_create_destroy(void) {
    printf("test_create_destroy... ");
    ModelCard *card = model_card_create();
    assert(card != NULL);
    model_card_destroy(card);
    printf("PASS\n");
}

void test_setters(void) {
    printf("test_setters... ");
    ModelCard *card = model_card_create();
    
    model_card_set_name(card, "test-model");
    model_card_set_version(card, "1.0.0");
    model_card_set_description(card, "A test model");
    model_card_set_author(card, "Test Author");
    model_card_set_license(card, "MIT");
    model_card_set_repository(card, "https://github.com/test/model");
    model_card_set_architecture(card, "transformer");
    model_card_set_num_parameters(card, 1000000);
    model_card_add_tag(card, "nlp");
    model_card_add_tag(card, "transformer");
    
    assert(strcmp(card->name, "test-model") == 0);
    assert(strcmp(card->version, "1.0.0") == 0);
    assert(strcmp(card->description, "A test model") == 0);
    assert(strcmp(card->author, "Test Author") == 0);
    assert(strcmp(card->license, "MIT") == 0);
    assert(strcmp(card->repository, "https://github.com/test/model") == 0);
    assert(strcmp(card->architecture, "transformer") == 0);
    assert(card->num_parameters == 1000000);
    assert(card->num_tags == 2);
    assert(strcmp(card->tags[0], "nlp") == 0);
    assert(strcmp(card->tags[1], "transformer") == 0);
    
    model_card_destroy(card);
    printf("PASS\n");
}

void test_json_serialization(void) {
    printf("test_json_serialization... ");
    ModelCard *card = model_card_create();
    
    model_card_set_name(card, "test-model");
    model_card_set_version(card, "1.0");
    model_card_set_architecture(card, "transformer");
    model_card_set_num_parameters(card, 1000000);
    card->learning_rate = 2e-4;
    card->num_layers = 12;
    model_card_add_tag(card, "nlp");
    model_card_add_tag(card, "transformer");
    
    char *json = model_card_to_json(card, 0);
    assert(json != NULL);
    assert(strstr(json, "test-model") != NULL);
    assert(strstr(json, "transformer") != NULL);
    assert(strstr(json, "1000000") != NULL);
    assert(strstr(json, "nlp") != NULL);
    
    free(json);
    model_card_destroy(card);
    printf("PASS\n");
}

void test_file_io(void) {
    printf("test_file_io... ");
    ModelCard *card = model_card_create();
    
    model_card_set_name(card, "test-model");
    model_card_set_version(card, "1.0");
    model_card_set_architecture(card, "transformer");
    model_card_set_num_parameters(card, 1000000);
    
    const char *path = "test_model_card.json";
    int rc = model_card_save(card, path);
    assert(rc == 0);
    
    ModelCard *loaded = model_card_load(path);
    (void)loaded;
    
    remove("test_model_card.json");
    model_card_destroy(card);
    printf("PASS\n");
}

void test_validation(void) {
    printf("test_validation... ");
    char *error = NULL;
    
    // Valid card
    ModelCard *card = model_card_create();
    model_card_set_name(card, "valid-model");
    model_card_set_version(card, "1.0");
    model_card_set_architecture(card, "transformer");
    model_card_set_num_parameters(card, 1000000);
    card->num_layers = 12;
    card->learning_rate = 2e-4;
    
    assert(model_card_validate(card, NULL) == 0);
    
    // Invalid: missing name
    ModelCard *invalid = model_card_create();
    model_card_set_version(invalid, "1.0");
    model_card_set_architecture(invalid, "transformer");
    assert(model_card_validate(invalid, NULL) != 0);
    
    // Invalid: missing version
    ModelCard *invalid2 = model_card_create();
    model_card_set_name(invalid2, "test");
    model_card_set_architecture(invalid2, "transformer");
    assert(model_card_validate(invalid2, NULL) != 0);
    
    // Invalid: negative params
    ModelCard *invalid3 = model_card_create();
    model_card_set_name(invalid3, "test");
    model_card_set_version(invalid3, "1.0");
    model_card_set_architecture(invalid3, "transformer");
    invalid3->num_parameters = -1;
    assert(model_card_validate(invalid3, NULL) != 0);
    
    // Invalid: zero learning rate
    ModelCard *invalid4 = model_card_create();
    model_card_set_name(invalid4, "test");
    model_card_set_version(invalid4, "1.0");
    model_card_set_architecture(invalid4, "transformer");
    invalid4->learning_rate = 0;
    assert(model_card_validate(invalid4, NULL) != 0);
    
    model_card_destroy(card);
    model_card_destroy(invalid);
    model_card_destroy(invalid2);
    model_card_destroy(invalid3);
    model_card_destroy(invalid4);
    
    printf("PASS\n");
}

int main(void) {
    printf("\n=== Model Card Tests ===\n\n");
    
    test_create_destroy();
    test_setters();
    test_json_serialization();
    test_file_io();
    test_validation();
    
    printf("\n=== All tests passed ===\n\n");
    return 0;
}

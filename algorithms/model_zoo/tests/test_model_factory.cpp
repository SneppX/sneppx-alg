#include "neural_core/model_zoo/model_factory.hpp"

#include <cstdio>
#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

/*
 * SNEPPX - Test Model Factory
 *
 * WHAT
 *   Test Model Factory.
 *
 * CONCEPT
 *   Provides the Test Model Factory.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


using namespace sneppx;

static int tests_passed = 0;
static int tests_total = 0;

#define SNEPPX_TEST(name) do { \
    printf("test_%s... ", name); \
    tests_total++; \
} while(0)

#define SNEPPX_PASS do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

namespace fs = std::filesystem;

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

void test_create_from_preset() {
    SNEPPX_TEST("create_from_preset");

    auto model = ModelFactory::create_model("llama2", "7B");
    assert(model.config().get() != nullptr);
    assert(model.config().get()->hidden_size == 4096);
    assert(model.config().get()->num_layers == 32);
    assert(model.config().get()->num_heads == 32);
    assert(model.config().get()->vocab_size == 32000);

    std::string id = model.model_id();
    assert(id.find("llama2") != std::string::npos);

    SNEPPX_PASS;
}

void test_create_from_preset_mistral() {
    SNEPPX_TEST("create_from_preset_mistral");

    auto model = ModelFactory::create_model("mistral", "7B");
    assert(model.config().get() != nullptr);
    assert(model.config().get()->hidden_size == 4096);
    assert(model.config().get()->num_layers == 32);
    assert(model.config().get()->num_heads == 32);

    SNEPPX_PASS;
}

void test_create_from_preset_qwen2() {
    SNEPPX_TEST("create_from_preset_qwen2");

    auto model = ModelFactory::create_model("qwen2", "7B");
    assert(model.config().get() != nullptr);
    assert(model.config().get()->hidden_size == 4096);
    assert(model.config().get()->num_layers == 32);

    SNEPPX_PASS;
}

void test_config_json_roundtrip() {
    SNEPPX_TEST("config_json_roundtrip");

    auto cfg = ModelConfigCpp::preset("llama3", "8B");
    std::string json = cfg.to_json(true);
    assert(!json.empty());

    auto cfg2 = ModelConfigCpp::from_json(json);
    assert(cfg2.get() != nullptr);
    assert(cfg2.get()->hidden_size == cfg.get()->hidden_size);
    assert(cfg2.get()->num_layers == cfg.get()->num_layers);
    assert(cfg2.get()->num_heads == cfg.get()->num_heads);

    SNEPPX_PASS;
}

void test_config_save_load() {
    SNEPPX_TEST("config_save_load");

    auto cfg = ModelConfigCpp::preset("bert", "base");
    std::string path = "test_bert_config.json";
    cfg.save(path);

    auto cfg2 = ModelConfigCpp::load(path);
    assert(cfg2.get() != nullptr);
    assert(cfg2.get()->hidden_size == cfg.get()->hidden_size);
    assert(cfg2.get()->num_layers == cfg.get()->num_layers);

    std::remove(path.c_str());

    SNEPPX_PASS;
}

void test_model_card_roundtrip() {
    SNEPPX_TEST("model_card_roundtrip");

    ModelCardCpp card;
    card.set_name("test-model");
    card.set_version("2.0.0");
    card.set_author("SneppX");
    card.add_tag("transformer");
    card.add_tag("llama");

    std::string json = card.to_json(true);
    assert(!json.empty());
    assert(json.find("test-model") != std::string::npos);
    assert(json.find("SneppX") != std::string::npos);

    auto card2 = ModelCardCpp::from_json(json);
    assert(card2.get() != nullptr);

    SNEPPX_PASS;
}

void test_model_factory_save_load() {
    SNEPPX_TEST("model_factory_save_load");

    auto model = ModelFactory::create_model("llama2", "7B");
    std::string dir = "test_model_save_load";
    ModelFactory::save_model(model, dir);

    auto loaded = ModelFactory::load_model(dir);
    assert(loaded.config().get() != nullptr);
    assert(loaded.config().get()->hidden_size == 4096);

    fs::remove_all(dir);

    SNEPPX_PASS;
}

void test_model_registry_integration() {
    SNEPPX_TEST("model_registry_integration");

    auto model = ModelFactory::create_model("llama3", "8B");
    ModelFactory::register_model(model);

    auto models = ModelFactory::list_registered_models();
    assert(!models.empty());

    SNEPPX_PASS;
}

void test_from_pretrained_stub() {
    SNEPPX_TEST("from_pretrained_stub");

    auto model = ModelFactory::from_pretrained("llama-2-7b");
    assert(model.config().get() != nullptr);
    assert(model.config().get()->hidden_size == 4096);

    SNEPPX_PASS;
}

void test_model_card_copy() {
    SNEPPX_TEST("model_card_copy");

    ModelCardCpp card1;
    card1.set_name("copy-test");
    card1.set_version("1.0.0");

    ModelCardCpp card2 = card1;
    card2.set_name("copy-test-2");

    SNEPPX_PASS;
}

void test_weight_collection() {
    SNEPPX_TEST("weight_collection");

    WeightCollectionCpp wc;
    assert(wc.count() == 0);

    std::vector<std::string> names = wc.tensor_names();
    assert(names.empty());

    SNEPPX_PASS;
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

int main() {
    printf("Running C++ Model Factory tests...\n\n");

    test_create_from_preset();
    test_create_from_preset_mistral();
    test_create_from_preset_qwen2();
    test_config_json_roundtrip();
    test_config_save_load();
    test_model_card_roundtrip();
    test_model_factory_save_load();
    test_model_registry_integration();
    test_from_pretrained_stub();
    test_model_card_copy();
    test_weight_collection();

    printf("\n%d/%d tests passed!\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}

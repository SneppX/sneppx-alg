#include "neural_core/model_zoo/model_factory.hpp"

#include <cstring>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>

/*
 * SNEPPX - Model Factory
 *
 * WHAT
 *   Model Factory.
 *
 * CONCEPT
 *   Provides the Model Factory.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace sneppx {

namespace fs = std::filesystem;

// =========================================================================
// ModelConfigCpp
// =========================================================================

ModelConfigCpp::ModelConfigCpp()
    : cfg_(model_config_create()) {
}

ModelConfigCpp::ModelConfigCpp(::ModelConfig* cfg)
    : cfg_(cfg) {
}

ModelConfigCpp::~ModelConfigCpp() {
    if (cfg_) {
        model_config_destroy(cfg_);
    }
}

ModelConfigCpp::ModelConfigCpp(const ModelConfigCpp& other)
    : cfg_(other.cfg_ ? model_config_copy(other.cfg_) : nullptr) {
}

ModelConfigCpp& ModelConfigCpp::operator=(const ModelConfigCpp& other) {
    if (this != &other) {
        if (cfg_) model_config_destroy(cfg_);
        cfg_ = other.cfg_ ? model_config_copy(other.cfg_) : nullptr;
    }
    return *this;
}

ModelConfigCpp::ModelConfigCpp(ModelConfigCpp&& other) noexcept
    : cfg_(other.cfg_) {
    other.cfg_ = nullptr;
}

ModelConfigCpp& ModelConfigCpp::operator=(ModelConfigCpp&& other) noexcept {
    if (this != &other) {
        if (cfg_) model_config_destroy(cfg_);
        cfg_ = other.cfg_;
        other.cfg_ = nullptr;
    }
    return *this;
}

::ModelConfig* ModelConfigCpp::release() {
    ::ModelConfig* ptr = cfg_;
    cfg_ = nullptr;
    return ptr;
}

void ModelConfigCpp::check() const {
    if (!cfg_) throw std::runtime_error("ModelConfigCpp: null config");
}

std::string ModelConfigCpp::to_json(bool pretty) const {
    check();
    char* json = model_config_to_json(cfg_, pretty ? 1 : 0);
    if (!json) throw std::runtime_error("ModelConfigCpp: to_json failed");
    std::string result(json);
    std::free(json);
    return result;
}

ModelConfigCpp ModelConfigCpp::from_json(const std::string& json) {
    ::ModelConfig* cfg = model_config_from_json(json.c_str());
    if (!cfg) throw std::runtime_error("ModelConfigCpp: from_json failed");
    return ModelConfigCpp(cfg);
}

void ModelConfigCpp::save(const std::string& path) const {
    check();
    if (model_config_save(cfg_, path.c_str()) != 0) {
        throw std::runtime_error("ModelConfigCpp: save failed");
    }
}

ModelConfigCpp ModelConfigCpp::load(const std::string& path) {
    ::ModelConfig* cfg = model_config_load(path.c_str());
    if (!cfg) throw std::runtime_error("ModelConfigCpp: load failed");
    return ModelConfigCpp(cfg);
}

ModelConfigCpp ModelConfigCpp::preset(const std::string& family, const std::string& size) {
    ::ModelConfig* cfg = nullptr;
    if (family == "llama2" && size == "7B") cfg = model_config_llama2_7b();
    else if (family == "llama2" && size == "13B") cfg = model_config_llama2_13b();
    else if (family == "llama3" && size == "8B") cfg = model_config_llama3_8b();
    else if (family == "mistral" && size == "7B") cfg = model_config_mistral_7b();
    else if (family == "qwen2" && size == "7B") cfg = model_config_qwen2_7b();
    else if (family == "bert" && size == "base") cfg = model_config_bert_base();
    else if (family == "vit" && size == "base") cfg = model_config_vit_base();
    else if (family == "sdxl") cfg = model_config_sdxl();
    else throw std::runtime_error("ModelConfigCpp: unknown preset '" + family + "/" + size + "'");

    if (!cfg) throw std::runtime_error("ModelConfigCpp: preset returned null");
    return ModelConfigCpp(cfg);
}

// =========================================================================
// ModelCardCpp
// =========================================================================

ModelCardCpp::ModelCardCpp()
    : card_(model_card_create()) {
}

ModelCardCpp::ModelCardCpp(::ModelCard* card)
    : card_(card) {
}

ModelCardCpp::~ModelCardCpp() {
    if (card_) model_card_destroy(card_);
}

ModelCardCpp::ModelCardCpp(const ModelCardCpp& other)
    : card_(nullptr) {
    if (other.card_) {
        std::string json = other.to_json();
        card_ = model_card_from_json(json.c_str());
    }
}

ModelCardCpp& ModelCardCpp::operator=(const ModelCardCpp& other) {
    if (this != &other) {
        if (card_) model_card_destroy(card_);
        card_ = nullptr;
        if (other.card_) {
            std::string json = other.to_json();
            card_ = model_card_from_json(json.c_str());
        }
    }
    return *this;
}

ModelCardCpp::ModelCardCpp(ModelCardCpp&& other) noexcept
    : card_(other.card_) {
    other.card_ = nullptr;
}

ModelCardCpp& ModelCardCpp::operator=(ModelCardCpp&& other) noexcept {
    if (this != &other) {
        if (card_) model_card_destroy(card_);
        card_ = other.card_;
        other.card_ = nullptr;
    }
    return *this;
}

::ModelCard* ModelCardCpp::release() {
    ::ModelCard* ptr = card_;
    card_ = nullptr;
    return ptr;
}

void ModelCardCpp::check() const {
    if (!card_) throw std::runtime_error("ModelCardCpp: null card");
}

std::string ModelCardCpp::to_json(bool pretty) const {
    check();
    char* json = model_card_to_json(card_, pretty ? 1 : 0);
    if (!json) throw std::runtime_error("ModelCardCpp: to_json failed");
    std::string result(json);
    std::free(json);
    return result;
}

ModelCardCpp ModelCardCpp::from_json(const std::string& json) {
    ::ModelCard* card = model_card_from_json(json.c_str());
    if (!card) throw std::runtime_error("ModelCardCpp: from_json failed");
    return ModelCardCpp(card);
}

void ModelCardCpp::set_name(const std::string& name) {
    check();
    model_card_set_name(card_, name.c_str());
}

void ModelCardCpp::set_version(const std::string& version) {
    check();
    model_card_set_version(card_, version.c_str());
}

void ModelCardCpp::set_author(const std::string& author) {
    check();
    model_card_set_author(card_, author.c_str());
}

void ModelCardCpp::add_tag(const std::string& tag) {
    check();
    model_card_add_tag(card_, tag.c_str());
}

// =========================================================================
// WeightCollectionCpp
// =========================================================================

WeightCollectionCpp::WeightCollectionCpp()
    : wc_(weight_collection_create()) {
}

WeightCollectionCpp::WeightCollectionCpp(::WeightCollection* wc)
    : wc_(wc) {
}

WeightCollectionCpp::~WeightCollectionCpp() {
    if (wc_) weight_collection_destroy(wc_);
}

WeightCollectionCpp::WeightCollectionCpp(WeightCollectionCpp&& other) noexcept
    : wc_(other.wc_) {
    other.wc_ = nullptr;
}

WeightCollectionCpp& WeightCollectionCpp::operator=(WeightCollectionCpp&& other) noexcept {
    if (this != &other) {
        if (wc_) weight_collection_destroy(wc_);
        wc_ = other.wc_;
        other.wc_ = nullptr;
    }
    return *this;
}

void WeightCollectionCpp::check() const {
    if (!wc_) throw std::runtime_error("WeightCollectionCpp: null collection");
}

int WeightCollectionCpp::count() const {
    check();
    return wc_->count;
}

bool WeightCollectionCpp::has_tensor(const std::string& name) const {
    check();
    return weight_collection_get_const(wc_, name.c_str()) != nullptr;
}

std::vector<std::string> WeightCollectionCpp::tensor_names() const {
    check();
    int count = 0;
    char** names = weight_collection_names(wc_, &count);
    std::vector<std::string> result;
    for (int i = 0; i < count; i++) {
        result.emplace_back(names[i]);
        std::free(names[i]);
    }
    std::free(names);
    return result;
}

WeightCollectionCpp WeightCollectionCpp::load(const std::string& path, WeightFormat format) {
    ::WeightCollection* wc = weights_load(path.c_str(), format);
    if (!wc) throw std::runtime_error("WeightCollectionCpp: load failed");
    return WeightCollectionCpp(wc);
}

// =========================================================================
// Model
// =========================================================================

Model::Model()
    : config_(), card_(), weights_() {
}

Model::Model(ModelConfigCpp config, ModelCardCpp card, WeightCollectionCpp weights)
    : config_(std::move(config)), card_(std::move(card)), weights_(std::move(weights)) {
}

std::string Model::model_id() const {
    const auto* c = config_.get();
    if (c && c->name) return c->name;
    return "unknown";
}

int64_t Model::num_parameters() const {
    const auto* c = config_.get();
    if (c) {
        int64_t params = 0;
        params += static_cast<int64_t>(c->vocab_size) * c->hidden_size * 2;
        params += static_cast<int64_t>(c->num_layers) * c->hidden_size * c->hidden_size * 4;
        params += static_cast<int64_t>(c->num_layers) * c->hidden_size * c->intermediate_size * 3;
        return params;
    }
    return 0;
}

// =========================================================================
// ModelFactory
// =========================================================================

Model ModelFactory::create_model(const std::string& family, const std::string& size) {
    auto config = ModelConfigCpp::preset(family, size);

    ModelCardCpp card;
    card.set_name(family + "-" + size);
    card.set_version("1.0.0");
    card.set_author("SneppX");

    WeightCollectionCpp weights;

    return Model(std::move(config), std::move(card), std::move(weights));
}

Model ModelFactory::create_model(const ModelConfigCpp& config) {
    auto cfg_copy = ModelConfigCpp(model_config_copy(config.get()));

    ModelCardCpp card;
    const auto* c = config.get();
    if (c->name) card.set_name(c->name);
    card.set_version("1.0.0");

    WeightCollectionCpp weights;

    return Model(std::move(cfg_copy), std::move(card), std::move(weights));
}

Model ModelFactory::create_model(const std::string& family, const std::string& size,
                                  const std::unordered_map<std::string, std::string>& overrides) {
    auto config = create_model(family, size);
    return config;
}

Model ModelFactory::from_pretrained(const std::string& model_id,
                                     const std::string& cache_dir) {
    (void)cache_dir;

    std::string lower = model_id;
    for (auto& c : lower) c = static_cast<char>(std::tolower(c));

    std::string family, size;
    std::string mapping[][3] = {
        {"llama-2-7b", "llama2", "7B"},
        {"llama-2-13b", "llama2", "13B"},
        {"llama-2-70b", "llama2", "70B"},
        {"llama-3-8b", "llama3", "8B"},
        {"llama-3-70b", "llama3", "70B"},
        {"mistral-7b", "mistral", "7B"},
        {"qwen2-7b", "qwen2", "7B"},
        {"qwen2-72b", "qwen2", "72B"},
        {"deepseek-v2", "llama2", "7B"},
    };

    bool found = false;
    for (const auto& m : mapping) {
        if (lower.find(m[0]) != std::string::npos) {
            family = m[1];
            size = m[2];
            found = true;
            break;
        }
    }

    if (!found) {
        throw std::runtime_error("ModelFactory: unknown model_id '" + model_id + "'");
    }

    auto model = create_model(family, size);
    model.config().get()->name = static_cast<char*>(std::realloc(model.config().get()->name, model_id.size() + 1));
    if (model.config().get()->name) {
        std::strncpy(model.config().get()->name, model_id.c_str(), model_id.size()); model.config().get()->name[model_id.size()] = '\0';
    }

    std::cout << "[SNEPPX from_pretrained] " << model_id
              << " -> family=" << family << ", size=" << size
              << ", hidden_size=" << model.config().get()->hidden_size
              << ", layers=" << model.config().get()->num_layers
              << ", heads=" << model.config().get()->num_heads << std::endl;

    return model;
}

void ModelFactory::save_model(const Model& model, const std::string& directory) {
    fs::create_directories(directory);

    std::string config_path = directory + "/config.json";
    model.config().save(config_path);
    std::cout << "Config saved to " << config_path << std::endl;

    std::string card_json = model.card().to_json(true);
    std::string card_path = directory + "/README.md";
    std::ofstream(card_path) << card_json;
    std::cout << "Model card saved to " << card_path << std::endl;
}

Model ModelFactory::load_model(const std::string& directory) {
    std::string config_path = directory + "/config.json";
    if (!fs::exists(config_path)) {
        throw std::runtime_error("ModelFactory: config not found at " + config_path);
    }

    auto config = ModelConfigCpp::load(config_path);

    ModelCardCpp card;
    std::string card_path = directory + "/README.md";
    if (fs::exists(card_path)) {
        std::ifstream ifs(card_path);
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        if (!content.empty()) {
            card = ModelCardCpp::from_json(content);
        }
    }

    WeightCollectionCpp weights;

    return Model(std::move(config), std::move(card), std::move(weights));
}

void ModelFactory::register_model(const Model& model) {
    auto* registry = model_registry_global();
    const auto* cfg = model.config().get();
    const auto* card = model.card().get();

    auto* rcard = model_card_create();
    if (card) {
        if (card->name) model_card_set_name(rcard, card->name);
        if (card->version) model_card_set_version(rcard, card->version);
        if (card->author) model_card_set_author(rcard, card->author);
    }

    std::string arch_name;
    if (cfg) {
        switch (cfg->architecture) {
            case MODEL_ARCH_TRANSFORMER: arch_name = "transformer"; break;
            case MODEL_ARCH_VIT: arch_name = "vit"; break;
            case MODEL_ARCH_DIFFUSION: arch_name = "diffusion"; break;
            case MODEL_ARCH_RNN: arch_name = "rnn"; break;
            default: arch_name = "custom"; break;
        }
    }

    model_registry_register(registry,
                            cfg && cfg->name ? cfg->name : "unknown",
                            card && card->version ? card->version : "1.0.0",
                            arch_name.c_str(),
                            cfg && cfg->description ? cfg->description : "",
                            card && card->author ? card->author : "",
                            card && card->license ? card->license : "",
                            "",
                            "",
                            "",
                            1);

    model_card_destroy(rcard);
}

std::vector<std::string> ModelFactory::list_registered_models() {
    auto* registry = model_registry_global();
    int count = 0;
    auto** entries = model_registry_list(registry, nullptr, &count);
    std::vector<std::string> result;
    for (int i = 0; i < count; i++) {
        if (entries[i] && entries[i]->name) {
            result.emplace_back(entries[i]->name);
        }
    }
    std::free(entries);
    return result;
}

ModelConfigCpp ModelFactory::apply_overrides(
    const ModelConfigCpp& base,
    const std::unordered_map<std::string, std::string>& overrides) {
    (void)base;
    (void)overrides;
    return ModelConfigCpp(model_config_copy(base.get()));
}

} // namespace sneppx

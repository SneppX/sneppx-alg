#ifndef NEURAL_CORE_MODEL_ZOO_MODEL_FACTORY_HPP
#define NEURAL_CORE_MODEL_ZOO_MODEL_FACTORY_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <functional>

#include "neural_core/kernel/model_config.h"
#include "neural_core/model_zoo/model_card.h"
#include "neural_core/model_zoo/weights.h"
#include "neural_core/model_zoo/registry.h"
#include "neural_core/kernel/model_zoo.h"

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

// =========================================================================
// C++ RAII wrappers around C types
// =========================================================================

class ModelConfigCpp {
public:
    ModelConfigCpp();
    explicit ModelConfigCpp(::ModelConfig* cfg);
    ~ModelConfigCpp();

    ModelConfigCpp(const ModelConfigCpp& other);
    ModelConfigCpp& operator=(const ModelConfigCpp& other);

    ModelConfigCpp(ModelConfigCpp&& other) noexcept;
    ModelConfigCpp& operator=(ModelConfigCpp&& other) noexcept;

    ::ModelConfig* get() { return cfg_; }
    const ::ModelConfig* get() const { return cfg_; }
    ::ModelConfig* release();

    std::string to_json(bool pretty = true) const;
    static ModelConfigCpp from_json(const std::string& json);
    void save(const std::string& path) const;
    static ModelConfigCpp load(const std::string& path);

    static ModelConfigCpp preset(const std::string& family, const std::string& size);

private:
    ::ModelConfig* cfg_;
    void check() const;
};

class ModelCardCpp {
public:
    ModelCardCpp();
    explicit ModelCardCpp(::ModelCard* card);
    ~ModelCardCpp();

    ModelCardCpp(const ModelCardCpp& other);
    ModelCardCpp& operator=(const ModelCardCpp& other);

    ModelCardCpp(ModelCardCpp&& other) noexcept;
    ModelCardCpp& operator=(ModelCardCpp&& other) noexcept;

    ::ModelCard* get() { return card_; }
    const ::ModelCard* get() const { return card_; }
    ::ModelCard* release();

    std::string to_json(bool pretty = true) const;
    static ModelCardCpp from_json(const std::string& json);

    void set_name(const std::string& name);
    void set_version(const std::string& version);
    void set_author(const std::string& author);
    void add_tag(const std::string& tag);

private:
    ::ModelCard* card_;
    void check() const;
};

class WeightCollectionCpp {
public:
    WeightCollectionCpp();
    explicit WeightCollectionCpp(::WeightCollection* wc);
    ~WeightCollectionCpp();

    WeightCollectionCpp(const WeightCollectionCpp&) = delete;
    WeightCollectionCpp& operator=(const WeightCollectionCpp&) = delete;

    WeightCollectionCpp(WeightCollectionCpp&& other) noexcept;
    WeightCollectionCpp& operator=(WeightCollectionCpp&& other) noexcept;

    ::WeightCollection* get() { return wc_; }
    const ::WeightCollection* get() const { return wc_; }

    int count() const;
    bool has_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    static WeightCollectionCpp load(const std::string& path, WeightFormat format = WEIGHT_FORMAT_AUTO);

private:
    ::WeightCollection* wc_;
    void check() const;
};

// =========================================================================
// Model - Full model object
// =========================================================================

class Model {
public:
    Model();
    Model(ModelConfigCpp config, ModelCardCpp card, WeightCollectionCpp weights);
    ~Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    ModelConfigCpp& config() { return config_; }
    const ModelConfigCpp& config() const { return config_; }

    ModelCardCpp& card() { return card_; }
    const ModelCardCpp& card() const { return card_; }

    WeightCollectionCpp& weights() { return weights_; }
    const WeightCollectionCpp& weights() const { return weights_; }

    std::string model_id() const;
    int64_t num_parameters() const;

private:
    ModelConfigCpp config_;
    ModelCardCpp card_;
    WeightCollectionCpp weights_;
};

// =========================================================================
// ModelFactory - Create and load models
// =========================================================================

class ModelFactory {
public:
    static Model create_model(const std::string& family, const std::string& size);

    static Model create_model(const ModelConfigCpp& config);

    static Model create_model(const std::string& family, const std::string& size,
                              const std::unordered_map<std::string, std::string>& overrides);

    static Model from_pretrained(const std::string& model_id,
                                 const std::string& cache_dir = "");

    static void save_model(const Model& model, const std::string& directory);

    static Model load_model(const std::string& directory);

    static void register_model(const Model& model);

    static std::vector<std::string> list_registered_models();

private:
    static ModelConfigCpp apply_overrides(const ModelConfigCpp& base,
                                           const std::unordered_map<std::string, std::string>& overrides);
};

} // namespace sneppx

#endif // NEURAL_CORE_MODEL_ZOO_MODEL_FACTORY_HPP

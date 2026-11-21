/*
 * SNEPPX - Optim Model Bindings
 *
 * WHAT
 *   Optim Model Bindings.
 *
 * CONCEPT
 *   Provides the Optim Model Bindings.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// Optimizer + LRScheduler + Model wrappers — extracted from bindings.cpp lines 605–800, 1445–1595
#pragma once

class PyOptimizer {
public:
    SNEPPXOptimizer* ptr;
    bool owner;

    PyOptimizer() : ptr(nullptr), owner(false) {}
    explicit PyOptimizer(SNEPPXOptimizer* o, bool own = true) : ptr(o), owner(own) {}
    ~PyOptimizer() { if (owner && ptr) SNEPPX_optimizer_destroy(ptr); }

    PyOptimizer(const PyOptimizer&) = delete;
    PyOptimizer& operator=(const PyOptimizer&) = delete;
    PyOptimizer(PyOptimizer&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyOptimizer& operator=(PyOptimizer&& o) noexcept {
        if (this != &o) { if (owner && ptr) SNEPPX_optimizer_destroy(ptr); ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false; }
        return *this;
    }

    void step(py::list params, py::list grads) {
        std::vector<SNEPPXTensor*> p(params.size()), g(grads.size());
        for (size_t i = 0; i < params.size(); ++i) p[i] = params[i].cast<PyTensor*>()->ptr;
        for (size_t i = 0; i < grads.size(); ++i) g[i] = grads[i].cast<PyTensor*>()->ptr;
        SNEPPX_optimizer_step(ptr, p.data(), g.data(), params.size());
    }

    static PyOptimizer sgd_create(float lr, float momentum, float weight_decay) {
        return PyOptimizer(SNEPPX_sgd_create(lr, momentum, weight_decay));
    }
    static PyOptimizer adam_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
        return PyOptimizer(SNEPPX_adam_create(lr, beta1, beta2, eps, weight_decay));
    }
    static PyOptimizer adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay) {
        return PyOptimizer(SNEPPX_adamw_create(lr, beta1, beta2, eps, weight_decay));
    }
    static PyOptimizer rmsprop_create(float lr, float alpha, float eps, float momentum, float weight_decay) {
        return PyOptimizer(SNEPPX_rmsprop_create(lr, alpha, eps, momentum, weight_decay));
    }
    static PyOptimizer adagrad_create(float lr, float eps, float weight_decay) {
        return PyOptimizer(SNEPPX_adagrad_create(lr, eps, weight_decay));
    }
};

class PyLRScheduler {
public:
    SNEPPXLRScheduler* ptr;
    bool owner;
    float* lr_ptr;
    bool owns_lr;

    PyLRScheduler() : ptr(nullptr), owner(false), lr_ptr(nullptr), owns_lr(false) {}
    PyLRScheduler(SNEPPXLRScheduler* s, float* lr, bool own = true, bool own_lr = false)
        : ptr(s), owner(own), lr_ptr(lr), owns_lr(own_lr) {}
    ~PyLRScheduler() {
        if (owner && ptr) SNEPPX_lr_scheduler_destroy(ptr);
        if (owns_lr && lr_ptr) delete lr_ptr;
    }

    void step(float current_loss = 0.0f) { SNEPPX_lr_scheduler_step(ptr, current_loss); }

    static PyLRScheduler step_lr(float& lr, float gamma, size_t step_size) {
        return PyLRScheduler(SNEPPX_lr_scheduler_step_lr(&lr, gamma, step_size), &lr);
    }
    static PyLRScheduler exponential(float& lr, float gamma) {
        return PyLRScheduler(SNEPPX_lr_scheduler_exponential(&lr, gamma), &lr);
    }
    static PyLRScheduler cosine(float& lr, float min_lr, float max_lr, size_t total_steps) {
        return PyLRScheduler(SNEPPX_lr_scheduler_cosine(&lr, min_lr, max_lr, total_steps), &lr);
    }
    static PyLRScheduler reduce_on_plateau(float& lr, float factor, size_t patience, int mode_min) {
        return PyLRScheduler(SNEPPX_lr_scheduler_reduce_on_plateau(&lr, factor, patience, mode_min), &lr);
    }
};

class PyModel {
public:
    SNEPPXModel* ptr;
    bool owner;

    PyModel() : ptr(nullptr), owner(false) {}
    explicit PyModel(SNEPPXModel* m, bool own = true) : ptr(m), owner(own) {}
    ~PyModel() { if (owner && ptr) SNEPPX_model_destroy(ptr); }
    PyModel(const PyModel&) = delete;
    PyModel& operator=(const PyModel&) = delete;
    PyModel(PyModel&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyModel& operator=(PyModel&& o) noexcept {
        if (this != &o) { if (owner && ptr) SNEPPX_model_destroy(ptr); ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false; }
        return *this;
    }

    static PyModel create(SNEPPXArchConfig& config) { return PyModel(SNEPPX_model_create(&config)); }

    PyTensor forward(const PyTensor& input) {
        SNEPPXTensor* out = nullptr;
        if (SNEPPX_model_forward(ptr, input.ptr, &out) != 0)
            throw std::runtime_error("model forward failed");
        return PyTensor(out);
    }

    std::vector<PyTensor> parameters() {
        size_t n = SNEPPX_model_get_params(ptr, nullptr, 0);
        std::vector<SNEPPXTensor*> raw(n);
        SNEPPX_model_get_params(ptr, raw.data(), n);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < n; ++i)
            out.emplace_back(raw[i], false);
        return out;
    }
};

class PyHSSModel {
public:
    SNEPPXHSSModel* ptr;
    bool owner;
    PyHSSModel() : ptr(nullptr), owner(false) {}
    explicit PyHSSModel(SNEPPXHSSModel* m, bool own = true) : ptr(m), owner(own) {}
    ~PyHSSModel() { if (owner && ptr) SNEPPX_hss_model_destroy(ptr); }

    static PyHSSModel create(const SNEPPXHSSConfig& config, unsigned int seed) {
        return PyHSSModel(SNEPPX_hss_model_create(&config, seed));
    }

    PyTensor forward(const PyTensor& input) {
        SNEPPXTensor* out = nullptr;
        if (SNEPPX_hss_forward(ptr, input.ptr, &out) != 0)
            throw std::runtime_error("HSS forward failed");
        return PyTensor(out);
    }

    std::vector<PyTensor> parameters() {
        size_t n = SNEPPX_hss_get_params(ptr, nullptr, 0);
        std::vector<SNEPPXTensor*> raw(n);
        SNEPPX_hss_get_params(ptr, raw.data(), n);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < n; ++i) out.emplace_back(raw[i], false);
        return out;
    }

    void discretize_layer(size_t layer_idx) {
        if (!ptr->layers) throw std::runtime_error("invalid layer");
        SNEPPX_hss_discretize(ptr->layers[layer_idx]);
    }
};

class PySERModel {
public:
    SNEPPXSERModel* ptr;
    bool owner;
    PySERModel() : ptr(nullptr), owner(false) {}
    explicit PySERModel(SNEPPXSERModel* m, bool own = true) : ptr(m), owner(own) {}
    ~PySERModel() { if (owner && ptr) SNEPPX_ser_model_destroy(ptr); }

    static PySERModel create(const SNEPPXSERConfig& config, unsigned int seed, size_t num_layers) {
        return PySERModel(SNEPPX_ser_model_create(&config, seed, num_layers));
    }

    std::vector<PyTensor> parameters() {
        size_t n = SNEPPX_ser_get_params(ptr, nullptr, 0);
        std::vector<SNEPPXTensor*> raw(n);
        SNEPPX_ser_get_params(ptr, raw.data(), n);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < n; ++i) out.emplace_back(raw[i], false);
        return out;
    }

    PyTensor forward(const PyTensor& input) {
        SNEPPXTensor* output = nullptr;
        SNEPPX_ser_forward(ptr->layers[0], input.ptr, &output);
        return PyTensor(output, true);
    }
};

class PyARCLayer {
public:
    SNEPPXARCLayer* ptr;
    bool owner;
    PyARCLayer() : ptr(nullptr), owner(false) {}
    explicit PyARCLayer(SNEPPXARCLayer* m, bool own = true) : ptr(m), owner(own) {}
    ~PyARCLayer() { if (owner && ptr) SNEPPX_arc_layer_destroy(ptr); }

    static PyARCLayer create(const SNEPPXARCConfig& config, size_t input_dim, size_t output_dim, unsigned int seed) {
        return PyARCLayer(SNEPPX_arc_layer_create(&config, input_dim, output_dim, seed));
    }
};

static void init_optim_model(py::module& m) {
    // Optimizer
    py::class_<PyOptimizer>(m, "_Optimizer")
        .def(py::init<>())
        .def("step", &PyOptimizer::step)
        .def_static("sgd_create", &PyOptimizer::sgd_create)
        .def_static("adam_create", &PyOptimizer::adam_create)
        .def_static("adamw_create", &PyOptimizer::adamw_create)
        .def_static("rmsprop_create", &PyOptimizer::rmsprop_create)
        .def_static("adagrad_create", &PyOptimizer::adagrad_create);

    // LRScheduler
    py::class_<PyLRScheduler>(m, "_LRScheduler")
        .def(py::init<>())
        .def("step", &PyLRScheduler::step)
        .def_static("step_lr", &PyLRScheduler::step_lr)
        .def_static("exponential", &PyLRScheduler::exponential)
        .def_static("cosine", &PyLRScheduler::cosine)
        .def_static("reduce_on_plateau", &PyLRScheduler::reduce_on_plateau);

    // Config structs
    py::class_<SNEPPXOptimizerConfig>(m, "SNEPPXOptimizerConfig")
        .def(py::init())
        .def_readwrite("learning_rate", &SNEPPXOptimizerConfig::learning_rate)
        .def_readwrite("momentum", &SNEPPXOptimizerConfig::momentum)
        .def_readwrite("weight_decay", &SNEPPXOptimizerConfig::weight_decay)
        .def_readwrite("grad_clip", &SNEPPXOptimizerConfig::grad_clip)
        .def_readwrite("type", &SNEPPXOptimizerConfig::type)
        .def_readwrite("beta1", &SNEPPXOptimizerConfig::beta1)
        .def_readwrite("beta2", &SNEPPXOptimizerConfig::beta2)
        .def_readwrite("epsilon", &SNEPPXOptimizerConfig::epsilon)
        .def_readwrite("nesterov", &SNEPPXOptimizerConfig::nesterov)
        .def("default", &SNEPPX_optimizer_config_default);

    py::class_<SNEPPXArchConfig>(m, "SNEPPXArchConfig")
        .def(py::init())
        .def_readwrite("input_dim", &SNEPPXArchConfig::input_dim)
        .def_readwrite("output_dim", &SNEPPXArchConfig::output_dim)
        .def_readwrite("seed", &SNEPPXArchConfig::seed)
        .def("default", &SNEPPX_arch_config_default);

    py::class_<SNEPPXTrainConfig>(m, "SNEPPXTrainConfig")
        .def(py::init())
        .def_readwrite("num_epochs", &SNEPPXTrainConfig::num_epochs)
        .def_readwrite("batch_size", &SNEPPXTrainConfig::batch_size)
        .def_readwrite("learning_rate", &SNEPPXTrainConfig::learning_rate)
        .def_readwrite("log_interval", &SNEPPXTrainConfig::log_interval)
        .def_readwrite("save_interval", &SNEPPXTrainConfig::save_interval)
        .def_readwrite("device", &SNEPPXTrainConfig::device)
        .def("default", &SNEPPX_train_config_default);

    py::class_<SNEPPXHSSConfig>(m, "SNEPPXHSSConfig")
        .def(py::init())
        .def_readwrite("state_dim", &SNEPPXHSSConfig::state_dim)
        .def_readwrite("input_dim", &SNEPPXHSSConfig::input_dim)
        .def_readwrite("output_dim", &SNEPPXHSSConfig::output_dim)
        .def_readwrite("num_layers", &SNEPPXHSSConfig::num_layers)
        .def_readwrite("seq_len", &SNEPPXHSSConfig::seq_len)
        .def_readwrite("dt_min", &SNEPPXHSSConfig::dt_min)
        .def_readwrite("dt_max", &SNEPPXHSSConfig::dt_max)
        .def_readwrite("use_hierarchical", &SNEPPXHSSConfig::use_hierarchical)
        .def("default", &SNEPPX_hss_config_default);

    py::class_<SNEPPXSERConfig>(m, "SNEPPXSERConfig")
        .def(py::init())
        .def_readwrite("num_experts", &SNEPPXSERConfig::num_experts)
        .def_readwrite("num_active", &SNEPPXSERConfig::num_active)
        .def_readwrite("input_dim", &SNEPPXSERConfig::input_dim)
        .def_readwrite("expert_dim", &SNEPPXSERConfig::expert_dim)
        .def_readwrite("output_dim", &SNEPPXSERConfig::output_dim)
        .def_readwrite("top_k_method", &SNEPPXSERConfig::top_k_method)
        .def_readwrite("load_balance_coef", &SNEPPXSERConfig::load_balance_coef)
        .def_readwrite("dropout_rate", &SNEPPXSERConfig::dropout_rate)
        .def_readwrite("use_mlp_gater", &SNEPPXSERConfig::use_mlp_gater)
        .def_readwrite("gater_hidden_dim", &SNEPPXSERConfig::gater_hidden_dim)
        .def("default", &SNEPPX_ser_config_default);

    py::class_<SNEPPXARCConfig>(m, "SNEPPXARCConfig")
        .def(py::init())
        .def_readwrite("input_guard_strength", &SNEPPXARCConfig::input_guard_strength)
        .def_readwrite("gradient_obfuscation_method", &SNEPPXARCConfig::gradient_obfuscation_method)
        .def_readwrite("gradient_noise_scale", &SNEPPXARCConfig::gradient_noise_scale)
        .def_readwrite("gradient_clip_max", &SNEPPXARCConfig::gradient_clip_max)
        .def_readwrite("output_verify_layers", &SNEPPXARCConfig::output_verify_layers)
        .def_readwrite("output_verify_threshold", &SNEPPXARCConfig::output_verify_threshold)
        .def_readwrite("adversarial_training", &SNEPPXARCConfig::adversarial_training)
        .def_readwrite("attack_simulation_types", &SNEPPXARCConfig::attack_simulation_types)
        .def_readwrite("attack_epsilon", &SNEPPXARCConfig::attack_epsilon)
        .def("default", &SNEPPX_arc_config_default);

    py::class_<SNEPPXNPEConfig>(m, "SNEPPXNPEConfig")
        .def(py::init())
        .def_readwrite("max_program_length", &SNEPPXNPEConfig::max_program_length)
        .def_readwrite("register_count", &SNEPPXNPEConfig::register_count)
        .def_readwrite("step_limit", &SNEPPXNPEConfig::step_limit)
        .def_readwrite("verification_mode", &SNEPPXNPEConfig::verification_mode)
        .def_readwrite("trace_execution", &SNEPPXNPEConfig::trace_execution)
        .def("default", &SNEPPX_npe_config_default);

    py::class_<SNEPPXFMConfig>(m, "SNEPPXFMConfig")
        .def(py::init())
        .def_readwrite("num_nodes", &SNEPPXFMConfig::num_nodes)
        .def_readwrite("memory_dim", &SNEPPXFMConfig::memory_dim)
        .def_readwrite("memory_capacity", &SNEPPXFMConfig::memory_capacity)
        .def_readwrite("sync_interval", &SNEPPXFMConfig::sync_interval)
        .def_readwrite("sync_method", &SNEPPXFMConfig::sync_method)
        .def_readwrite("compression_ratio", &SNEPPXFMConfig::compression_ratio)
        .def_readwrite("privacy_epsilon", &SNEPPXFMConfig::privacy_epsilon)
        .def_readwrite("catastrophic_forgetting_protection", &SNEPPXFMConfig::catastrophic_forgetting_protection)
        .def_readwrite("ewm_alpha", &SNEPPXFMConfig::ewm_alpha)
        .def("default", &SNEPPX_fm_config_default);

    py::class_<SNEPPXNPEInstruction>(m, "SNEPPXNPEInstruction")
        .def(py::init())
        .def_readwrite("opcode", &SNEPPXNPEInstruction::opcode)
        .def_readwrite("dest_reg", &SNEPPXNPEInstruction::dest_reg)
        .def_readwrite("src_reg_a", &SNEPPXNPEInstruction::src_reg_a)
        .def_readwrite("src_reg_b", &SNEPPXNPEInstruction::src_reg_b)
        .def_readwrite("immediate", &SNEPPXNPEInstruction::immediate)
        .def_property("shape_a",
            [](SNEPPXNPEInstruction& inst) { return py::make_tuple(inst.shape_a[0], inst.shape_a[1]); },
            [](SNEPPXNPEInstruction& inst, py::tuple t) {
                inst.shape_a[0] = t[0].cast<int>();
                inst.shape_a[1] = t[1].cast<int>();
            })
        .def_property("shape_b",
            [](SNEPPXNPEInstruction& inst) { return py::make_tuple(inst.shape_b[0], inst.shape_b[1]); },
            [](SNEPPXNPEInstruction& inst, py::tuple t) {
                inst.shape_b[0] = t[0].cast<int>();
                inst.shape_b[1] = t[1].cast<int>();
            });

    // Model wrappers
    py::class_<PyModel>(m, "_Model")
        .def(py::init<>())
        .def_static("create", &PyModel::create)
        .def("forward", &PyModel::forward)
        .def("parameters", &PyModel::parameters);

    py::class_<PyHSSModel>(m, "_HSSModel")
        .def(py::init<>())
        .def_static("create", &PyHSSModel::create)
        .def("forward", &PyHSSModel::forward)
        .def("parameters", &PyHSSModel::parameters)
        .def("discretize_layer", &PyHSSModel::discretize_layer);

    py::class_<PySERModel>(m, "_SERModel")
        .def(py::init<>())
        .def_static("create", &PySERModel::create)
        .def("parameters", &PySERModel::parameters)
        .def("forward", &PySERModel::forward);

    py::class_<PyARCLayer>(m, "_ARCLayer")
        .def(py::init<>())
        .def_static("create", &PyARCLayer::create);
}

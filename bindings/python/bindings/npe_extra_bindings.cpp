/*
 * SNEPPX - Npe Extra Bindings
 *
 * WHAT
 *   Npe Extra Bindings.
 *
 * CONCEPT
 *   Provides the Npe Extra Bindings.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// NPE, FM, Trainer, Memory, ThreadPool — extracted from bindings.cpp lines 802–983, 1597–1645
#pragma once

class PyNPEVM {
public:
    SNEPPXNPEVM* ptr;
    bool owner;
    PyNPEVM() : ptr(nullptr), owner(false) {}
    explicit PyNPEVM(SNEPPXNPEVM* m, bool own = true) : ptr(m), owner(own) {}
    ~PyNPEVM() { if (owner && ptr) SNEPPX_npe_vm_destroy(ptr); }

    static PyNPEVM create(const SNEPPXNPEConfig& config) {
        return PyNPEVM(SNEPPX_npe_vm_create(&config));
    }

    void load(PyNPEVM& vm, SNEPPXNPEProgram* prog) { SNEPPX_npe_vm_load(vm.ptr, prog); }

    PyTensor run(const PyTensor& input) {
        SNEPPXTensor* out = nullptr;
        if (SNEPPX_npe_vm_run(ptr, input.ptr, &out) != 0)
            throw std::runtime_error("NPE VM run failed");
        return PyTensor(out);
    }
};

class PyNPEProgram {
public:
    SNEPPXNPEProgram* ptr;
    bool owner;
    PyNPEProgram() : ptr(nullptr), owner(false) {}
    explicit PyNPEProgram(SNEPPXNPEProgram* m, bool own = true) : ptr(m), owner(own) {}
    ~PyNPEProgram() { if (owner && ptr) SNEPPX_npe_program_destroy(ptr); }

    static PyNPEProgram create(size_t max_instructions) {
        return PyNPEProgram(SNEPPX_npe_program_create(max_instructions));
    }

    void append(SNEPPXNPEInstruction inst) { SNEPPX_npe_program_append(ptr, inst); }

    static PyNPEProgram compile_attention(size_t seq_len, size_t dim) {
        return PyNPEProgram(SNEPPX_npe_compile_attention(seq_len, dim));
    }

    static PyNPEProgram compile_mlp(size_t dim, size_t hidden_dim) {
        return PyNPEProgram(SNEPPX_npe_compile_mlp(dim, hidden_dim));
    }
};

class PyFMController {
public:
    SNEPPXFMController* ptr;
    bool owner;
    PyFMController() : ptr(nullptr), owner(false) {}
    explicit PyFMController(SNEPPXFMController* m, bool own = true) : ptr(m), owner(own) {}
    ~PyFMController() { if (owner && ptr) SNEPPX_fm_controller_destroy(ptr); }

    static PyFMController create(const SNEPPXFMConfig& config) {
        return PyFMController(SNEPPX_fm_controller_create(&config));
    }

    PyTensor forward(size_t node_id, const PyTensor& input) {
        SNEPPXTensor* out = nullptr;
        if (SNEPPX_fm_forward(ptr, node_id, input.ptr, &out) != 0)
            throw std::runtime_error("FM forward failed");
        return PyTensor(out);
    }

    int sync_all_reduce() { return SNEPPX_fm_sync_all_reduce(ptr); }
    int sync_gossip(size_t num_pairs) { return SNEPPX_fm_sync_gossip(ptr, num_pairs); }
    int sync_topology() { return SNEPPX_fm_sync_topology(ptr); }
};

class PyTrainer {
public:
    SNEPPXTrainer* ptr;
    bool owner;
    SNEPPXModel* model_ptr;

    PyTrainer() : ptr(nullptr), owner(false), model_ptr(nullptr) {}
    PyTrainer(SNEPPXTrainer* t, SNEPPXModel* m, bool own = true)
        : ptr(t), owner(own), model_ptr(m) {}
    ~PyTrainer() { if (owner && ptr) SNEPPX_trainer_destroy(ptr); }

    static PyTrainer create(PyModel& model, const SNEPPXTrainConfig& config) {
        auto* t = SNEPPX_trainer_create(model.ptr, &config);
        return PyTrainer(t, model.ptr);
    }

    float train_step(const PyTensor& input, const PyTensor& target) {
        return SNEPPX_trainer_train_step(ptr, input.ptr, target.ptr);
    }

    float evaluate(const PyTensor& val_input, const PyTensor& val_target) {
        return SNEPPX_trainer_evaluate(ptr, val_input.ptr, val_target.ptr);
    }

    void save_checkpoint(const std::string& path) {
        if (SNEPPX_trainer_save_checkpoint(ptr, path.c_str()) != 0)
            throw std::runtime_error("failed to save checkpoint");
    }

    void load_checkpoint(const std::string& path) {
        if (SNEPPX_trainer_load_checkpoint(ptr, path.c_str()) != 0)
            throw std::runtime_error("failed to load checkpoint");
    }
};

class PyMemoryPool {
public:
    static int init() { return SNEPPX_mem_pool_init(); }
    static void destroy() { SNEPPX_mem_pool_destroy(); }
    static void tls_cache_init() { SNEPPX_tls_cache_init(); }
    static void tls_cache_destroy() { SNEPPX_tls_cache_destroy(); }
    static void* pool_alloc(size_t size) { return SNEPPX_pool_alloc(size); }
    static void pool_free(void* ptr, size_t size) { SNEPPX_pool_free(ptr, size); }
    static py::dict stats() {
        SNEPPXMemStats s;
        SNEPPX_mem_pool_stats(&s);
        py::dict d;
        d["total_pool_allocated"] = s.total_pool_allocated;
        d["total_pool_freed"] = s.total_pool_freed;
        d["total_chunks"] = s.total_chunks;
        d["total_pool_hits"] = s.total_pool_hits;
        d["total_tls_hits"] = s.total_tls_hits;
        d["active_tls_caches"] = s.active_tls_caches;
        return d;
    }
    static void print_stats() { SNEPPX_mem_pool_print_stats(); }
};

class PyThreadPool {
public:
    SNEPPXThreadPool* ptr;
    bool owner;
    PyThreadPool() : ptr(nullptr), owner(false) {}
    explicit PyThreadPool(SNEPPXThreadPool* p, bool own = true) : ptr(p), owner(own) {}
    ~PyThreadPool() { if (owner && ptr) SNEPPX_threadpool_destroy(ptr); }

    static PyThreadPool create(size_t num_threads) {
        return PyThreadPool(SNEPPX_threadpool_create(num_threads));
    }

    int submit(py::function func, py::args args) {
        auto* callable = new py::function(func);
        auto* callable_args = new py::args(args);
        SNEPPXTask task;
        task.arg = callable;
        task.func = [](void* arg) {
            auto* fn = static_cast<py::function*>(arg);
        };
        return SNEPPX_threadpool_submit(ptr, task);
    }

    void wait() { SNEPPX_threadpool_wait(ptr); }

    static size_t default_count() { return SNEPPX_threadpool_default_count(); }

    static void parallel_for(PyThreadPool& pool, size_t start, size_t end, py::function func) {
        auto* fn = new py::function(func);
        SNEPPX_parallel_for(pool.ptr, start, end,
            [](size_t s, size_t e, void* arg) {
                auto* f = static_cast<py::function*>(arg);
                (*f)(s, e);
            }, fn);
        delete fn;
    }
};

static void init_npe_extra(py::module& m) {
    py::class_<PyNPEVM>(m, "_NPEVM")
        .def(py::init<>())
        .def_static("create", &PyNPEVM::create)
        .def("load", &PyNPEVM::load)
        .def("run", &PyNPEVM::run);

    py::class_<PyNPEProgram>(m, "_NPEProgram")
        .def(py::init<>())
        .def_static("create", &PyNPEProgram::create)
        .def("append", &PyNPEProgram::append)
        .def_static("compile_attention", &PyNPEProgram::compile_attention)
        .def_static("compile_mlp", &PyNPEProgram::compile_mlp);

    py::class_<PyFMController>(m, "_FMController")
        .def(py::init<>())
        .def_static("create", &PyFMController::create)
        .def("forward", &PyFMController::forward)
        .def("sync_all_reduce", &PyFMController::sync_all_reduce)
        .def("sync_gossip", &PyFMController::sync_gossip)
        .def("sync_topology", &PyFMController::sync_topology);

    py::class_<PyTrainer>(m, "_Trainer")
        .def(py::init<>())
        .def_static("create", &PyTrainer::create)
        .def("train_step", &PyTrainer::train_step)
        .def("evaluate", &PyTrainer::evaluate)
        .def("save_checkpoint", &PyTrainer::save_checkpoint)
        .def("load_checkpoint", &PyTrainer::load_checkpoint);

    py::class_<PyMemoryPool>(m, "_MemoryPool")
        .def_static("init", &PyMemoryPool::init)
        .def_static("destroy", &PyMemoryPool::destroy)
        .def_static("tls_cache_init", &PyMemoryPool::tls_cache_init)
        .def_static("tls_cache_destroy", &PyMemoryPool::tls_cache_destroy)
        .def_static("pool_alloc", &PyMemoryPool::pool_alloc)
        .def_static("pool_free", &PyMemoryPool::pool_free)
        .def_static("stats", &PyMemoryPool::stats)
        .def_static("print_stats", &PyMemoryPool::print_stats);

    py::class_<PyThreadPool>(m, "_ThreadPool")
        .def(py::init<>())
        .def_static("create", &PyThreadPool::create)
        .def("submit", &PyThreadPool::submit)
        .def("wait", &PyThreadPool::wait)
        .def_static("default_count", &PyThreadPool::default_count)
        .def_static("parallel_for", &PyThreadPool::parallel_for);
}

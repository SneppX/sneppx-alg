/*
 * SNEPPX - Autograd Bindings
 *
 * WHAT
 *   Autograd Bindings.
 *
 * CONCEPT
 *   Provides automatic differentiation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// Variable + Tape wrappers — extracted from bindings.cpp lines 503–600, 1398–1443
#pragma once

class PyVariable {
public:
    SNEPPXVariable* ptr;
    bool owner;

    PyVariable() : ptr(nullptr), owner(false) {}
    explicit PyVariable(SNEPPXVariable* v, bool own = true) : ptr(v), owner(own) {
        if (own && !v)
            throw std::runtime_error(
                "SNEPPX op returned NULL (invalid variable creation)");
    }
    ~PyVariable() { if (owner && ptr) SNEPPX_variable_destroy(ptr); }

    /* Relinquish python-side ownership (the tape owns the variable once it
     * is recorded; issue #14 teardown double-free). */
    void release() { owner = false; }

    PyVariable(const PyVariable&) = delete;
    PyVariable& operator=(const PyVariable&) = delete;
    PyVariable(PyVariable&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyVariable& operator=(PyVariable&& o) noexcept {
        if (this != &o) {
            if (owner && ptr) SNEPPX_variable_destroy(ptr);
            ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false;
        }
        return *this;
    }

    static PyVariable create(PyTensor& data, int requires_grad) {
        // SNEPPX_variable_create TAKES OWNERSHIP of the tensor (its destroy
        // path frees var->data). Relinquish the python-side PyTensor's
        // ownership on success so the tensor is destroyed exactly once
        // (issue #14 — double-free at teardown otherwise).
        PyVariable v(SNEPPX_variable_create(data.ptr, requires_grad));
        if (v.ptr)
            data.release();
        return v;
    }

    void zero_grad() { SNEPPX_variable_zero_grad(ptr); }
    PyVariable detach() { return PyVariable(SNEPPX_variable_detach(ptr)); }
    PyVariable copy() { return PyVariable(SNEPPX_variable_copy(ptr)); }
    float item() { return SNEPPX_variable_item(ptr); }
    size_t numel() { return SNEPPX_variable_numel(ptr); }
    void set_requires_grad(int v) { SNEPPX_variable_set_requires_grad(ptr, v); }
    PyTensor get_data() const { return PyTensor(ptr->data, false); }
    PyTensor get_grad() const { return ptr->grad ? PyTensor(ptr->grad, false) : PyTensor(); }
    int get_requires_grad() const { return ptr->requires_grad; }
};

class PyTape {
public:
    SNEPPXTape* ptr;
    bool owner;

    PyTape() : ptr(nullptr), owner(false) {}
    explicit PyTape(SNEPPXTape* t, bool own = true) : ptr(t), owner(own) {}
    ~PyTape() { if (owner && ptr) SNEPPX_tape_destroy(ptr); }

    PyTape(const PyTape&) = delete;
    PyTape& operator=(const PyTape&) = delete;
    PyTape(PyTape&& o) noexcept : ptr(o.ptr), owner(o.owner) { o.ptr = nullptr; o.owner = false; }
    PyTape& operator=(PyTape&& o) noexcept {
        if (this != &o) {
            if (owner && ptr) SNEPPX_tape_destroy(ptr);
            ptr = o.ptr; owner = o.owner; o.ptr = nullptr; o.owner = false;
        }
        return *this;
    }

    static PyTape create() { return PyTape(SNEPPX_tape_create()); }

    /* Ops record their result on this tape when grad tracking is active —
     * recorded variables are owned (and destroyed) by the tape, so the
     * python wrapper must not also destroy them. When no_grad is active the
     * result is NOT recorded and python keeps ownership. */
    PyVariable wrap_result(SNEPPXVariable* v) {
        PyVariable r(v);
        if (r.ptr && !SNEPPX_no_grad_is_active())
            r.release();
        return r;
    }

    void record(PyVariable& var) {
        SNEPPX_tape_record(ptr, var.ptr);
        var.release();
    }
    void backward(PyVariable& loss) { SNEPPX_tape_backward(ptr, loss.ptr); }
    void zero_grad() { SNEPPX_tape_zero_grad(ptr); }
    float global_norm() { return SNEPPX_tape_global_norm(ptr); }
    void clip_grad_norm(float max_norm) { SNEPPX_tape_clip_grad_norm(ptr, max_norm); }

    PyVariable add(PyVariable& a, PyVariable& b) { return wrap_result(SNEPPX_add(ptr, a.ptr, b.ptr)); }
    PyVariable sub(PyVariable& a, PyVariable& b) { return wrap_result(SNEPPX_sub(ptr, a.ptr, b.ptr)); }
    PyVariable mul(PyVariable& a, PyVariable& b) { return wrap_result(SNEPPX_mul(ptr, a.ptr, b.ptr)); }
    PyVariable div(PyVariable& a, PyVariable& b) { return wrap_result(SNEPPX_div(ptr, a.ptr, b.ptr)); }
    PyVariable pow_(PyVariable& a, PyVariable& b) { return wrap_result(SNEPPX_pow(ptr, a.ptr, b.ptr)); }
    PyVariable neg(PyVariable& a) { return wrap_result(SNEPPX_neg(ptr, a.ptr)); }
    PyVariable matmul(PyVariable& a, PyVariable& b) { return wrap_result(SNEPPX_matmul(ptr, a.ptr, b.ptr)); }
    PyVariable relu(PyVariable& a) { return wrap_result(SNEPPX_relu(ptr, a.ptr)); }
    PyVariable gelu(PyVariable& a) { return wrap_result(SNEPPX_gelu(ptr, a.ptr)); }
    PyVariable silu(PyVariable& a) { return wrap_result(SNEPPX_silu(ptr, a.ptr)); }
    PyVariable sigmoid(PyVariable& a) { return wrap_result(SNEPPX_sigmoid(ptr, a.ptr)); }
    PyVariable tanh(PyVariable& a) { return wrap_result(SNEPPX_tanh(ptr, a.ptr)); }
    PyVariable softmax(PyVariable& a, size_t dim) { return wrap_result(SNEPPX_softmax(ptr, a.ptr, dim)); }
    PyVariable exp(PyVariable& a) { return wrap_result(SNEPPX_exp(ptr, a.ptr)); }
    PyVariable log(PyVariable& a) { return wrap_result(SNEPPX_log(ptr, a.ptr)); }
    PyVariable sum(PyVariable& a, size_t dim) { return wrap_result(SNEPPX_sum(ptr, a.ptr, dim)); }
    PyVariable mean(PyVariable& a, size_t dim) { return wrap_result(SNEPPX_mean(ptr, a.ptr, dim)); }
    PyVariable transpose(PyVariable& a, size_t dim1, size_t dim2) { return wrap_result(SNEPPX_transpose(ptr, a.ptr, dim1, dim2)); }
    PyVariable dropout(PyVariable& a, float rate, unsigned int seed) { return wrap_result(SNEPPX_dropout(ptr, a.ptr, rate, seed)); }
    PyVariable layer_norm(PyVariable& a, PyVariable& gamma, PyVariable& beta, float eps) { return wrap_result(SNEPPX_layer_norm(ptr, a.ptr, gamma.ptr, beta.ptr, eps)); }
    PyVariable conv2d(PyVariable& input, PyVariable& kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) { return wrap_result(SNEPPX_conv2d(ptr, input.ptr, kernel.ptr, stride_h, stride_w, pad_h, pad_w)); }
    PyVariable concat(py::args args) {
        size_t dim = args[args.size() - 1].cast<size_t>();
        size_t n = args.size() - 1;
        std::vector<SNEPPXVariable*> raw(n);
        for (size_t i = 0; i < n; ++i)
            raw[i] = args[i].cast<PyVariable*>()->ptr;
        return wrap_result(SNEPPX_concat(ptr, raw.data(), raw.size(), dim));
    }
    PyVariable mse_loss(PyVariable& pred, PyVariable& target) { return wrap_result(SNEPPX_mse_loss(ptr, pred.ptr, target.ptr)); }
};

static void init_autograd(py::module& m) {
    py::class_<PyVariable>(m, "_Variable")
        .def(py::init<>())
        .def_static("create", &PyVariable::create)
        .def("zero_grad", &PyVariable::zero_grad)
        .def("detach", &PyVariable::detach)
        .def("copy", &PyVariable::copy)
        .def("item", &PyVariable::item)
        .def("numel", &PyVariable::numel)
        .def("set_requires_grad", &PyVariable::set_requires_grad)
        .def_property_readonly("data", &PyVariable::get_data)
        .def_property_readonly("grad", &PyVariable::get_grad)
        .def_property_readonly("requires_grad", &PyVariable::get_requires_grad);

    py::class_<PyTape>(m, "_Tape")
        .def(py::init<>())
        .def_static("create", &PyTape::create)
        .def("record", &PyTape::record)
        .def("backward", &PyTape::backward)
        .def("zero_grad", &PyTape::zero_grad)
        .def("global_norm", &PyTape::global_norm)
        .def("clip_grad_norm", &PyTape::clip_grad_norm)
        .def("add", &PyTape::add)
        .def("sub", &PyTape::sub)
        .def("mul", &PyTape::mul)
        .def("div", &PyTape::div)
        .def("pow_", &PyTape::pow_)
        .def("neg", &PyTape::neg)
        .def("matmul", &PyTape::matmul)
        .def("relu", &PyTape::relu)
        .def("gelu", &PyTape::gelu)
        .def("silu", &PyTape::silu)
        .def("sigmoid", &PyTape::sigmoid)
        .def("tanh", &PyTape::tanh)
        .def("softmax", &PyTape::softmax)
        .def("exp", &PyTape::exp)
        .def("log", &PyTape::log)
        .def("sum", &PyTape::sum)
        .def("mean", &PyTape::mean)
        .def("transpose", &PyTape::transpose)
        .def("dropout", &PyTape::dropout)
        .def("layer_norm", &PyTape::layer_norm)
        .def("conv2d", &PyTape::conv2d)
        .def("concat", &PyTape::concat)
        .def("mse_loss", &PyTape::mse_loss);
}

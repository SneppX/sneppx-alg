/*
 * SNEPPX - Tensor Bindings
 *
 * WHAT
 *   Tensor Bindings.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// Tensor wrapper — extracted from bindings.cpp lines 86–498, 1252–1396
#pragma once

class PyTensor {
public:
    SNEPPXTensor* ptr;
    bool owner;

    PyTensor() : ptr(nullptr), owner(false) {}
    explicit PyTensor(SNEPPXTensor* t, bool own = true) : ptr(t), owner(own) {
        // Ops return NULL for invalid inputs (shape/dtype mismatch,
        // unsupported cases) — fail fast with a python-visible error
        // instead of letting a NULL ptr segfault later lifetimes (issue #14).
        if (own && !t)
            throw std::runtime_error(
                "SNEPPX op returned NULL (invalid shape/dtype combination for "
                "this op)");
    }
    ~PyTensor() { if (owner && ptr) SNEPPX_tensor_destroy(ptr); }

    /* Relinquish python-side ownership: used by PyVariable::create because
     * SNEPPX_variable_create takes ownership of the tensor pointer. */
    void release() { owner = false; }

    PyTensor(const PyTensor&) = delete;
    PyTensor& operator=(const PyTensor&) = delete;

    PyTensor(PyTensor&& other) noexcept : ptr(other.ptr), owner(other.owner) {
        other.ptr = nullptr;
        other.owner = false;
    }
    PyTensor& operator=(PyTensor&& other) noexcept {
        if (this != &other) {
            if (owner && ptr) SNEPPX_tensor_destroy(ptr);
            ptr = other.ptr;
            owner = other.owner;
            other.ptr = nullptr;
            other.owner = false;
        }
        return *this;
    }

    // ---- Factory methods ----
    static PyTensor create(py::array shape, SNEPPXDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(SNEPPX_tensor_create(sv.data(), sv.size(), dtype));
    }

    static PyTensor empty(py::array shape, SNEPPXDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(SNEPPX_tensor_empty(sv.data(), sv.size(), dtype));
    }

    static PyTensor zeros(py::array shape, SNEPPXDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(SNEPPX_tensor_zeros(sv.data(), sv.size(), dtype));
    }

    static PyTensor ones(py::array shape, SNEPPXDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(SNEPPX_tensor_ones(sv.data(), sv.size(), dtype));
    }

    static PyTensor full(py::array shape, SNEPPXDtype dtype, py::object val_obj) {
        auto sv = pyarray_to_sizevec(shape);
        std::vector<uint8_t> buf(SNEPPX_dtype_itemsize(dtype));
        if (dtype == SNEPPX_FLOAT32) {
            float v = val_obj.cast<float>();
            std::memcpy(buf.data(), &v, sizeof(float));
        } else if (dtype == SNEPPX_FLOAT64) {
            double v = val_obj.cast<double>();
            std::memcpy(buf.data(), &v, sizeof(double));
        } else if (dtype == SNEPPX_INT32) {
            int32_t v = val_obj.cast<int32_t>();
            std::memcpy(buf.data(), &v, sizeof(int32_t));
        } else if (dtype == SNEPPX_INT64) {
            int64_t v = val_obj.cast<int64_t>();
            std::memcpy(buf.data(), &v, sizeof(int64_t));
        } else if (dtype == SNEPPX_UINT8 || dtype == SNEPPX_BOOL) {
            uint8_t v = val_obj.cast<uint8_t>();
            std::memcpy(buf.data(), &v, sizeof(uint8_t));
        } else {
            throw std::runtime_error("full() unsupported dtype for value");
        }
        return PyTensor(SNEPPX_tensor_full(sv.data(), sv.size(), dtype, buf.data()));
    }

    static PyTensor arange(float start, float stop, float step, SNEPPXDtype dtype) {
        return PyTensor(SNEPPX_tensor_arange(start, stop, step, dtype));
    }

    static PyTensor linspace(float start, float stop, size_t steps, SNEPPXDtype dtype) {
        return PyTensor(SNEPPX_tensor_linspace(start, stop, steps, dtype));
    }

    static PyTensor eye(size_t n, SNEPPXDtype dtype) {
        return PyTensor(SNEPPX_tensor_eye(n, dtype));
    }

    static PyTensor randn(py::array shape, SNEPPXDtype dtype) {
        auto sv = pyarray_to_sizevec(shape);
        return PyTensor(SNEPPX_tensor_randn(sv.data(), sv.size(), dtype));
    }

    // ---- Properties ----
    py::tuple shape() const {
        py::tuple t(ptr->ndim);
        for (size_t i = 0; i < ptr->ndim; ++i)
            t[i] = py::int_(ptr->shape[i]);
        return t;
    }

    size_t ndim() const { return ptr->ndim; }
    size_t size() const { return ptr->size; }
    SNEPPXDtype dtype() const { return ptr->dtype; }
    SNEPPXDevice device() const { return ptr->device; }
    SNEPPXLayout layout() const { return ptr->layout; }
    std::string dtype_name() const { return SNEPPX_tensor_dtype_name(ptr->dtype); }
    size_t numel() const { return SNEPPX_tensor_numel(ptr); }
    int is_contiguous() const { return SNEPPX_tensor_is_contiguous(ptr); }

    // ---- Data access (read/write via numpy) ----
    py::array data() const {
        size_t itemsize = SNEPPX_dtype_itemsize(ptr->dtype);
        std::vector<py::ssize_t> np_shape(ptr->shape, ptr->shape + ptr->ndim);
        std::vector<py::ssize_t> np_strides(ptr->ndim);
        if (ptr->strides) {
            // SNEPPXTensor.strides are ELEMENT strides (see compute_offset in
            // kernel/tensor/tensor.c: `((float*)data)[offset]`), while numpy
            // expects BYTE strides — multiply by the item size (issue #14).
            for (size_t i = 0; i < ptr->ndim; ++i)
                np_strides[i] =
                    static_cast<py::ssize_t>(ptr->strides[i]) *
                    static_cast<py::ssize_t>(itemsize);
        } else {
            np_strides[ptr->ndim - 1] = itemsize;
            for (size_t i = ptr->ndim; i > 1; --i)
                np_strides[i - 2] = np_strides[i - 1] * static_cast<py::ssize_t>(ptr->shape[i - 1]);
        }
        std::string np_dt = SNEPPX_dtype_to_numpy(ptr->dtype);
        return py::array(py::dtype(np_dt), np_shape, np_strides, ptr->data, py::cast(*this));
    }

    void set_data(py::array arr) {
        py::buffer_info buf = arr.request();
        size_t nbytes = ptr->size * SNEPPX_dtype_itemsize(ptr->dtype);
        if (buf.size * buf.itemsize != nbytes)
            throw std::runtime_error("data size mismatch");
        std::memcpy(ptr->data, buf.ptr, nbytes);
    }

    // ---- Element access ----
    py::object get_item(py::args indices) const {
        std::vector<size_t> idx(ptr->ndim, 0);
        for (size_t i = 0; i < indices.size() && i < ptr->ndim; ++i)
            idx[i] = indices[i].cast<size_t>();
        switch (ptr->dtype) {
            case SNEPPX_FLOAT32: return py::float_(SNEPPX_tensor_get_f32(ptr, idx.data()));
            case SNEPPX_FLOAT64: return py::float_(SNEPPX_tensor_get_f64(ptr, idx.data()));
            case SNEPPX_INT32:   return py::int_(SNEPPX_tensor_get_i32(ptr, idx.data()));
            case SNEPPX_INT64:   return py::int_(SNEPPX_tensor_get_i64(ptr, idx.data()));
            case SNEPPX_BOOL:    return py::bool_(SNEPPX_tensor_get_bool(ptr, idx.data()));
            default: throw std::runtime_error("get_item unsupported dtype");
        }
    }

    void set_item(py::args args) {
        if (args.size() < 2) throw std::runtime_error("set_item needs indices + value");
        auto val = args[args.size() - 1];
        std::vector<size_t> idx(args.size() - 1, 0);
        for (size_t i = 0; i < args.size() - 1; ++i)
            idx[i] = args[i].cast<size_t>();
        switch (ptr->dtype) {
            case SNEPPX_FLOAT32: SNEPPX_tensor_set_f32(ptr, idx.data(), val.cast<float>()); break;
            case SNEPPX_FLOAT64: SNEPPX_tensor_set_f64(ptr, idx.data(), val.cast<double>()); break;
            case SNEPPX_INT32:   SNEPPX_tensor_set_i32(ptr, idx.data(), val.cast<int32_t>()); break;
            case SNEPPX_INT64:   SNEPPX_tensor_set_i64(ptr, idx.data(), val.cast<int64_t>()); break;
            case SNEPPX_BOOL:    SNEPPX_tensor_set_bool(ptr, idx.data(), val.cast<uint8_t>()); break;
            default: throw std::runtime_error("set_item unsupported dtype");
        }
    }

    // ---- Fill ----
    void fill_f32(float v) { SNEPPX_tensor_fill_f32(ptr, v); }
    void fill_f64(double v) { SNEPPX_tensor_fill_f64(ptr, v); }

    // ---- Unary ops ----
    PyTensor neg()    const { return PyTensor(SNEPPX_tensor_neg(ptr)); }
    PyTensor abs_()   const { return PyTensor(SNEPPX_tensor_abs(ptr)); }
    PyTensor sign()   const { return PyTensor(SNEPPX_tensor_sign(ptr)); }
    PyTensor floor()  const { return PyTensor(SNEPPX_tensor_floor(ptr)); }
    PyTensor ceil()   const { return PyTensor(SNEPPX_tensor_ceil(ptr)); }
    PyTensor round()  const { return PyTensor(SNEPPX_tensor_round(ptr)); }
    PyTensor trunc()  const { return PyTensor(SNEPPX_tensor_trunc(ptr)); }
    PyTensor exp()    const { return PyTensor(SNEPPX_tensor_exp(ptr)); }
    PyTensor log()    const { return PyTensor(SNEPPX_tensor_log(ptr)); }
    PyTensor sqrt()   const { return PyTensor(SNEPPX_tensor_sqrt(ptr)); }
    PyTensor sin()    const { return PyTensor(SNEPPX_tensor_sin(ptr)); }
    PyTensor cos()    const { return PyTensor(SNEPPX_tensor_cos(ptr)); }
    PyTensor tan()    const { return PyTensor(SNEPPX_tensor_tan(ptr)); }
    PyTensor asin()   const { return PyTensor(SNEPPX_tensor_asin(ptr)); }
    PyTensor acos()   const { return PyTensor(SNEPPX_tensor_acos(ptr)); }
    PyTensor atan()   const { return PyTensor(SNEPPX_tensor_atan(ptr)); }
    PyTensor sinh()   const { return PyTensor(SNEPPX_tensor_sinh(ptr)); }
    PyTensor cosh()   const { return PyTensor(SNEPPX_tensor_cosh(ptr)); }
    PyTensor tanh()   const { return PyTensor(SNEPPX_tensor_tanh(ptr)); }

    // ---- Binary arithmetic ----
    PyTensor add(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_add(ptr, b.ptr)); }
    PyTensor sub(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_sub(ptr, b.ptr)); }
    PyTensor mul(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_mul(ptr, b.ptr)); }
    PyTensor div(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_div(ptr, b.ptr)); }
    PyTensor pow_(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_pow(ptr, b.ptr)); }

    // ---- Comparison ----
    PyTensor eq(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_eq(ptr, b.ptr)); }
    PyTensor ne(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_ne(ptr, b.ptr)); }
    PyTensor lt(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_lt(ptr, b.ptr)); }
    PyTensor le(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_le(ptr, b.ptr)); }
    PyTensor gt(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_gt(ptr, b.ptr)); }
    PyTensor ge(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_ge(ptr, b.ptr)); }

    // ---- Activations ----
    PyTensor relu()     const { return PyTensor(SNEPPX_tensor_relu(ptr)); }
    PyTensor gelu()     const { return PyTensor(SNEPPX_tensor_gelu(ptr)); }
    PyTensor silu()     const { return PyTensor(SNEPPX_tensor_silu(ptr)); }
    PyTensor sigmoid()  const { return PyTensor(SNEPPX_tensor_sigmoid(ptr)); }
    PyTensor softmax(size_t dim) const { return PyTensor(SNEPPX_tensor_softmax(ptr, dim)); }
    PyTensor log_softmax(size_t dim) const { return PyTensor(SNEPPX_tensor_log_softmax(ptr, dim)); }
    PyTensor tanh_act() const { return PyTensor(SNEPPX_tensor_tanh(ptr)); }

    // ---- Reductions ----
    PyTensor sum(size_t dim) const { return PyTensor(SNEPPX_tensor_sum(ptr, dim)); }
    PyTensor mean(size_t dim) const { return PyTensor(SNEPPX_tensor_mean(ptr, dim)); }
    PyTensor var(size_t dim) const { return PyTensor(SNEPPX_tensor_var(ptr, dim)); }
    PyTensor std(size_t dim) const { return PyTensor(SNEPPX_tensor_std(ptr, dim)); }
    float min() const { return SNEPPX_tensor_min(ptr); }
    float max() const { return SNEPPX_tensor_max(ptr); }
    size_t argmin() const { return SNEPPX_tensor_argmin(ptr); }
    size_t argmax() const { return SNEPPX_tensor_argmax(ptr); }
    PyTensor cumsum(size_t dim) const { return PyTensor(SNEPPX_tensor_cumsum(ptr, dim)); }
    PyTensor cumprod(size_t dim) const { return PyTensor(SNEPPX_tensor_cumprod(ptr, dim)); }

    // ---- Linear algebra ----
    float dot(const PyTensor& b) const { return SNEPPX_tensor_dot(ptr, b.ptr); }
    PyTensor matmul(const PyTensor& b) const { return PyTensor(SNEPPX_tensor_matmul(ptr, b.ptr)); }
    PyTensor transpose(size_t dim1, size_t dim2) const { return PyTensor(SNEPPX_tensor_transpose(ptr, dim1, dim2)); }
    PyTensor inverse() const { return PyTensor(SNEPPX_tensor_inverse(ptr)); }
    float det() const { return SNEPPX_tensor_det(ptr); }

    // ---- Shape ops ----
    PyTensor reshape(py::array new_shape) const {
        auto sv = pyarray_to_sizevec(new_shape);
        return PyTensor(SNEPPX_tensor_reshape(ptr, sv.data(), sv.size()));
    }

    PyTensor permute(py::array axes) const {
        auto sv = pyarray_to_sizevec(axes);
        return PyTensor(SNEPPX_tensor_permute(ptr, sv.data()));
    }

    PyTensor expand(py::array new_shape) const {
        auto sv = pyarray_to_sizevec(new_shape);
        return PyTensor(SNEPPX_tensor_expand(ptr, sv.data(), sv.size()));
    }

    PyTensor squeeze(size_t dim) const { return PyTensor(SNEPPX_tensor_squeeze(ptr, dim)); }
    PyTensor unsqueeze(size_t dim) const { return PyTensor(SNEPPX_tensor_unsqueeze(ptr, dim)); }
    PyTensor slice(size_t dim, size_t start, size_t end) const { return PyTensor(SNEPPX_tensor_slice(ptr, dim, start, end)); }

    static PyTensor concat(py::list tensors, size_t dim) {
        std::vector<const SNEPPXTensor*> raw(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i)
            raw[i] = tensors[i].cast<PyTensor*>()->ptr;
        return PyTensor(SNEPPX_tensor_concat(raw.data(), raw.size(), dim));
    }

    std::vector<PyTensor> split(size_t num_splits, size_t dim) const {
        SNEPPXTensor** results = SNEPPX_tensor_split(ptr, num_splits, dim);
        std::vector<PyTensor> out;
        for (size_t i = 0; i < num_splits; ++i)
            out.emplace_back(results[i]);
        delete[] results;
        return out;
    }

    PyTensor tile(py::array reps) const {
        auto sv = pyarray_to_sizevec(reps);
        return PyTensor(SNEPPX_tensor_tile(ptr, sv.data(), sv.size()));
    }

    PyTensor repeat(size_t repeats, size_t dim) const {
        return PyTensor(SNEPPX_tensor_repeat(ptr, repeats, dim));
    }

    PyTensor gather(size_t dim, const PyTensor& indices) const {
        return PyTensor(SNEPPX_tensor_gather(ptr, dim, indices.ptr));
    }

    PyTensor scatter(size_t dim, const PyTensor& indices, const PyTensor& src) {
        return PyTensor(SNEPPX_tensor_scatter(ptr, dim, indices.ptr, src.ptr));
    }

    PyTensor masked_select(const PyTensor& mask) const {
        return PyTensor(SNEPPX_tensor_masked_select(ptr, mask.ptr));
    }

    PyTensor masked_fill(const PyTensor& mask, py::object val_obj) {
        SNEPPXDtype dt = ptr->dtype;
        std::vector<uint8_t> buf(SNEPPX_dtype_itemsize(dt));
        if (dt == SNEPPX_FLOAT32) {
            float v = val_obj.cast<float>(); std::memcpy(buf.data(), &v, sizeof(float));
        } else if (dt == SNEPPX_FLOAT64) {
            double v = val_obj.cast<double>(); std::memcpy(buf.data(), &v, sizeof(double));
        } else if (dt == SNEPPX_INT32) {
            int32_t v = val_obj.cast<int32_t>(); std::memcpy(buf.data(), &v, sizeof(int32_t));
        } else if (dt == SNEPPX_INT64) {
            int64_t v = val_obj.cast<int64_t>(); std::memcpy(buf.data(), &v, sizeof(int64_t));
        } else if (dt == SNEPPX_UINT8 || dt == SNEPPX_BOOL) {
            uint8_t v = val_obj.cast<uint8_t>(); std::memcpy(buf.data(), &v, sizeof(uint8_t));
        } else {
            throw std::runtime_error("masked_fill unsupported dtype");
        }
        return PyTensor(SNEPPX_tensor_masked_fill(ptr, mask.ptr, buf.data()));
    }

    static PyTensor where(const PyTensor& condition, const PyTensor& x, const PyTensor& y) {
        return PyTensor(SNEPPX_tensor_where(condition.ptr, x.ptr, y.ptr));
    }

    // ---- Cast / Device / Layout ----
    PyTensor cast(SNEPPXDtype dtype) const { return PyTensor(SNEPPX_tensor_cast(ptr, dtype)); }
    PyTensor to_device(SNEPPXDevice device) const { return PyTensor(SNEPPX_tensor_to_device(ptr, device)); }
    PyTensor to_layout(SNEPPXLayout layout) const { return PyTensor(SNEPPX_tensor_to_layout(ptr, layout)); }

    // ---- I/O ----
    void save(const std::string& path) const {
        if (SNEPPX_tensor_save(ptr, path.c_str()) != 0)
            throw std::runtime_error("failed to save tensor");
    }

    static PyTensor load(const std::string& path) {
        SNEPPXTensor* t = SNEPPX_tensor_load(path.c_str());
        if (!t) throw std::runtime_error("failed to load tensor");
        return PyTensor(t);
    }

    PyTensor copy() const { return PyTensor(SNEPPX_tensor_copy(ptr)); }
    PyTensor clone() const { return PyTensor(SNEPPX_tensor_clone(ptr)); }

    // ---- NN ops ----
    PyTensor conv1d(const PyTensor& kernel, size_t stride, size_t padding) const {
        return PyTensor(SNEPPX_tensor_conv1d(ptr, kernel.ptr, stride, padding));
    }

    PyTensor conv2d(const PyTensor& kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w) const {
        return PyTensor(SNEPPX_tensor_conv2d(ptr, kernel.ptr, stride_h, stride_w, pad_h, pad_w));
    }

    PyTensor pool1d(size_t kernel_size, size_t stride) const {
        return PyTensor(SNEPPX_tensor_pool1d(ptr, kernel_size, stride));
    }

    PyTensor pool2d(size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w) const {
        return PyTensor(SNEPPX_tensor_pool2d(ptr, kernel_h, kernel_w, stride_h, stride_w));
    }

    PyTensor dropout(float rate, unsigned int seed) const {
        return PyTensor(SNEPPX_tensor_dropout(ptr, rate, seed));
    }

    PyTensor layer_norm(const PyTensor& gamma, const PyTensor& beta, float eps) const {
        return PyTensor(SNEPPX_tensor_layer_norm(ptr, gamma.ptr, beta.ptr, eps));
    }

    PyTensor batch_norm(const PyTensor& gamma, const PyTensor& beta,
                        const PyTensor& running_mean, const PyTensor& running_var, float eps) const {
        return PyTensor(SNEPPX_tensor_batch_norm(ptr, gamma.ptr, beta.ptr,
                                                running_mean.ptr, running_var.ptr, eps));
    }

    PyTensor group_norm(const PyTensor& gamma, const PyTensor& beta, size_t num_groups, float eps) const {
        return PyTensor(SNEPPX_tensor_group_norm(ptr, gamma.ptr, beta.ptr, num_groups, eps));
    }

    PyTensor instance_norm(const PyTensor& gamma, const PyTensor& beta, float eps) const {
        return PyTensor(SNEPPX_tensor_instance_norm(ptr, gamma.ptr, beta.ptr, eps));
    }

    PyTensor embedding(const PyTensor& indices) const {
        return PyTensor(SNEPPX_tensor_embedding(ptr, indices.ptr));
    }

    // ---- Loss ----
    PyTensor mse_loss(const PyTensor& target) const { return PyTensor(SNEPPX_tensor_mse_loss(ptr, target.ptr)); }
    PyTensor cross_entropy(const PyTensor& target) const { return PyTensor(SNEPPX_tensor_cross_entropy(ptr, target.ptr)); }
    PyTensor mae_loss(const PyTensor& target) const { return PyTensor(SNEPPX_tensor_mae_loss(ptr, target.ptr)); }
    PyTensor nll_loss(const PyTensor& target) const { return PyTensor(SNEPPX_tensor_nll_loss(ptr, target.ptr)); }
    PyTensor kl_div(const PyTensor& target) const { return PyTensor(SNEPPX_tensor_kl_div(ptr, target.ptr)); }
    PyTensor binary_cross_entropy(const PyTensor& target) const { return PyTensor(SNEPPX_tensor_binary_cross_entropy(ptr, target.ptr)); }

    void print() const { SNEPPX_tensor_print(ptr); }

    std::string repr() const {
        std::string s = "Tensor(shape=[";
        for (size_t i = 0; i < ptr->ndim; ++i) {
            if (i) s += ", ";
            s += std::to_string(ptr->shape[i]);
        }
        s += "], dtype=" + std::string(SNEPPX_tensor_dtype_name(ptr->dtype));
        s += ")";
        return s;
    }

    // Python operators via trampolines
    PyTensor __add__(const PyTensor& b) const { return add(b); }
    PyTensor __sub__(const PyTensor& b) const { return sub(b); }
    PyTensor __mul__(const PyTensor& b) const { return mul(b); }
    PyTensor __truediv__(const PyTensor& b) const { return div(b); }
    PyTensor __pow__(const PyTensor& b) const { return pow_(b); }
    PyTensor __neg__() const { return neg(); }

    PyTensor __eq__(const PyTensor& b) const { return eq(b); }
    PyTensor __ne__(const PyTensor& b) const { return ne(b); }
    PyTensor __lt__(const PyTensor& b) const { return lt(b); }
    PyTensor __le__(const PyTensor& b) const { return le(b); }
    PyTensor __gt__(const PyTensor& b) const { return gt(b); }
    PyTensor __ge__(const PyTensor& b) const { return ge(b); }

    // Internal: expose raw pointer for C interop in Python
    uintptr_t _to_tensor_ptr() const { return reinterpret_cast<uintptr_t>(ptr); }
};

static void init_tensor(py::module& m) {
    py::class_<PyTensor>(m, "_Tensor")
        .def(py::init<>())
        .def_static("create", &PyTensor::create)
        .def_static("empty", &PyTensor::empty)
        .def_static("zeros", &PyTensor::zeros)
        .def_static("ones", &PyTensor::ones)
        .def_static("full", &PyTensor::full)
        .def_static("arange", &PyTensor::arange)
        .def_static("linspace", &PyTensor::linspace)
        .def_static("eye", &PyTensor::eye)
        .def_static("randn", &PyTensor::randn)
        .def_static("load", &PyTensor::load)

        .def_property_readonly("shape", &PyTensor::shape)
        .def_property_readonly("ndim", &PyTensor::ndim)
        .def_property_readonly("size_", &PyTensor::size)
        .def_property_readonly("dtype", &PyTensor::dtype)
        .def_property_readonly("device", &PyTensor::device)
        .def_property_readonly("layout", &PyTensor::layout)
        .def_property_readonly("dtype_name", &PyTensor::dtype_name)
        .def_property_readonly("numel", &PyTensor::numel)
        .def_property_readonly("is_contiguous", &PyTensor::is_contiguous)
        .def("data", &PyTensor::data)
        .def("set_data", &PyTensor::set_data)
        .def("__getitem__", &PyTensor::get_item)
        .def("__setitem__", &PyTensor::set_item)
        .def("fill_f32", &PyTensor::fill_f32)
        .def("fill_f64", &PyTensor::fill_f64)
        .def("save", &PyTensor::save)
        .def("copy", &PyTensor::copy)
        .def("clone", &PyTensor::clone)
        .def("cast", &PyTensor::cast)
        .def("to_device", &PyTensor::to_device)
        .def("to_layout", &PyTensor::to_layout)
        .def("print", &PyTensor::print)
        .def("__repr__", &PyTensor::repr)

        .def("neg", &PyTensor::neg)
        .def("abs", &PyTensor::abs_)
        .def("sign", &PyTensor::sign)
        .def("floor", &PyTensor::floor)
        .def("ceil", &PyTensor::ceil)
        .def("round", &PyTensor::round)
        .def("trunc", &PyTensor::trunc)
        .def("exp", &PyTensor::exp)
        .def("log", &PyTensor::log)
        .def("sqrt", &PyTensor::sqrt)
        .def("sin", &PyTensor::sin)
        .def("cos", &PyTensor::cos)
        .def("tan", &PyTensor::tan)
        .def("asin", &PyTensor::asin)
        .def("acos", &PyTensor::acos)
        .def("atan", &PyTensor::atan)
        .def("sinh", &PyTensor::sinh)
        .def("cosh", &PyTensor::cosh)
        .def("tanh", &PyTensor::tanh)

        .def("add", &PyTensor::add)
        .def("sub", &PyTensor::sub)
        .def("mul", &PyTensor::mul)
        .def("div", &PyTensor::div)
        .def("pow_", &PyTensor::pow_)
        .def("__add__", &PyTensor::__add__)
        .def("__sub__", &PyTensor::__sub__)
        .def("__mul__", &PyTensor::__mul__)
        .def("__truediv__", &PyTensor::__truediv__)
        .def("__pow__", &PyTensor::__pow__)
        .def("__neg__", &PyTensor::__neg__)

        .def("__eq__", &PyTensor::__eq__)
        .def("__ne__", &PyTensor::__ne__)
        .def("__lt__", &PyTensor::__lt__)
        .def("__le__", &PyTensor::__le__)
        .def("__gt__", &PyTensor::__gt__)
        .def("__ge__", &PyTensor::__ge__)

        .def("relu", &PyTensor::relu)
        .def("gelu", &PyTensor::gelu)
        .def("silu", &PyTensor::silu)
        .def("sigmoid", &PyTensor::sigmoid)
        .def("softmax", &PyTensor::softmax)
        .def("log_softmax", &PyTensor::log_softmax)
        .def("tanh_act", &PyTensor::tanh_act)

        .def("sum", &PyTensor::sum)
        .def("mean", &PyTensor::mean)
        .def("var", &PyTensor::var)
        .def("std", &PyTensor::std)
        .def("min", &PyTensor::min)
        .def("max", &PyTensor::max)
        .def("argmin", &PyTensor::argmin)
        .def("argmax", &PyTensor::argmax)
        .def("cumsum", &PyTensor::cumsum)
        .def("cumprod", &PyTensor::cumprod)

        .def("dot", &PyTensor::dot)
        .def("matmul", &PyTensor::matmul)
        .def("transpose", &PyTensor::transpose)
        .def("inverse", &PyTensor::inverse)
        .def("det", &PyTensor::det)

        .def("reshape", &PyTensor::reshape)
        .def("permute", &PyTensor::permute)
        .def("expand", &PyTensor::expand)
        .def("squeeze", &PyTensor::squeeze)
        .def("unsqueeze", &PyTensor::unsqueeze)
        .def("slice", &PyTensor::slice)
        .def_static("concat", &PyTensor::concat)
        .def("split", &PyTensor::split)
        .def("tile", &PyTensor::tile)
        .def("repeat", &PyTensor::repeat)
        .def("gather", &PyTensor::gather)
        .def("scatter", &PyTensor::scatter)
        .def("masked_select", &PyTensor::masked_select)
        .def("masked_fill", &PyTensor::masked_fill)
        .def_static("where", &PyTensor::where)

        .def("conv1d", &PyTensor::conv1d)
        .def("conv2d", &PyTensor::conv2d)
        .def("pool1d", &PyTensor::pool1d)
        .def("pool2d", &PyTensor::pool2d)
        .def("dropout", &PyTensor::dropout)
        .def("layer_norm", &PyTensor::layer_norm)
        .def("batch_norm", &PyTensor::batch_norm)
        .def("group_norm", &PyTensor::group_norm)
        .def("instance_norm", &PyTensor::instance_norm)
        .def("embedding", &PyTensor::embedding)

        .def("mse_loss", &PyTensor::mse_loss)
        .def("cross_entropy", &PyTensor::cross_entropy)
        .def("mae_loss", &PyTensor::mae_loss)
        .def("nll_loss", &PyTensor::nll_loss)
        .def("kl_div", &PyTensor::kl_div)
        .def("binary_cross_entropy", &PyTensor::binary_cross_entropy)

        .def("_to_tensor_ptr", &PyTensor::_to_tensor_ptr);
}

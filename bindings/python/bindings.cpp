#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

/*
 * SNEPPX - Bindings
 *
 * WHAT
 *   Bindings.
 *
 * CONCEPT
 *   Provides the Bindings.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#include "multidimensional_tensor_engine.h"
#include "polymorphic_memory_allocator.h"
#include "concurrent_workload_dispatch.h"
#include "automatic_differentiation_framework.h"
#include "gradient_optimization_suite.h"
#include "system_architecture_definitions.h"
#include "differentiable_training_pipeline.h"
#include "hierarchical_state_space.h"
#include "sparse_expert_routing.h"
#include "adversarial_robustness_certification.h"
#include "neural_programming_engine.h"
#include "fractal_memory_orchestrator.h"
#include "cryptographic_primitives_bundle.h"
}

#include <cstring>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: map SNEPPXDtype to NumPy dtype string
// ---------------------------------------------------------------------------
/**
 * @brief Perform Dtype To Numpy.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_dtype_to_numpy(SNEPPXDtype d) {
    switch (d) {
        case SNEPPX_FLOAT32:   return "float32";
        case SNEPPX_FLOAT64:   return "float64";
        case SNEPPX_FLOAT16:   return "float16";
        // NumPy has no portable bfloat16/float8 dtype. Expose their raw
        // storage representations while dtype_name preserves semantic identity.
        case SNEPPX_BFLOAT16:  return "uint16";
        case SNEPPX_FLOAT8:    return "uint8";
        case SNEPPX_INT32:     return "int32";
        case SNEPPX_INT64:     return "int64";
        case SNEPPX_INT16:     return "int16";
        case SNEPPX_INT8:      return "int8";
        case SNEPPX_UINT8:     return "uint8";
        case SNEPPX_BOOL:      return "bool";
        case SNEPPX_COMPLEX64:  return "complex64";
        case SNEPPX_COMPLEX128: return "complex128";
        default: throw std::runtime_error("unknown dtype");
    }
}

/**
 * @brief Perform Dtype.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXDtype numpy_to_SNEPPX_dtype(const std::string& dt) {
    if (dt == "float32")   return SNEPPX_FLOAT32;
    if (dt == "float64")   return SNEPPX_FLOAT64;
    if (dt == "float16")   return SNEPPX_FLOAT16;
    if (dt == "bfloat16")  return SNEPPX_BFLOAT16;
    if (dt == "float8")    return SNEPPX_FLOAT8;
    if (dt == "int32")     return SNEPPX_INT32;
    if (dt == "int64")     return SNEPPX_INT64;
    if (dt == "int16")     return SNEPPX_INT16;
    if (dt == "int8")      return SNEPPX_INT8;
    if (dt == "uint8")     return SNEPPX_UINT8;
    if (dt == "bool")      return SNEPPX_BOOL;
    if (dt == "complex64")  return SNEPPX_COMPLEX64;
    if (dt == "complex128") return SNEPPX_COMPLEX128;
    throw std::runtime_error("unsupported numpy dtype: " + dt);
}

/**
 * @brief Perform Dtype Itemsize.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_dtype_itemsize(SNEPPXDtype d) {
    return SNEPPX_tensor_dtype_size(d);
}

static std::vector<size_t> pyarray_to_sizevec(py::array a, size_t max_dims = 0) {
    if (!a) throw std::invalid_argument("shape/repetitions array is required");
    py::buffer_info original = a.request();
    if (original.ndim != 1)
        throw std::invalid_argument("shape/repetitions must be a 1-D integer array");
    const std::string kind = py::cast<std::string>(a.dtype().attr("kind"));
    if (kind != "i" && kind != "u")
        throw std::invalid_argument("shape/repetitions must contain integers");
    if (max_dims != 0 && static_cast<size_t>(original.size) > max_dims)
        throw std::invalid_argument("dimension count exceeds the operation limit");

    using ShapeArray = py::array_t<py::ssize_t,
        py::array::c_style | py::array::forcecast>;
    ShapeArray converted(a);
    py::buffer_info buf = converted.request();
    const auto* values = static_cast<const py::ssize_t*>(buf.ptr);
    std::vector<size_t> result;
    result.reserve(static_cast<size_t>(buf.size));
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (values[i] < 0)
            throw std::invalid_argument("shape/repetitions cannot contain negative values");
        result.push_back(static_cast<size_t>(values[i]));
    }
    return result;
}

// ---- Include split module files ----
#include "bindings/tensor_bindings.cpp"
#include "bindings/autograd_bindings.cpp"
#include "bindings/optim_model_bindings.cpp"
#include "bindings/npe_extra_bindings.cpp"
#include "bindings/crypto_bindings.cpp"

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_SNEPPX_c, m) {
    m.doc() = "SNEPPX Algo - Python bindings";

    // ---- Enums ----
    py::enum_<SNEPPXDtype>(m, "SNEPPXDtype")
        .value("FLOAT32", SNEPPX_FLOAT32)
        .value("FLOAT64", SNEPPX_FLOAT64)
        .value("FLOAT16", SNEPPX_FLOAT16)
        .value("BFLOAT16", SNEPPX_BFLOAT16)
        .value("FLOAT8", SNEPPX_FLOAT8)
        .value("INT32", SNEPPX_INT32)
        .value("INT64", SNEPPX_INT64)
        .value("INT16", SNEPPX_INT16)
        .value("INT8", SNEPPX_INT8)
        .value("UINT8", SNEPPX_UINT8)
        .value("BOOL", SNEPPX_BOOL)
        .value("COMPLEX64", SNEPPX_COMPLEX64)
        .value("COMPLEX128", SNEPPX_COMPLEX128)
        .export_values();

    py::enum_<SNEPPXDevice>(m, "SNEPPXDevice")
        .value("CPU", SNEPPX_DEVICE_CPU)
        .value("CUDA", SNEPPX_DEVICE_CUDA)
        .value("METAL", SNEPPX_DEVICE_METAL)
        .value("VULKAN", SNEPPX_DEVICE_VULKAN)
        .value("TPU", SNEPPX_DEVICE_TPU)
        .value("NPU", SNEPPX_DEVICE_NPU)
        .export_values();

    py::enum_<SNEPPXLayout>(m, "SNEPPXLayout")
        .value("ROW_MAJOR", SNEPPX_LAYOUT_ROW_MAJOR)
        .value("COL_MAJOR", SNEPPX_LAYOUT_COL_MAJOR)
        .value("CHANNELS_LAST", SNEPPX_LAYOUT_CHANNELS_LAST)
        .value("TILED", SNEPPX_LAYOUT_TILED)
        .export_values();

    py::enum_<SNEPPXOptimizerType>(m, "SNEPPXOptimizerType")
        .value("SGD", SNEPPX_OPTIMIZER_SGD)
        .value("ADAM", SNEPPX_OPTIMIZER_ADAM)
        .value("ADAMW", SNEPPX_OPTIMIZER_ADAMW)
        .value("ADAMAX", SNEPPX_OPTIMIZER_ADAMAX)
        .value("RMSPROP", SNEPPX_OPTIMIZER_RMSPROP)
        .value("ADAGRAD", SNEPPX_OPTIMIZER_ADAGRAD)
        .value("ADADELTA", SNEPPX_OPTIMIZER_ADADELTA)
        .export_values();

    py::enum_<SNEPPXLRSchedulerType>(m, "SNEPPXLRSchedulerType")
        .value("STEP", SNEPPX_LR_STEP)
        .value("EXPONENTIAL", SNEPPX_LR_EXPONENTIAL)
        .value("COSINE", SNEPPX_LR_COSINE)
        .value("REDUCE_ON_PLATEAU", SNEPPX_LR_REDUCE_ON_PLATEAU)
        .export_values();

    py::enum_<SNEPPXObfuscationMethod>(m, "SNEPPXObfuscationMethod")
        .value("NONE", SNEPPX_OBF_NONE)
        .value("NOISE", SNEPPX_OBF_NOISE)
        .value("CLAMP", SNEPPX_OBF_CLAMP)
        .value("MIXED", SNEPPX_OBF_MIXED)
        .export_values();

    py::enum_<SNEPPXAttackType>(m, "SNEPPXAttackType")
        .value("FGSM", SNEPPX_ATTACK_FGSM)
        .value("PGD", SNEPPX_ATTACK_PGD)
        .value("CW", SNEPPX_ATTACK_CW)
        .export_values();

    py::enum_<SNEPPXActivation>(m, "SNEPPXActivation")
        .value("RELU", SNEPPX_ACT_RELU)
        .value("GELU", SNEPPX_ACT_GELU)
        .value("SWISH", SNEPPX_ACT_SWISH)
        .export_values();

    py::enum_<SNEPPXTopKMethod>(m, "SNEPPXTopKMethod")
        .value("GREEDY", SNEPPX_TOPK_GREEDY)
        .value("NOISY", SNEPPX_TOPK_NOISY)
        .export_values();

    py::enum_<SNEPPXFMSyncMethod>(m, "SNEPPXFMSyncMethod")
        .value("ALL_REDUCE", SNEPPX_SYNC_ALL_REDUCE)
        .value("GOSSIP", SNEPPX_SYNC_GOSSIP)
        .value("TOPOLOGY", SNEPPX_SYNC_TOPOLOGY)
        .export_values();

    py::enum_<SNEPPXNPEOpCode>(m, "SNEPPXNPEOpCode")
        .value("NOP", SNEPPX_NOP)
        .value("LOAD", SNEPPX_LOAD)
        .value("STORE", SNEPPX_STORE)
        .value("ADD", SNEPPX_ADD)
        .value("SUB", SNEPPX_SUB)
        .value("MUL", SNEPPX_MUL)
        .value("DIV", SNEPPX_DIV)
        .value("MATMUL", SNEPPX_MATMUL)
        .value("RELU", SNEPPX_RELU)
        .value("SOFTMAX", SNEPPX_SOFTMAX)
        .value("LAYERNORM", SNEPPX_LAYERNORM)
        .value("ATTENTION", SNEPPX_ATTENTION)
        .value("BRANCH", SNEPPX_BRANCH)
        .value("HALT", SNEPPX_HALT)
        .value("EXP", SNEPPX_EXP)
        .value("LOG", SNEPPX_LOG)
        .value("SQRT", SNEPPX_SQRT)
        .value("POW", SNEPPX_POW)
        .value("SIN", SNEPPX_SIN)
        .value("COS", SNEPPX_COS)
        .value("TANH", SNEPPX_TANH)
        .value("SIGMOID", SNEPPX_SIGMOID)
        .value("GELU", SNEPPX_GELU)
        .value("SILU", SNEPPX_SILU)
        .value("DROPOUT", SNEPPX_DROPOUT)
        .value("CONV2D", SNEPPX_CONV2D)
        .value("POOL2D", SNEPPX_POOL2D)
        .value("BATCHNORM", SNEPPX_BATCHNORM)
        .value("EMBEDDING", SNEPPX_EMBEDDING)
        .value("CROSSENTROPY", SNEPPX_CROSSENTROPY)
        .value("MSE", SNEPPX_MSE)
        .value("CONCAT", SNEPPX_CONCAT)
        .value("SPLIT", SNEPPX_SPLIT)
        .export_values();

    // ---- Init split module classes ----
    init_tensor(m);
    init_autograd(m);
    init_optim_model(m);
    init_npe_extra(m);
    init_crypto(m);

    // ---- Utility functions ----
    m.def("dtype_size", &SNEPPX_tensor_dtype_size, "Get byte size of a dtype");
    m.def("dtype_name", &SNEPPX_tensor_dtype_name, "Get name of a dtype");
    m.def("dtype_to_numpy", &SNEPPX_dtype_to_numpy, "Map SNEPPXDtype to NumPy dtype string");

    m.def("no_grad_enter", &SNEPPX_no_grad_enter);
    m.def("no_grad_exit", &SNEPPX_no_grad_exit);
    m.def("no_grad_is_active", &SNEPPX_no_grad_is_active);
}

#ifndef SNEPPX_ONNX_FORMAT_H
#define SNEPPX_ONNX_FORMAT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Onnx Format
 *
 * WHAT
 *   Onnx Format.
 *
 * CONCEPT
 *   Provides the Onnx Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Load Onnx.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_onnx_load(const char* path);
/**
 * @brief Destroy Onnx.
 *
 * @param model [out] Model value.
 */
void SNEPPX_onnx_destroy(void* model);
/**
 * @brief Perform Onnx Get Input Count.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_input_count(void* model);
/**
 * @brief Perform Onnx Get Input Info.
 *
 * @param model [out] Model value.
 * @param idx [in] Idx value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param shape [out] Shape value.
 * @param ndim [out] Ndim value.
 * @param dtype [out] Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_input_info(void* model, int idx, char* name, size_t name_max, size_t** shape, size_t* ndim, int* dtype);
/**
 * @brief Perform Onnx Get Output Count.
 *
 * @param model [out] Model value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_get_output_count(void* model);
/**
 * @brief Run Onnx.
 *
 * @param model [out] Model value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 * @param num_outputs [in] Num Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_run(void* model, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs);
/**
 * @brief Perform Onnx Export.
 *
 * @param path [in] Path value.
 * @param graph [out] Graph value.
 * @param input_names [in] Input Names value.
 * @param num_inputs [in] Num Inputs value.
 * @param output_names [in] Output Names value.
 * @param num_outputs [in] Num Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_export(const char* path, void* graph, const char** input_names, size_t num_inputs, const char** output_names, size_t num_outputs);
/**
 * @brief Perform Onnx Check.
 *
 * @param path [in] Path value.
 * @param error_msg [out] Error Msg value.
 * @param error_max [in] Error Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_check(const char* path, char* error_msg, size_t error_max);
/**
 * @brief Export a linear (Gemm) ONNX model as a standard binary .onnx
 *        ModelProto (raw protobuf, loadable by onnxruntime). Emits a single
 *        Gemm node computing Y = X * W^T + B (transB=1).
 *
 * @param path [in] Output .onnx file path (raw protobuf ModelProto).
 * @param model_name [in] Graph/Model name.
 * @param input_name [in] Graph input name (e.g. "X").
 * @param output_name [in] Graph output name (e.g. "Y").
 * @param in_features [in] Input feature width (K).
 * @param out_features [in] Output feature width (N).
 * @param weights [in] Row-major weights [out_features][in_features] (N*K).
 * @param bias [in] Bias vector [out_features] (may be NULL for bias-free).
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_onnx_save_linear(const char* path, const char* model_name,
                            const char* input_name, const char* output_name,
                            size_t in_features, size_t out_features,
                            const float* weights, const float* bias);

/* ---- general-graph ONNX binary export (canonical IR field numbers) ----
 * These let a caller export an arbitrary graph (any op with a registered
 * ONNX op_type, string/int/int-list/float-list attributes) as a canonical
 * binary .onnx ModelProto. SneppXOnnxDim.param, when non-NULL, emits a symbolic
 * (dim_param) dimension such as "batch".*/

typedef struct {
    size_t value;
    const char* param; /* symbolic dim name (e.g. "batch"); non-NULL overrides value */
} SneppXOnnxDim;

typedef struct {
    const char* name;
    const SneppXOnnxDim* dims; /* may be NULL */
    size_t ndim;
    int elem_type;              /* 0 => FLOAT(1) */
} SneppXOnnxValueInfo;

typedef struct {
    const char* name;
    const size_t* dims;         /* NULL for scalar */
    size_t ndim;
    int elem_type;              /* 0 => FLOAT(1) */
    const float* data;          /* product(dims) float32 values */
    size_t nelem;
} SneppXOnnxInitializer;

typedef struct {
    const char* name;
    int type;                   /* 2=INT, 1=FLOAT, 7=INTS, 6=FLOATS */
    long long i;
    float f;
    const long long* ints;
    size_t nints;
    const float* floats;
    size_t nfloats;
} SneppXOnnxAttr;

typedef struct {
    const char* op_type;
    const char* name;
    const char** inputs;
    size_t ninputs;
    const char** outputs;
    size_t noutputs;
    const SneppXOnnxAttr* attrs;
    size_t nattrs;
} SneppXOnnxNode;

/**
 * @brief Export an arbitrary ONNX graph as a standard binary .onnx
 *        ModelProto (raw protobuf, loadable by onnxruntime).
 *
 * @param path [in] Output .onnx file path.
 * @param model_name [in] Graph/Model name.
 * @param producer_name [in] Model producer name.
 * @param producer_version [in] Model producer version.
 * @param ir_version [in] ONNX IR version (e.g. 10).
 * @param model_version [in] Model version.
 * @param opset_version [in] Default-domain (ai.onnx) opset version (e.g. 14).
 * @param inputs [in] Graph inputs.
 * @param ninputs [in] Number of graph inputs.
 * @param outputs [in] Graph outputs.
 * @param noutputs [in] Number of graph outputs.
 * @param initializers [in] Tensor initializers (weights/bias).
 * @param ninit [in] Number of initializers.
 * @param nodes [in] Graph nodes in topological order.
 * @param nnodes [in] Number of nodes.
 *
 * @return 0 on success, -1 on error.
 */
int SneppX_onnx_save_graph(const char* path, const char* model_name,
                           const char* producer_name, const char* producer_version,
                           int ir_version, long long model_version,
                           long long opset_version,
                           const SneppXOnnxValueInfo* inputs, size_t ninputs,
                           const SneppXOnnxValueInfo* outputs, size_t noutputs,
                            const SneppXOnnxInitializer* initializers, size_t ninit,
                            const SneppXOnnxNode* nodes, size_t nnodes);

/* ---- canonical ONNX model validation (standard IR field numbers) ---- */
#define SNEPPX_ONNX_ELEM_FLOAT 1
#define SNEPPX_ONNX_ELEM_FLOAT16 10
#define SNEPPX_ONNX_ELEM_DOUBLE 11

/**
 * @brief Validate a standard binary .onnx ModelProto (raw protobuf, with or
 *        without an "ONNX" magic prefix). Accepts files emitted by
 *        SneppX_onnx_save_linear / SneppX_onnx_save_graph.
 *
 * Checks performed: well-formed protobuf parse, ir_version present, opset
 * import present, graph name present, every node op_type non-empty, every node
 * input is declared (by a graph input or an initializer, or produced by an
 * earlier node), every graph output is produced by some node, initializer
 * data_type is a supported float/16/double, and initializer raw_data length
 * matches the product of its dims.
 *
 * @param path [in] .onnx file path.
 * @param error_msg [out] Buffer for a human-readable error (may be NULL).
 * @param error_max [in] Size of error_msg (ignored if error_msg is NULL).
 *
 * @return 0 if the model is valid, -1 if invalid (and error_msg is filled).
 */
int SneppX_onnx_validate(const char* path, char* error_msg, size_t error_max);
#ifdef __cplusplus
}
#endif
#endif

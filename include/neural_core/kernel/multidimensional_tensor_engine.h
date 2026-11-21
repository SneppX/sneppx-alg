#ifndef SNEPPX_TENSOR_H
#define SNEPPX_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/*
 * SNEPPX - Multidimensional Tensor Engine
 *
 * WHAT
 *   Multidimensional Tensor Engine.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_FLOAT32,
    SNEPPX_FLOAT64,
    SNEPPX_FLOAT16,
    SNEPPX_BFLOAT16,
    SNEPPX_FLOAT8,
    SNEPPX_INT32,
    SNEPPX_INT64,
    SNEPPX_INT16,
    SNEPPX_INT8,
    SNEPPX_UINT8,
    SNEPPX_BOOL,
    SNEPPX_COMPLEX64,
    SNEPPX_COMPLEX128,
} SNEPPXDtype;

#define SNEPPX_DTYPE_SIZE(dtype) SNEPPX_tensor_dtype_size(dtype)
#define SNEPPX_DTYPE_IS_FLOAT(dtype) ((dtype) == SNEPPX_FLOAT32 || (dtype) == SNEPPX_FLOAT64 || (dtype) == SNEPPX_FLOAT16 || (dtype) == SNEPPX_BFLOAT16 || (dtype) == SNEPPX_FLOAT8)
#define SNEPPX_DTYPE_IS_INT(dtype) ((dtype) == SNEPPX_INT32 || (dtype) == SNEPPX_INT64 || (dtype) == SNEPPX_INT16 || (dtype) == SNEPPX_INT8 || (dtype) == SNEPPX_UINT8 || (dtype) == SNEPPX_BOOL)
#define SNEPPX_DTYPE_IS_COMPLEX(dtype) ((dtype) == SNEPPX_COMPLEX64 || (dtype) == SNEPPX_COMPLEX128)

typedef enum {
    SNEPPX_LAYOUT_ROW_MAJOR,
    SNEPPX_LAYOUT_COL_MAJOR,
    SNEPPX_LAYOUT_CHANNELS_LAST,
    SNEPPX_LAYOUT_TILED,
} SNEPPXLayout;

typedef enum {
    SNEPPX_DEVICE_CPU,
    SNEPPX_DEVICE_CUDA,
    SNEPPX_DEVICE_METAL,
    SNEPPX_DEVICE_VULKAN,
    SNEPPX_DEVICE_TPU,
    SNEPPX_DEVICE_NPU,
} SNEPPXDevice;

typedef struct SNEPPXStorage {
    void* data;               /* raw data buffer */
    size_t num_bytes;         /* total bytes allocated */
    int ref_count;            /* how many tensors reference this storage */
} SNEPPXStorage;

/**
 * @brief Create Storage.
 *
 * @param num_bytes [in] Num Bytes value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXStorage* SNEPPX_storage_create(size_t num_bytes);
/**
 * @brief Perform Storage Retain.
 *
 * @param s [out] S value.
 */
void         SNEPPX_storage_retain(SNEPPXStorage* s);
/**
 * @brief Perform Storage Release.
 *
 * @param s [out] S value.
 */
void         SNEPPX_storage_release(SNEPPXStorage* s);

typedef struct SNEPPXTensor {
    SNEPPXStorage* storage;     /* ref-counted storage (NULL for unmanaged) */
    void* data;               /* convenience pointer: storage->data + offset * item_size */
    size_t offset;            /* element offset into storage->data */
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    size_t item_size;
    SNEPPXDtype dtype;
    SNEPPXDevice device;
    int device_id;
    SNEPPXLayout layout;
    int owns_data;
    void* backend_handle;
} SNEPPXTensor;

/**
 * @brief Create Tensor.
 *
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_create(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
/**
 * @brief Destroy Tensor.
 *
 * @param tensor [out] Tensor value.
 */
void SNEPPX_tensor_destroy(SNEPPXTensor* tensor);

/**
 * @brief Perform Tensor Get F32.
 *
 * @param tensor [in] Tensor value.
 * @param indices [in] Indices value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_tensor_get_f32(const SNEPPXTensor* tensor, const size_t* indices);
/**
 * @brief Perform Tensor Set F32.
 *
 * @param tensor [out] Tensor value.
 * @param indices [in] Indices value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_set_f32(SNEPPXTensor* tensor, const size_t* indices, float value);
/**
 * @brief Perform Tensor Get F64.
 *
 * @param tensor [in] Tensor value.
 * @param indices [in] Indices value.
 *
 * @return The result value, or 0 on error.
 */
double SNEPPX_tensor_get_f64(const SNEPPXTensor* tensor, const size_t* indices);
/**
 * @brief Perform Tensor Set F64.
 *
 * @param tensor [out] Tensor value.
 * @param indices [in] Indices value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_set_f64(SNEPPXTensor* tensor, const size_t* indices, double value);
/**
 * @brief Perform Tensor Get I32.
 *
 * @param tensor [in] Tensor value.
 * @param indices [in] Indices value.
 *
 * @return 0 on success, -1 on error.
 */
int32_t SNEPPX_tensor_get_i32(const SNEPPXTensor* tensor, const size_t* indices);
/**
 * @brief Perform Tensor Set I32.
 *
 * @param tensor [out] Tensor value.
 * @param indices [in] Indices value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_set_i32(SNEPPXTensor* tensor, const size_t* indices, int32_t value);
/**
 * @brief Perform Tensor Get I64.
 *
 * @param tensor [in] Tensor value.
 * @param indices [in] Indices value.
 *
 * @return 0 on success, -1 on error.
 */
int64_t SNEPPX_tensor_get_i64(const SNEPPXTensor* tensor, const size_t* indices);
/**
 * @brief Perform Tensor Set I64.
 *
 * @param tensor [out] Tensor value.
 * @param indices [in] Indices value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_set_i64(SNEPPXTensor* tensor, const size_t* indices, int64_t value);
/**
 * @brief Perform Tensor Get Bool.
 *
 * @param tensor [in] Tensor value.
 * @param indices [in] Indices value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_tensor_get_bool(const SNEPPXTensor* tensor, const size_t* indices);
/**
 * @brief Perform Tensor Set Bool.
 *
 * @param tensor [out] Tensor value.
 * @param indices [in] Indices value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_set_bool(SNEPPXTensor* tensor, const size_t* indices, uint8_t value);

/**
 * @brief Perform Tensor Fill F32.
 *
 * @param tensor [out] Tensor value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_fill_f32(SNEPPXTensor* tensor, float value);
/**
 * @brief Perform Tensor Fill F64.
 *
 * @param tensor [out] Tensor value.
 * @param value [in] Value value.
 */
void SNEPPX_tensor_fill_f64(SNEPPXTensor* tensor, double value);

/**
 * @brief Perform Tensor Empty.
 *
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_empty(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Zeros.
 *
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_zeros(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Ones.
 *
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_ones(const size_t* shape, size_t ndim, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Full.
 *
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 * @param value [in] Value value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_full(const size_t* shape, size_t ndim, SNEPPXDtype dtype, const void* value);
/**
 * @brief Perform Tensor Arange.
 *
 * @param start [in] Start value.
 * @param stop [in] Stop value.
 * @param step [in] Step value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_arange(float start, float stop, float step, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Linspace.
 *
 * @param start [in] Start value.
 * @param stop [in] Stop value.
 * @param steps [in] Steps value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_linspace(float start, float stop, size_t steps, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Eye.
 *
 * @param n [in] N value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_eye(size_t n, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Randn.
 *
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_randn(const size_t* shape, size_t ndim, SNEPPXDtype dtype);

/**
 * @brief Perform Tensor Copy.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_copy(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Clone.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_clone(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Slice.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 * @param start [in] Start value.
 * @param end [in] End value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_slice(const SNEPPXTensor* src, size_t dim, size_t start, size_t end);
/**
 * @brief Perform Tensor Reshape.
 *
 * @param src [in] Src value.
 * @param new_shape [in] New Shape value.
 * @param new_ndim [in] New Ndim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_reshape(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim);
/**
 * @brief Perform Tensor Permute.
 *
 * @param src [in] Src value.
 * @param axes [in] Axes value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_permute(const SNEPPXTensor* src, const size_t* axes);
/**
 * @brief Perform Tensor Expand.
 *
 * @param src [in] Src value.
 * @param new_shape [in] New Shape value.
 * @param new_ndim [in] New Ndim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_expand(const SNEPPXTensor* src, const size_t* new_shape, size_t new_ndim);
/**
 * @brief Perform Tensor Squeeze.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_squeeze(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Unsqueeze.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_unsqueeze(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Contiguous.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_contiguous(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor As Strided.
 *
 * @param src [in] Src value.
 * @param offset [in] Offset value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param strides [in] Strides value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_as_strided(const SNEPPXTensor* src, size_t offset, const size_t* shape, size_t ndim, const size_t* strides);
/**
 * @brief Perform Tensor Narrow.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 * @param start [in] Start value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_narrow(const SNEPPXTensor* src, size_t dim, size_t start, size_t size);

/**
 * @brief Perform Tensor Concat.
 *
 * @param tensors [in] Tensors value.
 * @param num_tensors [in] Num Tensors value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_concat(const SNEPPXTensor** tensors, size_t num_tensors, size_t dim);
/**
 * @brief Perform Tensor Split.
 *
 * @param src [in] Src value.
 * @param num_splits [in] Num Splits value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor** SNEPPX_tensor_split(const SNEPPXTensor* src, size_t num_splits, size_t dim);
/**
 * @brief Perform Tensor Tile.
 *
 * @param src [in] Src value.
 * @param reps [in] Reps value.
 * @param reps_ndim [in] Reps Ndim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_tile(const SNEPPXTensor* src, const size_t* reps, size_t reps_ndim);
/**
 * @brief Perform Tensor Repeat.
 *
 * @param src [in] Src value.
 * @param repeats [in] Repeats value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_repeat(const SNEPPXTensor* src, size_t repeats, size_t dim);
/**
 * @brief Perform Tensor Gather.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 * @param indices [in] Indices value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_gather(const SNEPPXTensor* src, size_t dim, const SNEPPXTensor* indices);
/**
 * @brief Perform Tensor Scatter.
 *
 * @param dest [out] Dest value.
 * @param dim [in] Dim value.
 * @param indices [in] Indices value.
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_scatter(SNEPPXTensor* dest, size_t dim, const SNEPPXTensor* indices, const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Masked Select.
 *
 * @param src [in] Src value.
 * @param mask [in] Mask value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_masked_select(const SNEPPXTensor* src, const SNEPPXTensor* mask);
/**
 * @brief Perform Tensor Masked Fill.
 *
 * @param src [out] Src value.
 * @param mask [in] Mask value.
 * @param value [in] Value value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_masked_fill(SNEPPXTensor* src, const SNEPPXTensor* mask, const void* value);
/**
 * @brief Perform Tensor Where.
 *
 * @param condition [in] Condition value.
 * @param x [in] X value.
 * @param y [in] Y value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_where(const SNEPPXTensor* condition, const SNEPPXTensor* x, const SNEPPXTensor* y);

/**
 * @brief Perform Tensor Cast.
 *
 * @param src [in] Src value.
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_cast(const SNEPPXTensor* src, SNEPPXDtype dtype);
/**
 * @brief Perform Tensor To Device.
 *
 * @param src [in] Src value.
 * @param device [in] Device value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_to_device(const SNEPPXTensor* src, SNEPPXDevice device);
/**
 * @brief Perform Tensor To Layout.
 *
 * @param src [in] Src value.
 * @param layout [in] Layout value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_to_layout(const SNEPPXTensor* src, SNEPPXLayout layout);
/**
 * @brief Save Tensor.
 *
 * @param src [in] Src value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_save(const SNEPPXTensor* src, const char* path);
/**
 * @brief Load Tensor.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_load(const char* path);

/**
 * @brief Perform Tensor Eq.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_eq(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Ne.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_ne(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Lt.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_lt(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Le.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_le(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Gt.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_gt(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Ge.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_ge(const SNEPPXTensor* a, const SNEPPXTensor* b);

/**
 * @brief Add Tensor.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_add(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Sub.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sub(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Mul.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_mul(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Div.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_div(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Minimum.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_minimum(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Maximum.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_maximum(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Pow.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_pow(const SNEPPXTensor* a, const SNEPPXTensor* b);

/**
 * @brief Perform Tensor Neg.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_neg(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Abs.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_abs(const SNEPPXTensor* src);
/**
 * @brief Sign Tensor.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sign(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Floor.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_floor(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Ceil.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_ceil(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Round.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_round(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Trunc.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_trunc(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Exp.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_exp(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Log.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_log(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Sqrt.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sqrt(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Sin.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sin(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Cos.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_cos(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Tan.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_tan(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Asin.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_asin(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Acos.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_acos(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Atan.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_atan(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Sinh.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sinh(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Cosh.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_cosh(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Tanh.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_tanh(const SNEPPXTensor* src);

/**
 * @brief Perform Tensor Sum.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sum(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Mean.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_mean(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Std.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_std(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Var.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_var(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Min.
 *
 * @param src [in] Src value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_tensor_min(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Max.
 *
 * @param src [in] Src value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_tensor_max(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Argmin.
 *
 * @param src [in] Src value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_tensor_argmin(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Argmax.
 *
 * @param src [in] Src value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_tensor_argmax(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Cumsum.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_cumsum(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Cumprod.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_cumprod(const SNEPPXTensor* src, size_t dim);

/**
 * @brief Perform Tensor Dot.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_tensor_dot(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Matmul.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b);
/**
 * @brief Perform Tensor Transpose.
 *
 * @param src [in] Src value.
 * @param dim1 [in] Dim1 value.
 * @param dim2 [in] Dim2 value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_transpose(const SNEPPXTensor* src, size_t dim1, size_t dim2);
/**
 * @brief Perform Tensor Inverse.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_inverse(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Det.
 *
 * @param src [in] Src value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_tensor_det(const SNEPPXTensor* src);

/**
 * @brief Perform Tensor Conv1d.
 *
 * @param input [in] Input value.
 * @param kernel [in] Kernel value.
 * @param stride [in] Stride value.
 * @param padding [in] Padding value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_conv1d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride, size_t padding);
/**
 * @brief Perform Tensor Conv2d.
 *
 * @param input [in] Input value.
 * @param kernel [in] Kernel value.
 * @param stride_h [in] Stride H value.
 * @param stride_w [in] Stride W value.
 * @param pad_h [in] Pad H value.
 * @param pad_w [in] Pad W value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_conv2d(const SNEPPXTensor* input, const SNEPPXTensor* kernel, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w);
/**
 * @brief Perform Tensor Pool1d.
 *
 * @param src [in] Src value.
 * @param kernel_size [in] Kernel Size value.
 * @param stride [in] Stride value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_pool1d(const SNEPPXTensor* src, size_t kernel_size, size_t stride);
/**
 * @brief Perform Tensor Pool2d.
 *
 * @param src [in] Src value.
 * @param kernel_h [in] Kernel H value.
 * @param kernel_w [in] Kernel W value.
 * @param stride_h [in] Stride H value.
 * @param stride_w [in] Stride W value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_pool2d(const SNEPPXTensor* src, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w);

/**
 * @brief Perform Tensor Softmax.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_softmax(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Log Softmax.
 *
 * @param src [in] Src value.
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_log_softmax(const SNEPPXTensor* src, size_t dim);
/**
 * @brief Perform Tensor Relu.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_relu(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Gelu.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_gelu(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Silu.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_silu(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Sigmoid.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_sigmoid(const SNEPPXTensor* src);
/**
 * @brief Perform Tensor Fake Quant (symmetric affine, straight-through).
 *
 * Quantizes src to `bits` using `scale` (symmetric, zero point = 0), then
 * de-quantizes back to float. Used for quantization-aware training: the
 * forward is the (non-differentiable-looking) quantize/dequantize, while the
 * backward is the straight-through estimator (identity) handled by the
 * autodiff op SNEPPX_fake_quant.
 *
 * @param src [in] Src value.
 * @param scale [in] Scale value (> 0).
 * @param bits [in] Bit width (2..16), e.g. 8 for INT8.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_fake_quant(const SNEPPXTensor* src, float scale, int bits);
/**
 * @brief Perform Tensor Dropout.
 *
 * @param src [in] Src value.
 * @param rate [in] Rate value.
 * @param seed [in] Seed value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_dropout(const SNEPPXTensor* src, float rate, unsigned int seed);
/**
 * @brief Perform Tensor Layer Norm.
 *
 * @param src [in] Src value.
 * @param gamma [in] Gamma value.
 * @param beta [in] Beta value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_layer_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
/**
 * @brief Perform Tensor Batch Norm.
 *
 * @param src [in] Src value.
 * @param gamma [in] Gamma value.
 * @param beta [in] Beta value.
 * @param running_mean [in] Running Mean value.
 * @param running_var [in] Running Var value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_batch_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, const SNEPPXTensor* running_mean, const SNEPPXTensor* running_var, float eps);
/**
 * @brief Perform Tensor Group Norm.
 *
 * @param src [in] Src value.
 * @param gamma [in] Gamma value.
 * @param beta [in] Beta value.
 * @param num_groups [in] Num Groups value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_group_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, size_t num_groups, float eps);
/**
 * @brief Perform Tensor Instance Norm.
 *
 * @param src [in] Src value.
 * @param gamma [in] Gamma value.
 * @param beta [in] Beta value.
 * @param eps [in] Eps value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_instance_norm(const SNEPPXTensor* src, const SNEPPXTensor* gamma, const SNEPPXTensor* beta, float eps);
/**
 * @brief Perform Tensor Embedding.
 *
 * @param weight [in] Weight value.
 * @param indices [in] Indices value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_embedding(const SNEPPXTensor* weight, const SNEPPXTensor* indices);

/**
 * @brief Perform Tensor Cross Entropy.
 *
 * @param pred [in] Pred value.
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target);
/**
 * @brief Perform Tensor Mse Loss.
 *
 * @param pred [in] Pred value.
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_mse_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
/**
 * @brief Perform Tensor Mae Loss.
 *
 * @param pred [in] Pred value.
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_mae_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
/**
 * @brief Perform Tensor Nll Loss.
 *
 * @param pred [in] Pred value.
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_nll_loss(const SNEPPXTensor* pred, const SNEPPXTensor* target);
/**
 * @brief Perform Tensor Kl Div.
 *
 * @param pred [in] Pred value.
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_kl_div(const SNEPPXTensor* pred, const SNEPPXTensor* target);
/**
 * @brief Perform Tensor Binary Cross Entropy.
 *
 * @param pred [in] Pred value.
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_binary_cross_entropy(const SNEPPXTensor* pred, const SNEPPXTensor* target);

/**
 * @brief Perform Tensor Print.
 *
 * @param tensor [in] Tensor value.
 */
void SNEPPX_tensor_print(const SNEPPXTensor* tensor);
/**
 * @brief Perform Tensor Dtype Size.
 *
 * @param dtype [in] Dtype value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_tensor_dtype_size(SNEPPXDtype dtype);
/**
 * @brief Perform Tensor Numel.
 *
 * @param tensor [in] Tensor value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_tensor_numel(const SNEPPXTensor* tensor);
/**
 * @brief Perform Tensor Is Contiguous.
 *
 * @param tensor [in] Tensor value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_is_contiguous(const SNEPPXTensor* tensor);
/**
 * @brief Perform Tensor Dtype Name.
 *
 * @param dtype [in] Dtype value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_tensor_dtype_name(SNEPPXDtype dtype);


#ifdef __cplusplus
}
#endif
#endif

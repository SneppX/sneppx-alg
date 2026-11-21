#ifndef SNEPPX_TENSOR_INTERNAL_H
#define SNEPPX_TENSOR_INTERNAL_H
/*
 * SNEPPX - Tensor Ops Impl
 *
 * WHAT
 *   Tensor Ops Impl.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Tensor Operations Internal — v0.5
 *
 * PURPOSE: Low-level tensor operation kernels: strided memory copy,
 * broadcasting logic, reduction kernels, and type conversion routines.
 * These are called by the public SNEPPX_tensor_* API but are factored
 * out for reuse by the autodiff and driver layers.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h, concurrent_workload_dispatch.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTensor SNEPPXTensor;

/* ---------- Strided memory ---------- */
/**
 * @brief Perform Tensor Strided Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param dst_strides [in] Dst Strides value.
 * @param src_strides [in] Src Strides value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param elem_size [in] Elem Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_strided_copy(void* dst, const void* src,
                              const size_t* dst_strides, const size_t* src_strides,
                              const size_t* shape, size_t ndim, size_t elem_size);

/**
 * @brief Perform Tensor Broadcast Strides.
 *
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param dst_shape [in] Dst Shape value.
 * @param dst_ndim [in] Dst Ndim value.
 * @param out_strides [out] Out Strides value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_broadcast_strides(const size_t* src_shape, size_t src_ndim,
                                  const size_t* dst_shape, size_t dst_ndim,
                                  size_t* out_strides);

/* ---------- Reduction kernels (v0.5) ---------- */
/**
 * @brief Perform Tensor Reduce Sum F32.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param reduce_dims [in] Reduce Dims value.
 * @param num_dims [in] Num Dims value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_reduce_sum_f32(const float* src, float* dst,
                                const size_t* src_shape, size_t src_ndim,
                                const size_t* reduce_dims, size_t num_dims);

/**
 * @brief Perform Tensor Reduce Mean F32.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param reduce_dims [in] Reduce Dims value.
 * @param num_dims [in] Num Dims value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_reduce_mean_f32(const float* src, float* dst,
                                 const size_t* src_shape, size_t src_ndim,
                                 const size_t* reduce_dims, size_t num_dims);

/**
 * @brief Perform Tensor Reduce Max F32.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param reduce_dims [in] Reduce Dims value.
 * @param num_dims [in] Num Dims value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_reduce_max_f32(const float* src, float* dst,
                                const size_t* src_shape, size_t src_ndim,
                                const size_t* reduce_dims, size_t num_dims);

/* ---------- Type conversion ---------- */
/**
 * @brief Perform Tensor Convert Dtype.
 *
 * @param dst [out] Dst value.
 * @param dst_dtype [in] Dst Dtype value.
 * @param src [in] Src value.
 * @param src_dtype [in] Src Dtype value.
 * @param num_elements [in] Num Elements value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_convert_dtype(void* dst, int dst_dtype,
                               const void* src, int src_dtype,
                               size_t num_elements);

/* ---------- Element-wise operations (v0.5) ---------- */
/**
 * @brief Perform Tensor Add F32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param out [out] Out value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_add_f32(const float* a, const float* b, float* out, size_t n);
/**
 * @brief Perform Tensor Mul F32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param out [out] Out value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_mul_f32(const float* a, const float* b, float* out, size_t n);
/**
 * @brief Perform Tensor Relu F32.
 *
 * @param a [in] A value.
 * @param out [out] Out value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_relu_f32(const float* a, float* out, size_t n);
/**
 * @brief Perform Tensor Sigmoid F32.
 *
 * @param a [in] A value.
 * @param out [out] Out value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_sigmoid_f32(const float* a, float* out, size_t n);

/* ---------- Parallel dispatch helpers ---------- */
typedef void (*SNEPPXTensorKernel1D)(void* ctx, size_t begin, size_t end);
/**
 * @brief Perform Tensor Parallel For.
 *
 * @param kernel [in] Kernel value.
 * @param ctx [out] Ctx value.
 * @param total_work [in] Total Work value.
 * @param min_grain [in] Min Grain value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_parallel_for(SNEPPXTensorKernel1D kernel, void* ctx,
                              size_t total_work, size_t min_grain);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_TENSOR_INTERNAL_H */

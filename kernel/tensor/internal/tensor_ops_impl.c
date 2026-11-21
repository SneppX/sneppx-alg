#include "tensor_ops_impl.h"
#include "concurrent_workload_dispatch.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

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


/**
 * @brief Perform Tensor Strided Copy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param dst_strides [in] Dst Strides value.
 * @param src_strides [in] Src Strides value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_strided_copy(void* dst, const void* src,
                              const size_t* dst_strides, const size_t* src_strides,
                              const size_t* shape, size_t ndim, size_t elem_size) {
    if (!dst || !src || !shape || ndim == 0) return -1;
    if (ndim == 1) { memcpy(dst, src, shape[0] * elem_size); return 0; }
    size_t* src_idx = (size_t*)calloc(ndim, sizeof(size_t));
    size_t* dst_idx = (size_t*)calloc(ndim, sizeof(size_t));
    if (!src_idx || !dst_idx) { free(src_idx); free(dst_idx); return -1; }
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) total *= shape[i];
    for (size_t flat = 0; flat < total; flat++) {
        size_t soff = 0, doff = 0, tmp = flat;
        for (size_t d = ndim; d > 0; d--) {
            size_t idx = tmp % shape[d - 1];
            tmp /= shape[d - 1];
            soff += idx * src_strides[d - 1];
            doff += idx * dst_strides[d - 1];
        }
        memcpy((unsigned char*)dst + doff * elem_size,
               (const unsigned char*)src + soff * elem_size, elem_size);
    }
    free(src_idx); free(dst_idx);
    return 0;
}

/**
 * @brief Perform Tensor Broadcast Strides.
 *
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param dst_shape [in] Dst Shape value.
 * @param dst_ndim [in] Dst Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_broadcast_strides(const size_t* src_shape, size_t src_ndim,
                                  const size_t* dst_shape, size_t dst_ndim,
                                  size_t* out_strides) {
    if (!src_shape || !dst_shape || !out_strides) return -1;
    if (dst_ndim < src_ndim) return -1;
    size_t diff = dst_ndim - src_ndim;
    size_t stride = 1;
    for (size_t d = dst_ndim; d > 0; d--) {
        size_t s_dim = (d - 1 < diff) ? 1 : src_shape[d - 1 - diff];
        size_t d_dim = dst_shape[d - 1];
        if (s_dim != d_dim && s_dim != 1) return -1;
        out_strides[d - 1] = (s_dim == 1) ? 0 : stride;
        stride *= d_dim;
    }
    return 0;
}

static int is_reduced_dim(size_t d, const size_t* reduce_dims, size_t num_dims) {
    for (size_t i = 0; i < num_dims; i++) if (reduce_dims[i] == d) return 1;
    return 0;
}

/**
 * @brief Perform Tensor Reduce Sum F32.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param reduce_dims [in] Reduce Dims value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_reduce_sum_f32(const float* src, float* dst,
                                const size_t* src_shape, size_t src_ndim,
                                const size_t* reduce_dims, size_t num_dims) {
    if (!src || !dst || !src_shape) return -1;
    size_t* src_strides = (size_t*)calloc(src_ndim, sizeof(size_t));
    size_t* dst_shape = (size_t*)calloc(src_ndim, sizeof(size_t));
    if (!src_strides || !dst_shape) { free(src_strides); free(dst_shape); return -1; }
    src_strides[src_ndim - 1] = 1;
    for (size_t i = src_ndim; i > 1; i--) src_strides[i - 2] = src_strides[i - 1] * src_shape[i - 1];
    size_t dst_ndim = 0;
    for (size_t i = 0; i < src_ndim; i++) {
        if (!is_reduced_dim(i, reduce_dims, num_dims)) dst_shape[dst_ndim++] = src_shape[i];
    }
    if (dst_ndim == 0) { dst_ndim = 1; dst_shape[0] = 1; }
    size_t dst_size = 1;
    for (size_t i = 0; i < dst_ndim; i++) dst_size *= dst_shape[i];
    memset(dst, 0, dst_size * sizeof(float));
    size_t* dst_strides = (size_t*)calloc(dst_ndim, sizeof(size_t));
    if (!dst_strides) { free(src_strides); free(dst_shape); return -1; }
    if (dst_ndim > 0) {
        dst_strides[dst_ndim - 1] = 1;
        for (size_t i = dst_ndim; i > 1; i--) dst_strides[i - 2] = dst_strides[i - 1] * dst_shape[i - 1];
    }
    size_t total = 1;
    for (size_t i = 0; i < src_ndim; i++) total *= src_shape[i];
    for (size_t flat = 0; flat < total; flat++) {
        size_t tmp = flat, dst_flat = 0;
        for (size_t d = src_ndim; d > 0; d--) {
            size_t idx = tmp % src_shape[d - 1];
            tmp /= src_shape[d - 1];
            if (!is_reduced_dim(d - 1, reduce_dims, num_dims)) {
                size_t dst_d = 0;
                for (size_t k = 0; k < d - 1; k++) if (!is_reduced_dim(k, reduce_dims, num_dims)) dst_d++;
                dst_flat += idx * dst_strides[dst_d];
            }
        }
        dst[dst_flat] += src[flat];
    }
    free(src_strides); free(dst_shape); free(dst_strides);
    return 0;
}

/**
 * @brief Perform Tensor Reduce Mean F32.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param reduce_dims [in] Reduce Dims value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_reduce_mean_f32(const float* src, float* dst,
                                 const size_t* src_shape, size_t src_ndim,
                                 const size_t* reduce_dims, size_t num_dims) {
    int ret = SNEPPX_tensor_reduce_sum_f32(src, dst, src_shape, src_ndim, reduce_dims, num_dims);
    if (ret != 0) return ret;
    float count = 1.0f;
    for (size_t i = 0; i < num_dims; i++) {
        if (reduce_dims[i] < src_ndim) count *= (float)src_shape[reduce_dims[i]];
    }
    size_t dst_size = 1;
    size_t dst_ndim = 0;
    for (size_t i = 0; i < src_ndim; i++) {
        if (!is_reduced_dim(i, reduce_dims, num_dims)) {
            size_t d = 1;
            for (size_t k = 0; k < i; k++) if (!is_reduced_dim(k, reduce_dims, num_dims)) d++;
            (void)d;
        }
    }
    for (size_t i = 0; i < src_ndim; i++) if (!is_reduced_dim(i, reduce_dims, num_dims)) dst_ndim++;
    if (dst_ndim == 0) { dst_ndim = 1; }
    dst_size = 1;
    for (size_t i = 0; i < src_ndim; i++) if (!is_reduced_dim(i, reduce_dims, num_dims)) dst_size *= src_shape[i];
    if (dst_size == 0) dst_size = 1;
    for (size_t i = 0; i < dst_size; i++) dst[i] /= count;
    return 0;
}

/**
 * @brief Perform Tensor Reduce Max F32.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param src_shape [in] Src Shape value.
 * @param src_ndim [in] Src Ndim value.
 * @param reduce_dims [in] Reduce Dims value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_reduce_max_f32(const float* src, float* dst,
                                const size_t* src_shape, size_t src_ndim,
                                const size_t* reduce_dims, size_t num_dims) {
    if (!src || !dst || !src_shape) return -1;
    size_t* src_strides = (size_t*)calloc(src_ndim, sizeof(size_t));
    size_t* dst_shape = (size_t*)calloc(src_ndim, sizeof(size_t));
    if (!src_strides || !dst_shape) { free(src_strides); free(dst_shape); return -1; }
    src_strides[src_ndim - 1] = 1;
    for (size_t i = src_ndim; i > 1; i--) src_strides[i - 2] = src_strides[i - 1] * src_shape[i - 1];
    size_t dst_ndim = 0;
    for (size_t i = 0; i < src_ndim; i++) if (!is_reduced_dim(i, reduce_dims, num_dims)) dst_shape[dst_ndim++] = src_shape[i];
    if (dst_ndim == 0) { dst_ndim = 1; dst_shape[0] = 1; }
    size_t dst_size = 1;
    for (size_t i = 0; i < dst_ndim; i++) dst_size *= dst_shape[i];
    for (size_t i = 0; i < dst_size; i++) dst[i] = -FLT_MAX;
    size_t* dst_strides = (size_t*)calloc(dst_ndim, sizeof(size_t));
    if (!dst_strides) { free(src_strides); free(dst_shape); return -1; }
    if (dst_ndim > 0) {
        dst_strides[dst_ndim - 1] = 1;
        for (size_t i = dst_ndim; i > 1; i--) dst_strides[i - 2] = dst_strides[i - 1] * dst_shape[i - 1];
    }
    size_t total = 1;
    for (size_t i = 0; i < src_ndim; i++) total *= src_shape[i];
    for (size_t flat = 0; flat < total; flat++) {
        size_t tmp = flat, dst_flat = 0;
        for (size_t d = src_ndim; d > 0; d--) {
            size_t idx = tmp % src_shape[d - 1];
            tmp /= src_shape[d - 1];
            if (!is_reduced_dim(d - 1, reduce_dims, num_dims)) {
                size_t dst_d = 0;
                for (size_t k = 0; k < d - 1; k++) if (!is_reduced_dim(k, reduce_dims, num_dims)) dst_d++;
                dst_flat += idx * dst_strides[dst_d];
            }
        }
        if (src[flat] > dst[dst_flat]) dst[dst_flat] = src[flat];
    }
    free(src_strides); free(dst_shape); free(dst_strides);
    return 0;
}

static int dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 4; case 1: return 1; case 2: return 1;
        case 3: return 2; case 4: return 2; case 5: return 8;
        case 6: return 8; case 7: return 1; default: return 4;
    }
}

/**
 * @brief Perform Tensor Convert Dtype.
 *
 * @param dst [out] Dst value.
 * @param dst_dtype [in] Dst Dtype value.
 * @param src [in] Src value.
 * @param src_dtype [in] Src Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_convert_dtype(void* dst, int dst_dtype, const void* src, int src_dtype, size_t num_elements) {
    if (!dst || !src || num_elements == 0) return 0;
    if (src_dtype == dst_dtype) { memcpy(dst, src, num_elements * (size_t)dtype_size(src_dtype)); return 0; }
    for (size_t i = 0; i < num_elements; i++) {
        float val;
        switch (src_dtype) {
            case 0: val = ((const float*)src)[i]; break;
            case 1: val = ((const signed char*)src)[i]; break;
            case 2: val = ((const unsigned char*)src)[i]; break;
            case 3: val = ((const short*)src)[i]; break;
            case 4: val = ((const unsigned short*)src)[i]; break;
            case 5: val = (float)((const double*)src)[i]; break;
            case 7: val = ((const unsigned char*)src)[i] ? 1.0f : 0.0f; break;
            default: val = ((const float*)src)[i]; break;
        }
        switch (dst_dtype) {
            case 0: ((float*)dst)[i] = val; break;
            case 1: ((signed char*)dst)[i] = (signed char)val; break;
            case 2: ((unsigned char*)dst)[i] = (unsigned char)val; break;
            case 3: ((short*)dst)[i] = (short)val; break;
            case 4: ((unsigned short*)dst)[i] = (unsigned short)val; break;
            case 5: ((double*)dst)[i] = (double)val; break;
            case 7: ((unsigned char*)dst)[i] = val != 0.0f; break;
            default: ((float*)dst)[i] = val; break;
        }
    }
    return 0;
}

/**
 * @brief Perform Tensor Add F32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_add_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out) return -1;
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
    return 0;
}

/**
 * @brief Perform Tensor Mul F32.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_mul_f32(const float* a, const float* b, float* out, size_t n) {
    if (!a || !b || !out) return -1;
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
    return 0;
}

/**
 * @brief Perform Tensor Relu F32.
 *
 * @param a [in] A value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_relu_f32(const float* a, float* out, size_t n) {
    if (!a || !out) return -1;
    for (size_t i = 0; i < n; i++) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    return 0;
}

/**
 * @brief Perform Tensor Sigmoid F32.
 *
 * @param a [in] A value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_sigmoid_f32(const float* a, float* out, size_t n) {
    if (!a || !out) return -1;
    for (size_t i = 0; i < n; i++) {
        float x = a[i];
        if (x < -80.0f) out[i] = 0.0f;
        else if (x > 80.0f) out[i] = 1.0f;
        else out[i] = 1.0f / (1.0f + expf(-x));
    }
    return 0;
}

/**
 * @brief Perform Tensor Parallel For.
 *
 * @param kernel [in] Kernel value.
 * @param ctx [out] Ctx value.
 * @param total_work [in] Total Work value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tensor_parallel_for(SNEPPXTensorKernel1D kernel, void* ctx,
                              size_t total_work, size_t min_grain) {
    if (!kernel || total_work == 0) return 0;
    if (total_work <= min_grain) { kernel(ctx, 0, total_work); return 0; }
    size_t num_chunks = (total_work + min_grain - 1) / min_grain;
    if (num_chunks > 8) num_chunks = 8;
    size_t chunk_size = (total_work + num_chunks - 1) / num_chunks;
    for (size_t c = 0; c < num_chunks; c++) {
        size_t begin = c * chunk_size;
        size_t end = begin + chunk_size;
        if (end > total_work) end = total_work;
        kernel(ctx, begin, end);
    }
    return 0;
}

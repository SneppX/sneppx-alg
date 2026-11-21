#ifndef SNEPPX_NUMPY_FORMAT_H
#define SNEPPX_NUMPY_FORMAT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Numpy Format
 *
 * WHAT
 *   Numpy Format.
 *
 * CONCEPT
 *   Provides the Numpy Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Load Npy.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npy_load(const char* path);
/**
 * @brief Destroy Npy.
 *
 * @param arr [out] Arr value.
 */
void SNEPPX_npy_destroy(void* arr);
/**
 * @brief Perform Npy Get Data.
 *
 * @param arr [out] Arr value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npy_get_data(void* arr);
/**
 * @brief Perform Npy Get Size.
 *
 * @param arr [out] Arr value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_npy_get_size(void* arr);
/**
 * @brief Perform Npy Get Ndim.
 *
 * @param arr [out] Arr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npy_get_ndim(void* arr);
/**
 * @brief Perform Npy Get Shape.
 *
 * @param arr [out] Arr value.
 *
 * @return The computed size/count, or 0 on error.
 */
const size_t* SNEPPX_npy_get_shape(void* arr);
/**
 * @brief Perform Npy Get Dtype.
 *
 * @param arr [out] Arr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npy_get_dtype(void* arr);
/**
 * @brief Save Npy.
 *
 * @param path [in] Path value.
 * @param data [in] Data value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npy_save(const char* path, const void* data, const size_t* shape, size_t ndim, int dtype);
/**
 * @brief Load Npz.
 *
 * @param path [in] Path value.
 * @param keys [out] Keys value.
 * @param arrays [out] Arrays value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npz_load(const char* path, char*** keys, void*** arrays, size_t* count);
/**
 * @brief Save Npz.
 *
 * @param path [in] Path value.
 * @param keys [in] Keys value.
 * @param data_ptrs [in] Data Ptrs value.
 * @param shapes [in] Shapes value.
 * @param ndims [in] Ndims value.
 * @param dtypes [in] Dtypes value.
 * @param count [in] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npz_save(const char* path, const char** keys, const void** data_ptrs, const size_t** shapes, const size_t* ndims, const int* dtypes, size_t count);
/**
 * @brief Free Npz.
 *
 * @param keys [out] Keys value.
 * @param arrays [out] Arrays value.
 * @param count [in] Count value.
 */
void SNEPPX_npz_free(char** keys, void** arrays, size_t count);
#ifdef __cplusplus
}
#endif
#endif

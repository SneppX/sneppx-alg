#ifndef SNEPPX_HDF5_FORMAT_H
#define SNEPPX_HDF5_FORMAT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Hdf5 Format
 *
 * WHAT
 *   Hdf5 Format.
 *
 * CONCEPT
 *   Provides the Hdf5 Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Open Hdf5.
 *
 * @param path [in] Path value.
 * @param mode [in] Mode value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_hdf5_open(const char* path, const char* mode);
/**
 * @brief Close Hdf5.
 *
 * @param file [out] File value.
 */
void SNEPPX_hdf5_close(void* file);
/**
 * @brief Perform Hdf5 Create Group.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_create_group(void* file, const char* path);
/**
 * @brief Perform Hdf5 Dataset Exists.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_dataset_exists(void* file, const char* path);
/**
 * @brief Set Hdf5 Read Data.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 * @param data [out] Data value.
 * @param dims [out] Dims value.
 * @param ndim [in] Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_read_dataset(void* file, const char* path, void* data, size_t* dims, size_t ndim);
/**
 * @brief Set Hdf5 Write Data.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 * @param data [in] Data value.
 * @param dims [in] Dims value.
 * @param ndim [in] Ndim value.
 * @param dtype [in] Dtype value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_write_dataset(void* file, const char* path, const void* data, const size_t* dims, size_t ndim, int dtype);
/**
 * @brief Perform Hdf5 Read Attribute.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 * @param attr_name [in] Attr Name value.
 * @param data [out] Data value.
 * @param size [out] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_read_attribute(void* file, const char* path, const char* attr_name, void* data, size_t* size);
/**
 * @brief Perform Hdf5 Write Attribute.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 * @param attr_name [in] Attr Name value.
 * @param data [in] Data value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_write_attribute(void* file, const char* path, const char* attr_name, const void* data, size_t size);
/**
 * @brief Perform Hdf5 List Children.
 *
 * @param file [out] File value.
 * @param path [in] Path value.
 * @param children [out] Children value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hdf5_list_children(void* file, const char* path, char*** children, size_t* count);
#ifdef __cplusplus
}
#endif
#endif

#ifndef SNEPPX_DRIVER_REGISTRY_H
#define SNEPPX_DRIVER_REGISTRY_H

#include <stddef.h>

#ifdef __cplusplus
/*
 * SNEPPX - Driver Registry
 *
 * WHAT
 *   Driver Registry.
 *
 * CONCEPT
 *   Provides the Driver Registry.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

#define SNEPPX_DRIVER_MAX_NAME 64
#define SNEPPX_DRIVER_MAX_REGISTERED 32

typedef struct {
    void* lib_handle;
    struct {
        int (*init)(void);
        int (*get_device_count)(int*);
        int (*get_device_props)(int, void*, size_t);
        void* (*create_context)(int);
        void (*destroy_context)(void*);
        int (*alloc)(void**, size_t);
        int (*free)(void*);
        int (*htod)(void*, const void*, size_t);
        int (*dtoh)(void*, const void*, size_t);
        int (*launch)(const char*, void*, void**, size_t, size_t, size_t);
        int (*synchronize)(void*);
    } api;
    int loaded;
} SNEPPXDriverEntry;

typedef struct {
    char name[SNEPPX_DRIVER_MAX_NAME];
    int count;
} SNEPPXDriverInfo;

/**
 * @brief Perform Driver Register.
 *
 * @param name [in] Name value.
 * @param entry [out] Entry value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_register(const char* name, SNEPPXDriverEntry* entry);
/**
 * @brief Perform Driver Unregister.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_unregister(const char* name);
/**
 * @brief Get Driver.
 *
 * @param name [in] Name value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXDriverEntry* SNEPPX_driver_get(const char* name);
/**
 * @brief Perform Driver Get Info.
 *
 * @param info [out] Info value.
 * @param max_info [in] Max Info value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_get_info(SNEPPXDriverInfo* info, size_t max_info, size_t* count);
/**
 * @brief Perform Driver Load Library.
 *
 * @param name [in] Name value.
 * @param lib_path [in] Lib Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_load_library(const char* name, const char* lib_path);
/**
 * @brief Perform Driver Unload All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_unload_all(void);

#ifdef __cplusplus
}
#endif

#endif

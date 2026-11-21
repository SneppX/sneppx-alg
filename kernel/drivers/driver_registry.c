#include "driver_registry.h"
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define MAX_ENTRIES SNEPPX_DRIVER_MAX_REGISTERED

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


static SNEPPXDriverEntry g_registry[MAX_ENTRIES];
static char g_names[MAX_ENTRIES][SNEPPX_DRIVER_MAX_NAME];
static int g_count = 0;

/**
 * @brief Perform Driver Register.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_register(const char* name, SNEPPXDriverEntry* entry) {
    if (!name || !entry || g_count >= MAX_ENTRIES) return -1;
    for (int i = 0; i < g_count; i++)
        if (strcmp(g_names[i], name) == 0) return -1;
    strncpy(g_names[g_count], name, SNEPPX_DRIVER_MAX_NAME - 1);
    g_names[g_count][SNEPPX_DRIVER_MAX_NAME - 1] = '\0';
    g_registry[g_count] = *entry;
    g_count++;
    return 0;
}

/**
 * @brief Perform Driver Unregister.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_unregister(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_count; i++) {
        if (strcmp(g_names[i], name) == 0) {
            memset(&g_registry[i], 0, sizeof(SNEPPXDriverEntry));
            memset(g_names[i], 0, SNEPPX_DRIVER_MAX_NAME);
            for (int j = i; j < g_count - 1; j++) {
                g_registry[j] = g_registry[j + 1];
                strncpy(g_names[j], g_names[j + 1], SNEPPX_DRIVER_MAX_NAME - 1);
            }
            g_count--;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Get Driver.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXDriverEntry* SNEPPX_driver_get(const char* name) {
    if (!name) return NULL;
    for (int i = 0; i < g_count; i++)
        if (strcmp(g_names[i], name) == 0)
            return &g_registry[i];
    return NULL;
}

/**
 * @brief Perform Driver Get Info.
 *
 * @param info [out] Info value.
 * @param max_info [in] Max Info value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_get_info(SNEPPXDriverInfo* info, size_t max_info, size_t* count) {
    if (!info || !count) return -1;
    size_t n = g_count < (int)max_info ? (size_t)g_count : max_info;
    for (size_t i = 0; i < n; i++) {
        strncpy(info[i].name, g_names[i], SNEPPX_DRIVER_MAX_NAME - 1);
        info[i].name[SNEPPX_DRIVER_MAX_NAME - 1] = '\0';
        info[i].count = 0;
    }
    *count = n;
    return 0;
}

/**
 * @brief Perform Driver Load Library.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_load_library(const char* name, const char* lib_path) {
    if (!name || !lib_path) return -1;
    if (SNEPPX_driver_get(name)) return -1;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA(lib_path);
    if (!lib) return -1;
    SNEPPXDriverEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.lib_handle = lib;
    entry.api.init = (int (*)(void))GetProcAddress(lib, "sneppx_driver_init");
    entry.api.get_device_count = (int (*)(int*))GetProcAddress(lib, "sneppx_driver_get_device_count");
    entry.api.alloc = (int (*)(void**, size_t))GetProcAddress(lib, "sneppx_driver_alloc");
    entry.api.htod = (int (*)(void*, const void*, size_t))GetProcAddress(lib, "sneppx_driver_htod");
    entry.api.dtoh = (int (*)(void*, const void*, size_t))GetProcAddress(lib, "sneppx_driver_dtoh");
    entry.api.launch = (int (*)(const char*, void*, void**, size_t, size_t, size_t))GetProcAddress(lib, "sneppx_driver_launch");
    entry.api.synchronize = (int (*)(void*))GetProcAddress(lib, "sneppx_driver_synchronize");
    entry.loaded = 1;
#else
    void* lib = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) return -1;
    SNEPPXDriverEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.lib_handle = lib;
    entry.api.init = (int (*)(void))dlsym(lib, "sneppx_driver_init");
    entry.api.get_device_count = (int (*)(int*))dlsym(lib, "sneppx_driver_get_device_count");
    entry.api.alloc = (int (*)(void**, size_t))dlsym(lib, "sneppx_driver_alloc");
    entry.api.htod = (int (*)(void*, const void*, size_t))dlsym(lib, "sneppx_driver_htod");
    entry.api.dtoh = (int (*)(void*, const void*, size_t))dlsym(lib, "sneppx_driver_dtoh");
    entry.api.launch = (int (*)(const char*, void*, void**, size_t, size_t, size_t))dlsym(lib, "sneppx_driver_launch");
    entry.api.synchronize = (int (*)(void*))dlsym(lib, "sneppx_driver_synchronize");
    entry.loaded = 1;
#endif
    if (entry.api.init && entry.api.init() != 0) {
#ifdef _WIN32
        FreeLibrary(lib);
#else
        dlclose(lib);
#endif
        return -1;
    }
    return SNEPPX_driver_register(name, &entry);
}

/**
 * @brief Perform Driver Unload All.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_driver_unload_all(void) {
    for (int i = 0; i < g_count; i++) {
#ifdef _WIN32
        if (g_registry[i].lib_handle) FreeLibrary((HMODULE)g_registry[i].lib_handle);
#else
        if (g_registry[i].lib_handle) dlclose(g_registry[i].lib_handle);
#endif
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


    }
    memset(g_registry, 0, sizeof(g_registry));
    memset(g_names, 0, sizeof(g_names));
    g_count = 0;
    return 0;
}

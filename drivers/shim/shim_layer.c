#include "shim_layer.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef SNEPPX_BUILD_SHIM

/*
 * SNEPPX - Shim Layer
 *
 * WHAT
 *   Shim Layer.
 *
 * CONCEPT
 *   Provides the Shim Layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* External driver symbols — linked from the same library */
/**
 * @brief Perform Amd Register.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_register(void);
/**
 * @brief Perform Amd Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_get_device_count(int*);
/**
 * @brief Perform Amd Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
extern void* SNEPPX_amd_create_context(int);
/**
 * @brief Perform Amd Destroy Context.
 *
 * @return The result value, or 0 on error.
 */
extern void SNEPPX_amd_destroy_context(void*);
/**
 * @brief Perform Amd Mem Alloc.
 *
 * @param size_t [in] Size T value.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_mem_alloc(void**, size_t);
/**
 * @brief Free Amd Mem.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_mem_free(void*);
/**
 * @brief Perform Amd Memcpy Htod.
 *
 * @param size_t [in] Size T value.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_memcpy_htod(void*, const void*, size_t);
/**
 * @brief Perform Amd Memcpy Dtoh.
 *
 * @param size_t [in] Size T value.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_memcpy_dtoh(void*, const void*, size_t);
/**
 * @brief Perform Amd Launch Kernel.
 *
 * @param size_t [in] Size T value.
 * @param size_t [in] Size T value.
 * @param size_t [in] Size T value.
 *
 * @return 0 on success, -1 on error.
 */
extern int SNEPPX_amd_launch_kernel(const char*, void*, void**, size_t, size_t, size_t);

typedef struct {
    char backend_type[32];
    int initialized;
    int device_count;
    int (*backend_register)(void);
    int (*backend_get_device_count)(int*);
    void* (*backend_create_context)(int);
    void (*backend_destroy_context)(void*);
    int (*backend_mem_alloc)(void**, size_t);
    int (*backend_mem_free)(void*);
    int (*backend_memcpy_htod)(void*, const void*, size_t);
    int (*backend_memcpy_dtoh)(void*, const void*, size_t);
    int (*backend_launch_kernel)(const char*, void*, void**, size_t, size_t, size_t);
} ShimState;

static ShimState g_shim;

/**
 * @brief Initialize Shim.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_init(const char* backend_type) {
    if (!backend_type) return SNEPPX_DRIVER_ERROR;
    memset(&g_shim, 0, sizeof(g_shim));
    strncpy(g_shim.backend_type, backend_type, sizeof(g_shim.backend_type) - 1);
    g_shim.backend_type[sizeof(g_shim.backend_type) - 1] = '\0';
    if (sneppx_ref_stricmp(backend_type, "amd") == 0) {
        g_shim.backend_register = SNEPPX_amd_register;
        g_shim.backend_get_device_count = SNEPPX_amd_get_device_count;
        g_shim.backend_create_context = (void* (*)(int))SNEPPX_amd_create_context;
        g_shim.backend_destroy_context = (void (*)(void*))SNEPPX_amd_destroy_context;
        g_shim.backend_mem_alloc = SNEPPX_amd_mem_alloc;
        g_shim.backend_mem_free = SNEPPX_amd_mem_free;
        g_shim.backend_memcpy_htod = SNEPPX_amd_memcpy_htod;
        g_shim.backend_memcpy_dtoh = SNEPPX_amd_memcpy_dtoh;
        g_shim.backend_launch_kernel = SNEPPX_amd_launch_kernel;
    } else {
        return SNEPPX_DRIVER_UNSUPPORTED;
    }
    if (g_shim.backend_register) {
        int ret = g_shim.backend_register();
        if (ret != SNEPPX_DRIVER_OK) return ret;
    }
    g_shim.initialized = 1;
    if (g_shim.backend_get_device_count)
        g_shim.backend_get_device_count(&g_shim.device_count);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Shim Cleanup.
 */
void SNEPPX_shim_cleanup(void) {
    memset(&g_shim, 0, sizeof(g_shim));
}

/**
 * @brief Perform Shim Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_get_device_count(int* count) {
    if (!g_shim.initialized) return SNEPPX_DRIVER_UNSUPPORTED;
    if (count) *count = g_shim.device_count;
    return g_shim.device_count > 0 ? SNEPPX_DRIVER_OK : SNEPPX_DRIVER_UNSUPPORTED;
}

/**
 * @brief Perform Shim Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param dev_type [out] Dev Type value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_get_device_props(int dev_id, char* name, size_t name_max, int* dev_type, size_t* global_mem) {
    (void)dev_id;
    if (!g_shim.initialized) return SNEPPX_DRIVER_UNSUPPORTED;
    if (name) snprintf(name, name_max, "Shim[%s] Device %d", g_shim.backend_type, dev_id);
    if (dev_type) *dev_type = 0;
    if (global_mem) *global_mem = 4ULL * 1024 * 1024 * 1024;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Shim Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_shim_create_context(int device_id) {
    if (!g_shim.initialized || !g_shim.backend_create_context) return NULL;
    return g_shim.backend_create_context(device_id);
}

/**
 * @brief Perform Shim Destroy Context.
 */
void SNEPPX_shim_destroy_context(void* ctx) {
    if (g_shim.backend_destroy_context) g_shim.backend_destroy_context(ctx);
}

/**
 * @brief Perform Shim Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_mem_alloc(void** dev_ptr, size_t bytes, void* ctx) {
    (void)ctx;
    if (!g_shim.initialized || !g_shim.backend_mem_alloc) return SNEPPX_DRIVER_UNSUPPORTED;
    return g_shim.backend_mem_alloc(dev_ptr, bytes);
}

/**
 * @brief Free Shim Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_mem_free(void* dev_ptr, void* ctx) {
    (void)ctx;
    if (!g_shim.initialized || !g_shim.backend_mem_free) return SNEPPX_DRIVER_UNSUPPORTED;
    return g_shim.backend_mem_free(dev_ptr);
}

/**
 * @brief Perform Shim Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx) {
    (void)ctx;
    if (!g_shim.initialized || !g_shim.backend_memcpy_htod) return SNEPPX_DRIVER_UNSUPPORTED;
    return g_shim.backend_memcpy_htod(dev_dst, host_src, bytes);
}

/**
 * @brief Perform Shim Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx) {
    (void)ctx;
    if (!g_shim.initialized || !g_shim.backend_memcpy_dtoh) return SNEPPX_DRIVER_UNSUPPORTED;
    return g_shim.backend_memcpy_dtoh(host_dst, dev_src, bytes);
}

/**
 * @brief Perform Shim Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param ctx [out] Ctx value.
 * @param args [out] Args value.
 * @param num_args [in] Num Args value.
 * @param global_size [in] Global Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_launch_kernel(const char* kernel_name, void* ctx, void** args, size_t num_args, size_t global_size, size_t local_size) {
    (void)ctx;
    if (!g_shim.initialized || !g_shim.backend_launch_kernel) return SNEPPX_DRIVER_UNSUPPORTED;
    return g_shim.backend_launch_kernel(kernel_name, ctx, args, num_args, global_size, local_size);
}

/**
 * @brief Perform Shim Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_synchronize(void* ctx) {
    (void)ctx;
    if (!g_shim.initialized) return SNEPPX_DRIVER_UNSUPPORTED;
    return SNEPPX_DRIVER_OK;
}

#else

/**
 * @brief Initialize Shim.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_init(const char* backend_type) { (void)backend_type; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Shim Cleanup.
 */
void SNEPPX_shim_cleanup(void) {}
/**
 * @brief Perform Shim Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Shim Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param dev_type [out] Dev Type value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_get_device_props(int dev_id, char* name, size_t name_max, int* dev_type, size_t* global_mem) { (void)dev_id; if (name) snprintf(name, name_max, "Shim Device %d", dev_id); if (dev_type) *dev_type = 0; if (global_mem) *global_mem = 4ULL*1024*1024*1024; return SNEPPX_DRIVER_OK; }
/**
 * @brief Perform Shim Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_shim_create_context(int device_id) { (void)device_id; return calloc(1, sizeof(void*)); }
/**
 * @brief Perform Shim Destroy Context.
 */
void SNEPPX_shim_destroy_context(void* ctx) { free(ctx); }
/**
 * @brief Perform Shim Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_mem_alloc(void** dev_ptr, size_t bytes, void* ctx) { (void)dev_ptr; (void)bytes; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Free Shim Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_mem_free(void* dev_ptr, void* ctx) { (void)dev_ptr; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Shim Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx) { (void)dev_dst; (void)host_src; (void)bytes; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Shim Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx) { (void)host_dst; (void)dev_src; (void)bytes; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Shim Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param ctx [out] Ctx value.
 * @param args [out] Args value.
 * @param num_args [in] Num Args value.
 * @param global_size [in] Global Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_launch_kernel(const char* kernel_name, void* ctx, void** args, size_t num_args, size_t global_size, size_t local_size) { (void)kernel_name; (void)ctx; (void)args; (void)num_args; (void)global_size; (void)local_size; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Shim Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_shim_synchronize(void* ctx) { (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }

#endif

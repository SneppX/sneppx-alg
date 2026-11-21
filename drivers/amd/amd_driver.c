#include "amd_driver.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef SNEPPX_BUILD_AMD

/*
 * SNEPPX - Amd Driver
 *
 * WHAT
 *   Amd Driver.
 *
 * CONCEPT
 *   Provides the Amd Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int g_amd_device_count = 1;

typedef struct {
    int device_id;
    float* staging;
    size_t staging_size;
    int initialized;
} AMDContext;

/**
 * @brief Perform Amd Register.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_register(void) {
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Amd Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_get_device_count(int* count) {
    if (count) *count = g_amd_device_count;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Amd Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param global_mem [out] Global Mem value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem, int* compute_units) {
    (void)dev_id;
    if (name) snprintf(name, name_max, "AMD GPU (reference)");
    if (global_mem) *global_mem = 8ULL * 1024 * 1024 * 1024;
    if (compute_units) *compute_units = 40;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Amd Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_amd_create_context(int device_id) {
    AMDContext* ctx = (AMDContext*)calloc(1, sizeof(AMDContext));
    if (!ctx) return NULL;
    ctx->device_id = device_id;
    ctx->staging = NULL;
    ctx->staging_size = 0;
    ctx->initialized = 1;
    return ctx;
}

/**
 * @brief Perform Amd Destroy Context.
 */
void SNEPPX_amd_destroy_context(void* ctx) {
    if (!ctx) return;
    AMDContext* c = (AMDContext*)ctx;
    free(c->staging);
    free(c);
}

/**
 * @brief Perform Amd Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_mem_alloc(void** dev_ptr, size_t bytes) {
    if (!dev_ptr) return SNEPPX_DRIVER_ERROR;
    *dev_ptr = bytes ? malloc(bytes) : malloc(1);
    if (!*dev_ptr) return SNEPPX_DRIVER_ERROR;
    memset(*dev_ptr, 0, bytes ? bytes : 1);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Free Amd Mem.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_mem_free(void* dev_ptr) {
    free(dev_ptr);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Amd Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes) {
    if (!dev_dst || !host_src) return SNEPPX_DRIVER_ERROR;
    memcpy(dev_dst, host_src, bytes);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Amd Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes) {
    if (!host_dst || !dev_src) return SNEPPX_DRIVER_ERROR;
    memcpy(host_dst, dev_src, bytes);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Amd Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param queue [out] Queue value.
 * @param args [out] Args value.
 * @param num_args [in] Num Args value.
 * @param global_size [in] Global Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_launch_kernel(const char* kernel_name, void* queue, void** args, size_t num_args, size_t global_size, size_t local_size) {
    (void)queue; (void)local_size;
    if (!kernel_name || !args || num_args < 3) return SNEPPX_DRIVER_ERROR;
    float* a = (float*)args[0];
    float* b = (float*)args[1];
    float* c = (float*)args[2];
    if (!a || !b || !c) return SNEPPX_DRIVER_ERROR;
    size_t n = (size_t)global_size;
    if (sneppx_ref_stricmp(kernel_name, "gemm") == 0) {
        sneppx_ref_gemm((int)n, (int)n, (int)n, a, b, c);
    } else {
        sneppx_ref_elementwise(kernel_name, c, a, n, 1.0f);
    }
    return SNEPPX_DRIVER_OK;
}

#else

/**
 * @brief Perform Amd Register.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_register(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Amd Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Amd Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param global_mem [out] Global Mem value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_get_device_props(int dev_id, char* name, size_t name_max, size_t* global_mem, int* compute_units) { (void)dev_id; if (name) snprintf(name, name_max, "AMD Device %d", dev_id); if (global_mem) *global_mem = 8ULL*1024*1024*1024; if (compute_units) *compute_units = 40; return SNEPPX_DRIVER_OK; }
/**
 * @brief Perform Amd Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_amd_create_context(int device_id) { (void)device_id; return calloc(1, sizeof(void*)); }
/**
 * @brief Perform Amd Destroy Context.
 */
void SNEPPX_amd_destroy_context(void* ctx) { free(ctx); }
/**
 * @brief Perform Amd Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_mem_alloc(void** dev_ptr, size_t bytes) { (void)dev_ptr; (void)bytes; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Free Amd Mem.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_mem_free(void* dev_ptr) { (void)dev_ptr; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Amd Memcpy Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_memcpy_htod(void* dev_dst, const void* host_src, size_t bytes) { (void)dev_dst; (void)host_src; (void)bytes; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Amd Memcpy Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_memcpy_dtoh(void* host_dst, const void* dev_src, size_t bytes) { (void)host_dst; (void)dev_src; (void)bytes; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Amd Launch Kernel.
 *
 * @param kernel_name [in] Kernel Name value.
 * @param queue [out] Queue value.
 * @param args [out] Args value.
 * @param num_args [in] Num Args value.
 * @param global_size [in] Global Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_amd_launch_kernel(const char* kernel_name, void* queue, void** args, size_t num_args, size_t global_size, size_t local_size) { (void)kernel_name; (void)queue; (void)args; (void)num_args; (void)global_size; (void)local_size; return SNEPPX_DRIVER_UNSUPPORTED; }

#endif

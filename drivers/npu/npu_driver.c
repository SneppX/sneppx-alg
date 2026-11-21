#include "npu_driver.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>

#ifdef SNEPPX_BUILD_NPU

/*
 * SNEPPX - Npu Driver
 *
 * WHAT
 *   Npu Driver.
 *
 * CONCEPT
 *   Provides the Npu Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int g_npu_device_count = 1;

typedef struct {
    int device_id;
    float* local_mem;
    size_t local_mem_size;
} NPUContext;

/**
 * @brief Perform Npu Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_register_driver(void) {
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Npu Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_get_device_count(int* count) {
    if (count) *count = g_npu_device_count;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Npu Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_get_device_props(int dev_id, void* props) {
    (void)dev_id;
    if (props) memset(props, 0, 128);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Npu Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npu_create_context(int device_id) {
    NPUContext* ctx = (NPUContext*)calloc(1, sizeof(NPUContext));
    if (!ctx) return NULL;
    ctx->device_id = device_id;
    ctx->local_mem = NULL;
    ctx->local_mem_size = 0;
    return ctx;
}

/**
 * @brief Perform Npu Destroy Context.
 */
void SNEPPX_npu_destroy_context(void* ctx) {
    if (!ctx) return;
    NPUContext* c = (NPUContext*)ctx;
    free(c->local_mem);
    free(c);
}

/**
 * @brief Perform Npu Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_alloc(void** dev_ptr, size_t bytes, void* ctx) {
    (void)ctx;
    if (!dev_ptr) return SNEPPX_DRIVER_ERROR;
    *dev_ptr = bytes ? malloc(bytes) : malloc(1);
    if (!*dev_ptr) return SNEPPX_DRIVER_ERROR;
    memset(*dev_ptr, 0, bytes ? bytes : 1);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Free Npu Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_free(void* dev_ptr, void* ctx) {
    (void)ctx;
    free(dev_ptr);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Npu Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx) {
    (void)ctx;
    if (!dev_dst || !host_src) return SNEPPX_DRIVER_ERROR;
    memcpy(dev_dst, host_src, bytes);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Npu Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx) {
    (void)ctx;
    if (!host_dst || !dev_src) return SNEPPX_DRIVER_ERROR;
    memcpy(host_dst, dev_src, bytes);
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Npu Execute.
 *
 * @param exec [out] Exec value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 * @param num_outputs [in] Num Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_execute(void* exec, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs, void* ctx) {
    (void)exec; (void)ctx;
    if (!inputs || !outputs || num_inputs == 0 || num_outputs == 0) return SNEPPX_DRIVER_ERROR;
    float* in = (float*)inputs[0];
    float* out = (float*)outputs[0];
    if (!in || !out) return SNEPPX_DRIVER_ERROR;
    size_t n = (size_t)num_inputs > 0 ? 256 : 1;
    sneppx_ref_elementwise("relu", out, in, n, 1.0f);
    return SNEPPX_DRIVER_OK;
}

#else

/**
 * @brief Perform Npu Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_register_driver(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Npu Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Npu Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param props [out] Props value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_get_device_props(int dev_id, void* props) { (void)dev_id; if (props) memset(props, 0, 128); return SNEPPX_DRIVER_OK; }
/**
 * @brief Perform Npu Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npu_create_context(int device_id) { (void)device_id; return calloc(1, 64); }
/**
 * @brief Perform Npu Destroy Context.
 */
void SNEPPX_npu_destroy_context(void* ctx) { free(ctx); }
/**
 * @brief Perform Npu Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_alloc(void** dev_ptr, size_t bytes, void* ctx) { (void)dev_ptr; (void)bytes; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Free Npu Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_free(void* dev_ptr, void* ctx) { (void)dev_ptr; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Npu Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_htod(void* dev_dst, const void* host_src, size_t bytes, void* ctx) { (void)dev_dst; (void)host_src; (void)bytes; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Npu Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes, void* ctx) { (void)host_dst; (void)dev_src; (void)bytes; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Npu Execute.
 *
 * @param exec [out] Exec value.
 * @param inputs [out] Inputs value.
 * @param num_inputs [in] Num Inputs value.
 * @param outputs [out] Outputs value.
 * @param num_outputs [in] Num Outputs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npu_execute(void* exec, void** inputs, size_t num_inputs, void** outputs, size_t num_outputs, void* ctx) { (void)exec; (void)inputs; (void)num_inputs; (void)outputs; (void)num_outputs; (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }

#endif

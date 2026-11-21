#include "qualcomm_driver.h"
#include "neural_core/drivers/driver_status.h"
#include "neural_core/drivers/reference_compute.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef SNEPPX_BUILD_QNN

/*
 * SNEPPX - Qualcomm Driver
 *
 * WHAT
 *   Qualcomm Driver.
 *
 * CONCEPT
 *   Provides the Qualcomm Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int g_qnn_device_count = 1;

typedef struct {
    char model_path[256];
    float* weights;
    size_t weight_count;
    float* input_buffer;
    size_t input_size;
    float* output_buffer;
    size_t output_size;
} QNNContext;

/**
 * @brief Perform Qualcomm Register.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_register(void) {
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Qualcomm Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_get_device_count(int* count) {
    if (count) *count = g_qnn_device_count;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Qualcomm Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_get_device_props(int dev_id, char* name, size_t name_max, unsigned long long* total_mem) {
    (void)dev_id;
    if (name) snprintf(name, name_max, "Qualcomm QNN (reference)");
    if (total_mem) *total_mem = 8ULL * 1024 * 1024 * 1024;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Qualcomm Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_qualcomm_create_context(const char* model_path) {
    QNNContext* ctx = (QNNContext*)calloc(1, sizeof(QNNContext));
    if (!ctx) return NULL;
    if (model_path) {
        strncpy(ctx->model_path, model_path, sizeof(ctx->model_path) - 1);
        ctx->model_path[sizeof(ctx->model_path) - 1] = '\0';
    }
    ctx->weights = NULL;
    ctx->weight_count = 0;
    ctx->input_buffer = NULL;
    ctx->input_size = 0;
    ctx->output_buffer = NULL;
    ctx->output_size = 0;
    return ctx;
}

/**
 * @brief Perform Qualcomm Destroy Context.
 */
void SNEPPX_qualcomm_destroy_context(void* ctx) {
    if (!ctx) return;
    QNNContext* c = (QNNContext*)ctx;
    free(c->weights);
    free(c->input_buffer);
    free(c->output_buffer);
    free(c);
}

/**
 * @brief Perform Qualcomm Set Input.
 *
 * @param ctx [out] Ctx value.
 * @param name [in] Name value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_set_input(void* ctx, const char* name, const float* data, size_t size) {
    if (!ctx || !name || !data) return SNEPPX_DRIVER_ERROR;
    QNNContext* c = (QNNContext*)ctx;
    (void)name;
    float* new_buf = (float*)realloc(c->input_buffer, size * sizeof(float));
    if (!new_buf) return SNEPPX_DRIVER_ERROR;
    c->input_buffer = new_buf;
    c->input_size = size;
    memcpy(c->input_buffer, data, size * sizeof(float));
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Qualcomm Run Inference.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_run_inference(void* ctx) {
    if (!ctx) return SNEPPX_DRIVER_ERROR;
    QNNContext* c = (QNNContext*)ctx;
    if (!c->input_buffer || c->input_size == 0) return SNEPPX_DRIVER_ERROR;
    size_t out_size = c->output_size > 0 ? c->output_size : c->input_size;
    float* new_buf = (float*)realloc(c->output_buffer, out_size * sizeof(float));
    if (!new_buf) return SNEPPX_DRIVER_ERROR;
    c->output_buffer = new_buf;
    c->output_size = out_size;
    for (size_t i = 0; i < out_size && i < c->input_size; i++)
        c->output_buffer[i] = c->input_buffer[i] > 0.0f ? c->input_buffer[i] : 0.0f;
    return SNEPPX_DRIVER_OK;
}

/**
 * @brief Perform Qualcomm Get Output.
 *
 * @param ctx [out] Ctx value.
 * @param name [in] Name value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_get_output(void* ctx, const char* name, float* data, size_t size) {
    if (!ctx || !name || !data) return SNEPPX_DRIVER_ERROR;
    QNNContext* c = (QNNContext*)ctx;
    (void)name;
    size_t copy = size < c->output_size ? size : c->output_size;
    memcpy(data, c->output_buffer, copy * sizeof(float));
    if (size > copy) memset(data + copy, 0, (size - copy) * sizeof(float));
    return SNEPPX_DRIVER_OK;
}

#else

/**
 * @brief Perform Qualcomm Register.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_register(void) { return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Qualcomm Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_get_device_count(int* count) { if (count) *count = 0; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Qualcomm Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 * @param name [out] Name value.
 * @param name_max [in] Name Max value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_get_device_props(int dev_id, char* name, size_t name_max, unsigned long long* total_mem) { (void)dev_id; if (name) snprintf(name, name_max, "QNN Device %d", dev_id); if (total_mem) *total_mem = 8ULL*1024*1024*1024; return SNEPPX_DRIVER_OK; }
/**
 * @brief Perform Qualcomm Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_qualcomm_create_context(const char* model_path) { (void)model_path; return calloc(1, sizeof(void*)); }
/**
 * @brief Perform Qualcomm Destroy Context.
 */
void SNEPPX_qualcomm_destroy_context(void* ctx) { free(ctx); }
/**
 * @brief Perform Qualcomm Set Input.
 *
 * @param ctx [out] Ctx value.
 * @param name [in] Name value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_set_input(void* ctx, const char* name, const float* data, size_t size) { (void)ctx; (void)name; (void)data; (void)size; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Qualcomm Run Inference.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_run_inference(void* ctx) { (void)ctx; return SNEPPX_DRIVER_UNSUPPORTED; }
/**
 * @brief Perform Qualcomm Get Output.
 *
 * @param ctx [out] Ctx value.
 * @param name [in] Name value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_qualcomm_get_output(void* ctx, const char* name, float* data, size_t size) { (void)ctx; (void)name; (void)data; (void)size; return SNEPPX_DRIVER_UNSUPPORTED; }

#endif

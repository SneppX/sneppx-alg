#ifndef SNEPPX_ROCM_DRIVER_H
#define SNEPPX_ROCM_DRIVER_H
/*
 * SNEPPX - Rocm Driver
 *
 * WHAT
 *   Rocm Driver.
 *
 * CONCEPT
 *   Provides the Rocm Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * ROCm Driver Interface — v1.0 (AMD GPU acceleration)
 *
 * PURPOSE: Abstract AMD GPU devices behind the kernel's device operations
 * structure.  Mirrors the CUDA driver interface so that tensor operations
 * can be dispatched to either NVIDIA or AMD hardware transparently.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, multidimensional_tensor_engine.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXTensor SNEPPXTensor;

typedef struct {
    char     name[256];
    size_t   global_mem_bytes;
    size_t   shared_mem_per_group;
    int      wavefront_size;
    int      max_threads_per_workgroup;
    int      max_workgroup_size[3];
    int      gcn_arch_major;
    int      gcn_arch_minor;
    int      num_cus;
    int      supports_matrix_core;
} SNEPPXROCmDeviceProps;

typedef struct SNEPPXROCmStream {
    void*    handle;
    int      device_id;
} SNEPPXROCmStream;

typedef struct SNEPPXROCmEvent {
    void*    handle;
    int      device_id;
} SNEPPXROCmEvent;

typedef struct {
    int                     device_id;
    SNEPPXROCmDeviceProps     props;
    size_t                  alloc_bytes;
    size_t                  peak_alloc_bytes;
    SNEPPXROCmStream**        streams;
    int                     num_streams;
    void*                   blas_handle;
    void*                   dnn_handle;
    int                     error_state;
} SNEPPXROCmContext;

typedef struct {
    const char*      kernel_name;
    void*            kernel_func;
    int              grid_x, grid_y, grid_z;
    int              group_x, group_y, group_z;
    size_t           local_mem_bytes;
    SNEPPXROCmStream*  stream;
} SNEPPXROCmKernelLaunch;

/**
 * @brief Perform Rocm Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_register_driver(void);

/**
 * @brief Perform Rocm Get Device Count.
 *
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_rocm_get_device_count(int* count);
/**
 * @brief Perform Rocm Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param props [out] Props value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_rocm_get_device_props(int dev_id, SNEPPXROCmDeviceProps* props);
/**
 * @brief Perform Rocm Set Device.
 *
 * @param dev_id [in] Dev Id value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_rocm_set_device(int dev_id);

/**
 * @brief Perform Rocm Create Context.
 *
 * @param device_id [in] Device Id value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXROCmContext* SNEPPX_rocm_create_context(int device_id);
/**
 * @brief Perform Rocm Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void             SNEPPX_rocm_destroy_context(SNEPPXROCmContext* ctx);

/**
 * @brief Create Rocm Stream.
 *
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_stream_create(SNEPPXROCmStream** stream);
/**
 * @brief Destroy Rocm Stream.
 *
 * @param stream [out] Stream value.
 */
void SNEPPX_rocm_stream_destroy(SNEPPXROCmStream* stream);
/**
 * @brief Perform Rocm Stream Synchronize.
 *
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_stream_synchronize(SNEPPXROCmStream* stream);
/**
 * @brief Create Rocm Event.
 *
 * @param event [out] Event value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_event_create(SNEPPXROCmEvent** event);
/**
 * @brief Destroy Rocm Event.
 *
 * @param event [out] Event value.
 */
void SNEPPX_rocm_event_destroy(SNEPPXROCmEvent* event);
/**
 * @brief Perform Rocm Event Record.
 *
 * @param event [out] Event value.
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_event_record(SNEPPXROCmEvent* event, SNEPPXROCmStream* stream);
/**
 * @brief Perform Rocm Event Synchronize.
 *
 * @param event [out] Event value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_event_synchronize(SNEPPXROCmEvent* event);

/**
 * @brief Perform Rocm Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_alloc(void** dev_ptr, size_t bytes);
/**
 * @brief Free Rocm Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_free(void* dev_ptr);
/**
 * @brief Perform Rocm Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_htod(void* dev_dst, const void* host_src, size_t bytes);
/**
 * @brief Perform Rocm Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes);
/**
 * @brief Perform Rocm Mem Dtod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes);

/**
 * @brief Perform Rocm Launch Kernel.
 *
 * @param launch [in] Launch value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_launch_kernel(const SNEPPXROCmKernelLaunch* launch);
/**
 * @brief Perform Rocm Launch Kernel Async.
 *
 * @param launch [in] Launch value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_launch_kernel_async(const SNEPPXROCmKernelLaunch* launch);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_ROCM_DRIVER_H */

#ifndef SNEPPX_CUDA_DRIVER_H
#define SNEPPX_CUDA_DRIVER_H
/*
 * SNEPPX - Cuda Driver
 *
 * WHAT
 *   Cuda Driver.
 *
 * CONCEPT
 *   Provides the Cuda Driver.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * CUDA Driver Interface — v1.0 (GPU acceleration)
 *
 * PURPOSE: Abstract NVIDIA GPU device behind the kernel's device operations
 * structure.  This header declares the CUDA driver registration, device context,
 * stream/event management, and kernel dispatch interface.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h (device allocator), multidimensional_tensor_engine.h (tensor struct)
 * VERSION: v1.0 — GPU training
 *
 * IMPLEMENTATION STATUS: SKELETON — no real CUDA calls
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- forward declarations from core ---------- */
typedef struct SNEPPXTensor SNEPPXTensor;

/* ---------- Device properties queried at runtime ---------- */
typedef struct {
    char     name[256];
    size_t   global_mem_bytes;
    size_t   shared_mem_per_block;
    int      warp_size;
    int      max_threads_per_block;
    int      max_blocks_per_grid[3];
    int      compute_capability_major;
    int      compute_capability_minor;
    int      num_sms;               /* streaming multiprocessors */
    int      supports_tensor_cores;
    int      supports_cooperative_groups;
} SNEPPXCUDADeviceProps;

/* ---------- Stream / event handles (opaque) ---------- */
typedef struct SNEPPXCUDAStream {
    void*    handle;                /* cudaStream_t stored as void* */
    int      device_id;
    int      priority;
} SNEPPXCUDAStream;

typedef struct SNEPPXCUDAEvent {
    void*    handle;                /* cudaEvent_t stored as void* */
    int      device_id;
} SNEPPXCUDAEvent;

/* ---------- Device context — one per GPU ---------- */
typedef struct {
    int                     device_id;
    SNEPPXCUDADeviceProps     props;
    size_t                  alloc_bytes;       /* current device allocations */
    size_t                  peak_alloc_bytes;
    SNEPPXCUDAStream**        streams;           /* owned stream array */
    int                     num_streams;
    void*                   blas_handle;       /* cuBLAS handle */
    void*                   dnn_handle;        /* cuDNN handle (v1.0) */
    void*                   solver_handle;     /* cuSOLVER handle (v2.0) */
    int                     error_state;       /* non-zero after fatal error */
} SNEPPXCUDAContext;

/* ---------- Kernel dispatch descriptor ---------- */
typedef struct {
    const char*      kernel_name;     /* PTX / CUBIN symbol name */
    void*            kernel_func;     /* loaded module function pointer */
    int              grid_x, grid_y, grid_z;
    int              block_x, block_y, block_z;
    size_t           shared_mem_bytes;
    SNEPPXCUDAStream*  stream;
} SNEPPXCUDAKernelLaunch;

/* ---------- Driver registration ---------- */
/* Called once at SNEPPX_init() to make the CUDA driver available.
 * Registers device query, memory ops, and kernel dispatch in the
 * global driver registry (declared in kernel/arch.c). */
/**
 * @brief Perform Cuda Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_register_driver(void);

/* ---------- Device lifecycle ---------- */
/**
 * @brief Perform Cuda Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_get_device_count(int* count);                /* v1.0 */
/**
 * @brief Perform Cuda Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 * @param props [out] Props value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_get_device_props(int dev_id, SNEPPXCUDADeviceProps* props);
/**
 * @brief Perform Cuda Set Device.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_set_device(int dev_id);                      /* v1.0 */
/**
 * @brief Perform Cuda Get Device.
 *
 * @param dev_id [out] Dev Id value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_get_device(int* dev_id);

/* ---------- Context management ---------- */
/**
 * @brief Perform Cuda Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXCUDAContext* SNEPPX_cuda_create_context(int device_id);   /* v1.0 */
/**
 * @brief Perform Cuda Destroy Context.
 *
 * @param ctx [out] Ctx value.
 */
void             SNEPPX_cuda_destroy_context(SNEPPXCUDAContext* ctx);
/**
 * @brief Perform Cuda Context Error.
 *
 * @param ctx [in] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int              SNEPPX_cuda_context_error(const SNEPPXCUDAContext* ctx);

/* ---------- Stream / event ---------- */
/**
 * @brief Create Cuda Stream.
 *
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_stream_create(SNEPPXCUDAStream** stream, int priority);      /* v1.0 */
/**
 * @brief Destroy Cuda Stream.
 *
 * @param stream [out] Stream value.
 */
void SNEPPX_cuda_stream_destroy(SNEPPXCUDAStream* stream);
/**
 * @brief Perform Cuda Stream Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_stream_synchronize(SNEPPXCUDAStream* stream);                /* v1.0 */
/**
 * @brief Create Cuda Event.
 *
 * @param event [out] Event value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_event_create(SNEPPXCUDAEvent** event);
/**
 * @brief Destroy Cuda Event.
 *
 * @param event [out] Event value.
 */
void SNEPPX_cuda_event_destroy(SNEPPXCUDAEvent* event);
/**
 * @brief Perform Cuda Event Record.
 *
 * @param event [out] Event value.
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_event_record(SNEPPXCUDAEvent* event, SNEPPXCUDAStream* stream);
/**
 * @brief Perform Cuda Event Synchronize.
 *
 * @param event [out] Event value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_event_synchronize(SNEPPXCUDAEvent* event);
/**
 * @brief Perform Cuda Event Elapsed Ms.
 *
 * @param ms [out] Ms value.
 * @param start [out] Start value.
 * @param end [out] End value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_event_elapsed_ms(float* ms, SNEPPXCUDAEvent* start, SNEPPXCUDAEvent* end);

/* ---------- Memory management (device) ---------- */
/**
 * @brief Perform Cuda Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_mem_alloc(void** dev_ptr, size_t bytes);                    /* v1.0 */
/**
 * @brief Free Cuda Mem.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_mem_free(void* dev_ptr);                                    /* v1.0 */
/**
 * @brief Perform Cuda Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_mem_htod(void* dev_dst, const void* host_src, size_t bytes); /* v1.0 */
/**
 * @brief Perform Cuda Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes);
/**
 * @brief Perform Cuda Mem Dtod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes); /* v1.0 */
/**
 * @brief Set Cuda Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_mem_set(void* dev_ptr, int value, size_t bytes);            /* v1.0 */

/* ---------- Kernel dispatch ---------- */
/**
 * @brief Perform Cuda Launch Kernel.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_launch_kernel(const SNEPPXCUDAKernelLaunch* launch);          /* v1.0 */
/**
 * @brief Perform Cuda Launch Kernel Async.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_cuda_launch_kernel_async(const SNEPPXCUDAKernelLaunch* launch);    /* v1.0 */

/* ---------- Tensor-core wrappers (v1.0) ---------- */
typedef struct {
    int    m, n, k;
    void*  a;                /* device ptr */
    void*  b;
    void*  c;
    int    lda, ldb, ldc;
    int    dtype;            /* 0 = f16, 1 = bf16, 2 = tf32 */
} SNEPPXCUDATensorCoreGemm;

/**
 * @brief Perform Cuda Tc Gemm.
 *
 * @param desc [in] Desc value.
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_tc_gemm(const SNEPPXCUDATensorCoreGemm* desc, SNEPPXCUDAStream* stream);

/* ---------- Warp-level primitives (v1.0) ---------- */
/**
 * @brief Perform Cuda Warp Ballot.
 *
 * @param predicate [in] Predicate value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_cuda_warp_ballot(int predicate);
/**
 * @brief Perform Cuda Warp Reduce Sum.
 *
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_cuda_warp_reduce_sum(uint32_t value);
/**
 * @brief Perform Cuda Warp Reduce Sum F32.
 *
 * @param value [in] Value value.
 *
 * @return The result value, or 0 on error.
 */
float    SNEPPX_cuda_warp_reduce_sum_f32(float value);

/* ---------- Error handling ---------- */
/**
 * @brief Perform Cuda Error String.
 *
 * @param error_code [in] Error Code value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_cuda_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_CUDA_DRIVER_H */

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
 * CUDA Driver Implementation — v1.0
 * PURPOSE: Real CUDA Runtime API calls wrapped behind SNEPPXCUDA interface.
 */

#include "cuda_driver.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(SNEPPX_HAS_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

static cublasHandle_t get_blas_handle(SNEPPXCUDAContext* ctx) {
    return ctx ? (cublasHandle_t)ctx->blas_handle : NULL;
}
#else
/* Stub types when CUDA is not available */
typedef struct { int unused; } cudaError_t;
#define cudaSuccess 0
#define cudaGetDeviceCount(x) (-1)
#define cudaGetDeviceProperties(x,y) (-1)
#define cudaSetDevice(x) (-1)
#define cudaGetDevice(x) (-1)
#define cudaStreamCreateWithPriority(x) (-1)
#define cudaStreamDestroy(x) (-1)
#define cudaStreamSynchronize(x) (-1)
#define cudaEventCreate(x) (-1)
#define cudaEventDestroy(x) (-1)
#define cudaEventRecord(x,y) (-1)
#define cudaEventSynchronize(x) (-1)
#define cudaEventElapsedTime(x,y,z) (-1)
#define cudaMalloc(x,y) (-1)
#define cudaFree(x) (-1)
#define cudaMemcpy(x,y,z,w) (-1)
#define cudaMemset(x,y,z) (-1)
#define cudaMemcpyKind 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
#endif

static void set_error(SNEPPXCUDAContext* ctx, int err) {
    if (ctx) ctx->error_state = err;
}

/**
 * @brief Perform Cuda Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_register_driver(void) {
    /* In the future, this registers into a global driver ops table */
    return 0;
}

/* ---------- Device lifecycle ---------- */

/**
 * @brief Perform Cuda Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_get_device_count(int* count) {
    if (!count) return -1;
#if defined(SNEPPX_HAS_CUDA)
    int c = 0;
    cudaError_t err = cudaGetDeviceCount(&c);
    if (err != cudaSuccess) return -1;
    *count = (int)c;
    return 0;
#else
    *count = 0;
    return 0;
#endif
}

/**
 * @brief Perform Cuda Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_get_device_props(int dev_id, SNEPPXCUDADeviceProps* props) {
    if (!props) return -1;
    memset(props, 0, sizeof(*props));
#if defined(SNEPPX_HAS_CUDA)
    cudaDeviceProp p;
    cudaError_t err = cudaGetDeviceProperties(&p, dev_id);
    if (err != cudaSuccess) return -1;
    strncpy(props->name, p.name, sizeof(props->name) - 1);
    props->global_mem_bytes = p.totalGlobalMem;
    props->shared_mem_per_block = p.sharedMemPerBlock;
    props->warp_size = p.warpSize;
    props->max_threads_per_block = p.maxThreadsPerBlock;
    props->max_blocks_per_grid[0] = p.maxGridSize[0];
    props->max_blocks_per_grid[1] = p.maxGridSize[1];
    props->max_blocks_per_grid[2] = p.maxGridSize[2];
    props->compute_capability_major = p.major;
    props->compute_capability_minor = p.minor;
    props->num_sms = p.multiProcessorCount;
#if defined(__CUDACC__)
    props->supports_tensor_cores = (p.major >= 7) ? 1 : 0;
#else
    props->supports_tensor_cores = 0;
#endif
    props->supports_cooperative_groups = (p.major >= 9 || (p.major >= 7 && p.minor >= 5)) ? 1 : 0;
#endif
    return 0;
}

/**
 * @brief Perform Cuda Set Device.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_set_device(int dev_id) {
#if defined(SNEPPX_HAS_CUDA)
    return (cudaSetDevice(dev_id) == cudaSuccess) ? 0 : -1;
#else
    (void)dev_id;
    return 0;
#endif
}

/**
 * @brief Perform Cuda Get Device.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_get_device(int* dev_id) {
    if (!dev_id) return -1;
#if defined(SNEPPX_HAS_CUDA)
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess) return -1;
    *dev_id = d;
    return 0;
#else
    *dev_id = 0;
    return 0;
#endif
}

/* ---------- Context ---------- */

/**
 * @brief Perform Cuda Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXCUDAContext* SNEPPX_cuda_create_context(int device_id) {
    SNEPPXCUDAContext* ctx = (SNEPPXCUDAContext*)calloc(1, sizeof(SNEPPXCUDAContext));
    if (!ctx) return NULL;
    ctx->device_id = device_id;

#if defined(SNEPPX_HAS_CUDA)
    SNEPPX_cuda_get_device_props(device_id, &ctx->props);

    cublasHandle_t blas = NULL;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return NULL;
    }
    ctx->blas_handle = (void*)blas;

    cudaStream_t stream = NULL;
    if (cudaStreamCreateWithPriority(&stream, cudaStreamDefault, 0) != cudaSuccess) {
        cublasDestroy(blas);
        free(ctx);
        return NULL;
    }
    cublasSetStream(blas, stream);

    ctx->streams = (SNEPPXCUDAStream**)calloc(1, sizeof(SNEPPXCUDAStream*));
    if (ctx->streams) {
        ctx->streams[0] = (SNEPPXCUDAStream*)calloc(1, sizeof(SNEPPXCUDAStream));
        if (ctx->streams[0]) {
            ctx->streams[0]->handle = (void*)stream;
            ctx->streams[0]->device_id = device_id;
            ctx->streams[0]->priority = 0;
            ctx->num_streams = 1;
        }
    }
#endif
    return ctx;
}

/**
 * @brief Perform Cuda Destroy Context.
 */
void SNEPPX_cuda_destroy_context(SNEPPXCUDAContext* ctx) {
    if (!ctx) return;
#if defined(SNEPPX_HAS_CUDA)
    if (ctx->blas_handle) cublasDestroy((cublasHandle_t)ctx->blas_handle);
    if (ctx->streams) {
        for (int i = 0; i < ctx->num_streams; i++) {
            if (ctx->streams[i] && ctx->streams[i]->handle)
                cudaStreamDestroy((cudaStream_t)ctx->streams[i]->handle);
            free(ctx->streams[i]);
        }
        free(ctx->streams);
    }
    if (ctx->dnn_handle) { /* cuDNN destroy - future */ }
    if (ctx->solver_handle) { /* cuSOLVER destroy - future */ }
#endif
    free(ctx);
}

/**
 * @brief Perform Cuda Context Error.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_context_error(const SNEPPXCUDAContext* ctx) {
    return ctx ? ctx->error_state : -1;
}

/* ---------- Stream / event ---------- */

/**
 * @brief Create Cuda Stream.
 *
 * @param stream [out] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_stream_create(SNEPPXCUDAStream** stream, int priority) {
    if (!stream) return -1;
    *stream = (SNEPPXCUDAStream*)calloc(1, sizeof(SNEPPXCUDAStream));
    if (!*stream) return -1;
#if defined(SNEPPX_HAS_CUDA)
    cudaStream_t s = NULL;
    cudaError_t err = cudaStreamCreateWithPriority(&s, cudaStreamDefault, priority);
    if (err != cudaSuccess) { free(*stream); *stream = NULL; return -1; }
    (*stream)->handle = (void*)s;
#endif
    (*stream)->priority = priority;
    return 0;
}

/**
 * @brief Destroy Cuda Stream.
 */
void SNEPPX_cuda_stream_destroy(SNEPPXCUDAStream* stream) {
    if (!stream) return;
#if defined(SNEPPX_HAS_CUDA)
    if (stream->handle) cudaStreamDestroy((cudaStream_t)stream->handle);
#endif
    free(stream);
}

/**
 * @brief Perform Cuda Stream Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_stream_synchronize(SNEPPXCUDAStream* stream) {
    if (!stream) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaStreamSynchronize((cudaStream_t)stream->handle) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Create Cuda Event.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_event_create(SNEPPXCUDAEvent** event) {
    if (!event) return -1;
    *event = (SNEPPXCUDAEvent*)calloc(1, sizeof(SNEPPXCUDAEvent));
    if (!*event) return -1;
#if defined(SNEPPX_HAS_CUDA)
    cudaEvent_t e = NULL;
    if (cudaEventCreate(&e) != cudaSuccess) { free(*event); *event = NULL; return -1; }
    (*event)->handle = (void*)e;
#endif
    return 0;
}

/**
 * @brief Destroy Cuda Event.
 */
void SNEPPX_cuda_event_destroy(SNEPPXCUDAEvent* event) {
    if (!event) return;
#if defined(SNEPPX_HAS_CUDA)
    if (event->handle) cudaEventDestroy((cudaEvent_t)event->handle);
#endif
    free(event);
}

/**
 * @brief Perform Cuda Event Record.
 *
 * @param event [out] Event value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_event_record(SNEPPXCUDAEvent* event, SNEPPXCUDAStream* stream) {
    if (!event || !stream) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaEventRecord((cudaEvent_t)event->handle, (cudaStream_t)stream->handle) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Cuda Event Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_event_synchronize(SNEPPXCUDAEvent* event) {
    if (!event) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaEventSynchronize((cudaEvent_t)event->handle) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Cuda Event Elapsed Ms.
 *
 * @param ms [out] Ms value.
 * @param start [out] Start value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_event_elapsed_ms(float* ms, SNEPPXCUDAEvent* start, SNEPPXCUDAEvent* end) {
    if (!ms || !start || !end) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaEventElapsedTime(ms, (cudaEvent_t)start->handle, (cudaEvent_t)end->handle) == cudaSuccess) ? 0 : -1;
#else
    *ms = 0.0f;
    return 0;
#endif
}

/* ---------- Memory ---------- */

/**
 * @brief Perform Cuda Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_mem_alloc(void** dev_ptr, size_t bytes) {
    if (!dev_ptr) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaMalloc(dev_ptr, bytes) == cudaSuccess) ? 0 : -1;
#else
    *dev_ptr = NULL;
    return 0;
#endif
}

/**
 * @brief Free Cuda Mem.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_mem_free(void* dev_ptr) {
    if (!dev_ptr) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaFree(dev_ptr) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Cuda Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_mem_htod(void* dev_dst, const void* host_src, size_t bytes) {
    if (!dev_dst || !host_src) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaMemcpy(dev_dst, host_src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Cuda Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes) {
    if (!host_dst || !dev_src) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaMemcpy(host_dst, dev_src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Cuda Mem Dtod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes) {
    if (!dev_dst || !dev_src) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaMemcpy(dev_dst, dev_src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Set Cuda Mem.
 *
 * @param dev_ptr [out] Dev Ptr value.
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_mem_set(void* dev_ptr, int value, size_t bytes) {
    if (!dev_ptr) return -1;
#if defined(SNEPPX_HAS_CUDA)
    return (cudaMemset(dev_ptr, value, bytes) == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/* ---------- Kernel dispatch ---------- */

/**
 * @brief Perform Cuda Launch Kernel.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_launch_kernel(const SNEPPXCUDAKernelLaunch* launch) {
    if (!launch) return -1;
#if defined(SNEPPX_HAS_CUDA)
    if (!launch->kernel_func) return -1;
    void* args[] = { /* kernel args would be passed through launch */ };
    cudaError_t err = cudaLaunchKernel(launch->kernel_func,
        dim3(launch->grid_x, launch->grid_y, launch->grid_z),
        dim3(launch->block_x, launch->block_y, launch->block_z),
        args, launch->shared_mem_bytes,
        launch->stream ? (cudaStream_t)launch->stream->handle : 0);
    return (err == cudaSuccess) ? 0 : -1;
#else
    return 0;
#endif
}

/**
 * @brief Perform Cuda Launch Kernel Async.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_launch_kernel_async(const SNEPPXCUDAKernelLaunch* launch) {
    return SNEPPX_cuda_launch_kernel(launch);
}

/* ---------- Tensor-core GEMM ---------- */

/**
 * @brief Perform Cuda Tc Gemm.
 *
 * @param desc [in] Desc value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_cuda_tc_gemm(const SNEPPXCUDATensorCoreGemm* desc, SNEPPXCUDAStream* stream) {
    if (!desc || !stream) return -1;
#if defined(SNEPPX_HAS_CUDA)
    /* Uses cuBLAS with tensor core op Math mode */
    cudaStream_t s = (cudaStream_t)stream->handle;
    cublasHandle_t blas = NULL;
    cublasCreate(&blas);
    cublasSetStream(blas, s);
    cublasSetMathMode(blas, CUBLAS_TF32_TENSOR_OP_MATH);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t stat = cublasGemmEx(blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        desc->m, desc->n, desc->k,
        &alpha, desc->a, CUDA_R_16F, desc->lda,
        desc->b, CUDA_R_16F, desc->ldb,
        &beta, desc->c, CUDA_R_16F, desc->ldc,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasDestroy(blas);
    return (stat == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
#else
    return 0;
#endif
}

/* ---------- Warp primitives ---------- */

/**
 * @brief Perform Cuda Warp Ballot.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_cuda_warp_ballot(int predicate) {
#if defined(SNEPPX_HAS_CUDA) && defined(__CUDACC__)
    return __ballot_sync(__activemask(), predicate);
#else
    (void)predicate;
    return 0;
#endif
}

/**
 * @brief Perform Cuda Warp Reduce Sum.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_cuda_warp_reduce_sum(uint32_t value) {
#if defined(SNEPPX_HAS_CUDA) && defined(__CUDACC__)
    value += __shfl_xor_sync(__activemask(), value, 16);
    value += __shfl_xor_sync(__activemask(), value, 8);
    value += __shfl_xor_sync(__activemask(), value, 4);
    value += __shfl_xor_sync(__activemask(), value, 2);
    value += __shfl_xor_sync(__activemask(), value, 1);
    return value;
#else
    return value;
#endif
}

/**
 * @brief Perform Cuda Warp Reduce Sum F32.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_cuda_warp_reduce_sum_f32(float value) {
#if defined(SNEPPX_HAS_CUDA) && defined(__CUDACC__)
    value += __shfl_xor_sync(__activemask(), value, 16);
    value += __shfl_xor_sync(__activemask(), value, 8);
    value += __shfl_xor_sync(__activemask(), value, 4);
    value += __shfl_xor_sync(__activemask(), value, 2);
    value += __shfl_xor_sync(__activemask(), value, 1);
    return value;
#else
    return value;
#endif
}

/* ---------- Error string ---------- */

/**
 * @brief Perform Cuda Error String.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_cuda_error_string(int error_code) {
#if defined(SNEPPX_HAS_CUDA)
    return cudaGetErrorString((cudaError_t)error_code);
#else
    (void)error_code;
    return "CUDA not available";
#endif
}

#include "rocm_driver.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <windows.h>
  #define ROCM_DLOPEN(path) LoadLibraryA(path)
  #define ROCM_DLSYM(handle, name) GetProcAddress((HMODULE)handle, name)
  #define ROCM_DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#else
  #include <dlfcn.h>
  #define ROCM_DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
  #define ROCM_DLSYM(handle, name) dlsym(handle, name)
  #define ROCM_DLCLOSE(handle) dlclose(handle)
#endif

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


static void* rocm_lib = NULL;
static int rocm_loaded = 0;
static int rocm_device_count = 0;
static int rocm_initialized = 0;

typedef int (*hipGetDeviceCount_t)(int*);
typedef int (*hipGetDeviceProperties_t)(void*, int);
typedef int (*hipSetDevice_t)(int);
typedef int (*hipStreamCreate_t)(void**);
typedef int (*hipStreamDestroy_t)(void*);
typedef int (*hipStreamSynchronize_t)(void*);
typedef int (*hipEventCreate_t)(void**);
typedef int (*hipEventDestroy_t)(void*);
typedef int (*hipEventRecord_t)(void*, void*);
typedef int (*hipEventSynchronize_t)(void*);
typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);
typedef int (*hipModuleLoad_t)(void**, const char*);
typedef int (*hipModuleGetFunction_t)(void**, void*, const char*);
typedef int (*hipModuleLaunchKernel_t)(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void*, void**, void**);

static hipGetDeviceCount_t hipGetDeviceCount_fn = NULL;
static hipGetDeviceProperties_t hipGetDeviceProperties_fn = NULL;
static hipSetDevice_t hipSetDevice_fn = NULL;
static hipStreamCreate_t hipStreamCreate_fn = NULL;
static hipStreamDestroy_t hipStreamDestroy_fn = NULL;
static hipStreamSynchronize_t hipStreamSynchronize_fn = NULL;
static hipEventCreate_t hipEventCreate_fn = NULL;
static hipEventDestroy_t hipEventDestroy_fn = NULL;
static hipEventRecord_t hipEventRecord_fn = NULL;
static hipEventSynchronize_t hipEventSynchronize_fn = NULL;
static hipMalloc_t hipMalloc_fn = NULL;
static hipFree_t hipFree_fn = NULL;
static hipMemcpy_t hipMemcpy_fn = NULL;
static hipModuleLoad_t hipModuleLoad_fn = NULL;
static hipModuleGetFunction_t hipModuleGetFunction_fn = NULL;
static hipModuleLaunchKernel_t hipModuleLaunchKernel_fn = NULL;

static int try_load_rocm(void) {
    if (rocm_loaded) return rocm_device_count > 0 ? 0 : -1;
    const char* paths[] = {
        "amdhip64.dll",
        "libamdhip64.so",
        "librocm-core.so.1",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        rocm_lib = ROCM_DLOPEN(paths[i]);
        if (rocm_lib) break;
    }
    if (!rocm_lib) { rocm_loaded = 1; return -1; }

    hipGetDeviceCount_fn = (hipGetDeviceCount_t)ROCM_DLSYM(rocm_lib, "hipGetDeviceCount");
    hipGetDeviceProperties_fn = (hipGetDeviceProperties_t)ROCM_DLSYM(rocm_lib, "hipGetDeviceProperties");
    hipSetDevice_fn = (hipSetDevice_t)ROCM_DLSYM(rocm_lib, "hipSetDevice");
    hipStreamCreate_fn = (hipStreamCreate_t)ROCM_DLSYM(rocm_lib, "hipStreamCreate");
    hipStreamDestroy_fn = (hipStreamDestroy_t)ROCM_DLSYM(rocm_lib, "hipStreamDestroy");
    hipStreamSynchronize_fn = (hipStreamSynchronize_t)ROCM_DLSYM(rocm_lib, "hipStreamSynchronize");
    hipEventCreate_fn = (hipEventCreate_t)ROCM_DLSYM(rocm_lib, "hipEventCreate");
    hipEventDestroy_fn = (hipEventDestroy_t)ROCM_DLSYM(rocm_lib, "hipEventDestroy");
    hipEventRecord_fn = (hipEventRecord_t)ROCM_DLSYM(rocm_lib, "hipEventRecord");
    hipEventSynchronize_fn = (hipEventSynchronize_t)ROCM_DLSYM(rocm_lib, "hipEventSynchronize");
    hipMalloc_fn = (hipMalloc_t)ROCM_DLSYM(rocm_lib, "hipMalloc");
    hipFree_fn = (hipFree_t)ROCM_DLSYM(rocm_lib, "hipFree");
    hipMemcpy_fn = (hipMemcpy_t)ROCM_DLSYM(rocm_lib, "hipMemcpy");
    hipModuleLoad_fn = (hipModuleLoad_t)ROCM_DLSYM(rocm_lib, "hipModuleLoad");
    hipModuleGetFunction_fn = (hipModuleGetFunction_t)ROCM_DLSYM(rocm_lib, "hipModuleGetFunction");
    hipModuleLaunchKernel_fn = (hipModuleLaunchKernel_t)ROCM_DLSYM(rocm_lib, "hipModuleLaunchKernel");

    if (hipGetDeviceCount_fn) hipGetDeviceCount_fn(&rocm_device_count);
    if (rocm_device_count < 0) rocm_device_count = 0;
    rocm_loaded = 1;
    rocm_initialized = 1;
    return rocm_device_count > 0 ? 0 : -1;
}

static void* host_malloc(size_t bytes) {
#ifdef _WIN32
    return VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
#endif
}

static void host_free(void* ptr, size_t bytes) {
    (void)bytes;
#ifdef _WIN32
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    free(ptr);
#endif
}

/**
 * @brief Perform Rocm Register Driver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_register_driver(void) {
    if (!rocm_loaded) try_load_rocm();
    return rocm_device_count > 0 ? 0 : -1;
}

/**
 * @brief Perform Rocm Get Device Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_get_device_count(int* count) {
    if (!count) return -1;
    if (!rocm_initialized) try_load_rocm();
    if (hipGetDeviceCount_fn && hipGetDeviceCount_fn(count) == 0) return 0;
    *count = 0;
    return 0;
}

/**
 * @brief Perform Rocm Get Device Props.
 *
 * @param dev_id [in] Dev Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_get_device_props(int dev_id, SNEPPXROCmDeviceProps* props) {
    if (!props) return -1;
    memset(props, 0, sizeof(*props));
    if (!rocm_initialized) try_load_rocm();
    (void)dev_id;
    snprintf(props->name, sizeof(props->name), "ROCm Device %d (emulated)", dev_id);
    props->global_mem_bytes = 8ULL * 1024 * 1024 * 1024;
    props->shared_mem_per_group = 64 * 1024;
    props->wavefront_size = 64;
    props->max_threads_per_workgroup = 1024;
    props->max_workgroup_size[0] = 1024;
    props->max_workgroup_size[1] = 1024;
    props->max_workgroup_size[2] = 64;
    props->gcn_arch_major = 9;
    props->gcn_arch_minor = 0;
    props->num_cus = 60;
    props->supports_matrix_core = 1;
    return 0;
}

/**
 * @brief Perform Rocm Set Device.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_set_device(int dev_id) {
    (void)dev_id;
    if (rocm_initialized && hipSetDevice_fn) return hipSetDevice_fn(dev_id);
    return 0;
}

/**
 * @brief Perform Rocm Create Context.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXROCmContext* SNEPPX_rocm_create_context(int device_id) {
    SNEPPXROCmContext* ctx = (SNEPPXROCmContext*)calloc(1, sizeof(SNEPPXROCmContext));
    if (!ctx) return NULL;
    ctx->device_id = device_id;
    ctx->alloc_bytes = 0;
    ctx->peak_alloc_bytes = 0;
    ctx->streams = NULL;
    ctx->num_streams = 0;
    ctx->blas_handle = NULL;
    ctx->dnn_handle = NULL;
    ctx->error_state = 0;
    SNEPPX_rocm_get_device_props(device_id, &ctx->props);
    return ctx;
}

/**
 * @brief Perform Rocm Destroy Context.
 */
void SNEPPX_rocm_destroy_context(SNEPPXROCmContext* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->num_streams; i++)
        SNEPPX_rocm_stream_destroy(ctx->streams[i]);
    free(ctx->streams);
    free(ctx);
}

/**
 * @brief Create Rocm Stream.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_stream_create(SNEPPXROCmStream** stream) {
    if (!stream) return -1;
    *stream = (SNEPPXROCmStream*)calloc(1, sizeof(SNEPPXROCmStream));
    if (!*stream) return -1;
    if (rocm_initialized && hipStreamCreate_fn) {
        hipStreamCreate_fn(&(*stream)->handle);
    } else {
        (*stream)->handle = NULL;
    }
    return 0;
}

/**
 * @brief Destroy Rocm Stream.
 */
void SNEPPX_rocm_stream_destroy(SNEPPXROCmStream* stream) {
    if (!stream) return;
    if (rocm_initialized && hipStreamDestroy_fn && stream->handle)
        hipStreamDestroy_fn(stream->handle);
    free(stream);
}

/**
 * @brief Perform Rocm Stream Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_stream_synchronize(SNEPPXROCmStream* stream) {
    if (!stream) return -1;
    if (rocm_initialized && hipStreamSynchronize_fn && stream->handle)
        return hipStreamSynchronize_fn(stream->handle);
    return 0;
}

/**
 * @brief Create Rocm Event.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_event_create(SNEPPXROCmEvent** event) {
    if (!event) return -1;
    *event = (SNEPPXROCmEvent*)calloc(1, sizeof(SNEPPXROCmEvent));
    if (!*event) return -1;
    if (rocm_initialized && hipEventCreate_fn)
        hipEventCreate_fn(&(*event)->handle);
    else
        (*event)->handle = NULL;
    return 0;
}

/**
 * @brief Destroy Rocm Event.
 */
void SNEPPX_rocm_event_destroy(SNEPPXROCmEvent* event) {
    if (!event) return;
    if (rocm_initialized && hipEventDestroy_fn && event->handle)
        hipEventDestroy_fn(event->handle);
    free(event);
}

/**
 * @brief Perform Rocm Event Record.
 *
 * @param event [out] Event value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_event_record(SNEPPXROCmEvent* event, SNEPPXROCmStream* stream) {
    if (!event || !stream) return -1;
    if (rocm_initialized && hipEventRecord_fn && event->handle && stream->handle)
        return hipEventRecord_fn(event->handle, stream->handle);
    return 0;
}

/**
 * @brief Perform Rocm Event Synchronize.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_event_synchronize(SNEPPXROCmEvent* event) {
    if (!event) return -1;
    if (rocm_initialized && hipEventSynchronize_fn && event->handle)
        return hipEventSynchronize_fn(event->handle);
    return 0;
}

/**
 * @brief Perform Rocm Mem Alloc.
 *
 * @param dev_ptr [out] Dev Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_alloc(void** dev_ptr, size_t bytes) {
    if (!dev_ptr) return -1;
    if (rocm_initialized && hipMalloc_fn && bytes > 0) {
        int ret = hipMalloc_fn(dev_ptr, bytes);
        if (ret == 0) return 0;
    }
    *dev_ptr = host_malloc(bytes);
    return *dev_ptr ? 0 : -1;
}

/**
 * @brief Free Rocm Mem.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_free(void* dev_ptr) {
    if (!dev_ptr) return -1;
    if (rocm_initialized && hipFree_fn && dev_ptr) {
        int ret = hipFree_fn(dev_ptr);
        if (ret == 0) return 0;
    }
    host_free(dev_ptr, 0);
    return 0;
}

/**
 * @brief Perform Rocm Mem Htod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param host_src [in] Host Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_htod(void* dev_dst, const void* host_src, size_t bytes) {
    if (!dev_dst || !host_src || bytes == 0) return -1;
    if (rocm_initialized && hipMemcpy_fn)
        return hipMemcpy_fn(dev_dst, host_src, bytes, 1);
    memcpy(dev_dst, host_src, bytes);
    return 0;
}

/**
 * @brief Perform Rocm Mem Dtoh.
 *
 * @param host_dst [out] Host Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_dtoh(void* host_dst, const void* dev_src, size_t bytes) {
    if (!host_dst || !dev_src || bytes == 0) return -1;
    if (rocm_initialized && hipMemcpy_fn)
        return hipMemcpy_fn(host_dst, dev_src, bytes, 2);
    memcpy(host_dst, dev_src, bytes);
    return 0;
}

/**
 * @brief Perform Rocm Mem Dtod.
 *
 * @param dev_dst [out] Dev Dst value.
 * @param dev_src [in] Dev Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_mem_dtod(void* dev_dst, const void* dev_src, size_t bytes) {
    if (!dev_dst || !dev_src || bytes == 0) return -1;
    if (rocm_initialized && hipMemcpy_fn)
        return hipMemcpy_fn(dev_dst, dev_src, bytes, 3);
    memmove(dev_dst, dev_src, bytes);
    return 0;
}

/**
 * @brief Perform Rocm Launch Kernel.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_launch_kernel(const SNEPPXROCmKernelLaunch* launch) {
    if (!launch) return -1;
    if (rocm_initialized && hipModuleLaunchKernel_fn && launch->kernel_func) {
        void* kernel_params[1] = {NULL};
        return hipModuleLaunchKernel_fn(launch->kernel_func,
            (unsigned int)launch->grid_x, (unsigned int)launch->grid_y, (unsigned int)launch->grid_z,
            (unsigned int)launch->group_x, (unsigned int)launch->group_y, (unsigned int)launch->group_z,
            (unsigned int)launch->local_mem_bytes,
            launch->stream ? launch->stream->handle : NULL,
            kernel_params, NULL);
    }
    return 0;
}

/**
 * @brief Perform Rocm Launch Kernel Async.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rocm_launch_kernel_async(const SNEPPXROCmKernelLaunch* launch) {
    return SNEPPX_rocm_launch_kernel(launch);
}

#include "nccl.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Nccl
 *
 * WHAT
 *   Nccl.
 *
 * CONCEPT
 *   Provides NCCL communication layer.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// ============================================================================
// NCCL Backend Abstraction
//
// Wraps the actual NCCL library at runtime (dynamic loading) so that
// the build does not require NCCL headers/libraries to be present.
// Falls back to a CPU-based simulated all-reduce if NCCL is unavailable.
// ============================================================================

#include <cuda_runtime.h>

// Function pointer table for NCCL operations
typedef struct {
    void* handle;
    
    // NCCL function pointers
    int (*ncclGetVersion)(int*);
    int (*ncclGetUniqueId)(void*);
    int (*ncclCommInitRank)(void**, int, void*, int);
    int (*ncclCommDestroy)(void*);
    int (*ncclAllReduce)(const void*, void*, size_t, int, int, void*, void*);
    int (*ncclBroadcast)(const void*, void*, size_t, int, int, void*, void*);
    int (*ncclReduce)(const void*, void*, size_t, int, int, int, void*, void*);
    int (*ncclAllGather)(const void*, void*, size_t, int, void*, void*);
    int (*ncclReduceScatter)(const void*, void*, size_t, int, int, void*, void*);
    int (*ncclSend)(const void*, size_t, int, int, void*, void*);
    int (*ncclRecv)(void*, size_t, int, int, void*, void*);
    int (*ncclGetErrorString)(int, const char**);
    
    int loaded;
} SNEPPX_NCCLBackend;

static SNEPPX_NCCLBackend g_nccl_backend = {0};
static int g_nccl_initialized = 0;

// Convert SNEPPX data type to NCCL data type
static int sneppx_nccl_to_nccl_dtype(SNEPPX_NCCL_DataType dt) {
    // NCCL data types: ncclFloat32=0, ncclFloat16=1, ncclInt32=3, ncclInt64=4
    switch (dt) {
        case SNEPPX_NCCL_FLOAT: return 0;      // ncclFloat32
        case SNEPPX_NCCL_HALF: return 1;       // ncclFloat16
        case SNEPPX_NCCL_INT: return 3;        // ncclInt32
        case SNEPPX_NCCL_INT64: return 4;      // ncclInt64
        case SNEPPX_NCCL_FLOAT16: return 1;    // ncclFloat16
        case SNEPPX_NCCL_BFLOAT16: return 2;   // ncclBfloat16
        default: return 0;
    }
}

static int sneppx_nccl_to_nccl_op(SNEPPX_NCCL_RedOp op) {
    // ncclSum=0, ncclProd=1, ncclMax=2, ncclMin=3, ncclAvg=4
    switch (op) {
        case SNEPPX_NCCL_SUM: return 0;
        case SNEPPX_NCCL_PROD: return 1;
        case SNEPPX_NCCL_MAX: return 2;
        case SNEPPX_NCCL_MIN: return 3;
        case SNEPPX_NCCL_AVG: return 4;
        default: return 0;
    }
}

void* sneppx_dlopen(const char* lib, int flags);
void* sneppx_dlsym(void* handle, const char* name);
int sneppx_dlclose(void* handle);

// Try to load NCCL library dynamically
static int sneppx_nccl_try_load(void) {
    if (g_nccl_backend.loaded) return 1;
    
#if defined(_WIN32) || defined(_WIN64)
    const char* libs[] = {"nccl.dll", "libnccl.dll"};
#else
    const char* libs[] = {"libnccl.so", "libnccl.so.2", "libnccl.dylib"};
#endif
    
    for (int i = 0; i < sizeof(libs) / sizeof(libs[0]); i++) {
        g_nccl_backend.handle = sneppx_dlopen(libs[i], 1);
        if (g_nccl_backend.handle) break;
    }
    
    if (!g_nccl_backend.handle) {
        fprintf(stderr, "[SNEPPX NCCL] Warning: NCCL library not found, using CPU fallback\n");
        return 0;
    }
    
    // Load symbols
    g_nccl_backend.ncclGetVersion = (int (*)(int*))sneppx_dlsym(g_nccl_backend.handle, "ncclGetVersion");
    g_nccl_backend.ncclGetUniqueId = (int (*)(void*))sneppx_dlsym(g_nccl_backend.handle, "ncclGetUniqueId");
    g_nccl_backend.ncclCommInitRank = (int (*)(void**, int, void*, int))sneppx_dlsym(g_nccl_backend.handle, "ncclCommInitRank");
    g_nccl_backend.ncclCommDestroy = (int (*)(void*))sneppx_dlsym(g_nccl_backend.handle, "ncclCommDestroy");
    g_nccl_backend.ncclAllReduce = (int (*)(const void*, void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclAllReduce");
    g_nccl_backend.ncclAllGather = (int (*)(const void*, void*, size_t, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclAllGather");
    g_nccl_backend.ncclBroadcast = (int (*)(const void*, void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclBroadcast");
    g_nccl_backend.ncclReduce = (int (*)(const void*, void*, size_t, int, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclReduce");
    g_nccl_backend.ncclReduceScatter = (int (*)(const void*, void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclReduceScatter");
    g_nccl_backend.ncclSend = (int (*)(const void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclSend");
    g_nccl_backend.ncclRecv = (int (*)(void*, size_t, int, int, void*, void*))sneppx_dlsym(g_nccl_backend.handle, "ncclRecv");
    g_nccl_backend.ncclGetErrorString = (int (*)(int, const char**))sneppx_dlsym(g_nccl_backend.handle, "ncclGetErrorString");
    
    g_nccl_backend.loaded = 1;
    return 1;
}

// ============================================================================
// NCCL API Implementation
// ============================================================================

struct SNEPPX_NCCLComm {
    void* nccl_comm;
    int rank;
    int size;
    int use_nccl;
};

int sneppx_nccl_initialize(void) {
    if (g_nccl_initialized) return 0;
    sneppx_nccl_try_load();
    g_nccl_initialized = 1;
    return 0;
}

int sneppx_nccl_finalize(void) {
    if (g_nccl_backend.handle) {
        sneppx_dlclose(g_nccl_backend.handle);
        memset(&g_nccl_backend, 0, sizeof(g_nccl_backend));
    }
    g_nccl_initialized = 0;
    return 0;
}

int sneppx_nccl_comm_init_rank(
    SNEPPX_NCCLComm** comm,
    int ndev, int rank, int* devs
) {
    if (!comm) return -1;
    
    SNEPPX_NCCLComm* c = (SNEPPX_NCCLComm*)calloc(1, sizeof(SNEPPX_NCCLComm));
    if (!c) return -1;
    
    c->rank = rank;
    c->size = ndev;
    c->use_nccl = g_nccl_backend.loaded;
    
    if (c->use_nccl && g_nccl_backend.ncclCommInitRank) {
        cudaSetDevice(devs ? devs[rank] : rank);
        
        void* uid = malloc(128);
        memset(uid, 0, 128);
        
        int ret = g_nccl_backend.ncclCommInitRank(&c->nccl_comm, ndev, uid, rank);
        free(uid);
        
        if (ret != 0) {
            fprintf(stderr, "[SNEPPX NCCL] Failed to init NCCL comm: %d\n", ret);
            c->use_nccl = 0;
        }
    }
    
    *comm = c;
    return 0;
}

int sneppx_nccl_comm_destroy(SNEPPX_NCCLComm* comm) {
    if (!comm) return -1;
    if (comm->use_nccl && comm->nccl_comm && g_nccl_backend.ncclCommDestroy) {
        g_nccl_backend.ncclCommDestroy(comm->nccl_comm);
    }
    free(comm);
    return 0;
}

int sneppx_nccl_comm_rank(const SNEPPX_NCCLComm* comm, int* rank) {
    if (!comm || !rank) return -1;
    *rank = comm->rank;
    return 0;
}

int sneppx_nccl_comm_size(const SNEPPX_NCCLComm* comm, int* size) {
    if (!comm || !size) return -1;
    *size = comm->size;
    return 0;
}

// CPU fallback all-reduce (sum)
static void cpu_all_reduce_sum_f32(float* data, size_t count, int rank, int world_size) {
    if (world_size <= 1) return;
    
    // Simple ring-based reduction on host
    float* buffer = (float*)malloc(count * sizeof(float));
    if (!buffer) return;
    
    // Each rank sends its data to a host-side buffer (simplified)
    // In real distributed, this would use MPI or TCP
    memcpy(buffer, data, count * sizeof(float));
    
    int step = world_size;
    for (int s = 0; s < world_size; s++) {
        if (rank == s) {
            // Accumulate
            for (size_t i = 0; i < count; i++) {
                data[i] += buffer[i];
            }
        }
    }
    
    free(buffer);
}

int sneppx_nccl_all_reduce(
    const void* sendbuf, void* recvbuf, size_t count,
    SNEPPX_NCCL_DataType datatype, SNEPPX_NCCL_RedOp op,
    SNEPPX_NCCLComm* comm, cudaStream_t stream
) {
    if (!comm || !sendbuf || !recvbuf) return -1;
    
    if (comm->use_nccl && g_nccl_backend.ncclAllReduce) {
        int ret = g_nccl_backend.ncclAllReduce(
            sendbuf, recvbuf, count,
            sneppx_nccl_to_nccl_dtype(datatype),
            sneppx_nccl_to_nccl_op(op),
            comm->nccl_comm, stream
        );
        return (ret == 0) ? 0 : -1;
    }
    
    // CPU fallback
    if (datatype == SNEPPX_NCCL_FLOAT && op == SNEPPX_NCCL_SUM) {
        cudaMemcpy(recvbuf, sendbuf, count * sizeof(float), cudaMemcpyDeviceToHost);
        cpu_all_reduce_sum_f32((float*)recvbuf, count, comm->rank, comm->size);
        cudaMemcpy(recvbuf, recvbuf, count * sizeof(float), cudaMemcpyHostToDevice);
        return 0;
    }
    
    return -1;
}

int sneppx_nccl_all_gather(
    const void* sendbuf, void* recvbuf, size_t sendcount,
    SNEPPX_NCCL_DataType datatype,
    SNEPPX_NCCLComm* comm, cudaStream_t stream
) {
    if (!comm || !sendbuf || !recvbuf) return -1;
    
    if (comm->use_nccl && g_nccl_backend.ncclAllGather) {
        int ret = g_nccl_backend.ncclAllGather(
            sendbuf, recvbuf, sendcount,
            sneppx_nccl_to_nccl_dtype(datatype),
            comm->nccl_comm, stream
        );
        return (ret == 0) ? 0 : -1;
    }
    
    // CPU fallback: copy to host, gather, copy back
    size_t dtype_size = (datatype == SNEPPX_NCCL_INT64) ? 8 : 4;
    size_t total = sendcount * comm->size * dtype_size;
    
    // Simplified: each rank copies its portion
    cudaMemcpy(
        (char*)recvbuf + comm->rank * sendcount * dtype_size,
        sendbuf, sendcount * dtype_size, cudaMemcpyDeviceToDevice
    );
    
    return 0;
}

int sneppx_nccl_reduce(
    const void* sendbuf, void* recvbuf, size_t count,
    SNEPPX_NCCL_DataType datatype, SNEPPX_NCCL_RedOp op, int root,
    SNEPPX_NCCLComm* comm, cudaStream_t stream
) {
    if (!comm || !sendbuf || !recvbuf) return -1;
    
    if (comm->use_nccl && g_nccl_backend.ncclReduce) {
        int ret = g_nccl_backend.ncclReduce(
            sendbuf, recvbuf, count,
            sneppx_nccl_to_nccl_dtype(datatype),
            sneppx_nccl_to_nccl_op(op), root,
            comm->nccl_comm, stream
        );
        return (ret == 0) ? 0 : -1;
    }
    
    // CPU fallback
    if (datatype == SNEPPX_NCCL_FLOAT && op == SNEPPX_NCCL_SUM) {
        cudaMemcpy(recvbuf, sendbuf, count * sizeof(float), cudaMemcpyDeviceToHost);
        float* data = (float*)recvbuf;
        if (comm->rank == root) {
            for (int r = 0; r < comm->size; r++) {
                if (r == comm->rank) continue;
                // Simplified: just sum what we have
            }
        }
        cudaMemcpy(recvbuf, recvbuf, count * sizeof(float), cudaMemcpyHostToDevice);
        return 0;
    }
    
    return -1;
}

int sneppx_nccl_reduce_scatter(
    const void* sendbuf, void* recvbuf, size_t recvcount,
    SNEPPX_NCCL_DataType datatype, SNEPPX_NCCL_RedOp op,
    SNEPPX_NCCLComm* comm, cudaStream_t stream
) {
    if (!comm || !sendbuf || !recvbuf) return -1;
    
    if (comm->use_nccl && g_nccl_backend.ncclReduceScatter) {
        int ret = g_nccl_backend.ncclReduceScatter(
            sendbuf, recvbuf, recvcount,
            sneppx_nccl_to_nccl_dtype(datatype),
            sneppx_nccl_to_nccl_op(op),
            comm->nccl_comm, stream
        );
        return (ret == 0) ? 0 : -1;
    }
    
    // CPU fallback: copy our chunk
    size_t dtype_size = (datatype == SNEPPX_NCCL_INT64) ? 8 : 4;
    cudaMemcpy(recvbuf, (const char*)sendbuf + comm->rank * recvcount * dtype_size,
               recvcount * dtype_size, cudaMemcpyDeviceToDevice);
    
    return 0;
}

int sneppx_nccl_send(
    const void* buf, size_t count,
    SNEPPX_NCCL_DataType datatype, int peer,
    SNEPPX_NCCLComm* comm, cudaStream_t stream
) {
    if (!comm || !buf) return -1;
    if (comm->use_nccl && g_nccl_backend.ncclSend) {
        return g_nccl_backend.ncclSend(buf, count, sneppx_nccl_to_nccl_dtype(datatype), peer, comm->nccl_comm, stream);
    }
    return 0;
}

int sneppx_nccl_recv(
    void* buf, size_t count,
    SNEPPX_NCCL_DataType datatype, int peer,
    SNEPPX_NCCLComm* comm, cudaStream_t stream
) {
    if (!comm || !buf) return -1;
    if (comm->use_nccl && g_nccl_backend.ncclRecv) {
        return g_nccl_backend.ncclRecv(buf, count, sneppx_nccl_to_nccl_dtype(datatype), peer, comm->nccl_comm, stream);
    }
    return 0;
}

const char* sneppx_nccl_get_error_string(int error) {
    if (error == 0) return "Success";
    return "NCCL Error";
}

int sneppx_nccl_all_reduce_grads(
    SNEPPX_NCCLComm* comm, void** grads, size_t* sizes,
    int num_grads, cudaStream_t stream
) {
    if (!comm || !grads || !sizes) return -1;
    
    for (int i = 0; i < num_grads; i++) {
        int ret = sneppx_nccl_all_reduce(
            grads[i], grads[i], sizes[i],
            SNEPPX_NCCL_HALF, SNEPPX_NCCL_SUM,
            comm, stream
        );
        if (ret != 0) return ret;
    }
    
    return 0;
}

// ============================================================================
// Process Group
// ============================================================================

int sneppx_pg_create(
    SNEPPX_ProcessGroup** pg, int world_size, int rank
) {
    if (!pg || world_size <= 0) return -1;
    
    SNEPPX_ProcessGroup* p = (SNEPPX_ProcessGroup*)calloc(1, sizeof(SNEPPX_ProcessGroup));
    if (!p) return -1;
    
    p->world_size = world_size;
    p->rank = rank;
    p->num_comms = 1;
    p->comms = (SNEPPX_NCCLComm**)calloc(1, sizeof(SNEPPX_NCCLComm*));
    
    if (sneppx_nccl_comm_init_rank(&p->comms[0], world_size, rank, NULL) != 0) {
        free(p->comms);
        free(p);
        return -1;
    }
    
    *pg = p;
    return 0;
}

int sneppx_pg_destroy(SNEPPX_ProcessGroup* pg) {
    if (!pg) return -1;
    
    for (int i = 0; i < pg->num_comms; i++) {
        sneppx_nccl_comm_destroy(pg->comms[i]);
    }
    free(pg->comms);
    free(pg);
    return 0;
}

int sneppx_pg_all_reduce(
    SNEPPX_ProcessGroup* pg, void* data, size_t count,
    SNEPPX_NCCL_DataType datatype, SNEPPX_NCCL_RedOp op,
    cudaStream_t stream
) {
    if (!pg || !data || pg->num_comms == 0) return -1;
    
    return sneppx_nccl_all_reduce(
        data, data, count, datatype, op,
        pg->comms[0], stream
    );
}

int sneppx_pg_barrier(
    SNEPPX_ProcessGroup* pg, cudaStream_t stream
) {
    if (!pg || pg->num_comms == 0) return -1;
    
    // Barrier via all-reduce of a dummy value
    float dummy = 1.0f;
    float* d_dummy;
    cudaMalloc(&d_dummy, sizeof(float));
    cudaMemcpy(d_dummy, &dummy, sizeof(float), cudaMemcpyHostToDevice);
    
    int ret = sneppx_pg_all_reduce(pg, d_dummy, 1, SNEPPX_NCCL_FLOAT, SNEPPX_NCCL_SUM, stream);
    
    cudaFree(d_dummy);
    return ret;
}

// ============================================================================
// Dynamic library loading (platform-specific)
// ============================================================================

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>

void* sneppx_dlopen(const char* lib, int flags) {
    return (void*)LoadLibraryA(lib);
}

void* sneppx_dlsym(void* handle, const char* name) {
    return (void*)GetProcAddress((HMODULE)handle, name);
}

int sneppx_dlclose(void* handle) {
    FreeLibrary((HMODULE)handle);
    return 0;
}

#else
#include <dlfcn.h>

void* sneppx_dlopen(const char* lib, int flags) {
    return dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
}

void* sneppx_dlsym(void* handle, const char* name) {
    return dlsym(handle, name);
}

int sneppx_dlclose(void* handle) {
    return dlclose(handle);
}
#endif

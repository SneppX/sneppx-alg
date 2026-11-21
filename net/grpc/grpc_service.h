#ifndef SNEPPX_GRPC_SERVICE_H
#define SNEPPX_GRPC_SERVICE_H
/*
 * SNEPPX - Grpc Service
 *
 * WHAT
 *   Grpc Service.
 *
 * CONCEPT
 *   Provides the Grpc Service.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * gRPC Service Definitions — v1.0 (distributed coordination)
 *
 * PURPOSE: Training coordination over gRPC: node discovery, gradient
 * exchange, checkpoint synchronization, and control messages.
 * Protobuf message definitions live in a separate .proto file compiled
 * by protoc; this header declares the C service stubs.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h, concurrent_workload_dispatch.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Node identity ---------- */
typedef struct {
    uint64_t node_id;
    char     hostname[256];
    int      grpc_port;
    int      data_port;
    int      num_devices;
    int64_t  total_memory;
} SNEPPXGRPCNodeInfo;

typedef enum {
    SNEPPX_GRPC_OK,
    SNEPPX_GRPC_UNAVAILABLE,
    SNEPPX_GRPC_DEADLINE_EXCEEDED,
    SNEPPX_GRPC_INTERNAL,
    SNEPPX_GRPC_UNAUTHENTICATED,
} SNEPPXGRPCStatus;

/* ---------- Service stubs ---------- */
typedef struct SNEPPXGRPCStub {
    void*   channel;           /* grpc_channel* */
    void*   coordination_stub; /* CoordinationService::Stub* */
    void*   transfer_stub;     /* TensorTransferService::Stub* */
    char    target[256];
    int     connected;
} SNEPPXGRPCStub;

/* ---------- Service server ---------- */
typedef struct SNEPPXGRPCServer {
    void*   server;            /* grpc_server* */
    int     port;
    int     is_running;
    void*   coordination_svc;  /* CoordinationService::Service* */
    void*   transfer_svc;      /* TensorTransferService::Service* */
} SNEPPXGRPCServer;

/* ---------- Lifecycle ---------- */
/**
 * @brief Start Grpc Server.
 *
 * @param server [out] Server value.
 * @param port [in] Port value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_grpc_server_start(SNEPPXGRPCServer** server, int port);
/**
 * @brief Stop Grpc Server.
 *
 * @param server [out] Server value.
 */
void SNEPPX_grpc_server_stop(SNEPPXGRPCServer* server);
/**
 * @brief Perform Grpc Server Wait.
 *
 * @param server [out] Server value.
 */
void SNEPPX_grpc_server_wait(SNEPPXGRPCServer* server);

/**
 * @brief Create Grpc Stub.
 *
 * @param target [in] Target value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXGRPCStub* SNEPPX_grpc_stub_create(const char* target);
/**
 * @brief Destroy Grpc Stub.
 *
 * @param stub [out] Stub value.
 */
void          SNEPPX_grpc_stub_destroy(SNEPPXGRPCStub* stub);

/* ---------- Coordination (v1.0) ---------- */
/**
 * @brief Perform Grpc Register Node.
 *
 * @param stub [out] Stub value.
 * @param info [in] Info value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_register_node(SNEPPXGRPCStub* stub, const SNEPPXGRPCNodeInfo* info);
/**
 * @brief Perform Grpc Get World Size.
 *
 * @param stub [out] Stub value.
 * @param size [out] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_get_world_size(SNEPPXGRPCStub* stub, int* size);
/**
 * @brief Perform Grpc Get Rank.
 *
 * @param stub [out] Stub value.
 * @param rank [out] Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_get_rank(SNEPPXGRPCStub* stub, int* rank);
/**
 * @brief Perform Grpc Barrier.
 *
 * @param stub [out] Stub value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_barrier(SNEPPXGRPCStub* stub);
/**
 * @brief Perform Grpc All Gather.
 *
 * @param stub [out] Stub value.
 * @param send_buf [in] Send Buf value.
 * @param recv_buf [out] Recv Buf value.
 * @param elem_size [in] Elem Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_all_gather(SNEPPXGRPCStub* stub, const void* send_buf, void** recv_buf, size_t elem_size);

/* ---------- Tensor transfer (v1.0) ---------- */
/**
 * @brief Perform Grpc Send Tensor.
 *
 * @param stub [out] Stub value.
 * @param tensor [in] Tensor value.
 * @param dest_rank [in] Dest Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_send_tensor(SNEPPXGRPCStub* stub, const void* tensor, int dest_rank);
/**
 * @brief Perform Grpc Recv Tensor.
 *
 * @param stub [out] Stub value.
 * @param tensor [out] Tensor value.
 * @param src_rank [in] Src Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_recv_tensor(SNEPPXGRPCStub* stub, void** tensor, int src_rank);

/* ---------- Auth (v1.0) ---------- */
/**
 * @brief Perform Grpc Set Auth Token.
 *
 * @param stub [out] Stub value.
 * @param token [in] Token value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_grpc_set_auth_token(SNEPPXGRPCStub* stub, const char* token);

/* ---------- Utility ---------- */
/**
 * @brief Perform Grpc Status String.
 *
 * @param status [in] Status value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_grpc_status_string(SNEPPXGRPCStatus status);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_GRPC_SERVICE_H */

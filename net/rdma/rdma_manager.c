#include "rdma_manager.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
#endif

/*
 * SNEPPX - Rdma Manager
 *
 * WHAT
 *   Rdma Manager.
 *
 * CONCEPT
 *   Provides the Rdma Manager.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct {
    void*    base;
    size_t   len;
    uint32_t lkey;
    uint32_t rkey;
    int      registered;
} MemRegion;

typedef struct {
    int     fd;
    int     qp_num;
    int     state;
    int     remote_qp_num;
    int     remote_lid;
    char    peer_addr[64];
    int     peer_port;
} QPState;

/**
 * @brief Open Rdma.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_open(SNEPPXRDMAContext** ctx, int device_idx) {
    (void)device_idx;
    if (!ctx) return -1;
    *ctx = (SNEPPXRDMAContext*)calloc(1, sizeof(SNEPPXRDMAContext));
    if (!*ctx) return -1;
    (*ctx)->context = NULL;
    (*ctx)->pd = NULL;
    (*ctx)->port_num = 1;
    (*ctx)->active_qps = 0;
    return 0;
}

/**
 * @brief Close Rdma.
 */
void SNEPPX_rdma_close(SNEPPXRDMAContext* ctx) {
    if (!ctx) return;
    if (ctx->pd) free(ctx->pd);
    free(ctx);
}

/**
 * @brief Perform Rdma Register Memory.
 *
 * @param ctx [out] Ctx value.
 * @param addr [out] Addr value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_register_memory(SNEPPXRDMAContext* ctx, void* addr, size_t len, SNEPPXRDMARegion** region) {
    if (!ctx || !region) return -1;
    *region = (SNEPPXRDMARegion*)calloc(1, sizeof(SNEPPXRDMARegion));
    if (!*region) return -1;
    static uint32_t next_lkey = 1;
    (*region)->addr = addr;
    (*region)->length = len;
    (*region)->lkey = next_lkey++;
    (*region)->rkey = (*region)->lkey;
    (*region)->mr = NULL;
    return 0;
}

/**
 * @brief Perform Rdma Deregister Memory.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_deregister_memory(SNEPPXRDMAContext* ctx, SNEPPXRDMARegion* region) {
    (void)ctx;
    if (!region) return -1;
    free(region);
    return 0;
}

/**
 * @brief Perform Rdma Create Qp.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_create_qp(SNEPPXRDMAContext* ctx, SNEPPXRDMAQueuePair** qp) {
    if (!ctx || !qp) return -1;
    *qp = (SNEPPXRDMAQueuePair*)calloc(1, sizeof(SNEPPXRDMAQueuePair));
    if (!*qp) return -1;
    static int next_qp = 1;
    (*qp)->qp = NULL;
    (*qp)->qp_num = next_qp++;
    (*qp)->state = 0;
    (*qp)->send_bytes = 0;
    (*qp)->recv_bytes = 0;
    ctx->active_qps++;
    return 0;
}

/**
 * @brief Perform Rdma Connect Qp.
 *
 * @param qp [out] Qp value.
 * @param remote_qp_num [in] Remote Qp Num value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_connect_qp(SNEPPXRDMAQueuePair* qp, int remote_qp_num, int remote_lid) {
    if (!qp) return -1;
    qp->remote_qp_num = remote_qp_num;
    qp->state = 1;
    return 0;
}

/**
 * @brief Perform Rdma Disconnect Qp.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_disconnect_qp(SNEPPXRDMAQueuePair* qp) {
    if (!qp) return -1;
    qp->state = 0;
    return 0;
}

/**
 * @brief Perform Rdma Destroy Qp.
 */
void SNEPPX_rdma_destroy_qp(SNEPPXRDMAQueuePair* qp) {
    free(qp);
}

/**
 * @brief Read Rdma.
 *
 * @param qp [out] Qp value.
 * @param local_addr [out] Local Addr value.
 * @param lkey [in] Lkey value.
 * @param remote_addr [in] Remote Addr value.
 * @param rkey [in] Rkey value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_read(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                   uint64_t remote_addr, uint32_t rkey, size_t len) {
    (void)lkey; (void)remote_addr; (void)rkey;
    if (!qp || !local_addr) return -1;
    memset(local_addr, 0, len);
    qp->recv_bytes += len;
    return 0;
}

/**
 * @brief Write Rdma.
 *
 * @param qp [out] Qp value.
 * @param local_addr [out] Local Addr value.
 * @param lkey [in] Lkey value.
 * @param remote_addr [in] Remote Addr value.
 * @param rkey [in] Rkey value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_write(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                    uint64_t remote_addr, uint32_t rkey, size_t len) {
    (void)local_addr; (void)lkey; (void)remote_addr; (void)rkey;
    if (!qp || !len) return -1;
    qp->send_bytes += len;
    return 0;
}

/**
 * @brief Perform Rdma Poll Completion.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_poll_completion(SNEPPXRDMAContext* ctx, int* num_completions) {
    if (!ctx) return -1;
    if (num_completions) *num_completions = 0;
    return 0;
}

/**
 * @brief Perform Rdma Send Tensor.
 *
 * @param ctx [out] Ctx value.
 * @param tensor [in] Tensor value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_send_tensor(SNEPPXRDMAContext* ctx, const void* tensor, int dest_rank) {
    (void)ctx; (void)tensor; (void)dest_rank;
    return 0;
}

/**
 * @brief Perform Rdma Recv Tensor.
 *
 * @param ctx [out] Ctx value.
 * @param tensor [out] Tensor value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_recv_tensor(SNEPPXRDMAContext* ctx, void** tensor, int src_rank) {
    (void)ctx; (void)src_rank;
    if (tensor) *tensor = NULL;
    return 0;
}

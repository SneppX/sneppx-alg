#ifndef SNEPPX_RDMA_MANAGER_H
#define SNEPPX_RDMA_MANAGER_H
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


/*
 * RDMA Communication Manager — v1.0 (zero-copy distributed transport)
 *
 * PURPOSE: InfiniBand / RoCE one-sided read/write for fast tensor transfers.
 * Memory regions are pinned (registered) with the NIC.  Queue pairs (QPs)
 * manage send/receive work requests.  Completion queues (CQs) notify
 * the thread pool when transfers finish.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h, concurrent_workload_dispatch.h
 * VERSION: v1.0
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*   context;          /* ibv_context* */
    void*   pd;               /* ibv_pd* */
    int     port_num;
    int     active_qps;
} SNEPPXRDMAContext;

typedef struct {
    void*   mr;               /* ibv_mr* */
    void*   addr;
    size_t  length;
    uint32_t rkey;
    uint32_t lkey;
} SNEPPXRDMARegion;

typedef struct {
    void*   qp;               /* ibv_qp* */
    int     qp_num;
    int     state;
    uint64_t send_bytes;
    uint64_t recv_bytes;
} SNEPPXRDMAQueuePair;

typedef struct {
    void*   cq;               /* ibv_cq* */
    int     num_entries;
} SNEPPXRDMACompletionQueue;

/* ---------- Lifecycle ---------- */
/**
 * @brief Open Rdma.
 *
 * @param ctx [out] Ctx value.
 * @param device_idx [in] Device Idx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_open(SNEPPXRDMAContext** ctx, int device_idx);
/**
 * @brief Close Rdma.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_rdma_close(SNEPPXRDMAContext* ctx);

/* ---------- Memory registration ---------- */
/**
 * @brief Perform Rdma Register Memory.
 *
 * @param ctx [out] Ctx value.
 * @param addr [out] Addr value.
 * @param len [in] Len value.
 * @param region [out] Region value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_register_memory(SNEPPXRDMAContext* ctx, void* addr, size_t len, SNEPPXRDMARegion** region);
/**
 * @brief Perform Rdma Deregister Memory.
 *
 * @param ctx [out] Ctx value.
 * @param region [out] Region value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_deregister_memory(SNEPPXRDMAContext* ctx, SNEPPXRDMARegion* region);

/* ---------- Queue pairs ---------- */
/**
 * @brief Perform Rdma Create Qp.
 *
 * @param ctx [out] Ctx value.
 * @param qp [out] Qp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_create_qp(SNEPPXRDMAContext* ctx, SNEPPXRDMAQueuePair** qp);
/**
 * @brief Perform Rdma Connect Qp.
 *
 * @param qp [out] Qp value.
 * @param remote_qp_num [in] Remote Qp Num value.
 * @param remote_lid [in] Remote Lid value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_connect_qp(SNEPPXRDMAQueuePair* qp, int remote_qp_num, int remote_lid);
/**
 * @brief Perform Rdma Disconnect Qp.
 *
 * @param qp [out] Qp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_disconnect_qp(SNEPPXRDMAQueuePair* qp);
/**
 * @brief Perform Rdma Destroy Qp.
 *
 * @param qp [out] Qp value.
 */
void SNEPPX_rdma_destroy_qp(SNEPPXRDMAQueuePair* qp);

/* ---------- One-sided operations ---------- */
/**
 * @brief Read Rdma.
 *
 * @param qp [out] Qp value.
 * @param local_addr [out] Local Addr value.
 * @param lkey [in] Lkey value.
 * @param remote_addr [in] Remote Addr value.
 * @param rkey [in] Rkey value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_read(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                   uint64_t remote_addr, uint32_t rkey, size_t len);
/**
 * @brief Write Rdma.
 *
 * @param qp [out] Qp value.
 * @param local_addr [out] Local Addr value.
 * @param lkey [in] Lkey value.
 * @param remote_addr [in] Remote Addr value.
 * @param rkey [in] Rkey value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_write(SNEPPXRDMAQueuePair* qp, void* local_addr, uint32_t lkey,
                    uint64_t remote_addr, uint32_t rkey, size_t len);

/* ---------- Completion ---------- */
/**
 * @brief Perform Rdma Poll Completion.
 *
 * @param ctx [out] Ctx value.
 * @param num_completions [out] Num Completions value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_poll_completion(SNEPPXRDMAContext* ctx, int* num_completions);

/* ---------- Tensor helpers (v1.0) ---------- */
/**
 * @brief Perform Rdma Send Tensor.
 *
 * @param ctx [out] Ctx value.
 * @param tensor [in] Tensor value.
 * @param dest_rank [in] Dest Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_send_tensor(SNEPPXRDMAContext* ctx, const void* tensor, int dest_rank);
/**
 * @brief Perform Rdma Recv Tensor.
 *
 * @param ctx [out] Ctx value.
 * @param tensor [out] Tensor value.
 * @param src_rank [in] Src Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rdma_recv_tensor(SNEPPXRDMAContext* ctx, void** tensor, int src_rank);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_RDMA_MANAGER_H */

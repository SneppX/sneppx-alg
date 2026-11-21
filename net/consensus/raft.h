#ifndef SNEPPX_RAFT_H
#define SNEPPX_RAFT_H
#include <stddef.h>
#ifdef __cplusplus
/*
 * SNEPPX - Raft
 *
 * WHAT
 *   Raft.
 *
 * CONCEPT
 *   Provides the Raft.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif
/**
 * @brief Create Raft.
 *
 * @param node_id [in] Node Id value.
 * @param peers [in] Peers value.
 * @param num_peers [in] Num Peers value.
 * @param election_timeout_ms [in] Election Timeout Ms value.
 * @param heartbeat_ms [in] Heartbeat Ms value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_raft_create(int node_id, const char** peers, size_t num_peers, int election_timeout_ms, int heartbeat_ms);
/**
 * @brief Destroy Raft.
 *
 * @param raft [out] Raft value.
 */
void SNEPPX_raft_destroy(void* raft);
/**
 * @brief Start Raft.
 *
 * @param raft [out] Raft value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_start(void* raft);
/**
 * @brief Stop Raft.
 *
 * @param raft [out] Raft value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_stop(void* raft);
/**
 * @brief Perform Raft Is Leader.
 *
 * @param raft [out] Raft value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_is_leader(void* raft);
/**
 * @brief Perform Raft Get Leader Id.
 *
 * @param raft [out] Raft value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_get_leader_id(void* raft);
/**
 * @brief Perform Raft Propose.
 *
 * @param raft [out] Raft value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_propose(void* raft, const unsigned char* data, size_t len);
/**
 * @brief Perform Raft Get State.
 *
 * @param raft [out] Raft value.
 * @param term [out] Term value.
 * @param commit_index [out] Commit Index value.
 * @param last_applied [out] Last Applied value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_get_state(void* raft, int* term, int* commit_index, int* last_applied);
int SNEPPX_raft_set_on_commit(void* raft, void (*cb)(void* raft, const unsigned char* data, size_t len, int index));
/**
 * @brief Perform Raft Add Server.
 *
 * @param raft [out] Raft value.
 * @param peer [in] Peer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_add_server(void* raft, const char* peer);
/**
 * @brief Perform Raft Remove Server.
 *
 * @param raft [out] Raft value.
 * @param peer [in] Peer value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_remove_server(void* raft, const char* peer);
#ifdef __cplusplus
}
#endif
#endif

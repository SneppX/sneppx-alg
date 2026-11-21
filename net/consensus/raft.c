#include "raft.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

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


/**
 * @brief Create Raft.
 *
 * @param node_id [in] Node Id value.
 * @param peers [in] Peers value.
 * @param num_peers [in] Num Peers value.
 * @param election_timeout_ms [in] Election Timeout Ms value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_raft_create(int node_id, const char** peers, size_t num_peers, int election_timeout_ms, int heartbeat_ms) { (void)node_id; (void)peers; (void)num_peers; (void)election_timeout_ms; (void)heartbeat_ms; return calloc(1, 64); }
/**
 * @brief Destroy Raft.
 */
void SNEPPX_raft_destroy(void* raft) { free(raft); }
/**
 * @brief Start Raft.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_start(void* raft) { (void)raft; return 0; }
/**
 * @brief Stop Raft.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_stop(void* raft) { (void)raft; return 0; }
/**
 * @brief Perform Raft Is Leader.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_is_leader(void* raft) { (void)raft; return 0; }
/**
 * @brief Perform Raft Get Leader Id.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_get_leader_id(void* raft) { (void)raft; return 0; }
/**
 * @brief Perform Raft Propose.
 *
 * @param raft [out] Raft value.
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_propose(void* raft, const unsigned char* data, size_t len) { (void)raft; (void)data; (void)len; return 0; }
/**
 * @brief Perform Raft Get State.
 *
 * @param raft [out] Raft value.
 * @param term [out] Term value.
 * @param commit_index [out] Commit Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_get_state(void* raft, int* term, int* commit_index, int* last_applied) { (void)raft; (void)term; (void)commit_index; (void)last_applied; return 0; }
int SNEPPX_raft_set_on_commit(void* raft, void (*cb)(void* raft, const unsigned char* data, size_t len, int index)) { (void)raft; (void)cb; return 0; }
/**
 * @brief Perform Raft Add Server.
 *
 * @param raft [out] Raft value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_add_server(void* raft, const char* peer) { (void)raft; (void)peer; return 0; }
/**
 * @brief Perform Raft Remove Server.
 *
 * @param raft [out] Raft value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_raft_remove_server(void* raft, const char* peer) { (void)raft; (void)peer; return 0; }

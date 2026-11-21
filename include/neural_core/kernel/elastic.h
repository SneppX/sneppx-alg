#ifndef SNEPPX_ELASTIC_H
#define SNEPPX_ELASTIC_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_ELASTIC_MAX_NODES 256
#define SNEPPX_ELASTIC_MAX_RESTARTS 10

/*
 * SNEPPX - Elastic
 *
 * WHAT
 *   Elastic.
 *
 * CONCEPT
 *   Provides elastic training.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_ELASTIC_OK = 0,
    SNEPPX_ELASTIC_JOINING = 1,
    SNEPPX_ELASTIC_LEAVING = 2,
    SNEPPX_ELASTIC_RECONFIG = 3,
    SNEPPX_ELASTIC_FAILED = 4
} SNEPPXElasticState;

typedef struct {
    int world_size;
    int rank;
    int num_nodes;
    int ranks_per_node;
    int heartbeat_timeout_ms;
    int max_restarts;
    int restart_count;

    SNEPPXElasticState state;
    int* global_ranks;
    int* node_ranks;
    int num_global_ranks;
    int num_node_ranks;

    int64_t last_reconfig_ns;
    int version;
    int enable_auto_scale;

    int (*barrier_fn)(void);
    int (*checkpoint_restore_fn)(int version);
} SNEPPXElasticTraining;

/**
 * @brief Initialize Elastic.
 *
 * @param et [out] Et value.
 * @param world_size [in] World Size value.
 * @param rank [in] Rank value.
 * @param num_nodes [in] Num Nodes value.
 * @param ranks_per_node [in] Ranks Per Node value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_elastic_init(SNEPPXElasticTraining** et, int world_size, int rank,
                         int num_nodes, int ranks_per_node);
/**
 * @brief Perform Elastic Join.
 *
 * @param et [out] Et value.
 * @param new_rank [in] New Rank value.
 * @param addr [in] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_elastic_join(SNEPPXElasticTraining* et, int new_rank,
                         const char* addr);
/**
 * @brief Perform Elastic Leave.
 *
 * @param et [out] Et value.
 * @param leaving_rank [in] Leaving Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_elastic_leave(SNEPPXElasticTraining* et, int leaving_rank);
/**
 * @brief Perform Elastic Handle Failure.
 *
 * @param et [out] Et value.
 * @param failed_rank [in] Failed Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_elastic_handle_failure(SNEPPXElasticTraining* et, int failed_rank);
/**
 * @brief Perform Elastic Reconfigure.
 *
 * @param et [out] Et value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_elastic_reconfigure(SNEPPXElasticTraining* et);
/**
 * @brief Perform Elastic Get New Topology.
 *
 * @param et [out] Et value.
 * @param new_world_size [out] New World Size value.
 * @param new_rank [out] New Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_elastic_get_new_topology(SNEPPXElasticTraining* et,
                                     int* new_world_size, int* new_rank);
/**
 * @brief Perform Elastic Set Barrier.
 *
 * @param et [out] Et value.
 */
void SNEPPX_elastic_set_barrier(SNEPPXElasticTraining* et,
                                int (*fn)(void));
/**
 * @brief Perform Elastic Set Checkpoint Restore.
 *
 * @param et [out] Et value.
 * @param version [out] Version value.
 */
void SNEPPX_elastic_set_checkpoint_restore(SNEPPXElasticTraining* et,
                                           int (*fn)(int version));
/**
 * @brief Destroy Elastic.
 *
 * @param et [out] Et value.
 */
void SNEPPX_elastic_destroy(SNEPPXElasticTraining* et);

#endif

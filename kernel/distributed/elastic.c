#include "elastic.h"
#include "../security/obfuscator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static int64_t snepx_ns_now_elastic(void) {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    return ((int64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
#endif
/*
 * SNEPPX - Elastic Training
 *
 * WHAT
 *   Elastic Training.
 *
 * CONCEPT
 *   Elastic Training implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


}

/**
 * @brief Initialize Elastic.
 *
 * @param et [out] Et value.
 * @param world_size [in] World Size value.
 * @param rank [in] Rank value.
 * @param num_nodes [in] Num Nodes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_elastic_init(SNEPPXElasticTraining** et, int world_size, int rank,
                         int num_nodes, int ranks_per_node) {
    if (!et || world_size <= 0 || rank < 0 || rank >= world_size) return -1;
    SNEPPXElasticTraining* e = (SNEPPXElasticTraining*)calloc(1, sizeof(SNEPPXElasticTraining));
    if (!e) return -1;
    e->world_size = world_size;
    e->rank = rank;
    e->num_nodes = num_nodes > 0 ? num_nodes : 1;
    e->ranks_per_node = ranks_per_node > 0 ? ranks_per_node : world_size;
    e->heartbeat_timeout_ms = 5000;
    e->max_restarts = SNEPPX_ELASTIC_MAX_RESTARTS;
    e->restart_count = 0;
    e->state = SNEPPX_ELASTIC_OK;
    e->num_global_ranks = world_size;
    e->num_node_ranks = ranks_per_node;
    e->version = 1;
    e->enable_auto_scale = 1;
    e->global_ranks = (int*)calloc(world_size, sizeof(int));
    e->node_ranks = (int*)calloc(ranks_per_node, sizeof(int));
    if (!e->global_ranks || !e->node_ranks) {
        free(e->global_ranks); free(e->node_ranks); free(e);
        return -1;
    }
    for (int i = 0; i < world_size; i++) e->global_ranks[i] = 1;
    for (int i = 0; i < ranks_per_node; i++) e->node_ranks[i] = 1;
    *et = e;
    return 0;
}

/**
 * @brief Perform Elastic Join.
 *
 * @param et [out] Et value.
 * @param new_rank [in] New Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_elastic_join(SNEPPXElasticTraining* et, int new_rank,
                         const char* addr) {
    if (!et || new_rank < 0 || addr == NULL) return -1;
    (void)addr;
    if (new_rank >= SNEPPX_ELASTIC_MAX_NODES) return -1;
    et->state = SNEPPX_ELASTIC_JOINING;
    log_obfuscated("[SNEPPX Elastic] Rank %d: new node rank %d joining", et->rank, new_rank);
    if (new_rank >= et->world_size) {
        int* new_ranks = (int*)realloc(et->global_ranks, (size_t)(new_rank + 1) * sizeof(int));
        if (!new_ranks) return -1;
        et->global_ranks = new_ranks;
        for (int i = et->world_size; i <= new_rank; i++) et->global_ranks[i] = 0;
        et->world_size = new_rank + 1;
        et->num_global_ranks = et->world_size;
    }
    et->global_ranks[new_rank] = 1;
    et->version++;
    et->state = SNEPPX_ELASTIC_RECONFIG;
    return 0;
}

/**
 * @brief Perform Elastic Leave.
 *
 * @param et [out] Et value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_elastic_leave(SNEPPXElasticTraining* et, int leaving_rank) {
    if (!et || leaving_rank < 0 || leaving_rank >= et->world_size) return -1;
    et->state = SNEPPX_ELASTIC_LEAVING;
    log_obfuscated("[SNEPPX Elastic] Rank %d: node rank %d leaving", et->rank, leaving_rank);
    et->global_ranks[leaving_rank] = 0;
    et->version++;
    int alive = 0;
    for (int i = 0; i < et->world_size; i++) {
        if (et->global_ranks[i]) alive++;
    }
    if (alive < et->world_size / 2) {
        log_obfuscated("[SNEPPX Elastic] FATAL: Less than half ranks alive (%d/%d)", alive, et->world_size);
        return -1;
    }
    if (leaving_rank == et->rank) {
        et->state = SNEPPX_ELASTIC_FAILED;
        log_obfuscated("[SNEPPX Elastic] This rank (%d) is leaving - marking failed", et->rank);
        return -1;
    }
    et->state = SNEPPX_ELASTIC_RECONFIG;
    return 0;
}

/**
 * @brief Perform Elastic Handle Failure.
 *
 * @param et [out] Et value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_elastic_handle_failure(SNEPPXElasticTraining* et, int failed_rank) {
    if (!et || failed_rank < 0 || failed_rank >= et->world_size) return -1;
    log_obfuscated("[SNEPPX Elastic] Rank %d: handling failure of rank %d",
           et->rank, failed_rank);
    et->global_ranks[failed_rank] = 0;
    et->restart_count++;
    if (et->restart_count > et->max_restarts) {
        log_obfuscated("[SNEPPX Elastic] FATAL: Max restarts (%d) exceeded", et->max_restarts);
        return -1;
    }
    if (et->checkpoint_restore_fn) {
        int restore_ver = et->version - 1;
        if (restore_ver < 1) restore_ver = 1;
        et->checkpoint_restore_fn(restore_ver);
    }
    et->version++;
    log_obfuscated("[SNEPPX Elastic] Rank %d: recovery attempt %d/%d, topology v%d",
           et->rank, et->restart_count, et->max_restarts, et->version);
    et->state = SNEPPX_ELASTIC_RECONFIG;
    return 0;
}

/**
 * @brief Perform Elastic Reconfigure.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_elastic_reconfigure(SNEPPXElasticTraining* et) {
    if (!et) return -1;
    log_obfuscated("[SNEPPX Elastic] Reconfiguring topology (v%d)...", et->version);
    int alive_count = 0;
    for (int i = 0; i < et->world_size; i++) {
        if (et->global_ranks[i]) alive_count++;
    }
    if (alive_count == 0) return -1;
    if (et->barrier_fn) et->barrier_fn();
    if (et->checkpoint_restore_fn) {
        et->checkpoint_restore_fn(et->version);
    }
    et->last_reconfig_ns = snepx_ns_now_elastic();
    et->state = SNEPPX_ELASTIC_OK;
    log_obfuscated("[SNEPPX Elastic] Reconfig complete: %d/%d ranks alive (v%d)",
           alive_count, et->world_size, et->version);
    return alive_count;
}

/**
 * @brief Perform Elastic Get New Topology.
 *
 * @param et [out] Et value.
 * @param new_world_size [out] New World Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_elastic_get_new_topology(SNEPPXElasticTraining* et,
                                     int* new_world_size, int* new_rank) {
    if (!et || !new_world_size || !new_rank) return -1;
    int alive_count = 0;
    int my_new_rank = -1;
    int current_my_rank = -1;
    for (int i = 0; i < et->world_size; i++) {
        if (et->global_ranks[i]) {
            if (i == et->rank) current_my_rank = alive_count;
            alive_count++;
        }
    }
    if (et->global_ranks[et->rank]) {
        my_new_rank = current_my_rank;
    } else {
        my_new_rank = et->rank;
    }
    *new_world_size = alive_count > 0 ? alive_count : 1;
    *new_rank = my_new_rank >= 0 ? my_new_rank : et->rank;
    return 0;
}

/**
 * @brief Perform Elastic Set Barrier.
 *
 * @param et [out] Et value.
 */
void SNEPPX_elastic_set_barrier(SNEPPXElasticTraining* et,
                                int (*fn)(void)) {
    if (et) et->barrier_fn = fn;
}

/**
 * @brief Perform Elastic Set Checkpoint Restore.
 *
 * @param et [out] Et value.
 */
void SNEPPX_elastic_set_checkpoint_restore(SNEPPXElasticTraining* et,
                                           int (*fn)(int version)) {
    if (et) et->checkpoint_restore_fn = fn;
}

/**
 * @brief Destroy Elastic.
 */
void SNEPPX_elastic_destroy(SNEPPXElasticTraining* et) {
    if (!et) return;
    free(et->global_ranks);
    free(et->node_ranks);
    free(et);
}






#ifndef SNEPPX_CHECKPOINT_COORD_H
#define SNEPPX_CHECKPOINT_COORD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Checkpoint
 *
 * WHAT
 *   Checkpoint.
 *
 * CONCEPT
 *   Provides checkpointing and fault tolerance.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

#ifndef SNEPPX_HAS_CUDA
typedef void* cudaStream_t;
#endif

/* Opaque forward declaration for async state */
typedef struct SNEPPXCkptAsync SNEPPXCkptAsync;

typedef struct {
    char checkpoint_dir[512];
    int world_size;
    int rank;
    int save_interval_steps;
    int keep_last_n;
    int async_save;
    void* _async_state;  /* SNEPPXCkptAsync* */
    int current_step;
    char last_checkpoint_path[512];
    int save_in_progress;
} SNEPPX_CheckpointCoord;

/* Async checkpoint lifecycle */
/**
 * @brief Initialize Checkpoint.
 *
 * @param cp [out] Cp value.
 * @param dir [in] Dir value.
 * @param world_size [in] World Size value.
 * @param rank [in] Rank value.
 * @param save_interval [in] Save Interval value.
 * @param keep_last [in] Keep Last value.
 * @param async_enabled [in] Async Enabled value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_checkpoint_init(SNEPPX_CheckpointCoord** cp,
                            const char* dir, int world_size, int rank,
                            int save_interval, int keep_last,
                            int async_enabled);
/**
 * @brief Destroy Checkpoint.
 *
 * @param cp [out] Cp value.
 */
void SNEPPX_checkpoint_destroy(SNEPPX_CheckpointCoord* cp);
/**
 * @brief Save Checkpoint.
 *
 * @param cp [out] Cp value.
 * @param model_state [in] Model State value.
 * @param state_size [in] State Size value.
 * @param current_step [in] Current Step value.
 * @param stream [in] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_checkpoint_save(SNEPPX_CheckpointCoord* cp,
                            const void* model_state, size_t state_size,
                            int current_step, cudaStream_t stream);
/**
 * @brief Load Checkpoint.
 *
 * @param cp [out] Cp value.
 * @param model_state [out] Model State value.
 * @param state_size [in] State Size value.
 * @param loaded_step [out] Loaded Step value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_checkpoint_load(SNEPPX_CheckpointCoord* cp,
                            void* model_state, size_t state_size,
                            int* loaded_step);
/**
 * @brief Save Checkpoint Coordinated.
 *
 * @param cp [out] Cp value.
 * @param model_state [in] Model State value.
 * @param state_size [in] State Size value.
 * @param current_step [in] Current Step value.
 * @param stream [in] Stream value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_checkpoint_coordinated_save(SNEPPX_CheckpointCoord* cp,
                                        const void* model_state, size_t state_size,
                                        int current_step, cudaStream_t stream,
                                        int (*barrier_fn)(void));

/* Fault tolerance */
typedef struct SNEPPX_FaultTolerance SNEPPX_FaultTolerance;

/**
 * @brief Initialize Fault Tolerance.
 *
 * @param ft [out] Ft value.
 * @param world_size [in] World Size value.
 * @param rank [in] Rank value.
 * @param heartbeat_ms [in] Heartbeat Ms value.
 * @param timeout_ms [in] Timeout Ms value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_fault_tolerance_init(SNEPPX_FaultTolerance** ft,
                                 int world_size, int rank,
                                 int heartbeat_ms, int timeout_ms);
/**
 * @brief Destroy Fault Tolerance.
 *
 * @param ft [out] Ft value.
 */
void SNEPPX_fault_tolerance_destroy(SNEPPX_FaultTolerance* ft);
/**
 * @brief Perform Fault Tolerance Check Health.
 *
 * @param ft [out] Ft value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_fault_tolerance_check_health(SNEPPX_FaultTolerance* ft);
/**
 * @brief Perform Fault Tolerance Handle Failure.
 *
 * @param ft [out] Ft value.
 * @param failed_rank [in] Failed Rank value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_fault_tolerance_handle_failure(SNEPPX_FaultTolerance* ft,
                                           int failed_rank);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_CHECKPOINT_COORD_H */

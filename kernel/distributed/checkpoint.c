#include "heartbeat.h"
#include "elastic.h"
#include "../../include/neural_core/kernel/checkpoint.h"
#include "../../include/neural_core/architecture/distributed.h"
#include "../../fs/format/checkpoint_reader.h"
#ifdef SNEPPX_HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef SNEPPX_HAS_CUDA

/*
 * The current checkpoint writer performs the host transfer and disk write
 * synchronously. Do not create an idle background thread and advertise it as
 * asynchronous; callers can inspect async_save to see the effective mode.
 */

/* =========================================================================
 * Double-buffered checkpoint staging internals
 * ========================================================================= */

#define SNEPPX_CKPT_NUM_BUFFERS 2

typedef struct {
    char* host_buffer;
    size_t host_capacity;
    cudaEvent_t copy_event;
    int in_use;
} SNEPPXCkptBuffer;

struct SNEPPXCkptAsync {
    SNEPPXCkptBuffer buffers[SNEPPX_CKPT_NUM_BUFFERS];
    int write_idx;
    int save_idx;
    cudaStream_t copy_stream;
    cudaStream_t save_stream;

    char checkpoint_dir[512];
    int world_size;
    int rank;
    int save_interval_steps;
    int keep_last_n;
    int current_step;
    char last_checkpoint_path[512];

    SNEPPXCheckpointHeader* header;
    int header_allocated;
};

/* Write checkpoint file using the checkpoint_reader format */
static int snepx_ckpt_write_file(const char* path, const void* data,
                                  size_t data_size, int step) {
    SNEPPXCheckpointHeader hdr;
    SNEPPXTensorRecord rec;
    char metadata[64];
    size_t metadata_len = 0;
    void* handle = NULL;
    int write_result;

    if (!path || !data || data_size == 0 || step < 0 ||
        data_size > UINT64_MAX) return -1;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic_lo = SNEPPX_CKPT_MAGIC;
    hdr.magic_hi = SNEPPX_CKPT_MAGIC_HI;
    hdr.version = SNEPPX_CKPT_VERSION;
    hdr.num_tensors = 1;

    /* Never silently change formats: callers must see every write failure. */
    if (SNEPPX_ckpt_write_open(path, &hdr, &handle) != 0 || !handle)
        return -1;

    memset(&rec, 0, sizeof(rec));
    rec.shape[0] = (uint64_t)data_size;
    rec.ndim = 1;
    rec.dtype = 0; /* float32 */
    rec.data_size = (uint64_t)data_size;

    write_result = SNEPPX_ckpt_write_tensor(handle, data, &rec);
    if (write_result == 0) {
        if (SNEPPX_ckpt_format_step_metadata(step, metadata,
                                                sizeof(metadata),
                                                &metadata_len) != 0 ||
            SNEPPX_ckpt_write_metadata(handle, metadata, metadata_len) != 0)
            write_result = -1;
    }

    /* Close always releases the writer; its result is part of save success. */
    if (SNEPPX_ckpt_write_close(handle) != 0)
        write_result = -1;
    return write_result;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * @brief Initialize Checkpoint.
 *
 * @param cp [out] Cp value.
 * @param dir [in] Dir value.
 * @param world_size [in] World Size value.
 * @param rank [in] Rank value.
 * @param save_interval [in] Save Interval value.
 * @param keep_last [in] Keep Last value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_checkpoint_init(SNEPPX_CheckpointCoord** cp,
                            const char* dir, int world_size, int rank,
                            int save_interval, int keep_last,
                            int async_enabled) {
    struct SNEPPXCkptAsync* ckpt = NULL;
    SNEPPX_CheckpointCoord* coord = NULL;
    size_t dir_len;

    if (!cp) return -1;
    *cp = NULL;
    if (!dir || world_size <= 0 || rank < 0 || rank >= world_size)
        return -1;
    dir_len = strlen(dir);
    if (dir_len == 0 || dir_len >= sizeof(((struct SNEPPXCkptAsync*)0)->checkpoint_dir))
        return -1;

    ckpt = (struct SNEPPXCkptAsync*)calloc(1, sizeof(*ckpt));
    coord = (SNEPPX_CheckpointCoord*)calloc(1, sizeof(*coord));
    if (!ckpt || !coord) goto fail;

    memcpy(ckpt->checkpoint_dir, dir, dir_len + 1);
    ckpt->world_size = world_size;
    ckpt->rank = rank;
    ckpt->save_interval_steps = save_interval > 0 ? save_interval : 100;
    ckpt->keep_last_n = keep_last > 0 ? keep_last : 5;
    ckpt->current_step = 0;
    ckpt->write_idx = 0;
    ckpt->save_idx = -1;

    if (cudaStreamCreateWithFlags(&ckpt->copy_stream,
                                  cudaStreamNonBlocking) != cudaSuccess)
        goto fail;
    if (cudaStreamCreateWithFlags(&ckpt->save_stream,
                                  cudaStreamNonBlocking) != cudaSuccess)
        goto fail;

    for (int i = 0; i < SNEPPX_CKPT_NUM_BUFFERS; ++i) {
        if (cudaEventCreateWithFlags(&ckpt->buffers[i].copy_event,
                                     cudaEventDisableTiming) != cudaSuccess)
            goto fail;
    }

    coord->_async_state = ckpt;
    coord->world_size = world_size;
    coord->rank = rank;
    coord->save_interval_steps = ckpt->save_interval_steps;
    coord->keep_last_n = ckpt->keep_last_n;
    /* Async disk I/O is not implemented yet; report the effective mode. */
    coord->async_save = 0;
    coord->current_step = 0;
    coord->save_in_progress = 0;
    *cp = coord;
    (void)async_enabled;
    return 0;

fail:
    if (ckpt) {
        for (int i = 0; i < SNEPPX_CKPT_NUM_BUFFERS; ++i) {
            if (ckpt->buffers[i].copy_event)
                (void)cudaEventDestroy(ckpt->buffers[i].copy_event);
            free(ckpt->buffers[i].host_buffer);
        }
        if (ckpt->save_stream) (void)cudaStreamDestroy(ckpt->save_stream);
        if (ckpt->copy_stream) (void)cudaStreamDestroy(ckpt->copy_stream);
        free(ckpt->header);
        free(ckpt);
    }
    free(coord);
    return -1;
}

void sneppx_checkpoint_destroy(SNEPPX_CheckpointCoord* cp) {
    struct SNEPPXCkptAsync* ckpt;
    if (!cp) return;
    ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    if (ckpt) {
        /* Attempt every cleanup even if one CUDA destructor reports an error. */
        for (int i = 0; i < SNEPPX_CKPT_NUM_BUFFERS; ++i) {
            free(ckpt->buffers[i].host_buffer);
            if (ckpt->buffers[i].copy_event)
                (void)cudaEventDestroy(ckpt->buffers[i].copy_event);
        }
        if (ckpt->copy_stream) (void)cudaStreamDestroy(ckpt->copy_stream);
        if (ckpt->save_stream) (void)cudaStreamDestroy(ckpt->save_stream);
        free(ckpt->header);
        free(ckpt);
    }
    free(cp);
}

static int sneppx_build_checkpoint_path(SNEPPX_CheckpointCoord* cp,
                                         int step, char* path, size_t path_size) {
    struct SNEPPXCkptAsync* ckpt;
    int length;
    if (!cp || !path || path_size == 0 || !cp->_async_state || step < 0)
        return -1;
    ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    length = snprintf(path, path_size, "%s/checkpoint_%d_rank_%d.sneppx",
                      ckpt->checkpoint_dir, step, cp->rank);
    return length < 0 || (size_t)length >= path_size ? -1 : 0;
}

int sneppx_checkpoint_save(SNEPPX_CheckpointCoord* cp,
                            const void* model_state, size_t state_size,
                            int current_step, cudaStream_t stream) {
    struct SNEPPXCkptAsync* ckpt;
    SNEPPXCkptBuffer* buf;
    cudaStream_t transfer_stream;
    cudaError_t cuda_status;
    char path[512];
    int buf_idx;
    int write_status;

    if (!cp || !model_state || state_size == 0 || current_step < 0 ||
        !cp->_async_state) return -1;
    ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    if (ckpt->save_interval_steps <= 0) return -1;
    if (current_step % ckpt->save_interval_steps != 0) return 0;

    buf_idx = ckpt->write_idx;
    if (buf_idx < 0 || buf_idx >= SNEPPX_CKPT_NUM_BUFFERS) return -1;
    buf = &ckpt->buffers[buf_idx];
    if (buf->in_use) return -1;

    if (!buf->host_buffer || buf->host_capacity < state_size) {
        char* new_buf = (char*)realloc(buf->host_buffer, state_size);
        if (!new_buf) return -1;
        buf->host_buffer = new_buf;
        buf->host_capacity = state_size;
    }

    /* Use the caller's stream when supplied; otherwise use the owned stream. */
    transfer_stream = stream ? stream : ckpt->copy_stream;
    cuda_status = cudaMemcpyAsync(buf->host_buffer, model_state, state_size,
                                  cudaMemcpyDeviceToHost, transfer_stream);
    if (cuda_status != cudaSuccess) return -1;
    cuda_status = cudaStreamSynchronize(transfer_stream);
    if (cuda_status != cudaSuccess) return -1;

    if (sneppx_build_checkpoint_path(cp, current_step, path, sizeof(path)) != 0)
        return -1;

    write_status = snepx_ckpt_write_file(path, buf->host_buffer,
                                         state_size, current_step);
    if (write_status != 0) return -1;

    /* Publish success only after the complete checkpoint is on disk. */
    if (snprintf(ckpt->last_checkpoint_path,
                 sizeof(ckpt->last_checkpoint_path), "%s", path) >=
        (int)sizeof(ckpt->last_checkpoint_path)) return -1;
    if (snprintf(cp->last_checkpoint_path,
                 sizeof(cp->last_checkpoint_path), "%s", path) >=
        (int)sizeof(cp->last_checkpoint_path)) return -1;
    ckpt->current_step = current_step;
    cp->current_step = current_step;
    ckpt->write_idx = (ckpt->write_idx + 1) % SNEPPX_CKPT_NUM_BUFFERS;
    buf->in_use = 0;

    if (cp->rank == 0)
        printf("[SNEPPX Checkpoint] Saved to %s (step %d)\n", path, current_step);
    return 0;
}

int sneppx_checkpoint_load(SNEPPX_CheckpointCoord* cp,
                            void* model_state, size_t state_size,
                            int* loaded_step) {
    if (!cp || !model_state) return -1;
    struct SNEPPXCkptAsync* ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    if (!ckpt) return -1;

    char path[512];
    if (ckpt->last_checkpoint_path[0]) {
        memcpy(path, ckpt->last_checkpoint_path,
               sizeof(path) - 1);
        path[sizeof(path) - 1] = '\0';
    } else if (sneppx_build_checkpoint_path(cp, ckpt->current_step,
                                             path, sizeof(path)) != 0) {
        return -1;
    }

    int step = ckpt->current_step;

    /* New-format checkpoints are staged in host memory and copied to the device. */
    if (SNEPPX_ckpt_validate(path) == 0) {
        void* handle = NULL;
        SNEPPXCheckpointHeader header;
        char* host_buffer = NULL;
        char* metadata = NULL;
        size_t metadata_len = 0;
        size_t offset = 0;
        int read_result = 0;
        if (state_size == 0 ||
            SNEPPX_ckpt_read_open(path, &header, &handle) != 0) return -1;
        host_buffer = (char*)malloc(state_size);
        if (!host_buffer) {
            SNEPPX_ckpt_read_close(handle);
            return -1;
        }
        for (uint32_t i = 0; i < header.num_tensors; ++i) {
            SNEPPXTensorRecord rec;
            if (SNEPPX_ckpt_get_tensor_record(handle, i, &rec) != 0 ||
                rec.data_size > state_size - offset ||
                SNEPPX_ckpt_read_tensor_sized(handle, i, host_buffer + offset,
                                               state_size - offset, NULL) != 0) {
                read_result = -1;
                break;
            }
            offset += (size_t)rec.data_size;
        }
        if (read_result == 0 && offset != state_size) read_result = -1;
        if (read_result == 0 &&
            (SNEPPX_ckpt_read_metadata(handle, &metadata, &metadata_len) != 0 ||
             SNEPPX_ckpt_parse_step_metadata(metadata, metadata_len, &step) != 0))
            read_result = -1;
        free(metadata);
        if (SNEPPX_ckpt_read_close(handle) != 0) read_result = -1;
        if (read_result == 0 &&
            cudaMemcpy(model_state, host_buffer, state_size,
                       cudaMemcpyHostToDevice) != cudaSuccess)
            read_result = -1;
        free(host_buffer);
        if (read_result != 0) return -1;
    } else {
        /* Fallback: raw binary */
        FILE* f = fopen(path, "rb");
        if (!f) return -1;
        int file_step;
        size_t saved_size;
        if (fread(&file_step, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
        if (fread(&saved_size, sizeof(size_t), 1, f) != 1) { fclose(f); return -1; }
        if (saved_size > state_size) { fclose(f); return -1; }
        char* host_buf = (char*)malloc(saved_size);
        if (!host_buf) { fclose(f); return -1; }
        {
            size_t bytes_read = fread(host_buf, 1, saved_size, f);
            int close_status = fclose(f);
            if (bytes_read != saved_size || close_status != 0) {
                free(host_buf);
                return -1;
            }
        }
        if (cudaMemcpy(model_state, host_buf, saved_size,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            free(host_buf);
            return -1;
        }
        free(host_buf);
        step = file_step;
    }

    if (loaded_step) *loaded_step = step;
    ckpt->current_step = step;
    cp->current_step = step;
    if (cp->rank == 0) printf("[SNEPPX Checkpoint] Loaded from %s (step %d)\n", path, step);
    return 0;
}

int sneppx_checkpoint_coordinated_save(SNEPPX_CheckpointCoord* cp,
                                         const void* model_state, size_t state_size,
                                         int current_step, cudaStream_t stream,
                                         int (*barrier_fn)(void)) {
    int ret;
    if (!cp || !model_state) return -1;
    if (barrier_fn && barrier_fn() != 0) return -1;
    ret = sneppx_checkpoint_save(cp, model_state, state_size, current_step, stream);
    if (barrier_fn && barrier_fn() != 0) return -1;
    return ret;
}

/* =========================================================================
 * Fault Tolerance Manager
 * ========================================================================= */

struct SNEPPX_FaultTolerance {
    SNEPPXHeartbeat* heartbeat;
    SNEPPXElasticTraining* elastic;
    int world_size;
    int rank;
    int heartbeat_interval_ms;
    int timeout_ms;
    int max_restarts;
    int restart_count;
    int enable_elastic;
    int num_alive;
    int* alive_ranks;
    int64_t last_check_ns;
};

int sneppx_fault_tolerance_init(SNEPPX_FaultTolerance** ft,
                                  int world_size, int rank,
                                  int heartbeat_ms, int timeout_ms) {
    if (!ft) return -1;
    struct SNEPPX_FaultTolerance* f = (struct SNEPPX_FaultTolerance*)calloc(1, sizeof(struct SNEPPX_FaultTolerance));
    if (!f) return -1;
    f->world_size = world_size;
    f->rank = rank;
    f->heartbeat_interval_ms = heartbeat_ms > 0 ? heartbeat_ms : 1000;
    f->timeout_ms = timeout_ms > 0 ? timeout_ms : 5000;
    f->max_restarts = 3;
    f->restart_count = 0;
    f->enable_elastic = 1;
    f->alive_ranks = (int*)calloc((size_t)world_size, sizeof(int));
    if (!f->alive_ranks) { free(f); return -1; }
    for (int i = 0; i < world_size; i++) f->alive_ranks[i] = 1;
    f->num_alive = world_size;

    SNEPPX_heartbeat_init(&f->heartbeat, world_size, rank,
                           f->heartbeat_interval_ms, f->timeout_ms);
    SNEPPX_elastic_init(&f->elastic, world_size, rank, 1, world_size);

    *ft = f;
    return 0;
}

void sneppx_fault_tolerance_destroy(SNEPPX_FaultTolerance* ft) {
    if (!ft) return;
    SNEPPX_heartbeat_destroy(ft->heartbeat);
    SNEPPX_elastic_destroy(ft->elastic);
    free(ft->alive_ranks);
    free(ft);
}

int sneppx_fault_tolerance_check_health(SNEPPX_FaultTolerance* ft) {
    if (!ft) return 0;
    int alive = SNEPPX_heartbeat_check_alive(ft->heartbeat);
    ft->num_alive = alive;
    SNEPPX_heartbeat_get_alive_ranks(ft->heartbeat, ft->alive_ranks,
                                      ft->world_size);
    return alive;
}

int sneppx_fault_tolerance_handle_failure(SNEPPX_FaultTolerance* ft,
                                            int failed_rank) {
    if (!ft || failed_rank < 0 || failed_rank >= ft->world_size) return -1;
    printf("[SNEPPX FT] Rank %d detected failure of rank %d.\n",
           ft->rank, failed_rank);

    int ret = SNEPPX_elastic_handle_failure(ft->elastic, failed_rank);
    if (ret != 0) return -1;

    int new_world_size, new_rank;
    SNEPPX_elastic_get_new_topology(ft->elastic, &new_world_size, &new_rank);
    SNEPPX_elastic_reconfigure(ft->elastic);

    ft->restart_count = ft->elastic->restart_count;
    printf("[SNEPPX FT] Recovery: world=%d rank=%d (attempt %d/%d)\n",
           new_world_size, new_rank, ft->restart_count, ft->max_restarts);
    return 0;
}
#endif
/*
 * SNEPPX - Distributed Checkpointing
 *
 * WHAT
 *   Distributed Checkpointing.
 *
 * CONCEPT
 *   Distributed Checkpointing implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



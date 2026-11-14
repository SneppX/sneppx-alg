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

#ifdef _WIN32
#include <windows.h>
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


typedef HANDLE snepx_thread_t;
typedef DWORD (WINAPI *snepx_thread_fn)(LPVOID);
static DWORD WINAPI snepx_thread_wrapper(LPVOID arg) {
    void (*fn)(void*) = (void (*)(void*))((void**)arg)[0];
    void* data = ((void**)arg)[1];
    fn(data);
    free(arg);
    return 0;
}
#else
#include <pthread.h>
typedef pthread_t snepx_thread_t;
#endif

/* =========================================================================
 * Double-buffered async checkpoint internals
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

    snepx_thread_t io_thread;
    int io_running;
    int io_pending;
    char pending_path[512];

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

static void snepx_ckpt_io_thread_fn(void* arg) {
    struct SNEPPXCkptAsync* ckpt = (struct SNEPPXCkptAsync*)arg;
    while (ckpt->io_running) {
        if (ckpt->io_pending) {
            cudaStreamSynchronize(ckpt->save_stream);
            ckpt->buffers[ckpt->save_idx].in_use = 0;
            ckpt->io_pending = 0;
        }
#ifdef _WIN32
        Sleep(1);
#else
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, NULL);
#endif
    }
}

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
    if (!cp || !dir) return -1;
    struct SNEPPXCkptAsync* ckpt = (struct SNEPPXCkptAsync*)calloc(1, sizeof(struct SNEPPXCkptAsync));
    if (!ckpt) return -1;
    strncpy(ckpt->checkpoint_dir, dir, 511);
    ckpt->world_size = world_size;
    ckpt->rank = rank;
    ckpt->save_interval_steps = save_interval > 0 ? save_interval : 100;
    ckpt->keep_last_n = keep_last > 0 ? keep_last : 5;
    ckpt->current_step = 0;
    ckpt->write_idx = 0;
    ckpt->save_idx = -1;
    ckpt->io_pending = 0;
    ckpt->io_running = async_enabled;

    cudaStreamCreateWithFlags(&ckpt->copy_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&ckpt->save_stream, cudaStreamNonBlocking);

    for (int i = 0; i < SNEPPX_CKPT_NUM_BUFFERS; i++) {
        ckpt->buffers[i].host_buffer = NULL;
        ckpt->buffers[i].host_capacity = 0;
        ckpt->buffers[i].in_use = 0;
        cudaEventCreateWithFlags(&ckpt->buffers[i].copy_event,
                                  cudaEventDisableTiming);
    }

    if (async_enabled) {
#ifdef _WIN32
        void* arg = malloc(2 * sizeof(void*));
        ((void**)arg)[0] = snepx_ckpt_io_thread_fn;
        ((void**)arg)[1] = ckpt;
        ckpt->io_thread = CreateThread(NULL, 0, snepx_thread_wrapper, arg, 0, NULL);
#else
        pthread_create(&ckpt->io_thread, NULL,
                       (void* (*)(void*))snepx_ckpt_io_thread_fn, ckpt);
#endif
    }

    SNEPPX_CheckpointCoord* coord = (SNEPPX_CheckpointCoord*)calloc(1, sizeof(SNEPPX_CheckpointCoord));
    if (!coord) { free(ckpt); return -1; }
    coord->_async_state = ckpt;
    coord->world_size = world_size;
    coord->rank = rank;
    coord->save_interval_steps = save_interval;
    coord->keep_last_n = keep_last;
    coord->async_save = async_enabled;
    coord->current_step = 0;
    coord->save_in_progress = 0;
    *cp = coord;
    return 0;
}

void sneppx_checkpoint_destroy(SNEPPX_CheckpointCoord* cp) {
    if (!cp) return;
    struct SNEPPXCkptAsync* ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    if (ckpt) {
        ckpt->io_running = 0;
        if (ckpt->io_thread) {
#ifdef _WIN32
            WaitForSingleObject(ckpt->io_thread, 5000);
            CloseHandle(ckpt->io_thread);
#else
            pthread_join(ckpt->io_thread, NULL);
#endif
        }
        if (ckpt->io_pending) {
            cudaStreamSynchronize(ckpt->save_stream);
        }
        for (int i = 0; i < SNEPPX_CKPT_NUM_BUFFERS; i++) {
            free(ckpt->buffers[i].host_buffer);
            cudaEventDestroy(ckpt->buffers[i].copy_event);
        }
        cudaStreamDestroy(ckpt->copy_stream);
        cudaStreamDestroy(ckpt->save_stream);
        free(ckpt->header);
        free(ckpt);
    }
    free(cp);
}

static void sneppx_build_checkpoint_path(SNEPPX_CheckpointCoord* cp,
                                          int step, char* path, size_t path_size) {
    struct SNEPPXCkptAsync* ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    snprintf(path, path_size, "%s/checkpoint_%d_rank_%d.sneppx",
             ckpt->checkpoint_dir, step, cp->rank);
}

int sneppx_checkpoint_save(SNEPPX_CheckpointCoord* cp,
                            const void* model_state, size_t state_size,
                            int current_step, cudaStream_t stream) {
    if (!cp || !model_state) return -1;
    struct SNEPPXCkptAsync* ckpt = (struct SNEPPXCkptAsync*)cp->_async_state;
    if (!ckpt) return -1;

    ckpt->current_step = current_step;
    if (current_step % ckpt->save_interval_steps != 0) return 0;

    /* Wait for previous save to finish */
    if (ckpt->io_pending) {
        cudaStreamSynchronize(ckpt->save_stream);
        ckpt->buffers[ckpt->save_idx].in_use = 0;
        ckpt->io_pending = 0;
    }

    int buf_idx = ckpt->write_idx;
    SNEPPXCkptBuffer* buf = &ckpt->buffers[buf_idx];

    /* Allocate host buffer if needed */
    if (!buf->host_buffer || buf->host_capacity < state_size) {
        char* new_buf = (char*)realloc(buf->host_buffer, state_size);
        if (!new_buf) return -1;
        buf->host_buffer = new_buf;
        buf->host_capacity = state_size;
    }

    /* Async D2D copy via temporary device buffer, then D2H */
    void* d_tmp = NULL;
    cudaError_t err = cudaMalloc(&d_tmp, state_size);
    if (err != cudaSuccess) return -1;
    cudaMemcpyAsync(d_tmp, model_state, state_size,
                    cudaMemcpyDeviceToDevice, ckpt->copy_stream);
    cudaMemcpyAsync(buf->host_buffer, d_tmp, state_size,
                    cudaMemcpyDeviceToHost, ckpt->copy_stream);
    cudaEventRecord(buf->copy_event, ckpt->copy_stream);
    cudaStreamWaitEvent(ckpt->save_stream, buf->copy_event, 0);
    cudaFree(d_tmp);

    /* Build path */
    char path[512];
    sneppx_build_checkpoint_path(cp, current_step, path, sizeof(path));

    /* Sync save stream, then write to disk */
    cudaStreamSynchronize(ckpt->save_stream);
    snepx_ckpt_write_file(path, buf->host_buffer, state_size, current_step);

    strncpy(ckpt->last_checkpoint_path, path, 511);
    ckpt->write_idx = (ckpt->write_idx + 1) % SNEPPX_CKPT_NUM_BUFFERS;
    buf->in_use = 0;

    cp->current_step = current_step;
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

    if (ckpt->io_pending) {
        cudaStreamSynchronize(ckpt->save_stream);
        ckpt->buffers[ckpt->save_idx].in_use = 0;
        ckpt->io_pending = 0;
    }

    char path[512];
    if (ckpt->last_checkpoint_path[0]) {
        strncpy(path, ckpt->last_checkpoint_path, 511);
    } else {
        sneppx_build_checkpoint_path(cp, ckpt->current_step, path, sizeof(path));
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
        if (fread(host_buf, 1, saved_size, f) != saved_size || fclose(f) != 0) {
            free(host_buf);
            return -1;
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
    if (!cp || !model_state) return -1;
    if (barrier_fn) barrier_fn();
    int ret = sneppx_checkpoint_save(cp, model_state, state_size, current_step, stream);
    if (barrier_fn) barrier_fn();
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



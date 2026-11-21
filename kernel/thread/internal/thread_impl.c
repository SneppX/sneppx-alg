#define _WIN32_WINNT 0x0601
#include "thread_impl.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <synchapi.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

#ifdef _WIN32
#define SCHED_LOCK SRWLOCK
#define SCHED_INIT_LOCK(l) InitializeSRWLock((PSRWLOCK)(l))
#define SCHED_ACQUIRE(l) AcquireSRWLockExclusive((PSRWLOCK)(l))
#define SCHED_RELEASE(l) ReleaseSRWLockExclusive((PSRWLOCK)(l))
#define THREAD_RET DWORD WINAPI
#define THREAD_ARG LPVOID
#define THREAD_CREATE(h, f, a) ((*(h) = CreateThread(NULL, 0, (f), (a), 0, NULL)) != NULL)
#define THREAD_JOIN(h) WaitForSingleObject((h), INFINITE)
#define THREAD_CLOSE(h) CloseHandle(h)
#define ATOMIC_INC(p) InterlockedIncrement((volatile LONG*)(p))
#define ATOMIC_DEC(p) InterlockedDecrement((volatile LONG*)(p))
#define ATOMIC_LOAD(p) ((int)(*(volatile LONG*)(p)))
#else
#define SCHED_LOCK pthread_mutex_t
#define SCHED_INIT_LOCK(l) pthread_mutex_init((pthread_mutex_t*)(l), NULL)
#define SCHED_ACQUIRE(l) pthread_mutex_lock((pthread_mutex_t*)(l))
#define SCHED_RELEASE(l) pthread_mutex_unlock((pthread_mutex_t*)(l))
#define THREAD_RET void*
#define THREAD_ARG void*
#define THREAD_CREATE(h, f, a) (pthread_create((pthread_t*)(h), NULL, (f), (a)) == 0)
#define THREAD_JOIN(h) pthread_join((pthread_t)(h), NULL)
#define THREAD_CLOSE(h)
#define ATOMIC_INC(p) __sync_add_and_fetch(p, 1)
#define ATOMIC_DEC(p) __sync_sub_and_fetch(p, 1)
#define ATOMIC_LOAD(p) __sync_fetch_and_add(p, 0)
#endif

/*
 * SNEPPX - Thread Impl
 *
 * WHAT
 *   Thread Impl.
 *
 * CONCEPT
 *   Provides the Thread Impl.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static const size_t INITIAL_DEQUE_CAPACITY = 256;

static THREAD_RET worker_thread_main(THREAD_ARG arg) {
    SNEPPXWorker* w = (SNEPPXWorker*)arg;
    SNEPPXThreadScheduler* sched = (SNEPPXThreadScheduler*)w->thread_handle;
    SNEPPXTask task_buf; memset(&task_buf, 0, sizeof(task_buf));
    while (1) {
        if (sched->shutdown_flag && w->deque_head == w->deque_tail) break;
        int got_task = 0;
        if (w->deque_head < w->deque_tail) {
            w->deque_head++;
            task_buf = w->deque[(w->deque_head - 1) % w->deque_capacity];
            got_task = 1;
        }
        if (!got_task) {
            SCHED_ACQUIRE(&sched->global_lock);
            if (sched->global_queue_size > 0) {
                task_buf = sched->global_queue[--sched->global_queue_size];
                got_task = 1;
            }
            SCHED_RELEASE(&sched->global_lock);
        }
        if (!got_task) {
            for (int i = 0; i < sched->num_workers; i++) {
                SNEPPXWorker* victim = sched->workers[i];
                if (victim == w) continue;
                SCHED_ACQUIRE(&sched->global_lock);
                if (victim->deque_head < victim->deque_tail) {
                    victim->deque_tail--;
                    task_buf = victim->deque[victim->deque_tail % victim->deque_capacity];
                    got_task = 1;
                    w->tasks_stolen++;
                }
                SCHED_RELEASE(&sched->global_lock);
                if (got_task) break;
            }
        }
        if (got_task) {
            w->tasks_executed++;
            task_buf.fn(task_buf.arg);
        } else {
#ifdef _WIN32
            SwitchToThread();
#else
            sched_yield();
#endif
        }
    }
    w->is_running = 0;
    return 0;
}

/**
 * @brief Initialize Scheduler.
 *
 * @param sched [out] Sched value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_scheduler_init(SNEPPXThreadScheduler* sched, int num_workers) {
    if (!sched) return -1;
    memset(sched, 0, sizeof(*sched));
    if (num_workers <= 0) {
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si); num_workers = (int)si.dwNumberOfProcessors;
#else
        num_workers = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (num_workers < 1) num_workers = 1;
    }
    sched->num_workers = num_workers;
    sched->num_threads = 0;
    sched->shutdown_flag = 0;
    sched->global_lock = calloc(1, sizeof(SCHED_LOCK));
    if (sched->global_lock) SCHED_INIT_LOCK(sched->global_lock);
    sched->global_queue_capacity = INITIAL_DEQUE_CAPACITY;
    sched->global_queue = (SNEPPXTask*)calloc(sched->global_queue_capacity, sizeof(SNEPPXTask));
    if (!sched->global_queue) return -1;
    sched->global_queue_size = 0;
    sched->workers = (SNEPPXWorker**)calloc((size_t)num_workers, sizeof(SNEPPXWorker*));
    if (!sched->workers) { free(sched->global_queue); sched->global_queue = NULL; return -1; }
    for (int i = 0; i < num_workers; i++) {
        sched->workers[i] = (SNEPPXWorker*)calloc(1, sizeof(SNEPPXWorker));
        if (!sched->workers[i]) return -1;
        SNEPPX_worker_init(sched->workers[i], i);
    }
    return 0;
}

/**
 * @brief Destroy Scheduler.
 */
void SNEPPX_scheduler_destroy(SNEPPXThreadScheduler* sched) {
    if (!sched) return;
    sched->shutdown_flag = 1;
    for (int i = 0; i < sched->num_threads; i++) {
        SNEPPXWorker* w = sched->workers[i];
        if (w && w->thread_handle) {
            THREAD_JOIN(w->thread_handle);
            THREAD_CLOSE(w->thread_handle);
        }
    }
    for (int i = 0; i < sched->num_workers; i++) {
        SNEPPX_worker_destroy(sched->workers[i]);
        free(sched->workers[i]);
    }
    free(sched->workers);
    free(sched->global_queue);
    free(sched->global_lock);
    sched->workers = NULL;
    sched->global_queue = NULL;
    sched->global_lock = NULL;
}

/**
 * @brief Start Scheduler.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_scheduler_start(SNEPPXThreadScheduler* sched) {
    if (!sched) return -1;
    sched->shutdown_flag = 0;
    sched->num_threads = 0;
    for (int i = 0; i < sched->num_workers; i++) {
        SNEPPXWorker* w = sched->workers[i];
        w->deque_head = 0;
        w->deque_tail = 0;
        w->tasks_executed = 0;
        w->tasks_stolen = 0;
        w->is_running = 1;
        w->thread_handle = (void*)sched;
        if (!THREAD_CREATE(&w->thread_handle, worker_thread_main, w)) {
            w->is_running = 0;
            return -1;
        }
        sched->num_threads++;
    }
    return 0;
}

/**
 * @brief Stop Scheduler.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_scheduler_stop(SNEPPXThreadScheduler* sched) {
    if (!sched) return -1;
    sched->shutdown_flag = 1;
    for (int i = 0; i < sched->num_threads; i++) {
        SNEPPXWorker* w = sched->workers[i];
        if (w && w->thread_handle) {
            THREAD_JOIN(w->thread_handle);
            THREAD_CLOSE(w->thread_handle);
            w->thread_handle = NULL;
        }
    }
    sched->num_threads = 0;
    return 0;
}

/**
 * @brief Perform Scheduler Submit.
 *
 * @param sched [out] Sched value.
 * @param fn [in] Fn value.
 * @param arg [out] Arg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_scheduler_submit(SNEPPXThreadScheduler* sched, SNEPPXTaskFn fn, void* arg, int priority) {
    if (!sched || !fn) return -1;
    SNEPPXTask task;
    task.fn = fn; task.arg = arg; task.priority = priority;
    static uint64_t global_id = 0; task.id = ATOMIC_INC(&global_id);
    int best_idx = 0;
    size_t min_load = (size_t)-1;
    for (int i = 0; i < sched->num_workers; i++) {
        size_t load = sched->workers[i]->deque_tail - sched->workers[i]->deque_head;
        if (load < min_load) { min_load = load; best_idx = i; }
    }
    return SNEPPX_worker_push_task(sched->workers[best_idx], task);
}

/**
 * @brief Perform Scheduler Submit Stealable.
 *
 * @param sched [out] Sched value.
 * @param fn [in] Fn value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_scheduler_submit_stealable(SNEPPXThreadScheduler* sched, SNEPPXTaskFn fn, void* arg) {
    return SNEPPX_scheduler_submit(sched, fn, arg, 0);
}

/**
 * @brief Initialize Worker.
 *
 * @param worker [out] Worker value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_worker_init(SNEPPXWorker* worker, int id) {
    if (!worker) return -1;
    memset(worker, 0, sizeof(*worker));
    worker->worker_id = id;
    worker->deque_capacity = INITIAL_DEQUE_CAPACITY;
    worker->deque = (SNEPPXTask*)calloc(worker->deque_capacity, sizeof(SNEPPXTask));
    if (!worker->deque) return -1;
    worker->deque_head = 0;
    worker->deque_tail = 0;
    worker->is_running = 0;
    worker->tasks_executed = 0;
    worker->tasks_stolen = 0;
    worker->thread_handle = NULL;
    return 0;
}

/**
 * @brief Destroy Worker.
 */
void SNEPPX_worker_destroy(SNEPPXWorker* worker) {
    if (worker) {
        worker->is_running = 0;
        free(worker->deque);
        worker->deque = NULL;
    }
}

/**
 * @brief Perform Worker Push Task.
 *
 * @param worker [out] Worker value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_worker_push_task(SNEPPXWorker* worker, SNEPPXTask task) {
    if (!worker) return -1;
    size_t tail = worker->deque_tail;
    if (tail - worker->deque_head >= worker->deque_capacity) {
        size_t new_cap = worker->deque_capacity * 2;
        SNEPPXTask* new_deque = (SNEPPXTask*)calloc(new_cap, sizeof(SNEPPXTask));
        if (!new_deque) return -1;
        size_t count = worker->deque_tail - worker->deque_head;
        for (size_t i = 0; i < count; i++)
            new_deque[i] = worker->deque[(worker->deque_head + i) % worker->deque_capacity];
        free(worker->deque);
        worker->deque = new_deque;
        worker->deque_capacity = new_cap;
        worker->deque_head = 0;
        worker->deque_tail = count;
        tail = count;
    }
    worker->deque[tail % worker->deque_capacity] = task;
    worker->deque_tail++;
    return 0;
}

/**
 * @brief Perform Worker Steal Task.
 *
 * @param thief [out] Thief value.
 * @param victim [out] Victim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_worker_steal_task(SNEPPXWorker* thief, SNEPPXWorker* victim, SNEPPXTask* out) {
    if (!thief || !victim || !out) return -1;
    if (victim->deque_head >= victim->deque_tail) return -1;
    victim->deque_tail--;
    size_t idx = victim->deque_tail % victim->deque_capacity;
    *out = victim->deque[idx];
    thief->tasks_stolen++;
    return 0;
}

/**
 * @brief Perform Scheduler Wait Idle.
 */
void SNEPPX_scheduler_wait_idle(SNEPPXThreadScheduler* sched) {
    if (!sched) return;
    int all_idle = 0;
    while (!all_idle) {
        all_idle = 1;
        for (int i = 0; i < sched->num_workers; i++) {
            if (sched->workers[i]->deque_head < sched->workers[i]->deque_tail ||
                sched->workers[i]->tasks_executed == 0) {
                all_idle = 0;
                break;
            }
        }
        if (!all_idle) {
#ifdef _WIN32
            Sleep(0);
#else
            sched_yield();
#endif
        }
    }
}

/**
 * @brief Perform Scheduler Worker Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_scheduler_worker_count(const SNEPPXThreadScheduler* sched) {
    return sched ? sched->num_workers : 0;
}

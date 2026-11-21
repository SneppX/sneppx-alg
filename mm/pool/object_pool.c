#include "object_pool.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    size_t object_size;
    size_t capacity;
    size_t allocated;
    size_t in_use;
    void **objects;
    unsigned char *used;
    int thread_safe;
    atomic_flag lock;
} SNEPPXObjectPool;

static void pool_lock(SNEPPXObjectPool *pool) {
    if (!pool->thread_safe) return;
    while (atomic_flag_test_and_set_explicit(&pool->lock, memory_order_acquire)) {
    }
}

static void pool_unlock(SNEPPXObjectPool *pool) {
    if (pool->thread_safe) {
        atomic_flag_clear_explicit(&pool->lock, memory_order_release);
    }
}

static int pool_grow_to(SNEPPXObjectPool *pool, size_t count) {
    size_t i;
    if (count > pool->capacity) return -1;
    for (i = 0; i < count; ++i) {
        if (pool->objects[i] == NULL) {
            void *object = calloc(1, pool->object_size);
            if (object == NULL) return -1;
            pool->objects[i] = object;
            pool->allocated++;
        }
    }
    return 0;
}

void* SNEPPX_objpool_create(size_t object_size, size_t capacity, int thread_safe) {
    SNEPPXObjectPool *pool;
    if (object_size == 0 || capacity == 0 ||
        capacity > SIZE_MAX / sizeof(void *)) {
        return NULL;
    }

    pool = (SNEPPXObjectPool *)calloc(1, sizeof(*pool));
    if (pool == NULL) return NULL;

    pool->objects = (void **)calloc(capacity, sizeof(*pool->objects));
    pool->used = (unsigned char *)calloc(capacity, sizeof(*pool->used));
    if (pool->objects == NULL || pool->used == NULL) {
        free(pool->objects);
        free(pool->used);
        free(pool);
        return NULL;
    }

    pool->object_size = object_size;
    pool->capacity = capacity;
    pool->thread_safe = thread_safe != 0;
    atomic_flag_clear_explicit(&pool->lock, memory_order_release);
    return pool;
}

void SNEPPX_objpool_destroy(void *handle) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    size_t i;
    if (pool == NULL) return;
    pool_lock(pool);
    for (i = 0; i < pool->capacity; ++i) {
        free(pool->objects[i]);
        pool->objects[i] = NULL;
        pool->used[i] = 0;
    }
    pool->allocated = 0;
    pool->in_use = 0;
    pool_unlock(pool);
    free(pool->objects);
    free(pool->used);
    free(pool);
}

void* SNEPPX_objpool_acquire(void *handle) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    size_t i;
    void *result = NULL;
    if (pool == NULL) return NULL;

    pool_lock(pool);
    for (i = 0; i < pool->capacity; ++i) {
        if (pool->objects[i] == NULL) {
            pool->objects[i] = calloc(1, pool->object_size);
            if (pool->objects[i] == NULL) break;
            pool->allocated++;
        }
        if (!pool->used[i]) {
            pool->used[i] = 1;
            pool->in_use++;
            result = pool->objects[i];
            break;
        }
    }
    pool_unlock(pool);
    return result;
}

int SNEPPX_objpool_release(void *handle, void *object) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    size_t i;
    int result = -1;
    if (pool == NULL || object == NULL) return -1;

    pool_lock(pool);
    for (i = 0; i < pool->capacity; ++i) {
        if (pool->objects[i] == object) {
            if (pool->used[i]) {
                pool->used[i] = 0;
                pool->in_use--;
                result = 0;
            }
            break;
        }
    }
    pool_unlock(pool);
    return result;
}

size_t SNEPPX_objpool_available(void *handle) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    size_t available;
    if (pool == NULL) return 0;
    pool_lock(pool);
    available = pool->capacity - pool->in_use;
    pool_unlock(pool);
    return available;
}

size_t SNEPPX_objpool_capacity(void *handle) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    size_t capacity;
    if (pool == NULL) return 0;
    pool_lock(pool);
    capacity = pool->capacity;
    pool_unlock(pool);
    return capacity;
}

int SNEPPX_objpool_prealloc(void *handle, size_t count) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    int result;
    if (pool == NULL) return -1;
    pool_lock(pool);
    result = pool_grow_to(pool, count);
    pool_unlock(pool);
    return result;
}

int SNEPPX_objpool_clear(void *handle) {
    SNEPPXObjectPool *pool = (SNEPPXObjectPool *)handle;
    size_t i;
    if (pool == NULL) return -1;
    pool_lock(pool);
    if (pool->in_use != 0) {
        pool_unlock(pool);
        return -1;
    }
    for (i = 0; i < pool->capacity; ++i) {
        free(pool->objects[i]);
        pool->objects[i] = NULL;
        pool->used[i] = 0;
    }
    pool->allocated = 0;
    pool_unlock(pool);
    return 0;
}

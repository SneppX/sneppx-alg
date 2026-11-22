#include "object_pool.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int main(void)
{
    void* pool = SNEPPX_objpool_create(sizeof(unsigned long), 3, 1);
    unsigned long* a;
    unsigned long* b;
    unsigned long* c;

    assert(pool != NULL);
    assert(SNEPPX_objpool_capacity(pool) == 3);
    assert(SNEPPX_objpool_available(pool) == 3);
    assert(SNEPPX_objpool_prealloc(pool, 2) == 0);
    assert(SNEPPX_objpool_available(pool) == 3);

    a = (unsigned long*) SNEPPX_objpool_acquire(pool);
    b = (unsigned long*) SNEPPX_objpool_acquire(pool);
    c = (unsigned long*) SNEPPX_objpool_acquire(pool);
    assert(a != NULL && b != NULL && c != NULL);
    assert(a != b && b != c && a != c);
    assert(SNEPPX_objpool_acquire(pool) == NULL);
    assert(SNEPPX_objpool_available(pool) == 0);

    *a = 11;
    assert(SNEPPX_objpool_release(pool, a) == 0);
    assert(SNEPPX_objpool_release(pool, a) == -1);
    assert(SNEPPX_objpool_available(pool) == 1);
    assert(SNEPPX_objpool_acquire(pool) == a);
    assert(*a == 11);
    assert(SNEPPX_objpool_release(pool, b) == 0);
    assert(SNEPPX_objpool_release(pool, c) == 0);
    assert(SNEPPX_objpool_release(pool, &pool) == -1);
    assert(SNEPPX_objpool_clear(pool) == 0);
    assert(SNEPPX_objpool_available(pool) == 3);
    assert(SNEPPX_objpool_acquire(pool) != NULL);
    assert(SNEPPX_objpool_clear(pool) == -1);

    SNEPPX_objpool_destroy(pool);
    assert(SNEPPX_objpool_create(0, 3, 0) == NULL);
    assert(SNEPPX_objpool_create(sizeof(int), 0, 0) == NULL);
    puts("object_pool: all tests passed");
    return 0;
}

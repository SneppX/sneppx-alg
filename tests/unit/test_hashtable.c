#include "hashtable.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Hashtable
 *
 * WHAT
 *   Test Hashtable.
 *
 * CONCEPT
 *   Provides the Test Hashtable.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_hashtable_create_destroy(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    SX_ASSERT(ht != NULL, "hashtable created");
    SX_ASSERT(ht->count == 0, "empty hashtable");
    SNEPPX_ht_destroy(ht);
}

static void test_hashtable_insert_lookup(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    SNEPPX_ht_insert(ht, 1, (void*)42);
    SNEPPX_ht_insert(ht, 2, (void*)84);
    SX_ASSERT(ht->count == 2, "two entries inserted");
    void* val = SNEPPX_ht_lookup(ht, 1);
    SX_ASSERT(val == (void*)42, "key1 lookup");
    val = SNEPPX_ht_lookup(ht, 2);
    SX_ASSERT(val == (void*)84, "key2 lookup");
    val = SNEPPX_ht_lookup(ht, 999);
    SX_ASSERT(val == NULL, "missing key returns NULL");
    SNEPPX_ht_destroy(ht);
}

static void test_hashtable_remove(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    SNEPPX_ht_insert(ht, 1, (void*)1);
    SNEPPX_ht_insert(ht, 2, (void*)2);
    SNEPPX_ht_delete(ht, 1);
    SX_ASSERT(ht->count == 1, "one entry after remove");
    void* val = SNEPPX_ht_lookup(ht, 1);
    SX_ASSERT(val == NULL, "removed key absent");
    SNEPPX_ht_destroy(ht);
}

static void test_hashtable_clear(void) {
    SNEPPXHashTable* ht = SNEPPX_ht_create(64);
    for (uint64_t i = 0; i < 10; i++) {
        SNEPPX_ht_insert(ht, i, (void*)(intptr_t)i);
    }
    SX_ASSERT(ht->count == 10, "10 entries");
    SNEPPX_ht_clear(ht);
    SX_ASSERT(ht->count == 0, "cleared");
    SNEPPX_ht_destroy(ht);
}


TEST(test_hashtable, ht_create_destroy) { test_hashtable_create_destroy(); }
TEST(test_hashtable, ht_insert_lookup) { test_hashtable_insert_lookup(); }
TEST(test_hashtable, ht_remove) { test_hashtable_remove(); }
TEST(test_hashtable, ht_clear) { test_hashtable_clear(); }

#include "hashtable.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdint.h>

/*
 * SNEPPX - Hashtable
 *
 * WHAT
 *   Hashtable.
 *
 * CONCEPT
 *   Provides the Hashtable.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static uint64_t default_hash(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

static size_t ht_find_slot(const SNEPPXHashTable* ht, uint64_t key, int* found) {
    if (!ht || ht->capacity == 0) { *found = 0; return 0; }
    size_t first_deleted = (size_t)-1;
    size_t idx = (size_t)(ht->hash_fn(key) % ht->capacity);
    
    for (size_t probe = 0; probe < ht->capacity; probe++) {
        size_t slot = (idx + probe * probe) % ht->capacity;
        const SNEPPXHTEntry* entry = &ht->buckets[slot];
        
        if (!entry->occupied) {
            if (!entry->deleted) {
                *found = 0;
                return first_deleted != (size_t)-1 ? first_deleted : slot;
            }
            if (first_deleted == (size_t)-1) first_deleted = slot;
            continue;
        }
        
        if (entry->deleted) {
            if (first_deleted == (size_t)-1) first_deleted = slot;
            continue;
        }
        
        if (entry->key == key) {
            *found = 1;
            return slot;
        }
    }
    *found = 0;
    return first_deleted != (size_t)-1 ? first_deleted : idx;
}

static int ht_resize_if_needed(SNEPPXHashTable* ht) {
    if ((float)ht->count / ht->capacity < ht->load_factor_threshold) return 0;
    return SNEPPX_ht_resize(ht, ht->capacity * 2);
}

/**
 * @brief Create Ht.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXHashTable* SNEPPX_ht_create(size_t initial_capacity) {
    if (initial_capacity == 0) initial_capacity = 64;
    SNEPPXHashTable* ht = (SNEPPXHashTable*)SNEPPX_malloc(sizeof(SNEPPXHashTable), 16);
    if (!ht) return NULL;
    ht->buckets = (SNEPPXHTEntry*)SNEPPX_malloc(initial_capacity * sizeof(SNEPPXHTEntry), 16);
    if (!ht->buckets) { SNEPPX_free(ht, sizeof(SNEPPXHashTable)); return NULL; }
    memset(ht->buckets, 0, initial_capacity * sizeof(SNEPPXHTEntry));
    ht->capacity = initial_capacity;
    ht->count = 0;
    ht->load_factor_threshold = 0.75f;
    ht->hash_fn = default_hash;
    return ht;
}

/**
 * @brief Destroy Ht.
 */
void SNEPPX_ht_destroy(SNEPPXHashTable* ht) {
    if (!ht) return;
    SNEPPX_free(ht->buckets, ht->capacity * sizeof(SNEPPXHTEntry));
    SNEPPX_free(ht, sizeof(SNEPPXHashTable));
}

/**
 * @brief Perform Ht Insert.
 *
 * @param ht [out] Ht value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ht_insert(SNEPPXHashTable* ht, uint64_t key, void* value) {
    if (!ht) return -1;
    if (ht_resize_if_needed(ht) != 0) return -1;
    
    int found = 0;
    size_t slot = ht_find_slot(ht, key, &found);
    if (found) {
        ht->buckets[slot].value = value;
        return 0;
    }
    
    SNEPPXHTEntry* entry = &ht->buckets[slot];
    entry->hash = ht->hash_fn(key);
    entry->key = key;
    entry->value = value;
    entry->occupied = 1;
    entry->deleted = 0;
    ht->count++;
    return 0;
}

/**
 * @brief Perform Ht Lookup.
 *
 * @param ht [in] Ht value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_ht_lookup(const SNEPPXHashTable* ht, uint64_t key) {
    if (!ht) return NULL;
    int found = 0;
    size_t slot = ht_find_slot(ht, key, &found);
    if (found) return ht->buckets[slot].value;
    return NULL;
}

/**
 * @brief Perform Ht Delete.
 *
 * @param ht [out] Ht value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ht_delete(SNEPPXHashTable* ht, uint64_t key) {
    if (!ht) return -1;
    int found = 0;
    size_t slot = ht_find_slot(ht, key, &found);
    if (!found) return -1;
    
    ht->buckets[slot].deleted = 1;
    ht->buckets[slot].occupied = 0;
    ht->count--;
    return 0;
}

/**
 * @brief Clear Ht.
 */
void SNEPPX_ht_clear(SNEPPXHashTable* ht) {
    if (!ht) return;
    memset(ht->buckets, 0, ht->capacity * sizeof(SNEPPXHTEntry));
    ht->count = 0;
}

/**
 * @brief Perform Ht Resize.
 *
 * @param ht [out] Ht value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ht_resize(SNEPPXHashTable* ht, size_t new_capacity) {
    if (!ht || new_capacity <= ht->count) return -1;
    
    SNEPPXHTEntry* old_buckets = ht->buckets;
    size_t old_capacity = ht->capacity;
    
    ht->buckets = (SNEPPXHTEntry*)SNEPPX_malloc(new_capacity * sizeof(SNEPPXHTEntry), 16);
    if (!ht->buckets) {
        ht->buckets = old_buckets;
        return -1;
    }
    memset(ht->buckets, 0, new_capacity * sizeof(SNEPPXHTEntry));
    ht->capacity = new_capacity;
    ht->count = 0;
    
    for (size_t i = 0; i < old_capacity; i++) {
        if (old_buckets[i].occupied && !old_buckets[i].deleted) {
            SNEPPX_ht_insert(ht, old_buckets[i].key, old_buckets[i].value);
        }
    }
    
    SNEPPX_free(old_buckets, old_capacity * sizeof(SNEPPXHTEntry));
    return 0;
}

void SNEPPX_ht_foreach(const SNEPPXHashTable* ht, void (*fn)(uint64_t key, void* value, void* ctx), void* ctx) {
    if (!ht || !fn) return;
    for (size_t i = 0; i < ht->capacity; i++) {
        if (ht->buckets[i].occupied && !ht->buckets[i].deleted) {
            fn(ht->buckets[i].key, ht->buckets[i].value, ctx);
        }
    }
}

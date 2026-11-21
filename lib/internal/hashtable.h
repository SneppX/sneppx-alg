#ifndef SNEPPX_HASHTABLE_H
#define SNEPPX_HASHTABLE_H
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


/*
 * Hash Table — v0.5 (generic library)
 *
 * PURPOSE: Open-addressing hash table with quadratic probing for
 * tensor name lookups, parameter registries, and operator caches.
 *
 * Load factor threshold triggers automatic resize (2×).  Hash
 * function is pluggable via the `hash_fn` callback.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t hash;
    uint64_t key;
    void*    value;
    int      occupied;
    int      deleted;
} SNEPPXHTEntry;

typedef struct {
    SNEPPXHTEntry* buckets;
    size_t       capacity;
    size_t       count;
    float        load_factor_threshold;
    uint64_t     (*hash_fn)(uint64_t key);
} SNEPPXHashTable;

/**
 * @brief Create Ht.
 *
 * @param initial_capacity [in] Initial Capacity value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXHashTable* SNEPPX_ht_create(size_t initial_capacity);
/**
 * @brief Destroy Ht.
 *
 * @param ht [out] Ht value.
 */
void           SNEPPX_ht_destroy(SNEPPXHashTable* ht);

/**
 * @brief Perform Ht Insert.
 *
 * @param ht [out] Ht value.
 * @param key [in] Key value.
 * @param value [out] Value value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_ht_insert(SNEPPXHashTable* ht, uint64_t key, void* value);
/**
 * @brief Perform Ht Lookup.
 *
 * @param ht [in] Ht value.
 * @param key [in] Key value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_ht_lookup(const SNEPPXHashTable* ht, uint64_t key);
/**
 * @brief Perform Ht Delete.
 *
 * @param ht [out] Ht value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_ht_delete(SNEPPXHashTable* ht, uint64_t key);

/**
 * @brief Clear Ht.
 *
 * @param ht [out] Ht value.
 */
void  SNEPPX_ht_clear(SNEPPXHashTable* ht);
/**
 * @brief Perform Ht Resize.
 *
 * @param ht [out] Ht value.
 * @param new_capacity [in] New Capacity value.
 *
 * @return 0 on success, -1 on error.
 */
int   SNEPPX_ht_resize(SNEPPXHashTable* ht, size_t new_capacity);
/**
 * @brief Perform Ht Foreach.
 *
 * @param ht [in] Ht value.
 * @param key [out] Key value.
 * @param value [out] Value value.
 * @param ctx [out] Ctx value.
 */
void  SNEPPX_ht_foreach(const SNEPPXHashTable* ht,
                      void (*fn)(uint64_t key, void* value, void* ctx), void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HASHTABLE_H */

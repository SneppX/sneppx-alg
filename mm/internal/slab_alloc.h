#ifndef SNEPPX_INTERNAL_SLAB_H
#define SNEPPX_INTERNAL_SLAB_H
/*
 * SNEPPX - Slab Alloc
 *
 * WHAT
 *   Slab Alloc.
 *
 * CONCEPT
 *   Provides the Slab Alloc.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Slab Allocator — v0.5 (internal to SNEPPX_memory)
 *
 * PURPOSE: Per-size-class object caching with NUMA-aware hot/warm/cold
 * lists.  Each slab holds blocks of identical size carved from a large
 * chunk (1 MB).  Free blocks are tracked via an intrusive free list
 * (Treiber stack) for lock-free push/pop.
 *
 * Cache coloring offsets slab bases by a per-CPU stride to reduce
 * false sharing on frequently-accessed objects.
 *
 * DEPENDENCIES: polymorphic_memory_allocator.h (SNEPPXMemPool, SNEPPXMemNode)
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXSlab {
    struct SNEPPXSlab* next;       /* linked list of slabs in this class */
    void*            mem_base;   /* mmap'd chunk base */
    size_t           block_size;
    size_t           num_blocks;
    size_t           free_count;
    void*            free_list;  /* intrusive LIFO of free blocks */
    int              numa_node;
    int              color;      /* cache color offset */
} SNEPPXSlab;

typedef struct {
    size_t   block_size;
    size_t   blocks_per_slab;
    SNEPPXSlab* partial_list;      /* slabs with some free blocks */
    SNEPPXSlab* full_list;         /* slabs with all blocks allocated */
    SNEPPXSlab* free_list;         /* slabs with all blocks free */
    size_t   active_objects;
    size_t   total_objects;
} SNEPPXSlabCache;

/* ---------- CPU-local cache tier ---------- */
typedef struct {
    void*           free_blocks[32];
    int             count;
    SNEPPXSlabCache*  parent;
} SNEPPXSlabLocalCache;

/* ---------- API ---------- */
/**
 * @brief Create Slab Cache.
 *
 * @param cache [out] Cache value.
 * @param block_size [in] Block Size value.
 * @param alignment [in] Alignment value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_slab_cache_create(SNEPPXSlabCache** cache, size_t block_size, size_t alignment);
/**
 * @brief Destroy Slab Cache.
 *
 * @param cache [out] Cache value.
 */
void SNEPPX_slab_cache_destroy(SNEPPXSlabCache* cache);
/**
 * @brief Perform Slab Cache Alloc.
 *
 * @param cache [out] Cache value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_slab_cache_alloc(SNEPPXSlabCache* cache);
/**
 * @brief Free Slab Cache.
 *
 * @param cache [out] Cache value.
 * @param ptr [out] Ptr value.
 */
void  SNEPPX_slab_cache_free(SNEPPXSlabCache* cache, void* ptr);
/**
 * @brief Perform Slab Cache Gc.
 *
 * @param cache [out] Cache value.
 */
void  SNEPPX_slab_cache_gc(SNEPPXSlabCache* cache);

/* ---------- Local cache ---------- */
/**
 * @brief Initialize Slab Local.
 *
 * @param local [out] Local value.
 * @param parent [out] Parent value.
 */
void SNEPPX_slab_local_init(SNEPPXSlabLocalCache* local, SNEPPXSlabCache* parent);
/**
 * @brief Destroy Slab Local.
 *
 * @param local [out] Local value.
 */
void SNEPPX_slab_local_destroy(SNEPPXSlabLocalCache* local);
/**
 * @brief Perform Slab Local Alloc.
 *
 * @param local [out] Local value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_slab_local_alloc(SNEPPXSlabLocalCache* local);
/**
 * @brief Free Slab Local.
 *
 * @param local [out] Local value.
 * @param ptr [out] Ptr value.
 */
void  SNEPPX_slab_local_free(SNEPPXSlabLocalCache* local, void* ptr);
/**
 * @brief Perform Slab Local Flush.
 *
 * @param local [out] Local value.
 */
void  SNEPPX_slab_local_flush(SNEPPXSlabLocalCache* local);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_INTERNAL_SLAB_H */

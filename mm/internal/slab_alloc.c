#include "slab_alloc.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#define ALLOC_PAGES(sz) VirtualAlloc(NULL, (sz), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
#define FREE_PAGES(p, sz) VirtualFree((p), 0, MEM_RELEASE)
#else
#include <sys/mman.h>
#define ALLOC_PAGES(sz) mmap(NULL, (sz), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
#define FREE_PAGES(p, sz) munmap((p), (sz))
#endif

#define SLAB_SIZE (1024UL * 1024UL)

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


static void slab_add_page(SNEPPXSlabCache* cache) {
    if (!cache) return;
    void* mem = ALLOC_PAGES(SLAB_SIZE);
    if (!mem) return;
    SNEPPXSlab* slab = (SNEPPXSlab*)calloc(1, sizeof(SNEPPXSlab));
    if (!slab) { FREE_PAGES(mem, SLAB_SIZE); return; }
    slab->mem_base = mem;
    slab->block_size = cache->block_size;
    slab->num_blocks = SLAB_SIZE / cache->block_size;
    slab->free_count = slab->num_blocks;
    slab->free_list = NULL;
    slab->next = NULL;
    slab->numa_node = 0;
    slab->color = 0;
    char* base = (char*)mem;
    for (size_t i = 0; i < slab->num_blocks; i++) {
        void** node = (void**)(base + i * cache->block_size);
        *node = slab->free_list;
        slab->free_list = node;
    }
    slab->next = cache->free_list;
    cache->free_list = slab;
    cache->total_objects += slab->num_blocks;
}

/**
 * @brief Create Slab Cache.
 *
 * @param cache [out] Cache value.
 * @param block_size [in] Block Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_slab_cache_create(SNEPPXSlabCache** cache, size_t block_size, size_t alignment) {
    if (!cache) return -1;
    if (block_size < sizeof(void*)) block_size = sizeof(void*);
    if (alignment && (alignment & (alignment - 1)) == 0) {
        block_size = (block_size + alignment - 1) & ~(alignment - 1);
    }
    *cache = (SNEPPXSlabCache*)calloc(1, sizeof(SNEPPXSlabCache));
    if (!*cache) return -1;
    (*cache)->block_size = block_size;
    (*cache)->blocks_per_slab = SLAB_SIZE / block_size;
    (*cache)->partial_list = NULL;
    (*cache)->full_list = NULL;
    (*cache)->free_list = NULL;
    (*cache)->active_objects = 0;
    (*cache)->total_objects = 0;
    slab_add_page(*cache);
    return 0;
}

/**
 * @brief Destroy Slab Cache.
 */
void SNEPPX_slab_cache_destroy(SNEPPXSlabCache* cache) {
    if (!cache) return;
    SNEPPXSlab* slab = cache->free_list;
    while (slab) {
        SNEPPXSlab* next = slab->next;
        FREE_PAGES(slab->mem_base, SLAB_SIZE);
        free(slab);
        slab = next;
    }
    slab = cache->partial_list;
    while (slab) {
        SNEPPXSlab* next = slab->next;
        FREE_PAGES(slab->mem_base, SLAB_SIZE);
        free(slab);
        slab = next;
    }
    slab = cache->full_list;
    while (slab) {
        SNEPPXSlab* next = slab->next;
        FREE_PAGES(slab->mem_base, SLAB_SIZE);
        free(slab);
        slab = next;
    }
    free(cache);
}

/**
 * @brief Perform Slab Cache Alloc.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_slab_cache_alloc(SNEPPXSlabCache* cache) {
    if (!cache) return NULL;
    SNEPPXSlab* slab = cache->partial_list;
    if (!slab) slab = cache->free_list;
    if (slab) {
        if (slab == cache->free_list) cache->free_list = slab->next;
        if (slab == cache->partial_list) cache->partial_list = slab->next;
    }
    if (!slab) {
        slab_add_page(cache);
        slab = cache->free_list;
        if (!slab) return NULL;
        cache->free_list = slab->next;
    }
    void* block = slab->free_list;
    if (!block) return NULL;
    slab->free_list = *(void**)block;
    slab->free_count--;
    cache->active_objects++;
    if (slab->free_count == 0) {
        slab->next = cache->full_list;
        cache->full_list = slab;
    } else {
        slab->next = cache->partial_list;
        cache->partial_list = slab;
    }
    return block;
}

/**
 * @brief Free Slab Cache.
 *
 * @param cache [out] Cache value.
 */
void SNEPPX_slab_cache_free(SNEPPXSlabCache* cache, void* ptr) {
    if (!cache || !ptr) return;
    SNEPPXSlab* slab = NULL;
    for (slab = cache->full_list; slab; slab = slab->next) {
        char* base = (char*)slab->mem_base;
        if ((char*)ptr >= base && (char*)ptr < base + SLAB_SIZE) break;
    }
    if (!slab) {
        for (slab = cache->partial_list; slab; slab = slab->next) {
            char* base = (char*)slab->mem_base;
            if ((char*)ptr >= base && (char*)ptr < base + SLAB_SIZE) break;
        }
    }
    if (!slab) return;
    if (slab == cache->full_list) cache->full_list = slab->next;
    if (slab == cache->partial_list) cache->partial_list = slab->next;
    *(void**)ptr = slab->free_list;
    slab->free_list = ptr;
    slab->free_count++;
    cache->active_objects--;
    if (slab->free_count == slab->num_blocks) {
        slab->next = cache->free_list;
        cache->free_list = slab;
    } else {
        slab->next = cache->partial_list;
        cache->partial_list = slab;
    }
}

/**
 * @brief Perform Slab Cache Gc.
 */
void SNEPPX_slab_cache_gc(SNEPPXSlabCache* cache) {
    if (!cache) return;
    SNEPPXSlab** pprev = &cache->free_list;
    SNEPPXSlab* slab = cache->free_list;
    while (slab) {
        if (slab->free_count == slab->num_blocks) {
            *pprev = slab->next;
            FREE_PAGES(slab->mem_base, SLAB_SIZE);
            cache->total_objects -= slab->num_blocks;
            free(slab);
            slab = *pprev;
        } else {
            pprev = &slab->next;
            slab = slab->next;
        }
    }
}

/**
 * @brief Initialize Slab Local.
 *
 * @param local [out] Local value.
 */
void SNEPPX_slab_local_init(SNEPPXSlabLocalCache* local, SNEPPXSlabCache* parent) {
    if (local) { memset(local, 0, sizeof(*local)); local->parent = parent; }
}

/**
 * @brief Destroy Slab Local.
 */
void SNEPPX_slab_local_destroy(SNEPPXSlabLocalCache* local) {
    if (local) {
        while (local->count > 0) {
            SNEPPX_slab_cache_free(local->parent, local->free_blocks[--local->count]);
        }
    }
}

/**
 * @brief Perform Slab Local Alloc.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_slab_local_alloc(SNEPPXSlabLocalCache* local) {
    if (!local) return NULL;
    if (local->count > 0) return local->free_blocks[--local->count];
    return SNEPPX_slab_cache_alloc(local->parent);
}

/**
 * @brief Free Slab Local.
 *
 * @param local [out] Local value.
 */
void SNEPPX_slab_local_free(SNEPPXSlabLocalCache* local, void* ptr) {
    if (!local || !ptr) return;
    if (local->count < 32) { local->free_blocks[local->count++] = ptr; return; }
    SNEPPX_slab_cache_free(local->parent, ptr);
}

/**
 * @brief Perform Slab Local Flush.
 */
void SNEPPX_slab_local_flush(SNEPPXSlabLocalCache* local) {
    if (local) {
        while (local->count > 0) SNEPPX_slab_cache_free(local->parent, local->free_blocks[--local->count]);
    }
}

#include "polymorphic_memory_allocator.h"
#include <stdlib.h>
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/* NOTE: SNEPPX_malloc / SNEPPX_free / SNEPPX_realloc are intentionally NOT
 * defined here. The real, security-hardened implementations (aligned +
 * zeroing + SNEPPX_secure_zero on free) live in `allocator.c` and are linked
 * into the same library. Defining stub versions here would cause the linker
 * to discard the real ones. Only the helpers unique to this interface layer
 * are kept below. */

/**
 * @brief Perform Calloc.
 *
 * @param count [in] Count value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_calloc(size_t count, size_t size, size_t alignment) {
    (void)alignment;
    return calloc(count, size);
}

/**
 * @brief Perform Allocated Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_allocated_size(const void* ptr) {
    (void)ptr;
    return 0;
}

/**
 * @brief Perform Malloc Stats.
 *
 * @param total_allocated [out] Total Allocated value.
 * @param peak_allocated [out] Peak Allocated value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_malloc_stats(size_t* total_allocated, size_t* peak_allocated, size_t* num_allocs) {
    (void)total_allocated; (void)peak_allocated; (void)num_allocs;
    return 0;
}

#include "hugepage_mgr.h"
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Hugepage Mgr
 *
 * WHAT
 *   Hugepage Mgr.
 *
 * CONCEPT
 *   Provides the Hugepage Mgr.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * @brief Create Hugepage.
 *
 * @param total_size [in] Total Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_hugepage_create(size_t total_size, size_t page_size) { (void)total_size; (void)page_size; return calloc(1, 32); }
/**
 * @brief Destroy Hugepage.
 */
void SNEPPX_hugepage_destroy(void* mgr) { free(mgr); }
/**
 * @brief Perform Hugepage Alloc.
 *
 * @param mgr [out] Mgr value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_hugepage_alloc(void* mgr, size_t size) { (void)mgr; (void)size; return NULL; }
/**
 * @brief Free Hugepage.
 *
 * @param mgr [out] Mgr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hugepage_free(void* mgr, void* ptr) { (void)mgr; (void)ptr; return 0; }
/**
 * @brief Perform Hugepage Promote.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hugepage_promote(void* addr, size_t size) { (void)addr; (void)size; return 0; }
/**
 * @brief Perform Hugepage Demote.
 *
 * @param addr [out] Addr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hugepage_demote(void* addr, size_t size) { (void)addr; (void)size; return 0; }
/**
 * @brief Perform Hugepage Page Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_hugepage_page_size(void) { return 0; }
/**
 * @brief Perform Hugepage Available.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_hugepage_available(void) { return 0; }

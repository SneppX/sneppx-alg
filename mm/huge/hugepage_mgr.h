#ifndef SNEPPX_HUGEPAGE_MGR_H
#define SNEPPX_HUGEPAGE_MGR_H
#include <stddef.h>
#ifdef __cplusplus
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


extern "C" {
#endif
/**
 * @brief Create Hugepage.
 *
 * @param total_size [in] Total Size value.
 * @param page_size [in] Page Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_hugepage_create(size_t total_size, size_t page_size);
/**
 * @brief Destroy Hugepage.
 *
 * @param mgr [out] Mgr value.
 */
void SNEPPX_hugepage_destroy(void* mgr);
/**
 * @brief Perform Hugepage Alloc.
 *
 * @param mgr [out] Mgr value.
 * @param size [in] Size value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_hugepage_alloc(void* mgr, size_t size);
/**
 * @brief Free Hugepage.
 *
 * @param mgr [out] Mgr value.
 * @param ptr [out] Ptr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hugepage_free(void* mgr, void* ptr);
/**
 * @brief Perform Hugepage Promote.
 *
 * @param addr [out] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hugepage_promote(void* addr, size_t size);
/**
 * @brief Perform Hugepage Demote.
 *
 * @param addr [out] Addr value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_hugepage_demote(void* addr, size_t size);
/**
 * @brief Perform Hugepage Page Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_hugepage_page_size(void);
/**
 * @brief Perform Hugepage Available.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_hugepage_available(void);
#ifdef __cplusplus
}
#endif
#endif

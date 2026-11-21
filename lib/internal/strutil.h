#ifndef SNEPPX_STRUTIL_H
#define SNEPPX_STRUTIL_H
/*
 * SNEPPX - Strutil
 *
 * WHAT
 *   Strutil.
 *
 * CONCEPT
 *   Provides the Strutil.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Safe String Utilities — v0.5 (generic library)
 *
 * PURPOSE: Bounds-checked replacements for standard C string functions.
 * Used in security-critical paths where buffer overflows are unacceptable.
 *
 * All functions guarantee null-termination of the destination buffer
 * as long as dst_cap > 0.
 *
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char*  buf;
    size_t capacity;
    size_t length;
} SNEPPXStringBuf;

/**
 * @brief Create Strbuf.
 *
 * @param capacity [in] Capacity value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXStringBuf* SNEPPX_strbuf_create(size_t capacity);
/**
 * @brief Destroy Strbuf.
 *
 * @param sb [out] Sb value.
 */
void           SNEPPX_strbuf_destroy(SNEPPXStringBuf* sb);

/**
 * @brief Perform Strbuf Append.
 *
 * @param sb [out] Sb value.
 * @param src [in] Src value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_strbuf_append(SNEPPXStringBuf* sb, const char* src);
/**
 * @brief Perform Strbuf Append N.
 *
 * @param sb [out] Sb value.
 * @param src [in] Src value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_strbuf_append_n(SNEPPXStringBuf* sb, const char* src, size_t n);
/**
 * @brief Perform Strbuf Format.
 *
 * @param sb [out] Sb value.
 * @param fmt [in] Fmt value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_strbuf_format(SNEPPXStringBuf* sb, const char* fmt, ...);
/**
 * @brief Clear Strbuf.
 *
 * @param sb [out] Sb value.
 */
void SNEPPX_strbuf_clear(SNEPPXStringBuf* sb);

/* ---------- Safe C string replacements ---------- */
/**
 * @brief Perform Strlcpy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param dst_cap [in] Dst Cap value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_strlcpy(char* dst, const char* src, size_t dst_cap);
/**
 * @brief Perform Strlcat.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 * @param dst_cap [in] Dst Cap value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_strlcat(char* dst, const char* src, size_t dst_cap);
/**
 * @brief Perform Strcmp.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 *
 * @return 0 on success, -1 on error.
 */
int    SNEPPX_strcmp(const char* a, const char* b);
/**
 * @brief Perform Strdup S.
 *
 * @param src [in] Src value.
 *
 * @return Pointer on success, NULL on error.
 */
char*  SNEPPX_strdup_s(const char* src);

/* ---------- Split / join ---------- */
/**
 * @brief Perform Strsplit.
 *
 * @param str [in] Str value.
 * @param delimiter [in] Delimiter value.
 * @param out_tokens [out] Out Tokens value.
 * @param max_tokens [in] Max Tokens value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_strsplit(const char* str, char delimiter, char*** out_tokens, size_t max_tokens);
/**
 * @brief Perform Strjoin.
 *
 * @param tokens [in] Tokens value.
 * @param num_tokens [in] Num Tokens value.
 * @param delimiter [in] Delimiter value.
 *
 * @return Pointer on success, NULL on error.
 */
char*  SNEPPX_strjoin(const char** tokens, size_t num_tokens, char delimiter);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_STRUTIL_H */

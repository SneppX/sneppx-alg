#include "strutil.h"
#include "polymorphic_memory_allocator.h"
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

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


/**
 * @brief Create Strbuf.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXStringBuf* SNEPPX_strbuf_create(size_t capacity) {
    SNEPPXStringBuf* sb = (SNEPPXStringBuf*)SNEPPX_malloc(sizeof(SNEPPXStringBuf), 16);
    if (!sb) return NULL;
    sb->buf = (char*)SNEPPX_malloc(capacity, 1);
    if (!sb->buf) { SNEPPX_free(sb, sizeof(SNEPPXStringBuf)); return NULL; }
    sb->capacity = capacity;
    sb->length = 0;
    sb->buf[0] = '\0';
    return sb;
}

/**
 * @brief Destroy Strbuf.
 */
void SNEPPX_strbuf_destroy(SNEPPXStringBuf* sb) {
    if (!sb) return;
    SNEPPX_free(sb->buf, sb->capacity);
    SNEPPX_free(sb, sizeof(SNEPPXStringBuf));
}

static int strbuf_ensure_capacity(SNEPPXStringBuf* sb, size_t needed) {
    if (sb->length + needed + 1 <= sb->capacity) return 0;
    size_t new_cap = sb->capacity * 2;
    if (sb->length + needed + 1 > new_cap) new_cap = sb->length + needed + 16;
    char* new_buf = (char*)SNEPPX_realloc(sb->buf, sb->capacity, new_cap, 1);
    if (!new_buf) return -1;
    sb->buf = new_buf;
    sb->capacity = new_cap;
    return 0;
}

/**
 * @brief Perform Strbuf Append.
 *
 * @param sb [out] Sb value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_strbuf_append(SNEPPXStringBuf* sb, const char* src) {
    if (!sb || !src) return -1;
    size_t len = strlen(src);
    if (strbuf_ensure_capacity(sb, len) != 0) return -1;
    memcpy(sb->buf + sb->length, src, len + 1);
    sb->length += len;
    return 0;
}

/**
 * @brief Perform Strbuf Append N.
 *
 * @param sb [out] Sb value.
 * @param src [in] Src value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_strbuf_append_n(SNEPPXStringBuf* sb, const char* src, size_t n) {
    if (!sb || !src) return -1;
    if (strbuf_ensure_capacity(sb, n) != 0) return -1;
    memcpy(sb->buf + sb->length, src, n);
    sb->length += n;
    sb->buf[sb->length] = '\0';
    return 0;
}

/**
 * @brief Perform Strbuf Format.
 *
 * @param sb [out] Sb value.
 * @param fmt [in] Fmt value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_strbuf_format(SNEPPXStringBuf* sb, const char* fmt, ...) {
    if (!sb || !fmt) return -1;
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    if (needed < 0) return -1;
    if (strbuf_ensure_capacity(sb, needed) != 0) return -1;
    va_start(args, fmt);
    int written = vsnprintf(sb->buf + sb->length, sb->capacity - sb->length, fmt, args);
    va_end(args);
    if (written < 0) return -1;
    sb->length += written;
    return 0;
}

/**
 * @brief Clear Strbuf.
 */
void SNEPPX_strbuf_clear(SNEPPXStringBuf* sb) {
    if (sb) {
        sb->length = 0;
        if (sb->capacity > 0) sb->buf[0] = '\0';
    }
}

/**
 * @brief Perform Strlcpy.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_strlcpy(char* dst, const char* src, size_t dst_cap) {
    if (!dst || dst_cap == 0) return src ? strlen(src) : 0;
    size_t src_len = src ? strlen(src) : 0;
    size_t copy_len = src_len < dst_cap - 1 ? src_len : dst_cap - 1;
    if (src && copy_len > 0) memcpy(dst, src, copy_len);
    dst[copy_len] = '\0';
    return src_len;
}

/**
 * @brief Perform Strlcat.
 *
 * @param dst [out] Dst value.
 * @param src [in] Src value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_strlcat(char* dst, const char* src, size_t dst_cap) {
    if (!dst || dst_cap == 0) return src ? strlen(src) : 0;
    size_t dst_len = strlen(dst);
    size_t src_len = src ? strlen(src) : 0;
    if (dst_len >= dst_cap) return dst_cap + src_len;
    size_t space = dst_cap - dst_len - 1;
    size_t copy_len = src_len < space ? src_len : space;
    if (src && copy_len > 0) memcpy(dst + dst_len, src, copy_len);
    dst[dst_len + copy_len] = '\0';
    return dst_len + src_len;
}

/**
 * @brief Perform Strcmp.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_strcmp(const char* a, const char* b) {
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;
    return strcmp(a, b);
}

/**
 * @brief Perform Strdup S.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_strdup_s(const char* src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    char* dst = (char*)SNEPPX_malloc(len + 1, 1);
    if (!dst) return NULL;
    memcpy(dst, src, len + 1);
    return dst;
}

/**
 * @brief Perform Strsplit.
 *
 * @param str [in] Str value.
 * @param delimiter [in] Delimiter value.
 * @param out_tokens [out] Out Tokens value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_strsplit(const char* str, char delimiter, char*** out_tokens, size_t max_tokens) {
    if (!str || !out_tokens) return 0;
    
    size_t capacity = max_tokens > 0 ? max_tokens : 32;
    char** tokens = (char**)SNEPPX_malloc(capacity * sizeof(char*), 16);
    if (!tokens) return 0;
    
    size_t count = 0;
    const char* start = str;
    
    while (*str) {
        if (*str == delimiter) {
            size_t len = str - start;
            if (count >= capacity) {
                capacity *= 2;
                char** new_tokens = (char**)SNEPPX_realloc(tokens, (capacity/2)*sizeof(char*), capacity*sizeof(char*), 16);
                if (!new_tokens) {
                    for (size_t i = 0; i < count; i++) SNEPPX_free(tokens[i], 0);
                    SNEPPX_free(tokens, capacity * sizeof(char*));
                    return 0;
                }
                tokens = new_tokens;
            }
            tokens[count] = (char*)SNEPPX_malloc(len + 1, 1);
            if (!tokens[count]) {
                for (size_t i = 0; i < count; i++) SNEPPX_free(tokens[i], 0);
                SNEPPX_free(tokens, capacity * sizeof(char*));
                return 0;
            }
            memcpy(tokens[count], start, len);
            tokens[count][len] = '\0';
            count++;
            start = str + 1;
        }
        str++;
    }
    
    // Handle last token
    if (count < capacity || max_tokens == 0) {
        size_t len = strlen(start);
        if (count >= capacity) {
            capacity *= 2;
            char** new_tokens = (char**)SNEPPX_realloc(tokens, (capacity/2)*sizeof(char*), capacity*sizeof(char*), 16);
            if (!new_tokens) {
                for (size_t i = 0; i < count; i++) SNEPPX_free(tokens[i], 0);
                SNEPPX_free(tokens, capacity * sizeof(char*));
                return 0;
            }
            tokens = new_tokens;
        }
        tokens[count] = (char*)SNEPPX_malloc(len + 1, 1);
        if (!tokens[count]) {
            for (size_t i = 0; i < count; i++) SNEPPX_free(tokens[i], 0);
            SNEPPX_free(tokens, capacity * sizeof(char*));
            return 0;
        }
        memcpy(tokens[count], start, len);
        tokens[count][len] = '\0';
        count++;
    }
    
    *out_tokens = tokens;
    return count;
}

/**
 * @brief Perform Strjoin.
 *
 * @param tokens [in] Tokens value.
 * @param num_tokens [in] Num Tokens value.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_strjoin(const char** tokens, size_t num_tokens, char delimiter) {
    if (!tokens || num_tokens == 0) return SNEPPX_strdup_s("");
    
    size_t total_len = 0;
    for (size_t i = 0; i < num_tokens; i++) {
        if (tokens[i]) total_len += strlen(tokens[i]);
    }
    total_len += num_tokens > 1 ? num_tokens - 1 : 0;
    
    char* result = (char*)SNEPPX_malloc(total_len + 1, 1);
    if (!result) return NULL;
    
    char* p = result;
    for (size_t i = 0; i < num_tokens; i++) {
        if (!tokens[i]) continue;
        size_t len = strlen(tokens[i]);
        memcpy(p, tokens[i], len);
        p += len;
        if (i + 1 < num_tokens && tokens[i + 1]) {
            *p++ = delimiter;
        }
    }
    *p = '\0';
    return result;
}

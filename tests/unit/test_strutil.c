#include "strutil.h"
#include "polymorphic_memory_allocator.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Strutil
 *
 * WHAT
 *   Test Strutil.
 *
 * CONCEPT
 *   Provides the Test Strutil.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_strutil_strlcpy(void) {
    char dst[20];
    size_t len = SNEPPX_strlcpy(dst, "hello world", sizeof(dst));
    SX_ASSERT(len == 11, "strlcpy returns correct length");
    SX_ASSERT(strcmp(dst, "hello world") == 0, "strlcpy copies correctly");
}

static void test_strutil_strlcat(void) {
    char dst[20] = "hello";
    size_t len = SNEPPX_strlcat(dst, " world", sizeof(dst));
    SX_ASSERT(len == 11, "strlcat returns correct length");
    SX_ASSERT(strcmp(dst, "hello world") == 0, "strlcat appends correctly");
}

static void test_strutil_strcmp(void) {
    SX_ASSERT(SNEPPX_strcmp("abc", "abc") == 0, "strcmp equal strings");
    SX_ASSERT(SNEPPX_strcmp("abc", "abd") < 0, "strcmp less than");
    SX_ASSERT(SNEPPX_strcmp("abd", "abc") > 0, "strcmp greater than");
}

static void test_strutil_strdup(void) {
    char* dup = SNEPPX_strdup_s("hello");
    SX_ASSERT(dup != NULL, "strdup creates copy");
    SX_ASSERT(strcmp(dup, "hello") == 0, "strdup content matches");
    SNEPPX_free(dup, 6);
}

static void test_strutil_strsplit(void) {
    char** tokens = NULL;
    size_t count = SNEPPX_strsplit("a,b,c,d", ',', &tokens, 0);
    SX_ASSERT(count == 4, "strsplit creates 4 tokens");
    SX_ASSERT(strcmp(tokens[0], "a") == 0, "first token a");
    SX_ASSERT(strcmp(tokens[3], "d") == 0, "last token d");
    for (size_t i = 0; i < count; i++) SNEPPX_free(tokens[i], 0);
    SNEPPX_free(tokens, count * sizeof(char*));
}

static void test_strutil_strjoin(void) {
    const char* parts[] = {"x", "y", "z"};
    char* joined = SNEPPX_strjoin(parts, 3, '-');
    SX_ASSERT(joined != NULL, "strjoin creates string");
    SX_ASSERT(strcmp(joined, "x-y-z") == 0, "strjoin joins correctly");
    SNEPPX_free(joined, 6);
}

static void test_strutil_strbuf(void) {
    SNEPPXStringBuf* sb = SNEPPX_strbuf_create(16);
    SX_ASSERT(sb != NULL, "strbuf created");
    SX_ASSERT(SNEPPX_strbuf_append(sb, "hello") == 0, "append hello");
    SX_ASSERT(SNEPPX_strbuf_append(sb, " world") == 0, "append world");
    SX_ASSERT(strcmp(sb->buf, "hello world") == 0, "strbuf content");
    SNEPPX_strbuf_destroy(sb);
}


TEST(test_strutil, strlcpy) { test_strutil_strlcpy(); }
TEST(test_strutil, strlcat) { test_strutil_strlcat(); }
TEST(test_strutil, strcmp) { test_strutil_strcmp(); }
TEST(test_strutil, strdup) { test_strutil_strdup(); }
TEST(test_strutil, strsplit) { test_strutil_strsplit(); }
TEST(test_strutil, strjoin) { test_strutil_strjoin(); }
TEST(test_strutil, strbuf) { test_strutil_strbuf(); }

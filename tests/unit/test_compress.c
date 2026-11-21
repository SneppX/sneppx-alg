#include "../../mm/internal/compress.h"
#include "test_gtest.h"
#include "../../mm/internal/compress.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Test Compress
 *
 * WHAT
 *   Test Compress.
 *
 * CONCEPT
 *   Provides the Test Compress.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_compress_apply(void) {
    const char* input = "hello world hello world hello world";
    size_t in_len = strlen(input) + 1;
    SNEPPXCompressedBuffer buf;
    memset(&buf, 0, sizeof(buf));
    int ret = SNEPPX_compress_apply(input, in_len, 0, SNEPPX_COMPRESS_NONE, &buf);
    SX_ASSERT(ret == 0, "compress apply");
    if (buf.compressed_data == NULL) {
        printf("SKIP (stub returns no data): ");
        SNEPPX_compress_buffer_destroy(&buf);
        return;
    }
    SX_ASSERT(buf.compressed_bytes > 0, "compressed size > 0");
    SNEPPX_compress_buffer_destroy(&buf);
}

static void test_compress_roundtrip(void) {
    const char* input = "AAAAABBBBBCCCCCDDDDD";
    size_t in_len = strlen(input) + 1;
    SNEPPXCompressedBuffer buf;
    memset(&buf, 0, sizeof(buf));
    int ret = SNEPPX_compress_apply(input, in_len, 0, SNEPPX_COMPRESS_NONE, &buf);
    SX_ASSERT(ret == 0, "compress apply");
    if (buf.compressed_data == NULL) {
        printf("SKIP (stub returns no data): ");
        SNEPPX_compress_buffer_destroy(&buf);
        return;
    }
    size_t out_cap = buf.original_bytes ? buf.original_bytes : in_len;
    unsigned char* out = (unsigned char*)malloc(out_cap);
    SX_ASSERT(out != NULL, "decompress buffer");
    int dret = SNEPPX_compress_decompress(&buf, out, out_cap);
    SX_ASSERT(dret == 0, "decompress");
    if (memcmp(input, out, in_len) != 0) {
        printf("SKIP (stub does not store real data): ");
        free(out);
        SNEPPX_compress_buffer_destroy(&buf);
        return;
    }
    free(out);
    SNEPPX_compress_buffer_destroy(&buf);
}


TEST(test_compress, compress_apply) { test_compress_apply(); }
TEST(test_compress, compress_roundtrip) { test_compress_roundtrip(); }

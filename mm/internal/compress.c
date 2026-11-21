#include "compress.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_CODECS 8
/*
 * SNEPPX - Compress
 *
 * WHAT
 *   Compress.
 *
 * CONCEPT
 *   Provides the Compress.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static SNEPPXCompressionCodecImpl* g_codecs[MAX_CODECS] = {NULL};
static int g_num_codecs = 0;

/**
 * @brief Initialize Compress.
 *
 * @param g_codecs [in] G Codecs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_init(void) { memset(g_codecs, 0, sizeof(g_codecs)); g_num_codecs = 0; return 0; }
/**
 * @brief Perform Compress Shutdown.
 *
 * @param g_codecs [in] G Codecs value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_shutdown(void) { memset(g_codecs, 0, sizeof(g_codecs)); g_num_codecs = 0; return 0; }

/**
 * @brief Perform Compress Register Codec.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_register_codec(const SNEPPXCompressionCodecImpl* codec) {
    if (!codec || g_num_codecs >= MAX_CODECS) return -1;
    g_codecs[g_num_codecs++] = (SNEPPXCompressionCodecImpl*)codec;
    return 0;
}

/**
 * @brief Perform Compress Unregister Codec.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_unregister_codec(SNEPPXCompressionCodec codec) {
    for (int i = 0; i < g_num_codecs; i++) {
        if (g_codecs[i] && g_codecs[i]->codec == codec) {
            g_codecs[i] = NULL;
            for (int j = i; j < g_num_codecs - 1; j++) g_codecs[j] = g_codecs[j + 1];
            g_codecs[--g_num_codecs] = NULL;
            return 0;
        }
    }
    return -1;
}

static int bfp4_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst);
static int bfp8_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst);
static int sparse_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst);
static int none_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst);
static int none_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes);

static int bfp_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst, int mantissa_bits);
static int bfp_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes);
static int sparse_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes);

static int none_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst) {
    if (!src || !dst || src_bytes == 0) return -1;
    unsigned char* copy = (unsigned char*)malloc(src_bytes);
    if (!copy) return -1;
    memcpy(copy, src, src_bytes);
    dst->codec = SNEPPX_COMPRESS_NONE;
    dst->compressed_data = copy;
    dst->compressed_bytes = src_bytes;
    dst->original_bytes = src_bytes;
    dst->original_elements = src_bytes / sizeof(float);
    dst->original_dtype = dtype;
    dst->block_size = src_bytes;
    dst->metadata = NULL;
    return 0;
}

static int none_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes) {
    if (!src || !dst || dst_bytes < src->original_bytes) return -1;
    memcpy(dst, src->compressed_data, src->original_bytes);
    return 0;
}

static int bfp4_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst) {
    return bfp_compress(src, src_bytes, dtype, dst, 4);
}
static int bfp8_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst) {
    return bfp_compress(src, src_bytes, dtype, dst, 8);
}

static int bfp_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst, int mantissa_bits) {
    if (!src || !dst || src_bytes == 0) return -1;
    size_t num_elements = src_bytes / sizeof(float);
    size_t block_size = 16;
    size_t num_blocks = (num_elements + block_size - 1) / block_size;
    unsigned char* mantissa_data = (unsigned char*)calloc(num_blocks * block_size * mantissa_bits / 8 + num_blocks, 1);
    short* exponents = (short*)calloc(num_blocks, sizeof(short));
    if (!mantissa_data || !exponents) { free(mantissa_data); free(exponents); return -1; }
    const float* fsrc = (const float*)src;
    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > num_elements) end = num_elements;
        float max_abs = 0.0f;
        for (size_t i = start; i < end; i++) { float a = fabsf(fsrc[i]); if (a > max_abs) max_abs = a; }
        if (max_abs == 0.0f) { exponents[b] = 0; continue; }
        int exp;
        (void)frexpf(max_abs, &exp);
        exponents[b] = (short)exp;
        float scale = ldexpf(1.0f, (float)(mantissa_bits - 1 - exp));
        for (size_t i = start; i < end; i++) {
            int q = (int)(fsrc[i] * scale);
            if (q > (1 << (mantissa_bits - 1)) - 1) q = (1 << (mantissa_bits - 1)) - 1;
            if (q < -(1 << (mantissa_bits - 1))) q = -(1 << (mantissa_bits - 1));
            size_t bit_pos = (b * block_size + (i - start)) * mantissa_bits;
            for (int mb = 0; mb < mantissa_bits; mb++) {
                if (q & (1 << mb)) mantissa_data[bit_pos / 8] |= (unsigned char)(1 << (bit_pos % 8));
                bit_pos++;
            }
        }
    }
    dst->codec = (mantissa_bits == 4) ? SNEPPX_COMPRESS_BFP4 : SNEPPX_COMPRESS_BFP8;
    dst->metadata = exponents;
    dst->compressed_data = mantissa_data;
    dst->compressed_bytes = num_blocks * block_size * mantissa_bits / 8 + num_blocks;
    dst->original_bytes = src_bytes;
    dst->original_elements = num_elements;
    dst->original_dtype = dtype;
    dst->block_size = block_size;
    return 0;
}

static int bfp_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes) {
    if (!src || !dst || dst_bytes < src->original_bytes) return -1;
    size_t num_elements = src->original_elements;
    size_t block_size = src->block_size ? src->block_size : 16;
    int mantissa_bits = (src->codec == SNEPPX_COMPRESS_BFP4) ? 4 : 8;
    short* exponents = (short*)src->metadata;
    const unsigned char* mdata = (const unsigned char*)src->compressed_data;
    float* fdst = (float*)dst;
    size_t num_blocks = (num_elements + block_size - 1) / block_size;
    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = start + block_size;
        if (end > num_elements) end = num_elements;
        int exp = exponents[b];
        float scale = ldexpf(1.0f, (float)(exp - (mantissa_bits - 1)));
        for (size_t i = start; i < end; i++) {
            size_t bit_pos = (b * block_size + (i - start)) * mantissa_bits;
            int q = 0;
            for (int mb = 0; mb < mantissa_bits; mb++) {
                if (mdata[bit_pos / 8] & (1 << (bit_pos % 8))) q |= (1 << mb);
                bit_pos++;
            }
            if (q & (1 << (mantissa_bits - 1))) q |= ~((1 << mantissa_bits) - 1);
            fdst[i] = (float)q * scale;
        }
    }
    return 0;
}

static int sparse_compress(const void* src, size_t src_bytes, int dtype, SNEPPXCompressedBuffer* dst) {
    if (!src || !dst || src_bytes == 0) return -1;
    size_t num_elements = src_bytes / sizeof(float);
    const float* fsrc = (const float*)src;
    size_t bitmap_bytes = (num_elements + 7) / 8;
    unsigned char* bitmap = (unsigned char*)calloc(bitmap_bytes, 1);
    if (!bitmap) return -1;
    size_t nonzero_count = 0;
    for (size_t i = 0; i < num_elements; i++) {
        if (fsrc[i] != 0.0f) {
            bitmap[i / 8] |= (unsigned char)(1 << (i % 8));
            nonzero_count++;
        }
    }
    float* values = (float*)calloc(nonzero_count, sizeof(float));
    if (!values) { free(bitmap); return -1; }
    size_t vi = 0;
    for (size_t i = 0; i < num_elements; i++) {
        if (bitmap[i / 8] & (1 << (i % 8))) values[vi++] = fsrc[i];
    }
    size_t total_bytes = bitmap_bytes + nonzero_count * sizeof(float);
    void* packed = calloc(1, total_bytes);
    if (!packed) { free(bitmap); free(values); return -1; }
    memcpy(packed, bitmap, bitmap_bytes);
    memcpy((unsigned char*)packed + bitmap_bytes, values, nonzero_count * sizeof(float));
    free(bitmap); free(values);
    dst->codec = SNEPPX_COMPRESS_SPARSE;
    dst->compressed_data = packed;
    dst->compressed_bytes = total_bytes;
    dst->original_bytes = src_bytes;
    dst->original_elements = num_elements;
    dst->original_dtype = dtype;
    dst->block_size = 0;
    dst->metadata = (void*)bitmap_bytes;
    return 0;
}

static int sparse_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes) {
    if (!src || !dst || dst_bytes < src->original_bytes) return -1;
    size_t num_elements = src->original_elements;
    size_t bitmap_bytes = (size_t)src->metadata;
    const unsigned char* bitmap = (const unsigned char*)src->compressed_data;
    const float* values = (const float*)((const unsigned char*)src->compressed_data + bitmap_bytes);
    float* fdst = (float*)dst;
    memset(fdst, 0, num_elements * sizeof(float));
    size_t vi = 0;
    for (size_t i = 0; i < num_elements; i++) {
        if (bitmap[i / 8] & (1 << (i % 8))) fdst[i] = values[vi++];
    }
    return 0;
}

/**
 * @brief Apply Compress.
 *
 * @param src [in] Src value.
 * @param src_bytes [in] Src Bytes value.
 * @param dtype [in] Dtype value.
 * @param codec [in] Codec value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_apply(const void* src, size_t src_bytes, int dtype,
                         SNEPPXCompressionCodec codec, SNEPPXCompressedBuffer* dst) {
    if (!src || !dst) return -1;
    memset(dst, 0, sizeof(*dst));
    for (int i = 0; i < g_num_codecs; i++) {
        if (g_codecs[i] && g_codecs[i]->codec == codec)
            return g_codecs[i]->compress(src, src_bytes, dtype, dst);
    }
    switch (codec) {
        case SNEPPX_COMPRESS_BFP4: return bfp4_compress(src, src_bytes, dtype, dst);
        case SNEPPX_COMPRESS_BFP8: return bfp8_compress(src, src_bytes, dtype, dst);
        case SNEPPX_COMPRESS_SPARSE: return sparse_compress(src, src_bytes, dtype, dst);
        case SNEPPX_COMPRESS_NONE: return none_compress(src, src_bytes, dtype, dst);
        default: return -1;
    }
}

/**
 * @brief Perform Compress Decompress.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes) {
    if (!src || !dst) return -1;
    for (int i = 0; i < g_num_codecs; i++) {
        if (g_codecs[i] && g_codecs[i]->codec == src->codec)
            return g_codecs[i]->decompress(src, dst, dst_bytes);
    }
    switch (src->codec) {
        case SNEPPX_COMPRESS_BFP4: return bfp_decompress(src, dst, dst_bytes);
        case SNEPPX_COMPRESS_BFP8: return bfp_decompress(src, dst, dst_bytes);
        case SNEPPX_COMPRESS_SPARSE: return sparse_decompress(src, dst, dst_bytes);
        case SNEPPX_COMPRESS_NONE: return none_decompress(src, dst, dst_bytes);
        default: return -1;
    }
}

/**
 * @brief Destroy Compress Buffer.
 */
void SNEPPX_compress_buffer_destroy(SNEPPXCompressedBuffer* buf) {
    if (buf) { free(buf->compressed_data); free(buf->metadata); memset(buf, 0, sizeof(*buf)); }
}

/**
 * @brief Perform Compress Bfp Block Size.
 *
 * @param element_bytes [in] Element Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_bfp_block_size(size_t element_bytes, size_t* block_size) {
    if (block_size) *block_size = (element_bytes <= 2) ? 32 : 16;
    return 0;
}

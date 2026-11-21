#ifndef SNEPPX_INTERNAL_COMPRESS_H
#define SNEPPX_INTERNAL_COMPRESS_H
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


/*
 * Memory Compression — v0.5 (internal to SNEPPX_memory)
 *
 * PURPOSE: Block-floating-point (BFP) and sparse compression of tensor
 * data to reduce memory bandwidth pressure.  Compression is transparent
 * to the tensor API: a compressed buffer is tagged with codec ID and
 * decompressed on access.
 *
 * BFP: groups elements into blocks, shared exponent per block.
 * Sparse: bitmask of non-zero elements + values array.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNEPPX_COMPRESS_NONE,
    SNEPPX_COMPRESS_BFP4,      /* block floating-point, 4-bit mantissa */
    SNEPPX_COMPRESS_BFP8,      /* block floating-point, 8-bit mantissa */
    SNEPPX_COMPRESS_SPARSE,    /* sparse bitmap encoding */
} SNEPPXCompressionCodec;

typedef struct {
    SNEPPXCompressionCodec  codec;
    void*                 compressed_data;
    size_t                compressed_bytes;
    size_t                original_bytes;
    size_t                original_elements;
    int                   original_dtype;
    size_t                block_size;       /* for BFP */
    void*                 metadata;         /* codec-specific (shared exponents, bitmask) */
} SNEPPXCompressedBuffer;

typedef struct {
    SNEPPXCompressionCodec codec;
    int                  (*compress)(const void* src, size_t src_bytes, int dtype,
                                     SNEPPXCompressedBuffer* dst);
    int                  (*decompress)(const SNEPPXCompressedBuffer* src,
                                       void* dst, size_t dst_bytes);
    void                 (*destroy)(SNEPPXCompressedBuffer* buf);
    const char*          name;
} SNEPPXCompressionCodecImpl;

/* ---------- API ---------- */
/**
 * @brief Initialize Compress.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_init(void);
/**
 * @brief Perform Compress Shutdown.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_shutdown(void);

/**
 * @brief Perform Compress Register Codec.
 *
 * @param codec [in] Codec value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_register_codec(const SNEPPXCompressionCodecImpl* codec);
/**
 * @brief Perform Compress Unregister Codec.
 *
 * @param codec [in] Codec value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_unregister_codec(SNEPPXCompressionCodec codec);

/**
 * @brief Apply Compress.
 *
 * @param src [in] Src value.
 * @param src_bytes [in] Src Bytes value.
 * @param dtype [in] Dtype value.
 * @param codec [in] Codec value.
 * @param dst [out] Dst value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_apply(const void* src, size_t src_bytes, int dtype,
                        SNEPPXCompressionCodec codec, SNEPPXCompressedBuffer* dst);
/**
 * @brief Perform Compress Decompress.
 *
 * @param src [in] Src value.
 * @param dst [out] Dst value.
 * @param dst_bytes [in] Dst Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_decompress(const SNEPPXCompressedBuffer* src, void* dst, size_t dst_bytes);
/**
 * @brief Destroy Compress Buffer.
 *
 * @param buf [out] Buf value.
 */
void SNEPPX_compress_buffer_destroy(SNEPPXCompressedBuffer* buf);

/* ---------- BFP helpers ---------- */
/**
 * @brief Perform Compress Bfp Block Size.
 *
 * @param element_bytes [in] Element Bytes value.
 * @param block_size [out] Block Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_compress_bfp_block_size(size_t element_bytes, size_t* block_size);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_INTERNAL_COMPRESS_H */

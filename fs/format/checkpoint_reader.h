#ifndef SNEPPX_CHECKPOINT_FORMAT_H
#define SNEPPX_CHECKPOINT_FORMAT_H
/*
 * SNEPPX - Checkpoint Reader
 *
 * WHAT
 *   Checkpoint Reader.
 *
 * CONCEPT
 *   Provides checkpointing and fault tolerance.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * Checkpoint File Format — v0.5 (training persistence)
 *
 * PURPOSE: Binary format for saving/loading model weights and optimizer
 * state.  Format: magic | header | tensor records | metadata.
 *   - Magic: 8 bytes (0x41524958 0x434B5054 = "SNEPPXCKPT")
 *   - Header: version, num_tensors, metadata_offset
 *   - Tensor records: shape, dtype, offset, size (repeated)
 *   - Metadata: JSON-like key-value pairs (optimizer step, epoch, RNG seed)
 *
 * Endianness: little-endian.  Alignment: 8-byte boundaries.
 * Forward compat: version field allows readers to reject newer formats.
 *
 * DEPENDENCIES: multidimensional_tensor_engine.h
 * VERSION: v0.5
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_CKPT_MAGIC      0x41524958  /* "SNEPPX" */
#define SNEPPX_CKPT_MAGIC_HI   0x434B5054  /* "CKPT" */
#define SNEPPX_CKPT_VERSION    1

#pragma pack(push, 1)
typedef struct {
    uint32_t magic_lo;
    uint32_t magic_hi;
    uint32_t version;
    uint32_t num_tensors;
    uint64_t metadata_offset;
    uint64_t metadata_size;
    uint64_t total_size;
    uint8_t  reserved[32];
} SNEPPXCheckpointHeader;

typedef struct {
    uint64_t shape[8];
    uint32_t ndim;
    uint32_t dtype;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t stride[8];
} SNEPPXTensorRecord;
#pragma pack(pop)

/* ---------- Writer ---------- */
/**
 * @brief Open Ckpt Write.
 *
 * @param path [in] Path value.
 * @param header [out] Header value.
 * @param handle [out] Handle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_open(const char* path, SNEPPXCheckpointHeader* header, void** handle);
/**
 * @brief Perform Ckpt Write Tensor.
 *
 * @param handle [out] Handle value.
 * @param tensor_data [in] Tensor Data value.
 * @param record [in] Record value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_tensor(void* handle, const void* tensor_data, const SNEPPXTensorRecord* record);
/**
 * @brief Perform Ckpt Write Metadata.
 *
 * @param handle [out] Handle value.
 * @param metadata_json [in] Metadata Json value.
 * @param json_len [in] Json Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_metadata(void* handle, const char* metadata_json, size_t json_len);
/**
 * @brief Close Ckpt Write.
 *
 * @param handle [out] Handle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_close(void* handle);

/* ---------- Reader ---------- */
/**
 * @brief Open Ckpt Read.
 *
 * @param path [in] Path value.
 * @param header [out] Header value.
 * @param handle [out] Handle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_open(const char* path, SNEPPXCheckpointHeader* header, void** handle);
/**
 * @brief Perform Ckpt Read Tensor.
 *
 * @param handle [out] Handle value.
 * @param tensor_idx [in] Tensor Idx value.
 * @param tensor_data [out] Tensor Data value.
 * @param record [out] Record value.
 *
 * @return 0 on success, -1 on error.
 */
/* Read validated tensor metadata before allocating the destination buffer. */
int SNEPPX_ckpt_get_tensor_record(void* handle, size_t tensor_idx, SNEPPXTensorRecord* record);
/* Capacity-checked read; rejects undersized buffers before writing any bytes. */
int SNEPPX_ckpt_read_tensor_sized(void* handle, size_t tensor_idx, void* tensor_data,
                                  size_t buffer_size, SNEPPXTensorRecord* record);
/* Legacy ABI: caller must provide at least record->data_size bytes. */
int SNEPPX_ckpt_read_tensor(void* handle, size_t tensor_idx, void* tensor_data, SNEPPXTensorRecord* record);
/**
 * @brief Perform Ckpt Read Metadata.
 *
 * @param handle [out] Handle value.
 * @param metadata_json [out] Metadata Json value.
 * @param json_len [out] Json Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_metadata(void* handle, char** metadata_json, size_t* json_len);
/**
 * @brief Close Ckpt Read.
 *
 * @param handle [out] Handle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_close(void* handle);

/* ---------- Utility ---------- */
/**
 * @brief Perform Ckpt Validate.
 *
 * @param path [in] Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ckpt_validate(const char* path);
/**
 * @brief Perform Ckpt Supports Version.
 *
 * @param version [in] Version value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ckpt_supports_version(uint32_t version);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_CHECKPOINT_FORMAT_H */

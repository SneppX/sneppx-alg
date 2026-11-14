#include "checkpoint_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_WIN32) || defined(_WIN64)
    #define FSEEK64(fp, offset, origin) _fseeki64(fp, offset, origin)
    #define FTELL64(fp) _ftelli64(fp)
    #define OFF64_T __int64
#else
    #define FSEEK64(fp, offset, origin) fseeko(fp, offset, origin)
    #define FTELL64(fp) ftello(fp)
    #define OFF64_T off_t
#endif

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


typedef struct {
    FILE* fp;
    SNEPPXCheckpointHeader header;
    SNEPPXTensorRecord* records;
    uint32_t num_records;
} SNEPPXCkptHandle;

#define SNEPPX_CKPT_MAX_METADATA_SIZE (64u * 1024u * 1024u)
#define SNEPPX_CKPT_MAX_RECORDS 1000000u

static int ckpt_file_size(FILE *fp, uint64_t *size) {
    if (!fp || !size || FSEEK64(fp, 0, SEEK_END) != 0) return -1;
    OFF64_T end = FTELL64(fp);
    if (end < 0) return -1;
    *size = (uint64_t)end;
    return 0;
}

static int ckpt_range_valid(uint64_t offset, uint64_t length, uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

static int ckpt_validate_layout(FILE *fp, const SNEPPXCheckpointHeader *header,
                                SNEPPXTensorRecord **records_out) {
    uint64_t file_size = 0, table_size, table_end, data_limit;
    SNEPPXTensorRecord *records = NULL;
    if (!fp || !header || !records_out) return -1;
    *records_out = NULL;
    if (ckpt_file_size(fp, &file_size) != 0 ||
        file_size < sizeof(SNEPPXCheckpointHeader)) return -1;
    if (header->magic_lo != SNEPPX_CKPT_MAGIC ||
        header->magic_hi != SNEPPX_CKPT_MAGIC_HI ||
        header->version == 0 || header->version > SNEPPX_CKPT_VERSION ||
        header->num_tensors > SNEPPX_CKPT_MAX_RECORDS ||
        header->total_size != file_size) return -1;
    if ((uint64_t)header->num_tensors >
        (UINT64_MAX - (uint64_t)sizeof(SNEPPXCheckpointHeader)) /
        (uint64_t)sizeof(SNEPPXTensorRecord)) return -1;
    table_size = (uint64_t)header->num_tensors * sizeof(SNEPPXTensorRecord);
    table_end = sizeof(SNEPPXCheckpointHeader) + table_size;
    if (table_end > file_size) return -1;
    if (header->metadata_size > SNEPPX_CKPT_MAX_METADATA_SIZE) return -1;
    if (header->metadata_size == 0) {
        if (header->metadata_offset != 0) return -1;
        data_limit = file_size;
    } else {
        if (header->metadata_offset < table_end ||
            !ckpt_range_valid(header->metadata_offset, header->metadata_size,
                              file_size) ||
            header->metadata_offset + header->metadata_size != file_size)
            return -1;
        data_limit = header->metadata_offset;
    }
    if (header->num_tensors != 0) {
        records = (SNEPPXTensorRecord *)calloc(header->num_tensors,
                                               sizeof(SNEPPXTensorRecord));
        if (!records) return -1;
        if (FSEEK64(fp, (OFF64_T)sizeof(SNEPPXCheckpointHeader), SEEK_SET) != 0 ||
            fread(records, sizeof(SNEPPXTensorRecord),
                  header->num_tensors, fp) != header->num_tensors) {
            free(records);
            return -1;
        }
        for (uint32_t i = 0; i < header->num_tensors; ++i) {
            const SNEPPXTensorRecord *record = &records[i];
            if (record->ndim > 8 || record->data_offset < table_end ||
                !ckpt_range_valid(record->data_offset, record->data_size,
                                  data_limit)) {
                free(records);
                return -1;
            }
        }
    }
    *records_out = records;
    return 0;
}

/**
 * @brief Open Ckpt Write.
 *
 * @param path [in] Path value.
 * @param header [out] Header value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_open(const char* path, SNEPPXCheckpointHeader* header, void** handle) {
    if (!path || !header || !handle) return -1;
    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)calloc(1, sizeof(SNEPPXCkptHandle));
    if (!h) { fclose(fp); return -1; }
    h->fp = fp;
    memcpy(&h->header, header, sizeof(SNEPPXCheckpointHeader));
    h->header.magic_lo = SNEPPX_CKPT_MAGIC;
    h->header.magic_hi = SNEPPX_CKPT_MAGIC_HI;
    h->header.version = SNEPPX_CKPT_VERSION;
    
    // Write header
    fwrite(&h->header, sizeof(SNEPPXCheckpointHeader), 1, fp);
    
    // Write placeholder records (will be filled in write_tensor)
    if (header->num_tensors > 0) {
        h->records = (SNEPPXTensorRecord*)calloc(header->num_tensors, sizeof(SNEPPXTensorRecord));
        if (!h->records) { fclose(fp); return -1; }
        h->num_records = header->num_tensors;
        // Write placeholder records
        fwrite(h->records, sizeof(SNEPPXTensorRecord), header->num_tensors, fp);
    }
    *handle = h;
    return 0;
}

/**
 * @brief Perform Ckpt Write Tensor.
 *
 * @param handle [out] Handle value.
 * @param tensor_data [in] Tensor Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_tensor(void* handle, const void* tensor_data, const SNEPPXTensorRecord* record) {
    if (!handle || !tensor_data || !record) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    if (h->num_records == 0) return -1;
    uint32_t idx = h->header.num_tensors - h->num_records;
    
    // Get current file position (this is where tensor data will be written)
    FSEEK64(h->fp, 0, SEEK_END);
    OFF64_T data_offset = ftell(h->fp);
    
    // Update the record in memory with the data offset
    h->records[idx] = *record;
    h->records[idx].data_offset = (uint64_t)data_offset;
    
    // Update the record in the file (seek to record position in the records array)
    uint64_t record_pos = sizeof(SNEPPXCheckpointHeader) + idx * sizeof(SNEPPXTensorRecord);
    FSEEK64(h->fp, (OFF64_T)record_pos, SEEK_SET);
    fwrite(&h->records[idx], sizeof(SNEPPXTensorRecord), 1, h->fp);
    
    // Seek back to end and write tensor data
    FSEEK64(h->fp, 0, SEEK_END);
    fwrite(tensor_data, 1, record->data_size, h->fp);
    
    h->num_records--;
    return 0;
}

/**
 * @brief Perform Ckpt Write Metadata.
 *
 * @param handle [out] Handle value.
 * @param metadata_json [in] Metadata Json value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_metadata(void* handle, const char* metadata_json, size_t json_len) {
    if (!handle || !metadata_json) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    FSEEK64(h->fp, 0, SEEK_END);
    h->header.metadata_offset = (uint64_t)ftell(h->fp);
    h->header.metadata_size = (uint64_t)json_len;
    fwrite(metadata_json, 1, json_len, h->fp);
    return 0;
}

/**
 * @brief Close Ckpt Write.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_write_close(void* handle) {
    if (!handle) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    
    // Update header with final total_size
    FSEEK64(h->fp, 0, SEEK_END);
    long end_pos = ftell(h->fp);
    h->header.total_size = (uint64_t)end_pos;
    
    // Write final header
    FSEEK64(h->fp, 0, SEEK_SET);
    fwrite(&h->header, sizeof(SNEPPXCheckpointHeader), 1, h->fp);
    
    fclose(h->fp);
    free(h->records);
    free(h);
    return 0;
}

/**
 * @brief Open Ckpt Read.
 *
 * @param path [in] Path value.
 * @param header [out] Header value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_read_open(const char* path, SNEPPXCheckpointHeader* header, void** handle) {
    SNEPPXCkptHandle *h;
    SNEPPXCheckpointHeader parsed;
    SNEPPXTensorRecord *records = NULL;
    if (!path || !header || !handle) return -1;
    *handle = NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fread(&parsed, sizeof(parsed), 1, fp) != 1 ||
        ckpt_validate_layout(fp, &parsed, &records) != 0) {
        fclose(fp);
        return -1;
    }
    h = (SNEPPXCkptHandle *)calloc(1, sizeof(*h));
    if (!h) {
        free(records);
        fclose(fp);
        return -1;
    }
    h->fp = fp;
    h->header = parsed;
    h->records = records;
    h->num_records = parsed.num_tensors;
    *header = parsed;
    *handle = h;
    return 0;
}

int SNEPPX_ckpt_read_tensor(void* handle, size_t tensor_idx, void* tensor_data, SNEPPXTensorRecord* record) {
    SNEPPXCkptHandle *h = (SNEPPXCkptHandle *)handle;
    const SNEPPXTensorRecord *entry;
    if (!h || !tensor_data || tensor_idx >= h->header.num_tensors) return -1;
    entry = &h->records[tensor_idx];
    if (entry->data_size > SIZE_MAX ||
        FSEEK64(h->fp, (OFF64_T)entry->data_offset, SEEK_SET) != 0 ||
        fread(tensor_data, 1, (size_t)entry->data_size, h->fp) !=
            (size_t)entry->data_size) return -1;
    if (record) memcpy(record, entry, sizeof(*record));
    return 0;
}

int SNEPPX_ckpt_read_metadata(void* handle, char** metadata_json, size_t* json_len) {
    SNEPPXCkptHandle *h = (SNEPPXCkptHandle *)handle;
    char *buffer;
    size_t length;
    if (!h || !metadata_json || !json_len) return -1;
    *metadata_json = NULL;
    *json_len = 0;
    if (h->header.metadata_size == 0) return 0;
    if (h->header.metadata_size > SIZE_MAX - 1 ||
        h->header.metadata_size > SNEPPX_CKPT_MAX_METADATA_SIZE) return -1;
    length = (size_t)h->header.metadata_size;
    buffer = (char *)malloc(length + 1);
    if (!buffer) return -1;
    if (FSEEK64(h->fp, (OFF64_T)h->header.metadata_offset, SEEK_SET) != 0 ||
        fread(buffer, 1, length, h->fp) != length) {
        free(buffer);
        return -1;
    }
    buffer[length] = '\0';
    *metadata_json = buffer;
    *json_len = length;
    return 0;
}

int SNEPPX_ckpt_read_close(void* handle) {
    if (!handle) return -1;
    SNEPPXCkptHandle* h = (SNEPPXCkptHandle*)handle;
    fclose(h->fp);
    free(h->records);
    free(h);
    return 0;
}

/**
 * @brief Perform Ckpt Validate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ckpt_validate(const char* path) {
    SNEPPXCheckpointHeader header;
    SNEPPXTensorRecord *records = NULL;
    FILE *fp;
    int result;
    if (!path) return -1;
    fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    result = ckpt_validate_layout(fp, &header, &records);
    free(records);
    fclose(fp);
    return result;
}

int SNEPPX_ckpt_supports_version(uint32_t version) {
    return version <= SNEPPX_CKPT_VERSION ? 1 : 0;
}

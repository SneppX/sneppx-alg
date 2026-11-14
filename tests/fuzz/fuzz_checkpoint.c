#include "checkpoint_reader.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PATH "sneppx_checkpoint_integrity_test.ckpt"

static int write_checkpoint(void) {
    SNEPPXCheckpointHeader header = {0};
    SNEPPXTensorRecord record = {0};
    const uint32_t values[4] = {11u, 22u, 33u, 44u};
    void *writer = NULL;
    header.num_tensors = 1;
    if (SNEPPX_ckpt_write_open(TEST_PATH, &header, &writer) != 0) return -1;
    record.ndim = 1;
    record.shape[0] = 4;
    record.data_size = sizeof(values);
    if (SNEPPX_ckpt_write_tensor(writer, values, &record) != 0) {
        SNEPPX_ckpt_write_close(writer);
        return -1;
    }
    return SNEPPX_ckpt_write_close(writer);
}

static int read_file(unsigned char **bytes, size_t *length) {
    FILE *fp = fopen(TEST_PATH, "rb");
    long size;
    if (!fp) return -1;
    if (fseek(fp, 0, SEEK_END) != 0 || (size = ftell(fp)) < 0 ||
        fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    *bytes = (unsigned char *)malloc((size_t)size);
    if (!*bytes) { fclose(fp); return -1; }
    *length = (size_t)size;
    if (fread(*bytes, 1, *length, fp) != *length) {
        free(*bytes);
        *bytes = NULL;
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int write_bytes(const unsigned char *bytes, size_t length) {
    FILE *fp = fopen(TEST_PATH, "wb");
    int ok;
    if (!fp) return -1;
    ok = fwrite(bytes, 1, length, fp) == length && fclose(fp) == 0;
    return ok ? 0 : -1;
}


static int test_step_metadata(void) {
    char buffer[32];
    size_t written = 0;
    int step = -1;
    const char valid[] = "{\"step\":12345}";
    const char overflow[] = "{\"step\":2147483648}";
    const char malformed[] = "{\"step\":\"12345\"}";
    if (SNEPPX_ckpt_format_step_metadata(12345, buffer, sizeof(buffer),
                                         &written) != 0 ||
        written != sizeof(valid) - 1 ||
        memcmp(buffer, valid, written) != 0 ||
        SNEPPX_ckpt_parse_step_metadata(buffer, written, &step) != 0 ||
        step != 12345) return 1;
    if (SNEPPX_ckpt_parse_step_metadata(malformed, sizeof(malformed) - 1,
                                        &step) == 0 ||
        SNEPPX_ckpt_parse_step_metadata(overflow, sizeof(overflow) - 1,
                                        &step) == 0 ||
        SNEPPX_ckpt_format_step_metadata(-1, buffer, sizeof(buffer),
                                         &written) == 0 ||
        SNEPPX_ckpt_format_step_metadata(1, buffer, 2, &written) == 0)
        return 2;
    return 0;
}

static int test_writer_contract(void) {
    SNEPPXCheckpointHeader header = {0};
    SNEPPXTensorRecord record = {0};
    const uint32_t value = 99u;
    const char metadata[] = "{\"epoch\":1}";
    void *writer = NULL;
    void *reader = NULL;
    char *read_metadata = NULL;
    size_t metadata_len = 0;

    header.num_tensors = 1;
    if (SNEPPX_ckpt_write_open(TEST_PATH, &header, &writer) != 0) return 1;
    if (SNEPPX_ckpt_write_metadata(writer, metadata, sizeof(metadata) - 1) == 0) {
        SNEPPX_ckpt_write_close(writer);
        return 2;
    }
    record.ndim = 1;
    record.shape[0] = 1;
    record.data_size = sizeof(value);
    if (SNEPPX_ckpt_write_tensor(writer, &value, &record) != 0 ||
        SNEPPX_ckpt_write_metadata(writer, metadata, sizeof(metadata) - 1) != 0 ||
        SNEPPX_ckpt_write_metadata(writer, metadata, sizeof(metadata) - 1) == 0 ||
        SNEPPX_ckpt_write_close(writer) != 0 ||
        SNEPPX_ckpt_validate(TEST_PATH) != 0) return 3;

    if (SNEPPX_ckpt_read_open(TEST_PATH, &header, &reader) != 0 ||
        SNEPPX_ckpt_read_metadata(reader, &read_metadata, &metadata_len) != 0 ||
        metadata_len != sizeof(metadata) - 1 ||
        memcmp(read_metadata, metadata, metadata_len) != 0) {
        if (reader) SNEPPX_ckpt_read_close(reader);
        free(read_metadata);
        return 4;
    }
    free(read_metadata);
    SNEPPX_ckpt_read_close(reader);

    /* An incomplete checkpoint must never be finalized as valid. */
    memset(&header, 0, sizeof(header));
    header.num_tensors = 1;
    if (SNEPPX_ckpt_write_open(TEST_PATH, &header, &writer) != 0) return 5;
    if (SNEPPX_ckpt_write_close(writer) == 0 ||
        SNEPPX_ckpt_validate(TEST_PATH) == 0) return 6;
    return 0;
}

int main(void) {
    unsigned char *bytes = NULL;
    size_t length = 0;
    SNEPPXCheckpointHeader header;
    SNEPPXTensorRecord *record;
    void *reader = NULL;
    uint32_t output[4] = {0};

    if (write_checkpoint() != 0 ||
        SNEPPX_ckpt_validate(TEST_PATH) != 0 ||
        read_file(&bytes, &length) != 0) return 1;

    if (SNEPPX_ckpt_read_open(TEST_PATH, &header, &reader) != 0) {
        free(bytes);
        return 2;
    }
    SNEPPXTensorRecord queried = {0};
    unsigned char guarded[sizeof(output)];
    memset(guarded, 0xA5, sizeof(guarded));
    if (SNEPPX_ckpt_get_tensor_record(reader, 0, &queried) != 0 ||
        queried.data_size != sizeof(output) ||
        SNEPPX_ckpt_read_tensor_sized(reader, 0, guarded, sizeof(guarded) - 1, NULL) == 0 ||
        guarded[0] != 0xA5 || guarded[sizeof(guarded) - 1] != 0xA5 ||
        SNEPPX_ckpt_read_tensor_sized(reader, 0, output, sizeof(output), NULL) != 0 ||
        memcmp(output, (uint32_t[]){11u, 22u, 33u, 44u}, sizeof(output)) != 0) {
        SNEPPX_ckpt_read_close(reader);
        free(bytes);
        return 3;
    }
    unsigned char oversized[sizeof(output) + 8] = {0};
    if (SNEPPX_ckpt_read_tensor_sized(reader, 0, oversized, sizeof(oversized), NULL) != 0 ||
        memcmp(oversized, output, sizeof(output)) != 0) {
        SNEPPX_ckpt_read_close(reader);
        free(bytes);
        return 8;
    }
    SNEPPX_ckpt_read_close(reader);

    /* Reject a truncated header. */
    if (write_bytes(bytes, sizeof(SNEPPXCheckpointHeader) - 1) != 0 ||
        SNEPPX_ckpt_validate(TEST_PATH) == 0) { free(bytes); return 4; }

    /* Reject an attacker-controlled tensor count before allocating records. */
    ((SNEPPXCheckpointHeader *)bytes)->num_tensors = UINT32_MAX;
    if (write_bytes(bytes, length) != 0 ||
        SNEPPX_ckpt_validate(TEST_PATH) == 0) { free(bytes); return 5; }

    /* Reject a tensor record whose data offset wraps/out-ranges the file. */
    memcpy(bytes, &header, sizeof(header));
    record = (SNEPPXTensorRecord *)(bytes + sizeof(SNEPPXCheckpointHeader));
    record->data_offset = UINT64_MAX;
    if (write_bytes(bytes, length) != 0 ||
        SNEPPX_ckpt_validate(TEST_PATH) == 0) { free(bytes); return 6; }

    free(bytes);
    if (test_writer_contract() != 0) return 7;
    if (test_step_metadata() != 0) return 9;
    remove(TEST_PATH);
    return 0;
}

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
    if (SNEPPX_ckpt_read_tensor(reader, 0, output, NULL) != 0 ||
        memcmp(output, (uint32_t[]){11u, 22u, 33u, 44u}, sizeof(output)) != 0) {
        SNEPPX_ckpt_read_close(reader);
        free(bytes);
        return 3;
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
    remove(TEST_PATH);
    return 0;
}

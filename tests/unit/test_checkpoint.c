#include "checkpoint_reader.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Checkpoint
 *
 * WHAT
 *   Test Checkpoint.
 *
 * CONCEPT
 *   Provides checkpointing and fault tolerance.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_checkpoint_validate(void) {
    int ret = SNEPPX_ckpt_validate("/tmp/nonexistent.ckpt");
    SX_ASSERT(ret == -1, "validate nonexistent file returns -1");
}

static void test_checkpoint_supports_version(void) {
    SX_ASSERT(SNEPPX_ckpt_supports_version(1) == 1, "supports version 1");
    SX_ASSERT(SNEPPX_ckpt_supports_version(2) == 0, "does not support version 2");
}

static void test_checkpoint_open(void) {
    void* handle = NULL;
    SNEPPXCheckpointHeader header = {0};
    int ret = SNEPPX_ckpt_read_open("/nonexistent.ckpt", &header, NULL);
    SX_ASSERT(ret != 0, "open nonexistent returns error");
}

static void test_checkpoint_write_read(void) {
    const char* path = "C:\\Users\\PC\\test_checkpoint.ckpt";
    SNEPPXCheckpointHeader header = {0};
    header.num_tensors = 1;
    void* handle = NULL;
    
    int ret = SNEPPX_ckpt_write_open(path, &header, &handle);
    SX_ASSERT(ret == 0, "write open succeeds");
    
    SNEPPXTensorRecord record = {0};
    record.ndim = 1;
    record.shape[0] = 4;
    record.dtype = 0;
    record.data_size = 16;
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    ret = SNEPPX_ckpt_write_tensor(handle, data, &record);
    SX_ASSERT(ret == 0, "write tensor succeeds");
    
    ret = SNEPPX_ckpt_write_close(handle);
    SX_ASSERT(ret == 0, "write close succeeds");
    
    // Read back
    SNEPPXCheckpointHeader read_header = {0};
    void* read_handle = NULL;
    ret = SNEPPX_ckpt_read_open(path, &read_header, &handle);
    SX_ASSERT(ret == 0, "read open succeeds");
    SX_ASSERT(read_header.num_tensors == 1, "num_tensors matches");
    
    float read_data[4] = {0};
    SNEPPXTensorRecord read_record = {0};
    ret = SNEPPX_ckpt_read_tensor(handle, 0, read_data, &read_record);
    SX_ASSERT(ret == 0, "read tensor succeeds");
    SX_ASSERT(read_data[0] == 1.0f && read_data[3] == 4.0f, "data matches");
    
    SNEPPX_ckpt_read_close(handle);
}


TEST(test_checkpoint, checkpoint_validate) { test_checkpoint_validate(); }
TEST(test_checkpoint, checkpoint_supports_version) { test_checkpoint_supports_version(); }
TEST(test_checkpoint, checkpoint_open) { test_checkpoint_open(); }
TEST(test_checkpoint, checkpoint_write_read) { test_checkpoint_write_read(); }

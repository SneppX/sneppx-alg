#include "data_pipeline.h"
#include "test_gtest.h"
#include "multidimensional_tensor_engine.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Test Data Pipeline
 *
 * WHAT
 *   Test Data Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_pipeline_create(void) {
    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(256);
    SX_ASSERT(pipe != NULL, "pipeline created");
    SX_ASSERT(SNEPPX_data_pipeline_get_batch_size(pipe) == 256, "batch size set");
    SNEPPX_data_pipeline_destroy(pipe);
}

static void test_pipeline_load_csv(void) {
    const char* csv_data = "1.0,2.0,3.0,0.0\n4.0,5.0,6.0,1.0\n7.0,8.0,9.0,0.0\n";
    const char* tmpfile = "test_data_pipeline_tmp.csv";
    FILE* f = fopen(tmpfile, "w");
    SX_ASSERT(f != NULL, "temp file created");
    fputs(csv_data, f);
    fclose(f);

    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(2);
    SX_ASSERT(pipe != NULL, "pipeline created");

    SNEPPXTensor* data = NULL;
    SNEPPXTensor* labels = NULL;
    int ret = SNEPPX_data_pipeline_load(tmpfile, pipe, &data, &labels);
    SX_ASSERT(ret == 0, "load csv returns 0");
    SX_ASSERT(data != NULL, "data tensor created");
    SX_ASSERT(labels != NULL, "labels tensor created");
    SX_ASSERT(data->shape[0] == 3, "3 samples loaded");
    SX_ASSERT(data->shape[1] == 3, "3 features per sample");
    SNEPPX_tensor_destroy(data);
    SNEPPX_tensor_destroy(labels);

    ret = SNEPPX_data_pipeline_load(tmpfile, pipe, NULL, NULL);
    SX_ASSERT(ret == 0, "load with NULL tensor pointers");

    SNEPPX_data_pipeline_destroy(pipe);
    remove(tmpfile);
}

static void test_pipeline_get_batch(void) {
    const char* csv_data = "1.0,2.0,0.0\n4.0,5.0,1.0\n7.0,8.0,0.0\n9.0,10.0,1.0\n";
    const char* tmpfile = "test_pipeline_batch_tmp.csv";
    FILE* f = fopen(tmpfile, "w");
    fputs(csv_data, f);
    fclose(f);

    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(2);
    SNEPPX_data_pipeline_load(tmpfile, pipe, NULL, NULL);
    SNEPPX_data_pipeline_shuffle(pipe);

    SNEPPXTensor* batch = NULL;
    SNEPPXTensor* labels = NULL;
    int ret = SNEPPX_data_pipeline_get_batch(pipe, &batch, &labels);
    SX_ASSERT(ret == 0, "get_batch returns 0");
    SX_ASSERT(batch != NULL, "batch tensor created");
    SX_ASSERT(labels != NULL, "labels tensor created");
    SX_ASSERT(batch->shape[0] == 2, "batch size matches");
    SNEPPX_tensor_destroy(batch);
    SNEPPX_tensor_destroy(labels);

    SNEPPX_data_pipeline_destroy(pipe);
    remove(tmpfile);
}

static void test_pipeline_shuffle(void) {
    SNEPPXDataPipeline* pipe = SNEPPX_data_pipeline_create(64);
    SNEPPX_data_pipeline_shuffle(pipe);
    SX_ASSERT(pipe != NULL, "shuffle does not crash");
    SNEPPX_data_pipeline_destroy(pipe);
}


TEST(test_data_pipeline, pipeline_create) { test_pipeline_create(); }
TEST(test_data_pipeline, pipeline_load_csv) { test_pipeline_load_csv(); }
TEST(test_data_pipeline, pipeline_get_batch) { test_pipeline_get_batch(); }
TEST(test_data_pipeline, pipeline_shuffle) { test_pipeline_shuffle(); }

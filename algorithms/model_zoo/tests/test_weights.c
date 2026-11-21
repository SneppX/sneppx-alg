#include "weights.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

/*
 * SNEPPX - Test Weights
 *
 * WHAT
 *   Test Weights.
 *
 * CONCEPT
 *   Provides the Test Weights.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


void test_create_destroy(void) {
    printf("test_create_destroy... ");
    WeightCollection *wc = weight_collection_create();
    assert(wc != NULL);
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_add_remove(void) {
    printf("test_add_remove... ");
    WeightCollection *wc = weight_collection_create();
    
    // Add a tensor
    int64_t shape[] = {2, 3};
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int rc = weight_collection_add(wc, "test_tensor", shape, 2, "float32", data, 6 * sizeof(float), 1);
    assert(rc == 0);
    
    WeightTensor *t = weight_collection_get(wc, "test_tensor");
    assert(t != NULL);
    assert(strcmp(t->name, "test_tensor") == 0);
    assert(t->ndim == 2);
    assert(t->shape[0] == 2 && t->shape[1] == 3);
    assert(strcmp(t->dtype, "float32") == 0);
    
    // Get non-existent
    WeightTensor *nt = weight_collection_get(wc, "nonexistent");
    assert(nt == NULL);
    
    // List names
    int count = 0;
    char **names = weight_collection_names(wc, &count);
    assert(count == 1);
    assert(strcmp(names[0], "test_tensor") == 0);
    free(names[0]);
    free(names);
    
    // Remove
    int rc2 = weight_collection_remove(wc, "test_tensor");
    assert(rc2 == 0);
    t = weight_collection_get(wc, "test_tensor");
    assert(t == NULL);
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_convert_dtype(void) {
    printf("test_convert_dtype... ");
    WeightCollection *wc = weight_collection_create();
    
    int64_t shape[] = {4};
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    weight_collection_add(wc, "fp32_tensor", shape, 1, "float32", data, 4 * sizeof(float), 1);
    
    WeightTensor *t = weight_collection_get(wc, "fp32_tensor");
    assert(strcmp(t->dtype, "float32") == 0);
    
    int rc = weight_tensor_convert_dtype(t, "float16");
    assert(rc == 0);
    assert(strcmp(t->dtype, "float16") == 0);
    assert(t->data_size == 4 * 2);
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_quantize_int8(void) {
    printf("test_quantize_int8... ");
    WeightCollection *wc = weight_collection_create();
    
    int64_t shape[] = {8};
    float data[8] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, -2.0f, 0.25f};
    weight_collection_add(wc, "quant_tensor", shape, 1, "float32", data, 8 * sizeof(float), 1);
    
    WeightTensor *t = weight_collection_get(wc, "quant_tensor");
    assert(strcmp(t->dtype, "float32") == 0);
    
    int rc = weight_tensor_quantize_int8(t);
    assert(rc == 0);
    assert(strcmp(t->dtype, "int8") == 0);
    assert(t->data_size == 8);
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_format_utils(void) {
    printf("test_format_utils... ");
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_SAFETENSORS), "safetensors") == 0);
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_GGUF), "gguf") == 0);
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_NPZ), "npz") == 0);
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_AUTO), "auto") == 0);
    
    assert(weight_format_from_str("safetensors") == WEIGHT_FORMAT_SAFETENSORS);
    assert(weight_format_from_str("gguf") == WEIGHT_FORMAT_GGUF);
    assert(weight_format_from_str("npz") == WEIGHT_FORMAT_NPZ);
    assert(weight_format_from_str("unknown") == WEIGHT_FORMAT_AUTO);
    assert(weight_format_from_str(NULL) == WEIGHT_FORMAT_AUTO);
    
    printf("PASS\n");
}

int main(void) {
    printf("\n=== Weights Tests ===\n\n");
    
    test_create_destroy();
    test_add_remove();
    test_convert_dtype();
    test_quantize_int8();
    test_format_utils();
    
    printf("\n=== All tests passed ===\n\n");
    return 0;
}

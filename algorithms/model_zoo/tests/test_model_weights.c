#include "model_config.h"
#include <neural_core/model_zoo/weights.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/*
 * SNEPPX - Test Model Weights
 *
 * WHAT
 *   Test Model Weights.
 *
 * CONCEPT
 *   Provides the Test Model Weights.
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
    assert(wc->count == 0);
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_add_remove(void) {
    printf("test_add_remove... ");
    WeightCollection *wc = weight_collection_create();
    
    int64_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    int rc = weight_collection_add(wc, "test_tensor", shape, 2, "float32", data, sizeof(data), 0);
    assert(rc == 0);
    assert(wc->count == 1);
    
    WeightTensor *t = weight_collection_get(wc, "test_tensor");
    assert(t != NULL);
    assert(strcmp(t->name, "test_tensor") == 0);
    assert(t->ndim == 2);
    assert(t->shape[0] == 2 && t->shape[1] == 3);
    assert(t->numel == 6);
    assert(strcmp(t->dtype, "float32") == 0);
    
    // Duplicate name should fail
    int rc2 = weight_collection_add(wc, "test_tensor", shape, 2, "float32", data, sizeof(data), 0);
    assert(rc2 != 0);
    
    // Remove
    int rc3 = weight_collection_remove(wc, "test_tensor");
    assert(rc3 == 0);
    assert(wc->count == 0);
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_get_names(void) {
    printf("test_get_names... ");
    WeightCollection *wc = weight_collection_create();
    
    int64_t shape1[] = {2};
    int64_t shape2[] = {3, 4};
    float data1[] = {1.0f, 2.0f};
    float data2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    weight_collection_add(wc, "tensor_a", shape1, 1, "float32", &data1[0], sizeof(data1), 0);
    weight_collection_add(wc, "tensor_b", shape2, 2, "float32", &data2[0], sizeof(data2), 0);
    
    int count = 0;
    char **names = weight_collection_names(wc, &count);
    assert(count == 2);
    int found_a = 0, found_b = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], "tensor_a") == 0) found_a = 1;
        if (strcmp(names[i], "tensor_b") == 0) found_b = 1;
        free(names[i]);
    }
    free(names);
    assert(found_a && found_b);
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_dtype_conversion(void) {
    printf("test_dtype_conversion... ");
    WeightCollection *wc = weight_collection_create();
    
    int64_t shape[] = {4};
    float data[] = {1.0f, -2.0f, 3.0f, -4.0f};
    
    weight_collection_add(wc, "fp32_tensor", shape, 1, "float32", data, sizeof(data), 0);
    WeightTensor *t = weight_collection_get(wc, "fp32_tensor");
    assert(t != NULL);
    assert(strcmp(t->dtype, "float32") == 0);
    
    int rc = weight_tensor_convert_dtype(t, "float16");
    assert(rc == 0);
    assert(strcmp(t->dtype, "float16") == 0);
    assert(t->data_size == 8); // 4 elements * 2 bytes
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_quantize_int8(void) {
    printf("test_quantize_int8... ");
    WeightCollection *wc = weight_collection_create();
    
    int64_t shape[] = {4};
    float data[] = {100.0f, -50.0f, 200.0f, -25.0f};
    
    weight_collection_add(wc, "quant_test", shape, 1, "float32", data, sizeof(data), 0);
    WeightTensor *t = weight_collection_get(wc, "quant_test");
    assert(t != NULL);
    assert(strcmp(t->dtype, "float32") == 0);
    
    int rc = weight_tensor_quantize_int8(t);
    assert(rc == 0);
    assert(strcmp(t->dtype, "int8") == 0);
    assert(t->data_size == 4);
    
    weight_collection_destroy(wc);
    printf("PASS\n");
}

void test_format_utils(void) {
    printf("test_format_utils... ");
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_SAFETENSORS), "safetensors") == 0);
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_GGUF), "gguf") == 0);
    assert(strcmp(weight_format_to_str(WEIGHT_FORMAT_NPZ), "npz") == 0);
    
    assert(weight_format_from_str("safetensors") == WEIGHT_FORMAT_SAFETENSORS);
    assert(weight_format_from_str("gguf") == WEIGHT_FORMAT_GGUF);
    assert(weight_format_from_str("npz") == WEIGHT_FORMAT_NPZ);
    assert(weight_format_from_str("unknown") == WEIGHT_FORMAT_AUTO);
    
    printf("PASS\n");
}

void test_safetensors_load(void) {
    printf("test_safetensors_load... ");
    // Create a minimal safetensors file for testing
    // This is a stub test - the actual loading is stubbed
    WeightCollection *wc = weights_load_safetensors("/nonexistent.safetensors");
    // Should return NULL for nonexistent file
    assert(wc == NULL);
    printf("PASS\n");
}

int main(void) {
    printf("\n=== Model Weights Tests ===\n\n");
    
    test_create_destroy();
    test_add_remove();
    test_get_names();
    test_dtype_conversion();
    test_quantize_int8();
    test_format_utils();
    test_safetensors_load();
    
    printf("\n=== All tests passed ===\n\n");
    return 0;
}

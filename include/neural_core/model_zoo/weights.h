#ifndef NEURAL_CORE_MODEL_ZOO_WEIGHTS_H
#define NEURAL_CORE_MODEL_ZOO_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Weights
 *
 * WHAT
 *   Weights.
 *
 * CONCEPT
 *   Provides the Weights.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

// Weight format types
typedef enum {
    WEIGHT_FORMAT_AUTO = 0,
    WEIGHT_FORMAT_SAFETENSORS = 1,
    WEIGHT_FORMAT_GGUF = 2,
    WEIGHT_FORMAT_NPZ = 3,
    WEIGHT_FORMAT_PYTORCH = 4,
    WEIGHT_FORMAT_ONNX = 5,
} WeightFormat;

// Tensor storage
typedef struct WeightTensor WeightTensor;
struct WeightTensor {
    char *name;
    int64_t *shape;
    int ndim;
    size_t numel;
    char *dtype;
    void *data;
    size_t data_size;
    int owns_data;
};

// Weight collection
typedef struct WeightCollection WeightCollection;
struct WeightCollection {
    WeightTensor **tensors;
    int count;
    int capacity;
    WeightFormat format;
    char *source_path;
    char *metadata_json;  // Raw metadata from file format
};

// Format conversion
WeightFormat weight_format_from_str(const char *str);
const char *weight_format_to_str(WeightFormat fmt);

// Load weights from file
WeightCollection *weights_load(const char *path, WeightFormat format);
WeightCollection *weights_load_safetensors(const char *path);
WeightCollection *weights_load_gguf(const char *path);
WeightCollection *weights_load_npz(const char *path);

// Save weights
int weights_save(const WeightCollection *wc, const char *path, WeightFormat format);

// Collection management
WeightCollection *weight_collection_create(void);
void weight_collection_destroy(WeightCollection *wc);
int weight_collection_add(WeightCollection *wc, const char *name, const int64_t *shape,
                          int ndim, const char *dtype, const void *data, size_t data_size,
                          int owns_data);
WeightTensor *weight_collection_get(WeightCollection *wc, const char *name);
const WeightTensor *weight_collection_get_const(const WeightCollection *wc, const char *name);
int weight_collection_remove(WeightCollection *wc, const char *name);
char **weight_collection_names(const WeightCollection *wc, int *count_out);

// Dtype conversion
int weight_tensor_convert_dtype(WeightTensor *tensor, const char *target_dtype);

// Quantization
int weight_tensor_quantize_int8(WeightTensor *tensor);
int weight_tensor_quantize_int4(WeightTensor *tensor);

// Utility
void weight_tensor_destroy(WeightTensor *tensor);
int weight_collection_mmap(WeightCollection *wc, const char *path);

// Helper
size_t dtype_size(const char *dtype);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_CORE_MODEL_ZOO_WEIGHTS_H

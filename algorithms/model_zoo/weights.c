#include "model_config.h"
#include <neural_core/model_zoo/weights.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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


// ── Internal helpers ────────────────────────────────────────────────────────

static char *dup_str(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (copy) memcpy(copy, s, len + 1);
    return copy;
}

// ── WeightCollection ────────────────────────────────────────────────────────

WeightCollection *weight_collection_create(void) {
    WeightCollection *wc = (WeightCollection *)calloc(1, sizeof(WeightCollection));
    if (!wc) return NULL;
    wc->capacity = 8;
    wc->tensors = (WeightTensor **)calloc(wc->capacity, sizeof(WeightTensor *));
    return wc;
}

void weight_collection_destroy(WeightCollection *wc) {
    if (!wc) return;
    for (int i = 0; i < wc->count; i++) {
        if (wc->tensors[i]) weight_tensor_destroy(wc->tensors[i]);
    }
    free(wc->tensors);
    free(wc->source_path);
    free(wc->metadata_json);
    free(wc);
}

// ── WeightTensor ────────────────────────────────────────────────────────────

void weight_tensor_destroy(WeightTensor *tensor) {
    if (!tensor) return;
    free(tensor->name);
    free(tensor->shape);
    free(tensor->dtype);
    if (tensor->owns_data && tensor->data) free(tensor->data);
    free(tensor);
}

// ── Tensor access ───────────────────────────────────────────────────────────

WeightTensor *weight_collection_get(WeightCollection *wc, const char *name) {
    if (!wc || !name) return NULL;
    for (int i = 0; i < wc->count; i++) {
        if (wc->tensors[i] && strcmp(wc->tensors[i]->name, name) == 0) {
            return wc->tensors[i];
        }
    }
    return NULL;
}

const WeightTensor *weight_collection_get_const(const WeightCollection *wc, const char *name) {
    if (!wc || !name) return NULL;
    for (int i = 0; i < wc->count; i++) {
        if (wc->tensors[i] && strcmp(wc->tensors[i]->name, name) == 0) {
            return wc->tensors[i];
        }
    }
    return NULL;
}

// ── Add/remove tensors ──────────────────────────────────────────────────────

int weight_collection_add(WeightCollection *wc, const char *name, const int64_t *shape,
                          int ndim, const char *dtype, const void *data, size_t data_size, int owns_data) {
    if (!wc || !name || !shape || ndim <= 0 || !dtype) return -1;
    
    // Check for duplicate
    for (int i = 0; i < wc->count; i++) {
        if (wc->tensors[i] && strcmp(wc->tensors[i]->name, name) == 0) {
            return -1;
        }
    }
    
    if (wc->count >= wc->capacity) {
        int new_cap = wc->capacity ? wc->capacity * 2 : 8;
        WeightTensor **new_tensors = (WeightTensor **)realloc(wc->tensors, new_cap * sizeof(WeightTensor *));
        if (!new_tensors) return -1;
        wc->tensors = new_tensors;
        wc->capacity = new_cap;
    }
    
    WeightTensor *t = (WeightTensor *)calloc(1, sizeof(WeightTensor));
    if (!t) return -1;
    
    t->name = strdup(name);
    t->ndim = ndim;
    t->shape = (int64_t *)malloc(ndim * sizeof(int64_t));
    if (!t->shape) { free(t->name); free(t); return -1; }
    memcpy(t->shape, shape, ndim * sizeof(int64_t));
    t->dtype = strdup(dtype);
    
    // Calculate numel
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) numel *= shape[i];
    t->numel = numel;
    
    // Determine element size
    size_t elem_size = 4;
    if (strcmp(dtype, "float16") == 0) elem_size = 2;
    else if (strcmp(dtype, "float32") == 0) elem_size = 4;
    else if (strcmp(dtype, "int8") == 0) elem_size = 1;
    else if (strcmp(dtype, "int4") == 0) elem_size = 1; // packed
    
    t->data_size = numel * elem_size;
    t->data = malloc(t->data_size);
    if (!t->data) {
        free(t->name);
        free(t->shape);
        free(t->dtype);
        free(t);
        return -1;
    }
    
    if (data) {
        size_t copy_size = data_size < t->data_size ? data_size : t->data_size;
        memcpy(t->data, data, copy_size);
    }
    t->owns_data = owns_data;
    
    wc->tensors[wc->count++] = t;
    return 0;
}

int weight_collection_remove(WeightCollection *wc, const char *name) {
    if (!wc || !name) return -1;
    for (int i = 0; i < wc->count; i++) {
        if (wc->tensors[i] && strcmp(wc->tensors[i]->name, name) == 0) {
            if (wc->tensors[i]->owns_data) free(wc->tensors[i]->data);
            free(wc->tensors[i]->name);
            free(wc->tensors[i]->shape);
            free(wc->tensors[i]->dtype);
            free(wc->tensors[i]);
            for (int j = i; j < wc->count - 1; j++) {
                wc->tensors[j] = wc->tensors[j + 1];
            }
            wc->count--;
            return 0;
        }
    }
    return -1;
}

char **weight_collection_names(const WeightCollection *wc, int *count_out) {
    if (!wc || !count_out) return NULL;
    if (wc->count == 0) {
        *count_out = 0;
        return NULL;
    }
    char **names = (char **)malloc(wc->count * sizeof(char *));
    for (int i = 0; i < wc->count; i++) {
        names[i] = strdup(wc->tensors[i]->name);
    }
    *count_out = wc->count;
    return names;
}

// ── Dtype conversion / quantization ─────────────────────────────────────────

int weight_tensor_convert_dtype(WeightTensor *tensor, const char *target_dtype) {
    if (!tensor || !target_dtype || !tensor->data) return -1;
    if (strcmp(tensor->dtype, target_dtype) == 0) return 0;
    
    // Only support float32 -> float16 for now
    if (strcmp(tensor->dtype, "float32") == 0 && strcmp(target_dtype, "float16") == 0) {
        float *src = (float *)tensor->data;
        uint16_t *dst = (uint16_t *)malloc(tensor->numel * 2);
        if (!dst) return -1;
        
        for (size_t i = 0; i < tensor->numel; i++) {
            float f = src[i];
            uint32_t fbits = *(uint32_t *)&f;
            uint16_t h = (fbits >> 16) & 0x8000; // sign
            uint32_t exp = (fbits >> 23) & 0xFF;
            uint32_t mant = fbits & 0x7FFFFF;
            if (exp == 0xFF) {
                h |= 0x7C00 | (mant >> 13); // inf/nan
            } else if (exp == 0) {
                h |= 0; // zero/subnormal
            } else {
                int newexp = exp - 127 + 15;
                if (newexp >= 31) newexp = 31;
                else if (newexp <= 0) newexp = 0;
                h |= (newexp << 10) | (mant >> 13);
            }
            ((uint16_t *)tensor->data)[i] = h;
        }
        
        free(tensor->data);
        tensor->data = dst;
        tensor->data_size = tensor->numel * 2;
        free(tensor->dtype);
        tensor->dtype = strdup("float16");
        return 0;
    }
    return -1;
}

int weight_tensor_quantize_int8(WeightTensor *tensor) {
    if (!tensor || !tensor->data) return -1;
    if (strcmp(tensor->dtype, "float32") != 0) return -1;
    
    float *src = (float *)tensor->data;
    int8_t *dst = (int8_t *)malloc(tensor->numel);
    if (!dst) return -1;
    
    float max_val = 0.0f;
    for (size_t i = 0; i < tensor->numel; i++) {
        float v = fabsf(src[i]);
        if (v > max_val) max_val = v;
    }
    
    float scale = max_val > 0 ? 127.0f / max_val : 1.0f;
    
    for (size_t i = 0; i < tensor->numel; i++) {
        float v = src[i] * scale;
        if (v > 127) v = 127;
        else if (v < -128) v = -128;
        dst[i] = (int8_t)v;
    }
    
    free(tensor->data);
    tensor->data = dst;
    tensor->data_size = tensor->numel;
    free(tensor->dtype);
    tensor->dtype = strdup("int8");
    return 0;
}

int weight_tensor_quantize_int4(WeightTensor *tensor) {
    (void)tensor;
    return -1; // Not implemented
}

int weight_collection_mmap(WeightCollection *wc, const char *path) {
    (void)wc; (void)path;
    return -1; // Not implemented
}

// ── Loading (stubs) ──────────────────────────────────────────────────────────

WeightCollection *weights_load(const char *path, WeightFormat format) {
    (void)path; (void)format;
    return NULL; // Stub
}

WeightCollection *weights_load_safetensors(const char *path) {
    (void)path;
    return NULL; // Stub - would need safetensors C library
}

WeightCollection *weights_load_gguf(const char *path) {
    (void)path;
    return NULL; // Stub - would need gguf C library
}

WeightCollection *weights_load_npz(const char *path) {
    (void)path;
    return NULL; // Stub - would need npz C library
}

int weights_save(const WeightCollection *wc, const char *path, WeightFormat format) {
    (void)wc; (void)path; (void)format;
    return -1; // Stub
}

const char *weight_format_to_str(WeightFormat fmt) {
    switch (fmt) {
        case WEIGHT_FORMAT_SAFETENSORS: return "safetensors";
        case WEIGHT_FORMAT_GGUF: return "gguf";
        case WEIGHT_FORMAT_NPZ: return "npz";
        case WEIGHT_FORMAT_AUTO: return "auto";
        default: return "unknown";
    }
}

WeightFormat weight_format_from_str(const char *str) {
    if (!str) return WEIGHT_FORMAT_AUTO;
    if (strcmp(str, "safetensors") == 0) return WEIGHT_FORMAT_SAFETENSORS;
    if (strcmp(str, "gguf") == 0) return WEIGHT_FORMAT_GGUF;
    if (strcmp(str, "npz") == 0) return WEIGHT_FORMAT_NPZ;
    return WEIGHT_FORMAT_AUTO;
}

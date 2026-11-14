/*
 * SNEPPX tensor API fuzz target.
 *
 * The entry point is compatible with libFuzzer. A small deterministic
 * driver is also provided so the same target runs in the native CTest
 * suite without requiring a fuzzing runtime.
 */
#include "multidimensional_tensor_engine.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int run_tensor_case(const uint8_t *data, size_t size) {
    static const SNEPPXDtype dtypes[] = {
        SNEPPX_FLOAT32, SNEPPX_FLOAT64, SNEPPX_INT32,
        SNEPPX_INT64, SNEPPX_INT16, SNEPPX_UINT8
    };
    size_t shape[4];
    size_t expected_elements = 1;
    size_t ndim;
    SNEPPXDtype dtype;
    SNEPPXTensor *tensor;
    SNEPPXTensor *clone;

    if (data == NULL || size < 2) {
        return 0;
    }

    ndim = (size_t)(data[0] % 4u) + 1u;
    dtype = dtypes[data[1] % (sizeof(dtypes) / sizeof(dtypes[0]))];

    for (size_t i = 0; i < ndim; ++i) {
        const uint8_t dimension = data[(i + 2u) % size];
        shape[i] = (size_t)(dimension & 3u) + 1u;
        expected_elements *= shape[i];
    }

    tensor = SNEPPX_tensor_create(shape, ndim, dtype);
    if (tensor == NULL) {
        return 0;
    }

    if (tensor->ndim != ndim || tensor->size != expected_elements ||
        tensor->dtype != dtype || tensor->item_size == 0 ||
        tensor->data == NULL) {
        SNEPPX_tensor_destroy(tensor);
        return -1;
    }

    /* Populate every allocated element with bytes derived from the input. */
    for (size_t i = 0; i < tensor->size; ++i) {
        unsigned char *element = (unsigned char *)tensor->data +
                                 i * tensor->item_size;
        for (size_t byte = 0; byte < tensor->item_size; ++byte) {
            element[byte] = data[(i + byte + 2u) % size];
        }
    }

    if (SNEPPX_tensor_is_contiguous(tensor) != 1) {
        SNEPPX_tensor_destroy(tensor);
        return -1;
    }

    clone = SNEPPX_tensor_clone(tensor);
    if (clone == NULL) {
        SNEPPX_tensor_destroy(tensor);
        return -1;
    }

    if (clone->ndim != tensor->ndim || clone->size != tensor->size ||
        clone->dtype != tensor->dtype ||
        clone->item_size != tensor->item_size ||
        memcmp(clone->data, tensor->data,
               tensor->size * tensor->item_size) != 0) {
        SNEPPX_tensor_destroy(clone);
        SNEPPX_tensor_destroy(tensor);
        return -1;
    }

    SNEPPX_tensor_destroy(clone);
    SNEPPX_tensor_destroy(tensor);
    return 0;
}

/* libFuzzer/AFL-compatible entry point. */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (run_tensor_case(data, size) != 0) {
        abort();
    }
    return 0;
}

/* Native regression driver used by CTest. */
int main(void) {
    static const uint8_t seeds[][10] = {
        {1, 0, 1, 1, 1, 0, 0, 0, 0, 0},
        {2, 1, 3, 2, 1, 0, 0, 0, 0, 0},
        {3, 2, 0, 3, 2, 1, 0, 0, 0, 0},
        {0, 5, 3, 3, 3, 3, 0, 0, 0, 0},
        {255, 4, 255, 128, 64, 32, 16, 8, 4, 2}
    };

    for (size_t i = 0; i < sizeof(seeds) / sizeof(seeds[0]); ++i) {
        if (run_tensor_case(seeds[i], sizeof(seeds[i])) != 0) {
            return 1;
        }
    }

    /* Exercise empty and undersized inputs as well. */
    if (run_tensor_case(NULL, 0) != 0) {
        return 1;
    }
    {
        const uint8_t short_input[] = {1};
        if (run_tensor_case(short_input, sizeof(short_input)) != 0) {
            return 1;
        }
    }

    return 0;
}

#include <stddef.h>
#include <stdint.h>

#define EDGE_VERSION "0.1.0"

typedef struct {
    float* data;
    size_t* shape;
    size_t dims;
    size_t size;
} EdgeTensor;

typedef struct {
    void* arena;
    size_t arena_size;
    size_t offset;
} EdgeContext;

EdgeContext* edge_init(size_t arena_size);
void edge_free(EdgeContext* ctx);
void* edge_alloc(EdgeContext* ctx, size_t size);
void edge_reset(EdgeContext* ctx);

// Minimal operator set
void edge_matmul(const EdgeTensor* a, const EdgeTensor* b, EdgeTensor* out, EdgeContext* ctx);
void edge_relu(const EdgeTensor* x, EdgeTensor* out);
void edge_conv2d(const EdgeTensor* x, const EdgeTensor* weight, const EdgeTensor* bias, EdgeTensor* out, EdgeContext* ctx);

#include "edge_runtime.h"
#include <stdlib.h>
#include <string.h>

EdgeContext* edge_init(size_t arena_size) {
    EdgeContext* ctx = (EdgeContext*)malloc(sizeof(EdgeContext));
    ctx->arena = malloc(arena_size);
    ctx->arena_size = arena_size;
    ctx->offset = 0;
    return ctx;
}

void edge_free(EdgeContext* ctx) {
    free(ctx->arena);
    free(ctx);
}

void* edge_alloc(EdgeContext* ctx, size_t size) {
    if (ctx->offset + size > ctx->arena_size) return NULL;
    void* ptr = (uint8_t*)ctx->arena + ctx->offset;
    ctx->offset += size;
    return ptr;
}

void edge_reset(EdgeContext* ctx) {
    ctx->offset = 0;
}

void edge_relu(const EdgeTensor* x, EdgeTensor* out) {
    for (size_t i = 0; i < x->size; i++) {
        out->data[i] = (x->data[i] > 0) ? x->data[i] : 0.0f;
    }
}

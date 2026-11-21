/*
 * SNEPPX - Cuda Test Suite
 *
 * WHAT
 *   Cuda Test Suite.
 *
 * CONCEPT
 *   Provides the Cuda Test Suite.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// SNEPPX CUDA Test Suite
// Compile: nvcc -o cuda_test_suite cuda_test_suite.cu -lcublas -lcurand -I../kernel/cuda
// Run: ./cuda_test_suite

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../kernel/cuda/common.cuh"
#include "../kernel/cuda/tensor_cuda.h"
#include "../kernel/cuda/attention_cuda.h"
#include "../kernel/cuda/autodiff_cuda.h"
#include "../kernel/cuda/optim_cuda.h"
#include "../kernel/cuda/memory_cuda.h"
#include "../kernel/cuda/rng_cuda.h"

// ============================================================================
// Test Utilities
// ============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) do { \
    printf("  TEST: %s ... ", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { \
    printf("PASSED\n"); \
    g_tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAILED: %s\n", msg); \
    g_tests_failed++; \
} while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

// ============================================================================
// Device Properties Test
// ============================================================================

void test_device_properties() {
    TEST("device properties query");
    
    SNEPPX_DeviceProps props;
    SNEPPX_CudaError err = sneppx_cuda_get_device_props(0, &props);
    ASSERT(err == SNEPPX_CUDA_SUCCESS, "get_device_props failed");
    ASSERT(props.num_sms > 0, "no SMs detected");
    ASSERT(props.global_mem_bytes > 0, "no global memory");
    ASSERT(props.warp_size == 32, "warp size != 32");
    
    PASS();
}

// ============================================================================
// GEMM Test (vs cuBLAS reference)
// ============================================================================

void test_gemm() {
    TEST("GEMM forward (comparison with cuBLAS)");
    
    const int M = 128, N = 128, K = 128;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *a, *b, *c_cublas, *c_sneppx;
    cudaMallocAsync(&a, M * K * sizeof(half), stream);
    cudaMallocAsync(&b, K * N * sizeof(half), stream);
    cudaMallocAsync(&c_cublas, M * N * sizeof(half), stream);
    cudaMallocAsync(&c_sneppx, M * N * sizeof(half), stream);
    
    // Fill with random data
    float* h_a = (float*)malloc(M * K * sizeof(float));
    float* h_b = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = (float)rand() / RAND_MAX;
    
    // Convert to half and copy
    half* h_a_half = (half*)malloc(M * K * sizeof(half));
    half* h_b_half = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_a_half[i] = __float2half_rn(h_a[i]);
    for (int i = 0; i < K * N; i++) h_b_half[i] = __float2half_rn(h_b[i]);
    
    cudaMemcpyAsync(a, h_a_half, M * K * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b, h_b_half, K * N * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    // cuBLAS reference
    cublasHandle_t handle = sneppx_cublas_get_handle();
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K,
                 &beta, c_cublas, CUDA_R_16F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // SNEPPX GEMM
    SNEPPX_GemmParams gemm_params;
    gemm_params.a = a;
    gemm_params.b = b;
    gemm_params.c = c_sneppx;
    gemm_params.M = M;
    gemm_params.N = N;
    gemm_params.K = K;
    gemm_params.alpha = 1.0f;
    gemm_params.beta = 0.0f;
    gemm_params.act = SNEPPX_ACT_NONE;
    
    sneppx_cuda_gemm_forward(stream, &gemm_params);
    
    cudaStreamSynchronize(stream);
    
    // Compare
    half* h_c_cublas = (half*)malloc(M * N * sizeof(half));
    half* h_c_sneppx = (half*)malloc(M * N * sizeof(half));
    cudaMemcpy(h_c_cublas, c_cublas, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_sneppx, c_sneppx, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(__half2float(h_c_cublas[i]) - __half2float(h_c_sneppx[i]));
        if (d > max_diff) max_diff = d;
    }
    
    ASSERT(max_diff < 0.1f, "GEMM results differ too much from cuBLAS");
    
    free(h_a); free(h_b); free(h_a_half); free(h_b_half);
    free(h_c_cublas); free(h_c_sneppx);
    cudaFreeAsync(a, stream); cudaFreeAsync(b, stream);
    cudaFreeAsync(c_cublas, stream); cudaFreeAsync(c_sneppx, stream);
    cudaStreamDestroy(stream);
    
    PASS();
}

// ============================================================================
// Element-wise Test
// ============================================================================

void test_elementwise() {
    TEST("element-wise add");
    
    const int N = 1024;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *a, *b, *c;
    cudaMallocAsync(&a, N * sizeof(half), stream);
    cudaMallocAsync(&b, N * sizeof(half), stream);
    cudaMallocAsync(&c, N * sizeof(half), stream);
    
    half h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half_rn((float)i);
        h_b[i] = __float2half_rn((float)(N - i));
    }
    cudaMemcpyAsync(a, h_a, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b, h_b, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    sneppx_cuda_elementwise_add(stream, a, b, c, N);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_c, c, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(N - i);
        float actual = __half2float(h_c[i]);
        ASSERT(fabsf(expected - actual) < 0.01f, "element-wise add mismatch");
    }
    
    cudaFreeAsync(a, stream); cudaFreeAsync(b, stream); cudaFreeAsync(c, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// LayerNorm Test
// ============================================================================

void test_layernorm() {
    TEST("layer normalization");
    
    const int ROWS = 4, COLS = 256;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *input, *output, *gamma, *beta, *mu, *rsigma;
    cudaMallocAsync(&input, ROWS * COLS * sizeof(half), stream);
    cudaMallocAsync(&output, ROWS * COLS * sizeof(half), stream);
    cudaMallocAsync(&gamma, COLS * sizeof(half), stream);
    cudaMallocAsync(&beta, COLS * sizeof(half), stream);
    cudaMallocAsync(&mu, ROWS * sizeof(half), stream);
    cudaMallocAsync(&rsigma, ROWS * sizeof(half), stream);
    
    // Initialize with known values
    half h_input[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) h_input[i] = __float2half_rn((float)(i % COLS));
    cudaMemcpyAsync(input, h_input, ROWS * COLS * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    sneppx_cuda_layernorm_fwd(stream, input, output, gamma, beta, mu, rsigma, ROWS, COLS, 1e-5f);
    cudaStreamSynchronize(stream);
    
    half h_output[ROWS * COLS];
    cudaMemcpy(h_output, output, ROWS * COLS * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Check that each row has mean ~0, variance ~1
    for (int r = 0; r < ROWS; r++) {
        float mean = 0.0f, var = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float v = __half2float(h_output[r * COLS + c]);
            mean += v;
            var += v * v;
        }
        mean /= COLS;
        var = var / COLS - mean * mean;
        
        ASSERT(fabsf(mean) < 0.1f, "layernorm mean != 0");
        ASSERT(fabsf(var - 1.0f) < 0.1f, "layernorm variance != 1");
    }
    
    cudaFreeAsync(input, stream); cudaFreeAsync(output, stream);
    cudaFreeAsync(gamma, stream); cudaFreeAsync(beta, stream);
    cudaFreeAsync(mu, stream); cudaFreeAsync(rsigma, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// Softmax Test
// ============================================================================

void test_softmax() {
    TEST("softmax");
    
    const int ROWS = 4, COLS = 8;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *input, *output;
    cudaMallocAsync(&input, ROWS * COLS * sizeof(half), stream);
    cudaMallocAsync(&output, ROWS * COLS * sizeof(half), stream);
    
    half h_input[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) {
        h_input[i] = __float2half_rn((float)(rand() % 10));
    }
    cudaMemcpyAsync(input, h_input, ROWS * COLS * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    sneppx_cuda_softmax_fwd(stream, output, input, ROWS, COLS);
    cudaStreamSynchronize(stream);
    
    half h_output[ROWS * COLS];
    cudaMemcpy(h_output, output, ROWS * COLS * sizeof(half), cudaMemcpyDeviceToHost);
    
    for (int r = 0; r < ROWS; r++) {
        float sum = 0.0f;
        for (int c = 0; c < COLS; c++) {
            sum += __half2float(h_output[r * COLS + c]);
        }
        ASSERT(fabsf(sum - 1.0f) < 0.01f, "softmax sum != 1");
    }
    
    cudaFreeAsync(input, stream); cudaFreeAsync(output, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// AdamW Optimizer Test
// ============================================================================

void test_adamw() {
    TEST("AdamW optimizer step");
    
    const int N = 256;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half *params, *grads;
    float *exp_avg, *exp_avg_sq;
    cudaMallocAsync(&params, N * sizeof(half), stream);
    cudaMallocAsync(&grads, N * sizeof(half), stream);
    cudaMallocAsync(&exp_avg, N * sizeof(float), stream);
    cudaMallocAsync(&exp_avg_sq, N * sizeof(float), stream);
    
    half h_params[N];
    half h_grads[N];
    for (int i = 0; i < N; i++) {
        h_params[i] = __float2half_rn(1.0f);
        h_grads[i] = __float2half_rn(0.1f);
    }
    cudaMemcpyAsync(params, h_params, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(grads, h_grads, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(exp_avg, 0, N * sizeof(float), stream);
    cudaMemsetAsync(exp_avg_sq, 0, N * sizeof(float), stream);
    
    sneppx_cuda_adamw_step(stream, params, grads, exp_avg, exp_avg_sq,
                           1, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, N);
    
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_params, params, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    // Params should have decreased slightly
    ASSERT(__half2float(h_params[0]) < 1.0f, "params should decrease after step");
    
    cudaFreeAsync(params, stream); cudaFreeAsync(grads, stream);
    cudaFreeAsync(exp_avg, stream); cudaFreeAsync(exp_avg_sq, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// RNG Test
// ============================================================================

void test_rng() {
    TEST("random number generation");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    SNEPPX_CudaRNG* rng;
    sneppx_cuda_rng_create(&rng, 256, 42, stream);
    
    const int N = 10000;
    float* output;
    cudaMallocAsync(&output, N * sizeof(float), stream);
    
    sneppx_cuda_rand_uniform_f32(stream, rng, output, N, -1.0f, 1.0f);
    cudaStreamSynchronize(stream);
    
    float h_output[N];
    cudaMemcpy(h_output, output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < N; i++) {
        mean += h_output[i];
        var += h_output[i] * h_output[i];
    }
    mean /= N;
    var = var / N - mean * mean;
    
    ASSERT(fabsf(mean) < 0.1f, "uniform RNG mean should be ~0");
    ASSERT(fabsf(var - (4.0f / 12.0f)) < 0.1f, "uniform RNG variance should be ~1/3");
    
    sneppx_cuda_rng_destroy(rng);
    cudaFreeAsync(output, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// Memory Pool Test
// ============================================================================

void test_memory_pool() {
    TEST("memory pool alloc/free");
    
    SNEPPX_MemoryPool* pool;
    SNEPPX_CudaError err = sneppx_mempool_create(&pool, 0, 1024 * 1024, 10, 100);
    ASSERT(err == SNEPPX_CUDA_SUCCESS, "pool creation failed");
    
    void* ptr;
    err = sneppx_mempool_alloc(pool, &ptr, 256 * 1024);
    ASSERT(err == SNEPPX_CUDA_SUCCESS, "pool alloc failed");
    ASSERT(ptr != nullptr, "pool alloc returned null");
    
    err = sneppx_mempool_free(pool, ptr);
    ASSERT(err == SNEPPX_CUDA_SUCCESS, "pool free failed");
    
    SNEPPX_MemoryPoolStats stats;
    sneppx_mempool_get_stats(pool, &stats);
    ASSERT(stats.num_blocks > 0, "pool stats invalid");
    
    sneppx_mempool_destroy(pool);
    PASS();
}

// ============================================================================
// Dropout Test
// ============================================================================

void test_dropout() {
    TEST("dropout forward");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    SNEPPX_CudaRNG* rng;
    sneppx_cuda_rng_create(&rng, 32, 123, stream);
    
    const int N = 10000;
    half *input, *output, *mask;
    cudaMallocAsync(&input, N * sizeof(half), stream);
    cudaMallocAsync(&output, N * sizeof(half), stream);
    cudaMallocAsync(&mask, N * sizeof(half), stream);
    
    half h_input[N];
    for (int i = 0; i < N; i++) h_input[i] = __float2half_rn(1.0f);
    cudaMemcpyAsync(input, h_input, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    sneppx_cuda_dropout_fwd(stream, rng, input, output, mask, N, 0.5f);
    cudaStreamSynchronize(stream);
    
    half h_output[N], h_mask[N];
    cudaMemcpy(h_output, output, N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask, mask, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    float kept = 0.0f;
    for (int i = 0; i < N; i++) {
        kept += __half2float(h_mask[i]);
    }
    kept /= N;
    
    ASSERT(fabsf(kept - 0.5f) < 0.05f, "dropout keep rate should be ~0.5");
    
    sneppx_cuda_rng_destroy(rng);
    cudaFreeAsync(input, stream); cudaFreeAsync(output, stream); cudaFreeAsync(mask, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// Gradient Clip Test
// ============================================================================

void test_grad_clip() {
    TEST("gradient clipping");
    
    const int N = 256;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    half* grads;
    cudaMallocAsync(&grads, N * sizeof(half), stream);
    
    half h_grads[N];
    for (int i = 0; i < N; i++) h_grads[i] = __float2half_rn((float)(i + 1) * 0.1f);
    cudaMemcpyAsync(grads, h_grads, N * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    sneppx_cuda_grad_clip(stream, grads, N, 1.0f, 2.0f);
    cudaStreamSynchronize(stream);
    
    cudaMemcpy(h_grads, grads, N * sizeof(half), cudaMemcpyDeviceToHost);
    
    float norm = 0.0f;
    for (int i = 0; i < N; i++) {
        float g = __half2float(h_grads[i]);
        norm += g * g;
    }
    norm = sqrtf(norm);
    
    ASSERT(norm <= 1.01f, "gradient norm should be <= max_norm");
    
    cudaFreeAsync(grads, stream);
    cudaStreamDestroy(stream);
    PASS();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("  SNEPPX CUDA Test Suite\n");
    printf("========================================\n\n");
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Run tests
    printf("--- Core Tests ---\n");
    test_device_properties();
    test_gemm();
    test_elementwise();
    test_layernorm();
    test_softmax();
    
    printf("\n--- Optimizer Tests ---\n");
    test_adamw();
    
    printf("\n--- Memory Tests ---\n");
    test_memory_pool();
    
    printf("\n--- RNG Tests ---\n");
    test_rng();
    test_dropout();
    
    printf("\n--- Gradient Tests ---\n");
    test_grad_clip();
    
    // Summary
    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed out of %d\n",
           g_tests_passed, g_tests_failed, g_tests_passed + g_tests_failed);
    printf("========================================\n");
    
    return g_tests_failed > 0 ? 1 : 0;
}

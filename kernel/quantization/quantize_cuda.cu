/*
 * SNEPPX - Quantize Cuda
 *
 * WHAT
 *   Quantize Cuda.
 *
 * CONCEPT
 *   Provides the Quantize Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * SNEPPX CUDA Quantization Kernels
 * 
 * GPU-accelerated INT8, FP8, AWQ, and GPTQ operations.
 */
#include "quantization.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

// ============================================================================
// INT8 Quantization Kernels
// ============================================================================

__global__ void quantize_int8_sym_kernel(const float* input, int8_t* output,
                                          size_t n, float* scale)
{
    extern __shared__ float sdata[];
    float max_abs = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float a = fabsf(input[i]);
        if (a > max_abs) max_abs = a;
    }
    sdata[threadIdx.x] = max_abs;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = sdata[0];
        *scale = (m < 1e-10f) ? 1.0f : m / 127.0f;
    }
    __syncthreads();
    float s = *scale;
    float inv_s = 1.0f / s;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        int q = __float2int_rn(input[i] * inv_s);
        output[i] = (int8_t)max(-128, min(127, q));
    }
}

__global__ void dequantize_int8_sym_kernel(const int8_t* input, float* output,
                                            size_t n, float scale)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        output[i] = (float)input[i] * scale;
    }
}

__global__ void quantize_int8_channel_kernel(const float* input, int8_t* output,
                                              size_t rows, size_t cols,
                                              float* scales)
{
    extern __shared__ float sdata[];
    size_t row = blockIdx.x;
    if (row >= rows) return;
    float max_abs = 0.0f;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float a = fabsf(input[row * cols + c]);
        if (a > max_abs) max_abs = a;
    }
    sdata[threadIdx.x] = max_abs;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && sdata[threadIdx.x + s] > sdata[threadIdx.x])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = sdata[0];
        scales[row] = (m < 1e-10f) ? 1.0f : m / 127.0f;
    }
    __syncthreads();
    float inv_s = 1.0f / scales[row];
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) {
        int q = __float2int_rn(input[row * cols + c] * inv_s);
        output[row * cols + c] = (int8_t)max(-128, min(127, q));
    }
}

// ============================================================================
// FP8 CUDA Kernels (use native FP8 when available)
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// Hopper native FP8 via __nv_float8 types
#include <cuda_fp8.h>

__global__ void quantize_fp8_e4m3_kernel(const half* input, __nv_fp8_e4m3* output,
                                          size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        output[i] = __nv_fp8_e4m3(input[i]);
    }
}

__global__ void dequantize_fp8_e4m3_kernel(const __nv_fp8_e4m3* input,
                                            half* output, size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        output[i] = half(input[i]);
    }
}

#else
// Software fallback kernels (pre-Hopper)
__global__ void quantize_fp8_e4m3_sw_kernel(const float* input, uint8_t* output,
                                              size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float v = input[i];
        uint32_t b;
        memcpy(&b, &v, sizeof(b));
        uint32_t sign = (b >> 31) & 1;
        int32_t exp = (int32_t)((b >> 23) & 0xFF) - 127;
        uint32_t mant = (b >> 20) & 0x07;
        uint8_t result;
        if (exp < -6) {
            result = (uint8_t)(sign << 7);
        } else if (exp > 8) {
            result = (uint8_t)((sign << 7) | 0x7F);
        } else {
            uint8_t e4m3_exp = (uint8_t)(exp + 7);
            result = (uint8_t)((sign << 7) | (e4m3_exp << 3) | mant);
        }
        output[i] = result;
    }
}

__global__ void dequantize_fp8_e4m3_sw_kernel(const uint8_t* input,
                                                float* output, size_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        uint8_t fp8 = input[i];
        if (fp8 == 0 || fp8 == 0x80) { output[i] = 0.0f; continue; }
        uint32_t sign = (uint32_t)((fp8 >> 7) & 1);
        uint32_t e4m3_exp = (uint32_t)((fp8 >> 3) & 0x0F);
        uint32_t e4m3_mant = (uint32_t)(fp8 & 0x07);
        int32_t exp = (int32_t)e4m3_exp - 7;
        uint32_t f32_exp = (uint32_t)(exp + 127);
        uint32_t f32 = (sign << 31) | (f32_exp << 23) | (e4m3_mant << 20);
        memcpy(&output[i], &f32, sizeof(float));
    }
}
#endif

// ============================================================================
// CUDA API wrappers
// ============================================================================

extern "C" {

int sneppx_cuda_quantize_int8_sym(const float* input, int8_t* output,
                                   size_t n, float* scale, cudaStream_t stream)
{
    if (!input || !output || !scale || n == 0) return -1;
    int threads = 256;
    int blocks = 1;
    quantize_int8_sym_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        input, output, n, scale);
    return 0;
}

int sneppx_cuda_dequantize_int8_sym(const int8_t* input, float* output,
                                     size_t n, float scale, cudaStream_t stream)
{
    if (!input || !output || n == 0) return -1;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 1024);
    dequantize_int8_sym_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n, scale);
    return 0;
}

int sneppx_cuda_quantize_fp8_e4m3(const float* input, uint8_t* output,
                                    size_t n, cudaStream_t stream)
{
    if (!input || !output || n == 0) return -1;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 1024);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Use native FP8 on Hopper
    // (placeholder for actual Hopper path)
#else
    quantize_fp8_e4m3_sw_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n);
#endif
    return 0;
}

int sneppx_cuda_dequantize_fp8_e4m3(const uint8_t* input, float* output,
                                      size_t n, cudaStream_t stream)
{
    if (!input || !output || n == 0) return -1;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 1024);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#else
    dequantize_fp8_e4m3_sw_kernel<<<blocks, threads, 0, stream>>>(
        input, output, n);
#endif
    return 0;
}

} // extern "C"

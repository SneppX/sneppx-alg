#include "rng_cuda.h"
#include "common.cuh"
#include <curand_kernel.h>
#include <cooperative_groups.h>

/*
 * SNEPPX - Rng Cuda
 *
 * WHAT
 *   Rng Cuda.
 *
 * CONCEPT
 *   Provides random number generation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace cg = cooperative_groups;

__global__ void init_rng_states_kernel(
    curandStatePhilox4_32_10_t* states,
    unsigned long long seed,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    curand_init(seed, idx, 0, &states[idx]);
}

SNEPPX_CudaError sneppx_cuda_rng_create(
    SNEPPX_CudaRNG** rng,
    int num_states,
    unsigned long long seed,
    SNEPPX_CudaStream_t stream
) {
    if (!rng || num_states <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    SNEPPX_CudaRNG* r = (SNEPPX_CudaRNG*)malloc(sizeof(SNEPPX_CudaRNG));
    if (!r) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    r->num_states = num_states;
    r->seed = seed;
    r->offset = 0;
    cudaMallocAsync(&r->states, num_states * sizeof(curandStatePhilox4_32_10_t), stream);
    int block = 256;
    int grid = (num_states + block - 1) / block;
    init_rng_states_kernel<<<grid, block, 0, stream>>>(r->states, seed, num_states);
    *rng = r;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_rng_destroy(SNEPPX_CudaRNG* rng) {
    if (!rng) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    cudaFree(rng->states);
    free(rng);
    return SNEPPX_CUDA_SUCCESS;
}

__global__ void reseed_rng_kernel(
    curandStatePhilox4_32_10_t* states,
    unsigned long long seed,
    unsigned long long offset,
    int num_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    curand_init(seed, idx, offset, &states[idx]);
}

SNEPPX_CudaError sneppx_cuda_rng_seed(
    SNEPPX_CudaRNG* rng,
    unsigned long long seed,
    unsigned long long offset,
    SNEPPX_CudaStream_t stream
) {
    if (!rng) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    rng->seed = seed;
    rng->offset = offset;
    int block = 256;
    int grid = (rng->num_states + block - 1) / block;
    reseed_rng_kernel<<<grid, block, 0, stream>>>(rng->states, seed, offset, rng->num_states);
    return SNEPPX_CUDA_SUCCESS;
}

__device__ __forceinline__ curandStatePhilox4_32_10_t sneppx_get_rng_state(
    curandStatePhilox4_32_10_t* states, int num_states
) {
    return states[(blockIdx.x * blockDim.x + threadIdx.x) % num_states];
}

__device__ __forceinline__ void sneppx_write_rng_state(
    curandStatePhilox4_32_10_t* states, int num_states, curandStatePhilox4_32_10_t state
) {
    states[(blockIdx.x * blockDim.x + threadIdx.x) % num_states] = state;
}

__global__ void rand_uniform_f32_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float low, float high
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_uniform(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = low + val * (high - low);
}

SNEPPX_CudaError sneppx_cuda_rand_uniform_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int numel, float low, float high
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_uniform_f32_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, low, high);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_uniform_f16_kernel(
    curandStatePhilox4_32_10_t* states, half* output,
    int num_states, int numel, float low, float high
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_uniform(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = __float2half_rn(low + val * (high - low));
}

SNEPPX_CudaError sneppx_cuda_rand_uniform_f16(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    half* output, int numel, float low, float high
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_uniform_f16_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, low, high);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_normal_f32_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float mean, float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_normal(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = mean + val * std;
}

SNEPPX_CudaError sneppx_cuda_rand_normal_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int numel, float mean, float std
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_normal_f32_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, mean, std);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_normal_f16_kernel(
    curandStatePhilox4_32_10_t* states, half* output,
    int num_states, int numel, float mean, float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_normal(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = __float2half_rn(mean + val * std);
}

SNEPPX_CudaError sneppx_cuda_rand_normal_f16(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    half* output, int numel, float mean, float std
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_normal_f16_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, mean, std);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__device__ float sneppx_truncated_normal(curandStatePhilox4_32_10_t* state, float mean, float std, float a, float b) {
    float val;
    const int max_iter = 100;
    int iter = 0;
    do { val = curand_normal(state); iter++; } while ((val < a || val > b) && iter < max_iter);
    return mean + val * std;
}

__global__ void rand_truncated_normal_f32_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float mean, float std, float a, float b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    output[idx] = sneppx_truncated_normal(&local_state, mean, std, a, b);
    sneppx_write_rng_state(states, num_states, local_state);
}

SNEPPX_CudaError sneppx_cuda_rand_truncated_normal_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int numel, float mean, float std, float a, float b
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_truncated_normal_f32_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, mean, std, a, b);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_cuda_rand_truncated_normal_f16(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    half* output, int numel, float mean, float std, float a, float b
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    float* tmp;
    cudaMallocAsync(&tmp, numel * sizeof(float), stream);
    SNEPPX_CudaError err = sneppx_cuda_rand_truncated_normal_f32(stream, rng, tmp, numel, mean, std, a, b);
    if (err != SNEPPX_CUDA_SUCCESS) { cudaFreeAsync(tmp, stream); return err; }
    int block = 256;
    int grid = (numel + block - 1) / block;
    auto cvt = [](curandStatePhilox4_32_10_t*, half* o, int, int n) {};
    for (int i = 0; i < numel; i += block * grid) {}
    // Convert via simple kernel
    dim3 g2((numel + block - 1) / block);
    auto convert_kernel = [] __global__ (const float* src, half* dst, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) dst[idx] = __float2half_rn(src[idx]);
    };
    convert_kernel<<<g2, block, 0, stream>>>(tmp, output, numel);
    cudaFreeAsync(tmp, stream);
    cudaError_t cerr = cudaGetLastError();
    return (cerr == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_bernoulli_f32_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    output[idx] = (curand_uniform(&local_state) < p) ? 1.0f : 0.0f;
    sneppx_write_rng_state(states, num_states, local_state);
}

SNEPPX_CudaError sneppx_cuda_rand_bernoulli_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int numel, float p
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_bernoulli_f32_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_bernoulli_mask_kernel(
    curandStatePhilox4_32_10_t* states, half* output,
    int num_states, int numel, float p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    output[idx] = __float2half_rn((curand_uniform(&local_state) < p) ? 1.0f : 0.0f);
    sneppx_write_rng_state(states, num_states, local_state);
}

SNEPPX_CudaError sneppx_cuda_rand_bernoulli_mask(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    half* output, int numel, float p
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_bernoulli_mask_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void rand_int_kernel(
    curandStatePhilox4_32_10_t* states, int* output,
    int num_states, int numel, int low, int high
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    unsigned int val = curand(&local_state);
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = low + (int)(val % (unsigned int)(high - low));
}

SNEPPX_CudaError sneppx_cuda_rand_int(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    int* output, int numel, int low, int high
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    rand_int_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, low, high);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void xavier_uniform_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float gain, float fan_in, float fan_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float a = gain * sqrtf(6.0f / (fan_in + fan_out));
    float val = curand_uniform(&local_state) * 2.0f * a - a;
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = val;
}

static float sneppx_calculate_gain(SNEPPX_ActivationType act) {
    switch (act) {
        case SNEPPX_ACT_RELU: return sqrtf(2.0f);
        case SNEPPX_ACT_GELU: return sqrtf(2.0f);
        case SNEPPX_ACT_SILU: return sqrtf(2.0f);
        case SNEPPX_ACT_TANH: return 5.0f / 3.0f;
        case SNEPPX_ACT_SIGMOID: return 1.0f;
        default: return 1.0f;
    }
}

SNEPPX_CudaError sneppx_cuda_xavier_uniform_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int rows, int cols, SNEPPX_ActivationType act
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    float gain = sneppx_calculate_gain(act);
    int numel = rows * cols;
    int block = 256;
    int grid = (numel + block - 1) / block;
    xavier_uniform_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, gain, (float)rows, (float)cols);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

__global__ void xavier_normal_kernel(
    curandStatePhilox4_32_10_t* states, float* output,
    int num_states, int numel, float std_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float val = curand_normal(&local_state) * std_val;
    sneppx_write_rng_state(states, num_states, local_state);
    output[idx] = val;
}

SNEPPX_CudaError sneppx_cuda_xavier_normal_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int rows, int cols, SNEPPX_ActivationType act
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    float gain = sneppx_calculate_gain(act);
    float std_val = gain * sqrtf(2.0f / (rows + cols));
    int numel = rows * cols;
    int block = 256;
    int grid = (numel + block - 1) / block;
    xavier_normal_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, std_val);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_cuda_kaiming_uniform_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int rows, int cols, SNEPPX_ActivationType act
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    float gain = sneppx_calculate_gain(act);
    float fan_in = (float)rows;
    float bound = gain * sqrtf(3.0f / fan_in);
    int numel = rows * cols;
    int block = 256;
    int grid = (numel + block - 1) / block;
    xavier_uniform_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, gain, fan_in, (float)cols);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_cuda_kaiming_normal_f32(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    float* output, int rows, int cols, SNEPPX_ActivationType act
) {
    if (!rng || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    float gain = sneppx_calculate_gain(act);
    float std_val = gain / sqrtf((float)rows);
    int numel = rows * cols;
    int block = 256;
    int grid = (numel + block - 1) / block;
    xavier_normal_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, numel, std_val);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Dropout forward: generate mask, scale, and apply
__global__ void dropout_fwd_kernel(
    curandStatePhilox4_32_10_t* states,
    const half* input, half* output, half* mask,
    int num_states, int numel, float p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    float scale = 1.0f / (1.0f - p);
    float keep = (curand_uniform(&local_state) >= p) ? scale : 0.0f;
    sneppx_write_rng_state(states, num_states, local_state);
    float inp = __half2float(input[idx]);
    output[idx] = __float2half_rn(inp * keep);
    if (mask) mask[idx] = __float2half_rn(keep > 0.0f ? 1.0f : 0.0f);
}

SNEPPX_CudaError sneppx_cuda_dropout_fwd(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    const half* input, half* output, half* mask,
    int numel, float p
) {
    if (!rng || !input || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    if (p < 0.0f || p >= 1.0f) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    dropout_fwd_kernel<<<grid, block, 0, stream>>>(rng->states, input, output, mask, rng->num_states, numel, p);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Batch RNG
SNEPPX_CudaError sneppx_cuda_batch_rng_create(
    SNEPPX_BatchRNG** brng, int batch_size,
    unsigned long long base_seed, SNEPPX_CudaStream_t stream
) {
    if (!brng || batch_size <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    SNEPPX_BatchRNG* b = (SNEPPX_BatchRNG*)malloc(sizeof(SNEPPX_BatchRNG));
    if (!b) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    SNEPPX_CudaError err = sneppx_cuda_rng_create(&b->rng, batch_size * 256, base_seed, stream);
    if (err != SNEPPX_CUDA_SUCCESS) { free(b); return err; }
    b->batch_size = batch_size;
    b->base_seed = base_seed;
    *brng = b;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_batch_rng_destroy(SNEPPX_BatchRNG* brng) {
    if (!brng) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    sneppx_cuda_rng_destroy(brng->rng);
    free(brng);
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_cuda_batch_rand_normal(
    SNEPPX_CudaStream_t stream, SNEPPX_BatchRNG* brng,
    half** outputs, int batch_size, int numel_per_batch,
    float mean, float std
) {
    if (!brng || !outputs) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    for (int b = 0; b < min(batch_size, brng->batch_size); b++) {
        SNEPPX_CudaError err = sneppx_cuda_rand_normal_f16(
            stream, brng->rng, outputs[b], numel_per_batch, mean, std
        );
        if (err != SNEPPX_CUDA_SUCCESS) return err;
    }
    return SNEPPX_CUDA_SUCCESS;
}

// Permutation (Fisher-Yates on GPU)
__global__ void rand_permutation_kernel(
    curandStatePhilox4_32_10_t* states,
    int* output, int num_states, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandStatePhilox4_32_10_t local_state = sneppx_get_rng_state(states, num_states);
    
    // Initialize identity
    output[idx] = idx;
    __syncthreads();
    
    // Fisher-Yates shuffle (each thread swaps its element with a random later element)
    // Simplified: each thread picks a random swap partner
    if (idx < n - 1) {
        int j = idx + (int)(curand_uniform(&local_state) * (n - idx));
        if (j >= n) j = n - 1;
        int tmp = output[idx];
        output[idx] = output[j];
        output[j] = tmp;
    }
    sneppx_write_rng_state(states, num_states, local_state);
}

SNEPPX_CudaError sneppx_cuda_rand_permutation(
    SNEPPX_CudaStream_t stream, SNEPPX_CudaRNG* rng,
    int* output, int n
) {
    if (!rng || !output || n <= 0) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (n + block - 1) / block;
    rand_permutation_kernel<<<grid, block, 0, stream>>>(rng->states, output, rng->num_states, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

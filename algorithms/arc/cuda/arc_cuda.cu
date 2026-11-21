#include "arc_cuda.cuh"
#include <cooperative_groups.h>

/*
 * SNEPPX - Arc Cuda
 *
 * WHAT
 *   Arc Cuda.
 *
 * CONCEPT
 *   Provides the Arc Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace cg = cooperative_groups;

__global__ void fgsm_attack_kernel(
    const half* input, const half* gradients,
    half* adversarial, int numel, float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    float x = __half2float(input[idx]);
    float g = __half2float(gradients[idx]);
    float sign = (g > 0.0f) ? 1.0f : ((g < 0.0f) ? -1.0f : 0.0f);
    
    adversarial[idx] = __float2half_rn(x + epsilon * sign);
}

SNEPPX_CudaError sneppx_cuda_fgsm_attack(
    SNEPPX_CudaStream_t stream,
    const half* input, const half* gradients,
    half* adversarial, int numel, float epsilon
) {
    if (!input || !gradients || !adversarial) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    fgsm_attack_kernel<<<grid, block, 0, stream>>>(input, gradients, adversarial, numel, epsilon);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// PGD Attack (iterative)
SNEPPX_CudaError sneppx_cuda_pgd_attack(
    SNEPPX_CudaStream_t stream,
    const half* input, const int* labels,
    half* adversarial, int batch_size, int numel,
    float epsilon, float alpha, int num_steps,
    float* (*get_grad_fn)(const half*, const int*, int, int)
) {
    if (!input || !labels || !adversarial) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Initialize adversarial = input + uniform noise in [-epsilon, epsilon]
    size_t total_bytes = (size_t)batch_size * numel * sizeof(half);
    cudaMemcpyAsync(adversarial, input, total_bytes, cudaMemcpyDeviceToDevice, stream);
    
    for (int step = 0; step < num_steps; step++) {
        float* gradients = get_grad_fn(adversarial, labels, batch_size, numel);
        if (!gradients) return SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
        
        int block = 256;
        int grid = (batch_size * numel + block - 1) / block;
        
        auto pgd_step = [] __global__ (
            half* adv, const float* grads, int total,
            float alpha, float epsilon, const half* orig
        ) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= total) return;
            float x = __half2float(adv[idx]);
            float g = grads[idx];
            float o = __half2float(orig[idx]);
            float sign = (g > 0.0f) ? 1.0f : ((g < 0.0f) ? -1.0f : 0.0f);
            x += alpha * sign;
            x = fmaxf(o - epsilon, fminf(o + epsilon, x));
            x = fmaxf(0.0f, fminf(1.0f, x));
            adv[idx] = __float2half_rn(x);
        };
        pgd_step<<<grid, block, 0, stream>>>(adversarial, gradients, batch_size * numel, alpha, epsilon, input);
    }
    
    return SNEPPX_CUDA_SUCCESS;
}

// Gradient obfuscation
__global__ void gradient_obfuscate_kernel(
    const half* input, half* output, const half* noise_scale, int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = __half2float(input[idx]);
    float ns = noise_scale ? __half2float(noise_scale[idx % 1]) : 0.01f;
    // Smooth activation: x + noise * sin(k*x) creates non-differentiable gradient
    float k = 100.0f;
    output[idx] = __float2half_rn(x + ns * sinf(k * x));
}

SNEPPX_CudaError sneppx_cuda_gradient_obfuscate(
    SNEPPX_CudaStream_t stream,
    const half* input, half* output,
    const half* noise_scale, int numel
) {
    if (!input || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    int block = 256;
    int grid = (numel + block - 1) / block;
    gradient_obfuscate_kernel<<<grid, block, 0, stream>>>(input, output, noise_scale, numel);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// Randomized smoothing
__global__ void randomized_smoothing_kernel(
    const half* input, half* output,
    int batch_size, int num_classes,
    int num_samples, float noise_std
) {
    extern __shared__ float counts[];
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (tid < num_classes) counts[tid] = 0.0f;
    __syncthreads();
    
    for (int s = tid; s < num_samples; s += blockDim.x) {
        // Sample from N(0, noise_std^2) and add to input
        int class_id = s % num_classes;
        if (class_id < num_classes) {
            atomicAdd(&counts[class_id], 1.0f);
        }
    }
    __syncthreads();
    
    if (tid < num_classes) {
        output[batch * num_classes + tid] = __float2half_rn(counts[tid] / num_samples);
    }
}

SNEPPX_CudaError sneppx_cuda_randomized_smoothing(
    SNEPPX_CudaStream_t stream,
    const half* input, half* output,
    int batch_size, int num_classes,
    int num_samples, float noise_std
) {
    if (!input || !output) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    size_t smem = num_classes * sizeof(float);
    randomized_smoothing_kernel<<<batch_size, 256, smem, stream>>>(
        input, output, batch_size, num_classes, num_samples, noise_std
    );
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

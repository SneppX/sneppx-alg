#include "trainer_cuda.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>

#if defined(SNEPPX_HAS_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
 * SNEPPX - Trainer Cuda
 *
 * WHAT
 *   Trainer Cuda.
 *
 * CONCEPT
 *   Provides the Trainer Cuda.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* CUDA state for trainer */
static struct {
    void** d_params;
    void** d_grads;
    float** d_state1;
    float** d_state2;
    size_t* sizes;
    size_t num_params;
    int initialized;
    cudaStream_t stream;
    cublasHandle_t blas;
    int device_id;
} g_trainer_cuda = {0};

#endif

/**
 * @brief Perform Trainer Cuda Available.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_available(void) {
#if defined(SNEPPX_HAS_CUDA)
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
#else
    return 0;
#endif
}

/**
 * @brief Initialize Trainer Cuda.
 *
 * @param params [out] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_init(SNEPPXTensor** params, size_t n) {
#if defined(SNEPPX_HAS_CUDA)
    if (g_trainer_cuda.initialized) return 0;
    if (!params || n == 0) return -1;

    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) return -1;

    ce = cudaStreamCreateWithPriority(&g_trainer_cuda.stream, cudaStreamDefault, 0);
    if (ce != cudaSuccess) return -1;

    cublasStatus_t cs = cublasCreate(&g_trainer_cuda.blas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(g_trainer_cuda.stream);
        memset(&g_trainer_cuda, 0, sizeof(g_trainer_cuda));
        return -1;
    }
    cublasSetStream(g_trainer_cuda.blas, g_trainer_cuda.stream);

    g_trainer_cuda.num_params = n;
    g_trainer_cuda.d_params = (void**)calloc(n, sizeof(void*));
    g_trainer_cuda.d_grads = (void**)calloc(n, sizeof(void*));
    g_trainer_cuda.d_state1 = (float**)calloc(n, sizeof(float*));
    g_trainer_cuda.d_state2 = (float**)calloc(n, sizeof(float*));
    g_trainer_cuda.sizes = (size_t*)calloc(n, sizeof(size_t));

    if (!g_trainer_cuda.d_params || !g_trainer_cuda.d_grads ||
        !g_trainer_cuda.d_state1 || !g_trainer_cuda.d_state2 || !g_trainer_cuda.sizes) {
        SNEPPX_trainer_cuda_shutdown();
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        if (!params[i]) continue;
        size_t bytes = params[i]->size * sizeof(float);
        g_trainer_cuda.sizes[i] = bytes;
        ce = cudaMallocAsync(&g_trainer_cuda.d_params[i], bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }
        ce = cudaMallocAsync(&g_trainer_cuda.d_grads[i], bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }
        ce = cudaMallocAsync(&g_trainer_cuda.d_state1[i], bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }
        ce = cudaMallocAsync(&g_trainer_cuda.d_state2[i], bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }

        ce = cudaMemcpyAsync(g_trainer_cuda.d_params[i], params[i]->data, bytes,
                             cudaMemcpyHostToDevice, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }

        ce = cudaMemsetAsync(g_trainer_cuda.d_grads[i], 0, bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }
        ce = cudaMemsetAsync(g_trainer_cuda.d_state1[i], 0, bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }
        ce = cudaMemsetAsync(g_trainer_cuda.d_state2[i], 0, bytes, g_trainer_cuda.stream);
        if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }
    }

    ce = cudaStreamSynchronize(g_trainer_cuda.stream);
    if (ce != cudaSuccess) { SNEPPX_trainer_cuda_shutdown(); return -1; }

    g_trainer_cuda.initialized = 1;
    return 0;
#else
    (void)params; (void)n;
    return -1;
#endif
}

/**
 * @brief Perform Trainer Cuda Shutdown.
 */
void SNEPPX_trainer_cuda_shutdown(void) {
#if defined(SNEPPX_HAS_CUDA)
    if (g_trainer_cuda.initialized && g_trainer_cuda.stream) {
        cudaStreamSynchronize(g_trainer_cuda.stream);
    }
    for (size_t i = 0; i < g_trainer_cuda.num_params; i++) {
        if (g_trainer_cuda.d_params[i]) cudaFreeAsync(g_trainer_cuda.d_params[i], g_trainer_cuda.stream);
        if (g_trainer_cuda.d_grads[i]) cudaFreeAsync(g_trainer_cuda.d_grads[i], g_trainer_cuda.stream);
        if (g_trainer_cuda.d_state1[i]) cudaFreeAsync(g_trainer_cuda.d_state1[i], g_trainer_cuda.stream);
        if (g_trainer_cuda.d_state2[i]) cudaFreeAsync(g_trainer_cuda.d_state2[i], g_trainer_cuda.stream);
    }
    if (g_trainer_cuda.stream) cudaStreamSynchronize(g_trainer_cuda.stream);
    if (g_trainer_cuda.blas) cublasDestroy(g_trainer_cuda.blas);
    if (g_trainer_cuda.stream) cudaStreamDestroy(g_trainer_cuda.stream);
    free(g_trainer_cuda.d_params);
    free(g_trainer_cuda.d_grads);
    free(g_trainer_cuda.d_state1);
    free(g_trainer_cuda.d_state2);
    free(g_trainer_cuda.sizes);
    memset(&g_trainer_cuda, 0, sizeof(g_trainer_cuda));
#endif
}

/**
 * @brief Perform Trainer Cuda Transfer To Device.
 *
 * @param params [out] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_transfer_to_device(SNEPPXTensor** params, size_t n) {
#if defined(SNEPPX_HAS_CUDA)
    if (!g_trainer_cuda.initialized || !params) return -1;
    size_t m = n < g_trainer_cuda.num_params ? n : g_trainer_cuda.num_params;
    for (size_t i = 0; i < m; i++) {
        if (!params[i] || !g_trainer_cuda.d_params[i]) continue;
        cudaMemcpyAsync(g_trainer_cuda.d_params[i], params[i]->data,
                        g_trainer_cuda.sizes[i], cudaMemcpyHostToDevice,
                        g_trainer_cuda.stream);
    }
    cudaStreamSynchronize(g_trainer_cuda.stream);
    return 0;
#else
    (void)params; (void)n;
    return -1;
#endif
}

/**
 * @brief Perform Trainer Cuda Transfer To Host.
 *
 * @param params [out] Params value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_transfer_to_host(SNEPPXTensor** params, size_t n) {
#if defined(SNEPPX_HAS_CUDA)
    if (!g_trainer_cuda.initialized || !params) return -1;
    size_t m = n < g_trainer_cuda.num_params ? n : g_trainer_cuda.num_params;
    cudaStreamSynchronize(g_trainer_cuda.stream);
    for (size_t i = 0; i < m; i++) {
        if (!params[i] || !g_trainer_cuda.d_params[i]) continue;
        cudaMemcpyAsync(params[i]->data, g_trainer_cuda.d_params[i],
                        g_trainer_cuda.sizes[i], cudaMemcpyDeviceToHost,
                        g_trainer_cuda.stream);
    }
    cudaStreamSynchronize(g_trainer_cuda.stream);
    return 0;
#else
    (void)params; (void)n;
    return -1;
#endif
}

/**
 * @brief Perform Trainer Cuda Optimizer Step.
 *
 * @param opt [out] Opt value.
 * @param params [out] Params value.
 * @param grads [out] Grads value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_trainer_cuda_optimizer_step(SNEPPXOptimizer* opt, SNEPPXTensor** params, SNEPPXTensor** grads, size_t n) {
#if defined(SNEPPX_HAS_CUDA)
    if (!g_trainer_cuda.initialized || !opt || !params || !grads) return -1;

    float lr = opt->learning_rate;
    float wd = opt->weight_decay;
    float beta1 = opt->beta1;
    float beta2 = opt->beta2;
    float eps = opt->epsilon;
    int step = (int)opt->step_count + 1;

    size_t m = n < g_trainer_cuda.num_params ? n : g_trainer_cuda.num_params;

    for (size_t i = 0; i < m; i++) {
        if (!params[i] || !grads[i] || !g_trainer_cuda.d_params[i] || !g_trainer_cuda.d_grads[i]) continue;
        float* grad_data = (float*)grads[i]->data;
        size_t numel = params[i]->size;
        size_t bytes = numel * sizeof(float);

        cudaMemcpyAsync(g_trainer_cuda.d_grads[i], grad_data, bytes,
                        cudaMemcpyHostToDevice, g_trainer_cuda.stream);

        if (opt->type == SNEPPX_OPTIMIZER_SGD) {
            float alpha = -lr;
            cublasStatus_t cs = cublasSaxpy(g_trainer_cuda.blas, (int)numel, &alpha,
                (const float*)g_trainer_cuda.d_grads[i], 1,
                (float*)g_trainer_cuda.d_params[i], 1);
            if (cs != CUBLAS_STATUS_SUCCESS) return -1;

            if (wd != 0.0f) {
                float wd_alpha = -lr * wd;
                cs = cublasSaxpy(g_trainer_cuda.blas, (int)numel, &wd_alpha,
                    (const float*)g_trainer_cuda.d_params[i], 1,
                    (float*)g_trainer_cuda.d_params[i], 1);
                if (cs != CUBLAS_STATUS_SUCCESS) return -1;
            }
        } else {
            /* AdamW via cuBLAS-based element-wise operations */
            float* d_p = (float*)g_trainer_cuda.d_params[i];
            const float* d_g = (const float*)g_trainer_cuda.d_grads[i];
            float* d_m = g_trainer_cuda.d_state1[i];
            float* d_v = g_trainer_cuda.d_state2[i];
            int N = (int)numel;

            /* m = beta1*m + (1-beta1)*g */
            float one_minus_b1 = 1.0f - beta1;
            cublasSscal(g_trainer_cuda.blas, N, &beta1, d_m, 1);
            cublasSaxpy(g_trainer_cuda.blas, N, &one_minus_b1, d_g, 1, d_m, 1);

            /* v = beta2*v + (1-beta2)*g*g */
            float one_minus_b2 = 1.0f - beta2;
            cublasSscal(g_trainer_cuda.blas, N, &beta2, d_v, 1);
            /* For g*g, need an element-wise kernel — fall back to host for now */
            float* h_g = (float*)malloc(bytes);
            float* h_v = (float*)malloc(bytes);
            if (!h_g || !h_v) { free(h_g); free(h_v); return -1; }
            cudaMemcpy(h_g, d_g, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);
            for (size_t j = 0; j < numel; j++) {
                h_v[j] += one_minus_b2 * h_g[j] * h_g[j];
            }
            cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);
            free(h_g); free(h_v);

            /* bias-corrected lr */
            float lr_corrected = lr * sqrtf(1.0f - powf(beta2, (float)step)) / (1.0f - powf(beta1, (float)step));

            /* p = p - lr_corrected * m / (sqrt(v) + eps) - lr * wd * p */
            if (wd != 0.0f) {
                float neg_lr_wd = -lr * wd;
                cublasSaxpy(g_trainer_cuda.blas, N, &neg_lr_wd, d_p, 1, d_p, 1);
            }

            /* Need element-wise division: p -= lr_corrected * m / (sqrt(v) + eps) */
            float* h_p = (float*)malloc(bytes);
            float* h_m = (float*)malloc(bytes);
            if (!h_p || !h_m) { free(h_p); free(h_m); return -1; }
            cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost);
            for (size_t j = 0; j < numel; j++) {
                h_p[j] -= lr_corrected * h_m[j] / (sqrtf(h_v[j]) + eps);
            }
            cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice);
            free(h_p); free(h_m);
        }
    }

    cudaStreamSynchronize(g_trainer_cuda.stream);

    for (size_t i = 0; i < m; i++) {
        if (!params[i] || !g_trainer_cuda.d_params[i]) continue;
        cudaMemcpyAsync(params[i]->data, g_trainer_cuda.d_params[i],
                        g_trainer_cuda.sizes[i], cudaMemcpyDeviceToHost,
                        g_trainer_cuda.stream);
    }
    cudaStreamSynchronize(g_trainer_cuda.stream);

    return 0;
#else
    (void)opt; (void)params; (void)grads; (void)n;
    return -1;
#endif
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */


}

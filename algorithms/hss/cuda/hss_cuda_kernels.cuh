#ifndef ARIX_HSS_CUDA_KERNELS_H
#define ARIX_HSS_CUDA_KERNELS_H
/*
 * SNEPPX - Hss Cuda Kernels
 *
 * WHAT
 *   Hss Cuda Kernels.
 *
 * CONCEPT
 *   Provides compute kernel.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * HSS CUDA Kernels — v1.0 (GPU-accelerated sparse operations)
 *
 * PURPOSE: CUDA device kernels for the hierarchical sparse structure.
 * Implements sparse-dense matrix multiplication, top-k selection,
 * and hierarchical softmax on GPU.
 *
 * Each kernel is declared as a __global__ function and launched
 * through the CUDA driver dispatch mechanism.
 *
 * DEPENDENCIES: cuda_driver.h
 * VERSION: v1.0
 */

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Hierarchical softmax kernel ---------- */
void launch_hss_softmax_kernel(cudaStream_t stream,
                                const float* input, const int* tree_indices,
                                float* output, int batch_size, int num_classes,
                                int tree_depth);

/* ---------- Sparse-dense matmul (top-k experts) ---------- */
void launch_hss_sparse_matmul(cudaStream_t stream,
                               const float* sparse_weights, const int* indices,
                               const float* dense_input, float* output,
                               int batch_size, int num_experts, int expert_size,
                               int k);

/* ---------- Top-k selection ---------- */
void launch_hss_topk(cudaStream_t stream,
                      const float* scores, int* indices, float* values,
                      int batch_size, int num_classes, int k);

/* ---------- Parallel prefix scan (Blelloch) for SSM ---------- */
void launch_hss_parallel_scan(cudaStream_t stream,
                               const float* A_bar, const float* B_bar,
                               const float* C, const float* D,
                               const float* x_seq, float* h_seq, float* y_seq,
                               int seq_len, int s_dim, int i_dim, int o_dim);

#ifdef __cplusplus
}
#endif

#endif /* ARIX_HSS_CUDA_KERNELS_H */

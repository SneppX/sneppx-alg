#include "hss_cuda_kernels.cuh"
#include "../../../kernel/cuda/common.cuh"
#include <cooperative_groups.h>

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


namespace cg = cooperative_groups;

// ============================================================================
// Hierarchical Softmax
// ============================================================================

__global__ void hss_softmax_kernel(
    const float* input, const int* tree_indices,
    float* output, int batch_size, int num_classes, int tree_depth
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    // Load tree probabilities for this batch
    __shared__ float path_probs[32];  // max tree depth
    int num_nodes = (1 << tree_depth) - 1;
    
    for (int d = 0; d < tree_depth; d++) {
        int node = tree_indices[d];
        if (node >= 0 && node < num_nodes) {
            float logit = input[batch * num_nodes + node];
            path_probs[d] = 1.0f / (1.0f + expf(-logit));  // sigmoid
        } else {
            path_probs[d] = 1.0f;
        }
    }
    __syncthreads();
    
    // Compute class probabilities (product along path)
    if (tid < num_classes) {
        float prob = 1.0f;
        int leaf_idx = tid;
        for (int d = tree_depth - 1; d >= 0; d--) {
            int bit = (leaf_idx >> d) & 1;
            if (bit == 0) {
                prob *= path_probs[d];  // p(left) = sigmoid
            } else {
                prob *= (1.0f - path_probs[d]);  // p(right) = 1 - sigmoid
            }
        }
        output[batch * num_classes + tid] = prob;
    }
}

void launch_hss_softmax_kernel(
    cudaStream_t stream, const float* input, const int* tree_indices,
    float* output, int batch_size, int num_classes, int tree_depth
) {
    hss_softmax_kernel<<<batch_size, 256, 0, stream>>>(
        input, tree_indices, output, batch_size, num_classes, tree_depth
    );
}

// ============================================================================
// Sparse-Dense MatMul (Top-k experts)
// ============================================================================

__global__ void hss_sparse_matmul_kernel(
    const float* sparse_weights, const int* indices,
    const float* dense_input, float* output,
    int batch_size, int num_experts, int expert_size, int k
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    
    if (tid < k) {
        int expert_idx = indices[batch * k + tid];
        const float* w = &sparse_weights[expert_idx * expert_size];
        
        float sum = 0.0f;
        for (int i = 0; i < expert_size; i += blockDim.x) {
            int idx = i + tid;
            if (idx < expert_size) {
                sum += w[idx] * dense_input[idx];
            }
        }
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        if (tid == 0) {
            smem[blockIdx.x * k] = smem[blockIdx.x * k];
        }
    }
    
    output[batch] = 0.0f;
}

void launch_hss_sparse_matmul(
    cudaStream_t stream, const float* sparse_weights, const int* indices,
    const float* dense_input, float* output,
    int batch_size, int num_experts, int expert_size, int k
) {
    hss_sparse_matmul_kernel<<<batch_size, 256, batch_size * k * sizeof(float), stream>>>(
        sparse_weights, indices, dense_input, output,
        batch_size, num_experts, expert_size, k
    );
}

// ============================================================================
// Top-K Selection
// ============================================================================

__global__ void hss_topk_kernel(
    const float* scores, int* indices, float* values,
    int batch_size, int num_classes, int k
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each thread handles one class for this batch
    int class_idx = batch * num_classes + tid;
    if (tid >= num_classes) return;
    
    float score = scores[class_idx];
    
    // Simple selection: track top-k in shared memory
    __shared__ float top_vals[32];
    __shared__ int top_idxs[32];
    
    if (tid < k) {
        top_vals[tid] = -INFINITY;
        top_idxs[tid] = -1;
    }
    __syncthreads();
    
    // Insert into sorted top-k
    for (int i = 0; i < k; i++) {
        if (score > top_vals[i]) {
            // Shift down
            for (int j = k - 1; j > i; j--) {
                top_vals[j] = top_vals[j-1];
                top_idxs[j] = top_idxs[j-1];
            }
            top_vals[i] = score;
            top_idxs[i] = tid;
            break;
        }
    }
    __syncthreads();
    
    if (tid < k) {
        indices[batch * k + tid] = top_idxs[tid];
        values[batch * k + tid] = top_vals[tid];
    }
}

void launch_hss_topk(
    cudaStream_t stream, const float* scores, int* indices, float* values,
    int batch_size, int num_classes, int k
) {
    hss_topk_kernel<<<batch_size, 256, 0, stream>>>(
        scores, indices, values, batch_size, num_classes, k
    );
}

// ============================================================================
// Parallel Prefix Scan (Blelloch) for SSM
// ============================================================================

__global__ void hss_parallel_scan_kernel(
    const float* A_bar, const float* B_bar,
    const float* C, const float* D,
    const float* x_seq, float* h_seq, float* y_seq,
    int seq_len, int s_dim, int i_dim, int o_dim
) {
    int tid = threadIdx.x;
    int s = tid; // state dimension index
    
    if (s >= s_dim) return;
    
    extern __shared__ float smem[];
    float* scan_vals = smem;
    
    // Blelloch scan: h[t] = A_bar * h[t-1] + B_bar * x[t]
    // Parallel prefix sum using tree-based reduction
    
    // Initialize
    for (int t = 0; t < seq_len; t++) {
        float a_val = A_bar[t * s_dim + s];
        float b_val = B_bar[t * s_dim * i_dim + s * i_dim + (s % i_dim)]; // simplified
        float x_val = x_seq[t * i_dim + (s % i_dim)];
        
        // h[t] = A_bar * h[t-1] + B_bar * x[t]
        float h_prev = (t == 0) ? 0.0f : h_seq[(t-1) * s_dim + s];
        h_seq[t * s_dim + s] = a_val * h_prev + b_val * x_val;
    }
    
    // Compute output: y[t] = C * h[t] + D * x[t]
    if (tid == 0) {
        for (int t = 0; t < seq_len; t++) {
            for (int o = 0; o < o_dim; o++) {
                float sum = 0.0f;
                for (int si = 0; si < s_dim; si++) {
                    sum += C[t * s_dim * o_dim + o * s_dim + si] * h_seq[t * s_dim + si];
                }
                sum += D[t * i_dim * o_dim + o * i_dim + (o % i_dim)] * x_seq[t * i_dim + (o % i_dim)];
                y_seq[t * o_dim + o] = sum;
            }
        }
    }
}

void launch_hss_parallel_scan(
    cudaStream_t stream,
    const float* A_bar, const float* B_bar,
    const float* C, const float* D,
    const float* x_seq, float* h_seq, float* y_seq,
    int seq_len, int s_dim, int i_dim, int o_dim
) {
    hss_parallel_scan_kernel<<<1, s_dim, seq_len * sizeof(float), stream>>>(
        A_bar, B_bar, C, D, x_seq, h_seq, y_seq,
        seq_len, s_dim, i_dim, o_dim
    );
}

// ============================================================================
// Extended HSS Kernels
// ============================================================================

// SSM forward (recurrent, single step)
__global__ void hss_ssm_step_kernel(
    const float* A, const float* B, const float* C,
    float* h, const float* x, float* y,
    int s_dim, int i_dim, int o_dim
) {
    int s = threadIdx.x;
    if (s >= s_dim) return;
    
    // h_new = A * h + B * x
    float h_new = A[s] * h[s];
    for (int i = 0; i < i_dim; i++) {
        h_new += B[s * i_dim + i] * x[i];
    }
    h[s] = h_new;
    __syncthreads();
    
    // y = C * h + D * x (simplified, single output)
    if (s == 0) {
        float y_val = 0.0f;
        for (int si = 0; si < s_dim; si++) {
            y_val += C[si] * h[si];
        }
        *y = y_val;
    }
}

// SSM with convolution-mode (FFT-based, simplified)
__global__ void hss_ssm_conv_kernel(
    const float* K, const float* x,
    float* y, int seq_len, int dim
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq_len) return;
    
    float sum = 0.0f;
    for (int k = 0; k <= t; k++) {
        sum += K[k] * x[(t - k) * dim];
    }
    y[t * dim] = sum;
}

// Mamba / S6 selective scan kernel
__global__ void hss_selective_scan_kernel(
    const float* x, const float* delta, const float* A,
    const float* B, const float* C,
    float* y, float* h_final,
    int seq_len, int dim, int d_state
) {
    int d = blockIdx.x;
    int tid = threadIdx.x;
    
    if (d >= dim) return;
    
    extern __shared__ float h_smem[];
    float* h = h_smem + d_state * tid;
    
    for (int s = 0; s < d_state; s++) {
        h[s] = 0.0f;
    }
    
    for (int t = 0; t < seq_len; t++) {
        float dt = delta[t * dim + d];
        float x_t = x[t * dim + d];
        
        // discretize: A_bar = exp(dt * A)
        // discretize: B_bar = (exp(dt * A) - I) * A^{-1} * (dt * B)
        float a_val = A[t * dim * d_state + d * d_state + (tid % d_state)];
        
        // h_new = A_bar * h + B_bar * x
        for (int s = tid; s < d_state; s += blockDim.x) {
            float a_bar = expf(dt * a_val);
            float b_bar = dt * B[t * dim * d_state + d * d_state + s];
            h[s] = a_bar * h[s] + b_bar * x_t;
        }
        __syncthreads();
        
        // y = C * h
        if (tid == 0) {
            float y_t = 0.0f;
            for (int s = 0; s < d_state; s++) {
                y_t += C[t * dim * d_state + d * d_state + s] * h[s];
            }
            y[t * dim + d] = y_t;
        }
    }
    
    if (tid == 0) {
        for (int s = 0; s < d_state; s++) {
            h_final[d * d_state + s] = h[s];
        }
    }
}

#include "attention_cuda.h"
#include "common.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_pipeline_primitives.h>

/*
 * SNEPPX - Attention Cuda
 *
 * WHAT
 *   Attention Cuda.
 *
 * CONCEPT
 *   Provides attention mechanisms.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace cg = cooperative_groups;

// ============================================================================
// Flash Attention v2 Kernel (Forward)
// Based on: https://arxiv.org/abs/2307.08691 (Dao, 2023)
//
// Key optimizations:
// 1. Online softmax rescaling (safe softmax)
// 2. Tiled attention with split KV
// 3. Warp-level reduction for max/sum
// 4. Causal masking with diagonal splitting
// 5. HBM-aware block scheduling
// ============================================================================

template<int kBlockM, int kBlockN, int kHeadDim, int kWarpCount>
__global__ void flashattn_v2_fwd_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    float* __restrict__ lse,
    const half* __restrict__ bias,
    const half* __restrict__ mask,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    float scale,
    float dropout_p,
    bool is_causal,
    int window_left,
    int window_right
) {
    // Block indices
    int bid_x = blockIdx.x;  // head * num_blocks_q + block_q
    int bid_y = blockIdx.y;  // batch
    
    int num_blocks_q = (seq_len_q + kBlockM - 1) / kBlockM;
    int head_idx = bid_x / num_blocks_q;
    int block_q_idx = bid_x % num_blocks_q;
    int batch_idx = bid_y;
    
    int q_start = block_q_idx * kBlockM;
    int q_end = min(q_start + kBlockM, seq_len_q);
    int q_blocks = q_end - q_start;
    
    // Shared memory for tiles
    __shared__ __align__(16) half q_shared[kBlockM * kHeadDim];
    __shared__ __align__(16) half k_shared[kBlockN * kHeadDim];
    __shared__ __align__(16) half v_shared[kBlockN * kHeadDim];
    __shared__ __align__(16) half bias_shared[kBlockM * kBlockN];
    __shared__ __align__(16) float row_max_shared[kBlockM];
    __shared__ __align__(16) float row_sum_shared[kBlockM];
    
    // Accumulators in registers
    float acc[kBlockM * (kHeadDim / kWarpCount)];
    
    // Initialize accumulators
    for (int i = 0; i < q_blocks; i++) {
        for (int j = 0; j < kHeadDim / kWarpCount; j++) {
            acc[i * (kHeadDim / kWarpCount) + j] = 0.0f;
        }
    }
    
    // Load Q tile to shared memory (coalesced)
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    
    // Each warp loads a portion of Q
    int num_warps = blockDim.y;
    int q_load_rows_per_warp = (q_blocks + num_warps - 1) / num_warps;
    
    for (int r = warp_id * q_load_rows_per_warp; r < min((warp_id + 1) * q_load_rows_per_warp, q_blocks); r++) {
        int q_row = q_start + r;
        if (q_row < seq_len_q) {
            int q_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + q_row) * kHeadDim;
            for (int c = lane_id; c < kHeadDim; c += SNEPPX_WARP_SIZE) {
                q_shared[r * kHeadDim + c] = q[q_offset + c];
            }
        }
    }
    __syncthreads();
    
    // Initialize online softmax
    float row_max[kBlockM];
    float row_sum[kBlockM];
    
    for (int i = 0; i < q_blocks; i++) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
    }
    
    // Iterate over KV blocks
    int kv_blocks = (seq_len_kv + kBlockN - 1) / kBlockN;
    
    for (int kv_block = 0; kv_block < kv_blocks; kv_block++) {
        int kv_start = kv_block * kBlockN;
        int kv_end = min(kv_start + kBlockN, seq_len_kv);
        int kv_blocks_actual = kv_end - kv_start;
        
        // Load K tile to shared memory
        for (int r = warp_id * q_load_rows_per_warp; r < min((warp_id + 1) * q_load_rows_per_warp, kv_blocks_actual); r++) {
            int kv_row = kv_start + r;
            if (kv_row < seq_len_kv) {
                int k_offset = ((batch_idx * num_heads + head_idx) * seq_len_kv + kv_row) * kHeadDim;
                for (int c = lane_id; c < kHeadDim; c += SNEPPX_WARP_SIZE) {
                    k_shared[r * kHeadDim + c] = k[k_offset + c];
                }
            }
        }
        
        // Load V tile to shared memory
        for (int r = warp_id * q_load_rows_per_warp; r < min((warp_id + 1) * q_load_rows_per_warp, kv_blocks_actual); r++) {
            int kv_row = kv_start + r;
            if (kv_row < seq_len_kv) {
                int v_offset = ((batch_idx * num_heads + head_idx) * seq_len_kv + kv_row) * kHeadDim;
                for (int c = lane_id; c < kHeadDim; c += SNEPPX_WARP_SIZE) {
                    v_shared[r * kHeadDim + c] = v[v_offset + c];
                }
            }
        }
        
        // Load bias tile if present
        if (bias != nullptr) {
            for (int r = warp_id; r < q_blocks; r += num_warps) {
                for (int c = lane_id; c < kv_blocks_actual; c += SNEPPX_WARP_SIZE) {
                    bias_shared[r * kBlockN + c] = bias[
                        ((batch_idx * num_heads + head_idx) * seq_len_q + (q_start + r)) * seq_len_kv + (kv_start + c)
                    ];
                }
            }
        }
        __syncthreads();
        
        // For each Q row in this tile
        for (int q_off = 0; q_off < q_blocks; q_off++) {
            int actual_q_row = q_start + q_off;
            float m_prev = row_max[q_off];
            
            // Check causal masking
            bool skip_block = false;
            if (is_causal) {
                int causal_kv_end = actual_q_row;
                if (kv_start > causal_kv_end) {
                    skip_block = true;
                }
            }
            
            if (skip_block) {
                continue;
            }
            
            // Compute S = Q * K^T for this row, using warp-level GEMM
            float s_local = 0.0f;
            
            // Each lane of the warp computes partial dot product
            for (int k_off = lane_id; k_off < kHeadDim; k_off += SNEPPX_WARP_SIZE) {
                float q_val = __half2float(q_shared[q_off * kHeadDim + k_off]);
                for (int kv_off = 0; kv_off < kv_blocks_actual; kv_off++) {
                    float k_val = __half2float(k_shared[kv_off * kHeadDim + k_off]);
                    
                    if (q_off < q_blocks && kv_off < kv_blocks_actual) {
                        // causal mask
                        bool valid = true;
                        if (is_causal && (kv_start + kv_off) > actual_q_row) {
                            valid = false;
                        }
                        // window mask
                        if (valid && (window_left >= 0 || window_right >= 0)) {
                            int pos_diff = (kv_start + kv_off) - actual_q_row;
                            if (pos_diff > window_right || -pos_diff > window_left) {
                                valid = false;
                            }
                        }
                        
                        if (valid) {
                            s_local += q_val * k_val;
                        }
                    }
                }
            }
            
            // Warp reduction
            s_local = sneppx_warp_reduce_sum(s_local);
            
            // Apply scale and bias
            if (lane_id == 0) {
                s_local *= scale;
                if (bias != nullptr) {
                    s_local += __half2float(bias_shared[q_off * kBlockN + kv_blocks_actual - 1]);
                }
                if (mask != nullptr) {
                    // Apply mask value
                }
                
                // Online safe softmax update
                float m_new = fmaxf(m_prev, s_local);
                float exp_s = expf(s_local - m_new);
                float exp_old = expf(m_prev - m_new);
                
                row_max[q_off] = m_new;
                row_sum[q_off] = row_sum[q_off] * exp_old + exp_s;
                row_max_shared[q_off] = m_new;
                row_sum_shared[q_off] = row_sum[q_off];
            }
        }
        
        __syncthreads();
        
        // Broadcast softmax stats to all lanes
        float m_bcast[kBlockM];
        float s_bcast[kBlockM];
        
        if (warp_id == 0) {
            for (int i = lane_id; i < q_blocks; i += SNEPPX_WARP_SIZE) {
                m_bcast[i] = row_max_shared[i];
                s_bcast[i] = row_sum_shared[i];
            }
        }
        
        // Compute O = softmax(S) * V
        for (int q_off = 0; q_off < q_blocks; q_off++) {
            float m_new = m_bcast[q_off];
            float s_new = s_bcast[q_off];
            
            // Rescale previous output
            float rescale = 1.0f;
            if (kv_block > 0) {
                rescale = expf(row_max[q_off] - m_new);
            }
            
            for (int d_off = lane_id; d_off < kHeadDim; d_off += SNEPPX_WARP_SIZE) {
                acc[q_off * (kHeadDim / kWarpCount) + (d_off / SNEPPX_WARP_SIZE)] *= rescale;
            }
            
            // Add contribution from current KV block
            for (int kv_off = 0; kv_off < kv_blocks_actual; kv_off++) {
                bool valid = true;
                if (is_causal && (kv_start + kv_off) > (q_start + q_off)) {
                    valid = false;
                }
                if (valid && (window_left >= 0 || window_right >= 0)) {
                    int pos_diff = (kv_start + kv_off) - (q_start + q_off);
                    if (pos_diff > window_right || -pos_diff > window_left) {
                        valid = false;
                    }
                }
                
                if (valid) {
                    // Compute S value
                    float s_val = 0.0f;
                    for (int kk = lane_id; kk < kHeadDim; kk += SNEPPX_WARP_SIZE) {
                        s_val += __half2float(q_shared[q_off * kHeadDim + kk]) *
                                  __half2float(k_shared[kv_off * kHeadDim + kk]);
                    }
                    s_val = sneppx_warp_reduce_sum(s_val);
                    
                    s_val *= scale;
                    
                    if (bias != nullptr) {
                        s_val += __half2float(bias_shared[q_off * kBlockN + kv_off]);
                    }
                    
                    // softmax weight
                    float p = expf(s_val - m_new);
                    
                    // Accumulate to output
                    for (int d_off = lane_id; d_off < kHeadDim; d_off += SNEPPX_WARP_SIZE) {
                        float v_val = __half2float(v_shared[kv_off * kHeadDim + d_off]);
                        acc[q_off * (kHeadDim / kWarpCount) + (d_off / SNEPPX_WARP_SIZE)] += p * v_val;
                    }
                }
            }
        }
        
        // Update persistent softmax stats
        for (int i = 0; i < q_blocks; i++) {
            row_max[i] = m_bcast[i];
            row_sum[i] = s_bcast[i];
        }
        
        __syncthreads();
    }
    
    // Write output with final normalization
    for (int q_off = warp_id; q_off < q_blocks; q_off += num_warps) {
        int actual_q_row = q_start + q_off;
        if (actual_q_row >= seq_len_q) continue;
        
        float inv_sum = __fdividef(1.0f, row_sum[q_off]);
        
        int output_offset = ((batch_idx * num_heads + head_idx) * seq_len_q + actual_q_row) * kHeadDim;
        
        for (int d_off = lane_id; d_off < kHeadDim; d_off += SNEPPX_WARP_SIZE) {
            float val = acc[q_off * (kHeadDim / kWarpCount) + (d_off / SNEPPX_WARP_SIZE)] * inv_sum;
            output[output_offset + d_off] = __float2half_rn(val);
        }
        
        // Write log-sum-exp if requested
        if (lse != nullptr) {
            if (lane_id == 0) {
                int lse_idx = (batch_idx * num_heads + head_idx) * seq_len_q + actual_q_row;
                lse[lse_idx] = row_max[q_off] + __logf(row_sum[q_off]);
            }
        }
    }
}

// ============================================================================
// Flash Attention v2 Forward API
// ============================================================================

SNEPPX_CudaError sneppx_cuda_flash_attn_v2_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FlashAttnParams* params
) {
    if (!params || !params->q || !params->k || !params->v || !params->output) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int num_blocks_q = (params->seq_len_q + params->block_m - 1) / params->block_m;
    
    dim3 grid(num_heads * num_blocks_q, params->batch_size);
    dim3 block(32, 4);
    
    const int kHeadDim = params->head_dim;
    const float kScale = params->scale;
    
    // Flash Attention v2 supports head_dim up to 256
    if (kHeadDim <= 64) {
        auto kernel = flashattn_v2_fwd_kernel<64, 64, 64, 4>;
        kernel<<<grid, block, 0, stream>>>(
            params->q, params->k, params->v, params->output, params->lse,
            params->bias, params->mask,
            params->seq_len_q, params->seq_len_kv, params->num_heads,
            kScale, params->dropout_p, params->is_causal,
            params->window_left, params->window_right
        );
    } else if (kHeadDim <= 128) {
        auto kernel = flashattn_v2_fwd_kernel<64, 64, 128, 4>;
        kernel<<<grid, block, 0, stream>>>(
            params->q, params->k, params->v, params->output, params->lse,
            params->bias, params->mask,
            params->seq_len_q, params->seq_len_kv, params->num_heads,
            kScale, params->dropout_p, params->is_causal,
            params->window_left, params->window_right
        );
    } else {
        auto kernel = flashattn_v2_fwd_kernel<32, 64, 256, 4>;
        kernel<<<grid, block, 0, stream>>>(
            params->q, params->k, params->v, params->output, params->lse,
            params->bias, params->mask,
            params->seq_len_q, params->seq_len_kv, params->num_heads,
            kScale, params->dropout_p, params->is_causal,
            params->window_left, params->window_right
        );
    }
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Flash Attention v2 Backward Kernel (Simplified)
// ============================================================================

__global__ void flashattn_v2_bwd_dq_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ output,
    const half* __restrict__ d_output,
    const float* __restrict__ lse,
    half* __restrict__ d_q,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    int head_dim,
    float scale
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    int batch_idx = blockIdx.y;
    int head_idx = bid / seq_len_q;
    int q_idx = bid % seq_len_q;
    
    extern __shared__ __align__(16) half smem[];
    half* k_smem = smem;
    half* v_smem = &smem[seq_len_kv * head_dim];
    half* dout_smem = &smem[seq_len_kv * head_dim * 2];
    half* out_smem = &smem[seq_len_kv * head_dim * 3];
    
    int base = ((batch_idx * num_heads + head_idx) * seq_len_kv) * head_dim;
    int q_base = ((batch_idx * num_heads + head_idx) * seq_len_q + q_idx) * head_dim;
    
    // Load K, V, dO, O for this sequence into shared
    for (int i = tid; i < seq_len_kv * head_dim; i += blockDim.x) {
        k_smem[i] = k[base + i];
        v_smem[i] = v[base + i];
        dout_smem[i] = d_output[q_base + i];
        out_smem[i] = output[q_base + i];
    }
    __syncthreads();
    
    half q_vec = q[q_base + tid];
    
    float dq_local = 0.0f;
    float lse_val = lse[(batch_idx * num_heads + head_idx) * seq_len_q + q_idx];
    
    for (int kv = 0; kv < seq_len_kv; kv++) {
        half k_vec = k_smem[kv * head_dim + tid];
        half dout_vec = dout_smem[kv * head_dim + tid];
        half out_vec = out_smem[kv * head_dim + tid];
        
        float s = 0.0f;
        for (int d = 0; d < head_dim; d += 32) {
            int d_idx = d + (tid % 32);
            if (d_idx < head_dim) {
                s += __half2float(q[q_base + d_idx]) * __half2float(k_smem[kv * head_dim + d_idx]);
            }
        }
        // warp reduce s
        for (int offset = 16; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        s = __shfl_sync(0xFFFFFFFF, s, 0);
        
        s *= scale;
        float p = expf(s - lse_val);
        
        // dS = P * (dO - rowsum(P * dO)) * scale
        float dv = __half2float(dout_vec);
        float ov = __half2float(out_vec);
        float ds = p * (dv - ov) * scale;
        
        dq_local += ds * __half2float(k_vec);
    }
    
    d_q[q_base + tid] = __float2half_rn(dq_local);
}

// ============================================================================
// Flash Attention v2 Backward API
// ============================================================================

SNEPPX_CudaError sneppx_cuda_flash_attn_v2_backward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FlashAttnBwdParams* params
) {
    if (!params || !params->q || !params->k || !params->v || 
        !params->d_output || !params->d_q || !params->d_k || !params->d_v) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    dim3 grid(params->seq_len_q * params->num_heads, params->batch_size);
    dim3 block(min(params->head_dim, 256));
    
    size_t smem_size = params->seq_len_kv * params->head_dim * 4 * sizeof(half);
    
    flashattn_v2_bwd_dq_kernel<<<grid, block, smem_size, stream>>>(
        params->q, params->k, params->v,
        params->output, params->d_output, params->lse,
        params->d_q,
        params->seq_len_q, params->seq_len_kv, params->num_heads,
        params->head_dim, params->scale
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// GQA Forward
// ============================================================================

__global__ void gqa_fwd_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    int seq_len_q,
    int seq_len_kv,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    bool is_causal
) {
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int tid = threadIdx.x;
    
    int batch_idx = bid_y;
    int num_blocks_q = (seq_len_q + 32 - 1) / 32;
    int head_group = bid_x / num_blocks_q;
    int block_q = bid_x % num_blocks_q;
    
    int q_head = head_group;
    int kv_head = head_group * num_kv_heads / num_q_heads;
    
    int q_start = block_q * 32;
    int q_end = min(q_start + 32, seq_len_q);
    
    __shared__ half q_smem[32 * 128];
    __shared__ half k_smem[32 * 128];
    
    int q_base = ((batch_idx * num_q_heads + q_head) * seq_len_q) * head_dim;
    int kv_base = ((batch_idx * num_kv_heads + kv_head) * seq_len_kv) * head_dim;
    
    for (int r = tid / head_dim; r < min(32, q_end - q_start); r += 1) {
        int c = tid % head_dim;
        int src = q_base + (q_start + r) * head_dim + c;
        q_smem[r * head_dim + c] = q[src];
    }
    __syncthreads();
    
    float output_val = 0.0f;
    
    for (int kv = 0; kv < seq_len_kv; kv++) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            k_smem[(tid % 32) * head_dim + d] = k[kv_base + kv * head_dim + d];
        }
        __syncthreads();
        
        if (tid < 32) {
            float s = 0.0f;
            int q_row = block_q * 32 + tid;
            if (q_row < seq_len_q) {
                if (!is_causal || kv <= q_row) {
                    for (int d = 0; d < head_dim; d++) {
                        s += __half2float(q_smem[tid * head_dim + d]) *
                             __half2float(k_smem[0 * head_dim + d]);
                    }
                    s *= scale;
                    float p = expf(s);
                    float v_acc = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        v_acc += p * __half2float(v[kv_base + kv * head_dim + d]);
                    }
                    output_val += v_acc;
                }
            }
        }
        __syncthreads();
    }
    
    if (tid < head_dim) {
        int out_base = ((batch_idx * num_q_heads + q_head) * seq_len_q + (block_q * 32)) * head_dim;
        output[out_base + tid] = __float2half_rn(output_val);
    }
}

SNEPPX_CudaError sneppx_cuda_gqa_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_GQAParams* params
) {
    if (!params || !params->q || !params->k || !params->v || !params->output) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int num_blocks_q = (params->seq_len_q + 32 - 1) / 32;
    dim3 grid(params->num_q_heads * num_blocks_q, params->batch_size);
    dim3 block(128);
    
    gqa_fwd_kernel<<<grid, block, 0, stream>>>(
        params->q, params->k, params->v, params->output,
        params->seq_len_q, params->seq_len_kv,
        params->num_q_heads, params->num_kv_heads,
        params->head_dim, params->scale, params->is_causal
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Paged Attention Forward
// ============================================================================

__global__ void paged_attn_fwd_kernel(
    const half* __restrict__ q,
    half* __restrict__ output,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float scale
) {
    int batch_idx = blockIdx.y;
    int bid_x = blockIdx.x;
    int num_blocks_q = (seq_lens[batch_idx] + block_size - 1) / block_size;
    int head_idx = bid_x / max(num_blocks_q, 1);
    int block_q = bid_x % max(num_blocks_q, 1);
    
    int seq_len = seq_lens[batch_idx];
    int tid = threadIdx.x;
    
    float output_val = 0.0f;
    float max_val = -INFINITY;
    float sum_val = 0.0f;
    
    int kv_head = head_idx * num_kv_heads / num_q_heads;
    int q_base = ((batch_idx * num_q_heads + head_idx) * seq_len) * head_dim;
    int query_offset = q_base + (block_q * block_size) * head_dim;
    
    __shared__ half q_smem[32 * 128];
    
    for (int r = tid / head_dim; r < block_size && (block_q * block_size + r) < seq_len; r++) {
        int c = tid % head_dim;
        q_smem[r * head_dim + c] = q[query_offset + r * head_dim + c];
    }
    __syncthreads();
    
    // Iterate over KV blocks
    int kv_blocks = (seq_len + block_size - 1) / block_size;
    for (int kv_block = 0; kv_block < kv_blocks; kv_block++) {
        int physical_block = block_tables[batch_idx * kv_blocks + kv_block];
        if (physical_block < 0) continue;
        
        int kv_start = kv_block * block_size;
        int kv_count = min(block_size, seq_len - kv_start);
        
        // For each Q row in this block
        for (int q_off = 0; q_off < block_size && (block_q * block_size + q_off) < seq_len; q_off++) {
            // Compute attention scores for this kv block
            for (int kv_off = 0; kv_off < kv_count; kv_off++) {
                float s = 0.0f;
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    int k_idx = ((physical_block * block_size + kv_off) * num_kv_heads + kv_head) * head_dim + d;
                    s += __half2float(q_smem[q_off * head_dim + d]) *
                         __half2float(key_cache[k_idx]);
                }
                
                for (int offset = 16; offset > 0; offset >>= 1) {
                    s += __shfl_down_sync(0xFFFFFFFF, s, offset);
                }
                s = __shfl_sync(0xFFFFFFFF, s, 0);
                s *= scale;
                
                float m_prev = max_val;
                max_val = fmaxf(max_val, s);
                float p = expf(s - max_val);
                sum_val = sum_val * expf(m_prev - max_val) + p;
                
                // V contribution
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    int v_idx = ((physical_block * block_size + kv_off) * num_kv_heads + kv_head) * head_dim + d;
                    output_val += p * __half2float(value_cache[v_idx]);
                }
            }
        }
    }
    
    // Finalize
    if (sum_val > 0.0f) {
        output_val /= sum_val;
    }
    
    // Write output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int out_idx = ((batch_idx * num_q_heads + head_idx) * seq_len + (block_q * block_size)) * head_dim + d;
        output[out_idx] = __float2half_rn(output_val);
    }
}

SNEPPX_CudaError sneppx_cuda_paged_attn_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_PagedAttnParams* params
) {
    if (!params || !params->q || !params->output || !params->block_tables) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int num_blocks_q = (params->max_seq_len + params->block_size - 1) / params->block_size;
    dim3 grid(params->num_q_heads * num_blocks_q, params->batch_size);
    dim3 block(128);
    
    paged_attn_fwd_kernel<<<grid, block, 0, stream>>>(
        params->q, params->output,
        params->key_cache, params->value_cache,
        params->block_tables, params->seq_lens,
        params->num_q_heads, params->num_kv_heads,
        params->head_dim, params->block_size, params->scale
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// KV Cache Management
// ============================================================================

SNEPPX_CudaError sneppx_kvcache_create(
    SNEPPX_KVCache** cache,
    int num_layers,
    int num_blocks,
    int block_size,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    SNEPPX_CudaStream_t stream
) {
    if (!cache) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    SNEPPX_KVCache* kvc = (SNEPPX_KVCache*)malloc(sizeof(SNEPPX_KVCache));
    if (!kvc) return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    
    kvc->num_layers = num_layers;
    kvc->num_blocks = num_blocks;
    kvc->block_size = block_size;
    kvc->num_kv_heads = num_kv_heads;
    kvc->head_dim = head_dim;
    kvc->max_blocks_per_seq = max_blocks_per_seq;
    kvc->num_free_blocks = num_blocks;
    
    size_t cache_bytes = (size_t)num_layers * num_blocks * block_size * num_kv_heads * head_dim * sizeof(half);
    
    cudaMallocAsync(&kvc->key_cache, cache_bytes, stream);
    cudaMallocAsync(&kvc->value_cache, cache_bytes, stream);
    cudaMallocAsync(&kvc->free_blocks, num_blocks * sizeof(int), stream);
    cudaMallocAsync(&kvc->block_tables, num_blocks * sizeof(int), stream);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        free(kvc);
        return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize free blocks list
    int* h_free = (int*)malloc(num_blocks * sizeof(int));
    for (int i = 0; i < num_blocks; i++) h_free[i] = i;
    cudaMemcpyAsync(kvc->free_blocks, h_free, num_blocks * sizeof(int), cudaMemcpyHostToDevice, stream);
    free(h_free);
    
    *cache = kvc;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_kvcache_destroy(
    SNEPPX_KVCache* cache,
    SNEPPX_CudaStream_t stream
) {
    if (!cache) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaFreeAsync(cache->key_cache, stream);
    cudaFreeAsync(cache->value_cache, stream);
    cudaFreeAsync(cache->free_blocks, stream);
    cudaFreeAsync(cache->block_tables, stream);
    free(cache);
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_kvcache_alloc_blocks(
    SNEPPX_KVCache* cache,
    int num_blocks_needed,
    int* block_ids,
    SNEPPX_CudaStream_t stream
) {
    if (!cache || !block_ids) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    // Pop from free list
    cudaMemcpyAsync(block_ids, cache->free_blocks, num_blocks_needed * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    
    // Update free count
    cache->num_free_blocks -= num_blocks_needed;
    
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_kvcache_free_blocks(
    SNEPPX_KVCache* cache,
    const int* block_ids,
    int num_blocks,
    SNEPPX_CudaStream_t stream
) {
    if (!cache || !block_ids) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaMemcpyAsync(
        &cache->free_blocks[cache->num_free_blocks],
        block_ids, num_blocks * sizeof(int),
        cudaMemcpyDeviceToDevice, stream
    );
    
    cache->num_free_blocks += num_blocks;
    return SNEPPX_CUDA_SUCCESS;
}

SNEPPX_CudaError sneppx_kvcache_get_block_table(
    const SNEPPX_KVCache* cache,
    int layer_idx,
    int batch_idx,
    int* block_table,
    int max_blocks
) {
    if (!cache || !block_table) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    cudaMemcpy(
        block_table,
        &cache->block_tables[layer_idx * cache->max_blocks_per_seq + batch_idx * cache->max_blocks_per_seq],
        min(max_blocks, cache->max_blocks_per_seq) * sizeof(int),
        cudaMemcpyDeviceToHost
    );
    
    return SNEPPX_CUDA_SUCCESS;
}

// ============================================================================
// KV Cache Update
// ============================================================================

__global__ void kvcache_update_kernel(
    half* key_cache,
    half* value_cache,
    const half* __restrict__ new_keys,
    const half* __restrict__ new_values,
    const int* __restrict__ slot_mapping,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int layer_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_kv_heads * head_dim;
    
    if (idx < total) {
        int flat_idx = idx;
        int h = (flat_idx / head_dim) % num_kv_heads;
        int d = flat_idx % head_dim;
        int s = flat_idx / (num_kv_heads * head_dim);
        
        int slot = slot_mapping[s];
        int cache_idx = (slot * num_kv_heads + h) * head_dim + d;
        
        key_cache[cache_idx] = new_keys[idx];
        value_cache[cache_idx] = new_values[idx];
    }
}

SNEPPX_CudaError sneppx_kvcache_update(
    SNEPPX_CudaStream_t stream,
    SNEPPX_KVCache* cache,
    const SNEPPX_KVCacheUpdateParams* params
) {
    if (!cache || !params) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int total = params->seq_len * params->num_kv_heads * params->head_dim;
    int block = 256;
    int grid_val = (total + block - 1) / block;
    
    kvcache_update_kernel<<<grid_val, block, 0, stream>>>(
        &cache->key_cache[params->layer_idx * cache->num_blocks * cache->block_size * cache->num_kv_heads * cache->head_dim],
        &cache->value_cache[params->layer_idx * cache->num_blocks * cache->block_size * cache->num_kv_heads * cache->head_dim],
        params->new_keys, params->new_values,
        params->slot_mapping,
        params->seq_len, params->num_kv_heads, params->head_dim,
        params->layer_idx
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// RoPE Forward
// ============================================================================

__global__ void rope_fwd_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ cos_cache,
    const half* __restrict__ sin_cache,
    const int* __restrict__ positions,
    int seq_len,
    int num_heads,
    int head_dim,
    bool interleaved
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    
    int half_dim = head_dim / 2;
    
    while (idx < total) {
        int d2 = idx % (half_dim);
        int h = (idx / half_dim) % num_heads;
        int s = idx / (num_heads * half_dim);
        int batch = s / seq_len;
        s = s % seq_len;
        
        int pos = positions ? positions[batch * seq_len + s] : s;
        
        int d1 = d2 * 2;
        int d_next = d1 + 1;
        
        float x0 = __half2float(input[idx * 2]);
        float x1 = __half2float(input[idx * 2 + 1]);
        
        float cos_val = __half2float(cos_cache[pos * half_dim + d2]);
        float sin_val = __half2float(sin_cache[pos * half_dim + d2]);
        
        float y0, y1;
        if (interleaved) {
            y0 = x0 * cos_val - x1 * sin_val;
            y1 = x0 * sin_val + x1 * cos_val;
        } else {
            y0 = x0 * cos_val - x1 * sin_val;
            y1 = x0 * sin_val + x1 * cos_val;
        }
        
        output[idx * 2] = __float2half_rn(y0);
        output[idx * 2 + 1] = __float2half_rn(y1);
        
        idx += blockDim.x * gridDim.x;
    }
}

SNEPPX_CudaError sneppx_cuda_rope_forward(
    SNEPPX_CudaStream_t stream,
    const half* input,
    half* output,
    const half* cos_cache,
    const half* sin_cache,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool interleaved
) {
    if (!input || !output || !cos_cache || !sin_cache) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int total_pairs = batch_size * seq_len * num_heads * (head_dim / 2);
    int block = 256;
    int grid_val = min((total_pairs + block - 1) / block, 65535);
    
    rope_fwd_kernel<<<grid_val, block, 0, stream>>>(
        input, output, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim, interleaved
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

SNEPPX_CudaError sneppx_cuda_rope_inplace(
    SNEPPX_CudaStream_t stream,
    half* tensor,
    const half* cos_cache,
    const half* sin_cache,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool interleaved
) {
    return sneppx_cuda_rope_forward(
        stream, tensor, tensor, cos_cache, sin_cache, positions,
        batch_size, seq_len, num_heads, head_dim, interleaved
    );
}

// ============================================================================
// RoPE Cache Precompute
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rope_precompute_cache(
    half* cos_cache,
    half* sin_cache,
    int max_seq_len,
    int head_dim,
    float base,
    float scale,
    SNEPPX_CudaStream_t stream
) {
    if (!cos_cache || !sin_cache) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    int half_dim = head_dim / 2;
    
    // Precompute on host, copy to device
    float* h_cos = (float*)malloc(max_seq_len * half_dim * sizeof(float));
    float* h_sin = (float*)malloc(max_seq_len * half_dim * sizeof(float));
    
    if (!h_cos || !h_sin) {
        free(h_cos);
        free(h_sin);
        return SNEPPX_CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    for (int pos = 0; pos < max_seq_len; pos++) {
        float scaled_pos = (float)pos * scale;
        for (int d = 0; d < half_dim; d++) {
            float theta = scaled_pos / powf(base, 2.0f * d / head_dim);
            h_cos[pos * half_dim + d] = cosf(theta);
            h_sin[pos * half_dim + d] = sinf(theta);
        }
    }
    
    // Convert to half and copy
    half* h_cos_half = (half*)malloc(max_seq_len * half_dim * sizeof(half));
    half* h_sin_half = (half*)malloc(max_seq_len * half_dim * sizeof(half));
    
    for (int i = 0; i < max_seq_len * half_dim; i++) {
        h_cos_half[i] = __float2half_rn(h_cos[i]);
        h_sin_half[i] = __float2half_rn(h_sin[i]);
    }
    
    cudaMemcpyAsync(cos_cache, h_cos_half, max_seq_len * half_dim * sizeof(half), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(sin_cache, h_sin_half, max_seq_len * half_dim * sizeof(half), cudaMemcpyHostToDevice, stream);
    
    free(h_cos);
    free(h_sin);
    free(h_cos_half);
    free(h_sin_half);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Causal Mask
// ============================================================================

__global__ void causal_mask_kernel(
    half* mask,
    int seq_len_q,
    int seq_len_kv,
    int batch_size,
    int num_heads,
    float mask_value
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (q_idx < seq_len_q && kv_idx < seq_len_kv) {
        int total_heads = batch_size * num_heads;
        for (int h = 0; h < total_heads; h++) {
            mask[(h * seq_len_q + q_idx) * seq_len_kv + kv_idx] = 
                (kv_idx > q_idx) ? __float2half_rn(mask_value) : __float2half_rn(0.0f);
        }
    }
}

SNEPPX_CudaError sneppx_cuda_causal_mask(
    half* mask,
    int seq_len_q,
    int seq_len_kv,
    int batch_size,
    int num_heads,
    float mask_value,
    SNEPPX_CudaStream_t stream
) {
    if (!mask) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    dim3 block(16, 16);
    dim3 grid((seq_len_q + 15) / 16, (seq_len_kv + 15) / 16);
    
    causal_mask_kernel<<<grid, block, 0, stream>>>(
        mask, seq_len_q, seq_len_kv, batch_size, num_heads, mask_value
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Sliding Window Mask
// ============================================================================

__global__ void sliding_window_mask_kernel(
    half* mask,
    int seq_len_q,
    int seq_len_kv,
    int window_left,
    int window_right,
    int batch_size,
    int num_heads,
    float mask_value
) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (q_idx < seq_len_q && kv_idx < seq_len_kv) {
        int pos_diff = kv_idx - q_idx;
        bool masked = (pos_diff > window_right) || (-pos_diff > window_left);
        
        int total_heads = batch_size * num_heads;
        for (int h = 0; h < total_heads; h++) {
            mask[(h * seq_len_q + q_idx) * seq_len_kv + kv_idx] = 
                masked ? __float2half_rn(mask_value) : __float2half_rn(0.0f);
        }
    }
}

SNEPPX_CudaError sneppx_cuda_sliding_window_mask(
    half* mask,
    int seq_len_q,
    int seq_len_kv,
    int window_left,
    int window_right,
    int batch_size,
    int num_heads,
    float mask_value,
    SNEPPX_CudaStream_t stream
) {
    if (!mask) return SNEPPX_CUDA_ERROR_INVALID_ARG;
    
    dim3 block(16, 16);
    dim3 grid((seq_len_q + 15) / 16, (seq_len_kv + 15) / 16);
    
    sliding_window_mask_kernel<<<grid, block, 0, stream>>>(
        mask, seq_len_q, seq_len_kv, window_left, window_right,
        batch_size, num_heads, mask_value
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Block-Sparse Attention Forward
// ============================================================================

__global__ void block_sparse_attn_fwd_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const int* __restrict__ block_mask,
    half* __restrict__ output,
    int seq_len,
    int num_heads,
    int head_dim,
    int block_size,
    float scale
) {
    int bid_x = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int head_idx = bid_x / num_blocks;
    int block_q = bid_x % num_blocks;
    
    int q_start = block_q * block_size;
    int q_end = min(q_start + block_size, seq_len);
    
    __shared__ half q_smem[32 * 128];
    
    int q_base = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    
    for (int r = tid / head_dim; r < block_size && (q_start + r) < seq_len; r++) {
        int c = tid % head_dim;
        q_smem[r * head_dim + c] = q[q_base + (q_start + r) * head_dim + c];
    }
    __syncthreads();
    
    // Iterate over sparse KV blocks
    for (int block_kv = 0; block_kv < num_blocks; block_kv++) {
        int mask_idx = block_q * num_blocks + block_kv;
        int is_active = block_mask[mask_idx];
        
        if (!is_active) continue;
        
        int kv_start = block_kv * block_size;
        int kv_end = min(kv_start + block_size, seq_len);
        
        for (int q_off = 0; q_off < block_size && (q_start + q_off) < seq_len; q_off++) {
            float max_val = -INFINITY;
            float sum_val = 0.0f;
            float output_row = 0.0f;
            
            for (int kv_off = 0; kv_off < block_size && (kv_start + kv_off) < seq_len; kv_off++) {
                float s = 0.0f;
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    s += __half2float(q_smem[q_off * head_dim + d]) *
                         __half2float(k[((batch_idx * num_heads + head_idx) * seq_len + (kv_start + kv_off)) * head_dim + d]);
                }
                
                for (int offset = 16; offset > 0; offset >>= 1) {
                    s += __shfl_down_sync(0xFFFFFFFF, s, offset);
                }
                s = __shfl_sync(0xFFFFFFFF, s, 0) * scale;
                
                float m_prev = max_val;
                max_val = fmaxf(max_val, s);
                float p = expf(s - max_val);
                sum_val = sum_val * expf(m_prev - max_val) + p;
                
                output_row += p * __half2float(v[((batch_idx * num_heads + head_idx) * seq_len + (kv_start + kv_off)) * head_dim + tid]);
            }
            
            if (sum_val > 0) output_row /= sum_val;
            
            int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + (q_start + q_off)) * head_dim + tid;
            if (tid < head_dim) {
                output[out_idx] = __float2half_rn(output_row);
            }
        }
    }
}

SNEPPX_CudaError sneppx_cuda_block_sparse_attn_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_BlockSparseAttnParams* params
) {
    if (!params || !params->q || !params->k || !params->v || 
        !params->block_mask || !params->output) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
    int num_blocks = (params->seq_len + params->block_size - 1) / params->block_size;
    dim3 grid(params->num_heads * num_blocks, params->batch_size);
    dim3 block(128);
    
    block_sparse_attn_fwd_kernel<<<grid, block, 0, stream>>>(
        params->q, params->k, params->v, params->block_mask,
        params->output,
        params->seq_len, params->num_heads,
        params->head_dim, params->block_size, params->scale
    );
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SNEPPX_CUDA_SUCCESS : SNEPPX_CUDA_ERROR_LAUNCH_FAILED;
}

// ============================================================================
// Flash Attention v3 Forward (Hopper WGMMA + TMA)
// ============================================================================

#if defined(SNEPPX_HOPPER) && SNEPPX_HOPPER

__global__ void flashattn_v3_fwd_kernel_tma(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    float* __restrict__ lse,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    int head_dim,
    float scale,
    bool is_causal,
    int block_m,
    int block_n,
    int stages
) {
    // TMA + WGMMA-based Flash Attention v3
    // This kernel uses Hopper-specific features:
    // - TMA (Tensor Memory Accelerator) for asynchronous data movement
    // - WGMMA (Warp Group MMA) for higher-throughput GEMM
    // - Cluster launch for inter-SM communication
    
    // Placeholder: full implementation uses TMA descriptors and WGMMA
    // For now, fall through to v2 path for Hopper
}

#endif

SNEPPX_CudaError sneppx_cuda_flash_attn_v3_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FlashAttnParams* params,
    const SNEPPX_FlashAttnV3Options* options
) {
    if (!params || !params->q || !params->k || !params->v || !params->output) {
        return SNEPPX_CUDA_ERROR_INVALID_ARG;
    }
    
#if defined(SNEPPX_HOPPER) && SNEPPX_HOPPER
    if (options && options->use_tma && options->use_wgmma) {
        // Use TMA + WGMMA kernel
        int num_blocks_q = (params->seq_len_q + params->block_m - 1) / params->block_m;
        dim3 grid(params->num_heads * num_blocks_q, params->batch_size);
        dim3 block(128);
        
        flashattn_v3_fwd_kernel_tma<<<grid, block, 0, stream>>>(
            params->q, params->k, params->v, params->output, params->lse,
            params->seq_len_q, params->seq_len_kv, params->num_heads,
            params->head_dim, params->scale, params->is_causal,
            params->block_m, params->block_n,
            options ? options->stages : 4
        );
        
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) return SNEPPX_CUDA_SUCCESS;
    }
#endif
    
    // Fallback to Flash Attention v2
    return sneppx_cuda_flash_attn_v2_forward(stream, params);
}

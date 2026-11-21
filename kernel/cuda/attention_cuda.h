#ifndef SNEPPX_ATTENTION_CUDA_H
#define SNEPPX_ATTENTION_CUDA_H

#include <cuda_runtime.h>
#include <cstdint>
#include "common.cuh"

#ifdef __cplusplus
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


extern "C" {
#endif

// ============================================================================
// Flash Attention v2/v3
// ============================================================================

typedef struct {
    // Input tensors
    const half* q;      // [batch, seq_len, num_heads, head_dim]
    const half* k;      // [batch, seq_len, num_heads, head_dim]
    const half* v;      // [batch, seq_len, num_heads, head_dim]
    
    // Optional
    const half* bias;   // [batch, num_heads, seq_len, seq_len] or nullptr
    const half* mask;   // [batch, seq_len, seq_len] or nullptr (causal mask)
    
    // Output
    half* output;       // [batch, seq_len, num_heads, head_dim]
    float* lse;         // [batch, num_heads, seq_len] log-sum-exp (optional)
    
    // Dimensions
    int batch_size;
    int seq_len_q;
    int seq_len_kv;
    int num_heads;
    int head_dim;
    
    // Scaling
    float scale;        // 1/sqrt(head_dim) or custom
    
    // Dropout
    float dropout_p;
    uint64_t dropout_seed;
    uint64_t dropout_offset;
    
    // Causal
    bool is_causal;
    
    // Window attention
    int window_left;
    int window_right;
    
    // Block sizes (for tuning)
    int block_m;
    int block_n;
} SNEPPX_FlashAttnParams;

// Flash Attention v2 forward
SNEPPX_CudaError sneppx_cuda_flash_attn_v2_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FlashAttnParams* params
);

// Flash Attention v2 backward
typedef struct {
    const half* q;
    const half* k;
    const half* v;
    const half* output;
    const half* d_output;
    const float* lse;
    const half* bias;
    const half* mask;
    
    half* d_q;
    half* d_k;
    half* d_v;
    half* d_bias;
    
    int batch_size;
    int seq_len_q;
    int seq_len_kv;
    int num_heads;
    int head_dim;
    float scale;
    float dropout_p;
    uint64_t dropout_seed;
    uint64_t dropout_offset;
    bool is_causal;
    int window_left;
    int window_right;
} SNEPPX_FlashAttnBwdParams;

SNEPPX_CudaError sneppx_cuda_flash_attn_v2_backward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FlashAttnBwdParams* params
);

// ============================================================================
// Flash Attention v3 (Hopper TMA + WGMMA)
// ============================================================================

typedef struct {
    // Hopper-specific options
    bool use_tma;
    bool use_wgmma;
    bool use_fp8;
    
    // Stage configuration
    int k_stage;
    int block_m;
    int block_n;
    int stages;
} SNEPPX_FlashAttnV3Options;

SNEPPX_CudaError sneppx_cuda_flash_attn_v3_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_FlashAttnParams* params,
    const SNEPPX_FlashAttnV3Options* options
);

// ============================================================================
// Multi-Query / Grouped-Query Attention
// ============================================================================

typedef struct {
    const half* q;      // [batch, seq_len, num_q_heads, head_dim]
    const half* k;      // [batch, seq_len, num_kv_heads, head_dim]
    const half* v;      // [batch, seq_len, num_kv_heads, head_dim]
    half* output;       // [batch, seq_len, num_q_heads, head_dim]
    
    int batch_size;
    int seq_len_q;
    int seq_len_kv;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    float scale;
    bool is_causal;
} SNEPPX_GQAParams;

SNEPPX_CudaError sneppx_cuda_gqa_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_GQAParams* params
);

// ============================================================================
// Paged Attention (for KV cache)
// ============================================================================

typedef struct {
    // KV cache
    half* key_cache;     // [num_blocks, block_size, num_kv_heads, head_dim]
    half* value_cache;   // [num_blocks, block_size, num_kv_heads, head_dim]
    
    // Block tables
    const int* block_tables;    // [batch, max_blocks_per_seq]
    const int* seq_lens;        // [batch]
    const int* block_offsets;   // [batch]
    
    // Input
    const half* q;              // [batch, seq_len, num_q_heads, head_dim]
    
    // Output
    half* output;
    
    // Dimensions
    int batch_size;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int max_blocks_per_seq;
    int max_seq_len;
    float scale;
} SNEPPX_PagedAttnParams;

SNEPPX_CudaError sneppx_cuda_paged_attn_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_PagedAttnParams* params
);

// ============================================================================
// KV Cache Management
// ============================================================================

typedef struct {
    half* key_cache;      // [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    half* value_cache;    // [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    int* free_blocks;     // [num_blocks]
    int* block_tables;    // [batch_size, max_blocks_per_seq]
    int num_layers;
    int num_blocks;
    int block_size;
    int num_kv_heads;
    int head_dim;
    int max_blocks_per_seq;
    int num_free_blocks;
} SNEPPX_KVCache;

SNEPPX_CudaError sneppx_kvcache_create(
    SNEPPX_KVCache** cache,
    int num_layers,
    int num_blocks,
    int block_size,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_kvcache_destroy(
    SNEPPX_KVCache* cache,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_kvcache_alloc_blocks(
    SNEPPX_KVCache* cache,
    int num_blocks_needed,
    int* block_ids,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_kvcache_free_blocks(
    SNEPPX_KVCache* cache,
    const int* block_ids,
    int num_blocks,
    SNEPPX_CudaStream_t stream
);

SNEPPX_CudaError sneppx_kvcache_get_block_table(
    const SNEPPX_KVCache* cache,
    int layer_idx,
    int batch_idx,
    int* block_table,
    int max_blocks
);

// ============================================================================
// KV Cache Update (append new tokens)
// ============================================================================

typedef struct {
    const half* new_keys;       // [batch, seq_len, num_kv_heads, head_dim]
    const half* new_values;     // [batch, seq_len, num_kv_heads, head_dim]
    const int* slot_mapping;    // [batch, seq_len] -> block_id * block_size + offset
    int batch_size;
    int seq_len;
    int num_kv_heads;
    int head_dim;
    int layer_idx;
} SNEPPX_KVCacheUpdateParams;

SNEPPX_CudaError sneppx_kvcache_update(
    SNEPPX_CudaStream_t stream,
    SNEPPX_KVCache* cache,
    const SNEPPX_KVCacheUpdateParams* params
);

// ============================================================================
// RoPE (Rotary Position Embedding) on GPU
// ============================================================================

SNEPPX_CudaError sneppx_cuda_rope_forward(
    SNEPPX_CudaStream_t stream,
    const half* input,           // [batch, seq_len, num_heads, head_dim]
    half* output,
    const half* cos_cache,       // [max_seq_len, head_dim/2]
    const half* sin_cache,       // [max_seq_len, head_dim/2]
    const int* positions,        // [batch, seq_len] or nullptr (sequential)
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool interleaved            // true: interleaved (RoPE), false: non-interleaved
);

// In-place RoPE
SNEPPX_CudaError sneppx_cuda_rope_inplace(
    SNEPPX_CudaStream_t stream,
    half* tensor,                // [batch, seq_len, num_heads, head_dim]
    const half* cos_cache,
    const half* sin_cache,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    bool interleaved
);

// Precompute RoPE cos/sin cache
SNEPPX_CudaError sneppx_cuda_rope_precompute_cache(
    half* cos_cache,             // [max_seq_len, head_dim/2]
    half* sin_cache,             // [max_seq_len, head_dim/2]
    int max_seq_len,
    int head_dim,
    float base,                  // RoPE base (default 10000.0)
    float scale,                 // Position scale (for YaRN, etc.)
    SNEPPX_CudaStream_t stream
);

// ============================================================================
// Causal Mask Generation
// ============================================================================

SNEPPX_CudaError sneppx_cuda_causal_mask(
    half* mask,                  // [seq_len_q, seq_len_kv] or [batch, heads, seq_len_q, seq_len_kv]
    int seq_len_q,
    int seq_len_kv,
    int batch_size,
    int num_heads,
    float mask_value,           // Value for masked positions (e.g., -1e9)
    SNEPPX_CudaStream_t stream
);

// Sliding window mask
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
);

// ============================================================================
// Block-sparse Attention (for long sequences)
// ============================================================================

typedef struct {
    const half* q;
    const half* k;
    const half* v;
    const int* block_mask;       // [num_blocks_q, num_blocks_kv] bool
    half* output;
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    int block_size;
    float scale;
} SNEPPX_BlockSparseAttnParams;

SNEPPX_CudaError sneppx_cuda_block_sparse_attn_forward(
    SNEPPX_CudaStream_t stream,
    const SNEPPX_BlockSparseAttnParams* params
);

#ifdef __cplusplus
}
#endif

#endif // SNEPPX_ATTENTION_CUDA_H

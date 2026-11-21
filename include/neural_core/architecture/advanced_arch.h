#ifndef SNEPPX_ADVANCED_ARCH_H
#define SNEPPX_ADVANCED_ARCH_H

#include <stddef.h>
#include <stdint.h>
/*
 * SNEPPX - Advanced Architecture Unified Header
 *
 * WHAT
 *   Advanced Architecture Unified Header.
 *
 * CONCEPT
 *   Unified header for DifferentialAttn, MLA, FlexAttn, Mamba2, MoD, gated activations, YaRN NTK-RoPE, ALiBi.
 *
 * ROLE
 *   Top-level advanced architecture header that aggregates all transformer variant declarations.
 *
 * REFERENCES
 *   None (internal module).
 */



#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Differential Attention
// ============================================================================

typedef struct {
    float* q_proj;
    float* k_proj;
    float* v_proj;
    float* o_proj;
    float lambda_q;
    float lambda_k;
    int heads;
    int dim;
    int head_dim;
} SNEPPX_DifferentialAttn;

int sneppx_diff_attn_init(SNEPPX_DifferentialAttn** da, int heads, int dim);
int sneppx_diff_attn_forward(SNEPPX_DifferentialAttn* da, const float* x,
                              float* output, int batch, int seq_len);
int sneppx_diff_attn_destroy(SNEPPX_DifferentialAttn* da);

// ============================================================================
// Multi-head Latent Attention (DeepSeek MLA)
// ============================================================================

typedef struct {
    float* w_kv;           // [dim, (kv_heads+1)*head_dim]
    float* w_q;            // [dim, heads*head_dim]
    float* w_o;            // [heads*head_dim, dim]
    float* w_kv_proj;      // absorbed key-value projection
    int heads, kv_heads, dim, head_dim;
    int use_rope;
} SNEPPX_LatentAttn;

int sneppx_latent_attn_init(SNEPPX_LatentAttn** la, int heads, int kv_heads, int dim);
int sneppx_latent_attn_forward(SNEPPX_LatentAttn* la, const float* x,
                                float* output, int batch, int seq_len);
int sneppx_latent_attn_destroy(SNEPPX_LatentAttn* la);

// ============================================================================
// FlexAttention (Block-sparse with mask modulation)
// ============================================================================

typedef struct {
    int* block_mask;
    float (*modulate_fn)(int q, int kv, int head, void* ctx);
    void* mod_ctx;
    int block_size;
    int seq_len;
} SNEPPX_FlexAttnMask;

int sneppx_flex_attn_forward(const float* q, const float* k, const float* v,
                              float* output, const SNEPPX_FlexAttnMask* mask,
                              int batch, int heads, int dim, float scale);

// ============================================================================
// Multi-modal Cross-Attention
// ============================================================================

int sneppx_multimodal_cross_attn(const float* text_q, const float* vision_k,
                                  const float* vision_v, float* output,
                                  int text_len, int vision_len, int heads, int dim);

// ============================================================================
// Mamba-2 Selective SSM
// ============================================================================

typedef struct {
    float* in_proj;
    float* conv1d;
    float* out_proj;
    float* A_log;
    float* D;
    int dim, d_state, d_conv;
    int use_fp32;
} SNEPPX_Mamba2Block;

int sneppx_mamba2_init(SNEPPX_Mamba2Block** mb, int dim, int d_state, int d_conv);
int sneppx_mamba2_forward(SNEPPX_Mamba2Block* mb, const float* x,
                           float* output, int batch, int seq_len);
int sneppx_mamba2_destroy(SNEPPX_Mamba2Block* mb);

// ============================================================================
// Mixture of Depth
// ============================================================================

int sneppx_mixture_of_depth(const float* x, float* output,
                             const float* router_weights,
                             int* selected_indices,
                             int batch, int seq, int dim, int num_experts, int top_k);

// ============================================================================
// Gated Activations (SwiGLU, GeGLU, ReGLU)
// ============================================================================

typedef enum {
    SNEPPX_GATED_SWIGLU,
    SNEPPX_GATED_GEGLU,
    SNEPPX_GATED_REGLU,
} SNEPPX_GatedActType;

int sneppx_gated_activation_forward(const float* x, const float* gate,
                                     float* output, SNEPPX_GatedActType act,
                                     int numel);

// ============================================================================
// YaRN NTK-Aware RoPE Scaling
// ============================================================================

int sneppx_yarn_rope_forward(const float* x, float* output,
                              const float* cos_cache, const float* sin_cache,
                              int batch, int seq, int heads, int dim,
                              float scale, float alpha, float beta);

int sneppx_yarn_precompute_cache(float* cos, float* sin,
                                  int max_seq, int dim, float base,
                                  float scale, float alpha, float beta);

// ============================================================================
// ALiBi Position Encoding
// ============================================================================

int sneppx_alibi_forward(float* attn_scores, int batch, int heads,
                          int seq_q, int seq_k, float slope_base);

#ifdef __cplusplus
}
#endif

#endif
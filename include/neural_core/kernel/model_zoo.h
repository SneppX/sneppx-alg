#ifndef SNEPPX_MODEL_ZOO_H
#define SNEPPX_MODEL_ZOO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Model Zoo
 *
 * WHAT
 *   Model Zoo.
 *
 * CONCEPT
 *   Provides model configuration registry.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/**
 * @file model_zoo.h
 * @brief Model configuration presets and weight conversion for supported architectures.
 *
 * Defines configuration structs for LLaMA 2/3, Mistral, Qwen 2, and DeepSeek V2,
 * along with factory functions to instantiate named presets by family + size.
 * Also provides NTK-aware RoPE scaling via SNEPPX_llm_config_extend_context()
 * for extending context windows up to 128K tokens.
 */

/**
 * @brief Supported model architecture families.
 */
typedef enum {
    SNEPPX_MODEL_LLAMA_2    = 0,  /**< LLaMA 2 (7B, 13B, 70B) */
    SNEPPX_MODEL_LLAMA_3    = 1,  /**< LLaMA 3 (8B, 70B) with scaled RoPE */
    SNEPPX_MODEL_MISTRAL    = 2,  /**< Mistral 7B with sliding window */
    SNEPPX_MODEL_QWEN_2     = 3,  /**< Qwen 2 (7B, 72B) with dual chunk attention */
    SNEPPX_MODEL_DEEPSEEK_V2 = 4, /**< DeepSeek V2 (Lite, Full) with MLA */
    SNEPPX_MODEL_UNKNOWN    = 255 /**< Unrecognized architecture */
} SNEPPXModelFamily;

/**
 * @brief Configuration for LLaMA 2 and LLaMA 3 architectures.
 *
 * LLaMA 3 variants use `use_scaled_rope` and higher `rope_theta` (500000).
 * 70B models use GQA with 8 KV heads.
 */
typedef struct {
    SNEPPXModelFamily family;          /**< Must be SNEPPX_MODEL_LLAMA_2 or LLAMA_3 */
    size_t hidden_size;                /**< Hidden/embedding dimension (e.g. 4096) */
    size_t intermediate_size;          /**< FFN intermediate dimension */
    size_t num_hidden_layers;          /**< Number of transformer blocks */
    size_t num_attention_heads;        /**< Number of query heads */
    size_t num_key_value_heads;        /**< GQA: KV heads (can differ from num_attention_heads) */
    size_t vocab_size;                 /**< Vocabulary size */
    size_t max_position_embeddings;    /**< Maximum sequence length (base, before scaling) */
    float  rms_norm_eps;               /**< RMSNorm epsilon */
    float  rope_theta;                 /**< RoPE theta (10000 for LLaMA 2, 500000 for LLaMA 3) */
    int    use_scaled_rope;            /**< LLaMA 3: scaled RoPE variant */
    int    tie_word_embeddings;        /**< Tie input/output embeddings */
    int    hidden_act;                 /**< Hidden activation: 0=SiLU, 1=GELU, 2=ReLU */
    int    num_hidden_groups;          /**< Grouped-query attention groups */
    int    head_dim;                   /**< Per-head dimension (0 = auto from hidden/heads) */
    size_t num_experts;                /**< MoE expert count (0 if dense) */
    size_t num_experts_per_tok;        /**< Active experts per token */
    int    use_flash_attn;             /**< Enable Flash Attention kernel */
    int    use_sdpa;                   /**< Enable scaled dot-product attention */
    int    sliding_window;             /**< Sliding window size (0 = global attention) */
    float  attention_dropout;          /**< Attention dropout rate */
    float  hidden_dropout;             /**< Hidden dropout rate */
} SNEPPXLlamaConfig;

/**
 * @brief Configuration for Mistral architecture.
 *
 * Features sliding window attention (default 4096) and GQA with 8 KV heads.
 */
typedef struct {
    SNEPPXModelFamily family;          /**< Must be SNEPPX_MODEL_MISTRAL */
    size_t hidden_size;                /**< Hidden/embedding dimension */
    size_t intermediate_size;          /**< FFN intermediate dimension */
    size_t num_hidden_layers;          /**< Number of transformer blocks */
    size_t num_attention_heads;        /**< Number of query heads */
    size_t num_key_value_heads;        /**< GQA: key-value heads */
    size_t vocab_size;                 /**< Vocabulary size */
    size_t max_position_embeddings;    /**< Maximum sequence length */
    float  rms_norm_eps;               /**< RMSNorm epsilon */
    float  rope_theta;                 /**< RoPE theta */
    int    sliding_window;             /**< Sliding window size (4096 for Mistral) */
    int    head_dim;                   /**< Per-head dimension */
    float  attention_dropout;          /**< Attention dropout rate */
    float  hidden_dropout;             /**< Hidden dropout rate */
} SNEPPXMistralConfig;

/**
 * @brief Configuration for Qwen 2 architecture.
 *
 * Supports rope scaling (YARN / NTK-aware) via rope_scaling_factor
 * and dual chunk attention (DCA) for extended context.
 */
typedef struct {
    SNEPPXModelFamily family;          /**< Must be SNEPPX_MODEL_QWEN_2 */
    size_t hidden_size;                /**< Hidden/embedding dimension */
    size_t intermediate_size;          /**< FFN intermediate dimension */
    size_t num_hidden_layers;          /**< Number of transformer blocks */
    size_t num_attention_heads;        /**< Number of query heads */
    size_t num_key_value_heads;        /**< GQA: key-value heads */
    size_t vocab_size;                 /**< Vocabulary size */
    size_t max_position_embeddings;    /**< Base maximum sequence length */
    float  rms_norm_eps;               /**< RMSNorm epsilon */
    float  rope_theta;                 /**< RoPE theta */
    float  rope_scaling_factor;        /**< YARN / NTK scaling factor for context extension */
    int    use_rope_scaling;           /**< Enable rope scaling */
    int    head_dim;                   /**< Per-head dimension */
    float  attention_dropout;          /**< Attention dropout rate */
    float  hidden_dropout;             /**< Hidden dropout rate */
} SNEPPXQwen2Config;

/**
 * @brief Configuration for DeepSeek V2 architecture.
 *
 * Uses Multi-Head Latent Attention (MLA) with low-rank KV absorption.
 * kv_lora_rank = 512, q_lora_rank = 1536 for the V2 Full variant.
 */
typedef struct {
    SNEPPXModelFamily family;          /**< Must be SNEPPX_MODEL_DEEPSEEK_V2 */
    size_t hidden_size;                /**< Hidden/embedding dimension */
    size_t intermediate_size;          /**< FFN intermediate dimension */
    size_t num_hidden_layers;          /**< Number of transformer blocks */
    size_t num_attention_heads;        /**< Number of query heads */
    size_t num_key_value_heads;        /**< KV heads for MLA absorption */
    size_t vocab_size;                 /**< Vocabulary size */
    size_t max_position_embeddings;    /**< Maximum sequence length */
    float  rms_norm_eps;               /**< RMSNorm epsilon */
    float  rope_theta;                 /**< RoPE theta */
    int    head_dim;                   /**< Per-head dimension */
    int    kv_lora_rank;               /**< MLA: absorbed KV rank (DeepSeek V2: 512) */
    int    q_lora_rank;                /**< MLA: absorbed Q rank (optional) */
    float  attention_dropout;          /**< Attention dropout rate */
    float  hidden_dropout;             /**< Hidden dropout rate */
    int    use_flash_attn;             /**< Enable Flash Attention kernel */
} SNEPPXDeepSeekV2Config;

/**
 * @brief Generic LLM configuration union wrapper.
 *
 * Tagged union that holds any supported model configuration.
 * Used by all API functions for type-agnostic access.
 */
typedef struct {
    SNEPPXModelFamily family;          /**< Discriminant selecting the active union member */
    union {
        SNEPPXLlamaConfig      llama;  /**< LLaMA 2/3 config active when family is LLAMA_2 or LLAMA_3 */
        SNEPPXMistralConfig    mistral; /**< Mistral config */
        SNEPPXQwen2Config      qwen2;   /**< Qwen 2 config */
        SNEPPXDeepSeekV2Config deepseek_v2; /**< DeepSeek V2 config */
    } config;
} SNEPPXLLMConfig;

/**
 * @brief Create a named configuration preset.
 *
 * @param family  Model family name (e.g. "llama3", "mistral", "qwen2", "deepseek_v2").
 * @param size    Model size variant (e.g. "8B", "70B", "7B", "Lite", "Full").
 * @param out     Output configuration structure.
 * @return 0 on success, nonzero on error (unknown family or size).
 */
int SNEPPX_llm_config_from_name(const char* family, const char* size,
                                 SNEPPXLLMConfig* out);

/**
 * @brief Serialize an LLM configuration to JSON.
 *
 * @param cfg  Pointer to the configuration.
 * @return Newly allocated JSON string (caller must free with free()), or NULL on error.
 */
char* SNEPPX_llm_config_to_json(const SNEPPXLLMConfig* cfg);

/**
 * @brief Parse an LLM configuration from a JSON string.
 *
 * @param json  Null-terminated JSON string.
 * @param out   Output configuration structure.
 * @return 0 on success, nonzero on parse error.
 */
int SNEPPX_llm_config_from_json(const char* json, SNEPPXLLMConfig* out);

/**
 * @brief Extend a model's context window using NTK-aware and YaRN RoPE scaling.
 *
 * Adjusts max_position_embeddings, rope_theta, and scaling factors so the
 * model can handle sequences up to `target_len`. Supports 128K context
 * extension for all architectures (LLaMA 2/3, Mistral, Qwen 2, DeepSeek V2).
 *
 * @param cfg        Pointer to the configuration (modified in place).
 * @param target_len Desired maximum sequence length (e.g. 131072 for 128K).
 * @return 0 on success, nonzero if target_len is invalid or scaling fails.
 */
int SNEPPX_llm_config_extend_context(SNEPPXLLMConfig* cfg, size_t target_len);

/**
 * @brief Get the weight name prefix for a given model family.
 *
 * @param family  The model family.
 * @return String prefix (e.g. "model.layers" for LLaMA), or NULL for UNKNOWN.
 */
const char* SNEPPX_llm_weight_prefix(SNEPPXModelFamily family);

/**
 * @brief Compute the number of weight tensors for a model family.
 *
 * @param family     The model family.
 * @param num_layers Number of transformer layers.
 * @return The total number of named weight tensors.
 */
int SNEPPX_llm_num_weight_tensors(SNEPPXModelFamily family,
                                    size_t num_layers);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_MODEL_ZOO_H */

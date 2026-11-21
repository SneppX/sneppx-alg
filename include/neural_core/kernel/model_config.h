#ifndef NEURAL_CORE_KERNEL_MODEL_CONFIG_H
#define NEURAL_CORE_KERNEL_MODEL_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
/*
 * SNEPPX - Model Config
 *
 * WHAT
 *   Model Config.
 *
 * CONCEPT
 *   Provides the Model Config.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
#endif

/**
 * @file model_config.h
 * @brief Universal model configuration system.
 *
 * Provides a unified ModelConfig structure that can represent any model
 * architecture (Transformer, ViT, Diffusion, RNN, GAN, GCN, and sneppx
 * primitives HSS/NPE/SER/ARC). Supports JSON serialization, file I/O,
 * validation, and preset configurations.
 */

/** Forward declaration */
typedef struct ModelConfig ModelConfig;
/** Forward declaration */
typedef struct ModelConfigList ModelConfigList;

/**
 * @brief Supported model architecture types.
 */
typedef enum {
    MODEL_ARCH_TRANSFORMER = 0, /**< Decoder-only transformer (GPT, LLaMA, etc.) */
    MODEL_ARCH_VIT        = 1, /**< Vision Transformer */
    MODEL_ARCH_DIFFUSION  = 2, /**< Diffusion model (SDXL, etc.) */
    MODEL_ARCH_RNN        = 3, /**< Recurrent neural network (LSTM, GRU) */
    MODEL_ARCH_GAN        = 4, /**< Generative adversarial network */
    MODEL_ARCH_GCN        = 5, /**< Graph convolutional network */
    MODEL_ARCH_HSS        = 6, /**< Hierarchical State Space (sneppx primitive) */
    MODEL_ARCH_NPE        = 7, /**< Neural Predictive Engine (sneppx primitive) */
    MODEL_ARCH_SER        = 8, /**< Signal Enhancement & Reconstruction (sneppx primitive) */
    MODEL_ARCH_ARC        = 9, /**< Adaptive Resonance Compressor (sneppx primitive) */
    MODEL_ARCH_CUSTOM     = 255 /**< User-defined architecture */
} ModelArchitecture;

/**
 * @brief Activation function types.
 */
typedef enum {
    ACTIVATION_GELU   = 0, /**< Gaussian Error Linear Unit */
    ACTIVATION_RELU   = 1, /**< Rectified Linear Unit */
    ACTIVATION_SILU   = 2, /**< Sigmoid Linear Unit (SiLU / Swish) */
    ACTIVATION_GEGLU  = 3, /**< Gated GELU (GLU variant) */
    ACTIVATION_SWIGLU = 4  /**< Gated SiLU (GLU variant, used in LLaMA) */
} ActivationType;

/**
 * @brief Position encoding types.
 */
typedef enum {
    POS_ENCODING_LEARNED     = 0, /**< Learned absolute position embeddings */
    POS_ENCODING_ROPE        = 1, /**< Rotary Position Embeddings (RoPE) */
    POS_ENCODING_ALIBI       = 2, /**< ALiBi linear bias */
    POS_ENCODING_SINUSOIDAL  = 3  /**< Fixed sinusoidal encodings */
} PositionEncodingType;

/**
 * @brief Attention mechanism types.
 */
typedef enum {
    ATTENTION_CAUSAL         = 0, /**< Standard causal self-attention */
    ATTENTION_FLASH          = 1, /**< Flash Attention (memory-efficient) */
    ATTENTION_SLIDING_WINDOW = 2, /**< Sliding window attention */
    ATTENTION_BLOCK_SPARSE   = 3  /**< Block-sparse attention */
} AttentionType;

/**
 * @brief Universal model configuration structure.
 *
 * Covers identity metadata, architecture parameters, normalization,
 * activations, attention, position encoding, quantization, distributed
 * training, MoE, and custom key-value fields.
 */
struct ModelConfig {
    /* ----- Identity ----- */
    char *name;           /**< Model name */
    char *version;        /**< Model version */
    char *description;    /**< Human-readable description */
    char *author;         /**< Author or organization */
    char *license;        /**< License identifier */

    /* ----- Architecture ----- */
    ModelArchitecture architecture; /**< Model type */
    int vocab_size;                 /**< Vocabulary size */
    int hidden_size;                /**< Hidden/embedding dimension */
    int num_layers;                 /**< Number of transformer/backbone layers */
    int num_heads;                  /**< Number of attention heads */
    int num_kv_heads;               /**< KV heads for GQA (0 = equal to num_heads) */
    int intermediate_size;          /**< FFN intermediate dimension */
    int max_position_embeddings;    /**< Base maximum sequence length */
    int max_seq_len;                /**< Runtime maximum sequence length */

    /* ----- Normalization ----- */
    float layer_norm_eps;  /**< LayerNorm epsilon */
    float rms_norm_eps;    /**< RMSNorm epsilon */
    int use_rms_norm;      /**< 1 for RMSNorm, 0 for LayerNorm */

    /* ----- Activations ----- */
    ActivationType hidden_act; /**< Hidden layer activation */
    ActivationType ffn_act;    /**< FFN activation */
    int gated_ffn;             /**< 1 for gated FFN (SwiGLU/GEGLU) */

    /* ----- Attention ----- */
    AttentionType attention_type; /**< Causal, flash, sliding window, etc. */
    float attention_dropout;       /**< Attention dropout rate */
    float hidden_dropout;          /**< Hidden dropout rate */
    int use_flash_attention;       /**< Enable Flash Attention kernel */
    int sliding_window;            /**< Sliding window size (0 = global) */

    /* ----- Position encoding ----- */
    PositionEncodingType pos_encoding; /**< RoPE, ALiBi, learned, etc. */
    float rope_theta;                  /**< RoPE theta base frequency */
    int rope_scaling;                  /**< 0=none, 1=linear, 2=dynamic (NTK-aware/YaRN) */
    float rope_scaling_factor;         /**< Scaling factor for extended context */

    /* ----- Initialization ----- */
    float initializer_range; /**< Weight initialization range */
    int tie_word_embeddings; /**< Tie input and output embeddings */

    /* ----- Quantization ----- */
    int quantize;          /**< 1 if quantized */
    int quant_bits;        /**< Bit width: 4, 8, etc. */
    int quant_group_size;  /**< Group size for grouped quantization */

    /* ----- Distributed ----- */
    int tensor_parallel_size;   /**< TP degree */
    int pipeline_parallel_size; /**< PP degree */
    int sequence_parallel;      /**< 1 for sequence parallelism */

    /* ----- Training ----- */
    float learning_rate;                /**< Learning rate */
    float weight_decay;                 /**< Weight decay */
    float max_grad_norm;                /**< Gradient clipping norm */
    int warmup_steps;                   /**< LR warmup steps */
    int max_steps;                      /**< Total training steps */
    int gradient_accumulation_steps;    /**< Gradient accumulation steps */
    int mixed_precision;                /**< 1 for FP16/BF16 mixed precision */

    /* ----- MoE ----- */
    int num_experts;              /**< Total MoE experts (0 = dense) */
    int num_experts_per_token;    /**< Active experts per token */
    float router_aux_loss_coef;   /**< Auxiliary loss coefficient for load balancing */

    /* ----- Custom fields ----- */
    char **custom_keys;   /**< Custom key names */
    char **custom_values; /**< Custom key values */
    int custom_count;     /**< Number of custom key-value pairs */
};

/**
 * @brief A dynamic list of ModelConfig pointers.
 */
struct ModelConfigList {
    ModelConfig **configs; /**< Array of config pointers */
    int count;             /**< Current number of configs */
    int capacity;          /**< Allocated capacity */
};

/* ---- Lifecycle ---- */

/** @brief Create an empty ModelConfig (fields default to 0/NULL). */
ModelConfig *model_config_create(void);

/** @brief Destroy a ModelConfig and free all owned memory. */
void model_config_destroy(ModelConfig *config);

/** @brief Create an empty ModelConfigList. */
ModelConfigList *model_config_list_create(void);

/** @brief Destroy a ModelConfigList (including contained configs). */
void model_config_list_destroy(ModelConfigList *list);

/* ---- Serialization ---- */

/** @brief Serialize to JSON string (caller must free with free()). */
char *model_config_to_json(const ModelConfig *config, int pretty);

/** @brief Deserialize from JSON string. Returns NULL on error. */
ModelConfig *model_config_from_json(const char *json);

/* ---- File I/O ---- */

/** @brief Save config to a JSON file. Returns 0 on success. */
int model_config_save(const ModelConfig *config, const char *path);

/** @brief Load config from a JSON file. Returns NULL on error. */
ModelConfig *model_config_load(const char *path);

/* ---- Validation ---- */

/**
 * @brief Validate a ModelConfig.
 * @param config    Config to validate.
 * @param error_out If non-NULL and validation fails, set to an error string (caller must free).
 * @return 1 if valid, 0 if invalid.
 */
int model_config_validate(const ModelConfig *config, char **error_out);

/* ---- Presets ---- */

ModelConfig *model_config_llama2_7b(void);    /**< LLaMA 2 7B preset */
ModelConfig *model_config_llama2_13b(void);   /**< LLaMA 2 13B preset */
ModelConfig *model_config_llama3_8b(void);    /**< LLaMA 3 8B preset */
ModelConfig *model_config_mistral_7b(void);   /**< Mistral 7B preset */
ModelConfig *model_config_qwen2_7b(void);     /**< Qwen 2 7B preset */
ModelConfig *model_config_bert_base(void);    /**< BERT base preset */
ModelConfig *model_config_vit_base(void);     /**< ViT base preset */
ModelConfig *model_config_sdxl(void);         /**< SDXL preset */

/* ---- Utilities ---- */

/** @brief Deep copy a ModelConfig. Returns NULL on error. */
ModelConfig *model_config_copy(const ModelConfig *config);

/** @brief Merge override fields from second config onto first (in place). */
void model_config_merge(ModelConfig *base, const ModelConfig *override);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_CORE_KERNEL_MODEL_CONFIG_H

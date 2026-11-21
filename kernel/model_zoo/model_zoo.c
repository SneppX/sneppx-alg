#include "model_zoo.h"
#include "../../fs/format/json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
/*
 * SNEPPX - Model Zoo and Config
 *
 * WHAT
 *   Model Zoo and Config.
 *
 * CONCEPT
 *   Model Zoo and Config implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/* LLaMA 2 presets */
static const SNEPPXLlamaConfig llama2_7b = {
    .family = SNEPPX_MODEL_LLAMA_2,
    .hidden_size = 4096,
    .intermediate_size = 11008,
    .num_hidden_layers = 32,
    .num_attention_heads = 32,
    .num_key_value_heads = 32,
    .vocab_size = 32000,
    .max_position_embeddings = 4096,
    .rms_norm_eps = 1e-5f,
    .rope_theta = 10000.0f,
    .use_scaled_rope = 0,
    .tie_word_embeddings = 0,
    .hidden_act = 0,
    .head_dim = 128,
    .attention_dropout = 0.0f,
    .hidden_dropout = 0.0f,
};

static const SNEPPXLlamaConfig llama2_13b = {
    .family = SNEPPX_MODEL_LLAMA_2,
    .hidden_size = 5120,
    .intermediate_size = 13824,
    .num_hidden_layers = 40,
    .num_attention_heads = 40,
    .num_key_value_heads = 40,
    .vocab_size = 32000,
    .max_position_embeddings = 4096,
    .rms_norm_eps = 1e-5f,
    .rope_theta = 10000.0f,
    .head_dim = 128,
};

static const SNEPPXLlamaConfig llama2_70b = {
    .family = SNEPPX_MODEL_LLAMA_2,
    .hidden_size = 8192,
    .intermediate_size = 28672,
    .num_hidden_layers = 80,
    .num_attention_heads = 64,
    .num_key_value_heads = 8,
    .vocab_size = 32000,
    .max_position_embeddings = 4096,
    .rms_norm_eps = 1e-5f,
    .rope_theta = 10000.0f,
    .head_dim = 128,
};

/* LLaMA 3 presets */
static const SNEPPXLlamaConfig llama3_8b = {
    .family = SNEPPX_MODEL_LLAMA_3,
    .hidden_size = 4096,
    .intermediate_size = 14336,
    .num_hidden_layers = 32,
    .num_attention_heads = 32,
    .num_key_value_heads = 8,
    .vocab_size = 128256,
    .max_position_embeddings = 8192,
    .rms_norm_eps = 1e-5f,
    .rope_theta = 500000.0f,
    .use_scaled_rope = 1,
    .head_dim = 128,
};

static const SNEPPXLlamaConfig llama3_70b = {
    .family = SNEPPX_MODEL_LLAMA_3,
    .hidden_size = 8192,
    .intermediate_size = 28672,
    .num_hidden_layers = 80,
    .num_attention_heads = 64,
    .num_key_value_heads = 8,
    .vocab_size = 128256,
    .max_position_embeddings = 8192,
    .rms_norm_eps = 1e-5f,
    .rope_theta = 500000.0f,
    .use_scaled_rope = 1,
    .head_dim = 128,
};

/* Mistral presets */
static const SNEPPXMistralConfig mistral_7b = {
    .family = SNEPPX_MODEL_MISTRAL,
    .hidden_size = 4096,
    .intermediate_size = 14336,
    .num_hidden_layers = 32,
    .num_attention_heads = 32,
    .num_key_value_heads = 8,
    .vocab_size = 32000,
    .max_position_embeddings = 32768,
    .rms_norm_eps = 1e-5f,
    .rope_theta = 10000.0f,
    .sliding_window = 4096,
    .head_dim = 128,
};

/* Qwen 2 presets */
static const SNEPPXQwen2Config qwen2_7b = {
    .family = SNEPPX_MODEL_QWEN_2,
    .hidden_size = 3584,
    .intermediate_size = 18944,
    .num_hidden_layers = 28,
    .num_attention_heads = 28,
    .num_key_value_heads = 4,
    .vocab_size = 152064,
    .max_position_embeddings = 32768,
    .rms_norm_eps = 1e-6f,
    .rope_theta = 1000000.0f,
    .rope_scaling_factor = 1.0f,
    .use_rope_scaling = 0,
    .head_dim = 128,
};

static const SNEPPXQwen2Config qwen2_72b = {
    .family = SNEPPX_MODEL_QWEN_2,
    .hidden_size = 8192,
    .intermediate_size = 29568,
    .num_hidden_layers = 80,
    .num_attention_heads = 64,
    .num_key_value_heads = 8,
    .vocab_size = 152064,
    .max_position_embeddings = 32768,
    .rms_norm_eps = 1e-6f,
    .rope_theta = 1000000.0f,
    .rope_scaling_factor = 1.0f,
    .use_rope_scaling = 0,
    .head_dim = 128,
};

/* DeepSeek V2 presets */
static const SNEPPXDeepSeekV2Config deepseek_v2_lite = {
    .family = SNEPPX_MODEL_DEEPSEEK_V2,
    .hidden_size = 2048,
    .intermediate_size = 10944,
    .num_hidden_layers = 27,
    .num_attention_heads = 16,
    .num_key_value_heads = 16,
    .vocab_size = 102400,
    .max_position_embeddings = 4096,
    .rms_norm_eps = 1e-6f,
    .rope_theta = 10000.0f,
    .head_dim = 128,
    .kv_lora_rank = 512,
    .q_lora_rank = 1536,
};

static const SNEPPXDeepSeekV2Config deepseek_v2_full = {
    .family = SNEPPX_MODEL_DEEPSEEK_V2,
    .hidden_size = 5120,
    .intermediate_size = 12288,
    .num_hidden_layers = 60,
    .num_attention_heads = 64,
    .num_key_value_heads = 64,
    .vocab_size = 102400,
    .max_position_embeddings = 4096,
    .rms_norm_eps = 1e-6f,
    .rope_theta = 10000.0f,
    .head_dim = 128,
    .kv_lora_rank = 512,
    .q_lora_rank = 1536,
};

/* =========================================================================
 * API: create config from family + size name
 * ========================================================================= */

/**
 * @brief Perform Llm Config From Name.
 *
 * @param family [in] Family value.
 * @param size [in] Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_llm_config_from_name(const char* family, const char* size,
                                 SNEPPXLLMConfig* out) {
    if (!family || !size || !out) return -1;
    memset(out, 0, sizeof(SNEPPXLLMConfig));

    if (strcmp(family, "llama2") == 0 || strcmp(family, "llama-2") == 0) {
        out->family = SNEPPX_MODEL_LLAMA_2;
        if (strcmp(size, "7B") == 0 || strcmp(size, "7b") == 0) {
            out->config.llama = llama2_7b;
        } else if (strcmp(size, "13B") == 0 || strcmp(size, "13b") == 0) {
            out->config.llama = llama2_13b;
        } else if (strcmp(size, "70B") == 0 || strcmp(size, "70b") == 0) {
            out->config.llama = llama2_70b;
        } else return -1;
        return 0;
    }
    if (strcmp(family, "llama3") == 0 || strcmp(family, "llama-3") == 0) {
        out->family = SNEPPX_MODEL_LLAMA_3;
        if (strcmp(size, "8B") == 0 || strcmp(size, "8b") == 0) {
            out->config.llama = llama3_8b;
        } else if (strcmp(size, "70B") == 0 || strcmp(size, "70b") == 0) {
            out->config.llama = llama3_70b;
        } else return -1;
        return 0;
    }
    if (strcmp(family, "mistral") == 0) {
        out->family = SNEPPX_MODEL_MISTRAL;
        if (strcmp(size, "7B") == 0 || strcmp(size, "7b") == 0) {
            out->config.mistral = mistral_7b;
        } else return -1;
        return 0;
    }
    if (strcmp(family, "qwen2") == 0 || strcmp(family, "qwen") == 0) {
        out->family = SNEPPX_MODEL_QWEN_2;
        if (strcmp(size, "7B") == 0 || strcmp(size, "7b") == 0) {
            out->config.qwen2 = qwen2_7b;
        } else if (strcmp(size, "72B") == 0 || strcmp(size, "72b") == 0) {
            out->config.qwen2 = qwen2_72b;
        } else return -1;
        return 0;
    }
    if (strcmp(family, "deepseek") == 0 || strcmp(family, "deepseek-v2") == 0) {
        out->family = SNEPPX_MODEL_DEEPSEEK_V2;
        if (strcmp(size, "lite") == 0) {
            out->config.deepseek_v2 = deepseek_v2_lite;
        } else if (strcmp(size, "full") == 0 || strcmp(size, "236B") == 0) {
            out->config.deepseek_v2 = deepseek_v2_full;
        } else return -1;
        return 0;
    }
    return -1;
}

/* =========================================================================
 * API: serialize to JSON
 * ========================================================================= */

static void json_escape(char* out, size_t out_size, const char* in) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j < out_size - 1; i++) {
        if (in[i] == '"' || in[i] == '\\') {
            if (j < out_size - 2) out[j++] = '\\';
        }
        out[j++] = in[i];
    }
    out[j] = '\0';
}

/**
 * @brief Perform Llm Config To Json.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_llm_config_to_json(const SNEPPXLLMConfig* cfg) {
    if (!cfg) return NULL;
    char buf[4096];
    char family_buf[64];
    const char* family_str = "unknown";
    switch (cfg->family) {
        case SNEPPX_MODEL_LLAMA_2: family_str = "llama2"; break;
        case SNEPPX_MODEL_LLAMA_3: family_str = "llama3"; break;
        case SNEPPX_MODEL_MISTRAL: family_str = "mistral"; break;
        case SNEPPX_MODEL_QWEN_2:  family_str = "qwen2"; break;
        case SNEPPX_MODEL_DEEPSEEK_V2: family_str = "deepseek_v2"; break;
        default: break;
    }
    json_escape(family_buf, sizeof(family_buf), family_str);

    switch (cfg->family) {
        case SNEPPX_MODEL_LLAMA_2:
        case SNEPPX_MODEL_LLAMA_3: {
            const SNEPPXLlamaConfig* l = &cfg->config.llama;
            snprintf(buf, sizeof(buf),
                "{\"family\":\"%s\",\"hidden_size\":%zu,\"intermediate_size\":%zu,"
                "\"num_hidden_layers\":%zu,\"num_attention_heads\":%zu,"
                "\"num_key_value_heads\":%zu,\"vocab_size\":%zu,"
                "\"max_position_embeddings\":%zu,\"rms_norm_eps\":%g,"
                "\"rope_theta\":%g,\"use_scaled_rope\":%d,\"head_dim\":%d}",
                family_buf,
                l->hidden_size, l->intermediate_size, l->num_hidden_layers,
                l->num_attention_heads, l->num_key_value_heads, l->vocab_size,
                l->max_position_embeddings, l->rms_norm_eps,
                l->rope_theta, l->use_scaled_rope, l->head_dim);
            break;
        }
        case SNEPPX_MODEL_MISTRAL: {
            const SNEPPXMistralConfig* m = &cfg->config.mistral;
            snprintf(buf, sizeof(buf),
                "{\"family\":\"%s\",\"hidden_size\":%zu,\"intermediate_size\":%zu,"
                "\"num_hidden_layers\":%zu,\"num_attention_heads\":%zu,"
                "\"num_key_value_heads\":%zu,\"vocab_size\":%zu,"
                "\"max_position_embeddings\":%zu,\"rms_norm_eps\":%g,"
                "\"rope_theta\":%g,\"sliding_window\":%d,\"head_dim\":%d}",
                family_buf,
                m->hidden_size, m->intermediate_size, m->num_hidden_layers,
                m->num_attention_heads, m->num_key_value_heads, m->vocab_size,
                m->max_position_embeddings, m->rms_norm_eps,
                m->rope_theta, m->sliding_window, m->head_dim);
            break;
        }
        case SNEPPX_MODEL_QWEN_2: {
            const SNEPPXQwen2Config* q = &cfg->config.qwen2;
            snprintf(buf, sizeof(buf),
                "{\"family\":\"%s\",\"hidden_size\":%zu,\"intermediate_size\":%zu,"
                "\"num_hidden_layers\":%zu,\"num_attention_heads\":%zu,"
                "\"num_key_value_heads\":%zu,\"vocab_size\":%zu,"
                "\"max_position_embeddings\":%zu,\"rms_norm_eps\":%g,"
                "\"rope_theta\":%g,\"rope_scaling_factor\":%g,"
                "\"use_rope_scaling\":%d,\"head_dim\":%d}",
                family_buf,
                q->hidden_size, q->intermediate_size, q->num_hidden_layers,
                q->num_attention_heads, q->num_key_value_heads, q->vocab_size,
                q->max_position_embeddings, q->rms_norm_eps,
                q->rope_theta, q->rope_scaling_factor,
                q->use_rope_scaling, q->head_dim);
            break;
        }
        case SNEPPX_MODEL_DEEPSEEK_V2: {
            const SNEPPXDeepSeekV2Config* d = &cfg->config.deepseek_v2;
            snprintf(buf, sizeof(buf),
                "{\"family\":\"%s\",\"hidden_size\":%zu,\"intermediate_size\":%zu,"
                "\"num_hidden_layers\":%zu,\"num_attention_heads\":%zu,"
                "\"num_key_value_heads\":%zu,\"vocab_size\":%zu,"
                "\"max_position_embeddings\":%zu,\"rms_norm_eps\":%g,"
                "\"rope_theta\":%g,\"kv_lora_rank\":%d,\"q_lora_rank\":%d,"
                "\"head_dim\":%d}",
                family_buf,
                d->hidden_size, d->intermediate_size, d->num_hidden_layers,
                d->num_attention_heads, d->num_key_value_heads, d->vocab_size,
                d->max_position_embeddings, d->rms_norm_eps,
                d->rope_theta, d->kv_lora_rank, d->q_lora_rank, d->head_dim);
            break;
        }
        default:
            return NULL;
    }
#ifdef _WIN32
    return _strdup(buf);
#else
    return strdup(buf);
#endif
}

/* =========================================================================
 * API: parse from JSON (recursive descent parser)
 * ========================================================================= */

static int json_get_size_t(SNEPPX_JsonValue* root, const char* key, size_t* out) {
    SNEPPX_JsonValue* val = sneppx_json_find(root, key);
    if (!val || val->type != SNEPPX_JSON_NUMBER) return -1;
    *out = (size_t)val->number_val;
    return 0;
}

static int json_get_float(SNEPPX_JsonValue* root, const char* key, float* out) {
    SNEPPX_JsonValue* val = sneppx_json_find(root, key);
    if (!val || val->type != SNEPPX_JSON_NUMBER) return -1;
    *out = (float)val->number_val;
    return 0;
}

static int json_get_int(SNEPPX_JsonValue* root, const char* key, int* out) {
    SNEPPX_JsonValue* val = sneppx_json_find(root, key);
    if (!val || val->type != SNEPPX_JSON_NUMBER) return -1;
    *out = (int)val->number_val;
    return 0;
}

/**
 * @brief Perform Llm Config From Json.
 *
 * @param json_str [in] JSON string.
 * @param out [out] Config output.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_llm_config_from_json(const char* json_str, SNEPPXLLMConfig* out) {
    if (!json_str || !out) return -1;
    memset(out, 0, sizeof(SNEPPXLLMConfig));

    SNEPPX_JsonValue* root = sneppx_json_parse(json_str);
    if (!root) return -1;

    SNEPPX_JsonValue* family_val = sneppx_json_find(root, "family");
    if (!family_val || family_val->type != SNEPPX_JSON_STRING) {
        sneppx_json_free(root);
        return -1;
    }
    const char* family = family_val->string_val;

    if (strcmp(family, "llama2") == 0 || strcmp(family, "llama3") == 0) {
        out->family = (strcmp(family, "llama3") == 0) ? SNEPPX_MODEL_LLAMA_3 : SNEPPX_MODEL_LLAMA_2;
        SNEPPXLlamaConfig* l = &out->config.llama;
        l->family = out->family;
        json_get_size_t(root, "hidden_size", &l->hidden_size);
        json_get_size_t(root, "intermediate_size", &l->intermediate_size);
        json_get_size_t(root, "num_hidden_layers", &l->num_hidden_layers);
        json_get_size_t(root, "num_attention_heads", &l->num_attention_heads);
        json_get_size_t(root, "num_key_value_heads", &l->num_key_value_heads);
        json_get_size_t(root, "vocab_size", &l->vocab_size);
        json_get_size_t(root, "max_position_embeddings", &l->max_position_embeddings);
        json_get_float(root, "rms_norm_eps", &l->rms_norm_eps);
        json_get_float(root, "rope_theta", &l->rope_theta);
        json_get_int(root, "use_scaled_rope", &l->use_scaled_rope);
        json_get_int(root, "head_dim", &l->head_dim);
        sneppx_json_free(root);
        return 0;
    }
    if (strcmp(family, "mistral") == 0) {
        out->family = SNEPPX_MODEL_MISTRAL;
        SNEPPXMistralConfig* m = &out->config.mistral;
        m->family = out->family;
        json_get_size_t(root, "hidden_size", &m->hidden_size);
        json_get_size_t(root, "intermediate_size", &m->intermediate_size);
        json_get_size_t(root, "num_hidden_layers", &m->num_hidden_layers);
        json_get_size_t(root, "num_attention_heads", &m->num_attention_heads);
        json_get_size_t(root, "num_key_value_heads", &m->num_key_value_heads);
        json_get_size_t(root, "vocab_size", &m->vocab_size);
        json_get_size_t(root, "max_position_embeddings", &m->max_position_embeddings);
        json_get_float(root, "rms_norm_eps", &m->rms_norm_eps);
        json_get_float(root, "rope_theta", &m->rope_theta);
        json_get_int(root, "sliding_window", &m->sliding_window);
        json_get_int(root, "head_dim", &m->head_dim);
        sneppx_json_free(root);
        return 0;
    }
    if (strcmp(family, "qwen2") == 0) {
        out->family = SNEPPX_MODEL_QWEN_2;
        SNEPPXQwen2Config* q = &out->config.qwen2;
        q->family = out->family;
        json_get_size_t(root, "hidden_size", &q->hidden_size);
        json_get_size_t(root, "intermediate_size", &q->intermediate_size);
        json_get_size_t(root, "num_hidden_layers", &q->num_hidden_layers);
        json_get_size_t(root, "num_attention_heads", &q->num_attention_heads);
        json_get_size_t(root, "num_key_value_heads", &q->num_key_value_heads);
        json_get_size_t(root, "vocab_size", &q->vocab_size);
        json_get_size_t(root, "max_position_embeddings", &q->max_position_embeddings);
        json_get_float(root, "rms_norm_eps", &q->rms_norm_eps);
        json_get_float(root, "rope_theta", &q->rope_theta);
        json_get_float(root, "rope_scaling_factor", &q->rope_scaling_factor);
        json_get_int(root, "use_rope_scaling", &q->use_rope_scaling);
        json_get_int(root, "head_dim", &q->head_dim);
        sneppx_json_free(root);
        return 0;
    }
    if (strcmp(family, "deepseek_v2") == 0) {
        out->family = SNEPPX_MODEL_DEEPSEEK_V2;
        SNEPPXDeepSeekV2Config* d = &out->config.deepseek_v2;
        d->family = out->family;
        json_get_size_t(root, "hidden_size", &d->hidden_size);
        json_get_size_t(root, "intermediate_size", &d->intermediate_size);
        json_get_size_t(root, "num_hidden_layers", &d->num_hidden_layers);
        json_get_size_t(root, "num_attention_heads", &d->num_attention_heads);
        json_get_size_t(root, "num_key_value_heads", &d->num_key_value_heads);
        json_get_size_t(root, "vocab_size", &d->vocab_size);
        json_get_size_t(root, "max_position_embeddings", &d->max_position_embeddings);
        json_get_float(root, "rms_norm_eps", &d->rms_norm_eps);
        json_get_float(root, "rope_theta", &d->rope_theta);
        json_get_int(root, "kv_lora_rank", &d->kv_lora_rank);
        json_get_int(root, "q_lora_rank", &d->q_lora_rank);
        json_get_int(root, "head_dim", &d->head_dim);
        sneppx_json_free(root);
        return 0;
    }
    sneppx_json_free(root);
    return -1;
}

/* =========================================================================
 * API: extend context with NTK-aware / YaRN RoPE scaling
 * ========================================================================= */

/**
 * @brief Perform Llm Config Extend Context.
 *
 * @param cfg [out] Cfg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_llm_config_extend_context(SNEPPXLLMConfig* cfg, size_t target_len) {
    if (!cfg || target_len == 0) return -1;

    switch (cfg->family) {
        case SNEPPX_MODEL_LLAMA_2:
        case SNEPPX_MODEL_LLAMA_3: {
            SNEPPXLlamaConfig* l = &cfg->config.llama;
            if (target_len <= l->max_position_embeddings) return 0;
            float scale = (float)target_len / (float)l->max_position_embeddings;
            l->rope_theta *= scale > 1.0f ? scale : 1.0f;
            l->use_scaled_rope = 1;
            l->max_position_embeddings = target_len;
            return 0;
        }
        case SNEPPX_MODEL_MISTRAL: {
            SNEPPXMistralConfig* m = &cfg->config.mistral;
            if (target_len <= m->max_position_embeddings) return 0;
            float scale = (float)target_len / (float)m->max_position_embeddings;
            m->rope_theta *= scale > 1.0f ? scale : 1.0f;
            m->max_position_embeddings = target_len;
            return 0;
        }
        case SNEPPX_MODEL_QWEN_2: {
            SNEPPXQwen2Config* q = &cfg->config.qwen2;
            if (target_len <= q->max_position_embeddings) return 0;
            float scale = (float)target_len / (float)q->max_position_embeddings;
            q->rope_theta *= scale > 1.0f ? scale : 1.0f;
            q->rope_scaling_factor = scale;
            q->use_rope_scaling = 1;
            q->max_position_embeddings = target_len;
            return 0;
        }
        case SNEPPX_MODEL_DEEPSEEK_V2: {
            SNEPPXDeepSeekV2Config* d = &cfg->config.deepseek_v2;
            if (target_len <= d->max_position_embeddings) return 0;
            float scale = (float)target_len / (float)d->max_position_embeddings;
            d->rope_theta *= scale > 1.0f ? scale : 1.0f;
            d->max_position_embeddings = target_len;
            return 0;
        }
        default:
            return -1;
    }
}

/* =========================================================================
 * API: weight name prefix helpers
 * ========================================================================= */

/**
 * @brief Perform Llm Weight Prefix.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_llm_weight_prefix(SNEPPXModelFamily family) {
    switch (family) {
        case SNEPPX_MODEL_LLAMA_2:
        case SNEPPX_MODEL_LLAMA_3:   return "model.layers";
        case SNEPPX_MODEL_MISTRAL:   return "model.layers";
        case SNEPPX_MODEL_QWEN_2:    return "model.layers";
        case SNEPPX_MODEL_DEEPSEEK_V2: return "model.layers";
        default: return "";
    }
}

/**
 * @brief Perform Llm Num Weight Tensors.
 *
 * @param family [in] Family value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_llm_num_weight_tensors(SNEPPXModelFamily family, size_t num_layers) {
    (void)family;
    return (int)(num_layers * 8 + 2);
}

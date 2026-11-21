#include "model_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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


// ── Internal helpers ────────────────────────────────────────────────────────

static char *dup_str(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (copy) memcpy(copy, s, len + 1);
    return copy;
}

static void free_str(char *s) {
    free(s);
}

static char *trim(char *str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == '\0') return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

// ── Create/destroy ──────────────────────────────────────────────────────────

ModelConfig *model_config_create(void) {
    ModelConfig *cfg = (ModelConfig *)calloc(1, sizeof(ModelConfig));
    if (!cfg) return NULL;

    // Defaults
    cfg->architecture = MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = 32000;
    cfg->hidden_size = 4096;
    cfg->num_layers = 32;
    cfg->num_heads = 32;
    cfg->num_kv_heads = 32;
    cfg->intermediate_size = 11008;
    cfg->max_position_embeddings = 2048;
    cfg->max_seq_len = 2048;
    cfg->layer_norm_eps = 1e-5f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 1;
    cfg->hidden_act = ACTIVATION_SILU;
    cfg->ffn_act = ACTIVATION_SILU;
    cfg->gated_ffn = 1;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->attention_dropout = 0.0f;
    cfg->hidden_dropout = 0.0f;
    cfg->use_flash_attention = 1;
    cfg->sliding_window = 0;
    cfg->pos_encoding = POS_ENCODING_ROPE;
    cfg->rope_theta = 10000.0f;
    cfg->rope_scaling = 0;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->quantize = 0;
    cfg->quant_bits = 0;
    cfg->quant_group_size = 128;
    cfg->tensor_parallel_size = 1;
    cfg->pipeline_parallel_size = 1;
    cfg->sequence_parallel = 0;
    cfg->learning_rate = 2e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    cfg->warmup_steps = 0;
    cfg->max_steps = 0;
    cfg->gradient_accumulation_steps = 1;
    cfg->mixed_precision = 1;
    cfg->num_experts = 0;
    cfg->num_experts_per_token = 0;
    cfg->router_aux_loss_coef = 0.0f;
    cfg->custom_keys = NULL;
    cfg->custom_values = NULL;
    cfg->custom_count = 0;

    return cfg;
}

void model_config_destroy(ModelConfig *config) {
    if (!config) return;

    free_str(config->name);
    free_str(config->version);
    free_str(config->description);
    free_str(config->author);
    free_str(config->license);

    for (int i = 0; i < config->custom_count; i++) {
        free_str(config->custom_keys[i]);
        free_str(config->custom_values[i]);
    }
    free(config->custom_keys);
    free(config->custom_values);

    free(config);
}

ModelConfigList *model_config_list_create(void) {
    ModelConfigList *list = (ModelConfigList *)calloc(1, sizeof(ModelConfigList));
    return list;
}

void model_config_list_destroy(ModelConfigList *list) {
    if (!list) return;
    for (int i = 0; i < list->count; i++) {
        model_config_destroy(list->configs[i]);
    }
    free(list->configs);
    free(list);
}

// ── JSON serialization ────────────────────────────────────────────────────────

static const char *arch_to_str(ModelArchitecture arch) {
    switch (arch) {
        case MODEL_ARCH_TRANSFORMER: return "transformer";
        case MODEL_ARCH_VIT: return "vit";
        case MODEL_ARCH_DIFFUSION: return "diffusion";
        case MODEL_ARCH_RNN: return "rnn";
        case MODEL_ARCH_GAN: return "gan";
        case MODEL_ARCH_GCN: return "gcn";
        case MODEL_ARCH_HSS: return "hss";
        case MODEL_ARCH_NPE: return "npe";
        case MODEL_ARCH_SER: return "ser";
        case MODEL_ARCH_ARC: return "arc";
        case MODEL_ARCH_CUSTOM: return "custom";
        default: return "transformer";
    }
}

static ModelArchitecture str_to_arch(const char *s) {
    if (!strcmp(s, "transformer")) return MODEL_ARCH_TRANSFORMER;
    if (!strcmp(s, "vit")) return MODEL_ARCH_VIT;
    if (!strcmp(s, "diffusion")) return MODEL_ARCH_DIFFUSION;
    if (!strcmp(s, "rnn")) return MODEL_ARCH_RNN;
    if (!strcmp(s, "gan")) return MODEL_ARCH_GAN;
    if (!strcmp(s, "gcn")) return MODEL_ARCH_GCN;
    if (!strcmp(s, "hss")) return MODEL_ARCH_HSS;
    if (!strcmp(s, "npe")) return MODEL_ARCH_NPE;
    if (!strcmp(s, "ser")) return MODEL_ARCH_SER;
    if (!strcmp(s, "arc")) return MODEL_ARCH_ARC;
    if (!strcmp(s, "custom")) return MODEL_ARCH_CUSTOM;
    return MODEL_ARCH_TRANSFORMER;
}

static const char *act_to_str(ActivationType act) {
    switch (act) {
        case ACTIVATION_GELU: return "gelu";
        case ACTIVATION_RELU: return "relu";
        case ACTIVATION_SILU: return "silu";
        case ACTIVATION_GEGLU: return "geglu";
        case ACTIVATION_SWIGLU: return "swiglu";
        default: return "silu";
    }
}

static ActivationType str_to_act(const char *s) {
    if (!strcmp(s, "gelu")) return ACTIVATION_GELU;
    if (!strcmp(s, "relu")) return ACTIVATION_RELU;
    if (!strcmp(s, "silu")) return ACTIVATION_SILU;
    if (!strcmp(s, "geglu")) return ACTIVATION_GEGLU;
    if (!strcmp(s, "swiglu")) return ACTIVATION_SWIGLU;
    return ACTIVATION_SILU;
}

static const char *pos_to_str(PositionEncodingType pos) {
    switch (pos) {
        case POS_ENCODING_LEARNED: return "learned";
        case POS_ENCODING_ROPE: return "rope";
        case POS_ENCODING_ALIBI: return "alibi";
        case POS_ENCODING_SINUSOIDAL: return "sinusoidal";
        default: return "rope";
    }
}

static PositionEncodingType str_to_pos(const char *s) {
    if (!strcmp(s, "learned")) return POS_ENCODING_LEARNED;
    if (!strcmp(s, "rope")) return POS_ENCODING_ROPE;
    if (!strcmp(s, "alibi")) return POS_ENCODING_ALIBI;
    if (!strcmp(s, "sinusoidal")) return POS_ENCODING_SINUSOIDAL;
    return POS_ENCODING_ROPE;
}

static const char *attn_to_str(AttentionType attn) {
    switch (attn) {
        case ATTENTION_CAUSAL: return "causal";
        case ATTENTION_FLASH: return "flash";
        case ATTENTION_SLIDING_WINDOW: return "sliding_window";
        case ATTENTION_BLOCK_SPARSE: return "block_sparse";
        default: return "causal";
    }
}

static AttentionType str_to_attn(const char *s) {
    if (!strcmp(s, "causal")) return ATTENTION_CAUSAL;
    if (!strcmp(s, "flash")) return ATTENTION_FLASH;
    if (!strcmp(s, "sliding_window")) return ATTENTION_SLIDING_WINDOW;
    if (!strcmp(s, "block_sparse")) return ATTENTION_BLOCK_SPARSE;
    return ATTENTION_CAUSAL;
}

// Simple JSON value extraction (no external dependency)
static char *json_extract_str(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(json, pattern);
    if (!pos) return NULL;
    pos += strlen(pattern);
    while (*pos && *pos != ':') pos++;
    if (*pos != ':') return NULL;
    pos++;
    while (*pos && *pos != '"') pos++;
    if (*pos != '"') return NULL;
    pos++;
    char *start = pos;
    while (*pos && *pos != '"') pos++;
    if (*pos != '"') return NULL;
    size_t len = pos - start;
    char *result = (char *)malloc(len + 1);
    if (result) {
        memcpy(result, start, len);
        result[len] = '\0';
    }
    return result;
}

static int json_extract_int(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(json, pattern);
    if (!pos) return default_val;
    pos += strlen(pattern);
    while (*pos && *pos != ':') pos++;
    if (*pos != ':') return default_val;
    pos++;
    while (*pos && isspace((unsigned char)*pos)) pos++;
    return atoi(pos);
}

static double json_extract_double(const char *json, const char *key, double default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(json, pattern);
    if (!pos) return default_val;
    pos += strlen(pattern);
    while (*pos && *pos != ':') pos++;
    if (*pos != ':') return default_val;
    pos++;
    while (*pos && isspace((unsigned char)*pos)) pos++;
    return strtod(pos, NULL);
}

static int json_extract_bool(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    char *pos = strstr(json, pattern);
    if (!pos) return default_val;
    pos += strlen(pattern);
    while (*pos && *pos != ':') pos++;
    if (*pos != ':') return default_val;
    pos++;
    while (*pos && isspace((unsigned char)*pos)) pos++;
    if (!strncmp(pos, "true", 4)) return 1;
    if (!strncmp(pos, "false", 5)) return 0;
    return default_val;
}

char *model_config_to_json(const ModelConfig *config, int pretty) {
    if (!config) return NULL;

    const char *indent = pretty ? "  " : "";
    const char *nl = pretty ? "\n" : "";
    const char *sep = pretty ? ",\n" : ",";

    // Estimate size (generous)
    size_t buf_size = 8192;
    char *buf = (char *)malloc(buf_size);
    if (!buf) return NULL;

    int offset = snprintf(buf, buf_size,
        "{%s"
        "%s\"name\": \"%s\",%s"
        "%s\"version\": \"%s\",%s"
        "%s\"description\": \"%s\",%s"
        "%s\"author\": \"%s\",%s"
        "%s\"license\": \"%s\",%s"
        "%s\"architecture\": \"%s\",%s"
        "%s\"vocab_size\": %d,%s"
        "%s\"hidden_size\": %d,%s"
        "%s\"num_layers\": %d,%s"
        "%s\"num_heads\": %d,%s"
        "%s\"num_kv_heads\": %d,%s"
        "%s\"intermediate_size\": %d,%s"
        "%s\"max_position_embeddings\": %d,%s"
        "%s\"max_seq_len\": %d,%s"
        "%s\"layer_norm_eps\": %g,%s"
        "%s\"rms_norm_eps\": %g,%s"
        "%s\"use_rms_norm\": %d,%s"
        "%s\"hidden_act\": \"%s\",%s"
        "%s\"ffn_act\": \"%s\",%s"
        "%s\"gated_ffn\": %d,%s"
        "%s\"attention_type\": \"%s\",%s"
        "%s\"attention_dropout\": %g,%s"
        "%s\"hidden_dropout\": %g,%s"
        "%s\"use_flash_attention\": %d,%s"
        "%s\"sliding_window\": %d,%s"
        "%s\"pos_encoding\": \"%s\",%s"
        "%s\"rope_theta\": %g,%s"
        "%s\"rope_scaling\": %d,%s"
        "%s\"initializer_range\": %g,%s"
        "%s\"tie_word_embeddings\": %d,%s"
        "%s\"quantize\": %d,%s"
        "%s\"quant_bits\": %d,%s"
        "%s\"quant_group_size\": %d,%s"
        "%s\"tensor_parallel_size\": %d,%s"
        "%s\"pipeline_parallel_size\": %d,%s"
        "%s\"sequence_parallel\": %d,%s"
        "%s\"learning_rate\": %g,%s"
        "%s\"weight_decay\": %g,%s"
        "%s\"max_grad_norm\": %g,%s"
        "%s\"warmup_steps\": %d,%s"
        "%s\"max_steps\": %d,%s"
        "%s\"gradient_accumulation_steps\": %d,%s"
        "%s\"mixed_precision\": %d,%s"
        "%s\"num_experts\": %d,%s"
        "%s\"num_experts_per_token\": %d,%s"
        "%s\"router_aux_loss_coef\": %g%s"
        "%s}%s",
        nl,
        indent, config->name ? config->name : "", nl,
        indent, config->version ? config->version : "0.1.0", nl,
        indent, config->description ? config->description : "", nl,
        indent, config->author ? config->author : "", nl,
        indent, config->license ? config->license : "", nl,
        indent, arch_to_str(config->architecture), nl,
        indent, config->vocab_size, nl,
        indent, config->hidden_size, nl,
        indent, config->num_layers, nl,
        indent, config->num_heads, nl,
        indent, config->num_kv_heads, nl,
        indent, config->intermediate_size, nl,
        indent, config->max_position_embeddings, nl,
        indent, config->max_seq_len, nl,
        indent, config->layer_norm_eps, nl,
        indent, config->rms_norm_eps, nl,
        indent, config->use_rms_norm, nl,
        indent, act_to_str(config->hidden_act), nl,
        indent, act_to_str(config->ffn_act), nl,
        indent, config->gated_ffn, nl,
        indent, attn_to_str(config->attention_type), nl,
        indent, config->attention_dropout, nl,
        indent, config->hidden_dropout, nl,
        indent, config->use_flash_attention, nl,
        indent, config->sliding_window, nl,
        indent, pos_to_str(config->pos_encoding), nl,
        indent, config->rope_theta, nl,
        indent, config->rope_scaling, nl,
        indent, config->initializer_range, nl,
        indent, config->tie_word_embeddings, nl,
        indent, config->quantize, nl,
        indent, config->quant_bits, nl,
        indent, config->quant_group_size, nl,
        indent, config->tensor_parallel_size, nl,
        indent, config->pipeline_parallel_size, nl,
        indent, config->sequence_parallel, nl,
        indent, config->learning_rate, nl,
        indent, config->weight_decay, nl,
        indent, config->max_grad_norm, nl,
        indent, config->warmup_steps, nl,
        indent, config->max_steps, nl,
        indent, config->gradient_accumulation_steps, nl,
        indent, config->mixed_precision, nl,
        indent, config->num_experts, nl,
        indent, config->num_experts_per_token, nl,
        indent, config->router_aux_loss_coef, nl,
        indent,
        nl);

    // Add custom fields
    if (config->custom_count > 0) {
        char *tmp = (char *)realloc(buf, buf_size + 4096);
        if (tmp) {
            buf = tmp;
            buf[offset - 2] = ',';  // Replace the } before final newline
            for (int i = 0; i < config->custom_count; i++) {
                offset += snprintf(buf + offset, buf_size - offset,
                    "%s\"%s\": \"%s\"%s",
                    indent, config->custom_keys[i], config->custom_values[i],
                    (i < config->custom_count - 1) ? sep : "");
            }
            offset += snprintf(buf + offset, buf_size - offset, "%s}%s", nl, "");
        }
    }

    return buf;
}

ModelConfig *model_config_from_json(const char *json) {
    if (!json) return NULL;

    ModelConfig *cfg = model_config_create();
    if (!cfg) return NULL;

    char *name = json_extract_str(json, "name");
    char *version = json_extract_str(json, "version");
    char *description = json_extract_str(json, "description");
    char *author = json_extract_str(json, "author");
    char *license = json_extract_str(json, "license");

    char *arch_str = json_extract_str(json, "architecture");
    char *act_str = json_extract_str(json, "hidden_act");
    char *ffn_act_str = json_extract_str(json, "ffn_act");
    char *pos_str = json_extract_str(json, "pos_encoding");
    char *attn_str = json_extract_str(json, "attention_type");

    cfg->name = name;
    cfg->version = version;
    cfg->description = description;
    cfg->author = author;
    cfg->license = license;

    cfg->architecture = arch_str ? str_to_arch(arch_str) : MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = json_extract_int(json, "vocab_size", 32000);
    cfg->hidden_size = json_extract_int(json, "hidden_size", 4096);
    cfg->num_layers = json_extract_int(json, "num_layers", 32);
    cfg->num_heads = json_extract_int(json, "num_heads", 32);
    cfg->num_kv_heads = json_extract_int(json, "num_kv_heads", cfg->num_heads);
    cfg->intermediate_size = json_extract_int(json, "intermediate_size", 11008);
    cfg->max_position_embeddings = json_extract_int(json, "max_position_embeddings", 2048);
    cfg->max_seq_len = json_extract_int(json, "max_seq_len", 2048);
    cfg->layer_norm_eps = (float)json_extract_double(json, "layer_norm_eps", 1e-5);
    cfg->rms_norm_eps = (float)json_extract_double(json, "rms_norm_eps", 1e-6);
    cfg->use_rms_norm = json_extract_bool(json, "use_rms_norm", 1);
    cfg->hidden_act = act_str ? str_to_act(act_str) : ACTIVATION_SILU;
    cfg->ffn_act = ffn_act_str ? str_to_act(ffn_act_str) : ACTIVATION_SILU;
    cfg->gated_ffn = json_extract_bool(json, "gated_ffn", 1);
    cfg->attention_type = attn_str ? str_to_attn(attn_str) : ATTENTION_CAUSAL;
    cfg->attention_dropout = (float)json_extract_double(json, "attention_dropout", 0.0);
    cfg->hidden_dropout = (float)json_extract_double(json, "hidden_dropout", 0.0);
    cfg->use_flash_attention = json_extract_bool(json, "use_flash_attention", 1);
    cfg->sliding_window = json_extract_int(json, "sliding_window", 0);
    cfg->pos_encoding = pos_str ? str_to_pos(pos_str) : POS_ENCODING_ROPE;
    cfg->rope_theta = (float)json_extract_double(json, "rope_theta", 10000.0);
    cfg->rope_scaling = json_extract_int(json, "rope_scaling", 0);
    cfg->initializer_range = (float)json_extract_double(json, "initializer_range", 0.02);
    cfg->tie_word_embeddings = json_extract_bool(json, "tie_word_embeddings", 0);
    cfg->quantize = json_extract_bool(json, "quantize", 0);
    cfg->quant_bits = json_extract_int(json, "quant_bits", 0);
    cfg->quant_group_size = json_extract_int(json, "quant_group_size", 128);
    cfg->tensor_parallel_size = json_extract_int(json, "tensor_parallel_size", 1);
    cfg->pipeline_parallel_size = json_extract_int(json, "pipeline_parallel_size", 1);
    cfg->sequence_parallel = json_extract_bool(json, "sequence_parallel", 0);
    cfg->learning_rate = (float)json_extract_double(json, "learning_rate", 2e-4);
    cfg->weight_decay = (float)json_extract_double(json, "weight_decay", 0.01);
    cfg->max_grad_norm = (float)json_extract_double(json, "max_grad_norm", 1.0);
    cfg->warmup_steps = json_extract_int(json, "warmup_steps", 0);
    cfg->max_steps = json_extract_int(json, "max_steps", 0);
    cfg->gradient_accumulation_steps = json_extract_int(json, "gradient_accumulation_steps", 1);
    cfg->mixed_precision = json_extract_bool(json, "mixed_precision", 1);
    cfg->num_experts = json_extract_int(json, "num_experts", 0);
    cfg->num_experts_per_token = json_extract_int(json, "num_experts_per_token", 0);
    cfg->router_aux_loss_coef = (float)json_extract_double(json, "router_aux_loss_coef", 0.0);

    free(arch_str);
    free(act_str);
    free(ffn_act_str);
    free(pos_str);
    free(attn_str);

    return cfg;
}

// ── File I/O ──────────────────────────────────────────────────────────────────

int model_config_save(const ModelConfig *config, const char *path) {
    if (!config || !path) return -1;

    char *json = model_config_to_json(config, 1);
    if (!json) return -1;

    FILE *f = fopen(path, "w");
    if (!f) {
        free(json);
        return -1;
    }

    fputs(json, f);
    fclose(f);
    free(json);
    return 0;
}

ModelConfig *model_config_load(const char *path) {
    if (!path) return NULL;

    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size < 0 || size > 1024 * 1024) {
        fclose(f);
        return NULL;
    }

    char *buf = (char *)malloc(size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(buf, 1, size, f);
    fclose(f);
    buf[read] = '\0';

    ModelConfig *cfg = model_config_from_json(buf);
    free(buf);
    return cfg;
}

// ── Validation ────────────────────────────────────────────────────────────────

int model_config_validate(const ModelConfig *config, char **error_out) {
    if (!config) {
        if (error_out) *error_out = dup_str("Config is NULL");
        return -1;
    }

    if (!config->name) {
        if (error_out) *error_out = dup_str("Config name is required");
        return -1;
    }

    if (config->vocab_size <= 0) {
        if (error_out) *error_out = dup_str("vocab_size must be positive");
        return -1;
    }

    if (config->hidden_size <= 0) {
        if (error_out) *error_out = dup_str("hidden_size must be positive");
        return -1;
    }

    if (config->num_layers <= 0) {
        if (error_out) *error_out = dup_str("num_layers must be positive");
        return -1;
    }

    if (config->num_heads <= 0) {
        if (error_out) *error_out = dup_str("num_heads must be positive");
        return -1;
    }

    if (config->hidden_size % config->num_heads != 0) {
        if (error_out) *error_out = dup_str("hidden_size must be divisible by num_heads");
        return -1;
    }

    if (config->num_kv_heads > config->num_heads) {
        if (error_out) *error_out = dup_str("num_kv_heads cannot exceed num_heads");
        return -1;
    }

    if (config->intermediate_size <= 0) {
        if (error_out) *error_out = dup_str("intermediate_size must be positive");
        return -1;
    }

    if (config->max_position_embeddings <= 0) {
        if (error_out) *error_out = dup_str("max_position_embeddings must be positive");
        return -1;
    }

    if (config->learning_rate <= 0) {
        if (error_out) *error_out = dup_str("learning_rate must be positive");
        return -1;
    }

    return 0;
}

// ── Copy ──────────────────────────────────────────────────────────────────────

ModelConfig *model_config_copy(const ModelConfig *config) {
    if (!config) return NULL;

    ModelConfig *copy = model_config_create();
    if (!copy) return NULL;

    copy->name = dup_str(config->name);
    copy->version = dup_str(config->version);
    copy->description = dup_str(config->description);
    copy->author = dup_str(config->author);
    copy->license = dup_str(config->license);

    copy->architecture = config->architecture;
    copy->vocab_size = config->vocab_size;
    copy->hidden_size = config->hidden_size;
    copy->num_layers = config->num_layers;
    copy->num_heads = config->num_heads;
    copy->num_kv_heads = config->num_kv_heads;
    copy->intermediate_size = config->intermediate_size;
    copy->max_position_embeddings = config->max_position_embeddings;
    copy->max_seq_len = config->max_seq_len;
    copy->layer_norm_eps = config->layer_norm_eps;
    copy->rms_norm_eps = config->rms_norm_eps;
    copy->use_rms_norm = config->use_rms_norm;
    copy->hidden_act = config->hidden_act;
    copy->ffn_act = config->ffn_act;
    copy->gated_ffn = config->gated_ffn;
    copy->attention_type = config->attention_type;
    copy->attention_dropout = config->attention_dropout;
    copy->hidden_dropout = config->hidden_dropout;
    copy->use_flash_attention = config->use_flash_attention;
    copy->sliding_window = config->sliding_window;
    copy->pos_encoding = config->pos_encoding;
    copy->rope_theta = config->rope_theta;
    copy->rope_scaling = config->rope_scaling;
    copy->initializer_range = config->initializer_range;
    copy->tie_word_embeddings = config->tie_word_embeddings;
    copy->quantize = config->quantize;
    copy->quant_bits = config->quant_bits;
    copy->quant_group_size = config->quant_group_size;
    copy->tensor_parallel_size = config->tensor_parallel_size;
    copy->pipeline_parallel_size = config->pipeline_parallel_size;
    copy->sequence_parallel = config->sequence_parallel;
    copy->learning_rate = config->learning_rate;
    copy->weight_decay = config->weight_decay;
    copy->max_grad_norm = config->max_grad_norm;
    copy->warmup_steps = config->warmup_steps;
    copy->max_steps = config->max_steps;
    copy->gradient_accumulation_steps = config->gradient_accumulation_steps;
    copy->mixed_precision = config->mixed_precision;
    copy->num_experts = config->num_experts;
    copy->num_experts_per_token = config->num_experts_per_token;
    copy->router_aux_loss_coef = config->router_aux_loss_coef;

    // Copy custom fields
    for (int i = 0; i < config->custom_count; i++) {
        copy->custom_keys = (char **)realloc(copy->custom_keys, (i + 1) * sizeof(char *));
        copy->custom_values = (char **)realloc(copy->custom_values, (i + 1) * sizeof(char *));
        copy->custom_keys[i] = dup_str(config->custom_keys[i]);
        copy->custom_values[i] = dup_str(config->custom_values[i]);
        copy->custom_count = i + 1;
    }

    return copy;
}

// ── Merge ─────────────────────────────────────────────────────────────────────

void model_config_merge(ModelConfig *base, const ModelConfig *override) {
    if (!base || !override) return;

    if (override->name) { free_str(base->name); base->name = dup_str(override->name); }
    if (override->version) { free_str(base->version); base->version = dup_str(override->version); }
    if (override->description) { free_str(base->description); base->description = dup_str(override->description); }
    if (override->author) { free_str(base->author); base->author = dup_str(override->author); }
    if (override->license) { free_str(base->license); base->license = dup_str(override->license); }

    base->architecture = override->architecture;
    base->vocab_size = override->vocab_size;
    base->hidden_size = override->hidden_size;
    base->num_layers = override->num_layers;
    base->num_heads = override->num_heads;
    base->num_kv_heads = override->num_kv_heads;
    base->intermediate_size = override->intermediate_size;
    base->max_position_embeddings = override->max_position_embeddings;
    base->max_seq_len = override->max_seq_len;
    base->layer_norm_eps = override->layer_norm_eps;
    base->rms_norm_eps = override->rms_norm_eps;
    base->use_rms_norm = override->use_rms_norm;
    base->hidden_act = override->hidden_act;
    base->ffn_act = override->ffn_act;
    base->gated_ffn = override->gated_ffn;
    base->attention_type = override->attention_type;
    base->attention_dropout = override->attention_dropout;
    base->hidden_dropout = override->hidden_dropout;
    base->use_flash_attention = override->use_flash_attention;
    base->sliding_window = override->sliding_window;
    base->pos_encoding = override->pos_encoding;
    base->rope_theta = override->rope_theta;
    base->rope_scaling = override->rope_scaling;
    base->initializer_range = override->initializer_range;
    base->tie_word_embeddings = override->tie_word_embeddings;
    base->quantize = override->quantize;
    base->quant_bits = override->quant_bits;
    base->quant_group_size = override->quant_group_size;
    base->tensor_parallel_size = override->tensor_parallel_size;
    base->pipeline_parallel_size = override->pipeline_parallel_size;
    base->sequence_parallel = override->sequence_parallel;
    base->learning_rate = override->learning_rate;
    base->weight_decay = override->weight_decay;
    base->max_grad_norm = override->max_grad_norm;
    base->warmup_steps = override->warmup_steps;
    base->max_steps = override->max_steps;
    base->gradient_accumulation_steps = override->gradient_accumulation_steps;
    base->mixed_precision = override->mixed_precision;
    base->num_experts = override->num_experts;
    base->num_experts_per_token = override->num_experts_per_token;
    base->router_aux_loss_coef = override->router_aux_loss_coef;
}

// ── Preset configs ────────────────────────────────────────────────────────────

ModelConfig *model_config_llama2_7b(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("llama2-7b");
    cfg->version = dup_str("2.0.0");
    cfg->description = dup_str("LLaMA 2 7B parameter model");
    cfg->author = dup_str("Meta AI");
    cfg->license = dup_str("llama2-community");
    cfg->architecture = MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = 32000;
    cfg->hidden_size = 4096;
    cfg->num_layers = 32;
    cfg->num_heads = 32;
    cfg->num_kv_heads = 32;
    cfg->intermediate_size = 11008;
    cfg->max_position_embeddings = 4096;
    cfg->max_seq_len = 4096;
    cfg->layer_norm_eps = 1e-5f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 1;
    cfg->hidden_act = ACTIVATION_SILU;
    cfg->ffn_act = ACTIVATION_SILU;
    cfg->gated_ffn = 1;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->pos_encoding = POS_ENCODING_ROPE;
    cfg->rope_theta = 10000.0f;
    cfg->rope_scaling = 2;           /* NTK-aware scaling for 128K */
    cfg->rope_scaling_factor = 32.0f; /* 128K / 4K = 32 */
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 2e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

ModelConfig *model_config_llama2_13b(void) {
    ModelConfig *cfg = model_config_llama2_7b();
    cfg->name = dup_str("llama2-13b");
    cfg->hidden_size = 5120;
    cfg->num_layers = 40;
    cfg->num_heads = 40;
    cfg->num_kv_heads = 40;
    cfg->intermediate_size = 13824;
    return cfg;
}

ModelConfig *model_config_llama3_8b(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("llama3-8b");
    cfg->version = dup_str("3.0.0");
    cfg->description = dup_str("LLaMA 3 8B parameter model");
    cfg->author = dup_str("Meta AI");
    cfg->license = dup_str("llama3-community");
    cfg->architecture = MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = 128256;
    cfg->hidden_size = 4096;
    cfg->num_layers = 32;
    cfg->num_heads = 32;
    cfg->num_kv_heads = 8;
    cfg->intermediate_size = 14336;
    cfg->max_position_embeddings = 8192;
    cfg->max_seq_len = 8192;
    cfg->layer_norm_eps = 1e-5f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 1;
    cfg->hidden_act = ACTIVATION_SILU;
    cfg->ffn_act = ACTIVATION_SILU;
    cfg->gated_ffn = 1;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->pos_encoding = POS_ENCODING_ROPE;
    cfg->rope_theta = 500000.0f;
    cfg->rope_scaling = 1;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 2e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

ModelConfig *model_config_mistral_7b(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("mistral-7b");
    cfg->version = dup_str("0.1.0");
    cfg->description = dup_str("Mistral 7B parameter model with sliding window attention");
    cfg->author = dup_str("Mistral AI");
    cfg->license = dup_str("apache-2.0");
    cfg->architecture = MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = 32000;
    cfg->hidden_size = 4096;
    cfg->num_layers = 32;
    cfg->num_heads = 32;
    cfg->num_kv_heads = 8;
    cfg->intermediate_size = 14336;
    cfg->max_position_embeddings = 8192;
    cfg->max_seq_len = 8192;
    cfg->layer_norm_eps = 1e-5f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 1;
    cfg->hidden_act = ACTIVATION_SILU;
    cfg->ffn_act = ACTIVATION_SILU;
    cfg->gated_ffn = 1;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->pos_encoding = POS_ENCODING_ROPE;
    cfg->rope_theta = 10000.0f;
    cfg->sliding_window = 4096;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 2e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

ModelConfig *model_config_qwen2_7b(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("qwen2-7b");
    cfg->version = dup_str("2.0.0");
    cfg->description = dup_str("Qwen 2 7B parameter model");
    cfg->author = dup_str("Alibaba");
    cfg->license = dup_str("apache-2.0");
    cfg->architecture = MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = 151936;
    cfg->hidden_size = 4096;
    cfg->num_layers = 32;
    cfg->num_heads = 32;
    cfg->num_kv_heads = 32;
    cfg->intermediate_size = 11008;
    cfg->max_position_embeddings = 4096;
    cfg->max_seq_len = 4096;
    cfg->layer_norm_eps = 1e-6f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 1;
    cfg->hidden_act = ACTIVATION_SILU;
    cfg->ffn_act = ACTIVATION_SILU;
    cfg->gated_ffn = 1;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->pos_encoding = POS_ENCODING_ROPE;
    cfg->rope_theta = 1000000.0f;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 2e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

ModelConfig *model_config_bert_base(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("bert-base");
    cfg->version = dup_str("1.0.0");
    cfg->description = dup_str("BERT Base model (12 layers, 768 hidden, 12 heads)");
    cfg->author = dup_str("Google");
    cfg->license = dup_str("apache-2.0");
    cfg->architecture = MODEL_ARCH_TRANSFORMER;
    cfg->vocab_size = 30522;
    cfg->hidden_size = 768;
    cfg->num_layers = 12;
    cfg->num_heads = 12;
    cfg->num_kv_heads = 12;
    cfg->intermediate_size = 3072;
    cfg->max_position_embeddings = 512;
    cfg->max_seq_len = 512;
    cfg->layer_norm_eps = 1e-12f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 0;
    cfg->hidden_act = ACTIVATION_GELU;
    cfg->ffn_act = ACTIVATION_GELU;
    cfg->gated_ffn = 0;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->pos_encoding = POS_ENCODING_LEARNED;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 1e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

ModelConfig *model_config_vit_base(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("vit-base");
    cfg->version = dup_str("1.0.0");
    cfg->description = dup_str("Vision Transformer Base (12 layers, 768 hidden, 12 heads)");
    cfg->author = dup_str("Google");
    cfg->license = dup_str("apache-2.0");
    cfg->architecture = MODEL_ARCH_VIT;
    cfg->vocab_size = 1000;
    cfg->hidden_size = 768;
    cfg->num_layers = 12;
    cfg->num_heads = 12;
    cfg->num_kv_heads = 12;
    cfg->intermediate_size = 3072;
    cfg->max_position_embeddings = 196;
    cfg->max_seq_len = 196;
    cfg->layer_norm_eps = 1e-6f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 0;
    cfg->hidden_act = ACTIVATION_GELU;
    cfg->ffn_act = ACTIVATION_GELU;
    cfg->gated_ffn = 0;
    cfg->attention_type = ATTENTION_FLASH;
    cfg->pos_encoding = POS_ENCODING_LEARNED;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 1e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

ModelConfig *model_config_sdxl(void) {
    ModelConfig *cfg = model_config_create();
    cfg->name = dup_str("sdxl");
    cfg->version = dup_str("1.0.0");
    cfg->description = dup_str("Stable Diffusion XL (UNet + VAE + Text Encoders)");
    cfg->author = dup_str("Stability AI");
    cfg->license = dup_str("creative-ml-openraill");
    cfg->architecture = MODEL_ARCH_DIFFUSION;
    cfg->vocab_size = 32000;
    cfg->hidden_size = 4096;
    cfg->num_layers = 4;
    cfg->num_heads = 16;
    cfg->num_kv_heads = 16;
    cfg->intermediate_size = 11008;
    cfg->max_position_embeddings = 1024;
    cfg->max_seq_len = 1024;
    cfg->layer_norm_eps = 1e-5f;
    cfg->rms_norm_eps = 1e-6f;
    cfg->use_rms_norm = 1;
    cfg->hidden_act = ACTIVATION_SILU;
    cfg->ffn_act = ACTIVATION_SILU;
    cfg->gated_ffn = 1;
    cfg->attention_type = ATTENTION_CAUSAL;
    cfg->pos_encoding = POS_ENCODING_LEARNED;
    cfg->initializer_range = 0.02f;
    cfg->tie_word_embeddings = 0;
    cfg->learning_rate = 1e-4f;
    cfg->weight_decay = 0.01f;
    cfg->max_grad_norm = 1.0f;
    return cfg;
}

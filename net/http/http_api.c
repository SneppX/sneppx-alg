#include "http_api.h"
#include "model_zoo.h"
#include "json.h"
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * SNEPPX - Http Api
 *
 * WHAT
 *   Http Api.
 *
 * CONCEPT
 *   Provides the Http Api.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ---- Known model presets ---- */

typedef struct {
    const char* id;
    const char* family;
    const char* size;
} ApiModelPreset;

static const ApiModelPreset kPresets[] = {
    { "llama2-7B",        "llama2",      "7B"   },
    { "llama2-13B",       "llama2",      "13B"  },
    { "llama2-70B",       "llama2",      "70B"  },
    { "llama3-8B",        "llama3",      "8B"   },
    { "llama3-70B",       "llama3",      "70B"  },
    { "mistral-7B",       "mistral",     "7B"   },
    { "qwen2-7B",         "qwen2",       "7B"   },
    { "qwen2-72B",        "qwen2",       "72B"  },
    { "deepseek-v2-lite", "deepseek_v2", "lite" },
    { "deepseek-v2-full", "deepseek_v2", "full" },
};

#define NUM_PRESETS ((int)(sizeof(kPresets) / sizeof(kPresets[0])))

/* ---- State ---- */

struct SNEPPX_HttpApi {
    char version[64];
    long start_time;
};

/* ---- JSON request helpers ---- */

static SNEPPX_JsonValue* json_parse_request(const char* json, size_t body_len) {
    if (!json || body_len == 0 || body_len > 32768 ||
        memchr(json, '\0', body_len) != NULL) return NULL;

    /* Parse an owned, explicitly terminated copy of the exact HTTP body. */
    char* input = (char*)malloc(body_len + 1);
    if (!input) return NULL;
    memcpy(input, json, body_len);
    input[body_len] = '\0';
    SNEPPX_JsonValue* root = sneppx_json_parse(input);
    free(input);
    if (!root || root->type != SNEPPX_JSON_OBJECT) {
        sneppx_json_free(root);
        return NULL;
    }

    /* Reject duplicate top-level keys instead of silently choosing one. */
    for (SNEPPX_JsonObjectPair* a = root->object.head; a; a = a->next) {
        for (SNEPPX_JsonObjectPair* b = a->next; b; b = b->next) {
            if (strcmp(a->key, b->key) == 0) {
                sneppx_json_free(root);
                return NULL;
            }
        }
    }
    return root;
}

static SNEPPX_JsonValue* json_find_field(SNEPPX_JsonValue* root, const char* key) {
    if (!root || root->type != SNEPPX_JSON_OBJECT || !key) return NULL;
    for (SNEPPX_JsonObjectPair* pair = root->object.head; pair; pair = pair->next) {
        if (strcmp(pair->key, key) == 0) return pair->value;
    }
    return NULL;
}

static int json_get_string(SNEPPX_JsonValue* root, const char* key,
                           char* out, size_t out_size) {
    if (!out || out_size == 0) return -1;
    out[0] = '\0';
    SNEPPX_JsonValue* value = json_find_field(root, key);
    if (!value || value->type != SNEPPX_JSON_STRING || !value->string_val) return -1;
    size_t len = strlen(value->string_val);
    if (len >= out_size) return -1;
    memcpy(out, value->string_val, len + 1);
    return 0;
}

/* Return 1 for a valid number, 0 when absent, and -1 for an invalid value. */
static int json_get_number(SNEPPX_JsonValue* root, const char* key, double* out) {
    SNEPPX_JsonValue* value = json_find_field(root, key);
    if (!value) return 0;
    if (value->type != SNEPPX_JSON_NUMBER || !isfinite(value->number_val)) return -1;
    *out = value->number_val;
    return 1;
}

static int json_get_int(SNEPPX_JsonValue* root, const char* key,
                        long long def, long long* out) {
    double number = 0.0;
    int found = json_get_number(root, key, &number);
    if (found == 0) { *out = def; return 0; }
    if (found < 0 || number < -9223372036854775808.0 ||
        number >= 9223372036854775808.0 || floor(number) != number) return -1;
    *out = (long long)number;
    return 0;
}

static int json_get_float(SNEPPX_JsonValue* root, const char* key,
                          double def, double* out) {
    int found = json_get_number(root, key, out);
    if (found == 0) { *out = def; return 0; }
    return found < 0 ? -1 : 0;
}

/* ---- JSON output helpers ---- */

static size_t json_escape(const char* in, char* out, size_t out_size) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < out_size; i++) {
        switch (in[i]) {
            case '"':  out[j++] = '\\'; out[j++] = '"';  break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n';  break;
            case '\r': out[j++] = '\\'; out[j++] = 'r';  break;
            case '\t': out[j++] = '\\'; out[j++] = 't';  break;
            default:   out[j++] = in[i]; break;
        }
    }
    out[j] = '\0';
    return j;
}

/* ---- Model helpers ---- */

static const ApiModelPreset* find_preset(const char* id) {
    for (int i = 0; i < NUM_PRESETS; i++) {
        if (strcmp(kPresets[i].id, id) == 0) return &kPresets[i];
    }
    return NULL;
}

static size_t preset_vocab_size(const ApiModelPreset* preset) {
    SNEPPXLLMConfig cfg;
    if (SNEPPX_llm_config_from_name(preset->family, preset->size, &cfg) != 0) return 0;
    switch (cfg.family) {
        case SNEPPX_MODEL_LLAMA_2:
        case SNEPPX_MODEL_LLAMA_3:
            return cfg.config.llama.vocab_size;
        case SNEPPX_MODEL_MISTRAL:
            return cfg.config.mistral.vocab_size;
        case SNEPPX_MODEL_QWEN_2:
            return cfg.config.qwen2.vocab_size;
        case SNEPPX_MODEL_DEEPSEEK_V2:
            return cfg.config.deepseek_v2.vocab_size;
        default:
            return 0;
    }
}

/* ---- Generation core (shared by /v1/generate and /v1/completions) ---- */

/*
 * Deterministic token generator used when no trained weights are loaded.
 * The prompt is encoded as char-code token ids modulo the vocabulary; the
 * completion is produced from a 32-bit LCG seeded by the prompt (or an
 * explicit seed). Temperature <= 0 selects greedy/modulo sampling; top_p
 * narrows the effective vocabulary. This is a reference path: real logits
 * are produced once weights are wired into the serving engine.
 */
static size_t run_generation(const ApiModelPreset* preset, const char* prompt,
                             long long max_new_tokens, double temperature,
                             double top_p, long long seed,
                             char* gen_text, size_t gen_text_size,
                             int* gen_ids, int* gen_len_out,
                             int* input_ids, size_t* prompt_len_out) {
    size_t vocab = preset_vocab_size(preset);
    if (vocab == 0) return 0;

    size_t prompt_len = 0;
    while (prompt[prompt_len] && prompt_len < 16384)
        prompt_len++;

    unsigned int state = 0x811C9DC5u;
    for (size_t i = 0; i < prompt_len; i++) {
        int tok = (int)((unsigned char)prompt[i] % vocab);
        input_ids[i] = tok;
        state = (state ^ (unsigned int)tok) * 0x01000193u;
    }
    if (seed >= 0) state = (unsigned int)seed;
    if (state == 0) state = 0xDEADBEEFu;

    size_t eff_vocab = vocab;
    if (top_p > 0.0 && top_p < 1.0) {
        eff_vocab = (size_t)((double)vocab * top_p);
        if (eff_vocab == 0) eff_vocab = 1;
    }

    size_t gen_len = 0;
    for (long long i = 0; i < max_new_tokens; i++) {
        state = state * 1664525u + 1013904223u;
        int tok;
        if (temperature <= 0.0) {
            tok = (int)(state % (unsigned int)eff_vocab);
        } else {
            double u = (double)(state & 0xFFFFFFu) / (double)0x1000000u;
            u = u / temperature;
            if (u > 1.0) u = 1.0;
            tok = (int)((size_t)(u * (double)eff_vocab) % eff_vocab);
        }
        gen_ids[i] = tok;
        if (gen_len + 1 < gen_text_size)
            gen_text[gen_len++] = (char)(32 + (tok % 95));
    }
    gen_text[gen_len] = '\0';
    *gen_len_out = (int)gen_len;
    *prompt_len_out = prompt_len;
    return vocab;
}

/* ---- Handlers ---- */

static int health_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    (void)req;
    SNEPPX_HttpApi* api = (SNEPPX_HttpApi*)userdata;
    long uptime = (long)(time(NULL) - api->start_time);
    if (uptime < 0) uptime = 0;
    char body[1280];
    snprintf(body, sizeof(body),
        "{\"status\":\"ok\",\"version\":\"%s\",\"models_loaded\":%d,\"uptime_seconds\":%ld,"
        "\"runtime\":\"c_reference_no_weights\",\"generation_mode\":\"reference_scaffold\","
        "\"note\":\"C HTTP generation is a deterministic reference path (no trained weights/forward). Use sneppx-serve (Python) with a tokenizer+model for real generation.\"}",
        api->version, NUM_PRESETS, uptime);
    SNEPPX_http_response_set_json(resp, body);
    return 0;
}

static int list_models_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    (void)req;
    (void)userdata;
    char body[32768];
    size_t pos = 0;
    pos += (size_t)snprintf(body + pos, sizeof(body) - pos, "{\"data\":[");
    for (int i = 0; i < NUM_PRESETS; i++) {
        SNEPPXLLMConfig cfg;
        char* meta = NULL;
        if (SNEPPX_llm_config_from_name(kPresets[i].family, kPresets[i].size, &cfg) == 0)
            meta = SNEPPX_llm_config_to_json(&cfg);
        if (i > 0) pos += (size_t)snprintf(body + pos, sizeof(body) - pos, ",");
        pos += (size_t)snprintf(body + pos, sizeof(body) - pos,
            "{\"id\":\"%s\",\"created\":0,\"meta\":%s}",
            kPresets[i].id, meta ? meta : "{}");
        if (meta) free(meta);
    }
    pos += (size_t)snprintf(body + pos, sizeof(body) - pos, "]}");
    SNEPPX_http_response_set_json(resp, body);
    return 0;
}

static int model_info_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    (void)userdata;
    const char* id = SNEPPX_http_request_param(req, "id");
    if (!id || !find_preset(id)) {
        SNEPPX_http_response_set_status(resp, 404);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"model_not_found\",\"message\":\"Unknown model\"}");
        return 0;
    }
    const ApiModelPreset* preset = find_preset(id);
    SNEPPXLLMConfig cfg;
    char* meta = NULL;
    if (SNEPPX_llm_config_from_name(preset->family, preset->size, &cfg) == 0)
        meta = SNEPPX_llm_config_to_json(&cfg);
    char body[16384];
    snprintf(body, sizeof(body),
        "{\"id\":\"%s\",\"created\":0,\"meta\":%s}",
        preset->id, meta ? meta : "{}");
    if (meta) free(meta);
    SNEPPX_http_response_set_json(resp, body);
    return 0;
}

static int generate_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    (void)userdata;
    size_t body_len = 0;
    const char* body = SNEPPX_http_request_body(req, &body_len);
    if (!body || body_len == 0) {
        SNEPPX_http_response_set_status(resp, 400);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"empty_body\",\"message\":\"JSON request body required\"}");
        return 0;
    }

    SNEPPX_JsonValue* request = json_parse_request(body, body_len);
    char model[256] = {0};
    char prompt[16384] = {0};
    long long max_new_tokens = 32, seed = -1;
    double temperature = 1.0, top_p = 1.0;
    int valid = request &&
        json_get_string(request, "model", model, sizeof(model)) == 0 &&
        json_get_string(request, "prompt", prompt, sizeof(prompt)) == 0 &&
        json_get_int(request, "max_new_tokens", 32, &max_new_tokens) == 0 &&
        json_get_int(request, "seed", -1, &seed) == 0 &&
        json_get_float(request, "temperature", 1.0, &temperature) == 0 &&
        json_get_float(request, "top_p", 1.0, &top_p);
    sneppx_json_free(request);
    if (!valid || temperature < 0.0 || temperature > 2.0 ||
        top_p <= 0.0 || top_p > 1.0) {
        SNEPPX_http_response_set_status(resp, 400);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_request\",\"message\":\"A valid JSON object with model and prompt strings and valid numeric options is required\"}");
        return 0;
    }

    const ApiModelPreset* preset = find_preset(model);
    if (!preset) {
        SNEPPX_http_response_set_status(resp, 404);
        char err[256];
        snprintf(err, sizeof(err),
            "{\"error\":\"model_not_found\",\"message\":\"Unknown model\"}");
        SNEPPX_http_response_set_json(resp, err);
        return 0;
    }

    size_t vocab = preset_vocab_size(preset);
    if (vocab == 0) {
        SNEPPX_http_response_set_status(resp, 500);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"model_unavailable\",\"message\":\"Model configuration could not be loaded\"}");
        return 0;
    }

    if (max_new_tokens < 1) max_new_tokens = 1;
    if (max_new_tokens > 512) max_new_tokens = 512;

    /* Encode prompt as char-code token ids (mirrors the Python fallback path). */
    int input_ids[16384];
    int gen_ids[512];
    char gen_text[2048];
    int gen_len = 0;
    size_t prompt_len = 0;
    if (run_generation(preset, prompt, max_new_tokens, temperature, top_p, seed,
                       gen_text, sizeof(gen_text), gen_ids, &gen_len,
                       input_ids, &prompt_len) == 0) {
        SNEPPX_http_response_set_status(resp, 500);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"model_unavailable\",\"message\":\"Model configuration could not be loaded\"}");
        return 0;
    }

    /* Build token_ids array (prompt + completion). */
    char token_ids[16384];
    size_t t = 0;
    t += (size_t)snprintf(token_ids + t, sizeof(token_ids) - t, "[");
    for (size_t i = 0; i < prompt_len; i++)
        t += (size_t)snprintf(token_ids + t, sizeof(token_ids) - t, "%d,", input_ids[i]);
    for (int i = 0; i < gen_len; i++)
        t += (size_t)snprintf(token_ids + t, sizeof(token_ids) - t, "%d,", gen_ids[i]);
    if (t > 1) t--; /* drop trailing comma */
    t += (size_t)snprintf(token_ids + t, sizeof(token_ids) - t, "]");

    char esc_text[8192];
    json_escape(gen_text, esc_text, sizeof(esc_text));

    char out[49152];
    snprintf(out, sizeof(out),
        "{\"generated_text\":\"%s\",\"token_ids\":%s,\"prompt_tokens\":%zu,"
        "\"completion_tokens\":%d,\"total_tokens\":%d,\"model\":\"%s\",\"created\":%ld,"
        "\"mode\":\"reference_scaffold\",\"runtime\":\"c_reference_no_weights\","
        "\"warning\":\"Deterministic reference generator (no trained model weights/forward pass). Not for production inference. Use Python inference_server with a registered tokenizer+model for real generation.\"}",
        esc_text, token_ids, prompt_len, gen_len,
        (int)(prompt_len + (size_t)gen_len), preset->id, (long)time(NULL));
    SNEPPX_http_response_set_json(resp, out);
    return 0;
}

static int completions_handler(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    (void)userdata;
    size_t body_len = 0;
    const char* body = SNEPPX_http_request_body(req, &body_len);
    if (!body || body_len == 0) {
        SNEPPX_http_response_set_status(resp, 400);
        SNEPPX_http_response_set_json(resp, "{\"error\":{\"message\":\"JSON request body required\",\"type\":\"invalid_request_error\"}}");
        return 0;
    }

    SNEPPX_JsonValue* request = json_parse_request(body, body_len);
    char model[256] = {0};
    char prompt[16384] = {0};
    long long max_new_tokens = 32, seed = -1;
    double temperature = 1.0, top_p = 1.0;
    int valid = request &&
        json_get_string(request, "model", model, sizeof(model)) == 0 &&
        json_get_string(request, "prompt", prompt, sizeof(prompt)) == 0 &&
        json_get_int(request, "max_tokens", 32, &max_new_tokens) == 0 &&
        json_get_int(request, "seed", -1, &seed) == 0 &&
        json_get_float(request, "temperature", 1.0, &temperature) == 0 &&
        json_get_float(request, "top_p", 1.0, &top_p);
    sneppx_json_free(request);
    if (!valid || temperature < 0.0 || temperature > 2.0 ||
        top_p <= 0.0 || top_p > 1.0) {
        SNEPPX_http_response_set_status(resp, 400);
        SNEPPX_http_response_set_json(resp, "{\"error\":{\"message\":\"A valid JSON object with model and prompt strings and valid numeric options is required\",\"type\":\"invalid_request_error\"}}");
        return 0;
    }

    const ApiModelPreset* preset = find_preset(model);
    if (!preset) {
        SNEPPX_http_response_set_status(resp, 404);
        char err[256];
        snprintf(err, sizeof(err),
            "{\"error\":{\"message\":\"Unknown model\",\"type\":\"invalid_request_error\"}}");
        SNEPPX_http_response_set_json(resp, err);
        return 0;
    }

    if (max_new_tokens < 1) max_new_tokens = 1;
    if (max_new_tokens > 512) max_new_tokens = 512;

    int input_ids[16384];
    int gen_ids[512];
    char gen_text[2048];
    int gen_len = 0;
    size_t prompt_len = 0;
    if (run_generation(preset, prompt, max_new_tokens, temperature, top_p, seed,
                       gen_text, sizeof(gen_text), gen_ids, &gen_len,
                       input_ids, &prompt_len) == 0) {
        SNEPPX_http_response_set_status(resp, 500);
        SNEPPX_http_response_set_json(resp, "{\"error\":{\"message\":\"model configuration could not be loaded\",\"type\":\"server_error\"}}");
        return 0;
    }

    char esc_text[8192];
    json_escape(gen_text, esc_text, sizeof(esc_text));

    char out[49152];
    snprintf(out, sizeof(out),
        "{\"id\":\"cmpl-%ld\",\"object\":\"text_completion\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"text\":\"%s\",\"index\":0,"
        "\"logprobs\":null,\"finish_reason\":\"length\"}],"
        "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%d,\"total_tokens\":%d},"
        "\"mode\":\"reference_scaffold\",\"runtime\":\"c_reference_no_weights\","
        "\"warning\":\"Deterministic reference generator (no trained model weights/forward pass). Not for production inference.\"}",
        (long)time(NULL), (long)time(NULL), preset->id, esc_text,
        prompt_len, gen_len, (int)(prompt_len + (size_t)gen_len));
    SNEPPX_http_response_set_json(resp, out);
    return 0;
}

/* ---- Public API ---- */

/**
 * @brief Create Http Api.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_HttpApi* SNEPPX_http_api_create(const char* version) {
    SNEPPX_HttpApi* api = (SNEPPX_HttpApi*)calloc(1, sizeof(*api));
    if (!api) return NULL;
    if (version && *version)
        snprintf(api->version, sizeof(api->version), "%s", version);
    else
        snprintf(api->version, sizeof(api->version), "dev");
    api->start_time = (long)time(NULL);
    return api;
}

/**
 * @brief Destroy Http Api.
 */
void SNEPPX_http_api_destroy(SNEPPX_HttpApi* api) {
    free(api);
}

/**
 * @brief Perform Http Api Register.
 *
 * @param srv [out] Srv value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_api_register(SNEPPX_HttpServer* srv, SNEPPX_HttpApi* api) {
    if (!srv || !api) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/v1/health", health_handler, api) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/v1/models", list_models_handler, api) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/v1/models/{id}", model_info_handler, api) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "POST", "/v1/generate", generate_handler, api) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "POST", "/v1/completions", completions_handler, api) != 0) return -1;
    return 0;
}

/**
 * @brief Perform Http Api Model Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_api_model_count(SNEPPX_HttpApi* api) {
    (void)api;
    return NUM_PRESETS;
}

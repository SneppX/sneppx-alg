#include <neural_core/model_zoo/model_card.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/*
 * SNEPPX - Model Card
 *
 * WHAT
 *   Model Card.
 *
 * CONCEPT
 *   Provides the Model Card.
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

// ── Create/destroy ──────────────────────────────────────────────────────────

ModelCard *model_card_create(void) {
    ModelCard *card = (ModelCard *)calloc(1, sizeof(ModelCard));
    return card;
}

void model_card_destroy(ModelCard *card) {
    if (!card) return;
    
    free_str(card->name);
    free_str(card->version);
    free_str(card->description);
    free_str(card->author);
    free_str(card->license);
    free_str(card->repository);
    free_str(card->homepage);
    free_str(card->architecture);
    free_str(card->model_type);
    free_str(card->framework);
    free_str(card->training_data);
    free_str(card->training_procedure);
    free_str(card->optimizer);
    free_str(card->precision);
    free_str(card->intended_use);
    free_str(card->limitations);
    free_str(card->ethical_considerations);
    free_str(card->citation);
    free_str(card->config_path);
    free_str(card->weights_path);
    free_str(card->tokenizer_path);
    
    for (int i = 0; i < card->num_tags; i++) {
        free_str(card->tags[i]);
    }
    free(card->tags);
    
    free(card);
}

// ── JSON serialization (simplified) ─────────────────────────────────────────

static void json_escape_append(const char *str, char **buf, size_t *buf_size, size_t *buf_len) {
    if (!str) {
        // append "null"
        if (*buf_len + 5 >= *buf_size) {
            *buf_size *= 2;
            *buf = (char *)realloc(*buf, *buf_size);
        }
        memcpy(*buf + *buf_len, "null", 4);
        *buf_len += 4;
        return;
    }
    
    // Ensure space
    size_t needed = strlen(str) * 2 + 3;
    while (*buf_len + needed >= *buf_size) {
        *buf_size *= 2;
        *buf = (char *)realloc(*buf, *buf_size);
    }
    
    (*buf)[(*buf_len)++] = '"';
    for (size_t i = 0; str[i]; i++) {
        char c = str[i];
        switch (c) {
            case '"': (*buf)[(*buf_len)++] = '\\'; (*buf)[(*buf_len)++] = '"'; break;
            case '\\': (*buf)[(*buf_len)++] = '\\'; (*buf)[(*buf_len)++] = '\\'; break;
            case '\n': (*buf)[(*buf_len)++] = '\\'; (*buf)[(*buf_len)++] = 'n'; break;
            case '\r': (*buf)[(*buf_len)++] = '\\'; (*buf)[(*buf_len)++] = 'r'; break;
            case '\t': (*buf)[(*buf_len)++] = '\\'; (*buf)[(*buf_len)++] = 't'; break;
            default: (*buf)[(*buf_len)++] = c; break;
        }
    }
    (*buf)[(*buf_len)++] = '"';
}

static void json_append_int64(char **buf, size_t *buf_size, size_t *buf_len, int64_t val) {
    char tmp[32];
    int len = snprintf(tmp, sizeof(tmp), "%lld", (long long)val);
    while (*buf_len + len + 1 >= *buf_size) {
        *buf_size *= 2;
        *buf = (char *)realloc(*buf, *buf_size);
    }
    memcpy(*buf + *buf_len, tmp, len);
    *buf_len += len;
}

static void json_append_double(char **buf, size_t *buf_size, size_t *buf_len, double val) {
    char tmp[64];
    int len = snprintf(tmp, sizeof(tmp), "%.6f", val);
    while (*buf_len + len + 1 >= *buf_size) {
        *buf_size *= 2;
        *buf = (char *)realloc(*buf, *buf_size);
    }
    memcpy(*buf + *buf_len, tmp, len);
    *buf_len += len;
}

char *model_card_to_json(const ModelCard *card, int pretty) {
    if (!card) return NULL;
    
    size_t buf_size = 4096;
    size_t buf_len = 0;
    char *buf = (char *)malloc(buf_size);
    if (!buf) return NULL;
    buf[0] = '\0';
    
    buf[buf_len++] = '{';
    if (pretty) buf[buf_len++] = '\n';
    
    // Helper macro for string fields
    #define ADD_STR(field) \
        do { \
            if (card->field) { \
                if (pretty) { buf[buf_len++] = ' '; buf[buf_len++] = ' '; } \
                size_t klen = strlen(#field); \
                while (buf_len + klen + 4 >= buf_size) { buf_size *= 2; buf = (char *)realloc(buf, buf_size); } \
                memcpy(buf + buf_len, "\"", 1); buf_len++; \
                memcpy(buf + buf_len, #field, klen); buf_len += klen; \
                memcpy(buf + buf_len, "\": ", 3); buf_len += 3; \
                json_escape_append(card->field, &buf, &buf_size, &buf_len); \
                buf[buf_len++] = ','; \
                if (pretty) buf[buf_len++] = '\n'; \
            } \
        } while (0)
    
    #define ADD_INT64(field) \
        do { \
            if (card->field != 0) { \
                if (pretty) { buf[buf_len++] = ' '; buf[buf_len++] = ' '; } \
                size_t klen = strlen(#field); \
                while (buf_len + klen + 32 >= buf_size) { buf_size *= 2; buf = (char *)realloc(buf, buf_size); } \
                memcpy(buf + buf_len, "\"", 1); buf_len++; \
                memcpy(buf + buf_len, #field, klen); buf_len += klen; \
                memcpy(buf + buf_len, "\": ", 3); buf_len += 3; \
                json_append_int64(&buf, &buf_size, &buf_len, card->field); \
                buf[buf_len++] = ','; \
                if (pretty) buf[buf_len++] = '\n'; \
            } \
        } while (0)
    
    #define ADD_DOUBLE(field) \
        do { \
            if (card->field != 0.0) { \
                if (pretty) { buf[buf_len++] = ' '; buf[buf_len++] = ' '; } \
                size_t klen = strlen(#field); \
                while (buf_len + klen + 32 >= buf_size) { buf_size *= 2; buf = (char *)realloc(buf, buf_size); } \
                memcpy(buf + buf_len, "\"", 1); buf_len++; \
                memcpy(buf + buf_len, #field, klen); buf_len += klen; \
                memcpy(buf + buf_len, "\": ", 3); buf_len += 3; \
                json_append_double(&buf, &buf_size, &buf_len, card->field); \
                buf[buf_len++] = ','; \
                if (pretty) buf[buf_len++] = '\n'; \
            } \
        } while (0)
    
    ADD_STR(name);
    ADD_STR(version);
    ADD_STR(description);
    ADD_STR(author);
    ADD_STR(license);
    ADD_STR(repository);
    ADD_STR(homepage);
    ADD_STR(architecture);
    ADD_STR(model_type);
    ADD_STR(framework);
    ADD_INT64(num_parameters);
    ADD_INT64(num_layers);
    ADD_INT64(hidden_size);
    ADD_INT64(vocab_size);
    ADD_INT64(max_seq_len);
    ADD_STR(training_data);
    ADD_STR(training_procedure);
    ADD_STR(optimizer);
    ADD_DOUBLE(learning_rate);
    ADD_INT64(batch_size);
    ADD_INT64(num_epochs);
    ADD_STR(precision);
    ADD_DOUBLE(perplexity);
    ADD_DOUBLE(accuracy);
    ADD_DOUBLE(f1_score);
    ADD_DOUBLE(latency_ms);
    ADD_DOUBLE(throughput);
    ADD_STR(intended_use);
    ADD_STR(limitations);
    ADD_STR(ethical_considerations);
    ADD_STR(citation);
    ADD_STR(config_path);
    ADD_STR(weights_path);
    ADD_STR(tokenizer_path);
    
    // Tags array
    if (card->num_tags > 0) {
        if (pretty) { buf[buf_len++] = ' '; buf[buf_len++] = ' '; }
        size_t klen = 5; // "tags"
        while (buf_len + 10 + card->num_tags * 32 >= buf_size) { buf_size *= 2; buf = (char *)realloc(buf, buf_size); }
        memcpy(buf + buf_len, "\"tags\": [", 9); buf_len += 9;
        for (int i = 0; i < card->num_tags; i++) {
            json_escape_append(card->tags[i], &buf, &buf_size, &buf_len);
            if (i < card->num_tags - 1) { buf[buf_len++] = ','; if (pretty) buf[buf_len++] = ' '; }
        }
        buf[buf_len++] = ']';
        buf[buf_len++] = ',';
        if (pretty) buf[buf_len++] = '\n';
    }
    
    // Remove trailing comma
    if (buf_len > 1 && buf[buf_len - 1] == ',') {
        buf_len--;
        if (pretty) buf[buf_len++] = '\n';
    }
    
    buf[buf_len++] = '}';
    if (pretty) buf[buf_len++] = '\n';
    buf[buf_len] = '\0';
    
    return buf;

    #undef ADD_STR
    #undef ADD_INT64
    #undef ADD_DOUBLE
}

// ── JSON parsing (simplified - stub) ────────────────────────────────────────

ModelCard *model_card_from_json(const char *json) {
    (void)json;
    return model_card_create(); // Stub
}

// ── File I/O ────────────────────────────────────────────────────────────────

int model_card_save(const ModelCard *card, const char *path) {
    if (!card || !path) return -1;
    
    char *json = model_card_to_json(card, 1);
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

ModelCard *model_card_load(const char *path) {
    if (!path) return NULL;
    
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size <= 0 || size > 1024 * 1024) {
        fclose(f);
        return NULL;
    }
    
    char *json = (char *)malloc(size + 1);
    if (!json) { fclose(f); return NULL; }
    
    size_t read = fread(json, 1, size, f);
    fclose(f);
    json[read] = '\0';
    
    ModelCard *card = model_card_from_json(json);
    free(json);
    return card;
}

// ── Validation ──────────────────────────────────────────────────────────────

int model_card_validate(const ModelCard *card, char **error_out) {
    if (!card) {
        if (error_out) *error_out = dup_str("Card is NULL");
        return -1;
    }
    
    if (!card->name || !*card->name) {
        if (error_out) *error_out = dup_str("Name is required");
        return -1;
    }
    
    if (!card->version || !*card->version) {
        if (error_out) *error_out = dup_str("Version is required");
        return -1;
    }
    
    if (!card->architecture || !*card->architecture) {
        if (error_out) *error_out = dup_str("Architecture is required");
        return -1;
    }
    
    if (card->num_parameters < 0) {
        if (error_out) *error_out = dup_str("num_parameters cannot be negative");
        return -1;
    }
    
    if (card->num_layers < 0) {
        if (error_out) *error_out = dup_str("num_layers cannot be negative");
        return -1;
    }
    
    if (card->learning_rate <= 0) {
        if (error_out) *error_out = dup_str("learning_rate must be positive");
        return -1;
    }
    
    return 0;
}

// ── Setters ────────────────────────────────────────────────────────────────

void model_card_set_name(ModelCard *card, const char *name) {
    if (!card) return;
    free_str(card->name);
    card->name = dup_str(name);
}

void model_card_set_version(ModelCard *card, const char *version) {
    if (!card) return;
    free_str(card->version);
    card->version = dup_str(version);
}

void model_card_set_description(ModelCard *card, const char *desc) {
    if (!card) return;
    free_str(card->description);
    card->description = dup_str(desc);
}

void model_card_set_author(ModelCard *card, const char *author) {
    if (!card) return;
    free_str(card->author);
    card->author = dup_str(author);
}

void model_card_set_license(ModelCard *card, const char *license) {
    if (!card) return;
    free_str(card->license);
    card->license = dup_str(license);
}

void model_card_set_repository(ModelCard *card, const char *repo) {
    if (!card) return;
    free_str(card->repository);
    card->repository = dup_str(repo);
}

void model_card_set_architecture(ModelCard *card, const char *arch) {
    if (!card) return;
    free_str(card->architecture);
    card->architecture = dup_str(arch);
}

void model_card_set_num_parameters(ModelCard *card, int64_t params) {
    if (!card) return;
    card->num_parameters = params;
}

void model_card_add_tag(ModelCard *card, const char *tag) {
    if (!card || !tag) return;
    
    card->tags = (char **)realloc(card->tags, (card->num_tags + 1) * sizeof(char *));
    card->tags[card->num_tags++] = dup_str(tag);
}

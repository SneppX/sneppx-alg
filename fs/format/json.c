/*
 * SNEPPX - JSON Format
 *
 * WHAT
 *   JSON parser â€” recursive descent parser with nested object/array support.
 *
 * CONCEPT
 *   Provides a structured SNEPPX_JsonValue AST with proper memory lifecycle
 *   (create + free), recursive descent parsing with escape sequence handling,
 *   and a find function for nested keys (e.g. "model.config.hidden_size").
 *
 * ROLE
 *   Replaces the fragile strstr-based JSON parsing scattered across the codebase.
 *   See docs/COMMENTING.md for the four-layer commenting standard.
 */

#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/* Forward declarations */
#define SNEPPX_JSON_MAX_DEPTH 128
static int json_hexval(char c);
static SNEPPX_JsonValue* parse_value(const char* json, size_t* pos, size_t len, unsigned depth);
static SNEPPX_JsonValue* parse_object(const char* json, size_t* pos, size_t len, unsigned depth);
static SNEPPX_JsonValue* parse_array(const char* json, size_t* pos, size_t len, unsigned depth);
static SNEPPX_JsonValue* parse_string(const char* json, size_t* pos, size_t len);
static SNEPPX_JsonValue* parse_number(const char* json, size_t* pos, size_t len);
static SNEPPX_JsonValue* parse_literal(const char* json, size_t* pos, size_t len);

static void skip_whitespace(const char* json, size_t* pos, size_t len) {
    while (*pos < len && isspace((unsigned char)json[*pos])) (*pos)++;
}

static char* unescape_string(const char* s, size_t len) {
    char* out = (char*)malloc(len + 1);
    if (!out) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '\\' && i + 1 < len) {
            i++;
            switch (s[i]) {
                case '"':  out[j++] = '"';  break;
                case '\\': out[j++] = '\\'; break;
                case '/':  out[j++] = '/';  break;
                case 'n':  out[j++] = '\n'; break;
                case 't':  out[j++] = '\t'; break;
                case 'r':  out[j++] = '\r'; break;
                case 'b':  out[j++] = '\b'; break;
                case 'f':  out[j++] = '\f'; break;
                case 'u': {
                    if (i + 4 >= len) { out[j] = '\0'; return out; }
                    unsigned int code = 0;
                    for (int k = 0; k < 4; k++) {
                        i++;
                        char c = (char)json_hexval(s[i]);
                        if (c < 0) { out[j] = '\0'; return out; }
                        code = (code << 4) | (unsigned int)c;
                    }
                    if (code < 0x80) {
                        out[j++] = (char)code;
                    } else if (code < 0x800) {
                        out[j++] = (char)(0xC0 | (code >> 6));
                        out[j++] = (char)(0x80 | (code & 0x3F));
                    } else {
                        out[j++] = (char)(0xE0 | (code >> 12));
                        out[j++] = (char)(0x80 | ((code >> 6) & 0x3F));
                        out[j++] = (char)(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default:
                    out[j++] = s[i];
                    break;
            }
        } else {
            out[j++] = s[i];
        }
    }
    out[j] = '\0';
    return out;
}

static int json_hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static SNEPPX_JsonValue* parse_value(const char* json, size_t* pos, size_t len, unsigned depth) {
    if (depth > SNEPPX_JSON_MAX_DEPTH) return NULL;
    skip_whitespace(json, pos, len);
    if (*pos >= len) return NULL;
    char c = json[*pos];
    if (c == '{') return parse_object(json, pos, len, depth);
    if (c == '[') return parse_array(json, pos, len, depth);
    if (c == '"') return parse_string(json, pos, len);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(json, pos, len);
    if (c == 't' || c == 'f' || c == 'n') return parse_literal(json, pos, len);
    return NULL;
}

static SNEPPX_JsonValue* new_value(SNEPPX_JsonType type) {
    SNEPPX_JsonValue* v = (SNEPPX_JsonValue*)calloc(1, sizeof(SNEPPX_JsonValue));
    if (!v) return NULL;
    v->type = type;
    return v;
}

static SNEPPX_JsonValue* parse_object(const char* json, size_t* pos, size_t len, unsigned depth) {
    SNEPPX_JsonValue* obj = new_value(SNEPPX_JSON_OBJECT);
    if (!obj) return NULL;
    (*pos)++; /* skip '{' */
    skip_whitespace(json, pos, len);
    if (*pos >= len || json[*pos] == '}') {
        (*pos)++;
        return obj;
    }
    while (1) {
        skip_whitespace(json, pos, len);
        if (*pos >= len || json[*pos] != '"') {
            sneppx_json_free(obj);
            return NULL;
        }
        SNEPPX_JsonValue* key = parse_string(json, pos, len);
        if (!key) { sneppx_json_free(obj); return NULL; }
        skip_whitespace(json, pos, len);
        if (*pos >= len || json[*pos] != ':') {
            sneppx_json_free(key);
            sneppx_json_free(obj);
            return NULL;
        }
        (*pos)++;
        SNEPPX_JsonValue* val = parse_value(json, pos, len, depth + 1);
        if (!val) { sneppx_json_free(key); sneppx_json_free(obj); return NULL; }
        /* Append key-value pair */
        SNEPPX_JsonObjectPair* pair = (SNEPPX_JsonObjectPair*)malloc(sizeof(SNEPPX_JsonObjectPair));
        if (!pair) { sneppx_json_free(key); sneppx_json_free(val); sneppx_json_free(obj); return NULL; }
        pair->key = key->string_val;
        pair->value = val;
        pair->next = NULL;
        key->string_val = NULL; /* ownership transferred */
        sneppx_json_free(key);
        if (!obj->object.head) {
            obj->object.head = pair;
        } else {
            SNEPPX_JsonObjectPair* tail = obj->object.head;
            while (tail->next) tail = tail->next;
            tail->next = pair;
        }
        skip_whitespace(json, pos, len);
        if (*pos >= len) { sneppx_json_free(obj); return NULL; }
        if (json[*pos] == '}') { (*pos)++; break; }
        if (json[*pos] != ',') { sneppx_json_free(obj); return NULL; }
        (*pos)++;
    }
    return obj;
}

static SNEPPX_JsonValue* parse_array(const char* json, size_t* pos, size_t len, unsigned depth) {
    SNEPPX_JsonValue* arr = new_value(SNEPPX_JSON_ARRAY);
    if (!arr) return NULL;
    (*pos)++; /* skip '[' */
    skip_whitespace(json, pos, len);
    if (*pos >= len || json[*pos] == ']') {
        (*pos)++;
        return arr;
    }
    size_t capacity = 8;
    arr->array.size = 0;
    arr->array.capacity = capacity;
    arr->array.items = (SNEPPX_JsonValue**)calloc(capacity, sizeof(SNEPPX_JsonValue*));
    if (!arr->array.items) { sneppx_json_free(arr); return NULL; }
    while (1) {
        SNEPPX_JsonValue* val = parse_value(json, pos, len, depth + 1);
        if (!val) { sneppx_json_free(arr); return NULL; }
        if (arr->array.size >= arr->array.capacity) {
            size_t new_cap = arr->array.capacity * 2;
            SNEPPX_JsonValue** new_items = (SNEPPX_JsonValue**)realloc(arr->array.items, new_cap * sizeof(SNEPPX_JsonValue*));
            if (!new_items) { sneppx_json_free(val); sneppx_json_free(arr); return NULL; }
            arr->array.items = new_items;
            arr->array.capacity = new_cap;
        }
        arr->array.items[arr->array.size++] = val;
        skip_whitespace(json, pos, len);
        if (*pos >= len) { sneppx_json_free(arr); return NULL; }
        if (json[*pos] == ']') { (*pos)++; break; }
        if (json[*pos] != ',') { sneppx_json_free(arr); return NULL; }
        (*pos)++;
    }
    return arr;
}

static SNEPPX_JsonValue* parse_string(const char* json, size_t* pos, size_t len) {
    (*pos)++; /* skip opening '"' */
    size_t start = *pos;
    size_t j = start;
    while (j < len && json[j] != '"') {
        unsigned char ch = (unsigned char)json[j];
        if (ch < 0x20) return NULL; /* JSON strings cannot contain raw controls. */
        if (json[j] == '\\') {
            if (j + 1 >= len) return NULL;
            char escape = json[j + 1];
            if (escape == 'u') {
                if (j + 5 >= len) return NULL;
                for (size_t k = j + 2; k <= j + 5; k++)
                    if (json_hexval(json[k]) < 0) return NULL;
                j += 6;
                continue;
            }
            if (escape != '"' && escape != '\\' && escape != '/' &&
                escape != 'b' && escape != 'f' && escape != 'n' &&
                escape != 'r' && escape != 't') return NULL;
            j += 2;
        } else {
            j++;
        }
    }
    if (j >= len) return NULL; /* unterminated string */
    SNEPPX_JsonValue* v = new_value(SNEPPX_JSON_STRING);
    if (!v) return NULL;
    v->string_val = unescape_string(json + start, j - start);
    if (!v->string_val) { free(v); return NULL; }
    *pos = j + 1; /* skip closing '"' */
    return v;
}

static SNEPPX_JsonValue* parse_number(const char* json, size_t* pos, size_t len) {
    size_t start = *pos;
    if (*pos < len && json[*pos] == '-') (*pos)++;
    if (*pos >= len) return NULL;

    if (json[*pos] == '0') {
        (*pos)++;
        if (*pos < len && isdigit((unsigned char)json[*pos])) return NULL;
    } else if (json[*pos] >= '1' && json[*pos] <= '9') {
        while (*pos < len && isdigit((unsigned char)json[*pos])) (*pos)++;
    } else {
        return NULL;
    }

    if (*pos < len && json[*pos] == '.') {
        (*pos)++;
        size_t fraction_start = *pos;
        while (*pos < len && isdigit((unsigned char)json[*pos])) (*pos)++;
        if (*pos == fraction_start) return NULL;
    }
    if (*pos < len && (json[*pos] == 'e' || json[*pos] == 'E')) {
        (*pos)++;
        if (*pos < len && (json[*pos] == '+' || json[*pos] == '-')) (*pos)++;
        size_t exponent_start = *pos;
        while (*pos < len && isdigit((unsigned char)json[*pos])) (*pos)++;
        if (*pos == exponent_start) return NULL;
    }

    char* end = NULL;
    double val = strtod(json + start, &end);
    if ((size_t)(end - json) != *pos || !isfinite(val)) return NULL;
    SNEPPX_JsonValue* v = new_value(SNEPPX_JSON_NUMBER);
    if (!v) return NULL;
    v->number_val = val;
    v->is_integer = 1;
    for (size_t i = start; i < *pos; i++) {
        if (json[i] == '.' || json[i] == 'e' || json[i] == 'E') { v->is_integer = 0; break; }
    }
    return v;
}

static SNEPPX_JsonValue* parse_literal(const char* json, size_t* pos, size_t len) {
    if (*pos + 4 <= len && strncmp(json + *pos, "true", 4) == 0) {
        *pos += 4;
        SNEPPX_JsonValue* v = new_value(SNEPPX_JSON_BOOL);
        if (!v) return NULL;
        v->bool_val = 1;
        return v;
    }
    if (*pos + 5 <= len && strncmp(json + *pos, "false", 5) == 0) {
        *pos += 5;
        SNEPPX_JsonValue* v = new_value(SNEPPX_JSON_BOOL);
        if (!v) return NULL;
        v->bool_val = 0;
        return v;
    }
    if (*pos + 4 <= len && strncmp(json + *pos, "null", 4) == 0) {
        *pos += 4;
        SNEPPX_JsonValue* v = new_value(SNEPPX_JSON_NULL);
        if (!v) return NULL;
        return v;
    }
    return NULL;
}

SNEPPX_JsonValue* sneppx_json_parse(const char* json) {
    if (!json) return NULL;
    size_t pos = 0;
    size_t len = strlen(json);
    SNEPPX_JsonValue* v = parse_value(json, &pos, len, 0);
    if (!v) return NULL;
    skip_whitespace(json, &pos, len);
    if (pos < len) { sneppx_json_free(v); return NULL; } /* trailing garbage */
    return v;
}

SNEPPX_JsonValue* sneppx_json_find(SNEPPX_JsonValue* root, const char* path) {
    if (!root || !path) return NULL;
    SNEPPX_JsonValue* cur = root;
    char buf[256];
    strncpy(buf, path, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char* token = strtok(buf, ".");
    while (token && cur) {
        if (cur->type == SNEPPX_JSON_OBJECT) {
            SNEPPX_JsonObjectPair* pair = cur->object.head;
            cur = NULL;
            while (pair) {
                if (strcmp(pair->key, token) == 0) {
                    cur = pair->value;
                    break;
                }
                pair = pair->next;
            }
        } else if (cur->type == SNEPPX_JSON_ARRAY) {
            int idx = atoi(token);
            if (idx < 0 || (size_t)idx >= cur->array.size) return NULL;
            cur = cur->array.items[idx];
        } else {
            return NULL;
        }
        token = strtok(NULL, ".");
    }
    return cur;
}

void sneppx_json_free(SNEPPX_JsonValue* v) {
    if (!v) return;
    switch (v->type) {
        case SNEPPX_JSON_STRING:
            free(v->string_val);
            break;
        case SNEPPX_JSON_ARRAY:
            if (v->array.items) {
                for (size_t i = 0; i < v->array.size; i++) {
                    sneppx_json_free(v->array.items[i]);
                }
                free(v->array.items);
            }
            break;
        case SNEPPX_JSON_OBJECT: {
            SNEPPX_JsonObjectPair* pair = v->object.head;
            while (pair) {
                SNEPPX_JsonObjectPair* next = pair->next;
                free(pair->key);
                sneppx_json_free(pair->value);
                free(pair);
                pair = next;
            }
            break;
        }
        default:
            break;
    }
    free(v);
}

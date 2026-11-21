/*
 * SNEPPX - JSON Format
 *
 * WHAT
 *   JSON parser — recursive descent with nested object/array support.
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

#ifndef SNEPPX_JSON_FORMAT_H
#define SNEPPX_JSON_FORMAT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
    SNEPPX_JSON_NULL = 0,
    SNEPPX_JSON_BOOL,
    SNEPPX_JSON_NUMBER,
    SNEPPX_JSON_STRING,
    SNEPPX_JSON_ARRAY,
    SNEPPX_JSON_OBJECT
} SNEPPX_JsonType;

typedef struct SNEPPX_JsonValue SNEPPX_JsonValue;

typedef struct {
    size_t size;
    size_t capacity;
    SNEPPX_JsonValue** items;
} SNEPPX_JsonArray;

typedef struct SNEPPX_JsonObjectPair {
    char* key;
    SNEPPX_JsonValue* value;
    struct SNEPPX_JsonObjectPair* next;
} SNEPPX_JsonObjectPair;

typedef struct {
    SNEPPX_JsonObjectPair* head;
} SNEPPX_JsonObject;

struct SNEPPX_JsonValue {
    SNEPPX_JsonType type;
    union {
        int bool_val;
        double number_val;
        long long int_val;
        char* string_val;
        SNEPPX_JsonArray array;
        SNEPPX_JsonObject object;
    };
    int is_integer;
};

/**
 * @brief Parse a JSON string into a structured AST.
 *
 * @param json [in] Null-terminated JSON string.
 * @return Root JsonValue on success, NULL on parse error. Caller must free via sneppx_json_free().
 */
SNEPPX_JsonValue* sneppx_json_parse(const char* json);

/**
 * @brief Find a value in a JSON AST by dotted path (e.g. "model.config.hidden_size").
 *
 * @param root [in] Root JsonValue.
 * @param path [in] Dotted key path (supports array indices).
 * @return Matched value or NULL if not found.
 */
SNEPPX_JsonValue* sneppx_json_find(SNEPPX_JsonValue* root, const char* path);

/**
 * @brief Free a JSON AST tree. Safe to call with NULL.
 * @param v Value to free.
 */
void sneppx_json_free(SNEPPX_JsonValue* v);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_JSON_FORMAT_H */

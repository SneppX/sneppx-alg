#include "json.h"
#include "test_gtest.h"
#include <string.h>

/*
 * SNEPPX - JSON Parser Regression Tests
 *
 * Covers strict container termination, Unicode escape decoding, and
 * re-entrant dotted-path lookup.
 */

static void test_json_rejects_truncated_containers(void)
{
    const char* invalid[] = {"{",
                             "[",
                             "{\"key\":",
                             "[1,",
                             "{\"key\":{",
                             "{\"key\":[",
                             "{\"key\":\"\\uD800\"}",
                             "{\"key\":\"\\uDC00\"}",
                             "{\"key\":\"\\uD800\\u0041\"}",
                             "{\"key\":\"\\u0000tail\"}",
                             "\v{}"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
    {
        SX_ASSERT_NULL(sneppx_json_parse(invalid[i]), "invalid JSON must be rejected");
    }
}

static void test_json_accepts_empty_containers(void)
{
    SNEPPX_JsonValue* root = sneppx_json_parse("{\"object\":{},\"array\":[]}");
    SX_ASSERT_NOT_NULL(root, "valid empty containers parse");
    SNEPPX_JsonValue* object = sneppx_json_find(root, "object");
    SNEPPX_JsonValue* array = sneppx_json_find(root, "array");
    SX_ASSERT_NOT_NULL(object, "empty object exists");
    SX_ASSERT_NOT_NULL(array, "empty array exists");
    SX_ASSERT_EQ(object->type, SNEPPX_JSON_OBJECT, "empty object type");
    SX_ASSERT_EQ(array->type, SNEPPX_JSON_ARRAY, "empty array type");
    SX_ASSERT_EQ(array->array.size, 0, "empty array length");
    sneppx_json_free(root);
}

static void test_json_decodes_unicode_surrogate_pairs(void)
{
    SNEPPX_JsonValue* root = sneppx_json_parse("{\"emoji\":\"\\uD83D\\uDE00\",\"accent\":\"\\u00E9\"}");
    SX_ASSERT_NOT_NULL(root, "valid Unicode escapes parse");
    SNEPPX_JsonValue* emoji = sneppx_json_find(root, "emoji");
    SNEPPX_JsonValue* accent = sneppx_json_find(root, "accent");
    SX_ASSERT_NOT_NULL(emoji, "emoji field exists");
    SX_ASSERT_NOT_NULL(accent, "accent field exists");
    SX_ASSERT_EQ(strcmp(emoji->string_val, "\xF0\x9F\x98\x80"), 0, "surrogate pair UTF-8");
    SX_ASSERT_EQ(strcmp(accent->string_val, "\xC3\xA9"), 0, "BMP UTF-8");
    sneppx_json_free(root);
}

static void test_json_find_array_paths_are_reentrant_and_strict(void)
{
    SNEPPX_JsonValue* root = sneppx_json_parse("{\"items\":[{\"name\":\"first\"},{\"name\":\"second\"}]}");
    SX_ASSERT_NOT_NULL(root, "path lookup input parses");
    SNEPPX_JsonValue* second = sneppx_json_find(root, "items.1.name");
    SX_ASSERT_NOT_NULL(second, "second array item path exists");
    SX_ASSERT_EQ(strcmp(second->string_val, "second"), 0, "second array item value");
    SX_ASSERT_NULL(sneppx_json_find(root, "items.99.name"), "out of range index rejected");
    SX_ASSERT_NULL(sneppx_json_find(root, "items.x.name"), "non numeric index rejected");
    SX_ASSERT_NULL(sneppx_json_find(root, "items..1"), "empty path component rejected");
    SX_ASSERT_NULL(sneppx_json_find(root, "items.1."), "trailing path separator rejected");
    /* Repeated independent lookups must not share strtok's process-global state. */
    SNEPPX_JsonValue* first = sneppx_json_find(root, "items.0.name");
    SX_ASSERT_NOT_NULL(first, "first array item path still works");
    SX_ASSERT_EQ(strcmp(first->string_val, "first"), 0, "first array item value");
    sneppx_json_free(root);
}

TEST(test_json_parser, reject_truncated_containers)
{
    test_json_rejects_truncated_containers();
}
TEST(test_json_parser, accept_empty_containers)
{
    test_json_accepts_empty_containers();
}
TEST(test_json_parser, decode_unicode_surrogate_pairs)
{
    test_json_decodes_unicode_surrogate_pairs();
}
TEST(test_json_parser, reentrant_array_path_lookup)
{
    test_json_find_array_paths_are_reentrant_and_strict();
}

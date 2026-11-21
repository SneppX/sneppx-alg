/**
 * SNEPPX Model Hub — C Client Implementation
 *
 * Implements the hub_client.h API using platform-agnostic HTTP via
 * WinHTTP (Windows) or libcurl (cross-platform fallback).
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "neural_core/hub_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Platform HTTP selection ---- */
#if defined(_WIN32)
    #define SNEPPX_HUB_USE_WINHTTP 1
    #include <windows.h>
    #include <winhttp.h>
    #pragma comment(lib, "winhttp.lib")
#else
    #define SNEPPX_HUB_USE_CURL 1
#endif

/* ---- Internal context ---- */
struct SNEPPXHubContext {
    char base_url[512];
    char api_key[256];
    char last_error[1024];

    /* Custom headers (simple array) */
    char custom_headers[16][512];
    int num_custom_headers;

#ifdef SNEPPX_HUB_USE_WINHTTP
    HINTERNET hSession;
#endif
};

/* ---- String helpers ---- */
static void set_error(SNEPPXHubContext *ctx, const char *msg) {
    if (ctx && msg) {
        strncpy(ctx->last_error, msg, sizeof(ctx->last_error) - 1);
        ctx->last_error[sizeof(ctx->last_error) - 1] = '\0';
    }
}

static char *str_dup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *copy = (char *)malloc(len);
    if (copy) memcpy(copy, s, len);
    return copy;
}

#ifdef SNEPPX_HUB_USE_WINHTTP
static int custom_header_contains_credentials(const SNEPPXHubContext *ctx) {
    static const char *sensitive_names[] = {
        "Authorization", "Proxy-Authorization", "Cookie", "X-API-Key",
        "API-Key", "X-Auth-Token", "X-Access-Token"
    };
    if (!ctx) return 0;
    for (int i = 0; i < ctx->num_custom_headers; i++) {
        const char *header = ctx->custom_headers[i];
        const char *colon = strchr(header, ':');
        if (!colon) continue;
        const char *name = header;
        while (name < colon && (*name == ' ' || *name == '\t')) name++;
        size_t name_len = (size_t)(colon - name);
        while (name_len && (name[name_len - 1] == ' ' || name[name_len - 1] == '\t')) name_len--;
        for (size_t j = 0; j < sizeof(sensitive_names) / sizeof(sensitive_names[0]); j++) {
            size_t expected_len = strlen(sensitive_names[j]);
            if (name_len == expected_len && _strnicmp(name, sensitive_names[j], name_len) == 0)
                return 1;
        }
    }
    return 0;
}

static void utf8_to_wide(WCHAR *dst, size_t dst_chars, const char *src) {
    if (!dst || dst_chars == 0) return;
    if (!src) {
        dst[0] = L'\0';
        return;
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, src, -1, dst, (int)dst_chars);
    if (n <= 0) {
        dst[0] = L'\0';
    } else if ((size_t)n >= dst_chars) {
        dst[dst_chars - 1] = L'\0';
    }
}
#endif

/* ---- JSON parsing (minimal, no external deps) ---- */

typedef struct {
    const char *json;
    size_t len;
    size_t pos;
} json_parser_t;

static void json_skip_ws(json_parser_t *p) {
    while (p->pos < p->len && (p->json[p->pos] == ' ' || p->json[p->pos] == '\t' ||
                                p->json[p->pos] == '\n' || p->json[p->pos] == '\r'))
        p->pos++;
}

static int json_peek(json_parser_t *p, char *out, size_t max) {
    json_skip_ws(p);
    if (p->pos >= p->len) return -1;
    if (p->json[p->pos] == '"') {
        p->pos++; /* skip opening quote */
        size_t i = 0;
        while (p->pos < p->len && p->json[p->pos] != '"' && i < max - 1) {
            out[i++] = p->json[p->pos++];
        }
        out[i] = '\0';
        if (p->pos < p->len) p->pos++; /* skip closing quote */
        return 0;
    }
    return -1;
}

static const char *json_find_str(const char *json, const char *key) {
    /* Very simple JSON string finder. Returns pointer to the value position or NULL. */
    size_t klen = strlen(key);
    const char *p = json;
    while (*p) {
        const char *found = strstr(p, key);
        if (!found) return NULL;
        /* Check it's a key (preceded by " and followed by ":) */
        if (found > json && found[-1] == '"') {
            const char *after = found + klen;
            while (*after && (*after == ' ' || *after == '\t' || *after == '\n')) after++;
            if (*after == '"') {
                after++;
                while (*after && *after != '"') {
                    if (*after == '\\') {
                        after++;
                        if (*after) after++;
                        continue;
                    }
                    /* Check if this is followed by : */
                    if (after[0] == '"' && after[1] == ':') {
                        return after + 1;
                    }
                    after++;
                }
            }
        }
        p = found + 1;
    }
    return NULL;
}

/* ---- Simple JSON value extraction ---- */
static int json_extract_str(const char *json, const char *key, char *out, size_t max) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *found = strstr(json, search);
    if (!found) return -1;
    found += strlen(search);
    while (*found && (*found == ' ' || *found == ':')) found++;
    if (*found == '"') {
        found++;
        size_t i = 0;
        while (*found && *found != '"' && i < max - 1) {
            out[i++] = *found++;
        }
        out[i] = '\0';
        return 0;
    }
    return -1;
}

static int json_extract_num(const char *json, const char *key, double *out) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *found = strstr(json, search);
    if (!found) return -1;
    found += strlen(search);
    while (*found && (*found == ' ' || *found == ':')) found++;
    *out = strtod(found, NULL);
    return 0;
}

static int json_extract_int(const char *json, const char *key, int *out) {
    double d;
    if (json_extract_num(json, key, &d) == 0) {
        *out = (int)d;
        return 0;
    }
    return -1;
}

/* ---- HTTP abstraction ---- */

typedef struct {
    int status_code;
    char *body;
    size_t body_len;
} http_response_t;

#ifdef SNEPPX_HUB_USE_WINHTTP

static int winhttp_request(
    SNEPPXHubContext *ctx,
    const char *method,
    const char *path,
    const char *body,
    http_response_t *resp
) {
    HINTERNET hConnect = NULL, hRequest = NULL;
    URL_COMPONENTS uc;
    WCHAR wide_base[512], wide_method[16], wide_path[2048], wide_auth[1024];
    char host[256], url_path[1024];

    memset(&uc, 0, sizeof(uc));
    uc.dwStructSize = sizeof(uc);
    uc.dwSchemeLength = (DWORD)-1;
    uc.dwHostNameLength = (DWORD)-1;
    uc.dwUrlPathLength = (DWORD)-1;
    uc.dwExtraInfoLength = (DWORD)-1;

    utf8_to_wide(wide_base, sizeof(wide_base) / sizeof(WCHAR), ctx->base_url);
    if (!WinHttpCrackUrl(wide_base, 0, 0, &uc)) {
        set_error(ctx, "Failed to parse base URL");
        return -1;
    }

    /* Convert wide chars to UTF-8 and validate the parsed URL scheme. */
    host[0] = '\\0';
    url_path[0] = '\\0';
    if (!WideCharToMultiByte(CP_UTF8, 0, uc.lpszHostName, -1, host, sizeof(host), NULL, NULL)) {
        set_error(ctx, "Failed to decode URL host");
        return -1;
    }
    if (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS) {
        set_error(ctx, "Only HTTP and HTTPS Hub URLs are supported");
        return -1;
    }
    int use_tls = (uc.nScheme == INTERNET_SCHEME_HTTPS);
    int is_loopback = (_stricmp(host, "localhost") == 0 ||
                       strcmp(host, "127.0.0.1") == 0 ||
                       strcmp(host, "::1") == 0 ||
                       strcmp(host, "[::1]") == 0);
    if ((ctx->api_key[0] || custom_header_contains_credentials(ctx)) &&
        !use_tls && !is_loopback) {
        set_error(ctx, "Refusing to send Hub credentials over non-local HTTP; use HTTPS");
        return -1;
    }

    hConnect = WinHttpConnect(ctx->hSession, uc.lpszHostName, uc.nPort, 0);
    if (!hConnect) {
        set_error(ctx, "Failed to connect to server");
        return -1;
    }

    utf8_to_wide(wide_method, sizeof(wide_method) / sizeof(WCHAR), method);
    utf8_to_wide(wide_path, sizeof(wide_path) / sizeof(WCHAR), path);
    hRequest = WinHttpOpenRequest(hConnect, wide_method, wide_path, NULL, NULL, NULL,
                                  use_tls ? WINHTTP_FLAG_SECURE : 0);
    if (!hRequest) {
        set_error(ctx, "Failed to create HTTP request");
        WinHttpCloseHandle(hConnect);
        return -1;
    }

    /* Add auth header if API key set */
    if (ctx->api_key[0]) {
        char auth[512];
        snprintf(auth, sizeof(auth), "Authorization: Bearer %s", ctx->api_key);
        utf8_to_wide(wide_auth, sizeof(wide_auth) / sizeof(WCHAR), auth);
        WinHttpAddRequestHeaders(hRequest, wide_auth, (DWORD)-1, WINHTTP_ADDREQ_FLAG_REPLACE | WINHTTP_ADDREQ_FLAG_ADD);
    }

    /* Add custom headers */
    for (int i = 0; i < ctx->num_custom_headers; i++) {
        utf8_to_wide(wide_auth, sizeof(wide_auth) / sizeof(WCHAR), ctx->custom_headers[i]);
        WinHttpAddRequestHeaders(hRequest, wide_auth, (DWORD)-1,
                                 WINHTTP_ADDREQ_FLAG_REPLACE | WINHTTP_ADDREQ_FLAG_ADD);
    }

    if (!WinHttpSendRequest(hRequest, NULL, 0, body ? (LPVOID)body : NULL,
                            body ? (DWORD)strlen(body) : 0,
                            body ? (DWORD)strlen(body) : 0, 0)) {
        set_error(ctx, "Failed to send HTTP request");
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        return -1;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        set_error(ctx, "Failed to receive response");
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        return -1;
    }

    /* Get status code */
    DWORD statusCode = 0;
    DWORD statusSize = sizeof(statusCode);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE,
                        NULL, &statusCode, &statusSize, NULL);

    /* Read body */
    char *buffer = NULL;
    size_t total = 0;
    char chunk[8192];
    DWORD read = 0;

    while (WinHttpReadData(hRequest, chunk, sizeof(chunk), &read) && read > 0) {
        buffer = (char *)realloc(buffer, total + read + 1);
        memcpy(buffer + total, chunk, read);
        total += read;
    }
    buffer[total] = '\0';

    resp->status_code = statusCode;
    resp->body = buffer;
    resp->body_len = total;

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    return 0;
}

#else
/* ---- libcurl fallback (not implemented here) ---- */
/* For a production build, link against libcurl or use a simple blocking socket. */
#endif

/* ---- Public API ---- */

SNEPPX_HUB_API SNEPPXHubContext *sneppx_hub_init(const char *base_url, const char *api_key) {
    if (!base_url) {
        return NULL;
    }

    SNEPPXHubContext *ctx = (SNEPPXHubContext *)calloc(1, sizeof(SNEPPXHubContext));
    if (!ctx) {
        return NULL;
    }

    strncpy(ctx->base_url, base_url, sizeof(ctx->base_url) - 1);
    if (api_key) {
        strncpy(ctx->api_key, api_key, sizeof(ctx->api_key) - 1);
    }

#ifdef SNEPPX_HUB_USE_WINHTTP
    ctx->hSession = WinHttpOpen(L"SNEPPX-Hub/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!ctx->hSession) {
        set_error(ctx, "Failed to initialize WinHTTP");
        free(ctx);
        return NULL;
    }
#endif

    return ctx;
}

SNEPPX_HUB_API void sneppx_hub_destroy(SNEPPXHubContext *ctx) {
    if (!ctx) return;
#ifdef SNEPPX_HUB_USE_WINHTTP
    if (ctx->hSession) {
        WinHttpCloseHandle(ctx->hSession);
    }
#endif
    free(ctx);
}

SNEPPX_HUB_API int sneppx_hub_is_reachable(SNEPPXHubContext *ctx) {
    if (!ctx) return 0;
    http_response_t resp = {0, NULL, 0};
#ifdef SNEPPX_HUB_USE_WINHTTP
    int r = winhttp_request(ctx, "GET", "/api/v1/stats", NULL, &resp);
    if (r == 0) {
        if (resp.body) free(resp.body);
        return resp.status_code == 200;
    }
#else
    /* curl or socket-based check */
#endif
    set_error(ctx, "HTTP implementation not available");
    return 0;
}

SNEPPX_HUB_API const char *sneppx_hub_strerror(SNEPPXHubContext *ctx) {
    if (!ctx) return "NULL context";
    return ctx->last_error[0] ? ctx->last_error : "no error";
}

SNEPPX_HUB_API void sneppx_hub_set_header(SNEPPXHubContext *ctx, const char *key, const char *value) {
    if (!ctx || !key || !value) return;
    if (ctx->num_custom_headers < 16) {
        snprintf(ctx->custom_headers[ctx->num_custom_headers], sizeof(ctx->custom_headers[0]),
                 "%s: %s", key, value);
        ctx->num_custom_headers++;
    }
}

SNEPPX_HUB_API int sneppx_hub_list_models(
    SNEPPXHubContext *ctx,
    sneppx_model_entry_t **out_models,
    int *out_count,
    const sneppx_hub_search_params_t *params
) {
    if (!ctx || !out_models || !out_count) {
        if (ctx) set_error(ctx, "NULL parameter");
        return -1;
    }

    *out_models = NULL;
    *out_count = 0;

    /* Build query string */
    char path[1024];
    snprintf(path, sizeof(path), "/api/v1/models?page=%d&page_size=%d",
             params ? params->page : 1, params ? (params->page_size > 0 ? params->page_size : 20) : 20);

    if (params) {
        if (params->query) {
            char buf[512];
            snprintf(buf, sizeof(buf), "&q=%s", params->query);
            strncat(path, buf, sizeof(path) - strlen(path) - 1);
        }
        if (params->task) {
            char buf[256];
            snprintf(buf, sizeof(buf), "&task=%s", params->task);
            strncat(path, buf, sizeof(path) - strlen(path) - 1);
        }
    }

    http_response_t resp = {0, NULL, 0};
#ifdef SNEPPX_HUB_USE_WINHTTP
    if (winhttp_request(ctx, "GET", path, NULL, &resp) != 0) {
        return -1;
    }
#else
    set_error(ctx, "HTTP implementation not available on this platform");
    return -1;
#endif

    if (resp.status_code != 200) {
        set_error(ctx, "HTTP request failed");
        if (resp.body) free(resp.body);
        return -1;
    }

    /* Parse JSON: find "items" array and count objects */
    const char *json = resp.body;
    const char *items = strstr(json, "\"items\"");
    if (!items) items = json; /* fallback to whole body */

    /* Count objects in the response (simple heuristic: count { not nested) */
    int count = 0;
    const char *p = items;
    while ((p = strstr(p, "\"name\""))) {
        count++;
        p += 6;
    }

    if (count == 0) {
        if (resp.body) free(resp.body);
        return 0;
    }

    sneppx_model_entry_t *models = (sneppx_model_entry_t *)calloc(count, sizeof(sneppx_model_entry_t));
    if (!models) {
        if (resp.body) free(resp.body);
        return -1;
    }

    /* Parse individual model entries */
    /* This is a simplified parser — a real implementation would use a proper JSON library */
    p = json;
    int idx = 0;
    while (p && idx < count) {
        const char *next = strstr(p, "\"name\"");
        if (!next) break;
        p = next;

        json_extract_str(p, "name", models[idx].name, sizeof(models[idx].name));
        json_extract_str(p, "version", models[idx].version, sizeof(models[idx].version));
        json_extract_str(p, "description", models[idx].description, sizeof(models[idx].description));
        json_extract_str(p, "architecture", models[idx].architecture, sizeof(models[idx].architecture));
        json_extract_str(p, "task", models[idx].task, sizeof(models[idx].task));
        json_extract_str(p, "format", models[idx].format, sizeof(models[idx].format));
        json_extract_str(p, "license", models[idx].license, sizeof(models[idx].license));
        json_extract_str(p, "visibility", models[idx].visibility, sizeof(models[idx].visibility));
        json_extract_str(p, "tags", models[idx].tags, sizeof(models[idx].tags));
        json_extract_str(p, "created_at", models[idx].created_at, sizeof(models[idx].created_at));
        json_extract_str(p, "updated_at", models[idx].updated_at, sizeof(models[idx].updated_at));
        double total_size = 0;
        json_extract_num(p, "total_size", &total_size);
        models[idx].total_size = (size_t)total_size;
        json_extract_int(p, "is_lfs", &models[idx].is_lfs);

        idx++;
        /* Move past this object */
        p = strchr(p, '{');
        if (p) p++;
    }

    *out_models = models;
    *out_count = count;
    if (resp.body) free(resp.body);
    return 0;
}

SNEPPX_HUB_API int sneppx_hub_get_model_info(
    SNEPPXHubContext *ctx,
    const char *model_name,
    const char *version,
    sneppx_model_card_t *out_card
) {
    if (!ctx || !model_name || !out_card) {
        if (ctx) set_error(ctx, "NULL parameter");
        return -1;
    }

    memset(out_card, 0, sizeof(sneppx_model_card_t));

    char path[1024];
    const char *ver = version ? version : "latest";
    snprintf(path, sizeof(path), "/api/v1/models/%s/%s", model_name, ver);

    http_response_t resp = {0, NULL, 0};
#ifdef SNEPPX_HUB_USE_WINHTTP
    if (winhttp_request(ctx, "GET", path, NULL, &resp) != 0) {
        return -1;
    }
#else
    set_error(ctx, "HTTP implementation not available on this platform");
    return -1;
#endif

    if (resp.status_code != 200) {
        set_error(ctx, "Model not found or inaccessible");
        if (resp.body) free(resp.body);
        return -1;
    }

    const char *json = resp.body;
    json_extract_str(json, "name", out_card->name, sizeof(out_card->name));
    json_extract_str(json, "organization", out_card->organization, sizeof(out_card->organization));
    json_extract_str(json, "description", out_card->description, sizeof(out_card->description));
    json_extract_str(json, "version", out_card->version, sizeof(out_card->version));
    json_extract_str(json, "architecture", out_card->architecture, sizeof(out_card->architecture));
    json_extract_str(json, "format", out_card->format, sizeof(out_card->format));
    json_extract_str(json, "task", out_card->task, sizeof(out_card->task));
    json_extract_str(json, "license", out_card->license, sizeof(out_card->license));
    json_extract_str(json, "visibility", out_card->visibility, sizeof(out_card->visibility));
    json_extract_str(json, "language", out_card->language, sizeof(out_card->language));
    json_extract_str(json, "author", out_card->author, sizeof(out_card->author));
    json_extract_str(json, "created_at", out_card->created_at, sizeof(out_card->created_at));
    json_extract_str(json, "updated_at", out_card->updated_at, sizeof(out_card->updated_at));
    json_extract_num(json, "total_size", (double *)&out_card->total_size);

    if (resp.body) free(resp.body);
    return 0;
}

SNEPPX_HUB_API void sneppx_hub_free_model_card(sneppx_model_card_t *card) {
    if (card && card->files) {
        free(card->files);
        card->files = NULL;
    }
}

SNEPPX_HUB_API int sneppx_hub_get_leaderboard(
    SNEPPXHubContext *ctx,
    sneppx_leaderboard_entry_t **out_entries,
    int *out_count,
    const char *task,
    const char *metric
) {
    if (!ctx || !out_entries || !out_count) {
        if (ctx) set_error(ctx, "NULL parameter");
        return -1;
    }

    *out_entries = NULL;
    *out_count = 0;

    char path[1024];
    snprintf(path, sizeof(path), "/api/v1/leaderboard");
    if (task) snprintf(path + strlen(path), sizeof(path) - strlen(path), "?task=%s", task);
    if (metric) snprintf(path + strlen(path), sizeof(path) - strlen(path), "%s&metric=%s",
                         strchr(path, '?') ? "" : "?", metric);

    http_response_t resp = {0, NULL, 0};
#ifdef SNEPPX_HUB_USE_WINHTTP
    if (winhttp_request(ctx, "GET", path, NULL, &resp) != 0) {
        return -1;
    }
#else
    set_error(ctx, "HTTP implementation not available on this platform");
    return -1;
#endif

    if (resp.status_code != 200) {
        if (resp.body) free(resp.body);
        return -1;
    }

    /* Count entries */
    int count = 0;
    const char *p = resp.body;
    while ((p = strstr(p, "\"model_name\""))) {
        count++;
        p += 13;
    }

    if (count == 0) {
        if (resp.body) free(resp.body);
        return 0;
    }

    sneppx_leaderboard_entry_t *entries = (sneppx_leaderboard_entry_t *)calloc(count, sizeof(sneppx_leaderboard_entry_t));
    if (!entries) {
        if (resp.body) free(resp.body);
        return -1;
    }

    p = resp.body;
    int idx = 0;
    while (p && idx < count) {
        const char *next = strstr(p, "\"model_name\"");
        if (!next) break;
        p = next;

        json_extract_str(p, "model_name", entries[idx].model_name, sizeof(entries[idx].model_name));
        json_extract_str(p, "version", entries[idx].version, sizeof(entries[idx].version));
        json_extract_str(p, "task", entries[idx].task, sizeof(entries[idx].task));
        json_extract_str(p, "metric", entries[idx].metric, sizeof(entries[idx].metric));
        json_extract_num(p, "score", &entries[idx].score);
        json_extract_int(p, "rank", &entries[idx].rank);
        json_extract_str(p, "dataset", entries[idx].dataset, sizeof(entries[idx].dataset));
        json_extract_str(p, "evaluated_at", entries[idx].evaluated_at, sizeof(entries[idx].evaluated_at));

        idx++;
        p = next + 13;
    }

    *out_entries = entries;
    *out_count = count;
    if (resp.body) free(resp.body);
    return 0;
}

SNEPPX_HUB_API int sneppx_hub_submit_benchmark(
    SNEPPXHubContext *ctx,
    const sneppx_benchmark_result_t *result
) {
    if (!ctx || !result) {
        if (ctx) set_error(ctx, "NULL parameter");
        return -1;
    }

    /* Build JSON body */
    char body[2048];
    snprintf(body, sizeof(body),
        "{\"model_name\":\"%s\",\"version\":\"%s\",\"task\":\"%s\",\"metric\":\"%s\","
        "\"score\":%.6f,\"higher_is_better\":%d,\"dataset\":\"%s\"}",
        result->model_name, result->version, result->task, result->metric,
        result->score, result->higher_is_better, result->dataset);

    http_response_t resp = {0, NULL, 0};
#ifdef SNEPPX_HUB_USE_WINHTTP
    if (winhttp_request(ctx, "POST", "/api/v1/leaderboard/submit", body, &resp) != 0) {
        return -1;
    }
#else
    set_error(ctx, "HTTP implementation not available on this platform");
    return -1;
#endif

    int rc = (resp.status_code == 200) ? 0 : -1;
    if (rc != 0) set_error(ctx, "Benchmark submission failed");
    if (resp.body) free(resp.body);
    return rc;
}

SNEPPX_HUB_API int sneppx_hub_search(
    SNEPPXHubContext *ctx,
    const char *query,
    sneppx_model_entry_t **out_models,
    int *out_count,
    const sneppx_hub_search_params_t *params
) {
    if (!ctx || !query || !out_models || !out_count) {
        if (ctx) set_error(ctx, "NULL parameter");
        return -1;
    }

    sneppx_hub_search_params_t default_params;
    if (!params) {
        memset(&default_params, 0, sizeof(default_params));
        default_params.page = 1;
        default_params.page_size = 20;
        params = &default_params;
    }

    /* Copy params and set query */
    sneppx_hub_search_params_t local_params = *params;
    local_params.query = query;

    return sneppx_hub_list_models(ctx, out_models, out_count, &local_params);
}

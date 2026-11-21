/**
 * SNEPPX Model Hub — C Client Library for Inference
 *
 * Provides a lightweight C API for:
 *   - Discovering models on the hub
 *   - Downloading model weights
 *   - Loading models for inference
 *
 * Usage:
 *   #include "neural_core/hub_client.h"
 *   sneppx_hub_ctx_t *ctx = sneppx_hub_init("http://localhost:8000", "api-key");
 *   sneppx_model_entry_t *models = NULL; int n = 0;
 *   sneppx_hub_list_models(ctx, &models, &n);
 *   sneppx_hub_download_model(ctx, "sneppx/llama-7b", "v1.0.0", "/tmp/models");
 *   sneppx_hub_destroy(ctx);
 */
#ifndef SNEPPX_HUB_CLIENT_H
#define SNEPPX_HUB_CLIENT_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
    #ifdef SNEPPX_HUB_EXPORTS
        #define SNEPPX_HUB_API __declspec(dllexport)
    #else
        #define SNEPPX_HUB_API __declspec(dllimport)
    #endif
#else
    #define SNEPPX_HUB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque types */
typedef struct SNEPPXHubContext SNEPPXHubContext;

/* HTTP method enum */
typedef enum {
    SNEPPX_HUB_GET,
    SNEPPX_HUB_POST,
    SNEPPX_HUB_PUT,
    SNEPPX_HUB_DELETE,
} sneppx_hub_http_method_t;

/* Model entry (lightweight, for listings) */
typedef struct {
    char name[256];
    char version[32];
    char description[1024];
    char architecture[128];
    char task[64];
    char format[32];
    char license[32];
    char visibility[16];
    char tags[256];
    char created_at[32];
    char updated_at[32];
    char download_url[512];
    size_t total_size;
    int is_lfs;
} sneppx_model_entry_t;

/* Model file descriptor */
typedef struct {
    char filename[256];
    char sha256[65];
    size_t size;
    int is_lfs;
    char url[512];
} sneppx_model_file_t;

/* Full model card */
typedef struct {
    char name[256];
    char organization[128];
    char description[1024];
    char version[32];
    char architecture[128];
    char format[32];
    char task[64];
    char license[32];
    char visibility[16];
    char language[32];
    char author[128];
    char created_at[32];
    char updated_at[32];
    size_t total_size;
    int num_files;
    sneppx_model_file_t *files;
} sneppx_model_card_t;

/* Leaderboard entry */
typedef struct {
    char model_name[256];
    char version[32];
    char task[64];
    char metric[64];
    double score;
    int rank;
    char dataset[128];
    char evaluated_at[32];
} sneppx_leaderboard_entry_t;

/* Benchmark result for submission */
typedef struct {
    char model_name[256];
    char version[32];
    char task[64];
    char metric[64];
    double score;
    int higher_is_better;
    char dataset[128];
} sneppx_benchmark_result_t;

/* Search/filter parameters */
typedef struct {
    const char *query;
    const char *task;
    const char *tag;
    const char *organization;
    int page;
    int page_size;
} sneppx_hub_search_params_t;

/**
 * Initialize a hub client context.
 *
 * @param base_url Base URL of the hub API (e.g. "http://localhost:8000").
 * @param api_key API key for authentication (or NULL for anonymous access).
 * @return New context, or NULL on error.
 */
SNEPPX_HUB_API SNEPPXHubContext *sneppx_hub_init(
    const char *base_url,
    const char *api_key
);

/**
 * Clean up a hub client context.
 */
SNEPPX_HUB_API void sneppx_hub_destroy(SNEPPXHubContext *ctx);

/**
 * List available models on the hub.
 *
 * @param ctx Hub context.
 * @param out_models Pointer to receive array of model entries.
 * @param out_count Pointer to receive number of entries.
 * @param params Optional search/filter params (or NULL for defaults).
 * @return 0 on success, -1 on error.
 */
SNEPPX_HUB_API int sneppx_hub_list_models(
    SNEPPXHubContext *ctx,
    sneppx_model_entry_t **out_models,
    int *out_count,
    const sneppx_hub_search_params_t *params
);

/**
 * Get detailed info about a specific model.
 *
 * @param ctx Hub context.
 * @param model_name Model name (e.g. "sneppx/llama-7b").
 * @param version Version string (e.g. "v1.0.0") or NULL for latest.
 * @param out_card Pointer to receive the model card. Must be freed with
 *                 sneppx_hub_free_model_card.
 * @return 0 on success, -1 on error.
 */
SNEPPX_HUB_API int sneppx_hub_get_model_info(
    SNEPPXHubContext *ctx,
    const char *model_name,
    const char *version,
    sneppx_model_card_t *out_card
);

/**
 * Free resources associated with a model card.
 */
SNEPPX_HUB_API void sneppx_hub_free_model_card(sneppx_model_card_t *card);

/**
 * Download a model to a local directory.
 *
 * @param ctx Hub context.
 * @param model_name Model name.
 * @param version Version string or NULL for latest.
 * @param dest_dir Destination directory (must exist).
 * @param progress_callback Optional callback (fn(filename, downloaded, total, userdata)).
 * @param userdata Opaque pointer passed to progress_callback.
 * @return 0 on success, -1 on error.
 */
typedef void (*sneppx_hub_progress_cb)(const char *filename, size_t downloaded, size_t total, void *userdata);

SNEPPX_HUB_API int sneppx_hub_download_model(
    SNEPPXHubContext *ctx,
    const char *model_name,
    const char *version,
    const char *dest_dir,
    sneppx_hub_progress_cb progress_callback,
    void *userdata
);

/**
 * Get leaderboard entries.
 *
 * @param ctx Hub context.
 * @param out_entries Pointer to receive leaderboard entries.
 * @param out_count Pointer to receive number of entries.
 * @param task Optional task filter (or NULL for all).
 * @param metric Optional metric filter (or NULL for all).
 * @return 0 on success, -1 on error.
 */
SNEPPX_HUB_API int sneppx_hub_get_leaderboard(
    SNEPPXHubContext *ctx,
    sneppx_leaderboard_entry_t **out_entries,
    int *out_count,
    const char *task,
    const char *metric
);

/**
 * Submit a benchmark result to the leaderboard.
 *
 * @param ctx Hub context.
 * @param result The benchmark result to submit.
 * @return 0 on success, -1 on error.
 */
SNEPPX_HUB_API int sneppx_hub_submit_benchmark(
    SNEPPXHubContext *ctx,
    const sneppx_benchmark_result_t *result
);

/**
 * Search for models (convenience wrapper around list_models with query).
 */
SNEPPX_HUB_API int sneppx_hub_search(
    SNEPPXHubContext *ctx,
    const char *query,
    sneppx_model_entry_t **out_models,
    int *out_count,
    const sneppx_hub_search_params_t *params
);

/**
 * Check if the hub is reachable.
 * @return 1 if reachable, 0 if not.
 */
SNEPPX_HUB_API int sneppx_hub_is_reachable(SNEPPXHubContext *ctx);

/**
 * Get the last error message (thread-local).
 */
SNEPPX_HUB_API const char *sneppx_hub_strerror(SNEPPXHubContext *ctx);

/**
 * Set a custom HTTP header on the context.
 * This can be used to add authentication or other headers.
 */
SNEPPX_HUB_API void sneppx_hub_set_header(SNEPPXHubContext *ctx, const char *key, const char *value);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_HUB_CLIENT_H */

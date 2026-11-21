#ifndef SNEPPX_SERVING_ENGINE_H
#define SNEPPX_SERVING_ENGINE_H

#include "http_server.h"

#include <stddef.h>

/*
 * SNEPPX - Serving Engine
 *
 * WHAT
 *   Production-grade model serving control plane: model versioning, rolling
 *   updates, A/B traffic splitting, dynamic batching, Prometheus metrics,
 *   health/readiness probes, model warm-up, YAML config hot-reload, and
 *   horizontal worker configuration.
 *
 * CONCEPT
 *   Plugs into SNEPPX_HttpServer alongside http_api.h to expose additional
 *   endpoints (/metrics, /healthz, /readyz, /v1/models/{name}/versions,
 *   /v1/traffic, /v1/deploy). The engine itself is pure logic and is unit
 *   tested without binding to a socket.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_SERVING_MAX_MODELS  64
#define SNEPPX_SERVING_MAX_VERSIONS 8
#define SNEPPX_SERVING_MAX_ID_LEN  64
#define SNEPPX_SERVING_MAX_KEY_LEN 128

typedef struct {
    char version_id[SNEPPX_SERVING_MAX_ID_LEN];
    char description[256];
    int  weight;           /* traffic share 0..100 */
    long long deployed_at; /* monotonic ms */
} SNEPPX_ServingVersion;

typedef struct {
    char name[SNEPPX_SERVING_MAX_ID_LEN];
    SNEPPX_ServingVersion versions[SNEPPX_SERVING_MAX_VERSIONS];
    int num_versions;
    int active;            /* index of active (promote target) version */
} SNEPPX_ServingModel;

typedef struct {
    long long requests;
    long long errors;
    long long tokens;
    long long batches;
} SNEPPX_ServingCounts;

typedef struct SNEPPX_ServingEngine SNEPPX_ServingEngine;

/* ---- lifecycle ---- */
SNEPPX_ServingEngine* SNEPPX_serving_engine_create(void);
void SNEPPX_serving_engine_destroy(SNEPPX_ServingEngine* e);

/* ---- configuration (hot-reloadable via YAML) ---- */
void SNEPPX_serving_engine_configure(SNEPPX_ServingEngine* e,
                                     int max_batch_size, int batch_timeout_ms,
                                     int n_workers, int warmup_iters);
void SNEPPX_serving_set_config_path(SNEPPX_ServingEngine* e, const char* yaml_path);
/* Returns 1 if config was reloaded, 0 if unchanged, -1 on parse error. */
int SNEPPX_serving_config_reload(SNEPPX_ServingEngine* e);

/* ---- config accessors (read-back after configure/reload) ---- */
int SNEPPX_serving_max_batch_size(const SNEPPX_ServingEngine* e);
int SNEPPX_serving_batch_timeout_ms(const SNEPPX_ServingEngine* e);
int SNEPPX_serving_worker_count(const SNEPPX_ServingEngine* e);
int SNEPPX_serving_warmup_iters(const SNEPPX_ServingEngine* e);

/* ---- model catalogue + versioning / rolling updates ---- */
/* Register a named model with an initial version. Returns 0/-1. */
int SNEPPX_serving_register_model(SNEPPX_ServingEngine* e,
                                  const char* name,
                                  const char* version_id,
                                  const char* description);
/* Promote `version_id` to active (traffic 100% on it). */
int SNEPPX_serving_set_active_version(SNEPPX_ServingEngine* e,
                                      const char* name,
                                      const char* version_id);
/* Shift all weight away from `version_id` to the active version (rollback). */
int SNEPPX_serving_rollback(SNEPPX_ServingEngine* e, const char* name);
/* Set a single version's traffic share (0..100). */
int SNEPPX_serving_set_weight(SNEPPX_ServingEngine* e,
                              const char* name,
                              const char* version_id,
                              int weight);
const char* SNEPPX_serving_active_version(SNEPPX_ServingEngine* e, const char* name);

/* A/B routing: consistent, request-id-based weighted selection.
 * Writes the selected version_id into out (size >= SNEPPX_SERVING_MAX_ID_LEN).
 * Returns 0 on success, -1 if the model is unknown. */
int SNEPPX_serving_route(SNEPPX_ServingEngine* e,
                         const char* name,
                         const char* request_id,
                         char* out_version);

/* ---- dynamic batching ---- */
/* Enqueue a request key; returns the new pending count. */
int SNEPPX_serving_batch_submit(SNEPPX_ServingEngine* e, const char* request_id);
/* Drain a ready batch into out_ids (each SNEPPX_SERVING_MAX_ID_LEN).
 * Returns 0 / 1 (ready) and fills *out_count. */
int SNEPPX_serving_batch_drain(SNEPPX_ServingEngine* e,
                               char out_ids[][SNEPPX_SERVING_MAX_ID_LEN],
                               int max, int* out_count);
SNEPPX_ServingModel* SNEPPX_serving_get_model(SNEPPX_ServingEngine* e, const char* name);
int SNEPPX_serving_model_count(SNEPPX_ServingEngine* e);

/* ---- metrics ---- */
void SNEPPX_serving_record_request(SNEPPX_ServingEngine* e,
                                   long long latency_us,
                                   long long tokens,
                                   int error);
SNEPPX_ServingCounts SNEPPX_serving_counts(SNEPPX_ServingEngine* e);
double SNEPPX_serving_latency_p50(SNEPPX_ServingEngine* e);
double SNEPPX_serving_latency_p95(SNEPPX_ServingEngine* e);
double SNEPPX_serving_latency_p99(SNEPPX_ServingEngine* e);
double SNEPPX_serving_throughput_rps(SNEPPX_ServingEngine* e);
/* Render metrics into caller buffer. Returns bytes written. */
size_t SNEPPX_serving_metrics_json(const SNEPPX_ServingEngine* e,
                                     char* out, size_t out_size);
size_t SNEPPX_serving_metrics_prometheus(const SNEPPX_ServingEngine* e,
                                         char* out, size_t out_size);

/* ---- health / readiness / warm-up ---- */
void SNEPPX_serving_set_ready(SNEPPX_ServingEngine* e, int ready);
int SNEPPX_serving_is_ready(const SNEPPX_ServingEngine* e);
void SNEPPX_serving_warmup_start(SNEPPX_ServingEngine* e);
/* One warm-up tick; auto-ready once warmup_iters reached. */
int SNEPPX_serving_warmup_tick(SNEPPX_ServingEngine* e, long long latency_us, long long tokens);
long long SNEPPX_serving_uptime_ms(const SNEPPX_ServingEngine* e);

/* ---- HTTP wiring ---- */
/* Registers /metrics, /healthz, /readyz, /v1/models/{name}/versions,
 * /v1/traffic, /v1/deploy on an existing server. Returns 0/-1. */
int SNEPPX_serving_engine_register(SNEPPX_HttpServer* srv, SNEPPX_ServingEngine* e);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_SERVING_ENGINE_H */

#include "serving_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#endif

/*
 * SNEPPX - Serving Engine
 *
 * WHAT
 *   Implementation of the serving control plane.
 *
 * CONCEPT
 *   See serving_engine.h.
 */

/* ---- monotonic clock (ms) ---- */
static long long now_ms(void) {
#ifdef _WIN32
    return (long long)GetTickCount64();
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (long long)ts.tv_sec * 1000LL + (long long)(ts.tv_nsec / 1000000LL);
    return (long long)clock();
#endif
}

#define MS_PER_S 1000.0
static double now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* FNV-1a 64-bit — deterministic request-id hashing for A/B routing. */
static unsigned long long fnv1a64(const char* s) {
    unsigned long long h = 1469598103934665603ULL;
    if (!s) return h;
    for (const unsigned char* p = (const unsigned char*)s; *p; p++) {
        h ^= (unsigned long long)(*p);
        h *= 1099511628211ULL;
    }
    return h;
}

static void set_str(char* dst, size_t n, const char* src) {
    if (n == 0) return;
    if (src) {
        size_t len = strlen(src);
        if (len >= n) len = n - 1;
        memcpy(dst, src, len);
        dst[len] = '\0';
    } else {
        dst[0] = '\0';
    }
}

/* ---- minimal JSON value extractor (self-contained) ---- */

static const char* json_skip(const char* p) {
    while (p && *p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static int json_get_string(const char* json, const char* key, char* out, size_t out_size) {
    if (!json || !key || !out || out_size == 0) return -1;
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return -1;
    p = json_skip(p + strlen(pattern));
    if (*p != ':') return -1;
    p = json_skip(p + 1);
    if (*p != '"') return -1;
    p++;
    size_t n = 0;
    while (*p && *p != '"' && n + 1 < out_size) {
        if (*p == '\\' && p[1]) {
            switch (p[1]) {
                case 'n': out[n++] = '\n'; break;
                case 't': out[n++] = '\t'; break;
                case 'r': out[n++] = '\r'; break;
                case '\\': out[n++] = '\\'; break;
                case '"': out[n++] = '"'; break;
                default: out[n++] = p[1]; break;
            }
            p += 2;
        } else {
            out[n++] = *p++;
        }
    }
    out[n] = '\0';
    return 0;
}

static long long json_get_int(const char* json, const char* key, long long def) {
    char buf[128];
    if (json_get_string(json, key, buf, sizeof(buf)) != 0) return def;
    return atoll(buf);
}

/* ---- latency histogram (fixed buckets in microseconds) ---- */
#define N_BUCKETS 16
static const long long kBuckets[N_BUCKETS + 1] = {
    50LL, 100LL, 250LL, 500LL, 1000LL, 2500LL, 5000LL, 10000LL,
    25000LL, 50000LL, 100000LL, 250000LL, 500000LL, 1000000LL, 2500000LL, 100000000LL
};
static size_t bucket_index(long long us) {
    for (size_t i = 0; i < N_BUCKETS; i++)
        if (us <= kBuckets[i]) return i;
    return N_BUCKETS - 1;
}

struct SNEPPX_ServingEngine {
    int max_batch_size;
    int batch_timeout_ms;
    int n_workers;
    int warmup_iters;

    /* catalogue */
    SNEPPX_ServingModel models[SNEPPX_SERVING_MAX_MODELS];
    int num_models;

    /* dynamic batching state */
    char pending_keys[SNEPPX_SERVING_MAX_ID_LEN * 64]; /* ring buffer of ids */
    int pending_count;
    long long pending_first_ms;
    int batch_seq;

    /* metrics */
    long long requests;
    long long errors;
    long long tokens;
    long long batches;
    long long histogram[N_BUCKETS];
    double lat_sum_us;
    double lat_sum_sq_us;
    long long lat_n;
    long long start_ms;

    /* health / warm-up */
    int ready;
    int warmup_done;
    int warmup_remaining;
    long long uptime_offset_ms;

    /* config hot-reload */
    char config_path[512];
    long long config_mtime; /* best-effort; 0 = unknown */
};

static long long file_mtime_ms(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (long long)st.st_mtime * 1000LL;
}

/* ---- lifecycle ---- */

SNEPPX_ServingEngine* SNEPPX_serving_engine_create(void) {
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->max_batch_size = 8;
    e->batch_timeout_ms = 5;
    e->n_workers = 1;
    e->warmup_iters = 0;
    e->start_ms = now_ms();
    e->config_mtime = 0;
    return e;
}

void SNEPPX_serving_engine_destroy(SNEPPX_ServingEngine* e) {
    free(e);
}

void SNEPPX_serving_engine_configure(SNEPPX_ServingEngine* e,
                                     int max_batch_size, int batch_timeout_ms,
                                     int n_workers, int warmup_iters) {
    if (!e) return;
    if (max_batch_size > 0) e->max_batch_size = max_batch_size;
    e->batch_timeout_ms = batch_timeout_ms >= 0 ? batch_timeout_ms : e->batch_timeout_ms;
    if (n_workers > 0) e->n_workers = n_workers;
    e->warmup_iters = warmup_iters >= 0 ? warmup_iters : e->warmup_iters;
}

int SNEPPX_serving_max_batch_size(const SNEPPX_ServingEngine* e) { return e ? e->max_batch_size : 0; }
int SNEPPX_serving_batch_timeout_ms(const SNEPPX_ServingEngine* e) { return e ? e->batch_timeout_ms : 0; }
int SNEPPX_serving_worker_count(const SNEPPX_ServingEngine* e) { return e ? e->n_workers : 0; }
int SNEPPX_serving_warmup_iters(const SNEPPX_ServingEngine* e) { return e ? e->warmup_iters : 0; }

void SNEPPX_serving_set_config_path(SNEPPX_ServingEngine* e, const char* yaml_path) {
    if (!e) return;
    set_str(e->config_path, sizeof(e->config_path), yaml_path);
    /* Force a (re)load on the next reload(); mtime is snapshotted by reload. */
    e->config_mtime = 0;
}

/* ---- minimal YAML subset parser ----
 * Supports: dotted `key.sub: value` scalars, `serving:` parent blocks and
 * `model.<name>.canary <weight>` style entries. Sufficient for config hot-reload
 * without an external YAML dependency. */
static int parse_int_scalar(const char* s, int def) {
    char* end = NULL;
    long v = strtol(s, &end, 10);
    return (end && end != s) ? (int)v : def;
}

static void trim(char* s) {
    while (*s == ' ' || *s == '\t') memmove(s, s + 1, strlen(s));
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r' || s[n - 1] == '\n'))
        s[--n] = '\0';
}

static int model_index(SNEPPX_ServingEngine* e, const char* name) {
    for (int i = 0; i < e->num_models; i++)
        if (strcmp(e->models[i].name, name) == 0) return i;
    return -1;
}

static int version_index(const SNEPPX_ServingModel* m, const char* version_id) {
    for (int i = 0; i < m->num_versions; i++)
        if (strcmp(m->versions[i].version_id, version_id) == 0) return i;
    return -1;
}

int SNEPPX_serving_config_reload(SNEPPX_ServingEngine* e) {
    if (!e || !e->config_path || !*e->config_path) return 0;
    long long mt = file_mtime_ms(e->config_path);
    if (mt == 0) return 0; /* file not found */
    if (mt == e->config_mtime) return 0; /* unchanged */
    e->config_mtime = mt;

    FILE* f = fopen(e->config_path, "r");
    if (!f) return -1;
    char line[512];
    int changed = 0;
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0' || *p == '\r') continue;
        char key[256], val[256];
        /* strip trailing newline */
        char* nl = p + strlen(p);
        while (nl > p && (nl[-1] == '\n' || nl[-1] == '\r')) { *--nl = '\0'; }
        char* colon = strstr(p, ":");
        if (!colon) {
            /* model.<name>.canary <weight> form (no colon, space separated) */
            continue;
        }
        size_t kl = (size_t)(colon - p);
        if (kl >= sizeof(key)) kl = sizeof(key) - 1;
        memcpy(key, p, kl); key[kl] = '\0';
        memmove(val, colon + 1, strlen(colon + 1) + 1);
        trim(key); trim(val);
        if (strcmp(key, "serving.max_batch_size") == 0) {
            e->max_batch_size = parse_int_scalar(val, e->max_batch_size);
            changed = 1;
        } else if (strcmp(key, "serving.batch_timeout_ms") == 0) {
            e->batch_timeout_ms = parse_int_scalar(val, e->batch_timeout_ms);
            changed = 1;
        } else if (strcmp(key, "serving.n_workers") == 0) {
            e->n_workers = parse_int_scalar(val, e->n_workers);
            changed = 1;
        } else if (strcmp(key, "serving.warmup_iters") == 0) {
            e->warmup_iters = parse_int_scalar(val, e->warmup_iters);
            changed = 1;
        } else if (strncmp(key, "model.", 6) == 0 && val[0] != '\0') {
            /* model.<name>.canary <weight> (colon present, value is weight) */
            char rest[256];
            strncpy(rest, key + 6, sizeof(rest) - 1);
            rest[sizeof(rest) - 1] = '\0';
            /* rest = "<name>.canary" */
            char* dot = strrchr(rest, '.');
            if (dot) {
                *dot = '\0';
                int w = parse_int_scalar(val, -1);
                int mi = model_index(e, rest);
                if (mi >= 0 && w >= 0) {
                    int vi = e->models[mi].active;
                    e->models[mi].versions[vi].weight = w;
                    changed = 1;
                }
            }
        }
    }
    fclose(f);
    return changed ? 1 : 0;
}

/* ---- model registry ---- */

int SNEPPX_serving_register_model(SNEPPX_ServingEngine* e,
                                  const char* name,
                                  const char* version_id,
                                  const char* description) {
    if (!e || !name || !version_id) return -1;
    int mi = model_index(e, name);
    SNEPPX_ServingModel* m;
    if (mi < 0) {
        if (e->num_models >= SNEPPX_SERVING_MAX_MODELS) return -1;
        mi = e->num_models++;
        m = &e->models[mi];
        memset(m, 0, sizeof(*m));
        set_str(m->name, sizeof(m->name), name);
        m->active = 0;
    } else {
        m = &e->models[mi];
    }
    if (m->num_versions >= SNEPPX_SERVING_MAX_VERSIONS) return -1;
    if (version_index(m, version_id) >= 0) return -1; /* already exists */
    SNEPPX_ServingVersion* v = &m->versions[m->num_versions++];
    memset(v, 0, sizeof(*v));
    set_str(v->version_id, sizeof(v->version_id), version_id);
    set_str(v->description, sizeof(v->description), description);
    v->weight = (m->num_versions == 1) ? 100 : 0;
    v->deployed_at = now_ms();
    if (m->num_versions == 1) m->active = 0;
    return 0;
}

int SNEPPX_serving_set_active_version(SNEPPX_ServingEngine* e,
                                      const char* name,
                                      const char* version_id) {
    int mi = model_index(e, name);
    if (mi < 0) return -1;
    SNEPPX_ServingModel* m = &e->models[mi];
    int vi = version_index(m, version_id);
    if (vi < 0) return -1;
    m->active = vi;
    /* 100% traffic on the promoted version; others zero (rolling complete). */
    for (int i = 0; i < m->num_versions; i++)
        m->versions[i].weight = (i == vi) ? 100 : 0;
    return 0;
}

int SNEPPX_serving_rollback(SNEPPX_ServingEngine* e, const char* name) {
    int mi = model_index(e, name);
    if (mi < 0) return -1;
    SNEPPX_ServingModel* m = &e->models[mi];
    if (m->num_versions < 2) return -1;
    /* Roll back: drop the most recently added version (the canary),
     * restore full traffic to the active version. */
    m->num_versions--;
    if (m->active >= m->num_versions) m->active = 0;
    for (int i = 0; i < m->num_versions; i++)
        m->versions[i].weight = (i == m->active) ? 100 : 0;
    return 0;
}

int SNEPPX_serving_set_weight(SNEPPX_ServingEngine* e,
                              const char* name,
                              const char* version_id,
                              int weight) {
    int mi = model_index(e, name);
    if (mi < 0) return -1;
    SNEPPX_ServingModel* m = &e->models[mi];
    int vi = version_index(m, version_id);
    if (vi < 0) return -1;
    if (weight < 0) weight = 0;
    if (weight > 100) weight = 100;
    m->versions[vi].weight = weight;
    return 0;
}

const char* SNEPPX_serving_active_version(SNEPPX_ServingEngine* e, const char* name) {
    int mi = model_index(e, name);
    if (mi < 0) return NULL;
    return e->models[mi].versions[e->models[mi].active].version_id;
}

int SNEPPX_serving_route(SNEPPX_ServingEngine* e,
                         const char* name,
                         const char* request_id,
                         char* out_version) {
    int mi = model_index(e, name);
    if (mi < 0) return -1;
    SNEPPX_ServingModel* m = &e->models[mi];
    unsigned long long h = fnv1a64(request_id ? request_id : "0");
    int total = 0;
    for (int i = 0; i < m->num_versions; i++)
        total += m->versions[i].weight;
    if (total <= 0) {
        set_str(out_version, SNEPPX_SERVING_MAX_ID_LEN, m->versions[m->active].version_id);
        return 0;
    }
    int r = (int)((h % (unsigned long long)total) + 1); /* 1..total */
    int cum = 0;
    for (int i = 0; i < m->num_versions; i++) {
        cum += m->versions[i].weight;
        if (r <= cum) {
            set_str(out_version, SNEPPX_SERVING_MAX_ID_LEN, m->versions[i].version_id);
            return 0;
        }
    }
    set_str(out_version, SNEPPX_SERVING_MAX_ID_LEN, m->versions[m->active].version_id);
    return 0;
}

SNEPPX_ServingModel* SNEPPX_serving_get_model(SNEPPX_ServingEngine* e, const char* name) {
    int mi = model_index(e, name);
    return (mi >= 0) ? &e->models[mi] : NULL;
}

int SNEPPX_serving_model_count(SNEPPX_ServingEngine* e) {
    return e ? e->num_models : 0;
}

/* ---- dynamic batching ---- */

int SNEPPX_serving_batch_submit(SNEPPX_ServingEngine* e, const char* request_id) {
    if (!e || !request_id) return -1;
    if (e->pending_count == 0)
        e->pending_first_ms = now_ms();
    if (e->pending_count < 64)
        set_str(e->pending_keys + (size_t)e->pending_count * SNEPPX_SERVING_MAX_ID_LEN,
                SNEPPX_SERVING_MAX_ID_LEN, request_id);
    e->pending_count++;
    return e->pending_count;
}

int SNEPPX_serving_batch_drain(SNEPPX_ServingEngine* e,
                               char out_ids[][SNEPPX_SERVING_MAX_ID_LEN],
                               int max, int* out_count) {
    if (!e || !out_count) return -1;
    *out_count = 0;
    int ready = 0;
    long long elapsed = (e->pending_count > 0) ? (now_ms() - e->pending_first_ms) : 0;
    if (e->pending_count >= e->max_batch_size) ready = 1;
    else if (e->pending_count > 0 && elapsed >= e->batch_timeout_ms) ready = 1;
    if (!ready) return 0;
    int n = e->pending_count;
    if (n > max) n = max;
    for (int i = 0; i < n; i++)
        set_str(out_ids[i], SNEPPX_SERVING_MAX_ID_LEN,
                e->pending_keys + (size_t)i * SNEPPX_SERVING_MAX_ID_LEN);
    /* rotate remaining out */
    int left = e->pending_count - n;
    if (left > 0)
        memmove(e->pending_keys,
                e->pending_keys + (size_t)n * SNEPPX_SERVING_MAX_ID_LEN,
                (size_t)left * SNEPPX_SERVING_MAX_ID_LEN);
    e->pending_count -= n;
    *out_count = n;
    if (e->pending_count == 0) e->pending_first_ms = 0;
    e->batches++;
    e->batch_seq++;
    return 1;
}

/* ---- metrics ---- */

void SNEPPX_serving_record_request(SNEPPX_ServingEngine* e,
                                   long long latency_us,
                                   long long tokens,
                                   int error) {
    if (!e) return;
    e->requests++;
    if (error) e->errors++;
    e->tokens += tokens;
    e->lat_sum_us += (double)latency_us;
    e->lat_sum_sq_us += (double)latency_us * (double)latency_us;
    e->lat_n++;
    e->histogram[bucket_index(latency_us)]++;
}

SNEPPX_ServingCounts SNEPPX_serving_counts(SNEPPX_ServingEngine* e) {
    SNEPPX_ServingCounts c = {0, 0, 0, 0};
    if (e) {
        c.requests = e->requests;
        c.errors = e->errors;
        c.tokens = e->tokens;
        c.batches = e->batches;
    }
    return c;
}

static double percentile(SNEPPX_ServingEngine* e, double p) {
    if (!e || e->lat_n <= 0) return 0.0;
    long long target = (long long)(e->lat_n * p);
    if (target < 1) target = 1;
    long long cum = 0;
    for (size_t i = 0; i < N_BUCKETS; i++) {
        cum += e->histogram[i];
        if (cum >= target) return (double)kBuckets[i];
    }
    return (double)kBuckets[N_BUCKETS - 1];
}

double SNEPPX_serving_latency_p50(SNEPPX_ServingEngine* e) { return percentile(e, 0.50); }
double SNEPPX_serving_latency_p95(SNEPPX_ServingEngine* e) { return percentile(e, 0.95); }
double SNEPPX_serving_latency_p99(SNEPPX_ServingEngine* e) { return percentile(e, 0.99); }

double SNEPPX_serving_throughput_rps(SNEPPX_ServingEngine* e) {
    if (!e) return 0.0;
    long long up = now_ms() - e->start_ms;
    if (up <= 0) return (double)e->requests; /* degenerate: no measurable elapsed time */
    return (double)e->requests / ((double)up / 1000.0);
}

long long SNEPPX_serving_uptime_ms(const SNEPPX_ServingEngine* e) {
    return e ? (now_ms() - e->start_ms) : 0;
}

static size_t fmt_int(char* out, size_t sz, const char* name, long long val) {
    return (size_t)snprintf(out, sz, "\"%s\":%lld,", name, val);
}

size_t SNEPPX_serving_metrics_json(const SNEPPX_ServingEngine* e,
                                   char* out, size_t out_size) {
    if (!e || !out || out_size == 0) return 0;
    int n = e->num_models;
    size_t pos = 0;
    pos += (size_t)snprintf(out + pos, out_size - pos, "{");
    pos += (size_t)snprintf(out + pos, out_size - pos,
                            "\"uptime_ms\":%lld,"
                            "\"ready\":%s,"
                            "\"models_loaded\":%d,"
                            "\"max_batch_size\":%d,"
                            "\"batch_timeout_ms\":%d,"
                            "\"n_workers\":%d,",
                            SNEPPX_serving_uptime_ms(e),
                            e->ready ? "true" : "false",
                            n, e->max_batch_size, e->batch_timeout_ms, e->n_workers);
    pos += (size_t)snprintf(out + pos, out_size - pos, "\"counts\":{");
    pos += fmt_int(out + pos, out_size - pos, "requests", e->requests);
    pos += fmt_int(out + pos, out_size - pos, "errors", e->errors);
    pos += fmt_int(out + pos, out_size - pos, "tokens", e->tokens);
    pos += fmt_int(out + pos, out_size - pos, "batches", e->batches);
    /* strip trailing comma */
    if (pos > 0 && out[pos - 1] == ',') out[pos - 1] = '}';
    pos += (size_t)snprintf(out + pos, out_size - pos, "},");
    pos += (size_t)snprintf(out + pos, out_size - pos,
                            "\"latency\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f,\"throughput_rps\":%.3f}",
                            SNEPPX_serving_latency_p50((SNEPPX_ServingEngine*)e),
                            SNEPPX_serving_latency_p95((SNEPPX_ServingEngine*)e),
                            SNEPPX_serving_latency_p99((SNEPPX_ServingEngine*)e),
                            SNEPPX_serving_throughput_rps((SNEPPX_ServingEngine*)e));
    pos += (size_t)snprintf(out + pos, out_size - pos, "}");
    return pos;
}

size_t SNEPPX_serving_metrics_prometheus(const SNEPPX_ServingEngine* e,
                                         char* out, size_t out_size) {
    if (!e || !out || out_size == 0) return 0;
    size_t pos = 0;
    SNEPPX_ServingCounts c = SNEPPX_serving_counts((SNEPPX_ServingEngine*)e);
    pos += (size_t)snprintf(out + pos, out_size - pos,
        "# HELP sneppx_serving_requests_total Total requests.\n"
        "# TYPE sneppx_serving_requests_total counter\n"
        "sneppx_serving_requests_total %lld\n"
        "# HELP sneppx_serving_errors_total Total errors.\n"
        "# TYPE sneppx_serving_errors_total counter\n"
        "sneppx_serving_errors_total %lld\n"
        "# HELP sneppx_serving_tokens_total Tokens generated.\n"
        "# TYPE sneppx_serving_tokens_total counter\n"
        "sneppx_serving_tokens_total %lld\n"
        "# HELP sneppx_serving_batches_total Batches drained.\n"
        "# TYPE sneppx_serving_batches_total counter\n"
        "sneppx_serving_batches_total %lld\n",
        c.requests, c.errors, c.tokens, c.batches);
    pos += (size_t)snprintf(out + pos, out_size - pos,
        "# HELP sneppx_latency_ms Latency histogram (ms) by bucket.\n"
        "# TYPE sneppx_latency_ms histogram\n");
    long long sum_us = (long long)((SNEPPX_ServingEngine*)e)->lat_sum_us;
    double sum_ms = (double)sum_us / 1000.0;
    long long obs = ((SNEPPX_ServingEngine*)e)->lat_n;
    pos += (size_t)snprintf(out + pos, out_size - pos,
        "sneppx_latency_ms_bucket{le=\"%.3f\"} %lld\n",
        (double)kBuckets[0] / 1000.0, e->histogram[0]);
    for (size_t i = 1; i < N_BUCKETS; i++)
        pos += (size_t)snprintf(out + pos, out_size - pos,
            "sneppx_latency_ms_bucket{le=\"%.3f\"} %lld\n",
            (double)kBuckets[i] / 1000.0, e->histogram[i]);
    pos += (size_t)snprintf(out + pos, out_size - pos,
        "sneppx_latency_ms_bucket{le=\"+Inf\"} %lld\n"
        "sneppx_latency_ms_sum %f\n"
        "sneppx_latency_ms_count %lld\n",
        obs, sum_ms, obs);
    pos += (size_t)snprintf(out + pos, out_size - pos,
        "# HELP sneppx_serving_ready 1 if ready, 0 otherwise.\n"
        "# TYPE sneppx_serving_ready gauge\n"
        "sneppx_serving_ready %d\n"
        "# HELP sneppx_serving_models_loaded Number of registered models.\n"
        "# TYPE sneppx_serving_models_loaded gauge\n"
        "sneppx_serving_models_loaded %d\n"
        "# HELP sneppx_serving_throughput_rps Requests per second.\n"
        "# TYPE sneppx_serving_throughput_rps gauge\n"
        "sneppx_serving_throughput_rps %.3f\n",
        e->ready ? 1 : 0, e->num_models,
        (double)c.requests / ((SNEPPX_serving_uptime_ms((SNEPPX_ServingEngine*)e) > 0)
            ? (double)SNEPPX_serving_uptime_ms((SNEPPX_ServingEngine*)e) / 1000.0 : 1.0));
    return pos;
}

/* ---- health / warm-up ---- */

void SNEPPX_serving_set_ready(SNEPPX_ServingEngine* e, int ready) {
    if (e) e->ready = ready ? 1 : 0;
}

int SNEPPX_serving_is_ready(const SNEPPX_ServingEngine* e) {
    return e ? (e->ready ? 1 : 0) : 0;
}

void SNEPPX_serving_warmup_start(SNEPPX_ServingEngine* e) {
    if (!e) return;
    e->warmup_remaining = e->warmup_iters;
    e->warmup_done = 0;
}

int SNEPPX_serving_warmup_tick(SNEPPX_ServingEngine* e, long long latency_us, long long tokens) {
    if (!e) return 0;
    if (e->warmup_remaining > 0) {
        SNEPPX_serving_record_request(e, latency_us, tokens, 0);
        e->warmup_remaining--;
    }
    if (e->warmup_remaining <= 0 && !e->warmup_done) {
        e->warmup_done = 1;
        e->ready = 1;
        return 1; /* became ready this tick */
    }
    return 0;
}

/* ---- HTTP handlers ---- */

static int handle_metrics(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* ud) {
    (void)req;
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)ud;
    const char* accept = SNEPPX_http_request_header(req, "Accept");
    char buf[65536];
    if (accept && strstr(accept, "application/json")) {
        SNEPPX_serving_metrics_json(e, buf, sizeof(buf));
        SNEPPX_http_response_set_header(resp, "Content-Type", "application/json");
    } else {
        SNEPPX_serving_metrics_prometheus(e, buf, sizeof(buf));
        SNEPPX_http_response_set_header(resp, "Content-Type", "text/plain; version=0.0.4");
    }
    SNEPPX_http_response_set_body_str(resp, buf);
    return 0;
}

static int handle_healthz(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* ud) {
    (void)req;
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)ud;
    char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"status\":%s,\"uptime_ms\":%lld}",
        e->ready ? "true" : "false", SNEPPX_serving_uptime_ms(e));
    SNEPPX_http_response_set_json(resp, buf);
    return 0;
}

static int handle_readyz(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* ud) {
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)ud;
    if (!SNEPPX_serving_is_ready(e)) {
        SNEPPX_http_response_set_status(resp, 503);
        SNEPPX_http_response_set_json(resp, "{\"ready\":false}");
        return 0;
    }
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"ready\":true,\"uptime_ms\":%lld}", SNEPPX_serving_uptime_ms(e));
    SNEPPX_http_response_set_json(resp, buf);
    return 0;
}

static void model_json(SNEPPX_ServingEngine* e, const SNEPPX_ServingModel* m, char* out, size_t n) {
    size_t pos = 0;
    pos += (size_t)snprintf(out + pos, n - pos, "{\"name\":\"%s\",\"active\":\"%s\",\"versions\":[",
                           m->name, m->versions[m->active].version_id);
    for (int i = 0; i < m->num_versions; i++) {
        if (i) pos += (size_t)snprintf(out + pos, n - pos, ",");
        pos += (size_t)snprintf(out + pos, n - pos,
            "{\"id\":\"%s\",\"weight\":%d,\"deployed_at\":%lld}",
            m->versions[i].version_id, m->versions[i].weight, m->versions[i].deployed_at);
    }
    pos += (size_t)snprintf(out + pos, n - pos, "]}");
    (void)e;
}

static int handle_list_versions(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* ud) {
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)ud;
    const char* name = SNEPPX_http_request_param(req, "name");
    if (!name) name = SNEPPX_http_request_query(req, "model");
    char buf[4096];
    if (name) {
        SNEPPX_ServingModel* m = SNEPPX_serving_get_model(e, name);
        if (!m) {
            SNEPPX_http_response_set_status(resp, 404);
            SNEPPX_http_response_set_json(resp, "{\"error\":\"model_not_found\"}");
            return 0;
        }
        model_json(e, m, buf, sizeof(buf));
        SNEPPX_http_response_set_json(resp, buf);
        return 0;
    }
    /* list all */
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "{\"models\":[");
    for (int i = 0; i < e->num_models; i++) {
        if (i) pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, ",");
        model_json(e, &e->models[i], buf + pos, sizeof(buf) - pos);
        pos = strlen(buf);
    }
    pos += (size_t)snprintf(buf + pos, sizeof(buf) - pos, "]}");
    SNEPPX_http_response_set_json(resp, buf);
    return 0;
}

static int handle_traffic(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* ud) {
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)ud;
    if (strcmp(SNEPPX_http_request_method(req), "POST") != 0) {
        handle_list_versions(req, resp, ud);
        return 0;
    }
    size_t blen = 0;
    const char* body = SNEPPX_http_request_body(req, &blen);
    char name[128] = {0}, a[128] = {0}, b[128] = {0};
    long long pa = -1, pb = -1;
    /* parse simple JSON: {"model":"x","a":"v1","b":"v2","percent_a":70} */
    if (json_get_string(body, "model", name, sizeof(name)) != 0 || name[0] == '\0') {
        SNEPPX_http_response_set_status(resp, 400);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_traffic_request\"}");
        return 0;
    }
    json_get_string(body, "a", a, sizeof(a));
    json_get_string(body, "b", b, sizeof(b));
    pa = json_get_int(body, "percent_a", -1);
    pb = json_get_int(body, "percent_b", -1);
    int rc = 0;
    if (a[0] && pa >= 0) rc |= SNEPPX_serving_set_weight(e, name, a, (int)pa);
    if (b[0] && pb >= 0) rc |= SNEPPX_serving_set_weight(e, name, b, (int)pb);
    if (rc != 0) {
        SNEPPX_http_response_set_status(resp, 404);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"model_or_version_not_found\"}");
        return 0;
    }
    SNEPPX_http_response_set_json(resp, "{\"status\":\"ok\"}");
    return 0;
}

static int handle_deploy(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* ud) {
    SNEPPX_ServingEngine* e = (SNEPPX_ServingEngine*)ud;
    if (strcmp(SNEPPX_http_request_method(req), "POST") != 0) {
        SNEPPX_http_response_set_status(resp, 405);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"method_not_allowed\"}");
        return 0;
    }
    size_t blen = 0;
    const char* body = SNEPPX_http_request_body(req, &blen);
    char name[128] = {0}, version[128] = {0}, desc[256] = {0};
    if (json_get_string(body, "model", name, sizeof(name)) != 0 || name[0] == '\0' ||
        json_get_string(body, "version", version, sizeof(version)) != 0 || version[0] == '\0') {
        SNEPPX_http_response_set_status(resp, 400);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_deploy_request\"}");
        return 0;
    }
    json_get_string(body, "description", desc, sizeof(desc));
    int promote = (int)json_get_int(body, "promote", 0);
    if (SNEPPX_serving_register_model(e, name, version, desc) != 0) {
        SNEPPX_http_response_set_status(resp, 409);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"version_exists_or_full\"}");
        return 0;
    }
    if (promote)
        SNEPPX_serving_set_active_version(e, name, version);
    char out[512];
    snprintf(out, sizeof(out), "{\"status\":\"deployed\",\"model\":\"%s\",\"version\":\"%s\"}",
             name, version);
    SNEPPX_http_response_set_json(resp, out);
    return 0;
}

int SNEPPX_serving_engine_register(SNEPPX_HttpServer* srv, SNEPPX_ServingEngine* e) {
    if (!srv || !e) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/metrics", handle_metrics, e) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/healthz", handle_healthz, e) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/readyz", handle_readyz, e) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/v1/models/versions", handle_list_versions, e) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "GET", "/v1/models/versions/{name}", handle_list_versions, e) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "POST", "/v1/traffic", handle_traffic, e) != 0) return -1;
    if (SNEPPX_http_server_add_route(srv, "POST", "/v1/deploy", handle_deploy, e) != 0) return -1;
    return 0;
}

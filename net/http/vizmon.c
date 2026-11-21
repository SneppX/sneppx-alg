/*
 * SNEPPX - Visualization & Monitoring (vizmon) — control plane + HTTP/WS server.
 *
 * Pure-C implementation: thread-safe ring-buffer metric stores, PCA projection
 * (power-iteration SVD to 2-D), minimal HTTP file server + WebSocket broadcaster.
 * No third-party deps; on Windows link ws2_32.
 */
#include "neural_core/vizmon.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ---- cross-platform socket + thread boilerplate (mirrors net/http/http_server.c) ---- */
#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET snepx_socket_t;
#  define SNEPPX_VIZMON_INVALID_SOCK INVALID_SOCKET
#  define SNEPPX_VIZMON_SOCK_ERR     SOCKET_ERROR
#  define snepx_close_sock closesocket
#  define snepx_sleep_ms(ms) Sleep(ms)
   typedef HANDLE snepx_thread_t;
   typedef DWORD  (WINAPI *snepx_thread_fn)(void*);
   static snepx_thread_t snepx_thread_create(snepx_thread_fn fn, void *arg) {
       return CreateThread(NULL, 0, fn, arg, 0, NULL);
   }
   static void snepx_thread_join(snepx_thread_t t) { WaitForSingleObject(t, INFINITE); }
   static int snepx_set_reuseaddr(snepx_socket_t s) {
       BOOL opt = 1; return setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
   }
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <pthread.h>
   typedef int snepx_socket_t;
#  define SNEPPX_VIZMON_INVALID_SOCK (-1)
#  define SNEPPX_VIZMON_SOCK_ERR     (-1)
#  define snepx_close_sock close
#  define snepx_sleep_ms(ms) usleep((ms)*1000)
   typedef pthread_t snepx_thread_t;
    typedef void* (*snepx_thread_fn)(void*);
#  include <fcntl.h>
#  include <errno.h>
    static snepx_thread_t snepx_thread_create(snepx_thread_fn fn, void *arg) {
       pthread_t t; if (pthread_create(&t, NULL, fn, arg) != 0) return (pthread_t)0;
       return t;
   }
   static void snepx_thread_join(snepx_thread_t t) { pthread_join(t, NULL); }
   static int snepx_set_reuseaddr(snepx_socket_t s) {
       int opt = 1; return setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
   }
#endif

/* ---- mutex ---- */
#ifdef _WIN32
   typedef CRITICAL_SECTION snepx_mutex_t;
#  define snepx_mutex_init(m) InitializeCriticalSection(m)
#  define snepx_mutex_destroy(m) DeleteCriticalSection(m)
#  define snepx_mutex_lock(m) EnterCriticalSection(m)
#  define snepx_mutex_unlock(m) LeaveCriticalSection(m)
#else
   typedef pthread_mutex_t snepx_mutex_t;
#  define snepx_mutex_init(m) pthread_mutex_init(m, NULL)
#  define snepx_mutex_destroy(m) pthread_mutex_destroy(m)
#  define snepx_mutex_lock(m) pthread_mutex_lock(m)
#  define snepx_mutex_unlock(m) pthread_mutex_unlock(m)
#endif

/* ---- time ---- */
static double snepx_now_s(void) {
    double t = (double)clock() / (double)CLOCKS_PER_SEC;
#ifdef _WIN32
    static int once = 0;
    if (!once) { static double freq = 0; if (!freq) { LARGE_INTEGER f; QueryPerformanceFrequency(&f); freq = (double)f.QuadPart; } once = 1; LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart / freq; }
#endif
    return t;
}

/* ---- ring buffer helpers ---- */
typedef struct {
    SNEPPX_VizScalar *buf;
    int cap;
    int head;      /* next read */
    int tail;      /* next write */
    int count;
} ScalarRing;

static void ring_push(ScalarRing *r, const char *tag, double value, double ts) {
    SNEPPX_VizScalar *s = &r->buf[r->tail];
    strncpy(s->tag, tag, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    s->tag[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    s->value = value; s->ts = ts;
    r->tail = (r->tail + 1) % r->cap;
    if (r->count == r->cap) { r->head = (r->head + 1) % r->cap; }
    else { r->count++; }
}
static void ring_snapshot(ScalarRing *r, SNEPPX_VizScalar *out, int *n, int maxn) {
    int k = r->count < maxn ? r->count : maxn;
    *n = k;
    for (int i = 0; i < k; i++) {
        int idx = (r->head + i) % r->cap;
        out[i] = r->buf[idx];
    }
}

/* ---- series ---- */
typedef struct {
    char tag[SNEPPX_VIZMON_MAX_TAG_LEN];
    double *values;
    double *times;
    int count;
    int cap;
} SeriesStore;

static int series_find(SeriesStore *s, int n, const char *tag) {
    for (int i = 0; i < n; i++) if (strcmp(s[i].tag, tag) == 0) return i;
    return -1;
}
static SeriesStore *series_get_or_create(SeriesStore **s, int *n, int *cap, const char *tag) {
    int idx = series_find(*s, *n, tag);
    if (idx >= 0) return &(*s)[idx];
    if (*n >= *cap) {
        int nc = *cap ? *cap * 2 : 8;
        SeriesStore *ns = (SeriesStore*)realloc(*s, nc * sizeof(SeriesStore));
        if (!ns) return NULL;
        *s = ns; *cap = nc;
    }
    SeriesStore *e = &(*s)[*n];
    strncpy(e->tag, tag, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    e->tag[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    e->count = 0; e->cap = 256;
    e->values = (double*)calloc(e->cap, sizeof(double));
    e->times  = (double*)calloc(e->cap, sizeof(double));
    (*n)++;
    return e;
}
static void series_push(SeriesStore *e, double v, double ts) {
    if (e->count >= e->cap) {
        int nc = e->cap * 2;
        double *nv = (double*)realloc(e->values, nc * sizeof(double));
        double *nt = (double*)realloc(e->times,  nc * sizeof(double));
        if (!nv || !nt) return;
        e->values = nv; e->times = nt; e->cap = nc;
    }
    e->values[e->count] = v; e->times[e->count] = ts; e->count++;
}

/* ---- graph nodes ---- */
typedef struct {
    char name[SNEPPX_VIZMON_MAX_TAG_LEN];
    int in_nodes[SNEPPX_VIZMON_MAX_SCRATCH];
    int num_in;
    int out_node;
    double shape[SNEPPX_VIZMON_MAX_SCRATCH];
    int nshape;
} GraphNode;

/* ---- embeddings ---- */
#define EMB_CAP 8
typedef struct {
    char tag[SNEPPX_VIZMON_MAX_TAG_LEN];
    double *data;
    double *proj;
    int n_samples;
    int n_dim;
    int projected;
    int in_use;
} EmbeddingStore;

/* ---- samples ---- */
typedef struct {
    char tag[SNEPPX_VIZMON_MAX_TAG_LEN];
    SNEPPX_VizSampleKind kind;
    unsigned char *bytes;
    size_t len;
    double ts;
} SampleStore;

/* ---- histograms ---- */
typedef struct {
    char tag[SNEPPX_VIZMON_MAX_TAG_LEN];
    double *edges;
    int *counts;
    int n_bins;
    double ts;
    int in_use;
} HistStore;

/* ---- timeline ---- */
typedef struct {
    char name[SNEPPX_VIZMON_MAX_TAG_LEN];
    double start_ms;
    double dur_ms;
    size_t mem_bytes;
} TimelineEvent;

/* ---- sweeps ---- */
typedef struct {
    char config_id[SNEPPX_VIZMON_MAX_TAG_LEN];
    double final_loss;
    double final_acc;
    double duration_s;
} SweepResult;

/* ---- engine ---- */
struct SNEPPX_VizMon {
    snepx_mutex_t lock;

    /* scalars */
    ScalarRing scalars;

    /* series */
    SeriesStore *series;
    int series_n;
    int series_cap;

    /* graph */
    GraphNode *nodes;
    int n_nodes;
    int nodes_cap;

    /* embeddings */
    EmbeddingStore emb[EMB_CAP];
    int n_emb;

    /* samples (ring, most recent) */
    SampleStore *samples;
    int n_samples;
    int samples_cap;

    /* histograms */
    HistStore *hists;
    int n_hists;
    int hists_cap;

    /* timeline */
    TimelineEvent *timeline;
    int n_timeline;
    int timeline_cap;

    /* sweeps */
    SweepResult *sweeps;
    int n_sweeps;
    int sweeps_cap;

    /* server state */
    snepx_socket_t sock;
    snepx_thread_t thread;
    volatile int running;
    int m_port;
    volatile int ws_clients[SNEPPX_VIZMON_MAX_WS_CLIENTS];
    int n_ws_clients;
};

/* ---- engine lifecycle ---- */
SNEPPX_VIZMON_API SNEPPX_VizMon *SNEPPX_vizmon_create(void) {
    SNEPPX_VizMon *m = (SNEPPX_VizMon*)calloc(1, sizeof(SNEPPX_VizMon));
    if (!m) return NULL;
    snepx_mutex_init(&m->lock);
    m->scalars.cap = 1024;
    m->scalars.buf = (SNEPPX_VizScalar*)calloc(m->scalars.cap, sizeof(SNEPPX_VizScalar));
    m->sock = SNEPPX_VIZMON_INVALID_SOCK;
    m->running = 0;
    return m;
}

static void series_free(SNEPPX_VizMon *m) {
    if (m->series) {
        for (int i = 0; i < m->series_n; i++) {
            free(m->series[i].values);
            free(m->series[i].times);
        }
        free(m->series);
    }
}
static void nodes_free(SNEPPX_VizMon *m) { free(m->nodes); }
static void emb_free(SNEPPX_VizMon *m) {
    for (int i = 0; i < EMB_CAP; i++) {
        free(m->emb[i].data);
        free(m->emb[i].proj);
    }
}
static void samples_free(SNEPPX_VizMon *m) {
    if (m->samples) {
        for (int i = 0; i < m->n_samples; i++) free(m->samples[i].bytes);
        free(m->samples);
    }
}
static void hists_free(SNEPPX_VizMon *m) {
    if (m->hists) {
        for (int i = 0; i < m->n_hists; i++) {
            free(m->hists[i].edges);
            free(m->hists[i].counts);
        }
        free(m->hists);
    }
}
static void timeline_free(SNEPPX_VizMon *m) { free(m->timeline); }
static void sweeps_free(SNEPPX_VizMon *m) { free(m->sweeps); }

SNEPPX_VIZMON_API void SNEPPX_vizmon_destroy(SNEPPX_VizMon *m) {
    if (!m) return;
    SNEPPX_vizmon_stop(m);
    snepx_mutex_lock(&m->lock);
    free(m->scalars.buf);
    series_free(m); nodes_free(m); emb_free(m); samples_free(m);
    hists_free(m); timeline_free(m); sweeps_free(m);
    snepx_mutex_unlock(&m->lock);
    snepx_mutex_destroy(&m->lock);
    free(m);
}

/* ---- scalars ---- */
SNEPPX_VIZMON_API int SNEPPX_vizmon_push_scalar(SNEPPX_VizMon *m, const char *tag, double value, double ts) {
    if (!m || !tag) return -1;
    snepx_mutex_lock(&m->lock);
    ring_push(&m->scalars, tag, value, ts);
    /* also append to an append-only series */
    SeriesStore *e = series_get_or_create(&m->series, &m->series_n, &m->series_cap, tag);
    if (e) series_push(e, value, ts);
    snepx_mutex_unlock(&m->lock);
    return 0;
}

SNEPPX_VIZMON_API int SNEPPX_vizmon_get(SNEPPX_VizMon *m, const char *tag, SNEPPX_VizSeries *out_series) {
    if (!m || !tag || !out_series) return -1;
    snepx_mutex_lock(&m->lock);
    int idx = series_find(m->series, m->series_n, tag);
    if (idx < 0) { snepx_mutex_unlock(&m->lock); return -1; }
    SeriesStore *e = &m->series[idx];
    int n = e->count < SNEPPX_VIZMON_MAX_SAMPLES ? e->count : SNEPPX_VIZMON_MAX_SAMPLES;
    out_series->count = n;
    /* caller gets a copy to a freshly allocated buffer (owned by caller) */
    out_series->values = (double*)calloc(n, sizeof(double));
    out_series->times  = (double*)calloc(n, sizeof(double));
    if (!out_series->values || !out_series->times) {
        free(out_series->values); free(out_series->times);
        out_series->values = NULL; out_series->times = NULL;
        snepx_mutex_unlock(&m->lock); return -1;
    }
    memcpy(out_series->values, e->values, (size_t)n * sizeof(double));
    memcpy(out_series->times,  e->times,  (size_t)n * sizeof(double));
    strncpy(out_series->tag, e->tag, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    out_series->tag[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    snepx_mutex_unlock(&m->lock);
    return 0;
}

/* ---- graph ---- */
SNEPPX_VIZMON_API int SNEPPX_vizmon_add_node(SNEPPX_VizMon *m, const char *name, const int *in_nodes,
                           int num_in, int out_node, const double *shape, int nshape) {
    if (!m || !name || num_in < 0 || out_node < 0) return -1;
    if (num_in > SNEPPX_VIZMON_MAX_SCRATCH) num_in = SNEPPX_VIZMON_MAX_SCRATCH;
    if (nshape > SNEPPX_VIZMON_MAX_SCRATCH) nshape = SNEPPX_VIZMON_MAX_SCRATCH;
    snepx_mutex_lock(&m->lock);
    if (m->n_nodes >= m->nodes_cap) {
        int nc = m->nodes_cap ? m->nodes_cap * 2 : 16;
        GraphNode *nn = (GraphNode*)realloc(m->nodes, nc * sizeof(GraphNode));
        if (!nn) { snepx_mutex_unlock(&m->lock); return -1; }
        m->nodes = nn; m->nodes_cap = nc;
    }
    GraphNode *n = &m->nodes[m->n_nodes];
    strncpy(n->name, name, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    n->name[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    n->num_in = num_in;
    for (int i = 0; i < num_in; i++) n->in_nodes[i] = in_nodes[i];
    n->out_node = out_node;
    n->nshape = nshape;
    for (int i = 0; i < nshape; i++) n->shape[i] = shape[i];
    m->n_nodes++;
    snepx_mutex_unlock(&m->lock);
    return 0;
}

SNEPPX_VIZMON_API int SNEPPX_vizmon_get_graph(SNEPPX_VizMon *m, SNEPPX_VizNode *out, int *n_nodes) {
    if (!m || !out || !n_nodes) return -1;
    snepx_mutex_lock(&m->lock);
    int k = m->n_nodes < SNEPPX_VIZMON_MAX_NODES ? m->n_nodes : SNEPPX_VIZMON_MAX_NODES;
    *n_nodes = k;
    for (int i = 0; i < k; i++) {
        GraphNode *src = &m->nodes[i];
        strncpy(out[i].name, src->name, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
        out[i].name[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
        out[i].num_in = src->num_in;
        for (int j = 0; j < src->num_in; j++) out[i].in_nodes[j] = src->in_nodes[j];
        out[i].out_node = src->out_node;
        out[i].nshape = src->nshape;
        for (int j = 0; j < src->nshape; j++) out[i].shape[j] = src->shape[j];
    }
    snepx_mutex_unlock(&m->lock);
    return 0;
}

/* ---- embeddings ---- */
static EmbeddingStore *emb_find(SNEPPX_VizMon *m, const char *tag) {
    for (int i = 0; i < m->n_emb; i++)
        if (m->emb[i].in_use && strcmp(m->emb[i].tag, tag) == 0) return &m->emb[i];
    return NULL;
}
static EmbeddingStore *emb_alloc(SNEPPX_VizMon *m, const char *tag, int n_dim) {
    EmbeddingStore *e = NULL;
    for (int i = 0; i < EMB_CAP; i++) if (!m->emb[i].in_use) { e = &m->emb[i]; break; }
    if (!e) return NULL;
    e->in_use = 1;
    strncpy(e->tag, tag, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    e->tag[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    e->n_dim = n_dim; e->n_samples = 0; e->projected = 0;
    return e;
}

SNEPPX_VIZMON_API int SNEPPX_vizmon_push_embedding(SNEPPX_VizMon *m, const char *tag,
                                 const double *data, int n_samples, int n_dim) {
    if (!m || !tag || !data || n_samples <= 0 || n_dim <= 0) return -1;
    snepx_mutex_lock(&m->lock);
    EmbeddingStore *e = emb_find(m, tag);
    if (!e) {
        if (m->n_emb >= EMB_CAP) { snepx_mutex_unlock(&m->lock); return -1; }
        e = emb_alloc(m, tag, n_dim);
        m->n_emb++;
    } else {
        free(e->data); free(e->proj);
    }
    e->data = (double*)calloc((size_t)n_samples * n_dim, sizeof(double));
    e->n_samples = n_samples; e->n_dim = n_dim; e->projected = 0;
    if (!e->data) { snepx_mutex_unlock(&m->lock); return -1; }
    memcpy(e->data, data, (size_t)n_samples * n_dim * sizeof(double));
    snepx_mutex_unlock(&m->lock);
    return 0;
}

static void pca_project(EmbeddingStore *e) {
    /* Center data along each dimension. */
    int n = e->n_samples, d = e->n_dim;
    double *mean = (double*)calloc(d, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) mean[j] += e->data[i * d + j];
    for (int j = 0; j < d; j++) mean[j] /= n;
    double *X = (double*)calloc((size_t)n * d, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) X[i * d + j] = e->data[i * d + j] - mean[j];
    free(mean);
    /* Covariance = (X^T X)/n, power iteration for top-2 eigenvectors. */
    double *cov = (double*)calloc((size_t)d * d, sizeof(double));
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            double s = 0;
            for (int k = 0; k < n; k++) s += X[k * d + i] * X[k * d + j];
            cov[i * d + j] = s / n;
        }
    /* power iteration */
    double *v1 = (double*)calloc(d, sizeof(double)); v1[0] = 1.0;
    for (int it = 0; it < 100; it++) {
        double *nv = (double*)calloc(d, sizeof(double));
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) nv[i] += cov[i * d + j] * v1[j];
        double norm = 0; for (int i = 0; i < d; i++) norm += nv[i] * nv[i];
        norm = sqrt(norm); if (norm < 1e-12) { free(nv); break; }
        for (int i = 0; i < d; i++) v1[i] = nv[i] / norm;
        free(nv);
    }
    /* deflate */
    double *v2 = (double*)calloc(d, sizeof(double)); v2[1 % d] = 1.0;
    for (int it = 0; it < 100; it++) {
        double *nv = (double*)calloc(d, sizeof(double));
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) nv[i] += cov[i * d + j] * v2[j];
        /* remove v1 component */
        double dot = 0; for (int i = 0; i < d; i++) dot += v1[i] * nv[i];
        for (int i = 0; i < d; i++) nv[i] -= dot * v1[i];
        double norm = 0; for (int i = 0; i < d; i++) norm += nv[i] * nv[i];
        norm = sqrt(norm); if (norm < 1e-12) { free(nv); break; }
        for (int i = 0; i < d; i++) v2[i] = nv[i] / norm;
        free(nv);
    }
    e->proj = (double*)calloc((size_t)n * 2, sizeof(double));
    for (int i = 0; i < n; i++) {
        double p1 = 0, p2 = 0;
        for (int j = 0; j < d; j++) { p1 += X[i * d + j] * v1[j]; p2 += X[i * d + j] * v2[j]; }
        e->proj[i * 2] = p1; e->proj[i * 2 + 1] = p2;
    }
    e->projected = 1;
    free(v1); free(v2); free(X); free(cov);
}

SNEPPX_VIZMON_API int SNEPPX_vizmon_project_pca(SNEPPX_VizMon *m, int max_components) {
    (void)max_components;
    if (!m) return -1;
    snepx_mutex_lock(&m->lock);
    for (int i = 0; i < m->n_emb; i++)
        if (m->emb[i].in_use && !m->emb[i].projected) pca_project(&m->emb[i]);
    snepx_mutex_unlock(&m->lock);
    return 0;
}

SNEPPX_VIZMON_API SNEPPX_VizEmbedding *SNEPPX_vizmon_get_embedding(SNEPPX_VizMon *m, const char *tag) {
    if (!m || !tag) return NULL;
    snepx_mutex_lock(&m->lock);
    EmbeddingStore *e = emb_find(m, tag);
    if (!e) { snepx_mutex_unlock(&m->lock); return NULL; }
    SNEPPX_VizEmbedding *out = (SNEPPX_VizEmbedding*)calloc(1, sizeof(SNEPPX_VizEmbedding));
    if (!out) { snepx_mutex_unlock(&m->lock); return NULL; }
    out->data = (double*)calloc((size_t)e->n_samples * e->n_dim, sizeof(double));
    if (e->projected && e->proj) {
        out->proj = (double*)calloc((size_t)e->n_samples * 2, sizeof(double));
        memcpy(out->proj, e->proj, (size_t)e->n_samples * 2 * sizeof(double));
        out->projected = 1;
    }
    memcpy(out->data, e->data, (size_t)e->n_samples * e->n_dim * sizeof(double));
    out->n_samples = e->n_samples; out->n_dim = e->n_dim; out->projected = e->projected;
    snepx_mutex_unlock(&m->lock);
    return out;
}

/* ---- samples ---- */
SNEPPX_VIZMON_API int SNEPPX_vizmon_push_sample(SNEPPX_VizMon *m, const char *tag,
                              SNEPPX_VizSampleKind kind, const unsigned char *bytes,
                              size_t len, double ts) {
    if (!m || !tag || !bytes || kind > SNEPPX_VIZMON_AUD_WAV) return -1;
    /* cap sample size */
    if (len > 4 * 1024 * 1024) len = 4 * 1024 * 1024;
    snepx_mutex_lock(&m->lock);
    if (m->n_samples >= m->samples_cap) {
        int nc = m->samples_cap ? m->samples_cap * 2 : 16;
        SampleStore *ns = (SampleStore*)realloc(m->samples, nc * sizeof(SampleStore));
        if (!ns) { snepx_mutex_unlock(&m->lock); return -1; }
        m->samples = ns; m->samples_cap = nc;
    }
    SampleStore *s = &m->samples[m->n_samples];
    strncpy(s->tag, tag, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    s->tag[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    s->kind = kind; s->ts = ts;
    s->bytes = (unsigned char*)calloc(len, 1);
    if (!s->bytes) { snepx_mutex_unlock(&m->lock); return -1; }
    memcpy(s->bytes, bytes, len);
    s->len = len;
    m->n_samples++;
    /* keep only most recent SNEPPX_VIZMON_MAX_SAMPLES */
    if (m->n_samples > SNEPPX_VIZMON_MAX_SAMPLES) {
        size_t keep = SNEPPX_VIZMON_MAX_SAMPLES;
        size_t drop = m->n_samples - keep;
        for (size_t i = 0; i < drop; i++) free(m->samples[i].bytes);
        memmove(m->samples, m->samples + drop, keep * sizeof(SampleStore));
        m->n_samples = (int)keep;
    }
    snepx_mutex_unlock(&m->lock);
    return 0;
}

SNEPPX_VIZMON_API int SNEPPX_vizmon_get_samples(SNEPPX_VizMon *m, const char *tag, SNEPPX_VizSample *out, int *n) {
    if (!m || !tag || !out || !n) return -1;
    snepx_mutex_lock(&m->lock);
    int k = 0;
    for (int i = 0; i < m->n_samples; i++) {
        if (strcmp(m->samples[i].tag, tag) == 0) {
            if (k >= SNEPPX_VIZMON_MAX_SAMPLES) break;
            out[k].kind = m->samples[i].kind;
            out[k].len = m->samples[i].len;
            out[k].ts = m->samples[i].ts;
            out[k].bytes = (unsigned char*)calloc(m->samples[i].len, 1);
            if (out[k].bytes && m->samples[i].bytes)
                memcpy(out[k].bytes, m->samples[i].bytes, m->samples[i].len);
            k++;
        }
    }
    *n = k;
    snepx_mutex_unlock(&m->lock);
    return 0;
}

/* ---- histograms ---- */
static HistStore *hist_find(SNEPPX_VizMon *m, const char *tag) {
    for (int i = 0; i < m->n_hists; i++)
        if (strcmp(m->hists[i].tag, tag) == 0) return &m->hists[i];
    return NULL;
}
static int hist_push(SNEPPX_VizMon *m, const char *tag, const double *values, int n, int n_bins, double ts) {
    if (n_bins > SNEPPX_VIZMON_MAX_HIST_BINS) n_bins = SNEPPX_VIZMON_MAX_HIST_BINS;
    if (n_bins < 1) n_bins = 1;
    int idx = -1;
    for (int i = 0; i < m->n_hists; i++) if (strcmp(m->hists[i].tag, tag) == 0) { idx = i; break; }
    if (idx < 0) {
        if (m->n_hists >= m->hists_cap) {
            int nc = m->hists_cap ? m->hists_cap * 2 : 8;
            HistStore *ns = (HistStore*)realloc(m->hists, nc * sizeof(HistStore));
            if (!ns) return -1;
            m->hists = ns; m->hists_cap = nc;
        }
        idx = m->n_hists++;
        memset(&m->hists[idx], 0, sizeof(HistStore));
    }
    HistStore *h = &m->hists[idx];
    free(h->edges); free(h->counts);
    strncpy(h->tag, tag, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    h->tag[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    h->n_bins = n_bins; h->ts = ts;
    h->edges = (double*)calloc(n_bins + 1, sizeof(double));
    h->counts = (int*)calloc(n_bins, sizeof(int));
    if (!h->edges || !h->counts) return -1;
    double lo = values[0], hi = values[0];
    for (int i = 1; i < n; i++) { if (values[i] < lo) lo = values[i]; if (values[i] > hi) hi = values[i]; }
    if (hi <= lo) hi = lo + 1.0;
    double w = (hi - lo) / n_bins;
    for (int i = 0; i <= n_bins; i++) h->edges[i] = lo + i * w;
    for (int i = 0; i < n; i++) {
        int b = (int)((values[i] - lo) / w);
        if (b < 0) b = 0; if (b >= n_bins) b = n_bins - 1;
        h->counts[b]++;
    }
    return 0;
}
SNEPPX_VIZMON_API int SNEPPX_vizmon_push_histogram(SNEPPX_VizMon *m, const char *tag, const double *values, int n, int n_bins, double ts) {
    if (!m || !tag || !values || n <= 0) return -1;
    snepx_mutex_lock(&m->lock);
    int r = hist_push(m, tag, values, n, n_bins, ts);
    snepx_mutex_unlock(&m->lock);
    return r;
}
SNEPPX_VIZMON_API int SNEPPX_vizmon_get_histogram(SNEPPX_VizMon *m, const char *tag, SNEPPX_VizHistogram *out) {
    if (!m || !tag || !out) return -1;
    snepx_mutex_lock(&m->lock);
    HistStore *h = hist_find(m, tag);
    if (!h) { snepx_mutex_unlock(&m->lock); return -1; }
    out->ts = h->ts; out->n_bins = h->n_bins;
    out->edges = (double*)calloc(h->n_bins + 1, sizeof(double));
    out->counts = (int*)calloc(h->n_bins, sizeof(int));
    if (h->edges) memcpy(out->edges, h->edges, (h->n_bins + 1) * sizeof(double));
    if (h->counts) memcpy(out->counts, h->counts, h->n_bins * sizeof(int));
    snepx_mutex_unlock(&m->lock);
    return 0;
}

/* ---- timeline ---- */
SNEPPX_VIZMON_API int SNEPPX_vizmon_push_timeline(SNEPPX_VizMon *m, const char *name, double start_ms, double dur_ms, size_t mem_bytes) {
    if (!m || !name) return -1;
    snepx_mutex_lock(&m->lock);
    if (m->n_timeline >= m->timeline_cap) {
        int nc = m->timeline_cap ? m->timeline_cap * 2 : 32;
        TimelineEvent *ns = (TimelineEvent*)realloc(m->timeline, nc * sizeof(TimelineEvent));
        if (!ns) { snepx_mutex_unlock(&m->lock); return -1; }
        m->timeline = ns; m->timeline_cap = nc;
    }
    TimelineEvent *e = &m->timeline[m->n_timeline++];
    strncpy(e->name, name, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    e->name[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    e->start_ms = start_ms; e->dur_ms = dur_ms; e->mem_bytes = mem_bytes;
    if (m->n_timeline > SNEPPX_VIZMON_MAX_TIMELINE) {
        memmove(m->timeline, m->timeline + 1, (SNEPPX_VIZMON_MAX_TIMELINE - 1) * sizeof(TimelineEvent));
        m->n_timeline = SNEPPX_VIZMON_MAX_TIMELINE - 1;
    }
    snepx_mutex_unlock(&m->lock);
    return 0;
}
SNEPPX_VIZMON_API int SNEPPX_vizmon_get_timeline(SNEPPX_VizMon *m, SNEPPX_VizTimelineEvent *out, int *n) {
    if (!m || !out || !n) return -1;
    snepx_mutex_lock(&m->lock);
    int k = m->n_timeline < SNEPPX_VIZMON_MAX_TIMELINE ? m->n_timeline : SNEPPX_VIZMON_MAX_TIMELINE;
    *n = k;
    for (int i = 0; i < k; i++) {
        strncpy(out[i].name, m->timeline[i].name, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
        out[i].name[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
        out[i].start_ms = m->timeline[i].start_ms;
        out[i].dur_ms = m->timeline[i].dur_ms;
        out[i].mem_bytes = m->timeline[i].mem_bytes;
    }
    snepx_mutex_unlock(&m->lock);
    return 0;
}

/* ---- sweeps ---- */
SNEPPX_VIZMON_API int SNEPPX_vizmon_push_sweep(SNEPPX_VizMon *m, const char *config_id, double final_loss, double final_acc, double duration_s) {
    if (!m || !config_id) return -1;
    snepx_mutex_lock(&m->lock);
    if (m->n_sweeps >= m->sweeps_cap) {
        int nc = m->sweeps_cap ? m->sweeps_cap * 2 : 8;
        SweepResult *ns = (SweepResult*)realloc(m->sweeps, nc * sizeof(SweepResult));
        if (!ns) { snepx_mutex_unlock(&m->lock); return -1; }
        m->sweeps = ns; m->sweeps_cap = nc;
    }
    SweepResult *s = &m->sweeps[m->n_sweeps++];
    strncpy(s->config_id, config_id, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
    s->config_id[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
    s->final_loss = final_loss; s->final_acc = final_acc; s->duration_s = duration_s;
    snepx_mutex_unlock(&m->lock);
    return 0;
}
SNEPPX_VIZMON_API int SNEPPX_vizmon_get_sweeps(SNEPPX_VizMon *m, SNEPPX_VizSweepResult *out, int *n) {
    if (!m || !out || !n) return -1;
    snepx_mutex_lock(&m->lock);
    int k = m->n_sweeps < SNEPPX_VIZMON_MAX_SWEEPS ? m->n_sweeps : SNEPPX_VIZMON_MAX_SWEEPS;
    *n = k;
    for (int i = 0; i < k; i++) {
        strncpy(out[i].config_id, m->sweeps[i].config_id, SNEPPX_VIZMON_MAX_TAG_LEN - 1);
        out[i].config_id[SNEPPX_VIZMON_MAX_TAG_LEN - 1] = '\0';
        out[i].final_loss = m->sweeps[i].final_loss;
        out[i].final_acc = m->sweeps[i].final_acc;
        out[i].duration_s = m->sweeps[i].duration_s;
    }
    snepx_mutex_unlock(&m->lock);
    return 0;
}

/* ---- JSON emitter ---- */
static void json_escape(char *dst, const char *src, size_t cap) {
    size_t n = 0;
    for (const char *p = src; *p && n + 2 < cap; p++) {
        char c = *p;
        if (c == '"') { memcpy(dst + n, "\\\"", 2); n += 2; }
        else if (c == '\\') { memcpy(dst + n, "\\\\", 2); n += 2; }
        else if (c == '\n') { memcpy(dst + n, "\\n", 2); n += 2; }
        else if (c == '\r') { memcpy(dst + n, "\\r", 2); n += 2; }
        else if (c == '\t') { memcpy(dst + n, "\\t", 2); n += 2; }
        else dst[n++] = c;
    }
    dst[n] = '\0';
}

static char *sb_init(void) {
    char *buf = (char*)calloc(16384, 1);
    return buf; /* capacity tracked by realloc growth */
}
typedef struct { char *buf; size_t len; size_t cap; } SBuf;
static void sb_init2(SBuf *s, size_t cap) {
    s->buf = (char*)malloc(cap); s->cap = cap; s->len = 0; if (s->buf) s->buf[0] = '\0';
}
static void sb_append(SBuf *s, const char *fmt, ...) {
    va_list ap;
    for (;;) {
        va_start(ap, fmt);
        int need = vsnprintf(s->buf + s->len, s->cap - s->len, fmt, ap);
        va_end(ap);
        if (need < 0) return;
        if ((size_t)need < s->cap - s->len) { s->len += (size_t)need; return; }
        s->cap *= 2;
        char *nb = (char*)realloc(s->buf, s->cap);
        if (!nb) { s->len = 0; return; }
        s->buf = nb;
    }
}
static void sb_append_raw(SBuf *s, const char *data, size_t len) {
    while (s->len + len >= s->cap) { s->cap *= 2; char *nb = (char*)realloc(s->buf, s->cap); if (!nb) return; s->buf = nb; }
    memcpy(s->buf + s->len, data, len); s->len += len;
    s->buf[s->len] = '\0';
}
static int append_scalar_array(SBuf *s, ScalarRing *r) {
    SNEPPX_VizScalar *snap = (SNEPPX_VizScalar*)calloc(r->cap, sizeof(SNEPPX_VizScalar));
    int n; ring_snapshot(r, snap, &n, r->cap);
    sb_append(s, "\"scalars\":[");
    for (int i = 0; i < n; i++) {
        char esc[256]; json_escape(esc, snap[i].tag, sizeof(esc));
        sb_append(s, "%s{\"tag\":\"%s\",\"value\":%.6g,\"ts\":%.6g}", i ? "," : "", esc, snap[i].value, snap[i].ts);
    }
    sb_append(s, "],\"series\":{");
    return n > 0 ? 1 : 0; /* caller: close series object */
}
static void append_series_obj(SBuf *s, SNEPPX_VizMon *m) {
    sb_append(s, "}");
    (void)m;
}

SNEPPX_VIZMON_API char *SNEPPX_vizmon_snapshot_json(SNEPPX_VizMon *m) {
    if (!m) return NULL;
    SBuf s; sb_init2(&s, 32768);
    sb_append(&s, "{");
    snepx_mutex_lock(&m->lock);
    /* scalars (recent) + series (full) */
    SNEPPX_VizScalar *snap = (SNEPPX_VizScalar*)calloc(m->scalars.cap, sizeof(SNEPPX_VizScalar));
    int ns; ring_snapshot(&m->scalars, snap, &ns, m->scalars.cap);
    sb_append(&s, "\"scalars\":[");
    for (int i = 0; i < ns; i++) {
        char esc[256]; json_escape(esc, snap[i].tag, sizeof(esc));
        sb_append(&s, "%s{\"tag\":\"%s\",\"value\":%.6g,\"ts\":%.6g}", i ? "," : "", esc, snap[i].value, snap[i].ts);
    }
    sb_append(&s, "],\"series\":{");
    for (int i = 0; i < m->series_n; i++) {
        SeriesStore *e = &m->series[i];
        char esc[256]; json_escape(esc, e->tag, sizeof(esc));
        sb_append(&s, "%s\"%s\":[", i ? "," : "", esc);
        for (int j = 0; j < e->count; j++)
            sb_append(&s, "%s[%.6g,%.6g]", j ? "," : "", e->values[j], e->times[j]);
        sb_append(&s, "]");
    }
    sb_append(&s, "},");
    free(snap);
    /* graph */
    sb_append(&s, "\"graph\":[");
    for (int i = 0; i < m->n_nodes; i++) {
        GraphNode *n = &m->nodes[i];
        char esc[256]; json_escape(esc, n->name, sizeof(esc));
        sb_append(&s, "%s{\"name\":\"%s\",\"ins\":[", i ? "," : "", esc);
        for (int j = 0; j < n->num_in; j++) sb_append(&s, "%s%d", j ? "," : "", n->in_nodes[j]);
        sb_append(&s, "],\"out\":%d,\"shape\":[", n->out_node);
        for (int j = 0; j < n->nshape; j++) sb_append(&s, "%s%.0f", j ? "," : "", n->shape[j]);
        sb_append(&s, "]}");
    }
    sb_append(&s, "],");
    /* embeddings */
    sb_append(&s, "\"embeddings\":{");
    for (int i = 0; i < m->n_emb; i++) {
        if (!m->emb[i].in_use) continue;
        char esc[256]; json_escape(esc, m->emb[i].tag, sizeof(esc));
        sb_append(&s, "\"%s\":{\"n_samples\":%d,\"n_dim\":%d,\"projected\":%d", esc, m->emb[i].n_samples, m->emb[i].n_dim, m->emb[i].projected);
        if (m->emb[i].projected && m->emb[i].proj) {
            sb_append(&s, ",\"points\":[");
            int n = m->emb[i].n_samples;
            for (int j = 0; j < n; j++)
                sb_append(&s, "%s[%.6g,%.6g]", j ? "," : "", m->emb[i].proj[j*2], m->emb[i].proj[j*2+1]);
            sb_append(&s, "]");
        }
        sb_append(&s, "}");
    }
    sb_append(&s, "},");
    /* samples */
    sb_append(&s, "\"samples\":[");
    for (int i = 0; i < m->n_samples; i++) {
        SampleStore *s_ = &m->samples[i];
        char esc[256]; json_escape(esc, s_->tag, sizeof(esc));
        sb_append(&s, "%s{\"tag\":\"%s\",\"kind\":%d,\"len\":%zu}", i ? "," : "", esc, (int)s_->kind, (size_t)s_->len);
    }
    sb_append(&s, "],");
    /* histograms */
    sb_append(&s, "\"histograms\":{");
    for (int i = 0; i < m->n_hists; i++) {
        HistStore *h = &m->hists[i];
        char esc[256]; json_escape(esc, h->tag, sizeof(esc));
            sb_append(&s, "%s\"%s\":{\"n_bins\":%d,\"edges\":[", i ? "," : "", esc, h->n_bins);
        for (int j = 0; j <= h->n_bins; j++) sb_append(&s, "%s%.6g", j ? "," : "", h->edges[j]);
        sb_append(&s, "],\"counts\":[");
        for (int j = 0; j < h->n_bins; j++) sb_append(&s, "%s%d", j ? "," : "", h->counts[j]);
        sb_append(&s, "]}");
    }
    sb_append(&s, "},");
    /* timeline */
    sb_append(&s, "\"timeline\":[");
    for (int i = 0; i < m->n_timeline; i++) {
        TimelineEvent *e = &m->timeline[i];
        char esc[256]; json_escape(esc, e->name, sizeof(esc));
        sb_append(&s, "%s{\"name\":\"%s\",\"start_ms\":%.3f,\"dur_ms\":%.3f,\"mem_bytes\":%zu}", i ? "," : "", esc, e->start_ms, e->dur_ms, (size_t)e->mem_bytes);
    }
    sb_append(&s, "],");
    /* sweeps */
    sb_append(&s, "\"sweeps\":[");
    for (int i = 0; i < m->n_sweeps; i++) {
        SweepResult *sw = &m->sweeps[i];
        char esc[256]; json_escape(esc, sw->config_id, sizeof(esc));
        sb_append(&s, "%s{\"config_id\":\"%s\",\"final_loss\":%.6g,\"final_acc\":%.6g,\"duration_s\":%.3f}", i ? "," : "", esc, sw->final_loss, sw->final_acc, sw->duration_s);
    }
    sb_append(&s, "]");
    snepx_mutex_unlock(&m->lock);
    sb_append(&s, "}");
    return s.buf;
}

SNEPPX_VIZMON_API void SNEPPX_vizmon_free(void *ptr) { free(ptr); }

SNEPPX_VIZMON_API const char *SNEPPX_vizmon_frontend_html(void);
SNEPPX_VIZMON_API char *SNEPPX_vizmon_export_html(SNEPPX_VizMon *m);

/* ================================================================== */
/*                        HTTP + WebSocket server                     */
/* ================================================================== */

#define HTTP_BUF 65536

typedef struct {
    snepx_socket_t fd;
    int ws;
    char buf[HTTP_BUF];
    size_t len;
} ConnState;

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { *out_len = 0; return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); *out_len = 0; return NULL; }
    char *buf = (char*)calloc((size_t)sz + 1, 1);
    if (!buf) { fclose(f); *out_len = 0; return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f); fclose(f);
    *out_len = rd; return buf;
}

/* ---- base64 ---- */
static void base64_encode(const unsigned char *data, size_t len, char *out, size_t out_cap) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < len && o + 4 < out_cap; i += 3) {
        unsigned int v = (unsigned int)(data[i] << 16);
        int rem = 1;
        if (i + 1 < len) { v |= data[i+1] << 8; rem = 2; }
        if (i + 2 < len) { v |= data[i+2]; rem = 3; }
        out[o++] = tbl[(v >> 18) & 0x3f];
        out[o++] = tbl[(v >> 12) & 0x3f];
        out[o++] = rem >= 2 ? tbl[(v >> 6) & 0x3f] : '=';
        out[o++] = rem >= 3 ? tbl[v & 0x3f] : '=';
    }
    out[o] = '\0';
}

/* ---- minimal SHA-1 (no OpenSSL) for WebSocket accept key ---- */
typedef struct { uint32_t h[5]; uint64_t len; uint8_t buf[64]; uint32_t ctr; } Sha1;
static void sha1_init(Sha1 *s) {
    s->h[0]=0x67452301; s->h[1]=0xefcdab89; s->h[2]=0x98badcfe; s->h[3]=0x10325476; s->h[4]=0xc3d2e1f0;
    s->len=0; s->ctr=0; memset(s->buf,0,64);
}
static const uint32_t K[4]={0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xca62c1d6};
#define SHA1_ROL(v,n) (((v)<<(n))|((v)>>(32-(n))))
static void sha1_block(Sha1 *s, const uint8_t *p) {
    uint32_t w[80];
    for (int i=0;i<16;i++) w[i]=((uint32_t)p[i*4]<<24)|((uint32_t)p[i*4+1]<<16)|((uint32_t)p[i*4+2]<<8)|((uint32_t)p[i*4+3]);
    for (int i=16;i<80;i++) w[i]=SHA1_ROL(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4];
    for (int i=0;i<80;i++){
        uint32_t f; int k;
        if(i<20){f=(b&c)|((~b)&d);k=0;}
        else if(i<40){f=b^c^d;k=1;}
        else if(i<60){f=(b&c)|(b&d)|(c&d);k=2;}
        else{f=b^c^d;k=3;}
        uint32_t temp=SHA1_ROL(a,5)+f+e+K[k]+w[i];
        e=d;d=c;c=SHA1_ROL(b,30);b=a;a=temp;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;
}
static void sha1_update(Sha1 *s, const uint8_t *data, size_t len) {
    for (size_t i=0;i<len;i++){
        s->buf[s->ctr++]=data[i]; s->len++;
        if(s->ctr==64){ sha1_block(s,s->buf); s->ctr=0; }
    }
}
static void sha1_final(Sha1 *s, uint8_t out[20]) {
    s->buf[s->ctr++]=0x80;
    if(s->ctr>56){ while(s->ctr<64) s->buf[s->ctr++]=0; sha1_block(s,s->buf); s->ctr=0; }
    while(s->ctr<56) s->buf[s->ctr++]=0;
    uint64_t bitlen=s->len*8;
    for(int i=0;i<8;i++) s->buf[s->ctr++]=(uint8_t)(bitlen>>(56-i*8));
    sha1_block(s,s->buf);
    for(int i=0;i<5;i++){ out[i*4]=(s->h[i]>>24)&0xff; out[i*4+1]=(s->h[i]>>16)&0xff; out[i*4+2]=(s->h[i]>>8)&0xff; out[i*4+3]=s->h[i]&0xff; }
}

static void http_write_all(snepx_socket_t fd, const char *data, size_t len) {
    size_t off = 0;
    while (off < len) {
        int n = (int)send(fd, data + off, (int)(len - off), 0);
        if (n <= 0) return;
        off += (size_t)n;
    }
}

static int ws_send_text(snepx_socket_t fd, const char *msg, size_t len) {
    unsigned char hdr[10];
    size_t pos = 0;
    int ilen = (int)len;
    hdr[pos++] = 0x81;
    if (ilen <= 125) { hdr[pos++] = (unsigned char)(0x80 | ilen); }
    else if (ilen <= 65535) { hdr[pos++] = 0x80 | 126; hdr[pos++] = (unsigned char)((ilen >> 8) & 0xff); hdr[pos++] = (unsigned char)(ilen & 0xff); }
    else { hdr[pos++] = 0x80 | 127; for (int i = 7; i >= 0; i--) hdr[pos++] = (unsigned char)((len >> (i * 8)) & 0xff); }
    if (send(fd, (const char*)hdr, (int)pos, 0) != (int)pos) return -1;
    if (send(fd, msg, ilen, 0) != ilen) return -1;
    return 0;
}

typedef struct { SNEPPX_VizMon *m; snepx_socket_t client_fd; } AcceptArg;

static void handle_one_http(SNEPPX_VizMon *m, snepx_socket_t client_fd) {
    char *req = (char*)calloc(HTTP_BUF, 1);
    if (!req) { snepx_close_sock(client_fd); return; }
    int n = (int)recv(client_fd, req, HTTP_BUF - 1, 0);
    if (n <= 0) { free(req); snepx_close_sock(client_fd); return; }
    req[n] = '\0';

    char method[16] = {0}, path[1024] = {0};
    sscanf(req, "%15s %1023s", method, path);
    (void)method;

    /* WebSocket upgrade on /ws */
    if (strstr(path, "/ws") == path || strcmp(path, "/ws") == 0) {
        const char *hdr = strstr(req, "Sec-WebSocket-Key:");
        char key[128] = {0};
        if (hdr) { sscanf(hdr, "Sec-WebSocket-Key: %127[^\r\n]", key); }
        char accept_key[128] = "s3pPLMBiTxaQ9kV2SG40gN3pTkI=";
        if (strlen(key) > 0) {
            char combined[256];
            snprintf(combined, sizeof(combined), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key);
            Sha1 s; sha1_init(&s);
            sha1_update(&s, (const uint8_t*)combined, strlen(combined));
            uint8_t h[20]; sha1_final(&s, h);
            base64_encode(h, 20, accept_key, sizeof(accept_key));
        }
        char resp[512];
        int rl = snprintf(resp, sizeof(resp),
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Accept: %s\r\n\r\n", accept_key);
        http_write_all(client_fd, resp, (size_t)rl);

        snepx_mutex_lock(&m->lock);
        if (m->n_ws_clients < SNEPPX_VIZMON_MAX_WS_CLIENTS)
            m->ws_clients[m->n_ws_clients++] = (int)client_fd;
        snepx_mutex_unlock(&m->lock);

        for (int i = 0; i < 600 && m->running; i++) {
            char *snap = SNEPPX_vizmon_snapshot_json(m);
            if (snap) {
                if (ws_send_text(client_fd, snap, strlen(snap)) != 0) { free(snap); break; }
                free(snap);
            }
            snepx_sleep_ms(500);
        }
        snepx_close_sock(client_fd);
        snepx_mutex_lock(&m->lock);
        for (int i = 0; i < m->n_ws_clients; i++)
            if (m->ws_clients[i] == (int)client_fd) { m->ws_clients[i] = -1; break; }
        snepx_mutex_unlock(&m->lock);
        free(req);
        return;
    }

    if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
        const char *html = SNEPPX_vizmon_frontend_html();
        http_write_all(client_fd, html, strlen(html));
        snepx_close_sock(client_fd); free(req); return;
    }
    if (strcmp(path, "/snapshot") == 0) {
        char *json = SNEPPX_vizmon_snapshot_json(m);
        char hdr[256];
        int hl = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        http_write_all(client_fd, hdr, (size_t)hl);
        if (json) {
            char ln[32]; snprintf(ln, sizeof(ln), "%zu\r\n", strlen(json) + 2);
            /* simpler: no chunked; send body then close */
            http_write_all(client_fd, json, strlen(json));
            free(json);
        }
        snepx_close_sock(client_fd); free(req); return;
    }
    if (strcmp(path, "/export") == 0) {
        char *html = SNEPPX_vizmon_export_html(m);
        char hdr[256];
        int hl = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        http_write_all(client_fd, hdr, (size_t)hl);
        if (html) { http_write_all(client_fd, html, strlen(html)); free(html); }
        snepx_close_sock(client_fd); free(req); return;
    }
    const char *body = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found";
    http_write_all(client_fd, body, strlen(body));
    snepx_close_sock(client_fd); free(req);
}

static
#ifdef _WIN32
DWORD WINAPI
#else
void*
#endif
accept_thread(void *arg) {
    AcceptArg *a = (AcceptArg*)arg;
    handle_one_http(a->m, a->client_fd);
    free(a);
    return
#ifdef _WIN32
    0;
#else
    NULL;
#endif
}

static
#ifdef _WIN32
DWORD WINAPI
#else
void*
#endif
listen_thread(void *arg) {
    SNEPPX_VizMon *m = (SNEPPX_VizMon*)arg;
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
#endif
    m->sock = socket(AF_INET, SOCK_STREAM, 0);
    if (m->sock == SNEPPX_VIZMON_INVALID_SOCK) return
#ifdef _WIN32
    0;
#else
    NULL;
#endif
    snepx_set_reuseaddr(m->sock);
    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)m->m_port);
    if (bind(m->sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        snepx_close_sock(m->sock); m->sock = SNEPPX_VIZMON_INVALID_SOCK;
        return
#ifdef _WIN32
        0;
#else
        NULL;
#endif
    }
    if (listen(m->sock, 16) != 0) {
        snepx_close_sock(m->sock); m->sock = SNEPPX_VIZMON_INVALID_SOCK;
        return
#ifdef _WIN32
        0;
#else
        NULL;
#endif
    }
    /* non-blocking listen so the accept loop can poll m->running */
#ifdef _WIN32
    { u_long mode = 1; ioctlsocket(m->sock, FIONBIO, &mode); }
#else
    { int flags = fcntl(m->sock, F_GETFL, 0); if (flags >= 0) fcntl(m->sock, F_SETFL, flags | O_NONBLOCK); }
#endif
    while (m->running) {
        struct sockaddr_in caddr; int clen = sizeof(caddr);
        snepx_socket_t cfd = accept(m->sock, (struct sockaddr*)&caddr, &clen);
        if (cfd == SNEPPX_VIZMON_INVALID_SOCK) {
#ifdef _WIN32
            if (WSAGetLastError() == WSAEWOULDBLOCK) { snepx_sleep_ms(10); continue; }
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK) { snepx_sleep_ms(10); continue; }
#endif
            if (!m->running) break;
            snepx_sleep_ms(10);
            continue;
        }
        AcceptArg *a = (AcceptArg*)calloc(1, sizeof(AcceptArg));
        if (!a) { snepx_close_sock(cfd); continue; }
        a->m = m; a->client_fd = cfd;
        snepx_thread_t th = snepx_thread_create(accept_thread, a);
        if (!th) { snepx_close_sock(cfd); free(a); }
    }
    return
#ifdef _WIN32
    0;
#else
    NULL;
#endif
}

SNEPPX_VIZMON_API int SNEPPX_vizmon_start(SNEPPX_VizMon *m, int port) {
    if (!m || m->running) return -1;
    m->running = 1;
    m->m_port = port;
    m->thread = snepx_thread_create(listen_thread, m);
    if (!m->thread) { m->running = 0; return -1; }
    return 0;
}

SNEPPX_VIZMON_API void SNEPPX_vizmon_stop(SNEPPX_VizMon *m) {
    if (!m) return;
    m->running = 0;
    if (m->sock != SNEPPX_VIZMON_INVALID_SOCK) {
        snepx_close_sock(m->sock);
        m->sock = SNEPPX_VIZMON_INVALID_SOCK;
    }
    if (m->thread) { snepx_thread_join(m->thread); m->thread = 0; }
}

/* ---- inline Vue 3 frontend (CDN, no build) ---- */
static const char FRONTEND_HTML[] =
"<!doctype html><html><head><meta charset='utf-8'>"
"<meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>SNEPPX vizmon</title>"
"<style>"
"body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee;font-size:14px}"
".grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;height:calc(100vh-24px);box-sizing:border-box}"
".card{background:#1f1f2e;border-radius:8px;padding:12px;overflow:auto;display:flex;flex-direction:column}"
".card h2{margin:0 0 8px;font-size:16px}"
"canvas{width:100%;height:auto;background:#0f0f18;border-radius:4px}"
".tabs{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}"
".tab{padding:4px 10px;border:none;border-radius:4px;cursor:pointer;background:#2a2a3a;color:#eee;font-size:12px}"
".tab.sel,.tab:hover{background:#4f8}"
".tab.sel{background:#4f8}"
"table{border-collapse:collapse;width:100%;font-size:12px}"
"th,td{border:1px solid #333;padding:4px 8px;text-align:right}"
"th{background:#2a2a3a}"
"tr:nth-child(odd){background:#1a1a2a}"
".row{display:flex;gap:8px;align-items:center;font-size:12px}"
".mono{font-family:monospace}"
".hidden{display:none}"
".full{width:100%;grid-column:1/-1}"
"</style>"
"<script src='https://unpkg.com/vue@3/dist/vue.global.prod.js'></script>"
"<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'></script>"
"<script src='//cdn.jsdelivr.net/npm/moment@2/moment.min.js'></script>"
"</head><body>"
"<div id='app'>"
"<div class='card full'><div class='tabs'>"
"<button @click=\"tab='scalars'\":class=\"{'sel':tab==='scalars'}\">Scalars</button>"
"<button @click=\"tab='graph'\" :class=\"{'sel':tab==='graph'}\">Graph</button>"
"<button @click=\"tab='emb'\" :class=\"{'sel':tab==='emb'}\">Embeddings</button>"
"<button @click=\"tab='hist'\" :class=\"{'sel':tab==='hist'}\">Histograms</button>"
"<button @click=\"tab='time'\" :class=\"{'sel':tab==='time'}\">Timeline</button>"
"<button @click=\"tab='sweeps'\" :class=\"{'sel':tab==='sweeps'}\">Sweeps</button>"
"</div><span class='row'>live WS: {{wsState}}<span style='margin-left:auto'>snap at {{lastSnap}}</span></span></div>"
"<div class='grid'>"
"<div class='card' v-show=\"tab==='scalars'\">"
"<h2>Scalars (last 64)</h2>"
"<canvas id='c-scalar'></canvas></div>"
"<div class='card' v-show=\"tab==='graph'\">"
"<h2>Model Graph</h2>"
"<div v-if='!snap.graph||!snap.graph.length'>no nodes</div>"
"<div v-for='n in snap.graph' :key='n.name' class='row'>"
"<b>{{n.name}}</b><span>in:[{{n.ins}}] out={{n.out}} shape={{(n.shape||[]).join('x')}}</span></div></div>"
"<div class='card' v-show=\"tab==='emb'\">"
"<h2>Embeddings (PCA 2-D)</h2>"
"<div v-if='!snap.embeddings||!Object.keys(snap.embeddings).length'>no embeddings</div>"
"<div v-for='[k,e] in snap.embeddings' :key='k' class='row' style='margin-bottom:8px'>"
"<b>{{k}}</b><span>n={{e.n_samples}} dim={{e.n_dim}} proj={{e.projected?'yes':'no'}}</span>"
"<canvas :id=\"'c-emb-'+k\"></canvas></div></div>"
"<div class='card' v-show=\"tab==='hist'\">"
"<h2>Histograms</h2>"
"<div v-if='!snap.histograms||!Object.keys(snap.histograms).length'>no histograms</div>"
"<div v-for='[k,h] in snap.histograms' :key='k' class='row' style='margin-bottom:8px'>"
"<b>{{k}}</b><span>bins={{h.n_bins}}</span>"
"<canvas :id=\"'c-hist-'+k\"></canvas></div></div>"
"<div class='card' v-show=\"tab==='time'\">"
"<h2>Profiling Timeline</h2>"
"<table v-if='snap.timeline&&snap.timeline.length'>"
"<tr><th>name</th><th>start_ms</th><th>dur_ms</th><th>bytes</th></tr>"
"<tr v-for='e in snap.timeline'><td>{{e.name}}</td><td>{{e.start_ms}}</td><td>{{e.dur_ms}}</td><td>{{e.mem_bytes}}</td></tr></table>"
"<div v-else>no timeline events</div></div>"
"<div class='card' v-show=\"tab==='sweeps'\">"
"<h2>Sweep Comparison</h2>"
"<table v-if='snap.sweeps&&snap.sweeps.length'>"
"<tr><th>config_id</th><th>loss</th><th>acc</th><th>secs</th></tr>"
"<tr v-for='e in snap.sweeps'><td>{{e.config_id}}</td><td>{{e.final_loss}}</td><td>{{e.final_acc}}</td><td>{{e.duration_s}}</td></tr></table>"
"<div v-else>no sweeps</div></div>"
"</div>"
"</div>"
"<script>"
"const {createApp,onMounted,onUnmounted,ref,nextTick}=Vue;"
"let WS=null;let retry=null;"
"createApp({"
"setup(){"
"const tab=ref('scalars');const snap=ref({});const wsState=ref('offline');const lastSnap=ref('');"
"const charts={};"
"const drawScalar=()=>{const ctx=document.getElementById('c-scalar');if(!ctx)return;const s=snap.value.series||{},labels=[],ds=[];"
"const keys=Object.keys(s);const max=Math.max(...keys.map(k=>s[k].length),0);const tailN=Math.min(64,max);"
"const idxStart=max-tailN;for(let i=0;i<tailN;i++){labels.push(s[keys[0]]&&s[keys[0]][idxStart+i]?s[keys[0]][idxStart+i][1].toFixed(1):'');}"
"for(const k of keys){const d=s[k]||[];const pts=[];for(let i=idxStart;i<idxStart+tailN;i++){if(d[i])pts.push({x:d[i][1],y:d[i][0]})}ds.push({label:k,data:pts,fill:false,borderWidth:1})}"
"if(charts.scalar)charts.scalar.destroy();"
"charts.scalar=new Chart(ctx,{type:'line',data:{labels, datasets:ds},options:{animation:false,responsive:true,maintainAspectRatio:false,scales:{x:{type:'linear',title:{display:true,text:'time'}},y:{beginAtZero:false}},plugins:{legend:{labels:{font:{size:10}}}}}});"
"}"
"const drawEmb=()=>{if(!snap.value.embeddings)return;for(const [k,e] of Object.entries(snap.value.embeddings)){const id='c-emb-'+k;const el=document.getElementById(id);if(!el||!e.points)continue;if(charts[id])charts[id].destroy();new Chart(el,{type:'scatter',data:{datasets:[{data:e.points.map(p=>({x:p[0],y:p[1]})),backgroundColor:'#4fc3f7',pointRadius:3}]},options:{animation:false,responsive:true,maintainAspectRatio:false,plugins:{title:{display:true,text:k}}}});charts[id]=arguments[0].constructor===Chart?arguments[0]:null;const c=Chart.getChart(el);charts[id]=c;}};"
"const drawHist=()=>{if(!snap.value.histograms)return;for(const [k,h] of Object.entries(snap.value.histograms)){const id='c-hist-'+k;const el=document.getElementById(id);if(!el)continue;if(charts[id])charts[id].destroy();const labels=h.edges.map((v,i)=>i<h.edges.length-1?v.toFixed(3):'');const c=new Chart(el,{type:'bar',data:{labels,data:[h.counts]},options:{animation:false,responsive:true,maintainAspectRatio:false,plugins:{title:{display:true,text:k}}}});charts[id]=c;}};"
"const render=()=>{drawScalar();drawEmb();drawHist();};"
"const load=async()=>{try{const r=await fetch('/snapshot',{cache:'no-store'});snap.value=await r.json();lastSnap.value=new Date().toLocaleTimeString();if(tab.value!=='graph'&&tab.value!=='time'&&tab.value!=='sweeps')render();}catch(e){console.error(e)}};"
"const startWS=()=>{WS=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');WS.onopen=()=>{wsState.value='open'};WS.onmessage=(e)=>{try{snap.value=JSON.parse(e.data);lastSnap.value=new Date().toLocaleTimeString();if(tab.value!=='graph'&&tab.value!=='time'&&tab.value!=='sweeps')render();}catch(err){}};WS.onclose=()=>{wsState.value='offline';clearTimeout(retry);retry=setTimeout(startWS,800)};WS.onerror=()=>{WS.close()}};"
"load();setInterval(load,2000);startWS();"
"onMounted(()=>{});onUnmounted(()=>{if(WS)WS.close();clearTimeout(retry);});"
"return{tab,snap,wsState,lastSnap}},"
"template:'#app'"
"});"
"</script></body></html>";

SNEPPX_VIZMON_API const char *SNEPPX_vizmon_frontend_html(void) { return FRONTEND_HTML; }

SNEPPX_VIZMON_API char *SNEPPX_vizmon_export_html(SNEPPX_VizMon *m) {
    char *snap = m ? SNEPPX_vizmon_snapshot_json(m) : (char*)"{}";
    size_t snap_len = snap ? strlen(snap) : 2;
    char *out = (char*)calloc(strlen(FRONTEND_HTML) + snap_len + 72, 1);
    if (!out) { if (snap && snap != (char*)"{}") free(snap); return NULL; }
    strcpy(out, "<script>const SNAPSHOT=");
    if (snap) strcat(out, snap); else strcat(out, "{}");
    strcat(out, ";</script>");
    strcat(out, FRONTEND_HTML);
    if (snap && snap != (char*)"{}") free(snap);
    return out;
}


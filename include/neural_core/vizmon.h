#ifndef SNEPPX_VIZMON_H
#define SNEPPX_VIZMON_H

/*
 * SNEPPX - Visualization & Monitoring (vizmon)
 *
 * WHAT
 *   Lightweight C control plane for a live training dashboard with:
 *     - real-time scalar plots (loss/accuracy/learning rate) over WebSocket,
 *     - model graph visualization (tensor shapes + operations),
 *     - 2-D embedding projection (PCA),
 *     - image/audio sample display,
 *     - histogram tracking for weights/biases/grads,
 *     - profiling timeline (kernel exec + memory usage),
 *     - hyperparameter-sweep configuration comparison,
 *     - export dashboards as standalone HTML.
 *
 * CONCEPT
 *   Training loops call the push_*() API (thread-safe ring buffers); a
 *   self-contained HTTP + WebSocket server streams snapshots to a Vue 3
 *   single-file frontend. Compiled to a static library (libvizmon) with no
 *   third-party deps. On Windows this links ws2_32.
 *
 *   The engine is pure logic and is fully unit-testable without binding to a
 *   socket; only SNEPPX_vizmon_start()/stop() touch the network.
 */

#include <stddef.h>

#if defined(_WIN32) && defined(SNEPPX_VIZMON_EXPORTS)
#  define SNEPPX_VIZMON_API __declspec(dllexport)
#elif defined(_WIN32) && !defined(SNEPPX_VIZMON_STATIC)
#  define SNEPPX_VIZMON_API __declspec(dllimport)
#else
#  define SNEPPX_VIZMON_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Limits                                                            */
/* ------------------------------------------------------------------ */
#define SNEPPX_VIZMON_MAX_SCRATCH  32
#define SNEPPX_VIZMON_MAX_NODES    2048
#define SNEPPX_VIZMON_MAX_DIM      1024
#define SNEPPX_VIZMON_MAX_SAMPLES  8192
#define SNEPPX_VIZMON_MAX_HIST_BINS 128
#define SNEPPX_VIZMON_MAX_TIMELINE 4096
#define SNEPPX_VIZMON_MAX_SWEEPS   64
#define SNEPPX_VIZMON_MAX_SERIES   4096
#define SNEPPX_VIZMON_MAX_TAG_LEN  96
#define SNEPPX_VIZMON_MAX_WS_CLIENTS 64

/* ------------------------------------------------------------------ */
/* Core types                                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    char  tag[SNEPPX_VIZMON_MAX_TAG_LEN];   /* e.g. "loss", "accuracy" */
    double value;
    double ts;                              /* monotonic seconds */
} SNEPPX_VizScalar;

typedef struct {
    char  tag[SNEPPX_VIZMON_MAX_TAG_LEN];
    double *values;                         /* owned */
    double *times;                          /* owned */
    int    count;
    int    cap;
} SNEPPX_VizSeries;

typedef struct {
    char  name[SNEPPX_VIZMON_MAX_TAG_LEN];   /* operation, e.g. "matmul" */
    int   in_nodes[SNEPPX_VIZMON_MAX_SCRATCH];
    int   num_in;
    int   out_node;
    double shape[SNEPPX_VIZMON_MAX_SCRATCH]; /* output tensor shape */
    int   nshape;
} SNEPPX_VizNode;

typedef struct {
    double *data;          /* n_samples * n_dim, owned */
    double *proj;          /* n_samples * 2, owned (PCA output) */
    int    n_samples;
    int    n_dim;
    int    projected;      /* 1 after SNEPPX_vizmon_project_pca */
} SNEPPX_VizEmbedding;

typedef enum {
    SNEPPX_VIZMON_IMG_PNG = 0,
    SNEPPX_VIZMON_AUD_WAV = 1
} SNEPPX_VizSampleKind;

typedef struct {
    char                tag[SNEPPX_VIZMON_MAX_TAG_LEN];
    SNEPPX_VizSampleKind kind;
    unsigned char      *bytes;   /* owned */
    size_t              len;
    double              ts;
} SNEPPX_VizSample;

typedef struct {
    double *edges;        /* owned, n_bins+1 */
    int    *counts;       /* owned, n_bins */
    int    n_bins;
    double ts;
} SNEPPX_VizHistogram;

typedef struct {
    char   name[SNEPPX_VIZMON_MAX_TAG_LEN]; /* kernel / op name */
    double start_ms;                        /* monotonic */
    double dur_ms;
    size_t mem_bytes;                       /* peak bytes touched */
} SNEPPX_VizTimelineEvent;

typedef struct {
    char  config_id[SNEPPX_VIZMON_MAX_TAG_LEN]; /* sweep run id */
    double final_loss;
    double final_acc;
    double duration_s;
} SNEPPX_VizSweepResult;

/* ------------------------------------------------------------------ */
/* Engine handle                                                      */
/* ------------------------------------------------------------------ */
typedef struct SNEPPX_VizMon SNEPPX_VizMon;

/* Lifecycle */
SNEPPX_VIZMON_API SNEPPX_VizMon *SNEPPX_vizmon_create(void);
SNEPPX_VIZMON_API void           SNEPPX_vizmon_destroy(SNEPPX_VizMon *m);

/* ---- real-time scalars ---- */
SNEPPX_VIZMON_API int              SNEPPX_vizmon_push_scalar(SNEPPX_VizMon *m, const char *tag,
                               double value, double ts);
SNEPPX_VIZMON_API int              SNEPPX_vizmon_get(SNEPPX_VizMon *m, const char *tag,
                                 SNEPPX_VizSeries *out_series);

/* ---- model graph ---- */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_add_node(SNEPPX_VizMon *m, const char *name,
                             const int *in_nodes, int num_in,
                             int out_node, const double *shape, int nshape);
SNEPPX_VIZMON_API int  SNEPPX_vizmon_get_graph(SNEPPX_VizMon *m, SNEPPX_VizNode *out,
                              int *n_nodes);

/* ---- embedding projection (PCA to 2-D) ---- */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_push_embedding(SNEPPX_VizMon *m, const char *tag,
                                   const double *data, int n_samples, int n_dim);
SNEPPX_VIZMON_API int  SNEPPX_vizmon_project_pca(SNEPPX_VizMon *m, int max_components);
SNEPPX_VIZMON_API SNEPPX_VizEmbedding *SNEPPX_vizmon_get_embedding(SNEPPX_VizMon *m,
                                                  const char *tag);

/* ---- image / audio samples ---- */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_push_sample(SNEPPX_VizMon *m, const char *tag,
                                SNEPPX_VizSampleKind kind,
                                const unsigned char *bytes, size_t len,
                                double ts);
SNEPPX_VIZMON_API int  SNEPPX_vizmon_get_samples(SNEPPX_VizMon *m, const char *tag,
                                SNEPPX_VizSample *out, int *n);

/* ---- histograms (weights / biases / grads) ---- */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_push_histogram(SNEPPX_VizMon *m, const char *tag,
                                   const double *values, int n,
                                   int n_bins, double ts);
SNEPPX_VIZMON_API int  SNEPPX_vizmon_get_histogram(SNEPPX_VizMon *m, const char *tag,
                                  SNEPPX_VizHistogram *out);

/* ---- profiling timeline ---- */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_push_timeline(SNEPPX_VizMon *m, const char *name,
                                  double start_ms, double dur_ms,
                                  size_t mem_bytes);
SNEPPX_VIZMON_API int  SNEPPX_vizmon_get_timeline(SNEPPX_VizMon *m,
                                 SNEPPX_VizTimelineEvent *out, int *n);

/* ---- sweep config comparison ---- */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_push_sweep(SNEPPX_VizMon *m, const char *config_id,
                               double final_loss, double final_acc,
                               double duration_s);
SNEPPX_VIZMON_API int  SNEPPX_vizmon_get_sweeps(SNEPPX_VizMon *m, SNEPPX_VizSweepResult *out,
                               int *n);

/* ---- snapshot: build a single JSON string of all current state ----
 * Caller owns the returned buffer and must SNEPPX_vizmon_free() it.       */
SNEPPX_VIZMON_API char *SNEPPX_vizmon_snapshot_json(SNEPPX_VizMon *m);
SNEPPX_VIZMON_API void  SNEPPX_vizmon_free(void *ptr);

/* ---- network server ----
 * Starts a background thread serving HTTP (frontend) + WebSocket (/ws)
 * on `port`. Returns 0 on success. stop() drains and joins.            */
SNEPPX_VIZMON_API int  SNEPPX_vizmon_start(SNEPPX_VizMon *m, int port);
SNEPPX_VIZMON_API void SNEPPX_vizmon_stop(SNEPPX_VizMon *m);

/* ---- static frontend asset (Vue 3, CDN) ----
 * Returns a NUL-terminated string the caller does NOT free.            */
SNEPPX_VIZMON_API const char *SNEPPX_vizmon_frontend_html(void);

/* ---- HTML export of the current snapshot ----
 * Caller owns the returned buffer and must SNEPPX_vizmon_free() it.    */
SNEPPX_VIZMON_API char *SNEPPX_vizmon_export_html(SNEPPX_VizMon *m);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_VIZMON_H */

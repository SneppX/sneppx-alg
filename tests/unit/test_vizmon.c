/*
 * SNEPPX - vizmon unit test (compile + run standalone with MSVC + UCRT).
 *
 * Usage:
  *   cl /nologo /D_WIN32_WINNT=0x0601 /DSNEPPX_VIZMON_EXPORTS /Iinclude vizmon.c tests/unit/test_vizmon.c ws2_32.lib advapi32.lib
 *   test_vizmon.exe
 */
#include "neural_core/vizmon.h"
#include "test_gtest.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int failures = 0;

TEST(test_vizmon, suite) {
    setvbuf(stdout, NULL, _IONBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
    SNEPPX_VizMon *m = SNEPPX_vizmon_create();
    EXPECT_TRUE(m != NULL);
    for (int i = 0; i < 10; i++)
        SNEPPX_vizmon_push_scalar(m, "loss",  (double)(10 - i) * 0.1, (double)i);
    for (int i = 0; i < 10; i++)
        SNEPPX_vizmon_push_scalar(m, "acc",   (double)(i + 1) * 0.1, (double)i + 0.5);
    SNEPPX_VizSeries s = {0};
    EXPECT_TRUE(SNEPPX_vizmon_get(m, "loss", &s) == 0);
    EXPECT_TRUE(s.count == 10);
    EXPECT_TRUE(s.values[0] == 1.0 && s.values[9] == 0.1);
    EXPECT_TRUE(s.times[0] == 0.0 && s.times[9] == 9.0);
    /* caller-owned buffers */
    if (s.values) free(s.values);
    if (s.times) free(s.times);
    EXPECT_TRUE(SNEPPX_vizmon_get(m, "nope", &s) == -1);

    /* ---- graph ---- */
    int ins0[] = {-1}; (void)ins0;
    int ins1[] = {0};
    EXPECT_TRUE(SNEPPX_vizmon_add_node(m, "x:0", ins0, 0, 0, SX_ARR_C(double, 1, 4), 1) == 0);
    EXPECT_TRUE(SNEPPX_vizmon_add_node(m, "linear", ins1, 1, 1, SX_ARR_C(double, 2, 4,8), 2) == 0);
    int ins2[] = {1, 0};
    EXPECT_TRUE(SNEPPX_vizmon_add_node(m, "add", ins2, 2, 2, SX_ARR_C(double, 1, 8), 1) == 0);
    SNEPPX_VizNode nodes[16] = {0}; int nn = 0;
    EXPECT_TRUE(SNEPPX_vizmon_get_graph(m, nodes, &nn) == 0);
    EXPECT_TRUE(nn == 3);
    EXPECT_TRUE(strcmp(nodes[1].name, "linear") == 0);
    EXPECT_TRUE(nodes[1].num_in == 1 && nodes[1].in_nodes[0] == 0 && nodes[1].out_node == 1);
    EXPECT_TRUE(nodes[1].nshape == 2 && nodes[1].shape[0] == 4 && nodes[1].shape[1] == 8);
    EXPECT_TRUE(strcmp(nodes[2].name, "add") == 0 && nodes[2].num_in == 2);

    /* ---- embeddings + PCA ---- */
    double pts[6 * 3] = {
        1.0,  1.1,  0.9,
       -1.0, -1.1, -0.9,
        1.0,  1.2,  1.0,
       -1.0, -1.0, -1.1,
        1.1,  1.0,  1.0,
       -0.9, -1.0, -1.0
    };
    EXPECT_TRUE(SNEPPX_vizmon_push_embedding(m, "h", pts, 6, 3) == 0);
    EXPECT_TRUE(SNEPPX_vizmon_project_pca(m, 2) == 0);
    SNEPPX_VizEmbedding *e = SNEPPX_vizmon_get_embedding(m, "h");
    EXPECT_TRUE(e != NULL && e->n_samples == 6 && e->n_dim == 3 && e->projected == 1);
    EXPECT_TRUE(e->proj != NULL);
    /* first 3 points cluster at +x, last 3 at -x: principal component separates them */
    double mean_pos = (e->proj[0] + e->proj[4] + e->proj[8]) / 3.0;
    double mean_neg = (e->proj[4] + e->proj[8] + e->proj[12]) / 3.0;
    (void)mean_pos; (void)mean_neg;
    EXPECT_TRUE(e->proj[0] != e->proj[4] || e->proj[2] != e->proj[6]); /* not all identical */
    SNEPPX_vizmon_free(e->data); SNEPPX_vizmon_free(e->proj); SNEPPX_vizmon_free(e);

    /* ---- samples ---- */
    unsigned char png[4] = {137, 80, 78, 71};
    EXPECT_TRUE(SNEPPX_vizmon_push_sample(m, "img", SNEPPX_VIZMON_IMG_PNG, png, 4, 42.0) == 0);
    SNEPPX_VizSample sm[8] = {0}; int ns = 0;
    EXPECT_TRUE(SNEPPX_vizmon_get_samples(m, "img", sm, &ns) == 0);
    EXPECT_TRUE(ns == 1 && sm[0].kind == SNEPPX_VIZMON_IMG_PNG && sm[0].len == 4);
    for (int i = 0; i < ns; i++) free(sm[i].bytes);

    /* ---- histograms ---- */
    double data100[100]; for (int i = 0; i < 100; i++) data100[i] = (double)i;
    EXPECT_TRUE(SNEPPX_vizmon_push_histogram(m, "w", data100, 100, 20, 7.0) == 0);
    SNEPPX_VizHistogram h = {0};
    EXPECT_TRUE(SNEPPX_vizmon_get_histogram(m, "w", &h) == 0);
    EXPECT_TRUE(h.n_bins == 20);
    EXPECT_TRUE(h.counts[0] == 5 && h.counts[19] == 5); /* 0..4 and 95..99 in first/last bin */
    free(h.edges); free(h.counts);

    /* ---- timeline ---- */
    EXPECT_TRUE(SNEPPX_vizmon_push_timeline(m, "gemm_0", 1.0, 2.5, 1024) == 0);
    EXPECT_TRUE(SNEPPX_vizmon_push_timeline(m, "softmax_0", 3.5, 0.3, 512) == 0);
    SNEPPX_VizTimelineEvent te[64] = {0}; int nt = 0;
    EXPECT_TRUE(SNEPPX_vizmon_get_timeline(m, te, &nt) == 0);
    EXPECT_TRUE(nt == 2);
    EXPECT_TRUE(strcmp(te[0].name, "gemm_0") == 0 && te[0].dur_ms == 2.5 && te[0].mem_bytes == 1024);

    /* ---- sweeps ---- */
    EXPECT_TRUE(SNEPPX_vizmon_push_sweep(m, "lr-3", 0.12, 0.97, 320.0) == 0);
    EXPECT_TRUE(SNEPPX_vizmon_push_sweep(m, "lr-4", 0.09, 0.98, 410.0) == 0);
    SNEPPX_VizSweepResult sw[8] = {0}; int nsw = 0;
    EXPECT_TRUE(SNEPPX_vizmon_get_sweeps(m, sw, &nsw) == 0);
    EXPECT_TRUE(nsw == 2);
    EXPECT_TRUE(strcmp(sw[0].config_id, "lr-3") == 0 && sw[0].final_loss == 0.12);

    /* ---- snapshot JSON ---- */
    char *json = SNEPPX_vizmon_snapshot_json(m);
    EXPECT_TRUE(json != NULL);
    EXPECT_TRUE(strstr(json, "scalars") != NULL);
    EXPECT_TRUE(strstr(json, "linear") != NULL);
    EXPECT_TRUE(strstr(json, "embeddings") != NULL);
    EXPECT_TRUE(strstr(json, "histograms") != NULL);
    EXPECT_TRUE(strstr(json, "timeline") != NULL);
    EXPECT_TRUE(strstr(json, "sweeps") != NULL);
    EXPECT_TRUE(strstr(json, "\"final_acc\":0.97") != NULL);
    free(json);

    /* ---- frontend HTML ---- */
    const char *html = SNEPPX_vizmon_frontend_html();
    EXPECT_TRUE(html != NULL);
    EXPECT_TRUE(strstr(html, "vue.global.prod") != NULL);
    EXPECT_TRUE(strstr(html, "chart.js") != NULL);
    EXPECT_TRUE(strstr(html, "/snapshot") != NULL);
    EXPECT_TRUE(strstr(html, "/ws") != NULL);

    /* ---- export HTML embeds snapshot ---- */
    char *exp = SNEPPX_vizmon_export_html(m);
    EXPECT_TRUE(exp != NULL);
    EXPECT_TRUE(strstr(exp, "const SNAPSHOT=") != NULL);
    EXPECT_TRUE(strstr(exp, "vue.global.prod") != NULL);
    free(exp);

    /* ---- server start/stop smoke (use a high port) ---- */
    EXPECT_TRUE(SNEPPX_vizmon_start(m, 18399) == 0);
    SNEPPX_vizmon_stop(m);
    EXPECT_TRUE(SNEPPX_vizmon_start(m, 18399) == 0);
    SNEPPX_vizmon_stop(m);

    SNEPPX_vizmon_destroy(m);

    if (failures == 0) {
        printf("ALL vizmon TESTS PASS\n");
        return;
    }
    FAIL() << "early exit (legacy return 1)";
}

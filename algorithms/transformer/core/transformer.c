#include "transformer_model.h"
#include "multidimensional_tensor_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SNEPPX_T_MAXSEQ 8192

/*
 * SNEPPX - Transformer
 *
 * WHAT
 *   Transformer.
 *
 * CONCEPT
 *   Provides transformer architecture.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


struct SNEPPXTransformer {
    size_t vocab, hidden, heads, layers, ffn;
    int use_rope;
    float dropout;
    SNEPPXTensor* embedding;   /* [vocab, hidden] */
    SNEPPXTensor* pos_embed;   /* [max_seq, hidden] */
    SNEPPXTensor** Wq;         /* [layers][hidden, hidden] */
    SNEPPXTensor** Wk;
    SNEPPXTensor** Wv;
    SNEPPXTensor** Wo;
    SNEPPXTensor** W1;         /* [layers][ffn, hidden] */
    SNEPPXTensor** W2;         /* [layers][hidden, ffn] */
    SNEPPXTensor** ln1_g;      /* [layers][hidden] */
    SNEPPXTensor** ln1_b;
    SNEPPXTensor** ln2_g;
    SNEPPXTensor** ln2_b;
    SNEPPXTensor* ln_f_g;
    SNEPPXTensor* ln_f_b;
    SNEPPXTensor* lm_head;     /* [vocab, hidden] */
};

static void linear_fwd(const float* x, size_t m, const float* w, size_t k, size_t out, float* y) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < out; j++) {
            float s = 0.0f;
            for (size_t p = 0; p < k; p++) s += x[i * k + p] * w[j * k + p];
            y[i * out + j] = s;
        }
    }
}

static void layernorm_fwd(const float* x, const float* g, const float* b, size_t m, size_t hidden, float eps, float* y) {
    for (size_t i = 0; i < m; i++) {
        const float* row = x + i * hidden;
        float mean = 0.0f;
        for (size_t p = 0; p < hidden; p++) mean += row[p];
        mean /= (float)hidden;
        float var = 0.0f;
        for (size_t p = 0; p < hidden; p++) { float d = row[p] - mean; var += d * d; }
        var /= (float)hidden;
        float inv = 1.0f / sqrtf(var + eps);
        for (size_t p = 0; p < hidden; p++) {
            float v = (row[p] - mean) * inv;
            if (g) v *= g[p];
            if (b) v += b[p];
            y[i * hidden + p] = v;
        }
    }
}

static void rope(float* qk, size_t seq, size_t hidden, size_t heads, size_t head_dim) {
    for (size_t i = 0; i < seq; i++) {
        for (size_t h = 0; h < heads; h++) {
            for (size_t p = 0; p + 1 < head_dim; p += 2) {
                float theta = (float)i / powf(10000.0f, (float)(p) / (float)head_dim);
                float c = cosf(theta), s = sinf(theta);
                size_t off = (i * heads + h) * head_dim + p;
                float a = qk[off], b = qk[off + 1];
                qk[off] = a * c - b * s;
                qk[off + 1] = a * s + b * c;
            }
        }
    }
}

/**
 * @brief Create Transformer.
 *
 * @param vocab_size [in] Vocab Size value.
 * @param hidden_dim [in] Hidden Dim value.
 * @param num_heads [in] Num Heads value.
 * @param num_layers [in] Num Layers value.
 * @param ffn_dim [in] Ffn Dim value.
 * @param dropout [in] Dropout value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTransformer* SNEPPX_transformer_create(size_t vocab_size, size_t hidden_dim,
        size_t num_heads, size_t num_layers, size_t ffn_dim, float dropout, int use_rope) {
    if (hidden_dim == 0 || num_heads == 0 || hidden_dim % num_heads != 0 || num_layers == 0) return NULL;
    SNEPPXTransformer* m = (SNEPPXTransformer*)calloc(1, sizeof(SNEPPXTransformer));
    if (!m) return NULL;
    m->vocab = vocab_size; m->hidden = hidden_dim; m->heads = num_heads;
    m->layers = num_layers; m->ffn = ffn_dim; m->use_rope = use_rope; m->dropout = dropout;
    size_t h = hidden_dim;
    size_t shp_h[] = {h};
    m->embedding = SNEPPX_tensor_randn((size_t[]){vocab_size, h}, 2, SNEPPX_FLOAT32);
    m->pos_embed = SNEPPX_tensor_randn((size_t[]){SNEPPX_T_MAXSEQ, h}, 2, SNEPPX_FLOAT32);
    m->Wq = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->Wk = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->Wv = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->Wo = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->W1 = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->W2 = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->ln1_g = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->ln1_b = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->ln2_g = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    m->ln2_b = (SNEPPXTensor**)calloc(num_layers, sizeof(SNEPPXTensor*));
    for (size_t l = 0; l < num_layers; l++) {
        m->Wq[l] = SNEPPX_tensor_randn((size_t[]){h, h}, 2, SNEPPX_FLOAT32);
        m->Wk[l] = SNEPPX_tensor_randn((size_t[]){h, h}, 2, SNEPPX_FLOAT32);
        m->Wv[l] = SNEPPX_tensor_randn((size_t[]){h, h}, 2, SNEPPX_FLOAT32);
        m->Wo[l] = SNEPPX_tensor_randn((size_t[]){h, h}, 2, SNEPPX_FLOAT32);
        m->W1[l] = SNEPPX_tensor_randn((size_t[]){ffn_dim, h}, 2, SNEPPX_FLOAT32);
        m->W2[l] = SNEPPX_tensor_randn((size_t[]){h, ffn_dim}, 2, SNEPPX_FLOAT32);
        m->ln1_g[l] = SNEPPX_tensor_ones(shp_h, 1, SNEPPX_FLOAT32);
        m->ln1_b[l] = SNEPPX_tensor_zeros(shp_h, 1, SNEPPX_FLOAT32);
        m->ln2_g[l] = SNEPPX_tensor_ones(shp_h, 1, SNEPPX_FLOAT32);
        m->ln2_b[l] = SNEPPX_tensor_zeros(shp_h, 1, SNEPPX_FLOAT32);
    }
    m->ln_f_g = SNEPPX_tensor_ones(shp_h, 1, SNEPPX_FLOAT32);
    m->ln_f_b = SNEPPX_tensor_zeros(shp_h, 1, SNEPPX_FLOAT32);
    m->lm_head = SNEPPX_tensor_randn((size_t[]){vocab_size, h}, 2, SNEPPX_FLOAT32);
    if (!m->embedding || !m->lm_head) { SNEPPX_transformer_destroy(m); return NULL; }
    return m;
}

/**
 * @brief Destroy Transformer.
 */
void SNEPPX_transformer_destroy(void* model) {
    SNEPPXTransformer* m = (SNEPPXTransformer*)model;
    if (!m) return;
    SNEPPX_tensor_destroy(m->embedding); SNEPPX_tensor_destroy(m->pos_embed);
    SNEPPX_tensor_destroy(m->ln_f_g); SNEPPX_tensor_destroy(m->ln_f_b);
    SNEPPX_tensor_destroy(m->lm_head);
    for (size_t l = 0; l < m->layers; l++) {
        SNEPPX_tensor_destroy(m->Wq[l]); SNEPPX_tensor_destroy(m->Wk[l]); SNEPPX_tensor_destroy(m->Wv[l]);
        SNEPPX_tensor_destroy(m->Wo[l]); SNEPPX_tensor_destroy(m->W1[l]); SNEPPX_tensor_destroy(m->W2[l]);
        SNEPPX_tensor_destroy(m->ln1_g[l]); SNEPPX_tensor_destroy(m->ln1_b[l]);
        SNEPPX_tensor_destroy(m->ln2_g[l]); SNEPPX_tensor_destroy(m->ln2_b[l]);
    }
    free(m->Wq); free(m->Wk); free(m->Wv); free(m->Wo); free(m->W1); free(m->W2);
    free(m->ln1_g); free(m->ln1_b); free(m->ln2_g); free(m->ln2_b);
    free(m);
}

/**
 * @brief Run the forward pass for Transformer.
 *
 * @param model [out] Model value.
 * @param input_ids [in] Input Ids value.
 * @param seq_len [in] Seq Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transformer_forward(void* model, const int* input_ids, size_t seq_len, float* logits) {
    SNEPPXTransformer* m = (SNEPPXTransformer*)model;
    if (!m || !input_ids || !logits) return -1;
    if (seq_len > SNEPPX_T_MAXSEQ) seq_len = SNEPPX_T_MAXSEQ;
    size_t H = m->hidden, Nh = m->heads, hd = H / Nh;
    float* x = (float*)malloc(seq_len * H * sizeof(float));
    if (!x) return -1;
    /* embed + positional */
    float* emb = (float*)m->embedding->data;
    float* pos = (float*)m->pos_embed->data;
    for (size_t i = 0; i < seq_len; i++) {
        int id = input_ids[i]; if (id < 0) id = 0; if ((size_t)id >= m->vocab) id = (int)m->vocab - 1;
        memcpy(x + i * H, emb + (size_t)id * H, H * sizeof(float));
        for (size_t p = 0; p < H; p++) x[i * H + p] += pos[i * H + p];
    }
    float* q = (float*)malloc(seq_len * H * sizeof(float));
    float* k = (float*)malloc(seq_len * H * sizeof(float));
    float* v = (float*)malloc(seq_len * H * sizeof(float));
    float* attn = (float*)malloc(seq_len * H * sizeof(float));
    float* proj = (float*)malloc(seq_len * H * sizeof(float));
    float* xn = (float*)malloc(seq_len * H * sizeof(float));
    float* ffn = (float*)malloc(seq_len * m->ffn * sizeof(float));
    float* ffn2 = (float*)malloc(seq_len * H * sizeof(float));
    if (!q || !k || !v || !attn || !proj || !xn || !ffn || !ffn2) { free(x); free(q); free(k); free(v); free(attn); free(proj); free(xn); free(ffn); free(ffn2); return -1; }

    for (size_t l = 0; l < m->layers; l++) {
        layernorm_fwd(x, (float*)m->ln1_g[l]->data, (float*)m->ln1_b[l]->data, seq_len, H, 1e-5f, xn);
        linear_fwd(xn, seq_len, (float*)m->Wq[l]->data, H, H, q);
        linear_fwd(xn, seq_len, (float*)m->Wk[l]->data, H, H, k);
        linear_fwd(xn, seq_len, (float*)m->Wv[l]->data, H, H, v);
        if (m->use_rope) { rope(q, seq_len, H, Nh, hd); rope(k, seq_len, H, Nh, hd); }
        /* attention per head */
        memset(attn, 0, seq_len * H * sizeof(float));
        for (size_t hh = 0; hh < Nh; hh++) {
            for (size_t i = 0; i < seq_len; i++) {
                /* scores */
                float scores[SNEPPX_T_MAXSEQ];
                float mx = -1e30f;
                for (size_t j = 0; j < seq_len; j++) {
                    float s = 0.0f;
                    for (size_t p = 0; p < hd; p++) s += q[(i * Nh + hh) * hd + p] * k[(j * Nh + hh) * hd + p];
                    s /= sqrtf((float)hd);
                    if (j > i) s = -1e30f; /* causal */
                    scores[j] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
                float inv = sum > 0 ? 1.0f / sum : 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    float a = scores[j] * inv;
                    for (size_t p = 0; p < hd; p++)
                        attn[(i * Nh + hh) * hd + p] += a * v[(j * Nh + hh) * hd + p];
                }
            }
        }
        linear_fwd(attn, seq_len, (float*)m->Wo[l]->data, H, H, proj);
        for (size_t i = 0; i < seq_len * H; i++) x[i] += proj[i];
        layernorm_fwd(x, (float*)m->ln2_g[l]->data, (float*)m->ln2_b[l]->data, seq_len, H, 1e-5f, xn);
        linear_fwd(xn, seq_len, (float*)m->W1[l]->data, H, m->ffn, ffn);
        for (size_t i = 0; i < seq_len * m->ffn; i++) ffn[i] = ffn[i] > 0 ? ffn[i] : 0.0f; /* relu */
        linear_fwd(ffn, seq_len, (float*)m->W2[l]->data, m->ffn, H, ffn2);
        for (size_t i = 0; i < seq_len * H; i++) x[i] += ffn2[i];
    }
    layernorm_fwd(x, (float*)m->ln_f_g->data, (float*)m->ln_f_b->data, seq_len, H, 1e-5f, xn);
    /* logits = xn @ lm_head^T  -> [seq_len, vocab] */
    linear_fwd(xn, seq_len, (float*)m->lm_head->data, H, m->vocab, logits);
    free(x); free(q); free(k); free(v); free(attn); free(proj); free(xn); free(ffn); free(ffn2);
    return 0;
}

/**
 * @brief Perform Transformer Generate.
 *
 * @param model [out] Model value.
 * @param prompt [in] Prompt value.
 * @param prompt_len [in] Prompt Len value.
 * @param output [out] Output value.
 * @param max_len [in] Max Len value.
 * @param temperature [in] Temperature value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_transformer_generate(void* model, const int* prompt, size_t prompt_len,
        int* output, size_t max_len, int temperature, int top_k) {
    (void)temperature; (void)top_k;
    SNEPPXTransformer* m = (SNEPPXTransformer*)model;
    if (!m || !prompt || !output) return -1;
    int* ctx = (int*)malloc((prompt_len + max_len) * sizeof(int));
    if (!ctx) return -1;
    memcpy(ctx, prompt, prompt_len * sizeof(int));
    size_t total = prompt_len;
    for (size_t step = 0; step < max_len; step++) {
        float* logits = (float*)malloc(total * m->vocab * sizeof(float));
        if (!logits) { free(ctx); return -1; }
        SNEPPX_transformer_forward(m, ctx, total, logits);
        /* argmax over last position */
        float* last = logits + (total - 1) * m->vocab;
        int best = 0; float bv = last[0];
        for (size_t v = 1; v < m->vocab; v++) if (last[v] > bv) { bv = last[v]; best = (int)v; }
        free(logits);
        output[step] = best;
        ctx[total++] = best;
        if (best == 0) break;
    }
    free(ctx);
    return 0;
}

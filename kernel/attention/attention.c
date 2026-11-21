#include "multi_head_attention_module.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/*
 * SNEPPX - Attention Mechanisms
 *
 * WHAT
 *   Attention Mechanisms.
 *
 * CONCEPT
 *   Attention Mechanisms implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/* ========== Helpers ========== */

static float frand(unsigned int* seed) {
    *seed = *seed * 1103515245U + 12345U;
    return (float)((*seed >> 16) & 0x7FFF) / 32767.0f;
}

static float kaiming_init(unsigned int* seed, size_t fan) {
    float std = sqrtf(2.0f / (float)fan);
    float u1 = frand(seed), u2 = frand(seed);
    return std * sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.2831853f * u2);
}

/* ========== Config ========== */

/**
 * @brief Perform Attn Config Default.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXAttentionConfig SNEPPX_attn_config_default(void) {
    SNEPPXAttentionConfig cfg;
    cfg.num_heads = 8;
    cfg.head_dim = 64;
    cfg.d_model = 512;
    cfg.dropout = 0.0f;
    cfg.use_causal_mask = 1;
    cfg.use_rope = 1;
    cfg.rope_base = 10000.0f;
    return cfg;
}

/* ========== Weights ========== */

/**
 * @brief Create Attn Weights.
 *
 * @param cfg [in] Cfg value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXAttentionWeights* SNEPPX_attn_weights_create(SNEPPXAttentionConfig cfg, unsigned int seed) {
    SNEPPXAttentionWeights* w = SNEPPX_malloc(sizeof(SNEPPXAttentionWeights), 64);
    if (!w) return NULL;
    w->config = cfg;
    size_t d = cfg.d_model, dk = cfg.num_heads * cfg.head_dim;
    size_t shape2[] = {d, dk}, shape1[] = {dk};

    w->w_q = SNEPPX_tensor_create(shape2, 2, SNEPPX_FLOAT32);
    w->b_q = SNEPPX_tensor_zeros(shape1, 1, SNEPPX_FLOAT32);
    w->w_k = SNEPPX_tensor_create(shape2, 2, SNEPPX_FLOAT32);
    w->b_k = SNEPPX_tensor_zeros(shape1, 1, SNEPPX_FLOAT32);
    w->w_v = SNEPPX_tensor_create(shape2, 2, SNEPPX_FLOAT32);
    w->b_v = SNEPPX_tensor_zeros(shape1, 1, SNEPPX_FLOAT32);
    size_t shape2o[] = {dk, d};
    w->w_o = SNEPPX_tensor_create(shape2o, 2, SNEPPX_FLOAT32);
    w->b_o = SNEPPX_tensor_zeros(shape1, 1, SNEPPX_FLOAT32);

    SNEPPXTensor* weights[] = {w->w_q, w->w_k, w->w_v, w->w_o};
    for (int wi = 0; wi < 4; wi++) {
        float* dptr = (float*)weights[wi]->data;
        size_t n = weights[wi]->size;
        for (size_t i = 0; i < n; i++)
            dptr[i] = kaiming_init(&seed, d);
    }
    return w;
}

/**
 * @brief Destroy Attn Weights.
 */
void SNEPPX_attn_weights_destroy(SNEPPXAttentionWeights* w) {
    if (!w) return;
    SNEPPX_tensor_destroy(w->w_q); SNEPPX_tensor_destroy(w->b_q);
    SNEPPX_tensor_destroy(w->w_k); SNEPPX_tensor_destroy(w->b_k);
    SNEPPX_tensor_destroy(w->w_v); SNEPPX_tensor_destroy(w->b_v);
    SNEPPX_tensor_destroy(w->w_o); SNEPPX_tensor_destroy(w->b_o);
    SNEPPX_free(w, sizeof(SNEPPXAttentionWeights));
}

/**
 * @brief Perform Attn Num Params.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_attn_num_params(const SNEPPXAttentionWeights* w) {
    size_t n = 0;
    SNEPPXTensor* ts[] = {w->w_q, w->b_q, w->w_k, w->b_k, w->w_v, w->b_v, w->w_o, w->b_o};
    for (int i = 0; i < 8; i++) n += ts[i]->size;
    return n;
}

/**
 * @brief Perform Attn Get Params.
 *
 * @param w [in] W value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_attn_get_params(const SNEPPXAttentionWeights* w, SNEPPXTensor** out, size_t max) {
    if (!w || !out || max < 8) return -1;
    SNEPPXTensor* ts[] = {w->w_q, w->b_q, w->w_k, w->b_k, w->w_v, w->b_v, w->w_o, w->b_o};
    for (int i = 0; i < 8; i++) out[i] = ts[i];
    return 8;
}

/* ========== RoPE ========== */

/**
 * @brief Compute Rope Pre.
 *
 * @param seq_len [in] Seq Len value.
 * @param head_dim [in] Head Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_rope_precompute(size_t seq_len, size_t head_dim, float base) {
    size_t shape[] = {seq_len, head_dim};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    if (!t) return NULL;
    float* d = (float*)t->data;
    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t j = 0; j < head_dim / 2; j++) {
            float theta = (float)pos / powf(base, 2.0f * j / (float)head_dim);
            d[pos * head_dim + j] = cosf(theta);
            d[pos * head_dim + j + head_dim / 2] = sinf(theta);
        }
    }
    return t;
}

/**
 * @brief Apply Rope.
 *
 * @param q [out] Q value.
 * @param k [out] K value.
 * @param cos [in] Cos value.
 * @param sin [in] Sin value.
 */
void SNEPPX_rope_apply(SNEPPXTensor* q, SNEPPXTensor* k, const SNEPPXTensor* cos, const SNEPPXTensor* sin,
                     size_t offset) {
    if (!q || !k || !cos) return;
    size_t d = q->shape[q->ndim - 1];
    size_t half = d / 2;
    if (half == 0) return;
    size_t total = q->size;
    float* qd = (float*)q->data;
    float* kd = (float*)k->data;
    const float* cd = (const float*)cos->data;
    size_t cd_last = cos->shape[cos->ndim - 1];
    (void)sin;
    /* SNEPPX_rope_precompute returns a combined [S, d] table: first half =
       cos, second half = sin. Every caller passes this combined table as
       `cos` (and `sin` is NULL or the same pointer). Process each (x_i,
       x_{i+half}) pair together from the ORIGINAL values so the in-place
       rotation is not contaminated by earlier writes. */
    /* The [B,H,S,hd] layout means base/d is the flat [B*H*S] index, not the
       sequence position. pos must be (base/d)%S so multi-head tensors with
       S=1 do not index past the [S,hd] cos table. */
    size_t seq_len = (q->ndim >= 2) ? q->shape[q->ndim - 2] : 1;
    for (size_t base = 0; base < total; base += d) {
        size_t pos = ((base / d) % seq_len) + offset;
        for (size_t j = 0; j < half; j++) {
            float cosv = cd[pos * cd_last + j];
            float sinv = cd[pos * cd_last + half + j];
            float qv = qd[base + j];      float qvp = qd[base + j + half];
            float kv = kd[base + j];      float kvp = kd[base + j + half];
            qd[base + j]       = qv * cosv - qvp * sinv;
            qd[base + j + half] = qv * sinv + qvp * cosv;
            kd[base + j]       = kv * cosv - kvp * sinv;
            kd[base + j + half] = kv * sinv + kvp * cosv;
        }
    }
}

/* Correct RoPE rotation using original values only (no in-place contamination).
   x shape [..., d], cos_table shape [S, d]. First d/2 entries of each row: cos, second d/2: sin.
   Returns new tensor same shape as x. */
/**
 * @brief Perform Tensor Rope.
 *
 * @param x [in] X value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tensor_rope(const SNEPPXTensor* x, const SNEPPXTensor* cos_table) {
    if (!x || !cos_table) return NULL;
    size_t d = x->shape[x->ndim - 1];
    size_t half = d / 2;
    size_t total = x->size;
    SNEPPXTensor* out = SNEPPX_tensor_copy(x);
    if (!out) return NULL;
    float* xd = (float*)x->data;
    float* od = (float*)out->data;
    float* cd = (float*)cos_table->data;
    size_t cd_last = cos_table->shape[cos_table->ndim - 1];
    for (size_t i = 0; i < total; i++) {
        size_t pos = i / d;
        size_t j = i % d;
        size_t pair = j % half;
        float cos_val = cd[pos * cd_last + pair];
        float sin_val = cd[pos * cd_last + half + pair];
        size_t base = pos * d;
        size_t partner = (j < half) ? (base + j + half) : (base + j - half);
        float x_i = xd[i];
        float x_p = xd[partner];
        if (j < half)
            od[i] = x_i * cos_val - x_p * sin_val;
        else
            od[i] = x_p * sin_val + x_i * cos_val;
    }
    return out;
}

/* ========== Batched Matmul ========== */

/**
 * @brief Perform Batched Matmul.
 *
 * @param a [in] A value.
 * @param b [in] B value.
 * @param transpose_b [in] Transpose B value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_batched_matmul(const SNEPPXTensor* a, const SNEPPXTensor* b,
                                 int transpose_b, int transpose_a) {
    (void)transpose_a;
    if (!a || !b || a->ndim != 4 || b->ndim != 4) return NULL;
    size_t B = a->shape[0], H = a->shape[1], S = a->shape[2];
    size_t D, S2;
    if (transpose_b) {
        /* a @ b^T: a=[B,H,S,D], b=[B,H,S2,D] → out=[B,H,S,S2] */
        D = a->shape[3];
        if (b->shape[3] != D) return NULL;
        S2 = b->shape[2];
    } else {
        /* a @ b: a=[B,H,S,D1], b=[B,H,D1,D2] → out=[B,H,S,D2] */
        D = a->shape[3];
        if (b->shape[2] != D) return NULL;
        S2 = b->shape[3];
    }
    size_t out_shape[] = {B, H, S, S2};
    SNEPPXTensor* out = SNEPPX_tensor_zeros(out_shape, 4, SNEPPX_FLOAT32);
    if (!out) return NULL;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float* od = (float*)out->data;
    if (transpose_b) {
        for (size_t b_ = 0; b_ < B; b_++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t s = 0; s < S; s++) {
                    for (size_t s2 = 0; s2 < S2; s2++) {
                        float sum = 0.0f;
                        for (size_t d = 0; d < D; d++) {
                            sum += ad[((b_ * H + h) * S + s) * D + d]
                                 * bd[((b_ * H + h) * S2 + s2) * D + d];
                        }
                        od[((b_ * H + h) * S + s) * S2 + s2] = sum;
                    }
                }
            }
        }
    } else {
        for (size_t b_ = 0; b_ < B; b_++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t s = 0; s < S; s++) {
                    for (size_t d2 = 0; d2 < S2; d2++) {
                        float sum = 0.0f;
                        for (size_t d = 0; d < D; d++) {
                            sum += ad[((b_ * H + h) * S + s) * D + d]
                                 * bd[((b_ * H + h) * D + d) * S2 + d2];
                        }
                        od[((b_ * H + h) * S + s) * S2 + d2] = sum;
                    }
                }
            }
        }
    }
    return out;
}

/* ========== Causal Mask ========== */

static SNEPPXTensor* causal_mask(size_t S) {
    size_t shape[] = {S, S};
    SNEPPXTensor* m = SNEPPX_tensor_create(shape, 2, SNEPPX_FLOAT32);
    if (!m) return NULL;
    float* d = (float*)m->data;
    for (size_t i = 0; i < S; i++) {
        for (size_t j = 0; j < S; j++)
            d[i * S + j] = (j <= i) ? 0.0f : -1e9f;
    }
    return m;
}

/* ========== 3D Matmul Helper ========== */

static SNEPPXTensor* matmul_3d(const SNEPPXTensor* a, const SNEPPXTensor* b) {
    if (!a || !b) return NULL;
    /* a: [B, S, D], b: [D, Dk] → result: [B, S, Dk] */
    size_t B = a->shape[0], S = a->shape[1], D = a->shape[2];
    size_t Dk = b->shape[1];
    size_t result_shape[] = {B, S, Dk};
    SNEPPXTensor* result = SNEPPX_tensor_zeros(result_shape, 3, SNEPPX_FLOAT32);
    if (!result) return NULL;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    float* rd = (float*)result->data;
    for (size_t batch = 0; batch < B; batch++) {
        for (size_t s = 0; s < S; s++) {
            for (size_t k = 0; k < Dk; k++) {
                float sum = 0.0f;
                for (size_t d = 0; d < D; d++)
                    sum += ad[(batch * S + s) * D + d] * bd[d * Dk + k];
                rd[(batch * S + s) * Dk + k] = sum;
            }
        }
    }
    return result;
}

/* ========== Attention Forward ========== */

/**
 * @brief Run the forward pass for Attn.
 *
 * @param w [in] W value.
 * @param x [in] X value.
 * @param cos_t [out] Cos T value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_attn_forward(const SNEPPXAttentionWeights* w, const SNEPPXTensor* x,
                               SNEPPXTensor* cos_t, SNEPPXTensor* sin_t) {
    if (!w || !x) return NULL;
    /* Accept a 2D [seq_len, d_model] input as a single batch element. */
    int orig_ndim = x->ndim;
    SNEPPXTensor* x_work = (SNEPPXTensor*)x;
    if (x->ndim == 2) {
        size_t shp3[] = {1, x->shape[0], x->shape[1]};
        x_work = SNEPPX_tensor_reshape(x, shp3, 3);
        if (!x_work) return NULL;
    }
    size_t B = x_work->shape[0], S = x_work->shape[1], D = w->config.d_model;
    size_t H = w->config.num_heads, hd = w->config.head_dim;

    /* Q, K, V projections: [B, S, D] → [B, S, H*hd] */
    SNEPPXTensor* q = matmul_3d(x_work, w->w_q);
    SNEPPXTensor* k = matmul_3d(x_work, w->w_k);
    SNEPPXTensor* v = matmul_3d(x_work, w->w_v);
    if (!q || !k || !v) {
        SNEPPX_tensor_destroy(q); SNEPPX_tensor_destroy(k); SNEPPX_tensor_destroy(v);
        return NULL;
    }

    /* Reshape to [B, H, S, hd] */
    size_t qkv_shape[] = {B, H, S, hd};
    SNEPPXTensor* q_r = SNEPPX_tensor_reshape(q, qkv_shape, 4);
    SNEPPXTensor* k_r = SNEPPX_tensor_reshape(k, qkv_shape, 4);
    SNEPPXTensor* v_r = SNEPPX_tensor_reshape(v, qkv_shape, 4);
    SNEPPX_tensor_destroy(q); SNEPPX_tensor_destroy(k); SNEPPX_tensor_destroy(v);

    /* RoPE */
    if (w->config.use_rope && cos_t && sin_t)
        SNEPPX_rope_apply(q_r, k_r, cos_t, sin_t, 0);

    /* Attention scores: Q @ K^T, [B, H, S, S] */
    SNEPPXTensor* scores = SNEPPX_batched_matmul(q_r, k_r, 1, 0);
    if (!scores) { SNEPPX_tensor_destroy(q_r); SNEPPX_tensor_destroy(k_r); SNEPPX_tensor_destroy(v_r); return NULL; }

    /* Scale */
    float scale = 1.0f / sqrtf((float)hd);
    float* sd = (float*)scores->data;
    for (size_t i = 0; i < scores->size; i++) sd[i] *= scale;

    /* Causal mask */
    if (w->config.use_causal_mask && S > 1) {
        SNEPPXTensor* mask = causal_mask(S);
        if (mask) {
            float* md = (float*)mask->data;
            for (size_t b = 0; b < B; b++)
                for (size_t h = 0; h < H; h++)
                    for (size_t i = 0; i < S; i++)
                        for (size_t j = 0; j < S; j++)
                            sd[((b * H + h) * S + i) * S + j] += md[i * S + j];
            SNEPPX_tensor_destroy(mask);
        }
    }

    /* Softmax over last dim */
    for (size_t i = 0; i < scores->size; i += S) {
        float maxv = sd[i];
        for (size_t j = 1; j < S; j++)
            if (sd[i + j] > maxv) maxv = sd[i + j];
        float sum = 0.0f;
        for (size_t j = 0; j < S; j++) {
            sd[i + j] = expf(sd[i + j] - maxv);
            sum += sd[i + j];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (size_t j = 0; j < S; j++) sd[i + j] *= inv;
    }

    /* Apply to V: scores @ V, [B, H, S, hd] */
    SNEPPXTensor* out = SNEPPX_batched_matmul(scores, v_r, 0, 0);
    SNEPPX_tensor_destroy(scores);
    if (!out) { SNEPPX_tensor_destroy(q_r); SNEPPX_tensor_destroy(k_r); SNEPPX_tensor_destroy(v_r); return NULL; }

    /* Reshape back to [B, S, D] */
    size_t out_sh[] = {B, S, D};
    SNEPPXTensor* out_r = SNEPPX_tensor_reshape(out, out_sh, 3);
    SNEPPX_tensor_destroy(out);

    /* Output projection: [B, S, D] → [B, S, D] */
    SNEPPXTensor* result = matmul_3d(out_r, w->w_o);
    SNEPPX_tensor_destroy(out_r);
    SNEPPX_tensor_destroy(q_r); SNEPPX_tensor_destroy(k_r); SNEPPX_tensor_destroy(v_r);
    if (orig_ndim == 2) {
        size_t shp2[] = {x->shape[0], x->shape[1]};
        SNEPPXTensor* squeezed = SNEPPX_tensor_reshape(result, shp2, 2);
        SNEPPX_tensor_destroy(result);
        if (x_work != x) SNEPPX_tensor_destroy(x_work);
        return squeezed;
    }
    if (x_work != x) SNEPPX_tensor_destroy(x_work);
    return result;
}

/* ========== KV Cache ========== */

/**
 * @brief Create Kv Cache.
 *
 * @param max_batch [in] Max Batch value.
 * @param max_seq [in] Max Seq value.
 * @param num_heads [in] Num Heads value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXKVCache* SNEPPX_kv_cache_create(size_t max_batch, size_t max_seq, size_t num_heads, size_t head_dim) {
    SNEPPXKVCache* c = SNEPPX_malloc(sizeof(SNEPPXKVCache), 64);
    if (!c) return NULL;
    size_t shape[] = {max_batch, num_heads, max_seq, head_dim};
    c->k_cache = SNEPPX_tensor_zeros(shape, 4, SNEPPX_FLOAT32);
    c->v_cache = SNEPPX_tensor_zeros(shape, 4, SNEPPX_FLOAT32);
    c->seq_len = 0;
    if (!c->k_cache || !c->v_cache) {
        SNEPPX_tensor_destroy(c->k_cache); SNEPPX_tensor_destroy(c->v_cache);
        SNEPPX_free(c, sizeof(SNEPPXKVCache)); return NULL;
    }
    return c;
}

/**
 * @brief Destroy Kv Cache.
 */
void SNEPPX_kv_cache_destroy(SNEPPXKVCache* c) {
    if (!c) return;
    SNEPPX_tensor_destroy(c->k_cache);
    SNEPPX_tensor_destroy(c->v_cache);
    SNEPPX_free(c, sizeof(SNEPPXKVCache));
}

/**
 * @brief Clear Kv Cache.
 */
void SNEPPX_kv_cache_clear(SNEPPXKVCache* c) {
    if (c) c->seq_len = 0;
}

/**
 * @brief Perform Attn Forward Cached.
 *
 * @param w [in] W value.
 * @param x [in] X value.
 * @param cache [out] Cache value.
 * @param cos [out] Cos value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_attn_forward_cached(const SNEPPXAttentionWeights* w, const SNEPPXTensor* x,
                                      SNEPPXKVCache* cache, SNEPPXTensor* cos, SNEPPXTensor* sin) {
    if (!w || !x || !cache) return NULL;
    size_t B = x->shape[0], S = x->shape[1], D = w->config.d_model;
    size_t H = w->config.num_heads, hd = w->config.head_dim;
    size_t offset = cache->seq_len;
    size_t total_seq = offset + S;

    /* Q, K, V */
    SNEPPXTensor* q = SNEPPX_tensor_matmul(x, w->w_q);
    SNEPPXTensor* k = SNEPPX_tensor_matmul(x, w->w_k);
    SNEPPXTensor* v = SNEPPX_tensor_matmul(x, w->w_v);

    size_t qkv_shape[] = {B, H, S, hd};
    SNEPPXTensor* q_r = SNEPPX_tensor_reshape(q, qkv_shape, 4);
    SNEPPXTensor* k_r = SNEPPX_tensor_reshape(k, qkv_shape, 4);
    SNEPPXTensor* v_r = SNEPPX_tensor_reshape(v, qkv_shape, 4);
    SNEPPX_tensor_destroy(q); SNEPPX_tensor_destroy(k); SNEPPX_tensor_destroy(v);

    if (w->config.use_rope && cos && sin)
        SNEPPX_rope_apply(q_r, k_r, cos, sin, offset);

    /* Append to cache */
    float* kd = (float*)k_r->data;
    float* vd = (float*)v_r->data;
    float* kcd = (float*)cache->k_cache->data;
    float* vcd = (float*)cache->v_cache->data;
    size_t K_cache = cache->k_cache->shape[2];
    for (size_t b = 0; b < B; b++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t s = 0; s < S; s++) {
                for (size_t d = 0; d < hd; d++) {
                    size_t dst = ((b * H + h) * K_cache + offset + s) * hd + d;
                    size_t src = ((b * H + h) * S + s) * hd + d;
                    kcd[dst] = kd[src];
                    vcd[dst] = vd[src];
                }
            }
        }
    }
    SNEPPX_tensor_destroy(k_r); SNEPPX_tensor_destroy(v_r);
    cache->seq_len = total_seq;

    /* Use full K,V from cache */
    size_t full_kv_shape[] = {B, H, total_seq, hd};
    SNEPPXTensor* k_full = SNEPPX_tensor_as_strided(cache->k_cache, 0, full_kv_shape, 4,
                                                 cache->k_cache->strides);
    SNEPPXTensor* v_full = SNEPPX_tensor_as_strided(cache->v_cache, 0, full_kv_shape, 4,
                                                 cache->v_cache->strides);

    /* Attention scores */
    SNEPPXTensor* scores = SNEPPX_batched_matmul(q_r, k_full, 1, 0);
    float scale = 1.0f / sqrtf((float)hd);
    float* sd = (float*)scores->data;
    for (size_t i = 0; i < scores->size; i++) sd[i] *= scale;

    /* Causal mask for full sequence */
    if (w->config.use_causal_mask && total_seq > 1) {
        for (size_t b = 0; b < B; b++) {
            for (size_t h = 0; h < H; h++) {
                for (size_t i = 0; i < S; i++) {
                    for (size_t j = 0; j < total_seq; j++) {
                        if (j > offset + i)
                            sd[((b * H + h) * S + i) * total_seq + j] = -1e9f;
                    }
                }
            }
        }
    }

    /* Softmax */
    for (size_t i = 0; i < scores->size; i += total_seq) {
        float maxv = sd[i];
        for (size_t j = 1; j < total_seq; j++)
            if (sd[i + j] > maxv) maxv = sd[i + j];
        float sum = 0.0f;
        for (size_t j = 0; j < total_seq; j++) {
            sd[i + j] = expf(sd[i + j] - maxv);
            sum += sd[i + j];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (size_t j = 0; j < total_seq; j++) sd[i + j] *= inv;
    }

    /* Apply to V */
    SNEPPXTensor* attn_out = SNEPPX_batched_matmul(scores, v_full, 0, 0);
    SNEPPX_tensor_destroy(scores);
    SNEPPX_tensor_destroy(k_full); SNEPPX_tensor_destroy(v_full);

    size_t out_shape[] = {B, S, D};
    SNEPPXTensor* out_r = SNEPPX_tensor_reshape(attn_out, out_shape, 3);
    SNEPPX_tensor_destroy(attn_out);

    SNEPPXTensor* result = SNEPPX_tensor_matmul(out_r, w->w_o);
    SNEPPX_tensor_destroy(out_r);
    SNEPPX_tensor_destroy(q_r);
    return result;
}

/* ========== Autodiff Training Graph ========== */

/**
 * @brief Perform Attn Build Train Graph.
 *
 * @param w [out] W value.
 * @param tape [out] Tape value.
 * @param input_var [out] Input Var value.
 * @param weight_vars [out] Weight Vars value.
 * @param num_weights [in] Num Weights value.
 * @param output_var [out] Output Var value.
 * @param cos [out] Cos value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_attn_build_train_graph(SNEPPXAttentionWeights* w, SNEPPXTape* tape,
                                 SNEPPXVariable* input_var,
                                 SNEPPXVariable** weight_vars, size_t num_weights,
                                 SNEPPXVariable** output_var,
                                 SNEPPXTensor* cos, SNEPPXTensor* sin) {
    if (!w || !tape || !input_var || !weight_vars || !output_var || num_weights < 8) return -1;
    SNEPPXVariable* wq_v = weight_vars[0]; SNEPPXVariable* bq_v = weight_vars[1];
    SNEPPXVariable* wk_v = weight_vars[2]; SNEPPXVariable* bk_v = weight_vars[3];
    SNEPPXVariable* wv_v = weight_vars[4]; SNEPPXVariable* bv_v = weight_vars[5];
    SNEPPXVariable* wo_v = weight_vars[6]; SNEPPXVariable* bo_v = weight_vars[7];
    (void)bq_v; (void)bk_v; (void)bv_v; (void)bo_v;

    size_t D = w->config.d_model, H = w->config.num_heads, hd = w->config.head_dim;
    size_t B = input_var->data->shape[0], S = input_var->data->shape[1];
    size_t dk = H * hd;

    /* Flatten input: [B, S, D] -> [B*S, D] */
    size_t flat_sh[] = {B * S, D};
    SNEPPXVariable* x_flat = SNEPPX_reshape(tape, input_var, flat_sh, 2);

    /* Q, K, V projections: [B*S, D] @ [D, dk] -> [B*S, dk] (all 2D matmuls) */
    SNEPPXVariable* q_flat = SNEPPX_matmul(tape, x_flat, wq_v);
    SNEPPXVariable* k_flat = SNEPPX_matmul(tape, x_flat, wk_v);
    SNEPPXVariable* v_flat = SNEPPX_matmul(tape, x_flat, wv_v);

    /* Single-head: Q, K are [B*S, dk]; K^T = [dk, B*S]; scores = Q @ K^T = [B*S, B*S] */
    size_t k_t_sh[] = {dk, B * S};
    SNEPPXVariable* k_t = SNEPPX_reshape(tape, k_flat, k_t_sh, 2);
    SNEPPXVariable* scores = SNEPPX_matmul(tape, q_flat, k_t);

    /* Scale by 1/sqrt(dk) */
    float scale = 1.0f / sqrtf((float)dk);
    size_t one_s[] = {1};
    SNEPPXTensor* scale_t = SNEPPX_tensor_ones(one_s, 1, SNEPPX_FLOAT32);
    if (!scale_t) return -1;
    ((float*)scale_t->data)[0] = scale;
    SNEPPXVariable* scale_v = SNEPPX_variable_create(scale_t, 0);
    SNEPPX_tape_record(tape, scale_v);
    scores = SNEPPX_mul(tape, scores, scale_v);

    /* Causal mask */
    if (w->config.use_causal_mask && S > 1) {
        size_t mask_sh[] = {B * S, B * S};
        SNEPPXTensor* mask_t = SNEPPX_tensor_create(mask_sh, 2, SNEPPX_FLOAT32);
        if (!mask_t) return -1;
        float* md = (float*)mask_t->data;
        for (size_t i = 0; i < B * S; i++)
            for (size_t j = 0; j < B * S; j++)
                md[i * (B * S) + j] = (j <= i) ? 0.0f : -1e9f;
        SNEPPXVariable* mask_v = SNEPPX_variable_create(mask_t, 0);
        SNEPPX_tape_record(tape, mask_v);
        scores = SNEPPX_add(tape, scores, mask_v);
    }

    /* Softmax over last dim */
    SNEPPXVariable* attn = SNEPPX_softmax(tape, scores, 1);

    /* attn @ V: [B*S, B*S] @ [B*S, dk] -> [B*S, dk] */
    SNEPPXVariable* ctx = SNEPPX_matmul(tape, attn, v_flat);

    /* Output projection: [B*S, dk] @ [dk, D] -> [B*S, D] */
    SNEPPXVariable* out_flat = SNEPPX_matmul(tape, ctx, wo_v);

    /* Reshape back to [B, S, D] */
    size_t out_sh[] = {B, S, D};
    *output_var = SNEPPX_reshape(tape, out_flat, out_sh, 3);
    return 0;
}

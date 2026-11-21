#include "subword_tokenization_pipeline.h"
#include "polymorphic_memory_allocator.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define MAX_TOKEN_LEN 256
/*
 * SNEPPX - Tokenizer
 *
 * WHAT
 *   Tokenizer.
 *
 * CONCEPT
 *   Tokenizer implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



typedef struct {
    int left_id;
    int right_id;
    int new_id;
} BpeMerge;

typedef struct {
    int* ids;
    int len;
    int cap;
} IntArray;

struct SNEPPXTokenizer {
    SNEPPXTokenizerType type;
    int vocab_size;
    int capacity;
    char* token_bytes;
    int* token_lens;
    BpeMerge* merges;
    int num_merges;
    int merge_capacity;
    SNEPPXSpecialTokens special;
};

static IntArray* intarr_create(int cap) {
    IntArray* a = SNEPPX_malloc(sizeof(IntArray), 64);
    if (!a) return NULL;
    a->ids = SNEPPX_malloc((size_t)cap * sizeof(int), 64);
    a->len = 0; a->cap = cap;
    if (!a->ids) { SNEPPX_free(a, sizeof(IntArray)); return NULL; }
    return a;
}

static void intarr_free(IntArray* a) {
    if (!a) return;
    SNEPPX_free(a->ids, (size_t)a->cap * sizeof(int));
    SNEPPX_free(a, sizeof(IntArray));
}

static int intarr_push(IntArray* a, int id) {
    if (a->len >= a->cap) {
        int new_cap = a->cap ? a->cap * 2 : 256;
        int* new_ids = SNEPPX_realloc(a->ids, (size_t)a->cap * sizeof(int), (size_t)new_cap * sizeof(int), 64);
        if (!new_ids) return -1;
        a->ids = new_ids; a->cap = new_cap;
    }
    a->ids[a->len++] = id;
    return 0;
}

/**
 * @brief Create Tokenizer.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTokenizer* SNEPPX_tokenizer_create(int vocab_size) {
    SNEPPXTokenizer* tok = SNEPPX_malloc(sizeof(SNEPPXTokenizer), 64);
    if (!tok) return NULL;
    tok->type = SNEPPX_TOK_BPE;
    tok->vocab_size = 0;
    tok->capacity = vocab_size > 256 ? vocab_size : 256;
    tok->token_bytes = SNEPPX_malloc((size_t)tok->capacity * MAX_TOKEN_LEN, 64);
    tok->token_lens = SNEPPX_malloc((size_t)tok->capacity * sizeof(int), 64);
    tok->merges = NULL;
    tok->num_merges = 0; tok->merge_capacity = 0;
    if (!tok->token_bytes || !tok->token_lens) {
        SNEPPX_free(tok->token_bytes, (size_t)tok->capacity * MAX_TOKEN_LEN);
        SNEPPX_free(tok->token_lens, (size_t)tok->capacity * sizeof(int));
        SNEPPX_free(tok, sizeof(SNEPPXTokenizer)); return NULL;
    }
    tok->special.pad_id = -1; tok->special.bos_id = -1;
    tok->special.eos_id = -1; tok->special.unk_id = -1;
    for (int i = 0; i < 256; i++) {
        tok->token_bytes[i * MAX_TOKEN_LEN] = (char)i;
        tok->token_lens[i] = 1;
    }
    tok->vocab_size = 256;
    return tok;
}

/**
 * @brief Destroy Tokenizer.
 */
void SNEPPX_tokenizer_destroy(SNEPPXTokenizer* tok) {
    if (!tok) return;
    SNEPPX_free(tok->token_bytes, (size_t)tok->capacity * MAX_TOKEN_LEN);
    SNEPPX_free(tok->token_lens, (size_t)tok->capacity * sizeof(int));
    if (tok->merges) SNEPPX_free(tok->merges, (size_t)tok->merge_capacity * sizeof(BpeMerge));
    SNEPPX_free(tok, sizeof(SNEPPXTokenizer));
}

/**
 * @brief Perform Tokenizer Vocab Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tokenizer_vocab_size(const SNEPPXTokenizer* tok) { return tok ? tok->vocab_size : 0; }
/**
 * @brief Perform Tokenizer Special.
 *
 * @return The result value, or 0 on error.
 */
SNEPPXSpecialTokens SNEPPX_tokenizer_special(const SNEPPXTokenizer* tok) { return tok ? tok->special : (SNEPPXSpecialTokens){-1,-1,-1,-1}; }
/**
 * @brief Perform Tokenizer Set Special.
 *
 * @param tok [out] Tok value.
 */
void SNEPPX_tokenizer_set_special(SNEPPXTokenizer* tok, SNEPPXSpecialTokens sp) { if (tok) tok->special = sp; }

static int find_token(const SNEPPXTokenizer* tok, const char* bytes, int len) {
    for (int i = 0; i < tok->vocab_size; i++)
        if (tok->token_lens[i] == len && memcmp(tok->token_bytes + (size_t)i * MAX_TOKEN_LEN, bytes, (size_t)len) == 0)
            return i;
    return -1;
}

/**
 * @brief Perform Tokenizer Add Token.
 *
 * @param tok [out] Tok value.
 * @param token [in] Token value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tokenizer_add_token(SNEPPXTokenizer* tok, const char* token, int id) {
    if (!tok || id < 0 || id >= tok->capacity) return -1;
    int len = (int)strlen(token);
    if (len > MAX_TOKEN_LEN - 1 || find_token(tok, token, len) >= 0) return 0;
    if (id >= tok->vocab_size) tok->vocab_size = id + 1;
    memcpy(tok->token_bytes + (size_t)id * MAX_TOKEN_LEN, token, (size_t)len);
    tok->token_bytes[(size_t)id * MAX_TOKEN_LEN + len] = '\0';
    tok->token_lens[id] = len;
    return 0;
}

/**
 * @brief Perform Tokenizer Encode.
 *
 * @param tok [in] Tok value.
 * @param text [in] Text value.
 * @param out_ids [out] Out Ids value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tokenizer_encode(const SNEPPXTokenizer* tok, const char* text, int* out_ids, size_t max_len) {
    if (!tok || !text || !out_ids || max_len == 0) return -1;
    size_t text_len = strlen(text);
    IntArray* ids = intarr_create((int)(text_len + 1));
    if (!ids) return -1;
    for (size_t i = 0; i < text_len; i++) {
        int id = find_token(tok, text + i, 1);
        if (id < 0) id = tok->special.unk_id >= 0 ? tok->special.unk_id : (int)text[i];
        intarr_push(ids, id);
    }
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int r = 0; r < tok->num_merges; r++) {
            int l = tok->merges[r].left_id;
            int ri = tok->merges[r].right_id;
            int nid = tok->merges[r].new_id;
            for (int i = 0; i < ids->len - 1; i++) {
                if (ids->ids[i] == l && ids->ids[i + 1] == ri) {
                    ids->ids[i] = nid;
                    for (int j = i + 1; j < ids->len - 1; j++)
                        ids->ids[j] = ids->ids[j + 1];
                    ids->len--;
                    changed = 1;
                    i--;
                }
            }
        }
    }
    size_t n = (size_t)ids->len < max_len ? (size_t)ids->len : max_len;
    memcpy(out_ids, ids->ids, n * sizeof(int));
    int result = ids->len;
    intarr_free(ids);
    return result;
}

/**
 * @brief Perform Tokenizer Decode.
 *
 * @param tok [in] Tok value.
 * @param ids [in] Ids value.
 *
 * @return Pointer on success, NULL on error.
 */
char* SNEPPX_tokenizer_decode(const SNEPPXTokenizer* tok, const int* ids, size_t len) {
    if (!tok || !ids || len == 0) return NULL;
    size_t total = 0;
    for (size_t i = 0; i < len; i++)
        if (ids[i] >= 0 && ids[i] < tok->vocab_size)
            total += (size_t)tok->token_lens[ids[i]];
    char* out = SNEPPX_malloc(total + 1, 64);
    if (!out) return NULL;
    size_t pos = 0;
    for (size_t i = 0; i < len; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->vocab_size) {
            memcpy(out + pos, tok->token_bytes + (size_t)id * MAX_TOKEN_LEN, (size_t)tok->token_lens[id]);
            pos += (size_t)tok->token_lens[id];
        }
    }
    out[pos] = '\0';
    return out;
}

static IntArray* text_to_ids_bytewise(SNEPPXTokenizer* tok, const char* text) {
    size_t len = strlen(text);
    IntArray* ids = intarr_create((int)(len + 1));
    if (!ids) return NULL;
    for (size_t i = 0; i < len; i++) {
        char c = text[i];
        int id = find_token(tok, &c, 1);
        if (id < 0) id = (unsigned char)c;
        intarr_push(ids, id);
    }
    return ids;
}

static void apply_merge(IntArray* ids, int left_id, int right_id, int new_id) {
    for (int i = 0; i < ids->len - 1; i++) {
        if (ids->ids[i] == left_id && ids->ids[i + 1] == right_id) {
            ids->ids[i] = new_id;
            for (int j = i + 1; j < ids->len - 1; j++)
                ids->ids[j] = ids->ids[j + 1];
            ids->len--;
            i--;
        }
    }
}

static void count_pairs(IntArray** corpus, size_t num_texts, long long* pair_counts,
                         int max_id, int* pair_buf) {
    (void)pair_buf;
    for (size_t t = 0; t < num_texts; t++) {
        IntArray* ids = corpus[t];
        for (int i = 0; i < ids->len - 1; i++) {
            int l = ids->ids[i], r = ids->ids[i + 1];
            if (l >= 0 && l < max_id && r >= 0 && r < max_id)
                pair_counts[(size_t)l * (size_t)max_id + (size_t)r]++;
        }
    }
}

/**
 * @brief Perform Tokenizer Train Bpe.
 *
 * @param texts [in] Texts value.
 * @param num_texts [in] Num Texts value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTokenizer* SNEPPX_tokenizer_train_bpe(const char** texts, size_t num_texts, size_t target_vocab) {
    if (!texts || num_texts == 0 || target_vocab <= 256) return NULL;
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create((int)target_vocab + 256);
    if (!tok) return NULL;
    IntArray** corpus = SNEPPX_malloc(num_texts * sizeof(IntArray*), 64);
    if (!corpus) { SNEPPX_tokenizer_destroy(tok); return NULL; }
    for (size_t t = 0; t < num_texts; t++) {
        corpus[t] = text_to_ids_bytewise(tok, texts[t]);
        if (!corpus[t]) {
            for (size_t k = 0; k < t; k++) intarr_free(corpus[k]);
            SNEPPX_free(corpus, num_texts * sizeof(IntArray*));
            SNEPPX_tokenizer_destroy(tok); return NULL;
        }
    }
    int max_merges = (int)target_vocab - 256;
    int current_max_id = 256;
    for (int m = 0; m < max_merges; m++) {
        int pair_dim = current_max_id;
        long long* pair_counts = SNEPPX_malloc((size_t)pair_dim * (size_t)pair_dim * sizeof(long long), 64);
        if (!pair_counts) break;
        memset(pair_counts, 0, (size_t)pair_dim * (size_t)pair_dim * sizeof(long long));
        count_pairs(corpus, num_texts, pair_counts, pair_dim, NULL);
        long long best_count = 0;
        int best_l = -1, best_r = -1;
        for (int l = 0; l < pair_dim; l++) {
            for (int r = 0; r < pair_dim; r++) {
                long long c = pair_counts[(size_t)l * (size_t)pair_dim + (size_t)r];
                if (c > best_count) { best_count = c; best_l = l; best_r = r; }
            }
        }
        SNEPPX_free(pair_counts, (size_t)pair_dim * (size_t)pair_dim * sizeof(long long));
        if (best_l < 0 || best_count == 0) break;
        int new_id = tok->vocab_size;
        if (new_id >= tok->capacity) break;
        int l_len = tok->token_lens[best_l];
        int r_len = tok->token_lens[best_r];
        char* new_bytes = tok->token_bytes + (size_t)new_id * MAX_TOKEN_LEN;
        memcpy(new_bytes, tok->token_bytes + (size_t)best_l * MAX_TOKEN_LEN, (size_t)l_len);
        memcpy(new_bytes + l_len, tok->token_bytes + (size_t)best_r * MAX_TOKEN_LEN, (size_t)r_len);
        new_bytes[l_len + r_len] = '\0';
        tok->token_lens[new_id] = l_len + r_len;
        tok->vocab_size++;
        if (tok->num_merges >= tok->merge_capacity) {
            int nc = tok->merge_capacity ? tok->merge_capacity * 2 : 1024;
            BpeMerge* nm = SNEPPX_realloc(tok->merges, (size_t)tok->merge_capacity * sizeof(BpeMerge),
                                         (size_t)nc * sizeof(BpeMerge), 64);
            if (!nm) break;
            tok->merges = nm; tok->merge_capacity = nc;
        }
        tok->merges[tok->num_merges].left_id = best_l;
        tok->merges[tok->num_merges].right_id = best_r;
        tok->merges[tok->num_merges].new_id = new_id;
        tok->num_merges++;
        current_max_id = new_id + 1;
        for (size_t t = 0; t < num_texts; t++)
            apply_merge(corpus[t], best_l, best_r, new_id);
    }
    for (size_t t = 0; t < num_texts; t++) intarr_free(corpus[t]);
    SNEPPX_free(corpus, num_texts * sizeof(IntArray*));
    return tok;
}

/**
 * @brief Save Tokenizer.
 *
 * @param tok [in] Tok value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tokenizer_save(const SNEPPXTokenizer* tok, const char* path) {
    if (!tok || !path) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t magic = 0x4250544F;
    fwrite(&magic, sizeof(magic), 1, f);
    int vs = tok->vocab_size;
    fwrite(&vs, sizeof(vs), 1, f);
    fwrite(&tok->num_merges, sizeof(tok->num_merges), 1, f);
    fwrite(&tok->special, sizeof(tok->special), 1, f);
    for (int i = 0; i < tok->vocab_size; i++) {
        fwrite(&tok->token_lens[i], sizeof(int), 1, f);
        if (tok->token_lens[i] > 0)
            fwrite(tok->token_bytes + (size_t)i * MAX_TOKEN_LEN, 1, (size_t)tok->token_lens[i], f);
    }
    for (int i = 0; i < tok->num_merges; i++) {
        fwrite(&tok->merges[i].left_id, sizeof(int), 1, f);
        fwrite(&tok->merges[i].right_id, sizeof(int), 1, f);
        fwrite(&tok->merges[i].new_id, sizeof(int), 1, f);
    }
    fclose(f);
    return 0;
}

/**
 * @brief Load Tokenizer.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTokenizer* SNEPPX_tokenizer_load(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x4250544F) { fclose(f); return NULL; }
    int vs, nm;
    if (fread(&vs, sizeof(vs), 1, f) != 1 || fread(&nm, sizeof(nm), 1, f) != 1) { fclose(f); return NULL; }
    SNEPPXTokenizer* tok = SNEPPX_tokenizer_create(vs > 256 ? vs : 256);
    if (!tok) { fclose(f); return NULL; }
    fread(&tok->special, sizeof(tok->special), 1, f);
    tok->vocab_size = vs;
    for (int i = 0; i < vs; i++) {
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) { SNEPPX_tokenizer_destroy(tok); fclose(f); return NULL; }
        tok->token_lens[i] = len;
        if (len > 0 && fread(tok->token_bytes + (size_t)i * MAX_TOKEN_LEN, 1, (size_t)len, f) != (size_t)len) {
            SNEPPX_tokenizer_destroy(tok); fclose(f); return NULL;
        }
    }
    for (int i = 0; i < nm; i++) {
        BpeMerge m;
        if (fread(&m.left_id, sizeof(int), 1, f) != 1 || fread(&m.right_id, sizeof(int), 1, f) != 1
            || fread(&m.new_id, sizeof(int), 1, f) != 1) {
            SNEPPX_tokenizer_destroy(tok); fclose(f); return NULL;
        }
        if (tok->num_merges >= tok->merge_capacity) {
            int nc = tok->merge_capacity ? tok->merge_capacity * 2 : 1024;
            BpeMerge* nm2 = SNEPPX_realloc(tok->merges, (size_t)tok->merge_capacity * sizeof(BpeMerge),
                                          (size_t)nc * sizeof(BpeMerge), 64);
            if (!nm2) break;
            tok->merges = nm2; tok->merge_capacity = nc;
        }
        tok->merges[tok->num_merges++] = m;
    }
    fclose(f);
    return tok;
}

/**
 * @brief Perform Tokenizer Ids To Tensor.
 *
 * @param tok [in] Tok value.
 * @param ids [in] Ids value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTensor* SNEPPX_tokenizer_ids_to_tensor(const SNEPPXTokenizer* tok, const int* ids, size_t len) {
    (void)tok;
    size_t shape[] = {1, len};
    SNEPPXTensor* t = SNEPPX_tensor_create(shape, 2, SNEPPX_INT32);
    if (!t) return NULL;
    memcpy(t->data, ids, len * sizeof(int));
    return t;
}

/**
 * @brief Perform Tokenizer Tensor To Ids.
 *
 * @param t [in] T value.
 *
 * @return 0 on success, -1 on error.
 */
int* SNEPPX_tokenizer_tensor_to_ids(const SNEPPXTensor* t, size_t* out_len) {
    if (!t || t->dtype != SNEPPX_INT32) return NULL;
    size_t n = t->size;
    int* ids = SNEPPX_malloc(n * sizeof(int), 64);
    if (!ids) return NULL;
    memcpy(ids, t->data, n * sizeof(int));
    *out_len = n;
    return ids;
}

#include "numpy_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * SNEPPX - Numpy Format
 *
 * WHAT
 *   Numpy Format.
 *
 * CONCEPT
 *   Provides the Numpy Format.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Real NumPy .npy / .npz (stored ZIP) reader-writer. Little-endian. */

typedef enum {
    NPY_FLOAT32 = 0,
    NPY_FLOAT64,
    NPY_FLOAT16,
    NPY_INT64,
    NPY_INT32,
    NPY_INT16,
    NPY_INT8,
    NPY_UINT8,
    NPY_BOOL
} NPYDType;

typedef struct {
    void* data;
    size_t shape[8];
    size_t ndim;
    size_t size; /* bytes */
    int dtype;
} NPYArray;

static size_t npy_esize(int dtype) {
    switch (dtype) {
        case NPY_FLOAT64: return 8;
        case NPY_INT64: return 8;
        case NPY_FLOAT32: return 4;
        case NPY_INT32: return 4;
        case NPY_FLOAT16: return 2;
        case NPY_INT16: return 2;
        case NPY_INT8: return 1;
        case NPY_UINT8: return 1;
        case NPY_BOOL: return 1;
        default: return 0;
    }
}

static const char* npy_descr(int dtype) {
    switch (dtype) {
        case NPY_FLOAT64: return "<f8";
        case NPY_FLOAT32: return "<f4";
        case NPY_FLOAT16: return "<f2";
        case NPY_INT64: return "<i8";
        case NPY_INT32: return "<i4";
        case NPY_INT16: return "<i2";
        case NPY_INT8: return "<i1";
        case NPY_UINT8: return "|u1";
        case NPY_BOOL: return "|b1";
        default: return "<f4";
    }
}

static int npy_dtype_from_descr(const char* d) {
    if (!strcmp(d, "<f8")) return NPY_FLOAT64;
    if (!strcmp(d, "<f4")) return NPY_FLOAT32;
    if (!strcmp(d, "<f2")) return NPY_FLOAT16;
    if (!strcmp(d, "<i8")) return NPY_INT64;
    if (!strcmp(d, "<i4")) return NPY_INT32;
    if (!strcmp(d, "<i2")) return NPY_INT16;
    if (!strcmp(d, "<i1")) return NPY_INT8;
    if (!strcmp(d, "|u1")) return NPY_UINT8;
    if (!strcmp(d, "|b1")) return NPY_BOOL;
    return NPY_FLOAT32;
}

static char* read_whole(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char* buf = (char*)malloc((size_t)sz ? (size_t)sz : 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

/* Parse a Python-dict header: find 'descr' and 'shape'. */
static int parse_npy_header(const char* hdr, size_t hdr_len, int* dtype, size_t* shape, size_t* ndim) {
    char descr[16];
    descr[0] = '\0';
    const char* p = hdr;
    const char* end = hdr + hdr_len;
    /* descr */
    const char* d = strstr(p, "'descr'");
    if (d && d + 8 < end) {
        const char* q = strchr(d + 7, ':');
        if (q) {
            q = strchr(q, '\'');
            if (q) {
                q++;
                size_t i = 0;
                while (*q && *q != '\'' && i + 1 < sizeof(descr)) descr[i++] = *q++;
                descr[i] = '\0';
            }
        }
    }
    if (descr[0]) *dtype = npy_dtype_from_descr(descr);
    /* shape: 'shape': (a, b, ...) */
    size_t n = 0;
    const char* s = strstr(p, "'shape'");
    if (s && s + 7 < end) {
        const char* q = strchr(s + 7, '(');
        if (q) {
            q++;
            for (;;) {
                while (*q == ' ' || *q == '\n' || *q == '\t') q++;
                if (*q == ')' || *q == 0) break;
                long v = strtol(q, (char**)&q, 10);
                if (n < 8) shape[n++] = (size_t)v;
                while (*q == ' ' || *q == ',') q++;
            }
        }
    }
    *ndim = n;
    return 0;
}

/**
 * @brief Load Npy.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npy_load(const char* path) {
    size_t len = 0;
    char* buf = read_whole(path, &len);
    if (!buf || len < 10) { free(buf); return NULL; }
    if (buf[0] != '\x93' || buf[1] != 'N' || buf[2] != 'U' || buf[3] != 'M' || buf[4] != 'P' || buf[5] != 'Y') {
        free(buf); return NULL;
    }
    int vmajor = (unsigned char)buf[6];
    int hlen = (vmajor == 1) ? (buf[8] | (buf[9] << 8))
                             : (buf[8] | (buf[9] << 8) | (buf[10] << 16) | (buf[11] << 24));
    size_t header_off = (vmajor == 1) ? 10 : 12;
    if (header_off + (size_t)hlen > len) { free(buf); return NULL; }
    NPYArray* a = (NPYArray*)calloc(1, sizeof(NPYArray));
    if (!a) { free(buf); return NULL; }
    int dtype = NPY_FLOAT32;
    size_t shape[8] = {0}, ndim = 0;
    parse_npy_header(buf + header_off, (size_t)hlen, &dtype, shape, &ndim);
    a->dtype = dtype;
    a->ndim = ndim;
    for (size_t i = 0; i < ndim; i++) a->shape[i] = shape[i];
    size_t esz = npy_esize(dtype);
    size_t total = esz ? esz : 1;
    for (size_t i = 0; i < ndim; i++) total *= (shape[i] ? shape[i] : 1);
    a->size = total;
    size_t data_off = header_off + (size_t)hlen;
    if (data_off + total > len) { free(a); free(buf); return NULL; }
    a->data = malloc(total ? total : 1);
    if (!a->data) { free(a); free(buf); return NULL; }
    memcpy(a->data, buf + data_off, total);
    free(buf);
    return a;
}

/**
 * @brief Destroy Npy.
 */
void SNEPPX_npy_destroy(void* arr) {
    NPYArray* a = (NPYArray*)arr;
    if (!a) return;
    free(a->data);
    free(a);
}

/**
 * @brief Perform Npy Get Data.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_npy_get_data(void* arr) { NPYArray* a = (NPYArray*)arr; return a ? a->data : NULL; }
/**
 * @brief Perform Npy Get Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_npy_get_size(void* arr) { NPYArray* a = (NPYArray*)arr; return a ? a->size : 0; }
/**
 * @brief Perform Npy Get Ndim.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npy_get_ndim(void* arr) { NPYArray* a = (NPYArray*)arr; return a ? (int)a->ndim : 0; }
/**
 * @brief Perform Npy Get Shape.
 *
 * @return The computed size/count, or 0 on error.
 */
const size_t* SNEPPX_npy_get_shape(void* arr) { NPYArray* a = (NPYArray*)arr; return a ? a->shape : NULL; }
/**
 * @brief Perform Npy Get Dtype.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npy_get_dtype(void* arr) { NPYArray* a = (NPYArray*)arr; return a ? a->dtype : 0; }

/**
 * @brief Save Npy.
 *
 * @param path [in] Path value.
 * @param data [in] Data value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npy_save(const char* path, const void* data, const size_t* shape, size_t ndim, int dtype) {
    if (!path || !data) return -1;
    size_t total = npy_esize(dtype);
    total = total ? total : 1;
    for (size_t i = 0; i < ndim; i++) total *= (shape[i] ? shape[i] : 1);
    /* Build header. */
    char hdr[256];
    size_t hl = (size_t)snprintf(hdr, sizeof(hdr),
        "{'descr': '%s', 'fortran_order': False, 'shape': (", npy_descr(dtype));
    for (size_t i = 0; i < ndim; i++) {
        if (i) hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, ", %zu", shape[i]);
        else hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, "%zu", shape[i]);
    }
    if (ndim == 1) hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, ",)");
    else hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, ")");
    hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, "}\n");
    /* header length must make (10 + hlen) % 64 == 0 for v1. */
    size_t need = 10 + hl;
    size_t pad = (64 - (need % 64)) % 64;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char magic[8];
    magic[0] = 0x93; magic[1] = 'N'; magic[2] = 'U'; magic[3] = 'M';
    magic[4] = 'P'; magic[5] = 'Y'; magic[6] = 1; magic[7] = 0;
    if (fwrite(magic, 1, 8, f) != 8) { fclose(f); return -1; }
    unsigned short hl16 = (unsigned short)(hl + pad);
    if (fwrite(&hl16, 1, 2, f) != 2) { fclose(f); return -1; }
    if (fwrite(hdr, 1, hl, f) != hl) { fclose(f); return -1; }
    for (size_t i = 0; i < pad; i++) { char c = ' '; fwrite(&c, 1, 1, f); }
    if (total && fwrite(data, 1, total, f) != total) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

/* ---- minimal stored-ZIP support for .npz ---- */
static void* npy_from_bytes(const unsigned char* b, size_t blen, int* dtype, size_t* shape, size_t* ndim) {
    if (blen < 10 || b[0] != 0x93 || b[1] != 'N') return NULL;
    int vmajor = b[6];
    int hlen = (vmajor == 1) ? (b[8] | (b[9] << 8)) : (b[8] | (b[9] << 8) | (b[10] << 16) | (b[11] << 24));
    size_t header_off = (vmajor == 1) ? 10 : 12;
    int dt = NPY_FLOAT32; size_t sh[8] = {0}, nd = 0;
    parse_npy_header((const char*)(b + header_off), (size_t)hlen, &dt, sh, &nd);
    size_t esz = npy_esize(dt);
    size_t total = esz ? esz : 1;
    for (size_t i = 0; i < nd; i++) total *= (sh[i] ? sh[i] : 1);
    size_t data_off = header_off + (size_t)hlen;
    if (data_off + total > blen) return NULL;
    NPYArray* a = (NPYArray*)calloc(1, sizeof(NPYArray));
    if (!a) return NULL;
    a->dtype = dt; a->ndim = nd;
    for (size_t i = 0; i < nd; i++) a->shape[i] = sh[i];
    a->size = total;
    a->data = malloc(total ? total : 1);
    if (!a->data) { free(a); return NULL; }
    memcpy(a->data, b + data_off, total);
    if (dtype) *dtype = dt;
    if (shape) for (size_t i = 0; i < nd; i++) shape[i] = sh[i];
    if (ndim) *ndim = nd;
    return a;
}

/**
 * @brief Load Npz.
 *
 * @param path [in] Path value.
 * @param keys [out] Keys value.
 * @param arrays [out] Arrays value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npz_load(const char* path, char*** keys, void*** arrays, size_t* count) {
    size_t len = 0;
    char* buf = read_whole(path, &len);
    if (!buf) return -1;
    size_t cap = 8;
    char** kk = (char**)calloc(cap, sizeof(char*));
    void** aa = (void**)calloc(cap, sizeof(void*));
    size_t n = 0;
    size_t pos = 0;
    while (pos + 30 <= len) {
        unsigned sig = *(unsigned*)(buf + pos);
        if (sig != 0x04034b50u) break; /* reached central dir */
        unsigned short method = *(unsigned short*)(buf + pos + 8);
        unsigned comp_size = *(unsigned*)(buf + pos + 18);
        unsigned fname_len = *(unsigned short*)(buf + pos + 26);
        unsigned extra_len = *(unsigned short*)(buf + pos + 28);
        const char* fname = buf + pos + 30;
        size_t data_off = pos + 30 + fname_len + extra_len;
        if (data_off + comp_size > len) break;
        if (method == 0) { /* stored */
            char* name = (char*)malloc(fname_len + 1);
            memcpy(name, fname, fname_len);
            name[fname_len] = '\0';
            /* strip .npy extension for key */
            void* arr = npy_from_bytes((const unsigned char*)(buf + data_off), comp_size, NULL, NULL, NULL);
            if (n >= cap) { cap *= 2; kk = (char**)realloc(kk, cap*sizeof(char*)); aa = (void**)realloc(aa, cap*sizeof(void*)); }
            kk[n] = name; aa[n] = arr; n++;
        }
        pos = data_off + comp_size;
    }
    free(buf);
    *count = n; *keys = kk; *arrays = aa;
    return (int)n;
}

/**
 * @brief Save Npz.
 *
 * @param path [in] Path value.
 * @param keys [in] Keys value.
 * @param data_ptrs [in] Data Ptrs value.
 * @param shapes [in] Shapes value.
 * @param ndims [in] Ndims value.
 * @param dtypes [in] Dtypes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npz_save(const char* path, const char** keys, const void** data_ptrs, const size_t** shapes, const size_t* ndims, const int* dtypes, size_t count) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    /* Build each .npy in memory. */
    size_t* entry_off = (size_t*)calloc(count ? count : 1, sizeof(size_t));
    size_t* entry_len = (size_t*)calloc(count ? count : 1, sizeof(size_t));
    unsigned char** entry_data = (unsigned char**)calloc(count ? count : 1, sizeof(unsigned char*));
    unsigned cd_offset = 0;
    for (size_t i = 0; i < count; i++) {
        /* serialize npy to memory buffer */
        size_t total = npy_esize(dtypes[i]); total = total ? total : 1;
        for (size_t j = 0; j < ndims[i]; j++) total *= (shapes[i][j] ? shapes[i][j] : 1);
        char hdr[256];
        size_t hl = (size_t)snprintf(hdr, sizeof(hdr), "{'descr': '%s', 'fortran_order': False, 'shape': (", npy_descr(dtypes[i]));
        for (size_t j = 0; j < ndims[i]; j++) {
            if (j) hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, ", %zu", shapes[i][j]);
            else hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, "%zu", shapes[i][j]);
        }
        if (ndims[i] == 1) hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, ",)");
        else hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, ")");
        hl += (size_t)snprintf(hdr + hl, sizeof(hdr) - hl, "}\n");
        size_t need = 10 + hl; size_t pad = (64 - (need % 64)) % 64;
        size_t blen = 10 + hl + pad + total;
        unsigned char* b = (unsigned char*)malloc(blen);
        b[0] = 0x93; b[1]='N'; b[2]='U'; b[3]='M'; b[4]='P'; b[5]='Y'; b[6]=1; b[7]=0;
        unsigned short hl16 = (unsigned short)(hl + pad);
        memcpy(b + 8, &hl16, 2);
        memcpy(b + 10, hdr, hl);
        for (size_t p = 0; p < pad; p++) b[10 + hl + p] = ' ';
        memcpy(b + 10 + hl + pad, data_ptrs[i], total);
        /* local file header */
        unsigned sig = 0x04034b50u;
        unsigned short vneed = 20, method = 0, flags = 0;
        unsigned comp = (unsigned)total;
        unsigned fname_len = (unsigned)strlen(keys[i]) + 4; /* add .npy */
        char* fname = (char*)malloc(fname_len + 1);
        snprintf(fname, fname_len + 1, "%s.npy", keys[i]);
        fwrite(&sig, 1, 4, f);
        fwrite(&vneed, 1, 2, f); fwrite(&flags, 1, 2, f); fwrite(&method, 1, 2, f);
        unsigned short t0 = 0, d0 = 0; unsigned crc = 0;
        fwrite(&t0, 1, 2, f); fwrite(&d0, 1, 2, f);
        fwrite(&crc, 1, 4, f);
        fwrite(&comp, 1, 4, f); fwrite(&comp, 1, 4, f);
        fwrite(&fname_len, 1, 2, f);
        unsigned short extra = 0;
        fwrite(&extra, 1, 2, f);
        fwrite(fname, 1, fname_len, f);
        fwrite(b, 1, blen, f);
        entry_off[i] = cd_offset;
        entry_len[i] = (size_t)blen;
        entry_data[i] = b;
        cd_offset = (unsigned)(cd_offset + 30 + fname_len + blen);
        free(fname);
    }
    /* central directory */
    unsigned cd_start = (unsigned)cd_offset;
    for (size_t i = 0; i < count; i++) {
        unsigned sig = 0x02014b50u;
        unsigned short vmade = 20, vneed = 20, flags = 0, method = 0;
        unsigned short t0 = 0, d0 = 0; unsigned crc = 0;
        unsigned comp = (unsigned)entry_len[i];
        unsigned short fname_len = (unsigned short)(strlen(keys[i]) + 4);
        char* fname = (char*)malloc(fname_len + 1);
        snprintf(fname, fname_len + 1, "%s.npy", keys[i]);
        unsigned short extra = 0, comment = 0;
        unsigned short disk = 0, iattr = 0;
        unsigned attr = 0;
        fwrite(&sig, 1, 4, f);
        fwrite(&vmade, 1, 2, f); fwrite(&vneed, 1, 2, f); fwrite(&flags, 1, 2, f); fwrite(&method, 1, 2, f);
        fwrite(&t0, 1, 2, f); fwrite(&d0, 1, 2, f);
        fwrite(&crc, 1, 4, f);
        fwrite(&comp, 1, 4, f); fwrite(&comp, 1, 4, f);
        fwrite(&fname_len, 1, 2, f); fwrite(&extra, 1, 2, f); fwrite(&comment, 1, 2, f);
        fwrite(&disk, 1, 2, f); fwrite(&iattr, 1, 2, f); fwrite(&attr, 1, 4, f);
        fwrite(&entry_off[i], 1, 4, f);
        fwrite(fname, 1, fname_len, f);
        free(fname);
    }
    /* end of central directory */
    unsigned sig = 0x06054b50u;
    unsigned short disk = 0, cd_disk = 0;
    unsigned short n_this = (unsigned short)count, n_total = (unsigned short)count;
    unsigned cd_size = (unsigned)(cd_offset - cd_start);
    fwrite(&sig, 1, 4, f);
    fwrite(&disk, 1, 2, f); fwrite(&cd_disk, 1, 2, f);
    fwrite(&n_this, 1, 2, f); fwrite(&n_total, 1, 2, f);
    fwrite(&cd_size, 1, 4, f); fwrite(&cd_start, 1, 4, f);
    unsigned short clen = 0; fwrite(&clen, 1, 2, f);
    fclose(f);
    for (size_t i = 0; i < count; i++) free(entry_data[i]);
    free(entry_off); free(entry_len); free(entry_data);
    return 0;
}

/**
 * @brief Free Npz.
 *
 * @param keys [out] Keys value.
 * @param arrays [out] Arrays value.
 */
void SNEPPX_npz_free(char** keys, void** arrays, size_t count) {
    for (size_t i = 0; i < count; i++) {
        free(keys[i]);
        SNEPPX_npy_destroy(arrays[i]);
    }
    free(keys);
    free(arrays);
}

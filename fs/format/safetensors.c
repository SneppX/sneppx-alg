#include "safetensors.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * SNEPPX - Safetensors
 *
 * WHAT
 *   Safetensors.
 *
 * CONCEPT
 *   Provides tensor operations.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */

/* Portable string duplication for MSVC and POSIX toolchains. */
static char* safetensors_strdup(const char* value)
{
    if (!value)
        return NULL;
    const size_t length = strlen(value) + 1;
    char* copy = (char*) malloc(length);
    if (copy)
        memcpy(copy, value, length);
    return copy;
}

/* Minimal safetensors implementation (little-endian).
 *
 * Layout: uint64 header_length | utf-8 JSON header | byte buffer
 * Header JSON: { name: { dtype, shape:[..], data_offsets:[start,end] }, ... ,
 *                "__metadata__": { ... } }  (offsets are relative to the byte
 *                buffer that follows the header). */

/* Safe append to a growable buffer. Grows *cap if needed. */
static void safe_append(char** buf, size_t* len, size_t* cap, const char* fmt, ...)
{
    va_list args;
    while (1)
    {
        if (*len + 256 >= *cap)
        {
            *cap *= 2;
            char* nw = (char*) realloc(*buf, *cap);
            if (!nw)
                return;
            *buf = nw;
        }
        va_start(args, fmt);
        int n = vsnprintf(*buf + *len, *cap - *len, fmt, args);
        va_end(args);
        if (n < 0)
            return;
        if ((size_t) n < *cap - *len)
        {
            *len += (size_t) n;
            return;
        }
        /* Output was truncated, grow and retry */
        *cap *= 2;
    }
}

typedef struct
{
    char* name;
    int dtype;
    size_t ndim;
    size_t shape[8];
    size_t off_start;
    size_t off_end;
} STEntry;

typedef struct
{
    char* path;
    char mode; /* 'r' or 'w' */
    unsigned char* buf;
    size_t buf_len;
    size_t buf_cap;
    /* parsed (read) */
    STEntry* entries;
    size_t n_entries;
    char* meta_json;
    /* write accumulation */
    STEntry* w_entries;
    size_t w_cap;
    size_t w_count;
    unsigned char* w_data;
    size_t w_data_len;
    size_t w_data_cap;
} STHandle;

/**
 * @brief Perform Safetensors Dtype Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_safetensors_dtype_size(int dtype)
{
    switch (dtype)
    {
        case SNEPPX_ST_DTYPE_F64:
            return 8;
        case SNEPPX_ST_DTYPE_I64:
            return 8;
        case SNEPPX_ST_DTYPE_F32:
            return 4;
        case SNEPPX_ST_DTYPE_I32:
            return 4;
        case SNEPPX_ST_DTYPE_F16:
            return 2;
        case SNEPPX_ST_DTYPE_BF16:
            return 2;
        case SNEPPX_ST_DTYPE_F8_E5M2:
            return 1;
        case SNEPPX_ST_DTYPE_F8_E4M3FN:
            return 1;
        case SNEPPX_ST_DTYPE_I16:
            return 2;
        case SNEPPX_ST_DTYPE_I8:
            return 1;
        case SNEPPX_ST_DTYPE_U8:
            return 1;
        case SNEPPX_ST_DTYPE_BOOL:
            return 1;
        default:
            return 0;
    }
}

static const char* dtype_to_str(int dtype)
{
    switch (dtype)
    {
        case SNEPPX_ST_DTYPE_F64:
            return "F64";
        case SNEPPX_ST_DTYPE_F32:
            return "F32";
        case SNEPPX_ST_DTYPE_F16:
            return "F16";
        case SNEPPX_ST_DTYPE_BF16:
            return "BF16";
        case SNEPPX_ST_DTYPE_I64:
            return "I64";
        case SNEPPX_ST_DTYPE_I32:
            return "I32";
        case SNEPPX_ST_DTYPE_I16:
            return "I16";
        case SNEPPX_ST_DTYPE_I8:
            return "I8";
        case SNEPPX_ST_DTYPE_U8:
            return "U8";
        case SNEPPX_ST_DTYPE_BOOL:
            return "BOOL";
        case SNEPPX_ST_DTYPE_F8_E5M2:
            return "F8_E5M2";
        case SNEPPX_ST_DTYPE_F8_E4M3FN:
            return "F8_E4M3FN";
        default:
            return "F32";
    }
}

static int dtype_from_str(const char* s)
{
    if (!s)
        return SNEPPX_ST_DTYPE_F32;
    if (!strcmp(s, "F64"))
        return SNEPPX_ST_DTYPE_F64;
    if (!strcmp(s, "F32"))
        return SNEPPX_ST_DTYPE_F32;
    if (!strcmp(s, "F16"))
        return SNEPPX_ST_DTYPE_F16;
    if (!strcmp(s, "BF16"))
        return SNEPPX_ST_DTYPE_BF16;
    if (!strcmp(s, "I64"))
        return SNEPPX_ST_DTYPE_I64;
    if (!strcmp(s, "I32"))
        return SNEPPX_ST_DTYPE_I32;
    if (!strcmp(s, "I16"))
        return SNEPPX_ST_DTYPE_I16;
    if (!strcmp(s, "I8"))
        return SNEPPX_ST_DTYPE_I8;
    if (!strcmp(s, "U8"))
        return SNEPPX_ST_DTYPE_U8;
    if (!strcmp(s, "BOOL"))
        return SNEPPX_ST_DTYPE_BOOL;
    if (!strcmp(s, "F8_E5M2"))
        return SNEPPX_ST_DTYPE_F8_E5M2;
    if (!strcmp(s, "F8_E4M3FN"))
        return SNEPPX_ST_DTYPE_F8_E4M3FN;
    return SNEPPX_ST_DTYPE_F32;
}

/* ---- tiny JSON helpers (sufficient for the safetensors header) ---- */
static const char* skip_ws(const char* p)
{
    while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')
        p++;
    return p;
}
/* Parse a JSON string starting at '"'. Returns pointer past closing quote and
 * writes the unescaped string into out (caller buffer). Supports basic
 * escapes. Returns NULL on error. */
static const char* parse_string(const char* p, char* out, size_t out_max)
{
    if (*p != '"')
        return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < out_max)
    {
        if (*p == '\\')
        {
            p++;
            char c = *p;
            if (c == 'n')
                c = '\n';
            else if (c == 't')
                c = '\t';
            else if (c == 'r')
                c = '\r';
            out[i++] = c;
            p++;
        }
        else
        {
            out[i++] = *p++;
        }
    }
    if (*p != '"')
        return NULL;
    out[i] = '\0';
    return p + 1;
}
/* Parse an array of non-negative integers. Returns pointer past ']'. */
static const char* parse_int_array(const char* p, size_t* out, size_t max_n, size_t* n_out)
{
    if (*p != '[')
        return NULL;
    p++;
    size_t n = 0;
    for (;;)
    {
        p = skip_ws(p);
        if (*p == ']')
        {
            *n_out = n;
            return p + 1;
        }
        long v = strtol(p, (char**) &p, 10);
        if (n < max_n)
            out[n++] = (size_t) v;
        p = skip_ws(p);
        if (*p == ',')
        {
            p++;
            continue;
        }
        if (*p == ']')
        {
            *n_out = n;
            return p + 1;
        }
        return NULL;
    }
}

static STEntry* find_entry(STHandle* h, const char* name)
{
    for (size_t i = 0; i < h->n_entries; i++)
        if (strcmp(h->entries[i].name, name) == 0)
            return &h->entries[i];
    return NULL;
}

/**
 * @brief Open Safetensors.
 *
 * @param path [in] Path value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_safetensors_open(const char* path, const char* mode)
{
    STHandle* h = (STHandle*) calloc(1, sizeof(STHandle));
    if (!h)
        return NULL;
    h->path = safetensors_strdup(path);
    h->mode = (mode && *mode == 'w') ? 'w' : 'r';
    if (h->mode == 'r')
    {
        FILE* f = fopen(path, "rb");
        if (!f)
        {
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz <= 0)
        {
            fclose(f);
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        h->buf = (unsigned char*) malloc(sz ? (size_t) sz : 1);
        if (!h->buf)
        {
            fclose(f);
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        if (fread(h->buf, 1, (size_t) sz, f) != (size_t) sz)
        {
            fclose(f);
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        fclose(f);
        h->buf_len = (size_t) sz;
        if (h->buf_len < 8)
        {
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        uint64_t header_len = 0;
        memcpy(&header_len, h->buf, 8);
        if (8 + header_len > h->buf_len)
        {
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        const char* hdr = (const char*) (h->buf + 8);
        const char* p = skip_ws(hdr);
        if (*p != '{')
        {
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        p++;
        /* Count entries to size the array. */
        size_t cap = 8;
        h->entries = (STEntry*) calloc(cap, sizeof(STEntry));
        if (!h->entries)
        {
            SNEPPX_safetensors_close(h);
            return NULL;
        }
        for (;;)
        {
            p = skip_ws(p);
            if (*p == '}')
                break;
            if (*p == ',')
            {
                p++;
                continue;
            }
            char key[256];
            const char* after = parse_string(p, key, sizeof(key));
            if (!after)
                break;
            p = skip_ws(after);
            if (*p != ':')
                break;
            p = skip_ws(p + 1);
            if (strcmp(key, "__metadata__") == 0)
            {
                /* capture raw metadata object for get_metadata */
                const char* mstart = p;
                int depth = 0;
                const char* q = p;
                for (; *q; q++)
                {
                    if (*q == '{')
                        depth++;
                    else if (*q == '}')
                    {
                        depth--;
                        if (depth == 0)
                            break;
                    }
                }
                size_t mlen = (size_t) (q - mstart + 1);
                h->meta_json = (char*) malloc(mlen + 1);
                memcpy(h->meta_json, mstart, mlen);
                h->meta_json[mlen] = '\0';
                p = q + 1;
                continue;
            }
            /* tensor entry object */
            if (*p != '{')
                break;
            STEntry e;
            memset(&e, 0, sizeof(e));
            e.name = safetensors_strdup(key);
            p++;
            /* parse inner fields */
            for (;;)
            {
                p = skip_ws(p);
                if (*p == '}')
                {
                    p++;
                    break;
                }
                if (*p == ',')
                {
                    p++;
                    continue;
                }
                char fld[64];
                const char* af = parse_string(p, fld, sizeof(fld));
                if (!af)
                    break;
                p = skip_ws(af);
                if (*p != ':')
                    break;
                p = skip_ws(p + 1);
                if (!strcmp(fld, "dtype"))
                {
                    char dt[32];
                    const char* af2 = parse_string(p, dt, sizeof(dt));
                    if (!af2)
                        break;
                    e.dtype = dtype_from_str(dt);
                    p = af2;
                }
                else if (!strcmp(fld, "shape"))
                {
                    size_t tmp[8];
                    size_t tn = 0;
                    const char* af2 = parse_int_array(p, tmp, 8, &tn);
                    if (!af2)
                        break;
                    for (size_t k = 0; k < tn; k++)
                        e.shape[k] = tmp[k];
                    e.ndim = tn;
                    p = af2;
                }
                else if (!strcmp(fld, "data_offsets"))
                {
                    size_t tmp[2];
                    size_t tn = 0;
                    const char* af2 = parse_int_array(p, tmp, 2, &tn);
                    if (!af2)
                        break;
                    if (tn == 2)
                    {
                        e.off_start = tmp[0];
                        e.off_end = tmp[1];
                    }
                    p = af2;
                }
                else
                {
                    /* skip unknown value */
                    int depth = 0;
                    for (;;)
                    {
                        if (*p == '"')
                        {
                            p++;
                            while (*p && *p != '"')
                                p++;
                            if (*p == '"')
                                p++;
                        }
                        else if (*p == '{' || *p == '[')
                        {
                            depth++;
                            p++;
                        }
                        else if (*p == '}' || *p == ']')
                        {
                            depth--;
                            p++;
                            if (depth <= 0)
                                break;
                        }
                        else if (*p == ',' && depth == 0)
                        {
                            p++;
                            break;
                        }
                        else if (*p == 0)
                            break;
                        else
                            p++;
                    }
                }
            }
            if (h->n_entries >= cap)
            {
                cap *= 2;
                STEntry* nw = (STEntry*) realloc(h->entries, cap * sizeof(STEntry));
                if (!nw)
                    break;
                h->entries = nw;
            }
            h->entries[h->n_entries++] = e;
        }
    }
    else
    {
        h->w_entries = (STEntry*) calloc(8, sizeof(STEntry));
        h->w_cap = 8;
    }
    return h;
}

/**
 * @brief Close Safetensors.
 */
void SNEPPX_safetensors_close(void* st)
{
    STHandle* h = (STHandle*) st;
    if (!h)
        return;
    free(h->path);
    free(h->buf);
    free(h->meta_json);
    if (h->entries)
    {
        for (size_t i = 0; i < h->n_entries; i++)
            free(h->entries[i].name);
        free(h->entries);
    }
    if (h->w_entries)
    {
        for (size_t i = 0; i < h->w_count; i++)
            free(h->w_entries[i].name);
        free(h->w_entries);
    }
    free(h->w_data);
    free(h);
}

/**
 * @brief Perform Safetensors Get Tensor Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_get_tensor_count(void* st)
{
    STHandle* h = (STHandle*) st;
    if (!h)
        return 0;
    return (int) h->n_entries;
}

/**
 * @brief Perform Safetensors Get Tensor Names.
 *
 * @param st [out] St value.
 * @param names [out] Names value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_get_tensor_names(void* st, char*** names, size_t* count)
{
    STHandle* h = (STHandle*) st;
    if (!h || !names || !count)
        return -1;
    *count = h->n_entries;
    if (h->n_entries == 0)
    {
        *names = NULL;
        return 0;
    }
    char** out = (char**) malloc(h->n_entries * sizeof(char*));
    if (!out)
        return -1;
    for (size_t i = 0; i < h->n_entries; i++)
        out[i] = safetensors_strdup(h->entries[i].name);
    *names = out;
    return 0;
}

/**
 * @brief Perform Safetensors Read Tensor.
 *
 * @param st [out] St value.
 * @param name [in] Name value.
 * @param ndim [out] Ndim value.
 * @param shape [out] Shape value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_safetensors_read_tensor(void* st, const char* name, size_t* ndim, size_t** shape, int* dtype)
{
    STHandle* h = (STHandle*) st;
    if (!h || !name)
        return NULL;
    STEntry* e = find_entry(h, name);
    if (!e)
        return NULL;
    if (ndim)
        *ndim = e->ndim;
    if (dtype)
        *dtype = e->dtype;
    if (shape)
    {
        size_t* sh = (size_t*) malloc(sizeof(size_t) * (e->ndim ? e->ndim : 1));
        for (size_t i = 0; i < e->ndim; i++)
            sh[i] = e->shape[i];
        *shape = sh;
    }
    size_t off = 8 + (h->buf_len > 8 ? 0 : 0); /* header length unknown here; recompute */
    /* The header length is the first 8 bytes. */
    uint64_t header_len = 0;
    memcpy(&header_len, h->buf, 8);
    size_t data_base = 8 + (size_t) header_len + e->off_start;
    size_t nbytes = e->off_end - e->off_start;
    if (data_base + nbytes > h->buf_len)
        return NULL;
    void* out = malloc(nbytes ? nbytes : 1);
    if (!out)
        return NULL;
    memcpy(out, h->buf + data_base, nbytes);
    (void) off;
    return out;
}

/**
 * @brief Perform Safetensors Write Tensor.
 *
 * @param st [out] St value.
 * @param name [in] Name value.
 * @param data [in] Data value.
 * @param shape [in] Shape value.
 * @param ndim [in] Ndim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_write_tensor(void* st, const char* name, const void* data, const size_t* shape, size_t ndim,
                                    int dtype)
{
    STHandle* h = (STHandle*) st;
    if (!h || h->mode != 'w' || !name || !data)
        return -1;
    if (h->w_count >= h->w_cap)
    {
        h->w_cap *= 2;
        STEntry* nw = (STEntry*) realloc(h->w_entries, h->w_cap * sizeof(STEntry));
        if (!nw)
            return -1;
        h->w_entries = nw;
    }
    STEntry* e = &h->w_entries[h->w_count++];
    memset(e, 0, sizeof(*e));
    e->name = safetensors_strdup(name);
    e->dtype = dtype;
    e->ndim = ndim > 8 ? 8 : ndim;
    size_t total = 1;
    for (size_t i = 0; i < e->ndim; i++)
    {
        e->shape[i] = shape[i];
        total *= shape[i];
    }
    size_t esz = SNEPPX_safetensors_dtype_size(dtype);
    size_t nbytes = total * esz;
    e->off_start = h->w_data_len;
    /* append data */
    if (h->w_data_len + nbytes > h->w_data_cap)
    {
        size_t ncap = (h->w_data_len + nbytes) * 2 + 64;
        unsigned char* nw = (unsigned char*) realloc(h->w_data, ncap);
        if (!nw)
            return -1;
        h->w_data = nw;
        h->w_data_cap = ncap;
    }
    memcpy(h->w_data + h->w_data_len, data, nbytes);
    e->off_end = h->w_data_len + nbytes;
    h->w_data_len = e->off_end;
    return 0;
}

/**
 * @brief Save Safetensors.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_safetensors_save(void* st)
{
    STHandle* h = (STHandle*) st;
    if (!h || h->mode != 'w' || !h->path)
        return -1;
    /* Build JSON header. */
    size_t cap = 256 + h->w_count * 128;
    char* hdr = (char*) malloc(cap);
    if (!hdr)
        return -1;
    size_t len = 0;
    safe_append(&hdr, &len, &cap, "{");
    for (size_t i = 0; i < h->w_count; i++)
    {
        STEntry* e = &h->w_entries[i];
        if (i)
            safe_append(&hdr, &len, &cap, ",");
        safe_append(&hdr, &len, &cap, "\"%s\":{\"dtype\":\"%s\",\"shape\":[", e->name, dtype_to_str(e->dtype));
        for (size_t k = 0; k < e->ndim; k++)
        {
            if (k)
                safe_append(&hdr, &len, &cap, ",");
            safe_append(&hdr, &len, &cap, "%zu", e->shape[k]);
        }
        safe_append(&hdr, &len, &cap, "],\"data_offsets\":[%zu,%zu]}", e->off_start, e->off_end);
    }
    safe_append(&hdr, &len, &cap, "}");
    /* Pad header to 8-byte alignment (optional but friendly). */
    uint64_t header_len = (uint64_t) len;
    uint64_t pad = (8 - (header_len & 7)) & 7;
    /* Write file. */
    FILE* f = fopen(h->path, "wb");
    if (!f)
    {
        free(hdr);
        return -1;
    }
    if (fwrite(&header_len, 1, 8, f) != 8)
    {
        fclose(f);
        free(hdr);
        return -1;
    }
    if (fwrite(hdr, 1, len, f) != len)
    {
        fclose(f);
        free(hdr);
        return -1;
    }
    if (pad)
    {
        unsigned char z = 0;
        for (uint64_t i = 0; i < pad; i++)
            fwrite(&z, 1, 1, f);
    }
    if (h->w_data_len && fwrite(h->w_data, 1, h->w_data_len, f) != h->w_data_len)
    {
        fclose(f);
        free(hdr);
        return -1;
    }
    fclose(f);
    free(hdr);
    return 0;
}

/**
 * @brief Perform Safetensors Get Metadata.
 *
 * @param st [out] St value.
 * @param key [in] Key value.
 * @param value [out] Value value.
 *
 * @return The result value, or 0 on error.
 */
unsigned long long SNEPPX_safetensors_get_metadata(void* st, const char* key, char* value, size_t value_max)
{
    STHandle* h = (STHandle*) st;
    if (!h || !h->meta_json || !key)
        return 0;
    /* Naive search for "\"key\":\"value\" inside the metadata object. */
    char pat[128];
    (void) snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(h->meta_json, pat);
    if (!p)
        return 0;
    p = strchr(p, ':');
    if (!p)
        return 0;
    p++;
    p = skip_ws(p);
    if (*p != '"')
        return 0;
    const char* after = parse_string(p, value, value_max);
    if (!after)
        return 0;
    return (unsigned long long) strlen(value);
}

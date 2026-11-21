#include "data_pipeline.h"
#include "polymorphic_memory_allocator.h"
#include "multidimensional_tensor_engine.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
  #include <windows.h>
  #define snprintf _snprintf
#else
  #include <sys/mman.h>
  #include <unistd.h>
#endif
/*
 * SNEPPX - Kernel Module
 *
 * WHAT
 *   Kernel Module.
 *
 * CONCEPT
 *   Kernel Module implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



struct SNEPPXDataPipeline {
    size_t batch_size;
    float* data_buffer;
    float* label_buffer;
    size_t num_samples;
    size_t num_features;
    size_t num_labels;
    size_t* shuffle_indices;
    size_t current_pos;
    int owns_buffers;
};

/**
 * @brief Create Data Pipeline.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXDataPipeline* SNEPPX_data_pipeline_create(size_t batch_size) {
    if (batch_size == 0) return NULL;
    SNEPPXDataPipeline* pipe = (SNEPPXDataPipeline*)SNEPPX_malloc(sizeof(SNEPPXDataPipeline), 64);
    if (!pipe) return NULL;
    pipe->batch_size = batch_size;
    pipe->data_buffer = NULL;
    pipe->label_buffer = NULL;
    pipe->num_samples = 0;
    pipe->num_features = 0;
    pipe->num_labels = 0;
    pipe->shuffle_indices = NULL;
    pipe->current_pos = 0;
    pipe->owns_buffers = 1;
    return pipe;
}

/**
 * @brief Destroy Data Pipeline.
 */
void SNEPPX_data_pipeline_destroy(SNEPPXDataPipeline* pipe) {
    if (!pipe) return;
    if (pipe->owns_buffers) {
        SNEPPX_free(pipe->data_buffer, pipe->num_samples * pipe->num_features * sizeof(float));
        SNEPPX_free(pipe->label_buffer, pipe->num_samples * pipe->num_labels * sizeof(float));
    }
    SNEPPX_free(pipe->shuffle_indices, pipe->num_samples * sizeof(size_t));
    SNEPPX_free(pipe, sizeof(SNEPPXDataPipeline));
}

static int parse_csv_line(const char* line, float* out, size_t max_cols) {
    size_t col = 0;
    const char* p = line;
    while (*p && col < max_cols) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r') break;
        char* end = NULL;
        out[col++] = (float)strtod(p, &end);
        if (end == p) break;
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
    }
    return (int)col;
}

/**
 * @brief Load Data Pipeline.
 *
 * @param path [in] Path value.
 * @param pipe [out] Pipe value.
 * @param data [out] Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_pipeline_load(const char* path, SNEPPXDataPipeline* pipe, SNEPPXTensor** data, SNEPPXTensor** labels) {
    if (!path || !pipe) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = (char*)SNEPPX_malloc((size_t)fsize + 1, 64);
    if (!text) { fclose(f); return -1; }
    size_t nread = fread(text, 1, (size_t)fsize, f);
    fclose(f);
    text[nread] = '\0';

    size_t max_lines = 0;
    int has_cr = 0;
    for (size_t i = 0; i < nread; i++) {
        if (text[i] == '\n') max_lines++;
        if (text[i] == '\r') has_cr = 1;
    }
    if (max_lines == 0) { SNEPPX_free(text, (size_t)fsize + 1); return -1; }

    float* temp = (float*)SNEPPX_malloc(max_lines * 1024 * sizeof(float), 64);
    if (!temp) { SNEPPX_free(text, (size_t)fsize + 1); return -1; }

    size_t actual_lines = 0;
    size_t max_cols = 0;
    char* line_start = text;
    for (size_t i = 0; i <= nread && actual_lines < max_lines; i++) {
        if (i == nread || text[i] == '\n' || text[i] == '\r') {
        if (text[i] == '\r' && i + 1 < nread && text[i + 1] == '\n') {
            text[i] = '\0';
            i++;
        }
        text[i] = '\0';
            if (line_start < text + i && *line_start != '\0' && *line_start != '#') {
                size_t cols = (size_t)parse_csv_line(line_start, temp + actual_lines * 1024, 1024);
                if (cols > 0) {
                    if (cols > max_cols) max_cols = cols;
                    actual_lines++;
                }
            }
            line_start = text + i + 1;
        }
    }

    SNEPPX_free(text, (size_t)fsize + 1);

    if (actual_lines == 0 || max_cols < 2) {
        SNEPPX_free(temp, max_lines * 1024 * sizeof(float));
        return -1;
    }

    size_t num_features = max_cols - 1;
    size_t num_labels = 1;
    pipe->data_buffer = (float*)SNEPPX_malloc(actual_lines * num_features * sizeof(float), 64);
    pipe->label_buffer = (float*)SNEPPX_malloc(actual_lines * num_labels * sizeof(float), 64);
    if (!pipe->data_buffer || !pipe->label_buffer) {
        SNEPPX_free(pipe->data_buffer, actual_lines * num_features * sizeof(float));
        SNEPPX_free(pipe->label_buffer, actual_lines * num_labels * sizeof(float));
        SNEPPX_free(temp, max_lines * 1024 * sizeof(float));
        return -1;
    }

    for (size_t i = 0; i < actual_lines; i++) {
        for (size_t j = 0; j < num_features; j++)
            pipe->data_buffer[i * num_features + j] = temp[i * 1024 + j];
        pipe->label_buffer[i] = temp[i * 1024 + num_features];
    }
    SNEPPX_free(temp, max_lines * 1024 * sizeof(float));

    pipe->num_samples = actual_lines;
    pipe->num_features = num_features;
    pipe->num_labels = num_labels;
    pipe->current_pos = 0;

    SNEPPX_free(pipe->shuffle_indices, pipe->num_samples * sizeof(size_t));
    pipe->shuffle_indices = (size_t*)SNEPPX_malloc(actual_lines * sizeof(size_t), 64);
    if (pipe->shuffle_indices) {
        for (size_t i = 0; i < actual_lines; i++) pipe->shuffle_indices[i] = i;
    }

    if (data) {
        size_t shape[] = {actual_lines, num_features};
        *data = SNEPPX_tensor_empty(shape, 2, SNEPPX_FLOAT32);
        if (*data) memcpy((*data)->data, pipe->data_buffer, actual_lines * num_features * sizeof(float));
    }
    if (labels) {
        size_t shape[] = {actual_lines, 1};
        *labels = SNEPPX_tensor_empty(shape, 2, SNEPPX_FLOAT32);
        if (*labels) memcpy((*labels)->data, pipe->label_buffer, actual_lines * sizeof(float));
    }
    return 0;
}

/**
 * @brief Perform Data Pipeline Get Batch.
 *
 * @param pipe [out] Pipe value.
 * @param batch [out] Batch value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_pipeline_get_batch(SNEPPXDataPipeline* pipe, SNEPPXTensor** batch, SNEPPXTensor** labels) {
    if (!pipe || !pipe->data_buffer || !batch || !labels) return -1;
    if (pipe->num_samples == 0) return -1;
    size_t actual = pipe->batch_size;
    if (pipe->current_pos + actual > pipe->num_samples) actual = pipe->num_samples - pipe->current_pos;
    if (actual == 0) { pipe->current_pos = 0; actual = pipe->batch_size; if (actual > pipe->num_samples) actual = pipe->num_samples; }

    float* batch_data = (float*)SNEPPX_malloc(actual * pipe->num_features * sizeof(float), 64);
    float* batch_labels = (float*)SNEPPX_malloc(actual * pipe->num_labels * sizeof(float), 64);
    if (!batch_data || !batch_labels) {
        SNEPPX_free(batch_data, actual * pipe->num_features * sizeof(float));
        SNEPPX_free(batch_labels, actual * pipe->num_labels * sizeof(float));
        return -1;
    }

    for (size_t i = 0; i < actual; i++) {
        size_t idx = pipe->shuffle_indices ? pipe->shuffle_indices[pipe->current_pos + i] : pipe->current_pos + i;
        memcpy(batch_data + i * pipe->num_features, pipe->data_buffer + idx * pipe->num_features, pipe->num_features * sizeof(float));
        memcpy(batch_labels + i * pipe->num_labels, pipe->label_buffer + idx * pipe->num_labels, pipe->num_labels * sizeof(float));
    }
    pipe->current_pos += actual;

    {
        size_t shape[] = {actual, pipe->num_features};
        *batch = SNEPPX_tensor_empty(shape, 2, SNEPPX_FLOAT32);
        if (*batch) memcpy((*batch)->data, batch_data, actual * pipe->num_features * sizeof(float));
    }
    {
        size_t shape[] = {actual, pipe->num_labels};
        *labels = SNEPPX_tensor_empty(shape, 2, SNEPPX_FLOAT32);
        if (*labels) memcpy((*labels)->data, batch_labels, actual * pipe->num_labels * sizeof(float));
    }
    SNEPPX_free(batch_data, actual * pipe->num_features * sizeof(float));
    SNEPPX_free(batch_labels, actual * pipe->num_labels * sizeof(float));
    return 0;
}

/**
 * @brief Perform Data Pipeline Shuffle.
 */
void SNEPPX_data_pipeline_shuffle(SNEPPXDataPipeline* pipe) {
    if (!pipe || !pipe->shuffle_indices || pipe->num_samples < 2) return;
    for (size_t i = pipe->num_samples - 1; i > 0; i--) {
        size_t j = (size_t)((double)rand() / (RAND_MAX + 1.0) * (double)(i + 1));
        if (j > i) j = i;
        size_t tmp = pipe->shuffle_indices[i];
        pipe->shuffle_indices[i] = pipe->shuffle_indices[j];
        pipe->shuffle_indices[j] = tmp;
    }
    pipe->current_pos = 0;
}

/**
 * @brief Perform Data Pipeline Get Batch Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_data_pipeline_get_batch_size(const SNEPPXDataPipeline* pipe) {
    return pipe ? pipe->batch_size : 0;
}

struct SNEPPXTextDataset {
    unsigned char* token_data;
    size_t* sample_offsets;
    size_t num_samples;
    size_t seq_len;
    size_t alloc_size;
    SNEPPXTokenizer* tok;
};

/**
 * @brief Create Text Dataset.
 *
 * @param path [in] Path value.
 * @param tok [out] Tok value.
 * @param seq_len [in] Seq Len value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTextDataset* SNEPPX_text_dataset_create(const char* path, SNEPPXTokenizer* tok,
                                           size_t seq_len, int line_by_line) {
    if (!path || !tok || seq_len == 0) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    char* text = (char*)SNEPPX_malloc((size_t)fsize + 1, 64);
    if (!text) { fclose(f); return NULL; }
    size_t nread = fread(text, 1, (size_t)fsize, f);
    fclose(f);
    text[nread] = '\0';

    size_t max_lines = 0;
    for (size_t i = 0; i < nread; i++) if (text[i] == '\n') max_lines++;
    if (max_lines == 0) { SNEPPX_free(text, (size_t)fsize + 1); return NULL; }

    size_t max_token_buf = (size_t)fsize * sizeof(int);
    size_t max_samples = (size_t)fsize;
    size_t alloc_size = max_token_buf + max_samples * sizeof(size_t);
    unsigned char* token_data = (unsigned char*)SNEPPX_malloc(alloc_size, 64);
    if (!token_data) { SNEPPX_free(text, (size_t)fsize + 1); return NULL; }
    int* td = (int*)token_data;
    size_t* offsets = (size_t*)(token_data + max_token_buf);
    size_t actual_samples = 0;
    size_t token_cursor = 0;

    if (line_by_line) {
        char* line_start = text;
        for (size_t i = 0; i <= nread && actual_samples < max_lines; i++) {
            if (i == nread || text[i] == '\n') {
                text[i] = '\0';
                if (line_start < text + i) {
                    int n = SNEPPX_tokenizer_encode(tok, line_start, (int*)(td + token_cursor), (int)(max_token_buf / sizeof(int) - token_cursor));
                    if (n > 0) {
                        offsets[actual_samples] = token_cursor * sizeof(int);
                        token_cursor += (size_t)n;
                        actual_samples++;
                    }
                }
                line_start = text + i + 1;
            }
        }
    } else {
        int total_tokens = SNEPPX_tokenizer_encode(tok, text, (int*)td, (int)(max_token_buf / sizeof(int)));
        if (total_tokens <= 0) { SNEPPX_free(token_data, alloc_size); SNEPPX_free(text, (size_t)fsize + 1); return NULL; }
        size_t num_chunks = (size_t)total_tokens / seq_len;
        if (num_chunks == 0) { SNEPPX_free(token_data, alloc_size); SNEPPX_free(text, (size_t)fsize + 1); return NULL; }
        for (size_t i = 0; i < num_chunks; i++) {
            offsets[i] = i * seq_len * sizeof(int);
            actual_samples++;
        }
    }

    SNEPPX_free(text, (size_t)fsize + 1);

    SNEPPXTextDataset* ds = (SNEPPXTextDataset*)SNEPPX_malloc(sizeof(SNEPPXTextDataset), 64);
    if (!ds) { SNEPPX_free(token_data, alloc_size); return NULL; }
    ds->token_data = token_data;
    ds->sample_offsets = offsets;
    ds->num_samples = actual_samples;
    ds->seq_len = seq_len;
    ds->alloc_size = alloc_size;
    ds->tok = tok;
    return ds;
}

/**
 * @brief Destroy Text Dataset.
 */
void SNEPPX_text_dataset_destroy(SNEPPXTextDataset* ds) {
    if (!ds) return;
    SNEPPX_free(ds->token_data, ds->alloc_size);
    SNEPPX_free(ds, sizeof(SNEPPXTextDataset));
}

/**
 * @brief Perform Text Dataset Size.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_text_dataset_size(const SNEPPXTextDataset* ds) {
    return ds ? ds->num_samples : 0;
}

/**
 * @brief Perform Text Dataset Get Batch.
 *
 * @param ds [in] Ds value.
 * @param start_idx [in] Start Idx value.
 * @param batch_size [in] Batch Size value.
 * @param input_ids [out] Input Ids value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_text_dataset_get_batch(const SNEPPXTextDataset* ds, size_t start_idx, size_t batch_size,
                                 SNEPPXTensor** input_ids, SNEPPXTensor** target_ids) {
    if (!ds || !input_ids || !target_ids) return 1;
    if (start_idx >= ds->num_samples) return 1;
    size_t actual = batch_size;
    if (start_idx + actual > ds->num_samples) actual = ds->num_samples - start_idx;

    size_t in_shape[] = {actual, ds->seq_len};
    SNEPPXTensor* in_t = SNEPPX_tensor_empty(in_shape, 2, SNEPPX_INT32);
    SNEPPXTensor* tgt_t = SNEPPX_tensor_empty(in_shape, 2, SNEPPX_INT32);
    if (!in_t || !tgt_t) {
        SNEPPX_tensor_destroy(in_t); SNEPPX_tensor_destroy(tgt_t);
        return 1;
    }

    int* in_d = (int*)in_t->data;
    int* tgt_d = (int*)tgt_t->data;
    int* src = (int*)ds->token_data;

    for (size_t i = 0; i < actual; i++) {
        size_t sidx = start_idx + i;
        size_t sample_offset = ds->sample_offsets[sidx] / sizeof(int);
        for (size_t j = 0; j + 1 < ds->seq_len && sample_offset + j + 1 < ds->num_samples * ds->seq_len; j++) {
            in_d[i * ds->seq_len + j] = src[sample_offset + j];
            tgt_d[i * ds->seq_len + j] = src[sample_offset + j + 1];
        }
    }

    *input_ids = in_t;
    *target_ids = tgt_t;
    return 0;
}

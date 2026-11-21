#ifndef SNEPPX_DATA_PIPELINE_H
#define SNEPPX_DATA_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "subword_tokenization_pipeline.h"
#include "multidimensional_tensor_engine.h"
#include <stddef.h>

/*
 * SNEPPX - Data Pipeline
 *
 * WHAT
 *   Data Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef struct SNEPPXDataPipeline SNEPPXDataPipeline;

/**
 * @brief Create Data Pipeline.
 *
 * @param batch_size [in] Batch Size value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXDataPipeline* SNEPPX_data_pipeline_create(size_t batch_size);
/**
 * @brief Destroy Data Pipeline.
 *
 * @param pipe [out] Pipe value.
 */
void SNEPPX_data_pipeline_destroy(SNEPPXDataPipeline* pipe);
/**
 * @brief Load Data Pipeline.
 *
 * @param path [in] Path value.
 * @param pipe [out] Pipe value.
 * @param data [out] Data value.
 * @param labels [out] Labels value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_pipeline_load(const char* path, SNEPPXDataPipeline* pipe, SNEPPXTensor** data, SNEPPXTensor** labels);
/**
 * @brief Perform Data Pipeline Get Batch.
 *
 * @param pipe [out] Pipe value.
 * @param batch [out] Batch value.
 * @param labels [out] Labels value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_data_pipeline_get_batch(SNEPPXDataPipeline* pipe, SNEPPXTensor** batch, SNEPPXTensor** labels);
/**
 * @brief Perform Data Pipeline Shuffle.
 *
 * @param pipe [out] Pipe value.
 */
void SNEPPX_data_pipeline_shuffle(SNEPPXDataPipeline* pipe);
/**
 * @brief Perform Data Pipeline Get Batch Size.
 *
 * @param pipe [in] Pipe value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_data_pipeline_get_batch_size(const SNEPPXDataPipeline* pipe);

typedef struct SNEPPXTextDataset SNEPPXTextDataset;

/**
 * @brief Create Text Dataset.
 *
 * @param path [in] Path value.
 * @param tok [out] Tok value.
 * @param seq_len [in] Seq Len value.
 * @param line_by_line [in] Line By Line value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXTextDataset* SNEPPX_text_dataset_create(const char* path, SNEPPXTokenizer* tok,
                                            size_t seq_len, int line_by_line);
/**
 * @brief Destroy Text Dataset.
 *
 * @param ds [out] Ds value.
 */
void SNEPPX_text_dataset_destroy(SNEPPXTextDataset* ds);

/**
 * @brief Perform Text Dataset Size.
 *
 * @param ds [in] Ds value.
 *
 * @return The computed size/count, or 0 on error.
 */
size_t SNEPPX_text_dataset_size(const SNEPPXTextDataset* ds);

/**
 * @brief Perform Text Dataset Get Batch.
 *
 * @param ds [in] Ds value.
 * @param start_idx [in] Start Idx value.
 * @param batch_size [in] Batch Size value.
 * @param input_ids [out] Input Ids value.
 * @param target_ids [out] Target Ids value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_text_dataset_get_batch(const SNEPPXTextDataset* ds, size_t start_idx, size_t batch_size,
                                 SNEPPXTensor** input_ids, SNEPPXTensor** target_ids);


#ifdef __cplusplus
}
#endif
#endif

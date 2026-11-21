#ifndef SNEPPX_RECURRENT_NEURAL_NETWORK_H
#define SNEPPX_RECURRENT_NEURAL_NETWORK_H
#include <stddef.h>
#include <stdbool.h>
/*
 * SNEPPX - Recurrent Neural Network
 *
 * WHAT
 *   Recurrent Neural Network.
 *
 * CONCEPT
 *   RNN configuration and forward API for sequential data.
 *
 * ROLE
 *   Declares the RNN API for recurrent neural network operations within the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal module).
 */


#ifdef __cplusplus
extern "C" {
#endif

typedef struct SNEPPXRNN SNEPPXRNN;

/**
 * @brief Create Rnn.
 *
 * @param input_size [in] Input Size value.
 * @param hidden_size [in] Hidden Size value.
 * @param num_layers [in] Num Layers value.
 * @param bidirectional [in] Bidirectional value.
 * @param dropout [in] Dropout value.
 * @param rnn_type [in] Rnn Type value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXRNN* SNEPPX_rnn_create(size_t input_size, size_t hidden_size, size_t num_layers,
    int bidirectional, float dropout, const char* rnn_type);
/**
 * @brief Destroy Rnn.
 *
 * @param rnn [out] Rnn value.
 */
void SNEPPX_rnn_destroy(void* rnn);
/**
 * @brief Run the forward pass for Rnn.
 *
 * @param rnn [out] Rnn value.
 * @param input [in] Input value.
 * @param seq_len [in] Seq Len value.
 * @param batch_size [in] Batch Size value.
 * @param output [out] Output value.
 * @param hidden [out] Hidden value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_rnn_forward(void* rnn, const float* input, size_t seq_len, size_t batch_size,
    float* output, float* hidden);

#ifdef __cplusplus
}
#endif
#endif

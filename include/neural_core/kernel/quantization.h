#ifndef SNEPPX_QUANTIZATION_H
#define SNEPPX_QUANTIZATION_H

#include <stddef.h>
#include <stdint.h>
#include "multidimensional_tensor_engine.h"

/*
 * SNEPPX - Quantization
 *
 * WHAT
 *   Quantization.
 *
 * CONCEPT
 *   Provides quantization and compression.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_QUANT_NONE = 0,
    SNEPPX_QUANT_INT8_SYM,
    SNEPPX_QUANT_INT8_ASYM,
    SNEPPX_QUANT_INT4_SYM,
    SNEPPX_QUANT_FP8_E4M3,
    SNEPPX_QUANT_FP8_E5M2,
    SNEPPX_QUANT_AWQ,
    SNEPPX_QUANT_GPTQ,
} SNEPPXQuantMode;

typedef enum {
    SNEPPX_QUANT_PER_TENSOR = 0,
    SNEPPX_QUANT_PER_CHANNEL,
    SNEPPX_QUANT_PER_GROUP,
    SNEPPX_QUANT_PER_TOKEN,
} SNEPPXQuantGranularity;

typedef struct {
    float scale;
    int32_t zero_point;
    float qmin;
    float qmax;
    float scale_max;
} SNEPPXQuantParams;

typedef struct {
    float* scales;
    int32_t* zero_points;
    size_t num_params;
    SNEPPXQuantGranularity granularity;
} SNEPPXQuantState;

typedef struct {
    float* weights;        /* original high-precision weights */
    float* hessian;        /* Hessian matrix (GPTQ) */
    size_t rows;
    size_t cols;
    int group_size;
} SNEPPXGPTQState;

#ifdef __cplusplus
extern "C" {
#endif

/* INT8 symmetric quantization */
/**
 * @brief Perform Quantize Int8 Sym.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 * @param scale_out [out] Scale Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_int8_sym(const float* input, int8_t* output,
                              size_t n, float* scale_out);
/**
 * @brief Perform Dequantize Int8 Sym.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_int8_sym(const int8_t* input, float* output,
                                size_t n, float scale);

/* INT8 asymmetric quantization */
/**
 * @brief Perform Quantize Int8 Asym.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 * @param scale_out [out] Scale Out value.
 * @param zp_out [out] Zp Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_int8_asym(const float* input, int8_t* output,
                               size_t n, float* scale_out, int32_t* zp_out);
/**
 * @brief Perform Dequantize Int8 Asym.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 * @param scale [in] Scale value.
 * @param zp [in] Zp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_int8_asym(const int8_t* input, float* output,
                                 size_t n, float scale, int32_t zp);

/* Per-channel INT8 symmetric */
/**
 * @brief Perform Quantize Int8 Channel.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 * @param scales_out [out] Scales Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_int8_channel(const float* input, int8_t* output,
                                  size_t rows, size_t cols,
                                  float* scales_out);

/* INT4 symmetric (packed) */
/**
 * @brief Perform Quantize Int4 Sym.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 * @param scale_out [out] Scale Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_int4_sym(const float* input, uint8_t* output,
                              size_t n, float* scale_out);
/**
 * @brief Perform Dequantize Int4 Sym.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 * @param scale [in] Scale value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_int4_sym(const uint8_t* input, float* output,
                                size_t n, float scale);

/* FP8 E4M3 encode/decode (software fallback) */
/**
 * @brief Perform Float To Fp8 E4m3.
 *
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_float_to_fp8_e4m3(float value);
/**
 * @brief Perform Fp8 E4m3 To Float.
 *
 * @param fp8 [in] Fp8 value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_fp8_e4m3_to_float(uint8_t fp8);
/**
 * @brief Perform Quantize Fp8 E4m3.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_fp8_e4m3(const float* input, uint8_t* output, size_t n);
/**
 * @brief Perform Dequantize Fp8 E4m3.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_fp8_e4m3(const uint8_t* input, float* output, size_t n);

/* FP8 E5M2 encode/decode (software fallback) */
/**
 * @brief Perform Float To Fp8 E5m2.
 *
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_float_to_fp8_e5m2(float value);
/**
 * @brief Perform Fp8 E5m2 To Float.
 *
 * @param fp8 [in] Fp8 value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_fp8_e5m2_to_float(uint8_t fp8);
/**
 * @brief Perform Quantize Fp8 E5m2.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_fp8_e5m2(const float* input, uint8_t* output, size_t n);
/**
 * @brief Perform Dequantize Fp8 E5m2.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_fp8_e5m2(const uint8_t* input, float* output, size_t n);

/* AWQ: Activation-aware Weight Quantization */
/**
 * @brief Perform Awq Scale Weights.
 *
 * @param weights [out] Weights value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 * @param act_scales [in] Act Scales value.
 * @param scale_out [out] Scale Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_awq_scale_weights(float* weights, size_t rows, size_t cols,
                              const float* act_scales, float* scale_out);
/**
 * @brief Quantize Awq.
 *
 * @param weights [in] Weights value.
 * @param qweight [out] Qweight value.
 * @param act_scales [in] Act Scales value.
 * @param scales [out] Scales value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 * @param group_size [in] Group Size value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_awq_quantize(const float* weights, int8_t* qweight,
                         const float* act_scales, float* scales,
                         size_t rows, size_t cols, int group_size);

/* GPTQ: Post-training quantization */
/**
 * @brief Quantize Gptq.
 *
 * @param weights [out] Weights value.
 * @param rows [in] Rows value.
 * @param cols [in] Cols value.
 * @param group_size [in] Group Size value.
 * @param bits [in] Bits value.
 * @param qweight [out] Qweight value.
 * @param scales [out] Scales value.
 * @param zeros [out] Zeros value.
 * @param sym [in] Sym value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gptq_quantize(float* weights, size_t rows, size_t cols,
                          int group_size, int bits,
                          SNEPPXDtype* qweight, float* scales,
                          int32_t* zeros, int sym);
/**
 * @brief Perform Gptq Compute Hessian.
 *
 * @param hessian [out] Hessian value.
 * @param activation [in] Activation value.
 * @param n [in] N value.
 * @param dim [in] Dim value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gptq_compute_hessian(float* hessian, const float* activation,
                                 size_t n, size_t dim);
/**
 * @brief Perform Gptq Quantize Block.
 *
 * @param w [out] W value.
 * @param h_inv [out] H Inv value.
 * @param qw [out] Qw value.
 * @param scale [out] Scale value.
 * @param zero [out] Zero value.
 * @param block_size [in] Block Size value.
 * @param bits [in] Bits value.
 * @param sym [in] Sym value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_gptq_quantize_block(float* w, float* h_inv,
                                float* qw, float* scale, int32_t* zero,
                                size_t block_size, int bits, int sym);

/* Utility */
/**
 * @brief Create Quant Params.
 *
 * @param params [out] Params value.
 * @param mode [in] Mode value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quant_params_create(SNEPPXQuantParams* params, SNEPPXQuantMode mode);
/**
 * @brief Perform Quant Mode Name.
 *
 * @param mode [in] Mode value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_quant_mode_name(SNEPPXQuantMode mode);

#ifdef __cplusplus
}
#endif

#endif /* SNEPPX_QUANTIZATION_H */

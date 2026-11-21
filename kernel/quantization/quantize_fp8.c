#include "quantization.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
/*
 * SNEPPX - FP8 Quantization
 *
 * WHAT
 *   FP8 Quantization.
 *
 * CONCEPT
 *   FP8 Quantization implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/*
 * FP8 E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
 *   Range: ±448, supports 0, ±inf, NaN
 *   Bias: 7
 *
 * FP8 E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits
 *   Range: ±57344, supports 0, ±inf, NaN
 *   Bias: 15
 */

/* ---- E4M3 ---- */
/**
 * @brief Perform Float To Fp8 E4m3.
 *
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_float_to_fp8_e4m3(float value)
{
    uint32_t b;
    memcpy(&b, &value, sizeof(b));
    uint32_t sign = (b >> 31) & 1;
    uint32_t exp_bits = (b >> 23) & 0xFF;

    /* Handle special float32 values */
    if (exp_bits == 0xFF) {
        /* Inf or NaN -> FP8 Inf (0x78/0xF8) */
        return (uint8_t)((sign << 7) | 0x78);
    }
    if (exp_bits == 0) {
        /* Subnormal or zero -> FP8 zero */
        return (uint8_t)(sign << 7);
    }
    int32_t exp = (int32_t)exp_bits - 127;
    uint32_t mant = (b >> 20) & 0x07;

    if (exp < -6) {
        return (uint8_t)(sign << 7);
    }
    if (exp > 8) {
        return (uint8_t)((sign << 7) | 0x78); /* Inf */
    }
    uint8_t e4m3_exp = (uint8_t)(exp + 7);
    uint8_t e4m3 = (uint8_t)((sign << 7) | (e4m3_exp << 3) | mant);
    return e4m3;
}

/**
 * @brief Perform Fp8 E4m3 To Float.
 *
 * @param fp8 [in] Fp8 value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_fp8_e4m3_to_float(uint8_t fp8)
{
    uint32_t sign = (uint32_t)((fp8 >> 7) & 1);
    uint32_t e4m3_exp = (uint32_t)((fp8 >> 3) & 0x0F);
    uint32_t e4m3_mant = (uint32_t)(fp8 & 0x07);

    if (e4m3_exp == 0x0F) {
        /* Inf or NaN */
        if (e4m3_mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    if (e4m3_exp == 0 && e4m3_mant == 0) {
        return 0.0f;
    }
    int32_t exp = (int32_t)e4m3_exp - 7;
    uint32_t f32_exp = (uint32_t)(exp + 127);
    uint32_t f32 = (sign << 31) | (f32_exp << 23) | (e4m3_mant << 20);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

/**
 * @brief Perform Quantize Fp8 E4m3.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_fp8_e4m3(const float* input, uint8_t* output, size_t n)
{
    if (!input || !output) return -1;
    for (size_t i = 0; i < n; i++) {
        output[i] = SNEPPX_float_to_fp8_e4m3(input[i]);
    }
    return 0;
}

/**
 * @brief Perform Dequantize Fp8 E4m3.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_fp8_e4m3(const uint8_t* input, float* output, size_t n)
{
    if (!input || !output) return -1;
    for (size_t i = 0; i < n; i++) {
        output[i] = SNEPPX_fp8_e4m3_to_float(input[i]);
    }
    return 0;
}

/* ---- E5M2 ---- */
/**
 * @brief Perform Float To Fp8 E5m2.
 *
 * @param value [in] Value value.
 *
 * @return 0 on success, -1 on error.
 */
uint8_t SNEPPX_float_to_fp8_e5m2(float value)
{
    uint32_t b;
    memcpy(&b, &value, sizeof(b));
    uint32_t sign = (b >> 31) & 1;
    uint32_t exp_bits = (b >> 23) & 0xFF;

    if (exp_bits == 0xFF) {
        return (uint8_t)((sign << 7) | 0x7C); /* Inf */
    }
    if (exp_bits == 0) {
        return (uint8_t)(sign << 7);
    }
    int32_t exp = (int32_t)exp_bits - 127;
    uint32_t mant = (b >> 22) & 0x01;

    if (exp < -14) {
        return (uint8_t)(sign << 7);
    }
    if (exp > 15) {
        return (uint8_t)((sign << 7) | 0x7C); /* Inf */
    }
    uint8_t e5m2_exp = (uint8_t)(exp + 15);
    uint8_t e5m2 = (uint8_t)((sign << 7) | (e5m2_exp << 2) | mant);
    return e5m2;
}

/**
 * @brief Perform Fp8 E5m2 To Float.
 *
 * @param fp8 [in] Fp8 value.
 *
 * @return The result value, or 0 on error.
 */
float SNEPPX_fp8_e5m2_to_float(uint8_t fp8)
{
    uint32_t sign = (uint32_t)((fp8 >> 7) & 1);
    uint32_t e5m2_exp = (uint32_t)((fp8 >> 2) & 0x1F);
    uint32_t e5m2_mant = (uint32_t)(fp8 & 0x03);

    if (e5m2_exp == 0x1F) {
        if (e5m2_mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    if (e5m2_exp == 0 && e5m2_mant == 0) {
        return 0.0f;
    }
    int32_t exp = (int32_t)e5m2_exp - 15;
    uint32_t f32_exp = (uint32_t)(exp + 127);
    uint32_t f32 = (sign << 31) | (f32_exp << 23) | (e5m2_mant << 21);
    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

/**
 * @brief Perform Quantize Fp8 E5m2.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quantize_fp8_e5m2(const float* input, uint8_t* output, size_t n)
{
    if (!input || !output) return -1;
    for (size_t i = 0; i < n; i++) {
        output[i] = SNEPPX_float_to_fp8_e5m2(input[i]);
    }
    return 0;
}

/**
 * @brief Perform Dequantize Fp8 E5m2.
 *
 * @param input [in] Input value.
 * @param output [out] Output value.
 * @param n [in] N value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dequantize_fp8_e5m2(const uint8_t* input, float* output, size_t n)
{
    if (!input || !output) return -1;
    for (size_t i = 0; i < n; i++) {
        output[i] = SNEPPX_fp8_e5m2_to_float(input[i]);
    }
    return 0;
}

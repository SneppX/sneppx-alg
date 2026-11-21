#include "quantization.h"
#include <math.h>
#include <float.h>
#include <string.h>

#ifndef SNEPPX_MIN
#define SNEPPX_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef SNEPPX_MAX
#define SNEPPX_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef SNEPPX_CLAMP
#define SNEPPX_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif
/*
 * SNEPPX - INT8 Quantization
 *
 * WHAT
 *   INT8 Quantization.
 *
 * CONCEPT
 *   INT8 Quantization implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



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
                              size_t n, float* scale_out)
{
    if (!input || !output || !scale_out || n == 0) return -1;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(input[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < FLT_MIN) {
        *scale_out = 1.0f;
        memset(output, 0, n);
        return 0;
    }
    float scale = max_abs / 127.0f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; i++) {
        int q = (int)roundf(input[i] * inv_scale);
        output[i] = (int8_t)SNEPPX_CLAMP(q, -128, 127);
    }
    return 0;
}

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
                                size_t n, float scale)
{
    if (!input || !output || n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        output[i] = (float)input[i] * scale;
    }
    return 0;
}

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
                               size_t n, float* scale_out, int32_t* zp_out)
{
    if (!input || !output || !scale_out || !zp_out || n == 0) return -1;
    float vmin = input[0], vmax = input[0];
    for (size_t i = 1; i < n; i++) {
        if (input[i] < vmin) vmin = input[i];
        if (input[i] > vmax) vmax = input[i];
    }
    if (vmax - vmin < FLT_MIN) {
        *scale_out = 1.0f;
        *zp_out = 0;
        memset(output, 0, n);
        return 0;
    }
    float scale = (vmax - vmin) / 255.0f;
    int32_t zp = (int32_t)roundf(-vmin / scale);
    zp = SNEPPX_CLAMP(zp, 0, 255);
    *scale_out = scale;
    *zp_out = zp;
    for (size_t i = 0; i < n; i++) {
        int q = (int)roundf(input[i] / scale + (float)zp);
        output[i] = (int8_t)SNEPPX_CLAMP(q, 0, 255);
    }
    return 0;
}

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
                                 size_t n, float scale, int32_t zp)
{
    if (!input || !output || n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        output[i] = ((float)((uint8_t)input[i]) - (float)zp) * scale;
    }
    return 0;
}

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
                                  float* scales_out)
{
    if (!input || !output || !scales_out || rows == 0 || cols == 0) return -1;
    for (size_t r = 0; r < rows; r++) {
        float max_abs = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float a = fabsf(input[r * cols + c]);
            if (a > max_abs) max_abs = a;
        }
        float scale = (max_abs < FLT_MIN) ? 1.0f : max_abs / 127.0f;
        scales_out[r] = scale;
        float inv_scale = 1.0f / scale;
        for (size_t c = 0; c < cols; c++) {
            int q = (int)roundf(input[r * cols + c] * inv_scale);
            output[r * cols + c] = (int8_t)SNEPPX_CLAMP(q, -128, 127);
        }
    }
    return 0;
}

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
                              size_t n, float* scale_out)
{
    if (!input || !output || !scale_out || n == 0) return -1;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = fabsf(input[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < FLT_MIN) {
        *scale_out = 1.0f;
        memset(output, 0, (n + 1) / 2);
        return 0;
    }
    float scale = max_abs / 7.0f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; i += 2) {
        int q0 = (int)roundf(input[i] * inv_scale);
        q0 = SNEPPX_CLAMP(q0, -8, 7);
        uint8_t packed = (uint8_t)(q0 & 0x0F);
        if (i + 1 < n) {
            int q1 = (int)roundf(input[i + 1] * inv_scale);
            q1 = SNEPPX_CLAMP(q1, -8, 7);
            packed |= (uint8_t)((q1 & 0x0F) << 4);
        }
        output[i / 2] = packed;
    }
    return 0;
}

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
                                size_t n, float scale)
{
    if (!input || !output || n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        uint8_t packed = input[i / 2];
        int q;
        if (i % 2 == 0) {
            q = (int)(packed & 0x0F);
            if (q >= 8) q -= 16;
        } else {
            q = (int)((packed >> 4) & 0x0F);
            if (q >= 8) q -= 16;
        }
        output[i] = (float)q * scale;
    }
    return 0;
}

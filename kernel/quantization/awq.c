#include "quantization.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
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
#ifndef SNEPPX_DIV
#define SNEPPX_DIV(a, b) ((b) == 0.0f ? 0.0f : (a) / (b))
#endif
/*
 * SNEPPX - AWQ Weight Quantization
 *
 * WHAT
 *   AWQ Weight Quantization.
 *
 * CONCEPT
 *   AWQ Weight Quantization implementation.
 *
 * ROLE
 *   Core kernel module used throughout the SNEPPX-Algo system.
 *
 * REFERENCES
 *   None (internal kernel module).
 */



/*
 * AWQ (Activation-aware Weight Quantization):
 *
 * Instead of quantizing all channels uniformly, AWQ scales weights
 * by per-channel importance (activation scales) before quantization,
 * reducing quantization error for salient channels.
 *
 * Reference: "AWQ: Activation-aware Weight Quantization for LLM
 *             Compression and Acceleration" (Lin et al., 2024)
 */

static float _compute_best_scale(const float* w, size_t col, size_t cols,
                                  size_t group_size, const float* act_scale)
{
    float org_max = 0.0f;
    size_t start = (col / group_size) * group_size;
    size_t end = SNEPPX_MIN(start + group_size, cols);
    for (size_t j = start; j < end; j++) {
        float a = fabsf(w[j]);
        if (a > org_max) org_max = a;
    }
    if (org_max < FLT_MIN) return 1.0f;
    /*
     * Find best scale s in [0.5, 1.0] via grid search that minimizes
     * quantization error for this channel:
     *   loss(s) = || Q(w * s / s_max) * s_max / s - w ||
     * Quantize to INT8, dequantize, measure MSE, pick best s.
     */
    float best_s = 1.0f;
    float best_loss = 1e20f;
    float alpha = (act_scale) ? SNEPPX_MIN(act_scale[col], 1.0f) : 0.5f;
    float s_step = (1.0f - alpha) / 20.0f;
    if (s_step < 0.001f) s_step = 0.001f;
    for (float s = alpha; s <= 1.0f; s += s_step) {
        float inv_s = 1.0f / s;
        float max_q = 0.0f;
        for (size_t j = start; j < end; j++) {
            float q = fabsf(w[j] * s);
            if (q > max_q) max_q = q;
        }
        float q_scale = (max_q < FLT_MIN) ? 1.0f : max_q / 127.0f;
        float loss = 0.0f;
        for (size_t j = start; j < end; j++) {
            int qi = (int)roundf(w[j] * s / q_scale);
            qi = SNEPPX_CLAMP(qi, -128, 127);
            float dq = (float)qi * q_scale * inv_s;
            float diff = dq - w[j];
            loss += diff * diff;
        }
        loss /= (float)(end - start);
        if (loss < best_loss) {
            best_loss = loss;
            best_s = s;
        }
    }
    return best_s;
}

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
                              const float* act_scales, float* scale_out)
{
    if (!weights || !act_scales || !scale_out || rows == 0 || cols == 0) return -1;
    int group_size = 128;
    for (size_t r = 0; r < rows; r++) {
        float* row = weights + r * cols;
        for (size_t c = 0; c < cols; c++) {
            float s = _compute_best_scale(row, c, cols, group_size, act_scales);
            scale_out[r * cols + c] = s;
            row[c] *= s;
        }
    }
    return 0;
}

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
                         size_t rows, size_t cols, int group_size)
{
    if (!weights || !qweight || !scales || rows == 0 || cols == 0) return -1;
    if (group_size <= 0) group_size = 128;
    size_t num_groups = (cols + group_size - 1) / group_size;
    for (size_t r = 0; r < rows; r++) {
        for (size_t g = 0; g < num_groups; g++) {
            size_t g_start = g * group_size;
            size_t g_end = SNEPPX_MIN(g_start + group_size, cols);
            float max_abs = 0.0f;
            for (size_t c = g_start; c < g_end; c++) {
                float a = fabsf(weights[r * cols + c]);
                if (a > max_abs) max_abs = a;
            }
            float scale = (max_abs < FLT_MIN) ? 1.0f : max_abs / 127.0f;
            scales[r * num_groups + g] = scale;
            float inv_scale = 1.0f / scale;
            for (size_t c = g_start; c < g_end; c++) {
                int qi = (int)roundf(weights[r * cols + c] * inv_scale);
                qweight[r * cols + c] = (int8_t)SNEPPX_CLAMP(qi, -128, 127);
            }
        }
    }
    return 0;
}

/**
 * @brief Create Quant Params.
 *
 * @param params [out] Params value.
 * @param mode [in] Mode value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_quant_params_create(SNEPPXQuantParams* params, SNEPPXQuantMode mode)
{
    if (!params) return -1;
    params->scale = 1.0f;
    params->zero_point = 0;
    params->scale_max = 0.0f;
    switch (mode) {
        case SNEPPX_QUANT_INT8_SYM:
            params->qmin = -128.0f;
            params->qmax = 127.0f;
            break;
        case SNEPPX_QUANT_INT8_ASYM:
            params->qmin = 0.0f;
            params->qmax = 255.0f;
            break;
        case SNEPPX_QUANT_INT4_SYM:
            params->qmin = -8.0f;
            params->qmax = 7.0f;
            break;
        case SNEPPX_QUANT_FP8_E4M3:
        case SNEPPX_QUANT_FP8_E5M2:
            params->qmin = -448.0f;
            params->qmax = 448.0f;
            break;
        default:
            params->qmin = -127.0f;
            params->qmax = 127.0f;
            break;
    }
    return 0;
}

/**
 * @brief Perform Quant Mode Name.
 *
 * @param mode [in] Mode value.
 *
 * @return Pointer on success, NULL on error.
 */
const char* SNEPPX_quant_mode_name(SNEPPXQuantMode mode)
{
    switch (mode) {
        case SNEPPX_QUANT_NONE:     return "none";
        case SNEPPX_QUANT_INT8_SYM:  return "int8_sym";
        case SNEPPX_QUANT_INT8_ASYM: return "int8_asym";
        case SNEPPX_QUANT_INT4_SYM:  return "int4_sym";
        case SNEPPX_QUANT_FP8_E4M3:  return "fp8_e4m3";
        case SNEPPX_QUANT_FP8_E5M2:  return "fp8_e5m2";
        case SNEPPX_QUANT_AWQ:       return "awq";
        case SNEPPX_QUANT_GPTQ:      return "gptq";
        default: return "unknown";
    }
}

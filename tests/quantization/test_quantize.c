/*
 * SNEPPX - Test Quantize
 *
 * WHAT
 *   Test Quantize.
 *
 * CONCEPT
 *   Provides the Test Quantize.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/**
 * SNEPPX Quantization C Test Suite
 * Tests INT8, FP8, AWQ, GPTQ host-side implementations.
 */
#include "quantization.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int test_int8_sym_roundtrip(void)
{
    float input[4] = {-1.0f, 0.0f, 1.0f, 2.5f};
    int8_t quantized[4];
    float output[4];
    float scale;
    int ret = SNEPPX_quantize_int8_sym(input, quantized, 4, &scale);
    assert(ret == 0);
    ret = SNEPPX_dequantize_int8_sym(quantized, output, 4, scale);
    assert(ret == 0);
    float max_err = 0.0f;
    for (int i = 0; i < 4; i++) {
        float e = fabsf(output[i] - input[i]);
        if (e > max_err) max_err = e;
    }
    printf("  INT8 sym roundtrip max error: %.6f (scale=%f)\n", max_err, scale);
    assert(max_err < 0.02f);
    return 0;
}

static int test_int8_sym_zero(void)
{
    float input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int8_t quantized[4];
    float output[4];
    float scale;
    SNEPPX_quantize_int8_sym(input, quantized, 4, &scale);
    SNEPPX_dequantize_int8_sym(quantized, output, 4, scale);
    for (int i = 0; i < 4; i++) assert(output[i] == 0.0f);
    printf("  INT8 sym zero PASS\n");
    return 0;
}

static int test_int8_asym_roundtrip(void)
{
    float input[5] = {0.5f, 1.5f, 2.0f, 0.0f, -1.0f};
    int8_t quantized[5];
    float output[5];
    float scale;
    int32_t zp;
    int ret = SNEPPX_quantize_int8_asym(input, quantized, 5, &scale, &zp);
    assert(ret == 0);
    ret = SNEPPX_dequantize_int8_asym(quantized, output, 5, scale, zp);
    assert(ret == 0);
    float max_err = 0.0f;
    for (int i = 0; i < 5; i++) {
        float e = fabsf(output[i] - input[i]);
        if (e > max_err) max_err = e;
    }
    printf("  INT8 asym roundtrip max error: %.6f (scale=%f, zp=%d)\n", max_err, scale, (int)zp);
    assert(max_err < 0.05f);
    return 0;
}

static int test_int8_channel(void)
{
    float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f};
    int8_t output[6];
    float scales[2];
    int ret = SNEPPX_quantize_int8_channel(input, output, 2, 3, scales);
    assert(ret == 0);
    printf("  INT8 channel scales: %f, %f\n", scales[0], scales[1]);
    return 0;
}

static int test_int4_roundtrip(void)
{
    float input[6] = {1.0f, -1.0f, 3.5f, -3.0f, 0.5f, -0.5f};
    uint8_t packed[3];
    float output[6], scale;
    SNEPPX_quantize_int4_sym(input, packed, 6, &scale);
    SNEPPX_dequantize_int4_sym(packed, output, 6, scale);
    float max_err = 0.0f;
    for (int i = 0; i < 6; i++) {
        float e = fabsf(output[i] - input[i]);
        if (e > max_err) max_err = e;
    }
    printf("  INT4 sym roundtrip max error: %.6f (scale=%f)\n", max_err, scale);
    assert(max_err < 1.0f);
    return 0;
}

static int test_fp8_e4m3_encdec(void)
{
    float test_vals[] = {0.0f, 1.0f, -1.0f, 2.5f, 127.0f, -64.0f, 0.125f};
    for (int i = 0; i < 7; i++) {
        uint8_t fp8 = SNEPPX_float_to_fp8_e4m3(test_vals[i]);
        float dec = SNEPPX_fp8_e4m3_to_float(fp8);
        printf("  E4M3: %.4f -> 0x%02X -> %.4f\n", test_vals[i], fp8, dec);
    }
    return 0;
}

static int test_fp8_e5m2_encdec(void)
{
    float test_vals[] = {0.0f, 1.0f, -1.0f, 256.0f, -128.0f, 0.25f};
    for (int i = 0; i < 6; i++) {
        uint8_t fp8 = SNEPPX_float_to_fp8_e5m2(test_vals[i]);
        float dec = SNEPPX_fp8_e5m2_to_float(fp8);
        printf("  E5M2: %.4f -> 0x%02X -> %.4f\n", test_vals[i], fp8, dec);
    }
    return 0;
}

static int test_awq_quantize(void)
{
    float weights[8] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    float act_scales[4] = {0.8f, 1.0f, 0.6f, 0.9f};
    int8_t qweight[8];
    float scales[4];
    int ret = SNEPPX_awq_quantize(weights, qweight, act_scales, scales, 2, 4, 2);
    assert(ret == 0);
    printf("  AWQ quantize: scales = %f, %f, %f, %f\n", scales[0], scales[1], scales[2], scales[3]);
    return 0;
}

static int test_gptq_hessian(void)
{
    float acts[12] = {1,0, 2,0, 3,0, 4,0, 5,0, 6,0, 7,0, 8,0, 9,0, 10,0, 11,0, 12,0};
    float H[9];
    int ret = SNEPPX_gptq_compute_hessian(H, acts, 4, 3);
    assert(ret == 0);
    printf("  GPTQ Hessian: %.2f %.2f %.2f / %.2f %.2f %.2f / %.2f %.2f %.2f\n",
           H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]);
    return 0;
}

static int test_quant_mode_names(void)
{
    assert(strcmp(SNEPPX_quant_mode_name(SNEPPX_QUANT_INT8_SYM), "int8_sym") == 0);
    assert(strcmp(SNEPPX_quant_mode_name(SNEPPX_QUANT_FP8_E4M3), "fp8_e4m3") == 0);
    assert(strcmp(SNEPPX_quant_mode_name(SNEPPX_QUANT_AWQ), "awq") == 0);
    assert(strcmp(SNEPPX_quant_mode_name(SNEPPX_QUANT_GPTQ), "gptq") == 0);
    printf("  Quant mode names PASS\n");
    return 0;
}

int main(void)
{
    printf("=== SNEPPX Quantization C Tests ===\n\n");
    test_int8_sym_roundtrip();
    test_int8_sym_zero();
    test_int8_asym_roundtrip();
    test_int8_channel();
    test_int4_roundtrip();
    test_fp8_e4m3_encdec();
    test_fp8_e5m2_encdec();
    test_awq_quantize();
    test_gptq_hessian();
    test_quant_mode_names();
    printf("\nAll C quantization tests PASSED.\n");
    return 0;
}

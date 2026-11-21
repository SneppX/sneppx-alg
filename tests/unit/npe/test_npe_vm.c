#include "neural_programming_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Npe Vm
 *
 * WHAT
 *   Test Npe Vm.
 *
 * CONCEPT
 *   Provides the Test Npe Vm.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_vm_create(void) {
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SX_ASSERT(vm != NULL, "vm not null");
    SX_ASSERT(vm->program == NULL, "no program loaded");
    SNEPPX_npe_vm_destroy(vm);
}

static void test_vm_nop(void) {
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(p, inst);

    SNEPPX_npe_vm_load(vm, p);
    size_t sh[] = {2, 3};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(sh, 2, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;
    int r = SNEPPX_npe_vm_run(vm, input, &output);
    SX_ASSERT(r == 0, "run success");
    SNEPPX_tensor_destroy(input);
    if (output) SNEPPX_tensor_destroy(output);
    SNEPPX_npe_program_destroy(p);
    SNEPPX_npe_vm_destroy(vm);
}

static void test_vm_add(void) {
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(32);

    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {10.0f, 20.0f, 30.0f, 40.0f};
    memcpy((float*)p->memory->data + 0, a_vals, 4 * sizeof(float));
    memcpy((float*)p->memory->data + 4, b_vals, 4 * sizeof(float));

    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = 4;
    SNEPPX_npe_program_append(p, inst);

    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 2; inst.immediate = 4;
    inst.shape_a[0] = 4;
    SNEPPX_npe_program_append(p, inst);

    inst.opcode = SNEPPX_ADD; inst.dest_reg = 3; inst.src_reg_a = 1; inst.src_reg_b = 2;
    SNEPPX_npe_program_append(p, inst);

    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(p, inst);

    SNEPPX_npe_vm_load(vm, p);
    size_t sh[] = {4};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;
    int r = SNEPPX_npe_vm_run(vm, input, &output);
    SX_ASSERT(r == 0, "run success");
    SX_ASSERT(output != NULL, "output not null");
    float* od = (float*)output->data;
    SX_ASSERT(fabsf(od[0] - 11.0f) < 1e-5f, "1+10=11");
    SX_ASSERT(fabsf(od[1] - 22.0f) < 1e-5f, "2+20=22");
    SX_ASSERT(fabsf(od[2] - 33.0f) < 1e-5f, "3+30=33");
    SX_ASSERT(fabsf(od[3] - 44.0f) < 1e-5f, "4+40=44");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_npe_program_destroy(p);
    SNEPPX_npe_vm_destroy(vm);
}

static void test_vm_matmul(void) {
    SNEPPXNPEConfig cfg = SNEPPX_npe_config_default();
    SNEPPXNPEVM* vm = SNEPPX_npe_vm_create(&cfg);
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(32);

    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    memcpy((float*)p->memory->data + 0, a_vals, 4 * sizeof(float));
    memcpy((float*)p->memory->data + 4, b_vals, 8 * sizeof(float));

    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = 2; inst.shape_a[1] = 2;
    SNEPPX_npe_program_append(p, inst);

    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 2; inst.immediate = 4;
    inst.shape_a[0] = 2; inst.shape_a[1] = 4;
    SNEPPX_npe_program_append(p, inst);

    inst.opcode = SNEPPX_MATMUL; inst.dest_reg = 3; inst.src_reg_a = 1; inst.src_reg_b = 2;
    SNEPPX_npe_program_append(p, inst);

    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(p, inst);

    SNEPPX_npe_vm_load(vm, p);
    size_t sh[] = {1};
    SNEPPXTensor* input = SNEPPX_tensor_zeros(sh, 1, SNEPPX_FLOAT32);
    SNEPPXTensor* output = NULL;
    int r = SNEPPX_npe_vm_run(vm, input, &output);
    SX_ASSERT(r == 0, "run success");
    SX_ASSERT(output != NULL, "output not null");
    SX_ASSERT(output->shape[0] == 2 && output->shape[1] == 4, "output [2x4]");

    SNEPPX_tensor_destroy(input);
    SNEPPX_tensor_destroy(output);
    SNEPPX_npe_program_destroy(p);
    SNEPPX_npe_vm_destroy(vm);
}


TEST(test_npe_vm, test_vm_create) { test_vm_create(); }
TEST(test_npe_vm, test_vm_nop) { test_vm_nop(); }
TEST(test_npe_vm, test_vm_add) { test_vm_add(); }
TEST(test_npe_vm, test_vm_matmul) { test_vm_matmul(); }

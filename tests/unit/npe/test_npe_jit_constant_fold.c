#include "neural_programming_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
 * SNEPPX - Test Npe Jit Constant Fold
 *
 * WHAT
 *   Test Npe Jit Constant Fold.
 *
 * CONCEPT
 *   Provides the Test Npe Jit Constant Fold.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_constant_fold_identity(void) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(64);
    SX_ASSERT(prog != NULL, "program created");

    SNEPPXNPEInstruction load = {SNEPPX_LOAD, 0, -1, -1, 0, {1, 1}, {0, 0}};
    SNEPPX_npe_program_append(prog, load);

    SNEPPXNPEInstruction halt = {SNEPPX_HALT, -1, -1, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, halt);

    SNEPPXTensor mem;
    float mem_data[] = {42.0f};
    mem.data = mem_data;
    mem.size = 1;

    SNEPPXNPEProgram* folded = SNEPPX_npe_jit_constant_fold(prog, &mem);
    SX_ASSERT(folded != NULL, "folded program created");
    SX_ASSERT(folded->num_instructions == 2, "same number of instructions");

    SNEPPX_npe_program_destroy(folded);
    SNEPPX_npe_program_destroy(prog);
}

static void test_constant_fold_add(void) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(64);
    SX_ASSERT(prog != NULL, "program created");

    SNEPPXNPEInstruction load_a = {SNEPPX_LOAD, 0, -1, -1, 0, {1, 1}, {0, 0}};
    SNEPPX_npe_program_append(prog, load_a);
    SNEPPXNPEInstruction load_b = {SNEPPX_LOAD, 1, -1, -1, 1, {1, 1}, {0, 0}};
    SNEPPX_npe_program_append(prog, load_b);
    SNEPPXNPEInstruction add = {SNEPPX_ADD, 2, 0, 1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, add);
    SNEPPXNPEInstruction halt = {SNEPPX_HALT, -1, -1, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, halt);

    float mem_data[] = {10.0f, 20.0f};
    SNEPPXTensor mem;
    mem.data = mem_data;
    mem.size = 2;

    SNEPPXNPEProgram* folded = SNEPPX_npe_jit_constant_fold(prog, &mem);
    SX_ASSERT(folded != NULL, "folded program created");
    SX_ASSERT(folded->num_instructions == 4, "instructions preserved");

    SNEPPX_npe_program_destroy(folded);
    SNEPPX_npe_program_destroy(prog);
}

static void test_constant_fold_null_memory(void) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction load = {SNEPPX_LOAD, 0, -1, -1, 0, {1, 1}, {0, 0}};
    SNEPPX_npe_program_append(prog, load);
    SNEPPXNPEInstruction halt = {SNEPPX_HALT, -1, -1, -1, 0, {0, 0}, {0, 0}};
    SNEPPX_npe_program_append(prog, halt);

    SNEPPXNPEProgram* folded = SNEPPX_npe_jit_constant_fold(prog, NULL);
    SX_ASSERT(folded != NULL, "folded with null memory");

    SNEPPX_npe_program_destroy(folded);
    SNEPPX_npe_program_destroy(prog);
}


TEST(test_npe_jit_constant_fold, constant_fold_identity) { test_constant_fold_identity(); }
TEST(test_npe_jit_constant_fold, constant_fold_add) { test_constant_fold_add(); }
TEST(test_npe_jit_constant_fold, constant_fold_null_memory) { test_constant_fold_null_memory(); }

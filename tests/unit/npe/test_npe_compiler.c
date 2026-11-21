#include "neural_programming_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Npe Compiler
 *
 * WHAT
 *   Test Npe Compiler.
 *
 * CONCEPT
 *   Provides message passing interface.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_compile_attention(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_compile_attention(8, 16);
    SX_ASSERT(p != NULL, "program not null");
    SX_ASSERT(p->num_instructions > 5, "more than 5 instructions");
    int has_softmax = 0, has_matmul = 0;
    for (size_t i = 0; i < p->num_instructions; i++) {
        if (p->instructions[i].opcode == SNEPPX_SOFTMAX) has_softmax = 1;
        if (p->instructions[i].opcode == SNEPPX_MATMUL) has_matmul = 1;
    }
    SX_ASSERT(has_softmax, "contains SOFTMAX");
    SX_ASSERT(has_matmul, "contains MATMUL");
    SNEPPX_npe_program_destroy(p);
}

static void test_compile_mlp(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_compile_mlp(8, 16);
    SX_ASSERT(p != NULL, "program not null");
    SX_ASSERT(p->num_instructions > 5, "more than 5 instructions");
    int has_relu = 0, has_matmul = 0, has_add = 0;
    for (size_t i = 0; i < p->num_instructions; i++) {
        if (p->instructions[i].opcode == SNEPPX_RELU) has_relu = 1;
        if (p->instructions[i].opcode == SNEPPX_MATMUL) has_matmul = 1;
        if (p->instructions[i].opcode == SNEPPX_ADD) has_add = 1;
    }
    SX_ASSERT(has_relu, "contains RELU");
    SX_ASSERT(has_matmul, "contains MATMUL");
    SX_ASSERT(has_add, "contains ADD");
    SNEPPX_npe_program_destroy(p);
}


TEST(test_npe_compiler, test_compile_attention) { test_compile_attention(); }
TEST(test_npe_compiler, test_compile_mlp) { test_compile_mlp(); }

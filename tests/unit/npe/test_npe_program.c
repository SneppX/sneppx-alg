#include "neural_programming_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test Npe Program
 *
 * WHAT
 *   Test Npe Program.
 *
 * CONCEPT
 *   Provides the Test Npe Program.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_program_create(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(64);
    SX_ASSERT(p != NULL, "program not null");
    SX_ASSERT(p->num_instructions == 0, "empty program");
    SX_ASSERT(p->max_instructions == 64, "max 64");
    SX_ASSERT(p->registers[0] == NULL, "reg 0 null");
    SX_ASSERT(p->registers[15] == NULL, "reg 15 null");
    SX_ASSERT(p->memory != NULL, "memory not null");
    SNEPPX_npe_program_destroy(p);
}

static void test_program_append(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(64);
    SX_ASSERT(p != NULL, "program not null");
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_NOP;
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SNEPPX_npe_program_append(p, inst);
    SX_ASSERT(p->num_instructions == 5, "5 instructions");
    SX_ASSERT(p->instructions[0].opcode == SNEPPX_NOP, "opcode NOP");
    SNEPPX_npe_program_destroy(p);
}


TEST(test_npe_program, test_program_create) { test_program_create(); }
TEST(test_npe_program, test_program_append) { test_program_append(); }

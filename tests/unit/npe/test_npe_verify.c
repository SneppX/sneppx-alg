#include "neural_programming_engine.h"
#include "test_gtest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * SNEPPX - Test Npe Verify
 *
 * WHAT
 *   Test Npe Verify.
 *
 * CONCEPT
 *   Provides the Test Npe Verify.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_verify_valid(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_NOP;
    SNEPPX_npe_program_append(p, inst);
    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(p, inst);

    char* err = NULL;
    size_t err_len = 0;
    int r = SNEPPX_npe_verify_program(p, &err, &err_len);
    SX_ASSERT(r != 0, "valid program passes");
    free(err);
    SNEPPX_npe_program_destroy(p);
}

static void test_verify_invalid_reg(void) {
    SNEPPXNPEProgram* p = SNEPPX_npe_program_create(16);
    SNEPPXNPEInstruction inst; memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_NOP; inst.dest_reg = 20;
    SNEPPX_npe_program_append(p, inst);
    inst.opcode = SNEPPX_HALT; inst.dest_reg = -1;
    SNEPPX_npe_program_append(p, inst);

    char* err = NULL;
    size_t err_len = 0;
    int r = SNEPPX_npe_verify_program(p, &err, &err_len);
    SX_ASSERT(r == 0, "invalid reg fails");
    SX_ASSERT(err_len > 0, "error message not empty");
    free(err);
    SNEPPX_npe_program_destroy(p);
}


TEST(test_npe_verify, test_verify_valid) { test_verify_valid(); }
TEST(test_npe_verify, test_verify_invalid_reg) { test_verify_invalid_reg(); }

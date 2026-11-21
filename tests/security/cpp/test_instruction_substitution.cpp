#include "instruction_obfuscation_engine.h"
#include "control_flow_obfuscation.h"
#include <stdio.h>
#include <cassert>

/*
 * SNEPPX - Test Instruction Substitution
 *
 * WHAT
 *   Test Instruction Substitution.
 *
 * CONCEPT
 *   Provides the Test Instruction Substitution.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_nand_and_substitution(void) {
    SNEPPX::SNEPPXObfSubst subst;
    SNEPPX::SNEPPXObfCFG cfg;
    uint64_t block_id = cfg.add_block();
    auto block = cfg.blocks[block_id];

    SNEPPX::SNEPPXObfInstruction inst;
    inst.type = SNEPPX::SNEPPXObfInstType::AND;
    inst.operand1 = "a";
    inst.operand2 = "b";
    inst.result = "r";
    block->instructions.push_back(inst);

    subst.substitute_logic(*block);

    /* AND should be replaced with NAND-based sequence */
    bool found_nand = false;
    for (auto& instr : block->instructions) {
        if (instr.type == SNEPPX::SNEPPXObfInstType::NAND) found_nand = true;
    }
    ASSERT(found_nand, "AND substitution uses NAND instructions");
}

static void test_nand_or_substitution(void) {
    SNEPPX::SNEPPXObfSubst subst;
    SNEPPX::SNEPPXObfCFG cfg;
    uint64_t block_id = cfg.add_block();
    auto block = cfg.blocks[block_id];

    SNEPPX::SNEPPXObfInstruction inst;
    inst.type = SNEPPX::SNEPPXObfInstType::OR;
    inst.operand1 = "x";
    inst.operand2 = "y";
    inst.result = "z";
    block->instructions.push_back(inst);

    subst.substitute_logic(*block);

    bool found_nand = false;
    for (auto& instr : block->instructions) {
        if (instr.type == SNEPPX::SNEPPXObfInstType::NAND) found_nand = true;
    }
    ASSERT(found_nand, "OR substitution uses NAND instructions");
}

static void test_nand_xor_substitution(void) {
    SNEPPX::SNEPPXObfSubst subst;
    SNEPPX::SNEPPXObfCFG cfg;
    uint64_t block_id = cfg.add_block();
    auto block = cfg.blocks[block_id];

    SNEPPX::SNEPPXObfInstruction inst;
    inst.type = SNEPPX::SNEPPXObfInstType::XOR;
    inst.operand1 = "p";
    inst.operand2 = "q";
    inst.result = "r";
    block->instructions.push_back(inst);

    subst.substitute_logic(*block);

    bool found_nand = false;
    int nand_count = 0;
    for (auto& instr : block->instructions) {
        if (instr.type == SNEPPX::SNEPPXObfInstType::NAND) {
            found_nand = true;
            nand_count++;
        }
        ASSERT(instr.type != SNEPPX::SNEPPXObfInstType::NOP, "no NOP instructions remain");
    }
    ASSERT(found_nand, "XOR substitution uses NAND instructions");
    ASSERT(nand_count >= 3, "XOR requires at least 3 NANDs");
}

int main(void) {
    run_test("nand_and_substitution", test_nand_and_substitution);
    run_test("nand_or_substitution", test_nand_or_substitution);
    run_test("nand_xor_substitution", test_nand_xor_substitution);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

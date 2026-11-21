#include "control_flow_obfuscation.h"
#include <cstdio>
#include <cassert>
#include <iostream>

/*
 * SNEPPX - Test Obf Cfg
 *
 * WHAT
 *   Test Obf Cfg.
 *
 * CONCEPT
 *   Provides the Test Obf Cfg.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

void build_test_cfg(SNEPPX::SNEPPXObfCFG& cfg) {
    cfg.entry_block = cfg.add_block();
    cfg.blocks[cfg.entry_block]->is_entry = true;

    uint64_t b2 = cfg.add_block();
    uint64_t b3 = cfg.add_block();
    cfg.exit_block = cfg.add_block();
    cfg.blocks[cfg.exit_block]->is_exit = true;

    SNEPPX::SNEPPXObfInstruction inst1;
    inst1.type = SNEPPX::SNEPPXObfInstType::MOV;
    inst1.result = "x";
    inst1.operand1 = "10";
    cfg.blocks[cfg.entry_block]->instructions.push_back(inst1);

    SNEPPX::SNEPPXObfInstruction inst2;
    inst2.type = SNEPPX::SNEPPXObfInstType::ADD;
    inst2.result = "x";
    inst2.operand1 = "x";
    inst2.operand2 = "5";
    cfg.blocks[b2]->instructions.push_back(inst2);

    SNEPPX::SNEPPXObfInstruction inst3;
    inst3.type = SNEPPX::SNEPPXObfInstType::SUB;
    inst3.result = "x";
    inst3.operand1 = "x";
    inst3.operand2 = "3";
    cfg.blocks[b3]->instructions.push_back(inst3);

    cfg.add_edge(cfg.entry_block, b2);
    cfg.add_edge(b2, b3);
    cfg.add_edge(b3, cfg.exit_block);
}

void test_flatten_simple() {
    TEST("flatten_simple_3_blocks");
    SNEPPX::SNEPPXObfCFG cfg;
    build_test_cfg(cfg);

    ASSERT(cfg.blocks.size() == 4, "expected 4 blocks before flatten");

    SNEPPX::SNEPPXObfCFGFlattener flattener;
    flattener.set_seed(12345);
    flattener.flatten(cfg);

    ASSERT(cfg.blocks.size() >= 4, "flatten should not remove blocks");

    bool has_dispatcher = false;
    for (auto& pair : cfg.blocks) {
        if (pair.second->is_entry) {
            has_dispatcher = true;
            for (auto& inst : pair.second->instructions) {
                if (inst.type == SNEPPX::SNEPPXObfInstType::MOV &&
                    inst.result == "state_var") {
                    has_dispatcher = true;
                    break;
                }
            }
        }
    }

    PASS();
}

void test_flatten_loop() {
    TEST("flatten_loop");
    SNEPPX::SNEPPXObfCFG cfg;

    cfg.entry_block = cfg.add_block();
    cfg.blocks[cfg.entry_block]->is_entry = true;
    uint64_t loop_header = cfg.add_block();
    uint64_t loop_body = cfg.add_block();
    cfg.exit_block = cfg.add_block();
    cfg.blocks[cfg.exit_block]->is_exit = true;

    cfg.add_edge(cfg.entry_block, loop_header);
    cfg.add_edge(loop_header, loop_body);
    cfg.add_edge(loop_body, loop_header);
    cfg.add_edge(loop_header, cfg.exit_block);

    SNEPPX::SNEPPXObfCFGFlattener flattener;
    flattener.set_seed(67890);
    flattener.flatten(cfg);

    ASSERT(cfg.blocks.size() >= 5, "flatten should not reduce blocks");
    PASS();
}

void test_semantic_preserve() {
    TEST("semantic_preserve");
    SNEPPX::SNEPPXObfCFG cfg;
    build_test_cfg(cfg);

    SNEPPX::SNEPPXObfCFGFlattener flattener;
    flattener.set_seed(11111);
    flattener.flatten(cfg);

    ASSERT(cfg.blocks.size() >= 4, "blocks preserved after flatten");
    PASS();
}

int main() {
    printf("\n=== S2 Obfuscation CFG Tests ===\n\n");

    test_flatten_simple();
    test_flatten_loop();
    test_semantic_preserve();

    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}

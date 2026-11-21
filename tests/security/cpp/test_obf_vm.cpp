#include "virtualized_code_execution.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <vector>

/*
 * SNEPPX - Test Obf Vm
 *
 * WHAT
 *   Test Obf Vm.
 *
 * CONCEPT
 *   Provides the Test Obf Vm.
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

void test_vm_add() {
    TEST("vm_add_registers");
    SNEPPX::SNEPPXObfVM vm;

    std::vector<uint8_t> bytecode;

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(0);
    bytecode.push_back(10);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(1);
    bytecode.push_back(20);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::ADD));
    bytecode.push_back(0);
    bytecode.push_back(1);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::HALT));
    bytecode.push_back(0);
    bytecode.push_back(0);
    bytecode.push_back(0);

    vm.load_bytecode(bytecode);
    vm.vm_execute(bytecode.data(), bytecode.size());

    ASSERT(vm.state().regs[0] == 30, "expected 10 + 20 = 30");
    PASS();
}

void test_vm_loop() {
    TEST("vm_simple_loop");
    SNEPPX::SNEPPXObfVM vm;

    std::vector<uint8_t> bytecode;

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(0);
    bytecode.push_back(0);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(1);
    bytecode.push_back(5);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::ADD));
    bytecode.push_back(0);
    bytecode.push_back(1);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::HALT));
    bytecode.push_back(0);
    bytecode.push_back(0);
    bytecode.push_back(0);

    vm.load_bytecode(bytecode);
    vm.vm_execute(bytecode.data(), bytecode.size());

    ASSERT(vm.state().regs[0] == 5, "expected 0 + 5 = 5");
    PASS();
}

void test_handler_encryption() {
    TEST("handler_encryption");
    SNEPPX::SNEPPXObfVM vm;

    vm.encrypt_handler_table();

    std::vector<uint8_t> bytecode;
    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(0);
    bytecode.push_back(42);
    bytecode.push_back(0);
    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::HALT));
    bytecode.push_back(0);
    bytecode.push_back(0);
    bytecode.push_back(0);

    vm.load_bytecode(bytecode);
    vm.vm_execute(bytecode.data(), bytecode.size());

    ASSERT(vm.state().regs[0] == 42, "expected handler to decrypt and execute");
    PASS();
}

void test_vm_arithmetic() {
    TEST("vm_arithmetic_chain");
    SNEPPX::SNEPPXObfVM vm;

    std::vector<uint8_t> bytecode;

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(0);
    bytecode.push_back(7);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(1);
    bytecode.push_back(3);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::MUL));
    bytecode.push_back(0);
    bytecode.push_back(1);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::LOAD));
    bytecode.push_back(2);
    bytecode.push_back(2);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::SUB));
    bytecode.push_back(0);
    bytecode.push_back(2);
    bytecode.push_back(0);

    bytecode.push_back(static_cast<uint8_t>(SNEPPX::SNEPPXObfBytecode::HALT));
    bytecode.push_back(0);
    bytecode.push_back(0);
    bytecode.push_back(0);

    vm.load_bytecode(bytecode);
    vm.vm_execute(bytecode.data(), bytecode.size());

    ASSERT(vm.state().regs[0] == 19, "expected (7*3)-2 = 19");
    PASS();
}

int main() {
    printf("\n=== S2 Obfuscation VM Tests ===\n\n");

    test_vm_add();
    test_vm_loop();
    test_handler_encryption();
    test_vm_arithmetic();

    printf("\nResults: %d passed, %d failed out of %d\n",
           tests_passed, tests_failed, tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}

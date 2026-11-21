#include "code_obfuscation_interface.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>

/*
 * SNEPPX - Obf Demo
 *
 * WHAT
 *   Obf Demo.
 *
 * CONCEPT
 *   Provides the Obf Demo.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int original_add(int a, int b) {
    return a + b;
}

int main() {
    printf("\n=== SNEPPX S2 Obfuscation Demo ===\n\n");

    // Build a simple CFG representing add(a, b)
    SNEPPX::SNEPPXObfCFG cfg;

    cfg.entry_block = cfg.add_block();
    cfg.blocks[cfg.entry_block]->is_entry = true;

    uint64_t body = cfg.add_block();
    cfg.exit_block = cfg.add_block();
    cfg.blocks[cfg.exit_block]->is_exit = true;

    // Block 0: setup params
    SNEPPX::SNEPPXObfInstruction load_a;
    load_a.type = SNEPPX::SNEPPXObfInstType::MOV;
    load_a.result = "a";
    load_a.operand1 = "input_a";
    cfg.blocks[cfg.entry_block]->instructions.push_back(load_a);

    SNEPPX::SNEPPXObfInstruction load_b;
    load_b.type = SNEPPX::SNEPPXObfInstType::MOV;
    load_b.result = "b";
    load_b.operand1 = "input_b";
    cfg.blocks[cfg.entry_block]->instructions.push_back(load_b);

    // Block 1: add
    SNEPPX::SNEPPXObfInstruction add_inst;
    add_inst.type = SNEPPX::SNEPPXObfInstType::ADD;
    add_inst.result = "result";
    add_inst.operand1 = "a";
    add_inst.operand2 = "b";
    cfg.blocks[body]->instructions.push_back(add_inst);

    // Block 2: return
    SNEPPX::SNEPPXObfInstruction ret_inst;
    ret_inst.type = SNEPPX::SNEPPXObfInstType::RET;
    ret_inst.operand1 = "result";
    cfg.blocks[cfg.exit_block]->instructions.push_back(ret_inst);

    cfg.add_edge(cfg.entry_block, body);
    cfg.add_edge(body, cfg.exit_block);

    // Original execution timing
    auto orig_start = std::chrono::high_resolution_clock::now();
    int orig_result = original_add(2, 3);
    auto orig_end = std::chrono::high_resolution_clock::now();
    auto orig_us = std::chrono::duration_cast<std::chrono::nanoseconds>(orig_end - orig_start).count();

    printf("Original add(2, 3): %d (took %lld ns)\n", orig_result, (long long)orig_us);

    // Configure obfuscator at HEAVY level
    SNEPPX::SNEPPXObfuscator obfuscator;
    obfuscator.anti_debug().set_action(SNEPPX::SNEPPXAntiDebugAction::DELAY);
    obfuscator.configure(SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_HEAVY);

    auto obf_start = std::chrono::high_resolution_clock::now();
    SNEPPX::SNEPPXObfuscationReport report = obfuscator.obfuscate(cfg);
    auto obf_end = std::chrono::high_resolution_clock::now();
    auto obf_us = std::chrono::duration_cast<std::chrono::microseconds>(obf_end - obf_start).count();

    printf("\n--- Obfuscation Report ---\n");
    printf("Level:                    ");
    switch (report.level) {
        case SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_NONE:    printf("NONE\n"); break;
        case SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_LIGHT:   printf("LIGHT\n"); break;
        case SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_MEDIUM:  printf("MEDIUM\n"); break;
        case SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_HEAVY:   printf("HEAVY\n"); break;
        case SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_MAXIMUM: printf("MAXIMUM\n"); break;
    }
    printf("Transformations applied:  %zu\n", report.transformations_applied);
    printf("Blocks flattened:         %zu\n", report.blocks_flattened);
    printf("Strings encrypted:        %zu\n", report.strings_encrypted);
    printf("Instructions substituted: %zu\n", report.instructions_substituted);
    printf("Opaque predicates:        %zu\n", report.opaque_predicates_inserted);
    printf("Bytecode size:            %zu bytes\n", report.bytecode_size);
    printf("Virtualization:           %s\n", report.virtualization_applied ? "yes" : "no");
    printf("Anti-debug:               %s\n", report.anti_debug_applied ? "yes" : "no");
    printf("Obfuscation time:         %.3f ms\n", report.execution_time_ms);

    printf("\n--- Decrypted Strings ---\n");
    for (size_t i = 0; i < obfuscator.string_pool().count(); i++) {
        const char* s = obfuscator.string_pool().get_string(i);
        if (s) printf("  [%zu]: %s\n", i, s);
    }

    printf("\n--- Block Structure ---\n");
    printf("Total blocks: %zu\n", cfg.blocks.size());
    for (auto& pair : cfg.blocks) {
        printf("  Block %llu %s %s (instrs: %zu)\n",
               (unsigned long long)pair.first,
               pair.second->is_entry ? "[ENTRY]" : "",
               pair.second->is_exit ? "[EXIT]" : "",
               pair.second->instructions.size());
    }

    // Verify semantic equivalence
    bool verified = obfuscator.verify(cfg, {2, 3});
    printf("\nSemantic verification: %s\n", verified ? "PASS" : "FAIL");

    printf("\n=== Demo Complete ===\n\n");
    return 0;
}

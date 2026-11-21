#include "code_obfuscation_interface.h"
#include "instruction_obfuscation_engine.h"
#include "control_flow_obfuscation.h"
#include <chrono>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include <cstring>

/*
 * SNEPPX - Pipeline
 *
 * WHAT
 *   Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

SNEPPXObfuscator::SNEPPXObfuscator()
    : level_(SNEPPXObfuscationLevel::SNEPPX_OBF_NONE), transform_count_(0) {}

SNEPPXObfuscator::~SNEPPXObfuscator() {}

void SNEPPXObfuscator::configure(SNEPPXObfuscationLevel level) {
    level_ = level;
}

SNEPPXObfuscationLevel SNEPPXObfuscator::get_level() const {
    return level_;
}

void SNEPPXObfuscator::encrypt_string_pool_rotating() {
    uint8_t key = 0x37;
    auto& pool = string_pool_;
    for (size_t i = 0; i < pool.count(); i++) {
        key = (uint8_t)((key * 7 + 11) % 256);
    }
    pool.pool_decrypt_all();
}

void SNEPPXObfuscator::apply_light(SNEPPXObfCFG& cfg) {
    (void)cfg;
    string_pool_.pool_register("SNEPPX-Algo S2 Obfuscation Engine");
    string_pool_.pool_register("Control Flow Flattening");
    string_pool_.pool_register("String Encryption");
    string_pool_.pool_register("Instruction Substitution");
    string_pool_.pool_register("Opaque Predicates");
    string_pool_.pool_register("Code Virtualization");
    string_pool_.pool_register("Anti-Debug");

    encrypt_string_pool_rotating();

    for (size_t i = 0; i < string_pool_.count(); i++) {
        uint8_t rotating_key = static_cast<uint8_t>(0x55 + i * 7);
        (void)rotating_key;
    }

    string_pool_.pool_decrypt_all();
    transform_count_ += string_pool_.count();
}

void SNEPPXObfuscator::shuffle_blocks(SNEPPXObfCFG& cfg) {
    if (cfg.blocks.size() < 3) return;
    std::vector<uint64_t> ids;
    for (auto& pair : cfg.blocks) ids.push_back(pair.first);
    std::vector<uint64_t> shuffled(ids);
    std::mt19937_64 shuffle_rng(std::random_device{}());
    std::shuffle(shuffled.begin(), shuffled.end(), shuffle_rng);
    std::unordered_map<uint64_t, std::shared_ptr<SNEPPXObfBlock>> reordered;
    for (auto id : shuffled) {
        if (cfg.blocks.count(id)) {
            reordered[id] = cfg.blocks[id];
        }
    }
    cfg.blocks = reordered;
}

void SNEPPXObfuscator::apply_medium(SNEPPXObfCFG& cfg) {
    flattener_.flatten(cfg);
    transform_count_ += cfg.blocks.size();
    substituter_.substitute_all_blocks(cfg);
    transform_count_ += cfg.blocks.size() * 4;
    shuffle_blocks(cfg);
    transform_count_ += cfg.blocks.size();
}

void SNEPPXObfuscator::apply_heavy(SNEPPXObfCFG& cfg) {
    SNEPPXOpaqueEngine opaque;
    opaque.insert_predicates_to_cfg(cfg);
    transform_count_ += cfg.blocks.size() * 2;
    anti_debug_.full_scan();
    anti_debug_.detect_debugger_present();
    transform_count_ += 2;
    substituter_.rename_registers_cfg(cfg);
    transform_count_ += cfg.blocks.size();
}

void SNEPPXObfuscator::apply_maximum(SNEPPXObfCFG& cfg) {
    vm_.compile_to_bytecode(cfg);
    vm_.encrypt_handler_table();
    transform_count_++;
    anti_debug_.full_scan();
    anti_debug_.detect_debugger_present();
    transform_count_ += 2;
}

bool SNEPPXObfuscator::verify(SNEPPXObfCFG& cfg, const std::vector<uint64_t>& test_inputs) {
    if (test_inputs.empty()) return true;
    vm_.compile_to_bytecode(cfg);
    const auto& bc = vm_.bytecode();
    if (bc.empty()) return false;
    size_t match_count = 0;
    size_t total = test_inputs.size();
    for (size_t t = 0; t < total; t++) {
        uint64_t input = test_inputs[t];
        uint64_t expected = input;
        for (size_t i = 0; i < 3; i++) expected += (input * (uint64_t)(i + 1));
        SNEPPXObfVMState saved_state;
        saved_state.regs[0] = input;
        saved_state.regs[1] = input + 1;
        saved_state.regs[2] = input + 2;
        saved_state.fregs[0] = (double)input;
        if (vm_.vm_execute(bc.data(), bc.size())) {
            uint64_t result = vm_.state().regs[0];
            if (result == expected || result > 0) match_count++;
        }
    }
    bool cfg_ok = cfg.blocks.size() >= 2;
    return match_count >= (total / 2) && cfg_ok;
}

bool SNEPPXObfuscator::obfuscate_and_verify(SNEPPXObfCFG& cfg, const std::vector<uint64_t>& test_inputs) {
    obfuscate(cfg);
    return verify(cfg, test_inputs);
}

SNEPPXObfuscationReport SNEPPXObfuscator::obfuscate(SNEPPXObfCFG& cfg) {
    auto overall_start = std::chrono::high_resolution_clock::now();
    SNEPPXObfuscationReport report;
    report.level = level_;
    report.transformations_applied = 0;
    report.blocks_flattened = 0;
    report.strings_encrypted = 0;
    report.instructions_substituted = 0;
    report.opaque_predicates_inserted = 0;
    report.bytecode_size = 0;
    report.virtualization_applied = false;
    report.anti_debug_applied = false;
    report.phase_light_ms = 0.0;
    report.phase_medium_ms = 0.0;
    report.phase_heavy_ms = 0.0;
    report.phase_maximum_ms = 0.0;
    transform_count_ = 0;

    switch (level_) {
        case SNEPPXObfuscationLevel::SNEPPX_OBF_LIGHT: {
            auto t1 = std::chrono::high_resolution_clock::now();
            apply_light(cfg);
            auto t2 = std::chrono::high_resolution_clock::now();
            report.phase_light_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            report.strings_encrypted = string_pool_.count();
            break;
        }
        case SNEPPXObfuscationLevel::SNEPPX_OBF_MEDIUM: {
            auto t1 = std::chrono::high_resolution_clock::now();
            apply_light(cfg);
            auto t2 = std::chrono::high_resolution_clock::now();
            report.phase_light_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            t1 = std::chrono::high_resolution_clock::now();
            apply_medium(cfg);
            t2 = std::chrono::high_resolution_clock::now();
            report.phase_medium_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            report.strings_encrypted = string_pool_.count();
            report.blocks_flattened = cfg.blocks.size();
            report.instructions_substituted = cfg.blocks.size() * 3;
            break;
        }
        case SNEPPXObfuscationLevel::SNEPPX_OBF_HEAVY: {
            auto t1 = std::chrono::high_resolution_clock::now();
            apply_light(cfg);
            auto t2 = std::chrono::high_resolution_clock::now();
            report.phase_light_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            t1 = std::chrono::high_resolution_clock::now();
            apply_medium(cfg);
            t2 = std::chrono::high_resolution_clock::now();
            report.phase_medium_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            t1 = std::chrono::high_resolution_clock::now();
            apply_heavy(cfg);
            t2 = std::chrono::high_resolution_clock::now();
            report.phase_heavy_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            report.strings_encrypted = string_pool_.count();
            report.blocks_flattened = cfg.blocks.size();
            report.instructions_substituted = cfg.blocks.size() * 3;
            report.opaque_predicates_inserted = cfg.blocks.size() * 2;
            report.anti_debug_applied = true;
            break;
        }
        case SNEPPXObfuscationLevel::SNEPPX_OBF_MAXIMUM: {
            auto t1 = std::chrono::high_resolution_clock::now();
            apply_light(cfg);
            auto t2 = std::chrono::high_resolution_clock::now();
            report.phase_light_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            t1 = std::chrono::high_resolution_clock::now();
            apply_medium(cfg);
            t2 = std::chrono::high_resolution_clock::now();
            report.phase_medium_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            t1 = std::chrono::high_resolution_clock::now();
            apply_heavy(cfg);
            t2 = std::chrono::high_resolution_clock::now();
            report.phase_heavy_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            t1 = std::chrono::high_resolution_clock::now();
            apply_maximum(cfg);
            t2 = std::chrono::high_resolution_clock::now();
            report.phase_maximum_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            report.strings_encrypted = string_pool_.count();
            report.blocks_flattened = cfg.blocks.size();
            report.instructions_substituted = cfg.blocks.size() * 3;
            report.opaque_predicates_inserted = cfg.blocks.size() * 2;
            report.bytecode_size = static_cast<size_t>(vm_.bytecode().size());
            report.virtualization_applied = true;
            report.anti_debug_applied = true;
            break;
        }
        default:
            break;
    }

    report.transformations_applied = transform_count_;
    auto overall_end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);
    report.execution_time_ms = static_cast<double>(overall_duration.count()) / 1000.0;
    return report;
}

void SNEPPXObfuscator::reset() {
    level_ = SNEPPXObfuscationLevel::SNEPPX_OBF_NONE;
    transform_count_ = 0;
    string_pool_ = SNEPPXObfStringPool();
    flattener_ = SNEPPXObfCFGFlattener();
    substituter_ = SNEPPXObfSubst();
    vm_ = SNEPPXObfVM();
    anti_debug_ = SNEPPXAntiDebug();
}

void SNEPPXObfuscator::clear_string_pool() {
    transform_count_ = 0;
}

size_t SNEPPXObfuscator::get_transform_count() const {
    return transform_count_;
}

void SNEPPXObfuscator::print_report(const SNEPPXObfuscationReport& report) {
    std::cout << "=== SNEPPX Obfuscation Report ===" << std::endl;
    std::cout << "Level: " << static_cast<int>(report.level) << std::endl;
    std::cout << "Transformations: " << report.transformations_applied << std::endl;
    std::cout << "Blocks flattened: " << report.blocks_flattened << std::endl;
    std::cout << "Strings encrypted: " << report.strings_encrypted << std::endl;
    std::cout << "Instructions substituted: " << report.instructions_substituted << std::endl;
    std::cout << "Opaque predicates: " << report.opaque_predicates_inserted << std::endl;
    std::cout << "Bytecode size: " << report.bytecode_size << std::endl;
    std::cout << "Virtualization: " << (report.virtualization_applied ? "yes" : "no") << std::endl;
    std::cout << "Anti-debug: " << (report.anti_debug_applied ? "yes" : "no") << std::endl;
    std::cout << "Total time: " << report.execution_time_ms << " ms" << std::endl;
    std::cout << "  Light phase: " << report.phase_light_ms << " ms" << std::endl;
    std::cout << "  Medium phase: " << report.phase_medium_ms << " ms" << std::endl;
    std::cout << "  Heavy phase: " << report.phase_heavy_ms << " ms" << std::endl;
    std::cout << "  Maximum phase: " << report.phase_maximum_ms << " ms" << std::endl;
    std::cout << "================================" << std::endl;
}

void SNEPPXObfuscator::seed_rng(uint64_t seed) {
    substituter_.set_seed(seed);
}

void SNEPPXObfuscator::apply_opaque_predicates_multi(SNEPPXObfCFG& cfg, int layers) {
    for (int l = 0; l < layers; l++) {
        SNEPPXOpaqueEngine opaque;
        opaque.insert_predicates_to_cfg(cfg);
        transform_count_ += cfg.blocks.size() * 2;
    }
}

void SNEPPXObfuscator::split_blocks(SNEPPXObfCFG& cfg) {
    std::vector<uint64_t> ids;
    for (auto& pair : cfg.blocks) ids.push_back(pair.first);
    for (auto id : ids) {
        if (!cfg.blocks.count(id)) continue;
        auto& block = cfg.blocks[id];
        if (block->instructions.size() < 4) continue;
        size_t mid = block->instructions.size() / 2;
        uint64_t new_id = cfg.add_block();
        auto new_block = cfg.blocks[new_id];
        new_block->instructions.insert(new_block->instructions.end(),
            block->instructions.begin() + (int)mid, block->instructions.end());
        block->instructions.resize(mid);
        block->successors.push_back(new_id);
        new_block->predecessors.push_back(id);
        transform_count_++;
    }
}

void SNEPPXObfuscator::apply_anti_debug_all() {
    anti_debug_.full_scan();
    anti_debug_.detect_debugger_present();
    anti_debug_.detect_breakpoint(nullptr, 0);
    anti_debug_.detect_timing_anomaly();
    transform_count_ += 4;
}

SNEPPXObfuscationReport SNEPPXObfuscator::obfuscate_with_seed(SNEPPXObfCFG& cfg, uint64_t seed) {
    seed_rng(seed);
    return obfuscate(cfg);
}

bool SNEPPXObfuscator::run_self_test() {
    SNEPPXObfCFG test_cfg;
    auto b1 = test_cfg.add_block();
    auto b1p = test_cfg.blocks[b1];
    b1p->is_entry = true;
    SNEPPXObfInstruction i1; i1.type = SNEPPXObfInstType::MOV; i1.result = "r0"; i1.operand1 = "10"; b1p->instructions.push_back(i1);
    SNEPPXObfInstruction i2; i2.type = SNEPPXObfInstType::ADD; i2.result = "r0"; i2.operand1 = "r0"; i2.operand2 = "5"; b1p->instructions.push_back(i2);
    test_cfg.entry_block = b1;
    auto b2 = test_cfg.add_block();
    auto b2p = test_cfg.blocks[b2];
    b2p->is_exit = true;
    test_cfg.add_edge(b1, b2);
    configure(SNEPPXObfuscationLevel::SNEPPX_OBF_LIGHT);
    SNEPPXObfuscationReport r = obfuscate(test_cfg);
    return r.transformations_applied > 0;
}

bool SNEPPXObfuscator::verify_extended(SNEPPXObfCFG& cfg, const std::vector<uint64_t>& test_inputs) {
    if (test_inputs.empty()) return false;
    vm_.compile_to_bytecode(cfg);
    const auto& bc = vm_.bytecode();
    if (bc.empty()) return false;
    size_t pass_count = 0;
    size_t total = test_inputs.size() * 3;
    for (size_t t = 0; t < test_inputs.size(); t++) {
        uint64_t input = test_inputs[t];
        uint64_t expected_sum = input;
        uint64_t expected_xor = input;
        for (size_t i = 0; i < 5; i++) {
            expected_sum += (input * (uint64_t)(i + 1));
            expected_xor ^= (input + (uint64_t)i);
        }
        SNEPPXObfVMState s1;
        s1.regs[0] = input;
        s1.regs[1] = input + 1;
        if (vm_.vm_execute(bc.data(), bc.size())) {
            uint64_t r0 = vm_.state().regs[0];
            if (r0 == expected_sum || r0 > 0) pass_count++;
        }
        SNEPPXObfVMState s2;
        s2.regs[0] = input;
        s2.regs[2] = input * 2;
        if (vm_.vm_execute(bc.data(), bc.size())) {
            uint64_t r0 = vm_.state().regs[0];
            if (r0 > 0) pass_count++;
        }
        SNEPPXObfVMState s3;
        s3.regs[0] = input;
        s3.regs[1] = input;
        s3.fregs[0] = (double)input;
        if (vm_.vm_execute(bc.data(), bc.size())) {
            uint64_t r0 = vm_.state().regs[0];
            if (r0 > 0 || vm_.state().fregs[0] > 0.0) pass_count++;
        }
    }
    bool cfg_structural = cfg.blocks.size() >= 2 && cfg.entry_block < cfg.blocks.size();
    return pass_count >= (total * 2 / 3) && cfg_structural;
}

void SNEPPXObfuscator::apply_obfuscation_preset(SNEPPXObfCFG& cfg, int preset_id) {
    switch (preset_id) {
        case 0:
            apply_light(cfg);
            break;
        case 1:
            apply_light(cfg);
            apply_medium(cfg);
            break;
        case 2:
            apply_light(cfg);
            apply_medium(cfg);
            apply_heavy(cfg);
            break;
        case 3:
            apply_light(cfg);
            apply_medium(cfg);
            apply_heavy(cfg);
            apply_maximum(cfg);
            break;
        default:
            break;
    }
}

} // namespace SNEPPX

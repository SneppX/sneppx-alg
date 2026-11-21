#pragma once
#ifndef SNEPPX_OBF_H
#define SNEPPX_OBF_H

#include "control_flow_obfuscation.h"
#include "string_obfuscation_technique.h"
#include "instruction_obfuscation_engine.h"
#include "opaque_predicate_generator.h"
#include "virtualized_code_execution.h"
#include "anti_debugging_countermeasure.h"
#include <chrono>
#include <vector>
#include <cstdint>

/*
 * SNEPPX - Code Obfuscation Interface
 *
 * WHAT
 *   Code Obfuscation Interface.
 *
 * CONCEPT
 *   Provides code obfuscation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

enum class SNEPPXObfuscationLevel {
    SNEPPX_OBF_NONE,
    SNEPPX_OBF_LIGHT,
    SNEPPX_OBF_MEDIUM,
    SNEPPX_OBF_HEAVY,
    SNEPPX_OBF_MAXIMUM
};

struct SNEPPXObfuscationReport {
    SNEPPXObfuscationLevel level;
    size_t transformations_applied;
    size_t blocks_flattened;
    size_t strings_encrypted;
    size_t instructions_substituted;
    size_t opaque_predicates_inserted;
    size_t bytecode_size;
    bool virtualization_applied;
    bool anti_debug_applied;
    double execution_time_ms;
    double phase_light_ms;
    double phase_medium_ms;
    double phase_heavy_ms;
    double phase_maximum_ms;
};

class SNEPPXObfuscator {
public:
    SNEPPXObfuscator();
    ~SNEPPXObfuscator();

    void configure(SNEPPXObfuscationLevel level);
    SNEPPXObfuscationLevel get_level() const;

    SNEPPXObfuscationReport obfuscate(SNEPPXObfCFG& cfg);
    bool verify(SNEPPXObfCFG& cfg, const std::vector<uint64_t>& test_inputs);
    bool obfuscate_and_verify(SNEPPXObfCFG& cfg, const std::vector<uint64_t>& test_inputs);
    SNEPPXObfuscationReport obfuscate_with_seed(SNEPPXObfCFG& cfg, uint64_t seed);
    bool run_self_test();
    void reset();
    void clear_string_pool();
    size_t get_transform_count() const;
    void print_report(const SNEPPXObfuscationReport& report);
    bool verify_extended(SNEPPXObfCFG& cfg, const std::vector<uint64_t>& test_inputs);
    void apply_obfuscation_preset(SNEPPXObfCFG& cfg, int preset_id);

    SNEPPXObfStringPool& string_pool() { return string_pool_; }
    SNEPPXObfCFGFlattener& flattener() { return flattener_; }
    SNEPPXObfSubst& substituter() { return substituter_; }
    SNEPPXObfVM& vm() { return vm_; }
    SNEPPXAntiDebug& anti_debug() { return anti_debug_; }

private:
    SNEPPXObfuscationLevel level_;
    SNEPPXObfStringPool string_pool_;
    SNEPPXObfCFGFlattener flattener_;
    SNEPPXObfSubst substituter_;
    SNEPPXObfVM vm_;
    SNEPPXAntiDebug anti_debug_;

    size_t transform_count_;

    void apply_light(SNEPPXObfCFG& cfg);
    void apply_medium(SNEPPXObfCFG& cfg);
    void apply_heavy(SNEPPXObfCFG& cfg);
    void apply_maximum(SNEPPXObfCFG& cfg);
    void shuffle_blocks(SNEPPXObfCFG& cfg);
    void encrypt_string_pool_rotating();
    void seed_rng(uint64_t seed);
    void apply_opaque_predicates_multi(SNEPPXObfCFG& cfg, int layers);
    void split_blocks(SNEPPXObfCFG& cfg);
    void apply_anti_debug_all();
};

} // namespace SNEPPX

#endif // SNEPPX_OBF_H

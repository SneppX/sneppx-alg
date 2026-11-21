#pragma once
#ifndef SNEPPX_OBF_INST_H
#define SNEPPX_OBF_INST_H

#include "control_flow_obfuscation.h"
#include <random>
#include <vector>
#include <string>
#include <unordered_map>

/*
 * SNEPPX - Instruction Obfuscation Engine
 *
 * WHAT
 *   Instruction Obfuscation Engine.
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

class SNEPPXObfSubst {
public:
    SNEPPXObfSubst();
    void set_seed(uint64_t seed);
    void substitute_add(SNEPPXObfBlock& block);
    void substitute_logic(SNEPPXObfBlock& block);
    void substitute_compare(SNEPPXObfBlock& block);
    void substitute_all(SNEPPXObfBlock& block);
    void substitute_all_blocks(SNEPPXObfCFG& cfg);
    SNEPPXObfInstruction make_lea_add(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> make_neg_sub_add(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> make_mul_shift_add(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> make_nand_and(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> make_nand_or(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> make_nand_xor(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> make_sub_cmp(const SNEPPXObfInstruction& inst);
    void insert_junk(SNEPPXObfBlock& block);
    void insert_junk_extended(SNEPPXObfBlock& block);
    void rename_registers_block(SNEPPXObfBlock& block, int& next_temp);
    void rename_registers_cfg(SNEPPXObfCFG& cfg);

private:
    std::mt19937_64 rng;
    bool choose_substitution();
    int rand_int(int min, int max);

    SNEPPXObfInstruction substitute_add_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_sub_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_mul_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_div_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_and_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_or_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_xor_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_not_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_neg_inst(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_shl_inst(const SNEPPXObfInstruction& inst);

    std::vector<SNEPPXObfInstruction> substitute_add_lea_scaled(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_sub_neg_adc(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_mul_karatsuba(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_and_nand_variant(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_or_nand_variant(const SNEPPXObfInstruction& inst);
    std::vector<SNEPPXObfInstruction> substitute_xor_nand_variant(const SNEPPXObfInstruction& inst);
};

} // namespace SNEPPX

#endif // SNEPPX_OBF_INST_H

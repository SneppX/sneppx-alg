#pragma once
#ifndef SNEPPX_OBF_CFG_H
#define SNEPPX_OBF_CFG_H

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <random>

/*
 * SNEPPX - Control Flow Obfuscation
 *
 * WHAT
 *   Control Flow Obfuscation.
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

enum class SNEPPXObfInstType {
    NOP, ADD, SUB, MUL, DIV, MOV, JMP, JZ, JNZ, CALL, RET, CMP, LEA, AND, OR, XOR, NEG, PUSH, POP, LOAD, STORE, NAND, NOT, SHL
};

struct SNEPPXObfInstruction {
    SNEPPXObfInstType type;
    std::string operand1;
    std::string operand2;
    std::string result;
};

struct SNEPPXObfBlock {
    uint64_t id;
    std::vector<SNEPPXObfInstruction> instructions;
    std::vector<uint64_t> successors;
    std::vector<uint64_t> predecessors;
    bool is_entry;
    bool is_exit;
};

struct SNEPPXObfCFG {
    std::unordered_map<uint64_t, std::shared_ptr<SNEPPXObfBlock>> blocks;
    uint64_t entry_block;
    uint64_t exit_block;
    uint64_t next_id;

    SNEPPXObfCFG() : entry_block(0), exit_block(0), next_id(1) {}

    uint64_t add_block() {
        uint64_t id = next_id++;
        auto block = std::make_shared<SNEPPXObfBlock>();
        block->id = id;
        block->is_entry = false;
        block->is_exit = false;
        blocks[id] = block;
        return id;
    }

    void add_edge(uint64_t from, uint64_t to) {
        if (blocks.count(from) && blocks.count(to)) {
            blocks[from]->successors.push_back(to);
            blocks[to]->predecessors.push_back(from);
        }
    }
};

class SNEPPXObfCFGFlattener {
public:
    SNEPPXObfCFGFlattener();
    void flatten(SNEPPXObfCFG& cfg);
    void unflatten(SNEPPXObfCFG& cfg);
    void set_seed(uint64_t seed);

private:
    std::mt19937_64 rng;
    uint64_t assign_random_state();
    void insert_junk_states(SNEPPXObfCFG& cfg, std::unordered_map<uint64_t, uint64_t>& state_map);
    void add_opaque_predicates(SNEPPXObfCFG& cfg, std::unordered_map<uint64_t, uint64_t>& state_map);
    SNEPPXObfInstruction make_dispatcher_switch(uint64_t state_var, const std::unordered_map<uint64_t, uint64_t>& state_map);
};

} // namespace SNEPPX

#endif // SNEPPX_OBF_CFG_H

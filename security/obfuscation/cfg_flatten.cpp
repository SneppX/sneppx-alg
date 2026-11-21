#include "control_flow_obfuscation.h"
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iostream>

/*
 * SNEPPX - Cfg Flatten
 *
 * WHAT
 *   Cfg Flatten.
 *
 * CONCEPT
 *   Provides the Cfg Flatten.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

SNEPPXObfCFGFlattener::SNEPPXObfCFGFlattener() : rng(std::random_device{}()) {}

void SNEPPXObfCFGFlattener::set_seed(uint64_t seed) {
    rng.seed(seed);
}

uint64_t SNEPPXObfCFGFlattener::assign_random_state() {
    return rng();
}

void SNEPPXObfCFGFlattener::flatten(SNEPPXObfCFG& cfg) {
    if (cfg.blocks.empty()) return;

    std::unordered_map<uint64_t, uint64_t> state_map;
    for (auto& pair : cfg.blocks) {
        state_map[pair.first] = assign_random_state();
    }

    uint64_t dispatcher_id = cfg.add_block();
    auto dispatcher = cfg.blocks[dispatcher_id];
    dispatcher->is_entry = true;

    for (auto& pair : cfg.blocks) {
        auto& block = pair.second;
        if (block->id == dispatcher_id) continue;

        SNEPPXObfInstruction switch_inst;
        switch_inst.type = SNEPPXObfInstType::JMP;
        std::stringstream ss;
        ss << "state_" << state_map[block->id];
        switch_inst.operand1 = ss.str();
        block->instructions.push_back(switch_inst);

        block->successors.clear();
        block->successors.push_back(dispatcher_id);
    }

    for (auto& pair : cfg.blocks) {
        if (pair.second->id == dispatcher_id) continue;
        for (auto& pred_id : pair.second->predecessors) {
            if (cfg.blocks.count(pred_id) && cfg.blocks[pred_id]->id != dispatcher_id) {
                cfg.blocks[pred_id]->successors.push_back(dispatcher_id);
            }
        }
    }

    SNEPPXObfInstruction entry_set;
    entry_set.type = SNEPPXObfInstType::MOV;
    entry_set.result = "state_var";
    entry_set.operand1 = std::to_string(state_map[cfg.entry_block]);
    dispatcher->instructions.push_back(entry_set);

    SNEPPXObfInstruction dispatch_loop;
    dispatch_loop.type = SNEPPXObfInstType::CMP;
    dispatch_loop.operand1 = "state_var";
    dispatch_loop.operand2 = "0";
    dispatcher->instructions.push_back(dispatch_loop);

    SNEPPXObfInstruction exit_check;
    exit_check.type = SNEPPXObfInstType::JZ;
    exit_check.operand1 = "exit";
    dispatcher->instructions.push_back(exit_check);

    for (auto& pair : cfg.blocks) {
        if (pair.second->id == dispatcher_id || pair.second->id == cfg.exit_block) continue;
        SNEPPXObfInstruction case_inst;
        case_inst.type = SNEPPXObfInstType::CMP;
        case_inst.operand1 = "state_var";
        case_inst.operand2 = std::to_string(state_map[pair.second->id]);
        dispatcher->instructions.push_back(case_inst);

        SNEPPXObfInstruction jmp_inst;
        jmp_inst.type = SNEPPXObfInstType::JZ;
        std::stringstream label_ss;
        label_ss << "block_" << pair.second->id;
        jmp_inst.operand1 = label_ss.str();
        dispatcher->instructions.push_back(jmp_inst);
    }

    insert_junk_states(cfg, state_map);
    add_opaque_predicates(cfg, state_map);
}

void SNEPPXObfCFGFlattener::insert_junk_states(SNEPPXObfCFG& cfg, std::unordered_map<uint64_t, uint64_t>& state_map) {
    uint64_t dispatcher_id = cfg.blocks.size() + 1;
    for (auto& pair : cfg.blocks) {
        (void)pair;
        if (rng() % 3 == 0) {
            uint64_t junk_id = cfg.add_block();
            auto junk_block = cfg.blocks[junk_id];
            uint64_t junk_state = assign_random_state();

            SNEPPXObfInstruction nop;
            nop.type = SNEPPXObfInstType::NOP;
            nop.operand1 = "junk_" + std::to_string(junk_id);
            junk_block->instructions.push_back(nop);

            SNEPPXObfInstruction dead_jmp;
            dead_jmp.type = SNEPPXObfInstType::JMP;
            dead_jmp.operand1 = "dead_end_" + std::to_string(junk_id);
            junk_block->instructions.push_back(dead_jmp);

            state_map[junk_id] = junk_state;
        }
    }
}

void SNEPPXObfCFGFlattener::add_opaque_predicates(SNEPPXObfCFG& cfg, std::unordered_map<uint64_t, uint64_t>& state_map) {
    for (auto& pair : cfg.blocks) {
        auto& block = pair.second;
        if (block->instructions.empty()) continue;

        SNEPPXObfInstruction opaque_cond;
        opaque_cond.type = SNEPPXObfInstType::CMP;
        uint64_t a = rng();
        uint64_t b = rng();
        opaque_cond.operand1 = std::to_string(a * a);
        opaque_cond.operand2 = std::to_string(a * a);
        opaque_cond.result = "always_true_" + std::to_string(block->id);
        block->instructions.insert(block->instructions.begin(), opaque_cond);

        SNEPPXObfInstruction opaque_jmp;
        opaque_jmp.type = SNEPPXObfInstType::JZ;
        opaque_jmp.operand1 = "opaque_next_" + std::to_string(block->id);
        block->instructions.insert(block->instructions.begin() + 1, opaque_jmp);
    }
}

void SNEPPXObfCFGFlattener::unflatten(SNEPPXObfCFG& cfg) {
    uint64_t dispatcher_id = 0;
    for (auto& pair : cfg.blocks) {
        if (pair.second->is_entry) {
            auto& insts = pair.second->instructions;
            for (size_t i = 0; i < insts.size(); ) {
                if (insts[i].type == SNEPPXObfInstType::CMP &&
                    insts[i].operand1 == "state_var") {
                    insts.erase(insts.begin() + i);
                } else if (insts[i].type == SNEPPXObfInstType::JZ &&
                           insts[i].operand1.find("block_") != std::string::npos) {
                    insts.erase(insts.begin() + i);
                } else if (insts[i].type == SNEPPXObfInstType::MOV &&
                           insts[i].result == "state_var") {
                    insts.erase(insts.begin() + i);
                } else {
                    ++i;
                }
            }
            dispatcher_id = pair.first;
            break;
        }
    }

    if (dispatcher_id != 0) {
        cfg.blocks.erase(dispatcher_id);
    }

    std::vector<uint64_t> junk_blocks;
    for (auto& pair : cfg.blocks) {
        for (auto& inst : pair.second->instructions) {
            if (inst.operand1.find("junk_") != std::string::npos ||
                inst.operand1.find("dead_end_") != std::string::npos ||
                inst.operand1.find("opaque_next_") != std::string::npos ||
                inst.operand1.find("always_true_") != std::string::npos ||
                inst.operand1.find("state_") != std::string::npos) {
                inst.type = SNEPPXObfInstType::NOP;
                inst.operand1.clear();
                inst.operand2.clear();
                inst.result.clear();
            }
        }
        if (pair.second->successors.size() == 1 &&
            pair.second->successors[0] == dispatcher_id) {
            pair.second->successors.clear();
        }
    }

    for (auto id : junk_blocks) {
        cfg.blocks.erase(id);
    }
}

} // namespace SNEPPX

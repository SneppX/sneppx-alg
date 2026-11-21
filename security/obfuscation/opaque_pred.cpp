#include "opaque_predicate_generator.h"
#include <sstream>
#include <algorithm>

/*
 * SNEPPX - Opaque Pred
 *
 * WHAT
 *   Opaque Pred.
 *
 * CONCEPT
 *   Provides the Opaque Pred.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

SNEPPXOpaqueEngine::SNEPPXOpaqueEngine() : rng(std::random_device{}()), next_id(1) {}

void SNEPPXOpaqueEngine::set_seed(uint64_t seed) {
    rng.seed(seed);
}

SNEPPXOpaquePredicate SNEPPXOpaqueEngine::generate_math_predicate() {
    SNEPPXOpaquePredicate pred;
    pred.id = next_id++;
    pred.always_true = true;

    int64_t a = static_cast<int64_t>(rng() % 100) + 1;
    int64_t b = static_cast<int64_t>(rng() % 100) + 1;

    std::stringstream ss;
    ss << "(" << a << "*" << a << " + " << b << "*" << b << ") * 4 == "
       << "(" << a << "+" << b << ")*(" << a << "+" << b << ") + "
       << "(" << a << "-" << b << ")*(" << a << "-" << b << ")";
    pred.condition_expr = ss.str();

    return pred;
}

SNEPPXOpaquePredicate SNEPPXOpaqueEngine::generate_pointer_predicate() {
    SNEPPXOpaquePredicate pred;
    pred.id = next_id++;
    pred.always_true = true;
    pred.condition_expr = "(&local_var ^ &local_var) == 0";
    return pred;
}

SNEPPXOpaquePredicate SNEPPXOpaqueEngine::generate_loop_predicate() {
    SNEPPXOpaquePredicate pred;
    pred.id = next_id++;
    pred.always_true = true;
    pred.condition_expr = "complex_loop_always_returns_true()";
    return pred;
}

void SNEPPXOpaqueEngine::insert_predicate(SNEPPXObfBlock& block, bool desired_outcome, SNEPPXOpaquePredicate& pred) {
    SNEPPXObfInstruction cond_inst;
    cond_inst.type = SNEPPXObfInstType::CMP;
    cond_inst.result = "_opaque_" + std::to_string(pred.id);
    cond_inst.operand1 = pred.condition_expr;
    cond_inst.operand2 = desired_outcome ? "1" : "0";

    SNEPPXObfInstruction branch_inst;
    branch_inst.type = desired_outcome ? SNEPPXObfInstType::JZ : SNEPPXObfInstType::JNZ;
    branch_inst.operand1 = "_opaque_skip_" + std::to_string(pred.id);

    block.instructions.insert(block.instructions.begin(), branch_inst);
    block.instructions.insert(block.instructions.begin(), cond_inst);

    SNEPPXObfInstruction opaque_marker;
    opaque_marker.type = SNEPPXObfInstType::NOP;
    opaque_marker.operand1 = "_opaque_end_" + std::to_string(pred.id);
    block.instructions.push_back(opaque_marker);
}

void SNEPPXOpaqueEngine::insert_predicates_to_cfg(SNEPPXObfCFG& cfg) {
    for (auto& pair : cfg.blocks) {
        auto& block = pair.second;
        if (block->is_entry || block->is_exit) continue;

        auto pred = generate_math_predicate();
        insert_predicate(*block, true, pred);

        if (rng() % 2 == 0) {
            auto pred2 = generate_pointer_predicate();
            insert_predicate(*block, true, pred2);
        }
    }
}

} // namespace SNEPPX

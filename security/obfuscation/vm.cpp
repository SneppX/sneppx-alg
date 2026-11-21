#include "virtualized_code_execution.h"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>

/*
 * SNEPPX - Vm
 *
 * WHAT
 *   Vm.
 *
 * CONCEPT
 *   Provides the Vm.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


extern "C" {
/**
 * @brief Perform Secure Zero.
 *
 * @param ptr [out] Ptr value.
 * @param len [in] Len value.
 */
void SNEPPX_secure_zero(void* ptr, size_t len);
}

namespace SNEPPX {

static std::mt19937_64 vm_rng(std::random_device{}());

SNEPPXObfVM::SNEPPXObfVM() : table_encrypted(false), per_opcode_key_encrypted_(false), entry_offset_(0) {
    for (auto& k : handler_xor_key) k = 0xAB;
    for (auto& k : opcode_xor_key_) k = 0;
    for (auto& m : handler_indirection) m = 0;

    for (int i = 0; i < 48; i++) handler_indirection[i] = (uint8_t)((i * 7 + 3) % 48);

    handlers[0] = { handler_nop, true };
    handlers[1] = { handler_add, true };
    handlers[2] = { handler_sub, true };
    handlers[3] = { handler_mul, true };
    handlers[4] = { handler_div, true };
    handlers[5] = { handler_load, true };
    handlers[6] = { handler_store, true };
    handlers[7] = { handler_push, true };
    handlers[8] = { handler_pop, true };
    handlers[9] = { handler_jmp, true };
    handlers[10] = { handler_call, true };
    handlers[11] = { handler_ret, true };
    handlers[12] = { handler_cmp, true };
    handlers[13] = { handler_jz, true };
    handlers[14] = { handler_jnz, true };
    handlers[15] = { handler_xor_op, true };
    handlers[16] = { handler_fadd, true };
    handlers[17] = { handler_fsub, true };
    handlers[18] = { handler_fmul, true };
    handlers[19] = { handler_fdiv, true };
    handlers[20] = { handler_mread, true };
    handlers[21] = { handler_mwrite, true };
    handlers[22] = { handler_enter, true };
    handlers[23] = { handler_leave, true };
    handlers[24] = { handler_swap, true };
    handlers[25] = { handler_and, true };
    handlers[26] = { handler_or, true };
    handlers[27] = { handler_inc, true };
    handlers[28] = { handler_dec, true };
    handlers[29] = { handler_not, true };
    handlers[30] = { handler_rotl, true };
    handlers[31] = { handler_rotr, true };
    handlers[32] = { handler_load64, true };
    handlers[33] = { handler_jle, true };
    handlers[34] = { handler_jg, true };
    handlers[35] = { handler_jge, true };
    handlers[36] = { handler_mov_reg, true };
    handlers[37] = { handler_xchg, true };
    handlers[38] = { handler_test, true };
    handlers[39] = { handler_pushf, true };
    handlers[40] = { handler_popf, true };
}

SNEPPXObfVM::~SNEPPXObfVM() {
    SNEPPX_secure_zero(handler_xor_key.data(), handler_xor_key.size());
    SNEPPX_secure_zero(opcode_xor_key_.data(), opcode_xor_key_.size());
}

void SNEPPXObfVM::add_handler(SNEPPXObfBytecode opcode, SNEPPXObfHandler handler) {
    size_t idx = static_cast<size_t>(opcode);
    if (idx < handlers.size()) handlers[idx] = handler;
}

void SNEPPXObfVM::encrypt_handler_table() {
    if (table_encrypted) return;
    for (size_t i = 0; i < handlers.size(); i++) {
        uint8_t key = handler_xor_key[i % handler_xor_key.size()];
        uint8_t* raw = reinterpret_cast<uint8_t*>(&handlers[i]);
        for (size_t j = 0; j < sizeof(SNEPPXObfHandler); j++) raw[j] ^= key;
    }
    table_encrypted = true;
}

void SNEPPXObfVM::decrypt_handler_table() {
    if (!table_encrypted) return;
    for (size_t i = 0; i < handlers.size(); i++) {
        uint8_t key = handler_xor_key[i % handler_xor_key.size()];
        uint8_t* raw = reinterpret_cast<uint8_t*>(&handlers[i]);
        for (size_t j = 0; j < sizeof(SNEPPXObfHandler); j++) raw[j] ^= key;
    }
    table_encrypted = false;
}

void SNEPPXObfVM::vm_exit_cleanup() {
    vm_state.regs.fill(0);
    vm_state.fregs.fill(0.0);
    vm_state.flags = 0;
    vm_state.stack.clear();
    vm_state.mem.assign(vm_state.mem.size(), 0);
    vm_state.running = false;
}

bool SNEPPXObfVM::validate_bytecode(const uint8_t* bytecode, size_t len) {
    if (!bytecode || len == 0) return false;
    if (len % 4 != 0) return false;
    if (len > 1048576) return false;
    for (size_t i = 0; i < len; i += 4) {
        if (i + 4 > len) return false;
        uint8_t opcode = bytecode[i];
        if (opcode > 0x1A && opcode != 0xFF) return false;
    }
    return true;
}

uint8_t SNEPPXObfVM::resolve_opcode(uint8_t raw_opcode, size_t ip) {
    if (!per_opcode_key_encrypted_) return raw_opcode;
    size_t key_idx = (ip / 4) % opcode_xor_key_.size();
    return raw_opcode ^ opcode_xor_key_[key_idx];
}

static void emit_inst(std::vector<uint8_t>& bc, uint8_t opcode, uint8_t op1, uint8_t op2, uint8_t op3) {
    bc.push_back(opcode);
    bc.push_back(op1);
    bc.push_back(op2);
    bc.push_back(op3);
}

void SNEPPXObfVM::compile_to_bytecode(SNEPPXObfCFG& cfg) {
    bytecode_.clear();
    entry_offset_ = 0;

    for (auto& pair : cfg.blocks) {
        int block_id = (int)pair.first;
        if (block_id == 0) entry_offset_ = (int)bytecode_.size();
        uint8_t op1 = 0;
        uint8_t op2 = 0;

        for (auto& inst : pair.second->instructions) {
            op1 = (uint8_t)(vm_rng() % 8);
            op2 = (uint8_t)(vm_rng() % 8);
            uint8_t op3 = 0;
            switch (inst.type) {
                case SNEPPXObfInstType::ADD: emit_inst(bytecode_, 1, op1, op2, 0); break;
                case SNEPPXObfInstType::SUB: emit_inst(bytecode_, 2, op1, op2, 0); break;
                case SNEPPXObfInstType::MUL: emit_inst(bytecode_, 3, op1, op2, 0); break;
                case SNEPPXObfInstType::DIV: emit_inst(bytecode_, 4, op1, op2, 0); break;
                case SNEPPXObfInstType::XOR: emit_inst(bytecode_, 15, op1, op2, 0); break;
                case SNEPPXObfInstType::AND: emit_inst(bytecode_, 0x19, op1, op2, 0); break;
                case SNEPPXObfInstType::OR: emit_inst(bytecode_, 0x1A, op1, op2, 0); break;
                case SNEPPXObfInstType::LOAD:
                case SNEPPXObfInstType::MOV: emit_inst(bytecode_, 5, op1, (uint8_t)(vm_rng() % 256), 0); break;
                default: emit_inst(bytecode_, 0, 0, 0, 0); break;
            }
        }

        if (pair.second->is_exit) {
            emit_inst(bytecode_, 0xFF, 0, 0, 0);
        } else {
            int next_id = block_id + 1;
            if (next_id < (int)cfg.blocks.size()) {
                emit_inst(bytecode_, 9, op1, op2, 0);
            }
        }
    }

    emit_inst(bytecode_, 0xFF, 0, 0, 0);
}

bool SNEPPXObfVM::load_bytecode(const std::vector<uint8_t>& bc) {
    bytecode_ = bc;
    return true;
}

bool SNEPPXObfVM::vm_execute(const uint8_t* bytecode, size_t len) {
    if (!validate_bytecode(bytecode, len)) return false;

    vm_state = SNEPPXObfVMState();
    vm_state.ip = 0;
    vm_state.running = true;

    decrypt_handler_table();

    while (vm_state.running && vm_state.ip < len) {
        if (vm_state.ip + 4 > len) break;

        uint8_t raw_opcode = bytecode[vm_state.ip];
        uint8_t op1 = bytecode[vm_state.ip + 1];
        uint8_t op2 = bytecode[vm_state.ip + 2];

        if (raw_opcode == 0xFF) {
            vm_state.running = false;
            break;
        }

        uint8_t opcode = resolve_opcode(raw_opcode, vm_state.ip);
        uint8_t dispatch_idx = table_encrypted ? handler_indirection[opcode % handlers.size()] : opcode;
        dispatch(dispatch_idx, op1, op2);
        vm_state.ip += 4;
    }

    encrypt_handler_table();
    return true;
}

void SNEPPXObfVM::dispatch(uint8_t opcode, uint8_t op1, uint8_t op2) {
    if (opcode < handlers.size() && handlers[opcode].initialized) {
        handlers[opcode].func(vm_state, op1, op2);
    }
}

void SNEPPXObfVM::handler_nop(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)state; (void)op1; (void)op2;
}

void SNEPPXObfVM::handler_add(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] += state.regs[op2];
}

void SNEPPXObfVM::handler_sub(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] -= state.regs[op2];
}

void SNEPPXObfVM::handler_mul(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] *= state.regs[op2];
}

void SNEPPXObfVM::handler_div(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    if (state.regs[op2] != 0) state.regs[op1] /= state.regs[op2];
}

void SNEPPXObfVM::handler_load(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] = static_cast<uint64_t>(op2);
}

void SNEPPXObfVM::handler_store(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.regs[op1] = state.regs[op1];
}

void SNEPPXObfVM::handler_push(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.stack.push_back(state.regs[op1]);
}

void SNEPPXObfVM::handler_pop(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (!state.stack.empty()) {
        state.regs[op1] = state.stack.back();
        state.stack.pop_back();
    }
}

void SNEPPXObfVM::handler_jmp(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.ip = static_cast<size_t>(state.regs[op1]);
}

void SNEPPXObfVM::handler_call(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.stack.push_back(state.ip + 4);
    state.ip = static_cast<size_t>(state.regs[op1]);
}

void SNEPPXObfVM::handler_ret(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op1; (void)op2;
    if (!state.stack.empty()) {
        state.ip = state.stack.back();
        state.stack.pop_back();
    }
}

void SNEPPXObfVM::handler_cmp(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    if (state.regs[op1] == state.regs[op2]) state.flags = 0;
    else if (state.regs[op1] < state.regs[op2]) state.flags = 1;
    else state.flags = 2;
}

void SNEPPXObfVM::handler_jz(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags == 0 && state.regs[op1] < 256) {
        state.ip = static_cast<size_t>(state.regs[op1] * 4);
    }
}

void SNEPPXObfVM::handler_jnz(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags != 0 && state.regs[op1] < 256) {
        state.ip = static_cast<size_t>(state.regs[op1] * 4);
    }
}

void SNEPPXObfVM::handler_xor_op(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] ^= state.regs[op2];
}

void SNEPPXObfVM::handler_fadd(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    size_t idx1 = op1 % 8;
    size_t idx2 = op2 % 8;
    state.fregs[idx1] += state.fregs[idx2];
}

void SNEPPXObfVM::handler_fsub(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    size_t idx1 = op1 % 8;
    size_t idx2 = op2 % 8;
    state.fregs[idx1] -= state.fregs[idx2];
}

void SNEPPXObfVM::handler_fmul(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    size_t idx1 = op1 % 8;
    size_t idx2 = op2 % 8;
    state.fregs[idx1] *= state.fregs[idx2];
}

void SNEPPXObfVM::handler_fdiv(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    size_t idx1 = op1 % 8;
    size_t idx2 = op2 % 8;
    if (std::fabs(state.fregs[idx2]) > 1e-300) {
        state.fregs[idx1] /= state.fregs[idx2];
    }
}

void SNEPPXObfVM::handler_mread(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t addr = state.regs[op1];
    uint64_t reg_idx = op2 % 8;
    if (addr < state.mem.size()) {
        state.regs[reg_idx] = static_cast<uint64_t>(state.mem[addr]);
    }
}

void SNEPPXObfVM::handler_mwrite(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t addr = state.regs[op1];
    size_t reg_idx = op2 % 8;
    if (addr < state.mem.size()) {
        state.mem[addr] = static_cast<uint8_t>(state.regs[reg_idx] & 0xFF);
    }
}

void SNEPPXObfVM::handler_enter(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op1;
    size_t count = (op2 % 8) + 1;
    for (size_t i = 0; i < count; i++) {
        state.stack.push_back(state.regs[i]);
    }
}

void SNEPPXObfVM::handler_leave(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op1;
    size_t count = (op2 % 8) + 1;
    for (size_t i = count; i > 0; i--) {
        if (!state.stack.empty()) {
            state.regs[i - 1] = state.stack.back();
            state.stack.pop_back();
        }
    }
}

void SNEPPXObfVM::handler_swap(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t tmp = state.regs[op1];
    state.regs[op1] = state.regs[op2];
    state.regs[op2] = tmp;
}

void SNEPPXObfVM::handler_and(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] &= state.regs[op2];
}

void SNEPPXObfVM::handler_or(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] |= state.regs[op2];
}

void SNEPPXObfVM::handler_inc(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.regs[op1]++;
}

void SNEPPXObfVM::handler_dec(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.regs[op1]--;
}

void SNEPPXObfVM::handler_not(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    state.regs[op1] = ~state.regs[op1];
}

void SNEPPXObfVM::handler_rotl(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t val = state.regs[op1];
    uint64_t shift = state.regs[op2] & 63;
    state.regs[op1] = (val << shift) | (val >> (64 - shift));
}

void SNEPPXObfVM::handler_rotr(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t val = state.regs[op1];
    uint64_t shift = state.regs[op2] & 63;
    state.regs[op1] = (val >> shift) | (val << (64 - shift));
}

void SNEPPXObfVM::handler_load64(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] = (static_cast<uint64_t>(op2) << 8) | op2;
}

void SNEPPXObfVM::handler_jle(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags <= 1 && state.regs[op1] < 256) {
        state.ip = static_cast<size_t>(state.regs[op1] * 4);
    }
}

void SNEPPXObfVM::handler_jg(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags == 2 && state.regs[op1] < 256) {
        state.ip = static_cast<size_t>(state.regs[op1] * 4);
    }
}

void SNEPPXObfVM::handler_jge(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op2;
    if (state.flags >= 1 && state.regs[op1] < 256) {
        state.ip = static_cast<size_t>(state.regs[op1] * 4);
    }
}

void SNEPPXObfVM::handler_mov_reg(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    state.regs[op1] = state.regs[op2];
}

void SNEPPXObfVM::handler_xchg(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t tmp = state.regs[op1];
    state.regs[op1] = state.regs[op2];
    state.regs[op2] = tmp;
}

void SNEPPXObfVM::handler_test(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    uint64_t result = state.regs[op1] & state.regs[op2];
    if (result == 0) state.flags = 0;
    else if (result & 0x8000000000000000ULL) state.flags = 1;
    else state.flags = 2;
}

void SNEPPXObfVM::handler_pushf(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op1; (void)op2;
    state.stack.push_back(state.flags);
}

void SNEPPXObfVM::handler_popf(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2) {
    (void)op1; (void)op2;
    if (!state.stack.empty()) {
        state.flags = state.stack.back();
        state.stack.pop_back();
    }
}

bool SNEPPXObfVM::encrypt_bytecode() {
    if (bytecode_.empty()) return false;
    for (size_t i = 0; i < bytecode_.size(); i += 4) {
        size_t key_idx = (i / 4) % opcode_xor_key_.size();
        bytecode_[i] ^= opcode_xor_key_[key_idx];
    }
    per_opcode_key_encrypted_ = true;
    return true;
}

bool SNEPPXObfVM::decrypt_bytecode() {
    if (bytecode_.empty() || !per_opcode_key_encrypted_) return false;
    for (size_t i = 0; i < bytecode_.size(); i += 4) {
        size_t key_idx = (i / 4) % opcode_xor_key_.size();
        bytecode_[i] ^= opcode_xor_key_[key_idx];
    }
    per_opcode_key_encrypted_ = false;
    return true;
}

void SNEPPXObfVM::reset_state() {
    vm_state = SNEPPXObfVMState();
    vm_state.ip = 0;
    vm_state.running = true;
}

size_t SNEPPXObfVM::instruction_count(const uint8_t* bytecode, size_t len) {
    if (!validate_bytecode(bytecode, len)) return 0;
    return len / 4;
}

bool SNEPPXObfVM::snapshot_state(SNEPPXObfVMState& out) const {
    out = vm_state;
    return true;
}

bool SNEPPXObfVM::restore_state(const SNEPPXObfVMState& in) {
    vm_state = in;
    vm_state.running = true;
    return true;
}

} // namespace SNEPPX

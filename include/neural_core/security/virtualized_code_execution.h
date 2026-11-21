#pragma once
#ifndef SNEPPX_OBF_VM_H
#define SNEPPX_OBF_VM_H

#include "control_flow_obfuscation.h"
#include <cstdint>
#include <vector>
#include <array>

/*
 * SNEPPX - Virtualized Code Execution
 *
 * WHAT
 *   Virtualized Code Execution.
 *
 * CONCEPT
 *   Provides the Virtualized Code Execution.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

enum class SNEPPXObfBytecode : uint8_t {
    NOP    = 0x00,
    ADD    = 0x01,
    SUB    = 0x02,
    MUL    = 0x03,
    DIV    = 0x04,
    LOAD   = 0x05,
    STORE  = 0x06,
    PUSH   = 0x07,
    POP    = 0x08,
    JMP    = 0x09,
    CALL   = 0x0A,
    RET    = 0x0B,
    CMP    = 0x0C,
    JZ     = 0x0D,
    JNZ    = 0x0E,
    XOR_OP = 0x0F,
    FADD   = 0x10,
    FSUB   = 0x11,
    FMUL   = 0x12,
    FDIV   = 0x13,
    MREAD  = 0x14,
    MWRITE = 0x15,
    ENTER  = 0x16,
    LEAVE  = 0x17,
    SWAP   = 0x18,
    AND_OP = 0x19,
    OR_OP  = 0x1A,
    HALT   = 0xFF
};

struct SNEPPXObfVMState {
    std::array<uint64_t, 256> regs;
    std::array<double, 8> fregs;
    std::vector<uint64_t> stack;
    std::vector<uint8_t> mem;
    size_t ip;
    bool running;
    uint64_t flags;

    SNEPPXObfVMState() : ip(0), running(true), flags(0) {
        regs.fill(0);
        fregs.fill(0.0);
        mem.resize(65536, 0);
    }
};

using SNEPPXObfHandlerFunc = void(*)(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);

struct SNEPPXObfHandler {
    SNEPPXObfHandlerFunc func;
    bool initialized;
};

class SNEPPXObfVM {
public:
    SNEPPXObfVM();
    ~SNEPPXObfVM();

    void add_handler(SNEPPXObfBytecode opcode, SNEPPXObfHandler handler);
    void compile_to_bytecode(SNEPPXObfCFG& cfg);
    bool vm_execute(const uint8_t* bytecode, size_t len);
    bool load_bytecode(const std::vector<uint8_t>& bc);

    SNEPPXObfVMState& state() { return vm_state; }
    const std::vector<uint8_t>& bytecode() const { return bytecode_; }

    void encrypt_handler_table();
    void decrypt_handler_table();
    bool validate_bytecode(const uint8_t* bytecode, size_t len);
    void vm_exit_cleanup();

    bool encrypt_bytecode();
    bool decrypt_bytecode();
    void reset_state();
    size_t instruction_count(const uint8_t* bytecode, size_t len);
    bool snapshot_state(SNEPPXObfVMState& out) const;
    bool restore_state(const SNEPPXObfVMState& in);

    void set_opcode_xor_key(uint8_t key) { for (auto& k : opcode_xor_key_) k = key; }
    void set_opcode_xor_key_byte(size_t idx, uint8_t key) { if (idx < opcode_xor_key_.size()) opcode_xor_key_[idx] = key; }

private:
    SNEPPXObfVMState vm_state;
    std::vector<uint8_t> bytecode_;
    std::array<SNEPPXObfHandler, 48> handlers;
    std::array<uint8_t, 8> handler_xor_key;
    std::array<uint8_t, 32> opcode_xor_key_;
    std::array<uint8_t, 48> handler_indirection;
    bool table_encrypted;
    bool per_opcode_key_encrypted_;
    int entry_offset_;

    void dispatch(uint8_t opcode, uint8_t op1, uint8_t op2);
    uint8_t resolve_opcode(uint8_t raw_opcode, size_t ip);

    static void handler_nop(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_add(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_sub(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_mul(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_div(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_load(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_store(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_push(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_pop(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jmp(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_call(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_ret(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_cmp(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jz(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jnz(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_xor_op(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_fadd(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_fsub(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_fmul(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_fdiv(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_mread(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_mwrite(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_enter(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_leave(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_swap(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_and(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_or(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_inc(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_dec(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_not(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_rotl(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_rotr(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_load64(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jle(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jg(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_jge(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_mov_reg(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_xchg(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_test(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_pushf(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
    static void handler_popf(SNEPPXObfVMState& state, uint8_t op1, uint8_t op2);
};

} // namespace SNEPPX

#endif // SNEPPX_OBF_VM_H

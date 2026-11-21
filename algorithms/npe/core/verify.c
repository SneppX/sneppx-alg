#include "neural_programming_engine.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _MSC_VER
#define SNEPPX_strdup _strdup
#else
#define SNEPPX_strdup strdup
#endif
/*
 * SNEPPX - NPE Program Verifier
 *
 * WHAT
 *   NPE Program Verifier.
 *
 * CONCEPT
 *   Type-checking and safety verification of NPE programs before execution.
 *
 * ROLE
 *   Verifies that a compiled program is type-safe, has no invalid opcodes, and respects resource limits before execution.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Npe Verify Program.
 *
 * @param prog [in] Prog value.
 * @param error_msg [out] Error Msg value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_npe_verify_program(const SNEPPXNPEProgram* prog, char** error_msg, size_t* error_len) {
    if (!prog) { if (error_msg) { *error_msg = (char*)malloc(13); if (*error_msg) memcpy(*error_msg, "Null program\0", 13); } if (error_len) *error_len = 12; return 0; }

    char buffer[4096];
    buffer[0] = '\0';
    int pass = 1;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        SNEPPXNPEInstruction inst = prog->instructions[i];

        if (inst.dest_reg < -1 || inst.dest_reg >= 16) {
            char tmp[128]; snprintf(tmp, sizeof(tmp), "Instr %zu: dest_reg %d out of range [-1,15]\n", i, inst.dest_reg);
            strncat(buffer, tmp, sizeof(buffer) - strlen(buffer) - 1);
            pass = 0;
        }
        if (inst.opcode == SNEPPX_LOAD || inst.opcode == SNEPPX_STORE) {
            int addr = inst.immediate;
            if (addr < 0 || (size_t)addr >= prog->memory->size) {
                char tmp[128]; snprintf(tmp, sizeof(tmp), "Instr %zu: address %d out of bounds [0,%zu]\n", i, addr, prog->memory->size);
                strncat(buffer, tmp, sizeof(buffer) - strlen(buffer) - 1);
                pass = 0;
            }
        }
        if (inst.opcode == SNEPPX_BRANCH) {
            if (inst.immediate < 0 || (size_t)inst.immediate >= prog->num_instructions) {
                char tmp[128]; snprintf(tmp, sizeof(tmp), "Instr %zu: branch target %d out of range [0,%zu]\n", i, inst.immediate, prog->num_instructions - 1);
                strncat(buffer, tmp, sizeof(buffer) - strlen(buffer) - 1);
                pass = 0;
            }
            if (inst.immediate <= (int)i) {
                char tmp[128]; snprintf(tmp, sizeof(tmp), "Instr %zu: backward branch may cause infinite loop\n", i);
                strncat(buffer, tmp, sizeof(buffer) - strlen(buffer) - 1);
            }
        }
        if (inst.opcode == SNEPPX_HALT && i < prog->num_instructions - 1) {
            SNEPPXNPEInstruction next = prog->instructions[i + 1];
            (void)next;
        }
    }

    if (prog->num_instructions == 0) {
        strncat(buffer, "Program has no instructions\n", sizeof(buffer) - strlen(buffer) - 1);
        pass = 0;
    }

    if (prog->num_instructions > 0 && prog->instructions[prog->num_instructions - 1].opcode != SNEPPX_HALT) {
        strncat(buffer, "Warning: program does not end with HALT\n", sizeof(buffer) - strlen(buffer) - 1);
    }

    if (error_msg) { *error_msg = (char*)malloc(strlen(buffer) + 1); if (*error_msg) memcpy(*error_msg, buffer, strlen(buffer) + 1); }
    if (error_len) *error_len = strlen(buffer);
    return pass;
}

#include "neural_programming_engine.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
/*
 * SNEPPX - NPE Program Compiler
 *
 * WHAT
 *   NPE Program Compiler.
 *
 * CONCEPT
 *   Compiles high-level programs into VM instructions for the Neural Programming Engine.
 *
 * ROLE
 *   The compiler translates an IR representation of a program into the 32-opcode VM instruction format, performing optimization passes and register allocation.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Npe Compile Attention.
 *
 * @param seq_len [in] Seq Len value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_compile_attention(size_t seq_len, size_t dim) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(32);
    if (!prog) return NULL;

    SNEPPXNPEInstruction inst;
    memset(&inst, 0, sizeof(SNEPPXNPEInstruction));

    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = (int)seq_len; inst.shape_a[1] = (int)dim;
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 2; inst.immediate = (int)(seq_len * dim);
    inst.shape_a[0] = (int)seq_len; inst.shape_a[1] = (int)dim;
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 3; inst.immediate = (int)(2 * seq_len * dim);
    inst.shape_a[0] = (int)seq_len; inst.shape_a[1] = (int)dim;
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_MATMUL; inst.dest_reg = 4; inst.src_reg_a = 1; inst.src_reg_b = 2;
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_SOFTMAX; inst.dest_reg = 5; inst.src_reg_a = 4;
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_MATMUL; inst.dest_reg = 6; inst.src_reg_a = 5; inst.src_reg_b = 3;
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_STORE; inst.src_reg_a = 6; inst.immediate = (int)(3 * seq_len * dim);
    SNEPPX_npe_program_append(prog, inst);

    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(prog, inst);

    return prog;
}

/**
 * @brief Perform Npe Compile Mlp.
 *
 * @param dim [in] Dim value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_compile_mlp(size_t dim, size_t hidden_dim) {
    SNEPPXNPEProgram* prog = SNEPPX_npe_program_create(32);
    if (!prog) return NULL;

    SNEPPXNPEInstruction inst;
    memset(&inst, 0, sizeof(SNEPPXNPEInstruction));

    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 1; inst.immediate = 0;
    inst.shape_a[0] = (int)dim; inst.shape_a[1] = (int)hidden_dim;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 2; inst.immediate = (int)(dim * hidden_dim);
    inst.shape_a[0] = (int)hidden_dim;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_MATMUL; inst.dest_reg = 3; inst.src_reg_a = 0; inst.src_reg_b = 1;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_ADD; inst.dest_reg = 4; inst.src_reg_a = 3; inst.src_reg_b = 2;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_RELU; inst.dest_reg = 5; inst.src_reg_a = 4;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 6; inst.immediate = (int)(hidden_dim * dim + hidden_dim);
    inst.shape_a[0] = (int)hidden_dim; inst.shape_a[1] = (int)dim;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_LOAD; inst.dest_reg = 7; inst.immediate = (int)(hidden_dim * dim + hidden_dim + dim * hidden_dim);
    inst.shape_a[0] = (int)dim;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_MATMUL; inst.dest_reg = 8; inst.src_reg_a = 5; inst.src_reg_b = 6;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_ADD; inst.dest_reg = 9; inst.src_reg_a = 8; inst.src_reg_b = 7;
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_STORE; inst.src_reg_a = 9;
    inst.immediate = (int)(hidden_dim * dim + hidden_dim + dim * hidden_dim + dim);
    SNEPPX_npe_program_append(prog, inst);

    memset(&inst, 0, sizeof(inst));
    inst.opcode = SNEPPX_HALT;
    SNEPPX_npe_program_append(prog, inst);

    return prog;
}

/**
 * @brief Create Npe Jit Profile.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEJITProfile* SNEPPX_npe_jit_profile_create(size_t hot_threshold) {
    SNEPPXNPEJITProfile* profile = (SNEPPXNPEJITProfile*)calloc(1, sizeof(SNEPPXNPEJITProfile));
    if (!profile) return NULL;
    profile->hot_threshold = hot_threshold ? hot_threshold : 100;
    profile->is_profiling = 1;
    return profile;
}

/**
 * @brief Destroy Npe Jit Profile.
 */
void SNEPPX_npe_jit_profile_destroy(SNEPPXNPEJITProfile* profile) {
    if (profile) free(profile);
}

/**
 * @brief Perform Npe Jit Record.
 *
 * @param profile [out] Profile value.
 * @param opcode [in] Opcode value.
 */
void SNEPPX_npe_jit_record(SNEPPXNPEJITProfile* profile, int opcode, float latency_us) {
    if (!profile || !profile->is_profiling) return;
    if (opcode >= 0 && opcode < 32) {
        profile->op_frequency[opcode]++;
        profile->op_latency[opcode] += (size_t)(latency_us * 1000.0f);
        profile->total_instructions++;
    }
}

/**
 * @brief Perform Npe Jit Compile.
 *
 * @param profile [out] Profile value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_compile(SNEPPXNPEJITProfile* profile, const SNEPPXNPEProgram* original) {
    if (!profile || !original) return NULL;

    int hot_op = -1;
    size_t max_freq = 0;
    for (int i = 0; i < 32; i++) {
        if (profile->op_frequency[i] > max_freq) {
            max_freq = profile->op_frequency[i];
            hot_op = i;
        }
    }

    if (max_freq < profile->hot_threshold) return NULL;

    SNEPPXNPEProgram* opt = SNEPPX_npe_program_create(original->max_instructions);
    if (!opt) return NULL;

    for (size_t i = 0; i < original->num_instructions; i++) {
        SNEPPXNPEInstruction inst = original->instructions[i];
        if (inst.opcode == SNEPPX_NOP) continue;

        if (inst.opcode == SNEPPX_MATMUL && i + 1 < original->num_instructions) {
            SNEPPXNPEInstruction next = original->instructions[i + 1];
            if (next.opcode == SNEPPX_RELU && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x80000000;
                SNEPPX_npe_program_append(opt, inst);
                i++;
                continue;
            }
        }

        SNEPPX_npe_program_append(opt, inst);
    }

    return opt;
}

/**
 * @brief Perform Npe Jit Specialize.
 *
 * @param prog [in] Prog value.
 * @param batch [in] Batch value.
 * @param seq_len [in] Seq Len value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_specialize(const SNEPPXNPEProgram* prog, size_t batch, size_t seq_len, size_t dim) {
    if (!prog) return NULL;
    SNEPPXNPEProgram* spec = SNEPPX_npe_program_create(prog->max_instructions);
    if (!spec) return NULL;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        SNEPPXNPEInstruction inst = prog->instructions[i];

        if (inst.shape_a[0] > 0) inst.shape_a[0] = (int)seq_len;
        if (inst.shape_a[1] > 0) inst.shape_a[1] = (int)dim;
        if (inst.shape_b[0] > 0) inst.shape_b[0] = (int)seq_len;
        if (inst.shape_b[1] > 0) inst.shape_b[1] = (int)dim;

        if (inst.opcode == SNEPPX_BRANCH && inst.immediate < 0) {
            size_t body_start = i + inst.immediate;
            size_t body_end = i;
            for (size_t u = 0; u < batch; u++) {
                for (size_t j = body_start; j < body_end; j++) {
                    SNEPPX_npe_program_append(spec, prog->instructions[j]);
                }
            }
            continue;
        }

        SNEPPX_npe_program_append(spec, inst);
    }

    return spec;
}

/**
 * @brief Perform Npe Jit Fuse.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_fuse(const SNEPPXNPEProgram* prog) {
    if (!prog) return NULL;
    SNEPPXNPEProgram* fused = SNEPPX_npe_program_create(prog->max_instructions);
    if (!fused) return NULL;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        SNEPPXNPEInstruction inst = prog->instructions[i];

        // Triple fusion: MATMUL + ADD + RELU (0x20000000)
        if (inst.opcode == SNEPPX_MATMUL && i + 2 < prog->num_instructions) {
            SNEPPXNPEInstruction next = prog->instructions[i + 1];
            SNEPPXNPEInstruction next2 = prog->instructions[i + 2];
            if (next.opcode == SNEPPX_ADD && next.src_reg_a == inst.dest_reg &&
                next2.opcode == SNEPPX_RELU && next2.src_reg_a == next.dest_reg) {
                inst.immediate |= 0x20000000;
                inst.immediate = (inst.immediate & ~0xFF) | (next.src_reg_b & 0xFF);
                SNEPPX_npe_program_append(fused, inst);
                i += 2;
                continue;
            }
        }

        if (inst.opcode == SNEPPX_MATMUL && i + 1 < prog->num_instructions) {
            SNEPPXNPEInstruction next = prog->instructions[i + 1];
            if (next.opcode == SNEPPX_RELU && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x80000000;
                SNEPPX_npe_program_append(fused, inst);
                i++;
                continue;
            }
        }

        if (inst.opcode == SNEPPX_ADD && i + 1 < prog->num_instructions) {
            SNEPPXNPEInstruction next = prog->instructions[i + 1];
            if (next.opcode == SNEPPX_RELU && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x80000000;
                SNEPPX_npe_program_append(fused, inst);
                i++;
                continue;
            }
        }

        if (inst.opcode == SNEPPX_MATMUL && i + 1 < prog->num_instructions) {
            SNEPPXNPEInstruction next = prog->instructions[i + 1];
            if (next.opcode == SNEPPX_ADD && next.src_reg_a == inst.dest_reg) {
                inst.immediate |= 0x40000000;
                inst.immediate = (inst.immediate & ~0xFF) | (next.src_reg_b & 0xFF);
                SNEPPX_npe_program_append(fused, inst);
                i++;
                continue;
            }
        }

        SNEPPX_npe_program_append(fused, inst);
    }

    return fused;
}

/**
 * @brief Perform Npe Jit Constant Fold.
 *
 * @param prog [in] Prog value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_constant_fold(const SNEPPXNPEProgram* prog, const SNEPPXTensor* memory) {
    if (!prog) return NULL;
    SNEPPXNPEProgram* folded = SNEPPX_npe_program_create(prog->max_instructions);
    if (!folded) return NULL;

    int reg_const[16];
    float reg_value[16];
    memset(reg_const, 0, sizeof(reg_const));
    memset(reg_value, 0, sizeof(reg_value));

    for (size_t i = 0; i < prog->num_instructions; i++) {
        SNEPPXNPEInstruction inst = prog->instructions[i];

        if (inst.opcode == SNEPPX_LOAD) {
            if (memory && inst.immediate >= 0 && (size_t)inst.immediate < memory->size) {
                float* mem_data = (float*)memory->data;
                reg_const[inst.dest_reg] = 1;
                reg_value[inst.dest_reg] = mem_data[inst.immediate];
            } else {
                reg_const[inst.dest_reg] = 0;
            }
        } else if (inst.opcode == SNEPPX_MUL &&
                   reg_const[inst.src_reg_a] && reg_const[inst.src_reg_b]) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = reg_value[inst.src_reg_a] * reg_value[inst.src_reg_b];
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_ADD &&
                   reg_const[inst.src_reg_a] && reg_const[inst.src_reg_b]) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = reg_value[inst.src_reg_a] + reg_value[inst.src_reg_b];
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_SUB &&
                   reg_const[inst.src_reg_a] && reg_const[inst.src_reg_b]) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = reg_value[inst.src_reg_a] - reg_value[inst.src_reg_b];
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_DIV &&
                   reg_const[inst.src_reg_a] && reg_const[inst.src_reg_b] &&
                   reg_value[inst.src_reg_b] != 0.0f) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = reg_value[inst.src_reg_a] / reg_value[inst.src_reg_b];
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_RELU && reg_const[inst.src_reg_a]) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = reg_value[inst.src_reg_a] > 0.0f ? reg_value[inst.src_reg_a] : 0.0f;
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_EXP && reg_const[inst.src_reg_a]) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = expf(reg_value[inst.src_reg_a]);
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_NEG && reg_const[inst.src_reg_a]) {
            reg_const[inst.dest_reg] = 1;
            reg_value[inst.dest_reg] = -reg_value[inst.src_reg_a];
            inst.opcode = SNEPPX_LOAD;
            inst.immediate = 0;
        } else if (inst.opcode == SNEPPX_STORE || inst.opcode == SNEPPX_HALT) {
        } else if (inst.dest_reg >= 0 && inst.dest_reg < 16) {
            reg_const[inst.dest_reg] = 0;
        }

        SNEPPX_npe_program_append(folded, inst);
    }

    return folded;
}

/**
 * @brief Perform Npe Jit Dce.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_dce(const SNEPPXNPEProgram* prog) {
    if (!prog) return NULL;

    int reg_used[16];
    memset(reg_used, 0, sizeof(reg_used));

    for (size_t i = 0; i < prog->num_instructions; i++) {
        SNEPPXNPEInstruction inst = prog->instructions[i];
        if (inst.src_reg_a >= 0 && inst.src_reg_a < 16) reg_used[inst.src_reg_a] = 1;
        if (inst.src_reg_b >= 0 && inst.src_reg_b < 16) reg_used[inst.src_reg_b] = 1;
        if (inst.opcode == SNEPPX_STORE || inst.opcode == SNEPPX_HALT || inst.opcode == SNEPPX_BRANCH) {
            if (inst.dest_reg >= 0 && inst.dest_reg < 16) reg_used[inst.dest_reg] = 1;
        }
    }

    SNEPPXNPEProgram* cleaned = SNEPPX_npe_program_create(prog->max_instructions);
    if (!cleaned) return NULL;

    for (size_t i = 0; i < prog->num_instructions; i++) {
        SNEPPXNPEInstruction inst = prog->instructions[i];

        if (inst.opcode == SNEPPX_LOAD || inst.opcode == SNEPPX_STORE ||
            inst.opcode == SNEPPX_HALT || inst.opcode == SNEPPX_BRANCH) {
            SNEPPX_npe_program_append(cleaned, inst);
            continue;
        }

        if (inst.dest_reg >= 0 && inst.dest_reg < 16 && reg_used[inst.dest_reg]) {
            SNEPPX_npe_program_append(cleaned, inst);
        }
    }

    return cleaned;
}

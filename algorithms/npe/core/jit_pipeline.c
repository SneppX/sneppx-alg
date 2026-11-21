#include "neural_programming_engine.h"
#include "polymorphic_memory_allocator.h"
#include <string.h>
#include <stdlib.h>
/*
 * SNEPPX - NPE JIT Compilation Pipeline
 *
 * WHAT
 *   NPE JIT Compilation Pipeline.
 *
 * CONCEPT
 *   Just-in-time compilation pipeline for NPE programs.
 *
 * ROLE
 *   Implements the JIT pipeline that compiles programs to native code at runtime for accelerated execution.
 *
 * REFERENCES
 *   None (internal algorithm).
 */



/**
 * @brief Perform Npe Jit Optimize.
 *
 * @param profile [out] Profile value.
 * @param prog [in] Prog value.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPXNPEProgram* SNEPPX_npe_jit_optimize(SNEPPXNPEJITProfile* profile, const SNEPPXNPEProgram* prog, const SNEPPXTensor* memory) {
    if (!prog) return NULL;

    SNEPPXNPEProgram* cur = SNEPPX_npe_jit_dce(prog);
    if (!cur) return NULL;

    SNEPPXNPEProgram* folded = SNEPPX_npe_jit_constant_fold(cur, memory);
    SNEPPX_npe_program_destroy(cur);
    if (!folded) return NULL;

    SNEPPXNPEProgram* fused = SNEPPX_npe_jit_fuse(folded);
    SNEPPX_npe_program_destroy(folded);
    if (!fused) return NULL;

    if (profile) {
        SNEPPXNPEProgram* compiled = SNEPPX_npe_jit_compile(profile, fused);
        if (compiled) {
            SNEPPX_npe_program_destroy(fused);
            return compiled;
        }
    }

    return fused;
}

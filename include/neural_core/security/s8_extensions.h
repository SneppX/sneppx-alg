#ifndef SNEPPX_S8_EXTENSIONS_H
#define SNEPPX_S8_EXTENSIONS_H
/*
 * SNEPPX - S8 Extensions
 *
 * WHAT
 *   S8 Extensions.
 *
 * CONCEPT
 *   Provides the S8 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* S8 extensions: TLA+ parser, LTL model checker, symbolic execution,
   loop invariant inference, data flow analysis, Lean 4 proof export */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TLA_MAX_SPEC 4096
#define SNEPPX_LTL_MAX_FORMULA 256
#define SNEPPX_SYMEX_MAX_PATHS 1024

/* TLA+ specification parser (simplified) */
typedef struct {
    char spec[SNEPPX_TLA_MAX_SPEC];
    int parsed;
    int state_count;
} SNEPPXTLAParser;

/**
 * @brief Parse Tla.
 *
 * @param parser [out] Parser value.
 * @param spec_text [in] Spec Text value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tla_parse(SNEPPXTLAParser* parser, const char* spec_text);

/* LTL property verifier */
typedef struct {
    char formula[SNEPPX_LTL_MAX_FORMULA];
    int holds;
} SNEPPXLTLVerifier;

/**
 * @brief Initialize Ltl.
 *
 * @param ltl [out] Ltl value.
 * @param formula [in] Formula value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ltl_init(SNEPPXLTLVerifier* ltl, const char* formula);
/**
 * @brief Perform Ltl Check.
 *
 * @param ltl [out] Ltl value.
 * @param trace [out] Trace value.
 * @param trace_len [in] Trace Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ltl_check(SNEPPXLTLVerifier* ltl, int* trace, int trace_len);

/* Symbolic execution engine */
typedef struct {
    uint64_t explored_paths;
    uint64_t bounded_paths;
    int depth_limit;
} SNEPPXSymExEngine;

/**
 * @brief Initialize Symex.
 *
 * @param se [out] Se value.
 * @param depth_limit [in] Depth Limit value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_symex_init(SNEPPXSymExEngine* se, int depth_limit);
/**
 * @brief Perform Symex Explore.
 *
 * @param se [out] Se value.
 * @param bytecode [in] Bytecode value.
 * @param bc_len [in] Bc Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_symex_explore(SNEPPXSymExEngine* se, const uint8_t* bytecode, size_t bc_len);

/* Loop invariant inference */
/**
 * @brief Perform Loop Invariant Infer.
 *
 * @param loop_body [in] Loop Body value.
 * @param invariant_out [out] Invariant Out value.
 * @param inv_size [in] Inv Size value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_loop_invariant_infer(const char* loop_body, char* invariant_out, size_t inv_size);

/* Data flow analysis */
typedef struct {
    int taint_marks[256];
    int taint_count;
} SNEPPXDataFlow;

/**
 * @brief Initialize Data Flow.
 *
 * @param df [out] Df value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_data_flow_init(SNEPPXDataFlow* df);
/**
 * @brief Perform Data Flow Taint.
 *
 * @param df [out] Df value.
 * @param var_id [in] Var Id value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_data_flow_taint(SNEPPXDataFlow* df, int var_id);
/**
 * @brief Perform Data Flow Propagate.
 *
 * @param df [out] Df value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_data_flow_propagate(SNEPPXDataFlow* df);

/* Lean 4 proof export */
/**
 * @brief Perform Lean Export Proof.
 *
 * @param theorem_name [in] Theorem Name value.
 * @param proof_body [in] Proof Body value.
 * @param output_path [in] Output Path value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_lean_export_proof(const char* theorem_name, const char* proof_body, const char* output_path);

#ifdef __cplusplus
}
#endif
#endif

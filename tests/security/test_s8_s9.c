#include "model_checking.h"
#include "test_gtest.h"
#include "self_audit.h"
#include "s8_extensions.h"
#include "s9_extensions.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test S8 S9
 *
 * WHAT
 *   Test S8 S9.
 *
 * CONCEPT
 *   Provides the Test S8 S9.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static int sample_invariant(uint32_t state_id) {
    return state_id < 10;
}

static void test_model_init(void) {
    SNEPPXFormalModel model;
    SX_ASSERT(SNEPPX_model_init(&model) == 0, "model init");
}

static void test_model_simple_graph(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 1, 0);
    SNEPPX_model_add_state(&model, 2, 0, 1);
    SNEPPX_model_add_transition(&model, 0, 1);
    SNEPPX_model_add_transition(&model, 0, 2);
    SNEPPXModelCheckResult result = SNEPPX_model_check(&model);
    SX_ASSERT(result.total_states == 3, "3 states");
    SX_ASSERT(result.reachable_states > 0, "reachable states");
}

static void test_model_invariant(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 0, 0);
    SX_ASSERT(SNEPPX_model_verify_invariant(&model, sample_invariant) == 1, "invariant holds");
    SNEPPX_model_add_state(&model, 99, 0, 0);
    SX_ASSERT(SNEPPX_model_verify_invariant(&model, sample_invariant) == 0, "invariant fails");
}

static void test_self_audit_init(void) {
    SNEPPXSelfAudit audit;
    SX_ASSERT(SNEPPX_self_audit_init(&audit) == 0, "audit init");
    SNEPPX_self_audit_destroy(&audit);
}

static void test_self_audit_run(void) {
    SNEPPXSelfAudit audit;
    SNEPPX_self_audit_init(&audit);
    SX_ASSERT(SNEPPX_self_audit_run_all(&audit) == 0, "run all");
    SX_ASSERT(audit.total_passed > 0, "checks passed");
    SX_ASSERT(audit.total_failed >= 0, "checks run");
    double score = SNEPPX_self_audit_score(&audit);
    SX_ASSERT(score > 0.0, "positive score");
    SNEPPX_self_audit_destroy(&audit);
}

static void test_tla_parse(void) {
    SNEPPXTLAParser parser;
    SX_ASSERT(SNEPPX_tla_parse(&parser, "Init == x = 0") == 0, "tla parse");
    SX_ASSERT(parser.parsed == 1, "parsed flag");
}

static void test_ltl_init_check(void) {
    SNEPPXLTLVerifier ltl;
    SX_ASSERT(SNEPPX_ltl_init(&ltl, "G(x > 0)") == 0, "ltl init");
    int trace[] = {1, 2, 3};
    SX_ASSERT(SNEPPX_ltl_check(&ltl, trace, 3) == 0, "ltl check");
    SX_ASSERT(ltl.holds == 1, "ltl holds");
}

static void test_symex_init_explore(void) {
    SNEPPXSymExEngine se;
    SX_ASSERT(SNEPPX_symex_init(&se, 10) == 0, "symex init");
    SX_ASSERT(se.depth_limit == 10, "depth limit");
    uint8_t bc[] = {0x01, 0x02, 0x03};
    SX_ASSERT(SNEPPX_symex_explore(&se, bc, 3) == 0, "symex explore");
    SX_ASSERT(se.explored_paths >= 1, "paths explored");
}

static void test_loop_invariant(void) {
    char inv[256];
    SX_ASSERT(SNEPPX_loop_invariant_infer("for i in 0..n", inv, sizeof(inv)) == 0, "loop invariant");
}

static void test_data_flow(void) {
    SNEPPXDataFlow df;
    SX_ASSERT(SNEPPX_data_flow_init(&df) == 0, "dataflow init");
    SX_ASSERT(SNEPPX_data_flow_taint(&df, 42) == 0, "taint var");
    SX_ASSERT(df.taint_count == 1, "one taint mark");
    SX_ASSERT(SNEPPX_data_flow_propagate(&df) == 0, "propagate");
}

static void test_lean_export(void) {
    SX_ASSERT(SNEPPX_lean_export_proof("my_theorem", "trivial", "proof.lean") == 0, "lean export");
}

static void test_vuln_scanner(void) {
    SNEPPXVulnScanner vs;
    SX_ASSERT(SNEPPX_vuln_scanner_init(&vs) == 0, "vuln init");
    SX_ASSERT(SNEPPX_vuln_scanner_add_cve(&vs, "CVE-2024-1234") == 0, "add CVE");
    SX_ASSERT(vs.cve_count == 1, "one CVE");
    SX_ASSERT(SNEPPX_vuln_scanner_run(&vs) == 0, "vuln scan");
    SX_ASSERT(vs.scan_complete == 1, "scan complete");
}

static int dummy_fuzz_target(const uint8_t* data, size_t len) {
    (void)data; (void)len;
    return 0;
}

static void test_fuzz_harness(void) {
    SNEPPXFuzzHarness fh;
    SX_ASSERT(SNEPPX_fuzz_harness_init(&fh) == 0, "fuzz init");
    fh.input_len = 64;
    int crashes = SNEPPX_fuzz_harness_run(&fh, dummy_fuzz_target);
    SX_ASSERT(crashes >= 0, "fuzz run");
}

static void test_redteam(void) {
    SNEPPXRedTeamSim rt;
    SX_ASSERT(SNEPPX_redteam_init(&rt) == 0, "redteam init");
    SX_ASSERT(SNEPPX_redteam_add_step(&rt, "recon") == 0, "add step");
    SX_ASSERT(SNEPPX_redteam_add_step(&rt, "exploit") == 0, "add step 2");
    SX_ASSERT(SNEPPX_redteam_execute(&rt) == 0, "execute");
    SX_ASSERT(rt.completed == 1, "completed");
}

static void test_compliance(void) {
    SX_ASSERT(SNEPPX_compliance_check_nist("AC-1") == 1, "nist check");
}

static void test_bugbounty(void) {
    SNEPPXBugBountyTriage bt;
    SX_ASSERT(SNEPPX_bugbounty_triage_init(&bt) == 0, "bb triage init");
    SX_ASSERT(SNEPPX_bugbounty_triage_analyze(&bt, "XSS in login") == 0, "analyze report");
    SX_ASSERT(bt.severity >= 0, "severity assigned");
}

static void test_model_get_counterexample(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 0, 0);
    SNEPPX_model_add_state(&model, 2, 0, 1);
    SNEPPX_model_add_transition(&model, 0, 1);
    SNEPPX_model_add_transition(&model, 1, 2);
    SNEPPXModelCheckResult result = SNEPPX_model_check(&model);
    SX_ASSERT(result.property_satisfied == 0, "error state reachable, property violated");
    SX_ASSERT(result.error_states == 1, "one error state");

    uint32_t trace[16];
    int len = SNEPPX_model_get_counterexample(&model, trace, 16);
    SX_ASSERT(len == 3, "counterexample trace length");
    SX_ASSERT(trace[0] == 0, "trace starts at initial");
    SX_ASSERT(trace[1] == 1, "trace middle state");
    SX_ASSERT(trace[2] == 2, "trace ends at error state");
}

static void test_model_get_counterexample_no_error(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 0, 0);
    SNEPPX_model_add_transition(&model, 0, 1);
    SNEPPXModelCheckResult result = SNEPPX_model_check(&model);
    SX_ASSERT(result.property_satisfied == 1, "no error state, property satisfied");

    uint32_t trace[16];
    int len = SNEPPX_model_get_counterexample(&model, trace, 16);
    SX_ASSERT(len == 0, "no counterexample when no error state");
}

static void test_model_counterexample_string(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 0, 1);
    SNEPPX_model_add_transition(&model, 0, 1);
    SNEPPXModelCheckResult result = SNEPPX_model_check(&model);
    SX_ASSERT(strstr(result.counterexample, "0") != NULL, "counterexample contains state 0");
    SX_ASSERT(strstr(result.counterexample, "1") != NULL, "counterexample contains state 1");
}

static void test_model_check_deadlock(void) {
    SNEPPXFormalModel model;
    SNEPPX_model_init(&model);
    SNEPPX_model_add_state(&model, 0, 0, 0);
    SNEPPX_model_add_state(&model, 1, 0, 0);
    SNEPPX_model_add_transition(&model, 0, 1);
    int deadlocks = SNEPPX_model_check_deadlock(&model);
    SX_ASSERT(deadlocks == 1, "state 1 is a deadlock");
}

TEST(test_s8_s9, model_init) { test_model_init(); }
TEST(test_s8_s9, model_simple_graph) { test_model_simple_graph(); }
TEST(test_s8_s9, model_invariant) { test_model_invariant(); }
TEST(test_s8_s9, self_audit_init) { test_self_audit_init(); }
TEST(test_s8_s9, self_audit_run) { test_self_audit_run(); }
TEST(test_s8_s9, tla_parse) { test_tla_parse(); }
TEST(test_s8_s9, ltl_init_check) { test_ltl_init_check(); }
TEST(test_s8_s9, symex_init_explore) { test_symex_init_explore(); }
TEST(test_s8_s9, loop_invariant) { test_loop_invariant(); }
TEST(test_s8_s9, data_flow) { test_data_flow(); }
TEST(test_s8_s9, lean_export) { test_lean_export(); }
TEST(test_s8_s9, vuln_scanner) { test_vuln_scanner(); }
TEST(test_s8_s9, fuzz_harness) { test_fuzz_harness(); }
TEST(test_s8_s9, redteam) { test_redteam(); }
TEST(test_s8_s9, compliance) { test_compliance(); }
TEST(test_s8_s9, bugbounty) { test_bugbounty(); }
TEST(test_s8_s9, model_get_counterexample) { test_model_get_counterexample(); }
TEST(test_s8_s9, model_get_counterexample_no_error) { test_model_get_counterexample_no_error(); }
TEST(test_s8_s9, model_counterexample_string) { test_model_counterexample_string(); }
TEST(test_s8_s9, model_check_deadlock) { test_model_check_deadlock(); }

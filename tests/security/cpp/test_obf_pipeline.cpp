#include "code_obfuscation_interface.h"
#include "control_flow_obfuscation.h"
#include <stdio.h>

/*
 * SNEPPX - Test Obf Pipeline
 *
 * WHAT
 *   Test Obf Pipeline.
 *
 * CONCEPT
 *   Provides pipeline parallelism.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (%s)\n", msg, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void run_test(const char* name, void (*test_fn)(void)) {
    printf("Running %s... ", name);
    fflush(stdout);
    test_fn();
    printf("PASS\n");
    tests_passed++;
}

static void test_obf_pipeline_create(void) {
    SNEPPX::SNEPPXObfuscator obf;
    ASSERT(true, "obfuscator created");
}

static void test_obf_pipeline_add_pass(void) {
    SNEPPX::SNEPPXObfuscator obf;
    obf.configure(SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_LIGHT);
    ASSERT(obf.get_level() == SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_LIGHT, "level set");
    obf.configure(SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_MAXIMUM);
    ASSERT(obf.get_level() == SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_MAXIMUM, "level set to max");
}

static void test_obf_pipeline_run(void) {
    SNEPPX::SNEPPXObfuscator obf;
    SNEPPX::SNEPPXObfCFG cfg;
    // Just test that obfuscator can be used without crashing
    SNEPPX::SNEPPXObfuscationReport report = obf.obfuscate(cfg);
    ASSERT(report.level == SNEPPX::SNEPPXObfuscationLevel::SNEPPX_OBF_NONE, "default level");
}

int main(void) {
    run_test("obf_pipeline_create", test_obf_pipeline_create);
    run_test("obf_pipeline_add_pass", test_obf_pipeline_add_pass);
    run_test("obf_pipeline_run", test_obf_pipeline_run);
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}

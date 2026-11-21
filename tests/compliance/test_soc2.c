#include "security/compliance/soc2_framework.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Soc2
 *
 * WHAT
 *   Test Soc2.
 *
 * CONCEPT
 *   Provides the Test Soc2.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_soc2_control_assessment(void) {
    SNEPPXSOC2Framework fw;
    memset(&fw, 0, sizeof(fw));
    int ret = snepx_soc2_framework_init(&fw, "SOC2 Test", "1.0");
    assert(ret == 0);
    assert(fw.num_controls == 0);
    snepx_soc2_framework_destroy(&fw);
}

static void test_soc2_add_control(void) {
    SNEPPXSOC2Framework fw;
    snepx_soc2_framework_init(&fw, "SOC2 Controls", "1.0");
    SNEPPXSOC2Control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.control_ref = "CC6.1";
    ctrl.control_ref_len = 5;
    ctrl.status = SNEPPX_SOC2_CONTROL_PASSED;
    int ret = snepx_soc2_framework_add_control(&fw, &ctrl);
    assert(ret == 0);
    assert(fw.num_controls == 1);
    snepx_soc2_framework_destroy(&fw);
}

int main(void) {
    test_soc2_control_assessment();
    test_soc2_add_control();
    return 0;
}

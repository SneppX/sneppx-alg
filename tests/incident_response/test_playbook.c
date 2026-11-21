#include "security/incident_response/playbook_engine.h"
#include <assert.h>
#include <string.h>

/*
 * SNEPPX - Test Playbook
 *
 * WHAT
 *   Test Playbook.
 *
 * CONCEPT
 *   Provides the Test Playbook.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static void test_playbook_create(void) {
    SNEPPXIRPlaybook pb;
    memset(&pb, 0, sizeof(pb));
    int ret = snepx_ir_playbook_create(&pb, "test_playbook", SNEPPX_IR_PLAYBOOK_TYPE_INCIDENT_RESPONSE);
    assert(ret == 0);
    assert(pb.num_steps == 0);
    assert(pb.status == SNEPPX_IR_PLAYBOOK_STATUS_IDLE);
    snepx_ir_playbook_destroy(&pb);
}

static void test_playbook_add_step(void) {
    SNEPPXIRPlaybook pb;
    snepx_ir_playbook_create(&pb, "step_test", SNEPPX_IR_PLAYBOOK_TYPE_FORENSIC_ACQUISITION);
    SNEPPXIRPlaybookStep step;
    memset(&step, 0, sizeof(step));
    step.action_type = SNEPPX_IR_ACTION_COLLECT_EVIDENCE;
    step.action_description = "Collect memory dump";
    step.action_description_len = 19;
    step.command = "dumpmem";
    step.command_len = 7;
    step.critical = 1;
    int ret = snepx_ir_playbook_add_step(&pb, &step);
    assert(ret == 0);
    assert(pb.num_steps == 1);
    snepx_ir_playbook_destroy(&pb);
}

int main(void) {
    test_playbook_create();
    test_playbook_add_step();
    return 0;
}

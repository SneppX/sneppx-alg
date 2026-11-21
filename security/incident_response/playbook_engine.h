#ifndef SNEPPX_IR_PLAYBOOK_ENGINE_H
#define SNEPPX_IR_PLAYBOOK_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_IR_PLAYBOOK_MAX_STEPS 256
#define SNEPPX_IR_PLAYBOOK_MAX_CONDITIONS 64
#define SNEPPX_IR_PLAYBOOK_MAX_VARIABLES 128
#define SNEPPX_IR_PLAYBOOK_NESTING_MAX 16

/*
 * SNEPPX - Playbook Engine
 *
 * WHAT
 *   Playbook Engine.
 *
 * CONCEPT
 *   Provides the Playbook Engine.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_IR_PLAYBOOK_TYPE_INCIDENT_RESPONSE,
    SNEPPX_IR_PLAYBOOK_TYPE_SOC_RUNBOOK,
    SNEPPX_IR_PLAYBOOK_TYPE_FORENSIC_ACQUISITION,
    SNEPPX_IR_PLAYBOOK_TYPE_MALWARE_ANALYSIS,
    SNEPPX_IR_PLAYBOOK_TYPE_THREAT_HUNTING,
    SNEPPX_IR_PLAYBOOK_TYPE_REMEDIATION,
    SNEPPX_IR_PLAYBOOK_TYPE_RECOVERY,
    SNEPPX_IR_PLAYBOOK_TYPE_POST_MORTEM
} SNEPPXIRPlaybookType;

typedef enum {
    SNEPPX_IR_ACTION_RUN_COMMAND,
    SNEPPX_IR_ACTION_PARSE_OUTPUT,
    SNEPPX_IR_ACTION_DECISION,
    SNEPPX_IR_ACTION_COLLECT_EVIDENCE,
    SNEPPX_IR_ACTION_ANALYZE_FILE,
    SNEPPX_IR_ACTION_CREATE_ALERT,
    SNEPPX_IR_ACTION_CREATE_TICKET,
    SNEPPX_IR_ACTION_BLOCK_IOC,
    SNEPPX_IR_ACTION_ISOLATE_HOST,
    SNEPPX_IR_ACTION_KILL_PROCESS,
    SNEPPX_IR_ACTION_QUARANTINE_FILE,
    SNEPPX_IR_ACTION_REVERT_SNAPSHOT,
    SNEPPX_IR_ACTION_NOTIFY,
    SNEPPX_IR_ACTION_ESCALATE,
    SNEPPX_IR_ACTION_WAIT,
    SNEPPX_IR_ACTION_GOTO,
    SNEPPX_IR_ACTION_SUB_PLAYBOOK
} SNEPPXIRPlaybookAction;

typedef enum {
    SNEPPX_IR_CONDITION_EQUALS,
    SNEPPX_IR_CONDITION_CONTAINS,
    SNEPPX_IR_CONDITION_MATCHES_REGEX,
    SNEPPX_IR_CONDITION_GREATER_THAN,
    SNEPPX_IR_CONDITION_LESS_THAN,
    SNEPPX_IR_CONDITION_IN_RANGE,
    SNEPPX_IR_CONDITION_EXISTS,
    SNEPPX_IR_CONDITION_NOT_EXISTS,
    SNEPPX_IR_CONDITION_AND,
    SNEPPX_IR_CONDITION_OR,
    SNEPPX_IR_CONDITION_NOT
} SNEPPXIRConditionOp;

typedef enum {
    SNEPPX_IR_PLAYBOOK_STATUS_IDLE,
    SNEPPX_IR_PLAYBOOK_STATUS_RUNNING,
    SNEPPX_IR_PLAYBOOK_STATUS_PAUSED,
    SNEPPX_IR_PLAYBOOK_STATUS_COMPLETED,
    SNEPPX_IR_PLAYBOOK_STATUS_FAILED,
    SNEPPX_IR_PLAYBOOK_STATUS_TIMEOUT,
    SNEPPX_IR_PLAYBOOK_STATUS_ESCALATED,
    SNEPPX_IR_PLAYBOOK_STATUS_ABORTED
} SNEPPXIRPlaybookStatus;

typedef struct {
    char* variable_name;
    size_t name_len;
    uint8_t* variable_value;
    size_t value_len;
    uint8_t is_global : 1;
    uint8_t is_secret : 1;
} SNEPPXIRPlaybookVariable;

typedef struct {
    SNEPPXIRPlaybookAction action_type;
    char* action_description;
    size_t description_len;
    char* command;
    size_t command_len;
    uint32_t command_timeout_ms;
    char* on_success_goto;
    size_t on_success_goto_len;
    char* on_failure_goto;
    size_t on_failure_goto_len;
    uint32_t retry_count;
    uint32_t retry_delay_ms;
    uint8_t* expected_output;
    size_t expected_output_len;
    uint8_t critical : 1;
    uint8_t requires_approval : 1;
    uint8_t automated : 1;
} SNEPPXIRPlaybookStep;

typedef struct {
    char* condition_variable;
    size_t condition_variable_len;
    SNEPPXIRConditionOp op;
    uint8_t* condition_value;
    size_t condition_value_len;
    char* true_goto;
    size_t true_goto_len;
    char* false_goto;
    size_t false_goto_len;
} SNEPPXIRPlaybookCondition;

typedef struct {
    char* playbook_name;
    size_t playbook_name_len;
    char* playbook_version;
    size_t playbook_version_len;
    SNEPPXIRPlaybookType type;
    SNEPPXIRPlaybookStatus status;
    SNEPPXIRPlaybookStep* steps;
    SNEPPXIRPlaybookCondition* conditions;
    SNEPPXIRPlaybookVariable* variables;
    uint32_t num_steps;
    uint32_t num_conditions;
    uint32_t num_variables;
    uint32_t current_step_index;
    uint64_t started_at_ns;
    uint64_t completed_at_ns;
    uint8_t* incident_context;
    size_t incident_context_len;
    uint8_t* execution_log;
    size_t execution_log_len;
    uint8_t* error_log;
    size_t error_log_len;
    uint32_t execution_count;
    uint8_t running : 1;
} SNEPPXIRPlaybook;

int snepx_ir_playbook_create(SNEPPXIRPlaybook* playbook, const char* name, SNEPPXIRPlaybookType type);
int snepx_ir_playbook_add_step(SNEPPXIRPlaybook* playbook, const SNEPPXIRPlaybookStep* step);
int snepx_ir_playbook_add_condition(SNEPPXIRPlaybook* playbook, const SNEPPXIRPlaybookCondition* condition);
int snepx_ir_playbook_add_variable(SNEPPXIRPlaybook* playbook, const SNEPPXIRPlaybookVariable* variable);
int snepx_ir_playbook_execute(SNEPPXIRPlaybook* playbook);
int snepx_ir_playbook_execute_step(SNEPPXIRPlaybook* playbook, uint32_t step_index);
int snepx_ir_playbook_pause(SNEPPXIRPlaybook* playbook);
int snepx_ir_playbook_resume(SNEPPXIRPlaybook* playbook);
int snepx_ir_playbook_abort(SNEPPXIRPlaybook* playbook);
int snepx_ir_playbook_reset(SNEPPXIRPlaybook* playbook);
int snepx_ir_playbook_export(SNEPPXIRPlaybook* playbook, uint8_t* out, size_t* out_len);
int snepx_ir_playbook_import(SNEPPXIRPlaybook* playbook, const uint8_t* in, size_t in_len);
int snepx_ir_playbook_destroy(SNEPPXIRPlaybook* playbook);

// Automation rules engine
int snepx_ir_playbook_evaluate_condition(const SNEPPXIRPlaybookCondition* condition, const uint8_t* actual_value, size_t actual_val_len, uint8_t* result);
int snepx_ir_playbook_resolve_variable(const SNEPPXIRPlaybook* playbook, const char* var_name, uint8_t** value, size_t* value_len);
int snepx_ir_playbook_set_variable(SNEPPXIRPlaybook* playbook, const char* var_name, const uint8_t* value, size_t value_len);

#endif

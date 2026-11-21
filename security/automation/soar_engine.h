#ifndef SNEPPX_SOAR_ENGINE_H
#define SNEPPX_SOAR_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_SOAR_MAX_WORKFLOWS 1024
#define SNEPPX_SOAR_MAX_TRIGGERS 256
#define SNEPPX_SOAR_MAX_ACTIONS 64
#define SNEPPX_SOAR_MAX_INTEGRATIONS 128

/*
 * SNEPPX - Soar Engine
 *
 * WHAT
 *   Soar Engine.
 *
 * CONCEPT
 *   Provides the Soar Engine.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_SOAR_TRIGGER_BY_ALERT,
    SNEPPX_SOAR_TRIGGER_BY_SCHEDULE,
    SNEPPX_SOAR_TRIGGER_BY_IOC,
    SNEPPX_SOAR_TRIGGER_BY_THREAT_INTEL,
    SNEPPX_SOAR_TRIGGER_BY_EVENT,
    SNEPPX_SOAR_TRIGGER_BY_WEBHOOK,
    SNEPPX_SOAR_TRIGGER_BY_API_CALL,
    SNEPPX_SOAR_TRIGGER_MANUAL
} SNEPPXSOARTriggerType;

typedef enum {
    SNEPPX_SOAR_ACTION_BLOCK_IP,
    SNEPPX_SOAR_ACTION_BLOCK_DOMAIN,
    SNEPPX_SOAR_ACTION_BLOCK_URL,
    SNEPPX_SOAR_ACTION_QUARANTINE_HOST,
    SNEPPX_SOAR_ACTION_TERMINATE_PROCESS,
    SNEPPX_SOAR_ACTION_DISABLE_USER,
    SNEPPX_SOAR_ACTION_RESET_PASSWORD,
    SNEPPX_SOAR_ACTION_CREATE_TICKET,
    SNEPPX_SOAR_ACTION_SEND_EMAIL,
    SNEPPX_SOAR_ACTION_SEND_SLACK,
    SNEPPX_SOAR_ACTION_SEND_PAGERDUTY,
    SNEPPX_SOAR_ACTION_RUN_PLAYBOOK,
    SNEPPX_SOAR_ACTION_UPDATE_THREAT_INTEL,
    SNEPPX_SOAR_ACTION_SANDBOX_SUBMIT,
    SNEPPX_SOAR_ACTION_UPDATE_FIREWALL,
    SNEPPX_SOAR_ACTION_UPDATE_SIEM,
    SNEPPX_SOAR_ACTION_UPDATE_EDR,
    SNEPPX_SOAR_ACTION_SNAPSHOT_VM,
    SNEPPX_SOAR_ACTION_COLLECT_FORENSICS,
    SNEPPX_SOAR_ACTION_ESCALATE
} SNEPPXSOARActionType;

typedef struct {
    uint64_t integration_id;
    SNEPPXSOARActionType action_type;
    char* integration_name;
    size_t name_len;
    char* endpoint_url;
    size_t url_len;
    uint8_t* api_key;
    size_t api_key_len;
    uint8_t* auth_bearer;
    size_t auth_len;
    uint8_t* ca_cert;
    size_t ca_cert_len;
    uint32_t timeout_ms;
    uint32_t retry_limit;
    uint8_t enabled : 1;
    uint8_t verified : 1;
} SNEPPXSOARIntegration;

typedef struct {
    SNEPPXSOARTriggerType trigger;
    char* trigger_filter;
    size_t trigger_filter_len;
    SNEPPXSOARActionType* actions;
    uint32_t num_actions;
    uint32_t cooldown_seconds;
    uint8_t active : 1;
    uint8_t auto_approve : 1;
} SNEPPXSOARRule;

typedef struct {
    char* workflow_name;
    size_t workflow_name_len;
    SNEPPXSOARRule rules[SNEPPX_SOAR_MAX_ACTIONS];
    uint32_t num_rules;
    SNEPPXSOARIntegration* integrations;
    uint32_t num_integrations;
    uint32_t active_count;
    uint32_t total_executions;
    uint32_t success_count;
    uint32_t failure_count;
    uint8_t enabled : 1;
} SNEPPXSOARWorkflow;

int snepx_soar_workflow_create(SNEPPXSOARWorkflow* wf, const char* name);
int snepx_soar_workflow_add_rule(SNEPPXSOARWorkflow* wf, const SNEPPXSOARRule* rule);
int snepx_soar_workflow_add_integration(SNEPPXSOARWorkflow* wf, const SNEPPXSOARIntegration* integration);
int snepx_soar_workflow_execute(SNEPPXSOARWorkflow* wf, SNEPPXSOARTriggerType trigger, const uint8_t* context, size_t ctx_len);
int snepx_soar_workflow_execute_action(SNEPPXSOARWorkflow* wf, SNEPPXSOARActionType action, const uint8_t* params, size_t params_len, uint8_t* result, size_t* result_len);
int snepx_soar_workflow_get_stats(const SNEPPXSOARWorkflow* wf, uint8_t* stats_out, size_t* stats_len);
int snepx_soar_workflow_destroy(SNEPPXSOARWorkflow* wf);

// Pre-built integration helpers
int snepx_soar_action_block_ip(const SNEPPXSOARIntegration* integration, const char* ip);
int snepx_soar_action_block_domain(const SNEPPXSOARIntegration* integration, const char* domain);
int snepx_soar_action_quarantine_host(const SNEPPXSOARIntegration* integration, const char* hostname);
int snepx_soar_action_create_ticket(const SNEPPXSOARIntegration* integration, const char* title, const char* description, const char* priority);
int snepx_soar_action_send_email(const SNEPPXSOARIntegration* integration, const char* to, const char* subject, const char* body);
int snepx_soar_action_sandbox_submit(const SNEPPXSOARIntegration* integration, const uint8_t* sample, size_t sample_len, uint8_t* report_out, size_t* report_len);

#endif

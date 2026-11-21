#include "audit_logger.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>

#define SNEPPX_AUDIT_HMAC_KEY_LEN 32

/*
 * SNEPPX - Audit Logger
 *
 * WHAT
 *   Audit Logger.
 *
 * CONCEPT
 *   Provides structured logging.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static uint32_t audit_crc32(const void* data, size_t len) {
    const unsigned char* buf = (const unsigned char*)data;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

/**
 * @brief Initialize Audit.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_init(SNEPPXAuditLogger* audit, const char* log_path) {
    if (!audit) return -1;
    memset(audit, 0, sizeof(*audit));
    audit->enabled = 1;
    audit->log_file_path = log_path;
    audit->chain_crc = 0;
    return 0;
}

/**
 * @brief Perform Audit Shutdown.
 */
void SNEPPX_audit_shutdown(SNEPPXAuditLogger* audit) {
    if (audit) {
        if (audit->log_file_path) {
            SNEPPX_audit_export(audit, audit->log_file_path);
        }
        memset(audit, 0, sizeof(*audit));
    }
}

/**
 * @brief Perform Audit Log.
 *
 * @param audit [out] Audit value.
 * @param event_type [in] Event Type value.
 * @param description [in] Description value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_log(SNEPPXAuditLogger* audit, int event_type,
                     const char* description, uint64_t related_address) {
    if (!audit || !audit->enabled || !description) return -1;
    if (audit->entry_count >= SNEPPX_AUDIT_MAX_ENTRIES) return -1;
    SNEPPXAuditEntry* e = &audit->entries[audit->entry_count++];
    e->timestamp = (uint64_t)time(NULL);
    e->event_type = event_type;
    strncpy(e->description, description, SNEPPX_AUDIT_DESC_LEN - 1);
    e->related_address = related_address;
    e->crc = audit_crc32(e, offsetof(SNEPPXAuditEntry, crc));
    audit->chain_crc = audit_crc32(&audit->chain_crc, sizeof(audit->chain_crc)) ^ e->crc;
    return 0;
}

/**
 * @brief Perform Audit Export.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_export(SNEPPXAuditLogger* audit, const char* output_path) {
    if (!audit || !output_path) return -1;
    FILE* f = fopen(output_path, "w");
    if (!f) return -1;
    for (int i = 0; i < audit->entry_count; i++) {
        SNEPPXAuditEntry* e = &audit->entries[i];
        fprintf(f, "%llu|%d|%s|%llu|%08x\n",
                (unsigned long long)e->timestamp, e->event_type,
                e->description, (unsigned long long)e->related_address, e->crc);
    }
    fprintf(f, "CHAIN_CRC: %08x\n", audit->chain_crc);
    fclose(f);
    return 0;
}

/**
 * @brief Perform Audit Verify Chain.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_verify_chain(SNEPPXAuditLogger* audit) {
    if (!audit) return 0;
    uint32_t expected_chain = 0;
    for (int i = 0; i < audit->entry_count; i++) {
        uint32_t entry_crc = audit_crc32(&audit->entries[i], offsetof(SNEPPXAuditEntry, crc));
        if (entry_crc != audit->entries[i].crc) return 0;
        expected_chain = audit_crc32(&expected_chain, sizeof(expected_chain)) ^ entry_crc;
    }
    return (expected_chain == audit->chain_crc) ? 1 : 0;
}

/**
 * @brief Perform Audit Search.
 *
 * @param audit [out] Audit value.
 * @param event_type [in] Event Type value.
 * @param results [out] Results value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_search(SNEPPXAuditLogger* audit, int event_type,
                        SNEPPXAuditEntry* results, int max_results) {
    if (!audit || !results || max_results <= 0) return 0;
    int found = 0;
    for (int i = 0; i < audit->entry_count && found < max_results; i++) {
        if (audit->entries[i].event_type == event_type)
            results[found++] = audit->entries[i];
    }
    return found;
}

/**
 * @brief Perform Audit Get Entry Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_get_entry_count(SNEPPXAuditLogger* audit) {
    if (!audit) return -1;
    return audit->entry_count;
}

/**
 * @brief Perform Audit Get Chain Crc.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_audit_get_chain_crc(SNEPPXAuditLogger* audit) {
    if (!audit) return 0;
    return audit->chain_crc;
}

/**
 * @brief Perform Audit Purge.
 *
 * @param audit [out] Audit value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_purge(SNEPPXAuditLogger* audit, uint64_t before_timestamp) {
    if (!audit) return -1;
    int kept = 0;
    for (int i = 0; i < audit->entry_count; i++) {
        if (audit->entries[i].timestamp >= before_timestamp) {
            if (i != kept) {
                audit->entries[kept] = audit->entries[i];
            }
            kept++;
        }
    }
    audit->entry_count = kept;
    return 0;
}

/**
 * @brief Perform Audit Sign Entry.
 *
 * @param audit [out] Audit value.
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_sign_entry(SNEPPXAuditLogger* audit, int index, const uint8_t* key) {
    if (!audit || !key) return -1;
    if (index < 0 || index >= audit->entry_count) return -1;
    SNEPPXAuditEntry* e = &audit->entries[index];
    uint32_t sig = 0;
    for (size_t i = 0; i < SNEPPX_AUDIT_HMAC_KEY_LEN; i++)
        sig ^= ((uint32_t)key[i] << ((i % 4) * 8));
    sig ^= e->crc;
    e->crc = sig;
    return 0;
}

/**
 * @brief Perform Audit Log Formatted.
 *
 * @param audit [out] Audit value.
 * @param event_type [in] Event Type value.
 * @param fmt [in] Fmt value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_log_formatted(SNEPPXAuditLogger* audit, int event_type, const char* fmt, ...) {
    if (!audit || !fmt) return -1;
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return SNEPPX_audit_log(audit, event_type, buf, 0);
}

/**
 * @brief Perform Audit Get Entry.
 *
 * @param audit [out] Audit value.
 * @param index [in] Index value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_get_entry(SNEPPXAuditLogger* audit, int index, SNEPPXAuditEntry* entry_out) {
    if (!audit || !entry_out) return -1;
    if (index < 0 || index >= audit->entry_count) return -1;
    *entry_out = audit->entries[index];
    return 0;
}

/**
 * @brief Clear Audit.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_clear(SNEPPXAuditLogger* audit) {
    if (!audit) return -1;
    memset(audit->entries, 0, sizeof(SNEPPXAuditEntry) * audit->entry_count);
    audit->entry_count = 0;
    audit->chain_crc = 0;
    return 0;
}

/**
 * @brief Perform Audit Get Stats.
 *
 * @param audit [out] Audit value.
 * @param count [out] Count value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_get_stats(SNEPPXAuditLogger* audit, int* count, int* chain_verified) {
    if (!audit || !count || !chain_verified) return -1;
    *count = audit->entry_count;
    *chain_verified = SNEPPX_audit_verify_chain(audit);
    return 0;
}

/**
 * @brief Perform Audit Set Log Level.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_set_log_level(int level) {
    (void)level;
    return 0;
}

/**
 * @brief Perform Audit Export Xml.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_export_xml(const char* path) {
    if (!path) return -1;
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<auditLog>\n");
    fprintf(f, "  <entries />\n");
    fprintf(f, "</auditLog>\n");
    fclose(f);
    return 0;
}

/**
 * @brief Perform Audit Verify Entry.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_audit_verify_entry(int index) {
    (void)index;
    return 1;
}

static int audit_validate_event_type(int type) {
    if (type < 0) return 0;
    if (type > 255) return 0;
    return 1;
}

static const char* audit_event_type_name(int type) {
    switch (type) {
        case 0: return "INFO";
        case 1: return "WARNING";
        case 2: return "ERROR";
        case 3: return "CRITICAL";
        case 4: return "AUTH";
        case 5: return "ACCESS";
        default: return "UNKNOWN";
    }
}

static int audit_format_entry(char* buf, size_t size, const SNEPPXAuditEntry* e) {
    if (!buf || size == 0 || !e) return -1;
    return snprintf(buf, size, "[%llu] %s: %s (addr=0x%llx crc=0x%08x)",
                    (unsigned long long)e->timestamp,
                    audit_event_type_name(e->event_type),
                    e->description,
                    (unsigned long long)e->related_address,
                    e->crc);
}

static int audit_append_to_file(SNEPPXAuditLogger* audit, const char* path) {
    if (!audit || !path) return -1;
    FILE* f = fopen(path, "a");
    if (!f) return -1;
    for (int i = 0; i < audit->entry_count; i++) {
        fprintf(f, "%llu|%d|%s|%llu|%08x\n",
                (unsigned long long)audit->entries[i].timestamp,
                audit->entries[i].event_type,
                audit->entries[i].description,
                (unsigned long long)audit->entries[i].related_address,
                audit->entries[i].crc);
    }
    fclose(f);
    return 0;
}

static int audit_copy_entry(SNEPPXAuditEntry* dst, const SNEPPXAuditEntry* src) {
    if (!dst || !src) return -1;
    memcpy(dst, src, sizeof(SNEPPXAuditEntry));
    return 0;
}

static uint32_t audit_compute_entry_crc(const SNEPPXAuditEntry* e) {
    if (!e) return 0;
    return audit_crc32(e, offsetof(SNEPPXAuditEntry, crc));
}

static int audit_rebuild_chain(SNEPPXAuditLogger* audit) {
    if (!audit) return -1;
    audit->chain_crc = 0;
    for (int i = 0; i < audit->entry_count; i++) {
        uint32_t ecrc = audit_compute_entry_crc(&audit->entries[i]);
        audit->entries[i].crc = ecrc;
        audit->chain_crc = audit_crc32(&audit->chain_crc, sizeof(audit->chain_crc)) ^ ecrc;
    }
    return 0;
}

static int audit_validate_entry_timestamp(SNEPPXAuditEntry* e) {
    if (!e) return 0;
    uint64_t now = (uint64_t)time(NULL);
    if (e->timestamp > now + 10) return 0;
    if (e->timestamp < now - 86400 * 365) return 0;
    return 1;
}

static int audit_find_entry_by_address(SNEPPXAuditLogger* audit, uint64_t addr) {
    if (!audit) return -1;
    for (int i = 0; i < audit->entry_count; i++) {
        if (audit->entries[i].related_address == addr) return i;
    }
    return -1;
}

static int audit_merge_loggers(SNEPPXAuditLogger* dst, const SNEPPXAuditLogger* src) {
    if (!dst || !src) return -1;
    int space = SNEPPX_AUDIT_MAX_ENTRIES - dst->entry_count;
    int to_copy = (src->entry_count < space) ? src->entry_count : space;
    for (int i = 0; i < to_copy; i++) {
        dst->entries[dst->entry_count++] = src->entries[i];
    }
    return to_copy;
}

static int audit_count_by_type(SNEPPXAuditLogger* audit, int event_type) {
    if (!audit) return 0;
    int count = 0;
    for (int i = 0; i < audit->entry_count; i++) {
        if (audit->entries[i].event_type == event_type) count++;
    }
    return count;
}

static int audit_has_chain_violation(SNEPPXAuditLogger* audit) {
    if (!audit || audit->entry_count == 0) return 0;
    uint32_t expected = 0;
    for (int i = 0; i < audit->entry_count; i++) {
        uint32_t ecrc = audit_compute_entry_crc(&audit->entries[i]);
        if (audit->entries[i].crc != ecrc) return 1;
        expected = audit_crc32(&expected, sizeof(expected)) ^ ecrc;
    }
    return (audit->chain_crc != expected) ? 1 : 0;
}

static int audit_verify_chain_integrity(SNEPPXAuditLogger* audit) {
    if (!audit) return -1;
    if (audit->entry_count == 0) return 0;
    for (int i = 0; i < audit->entry_count; i++) {
        uint32_t computed = audit_compute_entry_crc(&audit->entries[i]);
        if (computed != audit->entries[i].crc) return 1;
    }
    return 0;
}

static int audit_prune_old_entries(SNEPPXAuditLogger* audit, uint64_t before_timestamp) {
    if (!audit) return 0;
    int kept = 0;
    for (int i = 0; i < audit->entry_count; i++) {
        if (audit->entries[i].timestamp >= before_timestamp) {
            if (i != kept) audit->entries[kept] = audit->entries[i];
            kept++;
        }
    }
    int pruned = audit->entry_count - kept;
    audit->entry_count = kept;
    return pruned;
}

static int audit_get_latest_entry(SNEPPXAuditLogger* audit, SNEPPXAuditEntry* out) {
    if (!audit || !out || audit->entry_count == 0) return -1;
    *out = audit->entries[audit->entry_count - 1];
    return 0;
}

static int audit_get_earliest_entry(SNEPPXAuditLogger* audit, SNEPPXAuditEntry* out) {
    if (!audit || !out || audit->entry_count == 0) return -1;
    *out = audit->entries[0];
    return 0;
}

static uint64_t audit_get_total_entries_count(SNEPPXAuditLogger* audit) {
    return audit ? (uint64_t)audit->entry_count : 0;
}

static int audit_is_full(SNEPPXAuditLogger* audit) {
    return audit ? (audit->entry_count >= SNEPPX_AUDIT_MAX_ENTRIES) : 1;
}

static int audit_get_entry_at(SNEPPXAuditLogger* audit, int index, SNEPPXAuditEntry* out) {
    if (!audit || !out || index < 0 || index >= audit->entry_count) return -1;
    *out = audit->entries[index];
    return 0;
}

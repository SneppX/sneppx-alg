#include "key_vault.h"
#include "test_gtest.h"
#include "audit_logger.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test S6 Ui
 *
 * WHAT
 *   Test S6 Ui.
 *
 * CONCEPT
 *   Provides the Test S6 Ui.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_key_vault_init(void) {
    SNEPPXKeyVault vault;
    SX_ASSERT(SNEPPX_key_vault_init(&vault) == 0, "init");
    SX_ASSERT(vault.is_locked == 1, "locked by default");
    SNEPPX_key_vault_destroy(&vault);
}

static void test_key_vault_generate_get(void) {
    SNEPPXKeyVault vault;
    SNEPPX_key_vault_init(&vault);
    SNEPPX_key_vault_unlock(&vault, NULL);
    uint8_t key_id[16];
    SX_ASSERT(SNEPPX_key_vault_generate_key(&vault, key_id, 3600) >= 0, "generate key");
    uint8_t key_out[32];
    SX_ASSERT(SNEPPX_key_vault_get_key(&vault, key_id, key_out) == 0, "get key");
    SNEPPX_key_vault_destroy(&vault);
}

static void test_key_vault_rotate_revoke(void) {
    SNEPPXKeyVault vault;
    SNEPPX_key_vault_init(&vault);
    SNEPPX_key_vault_unlock(&vault, NULL);
    uint8_t key_id[16];
    SNEPPX_key_vault_generate_key(&vault, key_id, 3600);
    uint8_t old_key[32];
    SNEPPX_key_vault_get_key(&vault, key_id, old_key);
    SX_ASSERT(SNEPPX_key_vault_rotate_key(&vault, key_id) == 0, "rotate");
    uint8_t new_key[32];
    SNEPPX_key_vault_get_key(&vault, key_id, new_key);
    int diff = 0;
    for (int i = 0; i < 32; i++) if (old_key[i] != new_key[i]) diff = 1;
    SX_ASSERT(diff, "key changed");
    SX_ASSERT(SNEPPX_key_vault_revoke_key(&vault, key_id) == 0, "revoke");
    SX_ASSERT(SNEPPX_key_vault_get_key(&vault, key_id, new_key) != 0, "revoked key fails");
    SNEPPX_key_vault_destroy(&vault);
}

static void test_audit_logger(void) {
    SNEPPXAuditLogger audit;
    SX_ASSERT(SNEPPX_audit_init(&audit, NULL) == 0, "audit init");
    SX_ASSERT(SNEPPX_audit_log(&audit, 1, "test event", 0x42) == 0, "log entry");
    SX_ASSERT(audit.entry_count == 1, "1 entry");
    SX_ASSERT(SNEPPX_audit_verify_chain(&audit) == 1, "chain valid");
    SNEPPX_audit_shutdown(&audit);
}

static void test_audit_search(void) {
    SNEPPXAuditLogger audit;
    SNEPPX_audit_init(&audit, NULL);
    SNEPPX_audit_log(&audit, 1, "first event", 0);
    SNEPPX_audit_log(&audit, 2, "second event", 0);
    SNEPPX_audit_log(&audit, 1, "third event", 0);
    SNEPPXAuditEntry results[10];
    int found = SNEPPX_audit_search(&audit, 1, results, 10);
    SX_ASSERT(found == 2, "found 2 type-1 events");
    SNEPPX_audit_shutdown(&audit);
}


TEST(test_s6_ui, key_vault_init) { test_key_vault_init(); }
TEST(test_s6_ui, key_vault_generate_get) { test_key_vault_generate_get(); }
TEST(test_s6_ui, key_vault_rotate_revoke) { test_key_vault_rotate_revoke(); }
TEST(test_s6_ui, audit_logger) { test_audit_logger(); }
TEST(test_s6_ui, audit_search) { test_audit_search(); }

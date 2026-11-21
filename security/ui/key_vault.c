#include "key_vault.h"
#include "cryptographic_random_generator.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/*
 * SNEPPX - Key Vault
 *
 * WHAT
 *   Key Vault.
 *
 * CONCEPT
 *   Provides key vault management.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef SNEPPXKeyVault SNEPPXVault;

typedef struct {
    int encrypted_blocks;
} SNEPPXVaultBackup;

#define SNEPPX_VAULT_AUTO_ROTATE_ACCESS 100

static int auto_rotate_threshold = SNEPPX_VAULT_AUTO_ROTATE_ACCESS;
static uint8_t master_key_hash[32];
static int master_key_set = 0;

/**
 * @brief Initialize Key Vault.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_init(SNEPPXKeyVault* vault) {
    if (!vault) return -1;
    memset(vault, 0, sizeof(*vault));
    vault->is_locked = 1;
    auto_rotate_threshold = SNEPPX_VAULT_AUTO_ROTATE_ACCESS;
    master_key_set = 0;
    return 0;
}

/**
 * @brief Destroy Key Vault.
 */
void SNEPPX_key_vault_destroy(SNEPPXKeyVault* vault) {
    if (!vault) return;
    for (int i = 0; i < vault->key_count; i++)
        memset(vault->keys[i].key_data, 0, SNEPPX_VAULT_KEY_LEN);
    memset(vault, 0, sizeof(*vault));
}

/**
 * @brief Perform Key Vault Generate Key.
 *
 * @param vault [out] Vault value.
 * @param key_id [out] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_generate_key(SNEPPXKeyVault* vault, uint8_t* key_id, uint64_t ttl_seconds) {
    if (!vault || vault->is_locked || !key_id || vault->key_count >= SNEPPX_VAULT_MAX_KEYS) return -1;
    SNEPPXVaultKey* k = &vault->keys[vault->key_count];
    SNEPPX_random_bytes(k->id, SNEPPX_VAULT_ID_LEN);
    SNEPPX_random_bytes(k->key_data, SNEPPX_VAULT_KEY_LEN);
    k->created_at = (uint64_t)time(NULL);
    k->expires_at = (ttl_seconds > 0) ? k->created_at + ttl_seconds : 0;
    k->is_active = 1;
    k->access_count = 0;
    if (key_id) memcpy(key_id, k->id, SNEPPX_VAULT_ID_LEN);
    return vault->key_count++;
}

/**
 * @brief Perform Key Vault Get Key.
 *
 * @param vault [out] Vault value.
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_key(SNEPPXKeyVault* vault, const uint8_t* key_id, uint8_t* key_out) {
    if (!vault || vault->is_locked || !key_id || !key_out) return -1;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active && memcmp(vault->keys[i].id, key_id, SNEPPX_VAULT_ID_LEN) == 0) {
            if (vault->keys[i].expires_at > 0 && (uint64_t)time(NULL) > vault->keys[i].expires_at) {
                vault->keys[i].is_active = 0;
                return -1;
            }
            memcpy(key_out, vault->keys[i].key_data, SNEPPX_VAULT_KEY_LEN);
            vault->keys[i].access_count++;
            if (auto_rotate_threshold > 0 && vault->keys[i].access_count >= auto_rotate_threshold) {
                SNEPPX_random_bytes(vault->keys[i].key_data, SNEPPX_VAULT_KEY_LEN);
                vault->keys[i].created_at = (uint64_t)time(NULL);
                vault->keys[i].access_count = 0;
            }
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Key Vault Rotate Key.
 *
 * @param vault [out] Vault value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_rotate_key(SNEPPXKeyVault* vault, const uint8_t* key_id) {
    if (!vault || vault->is_locked || !key_id) return -1;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active && memcmp(vault->keys[i].id, key_id, SNEPPX_VAULT_ID_LEN) == 0) {
            SNEPPX_random_bytes(vault->keys[i].key_data, SNEPPX_VAULT_KEY_LEN);
            vault->keys[i].created_at = (uint64_t)time(NULL);
            vault->keys[i].access_count = 0;
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Key Vault Revoke Key.
 *
 * @param vault [out] Vault value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_revoke_key(SNEPPXKeyVault* vault, const uint8_t* key_id) {
    if (!vault || vault->is_locked || !key_id) return -1;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active && memcmp(vault->keys[i].id, key_id, SNEPPX_VAULT_ID_LEN) == 0) {
            vault->keys[i].is_active = 0;
            memset(vault->keys[i].key_data, 0, SNEPPX_VAULT_KEY_LEN);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Key Vault Lock.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_lock(SNEPPXKeyVault* vault) {
    if (!vault) return -1;
    vault->is_locked = 1;
    return 0;
}

/**
 * @brief Perform Key Vault Unlock.
 *
 * @param vault [out] Vault value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_unlock(SNEPPXKeyVault* vault, const uint8_t* master_key) {
    if (!vault) return -1;
    if (master_key_set && master_key) {
        uint32_t check = 0;
        for (int i = 0; i < 32 && i < SNEPPX_VAULT_KEY_LEN; i++)
            check |= (uint8_t)(master_key[i] ^ master_key_hash[i]);
        if (check != 0) return -1;
    }
    if (master_key && !master_key_set) {
        for (int i = 0; i < 32 && i < SNEPPX_VAULT_KEY_LEN; i++)
            master_key_hash[i] = master_key[i];
        master_key_set = 1;
    }
    vault->is_locked = 0;
    return 0;
}

/**
 * @brief Perform Key Vault Get Key Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_key_count(SNEPPXKeyVault* vault) {
    if (!vault) return -1;
    return vault->key_count;
}

/**
 * @brief Perform Key Vault Get Key Info.
 *
 * @param vault [out] Vault value.
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_key_info(SNEPPXKeyVault* vault, const uint8_t* key_id, SNEPPXVaultKey* info_out) {
    if (!vault || !key_id || !info_out) return -1;
    for (int i = 0; i < vault->key_count; i++) {
        if (memcmp(vault->keys[i].id, key_id, SNEPPX_VAULT_ID_LEN) == 0) {
            memcpy(info_out->id, vault->keys[i].id, SNEPPX_VAULT_ID_LEN);
            info_out->created_at = vault->keys[i].created_at;
            info_out->expires_at = vault->keys[i].expires_at;
            info_out->is_active = vault->keys[i].is_active;
            info_out->access_count = vault->keys[i].access_count;
            memset(info_out->key_data, 0, SNEPPX_VAULT_KEY_LEN);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Perform Key Vault List Keys.
 *
 * @param vault [out] Vault value.
 * @param key_id_list [out] Key Id List value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_list_keys(SNEPPXKeyVault* vault, uint8_t* key_id_list, int max) {
    if (!vault || !key_id_list || max <= 0) return -1;
    int written = 0;
    for (int i = 0; i < vault->key_count && written < max; i++) {
        if (vault->keys[i].is_active) {
            memcpy(key_id_list + written * SNEPPX_VAULT_ID_LEN, vault->keys[i].id, SNEPPX_VAULT_ID_LEN);
            written++;
        }
    }
    return written;
}

/**
 * @brief Perform Key Vault Set Auto Rotate Threshold.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_set_auto_rotate_threshold(int access_count) {
    if (access_count < 0) return -1;
    auto_rotate_threshold = access_count;
    return 0;
}

/**
 * @brief Perform Key Vault Get Key Ttl.
 *
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_key_ttl(const uint8_t* key_id, uint64_t* ttl_out) {
    (void)key_id;
    if (!ttl_out) return -1;
    *ttl_out = 3600;
    return 0;
}

/**
 * @brief Perform Key Vault Set Key Ttl.
 *
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_set_key_ttl(const uint8_t* key_id, uint64_t ttl) {
    (void)key_id;
    (void)ttl;
    return 0;
}

/**
 * @brief Perform Key Vault Get Key Created.
 *
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_key_created(const uint8_t* key_id, uint64_t* created_out) {
    (void)key_id;
    if (!created_out) return -1;
    *created_out = (uint64_t)time(NULL);
    return 0;
}

/**
 * @brief Perform Key Vault Get Key Access Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_key_access_count(const uint8_t* key_id) {
    (void)key_id;
    return 0;
}

/**
 * @brief Perform Key Vault Get Active Key Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_active_key_count(void) {
    return 0;
}

/**
 * @brief Perform Key Vault Get Expired Key Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_get_expired_key_count(void) {
    return 0;
}

/**
 * @brief Perform Key Vault Purge Expired.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_purge_expired(void) {
    return 0;
}

/**
 * @brief Perform Key Vault Export Key Vault.
 *
 * @param vault [in] Vault value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_key_vault_export_key_vault(const uint8_t* vault, const uint8_t* key, size_t key_len) {
    if (!vault || !key) return -1;
    (void)vault;
    (void)key;
    (void)key_len;
    return 0;
}

static int vault_find_key_index(SNEPPXKeyVault* vault, const uint8_t* key_id) {
    if (!vault || !key_id) return -1;
    for (int i = 0; i < vault->key_count; i++) {
        if (memcmp(vault->keys[i].id, key_id, SNEPPX_VAULT_ID_LEN) == 0) return i;
    }
    return -1;
}

static int vault_count_active(SNEPPXKeyVault* vault) {
    if (!vault) return 0;
    int count = 0;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active) count++;
    }
    return count;
}

static int vault_count_expired(SNEPPXKeyVault* vault) {
    if (!vault) return 0;
    int count = 0;
    uint64_t now = (uint64_t)time(NULL);
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].expires_at > 0 && now > vault->keys[i].expires_at) count++;
    }
    return count;
}

static int vault_purge_inactive(SNEPPXKeyVault* vault) {
    if (!vault) return -1;
    int kept = 0;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active) {
            if (i != kept) vault->keys[kept] = vault->keys[i];
            kept++;
        }
    }
    vault->key_count = kept;
    return 0;
}

static int vault_validate_ttl(uint64_t ttl) {
    if (ttl == 0) return 1;
    if (ttl < 60) return 0;
    if (ttl > 86400 * 365) return 0;
    return 1;
}

static uint64_t vault_get_current_time(void) {
    return (uint64_t)time(NULL);
}

static int vault_key_matches(SNEPPXVaultKey* key, const uint8_t* key_id) {
    if (!key || !key_id) return 0;
    return memcmp(key->id, key_id, SNEPPX_VAULT_ID_LEN) == 0;
}

static void vault_scramble_key_data(uint8_t* data, size_t len) {
    if (!data || len == 0) return;
    for (size_t i = 0; i < len; i++) {
        data[i] ^= (uint8_t)((i * 13 + 7) % 256);
    }
}

static int vault_write_debug_log(const char* msg) {
    if (!msg) return -1;
    FILE* f = fopen("vault_debug.log", "a");
    if (!f) return -1;
    fprintf(f, "[%llu] %s\n", (unsigned long long)time(NULL), msg);
    fclose(f);
    return 0;
}

static int vault_rotate_secret_value(uint8_t* secret, size_t len) {
    if (!secret || len == 0) return -1;
    for (size_t i = 0; i < len; i++) secret[i] ^= (uint8_t)(i + 1);
    for (size_t i = 0; i < len; i++) secret[i] = (uint8_t)(secret[i] << 1 | secret[i] >> 7);
    return 0;
}

static int vault_compare_secrets(const uint8_t* a, size_t alen, const uint8_t* b, size_t blen) {
    if (!a || !b || alen != blen) return -1;
    return memcmp(a, b, alen);
}

static uint64_t vault_compute_backup_fingerprint(const uint8_t* data, size_t len) {
    if (!data || len == 0) return 0;
    uint64_t fp = 0;
    for (size_t i = 0; i < len; i++) fp = (fp * 31) + data[i];
    return fp;
}

static int vault_validate_backup_data(const uint8_t* data, size_t len) {
    if (!data || len < 4) return 0;
    uint32_t magic = ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) | ((uint32_t)data[2] << 8) | data[3];
    return (magic == 0x41525856) ? 1 : 0;
}

static int vault_setup_new_backup_key(const uint8_t* seed, size_t seed_len, uint8_t* key_out) {
    if (!seed || !key_out || seed_len < 16) return -1;
    for (size_t i = 0; i < 32; i++) key_out[i] = seed[i % seed_len] ^ (uint8_t)(i * 37);
    return 0;
}

static int vault_is_backup_encrypted(const SNEPPXVaultBackup* backup) {
    if (!backup) return 0;
    return (backup->encrypted_blocks > 0) ? 1 : 0;
}

static int vault_estimate_backup_size(size_t plaintext_len) {
    return (int)(plaintext_len * 2 + 128);
}

static int vault_compute_checksum(const uint8_t* data, size_t len, uint8_t* sum_out) {
    if (!data || !sum_out || len == 0) return -1;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
    }
    crc ^= 0xFFFFFFFF;
    for (int i = 0; i < 4; i++) sum_out[i] = (uint8_t)(crc >> (i * 8));
    return 0;
}

static int vault_validate_key_algorithm(const char* alg) {
    if (!alg) return 0;
    return (strcmp(alg, "AES-256") == 0 || strcmp(alg, "ChaCha20") == 0 || strcmp(alg, "AES-128") == 0) ? 1 : 0;
}

static int vault_check_key_expiry(SNEPPXVaultKey* key) {
    if (!key) return -1;
    uint64_t now = vault_get_current_time();
    return (key->expires_at > 0 && now > key->expires_at) ? 1 : 0;
}

static int vault_rotate_all_active_keys(SNEPPXVault* vault) {
    if (!vault) return -1;
    int rotated = 0;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active) {
            for (int j = 0; j < SNEPPX_VAULT_ID_LEN; j++) vault->keys[i].id[j] ^= (uint8_t)(i + j + 1);
            rotated++;
        }
    }
    return rotated;
}

static int vault_get_key_count_by_type(SNEPPXVault* vault, int type) {
    if (!vault) return 0;
    int count = 0;
    for (int i = 0; i < vault->key_count; i++) {
        if (vault->keys[i].is_active == type) count++;
    }
    return count;
}

static int vault_purge_expired_keys(SNEPPXVault* vault) {
    if (!vault) return 0;
    int purged = 0;
    for (int i = vault->key_count - 1; i >= 0; i--) {
        if (vault_check_key_expiry(&vault->keys[i]) == 1) {
            memset(&vault->keys[i], 0, sizeof(SNEPPXVaultKey));
            purged++;
        }
    }
    return purged;
}

static int vault_count_active_keys(SNEPPXVault* vault) {
    if (!vault) return 0;
    int count = 0;
    for (int i = 0; i < vault->key_count; i++) if (vault->keys[i].is_active) count++;
    return count;
}

static int vault_count_inactive_keys(SNEPPXVault* vault) {
    if (!vault) return 0;
    return vault->key_count - vault_count_active_keys(vault);
}

static int vault_has_reached_max_keys(SNEPPXVault* vault) {
    if (!vault) return 1;
    return (vault->key_count >= SNEPPX_VAULT_MAX_KEYS) ? 1 : 0;
}

static int vault_get_key_index_by_id(SNEPPXVault* vault, const uint8_t* key_id) {
    if (!vault || !key_id) return -1;
    for (int i = 0; i < vault->key_count; i++) {
        if (memcmp(vault->keys[i].id, key_id, SNEPPX_VAULT_ID_LEN) == 0) return i;
    }
    return -1;
}

static int vault_is_empty(SNEPPXVault* vault) {
    return vault ? (vault->key_count == 0) : 1;
}

static int vault_get_backup_block_count(SNEPPXVaultBackup* backup) {
    return backup ? backup->encrypted_blocks : 0;
}

static int vault_estimate_backup_time_ms(size_t data_size) {
    return (int)(data_size / 1024) + 50;
}

static int vault_dump_status(SNEPPXVault* vault, char* out, int out_size) {
    if (!vault || !out || out_size <= 0) return -1;
    snprintf(out, out_size, "keys=%d, active=%d, max=%d", vault->key_count, vault_count_active_keys(vault), SNEPPX_VAULT_MAX_KEYS);
    return 0;
}

static int vault_is_key_active(SNEPPXVault* vault, int index) {
    if (!vault || index < 0 || index >= vault->key_count) return -1;
    return vault->keys[index].is_active ? 1 : 0;
}

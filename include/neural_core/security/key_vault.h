#ifndef SNEPPX_KEY_VAULT_H
#define SNEPPX_KEY_VAULT_H
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


/*
 * S6 Security UI — Key Management Vault
 * Secure key generation, storage, rotation, and access auditing.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_VAULT_MAX_KEYS 64
#define SNEPPX_VAULT_KEY_LEN 32
#define SNEPPX_VAULT_ID_LEN 16

typedef struct {
    uint8_t id[SNEPPX_VAULT_ID_LEN];
    uint8_t key_data[SNEPPX_VAULT_KEY_LEN];
    uint64_t created_at;
    uint64_t expires_at;
    int is_active;
    int access_count;
} SNEPPXVaultKey;

typedef struct {
    SNEPPXVaultKey keys[SNEPPX_VAULT_MAX_KEYS];
    int key_count;
    int is_locked;
} SNEPPXKeyVault;

/**
 * @brief Initialize Key Vault.
 *
 * @param vault [out] Vault value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_init(SNEPPXKeyVault* vault);
/**
 * @brief Destroy Key Vault.
 *
 * @param vault [out] Vault value.
 */
void SNEPPX_key_vault_destroy(SNEPPXKeyVault* vault);
/**
 * @brief Perform Key Vault Generate Key.
 *
 * @param vault [out] Vault value.
 * @param key_id [out] Key Id value.
 * @param ttl_seconds [in] Ttl Seconds value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_generate_key(SNEPPXKeyVault* vault, uint8_t* key_id, uint64_t ttl_seconds);
/**
 * @brief Perform Key Vault Get Key.
 *
 * @param vault [out] Vault value.
 * @param key_id [in] Key Id value.
 * @param key_out [out] Key Out value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_get_key(SNEPPXKeyVault* vault, const uint8_t* key_id, uint8_t* key_out);
/**
 * @brief Perform Key Vault Rotate Key.
 *
 * @param vault [out] Vault value.
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_rotate_key(SNEPPXKeyVault* vault, const uint8_t* key_id);
/**
 * @brief Perform Key Vault Revoke Key.
 *
 * @param vault [out] Vault value.
 * @param key_id [in] Key Id value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_revoke_key(SNEPPXKeyVault* vault, const uint8_t* key_id);
/**
 * @brief Perform Key Vault Lock.
 *
 * @param vault [out] Vault value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_lock(SNEPPXKeyVault* vault);
/**
 * @brief Perform Key Vault Unlock.
 *
 * @param vault [out] Vault value.
 * @param master_key [in] Master Key value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_key_vault_unlock(SNEPPXKeyVault* vault, const uint8_t* master_key);

#ifdef __cplusplus
}
#endif
#endif

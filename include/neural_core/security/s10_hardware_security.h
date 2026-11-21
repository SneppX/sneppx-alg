#ifndef SNEPPX_S10_HARDWARE_SECURITY_H
#define SNEPPX_S10_HARDWARE_SECURITY_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_TPM_MAX_PCR_BANKS 16
#define SNEPPX_SE_KEY_SLOTS 32
#define SNEPPX_SGX_MAX_ENCLAVES 64
#define SNEPPX_TEE_SESSION_TIMEOUT 30000

/*
 * SNEPPX - S10 Hardware Security
 *
 * WHAT
 *   S10 Hardware Security.
 *
 * CONCEPT
 *   Provides the S10 Hardware Security.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_TEE_NONE,
    SNEPPX_TEE_SGX,
    SNEPPX_TEE_SEV_SNP,
    SNEPPX_TEE_TDX,
    SNEPPX_TEE_TRUSTZONE,
    SNEPPX_TEE_SEV_ES,
    SNEPPX_TEE_CCA
} SNEPPXTeeType;

typedef enum {
    SNEPPX_TPM_ALG_SHA256,
    SNEPPX_TPM_ALG_SHA384,
    SNEPPX_TPM_ALG_SHA512,
    SNEPPX_TPM_ALG_SM3_256
} SNEPPXTPMAlg;

typedef struct {
    uint8_t pcr_values[SNEPPX_TPM_MAX_PCR_BANKS][64];
    size_t pcr_count;
    SNEPPXTPMAlg pcr_bank_alg;
    uint32_t pcr_mask;
    uint8_t quote_signature[512];
    size_t quote_len;
} SNEPPXTPMAttestation;

typedef struct {
    uint8_t sealed_key[128];
    uint8_t auth_data[64];
    uint32_t key_slot;
    uint64_t policy_handle;
    uint32_t timeout_ms;
} SNEPPXTpmKey;

typedef struct {
    uint8_t measurement[64];
    uint8_t mrenclave[32];
    uint8_t mrsigner[32];
    uint8_t report_data[64];
    uint32_t attributes;
    uint16_t svn;
} SNEPPXSGXEnclave;

typedef struct {
    uint64_t physical_id;
    uint64_t api_major;
    uint64_t api_minor;
    uint64_t build_id;
    uint8_t policy[64];
    uint8_t chip_id[64];
    uint32_t vm_pl;
    uint32_t policy_flags;
} SNEPPXSevSnpAttestation;

typedef struct {
    uint8_t key_derivation_key[32];
    uint8_t sealing_key[32];
    uint32_t session_handle;
    uint64_t monotonic_counter;
} SNEPPXTeeSession;

int snepx_tee_initialize(SNEPPXTeeType tee_type);
int snepx_tee_create_session(SNEPPXTeeSession* session, const uint8_t* policy, size_t policy_len);
int snepx_tee_seal_data(SNEPPXTeeSession* session, const uint8_t* data, size_t data_len, uint8_t* sealed, size_t* sealed_len);
int snepx_tee_unseal_data(SNEPPXTeeSession* session, const uint8_t* sealed, size_t sealed_len, uint8_t* data, size_t* data_len);
int snepx_tee_attest(SNEPPXTeeSession* session, uint8_t* attestation, size_t* attestation_len);
int snepx_tee_verify_attestation(const uint8_t* attestation, size_t attestation_len, const uint8_t* expected_measurement);
int snepx_tee_destroy_session(SNEPPXTeeSession* session);

int snepx_tpm_pcr_read(uint32_t pcr_index, uint8_t* value, size_t* value_len);
int snepx_tpm_pcr_extend(uint32_t pcr_index, const uint8_t* value, size_t value_len);
int snepx_tpm_quote(SNEPPXTPMAttestation* attestation, const uint8_t* nonce, size_t nonce_len);
int snepx_tpm_seal(SNEPPXTpmKey* key, const uint8_t* data, size_t data_len);
int snepx_tpm_unseal(SNEPPXTpmKey* key, uint8_t* data, size_t* data_len);

int snepx_sgx_create_enclave(const uint8_t* enclave_binary, size_t binary_len, SNEPPXSGXEnclave* enclave, uint32_t* enclave_id);
int snepx_sgx_enter_enclave(uint32_t enclave_id, uint32_t function_id, void* input, size_t input_len, void* output, size_t* output_len);
int snepx_sgx_destroy_enclave(uint32_t enclave_id);

int snepx_sev_snp_attest(SNEPPXSevSnpAttestation* attestation, const uint8_t* report_data, size_t report_data_len);
int snepx_sev_snp_verify(const SNEPPXSevSnpAttestation* attestation, const uint8_t* trusted_root);

// Secure key storage
int snepx_hsm_store_key(uint32_t key_slot, const uint8_t* key, size_t key_len, const uint8_t* acl, size_t acl_len);
int snepx_hsm_load_key(uint32_t key_slot, uint8_t* key, size_t* key_len, const uint8_t* auth_token, size_t auth_len);
int snepx_hsm_generate_key(uint32_t key_slot, const char* algorithm);
int snepx_hsm_delete_key(uint32_t key_slot);

// Platform security
int snepx_secure_boot_verify(const uint8_t* firmware, size_t firmware_len, const uint8_t* signature, size_t sig_len);
int snepx_measured_boot_log(uint8_t* log, size_t* log_len);
int snepx_dma_remap_configure(uint64_t iova_start, uint64_t iova_end);
int snepx_smm_protection_enable(void);

#endif

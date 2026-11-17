#ifndef SNEPPX_S7_EXTENSIONS_H
#define SNEPPX_S7_EXTENSIONS_H
/*
 * SNEPPX - S7 Extensions
 *
 * WHAT
 *   S7 Extensions.
 *
 * CONCEPT
 *   Provides the S7 Extensions.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* S7 extensions: TUF compliance, bsdiff delta, A/B partitions, manifest
   verification, TPM attestation, canary rollout, offline bundles, dependency resolver */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNEPPX_TUF_MAX_KEYS 8
#define TPM_PCR_COUNT 24

/* TUF (The Update Framework) */
typedef struct {
    uint8_t root_key[32];
    uint8_t targets_key[32];
    uint8_t snapshot_key[32];
    uint8_t timestamp_key[32];
    int initialized;
} SNEPPXTUFMetadata;

/**
 * @brief Initialize Tuf.
 *
 * @param tuf [out] Tuf value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tuf_init(SNEPPXTUFMetadata* tuf);
/**
 * @brief Perform Tuf Sign Root.
 *
 * @param tuf [out] Tuf value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param sig [out] Sig value.
 * @param sig_len [out] Sig Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tuf_sign_root(SNEPPXTUFMetadata* tuf, const uint8_t* data, size_t len, uint8_t* sig, size_t* sig_len);
/**
 * @brief Perform Tuf Verify Targets.
 *
 * @param tuf [out] Tuf value.
 * @param targets_json [in] Targets Json value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tuf_verify_targets(SNEPPXTUFMetadata* tuf, const uint8_t* targets_json, size_t len);

/* bsdiff delta generation */
/**
 * @brief Perform Bsdiff.
 *
 * @param old_data [in] Old Data value.
 * @param old_len [in] Old Len value.
 * @param new_data [in] New Data value.
 * @param new_len [in] New Len value.
 * @param patch [out] Patch value.
 * @param patch_len [out] Patch Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bsdiff(const uint8_t* old_data, size_t old_len, const uint8_t* new_data, size_t new_len, uint8_t* patch, size_t* patch_len);
/**
 * @brief Perform Bspatch.
 *
 * @param old_data [in] Old Data value.
 * @param old_len [in] Old Len value.
 * @param patch [in] Patch value.
 * @param patch_len [in] Patch Len value.
 * @param new_data [out] New Data value.
 * @param new_len [out] New Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bspatch(const uint8_t* old_data, size_t old_len, const uint8_t* patch, size_t patch_len, uint8_t* new_data, size_t* new_len);

/* File-based delta helpers used by the update pipeline. */
int  SNEPPX_bsdiff_apply_patch(const char* old_path, const char* patch_path, const char* new_path);
int  SNEPPX_bspatch_apply_patch(const char* old_path, const char* patch_path, const char* new_path);

/* A/B partition management */
typedef struct {
    int active_slot;
    int inactive_slot;
    uint8_t slot_a_hash[32];
    uint8_t slot_b_hash[32];
    int swap_ready;
} SNEPPXABPartition;

/**
 * @brief Initialize Ab Partition.
 *
 * @param ab [out] Ab value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ab_partition_init(SNEPPXABPartition* ab);
/**
 * @brief Perform Ab Partition Mark Good.
 *
 * @param ab [out] Ab value.
 * @param slot [in] Slot value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ab_partition_mark_good(SNEPPXABPartition* ab, int slot);
/**
 * @brief Perform Ab Partition Swap.
 *
 * @param ab [out] Ab value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_ab_partition_swap(SNEPPXABPartition* ab);

/* Manifest verification */
/**
 * @brief Verify Manifest.
 *
 * @param manifest_path [in] Manifest Path value.
 * @param signature [in] Signature value.
 * @param sig_len [in] Sig Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_manifest_verify(const char* manifest_path, const uint8_t* signature, size_t sig_len);

/* TPM attestation */
/**
 * @brief Read Tpm Pcr.
 *
 * @param pcr_index [in] Pcr Index value.
 * @param out [out] Out value.
 * @param out_len [out] Out Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tpm_pcr_read(int pcr_index, uint8_t* out, size_t* out_len);
/**
 * @brief Perform Tpm Quote.
 *
 * @param nonce [in] Nonce value.
 * @param nonce_len [in] Nonce Len value.
 * @param quote [out] Quote value.
 * @param quote_len [out] Quote Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tpm_quote(const uint8_t* nonce, size_t nonce_len, uint8_t* quote, size_t* quote_len);

/* Canary rollout */
typedef struct {
    int total_nodes;
    int canary_nodes;
    int promoted;
} SNEPPXCanaryRollout;

/**
 * @brief Initialize Canary Rollout.
 *
 * @param cr [out] Cr value.
 * @param total [in] Total value.
 * @param canary [in] Canary value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_canary_rollout_init(SNEPPXCanaryRollout* cr, int total, int canary);
/**
 * @brief Perform Canary Rollout Promote.
 *
 * @param cr [out] Cr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_canary_rollout_promote(SNEPPXCanaryRollout* cr);

/* Offline update bundle */
typedef struct {
    uint8_t bundle_hash[32];
    size_t bundle_size;
    int signed_offline;
} SNEPPXOfflineBundle;

/**
 * @brief Create Offline Bundle.
 *
 * @param ob [out] Ob value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param signing_key [in] Signing Key value.
 * @param key_len [in] Key Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_offline_bundle_create(SNEPPXOfflineBundle* ob, const uint8_t* data, size_t data_len, const uint8_t* signing_key, size_t key_len);

/* Dependency resolver */
typedef struct {
    char name[64];
    uint32_t version_major, version_minor, version_patch;
    int resolved;
} SNEPPXDepResolver;

/**
 * @brief Initialize Dep Resolver.
 *
 * @param dr [out] Dr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_dep_resolver_init(SNEPPXDepResolver* dr);
/**
 * @brief Perform Dep Resolver Add Dep.
 *
 * @param dr [out] Dr value.
 * @param name [in] Name value.
 * @param maj [in] Maj value.
 * @param min [in] Min value.
 * @param pat [in] Pat value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_dep_resolver_add_dep(SNEPPXDepResolver* dr, const char* name, uint32_t maj, uint32_t min, uint32_t pat);
/**
 * @brief Perform Dep Resolver Resolve.
 *
 * @param dr [out] Dr value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_dep_resolver_resolve(SNEPPXDepResolver* dr);

#ifdef __cplusplus
}
#endif
#endif

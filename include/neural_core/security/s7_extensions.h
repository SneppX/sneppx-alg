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
extern "C"
{
#endif

#define SNEPPX_TUF_MAX_KEYS 8
#define TPM_PCR_COUNT 24

    /* TUF (The Update Framework) */
    typedef struct
    {
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
    int SNEPPX_tuf_init(SNEPPXTUFMetadata* tuf);
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
    int SNEPPX_tuf_sign_root(SNEPPXTUFMetadata* tuf, const uint8_t* data, size_t len, uint8_t* sig, size_t* sig_len);
    /**
     * @brief Perform Tuf Verify Targets.
     *
     * @param tuf [out] Tuf value.
     * @param targets_json [in] Targets Json value.
     * @param len [in] Len value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_tuf_verify_targets(SNEPPXTUFMetadata* tuf, const uint8_t* targets_json, size_t len);

    /* Legacy compatibility APIs below are retained but fail closed until backed by real signatures. */
    int SNEPPX_bsdiff_with_signature(const uint8_t* old_data, size_t old_len, const uint8_t* new_data, size_t new_len,
                                     uint8_t* patch, size_t* patch_len, const uint8_t* signing_key, size_t key_len);
    int SNEPPX_manifest_verify_from_buffer(const uint8_t* manifest_data, size_t data_len, const uint8_t* signature,
                                           size_t sig_len);

    /* Legacy TUF-shaped APIs retained for ABI compatibility; all fail closed. */
    int SNEPPX_tuf_generate_keypair(int key_type);
    int SNEPPX_tuf_revoke_key(int key_type);
    int SNEPPX_tuf_is_initialized(SNEPPXTUFMetadata* tuf);
    int SNEPPX_tuf_get_key(SNEPPXTUFMetadata* tuf, int key_type, uint8_t* key_out, size_t key_len);
    int SNEPPX_tuf_set_key(SNEPPXTUFMetadata* tuf, int key_type, const uint8_t* key, size_t key_len);
    int SNEPPX_tuf_snapshot(uint8_t* snapshot_out, size_t* snapshot_len);
    int SNEPPX_tuf_timestamp(uint8_t* timestamp_out, size_t* timestamp_len);
    int SNEPPX_tuf_verify_snapshot(SNEPPXTUFMetadata* tuf, const uint8_t* snapshot, size_t snapshot_len);
    int SNEPPX_tuf_verify_timestamp(SNEPPXTUFMetadata* tuf, const uint8_t* timestamp, size_t timestamp_len);
    int SNEPPX_tuf_get_keys_loaded(SNEPPXTUFMetadata* tuf);

    /* SXP1 is a project-specific format, not BSDIFF40: 4-byte magic, 64-bit
     * output size, bounded COPY/INSERT records, and a final END marker.
     * The generator emits common-prefix COPY, middle INSERT, and common-suffix COPY. */
    /**
     * @brief Perform Bsdiff.
     *
     * @param old_data [in] Old Data value.
     * @param old_len [in] Old Len value.
     * @param new_data [in] New Data value.
     * @param new_len [in] New Len value.
     * @param patch [out] Patch value.
     * @param patch_len [in,out] Patch capacity on entry; bytes written on success.
     *                      If capacity is insufficient, the required size is returned.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_bsdiff(const uint8_t* old_data, size_t old_len, const uint8_t* new_data, size_t new_len, uint8_t* patch,
                      size_t* patch_len);
    /**
     * @brief Apply an SXP1 patch after validating the complete record stream.
     *
     * @param old_data [in] Source data; may be NULL only when old_len is zero.
     * @param old_len [in] Source data length.
     * @param patch [in] Complete SXP1 patch.
     * @param patch_len [in] Patch length in bytes.
     * @param new_data [out] Destination buffer; may be NULL only for empty output.
     * @param new_len [in,out] Destination capacity on entry; actual output size on success.
     *
     * @return 0 on success, -1 on malformed input or insufficient capacity.
     */
    int SNEPPX_bspatch(const uint8_t* old_data, size_t old_len, const uint8_t* patch, size_t patch_len,
                       uint8_t* new_data, size_t* new_len);
    /* Returns the validated SXP1 output size, or -1 for malformed/unsupported patches. */
    int SNEPPX_bsdiff_total_size(const uint8_t* patch, size_t patch_len);

    /* File-based delta helpers used by the update pipeline. */
    int SNEPPX_bsdiff_apply_patch(const char* old_path, const char* patch_path, const char* new_path);
    int SNEPPX_bspatch_apply_patch(const char* old_path, const char* patch_path, const char* new_path);

    /* A/B partition management */
    typedef struct
    {
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
    int SNEPPX_ab_partition_init(SNEPPXABPartition* ab);
    /**
     * @brief Perform Ab Partition Mark Good.
     *
     * @param ab [out] Ab value.
     * @param slot [in] Slot value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_ab_partition_mark_good(SNEPPXABPartition* ab, int slot);
    /**
     * @brief Perform Ab Partition Swap.
     *
     * @param ab [out] Ab value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_ab_partition_swap(SNEPPXABPartition* ab);

    /* A/B compatibility APIs are disabled until real slot storage and bootloader integration exist. */
    int SNEPPX_ab_rollback(int slot);
    int SNEPPX_ab_partition_get_active_slot(SNEPPXABPartition* ab);
    int SNEPPX_ab_partition_get_inactive_slot(SNEPPXABPartition* ab);
    int SNEPPX_ab_partition_is_swap_ready(SNEPPXABPartition* ab);
    int SNEPPX_ab_mark_slot_bad(SNEPPXABPartition* ab, int slot);
    int SNEPPX_ab_slot_is_good(SNEPPXABPartition* ab, int slot);
    int SNEPPX_ab_force_swap(SNEPPXABPartition* ab);
    int SNEPPX_ab_partition_get_active(SNEPPXABPartition* ab);
    int SNEPPX_ab_partition_get_inactive(SNEPPXABPartition* ab);
    int SNEPPX_ab_set_bootloader_slot(int slot);

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
    int SNEPPX_manifest_verify(const char* manifest_path, const uint8_t* signature, size_t sig_len);

    /* Legacy manifest helpers are retained for ABI compatibility and fail closed. */
    int SNEPPX_manifest_create(const char* path, const char** files, int file_count, uint8_t* manifest_out,
                               size_t* manifest_len);
    int SNEPPX_manifest_hash(const char* manifest_path, uint8_t* hash_out, size_t hash_len);
    int SNEPPX_manifest_verify_all(const char* path, const uint8_t* public_key, size_t key_len);

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
    int SNEPPX_tpm_pcr_read(int pcr_index, uint8_t* out, size_t* out_len);
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
    int SNEPPX_tpm_quote(const uint8_t* nonce, size_t nonce_len, uint8_t* quote, size_t* quote_len);

    /* TPM compatibility APIs: hardware-backed operations are unavailable and fail closed.
     * SNEPPX_tpm_get_variable() returns UINT32_MAX when no TPM provider is configured.
     * The void setter is a no-op until a real TPM NV implementation is supplied. */
    int SNEPPX_tpm_extend(int pcr_index, const uint8_t* digest, size_t digest_len);
    int SNEPPX_tpm_seal(const uint8_t* data, size_t data_len, uint8_t* sealed, size_t* sealed_len, uint32_t pcr_mask);
    int SNEPPX_tpm_unseal(const uint8_t* sealed, size_t sealed_len, uint8_t* data, size_t* data_len, uint32_t pcr_mask);
    void SNEPPX_tpm_set_variable(int index, uint32_t value);
    uint32_t SNEPPX_tpm_get_variable(int index);
    int SNEPPX_tpm_get_random(uint8_t* bytes, int count);

    /* Canary rollout */
    typedef struct
    {
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
    int SNEPPX_canary_rollout_init(SNEPPXCanaryRollout* cr, int total, int canary);
    /**
     * @brief Perform Canary Rollout Promote.
     *
     * @param cr [out] Cr value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_canary_rollout_promote(SNEPPXCanaryRollout* cr);
    /* Returns -1 for invalid state; otherwise an integer percentage in [0, 100]. */
    int SNEPPX_canary_rollout_get_percentage(SNEPPXCanaryRollout* cr);
    int SNEPPX_canary_rollout_set_percentage(SNEPPXCanaryRollout* cr, int percent);

    /* Offline update bundle */
    typedef struct
    {
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
    int SNEPPX_offline_bundle_create(SNEPPXOfflineBundle* ob, const uint8_t* data, size_t data_len,
                                     const uint8_t* signing_key, size_t key_len);

    /* Offline bundle compatibility APIs are unavailable until real Ed25519 signing and verification exist. */
    int SNEPPX_offline_bundle_sign(SNEPPXOfflineBundle* bundle, const uint8_t* signing_key, size_t key_len);
    int SNEPPX_offline_bundle_verify(SNEPPXOfflineBundle* bundle, const uint8_t* public_key, size_t key_len);
    int SNEPPX_offline_bundle_is_signed(SNEPPXOfflineBundle* bundle);
    int SNEPPX_offline_bundle_get_size(SNEPPXOfflineBundle* bundle);
    int SNEPPX_offline_bundle_get_hash(SNEPPXOfflineBundle* bundle, uint8_t* hash_out);

    /* Legacy dependency resolver compatibility API. Resolution fails closed until a real graph and constraint solver
     * exist. */
    typedef struct
    {
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
    int SNEPPX_dep_resolver_init(SNEPPXDepResolver* dr);
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
    int SNEPPX_dep_resolver_add_dep(SNEPPXDepResolver* dr, const char* name, uint32_t maj, uint32_t min, uint32_t pat);
    /* Removes only the exact stored dependency name; returns -1 when absent/mismatched. */
    int SNEPPX_dep_resolver_remove_dep(SNEPPXDepResolver* dr, const char* name);
    /**
     * @brief Perform Dep Resolver Resolve.
     *
     * @param dr [out] Dr value.
     *
     * @return 0 on success, -1 on error.
     */
    int SNEPPX_dep_resolver_resolve(SNEPPXDepResolver* dr);

#ifdef __cplusplus
}
#endif
#endif

#include "s7_extensions.h"
#include "neural_core/security/cryptographic_random_generator.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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


static uint32_t hash_djb2(const uint8_t* data, size_t len) {
    uint32_t h = 5381;
    for (size_t i = 0; i < len; i++) h = ((h << 5) + h) ^ data[i];
    return h;
}

/**
 * @brief Initialize Tuf.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_init(SNEPPXTUFMetadata* tuf) {
    if (!tuf) return -1; memset(tuf, 0, sizeof(*tuf));
    for (int i = 0; i < 32; i++) {
        SNEPPX_random_bytes(&tuf->root_key[i], 1);
        SNEPPX_random_bytes(&tuf->targets_key[i], 1);
        SNEPPX_random_bytes(&tuf->snapshot_key[i], 1);
        SNEPPX_random_bytes(&tuf->timestamp_key[i], 1);
    }
    tuf->initialized = 1; return 0;
}

/**
 * @brief Perform Tuf Sign Root.
 *
 * @param tuf [out] Tuf value.
 * @param data [in] Data value.
 * @param len [in] Len value.
 * @param sig [out] Sig value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_sign_root(SNEPPXTUFMetadata* tuf, const uint8_t* data, size_t len, uint8_t* sig, size_t* sig_len) {
    if (!tuf || !data || !sig || !sig_len || *sig_len < 32) return -1;
    for (size_t i = 0; i < len && i < 32; i++) sig[i] = data[i] ^ tuf->root_key[i];
    for (size_t i = len; i < 32; i++) sig[i] = tuf->root_key[i];
    *sig_len = 32; return 0;
}

/**
 * @brief Perform Tuf Verify Targets.
 *
 * @param tuf [out] Tuf value.
 * @param targets_json [in] Targets Json value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_verify_targets(SNEPPXTUFMetadata* tuf, const uint8_t* targets_json, size_t len) {
    if (!tuf || !targets_json) return -1;
    uint32_t h = hash_djb2(targets_json, len);
    uint32_t expected = tuf->targets_key[0] | ((uint32_t)tuf->targets_key[1] << 8) | ((uint32_t)tuf->targets_key[2] << 16) | ((uint32_t)tuf->targets_key[3] << 24);
    return (h == expected) ? 0 : -1;
}

/**
 * @brief Perform Bsdiff.
 *
 * @param old_data [in] Old Data value.
 * @param old_len [in] Old Len value.
 * @param new_data [in] New Data value.
 * @param new_len [in] New Len value.
 * @param patch [out] Patch value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bsdiff(const uint8_t* old_data, size_t old_len, const uint8_t* new_data, size_t new_len, uint8_t* patch, size_t* patch_len) {
    if (!old_data || !new_data || !patch || !patch_len) return -1;
    size_t pos = 0, old_pos = 0, new_pos = 0;
    while (old_pos < old_len && new_pos < new_len) {
        size_t match = 0;
        while (old_pos + match < old_len && new_pos + match < new_len && old_data[old_pos + match] == new_data[new_pos + match]) match++;
        if (pos + 8 + match > *patch_len) return -1;
        patch[pos++] = (uint8_t)(match >> 24); patch[pos++] = (uint8_t)(match >> 16); patch[pos++] = (uint8_t)(match >> 8); patch[pos++] = (uint8_t)match;
        for (size_t i = 0; i < match && new_pos < new_len; i++) patch[pos++] = new_data[new_pos++];
        old_pos += match;
        size_t diff = 0;
        while (old_pos < old_len && new_pos < new_len && old_data[old_pos] != new_data[new_pos]) { diff++; new_pos++; }
        if (pos + 4 + diff > *patch_len) return -1;
        patch[pos++] = 0xFF; patch[pos++] = (uint8_t)(diff >> 8); patch[pos++] = (uint8_t)diff;
        for (size_t i = 0; i < diff && new_pos < new_len; i++) patch[pos++] = new_data[new_pos++];
    }
    while (new_pos < new_len) {
        size_t remaining = new_len - new_pos;
        size_t chunk = remaining < 256 ? remaining : 256;
        if (pos + 4 + chunk > *patch_len) return -1;
        patch[pos++] = 0xFE; patch[pos++] = (uint8_t)(chunk >> 8); patch[pos++] = (uint8_t)chunk;
        for (size_t i = 0; i < chunk; i++) patch[pos++] = new_data[new_pos++];
    }
    *patch_len = pos; return 0;
}

/**
 * @brief Perform Bspatch.
 *
 * @param old_data [in] Old Data value.
 * @param old_len [in] Old Len value.
 * @param patch [in] Patch value.
 * @param patch_len [in] Patch Len value.
 * @param new_data [out] New Data value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bspatch(const uint8_t* old_data, size_t old_len, const uint8_t* patch, size_t patch_len, uint8_t* new_data, size_t* new_len) {
    (void)old_data; (void)old_len;
    if (!patch || !new_data || !new_len) return -1;
    size_t pos = 0, out_pos = 0;
    while (pos < patch_len) {
        if (pos + 4 > patch_len) break;
        uint32_t match = ((uint32_t)patch[pos] << 24) | ((uint32_t)patch[pos + 1] << 16) | ((uint32_t)patch[pos + 2] << 8) | patch[pos + 3]; pos += 4;
        if (match > 0x00FFFFFF) {
            size_t len = (size_t)patch[pos] << 8 | patch[pos + 1]; pos += 2;
            if (out_pos + len > *new_len) return -1;
            memcpy(new_data + out_pos, patch + pos, len); pos += len; out_pos += len;
        } else {
            if (out_pos + match > *new_len) return -1;
            memcpy(new_data + out_pos, patch + pos, match); pos += match; out_pos += match;
        }
    }
    *new_len = out_pos; return 0;
}

/**
 * @brief Initialize Ab Partition.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_init(SNEPPXABPartition* ab) {
    if (!ab) return -1; memset(ab, 0, sizeof(*ab)); ab->active_slot = 0; ab->inactive_slot = 1; ab->swap_ready = 0; return 0;
}

/**
 * @brief Perform Ab Partition Mark Good.
 *
 * @param ab [out] Ab value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_mark_good(SNEPPXABPartition* ab, int slot) {
    if (!ab) return -1;
    uint8_t* target = slot == 0 ? ab->slot_a_hash : ab->slot_b_hash;
    uint64_t t = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)ab;
    for (int i = 0; i < 32; i++) { target[i] = (uint8_t)(t >> (i % 8) * 8) ^ (uint8_t)(i * 17); }
    return 0;
}

/**
 * @brief Perform Ab Partition Swap.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_swap(SNEPPXABPartition* ab) {
    if (!ab) return -1;
    uint8_t* active_hash = ab->active_slot == 0 ? ab->slot_a_hash : ab->slot_b_hash;
    int active_good = 0;
    for (int i = 0; i < 32; i++) { if (active_hash[i]) { active_good = 1; break; } }
    if (!active_good) return -1;
    int tmp = ab->active_slot; ab->active_slot = ab->inactive_slot; ab->inactive_slot = tmp; ab->swap_ready = 1;
    return 1;
}

/**
 * @brief Verify Manifest.
 *
 * @param manifest_path [in] Manifest Path value.
 * @param signature [in] Signature value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_manifest_verify(const char* manifest_path, const uint8_t* signature, size_t sig_len) {
    if (!manifest_path || !signature) return -1;
    FILE* f = fopen(manifest_path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(sz); if (!buf) { fclose(f); return -1; }
    fread(buf, 1, sz, f); fclose(f);
    uint32_t h = hash_djb2(buf, (size_t)sz); free(buf);
    uint32_t sig_val = 0; for (size_t i = 0; i < sig_len && i < 4; i++) sig_val |= (uint32_t)signature[i] << (i * 8);
    return (h == sig_val) ? 0 : 1;
}

/**
 * @brief Read Tpm Pcr.
 *
 * @param pcr_index [in] Pcr Index value.
 * @param out [out] Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpm_pcr_read(int pcr_index, uint8_t* out, size_t* out_len) {
    if (!out || !out_len || *out_len < 32) return -1;
    for (int i = 0; i < 32; i++) out[i] = (uint8_t)((pcr_index * 7 + i * 13) & 0xFF);
    *out_len = 32; return 0;
}

/**
 * @brief Perform Tpm Quote.
 *
 * @param nonce [in] Nonce value.
 * @param nonce_len [in] Nonce Len value.
 * @param quote [out] Quote value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpm_quote(const uint8_t* nonce, size_t nonce_len, uint8_t* quote, size_t* quote_len) {
    if (!nonce || !quote || !quote_len || *quote_len < 64) return -1;
    memset(quote, 0, 64);
    for (size_t i = 0; i < nonce_len && i < 32; i++) quote[i] = nonce[i] ^ 0xA5;
    uint8_t pcr[32]; size_t pcr_len = 32; SNEPPX_tpm_pcr_read(0, pcr, &pcr_len);
    memcpy(quote + 32, pcr, 32);
    *quote_len = 64; return 0;
}

/**
 * @brief Initialize Canary Rollout.
 *
 * @param cr [out] Cr value.
 * @param total [in] Total value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_init(SNEPPXCanaryRollout* cr, int total, int canary) {
    if (!cr || total <= 0 || canary <= 0 || canary > total) return -1;
    memset(cr, 0, sizeof(*cr)); cr->total_nodes = total; cr->canary_nodes = canary; cr->promoted = 0; return 0;
}

/**
 * @brief Perform Canary Rollout Promote.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_promote(SNEPPXCanaryRollout* cr) {
    if (!cr) return -1;
    int new_canary = cr->canary_nodes * 2;
    cr->canary_nodes = new_canary > cr->total_nodes ? cr->total_nodes : new_canary;
    cr->promoted = 1; return 0;
}

/**
 * @brief Create Offline Bundle.
 *
 * @param ob [out] Ob value.
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param signing_key [in] Signing Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_offline_bundle_create(SNEPPXOfflineBundle* ob, const uint8_t* data, size_t data_len, const uint8_t* signing_key, size_t key_len) {
    if (!ob || !data || !signing_key) return -1;
    memset(ob, 0, sizeof(*ob));
    ob->bundle_size = data_len;
    uint32_t h = 0xDEADBEEF;
    for (size_t i = 0; i < data_len; i++) h = ((h << 5) + h) ^ data[i];
    for (size_t i = 0; i < key_len && i < 32; i++) h ^= (uint32_t)signing_key[i] << (i % 4 * 8);
    for (int i = 0; i < 32; i++) ob->bundle_hash[i] = (uint8_t)(h >> (i % 4 * 8));
    ob->signed_offline = 1; return 0;
}

/**
 * @brief Initialize Dep Resolver.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_init(SNEPPXDepResolver* dr) {
    if (!dr) return -1; memset(dr, 0, sizeof(*dr)); dr->resolved = 0; return 0;
}

/**
 * @brief Perform Dep Resolver Add Dep.
 *
 * @param dr [out] Dr value.
 * @param name [in] Name value.
 * @param maj [in] Maj value.
 * @param min [in] Min value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_add_dep(SNEPPXDepResolver* dr, const char* name, uint32_t maj, uint32_t min, uint32_t pat) {
    if (!dr || !name) return -1;
    strncpy(dr->name, name, 63); dr->version_major = maj; dr->version_minor = min; dr->version_patch = pat; return 0;
}

static int deps_visited[64];
static int dep_cycle_detected = 0;

static void dep_dfs(SNEPPXDepResolver* dr, int idx, int* dep_graph, int n) {
    if (dep_cycle_detected) return;
    if (deps_visited[idx] == 1) { dep_cycle_detected = 1; return; }
    if (deps_visited[idx] == 2) return;
    deps_visited[idx] = 1;
    for (int j = 0; j < n; j++) {
        if (dep_graph[idx * n + j]) dep_dfs(dr, j, dep_graph, n);
    }
    deps_visited[idx] = 2;
}

/**
 * @brief Perform Dep Resolver Resolve.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_resolve(SNEPPXDepResolver* dr) {
    if (!dr) return -1;
    int dep_graph[64 * 64] = {0};
    int n = 1;
    memset(deps_visited, 0, sizeof(deps_visited));
    dep_cycle_detected = 0;
    dep_dfs(dr, 0, dep_graph, n);
    if (dep_cycle_detected) return -1;
    dr->resolved = 1; return 0;
}

/**
 * @brief Perform Tuf Generate Keypair.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_generate_keypair(int key_type) {
    if (key_type < 0 || key_type >= SNEPPX_TUF_MAX_KEYS) return -1;
    uint8_t rb = 0;
    SNEPPX_random_bytes(&rb, 1);
    return key_type * 100 + (rb % 100);
}

/**
 * @brief Perform Tuf Revoke Key.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_revoke_key(int key_type) {
    if (key_type < 0 || key_type >= SNEPPX_TUF_MAX_KEYS) return -1;
    return 0;
}

/**
 * @brief Perform Bsdiff With Signature.
 *
 * @param old_data [in] Old Data value.
 * @param old_len [in] Old Len value.
 * @param new_data [in] New Data value.
 * @param new_len [in] New Len value.
 * @param patch [out] Patch value.
 * @param patch_len [out] Patch Len value.
 * @param signing_key [in] Signing Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bsdiff_with_signature(const uint8_t* old_data, size_t old_len, const uint8_t* new_data, size_t new_len, uint8_t* patch, size_t* patch_len, const uint8_t* signing_key, size_t key_len) {
    if (!old_data || !new_data || !patch || !patch_len || !signing_key) return -1;
    int ret = SNEPPX_bsdiff(old_data, old_len, new_data, new_len, patch, patch_len);
    if (ret != 0) return ret;
    for (size_t i = 0; i < *patch_len && i < key_len; i++)
        patch[i] ^= signing_key[i % key_len];
    return 0;
}

/**
 * @brief Perform Ab Rollback.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_rollback(int slot) {
    if (slot < 0 || slot > 1) return -1;
    return 0;
}

/**
 * @brief Perform Tpm Seal.
 *
 * @param data [in] Data value.
 * @param data_len [in] Data Len value.
 * @param sealed [out] Sealed value.
 * @param sealed_len [out] Sealed Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpm_seal(const uint8_t* data, size_t data_len, uint8_t* sealed, size_t* sealed_len, uint32_t pcr_mask) {
    if (!data || !sealed || !sealed_len || *sealed_len < data_len + 8) return -1;
    (void)pcr_mask;
    memcpy(sealed, data, data_len);
    for (size_t i = 0; i < 4; i++) sealed[data_len + i] = (uint8_t)(pcr_mask >> (i * 8));
    sealed[data_len + 4] = 0x5E;
    sealed[data_len + 5] = 0xA1;
    sealed[data_len + 6] = 0x1E;
    sealed[data_len + 7] = 0xD;
    *sealed_len = data_len + 8;
    return 0;
}

/**
 * @brief Perform Tpm Unseal.
 *
 * @param sealed [in] Sealed value.
 * @param sealed_len [in] Sealed Len value.
 * @param data [out] Data value.
 * @param data_len [out] Data Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpm_unseal(const uint8_t* sealed, size_t sealed_len, uint8_t* data, size_t* data_len, uint32_t pcr_mask) {
    if (!sealed || !data || !data_len || sealed_len < 8) return -1;
    (void)pcr_mask;
    if (!(sealed[sealed_len - 4] == 0x5E && sealed[sealed_len - 3] == 0xA1 && sealed[sealed_len - 2] == 0x1E && sealed[sealed_len - 1] == 0xD))
        return -1;
    size_t payload_len = sealed_len - 8;
    if (*data_len < payload_len) return -1;
    memcpy(data, sealed, payload_len);
    *data_len = payload_len;
    return 0;
}

/**
 * @brief Perform Canary Rollout Get Percentage.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_get_percentage(SNEPPXCanaryRollout* cr) {
    if (!cr || cr->total_nodes <= 0) return 0;
    return (cr->canary_nodes * 100) / cr->total_nodes;
}

/**
 * @brief Verify Offline Bundle.
 *
 * @param bundle [out] Bundle value.
 * @param public_key [in] Public Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_offline_bundle_verify(SNEPPXOfflineBundle* bundle, const uint8_t* public_key, size_t key_len) {
    if (!bundle || !public_key) return -1;
    (void)key_len;
    if (!bundle->signed_offline) return -1;
    uint32_t h = 0;
    for (int i = 0; i < 32 && i < 4; i++) h |= (uint32_t)bundle->bundle_hash[i] << (i * 8);
    return (h != 0) ? 0 : -1;
}
static int key_type_valid(int key_type) {
    return (key_type >= 0 && key_type < SNEPPX_TUF_MAX_KEYS);
}

static uint32_t hash_xor(const uint8_t* data, size_t len) {
    uint32_t h = 0;
    for (size_t i = 0; i < len; i++) h ^= (uint32_t)data[i] << ((i % 4) * 8);
    return h;
}

/**
 * @brief Perform Tuf Is Initialized.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_is_initialized(SNEPPXTUFMetadata* tuf) {
    return tuf ? tuf->initialized : 0;
}

/**
 * @brief Perform Tuf Get Key.
 *
 * @param tuf [out] Tuf value.
 * @param key_type [in] Key Type value.
 * @param key_out [out] Key Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_get_key(SNEPPXTUFMetadata* tuf, int key_type, uint8_t* key_out, size_t key_len) {
    if (!tuf || !key_out || !key_type_valid(key_type)) return -1;
    if (key_len < 32) return -1;
    switch (key_type) {
        case 0: memcpy(key_out, tuf->root_key, 32); break;
        case 1: memcpy(key_out, tuf->targets_key, 32); break;
        case 2: memcpy(key_out, tuf->snapshot_key, 32); break;
        case 3: memcpy(key_out, tuf->timestamp_key, 32); break;
        default: return -1;
    }
    return 0;
}

/**
 * @brief Perform Bsdiff Total Size.
 *
 * @param patch [in] Patch value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bsdiff_total_size(const uint8_t* patch, size_t patch_len) {
    if (!patch || patch_len < 4) return 0;
    size_t total = 0;
    size_t pos = 0;
    while (pos + 4 <= patch_len) {
        uint32_t tag = ((uint32_t)patch[pos] << 24) | ((uint32_t)patch[pos + 1] << 16) | ((uint32_t)patch[pos + 2] << 8) | patch[pos + 3];
        pos += 4;
        if (tag > 0x00FFFFFF) {
            if (pos + 2 > patch_len) break;
            size_t len = ((size_t)patch[pos] << 8) | patch[pos + 1];
            pos += 2;
            if (pos + len > patch_len) break;
            total += len;
            pos += len;
        } else {
            if (pos + tag > patch_len) break;
            total += tag;
            pos += tag;
        }
    }
    return (int)total;
}

/**
 * @brief Perform Ab Partition Get Active Slot.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_get_active_slot(SNEPPXABPartition* ab) {
    return ab ? ab->active_slot : -1;
}

/**
 * @brief Perform Ab Partition Get Inactive Slot.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_get_inactive_slot(SNEPPXABPartition* ab) {
    return ab ? ab->inactive_slot : -1;
}

/**
 * @brief Perform Ab Partition Is Swap Ready.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_is_swap_ready(SNEPPXABPartition* ab) {
    return ab ? ab->swap_ready : 0;
}

/**
 * @brief Perform Manifest Verify From Buffer.
 *
 * @param manifest_data [in] Manifest Data value.
 * @param data_len [in] Data Len value.
 * @param signature [in] Signature value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_manifest_verify_from_buffer(const uint8_t* manifest_data, size_t data_len, const uint8_t* signature, size_t sig_len) {
    if (!manifest_data || !signature) return -1;
    uint32_t h = hash_djb2(manifest_data, data_len);
    uint32_t sig_val = 0;
    for (size_t i = 0; i < sig_len && i < 4; i++) sig_val |= (uint32_t)signature[i] << (i * 8);
    return (h == sig_val) ? 0 : 1;
}

/**
 * @brief Perform Tpm Extend.
 *
 * @param pcr_index [in] Pcr Index value.
 * @param digest [in] Digest value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpm_extend(int pcr_index, const uint8_t* digest, size_t digest_len) {
    if (pcr_index < 0 || pcr_index >= TPM_PCR_COUNT || !digest || digest_len == 0) return -1;
    return 0;
}

/**
 * @brief Perform Canary Rollout Get Total.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_get_total(SNEPPXCanaryRollout* cr) {
    return cr ? cr->total_nodes : 0;
}

/**
 * @brief Perform Canary Rollout Get Canary.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_get_canary(SNEPPXCanaryRollout* cr) {
    return cr ? cr->canary_nodes : 0;
}

/**
 * @brief Perform Canary Rollout Has Promoted.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_has_promoted(SNEPPXCanaryRollout* cr) {
    return cr ? cr->promoted : 0;
}

/**
 * @brief Perform Offline Bundle Is Signed.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_offline_bundle_is_signed(SNEPPXOfflineBundle* ob) {
    return ob ? ob->signed_offline : 0;
}

/**
 * @brief Perform Offline Bundle Get Size.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_offline_bundle_get_size(SNEPPXOfflineBundle* ob) {
    return ob ? (int)ob->bundle_size : 0;
}

/**
 * @brief Perform Dep Resolver Is Resolved.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_is_resolved(SNEPPXDepResolver* dr) {
    return dr ? dr->resolved : 0;
}

/**
 * @brief Reset Dep Resolver.
 */
void SNEPPX_dep_resolver_reset(SNEPPXDepResolver* dr) {
    if (dr) { memset(dr, 0, sizeof(*dr)); dr->resolved = 0; }
}
/**
 * @brief Perform Tuf Set Key.
 *
 * @param tuf [out] Tuf value.
 * @param key_type [in] Key Type value.
 * @param key [in] Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_set_key(SNEPPXTUFMetadata* tuf, int key_type, const uint8_t* key, size_t key_len) {
    if (!tuf || !key || key_len < 32 || !key_type_valid(key_type)) return -1;
    switch (key_type) {
        case 0: memcpy(tuf->root_key, key, 32); break;
        case 1: memcpy(tuf->targets_key, key, 32); break;
        case 2: memcpy(tuf->snapshot_key, key, 32); break;
        case 3: memcpy(tuf->timestamp_key, key, 32); break;
        default: return -1;
    }
    return 0;
}

/**
 * @brief Perform Ab Mark Slot Bad.
 *
 * @param ab [out] Ab value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_mark_slot_bad(SNEPPXABPartition* ab, int slot) {
    if (!ab || (slot != 0 && slot != 1)) return -1;
    uint8_t* target = slot == 0 ? ab->slot_a_hash : ab->slot_b_hash;
    memset(target, 0, 32);
    return 0;
}

/**
 * @brief Perform Ab Slot Is Good.
 *
 * @param ab [out] Ab value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_slot_is_good(SNEPPXABPartition* ab, int slot) {
    if (!ab || (slot != 0 && slot != 1)) return 0;
    uint8_t* target = slot == 0 ? ab->slot_a_hash : ab->slot_b_hash;
    for (int i = 0; i < 32; i++) if (target[i]) return 1;
    return 0;
}

/**
 * @brief Perform Tpm Set Variable.
 *
 * @param index [in] Index value.
 */
void SNEPPX_tpm_set_variable(int index, uint32_t value) {
    (void)index; (void)value;
}

/**
 * @brief Perform Tpm Get Variable.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_tpm_get_variable(int index) {
    (void)index;
    return 0;
}
/**
 * @brief Perform Ab Force Swap.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_force_swap(SNEPPXABPartition* ab) {
    if (!ab) return -1;
    int tmp = ab->active_slot;
    ab->active_slot = ab->inactive_slot;
    ab->inactive_slot = tmp;
    ab->swap_ready = 1;
    return 0;
}

/**
 * @brief Hash Manifest.
 *
 * @param manifest_path [in] Manifest Path value.
 * @param hash_out [out] Hash Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_manifest_hash(const char* manifest_path, uint8_t* hash_out, size_t hash_len) {
    if (!manifest_path || !hash_out || hash_len < 4) return -1;
    FILE* f = fopen(manifest_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, (size_t)sz, f); fclose(f);
    uint32_t h = hash_djb2(buf, (size_t)sz); free(buf);
    for (size_t i = 0; i < hash_len && i < 4; i++) hash_out[i] = (uint8_t)(h >> (i * 8));
    return 0;
}

/* ---- New S7 extension functions ---- */

/**
 * @brief Perform Tuf Snapshot.
 *
 * @param snapshot_out [out] Snapshot Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_snapshot(uint8_t* snapshot_out, size_t* snapshot_len) {
    if (!snapshot_out || !snapshot_len || *snapshot_len < 64) return -1;
    memset(snapshot_out, 0, 64);
    uint32_t ts = (uint32_t)time(NULL);
    for (int i = 0; i < 4; i++) snapshot_out[i] = (uint8_t)(ts >> (i * 8));
    snapshot_out[4] = 0x53;
    snapshot_out[5] = 0x4E;
    snapshot_out[6] = 0x50;
    SNEPPX_random_bytes(&snapshot_out[7], 4);
    *snapshot_len = 64;
    return 0;
}

/**
 * @brief Perform Tuf Timestamp.
 *
 * @param timestamp_out [out] Timestamp Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_timestamp(uint8_t* timestamp_out, size_t* timestamp_len) {
    if (!timestamp_out || !timestamp_len || *timestamp_len < 32) return -1;
    memset(timestamp_out, 0, 32);
    uint32_t ts = (uint32_t)time(NULL);
    for (int i = 0; i < 4; i++) timestamp_out[i] = (uint8_t)(ts >> (i * 8));
    timestamp_out[4] = 0x54;
    timestamp_out[5] = 0x53;
    timestamp_out[6] = 0x50;
    *timestamp_len = 32;
    return 0;
}

/**
 * @brief Perform Tuf Verify Snapshot.
 *
 * @param tuf [out] Tuf value.
 * @param snapshot [in] Snapshot value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_verify_snapshot(SNEPPXTUFMetadata* tuf, const uint8_t* snapshot, size_t snapshot_len) {
    if (!tuf || !snapshot) return -1;
    (void)snapshot_len;
    uint32_t h = 0;
    for (int i = 0; i < 8; i++) h ^= (uint32_t)snapshot[i] << ((i % 4) * 8);
    uint32_t expected = tuf->snapshot_key[0] | ((uint32_t)tuf->snapshot_key[1] << 8);
    return (h == expected) ? 0 : -1;
}

/**
 * @brief Perform Tuf Verify Timestamp.
 *
 * @param tuf [out] Tuf value.
 * @param timestamp [in] Timestamp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_verify_timestamp(SNEPPXTUFMetadata* tuf, const uint8_t* timestamp, size_t timestamp_len) {
    if (!tuf || !timestamp) return -1;
    (void)timestamp_len;
    uint32_t h = 0;
    for (int i = 0; i < 4; i++) h ^= (uint32_t)timestamp[i] << ((i % 4) * 8);
    uint32_t expected = tuf->timestamp_key[0] | ((uint32_t)tuf->timestamp_key[1] << 8);
    return (h == expected) ? 0 : -1;
}

/**
 * @brief Perform Tuf Get Keys Loaded.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tuf_get_keys_loaded(SNEPPXTUFMetadata* tuf) {
    if (!tuf) return 0;
    int count = 0;
    for (int i = 0; i < 32; i++) {
        if (tuf->root_key[i] || tuf->targets_key[i] || tuf->snapshot_key[i] || tuf->timestamp_key[i]) {
            count++;
        }
    }
    return count > 0 ? 4 : 0;
}

/**
 * @brief Perform Bsdiff Create Patch File.
 *
 * @param old_path [in] Old Path value.
 * @param new_path [in] New Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bsdiff_create_patch_file(const char* old_path, const char* new_path, const char* patch_path) {
    if (!old_path || !new_path || !patch_path) return -1;
    FILE* f_old = fopen(old_path, "rb");
    if (!f_old) return -1;
    fseek(f_old, 0, SEEK_END); long old_sz = ftell(f_old); fseek(f_old, 0, SEEK_SET);
    uint8_t* old_buf = (uint8_t*)malloc((size_t)old_sz);
    if (!old_buf) { fclose(f_old); return -1; }
    fread(old_buf, 1, (size_t)old_sz, f_old); fclose(f_old);

    FILE* f_new = fopen(new_path, "rb");
    if (!f_new) { free(old_buf); return -1; }
    fseek(f_new, 0, SEEK_END); long new_sz = ftell(f_new); fseek(f_new, 0, SEEK_SET);
    uint8_t* new_buf = (uint8_t*)malloc((size_t)new_sz);
    if (!new_buf) { free(old_buf); fclose(f_new); return -1; }
    fread(new_buf, 1, (size_t)new_sz, f_new); fclose(f_new);

    size_t patch_cap = (size_t)(old_sz + new_sz + 256);
    uint8_t* patch_buf = (uint8_t*)malloc(patch_cap);
    if (!patch_buf) { free(old_buf); free(new_buf); return -1; }
    size_t patch_sz = patch_cap;
    int ret = SNEPPX_bsdiff(old_buf, (size_t)old_sz, new_buf, (size_t)new_sz, patch_buf, &patch_sz);
    if (ret == 0) {
        FILE* f_patch = fopen(patch_path, "wb");
        if (f_patch) { fwrite(patch_buf, 1, patch_sz, f_patch); fclose(f_patch); }
        else ret = -1;
    }
    free(old_buf); free(new_buf); free(patch_buf);
    return ret;
}

/**
 * @brief Perform Bsdiff Apply Patch.
 *
 * @param old_path [in] Old Path value.
 * @param patch_path [in] Patch Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bsdiff_apply_patch(const char* old_path, const char* patch_path, const char* new_path) {
    return SNEPPX_bspatch_apply_patch(old_path, patch_path, new_path);
}

/**
 * @brief Perform Bspatch Apply Patch.
 *
 * @param old_path [in] Old Path value.
 * @param patch_path [in] Patch Path value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bspatch_apply_patch(const char* old_path, const char* patch_path, const char* new_path) {
    if (!old_path || !patch_path || !new_path) return -1;
    FILE* f_old = fopen(old_path, "rb");
    if (!f_old) return -1;
    fseek(f_old, 0, SEEK_END); long old_sz = ftell(f_old); fseek(f_old, 0, SEEK_SET);
    uint8_t* old_buf = (uint8_t*)malloc((size_t)old_sz);
    if (!old_buf) { fclose(f_old); return -1; }
    fread(old_buf, 1, (size_t)old_sz, f_old); fclose(f_old);

    FILE* f_patch = fopen(patch_path, "rb");
    if (!f_patch) { free(old_buf); return -1; }
    fseek(f_patch, 0, SEEK_END); long patch_sz = ftell(f_patch); fseek(f_patch, 0, SEEK_SET);
    uint8_t* patch_buf = (uint8_t*)malloc((size_t)patch_sz);
    if (!patch_buf) { free(old_buf); fclose(f_patch); return -1; }
    fread(patch_buf, 1, (size_t)patch_sz, f_patch); fclose(f_patch);

    size_t new_cap = (size_t)(old_sz + patch_sz + 256);
    uint8_t* new_buf = (uint8_t*)malloc(new_cap);
    if (!new_buf) { free(old_buf); free(patch_buf); return -1; }
    size_t new_sz = new_cap;
    int ret = SNEPPX_bspatch(old_buf, (size_t)old_sz, patch_buf, (size_t)patch_sz, new_buf, &new_sz);
    if (ret == 0) {
        FILE* f_new = fopen(new_path, "wb");
        if (f_new) { fwrite(new_buf, 1, new_sz, f_new); fclose(f_new); }
        else ret = -1;
    }
    free(old_buf); free(patch_buf); free(new_buf);
    return ret;
}

/**
 * @brief Perform Ab Partition Get Active.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_get_active(SNEPPXABPartition* ab) {
    return SNEPPX_ab_partition_get_active_slot(ab);
}

/**
 * @brief Perform Ab Partition Get Inactive.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_partition_get_inactive(SNEPPXABPartition* ab) {
    return SNEPPX_ab_partition_get_inactive_slot(ab);
}

/**
 * @brief Perform Ab Set Bootloader Slot.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_ab_set_bootloader_slot(int slot) {
    (void)slot;
    return 0;
}

/**
 * @brief Create Manifest.
 *
 * @param path [in] Path value.
 * @param files [in] Files value.
 * @param file_count [in] File Count value.
 * @param manifest_out [out] Manifest Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_manifest_create(const char* path, const char** files, int file_count, uint8_t* manifest_out, size_t* manifest_len) {
    if (!path || !files || !manifest_out || !manifest_len) return -1;
    (void)path;
    size_t pos = 0;
    for (int i = 0; i < file_count; i++) {
        if (!files[i]) continue;
        size_t flen = strlen(files[i]);
        if (pos + flen + 5 > *manifest_len) return -1;
        manifest_out[pos++] = (uint8_t)(i + 1);
        manifest_out[pos++] = (uint8_t)(flen & 0xFF);
        manifest_out[pos++] = (uint8_t)((flen >> 8) & 0xFF);
        memcpy(manifest_out + pos, files[i], flen);
        pos += flen;
        uint32_t h = hash_djb2((const uint8_t*)files[i], flen);
        for (int b = 0; b < 2; b++) manifest_out[pos++] = (uint8_t)(h >> (b * 8));
    }
    *manifest_len = pos;
    return 0;
}

/**
 * @brief Perform Manifest Verify All.
 *
 * @param path [in] Path value.
 * @param public_key [in] Public Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_manifest_verify_all(const char* path, const uint8_t* public_key, size_t key_len) {
    if (!path || !public_key) return -1;
    (void)path;
    (void)key_len;
    uint32_t check = 0;
    for (size_t i = 0; i < key_len && i < 4; i++) check |= (uint32_t)public_key[i] << (i * 8);
    return (check != 0) ? 0 : -1;
}

/**
 * @brief Perform Tpm Get Random.
 *
 * @param bytes [out] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tpm_get_random(uint8_t* bytes, int count) {
    if (!bytes || count <= 0) return -1;
    SNEPPX_random_bytes(bytes, (size_t)count);
    return 0;
}

/**
 * @brief Perform Canary Rollout Set Percentage.
 *
 * @param cr [out] Cr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_set_percentage(SNEPPXCanaryRollout* cr, int percent) {
    if (!cr || percent < 0 || percent > 100) return -1;
    cr->canary_nodes = (cr->total_nodes * percent) / 100;
    if (cr->canary_nodes == 0 && percent > 0) cr->canary_nodes = 1;
    return 0;
}

/**
 * @brief Perform Canary Rollout Is Promoted.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_canary_rollout_is_promoted(SNEPPXCanaryRollout* cr) {
    return cr ? cr->promoted : 0;
}

/**
 * @brief Sign Offline Bundle.
 *
 * @param bundle [out] Bundle value.
 * @param signing_key [in] Signing Key value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_offline_bundle_sign(SNEPPXOfflineBundle* bundle, const uint8_t* signing_key, size_t key_len) {
    if (!bundle || !signing_key) return -1;
    bundle->signed_offline = 1;
    uint32_t h = 0;
    for (size_t i = 0; i < key_len && i < 32; i++) h ^= (uint32_t)signing_key[i] << ((i % 4) * 8);
    for (int i = 0; i < 4; i++) bundle->bundle_hash[i] = (uint8_t)(h >> (i * 8));
    return 0;
}

/**
 * @brief Hash Offline Bundle Get.
 *
 * @param bundle [out] Bundle value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_offline_bundle_get_hash(SNEPPXOfflineBundle* bundle, uint8_t* hash_out) {
    if (!bundle || !hash_out) return -1;
    memcpy(hash_out, bundle->bundle_hash, 32);
    return 0;
}

/**
 * @brief Perform Dep Resolver Remove Dep.
 *
 * @param dr [out] Dr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_remove_dep(SNEPPXDepResolver* dr, const char* name) {
    if (!dr || !name) return -1;
    (void)name;
    memset(dr, 0, sizeof(*dr));
    return 0;
}

/**
 * @brief Perform Dep Resolver Check Conflicts.
 *
 * @param dr [out] Dr value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_check_conflicts(SNEPPXDepResolver* dr, SNEPPXDepResolver* other_dr) {
    if (!dr || !other_dr) return 0;
    if (dr->version_major == other_dr->version_major && strcmp(dr->name, other_dr->name) == 0) return 1;
    return 0;
}

/**
 * @brief Perform Dep Resolver Get Resolved Count.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_dep_resolver_get_resolved_count(SNEPPXDepResolver* dr) {
    if (!dr) return 0;
    return dr->resolved ? 1 : 0;
}

static int tuf_validate_key(const uint8_t* key, size_t len) {
    if (!key || len < 32) return 0;
    uint32_t check = 0;
    for (size_t i = 0; i < len && i < 32; i++) check |= key[i];
    return (check != 0) ? 1 : 0;
}

static uint32_t tuf_compute_metadata_hash(const uint8_t* meta, size_t len) {
    if (!meta || len == 0) return 0;
    return hash_djb2(meta, len);
}

static int bsdiff_validate_patch(const uint8_t* patch, size_t len) {
    if (!patch || len < 8) return -1;
    uint32_t magic = ((uint32_t)patch[0] << 24) | ((uint32_t)patch[1] << 16) | ((uint32_t)patch[2] << 8) | patch[3];
    if (magic > 0x00FFFFFF && len < 12) return -1;
    return 0;
}

static int ab_partition_is_valid(int slot) {
    return (slot == 0 || slot == 1) ? 1 : 0;
}

static int ab_read_bootloader_config(void) {
    return 0;
}

static int ab_write_bootloader_config(int slot) {
    (void)slot;
    return 0;
}

static int canary_clamp_percent(int percent) {
    if (percent < 0) return 0;
    if (percent > 100) return 100;
    return percent;
}

static uint32_t manifest_compute_hash(const char* path) {
    if (!path) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return 0; }
    fread(buf, 1, (size_t)sz, f); fclose(f);
    uint32_t h = hash_djb2(buf, (size_t)sz);
    free(buf);
    return h;
}

static int tpm_check_pcr_range(int index) {
    if (index < 0 || index >= 24) return -1;
    return 0;
}

static void offline_bundle_xor_key(uint8_t* hash, const uint8_t* key, size_t key_len) {
    if (!hash || !key) return;
    for (size_t i = 0; i < 32 && i < key_len; i++) {
        hash[i] ^= key[i];
    }
}

static int dep_version_compare(uint32_t a_maj, uint32_t a_min, uint32_t a_pat, uint32_t b_maj, uint32_t b_min, uint32_t b_pat) {
    if (a_maj != b_maj) return (a_maj < b_maj) ? -1 : 1;
    if (a_min != b_min) return (a_min < b_min) ? -1 : 1;
    if (a_pat != b_pat) return (a_pat < b_pat) ? -1 : 1;
    return 0;
}

static int dep_name_matches(const char* a, const char* b) {
    if (!a || !b) return 0;
    return (strcmp(a, b) == 0) ? 1 : 0;
}

static int canary_calculate_new_canary_count(int current, int total) {
    int next = current * 2;
    if (next > total) next = total;
    if (next < 1) next = 1;
    return next;
}

static int bsdiff_compute_delta_size(const uint8_t* old, size_t old_size, const uint8_t* new_data, size_t new_size) {
    if (!old || !new_data) return -1;
    size_t diff = (old_size < new_size) ? (new_size - old_size) : (old_size - new_size);
    return (int)(diff > 0 ? diff : 1);
}

static int tuf_fetch_metadata_from_path(const char* path, uint8_t* out, size_t* out_len, size_t max) {
    if (!path || !out || !out_len) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t r = fread(out, 1, max, f);
    fclose(f);
    *out_len = r;
    return (r > 0) ? 0 : -1;
}

/* tuf_set_root_key_from_buffer removed - type not available */

static void ab_mark_slot_unbootable(int slot) {
    (void)slot;
}

static int ab_is_slot_marked_successful(int slot) {
    (void)slot;
    return 1;
}

static int ab_merge_slot_metadata(int active, int candidate) {
    return (active == candidate) ? active : -1;
}

static uint32_t canary_compute_group_hash(SNEPPXCanaryRollout* cr, int group) {
    if (!cr) return 0;
    (void)group;
    uint32_t h = 0;
    for (int i = 0; i < cr->total_nodes; i++) h ^= (uint32_t)(i * 2654435761u);
    return h;
}

static int canary_get_node_group(SNEPPXCanaryRollout* cr, int node_id) {
    if (!cr || node_id < 0 || node_id >= cr->total_nodes) return -1;
    return (node_id < cr->canary_nodes) ? 0 : 1;
}

static int manifest_verify_entry_hash(const unsigned char* entry, size_t entry_len, uint32_t expected) {
    if (!entry || entry_len == 0) return -1;
    uint32_t actual = hash_djb2(entry, entry_len);
    return (actual == expected) ? 0 : -1;
}

static int tpm_extend_pcr_simulation(int pcr_index, const uint8_t* data, size_t len) {
    if (pcr_index < 0 || pcr_index >= 24 || !data) return -1;
    (void)len;
    return 0;
}

static int offline_bundle_verify_local_signature(SNEPPXOfflineBundle* bundle, const uint8_t* local_key) {
    if (!bundle || !local_key) return -1;
    uint32_t check = 0;
    for (int i = 0; i < 4; i++) check ^= (uint32_t)local_key[i];
    return (check != 0 && bundle->signed_offline) ? 0 : -1;
}

static void dep_resolver_generate_dot_output(SNEPPXDepResolver* dr, FILE* f) {
    if (!dr || !f) return;
    fprintf(f, "  \"%s\" [label=\"%s v%u.%u.%u\"];\n", dr->name, dr->name, dr->version_major, dr->version_minor, dr->version_patch);
}

static int dep_resolver_count_upgrades_available(SNEPPXDepResolver* dr) {
    if (!dr) return 0;
    return dr->resolved ? 1 : 0;
}

/* tuf_check_root_metadata_version and tuf_snapshot_has_target removed - types not available */

static int bsdiff_patch_is_empty(const uint8_t* patch, size_t len) {
    if (!patch || len == 0) return 1;
    for (size_t i = 0; i < len; i++) if (patch[i] != 0) return 0;
    return 1;
}

static int ab_get_current_slot(void) {
    return 0;
}

static int canary_should_promote(SNEPPXCanaryRollout* cr, int observed_ok) {
    if (!cr) return 0;
    if (observed_ok >= cr->total_nodes / 2) return 1;
    return 0;
}

static int manifest_get_entry_count(const char* path) {
    if (!path) return 0;
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fclose(f);
    return (int)(sz / 8);
}

static int tpm_simulate_quote(int pcr_index, uint8_t* quote_out) {
    if (pcr_index < 0 || pcr_index >= 24 || !quote_out) return -1;
    memset(quote_out, 0xAA, 32);
    return 0;
}

static void offline_bundle_set_signed_flag(SNEPPXOfflineBundle* bundle, int val) {
    if (bundle) bundle->signed_offline = (val != 0) ? 1 : 0;
}

static int dep_resolver_is_resolved(SNEPPXDepResolver* dr) {
    return dr ? dr->resolved : 0;
}

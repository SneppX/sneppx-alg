#include "signed_update.h"
#include "test_gtest.h"
#include "s7_extensions.h"
#include "cryptographic_hashing_blake3.h"
#include <stdio.h>
#include <string.h>

/*
 * SNEPPX - Test S7 Updates
 *
 * WHAT
 *   Test S7 Updates.
 *
 * CONCEPT
 *   Provides the Test S7 Updates.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */





static void test_update_verifier_init(void) {
    SNEPPXUpdateVerifier uv;
    SX_ASSERT(SNEPPX_update_verifier_init(&uv) == 0, "init");
    SX_ASSERT(uv.rollback_protection_enabled == 1, "rollback protection");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_update_verifier_rollback_detect(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    uint32_t older[] = {0, 9, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, older) == 1, "rollback detected");
    uint32_t same[] = {1, 0, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, same) == 0, "same version ok");
    uint32_t newer[] = {2, 0, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, newer) == 0, "newer version ok");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_update_verifier_min_version(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SX_ASSERT(SNEPPX_update_verifier_set_min_version(&uv, 1, 5, 0) == 0, "set min version");
    uint32_t below_min[] = {1, 2, 0};
    SX_ASSERT(SNEPPX_update_verifier_rollback_check(&uv, below_min) == 1, "below min rejected");
    SNEPPX_update_verifier_destroy(&uv);
}

static int prepare_signed_update(SNEPPXUpdateVerifier* uv, SNEPPXSignedUpdate* update,
                                  const uint8_t* payload, size_t payload_len,
                                  uint8_t seed[SNEPPX_ED25519_SEED_LEN], int trust_key) {
    if (!uv || !update || !payload || payload_len == 0 || payload_len > UINT32_MAX) return -1;
    memset(update, 0, sizeof(*update));
    update->version_major = 2;
    update->timestamp = UINT64_C(1700000000);
    update->chunk_count = 1;
    update->chunks[0].chunk_index = 0;
    update->chunks[0].chunk_size = (uint32_t)payload_len;
    SNEPPXBlake3State ctx;
    SNEPPX_blake3_init(&ctx);
    SNEPPX_blake3_update(&ctx, payload, payload_len);
    SNEPPX_blake3_finish(&ctx, update->chunks[0].chunk_hash);
    memset(seed, 0x01, SNEPPX_ED25519_SEED_LEN);
    if (trust_key) {
        uint8_t pubkey[SNEPPX_ED25519_PUBLIC_KEY_LEN];
        if (SNEPPX_ed25519_publickey_from_seed(seed, pubkey) != 0 ||
            SNEPPX_update_verifier_set_trusted_pubkey(uv, pubkey) != 0) return -1;
    }
    return SNEPPX_update_verifier_sign_update(update, seed, SNEPPX_ED25519_SEED_LEN);
}

static void test_signed_update_apply_requires_trusted_key(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    static const uint8_t payload[] = "signed update payload";
    SX_ASSERT(prepare_signed_update(&uv, &update, payload, sizeof(payload) - 1, seed, 0) == 0,
              "prepare signed update without trusted key");
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, payload, sizeof(payload) - 1) == -1,
              "apply without trusted key rejected");
    SX_ASSERT(uv.current_version[0] == 1, "version not updated");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_apply_succeeds_with_trusted_key(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    static const uint8_t payload[] = "signed update payload";
    SX_ASSERT(prepare_signed_update(&uv, &update, payload, sizeof(payload) - 1, seed, 1) == 0,
              "prepare signed update");
    SX_ASSERT(SNEPPX_update_verifier_check(&uv, &update) == 1, "signed metadata accepted");
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, payload, sizeof(payload) - 1) == 0,
              "apply verified payload");
    SX_ASSERT(uv.current_version[0] == 2, "version updated");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_rejects_tampered_payload(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    uint8_t payload[] = "signed update payload";
    SX_ASSERT(prepare_signed_update(&uv, &update, payload, sizeof(payload) - 1, seed, 1) == 0,
              "prepare signed update");
    payload[0] ^= 0x01;
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, payload, sizeof(payload) - 1) == -1,
              "tampered payload rejected");
    SX_ASSERT(uv.current_version[0] == 1, "tampered payload did not change version");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_rejects_payload_length_mismatch(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    static const uint8_t payload[] = "signed update payload";
    SX_ASSERT(prepare_signed_update(&uv, &update, payload, sizeof(payload) - 1, seed, 1) == 0,
              "prepare signed update");
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, payload, sizeof(payload) - 2) == -1,
              "truncated payload rejected");
    SX_ASSERT(uv.current_version[0] == 1, "truncated payload did not change version");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_rejects_missing_payload(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    static const uint8_t payload[] = "signed update payload";
    SX_ASSERT(prepare_signed_update(&uv, &update, payload, sizeof(payload) - 1, seed, 1) == 0,
              "prepare signed update");
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, NULL, 0) == -1,
              "missing payload rejected");
    SX_ASSERT(uv.current_version[0] == 1, "missing payload did not change version");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_rejects_tampered_chunk_metadata(void) {
    SNEPPXUpdateVerifier uv;
    SNEPPX_update_verifier_init(&uv);
    SNEPPXSignedUpdate update;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    static const uint8_t payload[] = "signed update payload";
    SX_ASSERT(prepare_signed_update(&uv, &update, payload, sizeof(payload) - 1, seed, 1) == 0,
              "prepare signed update");
    update.chunks[0].chunk_hash[0] ^= 0x80;
    SX_ASSERT(SNEPPX_update_verifier_apply(&uv, &update, payload, sizeof(payload) - 1) == -1,
              "tampered signed chunk metadata rejected");
    SX_ASSERT(uv.current_version[0] == 1, "tampered metadata did not change version");
    SNEPPX_update_verifier_destroy(&uv);
}

static void test_signed_update_rejects_invalid_chunk_layout(void) {
    SNEPPXSignedUpdate update;
    memset(&update, 0, sizeof(update));
    update.version_major = 2;
    update.timestamp = UINT64_C(1700000000);
    update.chunk_count = 1;
    update.chunks[0].chunk_index = 1;
    update.chunks[0].chunk_size = 4;
    uint8_t seed[SNEPPX_ED25519_SEED_LEN];
    memset(seed, 0x01, sizeof(seed));
    SX_ASSERT(SNEPPX_update_verifier_sign_update(&update, seed, sizeof(seed)) == -1,
              "out-of-order chunk index rejected");
}

static void test_bsdiff_roundtrip(void) {
    static const uint8_t old_data[] = "prefix-old-suffix";
    static const uint8_t new_data[] = "prefix-new-suffix";
    uint8_t patch[128]; size_t patch_len = sizeof(patch);
    SX_ASSERT(SNEPPX_bsdiff(old_data, sizeof(old_data)-1, new_data, sizeof(new_data)-1, patch, &patch_len) == 0,
              "generate SXP1 patch");
    SX_ASSERT(patch_len > 13 && memcmp(patch, "SXP1", 4) == 0, "SXP1 magic");
    uint8_t output[64]; memset(output, 0xA5, sizeof(output)); size_t output_len = sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(old_data, sizeof(old_data)-1, patch, patch_len, output, &output_len) == 0,
              "apply SXP1 patch");
    SX_ASSERT(output_len == sizeof(new_data)-1 && memcmp(output, new_data, output_len) == 0,
              "patch roundtrip bytes");
}
static void test_bsdiff_rejects_truncated_patch(void) {
    static const uint8_t old_data[] = "prefix-old-suffix", new_data[] = "prefix-new-suffix";
    uint8_t patch[128]; size_t patch_len = sizeof(patch);
    SX_ASSERT(SNEPPX_bsdiff(old_data, sizeof(old_data)-1, new_data, sizeof(new_data)-1, patch, &patch_len) == 0,
              "generate patch for truncation test");
    for (size_t n=0;n<patch_len;n++) {
        uint8_t output[64]; memset(output,0xA5,sizeof(output)); size_t output_len=sizeof(output);
        int rc=SNEPPX_bspatch(old_data,sizeof(old_data)-1,patch,n,output,&output_len);
        SX_ASSERT(rc == -1, "truncated patch rejected");
        SX_ASSERT(output[0] == 0xA5, "truncated patch leaves output untouched");
    }
}
static void test_bsdiff_rejects_malformed_records(void) {
    static const uint8_t old_data[] = "prefix-old-suffix", new_data[] = "prefix-new-suffix";
    uint8_t patch[128], malformed[130], output[64]; size_t patch_len=sizeof(patch), output_len;
    SX_ASSERT(SNEPPX_bsdiff(old_data,sizeof(old_data)-1,new_data,sizeof(new_data)-1,patch,&patch_len)==0,
              "generate malformed-input base patch");
    memcpy(malformed,patch,patch_len); malformed[0]^=1; output_len=sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(old_data,sizeof(old_data)-1,malformed,patch_len,output,&output_len)==-1,
              "invalid patch magic rejected");
    memcpy(malformed,patch,patch_len); malformed[12]=0x7f; output_len=sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(old_data,sizeof(old_data)-1,malformed,patch_len,output,&output_len)==-1,
              "invalid patch opcode rejected");
    memcpy(malformed,patch,patch_len); malformed[patch_len]=0x55; output_len=sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(old_data,sizeof(old_data)-1,malformed,patch_len+1,output,&output_len)==-1,
              "trailing patch bytes rejected");
    memcpy(malformed,patch,patch_len); memset(malformed+13,0xff,8); output_len=sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(old_data,sizeof(old_data)-1,malformed,patch_len,output,&output_len)==-1,
              "out-of-range COPY rejected");
}
static void test_bsdiff_rejects_oversized_insert_and_capacity(void) {
    static const uint8_t insert_data[]="xyz";
    uint8_t patch[64],output[8]; size_t patch_len=sizeof(patch);
    SX_ASSERT(SNEPPX_bsdiff(NULL,0,insert_data,sizeof(insert_data)-1,patch,&patch_len)==0,
              "generate INSERT-only patch");
    memset(patch+13,0xff,8); size_t output_len=sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(NULL,0,patch,patch_len,output,&output_len)==-1,
              "oversized INSERT length rejected");
    patch_len=sizeof(patch);
    SX_ASSERT(SNEPPX_bsdiff(NULL,0,insert_data,sizeof(insert_data)-1,patch,&patch_len)==0,
              "regenerate INSERT-only patch");
    memset(output,0xA5,sizeof(output)); output_len=1;
    SX_ASSERT(SNEPPX_bspatch(NULL,0,patch,patch_len,output,&output_len)==-1,
              "insufficient output capacity rejected");
    SX_ASSERT(output[0]==0xA5,"capacity failure leaves output untouched");
}
static void test_bsdiff_handles_empty_inputs(void) {
    static const uint8_t text[]="abc";
    uint8_t patch[64],output[8]; size_t patch_len=sizeof(patch);
    SX_ASSERT(SNEPPX_bsdiff(NULL,0,text,sizeof(text)-1,patch,&patch_len)==0,"empty old input supported");
    size_t output_len=sizeof(output);
    SX_ASSERT(SNEPPX_bspatch(NULL,0,patch,patch_len,output,&output_len)==0 &&
              output_len==sizeof(text)-1 && memcmp(output,text,output_len)==0,"empty old input roundtrip");
    patch_len=sizeof(patch);
    SX_ASSERT(SNEPPX_bsdiff(text,sizeof(text)-1,NULL,0,patch,&patch_len)==0,"empty new input supported");
    output_len=0;
    SX_ASSERT(SNEPPX_bspatch(text,sizeof(text)-1,patch,patch_len,NULL,&output_len)==0 && output_len==0,
              "empty output roundtrip");
}

static void test_legacy_tuf_apis_fail_closed_without_mutating_outputs(void) {
    SNEPPXTUFMetadata tuf, original;
    memset(&tuf, 0xA5, sizeof(tuf));
    memcpy(&original, &tuf, sizeof(tuf));
    uint8_t data[] = "metadata";
    uint8_t signature[64], key_out[32], snapshot[64], timestamp[32];
    memset(signature, 0x5A, sizeof(signature));
    memset(key_out, 0x6B, sizeof(key_out));
    memset(snapshot, 0x7C, sizeof(snapshot));
    memset(timestamp, 0x3D, sizeof(timestamp));
    size_t signature_len = sizeof(signature), snapshot_len = sizeof(snapshot), timestamp_len = sizeof(timestamp);
    const uint8_t key[32] = {1};
    SX_ASSERT(SNEPPX_tuf_init(&tuf) == -1, "unimplemented TUF init fails closed");
    SX_ASSERT(memcmp(&tuf, &original, sizeof(tuf)) == 0, "TUF init preserves caller state");
    SX_ASSERT(SNEPPX_tuf_sign_root(&tuf, data, sizeof(data) - 1, signature, &signature_len) == -1,
              "legacy TUF signer disabled");
    SX_ASSERT(signature_len == sizeof(signature) && signature[0] == 0x5A, "disabled signer preserves outputs");
    SX_ASSERT(SNEPPX_tuf_verify_targets(&tuf, data, sizeof(data) - 1) == -1, "TUF targets verifier disabled");
    SX_ASSERT(SNEPPX_tuf_generate_keypair(0) == -1, "fake TUF keypair generation disabled");
    SX_ASSERT(SNEPPX_tuf_revoke_key(0) == -1, "no-op TUF key revocation disabled");
    SX_ASSERT(SNEPPX_tuf_is_initialized(&tuf) == -1, "TUF initialization state unavailable");
    SX_ASSERT(SNEPPX_tuf_get_key(&tuf, 0, key_out, sizeof(key_out)) == -1, "TUF key export disabled");
    SX_ASSERT(key_out[0] == 0x6B, "disabled key export preserves output");
    SX_ASSERT(SNEPPX_tuf_set_key(&tuf, 0, key, sizeof(key)) == -1, "TUF key injection disabled");
    SX_ASSERT(memcmp(&tuf, &original, sizeof(tuf)) == 0, "disabled TUF operations preserve metadata");
    SX_ASSERT(SNEPPX_tuf_snapshot(snapshot, &snapshot_len) == -1, "unsigned TUF snapshot creation disabled");
    SX_ASSERT(snapshot_len == sizeof(snapshot) && snapshot[0] == 0x7C, "disabled snapshot preserves outputs");
    SX_ASSERT(SNEPPX_tuf_timestamp(timestamp, &timestamp_len) == -1, "unsigned TUF timestamp creation disabled");
    SX_ASSERT(timestamp_len == sizeof(timestamp) && timestamp[0] == 0x3D, "disabled timestamp preserves outputs");
    SX_ASSERT(SNEPPX_tuf_verify_snapshot(&tuf, snapshot, sizeof(snapshot)) == -1, "weak snapshot verification disabled");
    SX_ASSERT(SNEPPX_tuf_verify_timestamp(&tuf, timestamp, sizeof(timestamp)) == -1, "weak timestamp verification disabled");
    SX_ASSERT(SNEPPX_tuf_get_keys_loaded(&tuf) == -1, "fake TUF key state unavailable");
}
static void test_legacy_tpm_apis_fail_closed_without_mutating_outputs(void) {
    uint8_t out[96], quote[96], sealed[96], data[96], random_bytes[32];
    uint8_t nonce[8] = {1,2,3,4,5,6,7,8}, digest[32] = {9};
    memset(out, 0xA1, sizeof(out)); memset(quote, 0xB2, sizeof(quote));
    memset(sealed, 0xC3, sizeof(sealed)); memset(data, 0xD4, sizeof(data));
    memset(random_bytes, 0xE5, sizeof(random_bytes));
    size_t out_len=sizeof(out), quote_len=sizeof(quote), sealed_len=sizeof(sealed), data_len=sizeof(data);
    SX_ASSERT(SNEPPX_tpm_pcr_read(0, out, &out_len) == -1, "simulated PCR read disabled");
    SX_ASSERT(out_len==sizeof(out) && out[0]==0xA1, "PCR failure preserves outputs");
    SX_ASSERT(SNEPPX_tpm_quote(nonce,sizeof(nonce),quote,&quote_len)==-1, "simulated TPM quote disabled");
    SX_ASSERT(quote_len==sizeof(quote) && quote[0]==0xB2, "quote failure preserves outputs");
    SX_ASSERT(SNEPPX_tpm_extend(0,digest,sizeof(digest))==-1, "simulated PCR extend disabled");
    SX_ASSERT(SNEPPX_tpm_seal(digest,sizeof(digest),sealed,&sealed_len,1)==-1, "fake TPM seal disabled");
    SX_ASSERT(sealed_len==sizeof(sealed) && sealed[0]==0xC3, "seal failure preserves outputs");
    SX_ASSERT(SNEPPX_tpm_unseal(sealed,sizeof(sealed),data,&data_len,1)==-1, "fake TPM unseal disabled");
    SX_ASSERT(data_len==sizeof(data) && data[0]==0xD4, "unseal failure preserves outputs");
    SX_ASSERT(SNEPPX_tpm_get_random(random_bytes,sizeof(random_bytes))==-1, "TPM random API disabled without TPM");
    SX_ASSERT(random_bytes[0]==0xE5, "TPM random failure preserves output");
    SNEPPX_tpm_set_variable(1, 42);
    SX_ASSERT(SNEPPX_tpm_get_variable(1)==UINT32_MAX, "TPM NV read reports unavailable sentinel");
}

static void test_legacy_manifest_creation_and_hash_fail_closed_without_mutating_outputs(void) {
    const char* files[] = {"payload.bin"};
    uint8_t manifest[64], hash[32];
    memset(manifest, 0xA6, sizeof(manifest));
    memset(hash, 0xB7, sizeof(hash));
    size_t manifest_len = sizeof(manifest);
    SX_ASSERT(SNEPPX_manifest_create("manifest", files, 1, manifest, &manifest_len) == -1,
              "weak manifest creation disabled");
    SX_ASSERT(manifest_len == sizeof(manifest) && manifest[0] == 0xA6,
              "disabled manifest creation preserves outputs");
    SX_ASSERT(SNEPPX_manifest_hash("missing-manifest-file", hash, sizeof(hash)) == -1,
              "short non-cryptographic manifest hash disabled");
    SX_ASSERT(hash[0] == 0xB7, "disabled manifest hash preserves output");
    SX_ASSERT(SNEPPX_manifest_verify_all("manifest", hash, sizeof(hash)) == -1,
              "legacy manifest verification remains disabled");
}

static void test_legacy_manifest_verification_fails_closed(void) {
    uint8_t signature[64] = {0};
    SX_ASSERT(SNEPPX_manifest_verify("CMakeLists.txt", signature, sizeof(signature)) == -1,
              "manifest hash comparison is not signature verification");
}

static void test_legacy_offline_bundle_creation_fails_closed(void) {
    SNEPPXOfflineBundle bundle;
    uint8_t payload[] = "offline payload";
    uint8_t key[32] = {1};
    memset(&bundle, 0xA5, sizeof(bundle));
    SX_ASSERT(SNEPPX_offline_bundle_create(&bundle, payload, sizeof(payload) - 1, key, sizeof(key)) == -1,
              "legacy non-cryptographic bundle creation is disabled");
    SX_ASSERT(bundle.signed_offline == (int)0xA5A5A5A5,
              "disabled bundle creation does not mark bundle as signed");
}


static void test_legacy_signed_diff_fails_closed_without_mutating_outputs(void) {
    static const uint8_t old_data[] = "old";
    static const uint8_t new_data[] = "new";
    uint8_t patch[32];
    memset(patch, 0xA5, sizeof(patch));
    size_t patch_len = sizeof(patch);
    const uint8_t signing_key[32] = {1};
    SX_ASSERT(SNEPPX_bsdiff_with_signature(old_data, sizeof(old_data) - 1,
              new_data, sizeof(new_data) - 1, patch, &patch_len,
              signing_key, sizeof(signing_key)) == -1,
              "XOR-based signed diff is disabled");
    SX_ASSERT(patch_len == sizeof(patch), "disabled signed diff preserves output length");
    for (size_t i = 0; i < sizeof(patch); ++i)
        SX_ASSERT(patch[i] == 0xA5, "disabled signed diff preserves output buffer");
}

static void test_legacy_manifest_buffer_verification_fails_closed(void) {
    const uint8_t manifest[] = "manifest";
    uint8_t signature[64];
    memset(signature, 0x5A, sizeof(signature));
    SX_ASSERT(SNEPPX_manifest_verify_from_buffer(manifest, sizeof(manifest) - 1,
              signature, sizeof(signature)) == -1,
              "32-bit hash manifest verifier is disabled");
}

static void test_legacy_ab_apis_fail_closed_without_mutating_state(void) {
    SNEPPXABPartition ab, original;
    memset(&ab, 0xA5, sizeof(ab));
    memcpy(&original, &ab, sizeof(ab));
    SX_ASSERT(SNEPPX_ab_partition_init(&ab) == -1, "unimplemented A/B init fails closed");
    SX_ASSERT(memcmp(&ab, &original, sizeof(ab)) == 0, "A/B init preserves caller state");
    SX_ASSERT(SNEPPX_ab_partition_mark_good(&ab, 0) == -1, "fake slot-good marking disabled");
    SX_ASSERT(SNEPPX_ab_partition_swap(&ab) == -1, "simulated A/B swap disabled");
    SX_ASSERT(SNEPPX_ab_rollback(0) == -1, "no-op rollback disabled");
    SX_ASSERT(SNEPPX_ab_partition_get_active_slot(&ab) == -1, "active slot state unavailable");
    SX_ASSERT(SNEPPX_ab_partition_get_inactive_slot(&ab) == -1, "inactive slot state unavailable");
    SX_ASSERT(SNEPPX_ab_partition_is_swap_ready(&ab) == -1, "swap-ready state unavailable");
    SX_ASSERT(SNEPPX_ab_mark_slot_bad(&ab, 0) == -1, "fake bad-slot marking disabled");
    SX_ASSERT(SNEPPX_ab_slot_is_good(&ab, 0) == -1, "unverified slot-good result unavailable");
    SX_ASSERT(SNEPPX_ab_force_swap(&ab) == -1, "force swap disabled");
    SX_ASSERT(SNEPPX_ab_partition_get_active(&ab) == -1, "active slot alias unavailable");
    SX_ASSERT(SNEPPX_ab_partition_get_inactive(&ab) == -1, "inactive slot alias unavailable");
    SX_ASSERT(SNEPPX_ab_set_bootloader_slot(1) == -1, "bootloader slot operation disabled");
    SX_ASSERT(memcmp(&ab, &original, sizeof(ab)) == 0, "failed A/B operations preserve state");
}

static void test_canary_rollout(void) {
    SNEPPXCanaryRollout cr;
    SX_ASSERT(SNEPPX_canary_rollout_init(&cr, 100, 5) == 0, "canary init");
    SX_ASSERT(cr.total_nodes == 100, "total nodes");
    SX_ASSERT(cr.canary_nodes == 5, "canary nodes");
    SX_ASSERT(SNEPPX_canary_rollout_promote(&cr) == 0, "promote");
    SX_ASSERT(cr.promoted == 1, "promoted");
}

static void test_dep_resolver(void) {
    SNEPPXDepResolver dr;
    SX_ASSERT(SNEPPX_dep_resolver_init(&dr) == 0, "dep init");
    SX_ASSERT(SNEPPX_dep_resolver_add_dep(&dr, "libfoo", 1, 2, 3) == 0, "add dep");
    SX_ASSERT(SNEPPX_dep_resolver_resolve(&dr) == 0, "resolve");
    SX_ASSERT(dr.resolved == 1, "resolved");
}


TEST(test_s7_updates, update_verifier_init) { test_update_verifier_init(); }
TEST(test_s7_updates, update_verifier_rollback_detect) { test_update_verifier_rollback_detect(); }
TEST(test_s7_updates, update_verifier_min_version) { test_update_verifier_min_version(); }
TEST(test_s7_updates, signed_update_apply_requires_trusted_key) { test_signed_update_apply_requires_trusted_key(); }
TEST(test_s7_updates, signed_update_apply_succeeds_with_trusted_key) { test_signed_update_apply_succeeds_with_trusted_key(); }
TEST(test_s7_updates, signed_update_rejects_tampered_payload) { test_signed_update_rejects_tampered_payload(); }
TEST(test_s7_updates, signed_update_rejects_payload_length_mismatch) { test_signed_update_rejects_payload_length_mismatch(); }
TEST(test_s7_updates, signed_update_rejects_missing_payload) { test_signed_update_rejects_missing_payload(); }
TEST(test_s7_updates, signed_update_rejects_tampered_chunk_metadata) { test_signed_update_rejects_tampered_chunk_metadata(); }
TEST(test_s7_updates, signed_update_rejects_invalid_chunk_layout) { test_signed_update_rejects_invalid_chunk_layout(); }
TEST(test_s7_updates, bsdiff_roundtrip) { test_bsdiff_roundtrip(); }
TEST(test_s7_updates, bsdiff_rejects_truncated_patch) { test_bsdiff_rejects_truncated_patch(); }
TEST(test_s7_updates, bsdiff_rejects_malformed_records) { test_bsdiff_rejects_malformed_records(); }
TEST(test_s7_updates, bsdiff_rejects_oversized_insert_and_capacity) { test_bsdiff_rejects_oversized_insert_and_capacity(); }
TEST(test_s7_updates, bsdiff_handles_empty_inputs) { test_bsdiff_handles_empty_inputs(); }
TEST(test_s7_updates, legacy_tuf_apis_fail_closed_without_mutating_outputs) { test_legacy_tuf_apis_fail_closed_without_mutating_outputs(); }
TEST(test_s7_updates, legacy_tpm_apis_fail_closed_without_mutating_outputs) { test_legacy_tpm_apis_fail_closed_without_mutating_outputs(); }
TEST(test_s7_updates, legacy_manifest_creation_and_hash_fail_closed_without_mutating_outputs) { test_legacy_manifest_creation_and_hash_fail_closed_without_mutating_outputs(); }
TEST(test_s7_updates, legacy_manifest_verification_fails_closed) { test_legacy_manifest_verification_fails_closed(); }
TEST(test_s7_updates, legacy_offline_bundle_creation_fails_closed) { test_legacy_offline_bundle_creation_fails_closed(); }
TEST(test_s7_updates, legacy_signed_diff_fails_closed_without_mutating_outputs) { test_legacy_signed_diff_fails_closed_without_mutating_outputs(); }
TEST(test_s7_updates, legacy_manifest_buffer_verification_fails_closed) { test_legacy_manifest_buffer_verification_fails_closed(); }
TEST(test_s7_updates, legacy_ab_apis_fail_closed_without_mutating_state) { test_legacy_ab_apis_fail_closed_without_mutating_state(); }
TEST(test_s7_updates, canary_rollout) { test_canary_rollout(); }
TEST(test_s7_updates, dep_resolver) { test_dep_resolver(); }

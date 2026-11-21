#ifndef SNEPPX_IR_EVIDENCE_MANAGER_H
#define SNEPPX_IR_EVIDENCE_MANAGER_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_IR_EVIDENCE_MAX_ITEMS 65536
#define SNEPPX_IR_EVIDENCE_HASH_CHAIN_SIZE 4096

/*
 * SNEPPX - Evidence Manager
 *
 * WHAT
 *   Evidence Manager.
 *
 * CONCEPT
 *   Provides the Evidence Manager.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_IR_EVIDENCE_TYPE_MEMORY_DUMP,
    SNEPPX_IR_EVIDENCE_TYPE_DISK_IMAGE,
    SNEPPX_IR_EVIDENCE_TYPE_FILE_COPY,
    SNEPPX_IR_EVIDENCE_TYPE_NETWORK_CAPTURE,
    SNEPPX_IR_EVIDENCE_TYPE_PROCESS_DUMP,
    SNEPPX_IR_EVIDENCE_TYPE_REGISTRY_HIVE,
    SNEPPX_IR_EVIDENCE_TYPE_EVENT_LOG,
    SNEPPX_IR_EVIDENCE_TYPE_COMMAND_HISTORY,
    SNEPPX_IR_EVIDENCE_TYPE_PREFETCH,
    SNEPPX_IR_EVIDENCE_TYPE_AMCACHE,
    SNEPPX_IR_EVIDENCE_TYPE_JUMP_LIST,
    SNEPPX_IR_EVIDENCE_TYPE_USN_JOURNAL,
    SNEPPX_IR_EVIDENCE_TYPE_BROWSER_ARTIFACT,
    SNEPPX_IR_EVIDENCE_TYPE_EMAIL_MESSAGE,
    SNEPPX_IR_EVIDENCE_TYPE_SCHEDULED_TASK,
    SNEPPX_IR_EVIDENCE_TYPE_SERVICE_CONFIG,
    SNEPPX_IR_EVIDENCE_TYPE_PACKET_CAPTURE,
    SNEPPX_IR_EVIDENCE_TYPE_SANDBOX_REPORT,
    SNEPPX_IR_EVIDENCE_TYPE_MEMORY_PAGE
} SNEPPXIREvidenceType;

typedef enum {
    SNEPPX_IR_CHAIN_HASH_SHA256,
    SNEPPX_IR_CHAIN_HASH_SHA512,
    SNEPPX_IR_CHAIN_HASH_BLAKE2B,
    SNEPPX_IR_CHAIN_HASH_SHA3_256
} SNEPPXIRHashAlgorithm;

typedef struct {
    uint64_t evidence_id;
    SNEPPXIREvidenceType evidence_type;
    uint8_t* evidence_name;
    size_t evidence_name_len;
    uint8_t* source_host;
    size_t source_host_len;
    uint64_t acquisition_timestamp_ns;
    uint64_t recorded_timestamp_ns;
    uint8_t* evidence_data;
    size_t evidence_data_len;
    uint8_t* hash_digest;
    size_t hash_digest_len;
    SNEPPXIRHashAlgorithm hash_algo;
    uint8_t* chain_hash_prev;
    size_t chain_hash_prev_len;
    uint8_t* chain_hash_next;
    size_t chain_hash_next_len;
    uint8_t* chain_root_hash;
    size_t chain_root_hash_len;
    uint64_t chain_position;
    uint8_t* metadata_json;
    size_t metadata_json_len;
    uint8_t* tags;
    size_t tags_len;
    uint8_t verified : 1;
    uint8_t admisible : 1;
    uint8_t encrypted : 1;
    uint8_t compressed : 1;
    uint8_t tampered : 1;
} SNEPPXIREvidence;

typedef struct {
    char* case_name;
    size_t case_name_len;
    char* case_id;
    size_t case_id_len;
    SNEPPXIREvidence* evidence_items;
    uint32_t num_evidence;
    uint32_t evidence_capacity;
    uint8_t* chain_root_hash;
    size_t chain_root_hash_len;
    uint64_t evidence_counter;
    uint8_t* evidence_store_path;
    size_t evidence_store_path_len;
    uint8_t chain_integrity : 1;
} SNEPPXIREvidenceManager;

int snepx_ir_evidence_manager_init(SNEPPXIREvidenceManager* mgr, const char* case_name, const char* case_id);
int snepx_ir_evidence_add(SNEPPXIREvidenceManager* mgr, const SNEPPXIREvidence* evidence);
int snepx_ir_evidence_remove(SNEPPXIREvidenceManager* mgr, uint64_t evidence_id);
int snepx_ir_evidence_get(const SNEPPXIREvidenceManager* mgr, uint64_t evidence_id, SNEPPXIREvidence* evidence);
int snepx_ir_evidence_compute_hash(SNEPPXIREvidence* evidence, SNEPPXIRHashAlgorithm algo);
int snepx_ir_evidence_verify_hash(const SNEPPXIREvidence* evidence, uint8_t* computed_hash, size_t* hash_len);
int snepx_ir_evidence_chain_update(SNEPPXIREvidenceManager* mgr, SNEPPXIREvidence* evidence);
int snepx_ir_evidence_chain_verify(const SNEPPXIREvidenceManager* mgr);
int snepx_ir_evidence_encrypt(SNEPPXIREvidence* evidence, const uint8_t* key, size_t key_len);
int snepx_ir_evidence_decrypt(SNEPPXIREvidence* evidence, const uint8_t* key, size_t key_len);
int snepx_ir_evidence_compress(SNEPPXIREvidence* evidence);
int snepx_ir_evidence_decompress(SNEPPXIREvidence* evidence);
int snepx_ir_evidence_export(const SNEPPXIREvidenceManager* mgr, uint64_t evidence_id, uint8_t* out, size_t* out_len);
int snepx_ir_evidence_import(SNEPPXIREvidenceManager* mgr, const uint8_t* in, size_t in_len);
int snepx_ir_evidence_search(const SNEPPXIREvidenceManager* mgr, const uint8_t* query, size_t query_len, uint64_t* results, uint32_t* num_results);
int snepx_ir_evidence_manager_destroy(SNEPPXIREvidenceManager* mgr);

#endif

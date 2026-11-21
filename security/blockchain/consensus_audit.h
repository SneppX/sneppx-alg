#ifndef SNEPPX_BLOCKCHAIN_CONSENSUS_AUDIT_H
#define SNEPPX_BLOCKCHAIN_CONSENSUS_AUDIT_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_BC_MAX_VALIDATORS 1024
#define SNEPPX_BC_MAX_TRANSACTIONS 65536
#define SNEPPX_BC_BLOCK_HASH_SIZE 64
#define SNEPPX_BC_STATE_ROOT_SIZE 64

/*
 * SNEPPX - Consensus Audit
 *
 * WHAT
 *   Consensus Audit.
 *
 * CONCEPT
 *   Provides audit logging.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_BC_CONSENSUS_POW,
    SNEPPX_BC_CONSENSUS_POS,
    SNEPPX_BC_CONSENSUS_DPOS,
    SNEPPX_BC_CONSENSUS_POA,
    SNEPPX_BC_CONSENSUS_PBFT,
    SNEPPX_BC_CONSENSUS_TENDERMINT,
    SNEPPX_BC_CONSENSUS_HOTSTUFF,
    SNEPPX_BC_CONSENSUS_SNOW,
    SNEPPX_BC_CONSENSUS_AVALANCHE,
    SNEPPX_BC_CONSENSUS_HONEYBADGER
} SNEPPXBCConsensusType;

typedef enum {
    SNEPPX_BC_VALIDATOR_ACTIVE,
    SNEPPX_BC_VALIDATOR_SLASHED,
    SNEPPX_BC_VALIDATOR_JAILED,
    SNEPPX_BC_VALIDATOR_UNBONDING,
    SNEPPX_BC_VALIDATOR_UNBONDED
} SNEPPXBCValidatorStatus;

typedef enum {
    SNEPPX_BC_TX_VALID,
    SNEPPX_BC_TX_INVALID,
    SNEPPX_BC_TX_PENDING,
    SNEPPX_BC_TX_DROPPED,
    SNEPPX_BC_TX_REPLACED,
    SNEPPX_BC_TX_REVERTED
} SNEPPXBCTxStatus;

typedef struct {
    uint64_t validator_id;
    uint8_t* validator_pubkey;
    size_t pubkey_len;
    uint64_t stake_amount;
    SNEPPXBCValidatorStatus status;
    uint64_t blocks_proposed;
    uint64_t blocks_signed;
    uint64_t missed_blocks;
    uint64_t slashing_events;
    float commission_rate;
    uint8_t* metadata;
    size_t metadata_len;
    uint8_t active : 1;
    uint8_t jailed : 1;
} SNEPPXBCValidator;

typedef struct {
    uint64_t block_height;
    uint8_t block_hash[SNEPPX_BC_BLOCK_HASH_SIZE];
    uint8_t previous_hash[SNEPPX_BC_BLOCK_HASH_SIZE];
    uint8_t state_root[SNEPPX_BC_STATE_ROOT_SIZE];
    uint8_t* proposer;
    size_t proposer_len;
    uint64_t timestamp_ns;
    uint64_t total_transactions;
    uint64_t total_gas_used;
    uint8_t* transactions_root;
    size_t transactions_root_len;
    uint8_t* receipts_root;
    size_t receipts_root_len;
    uint8_t* extra_data;
    size_t extra_data_len;
    uint64_t total_validator_signatures;
    uint64_t required_signatures;
    uint8_t finalized : 1;
    uint8_t committed : 1;
    uint8_t reverted : 1;
} SNEPPXBCBlock;

typedef struct {
    uint64_t tx_id;
    uint8_t* tx_hash;
    size_t hash_len;
    uint8_t* from_address;
    size_t from_len;
    uint8_t* to_address;
    size_t to_len;
    uint64_t value;
    uint64_t gas_limit;
    uint64_t gas_price;
    uint64_t nonce;
    uint8_t* data;
    size_t data_len;
    SNEPPXBCTxStatus status;
    uint8_t* signature;
    size_t sig_len;
    uint64_t block_height;
    uint32_t tx_index;
    uint8_t* logs;
    size_t logs_len;
    uint8_t* events;
    size_t events_len;
} SNEPPXBCTx;

int snepx_bc_consensus_validate_block(SNEPPXBCBlock* block, SNEPPXBCValidator* validators, uint32_t num_validators, SNEPPXBCConsensusType type);
int snepx_bc_consensus_verify_signatures(SNEPPXBCBlock* block, SNEPPXBCValidator* validators, uint32_t num_validators);
int snepx_bc_consensus_verify_pow(SNEPPXBCBlock* block, uint32_t difficulty);
int snepx_bc_consensus_verify_pos(SNEPPXBCBlock* block, SNEPPXBCValidator* proposer, uint64_t total_stake);
int snepx_bc_consensus_verify_pbft(const SNEPPXBCBlock* block, const uint8_t** precommits, uint32_t num_precommits, uint32_t fault_tolerance);

int snepx_bc_audit_tx(SNEPPXBCTx* tx);
int snepx_bc_audit_tx_chain(SNEPPXBCTx* txs, uint32_t num_txs);
int snepx_bc_audit_validator_behavior(SNEPPXBCValidator* validator, SNEPPXBCBlock* blocks, uint32_t num_blocks, uint8_t* report_out, size_t* report_len);
int snepx_bc_audit_slashing(SNEPPXBCValidator* validator, uint64_t slashing_condition);
int snepx_bc_audit_governance(uint8_t* proposal, size_t proposal_len, uint8_t* votes, size_t votes_len, uint8_t* result_out, size_t* result_len);

int snepx_bc_block_create(SNEPPXBCBlock* block, uint64_t height, const uint8_t* proposer, size_t proposer_len);
int snepx_bc_block_add_tx(SNEPPXBCBlock* block, const SNEPPXBCTx* tx);
int snepx_bc_block_finalize(SNEPPXBCBlock* block);
int snepx_bc_block_destroy(SNEPPXBCBlock* block);

int snepx_bc_validator_create(SNEPPXBCValidator* validator, uint64_t id, const uint8_t* pubkey, size_t pubkey_len, uint64_t stake);
int snepx_bc_validator_slash(SNEPPXBCValidator* validator, uint64_t amount);
int snepx_bc_validator_jail(SNEPPXBCValidator* validator, uint64_t duration_ns);

#endif

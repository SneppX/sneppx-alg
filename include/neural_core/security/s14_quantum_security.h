#ifndef SNEPPX_S14_QUANTUM_SECURITY_H
#define SNEPPX_S14_QUANTUM_SECURITY_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_QUANTUM_MAX_QUBITS 1024
#define SNEPPX_QKD_MAX_KEY_LEN 4096
#define SNEPPX_QRNG_MAX_ENTROPY 8192

/*
 * SNEPPX - S14 Quantum Security
 *
 * WHAT
 *   S14 Quantum Security.
 *
 * CONCEPT
 *   Provides the S14 Quantum Security.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


typedef enum {
    SNEPPX_QRNG_SOURCE_QUANTUM,
    SNEPPX_QRNG_SOURCE_ATOMIC,
    SNEPPX_QRNG_SOURCE_PHOTONIC,
    SNEPPX_QRNG_SOURCE_CHAOTIC,
    SNEPPX_QRNG_SOURCE_HYBRID
} SNEPPXQRNGSourceType;

typedef struct {
    uint8_t* entropy_buffer;
    size_t buffer_size;
    SNEPPXQRNGSourceType source_type;
    uint64_t bits_generated;
    double entropy_per_bit;
    uint32_t min_entropy_h;
    uint8_t* health_test_state;
} SNEPPXQRNGState;

typedef enum {
    SNEPPX_QKD_BB84,
    SNEPPX_QKD_E91,
    SNEPPX_QKD_SARG04,
    SNEPPX_QKD_COHERENT_ONE_WAY,
    SNEPPX_QKD_DECOY_STATE,
    SNEPPX_QKD_MEASUREMENT_DEVICE_INDEPENDENT,
    SNEPPX_QKD_TWIN_FIELD
} SNEPPXQKDProtocol;

typedef struct {
    SNEPPXQKDProtocol protocol;
    uint64* basis_vector;
    uint64* measurement_results;
    size_t raw_key_len;
    uint8_t* sifted_key;
    size_t sifted_key_len;
    uint8_t* reconciled_key;
    size_t reconciled_key_len;
    double qber;
    double error_rate_threshold;
    uint32_t privacy_amplification_bits;
    uint32_t cascade_rounds;
    uint8_t authentication_tag[32];
} SNEPPXQKDSession;

typedef struct {
    uint64_t fock_state[SNEPPX_QUANTUM_MAX_QUBITS];
    uint64_t coherent_state_amplitude;
    uint64_t squeezing_parameter;
    uint64* wigner_function;
    uint64* density_matrix;
    size_t hilbert_space_dim;
    double decoherence_rate;
    uint32_t num_qubits;
} SNEPPXQuantumState;

typedef struct {
    uint8_t* ciphertext;
    size_t ciphertext_len;
    double security_level;
    uint32_t key_size;
    uint32_t ciphertext_expansion;
    uint8_t* authentication_data;
    size_t authentication_len;
} SNEPPXQuantumEncryption;

typedef struct {
    uint8_t* commitment;
    size_t commitment_len;
    uint64_t binding_security;
    uint64_t hiding_security;
    uint8_t* opening_key;
    size_t opening_key_len;
} SNEPPXQuantumCommitment;

typedef struct {
    uint8_t* secret_shared;
    size_t secret_len;
    uint32_t num_parties;
    uint32_t threshold;
    uint64_t* entanglement_graph;
    size_t graph_size;
} SNEPPXQuantumSecretSharing;

int snepx_qrng_init(SNEPPXQRNGState* qrng, SNEPPXQRNGSourceType source);
int snepx_qrng_generate(SNEPPXQRNGState* qrng, uint8_t* output, size_t output_len);
int snepx_qrng_entropy_test(SNEPPXQRNGState* qrng, double* entropy_out);
int snepx_qrng_health_check(SNEPPXQRNGState* qrng);
int snepx_qrng_reseed(SNEPPXQRNGState* qrng, const uint8_t* seed, size_t seed_len);

int snepx_qkd_init(SNEPPXQKDSession* session, SNEPPXQKDProtocol protocol, const uint8_t* basis, size_t basis_len);
int snepx_qkd_measure(SNEPPXQKDSession* session, const uint64_t* qubits, size_t qubit_count);
int snepx_qkd_sift(SNEPPXQKDSession* session, const uint64_t* other_basis, size_t basis_len);
int snepx_qkd_reconcile(SNEPPXQKDSession* session, const uint8_t* other_reconciliation, size_t rec_len);
int snepx_qkd_privacy_amplify(SNEPPXQKDSession* session, uint32_t output_bits);
int snepx_qkd_authenticate(SNEPPXQKDSession* session, const uint8_t* key, size_t key_len);
int snepx_qkd_estimate_qber(SNEPPXQKDSession* session, double* qber_out);

int snepx_quantum_encrypt(SNEPPXQuantumState* state, const uint8_t* plaintext, size_t plaintext_len, SNEPPXQuantumEncryption* encrypted);
int snepx_quantum_decrypt(SNEPPXQuantumState* state, const SNEPPXQuantumEncryption* encrypted, uint8_t* plaintext, size_t* plaintext_len);
int snepx_quantum_commit(SNEPPXQuantumState* state, const uint8_t* secret, size_t secret_len, SNEPPXQuantumCommitment* commitment);
int snepx_quantum_open(SNEPPXQuantumState* state, const SNEPPXQuantumCommitment* commitment, const uint8_t* opening, size_t opening_len);

int snepx_quantum_secret_share(SNEPPXQuantumState* state, const uint8_t* secret, size_t secret_len, uint32_t num_parties, uint32_t threshold, uint8_t** shares, size_t* share_lens);
int snepx_quantum_secret_reconstruct(SNEPPXQuantumState* state, const uint8_t** shares, const size_t* share_lens, uint32_t num_shares, uint8_t* secret_out, size_t* secret_len);

// Post-quantum cryptographic agility
typedef struct {
    uint32_t algorithm_id;
    char algorithm_name[64];
    uint32_t security_level;
    uint32_t key_size;
    uint32_t signature_size;
    double ops_per_second;
    uint8_t* public_key;
    size_t public_key_len;
    uint8_t* private_key;
    size_t private_key_len;
    uint32_t kem_encapsulation_size;
    uint32_t kem_decapsulation_size;
    uint32_t ciphertext_size;
} SNEPPXPqcSuite;

int snepx_pqc_suite_init(SNEPPXPqcSuite* suite, uint32_t algorithm_id);
int snepx_pqc_suite_keygen(SNEPPXPqcSuite* suite);
int snepx_pqc_suite_sign(SNEPPXPqcSuite* suite, const uint8_t* message, size_t message_len, uint8_t* signature, size_t* signature_len);
int snepx_pqc_suite_verify(SNEPPXPqcSuite* suite, const uint8_t* message, size_t message_len, const uint8_t* signature, size_t signature_len);
int snepx_pqc_suite_encapsulate(SNEPPXPqcSuite* suite, const uint8_t* public_key, size_t pk_len, uint8_t* ciphertext, size_t* ct_len, uint8_t* shared_secret, size_t* ss_len);
int snepx_pqc_suite_decapsulate(SNEPPXPqcSuite* suite, const uint8_t* ciphertext, size_t ct_len, uint8_t* shared_secret, size_t* ss_len);
int snepx_pqc_suite_benchmark(SNEPPXPqcSuite* suite, double* keygen_ms, double* sign_ms, double* verify_ms);

#endif

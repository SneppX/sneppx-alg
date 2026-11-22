#ifndef SNEPPX_CRYPTO_H
#define SNEPPX_CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#define SNEPPX_CRYPTO_VERSION "0.1.0"

/*
 * SNEPPX - Cryptographic Primitives Bundle
 *
 * WHAT
 *   Cryptographic Primitives Bundle.
 *
 * CONCEPT
 *   Provides the Cryptographic Primitives Bundle.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */

#if defined(__x86_64__) || defined(_M_X64)
#define SNEPPX_CRYPTO_ASM_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SNEPPX_CRYPTO_ASM_ARM64 1
#endif

#include "authenticated_encryption_module.h"
#include "chacha20_stream_cipher.h"
#include "constant_time_operations.h"
#include "cryptographic_hashing_blake3.h"
#include "cryptographic_random_generator.h"
#include "ed25519_signature_verification.h"
#include "keccak_sha3_hashing.h"
#include "memory_hard_key_derivation.h"
#include "polynomial_authentication_mac.h"
#include "sha512_hashing_implementation.h"

#include "address_space_randomization.h"
#include "container_breakout.h"
#include "container_security.h"
#include "ddos_mitigation.h"
#include "differential_privacy.h"
#include "dilithium.h"
#include "kyber.h"
#include "memory_leak_detector.h"
#include "network_fuzzer.h"
#include "power_analysis_mitigation.h"
#include "protected_memory_manager.h"
#include "rlhf_safety.h"
#include "secure_cache_management.h"
#include "sha256.h"
#include "side_channel_resistant_primitives.h"
#include "sphincsplus.h"
#include "stack_canary_protection.h"
#include "synchronization_lock_interface.h"
#include "timing_attack_countermeasure.h"

/* S4-S9 Security Extensions */
#include "audit_logger.h"
#include "data_poisoning_defense.h"
#include "identity_management.h"
#include "key_vault.h"
#include "model_checking.h"
#include "output_verifier.h"
#include "prompt_filter.h"
#include "self_audit.h"
#include "signed_update.h"
#include "transport_security.h"

/* S7-S9 Extension Headers */
#include "s7_extensions.h"
#include "s8_extensions.h"
#include "s9_extensions.h"

#endif

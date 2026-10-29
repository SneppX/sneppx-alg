/*
 * SNEPPX - Crypto Bindings
 *
 * WHAT
 *   Crypto Bindings.
 *
 * CONCEPT
 *   Provides the Crypto Bindings.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


// Crypto wrappers — extracted from bindings.cpp lines 988–1127, 1657–1672
#pragma once

#include "neural_core/security/sphincsplus.h"
#include "neural_core/security/dilithium.h"
#include "neural_core/security/kyber.h"

static void init_crypto(py::module& m) {
    py::module_ crypto = m.def_submodule("crypto", "Cryptographic functions");

    crypto.def("chacha20_encrypt", [](py::bytes key, py::bytes nonce, py::bytes plaintext, uint32_t counter) {
        std::string k = key, n = nonce, p = plaintext;
        std::string out(p);
        SNEPPXChaCha20State state;
        SNEPPX_chacha20_init(&state, (const uint8_t*)k.data(), (const uint8_t*)n.data(), counter);
        SNEPPX_chacha20_encrypt(&state, (uint8_t*)out.data(), out.size());
        return py::bytes(out);
    });

    crypto.def("sha512", [](py::bytes data) {
        std::string d = data;
        uint8_t hash[64];
        SNEPPX_sha512((const uint8_t*)d.data(), d.size(), hash);
        return py::bytes(reinterpret_cast<char*>(hash), 64);
    });

    crypto.def("sha3_256", [](py::bytes data) {
        std::string d = data;
        uint8_t hash[32];
        SNEPPXSHA3State state;
        SNEPPX_sha3_256_init(&state);
        SNEPPX_sha3_update(&state, (const uint8_t*)d.data(), d.size());
        SNEPPX_sha3_finish(&state, hash);
        return py::bytes(reinterpret_cast<char*>(hash), 32);
    });

    crypto.def("blake3", [](py::bytes data) {
        std::string d = data;
        uint8_t hash[32];
        SNEPPXBlake3State state;
        SNEPPX_blake3_init(&state);
        SNEPPX_blake3_update(&state, (const uint8_t*)d.data(), d.size());
        SNEPPX_blake3_finish(&state, hash);
        return py::bytes(reinterpret_cast<char*>(hash), 32);
    });

    crypto.def("ed25519_keypair", []() {
        SNEPPXEd25519Keypair kp;
        SNEPPX_ed25519_keypair_generate(&kp);
        return py::make_tuple(py::bytes(reinterpret_cast<char*>(kp.public_key), 32),
                              py::bytes(reinterpret_cast<char*>(kp.private_key), 64));
    });

    crypto.def("ed25519_sign", [](py::bytes message, py::bytes sk_bytes) {
        std::string m = message, sk = sk_bytes;
        if (sk.size() < 64) throw std::runtime_error("ed25519 private key must be 64 bytes");
        SNEPPXEd25519Keypair kp;
        memcpy(kp.public_key, sk.data() + 32, 32);
        SNEPPX_ed25519_secret_key_expand(kp.private_key, (const uint8_t*)sk.data());
        SNEPPXEd25519Signature sig;
        SNEPPX_ed25519_sign(&kp, (const uint8_t*)m.data(), m.size(), &sig);
        return py::bytes(reinterpret_cast<char*>(sig.data), 64);
    });

    crypto.def("ed25519_verify", [](py::bytes signature, py::bytes message, py::bytes pk_bytes) {
        std::string sig = signature, m = message, pk = pk_bytes;
        SNEPPXEd25519Signature sig_struct;
        memcpy(sig_struct.data, sig.data(), 64);
        return SNEPPX_ed25519_verify((const uint8_t*)pk.data(),
                                    (const uint8_t*)m.data(), m.size(),
                                    &sig_struct) == 1;
    });

    crypto.def("poly1305_mac", [](py::bytes key, py::bytes message) {
        std::string k = key, m = message;
        uint8_t mac[16];
        SNEPPXPoly1305State state;
        SNEPPX_poly1305_init(&state, (const uint8_t*)k.data());
        SNEPPX_poly1305_update(&state, (const uint8_t*)m.data(), m.size());
        SNEPPX_poly1305_finish(&state, mac);
        return py::bytes(reinterpret_cast<char*>(mac), 16);
    });

    crypto.def("aead_encrypt", [](py::bytes key, py::bytes nonce, py::bytes plaintext, py::bytes aad) {
        std::string k = key, n = nonce, p = plaintext, a = aad;
        std::string ct(p.size(), 0);
        uint8_t tag[16];
        SNEPPX_aead_encrypt((uint8_t*)ct.data(), tag,
                          (const uint8_t*)p.data(), p.size(),
                          (const uint8_t*)a.data(), a.size(),
                          (const uint8_t*)k.data(), (const uint8_t*)n.data());
        return py::bytes(ct + std::string(reinterpret_cast<char*>(tag), 16));
    });

    crypto.def("aead_decrypt", [](py::bytes key, py::bytes nonce, py::bytes ciphertext, py::bytes aad) {
        std::string k = key, n = nonce, ct = ciphertext, a = aad;
        if (ct.size() < 16) throw std::runtime_error("ciphertext too short");
        std::string pt(ct.size() - 16, 0);
        const uint8_t* tag = (const uint8_t*)(ct.data() + ct.size() - 16);
        if (0 != SNEPPX_aead_decrypt((uint8_t*)pt.data(),
                                   (const uint8_t*)ct.data(), ct.size() - 16,
                                   tag,
                                   (const uint8_t*)a.data(), a.size(),
                                   (const uint8_t*)k.data(), (const uint8_t*)n.data()))
            throw std::runtime_error("AEAD decryption failed");
        return py::bytes(pt);
    });

    crypto.def("argon2_hash", [](py::bytes password, py::bytes salt, size_t time_cost, size_t mem_cost, size_t parallelism, size_t hash_len) {
        std::string p = password, s = salt;
        std::string out(hash_len, 0);
        SNEPPXArgon2Config config;
        config.memory_kb = mem_cost;
        config.iterations = time_cost;
        config.parallelism = parallelism;
        config.hash_len = hash_len;
        SNEPPX_argon2id((const uint8_t*)p.data(), p.size(),
                      (const uint8_t*)s.data(), s.size(),
                      &config, (uint8_t*)out.data());
        return py::bytes(out);
    });

    crypto.def("random_bytes", [](size_t n) {
        std::string out(n, 0);
        SNEPPX_random_bytes((uint8_t*)out.data(), n);
        return py::bytes(out);
    });

    crypto.def("random_set_seed", [](py::bytes seed) {
        std::string s = seed;
        if (s.size() < 32) throw std::runtime_error("seed must be >= 32 bytes");
        SNEPPX_random_bytes_set_seed((const uint8_t*)s.data());
    });

    crypto.def("random_clear_seed", []() {
        SNEPPX_random_bytes_clear_seed();
    });

    crypto.def("secure_zero", [](py::bytes data) {
        std::string d = data;
        SNEPPX_secure_zero((void*)d.data(), d.size());
    });

    crypto.def("ct_compare", [](py::bytes a, py::bytes b) {
        std::string sa = a, sb = b;
        return SNEPPX_ct_equal((const uint8_t*)sa.data(), (const uint8_t*)sb.data(), sa.size());
    });

    // ---- SPHINCS+ (SLH-DSA) -------------------------------------------------
    // NOTE: the C implementation uses a single fixed parameter set
    // (pk = SPHINCS_PUBLICKEYBYTES, sk = SPHINCS_SECRETKEYBYTES,
    //  sig = SPHINCS_SIGBYTES); the `variant` argument is currently ignored.
    crypto.def("sphincs_keygen", [](int variant) {
        uint8_t pk[SPHINCS_PUBLICKEYBYTES];
        uint8_t sk[SPHINCS_SECRETKEYBYTES];
        // The keygen only fills the first 48 bytes (sk_seed || sk_prf ||
        // pub_seed); the remaining 16 bytes of the 64-byte secret key are
        // unused padding. Initialise the whole buffer so the tail is
        // deterministic rather than uninitialised stack memory.
        std::memset(sk, 0, SPHINCS_SECRETKEYBYTES);
        SNEPPX_sphincs_keygen(pk, sk, variant);
        return py::make_tuple(
            py::bytes(reinterpret_cast<char*>(pk), SPHINCS_PUBLICKEYBYTES),
            py::bytes(reinterpret_cast<char*>(sk), SPHINCS_SECRETKEYBYTES));
    });

    crypto.def("sphincs_sign", [](py::bytes sk_bytes, py::bytes msg, int variant) {
        std::string sk = sk_bytes, m = msg;
        if (sk.size() < SPHINCS_SECRETKEYBYTES)
            throw std::runtime_error("sphincs secret key must be >= 64 bytes");
        size_t siglen = 0;
        std::vector<uint8_t> sig(SPHINCS_SIGBYTES);
        SNEPPX_sphincs_sign(sig.data(), &siglen,
                            (const uint8_t*)m.data(), m.size(),
                            (const uint8_t*)sk.data(), variant);
        return py::bytes(reinterpret_cast<char*>(sig.data()), siglen);
    });

    crypto.def("sphincs_verify", [](py::bytes sig, py::bytes msg, py::bytes pk_bytes, int variant) {
        std::string s = sig, m = msg, pk = pk_bytes;
        return SNEPPX_sphincs_verify((const uint8_t*)s.data(), s.size(),
                                     (const uint8_t*)m.data(), m.size(),
                                     (const uint8_t*)pk.data(), variant) == 0;
    });

    // ---- Dilithium (ML-DSA) ------------------------------------------------
    // `variant` selects the parameter set (2 -> k=4, 3 -> k=6, 5 -> k=8).
    // pk = 32 + k*320, sk = 96 + k*608 (same formulas used by the C keygen).
    crypto.def("dilithium_keygen", [](int variant) {
        int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
        int eta = (variant == 2) ? 2 : 4;
        int s_bytes = (eta == 2) ? 608 : 672;
        int pk_size = 32 + k * 320;
        int sk_size = 96 + k * s_bytes;
        std::vector<uint8_t> pk(pk_size), sk(sk_size);
        SNEPPX_dilithium_keygen(pk.data(), sk.data(), variant);
        return py::make_tuple(
            py::bytes(reinterpret_cast<char*>(pk.data()), pk_size),
            py::bytes(reinterpret_cast<char*>(sk.data()), sk_size));
    });

    crypto.def("dilithium_sign", [](py::bytes sk_bytes, py::bytes msg, int variant) {
        std::string sk = sk_bytes, m = msg;
        int k = (variant == 2) ? 4 : (variant == 3) ? 6 : 8;
        int eta = (variant == 2) ? 2 : 4;
        int sk_size = 96 + k * (eta == 2 ? 608 : 672);
        if ((int)sk.size() < sk_size)
            throw std::runtime_error("dilithium secret key too short for variant");
        size_t siglen = 0;
        std::vector<uint8_t> sig(6000);
        SNEPPX_dilithium_sign(sig.data(), &siglen,
                              (const uint8_t*)m.data(), m.size(),
                              (const uint8_t*)sk.data(), variant);
        return py::bytes(reinterpret_cast<char*>(sig.data()), siglen);
    });

    crypto.def("dilithium_verify", [](py::bytes sig, py::bytes msg, py::bytes pk_bytes, int variant) {
        std::string s = sig, m = msg, pk = pk_bytes;
        return SNEPPX_dilithium_verify((const uint8_t*)s.data(), s.size(),
                                       (const uint8_t*)m.data(), m.size(),
                                       (const uint8_t*)pk.data(), variant);
    });

    // ---- Kyber (ML-KEM, FIPS 203) -------------------------------------------
    // variant: 2 -> ML-KEM-512, 3 -> ML-KEM-768, 4 -> ML-KEM-1024
    // pk sizes: 800 / 1184 / 1568; sk sizes: 1632 / 2400 / 3168; ct: 768 / 1088 / 1568
    crypto.def("kyber_keygen", [](int variant) {
        std::map<int, std::pair<int, int>> sizes = {
            {2, {800, 1632}}, {3, {1184, 2400}}, {4, {1568, 3168}}};
        auto it = sizes.find(variant);
        if (it == sizes.end()) throw std::runtime_error("unsupported kyber variant");
        int pk_size = it->second.first, sk_size = it->second.second;
        std::vector<uint8_t> pk(pk_size), sk(sk_size);
        if (SNEPPX_kyber_keygen(pk.data(), sk.data(), variant) != 0)
            throw std::runtime_error("kyber_keygen failed");
        return py::make_tuple(
            py::bytes(reinterpret_cast<char*>(pk.data()), pk_size),
            py::bytes(reinterpret_cast<char*>(sk.data()), sk_size));
    });

    crypto.def("kyber_encaps", [](py::bytes pk_bytes, int variant) {
        std::map<int, std::pair<int, int>> ct_sizes = {
            {2, {768, 800}}, {3, {1088, 1184}}, {4, {1568, 1568}}};
        auto it = ct_sizes.find(variant);
        if (it == ct_sizes.end()) throw std::runtime_error("unsupported kyber variant");
        int ct_size = it->second.first, pk_size = it->second.second;
        std::string pk = pk_bytes;
        if ((int)pk.size() != pk_size)
            throw std::runtime_error("kyber public key wrong size for variant");
        std::vector<uint8_t> ct(ct_size), ss(32);
        if (SNEPPX_kyber_encaps(ct.data(), ss.data(),
                                (const uint8_t*)pk.data(), variant) != 0)
            throw std::runtime_error("kyber_encaps failed");
        return py::make_tuple(
            py::bytes(reinterpret_cast<char*>(ct.data()), ct_size),
            py::bytes(reinterpret_cast<char*>(ss.data()), 32));
    });

    crypto.def("kyber_decaps", [](py::bytes ct_bytes, py::bytes sk_bytes, int variant) {
        std::map<int, std::pair<int, int>> ct_sk_sizes = {
            {2, {768, 1632}}, {3, {1088, 2400}}, {4, {1568, 3168}}};
        auto it = ct_sk_sizes.find(variant);
        if (it == ct_sk_sizes.end()) throw std::runtime_error("unsupported kyber variant");
        int ct_size = it->second.first, sk_size = it->second.second;
        std::string ct = ct_bytes, sk = sk_bytes;
        if ((int)ct.size() != ct_size)
            throw std::runtime_error("kyber ciphertext wrong size for variant");
        if ((int)sk.size() != sk_size)
            throw std::runtime_error("kyber secret key wrong size for variant");
        std::vector<uint8_t> ss(32);
        if (SNEPPX_kyber_decaps(ss.data(),
                                (const uint8_t*)ct.data(),
                                (const uint8_t*)sk.data(), variant) != 0)
            throw std::runtime_error("kyber_decaps failed");
        return py::bytes(reinterpret_cast<char*>(ss.data()), 32);
    });
}

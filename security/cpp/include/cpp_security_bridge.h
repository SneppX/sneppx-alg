#ifndef SNEPPX_SECURITY_CPP_H
#define SNEPPX_SECURITY_CPP_H
/*
 * SNEPPX - Cpp Security Bridge
 *
 * WHAT
 *   Cpp Security Bridge.
 *
 * CONCEPT
 *   Provides the Cpp Security Bridge.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/*
 * C++ Security Bindings — v0.5
 *
 * PURPOSE: C++ RAII wrappers for the C security library.  Provides
 * exception-safe handles for AEAD ciphers, hashers, and secure memory
 * buffers.  Used internally by the obfuscation layer and by C++ examples.
 *
 * DEPENDENCIES: authenticated_encryption_module.h, cryptographic_hashing_blake3.h, protected_memory_manager.h
 * VERSION: v0.5
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace SNEPPX {

/* ---------- Secure buffer (automatically zeroed on destruction) ---------- */
class SecureBuffer {
public:
    SecureBuffer();
    explicit SecureBuffer(size_t size);
    SecureBuffer(const uint8_t* data, size_t size);
    ~SecureBuffer();

    SecureBuffer(const SecureBuffer&) = delete;
    SecureBuffer& operator=(const SecureBuffer&) = delete;
    SecureBuffer(SecureBuffer&&) noexcept;
    SecureBuffer& operator=(SecureBuffer&&) noexcept;

    uint8_t*       data();
    const uint8_t* data() const;
    size_t         size() const;
    void           resize(size_t new_size);
    void           clear();

private:
    uint8_t* data_;
    size_t   size_;
};

/* ---------- AEAD cipher ---------- */
class AEADCipher {
public:
    AEADCipher(const uint8_t key[32]);
    ~AEADCipher();

    std::vector<uint8_t> encrypt(const uint8_t* plaintext, size_t len,
                                  const uint8_t* aad, size_t aad_len,
                                  const uint8_t nonce[12]);
    std::vector<uint8_t> decrypt(const uint8_t* ciphertext, size_t len,
                                  const uint8_t* aad, size_t aad_len,
                                  const uint8_t nonce[12], const uint8_t tag[16]);

private:
    uint8_t key_[32];
};

/* ---------- Blake3 hasher ---------- */
class Hasher {
public:
    Hasher();
    void update(const uint8_t* data, size_t len);
    void finalize(uint8_t* out, size_t out_len);

private:
    void* ctx_;
};

} /* namespace SNEPPX */

#endif /* SNEPPX_SECURITY_CPP_H */

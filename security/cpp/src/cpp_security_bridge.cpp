#include "../include/cpp_security_bridge.h"
#include "authenticated_encryption_module.h"
#include "cryptographic_hashing_blake3.h"
#include "constant_time_operations.h"
#include <cstring>
#include <stdexcept>

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


namespace SNEPPX {

/* ---------- SecureBuffer ---------- */
SecureBuffer::SecureBuffer() : data_(nullptr), size_(0) {}
SecureBuffer::SecureBuffer(size_t size) : data_(new uint8_t[size]()), size_(size) {}
SecureBuffer::SecureBuffer(const uint8_t* data, size_t size) : SecureBuffer(size) {
    if (data) std::memcpy(data_, data, size);
}
SecureBuffer::~SecureBuffer() { clear(); }
SecureBuffer::SecureBuffer(SecureBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr; other.size_ = 0;
}
SecureBuffer& SecureBuffer::operator=(SecureBuffer&& other) noexcept {
    if (this != &other) { clear(); data_ = other.data_; size_ = other.size_; other.data_ = nullptr; other.size_ = 0; }
    return *this;
}
uint8_t* SecureBuffer::data() { return data_; }
const uint8_t* SecureBuffer::data() const { return data_; }
size_t SecureBuffer::size() const { return size_; }
void SecureBuffer::resize(size_t new_size) {
    uint8_t* new_data = new uint8_t[new_size]();
    if (data_ && size_ > 0) std::memcpy(new_data, data_, (size_ < new_size) ? size_ : new_size);
    clear(); data_ = new_data; size_ = new_size;
}
void SecureBuffer::clear() { if (data_) { volatile char* p = reinterpret_cast<volatile char*>(data_); for (size_t i = 0; i < size_; i++) p[i] = 0; } delete[] data_; data_ = nullptr; size_ = 0; }

/* ---------- AEADCipher ---------- */
AEADCipher::AEADCipher(const uint8_t key[32]) { if (key) std::memcpy(key_, key, 32); else std::memset(key_, 0, 32); }
AEADCipher::~AEADCipher() { volatile char* p = reinterpret_cast<volatile char*>(key_); for (int i = 0; i < 32; i++) p[i] = 0; }

std::vector<uint8_t> AEADCipher::encrypt(const uint8_t* plaintext, size_t len,
                                           const uint8_t* aad, size_t aad_len,
                                           const uint8_t nonce[12]) {
    if (!plaintext || !key_ || !nonce) return std::vector<uint8_t>();
    std::vector<uint8_t> result(len);
    uint8_t tag[16];
    int ret = SNEPPX_aead_encrypt(result.data(), tag, plaintext, len, aad, aad_len, key_, nonce);
    if (ret != 0) return std::vector<uint8_t>();
    result.insert(result.end(), tag, tag + 16);
    return result;
}

std::vector<uint8_t> AEADCipher::decrypt(const uint8_t* ciphertext, size_t len,
                                           const uint8_t* aad, size_t aad_len,
                                           const uint8_t nonce[12], const uint8_t tag[16]) {
    if (!ciphertext || len < 16 || !key_ || !nonce || !tag) return std::vector<uint8_t>();
    size_t ct_len = len - 16;
    std::vector<uint8_t> result(ct_len);
    int ret = SNEPPX_aead_decrypt(result.data(), ciphertext, ct_len, tag, aad, aad_len, key_, nonce);
    if (ret != 0) return std::vector<uint8_t>();
    return result;
}

/* ---------- Hasher ---------- */
Hasher::Hasher() {
    ctx_ = new SNEPPXBlake3State();
    SNEPPX_blake3_init(static_cast<SNEPPXBlake3State*>(ctx_));
}
void Hasher::update(const uint8_t* data, size_t len) {
    if (data && len > 0 && ctx_) {
        SNEPPX_blake3_update(static_cast<SNEPPXBlake3State*>(ctx_), data, len);
    }
}
void Hasher::finalize(uint8_t* out, size_t out_len) {
    if (out && ctx_) {
        SNEPPX_blake3_finish(static_cast<SNEPPXBlake3State*>(ctx_), out);
    }
    delete static_cast<SNEPPXBlake3State*>(ctx_);
    ctx_ = nullptr;
}

} /* namespace SNEPPX */

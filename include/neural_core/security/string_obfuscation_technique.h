#pragma once
#ifndef SNEPPX_OBF_STRING_H
#define SNEPPX_OBF_STRING_H

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>

/*
 * SNEPPX - String Obfuscation Technique
 *
 * WHAT
 *   String Obfuscation Technique.
 *
 * CONCEPT
 *   Provides code obfuscation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


namespace SNEPPX {

struct SNEPPXObfString {
    std::vector<uint8_t> ciphertext;
    uint32_t key;
    size_t length;
    uint32_t iv;
};

class SNEPPXObfStringPool {
public:
    SNEPPXObfStringPool();
    ~SNEPPXObfStringPool();
    SNEPPXObfStringPool(SNEPPXObfStringPool&&) = default;
    SNEPPXObfStringPool& operator=(SNEPPXObfStringPool&&) = default;
    SNEPPXObfStringPool(const SNEPPXObfStringPool&) = delete;
    SNEPPXObfStringPool& operator=(const SNEPPXObfStringPool&) = delete;

    SNEPPXObfString encrypt(const std::string& plaintext, uint32_t key);
    void decrypt(const SNEPPXObfString& crypt, char* output, size_t output_len);
    void pool_register(const std::string& str);
    void pool_decrypt_all();
    const char* get_string(size_t index) const;
    size_t count() const;

private:
    struct PoolEntry {
        SNEPPXObfString crypt;
        std::unique_ptr<char[]> decrypted;
        bool used;
    };

    std::vector<PoolEntry> pool;
    uint32_t next_key;

    uint32_t generate_key();
    void xor_cipher(const uint8_t* input, uint8_t* output, size_t len, uint32_t key, uint32_t iv);
};

/**
 * @brief Perform Obf Compile Time Key.
 *
 * @param base [in] Base value.
 *
 * @return 0 on success, -1 on error.
 */
constexpr uint32_t SNEPPX_obf_compile_time_key(uint32_t base) {
    return base ^ 0xA71A75A7;
}

template<size_t N>
/**
 * @brief Perform Obf Encrypt Literal.
 *
 * @return 0 on success, -1 on error.
 */
constexpr std::array<uint8_t, N> SNEPPX_obf_encrypt_literal(const char(&str)[N], uint32_t key) {
    std::array<uint8_t, N> result{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = static_cast<uint8_t>(str[i]) ^ static_cast<uint8_t>((key >> ((i % 4) * 8)) & 0xFF);
    }
    return result;
}

} // namespace SNEPPX

#endif // SNEPPX_OBF_STRING_H

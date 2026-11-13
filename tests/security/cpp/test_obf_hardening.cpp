#include "obfuscation_advanced.h"

#include <cassert>
#include <cstdint>
#include <cstring>

int main() {
    SNEPPXJunkCodeGen generator{};
    assert(SNEPPX_junk_code_init(&generator) == 0);
    assert(generator.junk_count > 0 && generator.junk_count <= 64);

    uint8_t code[32] = {0x55, 0x48, 0x89, 0xE5};
    size_t code_len = 4;
    assert(SNEPPX_junk_code_insert(&generator, code, &code_len, sizeof(code), -1) == 0);
    assert(code_len > 4 && code_len <= sizeof(code));

    uint8_t unchanged[8] = {1, 2, 3, 4};
    const uint8_t before[8] = {1, 2, 3, 4};
    size_t too_small_len = 4;
    assert(SNEPPX_junk_code_insert(&generator, unchanged, &too_small_len, 4, 0) == -1);
    assert(too_small_len == 4);
    assert(std::memcmp(unchanged, before, sizeof(before)) == 0);

    generator.junk_count = 65;
    uint8_t guarded[16] = {7, 8, 9};
    size_t guarded_len = 3;
    assert(SNEPPX_junk_code_insert(&generator, guarded, &guarded_len, sizeof(guarded), 0) == -1);
    assert(guarded_len == 3);

    SNEPPXBogusCF control_flow{};
    assert(SNEPPX_bogus_cf_init(&control_flow) == 0);
    const uint8_t fake_block[] = {0x90, 0x90};
    assert(SNEPPX_bogus_cf_add_fake_block(&control_flow, fake_block, sizeof(fake_block)) == -1);

    uint8_t executable_bytes[] = {0x90, 0x90, 0xC3, 0xCC};
    const uint8_t executable_before[] = {0x90, 0x90, 0xC3, 0xCC};
    assert(SNEPPX_bogus_cf_redirect(&control_flow, executable_bytes, sizeof(executable_bytes)) == -1);
    assert(std::memcmp(executable_bytes, executable_before, sizeof(executable_before)) == 0);

    SNEPPXWhiteBoxAES whitebox{};
    const uint8_t key[16] = {0};
    const uint8_t input[16] = {1};
    uint8_t output[16];
    std::memset(output, 0xA5, sizeof(output));
    assert(SNEPPX_whitebox_aes_init(&whitebox, key) == -1);
    SNEPPX_whitebox_aes_encrypt(&whitebox, input, output);
    for (uint8_t byte : output) assert(byte == 0);

    return 0;
}

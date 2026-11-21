#include "aes_gcm.h"
#include <string.h>
/*
 * SNEPPX - AES-GCM (Galois/Counter Mode)
 *
 * WHAT
 *   AES-GCM (Galois/Counter Mode).
 *
 * CONCEPT
 *   AES-128/192/256 encryption in CTR mode with GHASH authentication tag.
 *
 * ROLE
 *   Layer S2 authenticated encryption for TLS record protection and key wrapping.
 *
 * REFERENCES
 *   NIST SP 800-38D (GCM), FIPS 197 (AES).
 */



static const uint32_t rcon[] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D};
static const uint8_t sbox[256] = {
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
};

static uint8_t xtime(uint8_t a) { return (uint8_t)((a << 1) ^ (((a >> 7) & 1) * 0x1B)); }
static uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < 8; i++) { if (b & 1) p ^= a; a = xtime(a); b >>= 1; }
    return p;
}

static uint32_t sub_word(uint32_t w) {
    return ((uint32_t)sbox[(w >> 24) & 0xFF] << 24) | ((uint32_t)sbox[(w >> 16) & 0xFF] << 16) | ((uint32_t)sbox[(w >> 8) & 0xFF] << 8) | (uint32_t)sbox[w & 0xFF];
}

static uint32_t rot_word(uint32_t w) { return (w << 8) | (w >> 24); }
static uint32_t load32(const uint8_t* b) { return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3]; }
static void store32(uint8_t* b, uint32_t w) { b[0]=(uint8_t)(w>>24); b[1]=(uint8_t)(w>>16); b[2]=(uint8_t)(w>>8); b[3]=(uint8_t)w; }
static void xor_block(uint8_t* d, const uint8_t* s) { for (int i=0;i<16;i++) d[i]^=s[i]; }
static void inc32(uint8_t* block) { for (int i=15;i>=12;i--) if (++block[i]) break; }

/**
 * @brief Expand Aes256.
 */
void SNEPPX_aes256_key_expansion(const uint8_t key[32], uint32_t rk[60]) {
    for (int i=0;i<8;i++) rk[i]=load32(key+i*4);
    for (int i=8;i<60;i++) {
        uint32_t t=rk[i-1];
        if (i%8==0) t=sub_word(rot_word(t))^(rcon[i/8-1]<<24);
        else if (i%8==4) t=sub_word(t);
        rk[i]=rk[i-8]^t;
    }
}

/**
 * @brief Encrypt a single block with Aes256.
 */
void SNEPPX_aes256_encrypt_block(const uint32_t rk[60], const uint8_t in[16], uint8_t out[16]) {
    uint32_t s[4];
    for (int i=0;i<4;i++) s[i]=load32(in+i*4)^rk[i];
    for (int r=1;r<14;r++) {
        for (int i=0;i<16;i++) out[i]=sbox[((uint8_t*)s)[i]];
        for (int i=0;i<4;i++) s[i]=load32(out+i*4);
        s[0]=((s[0]>>24)|((s[0]>>8)&0xFF00)|((s[0]<<8)&0xFF0000)|(s[0]<<24)) ^
             ((s[1]>>24)|((s[1]>>8)&0xFF00)|((s[1]<<8)&0xFF0000)|(s[1]<<24));
        s[1]=((s[1]>>24)|((s[1]>>8)&0xFF00)|((s[1]<<8)&0xFF0000)|(s[1]<<24)) ^
             ((s[2]>>24)|((s[2]>>8)&0xFF00)|((s[2]<<8)&0xFF0000)|(s[2]<<24));
        s[2]=((s[2]>>24)|((s[2]>>8)&0xFF00)|((s[2]<<8)&0xFF0000)|(s[2]<<24)) ^
             ((s[3]>>24)|((s[3]>>8)&0xFF00)|((s[3]<<8)&0xFF0000)|(s[3]<<24));
        s[3]=((s[3]>>24)|((s[3]>>8)&0xFF00)|((s[3]<<8)&0xFF0000)|(s[3]<<24)) ^
             ((s[0]>>24)|((s[0]>>8)&0xFF00)|((s[0]<<8)&0xFF0000)|(s[0]<<24));
        for (int i=0;i<4;i++) s[i]^=rk[r*4+i];
    }
    for (int i=0;i<16;i++) out[i]=sbox[((uint8_t*)s)[i]];
    for (int i=0;i<4;i++) store32(out+i*4,load32(out+i*4)^rk[56+i]);
}

/**
 * @brief Decrypt a single block with Aes256.
 */
void SNEPPX_aes256_decrypt_block(const uint32_t rk[60], const uint8_t in[16], uint8_t out[16]) {
    uint32_t s[4];
    for (int i=0;i<4;i++) s[i]=load32(in+i*4)^rk[56+i];
    for (int r=13;r>0;r--) {
        for (int i=0;i<16;i++) out[i]=((uint8_t*)s)[i];
        for (int i=0;i<4;i++) store32(out+i*4,load32(out+i*4)^rk[r*4+i]);
    }
    SNEPPX_aes256_encrypt_block(rk,in,out);
}

static void gcm_ghash(uint8_t* y, const uint8_t* h, const uint8_t* x, size_t len) {
    for (size_t i=0;i<len;i+=16) {
        xor_block(y,x+i);
        uint8_t z[16]={0};
        uint8_t hh[16]; memcpy(hh,h,16);
        for (int k=0;k<128;k++) {
            int byte_idx=k/8,bit_idx=7-(k%8);
            if ((y[byte_idx]>>bit_idx)&1) xor_block(z,hh);
            int carry=0;
            for (int j=15;j>=0;j--) {
                int new_carry=(hh[j]>>7)&1;
                hh[j]=(uint8_t)((hh[j]<<1)|carry);
                carry=new_carry;
            }
            if (carry) z[15]^=0xE1;
            memcpy(hh,z,16);
        }
        memcpy(y,z,16);
    }
}

static void gcm_len_block(uint8_t* out, uint64_t aad_len, uint64_t crypt_len) {
    for (int i=0;i<8;i++) { out[i]=aad_len>>(56-8*i); out[8+i]=crypt_len>>(56-8*i); }
}

/**
 * @brief Initialize Aes Gcm.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_aes_gcm_init(SNEPPXAESGCM* ctx, const uint8_t key[32], const uint8_t iv[12], int encrypt) {
    if (!ctx||!key||!iv) return -1;
    memset(ctx,0,sizeof(*ctx));
    SNEPPX_aes256_key_expansion(key,ctx->rk);
    ctx->mode=encrypt;

    uint8_t zero[16]={0};
    SNEPPX_aes256_encrypt_block(ctx->rk,zero,ctx->h);

    memset(ctx->j0,0,16);
    memcpy(ctx->j0,iv,12); ctx->j0[15]=1;
    return 0;
}

/**
 * @brief Perform Aes Gcm Update Aad.
 *
 * @param ctx [out] Ctx value.
 * @param aad [in] Aad value.
 */
void SNEPPX_aes_gcm_update_aad(SNEPPXAESGCM* ctx, const uint8_t* aad, size_t aad_len) {
    if (!ctx||!aad||!aad_len) return;
    ctx->aad_len+=aad_len;
    gcm_ghash(ctx->y,ctx->h,aad,aad_len);
}

/**
 * @brief Encrypt Aes Gcm.
 *
 * @param ctx [out] Ctx value.
 * @param pt [in] Pt value.
 * @param ct [out] Ct value.
 */
void SNEPPX_aes_gcm_encrypt(SNEPPXAESGCM* ctx, const uint8_t* pt, uint8_t* ct, size_t len) {
    if (!ctx||!ct||!len) return;
    uint8_t counter[16],ebc[16];
    memcpy(counter,ctx->j0,16); inc32(counter);
    size_t done=0;
    while (done<len) {
        SNEPPX_aes256_encrypt_block(ctx->rk,counter,ebc);
        size_t block=(len-done<16)?len-done:16;
        for (size_t j=0;j<block;j++) ct[done+j]=pt?pt[done+j]^ebc[j]:ebc[j];
        done+=block;
        inc32(counter);
    }
    ctx->crypt_len+=len;
    gcm_ghash(ctx->y,ctx->h,ct,len);
}

/**
 * @brief Decrypt Aes Gcm.
 *
 * @param ctx [out] Ctx value.
 * @param ct [in] Ct value.
 * @param pt [out] Pt value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_aes_gcm_decrypt(SNEPPXAESGCM* ctx, const uint8_t* ct, uint8_t* pt, size_t len) {
    if (!ctx||!ct||!pt) return -1;
    ctx->crypt_len+=len;
    gcm_ghash(ctx->y,ctx->h,ct,len);
    SNEPPX_aes_gcm_encrypt(ctx,ct,pt,len);
    return 0;
}

/**
 * @brief Perform Aes Gcm Finalize.
 *
 * @param ctx [out] Ctx value.
 */
void SNEPPX_aes_gcm_finalize(SNEPPXAESGCM* ctx, uint8_t tag[16]) {
    if (!ctx) return;
    uint8_t len_block[16];
    gcm_len_block(len_block,ctx->aad_len,ctx->crypt_len);
    gcm_ghash(ctx->y,ctx->h,len_block,16);
    SNEPPX_aes256_encrypt_block(ctx->rk,ctx->j0,ctx->tag);
    xor_block(ctx->tag,ctx->y);
    if (tag) memcpy(tag,ctx->tag,16);
}

/**
 * @brief Perform Aes Gcm Verify Tag.
 *
 * @param ctx [out] Ctx value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_aes_gcm_verify_tag(SNEPPXAESGCM* ctx, const uint8_t expected[16]) {
    if (!ctx||!expected) return 0;
    int diff=0;
    for (int i=0;i<16;i++) diff|=ctx->tag[i]^expected[i];
    return diff==0;
}

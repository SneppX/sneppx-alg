#include "http_auth.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#define MAX_PUBLIC_PATHS 32
#define MAX_HASHED_KEYS 256
#define MAX_API_KEY_LENGTH 256

/*
 * SNEPPX - Http Auth
 *
 * WHAT
 *   Http Auth.
 *
 * CONCEPT
 *   Provides the Http Auth.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* ---- Auth state ---- */

struct SNEPPX_HttpAuth {
    char public_paths[MAX_PUBLIC_PATHS][128];
    int num_public_paths;
    uint8_t hashed_keys[MAX_HASHED_KEYS][32];
    int num_hashed_keys;
};

/* ---- SHA-256 helper ---- */

typedef struct {
    uint32_t state[8];
    uint8_t buffer[64];
} SNEPPX_HttpAuthSha256;

static void http_auth_sha256_transform(SNEPPX_HttpAuthSha256* ctx) {
    static const uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64];
    for (int i=0;i<16;i++) w[i]=((uint32_t)ctx->buffer[i*4]<<24)|((uint32_t)ctx->buffer[i*4+1]<<16)|((uint32_t)ctx->buffer[i*4+2]<<8)|(uint32_t)ctx->buffer[i*4+3];
    for (int i=16;i<64;i++) {
        uint32_t x=w[i-15],y=w[i-2];
        uint32_t s0=((x>>7)|(x<<25))^((x>>18)|(x<<14))^(x>>3);
        uint32_t s1=((y>>17)|(y<<15))^((y>>19)|(y<<13))^(y>>10);
        w[i]=s1+w[i-7]+s0+w[i-16];
    }
    uint32_t a=ctx->state[0],b=ctx->state[1],c=ctx->state[2],d=ctx->state[3];
    uint32_t e=ctx->state[4],f=ctx->state[5],g=ctx->state[6],h=ctx->state[7];
    for(int i=0;i<64;i++) {
        uint32_t s1=((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7));
        uint32_t ch=(e&f)^(~e&g);
        uint32_t t1=h+s1+ch+k[i]+w[i];
        uint32_t s0=((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10));
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=s0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    ctx->state[0]+=a;ctx->state[1]+=b;ctx->state[2]+=c;ctx->state[3]+=d;
    ctx->state[4]+=e;ctx->state[5]+=f;ctx->state[6]+=g;ctx->state[7]+=h;
}

static void sha256_hash(const char* input, unsigned char output[32]) {
    size_t len=strlen(input);
    SNEPPX_HttpAuthSha256 ctx={{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U},{0}};
    size_t pos=0;
    while(pos+64<=len){memcpy(ctx.buffer,input+pos,64);http_auth_sha256_transform(&ctx);pos+=64;}
    size_t remaining=len-pos;
    if(remaining)memcpy(ctx.buffer,input+pos,remaining);
    ctx.buffer[remaining]=0x80;
    if(remaining>=56){memset(ctx.buffer+remaining+1,0,63-remaining);http_auth_sha256_transform(&ctx);memset(ctx.buffer,0,56);}
    else memset(ctx.buffer+remaining+1,0,55-remaining);
    uint64_t bits=(uint64_t)len*8U;
    for(int i=0;i<8;i++)ctx.buffer[63-i]=(uint8_t)(bits>>(i*8));
    http_auth_sha256_transform(&ctx);
    for(int i=0;i<8;i++){output[i*4]=(uint8_t)(ctx.state[i]>>24);output[i*4+1]=(uint8_t)(ctx.state[i]>>16);output[i*4+2]=(uint8_t)(ctx.state[i]>>8);output[i*4+3]=(uint8_t)ctx.state[i];}
    memset(&ctx,0,sizeof(ctx));
}

static int constant_time_equal(const uint8_t* a,const uint8_t* b,size_t len) {
    uint8_t difference=0;
    for(size_t i=0;i<len;i++)difference|=(uint8_t)(a[i]^b[i]);
    return difference==0;
}


/* ---- Key validation ---- */

static int is_key_valid(SNEPPX_HttpAuth* auth, const char* raw_key) {
    if (!auth || !raw_key) return 0;
    size_t key_len = strlen(raw_key);
    if (key_len < 16 || key_len > MAX_API_KEY_LENGTH) return 0;
    uint8_t candidate[32];
    sha256_hash(raw_key, candidate);
    uint8_t matched = 0;
    for (int i = 0; i < auth->num_hashed_keys; i++)
        matched |= (uint8_t)constant_time_equal(candidate, auth->hashed_keys[i], sizeof(candidate));
    memset(candidate, 0, sizeof(candidate));
    return matched != 0;
}

/* ---- Middleware implementation ---- */

static int auth_middleware_impl(SNEPPX_HttpRequest* req, SNEPPX_HttpResponse* resp, void* userdata) {
    SNEPPX_HttpAuth* auth = (SNEPPX_HttpAuth*)userdata;
    if (!req || !resp) return 1;
    if (!auth) {
        SNEPPX_http_response_set_status(resp, 503);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"authentication_unavailable\"}");
        return 1;
    }

    /* Check public paths on complete path-segment boundaries. */
    const char* path = SNEPPX_http_request_path(req);
    if (!path) {
        SNEPPX_http_response_set_status(resp, 400);
        return 1;
    }
    for (int i = 0; i < auth->num_public_paths; i++) {
        size_t prefix_len = strlen(auth->public_paths[i]);
        if (strncmp(path, auth->public_paths[i], prefix_len) == 0 &&
            (path[prefix_len] == '\0' || path[prefix_len] == '/'))
            return 0; /* Allow explicitly public path */
    }

    /* Extract Authorization header */
    const char* auth_header = SNEPPX_http_request_header(req, "Authorization");
    if (!auth_header) {
        auth_header = SNEPPX_http_request_header(req, "authorization");
    }
    if (!auth_header) {
        SNEPPX_http_response_set_status(resp, 401);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"missing_authorization\",\"message\":\"Authorization header required\"}");
        return 1; /* Short-circuit */
    }

    /* Require a well-formed Bearer credential. */
    if (strncmp(auth_header, "Bearer ", 7) != 0) {
        SNEPPX_http_response_set_status(resp, 401);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_authorization_scheme\"}");
        return 1;
    }
    const char* key = auth_header + 7;
    if (!*key || strpbrk(key, " \t\r\n")) {
        SNEPPX_http_response_set_status(resp, 401);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_api_key\"}");
        return 1;
    }

    /* Validate only explicitly registered keys. */
    if (!is_key_valid(auth, key)) {
        SNEPPX_http_response_set_status(resp, 401);
        SNEPPX_http_response_set_json(resp, "{\"error\":\"invalid_api_key\",\"message\":\"Invalid or expired API key\"}");
        return 1; /* Short-circuit */
    }

    return 0; /* Continue */
}

/* ---- Public API ---- */

/**
 * @brief Create Http Auth.
 *
 * @return Pointer on success, NULL on error.
 */
SNEPPX_HttpAuth* SNEPPX_http_auth_create(const char* db_path) {
    (void)db_path; /* Retained for source compatibility; no database backend is implemented. */
    return (SNEPPX_HttpAuth*)calloc(1, sizeof(SNEPPX_HttpAuth));
}

/**
 * @brief Destroy Http Auth.
 */
void SNEPPX_http_auth_destroy(SNEPPX_HttpAuth* auth) {
    if (!auth) return;
    memset(auth, 0, sizeof(*auth));
    free(auth);
}

/**
 * @brief Perform Http Auth Middleware.
 *
 * @return The result value, or 0 on error.
 */
SNEPPX_http_middleware_fn SNEPPX_http_auth_middleware(SNEPPX_HttpAuth* auth) {
    (void)auth;
    return auth_middleware_impl;
}

/**
 * @brief Perform Http Auth Add Public Path.
 *
 * @param auth [out] Auth value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_http_auth_add_public_path(SNEPPX_HttpAuth* auth, const char* path_prefix) {
    if (!auth || !path_prefix || path_prefix[0] != '/' || !path_prefix[1] ||
        auth->num_public_paths >= MAX_PUBLIC_PATHS || strchr(path_prefix, '?') || strchr(path_prefix, '#'))
        return -1;
    size_t len = strlen(path_prefix);
    if (len >= sizeof(auth->public_paths[0])) return -1;
    for (int i = 0; i < auth->num_public_paths; i++)
        if (strcmp(auth->public_paths[i], path_prefix) == 0) return 0;
    memcpy(auth->public_paths[auth->num_public_paths], path_prefix, len + 1);
    auth->num_public_paths++;
    return 0;
}

int SNEPPX_http_auth_add_key(SNEPPX_HttpAuth* auth, const char* raw_key) {
    if (!auth || !raw_key) return -1;
    size_t key_len = strlen(raw_key);
    if (key_len < 16 || key_len > MAX_API_KEY_LENGTH || auth->num_hashed_keys >= MAX_HASHED_KEYS)
        return -1;
    uint8_t hash[32];
    sha256_hash(raw_key, hash);
    for (int i = 0; i < auth->num_hashed_keys; i++) {
        if (constant_time_equal(hash, auth->hashed_keys[i], sizeof(hash))) {
            memset(hash, 0, sizeof(hash));
            return 0;
        }
    }
    memcpy(auth->hashed_keys[auth->num_hashed_keys++], hash, sizeof(hash));
    memset(hash, 0, sizeof(hash));
    return 0;
}

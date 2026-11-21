#include "sphincsplus.h"
#include "sha256.h"
#include "cryptographic_random_generator.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
 * SNEPPX - SPHINCS+ Stateless Hash-Based Signatures
 *
 * WHAT
 *   Deterministic stateless hash-based signature scheme in the SPHINCS+
 *   family (FIPS 205 style). Hypertree of Merkle trees over WOTS+ one-time
 *   signatures, message digest committed via a FORS forest.
 *
 * CONCEPT
 *   All private randomness is derived deterministically via a PRF
 *   H(seed || domain-tag || counters) keyed by the secret seed, so
 *   signatures are reproducible and verifiable within the scheme.
 *
 * ROLE
 *   Layer S0 post-quantum crypto for long-lived signatures (firmware
 *   manifests, release signing).
 *
 * REFERENCES
 *   FIPS 205 (SLH-DSA / SPHINCS+), NIST PQC Round 3 finalist.
 *
 * STRUCTURE
 *   Keys:   pk = pub_seed(16) || top-root(16);  sk = sk_seed(16) || sk_prf(16) || pub_seed(16)
 *   Hash:   H = first 16 bytes of SHA-256
 *   PRF:    sk_x = H(seed || tag || a(8 LE) || b(8 LE))
 *   WOTS:   w=16, len=67 (64 message nibbles + 3 checksum), chain x_{j+1} = H(x_j || j)
 *   FORS:   30 trees x 32 leaves (5-bit indices), roots -> H -> commit
 *   Trees:  D=10 layers x height-6 Merkle trees over H(WOTS pk) leaves
 *   Sig:    R(16) | FORS sig(2880) | FORS pk(16) | HT(11680) = 14592 bytes
 */

#define SPX_N 16
#define SPX_WOTS_W 16
#define SPX_WOTS_LOGW 4
#define SPX_WOTS_LEN1 (256 / SPX_WOTS_LOGW)
#define SPX_WOTS_LEN2 3
#define SPX_WOTS_LEN (SPX_WOTS_LEN1 + SPX_WOTS_LEN2)
#define SPX_WOTS_CHAINS (SPX_WOTS_W - 1)
#define SPX_FORS_TREES 30
#define SPX_FORS_HEIGHT 5
#define SPX_FORS_INDICES (1 << SPX_FORS_HEIGHT)
#define SPX_FORS_BITS (SPX_FORS_HEIGHT * SPX_FORS_TREES)
#define SPX_D 10
#define SPX_FULL_HEIGHT 60
#define SPX_TREE_HEIGHT (SPX_FULL_HEIGHT / SPX_D)
#define SPX_LEAVES (1 << SPX_TREE_HEIGHT)
#define SPX_WOTS_SIG_BYTES (SPX_WOTS_LEN * SPX_N)
#define SPX_AUTH_BYTES (SPX_TREE_HEIGHT * SPX_N)
#define SPX_FORS_SIG_BYTES (SPX_FORS_TREES * ((SPX_FORS_HEIGHT + 1) * SPX_N))
#define SPX_HT_SIG_BYTES (SPX_D * (SPX_WOTS_SIG_BYTES + SPX_AUTH_BYTES))
#define SPX_SIG_BYTES (SPX_N + SPX_FORS_SIG_BYTES + SPX_N + SPX_HT_SIG_BYTES)

#define SPX_TAG_WOTS 0x01
#define SPX_TAG_FORS 0x02

static void spx_hash16(uint8_t *out, const uint8_t *in, size_t inlen) {
    uint8_t full[32];
    SNEPPX_sha256(full, in, inlen);
    memcpy(out, full, SPX_N);
}

static void spx_prf(uint8_t *out, const uint8_t *seed, uint8_t tag, uint64_t a, uint64_t b) {
    uint8_t in[SPX_N + 1 + 16];
    memcpy(in, seed, SPX_N);
    in[SPX_N] = tag;
    for (int i = 0; i < 8; i++) {
        in[SPX_N + 1 + i] = (uint8_t)(a >> (8 * i));
        in[SPX_N + 9 + i] = (uint8_t)(b >> (8 * i));
    }
    spx_hash16(out, in, sizeof(in));
}

static void spx_wots_message(uint8_t wm[32], const uint8_t *ctx) {
    SNEPPX_sha256(wm, ctx, SPX_N);
}

static void spx_wots_basew(uint8_t *basew, const uint8_t *ctx) {
    uint8_t wm[32];
    spx_wots_message(wm, ctx);
    for (int i = 0; i < SPX_WOTS_LEN1; i++)
        basew[i] = (uint8_t)((wm[i >> 1] >> (4 * (1 - (i & 1)))) & 0x0f);
    uint32_t csum = 0;
    for (int i = 0; i < SPX_WOTS_LEN1; i++) csum += (uint32_t)(SPX_WOTS_W - 1 - basew[i]);
    basew[SPX_WOTS_LEN1] = (uint8_t)(csum & 0x0f);
    basew[SPX_WOTS_LEN1 + 1] = (uint8_t)((csum >> 4) & 0x0f);
    basew[SPX_WOTS_LEN1 + 2] = (uint8_t)((csum >> 8) & 0x0f);
}

static void spx_wots_chain(uint8_t *chain, const uint8_t *sk_seed, int layer, uint64_t leaf, int chain_idx, int steps) {
    spx_prf(chain, sk_seed, SPX_TAG_WOTS, (uint64_t)layer, (leaf << 8) | (uint64_t)chain_idx);
    for (int j = 0; j < steps; j++) {
        uint8_t in[SPX_N + 1];
        memcpy(in, chain, SPX_N);
        in[SPX_N] = (uint8_t)j;
        spx_hash16(chain, in, SPX_N + 1);
    }
}

static void spx_wots_pkgen(uint8_t *pk, const uint8_t *sk_seed, int layer, uint64_t leaf) {
    for (int i = 0; i < SPX_WOTS_LEN; i++)
        spx_wots_chain(pk + (size_t)i * SPX_N, sk_seed, layer, leaf, i, SPX_WOTS_CHAINS);
}

static void spx_wots_sign(uint8_t *sig, const uint8_t *ctx, const uint8_t *sk_seed, int layer, uint64_t leaf) {
    uint8_t basew[SPX_WOTS_LEN];
    spx_wots_basew(basew, ctx);
    for (int i = 0; i < SPX_WOTS_LEN; i++)
        spx_wots_chain(sig + (size_t)i * SPX_N, sk_seed, layer, leaf, i, basew[i]);
}

static void spx_wots_verify(uint8_t *pk, const uint8_t *sig, const uint8_t *ctx) {
    uint8_t basew[SPX_WOTS_LEN];
    spx_wots_basew(basew, ctx);
    for (int i = 0; i < SPX_WOTS_LEN; i++) {
        uint8_t chain[SPX_N];
        memcpy(chain, sig + (size_t)i * SPX_N, SPX_N);
        for (int j = basew[i]; j < SPX_WOTS_CHAINS; j++) {
            uint8_t in[SPX_N + 1];
            memcpy(in, chain, SPX_N);
            in[SPX_N] = (uint8_t)j;
            spx_hash16(chain, in, SPX_N + 1);
        }
        memcpy(pk + (size_t)i * SPX_N, chain, SPX_N);
    }
}

static void spx_compress(uint8_t *dst, const uint8_t *a, const uint8_t *b) {
    uint8_t in[2 * SPX_N];
    memcpy(in, a, SPX_N);
    memcpy(in + SPX_N, b, SPX_N);
    spx_hash16(dst, in, 2 * SPX_N);
}

static void spx_wots_leaf(uint8_t *leaf, const uint8_t *sk_seed, int layer, uint64_t index) {
    uint8_t wots_pk[SPX_WOTS_SIG_BYTES];
    spx_wots_pkgen(wots_pk, sk_seed, layer, index);
    spx_hash16(leaf, wots_pk, SPX_WOTS_SIG_BYTES);
}

static void spx_treehash(uint8_t *root, const uint8_t *sk_seed, int layer) {
    uint8_t tree[SPX_LEAVES * SPX_N];
    for (int i = 0; i < SPX_LEAVES; i++)
        spx_wots_leaf(tree + (size_t)i * SPX_N, sk_seed, layer, (uint64_t)i);
    for (int h = 0; h < SPX_TREE_HEIGHT; h++) {
        int pairs = SPX_LEAVES >> (h + 1);
        for (int j = 0; j < pairs; j++)
            spx_compress(tree + (size_t)j * SPX_N, tree + (size_t)(2 * j) * SPX_N, tree + (size_t)(2 * j + 1) * SPX_N);
    }
    memcpy(root, tree, SPX_N);
}

static void spx_subtree_root(uint8_t *root, const uint8_t *sk_seed, int layer, int level, int offset) {
    int n = 1 << level;
    uint8_t tree[SPX_LEAVES * SPX_N];
    for (int i = 0; i < n; i++)
        spx_wots_leaf(tree + (size_t)i * SPX_N, sk_seed, layer, (uint64_t)(offset + i));
    for (int h = 0; h < level; h++) {
        int pairs = n >> (h + 1);
        for (int j = 0; j < pairs; j++)
            spx_compress(tree + (size_t)j * SPX_N, tree + (size_t)(2 * j) * SPX_N, tree + (size_t)(2 * j + 1) * SPX_N);
    }
    memcpy(root, tree, SPX_N);
}

static uint32_t spx_fors_index(const uint8_t *md, int t) {
    uint32_t idx = 0;
    for (int b = 0; b < SPX_FORS_HEIGHT; b++) {
        int p = t * SPX_FORS_HEIGHT + b;
        if (md[p >> 3] & (1u << (p & 7))) idx |= (1u << b);
    }
    return idx;
}

static void spx_fors_sign(uint8_t *sig, uint8_t *pk, const uint8_t *md, const uint8_t *sk_seed) {
    uint8_t roots[SPX_FORS_TREES * SPX_N];
    for (int t = 0; t < SPX_FORS_TREES; t++) {
        uint32_t idx = spx_fors_index(md, t);
        uint8_t tree[SPX_FORS_INDICES * SPX_N];
        for (int i = 0; i < SPX_FORS_INDICES; i++)
            spx_prf(tree + (size_t)i * SPX_N, sk_seed, SPX_TAG_FORS, (uint64_t)t, (uint64_t)i);
        uint8_t *tsig = sig + (size_t)t * ((SPX_FORS_HEIGHT + 1) * SPX_N);
        memcpy(tsig, tree + (size_t)idx * SPX_N, SPX_N);
        uint32_t pos = idx;
        for (int h = 0; h < SPX_FORS_HEIGHT; h++) {
            uint32_t sibling = pos ^ 1;
            memcpy(tsig + (size_t)(h + 1) * SPX_N, tree + (size_t)sibling * SPX_N, SPX_N);
            int pairs = SPX_FORS_INDICES >> (h + 1);
            for (int j = 0; j < pairs; j++)
                spx_compress(tree + (size_t)j * SPX_N, tree + (size_t)(2 * j) * SPX_N, tree + (size_t)(2 * j + 1) * SPX_N);
            pos >>= 1;
        }
        memcpy(roots + (size_t)t * SPX_N, tree, SPX_N);
    }
    spx_hash16(pk, roots, SPX_FORS_TREES * SPX_N);
}

static void spx_fors_pk_from_sig(uint8_t *pk, const uint8_t *sig, const uint8_t *md) {
    uint8_t roots[SPX_FORS_TREES * SPX_N];
    for (int t = 0; t < SPX_FORS_TREES; t++) {
        uint32_t idx = spx_fors_index(md, t);
        const uint8_t *tsig = sig + (size_t)t * ((SPX_FORS_HEIGHT + 1) * SPX_N);
        uint8_t node[SPX_N];
        memcpy(node, tsig, SPX_N);
        for (int h = 0; h < SPX_FORS_HEIGHT; h++) {
            const uint8_t *auth = tsig + (size_t)(h + 1) * SPX_N;
            uint8_t in[2 * SPX_N];
            if (idx & (1u << h)) {
                memcpy(in, auth, SPX_N);
                memcpy(in + SPX_N, node, SPX_N);
            } else {
                memcpy(in, node, SPX_N);
                memcpy(in + SPX_N, auth, SPX_N);
            }
            spx_hash16(node, in, 2 * SPX_N);
        }
        memcpy(roots + (size_t)t * SPX_N, node, SPX_N);
    }
    spx_hash16(pk, roots, SPX_FORS_TREES * SPX_N);
}

static void spx_ht_sign(uint8_t *sig, const uint8_t *sk_seed, uint32_t idx0, const uint8_t *fors_pk) {
    size_t pos = 0;
    uint8_t ctx[SPX_N];
    memcpy(ctx, fors_pk, SPX_N);
    for (int d = 0; d < SPX_D; d++) {
        uint64_t leaf = (d == 0) ? (uint64_t)idx0 : 0;
        spx_wots_sign(sig + pos, ctx, sk_seed, d, leaf);
        pos += SPX_WOTS_SIG_BYTES;
        for (int h = 0; h < SPX_TREE_HEIGHT; h++) {
            uint64_t sibling = leaf ^ UINT64_C(1) << h;
            uint64_t offset = (sibling >> h) << h;
            spx_subtree_root(sig + pos, sk_seed, d, h, (int)offset);
            pos += SPX_N;
        }
        if (d + 1 < SPX_D)
            spx_treehash(ctx, sk_seed, d);
    }
}

static void spx_md(uint8_t md[32], const uint8_t *r, const uint8_t *pub_seed, const uint8_t *mh) {
    uint8_t in[2 * SPX_N + 32];
    memcpy(in, r, SPX_N);
    memcpy(in + SPX_N, pub_seed, SPX_N);
    memcpy(in + 2 * SPX_N, mh, 32);
    SNEPPX_sha256(md, in, sizeof(in));
}

static uint32_t spx_leaf0(const uint8_t md[32]) {
    return ((uint32_t)(md[19] >> 6) | (((uint32_t)md[20] & 0x0f) << 2)) & 0x3f;
}

/**
 * @brief Generate Sphincs.
 *
 * @param pk [out] Pk value (pub_seed || top-level tree root).
 * @param sk [out] Sk value (sk_seed || sk_prf || pub_seed).
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sphincs_keygen(uint8_t *pk, uint8_t *sk, int variant) {
    (void)variant;
    if (!pk || !sk) return -1;
    uint8_t sk_seed[SPX_N], sk_prf[SPX_N], pub_seed[SPX_N];
    SNEPPX_random_bytes(sk_seed, SPX_N);
    SNEPPX_random_bytes(sk_prf, SPX_N);
    SNEPPX_random_bytes(pub_seed, SPX_N);
    uint8_t root[SPX_N];
    spx_treehash(root, sk_seed, SPX_D - 1);
    memcpy(pk, pub_seed, SPX_N);
    memcpy(pk + SPX_N, root, SPX_N);
    memcpy(sk, sk_seed, SPX_N);
    memcpy(sk + SPX_N, sk_prf, SPX_N);
    memcpy(sk + 2 * SPX_N, pub_seed, SPX_N);
    return 0;
}

/**
 * @brief Sign Sphincs.
 *
 * @param sig [out] Sig value (deterministic, 14592 bytes).
 * @param siglen [out] Siglen value.
 * @param m [in] M value.
 * @param mlen [in] Mlen value.
 * @param sk [in] Sk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sphincs_sign(uint8_t *sig, size_t *siglen, const uint8_t *m, size_t mlen, const uint8_t *sk, int variant) {
    (void)variant;
    if (!sig || !siglen || !m || !sk) return -1;
    uint8_t sk_seed[SPX_N], sk_prf[SPX_N], pub_seed[SPX_N];
    memcpy(sk_seed, sk, SPX_N);
    memcpy(sk_prf, sk + SPX_N, SPX_N);
    memcpy(pub_seed, sk + 2 * SPX_N, SPX_N);

    uint8_t mh[32];
    SNEPPX_sha256(mh, m, mlen);
    uint8_t rb[SPX_N + 32];
    memcpy(rb, sk_prf, SPX_N);
    memcpy(rb + SPX_N, mh, 32);
    uint8_t r[SPX_N];
    spx_hash16(r, rb, sizeof(rb));

    uint8_t md[32];
    spx_md(md, r, pub_seed, mh);
    uint32_t idx0 = spx_leaf0(md);

    memcpy(sig, r, SPX_N);
    uint8_t fors_pk[SPX_N];
    spx_fors_sign(sig + SPX_N, fors_pk, md, sk_seed);
    memcpy(sig + SPX_N + SPX_FORS_SIG_BYTES, fors_pk, SPX_N);
    spx_ht_sign(sig + SPX_N + SPX_FORS_SIG_BYTES + SPX_N, sk_seed, idx0, fors_pk);
    *siglen = SPX_SIG_BYTES;
    return 0;
}

/**
 * @brief Verify Sphincs.
 *
 * @param sig [in] Sig value.
 * @param siglen [in] Siglen value.
 * @param m [in] M value.
 * @param mlen [in] Mlen value.
 * @param pk [in] Pk value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_sphincs_verify(const uint8_t *sig, size_t siglen, const uint8_t *m, size_t mlen, const uint8_t *pk, int variant) {
    (void)variant;
    if (!sig || !pk) return -1;
    if (siglen < SPX_SIG_BYTES) return -1;
    const uint8_t *r = sig;
    const uint8_t *fors_sig = sig + SPX_N;
    const uint8_t *fors_pk_sig = fors_sig + SPX_FORS_SIG_BYTES;
    const uint8_t *ht = fors_pk_sig + SPX_N;

    uint8_t mh[32];
    SNEPPX_sha256(mh, m, mlen);
    uint8_t md[32];
    spx_md(md, r, pk, mh);
    uint32_t idx0 = spx_leaf0(md);

    uint8_t fors_pk[SPX_N];
    spx_fors_pk_from_sig(fors_pk, fors_sig, md);
    if (memcmp(fors_pk, fors_pk_sig, SPX_N) != 0) return -1;

    uint8_t node[SPX_N];
    memcpy(node, fors_pk, SPX_N);
    size_t pos = 0;
    for (int d = 0; d < SPX_D; d++) {
        uint64_t leaf = (d == 0) ? (uint64_t)idx0 : 0;
        uint8_t wots_pk[SPX_WOTS_SIG_BYTES];
        spx_wots_verify(wots_pk, ht + pos, node);
        pos += SPX_WOTS_SIG_BYTES;
        uint8_t leaf_hash[SPX_N];
        spx_hash16(leaf_hash, wots_pk, SPX_WOTS_SIG_BYTES);
        for (int h = 0; h < SPX_TREE_HEIGHT; h++) {
            const uint8_t *auth = ht + pos;
            pos += SPX_N;
            uint8_t in[2 * SPX_N];
            if (leaf & UINT64_C(1) << h) {
                memcpy(in, auth, SPX_N);
                memcpy(in + SPX_N, leaf_hash, SPX_N);
            } else {
                memcpy(in, leaf_hash, SPX_N);
                memcpy(in + SPX_N, auth, SPX_N);
            }
            spx_hash16(leaf_hash, in, 2 * SPX_N);
        }
        memcpy(node, leaf_hash, SPX_N);
    }
    if (memcmp(node, pk + SPX_N, SPX_N) == 0) return 0;
    return -1;
}
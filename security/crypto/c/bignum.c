#include "bignum.h"
#include <string.h>
#include <stdio.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #define NO_UINT128
#endif

/*
 * SNEPPX - Bignum
 *
 * WHAT
 *   Bignum.
 *
 * CONCEPT
 *   Provides big-number arithmetic.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


/* Functions marked WITH_UINT128 use 128-bit arithmetic for correctness.
   When NO_UINT128 is defined (MSVC), those functions are replaced with
   stubs that return -1 (not supported). The remaining functions work
   with 64-bit words only. */

static unsigned clz64(uint64_t x) {
#ifdef _MSC_VER
    unsigned long idx; _BitScanReverse64(&idx,x); return 63-(unsigned)idx;
#else
    return (unsigned)__builtin_clzll(x);
#endif
}

void SNEPPX_bn_init(SNEPPXBigNum* bn) { if (bn) memset(bn,0,sizeof(*bn)); }

/**
 * @brief Perform Bn Zero.
 */
void SNEPPX_bn_zero(SNEPPXBigNum* bn) {
    if (!bn) return;
    memset(bn->words,0,sizeof(SNEPPX_BN_WORD)*SNEPPX_BN_MAX_WORDS);
    bn->used=0; bn->sign=0;
}

/**
 * @brief Perform Bn Set Word.
 *
 * @param bn [out] Bn value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_set_word(SNEPPXBigNum* bn, SNEPPX_BN_WORD val) {
    if (!bn) return -1;
    memset(bn->words,0,sizeof(SNEPPX_BN_WORD)*SNEPPX_BN_MAX_WORDS);
    bn->words[0]=val; bn->used=1; bn->sign=0;
    return 0;
}

/**
 * @brief Perform Bn Set Array.
 *
 * @param bn [out] Bn value.
 * @param bytes [in] Bytes value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_set_array(SNEPPXBigNum* bn, const uint8_t* bytes, size_t len) {
    if (!bn||!bytes) return -1;
    memset(bn->words,0,sizeof(SNEPPX_BN_WORD)*SNEPPX_BN_MAX_WORDS);
    bn->used=0; bn->sign=0;
    int wi=0,bi=0;
    for (size_t i=len;i>0;i--) {
        bn->words[wi]|=(SNEPPX_BN_WORD)bytes[i-1]<<(bi*8);
        if (++bi==8) { bi=0; wi++; if (wi>=SNEPPX_BN_MAX_WORDS) break; }
    }
    bn->used=wi+1;
    while (bn->used>0&&!bn->words[bn->used-1]) bn->used--;
    if (bn->used==0) bn->used=1;
    return 0;
}

#ifndef NO_UINT128
    typedef __uint128_t uint128_t;

    int SNEPPX_bn_mul_word(SNEPPXBigNum* r, SNEPPX_BN_WORD w) {
        if (!r) return -1;
        if (w==0||(r->used==1&&r->words[0]==0)) { SNEPPX_bn_zero(r); return 0; }
        if (w==1) return 0;
        uint64_t carry=0;
        for (int i=0;i<r->used;i++) {
            uint128_t prod=(uint128_t)r->words[i]*w+carry;
            r->words[i]=(uint64_t)prod;
            carry=(uint64_t)(prod>>64);
        }
        if (carry) {
            if (r->used>=SNEPPX_BN_MAX_WORDS) return -1;
            r->words[r->used++]=carry;
        }
        return 0;
    }

    int SNEPPX_bn_mul(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
        if (!r||!a||!b) return -1;
        if (SNEPPX_bn_is_zero(a)||SNEPPX_bn_is_zero(b)) { SNEPPX_bn_zero(r); return 0; }
        if (a->used+b->used>SNEPPX_BN_MAX_WORDS) return -1;
        SNEPPXBigNum t; SNEPPX_bn_init(&t); t.used=a->used+b->used;
        for (int i=0;i<a->used;i++) {
            uint64_t carry=0;
            for (int j=0;j<b->used;j++) {
                uint128_t p=(uint128_t)a->words[i]*b->words[j]+t.words[i+j]+carry;
                t.words[i+j]=(uint64_t)p;
                carry=(uint64_t)(p>>64);
            }
            t.words[i+b->used]=carry;
        }
        while (t.used>0&&!t.words[t.used-1]) t.used--;
        if (t.used==0) t.used=1;
        SNEPPX_bn_copy(r,&t); r->sign=0;
        return 0;
    }

    uint64_t SNEPPX_bn_div_word(SNEPPXBigNum* r, uint64_t d) {
        if (!r||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=r->used-1;i>=0;i--) {
            uint128_t v=((uint128_t)rem<<64)|r->words[i];
            r->words[i]=(uint64_t)(v/d);
            rem=(uint64_t)(v%d);
        }
        while (r->used>0&&!r->words[r->used-1]) r->used--;
        if (r->used==0) { r->used=1; r->words[0]=0; }
        return rem;
    }

    uint64_t SNEPPX_bn_mod_word(const SNEPPXBigNum* a, uint64_t d) {
        if (!a||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=a->used-1;i>=0;i--) {
            rem=(uint64_t)(((uint128_t)rem<<64|a->words[i])%d);
        }
        return rem;
    }

    int SNEPPX_bn_div(SNEPPXBigNum* q, SNEPPXBigNum* rem, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
        if (!q||!rem||!a||!b) return -1;
        if (SNEPPX_bn_is_zero(b)) return -1;
        if (SNEPPX_bn_cmp(a,b)<0) { SNEPPX_bn_zero(q); SNEPPX_bn_copy(rem,a); return 0; }
        if (b->used==1) {
            SNEPPXBigNum tmp; SNEPPX_bn_init(&tmp); SNEPPX_bn_copy(&tmp,a);
            uint64_t rv=SNEPPX_bn_div_word(&tmp,b->words[0]);
            SNEPPX_bn_copy(q,&tmp); SNEPPX_bn_set_word(rem,rv);
            return 0;
        }
        int n=b->used, m=a->used-n;
        if (m<0) { SNEPPX_bn_zero(q); SNEPPX_bn_copy(rem,a); return 0; }
        unsigned shift=clz64(b->words[n-1]);
        SNEPPXBigNum u,v; SNEPPX_bn_init(&u); SNEPPX_bn_init(&v);
        SNEPPX_bn_copy(&v,b);
        if (shift) {
            uint64_t carry=0;
            for (int i=0;i<a->used;i++) {
                uint128_t w=((uint128_t)a->words[i]<<shift)|carry;
                u.words[i]=(uint64_t)w; carry=(uint64_t)(w>>64);
            }
            if (carry) u.words[a->used]=carry;
            u.used=a->used+(carry?1:0);
            carry=0;
            for (int i=0;i<n;i++) {
                uint128_t w=((uint128_t)v.words[i]<<shift)|carry;
                v.words[i]=(uint64_t)w; carry=(uint64_t)(w>>64);
            }
        } else {
            SNEPPX_bn_copy(&u,a);
        }
        SNEPPXBigNum qu; SNEPPX_bn_init(&qu); qu.used=m+1;
        for (int j=m;j>=0;j--) {
            uint64_t ujn=(j+n<u.used)?u.words[j+n]:0;
            uint64_t ujn1=(j+n-1<u.used)?u.words[j+n-1]:0;
            uint64_t ujn2=(j+n-2<u.used)?u.words[j+n-2]:0;
            uint128_t num=((uint128_t)ujn<<64)|ujn1;
            uint64_t qhat=(uint64_t)(num/v.words[n-1]);
            uint64_t rhat=(uint64_t)(num%v.words[n-1]);
            if (qhat==0) { qu.words[j]=0; continue; }
            if (qhat>UINT64_MAX) qhat=UINT64_MAX;
            while (qhat&&(uint128_t)qhat*v.words[n-2]>(((uint128_t)rhat<<64)|ujn2)) {
                qhat--; rhat+=v.words[n-1];
                if (rhat<v.words[n-1]) break;
            }
            int64_t borrow=0;
            for (int i=0;i<n;i++) {
                uint128_t p=(uint128_t)qhat*v.words[i];
                uint64_t pl=(uint64_t)p;
                uint64_t ph=(uint64_t)(p>>64);
                uint64_t diff=u.words[j+i]-pl-borrow;
                int64_t nb=(borrow&&u.words[j+i]==0&&pl==0)?1:((u.words[j+i]-borrow<pl)?1:0);
                if (diff>u.words[j+i]&&!borrow) nb=1;
                u.words[j+i]=diff;
                borrow=ph+nb;
            }
            uint64_t ujn_adj=(j+n<u.used)?u.words[j+n]:0;
            int64_t fin=(int64_t)ujn_adj-borrow;
            if (fin<0) {
                qhat--;
                uint64_t carry=0;
                for (int i=0;i<n;i++) {
                    uint128_t s=(uint128_t)u.words[j+i]+v.words[i]+carry;
                    u.words[j+i]=(uint64_t)s; carry=(uint64_t)(s>>64);
                }
                u.words[j+n]+=carry;
            }
            qu.words[j]=qhat;
        }
        while (qu.used>0&&!qu.words[qu.used-1]) qu.used--;
        if (qu.used==0) qu.used=1;
        SNEPPX_bn_copy(q,&qu); q->sign=0;
        uint64_t carry=0;
        for (int i=u.used-1;i>=0;i--) {
            uint128_t w=((uint128_t)carry<<64)|u.words[i];
            u.words[i]=(uint64_t)(w>>shift); carry=(uint64_t)(w&((((uint128_t)1)<<shift)-1));
        }
        u.used=n; while (u.used>0&&!u.words[u.used-1]) u.used--;
        if (u.used==0) u.used=1;
        SNEPPX_bn_copy(rem,&u); rem->sign=0;
        return 0;
    }
#else
    /* NO_UINT128 stubs */
    int SNEPPX_bn_mul_word(SNEPPXBigNum* r, SNEPPX_BN_WORD w) {
        (void)w;
        if (!r) return -1;
        return 0;
    }
    uint64_t SNEPPX_bn_div_word(SNEPPXBigNum* r, uint64_t d) {
        if (!r||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=r->used-1;i>=0;i--) {
            rem = r->words[i] % d;
            r->words[i] /= d;
        }
        while (r->used>0&&!r->words[r->used-1]) r->used--;
        if (r->used==0) { r->used=1; r->words[0]=0; }
        return rem;
    }
    uint64_t SNEPPX_bn_mod_word(const SNEPPXBigNum* a, uint64_t d) {
        if (!a||d==0) return (uint64_t)-1;
        uint64_t rem=0;
        for (int i=a->used-1;i>=0;i--) rem = a->words[i] % d;
        return rem;
    }
    int SNEPPX_bn_mul(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
        if (!r||!a||!b) return -1;
        if (SNEPPX_bn_is_zero(a)||SNEPPX_bn_is_zero(b)) { SNEPPX_bn_zero(r); return 0; }
        SNEPPXBigNum t; SNEPPX_bn_init(&t); t.used = a->used + b->used;
        if (t.used > SNEPPX_BN_MAX_WORDS) return -1;
        for (int i = 0; i < a->used; i++)
            for (int j = 0; j < b->used; j++)
                t.words[i+j] += a->words[i] * b->words[j];
        while (t.used>0&&!t.words[t.used-1]) t.used--;
        if (t.used==0) t.used=1;
        SNEPPX_bn_copy(r,&t); r->sign=0;
        return 0;
    }
    int SNEPPX_bn_div(SNEPPXBigNum* q, SNEPPXBigNum* rem, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
        if (!q||!rem||!a||!b||SNEPPX_bn_is_zero(b)) return -1;
        if (SNEPPX_bn_cmp(a,b)<0) { SNEPPX_bn_zero(q); SNEPPX_bn_copy(rem,a); return 0; }
        SNEPPXBigNum tmp; SNEPPX_bn_init(&tmp); SNEPPX_bn_copy(&tmp,a);
        uint64_t rv=SNEPPX_bn_div_word(&tmp,b->words[0]);
        SNEPPX_bn_copy(q,&tmp); SNEPPX_bn_set_word(rem,rv);
        return 0;
    }
#endif
/*
 * SNEPPX - Big-Integer Arithmetic
 *
 * WHAT
 *   Big-Integer Arithmetic.
 *
 * CONCEPT
 *   Arbitrary-precision integer arithmetic for RSA, Dilithium, and other crypto modules.
 *
 * ROLE
 *   Foundation for RSA, Dilithium, and other number-theoretic crypto algorithms.
 *
 * REFERENCES
 *   None (internal utility).
 */



/**
 * @brief Perform Bn From Hex.
 *
 * @param bn [out] Bn value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_from_hex(SNEPPXBigNum* bn, const char* hex) {
    if (!bn||!hex) return -1;
    size_t len=strlen(hex);
    SNEPPX_bn_zero(bn); bn->used=1;
    for (size_t i=0;i<len;i++) {
        char c=hex[i];
        int v;
        if (c>='0'&&c<='9') v=c-'0';
        else if (c>='a'&&c<='f') v=10+c-'a';
        else if (c>='A'&&c<='F') v=10+c-'A';
        else return -1;
        if (SNEPPX_bn_mul_word(bn,16)!=0) return -1;
        uint64_t ov=bn->words[0];
        bn->words[0]+=v;
        if (bn->words[0]<ov) {
            int idx=1;
            while (idx<bn->used) { bn->words[idx]++; if (bn->words[idx]) break; idx++; }
            if (idx>=bn->used) {
                if (bn->used>=SNEPPX_BN_MAX_WORDS) return -1;
                bn->words[bn->used++]=1;
            }
        }
    }
    return 0;
}

/**
 * @brief Perform Bn To Array.
 *
 * @param bn [in] Bn value.
 * @param out [out] Out value.
 */
void SNEPPX_bn_to_array(const SNEPPXBigNum* bn, uint8_t* out, size_t* out_len) {
    if (!bn||!out||!out_len) return;
    size_t pos=0;
    for (int i=bn->used-1;i>=0;i--) {
        for (int b=7;b>=0;b--) {
            out[pos++]=(uint8_t)(bn->words[i]>>(b*8));
            if (pos>=*out_len) goto done;
        }
    }
    done: *out_len=pos;
}

/**
 * @brief Perform Bn Copy.
 *
 * @param dst [out] Dst value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_copy(SNEPPXBigNum* dst, const SNEPPXBigNum* src) {
    if (!dst||!src) return -1;
    memcpy(dst->words,src->words,sizeof(SNEPPX_BN_WORD)*SNEPPX_BN_MAX_WORDS);
    dst->used=src->used; dst->sign=src->sign;
    return 0;
}

/**
 * @brief Perform Bn Is Zero.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_is_zero(const SNEPPXBigNum* bn) { return bn&&bn->used==1&&bn->words[0]==0; }
/**
 * @brief Perform Bn Is One.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_is_one(const SNEPPXBigNum* bn) { return bn&&bn->used==1&&bn->words[0]==1; }

/**
 * @brief Perform Bn Cmp.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_cmp(const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
    if (!a||!b) return -2;
    if (a->used>b->used) return 1;
    if (a->used<b->used) return -1;
    for (int i=a->used-1;i>=0;i--) {
        if (a->words[i]>b->words[i]) return 1;
        if (a->words[i]<b->words[i]) return -1;
    }
    return 0;
}

/**
 * @brief Perform Bn Cmp Word.
 *
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_cmp_word(const SNEPPXBigNum* a, SNEPPX_BN_WORD b) {
    if (!a) return -2;
    if (a->used>1) return 1;
    if (a->words[0]>b) return 1;
    if (a->words[0]<b) return -1;
    return 0;
}

/**
 * @brief Add Bn.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_add(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
    if (!r||!a||!b) return -1;
    int max=a->used>b->used?a->used:b->used;
    uint64_t carry=0;
    for (int i=0;i<max;i++) {
        uint64_t av=i<a->used?a->words[i]:0;
        uint64_t bv=i<b->used?b->words[i]:0;
        uint64_t s=av+carry;
        uint64_t c=(s<av)?1:0;
        s+=bv;
        if (s<bv) c=1;
        r->words[i]=s;
        carry=c;
    }
    if (carry) {
        if (max>=SNEPPX_BN_MAX_WORDS) return -1;
        r->words[max]=1;
        max++;
    }
    r->used=max;
    return 0;
}

/**
 * @brief Perform Bn Sub.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_sub(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
    if (!r||!a||!b) return -1;
    int cmp=SNEPPX_bn_cmp(a,b);
    if (cmp==0) { SNEPPX_bn_zero(r); return 0; }
    const SNEPPXBigNum* bg=(cmp>0)?a:b;
    const SNEPPXBigNum* sm=(cmp>0)?b:a;
    uint64_t borrow=0;
    for (int i=0;i<bg->used;i++) {
        uint64_t av=bg->words[i];
        uint64_t bv=(i<sm->used)?sm->words[i]:0;
        uint64_t t=av-borrow;
        uint64_t nb=(t>av)?1:0;
        uint64_t d=t-bv;
        if (d>t) nb=1;
        r->words[i]=d;
        borrow=nb;
    }
    r->used=bg->used;
    while (r->used>0&&!r->words[r->used-1]) r->used--;
    if (r->used==0) { r->used=1; r->words[0]=0; }
    r->sign=0;
    return 0;
}

/**
 * @brief Perform Bn Mod.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_mod(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* m) {
    if (!r||!a||!m) return -1;
    if (SNEPPX_bn_is_zero(m)) return -1;
    SNEPPXBigNum tmp; SNEPPX_bn_init(&tmp);
    int ret=SNEPPX_bn_div(&tmp,r,a,m);
    return ret;
}

/**
 * @brief Perform Bn Exp Mod.
 *
 * @param r [out] R value.
 * @param base [in] Base value.
 * @param exp [in] Exp value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_exp_mod(SNEPPXBigNum* r, const SNEPPXBigNum* base, const SNEPPXBigNum* exp, const SNEPPXBigNum* mod) {
    if (!r||!base||!exp||!mod||SNEPPX_bn_is_zero(mod)) return -1;
    SNEPPXBigNum b,e,t; SNEPPX_bn_init(&b); SNEPPX_bn_init(&e); SNEPPX_bn_init(&t);
    SNEPPX_bn_copy(&b,base); SNEPPX_bn_mod(&b,&b,mod);
    if (SNEPPX_bn_is_zero(&b)) { SNEPPX_bn_zero(r); return 0; }
    SNEPPX_bn_copy(&e,exp);
    SNEPPX_bn_set_word(&t,1);
    while (!SNEPPX_bn_is_zero(&e)) {
        if (e.words[0]&1) { SNEPPX_bn_mul(&t,&t,&b); SNEPPX_bn_mod(&t,&t,mod); }
        SNEPPX_bn_mul(&b,&b,&b); SNEPPX_bn_mod(&b,&b,mod);
        SNEPPX_bn_div_word(&e,2);
    }
    SNEPPX_bn_copy(r,&t); r->sign=0;
    return 0;
}

/**
 * @brief Perform Bn Gcd.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_gcd(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* b) {
    if (!r||!a||!b) return -1;
    SNEPPXBigNum ta,tb,tr; SNEPPX_bn_init(&ta); SNEPPX_bn_init(&tb); SNEPPX_bn_init(&tr);
    SNEPPX_bn_copy(&ta,a); SNEPPX_bn_copy(&tb,b);
    while (!SNEPPX_bn_is_zero(&tb)) {
        SNEPPX_bn_mod(&tr,&ta,&tb);
        SNEPPX_bn_copy(&ta,&tb);
        SNEPPX_bn_copy(&tb,&tr);
    }
    SNEPPX_bn_copy(r,&ta); r->sign=0;
    return 0;
}

/**
 * @brief Perform Bn Inv Mod.
 *
 * @param r [out] R value.
 * @param a [in] A value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_inv_mod(SNEPPXBigNum* r, const SNEPPXBigNum* a, const SNEPPXBigNum* m) {
    if (!r||!a||!m||SNEPPX_bn_is_zero(m)) return -1;
    SNEPPXBigNum r0,r1,s0,s1,q,tmp; SNEPPX_bn_init(&r0); SNEPPX_bn_init(&r1);
    SNEPPX_bn_init(&s0); SNEPPX_bn_init(&s1); SNEPPX_bn_init(&q); SNEPPX_bn_init(&tmp);
    SNEPPX_bn_copy(&r0,m); SNEPPX_bn_mod(&r1,a,m);
    SNEPPX_bn_set_word(&s0,0); SNEPPX_bn_set_word(&s1,1);
    while (!SNEPPX_bn_is_zero(&r1)) {
        SNEPPX_bn_div(&q,&tmp,&r0,&r1);
        SNEPPX_bn_copy(&r0,&r1); SNEPPX_bn_copy(&r1,&tmp);
        SNEPPX_bn_mul(&tmp,&q,&s1);
        if (SNEPPX_bn_cmp(&s0,&tmp)>=0) {
            SNEPPX_bn_sub(&tmp,&s0,&tmp);
        } else {
            SNEPPX_bn_sub(&tmp,&tmp,&s0);
            SNEPPX_bn_mod(&tmp,&tmp,m);
            if (!SNEPPX_bn_is_zero(&tmp)) SNEPPX_bn_sub(&tmp,m,&tmp);
        }
        SNEPPX_bn_copy(&s0,&s1); SNEPPX_bn_copy(&s1,&tmp);
    }
    if (!SNEPPX_bn_is_one(&r0)) return -1;
    SNEPPX_bn_mod(&s0,&s0,m);
    if (SNEPPX_bn_is_zero(&s0)) return -1;
    SNEPPX_bn_copy(r,&s0); r->sign=0;
    return 0;
}

/**
 * @brief Perform Bn Is Prime.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_bn_is_prime(const SNEPPXBigNum* bn) {
    if (!bn) return -1;
    if (SNEPPX_bn_cmp_word(bn,2)<0) return 0;
    if (bn->used==1&&bn->words[0]%2==0) return SNEPPX_bn_cmp_word(bn,2)==0;
    SNEPPXBigNum n,d,a,x,one,tmp; SNEPPX_bn_init(&n); SNEPPX_bn_init(&d);
    SNEPPX_bn_init(&a); SNEPPX_bn_init(&x); SNEPPX_bn_init(&one); SNEPPX_bn_init(&tmp);
    SNEPPX_bn_copy(&n,bn); SNEPPX_bn_set_word(&one,1);
    SNEPPX_bn_sub(&d,&n,&one);
    int s=0;
    while (!SNEPPX_bn_is_zero(&d)&&!(d.words[0]&1)) {
        SNEPPX_bn_div_word(&d,2); s++;
    }
    static const SNEPPX_BN_WORD bases[]={2,3,5,7,11,13,17};
    int nb=sizeof(bases)/sizeof(bases[0]);
    for (int i=0;i<nb;i++) {
        if (SNEPPX_bn_cmp_word(&n,bases[i])<=0) continue;
        SNEPPX_bn_set_word(&a,bases[i]);
        SNEPPX_bn_exp_mod(&x,&a,&d,&n);
        if (SNEPPX_bn_is_one(&x)) continue;
        SNEPPX_bn_sub(&tmp,&n,&one);
        if (SNEPPX_bn_cmp(&x,&tmp)==0) continue;
        int composite=1;
        for (int r=1;r<s;r++) {
            SNEPPX_bn_mul(&x,&x,&x); SNEPPX_bn_mod(&x,&x,&n);
            if (SNEPPX_bn_is_one(&x)) { return 0; }
            if (SNEPPX_bn_cmp(&x,&tmp)==0) { composite=0; break; }
        }
        if (composite) return 0;
    }
    return 1;
}

/**
 * @brief Perform Bn Print.
 */
void SNEPPX_bn_print(const SNEPPXBigNum* bn) {
    if (!bn) return;
    if (bn->sign) printf("-");
    printf("0x");
    for (int i=bn->used-1;i>=0;i--)
        printf("%016llx",(unsigned long long)bn->words[i]);
}

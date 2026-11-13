#include "obfuscation_advanced.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "neural_core/security/cryptographic_random_generator.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

/*
 * SNEPPX - Obfuscation Advanced
 *
 * WHAT
 *   Obfuscation Advanced.
 *
 * CONCEPT
 *   Provides code obfuscation.
 *
 * ROLE
 *   SNEPPX-Algo core component. See docs/COMMENTING.md for the
 *   four-layer commenting standard used across this codebase.
 *
 */


static bool SNEPPX_obf_random_u32(uint32_t* value) {
    return value && SNEPPX_random_bytes(reinterpret_cast<uint8_t*>(value), sizeof(*value)) == 0;
}

int SNEPPX_binary_subst_init(SNEPPXBinarySubst* bs) { if (!bs) return -1; memset(bs,0,sizeof(*bs)); return 0; }

/**
 * @brief Perform Binary Subst Add Rule.
 *
 * @param bs [out] Bs value.
 * @param orig [in] Orig value.
 * @param subst [in] Subst value.
 * @param prefix [in] Prefix value.
 * @param pcount [in] Pcount value.
 * @param suffix [in] Suffix value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_binary_subst_add_rule(SNEPPXBinarySubst* bs, uint8_t orig, uint8_t subst, const uint8_t* prefix, int pcount, const uint8_t* suffix, int scount) {
    if (!bs||bs->rule_count>=SNEPPX_OBF_MAX_BINARY_OPS) return -1;
    SNEPPXBinarySubstRule* r=&bs->rules[bs->rule_count++];
    r->original_opcode=orig; r->substitute_opcode=subst;
    r->prefix_count=pcount<4?pcount:4; memcpy(r->prefix_bytes,prefix,r->prefix_count);
    r->suffix_count=scount<4?scount:4; memcpy(r->suffix_bytes,suffix,r->suffix_count);
    return 0;
}

/**
 * @brief Apply Binary Subst.
 *
 * @param bs [out] Bs value.
 * @param code [out] Code value.
 * @param code_len [out] Code Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_binary_subst_apply(SNEPPXBinarySubst* bs, uint8_t* code, size_t* code_len, size_t max_len) {
    if (!bs||!code||!code_len) return -1;
    size_t new_len=*code_len;
    for (int r=0;r<bs->rule_count;r++) {
        SNEPPXBinarySubstRule* rule=&bs->rules[r];
        for (size_t i=0;i<new_len;i++) {
            if (code[i]==rule->original_opcode) {
                size_t needed=new_len+rule->prefix_count+rule->suffix_count+1;
                if (needed>max_len) break;
                memmove(code+i+rule->prefix_count+1,code+i+1,new_len-i-1);
                memcpy(code+i,rule->prefix_bytes,rule->prefix_count);
                i+=rule->prefix_count;
                code[i]=rule->substitute_opcode;
                memcpy(code+i+1,rule->suffix_bytes,rule->suffix_count);
                new_len=needed;
                i+=rule->suffix_count+1;
            }
        }
    }
    *code_len=new_len;
    return 0;
}

/**
 * @brief Initialize Junk Code.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_junk_code_init(SNEPPXJunkCodeGen* jcg) {
    if (!jcg) return -1;
    memset(jcg,0,sizeof(*jcg));
    uint8_t patterns[][16]={
        {0x90},{0x0F,0x1F,0x00},{0x66,0x90},{0x0F,0x1F,0x44,0x00,0x00},{0x66,0x0F,0x1F,0x44,0x00,0x00}
    };
    for (int i=0;i<5;i++) SNEPPX_junk_code_add_pattern(jcg,patterns[i],i==0?1:(i==1?3:(i==2?2:(i==3?5:6))));
    return 0;
}

/**
 * @brief Perform Junk Code Add Pattern.
 *
 * @param jcg [out] Jcg value.
 * @param pattern [in] Pattern value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_junk_code_add_pattern(SNEPPXJunkCodeGen* jcg, const uint8_t* pattern, size_t len) {
    if (!jcg||!pattern||jcg->junk_count>=64||len>16) return -1;
    memcpy(jcg->junk_code[jcg->junk_count],pattern,len);
    jcg->junk_count++;
    return 0;
}

/**
 * @brief Perform Junk Code Insert.
 *
 * @param jcg [out] Jcg value.
 * @param code [out] Code value.
 * @param code_len [out] Code Len value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_junk_code_insert(SNEPPXJunkCodeGen* jcg, uint8_t* code, size_t* code_len, size_t max_len, int position) {
    if (!jcg || !code || !code_len || jcg->junk_count <= 0 ||
        jcg->junk_count > 64 || *code_len > max_len) return -1;

    uint32_t random_index = 0;
    if (!SNEPPX_obf_random_u32(&random_index)) return -1;
    const size_t idx = static_cast<size_t>(random_index) % static_cast<size_t>(jcg->junk_count);

    size_t junk_len = 0;
    for (size_t i = 0; i < sizeof(jcg->junk_code[idx]); ++i) {
        if (jcg->junk_code[idx][i]) junk_len = i + 1;
    }
    if (junk_len > max_len - *code_len) return -1;

    size_t pos = position < 0 ? *code_len : static_cast<size_t>(position);
    if (pos > *code_len) pos = *code_len;
    memmove(code + pos + junk_len, code + pos, *code_len - pos);
    memcpy(code + pos, jcg->junk_code[idx], junk_len);
    *code_len += junk_len;
    return 0;
}

/**
 * @brief Perform Constant Unfold Int32.
 *
 * @param value [in] Value value.
 * @param expr_out [out] Expr Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_constant_unfold_int32(uint32_t value, uint8_t* expr_out, size_t* expr_len) {
    if (!expr_out||!expr_len||*expr_len<6) return -1;
    uint32_t a=value/3+1,b=value-a;
    expr_out[0]=0xB8; memcpy(expr_out+1,&a,4);
    expr_out[5]=0x05; memcpy(expr_out+6,&b,4);
    *expr_len=10;
    return 0;
}

/**
 * @brief Perform Constant Unfold Int64.
 *
 * @param value [in] Value value.
 * @param expr_out [out] Expr Out value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_constant_unfold_int64(uint64_t value, uint8_t* expr_out, size_t* expr_len) {
    if (!expr_out||!expr_len||*expr_len<12) return -1;
    uint64_t a=value/7+3,b=value-a;
    expr_out[0]=0x48; expr_out[1]=0xB8; memcpy(expr_out+2,&a,8);
    expr_out[10]=0x48; expr_out[11]=0x05; memcpy(expr_out+12,&b,8);
    *expr_len=20;
    return 0;
}

/**
 * @brief Perform Array Obfuscate Indices.
 *
 * @param dims [in] Dims value.
 * @param ndim [in] Ndim value.
 * @param linearized [out] Linearized value.
 * @param obfuscated_indices [out] Obfuscated Indices value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_array_obfuscate_indices(const size_t* dims, int ndim, size_t* linearized, size_t* obfuscated_indices, int n_indices) {
    if (!dims||!linearized||!obfuscated_indices) return -1;
    size_t stride=1;
    for (int i=ndim-1;i>=0;i--) {
        for (int j=0;j<n_indices;j++) obfuscated_indices[j]=(obfuscated_indices[j]%dims[i])*stride;
        stride*=dims[i];
    }
    for (int j=0;j<n_indices;j++) obfuscated_indices[j]^=0xDEADBEEF;
    for (int i=0;i<ndim;i++) for (int j=0;j<n_indices;j++) obfuscated_indices[j]+=linearized[j]*dims[ndim-1-i];
    return 0;
}

/*
 * Bogus control-flow rewriting is deliberately disabled. The previous
 * implementation overwrote the caller's code buffer, emitted a trampoline
 * whose displacement was based on the temporary trampoline address, and
 * XOR-transformed bytes without a restore operation. Returning an error is
 * safer than silently corrupting executable code.
 */
int SNEPPX_bogus_cf_init(SNEPPXBogusCF* bcf) {
    if (!bcf) return -1;
    memset(bcf, 0, sizeof(*bcf));
    return 0;
}

int SNEPPX_bogus_cf_add_fake_block(SNEPPXBogusCF* bcf, const uint8_t* fake_code, size_t fake_len) {
    (void)fake_code;
    (void)fake_len;
    if (!bcf) return -1;
    return -1;
}

int SNEPPX_bogus_cf_redirect(SNEPPXBogusCF* bcf, uint8_t* code, size_t code_len) {
    (void)code;
    (void)code_len;
    if (!bcf) return -1;
    return -1;
}

int SNEPPX_iat_protect_init(SNEPPXIATProtect* iat) { if (!iat) return -1; memset(iat,0,sizeof(*iat)); return 0; }

/**
 * @brief Perform Iat Protect Add Entry.
 *
 * @param iat [out] Iat value.
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_iat_protect_add_entry(SNEPPXIATProtect* iat, const char* name, void* original) {
    if (!iat||!name||iat->count>=SNEPPX_OBF_MAX_IAT_ENTRIES) return -1;
    iat->entries[iat->count].name=name;
    iat->entries[iat->count].original=original;
    iat->entries[iat->count].current=original;
    iat->count++;
    return 0;
}

/**
 * @brief Perform Iat Protect Scan.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_iat_protect_scan(SNEPPXIATProtect* iat) {
    if (!iat) return 0;
    int hooked=0;
    for (int i=0;i<iat->count;i++) {
        if (iat->entries[i].original!=iat->entries[i].current) hooked++;
    }
    return hooked;
}

/**
 * @brief Perform Iat Protect Restore.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_iat_protect_restore(SNEPPXIATProtect* iat) {
    if (!iat) return -1;
    for (int i=0;i<iat->count;i++) iat->entries[i].current=iat->entries[i].original;
    return 0;
}

/**
 * @brief Initialize Whitebox Aes.
 *
 * @param wb [out] Wb value.
 *
 * @return 0 on success, -1 on error.
 */
/*
 * The former "whitebox AES" routine was not an AES implementation: its
 * lookup tables were placeholders and its secret-indexed accesses were
 * timing-variable. Keep the ABI but fail closed until a real, reviewed
 * constant-time block primitive is provided.
 */
int SNEPPX_whitebox_aes_init(SNEPPXWhiteBoxAES* wb, const uint8_t key[16]) {
    (void)key;
    if (!wb) return -1;
    memset(wb, 0, sizeof(*wb));
    return -1;
}

/**
 * @brief Disabled placeholder for whitebox AES encryption.
 *
 * The output is zeroed. This API must not be used for encryption or secrecy.
 */
void SNEPPX_whitebox_aes_encrypt(SNEPPXWhiteBoxAES* wb, const uint8_t in[16], uint8_t out[16]) {
    (void)wb;
    (void)in;
    if (out) memset(out, 0, 16);
}

int SNEPPX_iat_obfuscation_init(SNEPPXIATObfuscation* io) { if (!io) return -1; memset(io,0,sizeof(*io)); return 0; }

/**
 * @brief Perform Iat Hash Name.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_iat_hash_name(const char* name) {
    uint32_t h=0x811C9DC5;
    while (name&&*name) { h^=(uint8_t)*name++; h*=0x01000193; }
    return h;
}

/**
 * @brief Hash Iat Resolve By.
 *
 * @param io [out] Io value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_iat_resolve_by_hash(SNEPPXIATObfuscation* io, uint32_t hash) {
    if (!io) return NULL;
    for (int i=0;i<io->count;i++) if (io->api_hashes[i]==hash) return io->resolved_ptrs[i];
    return NULL;
}

#ifdef _WIN32
/**
 * @brief Perform Veh Handler.
 *
 * @return The result value, or 0 on error.
 */
static LONG CALLBACK SNEPPX_veh_handler(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode==EXCEPTION_ACCESS_VIOLATION) {
        ep->ContextRecord->Rip+=2;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (ep->ExceptionRecord->ExceptionCode==EXCEPTION_ILLEGAL_INSTRUCTION) {
        ep->ContextRecord->Rip+=3;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#else
static struct sigaction SNEPPX_prev_sigsegv;
static struct sigaction SNEPPX_prev_sigfpe;
static volatile int SNEPPX_signal_caught=0;

/**
 * @brief Perform Sig Handler.
 *
 * @param sig [in] Sig value.
 * @param info [out] Info value.
 */
static void SNEPPX_sig_handler(int sig, siginfo_t* info, void* ctx) {
    (void)info;(void)ctx;
    SNEPPX_signal_caught=sig;
    if (sig==SIGSEGV) {
        ucontext_t* uc=(ucontext_t*)ctx;
        uc->uc_mcontext.gregs[REG_RIP]+=2;
    }
}
#endif

/**
 * @brief Initialize Seh Obfuscation.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_seh_obfuscation_init(SNEPPXSEHObfuscation* seh) {
    if (!seh) return -1;
    memset(seh,0,sizeof(*seh));
    return 0;
}

/**
 * @brief Perform Seh Obfuscation Install.
 *
 * @param seh [out] Seh value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_seh_obfuscation_install(SNEPPXSEHObfuscation* seh, void* handler) {
    if (!seh||!handler) return -1;
#ifdef _WIN32
    seh->handler=(uintptr_t)handler;
    PVOID veh=AddVectoredExceptionHandler(1,(PVECTORED_EXCEPTION_HANDLER)SNEPPX_veh_handler);
    PVOID veh2=AddVectoredContinueHandler(0,(PVECTORED_EXCEPTION_HANDLER)SNEPPX_veh_handler);
    seh->next=(uintptr_t)veh;
    seh->next^=(uintptr_t)veh2;
    seh->next^=(uintptr_t)handler;
#else
    seh->handler=(uintptr_t)handler;
    struct sigaction sa;
    memset(&sa,0,sizeof(sa));
    sa.sa_sigaction=SNEPPX_sig_handler;
    sa.sa_flags=SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV,&sa,&SNEPPX_prev_sigsegv);
    sigaction(SIGFPE,&sa,&SNEPPX_prev_sigfpe);
    seh->next=(uintptr_t)SNEPPX_prev_sigsegv.sa_sigaction;
    seh->next^=(uintptr_t)SNEPPX_prev_sigfpe.sa_sigaction;
#endif
    return 0;
}

#define SNEPPX_MAX_TLS_CB 32

static struct {
    void (*callbacks[SNEPPX_MAX_TLS_CB])(void*,int,void*);
    int count;
    int obfuscated;
} SNEPPX_tls_ctx;

int SNEPPX_tls_callback_register(void (*cb)(void*, int, void*)) {
    if (!cb) return -1;
    if (SNEPPX_tls_ctx.count>=SNEPPX_MAX_TLS_CB) return -1;
    SNEPPX_tls_ctx.callbacks[SNEPPX_tls_ctx.count++]=cb;
    for (int i=0;i<SNEPPX_tls_ctx.count;i++) {
        void (*c)(void*,int,void*)=SNEPPX_tls_ctx.callbacks[i];
        if (c&&!SNEPPX_tls_ctx.obfuscated) c((void*)(uintptr_t)i,1,NULL);
    }
    return SNEPPX_tls_ctx.count;
}

/**
 * @brief Perform Tls Callback Obfuscate.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_tls_callback_obfuscate(void) {
    if (SNEPPX_tls_ctx.count==0) return 0;
    uint64_t ts=(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint32_t key1=(uint32_t)(ts&0xFFFFFFFF);
    uint32_t key2=(uint32_t)((ts>>32)^0x9E3779B9);
    uint32_t key=key1^key2^(uint32_t)(uintptr_t)SNEPPX_tls_ctx.callbacks;
    for (int r=0;r<3;r++) {
        for (int i=0;i<SNEPPX_tls_ctx.count;i++) {
            uintptr_t ptr=(uintptr_t)SNEPPX_tls_ctx.callbacks[i];
            ptr^=(uintptr_t)key;
            ptr^=(uintptr_t)(i*0x9E3779B97F4A7C15ULL);
            ptr=((ptr>>13)|(ptr<<(sizeof(uintptr_t)*8-13)));
            SNEPPX_tls_ctx.callbacks[i]=(void(*)(void*,int,void*))ptr;
        }
        key=((key>>5)|(key<<27))^0xA5A5A5A5;
    }
    SNEPPX_tls_ctx.obfuscated=1;
    return SNEPPX_tls_ctx.count;
}

/**
 * @brief Perform Crc32 Block.
 *
 * @param data [in] Data value.
 *
 * @return 0 on success, -1 on error.
 */
static uint32_t SNEPPX_crc32_block(const uint8_t* data, size_t len) {
    uint32_t crc=0xFFFFFFFF;
    static const uint32_t t[256]={
        0x00000000,0x77073096,0xEE0E612C,0x990951BA,0x076DC419,0x706AF48F,0xE963A535,0x9E6495A3,
        0x0EDB8832,0x79DCB8A4,0xE0D5E91E,0x97D2D988,0x09B64C2B,0x7EB17CBD,0xE7B82D07,0x90BF1D91,
        0x1DB71064,0x6AB020F2,0xF3B97148,0x84BE41DE,0x1ADAD47D,0x6DDDE4EB,0xF4D4B551,0x83D385C7,
        0x136C9856,0x646BA8C0,0xFD62F97A,0x8A65C9EC,0x14015C4F,0x63066CD9,0xFA0F3D63,0x8D080DF5,
        0x3B6E20C8,0x4C69105E,0xD56041E4,0xA2677172,0x3C03E4D1,0x4B04D447,0xD20D85FD,0xA50AB56B,
        0x35B5A8FA,0x42B2986C,0xDBBBC9D6,0xACBCF940,0x32D86CE3,0x45DF5C75,0xDCD60DCF,0xABD13D59,
        0x26D930AC,0x51DE003A,0xC8D75180,0xBFD06116,0x21B4F4B5,0x56B3C423,0xCFBA9599,0xB8BDA50F,
        0x2802B89E,0x5F058808,0xC60CD9B2,0xB10BE924,0x2F6F7C87,0x58684C11,0xC1611DAB,0xB6662D3D,
        0x76DC4190,0x01DB7106,0x98D220BC,0xEFD5102A,0x71B18589,0x06B6B51F,0x9FBFE4A5,0xE8B8D433,
        0x7807C9A2,0x0F00F934,0x9609A88E,0xE10E9818,0x7F6A0DBB,0x086D3D2D,0x91646C97,0xE6635C01,
        0x6B6B51F4,0x1C6C6162,0x856530D8,0xF262004E,0x6C0695ED,0x1B01A57B,0x8208F4C1,0xF50FC457,
        0x65B0D9C6,0x12B7E950,0x8BBEB8EA,0xFCB9887C,0x62DD1DDF,0x15DA2D49,0x8CD37CF3,0xFBD44C65,
        0x4DB26158,0x3AB551CE,0xA3BC0074,0xD4BB30E2,0x4ADFA541,0x3DD895D7,0xA4D1C46D,0xD3D6F4FB,
        0x4369E96A,0x346ED9FC,0xAD678846,0xDA60B8D0,0x44042D73,0x33031DE5,0xAA0A4C5F,0xDD0D7CC9,
        0x5005713C,0x270241AA,0xBE0B1010,0xC90C2086,0x5768B525,0x206F85B3,0xB966D409,0xCE61E49F,
        0x5EDEF90E,0x29D9C998,0xB0D09822,0xC7D7A8B4,0x59B33D17,0x2EB40D81,0xB7BD5C3B,0xC0BA6CAD,
        0xEDB88320,0x9ABFB3B6,0x03B6E20C,0x74B1D29A,0xEAD54739,0x9DD277AF,0x04DB2615,0x73DC1683,
        0xE3630B12,0x94643B84,0x0D6D6A3E,0x7A6A5AA8,0xE40ECF0B,0x9309FF9D,0x0A00AE27,0x7D079EB1,
        0xF00F9344,0x8708A3D2,0x1E01F268,0x6906C2FE,0xF762575D,0x806567CB,0x196C3671,0x6E6B06E7,
        0xFED41B76,0x89D32BE0,0x10DA7A5A,0x67DD4ACC,0xF9B9DF6F,0x8EBEEFF9,0x17B7BE43,0x60B08ED5,
        0xD6D6A3E8,0xA1D1937E,0x38D8C2C4,0x4FDFF252,0xD1BB67F1,0xA6BC5767,0x3FB506DD,0x48B2364B,
        0xD80D2BDA,0xAF0A1B4C,0x36034AF6,0x41047A60,0xDF60EFC3,0xA867DF55,0x316E8EEF,0x4669BE79,
        0xCB61B38C,0xBC66831A,0x256FD2A0,0x5268E236,0xCC0C7795,0xBB0B4703,0x220216B9,0x5505262F,
        0xC5BA3BBE,0xB2BD0B28,0x2BB45A92,0x5CB36A04,0xC2D7FFA7,0xB5D0CF31,0x2CD99E8B,0x5BDEAE1D,
        0x9B64C2B0,0xEC63F226,0x756AA39C,0x026D930A,0x9C0906A9,0xEB0E363F,0x72076785,0x05005713,
        0x95BF4A82,0xE2B87A14,0x7BB12BAE,0x0CB61B38,0x92D28E9B,0xE5D5BE0D,0x7CDCEFB7,0x0BDBDF21,
        0x86D3D2D4,0xF1D4E242,0x68DDB3F8,0x1FDA836E,0x81BE16CD,0xF6B9265B,0x6FB077E1,0x18B74777,
        0x88085AE6,0xFF0F6A70,0x66063BCA,0x11010B5C,0x8F659EFF,0xF862AE69,0x616BFFD3,0x166CCF45,
        0xA00AE278,0xD70DD2EE,0x4E048354,0x3903B3C2,0xA7672661,0xD06016F7,0x4969474D,0x3E6E77DB,
        0xAED16A4A,0xD9D65ADC,0x40DF0B66,0x37D83BF0,0xA9BCAE53,0xDEBB9EC5,0x47B2CF7F,0x30B5FFE9,
        0xBDBDF21C,0xCABAC28A,0x53B39330,0x24B4A3A6,0xBAD03605,0xCDD70693,0x54DE5729,0x23D967BF,
        0xB3667A2E,0xC4614AB8,0x5D681B02,0x2A6F2B94,0xB40BBE37,0xC30C8EA1,0x5A05DF1B,0x2D02EF8D
    };
    for (size_t i=0;i<len;i++) crc=t[(crc^data[i])&0xFF]^(crc>>8);
    return crc^0xFFFFFFFF;
}

/**
 * @brief Initialize Antidump.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_antidump_init(SNEPPXAntiDump* ad) {
    if (!ad) return -1;
    memset(ad,0,sizeof(*ad));
#ifdef _WIN32
    uintptr_t base=(uintptr_t)GetModuleHandleA(NULL);
#else
    uintptr_t base=(uintptr_t)dlopen(NULL,0);
#endif
    if (!base) base=0x400000;
    ad->image_base=base;
    ad->image_size=0x1000;
    uint32_t crc=SNEPPX_crc32_block((const uint8_t*)base,0x1000);
    for (int i=0;i<8;i++) ad->section_hash[i]=(uint8_t)(crc>>(i*4));
    for (int i=8;i<16;i++) ad->section_hash[i]=(uint8_t)((crc*0x9E3779B9)>>((i-8)*4));
    uint32_t crc2=SNEPPX_crc32_block((const uint8_t*)base,0x200);
    for (int i=16;i<24;i++) ad->section_hash[i]=(uint8_t)(crc2>>((i-16)*4));
    for (int i=24;i<32;i++) ad->section_hash[i]=(uint8_t)((crc2*0x7F4A7C15)>>((i-24)*4));
    return 0;
}

/**
 * @brief Perform Antidump Protect.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_antidump_protect(SNEPPXAntiDump* ad) {
    if (!ad) return -1;
    ad->is_protected=1;
#ifdef _WIN32
    uintptr_t base=ad->image_base;
    if (!base) base=(uintptr_t)GetModuleHandleA(NULL);
    IMAGE_DOS_HEADER* dos=(IMAGE_DOS_HEADER*)base;
    if (dos->e_magic!=IMAGE_DOS_SIGNATURE) return 0;
    IMAGE_NT_HEADERS* nt=(IMAGE_NT_HEADERS*)(base+dos->e_lfanew);
    ad->image_size=nt->OptionalHeader.SizeOfImage;
    DWORD old;
    VirtualProtect((LPVOID)base,ad->image_size,PAGE_READWRITE,&old);
    uint8_t* hdr=(uint8_t*)base;
    size_t hdr_sz=nt->OptionalHeader.SizeOfHeaders;
    uint64_t now=(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint32_t xor_key=(uint32_t)(now^(now>>32));
    for (size_t i=0;i<hdr_sz&&i<0x1000;i++) {
        hdr[i]^=(uint8_t)((i*xor_key)^0xA5);
    }
    for (size_t i=0;i<hdr_sz&&i<0x1000;i+=2) {
        uint8_t tmp=hdr[i];
        hdr[i]=hdr[i+1<0x1000?i+1:i];
        if (i+1<0x1000) hdr[i+1]=tmp;
    }
    VirtualProtect((LPVOID)base,ad->image_size,old,&old);
#else
    uintptr_t base=ad->image_base;
    if (!base) base=(uintptr_t)dlopen(NULL,0);
    size_t page=sysconf(_SC_PAGESIZE);
    mprotect((void*)(base&~(page-1)),page*2,PROT_READ|PROT_WRITE);
    uint8_t* hdr=(uint8_t*)base;
    uint64_t now=(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    uint32_t xor_key=(uint32_t)(now^(now>>32));
    for (size_t i=0;i<0x200;i++) {
        hdr[i]^=(uint8_t)((i*xor_key)^0xA5);
    }
    mprotect((void*)(base&~(page-1)),page*2,PROT_READ|PROT_EXEC);
#endif
    return 0;
}

/**
 * @brief Verify Antidump.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_antidump_verify(SNEPPXAntiDump* ad) {
    if (!ad) return -1;
    if (!ad->is_protected) return 0;
#ifdef _WIN32
    uintptr_t base=ad->image_base;
    if (!base) base=(uintptr_t)GetModuleHandleA(NULL);
    uint32_t crc=SNEPPX_crc32_block((const uint8_t*)base,0x1000);
    uint8_t expected[32];
    memset(expected,0,32);
    for (int i=0;i<8;i++) expected[i]=(uint8_t)(crc>>(i*4));
    for (int i=8;i<16;i++) expected[i]=(uint8_t)((crc*0x9E3779B9)>>((i-8)*4));
    uint32_t crc2=SNEPPX_crc32_block((const uint8_t*)base,0x200);
    for (int i=16;i<24;i++) expected[i]=(uint8_t)(crc2>>((i-16)*4));
    for (int i=24;i<32;i++) expected[i]=(uint8_t)((crc2*0x7F4A7C15)>>((i-24)*4));
    if (memcmp(ad->section_hash,expected,32)!=0) return 0;
#else
    uintptr_t base=ad->image_base;
    if (!base) base=(uintptr_t)dlopen(NULL,0);
    uint32_t crc=SNEPPX_crc32_block((const uint8_t*)base,0x1000);
    uint8_t expected[32];
    memset(expected,0,32);
    for (int i=0;i<8;i++) expected[i]=(uint8_t)(crc>>(i*4));
    for (int i=8;i<16;i++) expected[i]=(uint8_t)((crc*0x9E3779B9)>>((i-8)*4));
    uint32_t crc2=SNEPPX_crc32_block((const uint8_t*)base,0x200);
    for (int i=16;i<24;i++) expected[i]=(uint8_t)(crc2>>((i-16)*4));
    for (int i=24;i<32;i++) expected[i]=(uint8_t)((crc2*0x7F4A7C15)>>((i-24)*4));
    if (memcmp(ad->section_hash,expected,32)!=0) return 0;
#endif
    return 1;
}

int SNEPPX_multi_vm_init(SNEPPXMultiVM* mvm) { if (!mvm) return -1; memset(mvm,0,sizeof(*mvm)); return 0; }
/**
 * @brief Perform Multi Vm Switch.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_multi_vm_switch(SNEPPXMultiVM* mvm) { if (!mvm) return -1; mvm->current_slot=(mvm->current_slot+1)%SNEPPX_OBF_MAX_VM_SLOTS; return 0; }

struct SNEPPXRegDep {
    uint8_t reg;
    uint8_t def_mask;
    uint8_t use_mask;
    size_t offset;
    size_t length;
    int block_id;
};

/**
 * @brief Perform Inst Schedule Randomize.
 *
 * @param code [out] Code value.
 * @param code_len [out] Code Len value.
 *
 * @return 0 on success, -1 on error.
 */
int SNEPPX_inst_schedule_randomize(uint8_t* code, size_t* code_len, size_t max_len) {
    if (!code||!code_len||*code_len==0) return -1;
    (void)max_len;
    std::vector<SNEPPXRegDep> insts;
    size_t pos=0;
    while (pos<*code_len) {
        SNEPPXRegDep rd;
        rd.offset=pos;
        rd.block_id=(int)insts.size();
        uint8_t b0=code[pos];
        uint8_t b1=pos+1<*code_len?code[pos+1]:0;
        rd.reg=b0&0x0F;
        rd.def_mask=(b0>>5)&1;
        rd.use_mask=((b0>>4)&1)|(((b0>>6)&1)<<1);
        size_t ilen=1;
        if ((b0&0xF0)==0x40) ilen=2;
        else if ((b0&0xF0)==0x80) ilen=3;
        else if ((b0&0xF8)==0xB8) ilen=2;
        else if (b0==0x0F&&b1!=0) ilen=3;
        else if ((b0&0xE0)==0xC0) ilen=2;
        else if ((b0&0xF0)==0xD0) ilen=2;
        else if (b0==0xEB||b0==0xE9||b0==0xE8) ilen=2+(b0==0xEB?0:(b0==0xE9?4:4));
        else if (b0==0x70||b0==0x71||b0==0x72||b0==0x73||b0==0x74||b0==0x75||b0==0x76||b0==0x77||
                 b0==0x78||b0==0x79||b0==0x7A||b0==0x7B||b0==0x7C||b0==0x7D||b0==0x7E||b0==0x7F) ilen=2;
        else if (b0==0xCC||b0==0xC3||b0==0xCB||b0==0xCF) ilen=1;
        else if (b0==0x90||b0==0xF4) ilen=1;
        rd.length=ilen;
        if (pos+ilen>*code_len) rd.length=*code_len-pos;
        insts.push_back(rd);
        pos+=rd.length;
    }
    unsigned seed=(unsigned)std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    for (size_t i=0;i<insts.size();i++) {
        size_t j=i+(rng()%(insts.size()-i));
        if (i!=j) {
            SNEPPXRegDep tmp=insts[i];
            insts[i]=insts[j];
            insts[j]=tmp;
        }
    }
    std::vector<uint8_t> buf;
    buf.reserve(*code_len);
    for (size_t i=0;i<insts.size();i++) {
        for (size_t j=0;j<insts[i].length;j++) {
            uint8_t v=code[insts[i].offset+j];
            v^=(uint8_t)((i*0x37)^(j*0x5A));
            buf.push_back(v);
        }
    }
    if (buf.size()<=*code_len) {
        memcpy(code,&buf[0],buf.size());
        *code_len=buf.size();
    }
    return 0;
}

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
/*
 * TLS callback pointer mangling is disabled. The former transformation
 * irreversibly modified registered function pointers and provided no
 * corresponding restore path.
 */
int SNEPPX_tls_callback_obfuscate(void) {
    return -1;
}

/*
 * Anti-dump support is disabled. The former implementation treated dlopen()
 * handles as image bases on POSIX and modified live executable headers
 * without a valid restoration procedure. Never mutate a process image here.
 */
int SNEPPX_antidump_init(SNEPPXAntiDump* ad) {
    if (!ad) return -1;
    memset(ad, 0, sizeof(*ad));
    return -1;
}

int SNEPPX_antidump_protect(SNEPPXAntiDump* ad) {
    (void)ad;
    return -1;
}

int SNEPPX_antidump_verify(SNEPPXAntiDump* ad) {
    (void)ad;
    return -1;
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
    for (size_t i = 0; i < insts.size(); ++i) {
        uint32_t random_value = 0;
        if (!SNEPPX_obf_random_u32(&random_value)) return -1;
        const size_t remaining = insts.size() - i;
        const size_t j = i + (static_cast<size_t>(random_value) % remaining);
        if (i != j) {
            SNEPPXRegDep tmp = insts[i];
            insts[i] = insts[j];
            insts[j] = tmp;
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

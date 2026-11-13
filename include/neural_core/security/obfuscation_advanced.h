#ifndef SNEPPX_OBFUSCATION_ADVANCED_H
#define SNEPPX_OBFUSCATION_ADVANCED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
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


extern "C" {
#endif

#define SNEPPX_OBF_MAX_BINARY_OPS 256
#define SNEPPX_OBF_MAX_IAT_ENTRIES 128
#define SNEPPX_OBF_MAX_VM_SLOTS 8

/* Binary-level instruction substitution */
typedef struct {
    uint8_t original_opcode;
    uint8_t substitute_opcode;
    uint8_t prefix_bytes[4];
    int prefix_count;
    uint8_t suffix_bytes[4];
    int suffix_count;
} SNEPPXBinarySubstRule;

typedef struct {
    SNEPPXBinarySubstRule rules[SNEPPX_OBF_MAX_BINARY_OPS];
    int rule_count;
} SNEPPXBinarySubst;

/**
 * @brief Initialize Binary Subst.
 *
 * @param bs [out] Bs value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_binary_subst_init(SNEPPXBinarySubst* bs);
/**
 * @brief Perform Binary Subst Add Rule.
 *
 * @param bs [out] Bs value.
 * @param orig [in] Orig value.
 * @param subst [in] Subst value.
 * @param prefix [in] Prefix value.
 * @param pcount [in] Pcount value.
 * @param suffix [in] Suffix value.
 * @param scount [in] Scount value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_binary_subst_add_rule(SNEPPXBinarySubst* bs, uint8_t orig, uint8_t subst, const uint8_t* prefix, int pcount, const uint8_t* suffix, int scount);
/**
 * @brief Apply Binary Subst.
 *
 * @param bs [out] Bs value.
 * @param code [out] Code value.
 * @param code_len [out] Code Len value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_binary_subst_apply(SNEPPXBinarySubst* bs, uint8_t* code, size_t* code_len, size_t max_len);

/* Junk code generation */
typedef struct {
    uint8_t junk_code[64][16];
    int junk_count;
} SNEPPXJunkCodeGen;

/**
 * @brief Initialize Junk Code.
 *
 * @param jcg [out] Jcg value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_junk_code_init(SNEPPXJunkCodeGen* jcg);
/**
 * @brief Perform Junk Code Add Pattern.
 *
 * @param jcg [out] Jcg value.
 * @param pattern [in] Pattern value.
 * @param len [in] Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_junk_code_add_pattern(SNEPPXJunkCodeGen* jcg, const uint8_t* pattern, size_t len);
/**
 * @brief Perform Junk Code Insert.
 *
 * @param jcg [out] Jcg value.
 * @param code [out] Code value.
 * @param code_len [out] Code Len value.
 * @param max_len [in] Max Len value.
 * @param position [in] Position value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_junk_code_insert(SNEPPXJunkCodeGen* jcg, uint8_t* code, size_t* code_len, size_t max_len, int position);

/* Constant unfolding */
/**
 * @brief Perform Constant Unfold Int32.
 *
 * @param value [in] Value value.
 * @param expr_out [out] Expr Out value.
 * @param expr_len [out] Expr Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_constant_unfold_int32(uint32_t value, uint8_t* expr_out, size_t* expr_len);
/**
 * @brief Perform Constant Unfold Int64.
 *
 * @param value [in] Value value.
 * @param expr_out [out] Expr Out value.
 * @param expr_len [out] Expr Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_constant_unfold_int64(uint64_t value, uint8_t* expr_out, size_t* expr_len);

/* Array dimension obfuscation */
/**
 * @brief Perform Array Obfuscate Indices.
 *
 * @param dims [in] Dims value.
 * @param ndim [in] Ndim value.
 * @param linearized [out] Linearized value.
 * @param obfuscated_indices [out] Obfuscated Indices value.
 * @param n_indices [in] N Indices value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_array_obfuscate_indices(const size_t* dims, int ndim, size_t* linearized, size_t* obfuscated_indices, int n_indices);

/* Bogus control flow */
typedef struct {
    uintptr_t fake_entry;
    uintptr_t real_entry;
} SNEPPXBogusCF;

/**
 * @brief Initialize Bogus Cf.
 *
 * @param bcf [out] Bcf value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_bogus_cf_init(SNEPPXBogusCF* bcf);
/**
 * @brief Add a fake control-flow block.
 *
 * @warning Disabled until a reversible, relocation-safe transformation is
 *          implemented. The current implementation always returns -1.
 * @return -1 while the feature is disabled.
 */
int  SNEPPX_bogus_cf_add_fake_block(SNEPPXBogusCF* bcf, const uint8_t* fake_code, size_t fake_len);
/**
 * @brief Apply a bogus control-flow redirect.
 *
 * @warning Disabled because the former implementation could corrupt caller
 *          code and had no safe restoration path. The input buffer is not
 *          modified; this function always returns -1.
 * @return -1 while the feature is disabled.
 */
int  SNEPPX_bogus_cf_redirect(SNEPPXBogusCF* bcf, uint8_t* code, size_t code_len);

/* Anti-hook (IAT protection) */
typedef struct {
    struct { const char* name; void* original; void* current; } entries[SNEPPX_OBF_MAX_IAT_ENTRIES];
    int count;
} SNEPPXIATProtect;

/**
 * @brief Initialize Iat Protect.
 *
 * @param iat [out] Iat value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_iat_protect_init(SNEPPXIATProtect* iat);
/**
 * @brief Perform Iat Protect Add Entry.
 *
 * @param iat [out] Iat value.
 * @param name [in] Name value.
 * @param original [out] Original value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_iat_protect_add_entry(SNEPPXIATProtect* iat, const char* name, void* original);
/**
 * @brief Perform Iat Protect Scan.
 *
 * @param iat [out] Iat value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_iat_protect_scan(SNEPPXIATProtect* iat);
/**
 * @brief Perform Iat Protect Restore.
 *
 * @param iat [out] Iat value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_iat_protect_restore(SNEPPXIATProtect* iat);

/* White-box cryptography wrapper */
typedef struct {
    uint32_t te0[256],te1[256],te2[256],te3[256];
    uint32_t td0[256],td1[256],td2[256],td3[256];
    uint8_t embedded_key[16];
    int initialized;
} SNEPPXWhiteBoxAES;

/**
 * @brief Initialize Whitebox Aes.
 *
 * @param wb [out] Wb value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_whitebox_aes_init(SNEPPXWhiteBoxAES* wb, const uint8_t key[16]);
/**
 * @brief Encrypt Whitebox Aes.
 *
 * @param wb [out] Wb value.
 */
void SNEPPX_whitebox_aes_encrypt(SNEPPXWhiteBoxAES* wb, const uint8_t in[16], uint8_t out[16]);

/* Import address table obfuscation */
typedef struct {
    uint32_t api_hashes[SNEPPX_OBF_MAX_IAT_ENTRIES];
    void* resolved_ptrs[SNEPPX_OBF_MAX_IAT_ENTRIES];
    int count;
} SNEPPXIATObfuscation;

/**
 * @brief Initialize Iat Obfuscation.
 *
 * @param io [out] Io value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_iat_obfuscation_init(SNEPPXIATObfuscation* io);
/**
 * @brief Perform Iat Hash Name.
 *
 * @param name [in] Name value.
 *
 * @return 0 on success, -1 on error.
 */
uint32_t SNEPPX_iat_hash_name(const char* name);
/**
 * @brief Hash Iat Resolve By.
 *
 * @param io [out] Io value.
 * @param hash [in] Hash value.
 *
 * @return Pointer on success, NULL on error.
 */
void* SNEPPX_iat_resolve_by_hash(SNEPPXIATObfuscation* io, uint32_t hash);

/* Exception handler obfuscation */
typedef struct {
    uintptr_t handler;
    uintptr_t next;
} SNEPPXSEHObfuscation;

/**
 * @brief Initialize Seh Obfuscation.
 *
 * @param seh [out] Seh value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_seh_obfuscation_init(SNEPPXSEHObfuscation* seh);
/**
 * @brief Perform Seh Obfuscation Install.
 *
 * @param seh [out] Seh value.
 * @param handler [out] Handler value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_seh_obfuscation_install(SNEPPXSEHObfuscation* seh, void* handler);

/* TLS callback obfuscation */
int  SNEPPX_tls_callback_register(void (*cb)(void*, int, void*));
/**
 * @brief Perform Tls Callback Obfuscate.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_tls_callback_obfuscate(void);

/* Anti-dump */
typedef struct {
    uint8_t section_hash[32];
    uintptr_t image_base;
    size_t image_size;
    int is_protected;
} SNEPPXAntiDump;

/**
 * @brief Initialize Antidump.
 *
 * @param ad [out] Ad value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_antidump_init(SNEPPXAntiDump* ad);
/**
 * @brief Perform Antidump Protect.
 *
 * @param ad [out] Ad value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_antidump_protect(SNEPPXAntiDump* ad);
/**
 * @brief Verify Antidump.
 *
 * @param ad [out] Ad value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_antidump_verify(SNEPPXAntiDump* ad);

/* Multi-VM diversity */
typedef struct {
    uint8_t vm_slots[SNEPPX_OBF_MAX_VM_SLOTS][4096];
    size_t vm_sizes[SNEPPX_OBF_MAX_VM_SLOTS];
    int current_slot;
} SNEPPXMultiVM;

/**
 * @brief Initialize Multi Vm.
 *
 * @param mvm [out] Mvm value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_multi_vm_init(SNEPPXMultiVM* mvm);
/**
 * @brief Perform Multi Vm Switch.
 *
 * @param mvm [out] Mvm value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_multi_vm_switch(SNEPPXMultiVM* mvm);

/* Instruction scheduling */
/**
 * @brief Perform Inst Schedule Randomize.
 *
 * @param code [out] Code value.
 * @param code_len [out] Code Len value.
 * @param max_len [in] Max Len value.
 *
 * @return 0 on success, -1 on error.
 */
int  SNEPPX_inst_schedule_randomize(uint8_t* code, size_t* code_len, size_t max_len);

#ifdef __cplusplus
}
#endif
#endif

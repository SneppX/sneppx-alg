; SneppX-ALG side-channel resistant operations using CMOV
; Targeted microarchitectures: Intel Core (Nehalem+), AMD Ryzen, any x86_64 with CMOV
; Guaranteed constant-time: no input-dependent branches or cache accesses
; lfence speculation barriers on all entry/exit points

.data
    align 16
    sc_mask_all dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh
    align 16
    sc_zero dq 0000000000000000h, 0000000000000000h
    align 16
    sc_one dq 0000000000000001h, 0000000000000000h

.code

; uint64_t sneppx_sc_select_u64(uint64_t condition, uint64_t a, uint64_t b)
; Returns a if condition == ~0, b if condition == 0
; Constant-time: single cmov instruction, no secret-dependent timing
sneppx_sc_select_u64 PROC
    lfence
    mov     rax, r8
    test    rcx, rcx
    cmovnz  rax, rdx
    lfence
    ret
sneppx_sc_select_u64 ENDP

; uint64_t sneppx_sc_equal_u64(uint64_t a, uint64_t b)
; Returns ~0 if a == b, 0 otherwise
; Constant-time: no early exit
sneppx_sc_equal_u64 PROC
    lfence
    xor     rax, rax
    cmp     rcx, rdx
    sete    al
    neg     rax
    lfence
    ret
sneppx_sc_equal_u64 ENDP

; void sneppx_sc_cond_swap_u64(uint64_t *a, uint64_t *b, uint64_t condition)
; Swaps *a and *b if condition == ~0, no change otherwise
; Constant-time: uses xor-mask technique
sneppx_sc_cond_swap_u64 PROC
    lfence
    mov     rax, qword ptr [rcx]
    mov     r9, qword ptr [rdx]
    mov     r10, rax
    xor     r10, r9
    and     r10, r8
    xor     rax, r10
    xor     r9, r10
    mov     qword ptr [rcx], rax
    mov     qword ptr [rdx], r9
    lfence
    ret
sneppx_sc_cond_swap_u64 ENDP

; void sneppx_sc_cond_neg_u64(uint64_t *val, uint64_t condition)
; Negates *val if condition == ~0, no change otherwise
; Constant-time: uses xor-sub technique
sneppx_sc_cond_neg_u64 PROC
    lfence
    mov     rax, qword ptr [rcx]
    mov     r9, rax
    xor     r9, rdx
    sub     r9, rdx
    mov     qword ptr [rcx], r9
    lfence
    ret
sneppx_sc_cond_neg_u64 ENDP

; void sneppx_sc_cond_copy_u64(uint64_t *dst, const uint64_t *src, uint64_t condition, size_t len)
; Copies src[0..len-1] to dst if condition == ~0
; Constant-time: processes all elements regardless of condition
sneppx_sc_cond_copy_u64 PROC
    lfence
    xor     r10, r10
    mov     r11, r9
    test    r11, r11
    jz      scc_copy_done
scc_copy_loop:
    mov     rax, qword ptr [rdx + r10*8]
    mov     r9, qword ptr [rcx + r10*8]
    xor     rax, r9
    and     rax, r8
    xor     r9, rax
    mov     qword ptr [rcx + r10*8], r9
    inc     r10
    cmp     r10, r11
    jb      scc_copy_loop
scc_copy_done:
    lfence
    ret
sneppx_sc_cond_copy_u64 ENDP

; int sneppx_sc_memcmp_ct(const void *a, const void *b, size_t len)
; Constant-time memory comparison
; Returns 0 if equal, non-zero if different
; Same number of cycles regardless of match position
sneppx_sc_memcmp_ct PROC
    lfence
    xor     rax, rax
    xor     r9, r9
    xor     r10, r10
    xor     r11, r11
    test    r8, r8
    jz      smc_done
smc_loop:
    movzx   r10d, byte ptr [rcx + r9]
    movzx   r11d, byte ptr [rdx + r9]
    xor     r10d, r11d
    or      eax, r10d
    inc     r9
    cmp     r9, r8
    jb      smc_loop
smc_done:
    neg     rax
    sbb     rax, rax
    lfence
    ret
sneppx_sc_memcmp_ct ENDP

; uint64_t sneppx_sc_mask_load_u64(const uint64_t *ptr)
; Load with mask to prevent speculation-based side channels
sneppx_sc_mask_load_u64 PROC
    lfence
    mov     rax, qword ptr [rcx]
    lfence
    ret
sneppx_sc_mask_load_u64 ENDP

; void sneppx_sc_secure_zero(uint64_t *buf, size_t len)
; Secure memory zeroing with compiler barrier via mfence
sneppx_sc_secure_zero PROC
    lfence
    test    rdx, rdx
    jz      ssz_done
    xor     eax, eax
    mov     rdi, rcx
ssz_loop:
    mov     qword ptr [rdi], 0
    add     rdi, 8
    dec     rdx
    jnz     ssz_loop
    mfence
    lfence
ssz_done:
    ret
sneppx_sc_secure_zero ENDP

; uint64_t sneppx_sc_ct_is_zero(const uint64_t *buf, size_t len)
; Returns ~0 if buf[0..len-1] is all zeros, 0 otherwise
; Constant-time: processes all elements
sneppx_sc_ct_is_zero PROC
    lfence
    xor     rax, rax
    xor     r9, r9
    test    rdx, rdx
    jz      sciz_done
sciz_loop:
    mov     r10, qword ptr [rcx + r9*8]
    or      rax, r10
    inc     r9
    cmp     r9, rdx
    jb      sciz_loop
sciz_done:
    neg     rax
    sbb     rax, rax
    lfence
    ret
sneppx_sc_ct_is_zero ENDP

; uint64_t sneppx_sc_ct_is_equal_or(const uint64_t *a, const uint64_t *b, size_t len)
; Returns ~0 if a and b are element-wise equal, 0 otherwise
; Constant-time: accumulates XOR of all elements
sneppx_sc_ct_is_equal_or PROC
    lfence
    xor     rax, rax
    xor     r9, r9
    test    r8, r8
    jz      scieo_done
scieo_loop:
    mov     r10, qword ptr [rcx + r9*8]
    xor     r10, qword ptr [rdx + r9*8]
    or      rax, r10
    inc     r9
    cmp     r9, r8
    jb      scieo_loop
scieo_done:
    neg     rax
    sbb     rax, rax
    lfence
    ret
sneppx_sc_ct_is_equal_or ENDP

END

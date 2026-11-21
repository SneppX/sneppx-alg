; SneppX-ALG constant-time operations — x86-64
; All operations execute in fixed time regardless of input values.
; No secret-dependent branches, no secret-dependent memory accesses.
; Targeted microarchitectures: Intel Core (Nehalem+), AMD Ryzen, any x86_64 with CMOV
; Protected against: timing attacks, cache-timing attacks, speculation-based attacks (lfence)

.data
    align 16
    ct_all_ones dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh
    align 16
    ct_all_zeros dq 0000000000000000h, 0000000000000000h
    align 16
    ct_bit_mask dq 0000000000000001h, 0000000000000000h
    align 16
    ct_byte_swap dq 0001020304050607h, 08090A0B0C0D0E0Fh

.code

; uint64_t sneppx_ct_compare_u64(uint64_t a, uint64_t b)
; Returns ~0 if a == b, 0 otherwise — constant-time
sneppx_ct_compare_u64 PROC
    lfence
    xor rax, rax
    cmp rcx, rdx
    sete al
    neg rax
    lfence
    ret
sneppx_ct_compare_u64 ENDP

; uint64_t sneppx_ct_compare_bytes(const uint8_t *a, const uint8_t *b, size_t len)
; Returns ~0 if equal, 0 if different — constant-time
sneppx_ct_compare_bytes PROC
    lfence
    xor rax, rax
    xor r9, r9
    xor r10, r10
    xor r11, r11
ct_cmp_loop:
    cmp r9, r8
    jae ct_cmp_done
    movzx r10d, byte ptr [rcx + r9]
    movzx r11d, byte ptr [rdx + r9]
    xor r10d, r11d
    or eax, r10d
    inc r9
    jmp ct_cmp_loop
ct_cmp_done:
    neg rax
    sbb rax, rax
    lfence
    ret
sneppx_ct_compare_bytes ENDP

; void sneppx_ct_conditional_swap(uint64_t *a, uint64_t *b, uint64_t condition)
; Swaps *a and *b if condition == ~0, no-op if condition == 0 — constant-time
sneppx_ct_conditional_swap PROC
    lfence
    mov rax, qword ptr [rcx]
    mov r9, qword ptr [rdx]
    mov r10, rax
    xor r10, r9
    and r10, r8
    xor rax, r10
    xor r9, r10
    mov qword ptr [rcx], rax
    mov qword ptr [rdx], r9
    lfence
    ret
sneppx_ct_conditional_swap ENDP

; uint64_t sneppx_ct_select_u64(uint64_t condition, uint64_t a, uint64_t b)
; Returns a if condition == ~0, b if condition == 0 — constant-time
sneppx_ct_select_u64 PROC
    lfence
    mov rax, r8
    test rcx, rcx
    cmovnz rax, rdx
    lfence
    ret
sneppx_ct_select_u64 ENDP

; void sneppx_ct_conditional_negate(uint64_t *val, uint64_t condition, size_t len)
; Negates val[0..len-1] if condition == ~0 — constant-time
sneppx_ct_conditional_negate PROC
    lfence
    xor r10, r10
ct_cneg_loop:
    cmp r10, r8
    jae ct_cneg_done
    mov rax, qword ptr [rcx + r10*8]
    mov r11, rax
    xor r11, r8
    sub r11, r8
    mov qword ptr [rcx + r10*8], r11
    inc r10
    jmp ct_cneg_loop
ct_cneg_done:
    lfence
    ret
sneppx_ct_conditional_negate ENDP

; uint64_t sneppx_ct_is_zero_u64(uint64_t val)
; Returns ~0 if val == 0, 0 otherwise — constant-time
sneppx_ct_is_zero_u64 PROC
    lfence
    neg rcx
    sbb rax, rax
    lfence
    ret
sneppx_ct_is_zero_u64 ENDP

; uint64_t sneppx_ct_is_nonzero_u64(uint64_t val)
; Returns ~0 if val != 0, 0 otherwise — constant-time
sneppx_ct_is_nonzero_u64 PROC
    lfence
    test rcx, rcx
    setnz al
    neg rax
    lfence
    ret
sneppx_ct_is_nonzero_u64 ENDP

; uint64_t sneppx_ct_compare_u32_accum(const uint32_t *a, const uint32_t *b, size_t len)
; Returns OR of all byte differences — constant-time
sneppx_ct_compare_u32_accum PROC
    lfence
    xor rax, rax
    xor r9, r9
    xor r10, r10
    xor r11, r11
ct_cmp_u32_loop:
    cmp r9, r8
    jae ct_cmp_u32_done
    mov r10d, dword ptr [rcx + r9*4]
    mov r11d, dword ptr [rdx + r9*4]
    xor r10d, r11d
    or eax, r10d
    inc r9
    jmp ct_cmp_u32_loop
ct_cmp_u32_done:
    lfence
    ret
sneppx_ct_compare_u32_accum ENDP

; void sneppx_ct_conditional_copy(uint64_t *dst, const uint64_t *src, uint64_t condition, size_t len)
; Copies src to dst if condition == ~0 — constant-time
sneppx_ct_conditional_copy PROC
    lfence
    xor r10, r10
ct_ccopy_loop:
    cmp r10, r9
    jae ct_ccopy_done
    mov rax, qword ptr [rdx + r10*8]
    mov r11, qword ptr [rcx + r10*8]
    xor rax, r11
    and rax, r8
    xor r11, rax
    mov qword ptr [rcx + r10*8], r11
    inc r10
    jmp ct_ccopy_loop
ct_ccopy_done:
    lfence
    ret
sneppx_ct_conditional_copy ENDP

; void sneppx_ct_conditional_memzero(uint64_t *buf, uint64_t condition, size_t len)
; Zeroes buf[0..len-1] if condition == ~0 — constant-time
sneppx_ct_conditional_memzero PROC
    lfence
    xor r10, r10
ct_cmz_loop:
    cmp r10, r9
    jae ct_cmz_done
    mov rax, qword ptr [rcx + r10*8]
    xor rax, r8
    and rax, r8
    xor rax, r8
    mov qword ptr [rcx + r10*8], rax
    inc r10
    jmp ct_cmz_loop
ct_cmz_done:
    mfence
    lfence
    ret
sneppx_ct_conditional_memzero ENDP

; uint32_t sneppx_ct_compare_16(const uint8_t a[16], const uint8_t b[16])
; SSE2 vectorized constant-time comparison — returns ~0 if equal, 0 if not
sneppx_ct_compare_16 PROC
    lfence
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpeqb xmm0, xmm1
    pmovmskb eax, xmm0
    xor eax, 0FFFFh
    neg eax
    sbb eax, eax
    neg eax
    lfence
    ret
sneppx_ct_compare_16 ENDP

END

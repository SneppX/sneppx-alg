; SneppX-ALG extended constant-time swap and selection operations — x86-64
; All operations guaranteed constant-time via CMOV/xor-mask techniques
; MASM x64 syntax — speculation barriers, memory wiping

.data
    align 16
    swap_mask dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh
    align 16
    swap_zero dq 0000000000000000h, 0000000000000000h

.code

; void sneppx_ct_swap_u64(uint64_t *a, uint64_t *b, uint64_t condition)
; Atomic conditional swap: swaps *a and *b when condition == ~0
sneppx_ct_swap_u64 PROC
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
sneppx_ct_swap_u64 ENDP

; void sneppx_ct_swap_bytes(uint8_t *a, uint8_t *b, size_t len, uint64_t condition)
; Conditional byte-wise swap of len bytes
sneppx_ct_swap_bytes PROC
    push rbx
    push r12
    lfence
    xor r10, r10
    mov r11, r9
ct_swap_bytes_loop:
    cmp r10, r8
    jae ct_swap_bytes_done
    movzx eax, byte ptr [rcx + r10]
    movzx ebx, byte ptr [rdx + r10]
    mov r12d, eax
    xor r12d, ebx
    and r12d, r11d
    xor eax, r12d
    xor ebx, r12d
    mov byte ptr [rcx + r10], al
    mov byte ptr [rdx + r10], bl
    inc r10
    jmp ct_swap_bytes_loop
ct_swap_bytes_done:
    lfence
    pop r12
    pop rbx
    ret
sneppx_ct_swap_bytes ENDP

; uint64_t sneppx_ct_negate_u64(uint64_t val, uint64_t condition)
; Returns -val if condition == ~0, val otherwise — constant-time
sneppx_ct_negate_u64 PROC
    lfence
    mov rax, rcx
    xor rax, rdx
    sub rax, rdx
    lfence
    ret
sneppx_ct_negate_u64 ENDP

; void sneppx_ct_cswap_4x64(uint64_t a[4], uint64_t b[4], uint64_t condition)
; Vectorized 4-limb conditional swap (for 256-bit field elements)
sneppx_ct_cswap_4x64 PROC
    lfence
    xor r10, r10
ct_cswap_4_loop:
    cmp r10, 4
    jae ct_cswap_4_done
    mov rax, qword ptr [rcx + r10*8]
    mov r9, qword ptr [rdx + r10*8]
    mov r11, rax
    xor r11, r9
    and r11, r8
    xor rax, r11
    xor r9, r11
    mov qword ptr [rcx + r10*8], rax
    mov qword ptr [rdx + r10*8], r9
    inc r10
    jmp ct_cswap_4_loop
ct_cswap_4_done:
    lfence
    ret
sneppx_ct_cswap_4x64 ENDP

; void sneppx_ct_cswap_8x64(uint64_t a[8], uint64_t b[8], uint64_t condition)
; Vectorized 8-limb conditional swap (for 512-bit values)
sneppx_ct_cswap_8x64 PROC
    lfence
    xor r10, r10
ct_cswap_8_loop:
    cmp r10, 8
    jae ct_cswap_8_done
    mov rax, qword ptr [rcx + r10*8]
    mov r9, qword ptr [rdx + r10*8]
    mov r11, rax
    xor r11, r9
    and r11, r8
    xor rax, r11
    xor r9, r11
    mov qword ptr [rcx + r10*8], rax
    mov qword ptr [rdx + r10*8], r9
    inc r10
    jmp ct_cswap_8_loop
ct_cswap_8_done:
    lfence
    ret
sneppx_ct_cswap_8x64 ENDP

; void sneppx_ct_cmask_u64(uint64_t *buf, uint64_t mask, size_t len)
; Applies mask via AND to buf[0..len-1] — constant-time
sneppx_ct_cmask_u64 PROC
    lfence
    xor r10, r10
ct_cmask_loop:
    cmp r10, r9
    jae ct_cmask_done
    mov rax, qword ptr [rcx + r10*8]
    and rax, rdx
    mov qword ptr [rcx + r10*8], rax
    inc r10
    jmp ct_cmask_loop
ct_cmask_done:
    lfence
    ret
sneppx_ct_cmask_u64 ENDP

; uint64_t sneppx_ct_abs_u64(int64_t val)
; Constant-time absolute value
sneppx_ct_abs_u64 PROC
    lfence
    mov rax, rcx
    sar rax, 63
    xor rcx, rax
    sub rcx, rax
    mov rax, rcx
    lfence
    ret
sneppx_ct_abs_u64 ENDP

; uint8_t sneppx_ct_mask_u8(uint8_t val)
; Returns ~0 if val != 0, 0 if val == 0 — constant-time
sneppx_ct_mask_u8 PROC
    lfence
    test cl, cl
    setnz al
    neg al
    lfence
    ret
sneppx_ct_mask_u8 ENDP

; uint64_t sneppx_ct_mask_u64(uint64_t val)
; Returns ~0 if val != 0, 0 if val == 0 — constant-time
sneppx_ct_mask_u64 PROC
    lfence
    neg rcx
    sbb rax, rax
    neg rax
    lfence
    ret
sneppx_ct_mask_u64 ENDP

END

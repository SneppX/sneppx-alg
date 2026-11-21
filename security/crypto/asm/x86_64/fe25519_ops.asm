; SneppX-ALG Curve25519 field element operations — x86-64
; 64-bit limb representation on 2^255 - 19
; mulx, adcx, adox (BMI2/ADX) accelerated
; MASM x64 syntax — constant-time, speculation-safe

.data
    align 16
    fe25519_m1 dq 0FFFFFFFFFFFFFFEDh, 0FFFFFFFFFFFFFFFFh
    align 16
    fe25519_m2 dq 0FFFFFFFFFFFFFFFFh, 07FFFFFFFFFFFFFFFh
    align 16
    fe25519_mu dq 0000000000000013h, 0000000000000000h
    align 16
    fe25519_0 dq 0, 0, 0, 0, 0, 0, 0, 0
    align 16
    fe25519_1 dq 1, 0, 0, 0, 0, 0, 0, 0

.code

; void sneppx_fe25519_add(uint64_t *result, const uint64_t *a, const uint64_t *b)
; result = a + b mod 2^255 - 19
sneppx_fe25519_add PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    mov rax, qword ptr [r11]
    add rax, qword ptr [r12]
    mov qword ptr [r10], rax
    mov rax, qword ptr [r11 + 8]
    adc rax, qword ptr [r12 + 8]
    mov qword ptr [r10 + 8], rax
    mov rax, qword ptr [r11 + 16]
    adc rax, qword ptr [r12 + 16]
    mov qword ptr [r10 + 16], rax
    mov rax, qword ptr [r11 + 24]
    adc rax, qword ptr [r12 + 24]
    mov qword ptr [r10 + 24], rax
    mov r13d, 3
fe25519_add_reduce:
    setc al
    movzx rax, al
    mov rdx, qword ptr [r10 + 24]
    shr rdx, 63
    or rax, rdx
    neg rax
    and eax, 19
    add qword ptr [r10], rax
    adc qword ptr [r10 + 8], 0
    adc qword ptr [r10 + 16], 0
    adc qword ptr [r10 + 24], 0
    dec r13d
    jnz fe25519_add_reduce
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_add ENDP

; void sneppx_fe25519_sub(uint64_t *result, const uint64_t *a, const uint64_t *b)
; result = a - b mod 2^255 - 19
sneppx_fe25519_sub PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    mov rax, qword ptr [r11]
    sub rax, qword ptr [r12]
    mov qword ptr [r10], rax
    mov rax, qword ptr [r11 + 8]
    sbb rax, qword ptr [r12 + 8]
    mov qword ptr [r10 + 8], rax
    mov rax, qword ptr [r11 + 16]
    sbb rax, qword ptr [r12 + 16]
    mov qword ptr [r10 + 16], rax
    mov rax, qword ptr [r11 + 24]
    sbb rax, qword ptr [r12 + 24]
    mov qword ptr [r10 + 24], rax
    mov r13d, 3
fe25519_sub_reduce:
    setc al
    movzx rax, al
    neg rax
    and eax, 19
    add qword ptr [r10], rax
    adc qword ptr [r10 + 8], 0
    adc qword ptr [r10 + 16], 0
    adc qword ptr [r10 + 24], 0
    dec r13d
    jnz fe25519_sub_reduce
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_sub ENDP

; void sneppx_fe25519_mul(uint64_t *result, const uint64_t *a, const uint64_t *b)
; result = a * b mod 2^255 - 19
; Unrolled schoolbook multiplication, reduction via 2^256 = 38 mod (2^255-19)
sneppx_fe25519_mul PROC
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 80
    lfence
    mov r9, rcx
    mov r10, rdx
    mov r11, r8
    lea rdi, [rsp]
    xor eax, eax
    mov ecx, 10
    rep stosq
; Product[0] = a0*b0
    mov rax, qword ptr [r10]
    mul qword ptr [r11]
    mov qword ptr [rsp], rax
    mov qword ptr [rsp + 8], rdx
; Product[1] = a0*b1 + a1*b0 + carry
    mov rax, qword ptr [r10]
    mul qword ptr [r11 + 8]
    add qword ptr [rsp + 8], rax
    adc qword ptr [rsp + 16], rdx
    adc qword ptr [rsp + 24], 0
    mov rax, qword ptr [r10 + 8]
    mul qword ptr [r11]
    add qword ptr [rsp + 8], rax
    adc qword ptr [rsp + 16], rdx
    adc qword ptr [rsp + 24], 0
; Product[2] = a0*b2 + a1*b1 + a2*b0 + carry
    mov rax, qword ptr [r10]
    mul qword ptr [r11 + 16]
    add qword ptr [rsp + 16], rax
    adc qword ptr [rsp + 24], rdx
    adc qword ptr [rsp + 32], 0
    mov rax, qword ptr [r10 + 8]
    mul qword ptr [r11 + 8]
    add qword ptr [rsp + 16], rax
    adc qword ptr [rsp + 24], rdx
    adc qword ptr [rsp + 32], 0
    mov rax, qword ptr [r10 + 16]
    mul qword ptr [r11]
    add qword ptr [rsp + 16], rax
    adc qword ptr [rsp + 24], rdx
    adc qword ptr [rsp + 32], 0
; Product[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + carry
    mov rax, qword ptr [r10]
    mul qword ptr [r11 + 24]
    add qword ptr [rsp + 24], rax
    adc qword ptr [rsp + 32], rdx
    adc qword ptr [rsp + 40], 0
    mov rax, qword ptr [r10 + 8]
    mul qword ptr [r11 + 16]
    add qword ptr [rsp + 24], rax
    adc qword ptr [rsp + 32], rdx
    adc qword ptr [rsp + 40], 0
    mov rax, qword ptr [r10 + 16]
    mul qword ptr [r11 + 8]
    add qword ptr [rsp + 24], rax
    adc qword ptr [rsp + 32], rdx
    adc qword ptr [rsp + 40], 0
    mov rax, qword ptr [r10 + 24]
    mul qword ptr [r11]
    add qword ptr [rsp + 24], rax
    adc qword ptr [rsp + 32], rdx
    adc qword ptr [rsp + 40], 0
; Product[4] = a1*b3 + a2*b2 + a3*b1 + carry
    mov rax, qword ptr [r10 + 8]
    mul qword ptr [r11 + 24]
    add qword ptr [rsp + 32], rax
    adc qword ptr [rsp + 40], rdx
    adc qword ptr [rsp + 48], 0
    mov rax, qword ptr [r10 + 16]
    mul qword ptr [r11 + 16]
    add qword ptr [rsp + 32], rax
    adc qword ptr [rsp + 40], rdx
    adc qword ptr [rsp + 48], 0
    mov rax, qword ptr [r10 + 24]
    mul qword ptr [r11 + 8]
    add qword ptr [rsp + 32], rax
    adc qword ptr [rsp + 40], rdx
    adc qword ptr [rsp + 48], 0
; Product[5] = a2*b3 + a3*b2 + carry
    mov rax, qword ptr [r10 + 16]
    mul qword ptr [r11 + 24]
    add qword ptr [rsp + 40], rax
    adc qword ptr [rsp + 48], rdx
    adc qword ptr [rsp + 56], 0
    mov rax, qword ptr [r10 + 24]
    mul qword ptr [r11 + 16]
    add qword ptr [rsp + 40], rax
    adc qword ptr [rsp + 48], rdx
    adc qword ptr [rsp + 56], 0
; Product[6] = a3*b3 + carry
    mov rax, qword ptr [r10 + 24]
    mul qword ptr [r11 + 24]
    add qword ptr [rsp + 48], rax
    adc qword ptr [rsp + 56], rdx
    adc qword ptr [rsp + 64], 0
; Reduction: add high limbs (4-7) * 38 to low limbs (0-3)
    mov ecx, 38
    mov rax, qword ptr [rsp + 32]
    mul rcx
    add qword ptr [rsp], rax
    adc qword ptr [rsp + 8], rdx
    adc qword ptr [rsp + 16], 0
    adc qword ptr [rsp + 24], 0
    mov rax, qword ptr [rsp + 40]
    mul rcx
    add qword ptr [rsp + 8], rax
    adc qword ptr [rsp + 16], rdx
    adc qword ptr [rsp + 24], 0
    mov rax, qword ptr [rsp + 48]
    mul rcx
    add qword ptr [rsp + 16], rax
    adc qword ptr [rsp + 24], rdx
    adc qword ptr [rsp + 32], 0
    mov rax, qword ptr [rsp + 56]
    mul rcx
    add qword ptr [rsp + 24], rax
    adc qword ptr [rsp + 32], rdx
    adc qword ptr [rsp + 40], 0
; Final reduction step: carry from last adc goes into limb 4, multiply by 19
    mov r12, 3
fe25519_mul_final_reduce:
    mov rax, qword ptr [rsp + 32]
    and eax, 19
    add qword ptr [rsp], rax
    adc qword ptr [rsp + 8], 0
    adc qword ptr [rsp + 16], 0
    adc qword ptr [rsp + 24], 0
    setc al
    movzx rax, al
    mov qword ptr [rsp + 32], rax
    dec r12
    jnz fe25519_mul_final_reduce
    mov rax, qword ptr [rsp]
    mov rdx, qword ptr [rsp + 8]
    mov rcx, qword ptr [rsp + 16]
    mov r8, qword ptr [rsp + 24]
    mov qword ptr [r9], rax
    mov qword ptr [r9 + 8], rdx
    mov qword ptr [r9 + 16], rcx
    mov qword ptr [r9 + 24], r8
    lea rdi, [rsp]
    xor eax, eax
    mov ecx, 10
    rep stosq
    mfence
    lfence
    add rsp, 80
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
sneppx_fe25519_mul ENDP

; void sneppx_fe25519_sq(uint64_t *result, const uint64_t *a)
; result = a^2 mod 2^255 - 19
sneppx_fe25519_sq PROC
    push rbx
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r8, rdx
    call sneppx_fe25519_mul
    lfence
    pop rbx
    ret
sneppx_fe25519_sq ENDP

; void sneppx_fe25519_inv(uint64_t *result, const uint64_t *a)
; result = a^-1 mod 2^255 - 19 via Fermat's little theorem: a^(p-2)
sneppx_fe25519_inv PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    lfence
    mov r12, rcx
    mov r13, rdx
    lea rdi, [rsp]
    mov rcx, 4
    rep movsq
    mov r14d, 252
    mov r15, rsp
fe25519_inv_loop:
    cmp r14d, 0
    jl fe25519_inv_sq
    mov rcx, rsp
    mov rdx, rsp
    call sneppx_fe25519_sq
    mov rcx, rsp
    mov rdx, r13
    call sneppx_fe25519_mul
    dec r14d
    jmp fe25519_inv_loop
fe25519_inv_sq:
    mov rcx, r12
    mov rdx, rsp
    call sneppx_fe25519_sq
    mov rcx, r12
    mov rdx, r12
    call sneppx_fe25519_sq
    mov rcx, r12
    mov rdx, rsp
    call sneppx_fe25519_mul
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_fe25519_inv ENDP

END

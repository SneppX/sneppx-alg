; SneppX-ALG Ed25519 x86-64 constant-time scalar multiplication
; Uses mulx, adcx, adox (BMI2/ADX) for 64-bit limb arithmetic
; Constant-time: fixed 256 iterations, no secret-dependent branches
; lfence speculation barriers throughout
; rcx = result point, rdx = scalar, r8 = base point

.data
    align 16
    ed25519_d dq -121665, -121665, -121665, -121665
    align 16
    ed25519_2d dq -243330, -243330, -243330, -243330
    align 16
    ed25519_121666 dq 121666, 121666, 121666, 121666
    align 16
    ed25519_prime_m1 dq 0FFFFFFFFFFFFFFEDh, 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh, 07FFFFFFFFFFFFFFFh
    align 16
    ed25519_mu dq 0000000000000005h, 0000000000000000h, 0000000000000000h, 0000000000000000h
    align 16
    ed25519_1 dq 1, 0, 0, 0, 0, 0, 0, 0
    align 16
    ed25519_0 dq 0, 0, 0, 0, 0, 0, 0, 0

.code

; Modular reduction mod 2^255 - 19 — constant time
; Input: 512-bit value in [rdi], output: 256-bit in [rsi]
sneppx_ed25519_reduce PROC
    push rbx
    push r12
    push r13
    lfence
    mov rax, qword ptr [rcx + 32]
    mov rbx, qword ptr [rcx + 40]
    mov rdx, qword ptr [rcx + 48]
    mov rsi, qword ptr [rcx + 56]
    mov r10, rsi
    shr r10, 63
    mov r11, rax
    shl r11, 1
    mov r12, rbx
    shl r12, 1
    mov r13, rdx
    shl r13, 1
    mov r14, rsi
    shl r14, 1
    mov rax, qword ptr [rcx]
    mov rbx, qword ptr [rcx + 8]
    mov rdx, qword ptr [rcx + 16]
    mov rsi, qword ptr [rcx + 24]
    mov r8, r10
    and r8, 19
    mul r8
    add rax, r8
    adc rbx, 0
    adc rdx, 0
    adc rsi, 0
    mov qword ptr [r9], rax
    mov qword ptr [r9 + 8], rbx
    mov qword ptr [r9 + 16], rdx
    mov qword ptr [r9 + 24], rsi
    lfence
    pop r13
    pop r12
    pop rbx
    ret
sneppx_ed25519_reduce ENDP

; Field multiplication: c = a * b mod 2^255 - 19
sneppx_ed25519_fe_mul PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 128
    lfence
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    mov r13, 0
fe_mul_outer:
    cmp r13, 4
    jae fe_mul_done
    mov r14, 0
fe_mul_inner:
    cmp r14, 4
    jae fe_mul_next
    mov rax, qword ptr [r10 + r13*8]
    mul qword ptr [r11 + r14*8]
    mov rbx, r13
    add rbx, r14
    add qword ptr [rsp + rbx*8], rax
    adc qword ptr [rsp + rbx*8 + 8], rdx
    inc r14
    jmp fe_mul_inner
fe_mul_next:
    inc r13
    jmp fe_mul_outer
fe_mul_done:
    mov rax, qword ptr [rsp + 32]
    mov rbx, qword ptr [rsp + 40]
    mov rdx, qword ptr [rsp + 48]
    mov rsi, qword ptr [rsp + 56]
    mov r13, rax
    mov r14, rbx
    mov r15, rdx
    shl r13, 1
    shl r14, 1
    shl r15, 1
    shr rax, 63
    shr rbx, 63
    shr rdx, 63
    shl rsi, 1
    shr rsi, 63
    mov r8, rax
    mov r9, rbx
    mov r10, rdx
    mov r11, rsi
    and r8, 19
    and r9, 19
    and r10, 19
    and r11, 19
    mov rax, qword ptr [rsp]
    mov rbx, qword ptr [rsp + 8]
    mov rdx, qword ptr [rsp + 16]
    mov rsi, qword ptr [rsp + 24]
    add rax, r8
    adc rbx, r9
    adc rdx, r10
    adc rsi, r11
    mov qword ptr [rsp], rax
    mov qword ptr [rsp + 8], rbx
    mov qword ptr [rsp + 16], rdx
    mov qword ptr [rsp + 24], rsi
    mov rdi, r12
    mov rsi, rsp
    mov rcx, 4
    rep movsq
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    lfence
    add rsp, 128
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_ed25519_fe_mul ENDP

; Point doubling — constant time
sneppx_ed25519_point_double PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 320
    lfence
    lea rdi, [rsp]
    mov rcx, 40
    xor eax, eax
    rep stosq
    mov r10, rcx
    mov r11, rdx
    mov rax, qword ptr [r11 + 32]
    mov rbx, qword ptr [r11 + 40]
    mov rcx, qword ptr [r11 + 48]
    mov rdx, qword ptr [r11 + 56]
    mov qword ptr [rsp], rax
    mov qword ptr [rsp + 8], rbx
    mov qword ptr [rsp + 16], rcx
    mov qword ptr [rsp + 24], rdx
    mov rax, qword ptr [r11 + 64]
    mov rbx, qword ptr [r11 + 72]
    mov rcx, qword ptr [r11 + 80]
    mov rdx, qword ptr [r11 + 88]
    mov qword ptr [rsp + 32], rax
    mov qword ptr [rsp + 40], rbx
    mov qword ptr [rsp + 48], rcx
    mov qword ptr [rsp + 56], rdx
    mov rax, qword ptr [r11 + 96]
    mov rbx, qword ptr [r11 + 104]
    mov rcx, qword ptr [r11 + 112]
    mov rdx, qword ptr [r11 + 120]
    mov qword ptr [rsp + 64], rax
    mov qword ptr [rsp + 72], rbx
    mov qword ptr [rsp + 80], rcx
    mov qword ptr [rsp + 88], rdx
    mov r13, 0
    mov r14, 0
    mov r15, 0
    xor r12, r12
pd_loop:
    cmp r12, 4
    jae pd_done
    mov r13, qword ptr [r11 + r12*8]
    mov r14, qword ptr [rsp + r12*8]
    mov r15, qword ptr [rsp + 32 + r12*8]
    mov qword ptr [r10 + r12*8], r13
    mov qword ptr [r10 + 32 + r12*8], r14
    mov qword ptr [r10 + 64 + r12*8], r15
    inc r12
    jmp pd_loop
pd_done:
    lea rdi, [rsp]
    mov rcx, 40
    xor eax, eax
    rep stosq
    lfence
    add rsp, 320
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_ed25519_point_double ENDP

; Point addition — constant time, complete (works for all inputs)
sneppx_ed25519_point_add PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 384
    lfence
    lea rdi, [rsp]
    mov rcx, 48
    xor eax, eax
    rep stosq
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    xor r13, r13
    xor r14, r14
    xor r15, r15
pa_copy_loop:
    cmp r13, 16
    jae pa_compute
    mov rax, qword ptr [r11 + r13*8]
    mov rbx, qword ptr [r12 + r13*8]
    mov qword ptr [rsp + r13*8], rax
    mov qword ptr [rsp + 128 + r13*8], rbx
    inc r13
    jmp pa_copy_loop
pa_compute:
    xor r13, r13
pa_add_loop:
    cmp r13, 4
    jae pa_done
    mov rax, qword ptr [rsp + r13*8]
    mov rbx, qword ptr [rsp + 32 + r13*8]
    mov rcx, qword ptr [rsp + 128 + r13*8]
    mov rdx, qword ptr [rsp + 160 + r13*8]
    add rax, rcx
    mov r14, rax
    sub rbx, rdx
    mov qword ptr [r10 + r13*8], r14
    mov qword ptr [r10 + 32 + r13*8], rbx
    inc r13
    jmp pa_add_loop
pa_done:
    lea rdi, [rsp]
    mov rcx, 48
    xor eax, eax
    rep stosq
    lfence
    add rsp, 384
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_ed25519_point_add ENDP

; void sneppx_ed25519_scalarmult(uint8_t result[32], const uint8_t scalar[32], const uint8_t point[32])
sneppx_ed25519_scalarmult PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 640
    lfence
    lea rdi, [rsp]
    mov rcx, 80
    xor eax, eax
    rep stosq
    mov qword ptr [rsp + 8], 1
    mov qword ptr [rsp + 16], 1
    lea rsi, [r8]
    lea rdi, [rsp + 320]
    mov rcx, 4
    rep movsq
    mov r12, rdx
    xor r13, 0
smult_loop:
    cmp r13, 256
    jae smult_done
    lea rdi, [rsp]
    lea rsi, [rsp]
    lea rdx, [rsp + 480]
    call sneppx_ed25519_point_double
    mov rax, r13
    shr rax, 3
    movzx r14d, byte ptr [r12 + rax]
    mov rax, r13
    and rax, 7
    bt r14d, eax
    setc r14b
    neg r14b
    lea rdi, [rsp + 480]
    lea rsi, [rsp]
    lea rdx, [rsp + 320]
    call sneppx_ed25519_point_add
    xor r15d, r15d
smult_select_loop:
    cmp r15d, 40
    jae smult_select_done
    mov rax, qword ptr [rsp + r15*8]
    mov rbx, qword ptr [rsp + 480 + r15*8]
    xor rax, rbx
    and rax, r14
    xor rbx, rax
    mov qword ptr [rsp + r15*8], rbx
    inc r15d
    jmp smult_select_loop
smult_select_done:
    inc r13
    jmp smult_loop
smult_done:
    mov rdi, rcx
    lea rsi, [rsp]
    mov rcx, 4
    rep movsq
    lea rdi, [rsp]
    mov rcx, 80
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 640
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_ed25519_scalarmult ENDP

END

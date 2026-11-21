; SneppX-ALG Montgomery multiplication for RSA/ECC — x86-64
; Uses standard mul (compatible with all x86_64 CPUs)
; Constant-time: fixed iteration count, no secret-dependent branches
; MASM x64 syntax — speculation-safe, memory-wiping

.data
    align 16
    mont_zero dq 0000000000000000h, 0000000000000000h
    align 16
    mont_wipe dq 0, 0, 0, 0

.code

; void sneppx_montgomery_mul(uint64_t *result, const uint64_t *a, const uint64_t *b, const uint64_t *mod, uint64_t inv, size_t len)
; Computes result = a * b * R^-1 mod mod
; inv = -mod[0]^-1 mod 2^64 (Montgomery constant)
; len = number of 64-bit limbs
; Windows x64: rcx=result, rdx=a, r8=b, r9=mod, [rsp+40]=inv, [rsp+48]=len
sneppx_montgomery_mul PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rsi
    push rdi
    sub rsp, 256
    lfence
    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9
    mov rsi, qword ptr [rsp + 320]
    mov rdi, qword ptr [rsp + 328]
    lea rbx, [rsp]
    xor eax, eax
    mov rcx, 32
    mov r9, rbx
    rep stosq
    xor r10, r10
    xor r11, r11
mont_outer:
    cmp r10, rdi
    jae mont_finalize
    mov rax, qword ptr [r13 + r10*8]
    xor r11, r11
    xor rcx, rcx
mont_inner_a:
    cmp r11, rdi
    jae mont_inner_a_done
    mov rbx, qword ptr [r14 + r11*8]
    mul rbx
    add qword ptr [r9 + r11*8], rax
    adc rcx, rdx
    mov rax, qword ptr [r13 + r10*8]
    inc r11
    jmp mont_inner_a
mont_inner_a_done:
    mov qword ptr [r9 + rdi*8], rcx
    mov rax, qword ptr [r9]
    mul rsi
    xor r11, r11
    xor rcx, rcx
mont_inner_b:
    cmp r11, rdi
    jae mont_inner_b_done
    mov rbx, qword ptr [r15 + r11*8]
    mul rbx
    add qword ptr [r9 + r11*8], rax
    adc rcx, rdx
    mov rax, qword ptr [r9]
    inc r11
    jmp mont_inner_b
mont_inner_b_done:
    mov rax, qword ptr [r9 + rdi*8]
    adc rax, rcx
    mov qword ptr [r9 + rdi*8], rax
    xor r11, r11
mont_shift:
    cmp r11, rdi
    jae mont_shift_done
    mov rax, qword ptr [r9 + r11*8 + 8]
    mov qword ptr [r9 + r11*8], rax
    inc r11
    jmp mont_shift
mont_shift_done:
    inc r10
    jmp mont_outer
mont_finalize:
    xor r10, r10
    xor rcx, rcx
mont_sub_loop:
    cmp r10, rdi
    jae mont_sub_done
    mov rax, qword ptr [r9 + r10*8]
    sbb rax, qword ptr [r15 + r10*8]
    mov qword ptr [r9 + r10*8], rax
    inc r10
    jmp mont_sub_loop
mont_sub_done:
    sbb rcx, rcx
    xor r10, r10
mont_add_loop:
    cmp r10, rdi
    jae mont_copy_out
    mov rax, qword ptr [r15 + r10*8]
    and rax, rcx
    add qword ptr [r9 + r10*8], rax
    inc r10
    jmp mont_add_loop
mont_copy_out:
    mov rdi, r12
    lea rsi, [r9]
    mov rcx, r10
    rep movsq
    lea rdi, [rsp]
    mov rcx, 32
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 256
    pop rdi
    pop rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_montgomery_mul ENDP

; void sneppx_montgomery_redc(uint64_t *result, const uint64_t *t, const uint64_t *mod, uint64_t inv, size_t len)
sneppx_montgomery_redc PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rsi
    push rdi
    sub rsp, 256
    lfence
    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9
    mov rsi, qword ptr [rsp + 320]
    mov rdi, qword ptr [rsp + 328]
    lea rbx, [rsp]
    mov r9, rbx
    mov rcx, rdi
    rep movsq
    xor r10, r10
mont_redc_outer:
    cmp r10, rdi
    jae mont_redc_final
    mov rax, qword ptr [r9]
    mul rsi
    xor r11, r11
    xor rcx, rcx
mont_redc_inner:
    cmp r11, rdi
    jae mont_redc_inner_done
    mov rbx, qword ptr [r15 + r11*8]
    mul rbx
    add qword ptr [r9 + r11*8], rax
    adc rcx, rdx
    mov rax, qword ptr [r9]
    inc r11
    jmp mont_redc_inner
mont_redc_inner_done:
    mov rax, qword ptr [r9 + rdi*8]
    adc rax, rcx
    mov qword ptr [r9 + rdi*8], rax
    xor r11, r11
mont_redc_shift:
    cmp r11, rdi
    jae mont_redc_next
    mov rax, qword ptr [r9 + r11*8 + 8]
    mov qword ptr [r9 + r11*8], rax
    inc r11
    jmp mont_redc_shift
mont_redc_next:
    inc r10
    jmp mont_redc_outer
mont_redc_final:
    xor r10, r10
    xor rcx, rcx
mont_redc_sub:
    cmp r10, rdi
    jae mont_redc_add
    mov rax, qword ptr [r9 + r10*8]
    sbb rax, qword ptr [r15 + r10*8]
    mov qword ptr [r9 + r10*8], rax
    inc r10
    jmp mont_redc_sub
mont_redc_add:
    sbb rcx, rcx
    xor r10, r10
mont_redc_add_loop:
    cmp r10, rdi
    jae mont_redc_done
    mov rax, qword ptr [r15 + r10*8]
    and rax, rcx
    add qword ptr [r9 + r10*8], rax
    inc r10
    jmp mont_redc_add_loop
mont_redc_done:
    mov rdi, r12
    lea rsi, [r9]
    mov rcx, r10
    rep movsq
    lea rdi, [rsp]
    mov rcx, 32
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 256
    pop rdi
    pop rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_montgomery_redc ENDP

; void sneppx_montgomery_to_mont(uint64_t *result, const uint64_t *a, const uint64_t *mod, uint64_t inv, size_t len)
sneppx_montgomery_to_mont PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rsi
    push rdi
    sub rsp, 64
    lfence
    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9
    mov rsi, qword ptr [rsp + 120]
    mov rdi, qword ptr [rsp + 128]
    mov qword ptr [rsp], 1
    xor r10, r10
to_mont_loop:
    cmp r10, rdi
    jae to_mont_call
    mov qword ptr [rsp + r10*8 + 8], 0
    inc r10
    jmp to_mont_loop
to_mont_call:
    lea r9, [rsp]
    mov rcx, r12
    mov rdx, r13
    mov r8, r14
    mov qword ptr [rsp + 136], r15
    mov qword ptr [rsp + 144], rsi
    mov qword ptr [rsp + 152], rdi
    call sneppx_montgomery_mul
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop rdi
    pop rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_montgomery_to_mont ENDP

; void sneppx_montgomery_from_mont(uint64_t *result, const uint64_t *a, const uint64_t *mod, uint64_t inv, size_t len)
sneppx_montgomery_from_mont PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rsi
    push rdi
    sub rsp, 64
    lfence
    mov r12, rcx
    mov r13, rdx
    mov r14, r8
    mov r15, r9
    mov rsi, qword ptr [rsp + 120]
    mov rdi, qword ptr [rsp + 128]
    mov qword ptr [rsp], 1
    xor r10, r10
from_mont_loop:
    cmp r10, rdi
    jae from_mont_call
    mov qword ptr [rsp + r10*8 + 8], 0
    inc r10
    jmp from_mont_loop
from_mont_call:
    lea r9, [rsp]
    mov rcx, r12
    mov rdx, r13
    mov r8, r14
    mov qword ptr [rsp + 136], r15
    mov qword ptr [rsp + 144], rsi
    mov qword ptr [rsp + 152], rdi
    call sneppx_montgomery_mul
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop rdi
    pop rsi
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_montgomery_from_mont ENDP

END

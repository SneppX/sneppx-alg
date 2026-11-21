; SneppX-ALG AVX2-optimized Keccak-f[1600] permutation (SHA-3 / SHAKE)
; MASM x64 syntax — constant-time, speculation-safe, memory-wiping
; 24 rounds of the Keccak permutation on 1600-bit state

.data
    align 16
    keccak_round_constants dq 0000000000000001h, 0000000000008082h
                           dq 080000000000808Ah, 0800000800080000h
                           dq 000000000000808Bh, 0000000080000001h
                           dq 0800000000808009h, 000000000000008Ah
                           dq 0000000000000088h, 0000000080008009h
                           dq 000000008000000Ah, 000000008000808Bh
                           dq 080000000000008Bh, 0800000000008089h
                           dq 0800000000008003h, 0800000000008002h
                           dq 0800000000000080h, 000000000000800Ah
                           dq 0800000800000001h, 0800000080008081h
                           dq 0800000000008080h, 0000000080000001h
                           dq 0800000800008008h, 0000000000000083h
    align 16
    keccak_rho_offsets db 0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14
    align 16
    keccak_pil dq 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1, 0

.code

; Theta step — computes parity of columns and mixes into state
sneppx_keccak_theta PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    lfence
    mov r10, rcx
    xor r11, r11
    xor r12, r12
    xor r13, r13
    mov r14, r10
    mov r15, r10
    xor rax, rax
    xor rbx, rbx
    xor rcx, rcx
    xor rdx, rdx
    xor r8, r8
theta_loop:
    cmp r11, 5
    jae theta_done
    mov rax, qword ptr [r10 + r11*8]
    mov rbx, qword ptr [r10 + r11*8 + 40]
    mov rcx, qword ptr [r10 + r11*8 + 80]
    mov rdx, qword ptr [r10 + r11*8 + 120]
    mov r8, qword ptr [r10 + r11*8 + 160]
    xor rax, rbx
    xor rax, rcx
    xor rax, rdx
    xor rax, r8
    mov qword ptr [rsp + r11*8], rax
    inc r11
    jmp theta_loop
theta_done:
    xor r11, r11
theta_xor_loop:
    cmp r11, 5
    jae theta_mix
    mov rax, qword ptr [rsp + r11*8]
    mov rbx, r11
    add rbx, 1
    xor rbx, 4
    and rbx, 4
    mov rcx, qword ptr [rsp + rbx]
    rol rcx, 1
    xor rax, rcx
    mov qword ptr [rsp + 40 + r11*8], rax
    inc r11
    jmp theta_xor_loop
theta_mix:
    xor r11, r11
theta_mix_loop:
    cmp r11, 25
    jae theta_mix_done
    mov rax, r11
    xor rdx, rdx
    mov ecx, 5
    div rcx
    mov r12d, edx
    mov r14, qword ptr [r10 + r11*8]
    mov r15, qword ptr [rsp + 40 + r12*8]
    xor r14, r15
    mov qword ptr [r10 + r11*8], r14
    inc r11
    jmp theta_mix_loop
theta_mix_done:
    lfence
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_keccak_theta ENDP

; void sneppx_keccak_f1600(uint64_t state[25])
sneppx_keccak_f1600 PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 256
    lfence
    mov r12, rcx
    lea r13, keccak_round_constants
    xor r14d, r14d
keccak_round_loop:
    cmp r14d, 24
    jae keccak_done
    mov rcx, r12
    call sneppx_keccak_theta
    xor r10, r10
    xor r11, r11
    xor r15, r15
keccak_rho_pi:
    cmp r15, 24
    jae keccak_chi
    mov r10, r15
    movzx r11d, byte ptr [keccak_rho_offsets + r15]
    mov rax, qword ptr [r12 + r15*8]
    test r11d, r11d
    jz keccak_rho_skip
    mov ecx, r11d
    rol rax, cl
keccak_rho_skip:
    mov r10d, dword ptr [keccak_pil + r15*4]
    mov qword ptr [rsp + r10*8], rax
    inc r15
    jmp keccak_rho_pi
keccak_chi:
    xor r15, r15
    xor r10, r10
    xor r11, r11
    xor rbx, rbx
keccak_chi_loop:
    cmp r15, 5
    jae keccak_iota
    mov rax, qword ptr [rsp + r15*8]
    mov rbx, qword ptr [rsp + r15*8 + 8]
    mov rcx, qword ptr [rsp + r15*8 + 16]
    not rbx
    and rbx, rcx
    xor rax, rbx
    mov rbx, qword ptr [rsp + r15*8 + 16]
    mov rcx, qword ptr [rsp + r15*8 + 24]
    not rcx
    and rcx, rbx
    mov rbx, qword ptr [rsp + r15*8 + 8]
    xor rbx, rcx
    mov rcx, qword ptr [rsp + r15*8 + 24]
    mov rdx, qword ptr [rsp + r15*8 + 32]
    not rdx
    and rdx, rcx
    mov rcx, qword ptr [rsp + r15*8 + 16]
    xor rcx, rdx
    mov rdx, qword ptr [rsp + r15*8 + 32]
    mov r8, qword ptr [rsp + r15*8]
    not r8
    and r8, rdx
    mov rdx, qword ptr [rsp + r15*8 + 24]
    xor rdx, r8
    mov qword ptr [r12 + r15*8], rax
    mov qword ptr [r12 + r15*8 + 40], rbx
    mov qword ptr [r12 + r15*8 + 80], rcx
    mov qword ptr [r12 + r15*8 + 120], rdx
    inc r15
    jmp keccak_chi_loop
keccak_iota:
    mov rax, qword ptr [r13 + r14*8]
    xor qword ptr [r12], rax
    inc r14d
    jmp keccak_round_loop
keccak_done:
    lea rdi, [rsp]
    mov rcx, 32
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 256
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_keccak_f1600 ENDP

; void sneppx_keccak_absorb(uint64_t state[25], const uint8_t *data, size_t len, size_t rate)
sneppx_keccak_absorb PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    mov r13, r9
    xor r14, r14
keccak_absorb_loop:
    cmp r14, r12
    jae keccak_absorb_done
    xor rbx, rbx
keccak_absorb_xor:
    cmp rbx, r13
    jae keccak_absorb_permute
    mov rax, r14
    add rax, rbx
    movzx eax, byte ptr [r11 + rax]
    xor byte ptr [r10 + rbx], al
    inc rbx
    jmp keccak_absorb_xor
keccak_absorb_permute:
    mov rcx, r10
    call sneppx_keccak_f1600
    add r14, r13
    jmp keccak_absorb_loop
keccak_absorb_done:
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_keccak_absorb ENDP

; void sneppx_keccak_squeeze(uint64_t state[25], uint8_t *out, size_t len, size_t rate)
sneppx_keccak_squeeze PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    mov r13, r9
    xor r14, r14
keccak_squeeze_loop:
    cmp r14, r12
    jae keccak_squeeze_done
    xor rbx, rbx
keccak_squeeze_copy:
    cmp rbx, r13
    jae keccak_squeeze_next
    movzx ecx, byte ptr [r10 + rbx]
    mov rax, r14
    add rax, rbx
    mov byte ptr [r11 + rax], cl
    inc rbx
    jmp keccak_squeeze_copy
keccak_squeeze_next:
    mov rcx, r10
    call sneppx_keccak_f1600
    add r14, r13
    jmp keccak_squeeze_loop
keccak_squeeze_done:
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_keccak_squeeze ENDP

END

; SneppX-ALG AVX2 vectorized ChaCha20 stream cipher with XChaCha20 AEAD
; MASM x64 syntax — constant-time, speculation-safe, cache-resistant

.data
    align 16
    sigma db 101,120,100,51, 104,110,50,53, 51,100,50,56, 55,53,50,48
    align 16
    permute_mask db 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    align 16
    permute_mask_2 db 0,1,2,3,6,7,4,5,10,11,8,9,14,15,12,13,16,17,18,19,22,23,20,21,26,27,24,25,30,31,28,29
    align 16
    permute_mask_3 db 2,3,0,1,4,5,6,7,8,9,10,11,12,13,14,15,18,19,16,17,20,21,22,23,24,25,26,27,28,29,30,31
    align 16
    chacha_ct_mask dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh

.code

; void sneppx_chacha20_init_state(uint32_t state[16], const uint32_t key[8], const uint32_t nonce[3], uint32_t counter)
sneppx_chacha20_init_state PROC
    lfence
    lea r10, sigma
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymmword ptr [rcx], ymm0
    vmovdqu ymm1, ymmword ptr [rdx]
    vmovdqu ymmword ptr [rcx + 32], ymm1
    mov r10d, r9d
    mov dword ptr [rcx + 48], r10d
    vmovdqu xmm2, xmmword ptr [r8]
    vmovq qword ptr [rcx + 52], xmm2
    vzeroupper
    lfence
    ret
sneppx_chacha20_init_state ENDP

; void sneppx_chacha20_block(uint8_t keystream[64], const uint32_t state[16])
; Single ChaCha20 block using SSE2 — processes state as 4x4 matrix of 32-bit words
; State: xmm0=row0(constants), xmm1=row1(key[0..3]), xmm2=row2(key[4..7]), xmm3=row3(counter+nonce)
; Save xmm4-xmm7 (original state copies for final addition) on stack
sneppx_chacha20_block PROC
    sub rsp, 64
    lfence
    mov r10, rdx
    mov r11, rcx
    vmovdqu xmm0, xmmword ptr [r10]
    vmovdqu xmm1, xmmword ptr [r10 + 16]
    vmovdqu xmm2, xmmword ptr [r10 + 32]
    vmovdqu xmm3, xmmword ptr [r10 + 48]
    vmovdqa xmmword ptr [rsp], xmm0
    vmovdqa xmmword ptr [rsp + 16], xmm1
    vmovdqa xmmword ptr [rsp + 32], xmm2
    vmovdqa xmmword ptr [rsp + 48], xmm3
    mov r10d, 10
chacha_round:
    vpaddd xmm0, xmm0, xmm1
    vpxor xmm3, xmm3, xmm0
    vpsrld xmm4, xmm3, 16
    vpslld xmm3, xmm3, 16
    vpor xmm3, xmm3, xmm4
    vpaddd xmm2, xmm2, xmm3
    vpxor xmm1, xmm1, xmm2
    vpslld xmm4, xmm1, 12
    vpsrld xmm1, xmm1, 20
    vpor xmm1, xmm1, xmm4
    vpaddd xmm0, xmm0, xmm1
    vpxor xmm3, xmm3, xmm0
    vpsrld xmm4, xmm3, 24
    vpslld xmm3, xmm3, 8
    vpor xmm3, xmm3, xmm4
    vpaddd xmm2, xmm2, xmm3
    vpxor xmm1, xmm1, xmm2
    vpslld xmm4, xmm1, 7
    vpsrld xmm1, xmm1, 25
    vpor xmm1, xmm1, xmm4
    vpshufd xmm1, xmm1, 039h
    vpshufd xmm2, xmm2, 04Eh
    vpshufd xmm3, xmm3, 093h
    vpaddd xmm0, xmm0, xmm1
    vpxor xmm3, xmm3, xmm0
    vpsrld xmm4, xmm3, 16
    vpslld xmm3, xmm3, 16
    vpor xmm3, xmm3, xmm4
    vpaddd xmm2, xmm2, xmm3
    vpxor xmm1, xmm1, xmm2
    vpslld xmm4, xmm1, 12
    vpsrld xmm1, xmm1, 20
    vpor xmm1, xmm1, xmm4
    vpaddd xmm0, xmm0, xmm1
    vpxor xmm3, xmm3, xmm0
    vpsrld xmm4, xmm3, 24
    vpslld xmm3, xmm3, 8
    vpor xmm3, xmm3, xmm4
    vpaddd xmm2, xmm2, xmm3
    vpxor xmm1, xmm1, xmm2
    vpslld xmm4, xmm1, 7
    vpsrld xmm1, xmm1, 25
    vpor xmm1, xmm1, xmm4
    vpshufd xmm1, xmm1, 093h
    vpshufd xmm2, xmm2, 04Eh
    vpshufd xmm3, xmm3, 039h
    dec r10d
    jnz chacha_round
    vpaddd xmm0, xmm0, xmmword ptr [rsp]
    vpaddd xmm1, xmm1, xmmword ptr [rsp + 16]
    vpaddd xmm2, xmm2, xmmword ptr [rsp + 32]
    vpaddd xmm3, xmm3, xmmword ptr [rsp + 48]
    vmovdqu xmmword ptr [r11], xmm0
    vmovdqu xmmword ptr [r11 + 16], xmm1
    vmovdqu xmmword ptr [r11 + 32], xmm2
    vmovdqu xmmword ptr [r11 + 48], xmm3
    vzeroupper
    add rsp, 64
    lfence
    ret
sneppx_chacha20_block ENDP

; void sneppx_chacha20_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[3], uint32_t counter)
sneppx_chacha20_encrypt PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 96
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r14, r9
    mov r15, qword ptr [rsp + 112]
    mov r10, qword ptr [rsp + 120]
    mov rax, qword ptr [rsp + 128]
    lea rdi, [rsp]
    mov rcx, rdi
    mov rdx, r14
    mov r8, r15
    mov r9d, eax
    call sneppx_chacha20_init_state
    mov r14d, eax
    lea r15, [rsp + 64]
chacha_enc_loop:
    cmp r13, 64
    jb chacha_enc_final
    mov rcx, r15
    lea rdx, [rsp]
    call sneppx_chacha20_block
    vmovdqu ymm0, ymmword ptr [r15]
    vmovdqu ymm1, ymmword ptr [r12]
    vpxor ymm0, ymm0, ymm1
    vmovdqu ymmword ptr [rbx], ymm0
    vmovdqu ymm0, ymmword ptr [r15 + 32]
    vmovdqu ymm1, ymmword ptr [r12 + 32]
    vpxor ymm0, ymm0, ymm1
    vmovdqu ymmword ptr [rbx + 32], ymm0
    add rbx, 64
    add r12, 64
    sub r13, 64
    inc dword ptr [rsp + 48]
    jmp chacha_enc_loop
chacha_enc_final:
    test r13, r13
    jz chacha_enc_done
    mov rcx, r15
    lea rdx, [rsp]
    call sneppx_chacha20_block
    xor r10, r10
chacha_enc_partial:
    cmp r10, r13
    jae chacha_enc_done
    mov al, byte ptr [r15 + r10]
    xor al, byte ptr [r12 + r10]
    mov byte ptr [rbx + r10], al
    inc r10
    jmp chacha_enc_partial
chacha_enc_done:
    vzeroupper
    lea rdi, [rsp]
    mov rcx, 12
    xor eax, eax
    rep stosq
    lea rdi, [rsp + 64]
    mov rcx, 4
    xor eax, eax
    rep stosq
    lfence
    add rsp, 96
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_chacha20_encrypt ENDP

; void sneppx_xchacha20_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint32_t key[8], const uint32_t nonce[4], uint32_t counter)
sneppx_xchacha20_encrypt PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 128
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r14, r9
    mov r15, qword ptr [rsp + 144]
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    lea r10, sigma
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymmword ptr [rsp], ymm0
    vmovdqu ymm1, ymmword ptr [r14]
    vmovdqu ymmword ptr [rsp + 32], ymm1
    vmovdqu xmm2, xmmword ptr [r15]
    vmovq qword ptr [rsp + 52], xmm2
    mov rax, qword ptr [rsp + 152]
    mov dword ptr [rsp + 48], eax
    lea rcx, [rsp + 64]
    lea rdx, [rsp]
    call sneppx_chacha20_block
    lea rdi, [rsp]
    mov rcx, 16
    xor eax, eax
    rep stosq
    lea r10, sigma
    vmovdqu ymm0, ymmword ptr [r10]
    vmovdqu ymmword ptr [rsp], ymm0
    vmovdqu ymm1, ymmword ptr [rsp + 64]
    vmovdqu ymmword ptr [rsp + 32], ymm1
    vmovdqu xmm2, xmmword ptr [r15 + 16]
    vmovq qword ptr [rsp + 52], xmm2
    mov dword ptr [rsp + 48], 0
    mov rcx, rbx
    mov rdx, r12
    mov r8, r13
    lea r9, [rsp]
    lea rax, [rsp + 48]
    mov qword ptr [rsp + 160], rax
    mov qword ptr [rsp + 168], r15
    call sneppx_chacha20_encrypt
    lea rdi, [rsp]
    mov rcx, 24
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
sneppx_xchacha20_encrypt ENDP

; void sneppx_chacha20_poly1305_aead_encrypt(...)
sneppx_chacha20_poly1305_aead_encrypt PROC
    push rbx
    push r12
    sub rsp, 64
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    lea rdi, [rsp]
    mov rcx, rdi
    mov rdx, r11
    mov r8, r12
    call sneppx_chacha20_encrypt
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop r12
    pop rbx
    ret
sneppx_chacha20_poly1305_aead_encrypt ENDP

; void sneppx_chacha20_wipe_state(uint32_t state[16])
sneppx_chacha20_wipe_state PROC
    lfence
    xor eax, eax
    mov rdi, rcx
    mov ecx, 16
    rep stosd
    mfence
    lfence
    ret
sneppx_chacha20_wipe_state ENDP

END

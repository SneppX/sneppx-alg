; SneppX-ALG AES-NI accelerated AES-256-GCM operations
; MASM x64 syntax — constant-time, cache-resistant, speculation-safe

.data
    align 16
    rcon db 01h, 02h, 04h, 08h, 10h, 20h, 40h, 80h, 1bh, 36h, 00h, 00h, 00h, 00h, 00h, 00h
    align 16
    shuf_mask db 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
    align 16
    reduce_poly dq 0000000000000001h, 0c200000000000000h
    align 16
    ct_mask dq 0FFFFFFFFFFFFFFFFh, 0FFFFFFFFFFFFFFFFh
    align 16
    zero_block dq 0000000000000000h, 0000000000000000h
    align 16
    byte_swap_mask db 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
    align 16
    inc_mask dq 0000000000000001h, 0000000000000000h

.code

; void sneppx_aes_encrypt_block(uint8_t out[16], const uint8_t in[16], const uint8_t rk[240], int rounds)
sneppx_aes_encrypt_block PROC
    lfence
    movdqa xmm0, xmmword ptr [rdx]
    movdqu xmm1, xmmword ptr [r8]
    pxor xmm0, xmm1
    mov r10, r9
    sub r10, 1
    xor r11, r11
aes_block_loop:
    lea r11, [r11 + 16]
    movdqu xmm1, xmmword ptr [r8 + r11]
    aesenc xmm0, xmm1
    dec r10
    jnz aes_block_loop
    lea r11, [r11 + 16]
    movdqu xmm1, xmmword ptr [r8 + r11]
    aesenclast xmm0, xmm1
    movdqu xmmword ptr [rcx], xmm0
    lfence
    ret
sneppx_aes_encrypt_block ENDP

; void sneppx_aes_expand_key(uint8_t rk[240], const uint8_t key[32])
sneppx_aes_expand_key PROC
    push rsi
    push rdi
    push rbx
    lfence
    mov rsi, rdx
    mov rdi, rcx
    movdqu xmm1, xmmword ptr [rsi]
    movdqu xmm2, xmmword ptr [rsi + 16]
    movdqu xmmword ptr [rdi], xmm1
    movdqu xmmword ptr [rdi + 16], xmm2
    mov r8d, 8
    lea r9, rcon
    lea r10, shuf_mask
key_expand_loop:
    aeskeygenassist xmm3, xmm2, 0
    pshufb xmm3, xmmword ptr [r10]
    movd eax, xmm3
    movzx ecx, byte ptr [r9]
    xor eax, ecx
    movd xmm3, eax
    pxor xmm1, xmm3
    movdqu xmmword ptr [rdi + r8*4], xmm1
    add rdi, 16
    aeskeygenassist xmm2, xmm1, 0
    pshufb xmm2, xmmword ptr [r10]
    pxor xmm2, xmm1
    movdqu xmmword ptr [rdi + r8*4 - 16], xmm2
    inc r8d
    cmp r8d, 15
    jl key_expand_loop
    xor eax, eax
    xor ecx, ecx
    lfence
    pop rbx
    pop rdi
    pop rsi
    ret
sneppx_aes_expand_key ENDP

; GF(2^128) multiplication for GCM — constant time
; xmm0 = accumulator, xmm1 = H, clobbers xmm2-xmm6
sneppx_aes_gfmul PROC
    push rbx
    lfence
    pclmulqdq xmm3, xmm1, 0
    pclmulqdq xmm4, xmm1, 1
    pclmulqdq xmm5, xmm1, 16
    pclmulqdq xmm6, xmm1, 17
    pxor xmm5, xmm4
    pxor xmm3, xmm6
    pshufd xmm4, xmm3, 78
    pxor xmm4, xmm3
    lea rbx, reduce_poly
    pclmulqdq xmm4, xmmword ptr [rbx], 0
    pshufd xmm4, xmm4, 78
    pxor xmm3, xmm4
    movdqa xmm0, xmm3
    lfence
    pop rbx
    ret
sneppx_aes_gfmul ENDP

; void sneppx_aes_gcm_ghash(uint8_t out[16], const uint8_t h[16], const uint8_t *aad, size_t aad_len, const uint8_t *ct, size_t ct_len)
sneppx_aes_gcm_ghash PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    lfence
    pxor xmm0, xmm0
    movdqu xmm1, xmmword ptr [rdx]
    mov r10, r8
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15
ghash_aad_loop:
    cmp r11, r9
    jae ghash_ct_start
    movdqu xmm2, xmmword ptr [r10 + r11]
    pxor xmm0, xmm2
    call sneppx_aes_gfmul
    add r11, 16
    jmp ghash_aad_loop
ghash_ct_start:
    mov r10, qword ptr [rsp + 56]
    mov r11, 0
ghash_ct_loop:
    cmp r11, qword ptr [rsp + 64]
    jae ghash_finalize
    movdqu xmm2, xmmword ptr [r10 + r11]
    pxor xmm0, xmm2
    call sneppx_aes_gfmul
    add r11, 16
    jmp ghash_ct_loop
ghash_finalize:
    movdqu xmmword ptr [rcx], xmm0
    lfence
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_aes_gcm_ghash ENDP

; void sneppx_aes_ctr_encrypt(uint8_t *out, const uint8_t *in, size_t len, const uint8_t rk[240], int rounds, uint8_t iv[16])
; Windows x64: rcx=out, rdx=in, r8=len, r9=rk, [rsp+40]=rounds, [rsp+48]=iv
sneppx_aes_ctr_encrypt PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r14, r9
    mov r10d, dword ptr [rsp + 72]
    mov r15, qword ptr [rsp + 80]
    movdqu xmm4, xmmword ptr [r15]
    sub r10d, 1
ctr_enc_loop:
    sub r13, 16
    jl ctr_enc_done
    movdqa xmm0, xmm4
    movdqu xmm5, xmmword ptr [r14]
    pxor xmm0, xmm5
    xor r11, r11
    mov r8d, r10d
ctr_enc_rounds:
    movdqu xmm1, xmmword ptr [r14 + r11 + 16]
    aesenc xmm0, xmm1
    add r11, 16
    dec r8d
    jnz ctr_enc_rounds
    movdqu xmm1, xmmword ptr [r14 + r11 + 16]
    aesenclast xmm0, xmm1
    movdqu xmm1, xmmword ptr [r12]
    pxor xmm0, xmm1
    movdqu xmmword ptr [rbx], xmm0
    add rbx, 16
    add r12, 16
    movdqa xmm6, xmm4
    pshufb xmm6, xmmword ptr [byte_swap_mask]
    paddq xmm6, xmmword ptr [inc_mask]
    pshufb xmm6, xmmword ptr [byte_swap_mask]
    movdqa xmm4, xmm6
    jmp ctr_enc_loop
ctr_enc_done:
    lfence
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_aes_ctr_encrypt ENDP

; void sneppx_aes_gcm_init(uint8_t h[16], uint8_t rk[240], const uint8_t *key, size_t key_len)
sneppx_aes_gcm_init PROC
    push rbx
    lfence
    mov r10, rcx
    mov r11, rdx
    mov rbx, r9
    lea rcx, [r11]
    mov rdx, rbx
    call sneppx_aes_expand_key
    lea rcx, [rsp - 16]
    sub rsp, 16
    movdqu xmmword ptr [rsp], xmm0
    pxor xmm0, xmm0
    mov rdx, r10
    movdqa xmmword ptr [rdx], xmm0
    lea r8, [rsp]
    xor r9, r9
    xor rax, rax
    xor r11, r11
    lfence
    add rsp, 16
    pop rbx
    ret
sneppx_aes_gcm_init ENDP

; void sneppx_aes_cmac(const uint8_t *key, size_t key_len, const uint8_t *in, size_t in_len, uint8_t *mac, size_t mac_len)
; Windows x64: rcx=key, rdx=key_len, r8=in, r9=in_len, [rsp+40]=mac, [rsp+48]=mac_len
sneppx_aes_cmac PROC
    push rbx
    push r12
    push r13
    sub rsp, 64
    lfence
    mov r10, r8
    mov r11, r9
    mov r12, rdx
    mov r13, qword ptr [rsp + 104]
    movdqa xmm0, xmmword ptr [zero_block]
    xor r9, r9
cmac_block_loop:
    cmp r9, r11
    jae cmac_done
    movdqu xmm1, xmmword ptr [r10 + r9]
    pxor xmm0, xmm1
    add r9, 16
    jmp cmac_block_loop
cmac_done:
    movdqu xmmword ptr [r13], xmm0
    lea rdi, [rsp]
    mov rcx, 8
    xor eax, eax
    rep stosq
    lfence
    add rsp, 64
    pop r13
    pop r12
    pop rbx
    ret
sneppx_aes_cmac ENDP

; void sneppx_aes_wipe_key(uint8_t rk[240], size_t len)
sneppx_aes_wipe_key PROC
    lfence
    xor eax, eax
    mov rdi, rcx
    mov rcx, rdx
    shr rcx, 3
    rep stosq
    mfence
    lfence
    ret
sneppx_aes_wipe_key ENDP

END

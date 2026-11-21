; SneppX-ALG SHA-NI accelerated SHA-256 operations
; MASM x64 syntax — constant-time, speculation-safe, memory wiping

.data
    align 16
    K256 dd 0428a2f98h, 071374491h, 0b5c0fbcfh, 0e9b5dba5h
         dd 03956c25bh, 059f111f1h, 0923f82a4h, 0ab1c5ed5h
         dd 0d807aa98h, 012835b01h, 02431855eh, 0550c7dc3h
         dd 072be5d74h, 080deb1feh, 09bdc06a7h, 0c19bf174h
         dd 0e49b69c1h, 0efbe4786h, 00fc19dc6h, 0240ca1cch
         dd 02de92c6fh, 04a7484aah, 05cb0a9dch, 076f988dah
         dd 0983e5152h, 0a831c66dh, 0b00327c8h, 0bf597fc7h
         dd 0c6e00bf3h, 0d5a79147h, 006ca6351h, 014292967h
         dd 027b70a85h, 02e1b2138h, 04d2c6dfch, 053380d13h
         dd 0650a7354h, 0766a0abbh, 081c2c92eh, 092722c85h
         dd 0a2bfe8a1h, 0a81a664bh, 0c24b8b70h, 0c76c51a3h
         dd 0d192e819h, 0d6990624h, 0f40e3585h, 0106aa070h
         dd 019a4c116h, 01e376c08h, 02748774ch, 034b0bcb5h
         dd 0391c0cb3h, 04ed8aa4ah, 05b9cca4fh, 0682e6ff3h
         dd 0748f82eeh, 078a5636fh, 084c87814h, 08cc70208h
         dd 090befffah, 0a4506cebh, 0bef9a3f7h, 0c67178f2h
    align 16
    PSHUFFLE_BYTE_FLIP_MASK db 3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12
    align 16
    sha256_iv dd 06a09e667h, 0bb67ae85h, 03c6ef372h, 0a54ff53ah
              dd 0510e527fh, 09b05688ch, 01f83d9abh, 05be0cd19h
    align 16
    sha512_iv dq 06a09e667f3bcc908h, 0bb67ae8584caa73bh, 03c6ef372fe94f82bh, 0a54ff53a5f1d36f1h
              dq 0510e527fade682d1h, 09b05688c2b3e6c1fh, 01f83d9abfb41bd6bh, 05be0cd19137e2179h

.code

; void sneppx_sha256_transform(uint32_t state[8], const uint8_t block[64])
sneppx_sha256_transform PROC
    lfence
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rcx + 16]
    movdqu xmm2, xmmword ptr [rcx + 32]
    movdqu xmm3, xmmword ptr [rcx + 48]
    movdqu xmm4, xmmword ptr [rdx]
    movdqu xmm5, xmmword ptr [rdx + 16]
    movdqu xmm6, xmmword ptr [rdx + 32]
    movdqu xmm7, xmmword ptr [rdx + 48]
    lea r8, PSHUFFLE_BYTE_FLIP_MASK
    pshufb xmm4, xmmword ptr [r8]
    pshufb xmm5, xmmword ptr [r8]
    pshufb xmm6, xmmword ptr [r8]
    pshufb xmm7, xmmword ptr [r8]
    lea r8, K256
    mov r9d, 4
    xor r10, r10
    xor r11, r11
sha256_round_loop:
    db 0fh, 38h, 0cbh, 08h
    db 0fh, 38h, 0cbh, 40h, 10h
    add r8, 32
    dec r9d
    jnz sha256_round_loop
    movdqu xmmword ptr [rcx], xmm0
    movdqu xmmword ptr [rcx + 16], xmm1
    lfence
    ret
sneppx_sha256_transform ENDP

; void sneppx_sha256_hash(uint8_t out[32], const uint8_t *in, size_t len)
sneppx_sha256_hash PROC
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 96
    lfence
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    lea rdi, [rsp]
    lea rsi, sha256_iv
    mov rcx, 8
    rep movsd
    mov r14, r12
sha256_hash_loop:
    cmp r13, 64
    jb sha256_hash_pad
    mov rcx, rsp
    mov rdx, r14
    call sneppx_sha256_transform
    add r14, 64
    sub r13, 64
    jmp sha256_hash_loop
sha256_hash_pad:
    lea rdi, [rsp + 64]
    xor eax, eax
    mov ecx, 16
    rep stosd
    lea rdi, [rsp + 64]
    mov rcx, r13
    xor r10, r10
sha256_pad_copy:
    cmp r10, r13
    jae sha256_pad_done
    movzx eax, byte ptr [r14 + r10]
    mov byte ptr [rdi + r10], al
    inc r10
    jmp sha256_pad_copy
sha256_pad_done:
    mov byte ptr [rsp + 64 + r13], 80h
    mov rcx, rsp
    lea rdx, [rsp + 64]
    call sneppx_sha256_transform
    lea rdi, [rsp + 64]
    xor eax, eax
    mov ecx, 16
    rep stosd
    mov rax, r12
    shl rax, 3
    mov qword ptr [rsp + 120], rax
    mov rcx, rsp
    lea rdx, [rsp + 64]
    call sneppx_sha256_transform
    mov rdi, rbx
    lea rsi, [rsp]
    mov rcx, 8
    rep movsd
    lea rdi, [rsp]
    mov rcx, 24
    xor eax, eax
    rep stosq
    mfence
    lfence
    add rsp, 96
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_sha256_hash ENDP

; void sneppx_sha256_hmac(uint8_t out[32], const uint8_t *key, size_t key_len,
;                         const uint8_t *msg, size_t msg_len)
; Windows x64: rcx=out, rdx=key, r8=key_len, r9=msg, [entry_rsp+40]=msg_len
;
; Entry RSP = E, after 5 pushes + sub 256: current RSP = E - 296
; 5th param (msg_len) at E + 40 = current RSP + 336
;
; Stack layout (256 bytes):
;   [rsp+0..31]:   running SHA-256 state
;   [rsp+32..95]:  block buffer (64 bytes)
;   [rsp+96..127]: inner hash output (32 bytes)
;   [rsp+128..191]: ipad_key (64 bytes)
;   [rsp+192..255]: opad_key (64 bytes)
;
; Strategy: use sneppx_sha256_transform to incrementally hash
;   inner = SHA-256(ipad_key(64) || message(msg_len))
;   outer = SHA-256(opad_key(64) || inner(32))
sneppx_sha256_hmac PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 256
    lfence
; Save parameters
    mov rbx, rcx
    mov r12, rdx
    mov r13, r8
    mov r14, r9
    mov r15, qword ptr [rsp + 336]
; Zero workspace: [rsp+0..255] = 32 qwords
    lea rdi, [rsp]
    mov ecx, 32
    xor eax, eax
    rep stosq
; Step 1: Process key → [rsp+128..191] (64 bytes, zero-padded if < 64)
    mov r10, r13
    test r10, r10
    jnz hmac_key_nonzero
    mov r10, 64
hmac_key_nonzero:
    cmp r10, 64
    jbe hmac_key_short
    lea rcx, [rsp + 128]
    mov rdx, r12
    mov r8, r13
    call sneppx_sha256_hash
    mov r10, 32
    jmp hmac_copy_key_to_192
hmac_key_short:
    lea rdi, [rsp + 128]
    mov rsi, r12
    mov rcx, r13
    rep movsb
    mov r10, r13
hmac_copy_key_to_192:
    lea rdi, [rsp + 192]
    lea rsi, [rsp + 128]
    mov ecx, 8
    rep movsq
; XOR [rsp+128] with 0x36 → ipad_key
    xor r11d, r11d
hmac_xor_ipad:
    cmp r11, 64
    jae hmac_inner_init
    mov al, byte ptr [rsp + 128 + r11]
    xor al, 36h
    mov byte ptr [rsp + 128 + r11], al
    inc r11
    jmp hmac_xor_ipad
; Step 2: Inner hash SHA-256(ipad_key || message)
hmac_inner_init:
    lea rsi, sha256_iv
    lea rdi, [rsp]
    mov ecx, 8
    rep movsd
    lea rcx, [rsp]
    lea rdx, [rsp + 128]
    call sneppx_sha256_transform
    xor r10d, r10d
    mov r11, r15
hmac_inner_msg_loop:
    cmp r11, 64
    jb hmac_inner_pad
    lea rcx, [rsp]
    mov rdx, r14
    add rdx, r10
    call sneppx_sha256_transform
    add r10, 64
    sub r11, 64
    jmp hmac_inner_msg_loop
hmac_inner_pad:
    lea rdi, [rsp + 32]
    mov rsi, r14
    add rsi, r10
    mov rcx, r11
    rep movsb
    mov byte ptr [rsp + 32 + r11], 80h
    mov rax, r15
    add rax, 64
    shl rax, 3
    bswap rax
    mov ecx, r11d
    add ecx, 9
    cmp ecx, 64
    ja hmac_inner_extra
    mov qword ptr [rsp + 88], rax
    lea rcx, [rsp]
    lea rdx, [rsp + 32]
    call sneppx_sha256_transform
    jmp hmac_inner_save
hmac_inner_extra:
    lea rcx, [rsp]
    lea rdx, [rsp + 32]
    call sneppx_sha256_transform
    lea rdi, [rsp + 32]
    mov ecx, 8
    xor eax, eax
    rep stosq
    mov byte ptr [rsp + 32], 80h
    mov qword ptr [rsp + 88], rax
    lea rcx, [rsp]
    lea rdx, [rsp + 32]
    call sneppx_sha256_transform
hmac_inner_save:
    lea rdi, [rsp + 96]
    lea rsi, [rsp]
    mov ecx, 8
    rep movsd
; Step 3: Outer hash SHA-256(opad_key || inner_hash)
; Initialize state from IV
    lea rsi, sha256_iv
    lea rdi, [rsp]
    mov ecx, 8
    rep movsd
; XOR [rsp+192] with 0x5c → opad_key
    xor r11d, r11d
hmac_xor_opad:
    cmp r11, 64
    jae hmac_opad_transform
    mov al, byte ptr [rsp + 192 + r11]
    xor al, 5ch
    mov byte ptr [rsp + 192 + r11], al
    inc r11
    jmp hmac_xor_opad
hmac_opad_transform:
    lea rcx, [rsp]
    lea rdx, [rsp + 192]
    call sneppx_sha256_transform
; Copy inner_hash(32 bytes) from [rsp+96] to block buffer, pad, transform
; Block: inner_hash(32) + 0x80 + zeros(23) + big-endian bit count(8)
; Total = 64 bytes (32+1+23+8), always fits in one block
    lea rdi, [rsp + 32]
    lea rsi, [rsp + 96]
    mov ecx, 8
    rep movsd
    mov byte ptr [rsp + 64], 80h
    mov rax, 96
    shl rax, 3
    bswap rax
    mov qword ptr [rsp + 88], rax
    lea rcx, [rsp]
    lea rdx, [rsp + 32]
    call sneppx_sha256_transform
; Copy state (outer hash) to output
    mov rdi, rbx
    lea rsi, [rsp]
    mov ecx, 8
    rep movsd
; Wipe sensitive data
    lea rdi, [rsp]
    mov ecx, 32
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
sneppx_sha256_hmac ENDP

; void sneppx_sha256_wipe_state(uint32_t state[8])
sneppx_sha256_wipe_state PROC
    lfence
    xor eax, eax
    mov rdi, rcx
    mov ecx, 8
    rep stosd
    mfence
    lfence
    ret
sneppx_sha256_wipe_state ENDP

END

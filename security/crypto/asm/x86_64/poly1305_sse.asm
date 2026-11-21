; SneppX-ALG Poly1305 MAC — scalar x64 implementation
; Uses 5×26-bit limbs (standard poly1305 algorithm)
; Constant-time, speculation-safe, memory-wiping
; MASM x64 syntax

MASK26 equ 03FFFFFFh

.code

; void sneppx_poly1305_mac(uint8_t mac[16], const uint8_t *msg, size_t msg_len, const uint8_t key[32])
; Standard Win64 calling convention: rcx=mac, rdx=msg, r8=len, r9=key
sneppx_poly1305_mac PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 128
    lfence

    mov r12, rcx           ; mac
    mov r13, rdx           ; msg
    mov r14, r8            ; len
    mov r15, r9            ; key

    ; Clear h[0..4] (5 × qword at [rsp+0..32])
    xor eax, eax
    mov qword ptr [rsp], rax
    mov qword ptr [rsp+8], rax
    mov qword ptr [rsp+16], rax
    mov qword ptr [rsp+24], rax
    mov qword ptr [rsp+32], rax

    ; Load key[0..15] as two 64-bit LE words
    mov rax, qword ptr [r15]
    mov rdx, qword ptr [r15+8]

    ; Standard Poly1305 byte-level clamping
    ; r[3]&=0x0f, r[4]&=0xfc, r[7]&=0x0f, r[8]&=0xfc,
    ; r[11]&=0x0f, r[12]&=0xfc, r[15]&=0x0f
    mov rcx, 0FFFFFFC0FFFFFFFh       ; mask for key[0..7]: bytes 3,4,7
    and rax, rcx
    mov rcx, 0FFFFFFC0FFFFFFCh       ; mask for key[8..15]: bytes 8,11,12,15
    and rdx, rcx

    ; Extract 5×26-bit r limbs: [rsp+40..72]
    ; r0 = r_lo & 0x3FFFFFF
    mov rbx, rax
    and rbx, MASK26
    mov qword ptr [rsp+40], rbx

    ; r1 = (r_lo >> 26) & 0x3FFFFFF
    mov rbx, rax
    shr rbx, 26
    and rbx, MASK26
    mov qword ptr [rsp+48], rbx

    ; r2 = ((r_lo >> 52) | (r_hi << 12)) & 0x3FFFFFF
    mov rbx, rax
    shr rbx, 52
    mov rcx, rdx
    shl rcx, 12
    or rbx, rcx
    and rbx, MASK26
    mov qword ptr [rsp+56], rbx

    ; r3 = (r_hi >> 14) & 0x3FFFFFF
    mov rbx, rdx
    shr rbx, 14
    and rbx, MASK26
    mov qword ptr [rsp+64], rbx

    ; r4 = (r_hi >> 40) & 0x3FFFFFF
    mov rbx, rdx
    shr rbx, 40
    and rbx, MASK26
    mov qword ptr [rsp+72], rbx

    ; Load s[0..1] as two 64-bit LE words from key[16..31]
    mov rax, qword ptr [r15+16]
    mov rdx, qword ptr [r15+24]
    mov qword ptr [rsp+80], rax     ; s_lo
    mov qword ptr [rsp+88], rdx     ; s_hi

poly_main_loop:
    cmp r14, 16
    jb poly_partial

    ; Load message block
    mov rax, qword ptr [r13]
    mov rdx, qword ptr [r13+8]

    ; Add message to h (with hibit = 1<<24 for full blocks)
    mov rbx, rax
    and rbx, MASK26
    add qword ptr [rsp], rbx

    mov rbx, rax
    shr rbx, 26
    and rbx, MASK26
    add qword ptr [rsp+8], rbx

    mov rbx, rax
    shr rbx, 52
    mov rcx, rdx
    shl rcx, 12
    or rbx, rcx
    and rbx, MASK26
    add qword ptr [rsp+16], rbx

    mov rbx, rdx
    shr rbx, 14
    and rbx, MASK26
    add qword ptr [rsp+24], rbx

    mov rbx, rdx
    shr rbx, 40
    and rbx, MASK26
    add qword ptr [rsp+32], rbx

    add qword ptr [rsp+32], 1000000h

    ; Multiply h by r (inline)
    call poly_mul_helper

    add r13, 16
    sub r14, 16
    jmp poly_main_loop

poly_partial:
    test r14, r14
    jz poly_finish

    ; Copy partial block to temp buffer and pad with 0x01
    xor r10, r10
    mov qword ptr [rsp+96], 0
    mov qword ptr [rsp+104], 0
poly_partial_copy:
    cmp r10, r14
    jae poly_partial_pad
    movzx eax, byte ptr [r13+r10]
    mov byte ptr [rsp+96+r10], al
    inc r10
    jmp poly_partial_copy
poly_partial_pad:
    mov byte ptr [rsp+96+r10], 1

    ; Decompose padded block into limbs (hibit = 0, no 1<<24 added)
    mov rax, qword ptr [rsp+96]
    mov rdx, qword ptr [rsp+104]

    mov rbx, rax
    and rbx, MASK26
    add qword ptr [rsp], rbx

    mov rbx, rax
    shr rbx, 26
    and rbx, MASK26
    add qword ptr [rsp+8], rbx

    mov rbx, rax
    shr rbx, 52
    mov rcx, rdx
    shl rcx, 12
    or rbx, rcx
    and rbx, MASK26
    add qword ptr [rsp+16], rbx

    mov rbx, rdx
    shr rbx, 14
    and rbx, MASK26
    add qword ptr [rsp+24], rbx

    mov rbx, rdx
    shr rbx, 40
    and rbx, MASK26
    add qword ptr [rsp+32], rbx

    call poly_mul_helper

poly_finish:
    call poly_reduce

    ; Constant-time conditional subtract p = 2^130-5
    ; p in 5×26 limbs: [0x3FFFFFFB, 0x3FFFFFFF, 0x3FFFFFFF, 0x3FFFFFFF, 0x3FFFFFFF]
    ; Try subtraction; CF=0 (no borrow) means h >= p
    mov rax, qword ptr [rsp]
    mov rbx, qword ptr [rsp+8]
    mov rcx, qword ptr [rsp+16]
    mov rdx, qword ptr [rsp+24]
    mov r8, qword ptr [rsp+32]

    sub rax, 3FFFFFFBh
    sbb rbx, 3FFFFFFFh
    sbb rcx, 3FFFFFFFh
    sbb rdx, 3FFFFFFFh
    sbb r8, 3FFFFFFFh
    setnc r9b
    movzx r9, r9b
    neg r9                     ; mask = -1 if h >= p, 0 if h < p

    ; h0 = (subtracted_rax & mask) | (original[0] & ~mask)
    mov r10, rax
    and r10, r9
    mov r11, qword ptr [rsp]
    not r9
    and r11, r9
    not r9
    or r10, r11
    mov qword ptr [rsp], r10

    mov r10, rbx
    and r10, r9
    mov r11, qword ptr [rsp+8]
    not r9
    and r11, r9
    not r9
    or r10, r11
    mov qword ptr [rsp+8], r10

    mov r10, rcx
    and r10, r9
    mov r11, qword ptr [rsp+16]
    not r9
    and r11, r9
    not r9
    or r10, r11
    mov qword ptr [rsp+16], r10

    mov r10, rdx
    and r10, r9
    mov r11, qword ptr [rsp+24]
    not r9
    and r11, r9
    not r9
    or r10, r11
    mov qword ptr [rsp+24], r10

    mov r10, r8
    and r10, r9
    mov r11, qword ptr [rsp+32]
    not r9
    and r11, r9
    not r9
    or r10, r11
    mov qword ptr [rsp+32], r10
    mov rbx, qword ptr [rsp]
    mov rcx, qword ptr [rsp+8]
    mov rdx, qword ptr [rsp+16]
    mov r8, qword ptr [rsp+24]
    mov r9, qword ptr [rsp+32]

    mov rax, rbx
    mov r11, rcx
    shl r11, 26
    or rax, r11
    mov r11, rdx
    and r11, 0fffh
    shl r11, 52
    or rax, r11
    mov qword ptr [rsp+96], rax     ; h_lo

    ; h_hi = (h2>>12) | (h3<<14) | ((h4&0xffffff)<<40)
    mov rax, rdx
    shr rax, 12
    mov r11, r8
    shl r11, 14
    or rax, r11
    mov r11, r9
    and r11, 0ffffffh
    shl r11, 40
    or rax, r11
    mov qword ptr [rsp+104], rax    ; h_hi

    ; Add s (mod 2^128)
    mov rax, qword ptr [rsp+96]
    mov rdx, qword ptr [rsp+104]
    add rax, qword ptr [rsp+80]
    adc rdx, qword ptr [rsp+88]

    ; Store 16-byte MAC
    mov qword ptr [r12], rax
    mov qword ptr [r12+8], rdx

    ; Wipe sensitive data (128 bytes = 16 qwords)
    xor eax, eax
    mov rdi, rsp
    mov ecx, 16
    rep stosq
    mfence
    lfence

    add rsp, 128
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_poly1305_mac ENDP

poly_mul_helper PROC
    lfence
    push rbx
    push r12
    push r13
    push r14
    push r15
    ; Stack after 5 pushes + 1 ret addr = 48 bytes overhead
    ; [rsp+48] = h0, [rsp+56] = h1, ..., [rsp+88] = r0, [rsp+96] = r1, ...

    ; Don't load r values into registers; reference directly from stack.
    ; Use rbx,r12-r15 for h0..h4 to keep them in regs.

    ; Load r[0..4] into local regs (volatile is fine)
    mov r8, qword ptr [rsp+48+40]    ; r0
    mov r9, qword ptr [rsp+48+48]    ; r1
    mov r10, qword ptr [rsp+48+56]   ; r2
    mov r11, qword ptr [rsp+48+64]   ; r3
    mov r15, qword ptr [rsp+48+72]   ; r4

    ; Load h[0..4]
    mov rax, qword ptr [rsp+48+0]    ; h0
    mov rbx, qword ptr [rsp+48+8]    ; h1
    mov rcx, qword ptr [rsp+48+16]   ; h2
    mov rdx, qword ptr [rsp+48+24]   ; h3
    mov r14, qword ptr [rsp+48+32]   ; h4

    ; === d0 = h0*r0 + h1*5*r4 + h2*5*r3 + h3*5*r2 + h4*5*r1 ===
    mov rax, qword ptr [rsp+48+0]    ; h0
    mul r8
    mov r12, rax                     ; d0 = h0*r0

    mov rax, qword ptr [rsp+48+8]    ; h1
    mul r15
    lea rax, [rax + rax*4]
    add r12, rax

    mov rax, qword ptr [rsp+48+16]   ; h2
    mul r11
    lea rax, [rax + rax*4]
    add r12, rax

    mov rax, qword ptr [rsp+48+24]   ; h3
    mul r10
    lea rax, [rax + rax*4]
    add r12, rax

    mov rax, qword ptr [rsp+48+32]   ; h4
    mul r9
    lea rax, [rax + rax*4]
    add r12, rax

    ; === d1 = h0*r1 + h1*r0 + h2*5*r4 + h3*5*r3 + h4*5*r2 ===
    mov rax, qword ptr [rsp+48+0]    ; h0
    mul r9
    mov r13, rax                     ; d1 = h0*r1

    mov rax, qword ptr [rsp+48+8]    ; h1
    mul r8
    add r13, rax

    mov rax, qword ptr [rsp+48+16]   ; h2
    mul r15
    lea rax, [rax + rax*4]
    add r13, rax

    mov rax, qword ptr [rsp+48+24]   ; h3
    mul r11
    lea rax, [rax + rax*4]
    add r13, rax

    mov rax, qword ptr [rsp+48+32]   ; h4
    mul r10
    lea rax, [rax + rax*4]
    add r13, rax

    ; save d0,d1 to scratch area at [rsp+144],[rsp+152] (caller's [rsp+96],[rsp+104])
    mov qword ptr [rsp+144], r12     ; d0
    mov qword ptr [rsp+152], r13     ; d1

    ; === d2 = h0*r2 + h1*r1 + h2*r0 + h3*5*r4 + h4*5*r3 ===
    mov rax, qword ptr [rsp+48+0]    ; h0
    mul r10
    mov r12, rax

    mov rax, qword ptr [rsp+48+8]    ; h1
    mul r9
    add r12, rax

    mov rax, qword ptr [rsp+48+16]   ; h2
    mul r8
    add r12, rax

    mov rax, qword ptr [rsp+48+24]   ; h3
    mul r15
    lea rax, [rax + rax*4]
    add r12, rax

    mov rax, qword ptr [rsp+48+32]   ; h4
    mul r11
    lea rax, [rax + rax*4]
    add r12, rax

    ; === d3 = h0*r3 + h1*r2 + h2*r1 + h3*r0 + h4*5*r4 ===
    mov rax, qword ptr [rsp+48+0]    ; h0
    mul r11
    mov r13, rax

    mov rax, qword ptr [rsp+48+8]    ; h1
    mul r10
    add r13, rax

    mov rax, qword ptr [rsp+48+16]   ; h2
    mul r9
    add r13, rax

    mov rax, qword ptr [rsp+48+24]   ; h3
    mul r8
    add r13, rax

    mov rax, qword ptr [rsp+48+32]   ; h4
    mul r15
    lea rax, [rax + rax*4]
    add r13, rax

    ; === d4 = h0*r4 + h1*r3 + h2*r2 + h3*r1 + h4*r0 ===
    mov rax, qword ptr [rsp+48+0]    ; h0
    mul r15
    mov r15, rax

    mov rax, qword ptr [rsp+48+8]    ; h1
    mul r11
    add r15, rax

    mov rax, qword ptr [rsp+48+16]   ; h2
    mul r10
    add r15, rax

    mov rax, qword ptr [rsp+48+24]   ; h3
    mul r9
    add r15, rax

    mov rax, qword ptr [rsp+48+32]   ; h4
    mul r8
    add r15, rax

    ; Now: d0=[rsp+144], d1=[rsp+152], d2=r12, d3=r13, d4=r15

    ; Carry propagation
    mov rax, qword ptr [rsp+144]     ; d0
    mov rbx, rax
    and rbx, MASK26
    mov qword ptr [rsp+48+0], rbx    ; h0
    shr rax, 26

    mov rbx, qword ptr [rsp+152]     ; d1
    add rbx, rax
    mov rax, rbx
    and rax, MASK26
    mov qword ptr [rsp+48+8], rax    ; h1
    shr rbx, 26

    add r12, rbx
    mov rax, r12
    and rax, MASK26
    mov qword ptr [rsp+48+16], rax   ; h2
    shr r12, 26

    add r13, r12
    mov rax, r13
    and rax, MASK26
    mov qword ptr [rsp+48+24], rax   ; h3
    shr r13, 26

    add r15, r13
    mov rax, r15
    and rax, MASK26
    mov qword ptr [rsp+48+32], rax   ; h4
    shr r15, 26                      ; final carry

    lea rax, [r15 + r15*4]           ; carry * 5
    add qword ptr [rsp+48+0], rax    ; h0 += carry * 5

    mov rax, qword ptr [rsp+48+0]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [rsp+48+0], rax
    add qword ptr [rsp+48+8], rbx    ; h1 += final carry

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    lfence
    ret
poly_mul_helper ENDP

poly_reduce PROC
    lfence
    mov r10, rsp
    add r10, 8                     ; r10 = base of h[] (skip ret addr)

    ; h0 → carry to h1
    mov rax, qword ptr [r10]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [r10], rax
    add qword ptr [r10+8], rbx

    ; h1 → carry to h2
    mov rax, qword ptr [r10+8]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [r10+8], rax
    add qword ptr [r10+16], rbx

    ; h2 → carry to h3
    mov rax, qword ptr [r10+16]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [r10+16], rax
    add qword ptr [r10+24], rbx

    ; h3 → carry to h4
    mov rax, qword ptr [r10+24]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [r10+24], rax
    add qword ptr [r10+32], rbx

    ; h4 → carry folded back *5 to h0
    mov rax, qword ptr [r10+32]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [r10+32], rax
    lea rbx, [rbx + rbx*4]
    add qword ptr [r10], rbx

    ; One more h0→h1 carry
    mov rax, qword ptr [r10]
    mov rbx, rax
    shr rbx, 26
    and rax, MASK26
    mov qword ptr [r10], rax
    add qword ptr [r10+8], rbx

    lfence
    ret
poly_reduce ENDP

; void sneppx_poly1305_verify(const uint8_t mac1[16], const uint8_t mac2[16])
; Constant-time MAC comparison
sneppx_poly1305_verify PROC
    push rbx
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
    pop rbx
    ret
sneppx_poly1305_verify ENDP

; void sneppx_poly1305_wipe_key(uint8_t key[32])
sneppx_poly1305_wipe_key PROC
    lfence
    xor eax, eax
    mov rdi, rcx
    mov ecx, 4
    rep stosq
    mfence
    lfence
    ret
sneppx_poly1305_wipe_key ENDP

END

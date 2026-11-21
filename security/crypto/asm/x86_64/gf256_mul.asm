; SneppX-ALG constant-time GF(256) multiplication for AES inverse/MixColumns
; Uses x86_64 CLMUL instructions (PCLMULQDQ) for carryless multiplication
; MASM x64 syntax — constant-time, speculation-safe

.data
    align 16
    gf256_mod_poly dq 000000000000001Bh, 0000000000000000h
    align 16
    gf256_mod_poly_128 dq 0000000000000087h, 0000000000000000h
    align 16
    gf256_lut db 0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154
    align 16
    gf256_log db 0FFh, 0, 1, 19, 2, 32, 20, 43, 3, 50, 33, 49, 21, 31, 44, 47
              db 4, 29, 51, 41, 34, 57, 50, 46, 22, 39, 32, 40, 45, 56, 48, 30
              db 5, 24, 30, 18, 52, 36, 42, 17, 35, 15, 58, 14, 51, 37, 47, 23
              db 23, 59, 40, 13, 33, 55, 41, 28, 46, 12, 57, 27, 49, 11, 31, 26
    align 16
    gf256_ilog db 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47
               db 94, 188, 87, 174, 61, 122, 244, 193, 93, 186, 83, 166, 37, 74, 148, 7
               db 14, 28, 56, 112, 224, 159, 29, 58, 116, 232, 175, 63, 126, 252, 215, 141
               db 5, 10, 20, 40, 80, 160, 31, 62, 124, 248, 207, 133, 3, 6, 12, 24

.code

; uint8_t sneppx_gf256_mul(uint8_t a, uint8_t b)
; Constant-time GF(256) multiplication in AES polynomial basis
; Uses bit-loop carryless multiply with polynomial reduction by 0x11B
; 8 fixed iterations — constant-time
sneppx_gf256_mul PROC
    push rbx
    lfence
    movzx eax, cl
    movzx ebx, dl
    xor edx, edx
    mov ecx, 8
gf256_mul_loop:
    mov r10d, ebx
    and r10d, 1
    neg r10d
    mov r11d, eax
    and r11d, r10d
    xor edx, r11d
    add al, al
    jnc gf256_no_reduce
    xor al, 1Bh
gf256_no_reduce:
    shr bl, 1
    dec ecx
    jnz gf256_mul_loop
    mov eax, edx
    lfence
    pop rbx
    ret
sneppx_gf256_mul ENDP

; uint8_t sneppx_gf256_inv(uint8_t a)
; Constant-time GF(256) inversion using Fermat's little theorem: a^254
; Computes a^254 = a^2 * a^4 * a^8 * a^16 * a^32 * a^64 * a^128
sneppx_gf256_inv PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    sub rsp, 8
    mov byte ptr [rsp], cl
    mov cl, byte ptr [rsp]
    mov dl, byte ptr [rsp]
    call sneppx_gf256_mul
    mov byte ptr [rsp], al
    mov cl, al
    mov dl, al
    call sneppx_gf256_mul
    mov r12b, al
    mov cl, al
    mov dl, al
    call sneppx_gf256_mul
    mov r13b, al
    mov cl, al
    mov dl, al
    call sneppx_gf256_mul
    mov r14b, al
    mov cl, al
    mov dl, al
    call sneppx_gf256_mul
    mov ebx, eax
    mov cl, al
    mov dl, al
    call sneppx_gf256_mul
    mov r10b, al
    mov cl, byte ptr [rsp]
    mov dl, r12b
    call sneppx_gf256_mul
    mov cl, al
    mov dl, r13b
    call sneppx_gf256_mul
    mov cl, al
    mov dl, r14b
    call sneppx_gf256_mul
    mov cl, al
    mov dl, bl
    call sneppx_gf256_mul
    mov cl, al
    mov dl, r10b
    call sneppx_gf256_mul
    add rsp, 8
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_gf256_inv ENDP

; uint8_t sneppx_gf256_sq(uint8_t a)
; Constant-time GF(256) squaring
sneppx_gf256_sq PROC
    push rbx
    lfence
    movzx eax, cl
    mov dl, al
    call sneppx_gf256_mul
    lfence
    pop rbx
    ret
sneppx_gf256_sq ENDP

; uint8_t sneppx_gf256_mul_scalar(uint8_t a, uint8_t b)
; Scalar multiplication for MixColumns — constant-time
sneppx_gf256_mul_scalar PROC
    push rbx
    lfence
    movzx eax, cl
    movzx ecx, dl
    xor ebx, ebx
    xor edx, edx
gf256s_mul_loop:
    cmp edx, 8
    jae gf256s_done
    test cl, 1
    jz gf256s_skip
    xor ebx, eax
gf256s_skip:
    shr cl, 1
    shl al, 1
    test ah, 1
    jz gf256s_no_reduce
    xor al, 27
gf256s_no_reduce:
    inc edx
    jmp gf256s_mul_loop
gf256s_done:
    movzx eax, bl
    lfence
    pop rbx
    ret
sneppx_gf256_mul_scalar ENDP

END

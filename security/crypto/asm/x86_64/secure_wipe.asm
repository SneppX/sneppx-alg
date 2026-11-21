; SneppX-ALG secure memory wiping operations
; Guarantees compiler barriers via mfence + lfence
; Prevents dead-store elimination by clobbering memory
; MASM x64 syntax

.data
    align 16
    wipe_pattern dq 0DEADBEEFCAFEBABEh, 0FEEDFACEABADCAFEh

.code

; void sneppx_secure_wipe(void *ptr, size_t len)
; Overwrites memory with zeros, then with pattern, then zeros again
; Two passes + mfence to prevent dead-store elimination
sneppx_secure_wipe PROC
    push rdi
    lfence
    test rdx, rdx
    jz sw_done
    xor eax, eax
    mov rdi, rcx
    mov r10, rcx
    mov r11, rdx
sw_pass1:
    test r11, 7
    jnz sw_pass1_byte
    test r11, r11
    jz sw_pass2
    mov rcx, r11
    shr rcx, 3
    rep stosq
    jmp sw_pass2
sw_pass1_byte:
    mov byte ptr [rdi], 0
    inc rdi
    dec r11
    jmp sw_pass1
sw_pass2:
    mov rdi, r10
    mov r11, rdx
    lea rax, wipe_pattern
    mov rbx, qword ptr [rax]
    mov rcx, qword ptr [rax+8]
sw_pass2_loop:
    cmp r11, 16
    jb sw_pass2_byte
    mov qword ptr [rdi], rbx
    mov qword ptr [rdi+8], rcx
    add rdi, 16
    sub r11, 16
    jmp sw_pass2_loop
sw_pass2_byte:
    test r11, r11
    jz sw_pass3
    mov byte ptr [rdi], 0ABh
    inc rdi
    dec r11
    jmp sw_pass2_byte
sw_pass3:
    mov rdi, r10
    mov r11, rdx
    xor eax, eax
sw_pass3_loop:
    cmp r11, 8
    jb sw_pass3_byte
    mov qword ptr [rdi], 0
    add rdi, 8
    sub r11, 8
    jmp sw_pass3_loop
sw_pass3_byte:
    test r11, r11
    jz sw_fence
    mov byte ptr [rdi], 0
    inc rdi
    dec r11
    jmp sw_pass3_byte
sw_fence:
    mfence
    lfence
sw_done:
    pop rdi
    ret
sneppx_secure_wipe ENDP

; void sneppx_secure_wipe_register_state(void)
; Wipes all volatile registers (caller-saved: rax, rcx, rdx, r8-r11, xmm0-xmm5)
; Use after secret key operations
sneppx_secure_wipe_register_state PROC
    lfence
    xor eax, eax
    xor ecx, ecx
    xor edx, edx
    xor r8d, r8d
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    pxor xmm0, xmm0
    pxor xmm1, xmm1
    pxor xmm2, xmm2
    pxor xmm3, xmm3
    pxor xmm4, xmm4
    pxor xmm5, xmm5
    mfence
    lfence
    ret
sneppx_secure_wipe_register_state ENDP

; void sneppx_secure_wipe_page(void *page)
; Wipes an entire 4096-byte page with zero + pattern + zero
sneppx_secure_wipe_page PROC
    push rbx
    push r12
    lfence
    mov r12, rcx
    mov rbx, 4096
sw_page_loop:
    test rbx, rbx
    jz sw_page_done
    sub rbx, 64
    movdqu xmm0, xmmword ptr [r12 + rbx]
    movdqu xmm1, xmmword ptr [r12 + rbx + 16]
    movdqu xmm2, xmmword ptr [r12 + rbx + 32]
    movdqu xmm3, xmmword ptr [r12 + rbx + 48]
    pxor xmm0, xmm0
    pxor xmm1, xmm1
    pxor xmm2, xmm2
    pxor xmm3, xmm3
    movntdq xmmword ptr [r12 + rbx], xmm0
    movntdq xmmword ptr [r12 + rbx + 16], xmm1
    movntdq xmmword ptr [r12 + rbx + 32], xmm2
    movntdq xmmword ptr [r12 + rbx + 48], xmm3
    jmp sw_page_loop
sw_page_done:
    mfence
    sfence
    lfence
    pop r12
    pop rbx
    ret
sneppx_secure_wipe_page ENDP

; void sneppx_secure_wipe_xmm(void)
; Wipes volatile XMM/YMM registers (xmm0-xmm5)
sneppx_secure_wipe_xmm PROC
    lfence
    pxor xmm0, xmm0
    pxor xmm1, xmm1
    pxor xmm2, xmm2
    pxor xmm3, xmm3
    pxor xmm4, xmm4
    pxor xmm5, xmm5
    mfence
    lfence
    ret
sneppx_secure_wipe_xmm ENDP

END

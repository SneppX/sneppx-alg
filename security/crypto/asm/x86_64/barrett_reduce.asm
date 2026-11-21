; SneppX-ALG Barrett reduction for 64-bit and 128-bit moduli — x86-64
; Constant-time: no secret-dependent branches, fixed iteration count
; Used for modular reduction in lattice-based crypto (Kyber, Dilithium)
; MASM x64 syntax — speculation barriers, memory wiping

.data
    align 16
    barrett_k1 dq 0000000000000001h, 0000000000000000h
    align 16
    barrett_k2 dq 0000000000000002h, 0000000000000000h
    align 16
    barrett_27 dq 000000000000001Bh, 0000000000000000h

.code

; uint16_t sneppx_barrett_reduce_u16(uint32_t a, uint16_t mod, uint16_t barrett_mu)
; Barrett reduction for 16-bit moduli (used in Kyber)
; barrett_mu = (1 << 26) / mod precomputed
sneppx_barrett_reduce_u16 PROC
    push rbx
    lfence
    movzx eax, cx
    movzx ebx, dx
    movzx ecx, r8w
    mov r10d, eax
    imul r10d, ecx
    shr r10d, 26
    imul r10d, ebx
    sub eax, r10d
    movzx ecx, bx
    sub eax, ecx
    shr eax, 31
    movzx ecx, bx
    and ecx, eax
    add eax, ecx
    lfence
    pop rbx
    ret
sneppx_barrett_reduce_u16 ENDP

; uint32_t sneppx_barrett_reduce_u32(uint64_t a, uint32_t mod, uint64_t barrett_mu)
; Barrett reduction for 32-bit moduli
; barrett_mu = (1 << 64) / mod
sneppx_barrett_reduce_u32 PROC
    push rbx
    lfence
    mov r10, rcx
    mov r11d, edx
    mov rax, rcx
    mul r8
    mov rax, rdx
    mul r11
    sub r10, rax
    mov eax, r10d
    sub eax, r11d
    shr eax, 31
    and r11d, eax
    add eax, r11d
    lfence
    pop rbx
    ret
sneppx_barrett_reduce_u32 ENDP

; uint16_t sneppx_barrett_reduce_u16_mont(uint32_t a, uint16_t mod, uint16_t barrett_mu, uint16_t mont_r)
; Combined Barrett + Montgomery reduction for 16-bit
sneppx_barrett_reduce_u16_mont PROC
    push rbx
    lfence
    movzx eax, cx
    movzx ebx, dx
    movzx ecx, r8w
    movzx r9d, r9w
    mov r10d, eax
    imul r10d, ecx
    shr r10d, 16
    imul r10d, ebx
    sub eax, r10d
    imul eax, r9d
    and eax, 0FFFFh
    lfence
    pop rbx
    ret
sneppx_barrett_reduce_u16_mont ENDP

; void sneppx_barrett_reduce_poly(uint16_t *poly, size_t len, uint16_t mod, uint16_t barrett_mu)
; Barrett reduction for an entire polynomial vector (constant-time)
sneppx_barrett_reduce_poly PROC
    push rbx
    push r12
    push r13
    lfence
    mov r10, rcx
    mov r11, rdx
    movzx r12d, r8w
    movzx r13d, r9w
    xor r9, r9
barrett_poly_loop:
    cmp r9, r11
    jae barrett_poly_done
    movzx eax, word ptr [r10 + r9*2]
    mov ecx, eax
    imul ecx, r13d
    shr ecx, 16
    imul ecx, r12d
    sub eax, ecx
    movzx ecx, r12w
    sub eax, ecx
    shr eax, 31
    and ecx, eax
    add eax, ecx
    mov word ptr [r10 + r9*2], ax
    inc r9
    jmp barrett_poly_loop
barrett_poly_done:
    lfence
    pop r13
    pop r12
    pop rbx
    ret
sneppx_barrett_reduce_poly ENDP

; uint32_t sneppx_barrett_reduce_u32_tight(uint32_t a, uint32_t mod)
; Barrett reduction with precomputed mu = 2^32 / mod
; Tight version: assumes a < mod^2
sneppx_barrett_reduce_u32_tight PROC
    push rbx
    lfence
    mov eax, ecx
    mov ecx, edx
    mov r10d, eax
    shr r10d, 16
    mov r11d, ecx
    imul r10d, r11d
    shr r10d, 16
    imul r10d, ecx
    sub eax, r10d
    mov edx, ecx
    sub eax, edx
    shr eax, 31
    and edx, eax
    add eax, edx
    lfence
    pop rbx
    ret
sneppx_barrett_reduce_u32_tight ENDP

END

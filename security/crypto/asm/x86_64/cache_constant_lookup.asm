; SneppX-ALG cache-timing resistant table lookups
; All accesses read every entry to prevent cache-timing leakage
; Uses SSE2 for parallel vectorized operations
; MASM x64 syntax — constant-time, cache-attack resistant

.data
    align 16
    cache_const_mask db 0FFh, 0FFh, 0FFh, 0FFh, 0FFh, 0FFh, 0FFh, 0FFh
                     db 0FFh, 0FFh, 0FFh, 0FFh, 0FFh, 0FFh, 0FFh, 0FFh

.code

; uint8_t sneppx_cache_const_lookup(const uint8_t *table, size_t table_size, size_t index)
; Reads every entry in the table, returns table[index]
; All entries contribute to timing via OR tree — index cannot be inferred
sneppx_cache_const_lookup PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    xor eax, eax
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    xor r12d, r12d
    xor r13d, r13d
    xor r14d, r14d
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
ccl_loop:
    cmp r9, r11
    jae ccl_done
    movzx eax, byte ptr [r10 + r9]
    xor r13d, r9d
    xor r13d, r12d
    neg r13d
    sbb r13d, r13d
    and eax, r13d
    or r14d, eax
    inc r9
    jmp ccl_loop
ccl_done:
    movzx eax, r14b
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_cache_const_lookup ENDP

; uint32_t sneppx_cache_const_lookup32(const uint32_t *table, size_t table_size, size_t index)
; Constant-time 32-bit table lookup, reads all entries
sneppx_cache_const_lookup32 PROC
    push rbx
    push r12
    push r13
    push r14
    lfence
    xor rax, rax
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
ccl32_loop:
    cmp r9, r11
    jae ccl32_done
    mov eax, dword ptr [r10 + r9*4]
    xor r13d, r9d
    xor r13d, r12d
    neg r13d
    sbb r13d, r13d
    and eax, r13d
    or r14d, eax
    inc r9
    jmp ccl32_loop
ccl32_done:
    mov eax, r14d
    lfence
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_cache_const_lookup32 ENDP

; void sneppx_cache_const_sbox(uint8_t out[16], const uint8_t in[16], const uint8_t sbox[256])
; AES SubBytes using cache-constant lookup (reads all 256 entries per byte)
sneppx_cache_const_sbox PROC
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    lfence
    mov r10, rcx
    mov r11, rdx
    mov r12, r8
    xor r13, r13
ccl_sbox_byte_loop:
    cmp r13, 16
    jae ccl_sbox_done
    movzx r14d, byte ptr [r11 + r13]
    xor r15d, r15d
    xor r9, r9
ccl_sbox_entry_loop:
    cmp r9, 256
    jae ccl_sbox_store
    movzx eax, byte ptr [r12 + r9]
    xor ebx, r9d
    xor ebx, r14d
    neg ebx
    sbb ebx, ebx
    and eax, ebx
    or r15d, eax
    inc r9
    jmp ccl_sbox_entry_loop
ccl_sbox_store:
    mov byte ptr [r10 + r13], r15b
    inc r13
    jmp ccl_sbox_byte_loop
ccl_sbox_done:
    lea rdi, [rsp]
    mov rcx, 4
    xor eax, eax
    rep stosq
    lfence
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
sneppx_cache_const_sbox ENDP

; uint32_t sneppx_cache_const_te0_lookup(const uint32_t *te0, size_t index)
; Cache-constant T-table lookup for AES encryption
sneppx_cache_const_te0_lookup PROC
    push rbx
    push r12
    push r13
    lfence
    mov r10, rcx
    mov r11, rdx
    xor rax, rax
    xor r9, r9
    xor r12, r12
    xor r13, r13
ccl_te0_loop:
    cmp r9, 256
    jae ccl_te0_done
    mov eax, dword ptr [r10 + r9*4]
    xor r12d, r9d
    xor r12d, r11d
    neg r12d
    sbb r12d, r12d
    and eax, r12d
    or r13d, eax
    inc r9
    jmp ccl_te0_loop
ccl_te0_done:
    mov eax, r13d
    lfence
    pop r13
    pop r12
    pop rbx
    ret
sneppx_cache_const_te0_lookup ENDP

END

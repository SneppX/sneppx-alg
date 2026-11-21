; SNEPPX Constant-Time Operations (MASM x86-64)
; Side-channel resistant comparison and memory operations

.data
    align 16

.code

; int SNEPPX_ct_compare(const uint8_t* a, const uint8_t* b, size_t len)
SNEPPX_ct_compare PROC
    xor     eax, eax
    xor     ecx, ecx
    test    r8, r8
    jz      done
loop_start:
    movzx   r9d, BYTE PTR [rcx+rdx]
    movzx   r10d, BYTE PTR [rcx+r8]
    xor     r9d, r10d
    or      eax, r9d
    inc     ecx
    cmp     rcx, r8
    jb      loop_start
done:
    ret
SNEPPX_ct_compare ENDP

; void SNEPPX_ct_memzero(void* ptr, size_t len)
SNEPPX_ct_memzero PROC
    test    rdx, rdx
    jz      done
    xor     eax, eax
    mov     rcx, rdx
    xor     r8, r8
clear_loop:
    mov     BYTE PTR [rcx+r8], al
    inc     r8d
    cmp     r8, rdx
    jb      clear_loop
done:
    ret
SNEPPX_ct_memzero ENDP

END

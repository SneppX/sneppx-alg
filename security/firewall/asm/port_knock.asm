option casemap:none

.code

; BOOL firewall_verify_knock(DWORD* knock_seq, DWORD* expected_seq, DWORD len, LONGLONG window_ns, LONGLONG* timestamps)
;   RCX = knock_seq (array of DWORD ports actually hit)
;   RDX = expected_seq (array of DWORD expected ports)
;   R8D = len (number of knocks)
;   R9  = window_ns (time window in nanoseconds)
;   [rsp+40] = timestamps (array of LONGLONG, one per knock)
PUBLIC firewall_verify_knock
firewall_verify_knock PROC
    push    rbx
    push    rsi
    push    rdi
    mov     r10d, r8d
    mov     r11, r9
    mov     rdi, qword ptr [rsp+56]
    xor     ebx, ebx
    xor     esi, esi
seq_loop:
    cmp     ebx, r10d
    jae     time_check
    mov     eax, dword ptr [rcx + rbx*4]
    mov     edx, dword ptr [rdx + rbx*4]
    cmp     eax, edx
    jne     knock_fail
    inc     ebx
    jmp     seq_loop
time_check:
    cmp     r10d, 2
    jb      knock_ok
    mov     rcx, qword ptr [rdi]
    mov     rsi, qword ptr [rdi + 8]
    sub     rsi, rcx
    cmp     rsi, r11
    ja      knock_fail
knock_ok:
    mov     eax, 1
    jmp     knock_done
knock_fail:
    xor     eax, eax
knock_done:
    pop     rdi
    pop     rsi
    pop     rbx
    ret
firewall_verify_knock ENDP

END

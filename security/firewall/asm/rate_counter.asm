option casemap:none

.code

; BOOL firewall_rate_check(LONGLONG now_ns, LONGLONG window_start_ns, volatile LONG* counter, LONG limit)
;   RCX = now_ns
;   RDX = window_start_ns
;   R8  = pointer to counter (32-bit)
;   R9D = limit
; returns RAX = 1 if under limit (allow), 0 if over limit (deny)
PUBLIC firewall_rate_check
firewall_rate_check PROC
    cmp     rcx, rdx
    jae     check_window
    mov     dword ptr [r8], 0
check_window:
    mov     eax, dword ptr [r8]
    cmp     eax, r9d
    jae     rate_deny
    lock inc dword ptr [r8]
    mov     eax, 1
    ret
rate_deny:
    xor     eax, eax
    ret
firewall_rate_check ENDP

; LONGLONG firewall_rdtsc(VOID)
; returns TSC timestamp in RAX
PUBLIC firewall_rdtsc
firewall_rdtsc PROC
    rdtsc
    shl     rdx, 32
    or      rax, rdx
    ret
firewall_rdtsc ENDP

END

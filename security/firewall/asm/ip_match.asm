option casemap:none

.code

; BOOL firewall_ip_match(const BYTE* cidr_bytes, const BYTE* ip_bytes, BYTE prefix_len)
;   RCX = cidr_bytes (4 or 16 bytes network prefix)
;   RDX = ip_bytes   (4 or 16 bytes IP address)
;   R8B = prefix_len (0-32 for IPv4, 0-128 for IPv6)
; returns RAX = 1 if match, 0 otherwise
PUBLIC firewall_ip_match
firewall_ip_match PROC
    cmp     r8b, 0
    je      match_true
    cmp     r8b, 32
    ja      maybe_ipv6
    mov     r9d, 4
    jmp     match_loop
maybe_ipv6:
    mov     r9d, 16
match_loop:
    xor     r10d, r10d
    xor     eax, eax
loop_top:
    cmp     r10d, r9d
    jae     match_done
    movzx   r11d, byte ptr [rcx + r10]
    movzx   ecx,  byte ptr [rdx + r10]
    xor     ecx,  r11d
    test    ecx,  ecx
    jnz     match_false
    inc     r10d
    jmp     loop_top
match_true:
    mov     eax, 1
    ret
match_false:
    xor     eax, eax
    ret
match_done:
    mov     eax, 1
    ret
firewall_ip_match ENDP

; BOOL firewall_ip_match_cidr(const BYTE* network_bytes, const BYTE* ip_bytes, BYTE prefix_len)
;   Constant-time: XOR -> AND mask -> check zero
PUBLIC firewall_ip_match_cidr
firewall_ip_match_cidr PROC
    xor     eax, eax
    cmp     r8b, 0
    je      cidr_true
    movzx   r9d, r8b
    shr     r9d, 3
    mov     r10b, r8b
    and     r10b, 7
    xor     r11d, r11d
cidr_loop:
    cmp     r11d, r9d
    jae     cidr_partial
    movzx   eax, byte ptr [rcx + r11]
    movzx   edx, byte ptr [rdx + r11]
    xor     eax, edx
    test    eax, eax
    jnz     cidr_false
    inc     r11d
    jmp     cidr_loop
cidr_partial:
    test    r10b, r10b
    jz      cidr_true
    mov     cl, byte ptr [rcx + r9]
    mov     dl, byte ptr [rdx + r9]
    xor     cl, dl
    mov     dl, -1
    shl     dl, 8
    sub     dl, r10b
    and     cl, dl
    test    cl, cl
    jnz     cidr_false
cidr_true:
    mov     eax, 1
    ret
cidr_false:
    xor     eax, eax
    ret
firewall_ip_match_cidr ENDP

END

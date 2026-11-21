option casemap:none

.code

; DWORD firewall_hash_ip(DWORD ipv4)
;   RCX = ipv4 (32-bit integer)
; returns RAX = hash using CRC32
PUBLIC firewall_hash_ip
firewall_hash_ip PROC
    crc32   eax, ecx
    ret
firewall_hash_ip ENDP

; BOOL firewall_conn_track(DWORD ip, DWORD* state_table, DWORD table_size, DWORD now_sec, DWORD timeout_sec)
;   RCX = ip
;   RDX = state_table
;   R8D = table_size (must be power of 2)
;   R9D = now_sec
;   [rsp+40] = timeout_sec
PUBLIC firewall_conn_track
firewall_conn_track PROC
    push    rbx
    push    rsi
    mov     r10d, r9d
    mov     r11d, dword ptr [rsp+40]
    mov     eax, ecx
    crc32   eax, ecx
    dec     r8d
    and     eax, r8d
    shl     rax, 4
    add     rax, rdx
    mov     ebx, dword ptr [rax]
    mov     esi, dword ptr [rax+4]
    cmp     ebx, ecx
    jne     conn_new
    mov     edx, r10d
    sub     edx, esi
    cmp     edx, r11d
    ja      conn_expired
    inc     dword ptr [rax+8]
    mov     dword ptr [rax+4], r10d
    mov     eax, 1
    jmp     conn_done
conn_new:
    mov     dword ptr [rax], ecx
    mov     dword ptr [rax+4], r10d
    mov     dword ptr [rax+8], 1
    mov     eax, 1
    jmp     conn_done
conn_expired:
    mov     dword ptr [rax], ecx
    mov     dword ptr [rax+4], r10d
    mov     dword ptr [rax+8], 1
    mov     eax, 1
conn_done:
    pop     rsi
    pop     rbx
    ret
firewall_conn_track ENDP

END

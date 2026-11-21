// Rust Security Bindings — SKELETON
// VERSION: v0.5
//
// FFI wrappers around arix_security_c for Rust consumers.
// Declares extern "C" functions and provides safe wrappers.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::ptr;

extern "C" {
    fn arix_c_hash_blake3(data: *const u8, len: usize, out: *mut u8, out_len: usize) -> i32;
    fn arix_c_hash_sha3_256(data: *const u8, len: usize, out: *mut u8) -> i32;
    fn arix_c_chacha20_encrypt(key: *const u8, nonce: *const u8, plaintext: *const u8,
                                len: usize, ciphertext: *mut u8) -> i32;
    fn arix_c_chacha20_decrypt(key: *const u8, nonce: *const u8, ciphertext: *const u8,
                                len: usize, plaintext: *mut u8) -> i32;
    fn arix_c_ct_memzero(ptr: *mut c_void, len: usize);
    fn arix_c_ct_memcmp(a: *const c_void, b: *const c_void, len: usize) -> i32;
}

/// Compute a SHA3-256 hash.
pub fn sha3_256(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    unsafe { arix_c_hash_sha3_256(data.as_ptr(), data.len(), out.as_mut_ptr()); }
    out
}

/// Compute a Blake3 hash with the given output length.
pub fn blake3(data: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_len];
    unsafe { arix_c_hash_blake3(data.as_ptr(), data.len(), out.as_mut_ptr(), out_len); }
    out
}

/// Encrypt data with ChaCha20.
pub fn chacha20_encrypt(key: &[u8; 32], nonce: &[u8; 12], plaintext: &[u8]) -> Vec<u8> {
    let mut ciphertext = vec![0u8; plaintext.len()];
    unsafe {
        arix_c_chacha20_encrypt(
            key.as_ptr(), nonce.as_ptr(), plaintext.as_ptr(),
            plaintext.len(), ciphertext.as_mut_ptr(),
        );
    }
    ciphertext
}

/// Securely zero memory.
pub fn secure_zero(buf: &mut [u8]) {
    unsafe { arix_c_ct_memzero(buf.as_mut_ptr() as *mut c_void, buf.len()); }
}

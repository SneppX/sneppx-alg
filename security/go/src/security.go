// Go Security Bindings — SKELETON
// VERSION: v0.5
//
// cgo wrappers for the arix_security_c library.
// Package security exposes hashing, encryption, and secure memory.
package security

/*
#cgo LDFLAGS: -larix_security_c
#include "arix_c.h"
*/
import "C"
import (
	"unsafe"
)

// HashBlake3 computes a Blake3 hash of data.
func HashBlake3(data []byte, outLen int) ([]byte, error) {
	out := make([]byte, outLen)
	_ = C.arix_c_hash_blake3(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
		C.size_t(outLen),
	)
	return out, nil
}

// HashSHA3256 computes a SHA3-256 hash.
func HashSHA3256(data []byte) ([]byte, error) {
	out := make([]byte, 32)
	_ = C.arix_c_hash_sha3_256(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	return out, nil
}

// ChaCha20Encrypt encrypts data using ChaCha20.
func ChaCha20Encrypt(key, nonce, plaintext []byte) ([]byte, error) {
	ciphertext := make([]byte, len(plaintext))
	_ = C.arix_c_chacha20_encrypt(
		(*C.uint8_t)(unsafe.Pointer(&key[0])),
		(*C.uint8_t)(unsafe.Pointer(&nonce[0])),
		(*C.uint8_t)(unsafe.Pointer(&plaintext[0])),
		C.size_t(len(plaintext)),
		(*C.uint8_t)(unsafe.Pointer(&ciphertext[0])),
	)
	return ciphertext, nil
}

// ChaCha20Decrypt decrypts data using ChaCha20.
func ChaCha20Decrypt(key, nonce, ciphertext []byte) ([]byte, error) {
	plaintext := make([]byte, len(ciphertext))
	_ = C.arix_c_chacha20_decrypt(
		(*C.uint8_t)(unsafe.Pointer(&key[0])),
		(*C.uint8_t)(unsafe.Pointer(&nonce[0])),
		(*C.uint8_t)(unsafe.Pointer(&ciphertext[0])),
		C.size_t(len(ciphertext)),
		(*C.uint8_t)(unsafe.Pointer(&plaintext[0])),
	)
	return plaintext, nil
}

// SecureZero securely zeroes memory.
func SecureZero(ptr []byte) {
	C.arix_c_ct_memzero(unsafe.Pointer(&ptr[0]), C.size_t(len(ptr)))
}

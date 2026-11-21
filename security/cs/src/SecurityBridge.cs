/*
 * C# Security Bindings — SKELETON
 * VERSION: v0.5
 *
 * P/Invoke wrappers for the arix_security_c library.
 * Allows .NET applications to call ChaCha20, Blake3, AEAD, etc.
 */

using System;
using System.Runtime.InteropServices;

namespace Arix.Security
{
    public static class SecureMemory
    {
        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern void arix_c_ct_memzero(IntPtr ptr, UIntPtr len);

        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern int arix_c_ct_memcmp(IntPtr a, IntPtr b, UIntPtr len);
    }

    public static class Hash
    {
        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern int arix_c_hash_blake3(byte[] data, UIntPtr len, byte[] out_buf, UIntPtr out_len);

        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern int arix_c_hash_sha3_256(byte[] data, UIntPtr len, byte[] out_buf);
    }

    public static class Cipher
    {
        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern int arix_c_chacha20_encrypt(byte[] key, byte[] nonce, byte[] plaintext,
                                                          UIntPtr len, byte[] ciphertext);

        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern int arix_c_chacha20_decrypt(byte[] key, byte[] nonce, byte[] ciphertext,
                                                          UIntPtr len, byte[] plaintext);
    }

    public static class KeyDerivation
    {
        [DllImport("arix_security_c", CallingConvention = CallingConvention.Cdecl)]
        public static extern int arix_c_argon2_hash(string password, UIntPtr pwd_len,
                                                     byte[] salt, UIntPtr salt_len,
                                                     byte[] out_buf, UIntPtr out_len,
                                                     uint t_cost, uint m_cost, uint parallelism);
    }
}

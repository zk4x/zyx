// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Fnv hasher const initializable

use std::hash::Hasher;

pub struct FHasher(u64);

impl Default for FHasher {
    fn default() -> FHasher {
        FHasher(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for FHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let FHasher(mut hash) = *self;
        for byte in bytes.iter() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        *self = FHasher(hash);
    }
}

pub struct AHasher(u64);

impl Default for AHasher {
    fn default() -> Self {
        // non-zero seed
        Self(0xcbf29ce484222325 ^ 0x9e3779b97f4a7c15)
    }
}

impl Hasher for AHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;

        let mut i = 0;

        // process 8-byte chunks
        while i + 8 <= bytes.len() {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[i..i + 8]);
            let mut k = u64::from_le_bytes(buf);

            k = k.wrapping_mul(0x9e3779b97f4a7c15);
            k ^= k >> 33;

            h ^= k;
            h = h.wrapping_mul(0xc2b2ae3d27d4eb4f);
            h ^= h >> 29;

            i += 8;
        }

        // tail
        while i < bytes.len() {
            let mut k = bytes[i] as u64;
            k = k.wrapping_mul(0x9e3779b97f4a7c15);

            h ^= k;
            h = h.wrapping_mul(0xc2b2ae3d27d4eb4f);

            i += 1;
        }

        self.0 = h;
    }
}

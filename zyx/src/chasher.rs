//! Fnv hasher const initializable

pub struct CHasher(u64);

impl Default for CHasher {
    fn default() -> CHasher {
        CHasher(0xcbf2_9ce4_8422_2325)
    }
}

/*impl CHasher {
    pub(crate) const fn new() -> CHasher {
        CHasher(0xcbf29ce484222325)
    }
}*/

impl std::hash::Hasher for CHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let CHasher(mut hash) = *self;
        for byte in bytes.iter() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        *self = CHasher(hash);
    }
}

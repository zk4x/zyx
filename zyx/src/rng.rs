//! Just a few basic random generaton things from rand-core adjusted for purposes of zyx.
//! Copyright 2018 Developers of the Rand project.

use crate::{DType, Scalar};

//
pub(super) struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub(super) const fn seed_from_u64(mut state: u64) -> Self {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        let mut s = [0; 4];

        // Once rust supports for loops in const functions, this can be written as for loop

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        s[0] = z;

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        s[1] = z;

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        s[2] = z;

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        s[3] = z;

        // By using a non-zero PHI we are guaranteed to generate a non-zero state
        // Thus preventing a recursion between from_seed and seed_from_u64.
        //debug_assert_ne!(s, [0; 4]);
        Self { s }
    }

    const fn next_u64(&mut self) -> u64 {
        let res = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(45);

        res
    }

    pub(super) fn rand<T: Scalar>(&mut self) -> T {
        match T::dtype() {
            DType::BF16 => todo!(),
            DType::F8 => todo!(),
            DType::F16 => todo!(),
            DType::F32 => todo!(),
            DType::F64 => todo!(),
            DType::U8 => todo!(),
            DType::U16 => todo!(),
            DType::U32 => todo!(),
            DType::U64 => self.next_u64().cast(),
            DType::I8 => todo!(),
            DType::I16 => todo!(),
            DType::I32 => todo!(),
            DType::I64 => todo!(),
            DType::Bool => todo!(),
        }
    }
}

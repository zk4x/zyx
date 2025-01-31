//! Just a few basic random generaton things from rand-core adjusted for purposes of zyx.
//! Copyright 2018 Developers of the Rand project.

use crate::{DType, Scalar};

//
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub(super) const fn seed_from_u64(mut state: u64) -> Self {
        const PHI: u64 = 0x9e37_79b9_7f4a_7c15;

        const A: u64 = 0xbf58_476d_1ce4_e5b9;
        const B: u64 = 0x94d0_49bb_1331_11eb;

        let mut s = [0; 4];

        // Once rust supports for loops in const functions, this can be written as for loop

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(A);
        z = (z ^ (z >> 27)).wrapping_mul(B);
        z = z ^ (z >> 31);
        s[0] = z;

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(A);
        z = (z ^ (z >> 27)).wrapping_mul(B);
        z = z ^ (z >> 31);
        s[1] = z;

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(A);
        z = (z ^ (z >> 27)).wrapping_mul(B);
        z = z ^ (z >> 31);
        s[2] = z;

        state = state.wrapping_add(PHI);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(A);
        z = (z ^ (z >> 27)).wrapping_mul(B);
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

    const fn next_u32(&mut self) -> u32 {
        let x = self.next_u64().to_ne_bytes();
        u32::from_ne_bytes([x[0], x[1], x[2], x[3]])
    }

    const fn next_u16(&mut self) -> u16 {
        let x = self.next_u64().to_ne_bytes();
        u16::from_ne_bytes([x[0], x[1]])
    }

    const fn next_u8(&mut self) -> u8 {
        let x = self.next_u64().to_ne_bytes();
        u8::from_ne_bytes([x[0]])
    }

    #[allow(clippy::cast_precision_loss)]
    const fn next_f32(&mut self) -> f32 {
        const A: u32 = 1_664_525;
        const C: u32 = 1_013_904_223;
        const M: u32 = u32::MAX;

        let seed = self.next_u32();

        // Generate the next random number using the seed and LCG formula
        let next_seed = (A.wrapping_mul(seed).wrapping_add(C)) % M;

        // Convert the u32 result to a float in the range [0, 1)
        next_seed as f32 / M as f32
    }

    pub(super) fn rand<T: Scalar>(&mut self) -> T {
        match T::dtype() {
            DType::BF16 | DType::F16 | DType::F32 | DType::F64 => self.next_f32().cast(),
            DType::U8 | DType::Bool => self.next_u8().cast(),
            DType::U16 => self.next_u16().cast(),
            DType::U32 => self.next_u32().cast(),
            DType::U64 => self.next_u64().cast(),
            DType::I8 => unsafe { std::mem::transmute::<u8, i8>(self.next_u8()) }.cast(),
            DType::I16 => unsafe { std::mem::transmute::<u16, i16>(self.next_u16()) }.cast(),
            DType::I32 => unsafe { std::mem::transmute::<u32, i32>(self.next_u32()) }.cast(),
            DType::I64 => unsafe { std::mem::transmute::<u64, i64>(self.next_u64()) }.cast(),
        }
    }
}

/*#[test]
fn rng1() {
    let mut rng = Rng::seed_from_u64(42069);
    for _ in 0..10 {
        let x: f32 = rng.rand();
        println!("{x}");
    }
}*/

//! Exact floating-point remainder, `burn-flex`'s `((a % b) + b) % b` bit for
//! bit.
//!
//! # What Flex computes
//!
//! `burn-flex` 0.21 (`ops/float.rs`, `float_remainder`, `float_remainder_scalar`)
//! evaluates `((a % b) + b) % b` in `f32`: Rust's `%` is `fmodf`, **exact** (the
//! result is always representable, whatever the quotient), the `+ b` an IEEE
//! round-to-nearest-even add, the outer `%` exact again. The scalar form
//! converts its scalar as `to_f64() as f32` and evaluates the same formula.
//! Consequences, all of which the program reproduces:
//!
//! - `b = +-0`, `b = +-inf`, `a = +-inf` and a NaN either side give NaN
//!   (`fmod(x, 0)`, `fmod(inf, y)`, and `a % inf = a` then `a + inf = inf`,
//!   `inf % inf = NaN`); the program stores the canonical `0x7fc00000`, Flex's
//!   payload being the host libm's.
//! - the result's sign is `b`'s, a zero included: `-4 % 2 = +0`, `4 % -2 = -0`,
//!   `0 % b = copysign(0, b)`, and an `a` so small beside `b` that
//!   `fl(r + b) = b` gives `copysign(0, b)` too (`1e-20 % -1 = -0`, not `-1`).
//! - `a + b` can overflow when `|a| < |b|` are the same sign and huge
//!   (`3.0e38 % 3.3e38`: `r + b = inf`, `inf % b = NaN`).
//! - denormals are real operands and results: `fmod` is exact on them and
//!   Flex's `f32` add keeps them.
//!
//! # The algorithm
//!
//! With `r = fmod(a, b)` (`|r| < |b|`, sign of `a`) and `y = fl(r + b)`, the
//! outer `%` is, exactly, `y - b` when `|b| <= |y| < 2|b|` (Sterbenz: exact),
//! `0` when `|y| = 2|b|`, and `y` when `|y| < |b|` -- in each case with `b`'s
//! sign, a zero's too. In magnitudes, `M = |y|` runs twice through `if M >= B
//! { M -= B }`.
//!
//! The inexact work is `r`. `a = MA 2^(EA-150)`, `b = MB 2^(EB-150)` with
//! integer significands (`MA`, `MB < 2^24`; a denormal has `E = 1`, no hidden
//! bit). `MB` is first normalised into `[2^23, 2^24)` (`s` places left,
//! `EB -= s`, a no-op for a normal `b`) -- so `MA mod MB` is one conditional
//! subtraction -- and then `r = (MA mod MB) 2^d mod MB` for `d = EA - EB >= 0`
//! by `d` restoring steps `r = 2r; if r >= MB { r -= MB }` on integers under
//! 2^25. A lane whose `d` is smaller stops early (its step predicated off by a
//! per-lane counter); `d` is at most 276 (`EA = 254`, the least denormal `b`
//! normalised to `EB = -22`). A lane with `d < 0` has `|a| < |b|`, so `r = a`.
//!
//! `y` is then formed with the device's `SFPADD`, which flushes denormals. So
//! that it never sees one, `B` and `r` are scaled by `2^64` where `EB <= 100`:
//! both are then normal, `y`'s rounding is Flex's (the sum of two multiples
//! of `2^-149` below `2^-126` is exact, and above it the rounding is the same
//! at any scale), and the final `z` is unscaled by integer arithmetic into a
//! denormal where Flex has one. Where `EB > 100` a denormal `r` is below a
//! quarter of `ulp(b)`, so flushing it to zero changes nothing. The result is
//! Flex's bits for every input but a NaN's payload.
//!
//! # Cost
//!
//! 276 restoring steps of 8 instructions per row group, the body stored once
//! and pushed 32 times by the role runner (`super::super::super::code`).

use super::super::kernel::{A_ROW, B_ROW, OUT_ROW, SPILL_ROW};
use super::super::{Cond, Format, LReg, Program};
use tt_isa::isa::generated::encode;

const L0: LReg = LReg::L0;
const L1: LReg = LReg::L1;
const L2: LReg = LReg::L2;
const L3: LReg = LReg::L3;
const L4: LReg = LReg::L4;
const L5: LReg = LReg::L5;
const L6: LReg = LReg::L6;
const L7: LReg = LReg::L7;
const ZERO: LReg = LReg::ZERO;

/// Restoring steps in the long division: the most `d = EA - EB` can be.
pub const STEPS: u32 = 276;

/// `EB <= SCALE_UNTIL` scales `B` and `r` by `2^64`; a larger `EB` does not.
const SCALE_UNTIL: i32 = 100;

/// The positive NaN the program stores.
pub const NAN_BITS: u32 = 0x7fc0_0000;

/// What the program computes. Only [`Variant::Exact`] is an op; the others are
/// the negative controls of the gate (`step133_remainder`): each is a plausible
/// wrong remainder the corpus must catch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// Flex's `((a % b) + b) % b`.
    Exact,
    /// `r = a - b * trunc(a / b)` where `fmod` should be, the formula applied
    /// to its own result as Flex's is.
    TruncReduction,
    /// `fmod(a, b)` alone: the `+ b` and the outer `%` dropped, so the sign is
    /// `a`'s.
    NoCorrection,
    /// `b % a`: the operands the wrong way round.
    Swapped,
}

/// Where `b` comes from.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Divisor {
    /// The second operand's tile.
    Tile,
    /// A scalar, its bits.
    Scalar(f32),
}

/// `SFPPUSHC`/`SFPPOPC` without the builder's scope: the builder's flag
/// predicate would be `SFPSETCC`, here it is `SFPIADD`'s own condition code.
fn push(p: &mut Program) {
    p.raw(encode::sfppushc(0, 0).unwrap());
}

fn pop(p: &mut Program) {
    p.raw(encode::sfppopc(0, 0).unwrap());
}

/// `d = c + imm`, lanes kept (flag `>= 0` of the result) among those enabled.
fn iadd_imm_keep_nonneg(p: &mut Program, c: LReg, imm: i32, d: LReg) {
    assert!((-2048..2048).contains(&imm));
    // `SFPIADD_MOD1_ARG_IMM | SFPIADD_MOD1_CC_GTE0`.
    p.raw(encode::sfpiadd(imm as u32 & 0xfff, c.index(), d.index(), 1 | 8).unwrap());
}

/// `d = c + d`, lanes kept where the result is negative.
fn iadd_keep_neg(p: &mut Program, c: LReg, d: LReg) {
    // `SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_LT0`.
    p.raw(encode::sfpiadd(0, c.index(), d.index(), 0).unwrap());
}

/// The program: one row-group loop over the tile.
pub fn code(variant: Variant, divisor: Divisor) -> super::super::super::code::Code {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, o| match variant {
        Variant::TruncReduction => naive_body(p, o, divisor),
        v => exact_body(p, o, v, divisor),
    });
    p.finish_code()
}

/// `a` into `L0`, `b` into `L1`, as raw bits.
fn load_operands(p: &mut Program, o: u32, variant: Variant, divisor: Divisor) {
    let (a_at, b_at) = if variant == Variant::Swapped {
        (B_ROW + o, A_ROW + o)
    } else {
        (A_ROW + o, B_ROW + o)
    };
    p.load(LReg::L0, Format::Int32, a_at);
    match divisor {
        Divisor::Tile => p.load(LReg::L1, Format::Int32, b_at),
        Divisor::Scalar(s) => p.loadi_bits(LReg::L1, s.to_bits()),
    }
}

/// One row group of the exact program (see the module documentation), `a` and
/// `b` as raw bits throughout: nothing before `y` is floating-point.
fn exact_body(p: &mut Program, o: u32, variant: Variant, divisor: Divisor) {
    load_operands(p, o, variant, divisor);
    // L6 = a ^ b: bit 31 is whether `r` and `b` differ in sign.
    p.mov(L0, L6);
    p.xor(L1, L6);
    // A, B: the magnitudes.
    p.loadi_bits(L7, 0x7fff_ffff);
    p.and(L0, L7, L0);
    p.and(L1, L7, L1);
    // Exponent fields and significands (with the hidden bit).
    p.exponent(L0, false, L2);
    p.exponent(L1, false, L3);
    p.mantissa(L0, L4);
    p.mantissa(L1, L5);
    // A denormal (or zero): no hidden bit, `E = 1`.
    p.loadi_bits(L7, 1 << 23);
    p.if_(Cond::Eq0(L2), |p| {
        p.xor(L7, L4);
        p.loadi_bits(L2, 1);
    });
    p.if_(Cond::Eq0(L3), |p| {
        p.xor(L7, L5);
        p.loadi_bits(L3, 1);
    });
    // L2 = EA, L3 = EB, L4 = MA, L5 = MB. Normalise MB: `s = lz(MB) - 8`
    // places, none for a normal `b`.
    p.leading_zeros(L5, false, L7);
    p.iadd_imm(L7, -8, L7);
    p.shl_by(L7, L5);
    // L1 = EB' = EB - s; L7 = d = EA - EB'.
    p.mov(L7, L1);
    p.isub_from(L3, L1);
    p.mov(L1, L7);
    p.isub_from(L2, L7);
    // L3 = the exponent `r` carries: EB' where `d >= 0`, else `a`'s own.
    p.mov(L2, L3);
    p.if_(Cond::Gte0(L7), |p| p.mov(L1, L3));
    // L0 = r = MA, less MB where `d >= 0` and it is not below it. L2 = -MB.
    p.mov(L4, L0);
    p.mov(L5, L2);
    p.isub_from(ZERO, L2);
    p.if_(Cond::Gte0(L7), |p| {
        p.if_(Cond::LessEq(L5, L0), |p| p.iadd(L2, L0))
    });
    // The long division. L0 = r, L2 = -MB, L5 = MB, L7 = the step counter,
    // kept: L1 = EB', L3 = rexp, L6 = a ^ b.
    for _ in 0..STEPS {
        push(p);
        iadd_imm_keep_nonneg(p, L7, -1, L7);
        p.iadd(L0, L0);
        push(p);
        iadd_keep_neg(p, L2, L0);
        p.iadd(L5, L0);
        pop(p);
        pop(p);
    }
    // m = 64 where EB' <= SCALE_UNTIL, else 0 (L2).
    p.iadd_imm(L1, -(SCALE_UNTIL + 1), L7);
    p.mov(ZERO, L2);
    p.if_(Cond::Lt0(L7), |p| p.loadi_bits(L2, 64));
    // L4 = B' bits: ((EB' + m) << 23) | (MB & 0x7fffff).
    p.mov(L1, L4);
    p.iadd(L2, L4);
    p.shl(L4, 23, L4);
    p.loadi_bits(L7, 0x7f_ffff);
    p.and(L5, L7, L5);
    p.or(L4, L5, L4);
    // L1 = R' bits: r normalised, its exponent field `rexp + m + 8 - lz`.
    p.leading_zeros(L0, false, L7);
    p.mov(L3, L1);
    p.iadd(L2, L1);
    p.iadd_imm(L1, 8, L1);
    p.mov(L7, L3);
    p.isub_from(L1, L3);
    p.iadd_imm(L7, -8, L7);
    p.shl_by(L7, L0);
    p.loadi_bits(L7, 0x7f_ffff);
    p.and(L0, L7, L5);
    p.shl(L3, 23, L1);
    p.or(L1, L5, L1);
    // Zero where `r` is, and where the field would be below one (a denormal
    // `r` beside a large `b`: it cannot change `y`).
    p.if_(Cond::Eq0(L0), |p| p.mov(ZERO, L1));
    p.iadd_imm(L3, -1, L7);
    p.if_(Cond::Lt0(L7), |p| p.mov(ZERO, L1));
    p.mov(L2, L5);
    if variant == Variant::NoCorrection {
        // `fmod` alone: z = R'.
        p.mov(L1, L2);
    } else {
        // y = B' + R' with the sign `a ^ b` gives R'; z = y reduced twice.
        p.loadi_bits(L7, 0x8000_0000);
        p.and(L6, L7, L6);
        p.or(L1, L6, L1);
        p.add(L4, L1, L2);
        for _ in 0..2 {
            p.if_(Cond::LessEq(L4, L2), |p| p.sub(L2, L4, L2));
        }
    }
    // Unscale z' (L2) by `2^-m` (L5): a denormal where `e < 1`.
    p.exponent(L2, false, L7);
    p.mov(L5, L0);
    p.isub_from(L7, L0);
    p.shl(L5, 23, L1);
    p.mov(L1, L3);
    p.isub_from(L2, L3);
    p.iadd_imm(L0, -1, L7);
    p.if_(Cond::Lt0(L7), |p| {
        p.loadi_bits(L4, 0x7f_ffff);
        p.and(L2, L4, L5);
        p.loadi_bits(L4, 1 << 23);
        p.or(L5, L4, L5);
        p.shl_by(L7, L5);
        p.mov(L5, L3);
    });
    p.if_(Cond::Eq0(L2), |p| p.mov(ZERO, L3));
    // The sign, and the NaNs.
    load_operands(p, o, variant, divisor);
    p.loadi_bits(L7, 0x8000_0000);
    if variant == Variant::NoCorrection {
        p.and(L0, L7, L4);
    } else {
        p.and(L1, L7, L4);
    }
    p.or(L3, L4, L3);
    p.loadi_bits(L4, NAN_BITS);
    p.loadi_bits(L7, 0x7fff_ffff);
    p.and(L0, L7, L5);
    p.and(L1, L7, L6);
    p.loadi_bits(L7, 0x7f80_0000);
    p.if_(Cond::LessEq(L7, L5), |p| p.mov(L4, L3));
    p.if_(Cond::LessEq(L7, L6), |p| p.mov(L4, L3));
    p.if_(Cond::Eq0(L6), |p| p.mov(L4, L3));
    // `a + b` overflowed.
    p.if_(Cond::LessEq(L7, L2), |p| p.mov(L4, L3));
    p.store(L3, Format::Int32, OUT_ROW + o);
}

/// `r = a - b * trunc(a / b)` at `a_at`, `b_at`, into `out_at` (all `Dst`
/// rows): the formula a developer writes first, with the division's own
/// rounding.
fn naive_mod(p: &mut Program, a_at: u32, b_at: u32, out_at: u32, spill: u32) {
    use super::{divide, scale_large_divisor};
    p.load(L0, Format::Fp32, a_at);
    p.load(L1, Format::Fp32, b_at);
    p.loadi_bits(L6, f32::MAX.to_bits());
    p.loadi_bits(L7, 0x7f80_0000);
    scale_large_divisor(p, L0, L1, L3, L4);
    p.recip(L1, L2, L3, L4, L6);
    divide(p, L0, L1, L2, L5, L3, L4, L7);
    p.store(L5, Format::Fp32, spill);
    p.load(L0, Format::Int32, spill);
    super::super::round::body(p, super::super::round::RoundOp::Trunc);
    p.load(L0, Format::Fp32, a_at);
    p.load(L1, Format::Fp32, b_at);
    p.nmad(L1, L2, L0, L3);
    p.store(L3, Format::Fp32, out_at);
}

/// The first wrong remainder: `naive(naive(a, b) + b, b)`.
fn naive_body(p: &mut Program, o: u32, divisor: Divisor) {
    let (r1, y, r2) = (SPILL_ROW + o, SPILL_ROW + 64 + o, SPILL_ROW + 128 + o);
    let q = SPILL_ROW + 192 + o;
    let b_at = match divisor {
        Divisor::Tile => B_ROW + o,
        Divisor::Scalar(s) => {
            p.loadi_bits(L0, s.to_bits());
            p.store(L0, Format::Int32, SPILL_ROW + 256 + o);
            SPILL_ROW + 256 + o
        }
    };
    naive_mod(p, A_ROW + o, b_at, r1, q);
    p.load(L0, Format::Fp32, r1);
    p.load(L1, Format::Fp32, b_at);
    p.add(L0, L1, L2);
    p.store(L2, Format::Fp32, y);
    naive_mod(p, y, b_at, r2, q);
    p.load(L0, Format::Int32, r2);
    p.store(L0, Format::Int32, OUT_ROW + o);
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::sfpu::{interp::Vector, kernel};

    fn flex(a: f32, b: f32) -> f32 {
        ((a % b) + b) % b
    }

    fn run(variant: Variant, a: &[u32], b: &[u32]) -> Vec<u32> {
        let code = code(variant, Divisor::Tile);
        let mut v = Vector::new();
        v.put_tile(kernel::A_ROW as usize, a);
        v.put_tile(kernel::B_ROW as usize, b);
        v.run(&code.expand()).unwrap();
        v.tile(kernel::OUT_ROW as usize)
    }

    #[test]
    fn the_program_is_flexs_remainder_on_the_edges() {
        let vals: Vec<u32> = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            4.0,
            -4.0,
            2.0,
            -2.0,
            3.0,
            1e30,
            -1e30,
            1e-20,
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x7f_ffff),
            f32::from_bits(0x8000_0001),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            3.0e38,
            3.3e38,
            0.5,
            0.99999994,
            f32::from_bits(0x0000_0400),
            f32::from_bits(0x1234_5678),
        ]
        .iter()
        .map(|v| v.to_bits())
        .collect();
        let n = vals.len();
        let a: Vec<u32> = (0..1024).map(|i| vals[(i / n) % n]).collect();
        let b: Vec<u32> = (0..1024).map(|i| vals[i % n]).collect();
        let got = run(Variant::Exact, &a, &b);
        for i in 0..1024 {
            let (x, y) = (f32::from_bits(a[i]), f32::from_bits(b[i]));
            let want = flex(x, y);
            if want.is_nan() {
                assert!(f32::from_bits(got[i]).is_nan(), "{x:e} % {y:e}");
            } else {
                assert_eq!(
                    got[i],
                    want.to_bits(),
                    "{x:e} % {y:e}: device {:e}, flex {want:e}",
                    f32::from_bits(got[i])
                );
            }
        }
    }

    /// Operands from every region the algorithm treats differently.
    fn corpus(seed: u64, n: usize) -> Vec<u32> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        (0..n)
            .map(|_| {
                let (x, k) = (next(), next() >> 40);
                let sign = (x as u32) & 0x8000_0000;
                let frac = (x as u32) & 0x7f_ffff;
                let e = |lo: u32, span: u32| (((x >> 40) as u32) % span + lo) << 23;
                match k % 6 {
                    0 => x as u32,
                    1 => sign | frac,
                    2 => sign | e(100, 40) | frac,
                    3 => sign | e(1, 254) | frac,
                    4 => sign | e(0, 12) | frac,
                    _ => sign | e(243, 12) | frac,
                }
            })
            .collect()
    }

    #[test]
    fn a_random_corpus_is_flexs_remainder_bit_for_bit() {
        for seed in 1..=24u64 {
            let a = corpus(seed, 1024);
            let b = corpus(seed + 1000, 1024);
            let got = run(Variant::Exact, &a, &b);
            for i in 0..1024 {
                let (x, y) = (f32::from_bits(a[i]), f32::from_bits(b[i]));
                let want = flex(x, y);
                if want.is_nan() {
                    assert!(f32::from_bits(got[i]).is_nan(), "{x:e} % {y:e}");
                } else {
                    assert_eq!(got[i], want.to_bits(), "{:08x} % {:08x}", a[i], b[i]);
                }
            }
        }
    }

    #[test]
    fn the_program_fits_a_role_slot() {
        for divisor in [Divisor::Tile, Divisor::Scalar(3.5)] {
            let c = code(Variant::Exact, divisor);
            let (words, _) = c.stored().unwrap();
            eprintln!(
                "{divisor:?}: {} stored words, {} loops {:?}, {} instructions per tile",
                words.len(),
                c.loops.len(),
                c.loops,
                c.expand().len()
            );
            assert!(words.len() <= 8192);
        }
    }
}

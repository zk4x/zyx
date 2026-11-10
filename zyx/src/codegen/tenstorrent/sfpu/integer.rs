//! Full-width two's-complement ALU programs. No integer operand passes
//! through floating-point arithmetic or a sign-magnitude conversion.
use super::{Cond, LReg, Program};

/// Tensor kinds 0x170..0x17d, scalar forms 0x180..0x18d, NOT 0x17e.
pub fn operation(kind: u32) -> Option<(u32, bool)> {
    match kind {
        0x170..=0x17e => Some((kind - 0x170, false)),
        0x180..=0x18d => Some((kind - 0x180, true)),
        0x18e => Some((15, false)),
        0x18f => Some((16, false)),
        0x1a0 => Some((15, true)),
        0x1a1 => Some((16, true)),
        _ => None,
    }
}

/// LReg::L2 = op(LReg::L0, LReg::L1), LReg::L3..LReg::L7 scratch. Counts wrap modulo 32.
pub fn body(p: &mut Program, op: u32) {
    match op {
        0 => {
            p.mov(LReg::L0, LReg::L2);
            p.iadd(LReg::L1, LReg::L2);
        }
        1 => {
            p.mov(LReg::L1, LReg::L2);
            p.isub_from(LReg::L0, LReg::L2);
        }
        2 => {
            // a*b mod 2^32 = alo*blo + ((ahi*blo + alo*bhi) << 16).
            // SFPMUL24 returns 23 bits, so reconstruct the full 16x16
            // low product from its LOW and UPPER modes (VC is constant zero).
            p.loadi_bits(LReg::L3, 0xffff);
            p.and(LReg::L0, LReg::L3, LReg::L4);
            p.and(LReg::L1, LReg::L3, LReg::L5);
            p.mul24(LReg::L4, LReg::L5, false, LReg::L2);
            p.mul24(LReg::L4, LReg::L5, true, LReg::L6);
            p.shl(LReg::L6, 23, LReg::L6);
            p.or(LReg::L6, LReg::L2, LReg::L2);
            p.loadi_bits(LReg::L3, (-16i32) as u32);
            p.shr_by(LReg::L3, LReg::L0);
            p.shr_by(LReg::L3, LReg::L1);
            p.mul24(LReg::L0, LReg::L5, false, LReg::L6);
            p.shl(LReg::L6, 16, LReg::L6);
            p.iadd(LReg::L6, LReg::L2);
            p.mul24(LReg::L1, LReg::L4, false, LReg::L6);
            p.shl(LReg::L6, 16, LReg::L6);
            p.iadd(LReg::L6, LReg::L2);
        }
        3 => p.and(LReg::L0, LReg::L1, LReg::L2),
        4 => p.or(LReg::L0, LReg::L1, LReg::L2),
        5 => {
            p.mov(LReg::L0, LReg::L2);
            p.xor(LReg::L1, LReg::L2);
        }
        6 | 7 => {
            p.loadi_bits(LReg::L3, 31);
            p.and(LReg::L1, LReg::L3, LReg::L1);
            p.mov(LReg::L0, LReg::L2);
            if op == 6 {
                p.shl_by(LReg::L1, LReg::L2);
            } else {
                p.isub_from(LReg::ZERO, LReg::L1);
                p.shr_by(LReg::L1, LReg::L2);
                p.if_(Cond::Lt0(LReg::L0), |p| {
                    p.not(LReg::ZERO, LReg::L3);
                    p.shr_by(LReg::L1, LReg::L3);
                    p.not(LReg::L3, LReg::L3);
                    p.or(LReg::L3, LReg::L2, LReg::L2);
                });
            }
        }
        8..=13 => {
            // Invert the magnitude field of negative two's-complement words
            // to embed signed integer order in SFPGT's sign-magnitude order.
            p.loadi_bits(LReg::L3, 0x7fffffff);
            p.if_(Cond::Lt0(LReg::L0), |p| p.xor(LReg::L3, LReg::L0));
            p.if_(Cond::Lt0(LReg::L1), |p| p.xor(LReg::L3, LReg::L1));
            p.loadi_bits(LReg::L4, 1);
            p.mov(LReg::ZERO, LReg::L2);
            match op {
                8 | 9 => {
                    p.xor(LReg::L0, LReg::L1);
                    let cond = if op == 8 { Cond::Eq0(LReg::L1) } else { Cond::Ne0(LReg::L1) };
                    p.if_(cond, |p| p.mov(LReg::L4, LReg::L2));
                }
                10 => p.if_(Cond::Less(LReg::L1, LReg::L0), |p| p.mov(LReg::L4, LReg::L2)),
                11 => p.if_(Cond::LessEq(LReg::L1, LReg::L0), |p| p.mov(LReg::L4, LReg::L2)),
                12 => p.if_(Cond::Less(LReg::L0, LReg::L1), |p| p.mov(LReg::L4, LReg::L2)),
                _ => p.if_(Cond::LessEq(LReg::L0, LReg::L1), |p| p.mov(LReg::L4, LReg::L2)),
            }
        }
        14 => p.not(LReg::L0, LReg::L2),
        15 | 16 => divide(p, op == 16),
        _ => unreachable!("integer op"),
    }
}

/// Signed division truncated toward zero and Python-style remainder. Zero
/// divisors are rejected by the writer using separately packed domain flags.
/// MIN/-1 wraps to MIN, matching the backend's I64 intermediate cast to I32.
fn divide(p: &mut Program, remainder: bool) {
    p.mov(LReg::L0, LReg::L3);
    p.mov(LReg::L1, LReg::L4);
    p.if_(Cond::Lt0(LReg::L0), |p| p.isub_from(LReg::ZERO, LReg::L0));
    p.if_(Cond::Lt0(LReg::L1), |p| p.isub_from(LReg::ZERO, LReg::L1));
    p.mov(LReg::ZERO, LReg::L2);
    for bit in (0..32).rev() {
        let step = |p: &mut Program| {
            p.shl(LReg::L1, bit, LReg::L5);
            p.mov(LReg::L5, LReg::L6);
            p.isub_from(LReg::L0, LReg::L6);
            let subtract = |p: &mut Program| {
                p.mov(LReg::L6, LReg::L0);
                p.loadi_bits(LReg::L7, 1u32 << bit);
                p.or(LReg::L7, LReg::L2, LReg::L2);
            };
            // Rem <= 2^31. A shifted divisor >= 2^31 can only fit when
            // equal; smaller divisors use a non-overflowing signed difference
            // except when Rem=2^31, which always fits such a divisor.
            p.if_else(
                Cond::Lt0(LReg::L5),
                |p| {
                    p.mov(LReg::L5, LReg::L7);
                    p.xor(LReg::L0, LReg::L7);
                    p.if_(Cond::Eq0(LReg::L7), subtract);
                },
                |p| {
                    p.if_else(Cond::Lt0(LReg::L0), subtract, |p| {
                        p.if_(Cond::Gte0(LReg::L6), subtract);
                    });
                },
            );
        };
        if bit == 0 {
            step(p);
        } else {
            // Discard no significant divisor bit while shifting left.
            p.mov(LReg::L1, LReg::L6);
            p.loadi_bits(LReg::L7, (-((32 - bit) as i32)) as u32);
            p.shr_by(LReg::L7, LReg::L6);
            p.if_(Cond::Eq0(LReg::L6), step);
        }
    }
    p.mov(LReg::L4, LReg::L6);
    p.xor(LReg::L3, LReg::L6);
    if remainder {
        p.if_(Cond::Lt0(LReg::L3), |p| p.isub_from(LReg::ZERO, LReg::L0));
        p.if_(Cond::Ne0(LReg::L0), |p| {
            p.if_(Cond::Lt0(LReg::L6), |p| p.iadd(LReg::L4, LReg::L0));
        });
        p.mov(LReg::L0, LReg::L2);
    } else {
        p.if_(Cond::Lt0(LReg::L6), |p| p.isub_from(LReg::ZERO, LReg::L2));
    }
}

#[cfg(any())]
mod tests {
    #[test]
    fn division_and_domain_flags_match_integer_oracle() {
        use crate::sfpu::{interp::Vector, kernel, ops};
        let cases = [i32::MIN, i32::MAX, -1, 0, 1, -3, 3, 0x40000000];
        let a: Vec<_> = (0..1024).map(|i| cases[i / cases.len() % cases.len()] as u32).collect();
        let b: Vec<_> = (0..1024).map(|i| cases[i % cases.len()] as u32).collect();
        for kind in [ops::kind_sfpu::INT_DIV, ops::kind_sfpu::INT_REM] {
            let (_, code) = ops::code2(kind, [0.0; 2]).unwrap();
            let mut v = Vector::new();
            v.put_tile(kernel::A_ROW as usize, &a);
            v.put_tile(kernel::B_ROW as usize, &b);
            v.run(&code.expand()).unwrap();
            assert_eq!(v.tile(kernel::C_ROW as usize), b.iter().map(|&b| u32::from(b != 0)).collect::<Vec<_>>());
            for (i, got) in v.tile(kernel::OUT_ROW as usize).into_iter().enumerate() {
                let (a, b) = (a[i] as i32 as i64, b[i] as i32 as i64);
                if b != 0 {
                    let want = if kind == ops::kind_sfpu::INT_DIV {
                        a / b
                    } else {
                        ((a % b) + b) % b
                    };
                    assert_eq!(got, want as u32, "kind {kind:x}, lane {i}, {a}/{b}");
                }
            }
        }
    }
}

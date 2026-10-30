//! Deterministic F32 rounding and saturating I32 conversion on raw bits.
//! SFPSTOCHRND is deliberately not used: its bounded sign-magnitude formats
//! and ties-away/round-to-zero bugs do not implement these contracts.
use super::{Cond, LReg, Program};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoundOp {
    Even,
    Floor,
    Ceil,
    Trunc,
}

/// Round L0, preserving nonfinite values and signed zero, into L2.
pub fn body(p: &mut Program, op: RoundOp) {
    use RoundOp::*;
    p.loadi_bits(LReg::L3, 0x7fffffff);
    p.and(LReg::L0, LReg::L3, LReg::L1);
    p.mov(LReg::L0, LReg::L2);
    p.loadi_bits(LReg::L3, 0x4b000000); // 2^23: every finite F32 is integral above.
    p.if_(Cond::Less(LReg::L1, LReg::L3), |p| {
        p.loadi_bits(LReg::L3, 0x3f800000);
        p.if_else(
            Cond::Less(LReg::L1, LReg::L3),
            |p| {
                p.loadi_bits(LReg::L4, 0x80000000);
                p.and(LReg::L0, LReg::L4, LReg::L2);
                match op {
                    Trunc => {}
                    Floor => p.if_(Cond::Lt0(LReg::L0), |p| {
                        p.if_(Cond::Ne0(LReg::L1), |p| p.loadi(LReg::L2, -1.0))
                    }),
                    Ceil => p.if_(Cond::Gte0(LReg::L0), |p| {
                        p.if_(Cond::Ne0(LReg::L1), |p| p.loadi(LReg::L2, 1.0))
                    }),
                    Even => {
                        p.loadi_bits(LReg::L4, 0x3f000000);
                        p.if_(Cond::Less(LReg::L4, LReg::L1), |p| {
                            p.loadi_bits(LReg::L4, 0x3f800000);
                            p.or(LReg::L4, LReg::L2, LReg::L2);
                        });
                    }
                }
            },
            |p| {
                p.exponent(LReg::L0, true, LReg::L3);
                p.loadi_bits(LReg::L4, 23);
                p.isub_from(LReg::L4, LReg::L3); // discarded mantissa bit count
                p.loadi_bits(LReg::L4, 1);
                p.shl_by(LReg::L3, LReg::L4); // one integer ULP in raw bits
                p.iadd_imm(LReg::L4, -1, LReg::L5);
                p.and(LReg::L0, LReg::L5, LReg::L6); // discarded bits
                p.not(LReg::L5, LReg::L5);
                p.and(LReg::L0, LReg::L5, LReg::L2);
                match op {
                    Trunc => {}
                    Floor => p.if_(Cond::Lt0(LReg::L0), |p| {
                        p.if_(Cond::Ne0(LReg::L6), |p| p.iadd(LReg::L4, LReg::L2))
                    }),
                    Ceil => p.if_(Cond::Gte0(LReg::L0), |p| {
                        p.if_(Cond::Ne0(LReg::L6), |p| p.iadd(LReg::L4, LReg::L2))
                    }),
                    Even => {
                        p.mov(LReg::L4, LReg::L5);
                        p.loadi_bits(LReg::L7, (-1i32) as u32);
                        p.shr_by(LReg::L7, LReg::L5); // halfway
                        p.if_else(
                            Cond::Less(LReg::L5, LReg::L6),
                            |p| p.iadd(LReg::L4, LReg::L2),
                            |p| {
                                p.if_(Cond::LessEq(LReg::L5, LReg::L6), |p| {
                                    p.and(LReg::L2, LReg::L4, LReg::L7);
                                    p.if_(Cond::Ne0(LReg::L7), |p| p.iadd(LReg::L4, LReg::L2));
                                });
                            },
                        );
                    }
                }
            },
        );
    });
}

/// Rust's saturating `f32 as i32`: truncate, NaN zero, infinities/overflow clamp.
pub fn to_i32(p: &mut Program) {
    p.loadi_bits(LReg::L3, 0x7fffffff);
    p.and(LReg::L0, LReg::L3, LReg::L1);
    p.mov(LReg::ZERO, LReg::L2);
    p.loadi_bits(LReg::L3, 0x3f800000);
    p.if_(Cond::LessEq(LReg::L3, LReg::L1), |p| {
        p.loadi_bits(LReg::L3, 0x4f000000); // 2^31
        p.if_else(
            Cond::Less(LReg::L1, LReg::L3),
            |p| {
                p.exponent(LReg::L0, true, LReg::L3);
                p.iadd_imm(LReg::L3, -23, LReg::L3);
                p.mantissa(LReg::L0, LReg::L2);
                p.shl_by(LReg::L3, LReg::L2);
                p.if_(Cond::Lt0(LReg::L0), |p| p.isub_from(LReg::ZERO, LReg::L2));
            },
            |p| {
                p.loadi_bits(LReg::L3, 0x7f800000);
                p.if_(Cond::LessEq(LReg::L1, LReg::L3), |p| {
                    p.if_else(
                        Cond::Lt0(LReg::L0),
                        |p| p.loadi_bits(LReg::L2, 0x80000000),
                        |p| p.loadi_bits(LReg::L2, 0x7fffffff),
                    );
                });
            },
        );
    });
}

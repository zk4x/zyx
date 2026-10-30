//! Element-wise ops as SFPU programs (`hardware-coverage.md` S1): the
//! `crate::kind` ops, bit for bit `burn-flex`'s, and `kind_sfpu`'s beyond them.
//!
//! Each program reads its operands from the kernel's rows
//! (`super::kernel::A_ROW`, `B_ROW`) and writes `OUT_ROW`. `SFPMAD` rounds to
//! nearest even and flushes denormals, as Flex's `f32` does but for those
//! (`fma_bh`, which `fma_oracle` holds to `fma.c`); the
//! integer tests of `RELU` are the same predicate in `SFPGT`'s total order.
//! `step19_eltwise` holds every kind to `burn-flex`.

use super::super::kind;
use tt_isa::isa::Instruction;

use super::kernel::{bias_row, Operands, A_ROW, B_ROW, OUT_ROW};
use super::{Cond, Format, LReg, Program};
use super::super::elem::Elem;

pub mod rem;

/// `+inf`'s bits plus one: the first positive NaN.
const FIRST_NAN: u32 = 0x7f80_0001;

/// The element-wise ops beyond `crate::kind`'s, numbered above them.
pub mod kind_sfpu {
    /// Explicit SFPSTOCHRND modes: BF16 nearest/stochastic/toward-zero,
    /// then TF32 nearest/stochastic/toward-zero. These preserve hardware bugs.
    pub const HARDWARE_ROUND: u32 = 0x1b0;
    /// `1 / a`, within one ulp of the correctly rounded reciprocal
    /// (`Program::recip`).
    pub const RECIP: u32 = 0x100;
    /// `a / b`, within one ulp of the correctly rounded quotient.
    pub const DIV: u32 = 0x101;
    /// `a / s`.
    pub const DIV_SCALAR: u32 = 0x102;
    /// `e^a` (`exp_program`).
    pub const EXP: u32 = 0x103;
    /// `ln a` (`log_program`).
    pub const LOG: u32 = 0x104;
    /// `!a` of a `Bool` tensor (`hardware-coverage.md` D3).
    pub const BOOL_NOT: u32 = 0x105;
    /// `a && b`, `Bool`.
    pub const BOOL_AND: u32 = 0x106;
    /// `a || b`, `Bool`.
    pub const BOOL_OR: u32 = 0x107;
    /// `a != b`, `Bool`.
    pub const BOOL_XOR: u32 = 0x108;
    /// `-a`: the sign bit flipped, NaNs included (S2; 10.2c).
    pub const NEG: u32 = 0x109;
    /// `|a|`: the sign bit cleared, NaNs included.
    pub const ABS: u32 = 0x10a;
    /// `1`, `-1`, `+0` for `a > 0`, `a < 0`, `±0`; a NaN itself.
    pub const SIGN: u32 = 0x10b;
    /// `a.clamp(scalar, scalar2)`, as `f32::clamp`: bounds neither NaN nor
    /// crossed (a program is refused otherwise, as `f32::clamp` panics).
    pub const CLAMP: u32 = 0x10c;
    /// `a.max(scalar)` as Flex's `float_clamp_min`: a NaN on either side gives
    /// the other, and equal values (`±0`) give the scalar.
    pub const CLAMP_MIN: u32 = 0x10d;
    /// `a.min(scalar)`, as [`CLAMP_MIN`] mirrored.
    pub const CLAMP_MAX: u32 = 0x10e;
    /// `a >= 0 ? a : scalar * a`.
    pub const LEAKY_RELU: u32 = 0x10f;
    /// `(scalar * a + scalar2).clamp(0, 1)`, two roundings.
    pub const HARD_SIGMOID: u32 = 0x110;
    /// IEEE comparisons, `F32` with `F32` to `Bool`: `-0 == +0`, a NaN
    /// unordered (every one false but `NE`).
    pub const EQ: u32 = 0x111;
    pub const NE: u32 = 0x112;
    pub const GT: u32 = 0x113;
    pub const GE: u32 = 0x114;
    pub const LT: u32 = 0x115;
    pub const LE: u32 = 0x116;
    /// The same against `scalar`.
    pub const EQ_S: u32 = 0x117;
    pub const NE_S: u32 = 0x118;
    pub const GT_S: u32 = 0x119;
    pub const GE_S: u32 = 0x11a;
    pub const LT_S: u32 = 0x11b;
    pub const LE_S: u32 = 0x11c;
    /// `F32` to `Bool`.
    pub const IS_NAN: u32 = 0x11d;
    pub const IS_INF: u32 = 0x11e;
    /// `mask ? scalar : a`, `a` `F32`, `mask` `Bool` (as `B`, which may be a
    /// row or a column).
    pub const MASK_FILL: u32 = 0x11f;
    /// `mask ? c : a`: `F32`, `Bool`, `F32`, all one shape (ternary).
    pub const MASK_WHERE: u32 = 0x120;
    /// `a >= 0 ? a : b * a`, Flex's `prelu`: `LEAKY_RELU` with the slope a
    /// tensor -- a row of per-channel slopes, broadcast.
    pub const PRELU: u32 = 0x121;
    /// `scalar` everywhere, exact (Flex's `ones` for `powi_scalar(x, 0)`).
    pub const FILL: u32 = 0x122;
    /// `sqrt(a)`, within one ulp of the correctly rounded root (S4, 10.2d).
    pub const SQRT: u32 = 0x123;
    /// `1 / sqrt(a)`, within [`super::RSQRT_BOUND`].
    pub const RSQRT: u32 = 0x124;
    /// `ln(1 + a)`, within [`super::LOG1P_BOUND`].
    pub const LOG1P: u32 = 0x125;
    /// `a^b` as `powf` (`super::pow_program`), within [`super::pow_bound`].
    pub const POW: u32 = 0x126;
    /// `a^s`, `s` the scalar.
    pub const POW_S: u32 = 0x127;
    /// `a^b`, `b` an `I32` tensor (`as f32` first, as Flex's `powi`).
    pub const POW_I: u32 = 0x128;
    /// An `I32` as `F32`, `as f32`'s rounding (Flex's `int_into_float`).
    pub const I32_TO_F32: u32 = 0x129;
    /// Exact nonnegative integral F32 index below 2^23 to I32 bits.
    pub const INDEX_TO_I32: u32 = 0x160;
    // Integer and boolean selects.
    // Every one moves raw 32-bit words (`Format::Int32` loads and stores): no
    // float conversion ever sees a datum, so denormals and NaN payloads survive.
    /// `mask ? c : a`: `I32`, `Bool`, `I32`, all one shape (ternary).
    pub const INT_MASK_WHERE: u32 = 0x161;
    /// `mask ? c : a`: `Bool`, `Bool`, `Bool`, all one shape (ternary).
    pub const BOOL_MASK_WHERE: u32 = 0x162;
    /// `mask ? scalar : a`, `a` `I32` and `scalar` the raw 32-bit pattern
    /// (`f32::from_bits`; never a float conversion). `mask` as `MASK_FILL`'s.
    pub const INT_MASK_FILL: u32 = 0x163;
    /// `mask ? scalar : a`, `a` `Bool` and `scalar`'s bits `0` or `1`.
    pub const BOOL_MASK_FILL: u32 = 0x164;
    /// `|a|` of an `I32`, wrapping: `i32::MIN` stays `i32::MIN` (Flex's
    /// `wrapping_abs`).
    pub const INT_ABS: u32 = 0x165;
    /// `e^a - 1`, within [`super::EXPM1_BOUND`] (10.2e).
    pub const EXPM1: u32 = 0x12a;
    /// `1 / (1 + e^-a)` in Flex's two branches, within [`super::SIGMOID_BOUND`].
    pub const SIGMOID: u32 = 0x12b;
    /// `g * s * (1 - s)`, `s` the sigmoid's output (`A`) and `g` the gradient
    /// (`B`), exact as Flex's order.
    pub const SIGMOID_BACKWARD: u32 = 0x12c;
    /// `tanh a`, within [`super::TANH_BOUND`].
    pub const TANH: u32 = 0x12d;
    /// `erf a`, within [`super::ERF_BOUND`].
    pub const ERF: u32 = 0x12e;
    /// `0.5 a (1 + erf(a/sqrt 2))` (Flex's `gelu`), within
    /// [`super::GELU_BOUND`].
    pub const GELU: u32 = 0x12f;
    /// `g (Phi(a) + a phi(a))` from `a` (`A`) and the gradient `g` (`B`), within
    /// [`super::gelu_backward_bound`].
    pub const GELU_BACKWARD: u32 = 0x130;
    /// `sinh a`, within [`super::SINH_BOUND`] (`super::sinh_cosh_program`).
    pub const SINH: u32 = 0x131;
    /// `cosh a`, within [`super::COSH_BOUND`].
    pub const COSH: u32 = 0x132;
    /// `asinh a`, within [`super::ASINH_BOUND`] (`super::asinh_acosh_program`).
    pub const ASINH: u32 = 0x133;
    /// `acosh a`, within [`super::ACOSH_BOUND`].
    pub const ACOSH: u32 = 0x134;
    /// `atanh a`, within [`super::ATANH_BOUND`] (`super::atanh_program`).
    pub const ATANH: u32 = 0x135;
    /// `ln(sigmoid a)` (Flex's `log_sigmoid`), within
    /// [`super::LOG_SIGMOID_BOUND`] (`super::log_sigmoid_program`).
    pub const LOG_SIGMOID: u32 = 0x136;
    /// `g sigmoid(-a)` from `a` (`A`) and the gradient `g` (`B`), Flex's
    /// `log_sigmoid_backward`, within [`super::SIGMOID_BOUND`] and the
    /// product's rounding.
    pub const LOG_SIGMOID_BACKWARD: u32 = 0x137;
    /// `sin a`, within [`super::SIN_BOUND`] for every finite `a`
    /// (`super::sin_cos_program`; 10.2f).
    pub const SIN: u32 = 0x138;
    /// `cos a`, within [`super::COS_BOUND`].
    pub const COS: u32 = 0x139;
    /// `tan a`, within [`super::TAN_BOUND`] for every finite `a`
    /// (`super::tan_program`).
    pub const TAN: u32 = 0x13a;
    /// `atan a`, within [`super::ATAN_BOUND`] (`super::atan_program`).
    pub const ATAN: u32 = 0x13b;
    /// `atan2(a, b)` as `f32::atan2` (`a` the `y`), within
    /// [`super::ATAN2_BOUND`] (`super::atan2_program`).
    pub const ATAN2: u32 = 0x13c;
    /// `asin a`, within [`super::ASIN_BOUND`] (`super::asin_acos_program`).
    pub const ASIN: u32 = 0x13d;
    /// `acos a`, within [`super::ACOS_BOUND`].
    pub const ACOS: u32 = 0x13e;
    /// Boolean integer 0/1 to F32, exact.
    pub const BOOL_TO_F32: u32 = 0x13f;
    /// Boolean integer 0/1 copied to a typed I32 buffer, exact.
    pub const BOOL_TO_I32: u32 = 0x140;
    /// `((a % b) + b) % b` in `f32`, `burn-flex`'s `float_remainder`, bit for
    /// bit (`super::rem`).
    pub const REM: u32 = 0x141;
    /// [`REM`] with a scalar `b` (`float_remainder_scalar`).
    pub const REM_S: u32 = 0x142;
    pub const ROUND: u32 = 0x190;
    pub const FLOOR: u32 = 0x191;
    pub const CEIL: u32 = 0x192;
    pub const TRUNC: u32 = 0x193;
    pub const F32_TO_I32: u32 = 0x194;
    /// `2^floor(log2 |a|)`: the exponent field alone (`a & 0x7f800000`), exact. Zero and
    /// denormals give `+0`; infinities and NaNs give `+inf`.
    pub const POW2_FLOOR: u32 = 0x195;
    /// `2^-floor(log2 |a|)`: the exponent field `254 - e` (`0x7f000000 - (a & 0x7f800000)`), exact,
    /// so `a * POW2_FLOOR_INV(a)` is in `[1, 2)`. Zero and denormals (`e = 0`) give `2^127`;
    /// `e >= 254` (`|a| >= 2^127`, whose inverse is not a normal number, and infinities and
    /// NaNs) saturate to `2^-126`.
    pub const POW2_FLOOR_INV: u32 = 0x196;
    // I32 operations; scalar immediates carry raw I32 bits.
    pub const INT_ADD: u32 = 0x170;
    pub const INT_SUB: u32 = 0x171;
    pub const INT_MUL: u32 = 0x172;
    pub const INT_AND: u32 = 0x173;
    pub const INT_OR: u32 = 0x174;
    pub const INT_XOR: u32 = 0x175;
    pub const INT_SHL: u32 = 0x176;
    pub const INT_SHR: u32 = 0x177;
    pub const INT_EQ: u32 = 0x178;
    pub const INT_NE: u32 = 0x179;
    pub const INT_GT: u32 = 0x17a;
    pub const INT_GE: u32 = 0x17b;
    pub const INT_LT: u32 = 0x17c;
    pub const INT_LE: u32 = 0x17d;
    pub const INT_NOT: u32 = 0x17e;
    pub const INT_DIV: u32 = 0x18e;
    pub const INT_REM: u32 = 0x18f;
    pub const INT_DIV_S: u32 = 0x1a0;
    pub const INT_REM_S: u32 = 0x1a1;
    pub const INT_ADD_S: u32 = 0x180;
    pub const INT_SUB_S: u32 = 0x181;
    pub const INT_MUL_S: u32 = 0x182;
    pub const INT_AND_S: u32 = 0x183;
    pub const INT_OR_S: u32 = 0x184;
    pub const INT_XOR_S: u32 = 0x185;
    pub const INT_SHL_S: u32 = 0x186;
    pub const INT_SHR_S: u32 = 0x187;
    pub const INT_EQ_S: u32 = 0x188;
    pub const INT_NE_S: u32 = 0x189;
    pub const INT_GT_S: u32 = 0x18a;
    pub const INT_GE_S: u32 = 0x18b;
    pub const INT_LT_S: u32 = 0x18c;
    pub const INT_LE_S: u32 = 0x18d;
    /// `relu(a + b)`: compound fused addition and ReLU in one SFPU pass.
    pub const ADD_RELU: u32 = 0x1f0;
    /// End of the original contiguous activation range. Additional kinds
    /// occupy the documented integer, rounding and index ranges below.
    pub const LAST: u32 = BOOL_TO_I32;
}

/// The IEEE comparisons, tensor with tensor, with their scalar forms.
const COMPARES: [(u32, u32); 6] = [
    (kind_sfpu::EQ, kind_sfpu::EQ_S),
    (kind_sfpu::NE, kind_sfpu::NE_S),
    (kind_sfpu::GT, kind_sfpu::GT_S),
    (kind_sfpu::GE, kind_sfpu::GE_S),
    (kind_sfpu::LT, kind_sfpu::LT_S),
    (kind_sfpu::LE, kind_sfpu::LE_S),
];

/// `kind`'s tensor-with-tensor comparison, if it is one of either form.
fn compare_of(kind: u32) -> Option<u32> {
    COMPARES
        .iter()
        .find(|(t, s)| *t == kind || *s == kind)
        .map(|(t, _)| *t)
}

/// What `x (cmp) y` is for two FP32 values by IEEE, as Flex compares.
pub fn ieee_compare(cmp: u32, x: f32, y: f32) -> bool {
    match compare_of(cmp).expect("a comparison") {
        kind_sfpu::EQ => x == y,
        kind_sfpu::NE => x != y,
        kind_sfpu::GT => x > y,
        kind_sfpu::GE => x >= y,
        kind_sfpu::LT => x < y,
        _ => x <= y,
    }
}

/// An op's element types: one per operand, and its output's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Sig {
    pub inputs: &'static [Elem],
    pub out: Elem,
}

/// What `kind` computes on and what it produces (`crate::tensor::Elem`). The
/// session refuses any other operand (`TensorError::Elem`).
pub fn elems(kind: u32) -> Sig {
    use Elem::{Bool, F32};
    let sig = |inputs, out| Sig { inputs, out };
    // The Approx programs (`approx.rs`) are unary, `F32` to `F32`.
    if super::approx::is_approx(kind) {
        return sig(&[F32], F32);
    }
    if (kind_sfpu::HARDWARE_ROUND..kind_sfpu::HARDWARE_ROUND + 6).contains(&kind) {
        return sig(&[F32], F32);
    }
    if let Some((op, scalar)) = super::integer::operation(kind) {
        return sig(
            if scalar || op == 14 {
                &[Elem::I32]
            } else {
                &[Elem::I32, Elem::I32]
            },
            if (8..=13).contains(&op) {
                Bool
            } else {
                Elem::I32
            },
        );
    }
    match kind {
        kind_sfpu::BOOL_NOT => sig(&[Bool], Bool),
        kind_sfpu::BOOL_AND | kind_sfpu::BOOL_OR | kind_sfpu::BOOL_XOR => sig(&[Bool, Bool], Bool),
        kind_sfpu::EQ..=kind_sfpu::LE => sig(&[F32, F32], Bool),
        kind_sfpu::EQ_S..=kind_sfpu::IS_INF => sig(&[F32], Bool),
        kind_sfpu::MASK_FILL => sig(&[F32, Bool], F32),
        kind_sfpu::MASK_WHERE => sig(&[F32, Bool, F32], F32),
        kind_sfpu::INT_MASK_FILL => sig(&[Elem::I32, Bool], Elem::I32),
        kind_sfpu::INT_MASK_WHERE => sig(&[Elem::I32, Bool, Elem::I32], Elem::I32),
        kind_sfpu::BOOL_MASK_FILL => sig(&[Bool, Bool], Bool),
        kind_sfpu::BOOL_MASK_WHERE => sig(&[Bool, Bool, Bool], Bool),
        kind_sfpu::INT_ABS => sig(&[Elem::I32], Elem::I32),
        kind_sfpu::POW_I => sig(&[F32, Elem::I32], F32),
        kind_sfpu::I32_TO_F32 => sig(&[Elem::I32], F32),
        kind_sfpu::BOOL_TO_F32 => sig(&[Bool], F32),
        kind_sfpu::BOOL_TO_I32 => sig(&[Bool], Elem::I32),
        kind_sfpu::INDEX_TO_I32 | kind_sfpu::F32_TO_I32 => sig(&[F32], Elem::I32),
        kind_sfpu::ROUND..=kind_sfpu::TRUNC => sig(&[F32], F32),
        kind_sfpu::POW2_FLOOR | kind_sfpu::POW2_FLOOR_INV => sig(&[F32], F32),
        _ => sig(&[F32, F32], F32),
    }
}

/// Whether `kind` gives the host's bits exactly or an approximation held to a
/// derived bound -- which decides whether burn-tt's exact mode may run it on
/// the device (`burn-backend-parity.md` 4.3a).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Accuracy {
    /// Flex's bits, every input.
    Exact,
    /// Within a derived bound: `Program::recip`'s ulp, `EXP_BOUND`, ...
    Approximate,
}

/// [`Accuracy`] of `kind`.
pub fn accuracy(kind: u32) -> Accuracy {
    // The Approx programs are approximations, held to their bounds.
    if super::approx::is_approx(kind) {
        return Accuracy::Approximate;
    }
    match kind {
        kind_sfpu::RECIP
        | kind_sfpu::DIV
        | kind_sfpu::DIV_SCALAR
        | kind_sfpu::EXP
        | kind_sfpu::LOG
        | kind_sfpu::SQRT..=kind_sfpu::POW_I
        | kind_sfpu::EXPM1
        | kind_sfpu::SIGMOID
        | kind_sfpu::TANH
        | kind_sfpu::ERF
        | kind_sfpu::GELU
        | kind_sfpu::GELU_BACKWARD
        | kind_sfpu::SINH..=kind_sfpu::LOG_SIGMOID_BACKWARD
        | kind_sfpu::SIN..=kind_sfpu::ACOS => Accuracy::Approximate,
        _ => Accuracy::Exact,
    }
}

/// `e^x` of `x` into `d`, every register but `x` and `d` scratch.
///
/// `n = round(x log2 e)` by the magic-number rounding (adding `1.5 * 2^23`
/// leaves `n` in the low mantissa bits, round to nearest even), `r = x - n ln 2`
/// by Cody and Waite's two-part `ln 2` (the high part has nine significant
/// bits, so `n * ln2_hi` is exact for `|n| <= 128`), `|r| <= ln 2 / 2`; then
/// `e^r` by its degree-7 Taylor polynomial in Horner form, and `2^n` added to
/// the exponent field as an integer.
///
/// Error, as a fraction of `e^x`: the polynomial's remainder `r^8/8! * e^|r|`
/// is below `7.4e-9`; Horner's seven roundings, with `|r| <= 0.347` damping
/// all but the last two, contribute below `1.7 * 2^-24` (`1.0e-7`); the
/// reduction's rounding of `r` (one rounding of the low product, `|n ln2_lo|
/// < 2.8e-2`) moves `r` by under `2^-24 * |r| + 2^-30`, which `e^r` carries
/// one for one (`2.3e-8`). In all under [`EXP_BOUND`] `= 1.3e-7` of `e^x`
/// (about two ulps where the result's mantissa is near 2, one near 1).
/// Outside `[-87.34, 88.72]` the result is `+0`
/// (it would be denormal, which flushes) or `+inf`; a NaN stays one.
pub fn exp_program(p: &mut Program, x: LReg, d: LReg) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (k, magic, t, nf, r, c) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    p.loadi(k, std::f32::consts::LOG2_E);
    p.loadi_bits(magic, 0x4b40_0000);
    p.mad(x, k, magic, t);
    p.sub(t, magic, nf);
    p.loadi_bits(k, 0x3f31_8000); // ln2_hi = 0.693359375
    p.nmad(nf, k, x, r);
    p.loadi_bits(k, 0xb95e_8083); // ln2_lo = -2.12194440e-4
    p.nmad(nf, k, r, r);
    // e^r = 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r(1/120 + r(1/720 + r/5040))))))
    p.loadi(d, 1.0 / 5040.0);
    for coeff in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5] {
        p.loadi(c, coeff);
        p.mad(d, r, c, d);
    }
    p.mad(d, r, LReg::ONE, d);
    p.mad(d, r, LReg::ONE, d);
    // `n = bits(t) - bits(magic)`, into the exponent field.
    p.isub_from(t, magic);
    p.shl(magic, 23, magic);
    p.iadd(magic, d);
    // The range: below `ln 2^-126` a zero, above `ln f32::MAX` infinity
    // (which a positive NaN also passes), and a NaN restored.
    p.loadi_bits(k, 0xc2ae_ac50); // -87.336544
    p.if_(Cond::Less(x, k), |p| p.mov(LReg::ZERO, d));
    // `0x42b1_7218` (88.72284) is the first float above `ln f32::MAX`: from it
    // on, `n = 128` would carry the exponent field into 255 -- an infinity's,
    // or with mantissa bits a NaN's. (It was `>` alone until 10.2d's `pow`
    // met `e^(ln f32::MAX)`, whose `z` rounds to exactly this.)
    p.loadi_bits(k, 0x42b1_7218); // 88.72284
    p.if_(Cond::LessEq(k, x), |p| p.loadi_bits(d, 0x7f80_0000));
    // The magnitude by mask: `SFPABS` leaves a negative NaN negative, and
    // `e^-NaN` came out `0` (found in 10.2e).
    p.loadi_bits(k, 0x7fff_ffff);
    p.and(x, k, t);
    p.loadi_bits(k, 0x7f80_0000);
    p.if_(Cond::Less(k, t), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`exp_program`]'s derived bound, relative to `e^x`.
pub const EXP_BOUND: f64 = 1.3e-7;

/// [`log_program`]'s derived bound, relative to `ln x`.
pub const LOG_BOUND: f64 = 7.12 / 16_777_216.0;

/// `ln x` of `x` into `d`, every register but `x` and `d` scratch.
///
/// `x = 2^e m` with `m` in `[sqrt(2)/2, sqrt(2))` (`SFPEXEXP`, `SFPSETEXP`, and
/// a halving where `m` comes out above `sqrt 2`); `f = m - 1` exactly
/// (Sterbenz); `ln m = 2 atanh(s)`, `s = f / (2 + f)`, `|s| <= 0.1716`, by
/// `2s(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9)` -- the next term, `s^10/11`, is
/// below `2.0e-9` of the sum -- and `ln x = e ln2_hi + (e ln2_lo + ln m)`.
///
/// Error, relative, in units of `u = 2^-24`: `fl(2 + f)` is within `u` of
/// `2 + f`, and `s` within 1.5 ulps (`3u`) of `f / fl(2 + f)` (`divide`), so
/// `4u` of `f/(2+f)`; `ln m` is `2s (1 + O(s^2))` and carries that one for one
/// (`4.12u` with the `s^2 < 0.03` term), the series' own roundings damped by
/// `s^2` to under `0.04u`, plus `u` for the fma that adds them; the two final
/// fmas round once each, and `e ln 2` and `ln m` never cancel to less than
/// `|ln m|` (they share a sign unless `|e| = 1`, where `|ln x| >= 0.34 >=
/// |ln m|`). Under [`LOG_BOUND`] `= 7.12u` (`4.3e-7`) of `ln x` in all. `ln(±0)` (and
/// a denormal, which flushes) is `-inf`, `ln` of a negative number NaN,
/// `ln(+inf) = +inf`, a NaN stays one.
pub fn log_program(p: &mut Program, x: LReg, d: LReg) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (e, m, f, den, y, q) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    p.exponent(x, true, e);
    p.set_exponent(x, 127, m);
    // Positive: `SFPSETEXP` keeps the sign, and a negative `x` is a NaN below.
    p.set_sign(m, false, m);
    p.loadi_bits(f, 0x3fb5_04f3); // sqrt(2)
    p.if_(Cond::Less(f, m), |p| {
        p.set_exponent(m, 126, m);
        p.iadd_imm(e, 1, e);
    });
    p.sub(m, LReg::ONE, f);
    p.loadi(den, 2.0);
    p.add(f, den, den);
    // `s = f / (2 + f)`: `y = 1/den` uses `m` and `q` as scratch, `d` holds
    // `f32::MAX`; then the quotient into `q`, `m` and `d` scratch, `inf` in
    // `den` once it is spent.
    p.loadi_bits(d, f32::MAX.to_bits());
    p.recip(den, y, m, q, d);
    p.mul(f, y, q);
    p.abs(q, m);
    p.if_(Cond::Less(LReg::ZERO, m), |p| {
        p.nmad(den, q, f, m);
        p.mad(m, y, q, q);
    });
    // `s` in `q`; `s^2` into `f`; the series into `d`.
    p.mul(q, q, f);
    p.loadi(d, 1.0 / 9.0);
    for coeff in [1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0] {
        p.loadi(m, coeff);
        p.mad(d, f, m, d);
    }
    // ln m = 2(s + s * (s^2 * series)).
    p.mul(d, f, d);
    p.mad(q, d, q, d);
    p.add(d, d, d);
    // `e` as a float: `bits(1.5 * 2^23) + e` is `1.5 * 2^23 + e` exactly.
    p.loadi_bits(m, 0x4b40_0000);
    p.iadd(m, e);
    p.sub(e, m, e);
    p.loadi_bits(m, 0xb95e_8083); // ln2_lo
    p.mad(e, m, d, d);
    p.loadi_bits(m, 0x3f31_8000); // ln2_hi
    p.mad(e, m, d, d);
    // The special cases, each its own lanes (the total order of `SFPGT`).
    p.abs(x, m);
    // A zero or a denormal (which the arithmetic flushes): `-inf`.
    p.loadi_bits(f, 0x0080_0000);
    p.if_(Cond::Less(m, f), |p| p.loadi_bits(d, 0xff80_0000));
    // Anything else negative -- below minus the largest denormal in the total
    // order, `-inf` and negative NaNs included (whose `SFPABS` stays negative
    // and so took the branch above first): NaN.
    p.loadi_bits(f, 0x807f_ffff);
    p.if_(Cond::Less(x, f), |p| p.loadi_bits(d, 0x7fc0_0000));
    // `+inf`: `+inf`; a positive NaN: NaN.
    p.loadi_bits(f, f32::MAX.to_bits());
    p.if_(Cond::Less(f, x), |p| p.loadi_bits(d, 0x7f80_0000));
    p.loadi_bits(f, 0x7f80_0000);
    p.if_(Cond::Less(f, x), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`sqrt_program`]'s bound for the reciprocal root, relative.
pub const RSQRT_BOUND: f64 = 4.1 / 16_777_216.0;

/// `sqrt x` (or, `inverse`, `1/sqrt x`) of `x` into `d`; `x` kept, every
/// other register scratch.
///
/// The seed is the bit trick `0x5f3759df - bits(x) / 2` (within `e0 <=
/// 0.0344` of `1/sqrt x` for every positive normal `x`); three Newton steps
/// `y += y/2 (1 - x y^2)`, each taking `e` to `1.5 e^2` plus its four
/// roundings: `e1 <= 1.78e-3`, `e2 <= 4.8e-6`, `e3 <= 3.5e-11 + 4u` (`u =
/// 2^-24`) -- `x y` is formed before `y` multiplies it again, so `x y^2` never
/// leaves the normal range. So `1/sqrt x` within [`RSQRT_BOUND`] `= 4.1u`. The
/// root is `s = x y`, then one correction `s += (y/2)(x - s^2)` with the
/// residual in one rounding (`SFPMAD`): `s` within `5.1u` before it, so within
/// half an ulp plus `O(25u^2)` after -- at most one ulp from the correctly
/// rounded root, as `Program::recip` is from its reciprocal.
///
/// `sqrt(±0) = ±0`, `sqrt(+inf) = +inf`, anything negative (a negative
/// denormal included, as Flex has it) NaN, a NaN stays one; a positive
/// denormal flushes, so its root is `+0` (numerics row D). `1/sqrt`: `±inf`
/// for `±0`, `+0` for `+inf`, NaN as `sqrt`.
pub fn sqrt_program(p: &mut Program, x: LReg, d: LReg, inverse: bool) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (y, t, h, c) = (regs[0], regs[1], regs[2], regs[3]);
    // The seed: `magic - (bits(x) >> 1)`.
    p.mov(x, y);
    p.loadi_bits(c, (-1i32) as u32);
    p.shr_by(c, y);
    p.loadi_bits(c, 0x5f37_59df);
    p.isub_from(c, y);
    p.loadi(c, 0.5);
    for _ in 0..3 {
        // `h = x y`, `t = 1 - h y`, `y += (y/2) t`.
        p.mul(x, y, h);
        p.nmad(h, y, LReg::ONE, t);
        p.mul(y, c, h);
        p.mad(h, t, y, y);
    }
    if inverse {
        p.mov(y, d);
    } else {
        // `s = x y`; `s += (y/2)(x - s^2)`.
        p.mul(x, y, d);
        p.nmad(d, d, x, t);
        p.mul(y, c, h);
        p.mad(h, t, d, d);
    }
    // The special lanes, in the total order: `±0` and a positive denormal
    // (`|x| < 2^-126`), `+inf`, then everything negative but `-0`, and NaNs.
    p.abs(x, t);
    p.loadi_bits(c, 0x0080_0000);
    p.if_(Cond::Less(t, c), |p| {
        if inverse {
            p.loadi_bits(h, 0x7f80_0000);
            p.copy_sign(h, x, d);
        } else {
            // `±0` (a positive denormal flushes to `+0`).
            p.loadi_bits(h, 0x8000_0000);
            p.and(x, h, d);
        }
    });
    p.loadi_bits(c, 0x7f80_0000);
    p.if_(Cond::LessEq(c, x), |p| {
        if inverse {
            p.mov(LReg::ZERO, d);
        } else {
            p.mov(c, d);
        }
    });
    // Negative and not `-0`: below `-0` in the total order; a NaN above
    // `+inf` or below `-inf`.
    p.loadi_bits(c, 0x8000_0000);
    p.if_(Cond::Less(x, c), |p| p.loadi_bits(d, 0x7fc0_0000));
    p.loadi_bits(c, 0x7f80_0000);
    p.if_(Cond::Less(c, x), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`log1p_program`]'s derived bound, relative to `ln(1 + x)`.
pub const LOG1P_BOUND: f64 = 11.12 / 16_777_216.0;

/// `ln(1 + x)` of `x` into `d`, `x` read as raw bits and spilled to `Dst` at
/// `spill` (it outlives [`log_program`], which takes every register).
///
/// Kahan's: `u = fl(1 + x)`; where `u = 1` (`|x| <= 2^-24`, where
/// `ln(1 + x) = x (1 - x/2 + ...)` is `x` to under `2^-25` relative) the
/// result is `x` itself, bits and all; elsewhere `ln(u) * x / (u - 1)`. The
/// quotient `g(u) = ln(u)/(u - 1)` varies slowly enough that evaluating it at
/// `u` instead of `1 + x` costs at most `|g'/g| |u - (1 + x)| <= u` (`u =
/// 2^-24`, the rounding of `1 + x` being at most half an ulp of `u`), and `u -
/// 1` is exact (Sterbenz below 2; `1` a multiple of `u`'s ulp up to `2^24`).
/// Error, in units of `u`: `ln u` within `7.12` (`LOG_BOUND`), the quotient
/// `x / (u - 1)` within `2` (`divide`, one ulp), the product `1`, the
/// evaluation point `1`: under [`LOG1P_BOUND`] `= 11.12u` (`6.6e-7`). From `x =
/// 2^24` the result is `ln u` alone (`u = x`, and `ln(1 + x)` is `ln x` to
/// within `1/(x ln x) < 2^-27` relative).
/// `log1p(-1) = -inf`, below `-1` NaN, `log1p(+inf) = +inf`, a NaN stays one.
pub fn log1p_program(p: &mut Program, x: LReg, d: LReg, spill: u32) {
    use LReg as R;
    assert!(
        x == R::L0 && d == R::L7,
        "log1p_program's registers are fixed"
    );
    p.store(x, Format::Int32, spill);
    let u = R::L1;
    p.add(x, LReg::ONE, u);
    log_program(p, u, d);
    p.load(x, Format::Int32, spill);
    // `den = u - 1`; `q = x / den`; `d = ln(u) q`.
    let (den, y, q) = (R::L2, R::L3, R::L4);
    p.sub(u, LReg::ONE, den);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(den, y, R::L4, R::L5, R::L6);
    p.loadi_bits(R::L6, 0x7f80_0000);
    divide(p, x, den, y, q, R::L5, u, R::L6);
    // `ln u` in `d` survives: `divide` and `recip` use none of it. From
    // `x = 2^24` on, `u = x` and `ln(1 + x) = ln(x) (1 + O(1/x))`: `ln u` as it
    // is -- the quotient, `x / x`, would need `1/x`, which flushes from `2^126`.
    p.loadi_bits(y, 0x4b80_0000); // 2^24
    p.if_(Cond::Less(x, y), |p| p.mul(d, q, d));
    // Where `u = 1`: `x`. `u - 1` is then exactly `+0`.
    p.if_(Cond::Eq0(den), |p| p.mov(x, d));
    // `+inf`: `+inf` (the quotient `inf/inf` would be NaN).
    p.loadi_bits(y, 0x7f80_0000);
    p.if_(Cond::LessEq(y, x), |p| {
        p.if_(Cond::LessEq(x, y), |p| p.mov(y, d))
    });
}

/// [`expm1_program`]'s derived bound, relative to `e^x - 1`.
pub const EXPM1_BOUND: f64 = 4.5 / 16_777_216.0;

/// `e^x - 1` of `x` into `d`, every register but `x` and `d` scratch: accurate
/// near zero, where `e^x` then a subtraction would cancel.
///
/// The reduction is `exp_program`'s (`n = round(x log2 e)`, `r = x - n ln 2`
/// by Cody and Waite, `|r| <= ln2/2`); `p = e^r - 1 = r + r^2 q(r)`, `q` the
/// Taylor series of `(e^r - 1 - r)/r^2` to `r^6/8!` in Horner form (remainder
/// below `5.7e-10` of `p`); then `e^x - 1 = 2 (h p + (h - 1/2))` with `h =
/// 2^(n-1)`, one rounding in the fma, the doubling exact -- `h` rather than
/// `2^n` so that `n = 128` (`x` from 88.38) does not overflow, and `h - 1/2`
/// exact up to `n = 24`, beyond which the result is `e^x` to `2^-24` anyway.
///
/// Error, relative, in `u = 2^-24`: at `n = 0`, `r = x` exactly and the result
/// is `p` alone (`q`'s roundings damped by `r^2/2 / r <= 0.17`, `r^2`'s by the
/// same: `0.42u`, the fma `u`). Elsewhere `r` carries the reduction's rounding,
/// `2^-24 |r| + 2^-30 <= 2.2e-8` absolute, which the result amplifies by at
/// most `2^n e^r / |e^x - 1| <= 6.83` (at `n = 1`): `2.47u`; `p`'s error
/// enters as `2^n |p| / |e^x - 1| <= 2` times `0.42u`, and the fma rounds once:
/// under [`EXPM1_BOUND`] `= 4.5u` in all. Below `-18` the result is `-1`
/// (`e^-18 < 2^-25`), from 88.72284 `+inf`; a NaN stays one.
pub fn expm1_program(p: &mut Program, x: LReg, d: LReg) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (k, magic, t, nf, r, c) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    p.loadi(k, std::f32::consts::LOG2_E);
    p.loadi_bits(magic, 0x4b40_0000);
    p.mad(x, k, magic, t);
    p.sub(t, magic, nf);
    p.loadi_bits(k, 0x3f31_8000); // ln2_hi
    p.nmad(nf, k, x, r);
    p.loadi_bits(k, 0xb95e_8083); // ln2_lo
    p.nmad(nf, k, r, r);
    // q = 1/2 + r(1/6 + r(1/24 + r(1/120 + r(1/720 + r(1/5040 + r/40320)))))
    p.loadi(d, 1.0 / 40320.0);
    for coeff in [
        1.0 / 5040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
    ] {
        p.loadi(c, coeff);
        p.mad(d, r, c, d);
    }
    // p = r + r^2 q, into `d`.
    p.mul(r, r, nf);
    p.mad(nf, d, r, d);
    // h = 2^(n-1): `n = bits(t) - bits(magic)`, into the exponent of 0.5.
    p.isub_from(t, magic);
    p.shl(magic, 23, magic);
    p.loadi_bits(c, 0x3f00_0000);
    p.iadd(c, magic);
    p.loadi(c, 0.5);
    p.sub(magic, c, nf);
    p.mad(magic, d, nf, d);
    p.add(d, d, d);
    // The range, and a NaN.
    p.loadi(k, -18.0);
    p.if_(Cond::Less(x, k), |p| p.loadi(d, -1.0));
    p.loadi_bits(k, 0x42b1_7218); // 88.72284
    p.if_(Cond::LessEq(k, x), |p| p.loadi_bits(d, 0x7f80_0000));
    // The magnitude by mask: `SFPABS` leaves a negative NaN negative, and
    // `e^-NaN` came out `0` (found in 10.2e).
    p.loadi_bits(k, 0x7fff_ffff);
    p.and(x, k, t);
    p.loadi_bits(k, 0x7f80_0000);
    p.if_(Cond::Less(k, t), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`sigmoid_program`]'s derived bound, relative.
pub const SIGMOID_BOUND: f64 = EXP_BOUND + 4.0 / 16_777_216.0;

/// The sigmoid of `x` (in `L0`, spilled at `spill`) into `L7`, as Flex has it:
/// `1/(1 + e)` for `x >= 0` and `e/(1 + e)` below, `e = e^-|x|` -- one
/// exponential, never of a positive argument.
///
/// Error: `e` within `EXP_BOUND`, which `1/(1 + e)` carries scaled by `e/(1 +
/// e) <= 1/2` and `e/(1 + e)` by `1/(1 + e) <= 1`; `1 + e` rounds once, the
/// reciprocal is within an ulp (`2u`), the product rounds once: under
/// [`SIGMOID_BOUND`] `= EXP_BOUND + 4u`. `±inf` give `1` and `0`; a NaN stays
/// one; below `-87.3`, where the value would be denormal, `0`.
pub fn sigmoid_program(p: &mut Program, spill: u32) {
    use LReg as R;
    p.store(R::L0, Format::Int32, spill);
    p.abs(R::L0, R::L1);
    p.neg(R::L1, R::L1);
    exp_program(p, R::L1, R::L2);
    // `e` in `L2`; `den = 1 + e`; `y = 1/den`.
    p.add(R::L2, R::ONE, R::L3);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(R::L3, R::L7, R::L4, R::L5, R::L6);
    p.load(R::L0, Format::Int32, spill);
    p.if_(Cond::Lt0(R::L0), |p| p.mul(R::L2, R::L7, R::L7));
    // `-0` is `>= 0`: `1/2`, which the `x >= 0` branch gave.
    p.if_(Cond::Lt0(R::L0), |p| {
        p.loadi_bits(R::L5, 0x7fff_ffff);
        p.and(R::L0, R::L5, R::L4);
        p.if_(Cond::Eq0(R::L4), |p| p.loadi(R::L7, 0.5));
    });
}

/// [`tanh_program`]'s derived bound, relative.
pub const TANH_BOUND: f64 = 7.5 / 16_777_216.0;

/// `tanh x` of `x` (in `L0`, raw bits, spilled at `spill`) into `L7`: `sign(x) t/(t + 2)`, `t = e^(2|x|)
/// - 1` ([`expm1_program`], so no cancellation near zero).
///
/// Error: `t` within `EXPM1_BOUND` (`2|x|` exact), which the quotient carries
/// scaled by `2/(t + 2) <= 1`; `t + 2` rounds once (`u`), the quotient is
/// within an ulp (`2u`): under [`TANH_BOUND`] `= 7.5u`. Below `|x| = 2^-12`
/// the result is `x` itself (`tanh x = x (1 - x^2/3 + ...)`, `x^2/3 < 2^-26`),
/// bits and all -- a denormal kept, as the host keeps it; from `|x| = 9.01`
/// it is `±1` (`1 - tanh 9.01 < 2^-25`); a NaN stays one.
pub fn tanh_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let x = R::L0;
    // `expm1` keeps only its input and output: `x` waits in `Dst`.
    p.store(x, Format::Int32, spill);
    p.abs(x, R::L1);
    p.add(R::L1, R::L1, R::L1);
    expm1_program(p, R::L1, R::L7);
    p.load(x, Format::Int32, spill);
    let (t, den, y, q) = (R::L7, R::L2, R::L3, R::L4);
    p.loadi(den, 2.0);
    p.add(t, den, den);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(den, y, R::L1, R::L5, R::L6);
    p.loadi_bits(R::L6, 0x7f80_0000);
    divide(p, t, den, y, q, R::L5, R::L1, R::L6);
    p.copy_sign(q, x, R::L7);
    // Small: `x`. Large: `±1`. NaN: NaN (the quotient `inf/inf` is one too,
    // but its sign is not the input's).
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, R::L1);
    p.loadi_bits(R::L5, 0x3980_0000); // 2^-12
    p.if_(Cond::Less(R::L1, R::L5), |p| p.mov(x, R::L7));
    p.loadi(R::L5, 9.01);
    p.if_(Cond::LessEq(R::L5, R::L1), |p| {
        p.copy_sign(R::ONE, x, R::L7)
    });
    p.loadi_bits(R::L5, 0x7f80_0000);
    p.if_(Cond::Less(R::L5, R::L1), |p| {
        p.loadi_bits(R::L7, 0x7fc0_0000)
    });
}

/// [`sinh_cosh_program`]'s derived bound for `sinh`, relative.
pub const SINH_BOUND: f64 = 12.5 / 16_777_216.0;
/// The same for `cosh`.
pub const COSH_BOUND: f64 = SINH_BOUND;

/// `sinh x` (or `cosh x` where `cosh`) of `x` (in `L0`, raw bits, spilled at
/// `spill`) into `L7`, from one [`expm1_program`] of `a = |x|`: `t = e^a - 1`,
/// `e = t + 1`, `sinh a = (t + t/e)/2` (no cancellation near zero) and `cosh a
/// = (e + 1/e)/2`. From `a = 88`, where `e^a` nears `f32::MAX`, the argument
/// is `a/2` (exact) and the result `w (w/2)`, `w = e^(a/2)`: `e^-a` is then
/// below `2^-126` of `e^a`, and the product overflows only where the result
/// does (`a > 89.4159`).
///
/// Error, relative, in `u = 2^-24`, `E = EXPM1_BOUND = 4.5u` (`a` exact).
/// Below 88: `e = t + 1` is within `E t/(t+1) + u <= E + u`. `sinh`: `t/e`
/// within `E/(t+1) + u + 2u` (the quotient an ulp, `divide`), which the sum
/// `t + t/e` weighs by at most `1/2` (`t/e <= t`); the sum rounds once, the
/// halving is exact: `E + 1.5u + u = 7u`. `cosh`: `1/e` within `E + u + 2u`
/// (`recip`, an ulp), weighed by at most `1/2`; the sum rounds once: `E + 3u =
/// 7.5u`. From 88: `w` within `E + u`, twice in the product, which rounds
/// once: `2E + 3u = 12u`, and dropping `e^-a` costs below `2^-250`. Under
/// [`SINH_BOUND`] `= 12.5u` both, the second-order terms (products of these,
/// below `1e-13`) included. Below `|x| = 2^-12` `sinh x` is `x` itself (`x^2/6
/// < 2^-26`), bits and all; `±inf` give `±inf` (`cosh`: `+inf`), a NaN stays
/// one.
pub fn sinh_cosh_program(p: &mut Program, spill: u32, cosh: bool) {
    use LReg as R;
    let (x, a) = (R::L0, R::L1);
    // `expm1` keeps only its input and output: `x` waits in `Dst`.
    p.store(x, Format::Int32, spill);
    // The magnitude by mask (`SFPABS` leaves a negative NaN negative).
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, a);
    p.loadi(R::L5, 88.0);
    p.if_(Cond::LessEq(R::L5, a), |p| {
        p.loadi(R::L5, 0.5);
        p.mul(a, R::L5, a);
    });
    expm1_program(p, a, R::L7);
    let (t, e, y, q) = (R::L7, R::L2, R::L3, R::L4);
    p.add(t, R::ONE, e);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(e, y, R::L1, R::L5, R::L6);
    if cosh {
        p.add(e, y, q);
    } else {
        p.loadi_bits(R::L6, 0x7f80_0000);
        divide(p, t, e, y, q, R::L5, R::L1, R::L6);
        p.add(t, q, q);
    }
    p.loadi(R::L5, 0.5);
    p.mul(q, R::L5, q);
    p.load(x, Format::Int32, spill);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, a);
    p.loadi(R::L5, 88.0);
    p.if_(Cond::LessEq(R::L5, a), |p| {
        p.loadi(R::L5, 0.5);
        p.mul(e, R::L5, y);
        p.mul(e, y, q);
    });
    if cosh {
        p.mov(q, R::L7);
    } else {
        p.copy_sign(q, x, R::L7);
        p.loadi_bits(R::L5, 0x3980_0000); // 2^-12
        p.if_(Cond::Less(a, R::L5), |p| p.mov(x, R::L7));
    }
    p.loadi_bits(R::L5, 0x7f80_0000);
    p.if_(Cond::Less(R::L5, a), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
}

/// [`asinh_acosh_program`]'s derived bound for `asinh`, relative.
pub const ASINH_BOUND: f64 = 16.2 / 16_777_216.0;
/// The same for `acosh`.
pub const ACOSH_BOUND: f64 = ASINH_BOUND;

/// `asinh x` (or `acosh x` where `acosh`) of `x` (in `L0`, raw bits) into
/// `L7`, as one [`log1p_program`] of an argument `w` chosen per lane, `a =
/// |x|`: `asinh a = log1p(a + a^2/(1 + sqrt(1 + a^2)))` and, with `t = a - 1`
/// (exact below `2^24`), `acosh a = log1p(t + sqrt(t (t + 2)))` -- neither
/// cancels; from `a = 2^12` both are `ln(2a)` to within `1/(4a^2) < 2^-26`,
/// computed as `log1p(a - 1) + ln 2` so that `2a` never overflows. Spills at
/// `spill..spill + 192` (`log1p`'s own at `spill`).
///
/// Error, relative, in `u = 2^-24`; `log1p` carries its argument's relative
/// error scaled by `w/((1 + w) ln(1 + w)) <= 1`, and adds [`LOG1P_BOUND`] `=
/// 11.12u`; `sqrt` is within `3u` (an ulp and a half of the exact root, at
/// worst). `asinh`: `1 + a^2` within `2u`, its root `u + 3u`, `1 + root`
/// `5u`, the quotient `a^2/(..)` `u + 5u + 2u` (`divide`), weighed in `w` by
/// at most `1/2` (the quotient is below `a`), and `w`'s rounding: `5u`.
/// `acosh`: `t (t + 2)` within `2u`, its root `4u`, `w` `5u`. So `16.12u`
/// below `2^12`; above, `ln a` within `11.12u`, `ln 2`'s rounding and the
/// sum's: under `12.2u`. Under [`ASINH_BOUND`] `= 16.2u`. `asinh` is `x`
/// itself below `2^-12` (`x^2/6 < 2^-26`), odd, `±inf` for `±inf`; `acosh(1) =
/// +0`, below `1` NaN, `+inf` for `+inf`; a NaN stays one.
pub fn asinh_acosh_program(p: &mut Program, spill: u32, acosh: bool) {
    use LReg as R;
    let (x, a) = (R::L0, R::L1);
    let (sx, sv) = (spill + 64, spill + 128);
    p.store(x, Format::Int32, sx);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, a);
    if acosh {
        // `t = a - 1`; `w = t + sqrt(t (t + 2))`.
        p.sub(a, R::ONE, a);
        p.store(a, Format::Int32, sv);
        p.loadi(R::L5, 2.0);
        p.add(a, R::L5, R::L2);
        p.mul(R::L2, a, R::L2);
        sqrt_program(p, R::L2, R::L3, false);
        p.load(R::L1, Format::Int32, sv);
        p.add(R::L1, R::L3, R::L0);
    } else {
        // `m = a^2`; `w = a + m/(1 + sqrt(1 + m))`.
        p.mul(a, a, R::L2);
        p.store(R::L2, Format::Int32, sv);
        p.add(R::L2, R::ONE, R::L2);
        sqrt_program(p, R::L2, R::L3, false);
        let (m, den, y, q) = (R::L1, R::L2, R::L3, R::L4);
        p.add(R::L3, R::ONE, den);
        p.load(m, Format::Int32, sv);
        p.loadi_bits(R::L6, f32::MAX.to_bits());
        p.recip(den, y, R::L0, R::L5, R::L6);
        p.loadi_bits(R::L6, 0x7f80_0000);
        divide(p, m, den, y, q, R::L5, R::L0, R::L6);
        p.load(x, Format::Int32, sx);
        p.loadi_bits(R::L5, 0x7fff_ffff);
        p.and(x, R::L5, a);
        p.add(a, q, R::L0);
    }
    // From `a = 2^12`: `w = a - 1`, so that `log1p(w)` is `ln a`.
    p.load(R::L2, Format::Int32, sx);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(R::L2, R::L5, a);
    p.loadi_bits(R::L5, 0x4580_0000); // 2^12
    p.if_(Cond::LessEq(R::L5, a), |p| p.sub(a, R::ONE, R::L0));
    log1p_program(p, R::L0, R::L7, spill);
    p.load(x, Format::Int32, sx);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, a);
    p.loadi_bits(R::L5, 0x4580_0000);
    p.if_(Cond::LessEq(R::L5, a), |p| {
        p.loadi(R::L5, std::f32::consts::LN_2);
        p.add(R::L7, R::L5, R::L7);
    });
    if acosh {
        // Below `1` in the total order: everything negative too.
        p.if_(Cond::Less(x, R::ONE), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
    } else {
        p.mov(R::L7, R::L4);
        p.copy_sign(R::L4, x, R::L7);
        p.loadi_bits(R::L5, 0x3980_0000); // 2^-12
        p.if_(Cond::Less(a, R::L5), |p| p.mov(x, R::L7));
    }
    p.loadi_bits(R::L5, 0x7f80_0000);
    p.if_(Cond::Less(R::L5, a), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
}

/// [`atanh_program`]'s derived bound, relative.
pub const ATANH_BOUND: f64 = 14.2 / 16_777_216.0;

/// `atanh x` of `x` (in `L0`, raw bits) into `L7`: `sign(x) log1p(2a/(1 -
/// a))/2`, `a = |x|` -- one [`log1p_program`], no cancellation (`1 - a` exact
/// from `a = 1/2`, Sterbenz). Spills at `spill..spill + 128`.
///
/// Error, relative, in `u = 2^-24`: `1 - a` within `u`, `2a` exact, the
/// quotient `2u` (`divide`): `w` within `3u`, which `log1p` carries scaled by
/// at most 1, adding [`LOG1P_BOUND`] `= 11.12u`; the halving is exact: under
/// [`ATANH_BOUND`] `= 14.2u`. `x` itself below `|x| = 2^-12` (`x^2/3 <
/// 2^-25`), bits and all; `±1` give `±inf` (`1/(+0)` is `+inf`, and so is
/// `log1p(+inf)`); beyond `1` NaN, by name; a NaN stays one.
pub fn atanh_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let (x, a) = (R::L0, R::L1);
    let sx = spill + 64;
    p.store(x, Format::Int32, sx);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, a);
    let (den, y, q) = (R::L2, R::L3, R::L4);
    p.sub(R::ONE, a, den);
    p.add(a, a, a);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(den, y, R::L0, R::L5, R::L6);
    p.loadi_bits(R::L6, 0x7f80_0000);
    divide(p, a, den, y, q, R::L5, R::L0, R::L6);
    p.mov(q, R::L0);
    log1p_program(p, R::L0, R::L7, spill);
    p.loadi(R::L5, 0.5);
    p.mul(R::L7, R::L5, R::L4);
    p.load(x, Format::Int32, sx);
    p.copy_sign(R::L4, x, R::L7);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, a);
    p.loadi_bits(R::L5, 0x3980_0000); // 2^-12
    p.if_(Cond::Less(a, R::L5), |p| p.mov(x, R::L7));
    // Beyond `1`, NaNs and `±inf` included: NaN. The quotient alone does not
    // say so from `|x| ~ 2^126`, where `1/(1 - a)` flushes to a zero.
    p.if_(Cond::Less(R::ONE, a), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
}

/// [`log_sigmoid_program`]'s derived bound, relative.
pub const LOG_SIGMOID_BOUND: f64 = EXP_BOUND + LOG1P_BOUND + 1.0 / 16_777_216.0;

/// `ln(sigmoid x)` of `x` (in `L0`, raw bits) into `L7`, Flex's two branches
/// as one: `-l` for `x >= 0` and `x - l` below, `l = log1p(e)`, `e =
/// e^-|x|` -- one exponential, never of a positive argument, and one
/// [`log1p_program`]; neither side cancels (`x - l` adds magnitudes). Spills
/// at `spill..spill + 128`.
///
/// Error, relative: `e` within `EXP_BOUND` (`-|x|` exact), which `log1p`
/// carries scaled by `e/((1 + e) ln(1 + e)) <= 1`, adding [`LOG1P_BOUND`];
/// `x - l` weighs `l`'s error by `l/(|x| + l) <= 1` and rounds once: under
/// [`LOG_SIGMOID_BOUND`] `= EXP_BOUND + LOG1P_BOUND + u` (`14.3u`). From `x =
/// 87.3`, where `e` would be denormal, `-0`; `-inf` gives `-inf`, `+inf`
/// `-0`; a NaN stays one.
pub fn log_sigmoid_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let x = R::L0;
    let sx = spill + 64;
    p.store(x, Format::Int32, sx);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, R::L1);
    p.neg(R::L1, R::L1);
    exp_program(p, R::L1, R::L2);
    p.mov(R::L2, R::L0);
    log1p_program(p, R::L0, R::L7, spill);
    p.load(x, Format::Int32, sx);
    p.neg(R::L7, R::L1);
    p.if_(Cond::Lt0(x), |p| p.add(x, R::L1, R::L1));
    p.mov(R::L1, R::L7);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(x, R::L5, R::L1);
    p.loadi_bits(R::L5, 0x7f80_0000);
    p.if_(Cond::Less(R::L5, R::L1), |p| {
        p.loadi_bits(R::L7, 0x7fc0_0000)
    });
}

/// `2/pi`'s bits after the point, 24 to an entry: fdlibm's `ipio2`, as
/// `libm`'s `rem_pio2_large` has them. 216 bits; [`trig_reduce`] reads to bit
/// 204. A wrong bit `k` moves `g` by up to `2^(E - 126 - k)`, so one to about
/// `k = 150` shows in `transcendental::sin_is_within_its_derived_bound`
/// (watched failing at `1.2e35` with bit 145 flipped); the later ones are
/// below `2^-40` in `g`, which no `f32` result can see.
const TWO_OVER_PI: [u32; 9] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
];

/// Limb `n` of [`trig_reduce`]'s table: bits `23n..23n + 23` of `2/pi`'s
/// expansion preceded by 25 zeros, so that limb 0 begins at bit `k = -24`
/// (`b_k` of `2/pi = sum b_k 2^-k`, `b_k = 0` for `k <= 0`).
fn two_over_pi_limb(n: u32) -> u32 {
    (23 * n..23 * n + 23).fold(0, |v, t| {
        let k = t as i64 - 25;
        let b = if k < 1 {
            0
        } else {
            let i = (k - 1) as usize;
            (TWO_OVER_PI[i / 24] >> (23 - i % 24)) & 1
        };
        (v << 1) | b
    })
}

/// [`sin_cos_program`]'s derived bound for `sin`, relative.
pub const SIN_BOUND: f64 = 2.2 / 16_777_216.0;
/// The same for `cos`.
pub const COS_BOUND: f64 = SIN_BOUND;

/// The reduction of `x` (in `L0`, raw bits; spilled at `spill`) by `pi/2`:
/// `|x| = (q + g) pi/2`, `q` an integer, `|g| <= 1/2`, into `r = |g| pi/2`
/// as `r_hi` (`L5`) + `r_lo` (`L4`), `|r_lo| <= ulp(r_hi)/2`, and `q mod 4`
/// with `g`'s sign in bit 31 into `spill + 64`. Every register scratch.
///
/// Payne and Hanek's, in exact fixed point, for every finite `a = |x| =
/// M 2^(E-150)` from `pi/4` (`M` the 24-bit mantissa, `E` the biased
/// exponent): `a (2/pi) = sum_k M b_k 2^(E-150-k)`, and the terms with `k <=
/// E - 152` are multiples of 4, so only the window `W` of 92 bits from `k =
/// E - 151` matters: `a (2/pi) = M W 2^-90 (mod 4)`, less the bits past the
/// window, under `M 2^-90 < 2^-66`. `W` is four 23-bit limbs `W3..W0` cut
/// from the table ([`two_over_pi_limb`]) at bit `t0 = E - 126` (`0..129`):
/// whole limbs `j = t0 div 23` by five conditional shifts, then the shift
/// within, `sh = t0 mod 23`, by funnelling neighbours. `M W mod 2^92`, as four
/// limbs, is `2^23 W + M' W` (`M = 2^23 + M'`): each `M' W_i` from
/// `SFPMUL24`'s two halves, then the carries; the lowest limb only adds below
/// `2^-67`, so it is left out. The top two bits are `q mod 4`, the 67 below
/// the fraction; from one half up `q + 1` and the fraction's complement (`1 -
/// F`, less `2^-67`), with the sign set.
///
/// Error: the window and the dropped limb, under `2^-65` in `g` together.
/// The closest float to a multiple of `pi/2` is `16367173 2^72`, `|g| =
/// 2^-29.86` (every float scanned; `transcendental::the_hardest_reductions_
/// are_within_their_bounds` holds it), so `g` is within `2^-35` of itself,
/// relative. Then `g` to `f32`s: three exact limb conversions, an exact
/// two-sum of the top two, the third added with one rounding (relative
/// `2^-47`), and `pi/2` as two `f32`s, the product Dekker's (12-bit halves:
/// `SFPMAD` is not fused, numerics row E) and the cross terms rounded once
/// each: `r` within `2^-34` of itself, relative -- `6e-11`, `0.001u`.
/// Below `pi/4`, `r = a` exactly and `q = 0`. Infinities and NaNs give a
/// value the caller replaces.
pub fn trig_reduce(p: &mut Program, spill: u32) {
    use LReg as R;
    let (sx, sq) = (spill, spill + 64);
    let mask23 = 0x7f_ffff;
    p.store(R::L0, Format::Int32, sx);
    // `M'` waits where `q` will go; `sh = t0` in `L5`.
    p.loadi_bits(R::L7, mask23);
    p.and(R::L0, R::L7, R::L1);
    p.store(R::L1, Format::Int32, sq);
    p.exponent(R::L0, false, R::L5);
    p.iadd_imm(R::L5, -126, R::L5);
    // `L0..L4`: limbs `j..j + 5` of the table, `j` from 0, shifted down
    // while `sh >= 23`.
    let a = [R::L0, R::L1, R::L2, R::L3, R::L4];
    for (n, r) in a.iter().enumerate() {
        p.loadi_bits(*r, two_over_pi_limb(n as u32));
    }
    p.loadi_bits(R::L6, 23);
    for j in 1..=5 {
        p.if_(Cond::LessEq(R::L6, R::L5), |p| {
            for i in 0..4 {
                p.mov(a[i + 1], a[i]);
            }
            p.loadi_bits(R::L4, two_over_pi_limb(j + 4));
            p.iadd_imm(R::L5, -23, R::L5);
        });
    }
    // `W_i = (A_i << sh | A_(i+1) >> (23 - sh)) mod 2^23`, in `A_i`: `W3`
    // in `L0` .. `W0` in `L3`. Bits pushed past 32 are above the 23 kept.
    p.iadd_imm(R::L5, -23, R::L7);
    for i in 0..4 {
        p.mov(a[i + 1], R::L6);
        p.shr_by(R::L7, R::L6);
        p.shl_by(R::L5, a[i]);
        p.or(a[i], R::L6, a[i]);
    }
    // `W3` only meets `SFPMUL24`, which reads its operands' low 23 bits.
    p.loadi_bits(R::L7, mask23);
    for r in [R::L1, R::L2, R::L3] {
        p.and(r, R::L7, r);
    }
    // Limbs 1..3 of `M W mod 2^92`, in `L3`, `L2`, `L1`: limb `i` is `W_(i-1)
    // + hi(M' W_(i-1)) + lo(M' W_i)`.
    p.load(R::L4, Format::Int32, sq);
    for (hi_of, lo_of, into) in [
        (R::L3, R::L2, R::L3),
        (R::L2, R::L1, R::L2),
        (R::L1, R::L0, R::L1),
    ] {
        p.mul24(R::L4, hi_of, true, R::L5);
        p.iadd(R::L5, into);
        p.mul24(R::L4, lo_of, false, R::L5);
        p.iadd(R::L5, into);
    }
    // The carries (each limb below `3 * 2^23 + 2`), limb 3 kept to 23 bits.
    p.loadi_bits(R::L6, (-23i32) as u32);
    for (from, into) in [(R::L3, R::L2), (R::L2, R::L1)] {
        p.mov(from, R::L5);
        p.shr_by(R::L6, R::L5);
        p.and(from, R::L7, from);
        p.iadd(R::L5, into);
    }
    p.and(R::L1, R::L7, R::L1);
    // `q` (`L0`), the half bit (`L5`), the fraction's top 21 bits (`L1`).
    p.mov(R::L1, R::L0);
    p.loadi_bits(R::L6, (-21i32) as u32);
    p.shr_by(R::L6, R::L0);
    p.loadi_bits(R::L6, 0x10_0000);
    p.and(R::L1, R::L6, R::L5);
    p.loadi_bits(R::L6, 0x1f_ffff);
    p.and(R::L1, R::L6, R::L1);
    p.if_(Cond::Ne0(R::L5), |p| {
        p.iadd_imm(R::L0, 1, R::L0);
        p.xor(R::L6, R::L1);
        p.xor(R::L7, R::L2);
        p.xor(R::L7, R::L3);
    });
    p.shl(R::L5, 11, R::L5);
    p.or(R::L0, R::L5, R::L0);
    p.store(R::L0, Format::Int32, sq);
    // `|g| = f3 + f2 + f1`, the limbs at `2^-21`, `2^-44`, `2^-67`.
    for (r, scale) in [
        (R::L1, 0x3500_0000),
        (R::L2, 0x2980_0000),
        (R::L3, 0x1e00_0000),
    ] {
        p.sm32_to_float(r, r);
        p.loadi_bits(R::L6, scale);
        p.mul(r, R::L6, r);
    }
    // `hi = f3 + f2`, `lo` its exact error (`f3 = 0` or `f3 > f2`), plus `f1`.
    let (hi, lo) = (R::L4, R::L5);
    p.add(R::L1, R::L2, hi);
    p.sub(hi, R::L1, lo);
    p.sub(R::L2, lo, lo);
    p.add(lo, R::L3, lo);
    // `r = (hi + lo) (ph + pl)`: `ph hi` exactly as `r_hi + e` (Dekker,
    // negated: `L6 = r_hi - hi ph`), then `lo ph + hi pl - L6`.
    let ph = std::f32::consts::FRAC_PI_2;
    let pl = (std::f64::consts::FRAC_PI_2 - ph as f64) as f32;
    let (phh, phl) = split12_host(ph);
    p.loadi(R::L7, 4097.0);
    split12(p, hi, R::L1, R::L2, R::L7);
    p.loadi(R::L7, ph);
    p.mul(hi, R::L7, R::L3);
    p.loadi(R::L7, phh);
    p.nmad(R::L1, R::L7, R::L3, R::L6);
    p.loadi(R::L7, phl);
    p.nmad(R::L1, R::L7, R::L6, R::L6);
    p.loadi(R::L7, phh);
    p.nmad(R::L2, R::L7, R::L6, R::L6);
    p.loadi(R::L7, phl);
    p.nmad(R::L2, R::L7, R::L6, R::L6);
    p.loadi(R::L7, ph);
    p.mul(lo, R::L7, R::L1);
    p.loadi(R::L7, pl);
    p.mad(hi, R::L7, R::L1, R::L1);
    p.sub(R::L1, R::L6, R::L1);
    // Renormalised: `r_hi` the rounded sum (`L5`), `r_lo` its error (`L4`).
    p.add(R::L3, R::L1, R::L5);
    p.sub(R::L5, R::L3, R::L2);
    p.sub(R::L1, R::L2, R::L4);
    // Below `pi/4`: `r = a`, `q = 0`.
    p.load(R::L0, Format::Int32, sx);
    p.loadi_bits(R::L7, 0x7fff_ffff);
    p.and(R::L0, R::L7, R::L0);
    p.load(R::L6, Format::Int32, sq);
    p.loadi(R::L7, std::f32::consts::FRAC_PI_4);
    p.if_(Cond::Less(R::L0, R::L7), |p| {
        p.mov(R::L0, R::L5);
        p.mov(R::ZERO, R::L4);
        p.mov(R::ZERO, R::L6);
    });
    p.store(R::L6, Format::Int32, sq);
}

/// `sin r` into `spill + 128` and `cos r` into `L6`, of `r = r_hi + r_lo`
/// ([`trig_reduce`]'s `L5`, `L4`; `|r| <= pi/4`). Every register scratch.
///
/// `sin r = r_hi + (r_hi^3 P(z) + r_lo (1 - z/2))`, `z = r_hi^2`, `P` the
/// Taylor series to `z^4` (`r^11`; the next term below `1e-11` of the sum);
/// `cos r = 1 - (z/2 - (z^2 Q(z) - r_lo r_hi))`, `Q` to `z^4` (`r^12`; the
/// next below `5e-13`). Neither cancels: `cos r >= 0.707`.
///
/// Error, relative, in `u = 2^-24` (one rounding): `sin`: `z` within `u`,
/// `r_hi^3` within `2u`, Horner's `P` within `2.1u` (its last step and `-1/6`'s
/// rounding; `z/20` damps the rest), the multiply-add `u`: the correction
/// within `5.1u` of itself, and it is at most `0.103 r`, where `sin r >=
/// 0.900 r`: `0.59u`; `r_lo (1 - z/2)` misses `r_lo cos r` by `r_lo r^4/24`,
/// `0.02u`; the last add `u`: `1.61u`. `cos`: `z/2`'s error `0.31u`
/// absolute, the tail's (`<= 0.016`, its own `7u`, and `r_lo r_hi` for `r_lo
/// sin r_hi`, `0.064u`) `0.18u`, the inner difference's rounding `0.29u` (it
/// is below `0.293`): `0.78u` over `cos r >= 0.707` is `1.10u`, and the last
/// rounding `u`: `2.10u`. Both under [`SIN_BOUND`] `= COS_BOUND = 2.2u` --
/// `sin x` is `cos r` in odd quadrants -- `r`'s own `0.001u` included.
fn sin_cos_cores(p: &mut Program, spill: u32) {
    use LReg as R;
    let (r_hi, r_lo, z, half) = (R::L5, R::L4, R::L3, R::L0);
    let mut fact = 1.0f64;
    let taylor: Vec<f32> = (0..=12)
        .map(|n| {
            if n > 0 {
                fact *= n as f64;
            }
            let sign = if (n / 2) % 2 == 0 { 1.0 } else { -1.0 };
            (sign / fact) as f32
        })
        .collect();
    p.mul(r_hi, r_hi, z);
    p.loadi(R::L2, taylor[11]);
    for n in [9, 7, 5, 3] {
        p.loadi(R::L1, taylor[n]);
        p.mad(R::L2, z, R::L1, R::L2);
    }
    p.mul(r_hi, z, R::L1);
    p.loadi(half, 0.5);
    p.mul(z, half, half);
    p.nmad(r_lo, half, r_lo, R::L6);
    p.mad(R::L1, R::L2, R::L6, R::L6);
    p.add(r_hi, R::L6, R::L7);
    p.store(R::L7, Format::Int32, spill + 128);
    p.loadi(R::L2, taylor[12]);
    for n in [10, 8, 6, 4] {
        p.loadi(R::L1, taylor[n]);
        p.mad(R::L2, z, R::L1, R::L2);
    }
    p.mul(z, z, R::L1);
    p.mul(R::L1, R::L2, R::L1);
    p.nmad(r_lo, r_hi, R::L1, R::L1);
    p.sub(half, R::L1, R::L1);
    p.sub(R::ONE, R::L1, R::L6);
}

/// `sin x` or `cos x` of `x` (in `L0`, raw bits) into `L7`, for every finite
/// `x`: [`trig_reduce`], both of [`sin_cos_cores`], and the quadrant -- `sin
/// x = sign(x) [S, C, -S, -C][q]`, `cos x = [C, -S, -C, S][q]`, `S = sin r`
/// with `g`'s sign, `C = cos r` -- as sign bits. Spills at `spill..spill +
/// 192`.
///
/// Within [`SIN_BOUND`], [`COS_BOUND`] ([`sin_cos_cores`]); the quadrant is
/// exact. `sin` is `x` itself below `|x| = 2^-12` (`x^2/6 < 2^-26`), bits and
/// all; `cos` of a zero is `1`; `±inf` and NaNs give NaN.
pub fn sin_cos_program(p: &mut Program, spill: u32, cos: bool) {
    use LReg as R;
    trig_reduce(p, spill);
    sin_cos_cores(p, spill);
    let (x, qn, s, c, t, k) = (R::L0, R::L1, R::L2, R::L6, R::L3, R::L4);
    p.load(x, Format::Int32, spill);
    p.load(qn, Format::Int32, spill + 64);
    p.load(s, Format::Int32, spill + 128);
    p.loadi_bits(k, 0x8000_0000);
    p.and(qn, k, t);
    p.xor(t, s);
    // `q` odd takes the other core.
    p.loadi_bits(k, 1);
    p.and(qn, k, t);
    let (first, other) = if cos { (c, s) } else { (s, c) };
    p.mov(first, R::L7);
    p.if_(Cond::Ne0(t), |p| p.mov(other, R::L7));
    // The sign: `sin`'s from bit 1 of `q` and `x`'s; `cos`'s from bit 1 of
    // `q + 1` (set for `q` 1 and 2).
    if cos {
        p.iadd_imm(qn, 1, t);
    } else {
        p.mov(qn, t);
    }
    p.loadi_bits(k, 2);
    p.and(t, k, t);
    p.shl(t, 30, t);
    if !cos {
        p.loadi_bits(k, 0x8000_0000);
        p.and(x, k, k);
        p.xor(k, t);
    }
    p.xor(t, R::L7);
    p.loadi_bits(k, 0x7fff_ffff);
    p.and(x, k, t);
    if !cos {
        p.loadi_bits(k, 0x3980_0000); // 2^-12
        p.if_(Cond::Less(t, k), |p| p.mov(x, R::L7));
    }
    p.loadi_bits(k, 0x7f80_0000);
    p.if_(Cond::LessEq(k, t), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
}

/// [`tan_program`]'s derived bound, relative.
pub const TAN_BOUND: f64 = 5.8 / 16_777_216.0;

/// `tan x` of `x` (in `L0`, raw bits) into `L7`, for every finite `x`:
/// [`trig_reduce`], both of [`sin_cos_cores`], then `S/C` for even `q` and
/// `-C/S` for odd -- `S = sin r` with `g`'s sign, `C = cos r` -- and `x`'s
/// sign (`tan` is odd). Spills at `spill..spill + 192`.
///
/// Error, relative, in `u = 2^-24`: the cores' `1.61u` and `2.10u`
/// ([`sin_cos_cores`]), the quotient `2u` (`divide`, an ulp): under
/// [`TAN_BOUND`] `= 5.8u`, `r`'s `0.001u` included. Relative everywhere, the
/// poles' neighbours too: `r` is relative-accurate there (`|g| >= 2^-29.86`),
/// so `S` is, and `C/S` stays below `2^31`. `x` itself below `|x| = 2^-12`
/// (`x^2/3 < 2^-25`), bits and all; `±inf` and NaNs give NaN.
pub fn tan_program(p: &mut Program, spill: u32) {
    use LReg as R;
    trig_reduce(p, spill);
    sin_cos_cores(p, spill);
    let (num, den, odd) = (R::L1, R::L2, R::L5);
    p.load(R::L0, Format::Int32, spill + 64);
    p.load(R::L3, Format::Int32, spill + 128);
    p.loadi_bits(R::L4, 0x8000_0000);
    p.and(R::L0, R::L4, R::L7);
    p.xor(R::L7, R::L3);
    p.loadi_bits(R::L4, 1);
    p.and(R::L0, R::L4, odd);
    p.mov(R::L3, num);
    p.mov(R::L6, den);
    p.if_(Cond::Ne0(odd), |p| {
        p.mov(R::L6, num);
        p.mov(R::L3, den);
    });
    let (y, q) = (R::L4, R::L7);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(den, y, R::L0, R::L3, R::L6);
    p.loadi_bits(R::L6, 0x7f80_0000);
    divide(p, num, den, y, q, R::L3, R::L0, R::L6);
    // The sign: odd `q`'s negation and `x`'s.
    let (x, t, k) = (R::L0, R::L1, R::L2);
    p.load(x, Format::Int32, spill);
    p.load(t, Format::Int32, spill + 64);
    p.loadi_bits(k, 1);
    p.and(t, k, t);
    p.shl(t, 31, t);
    p.loadi_bits(k, 0x8000_0000);
    p.and(x, k, k);
    p.xor(k, t);
    p.xor(t, q);
    p.loadi_bits(k, 0x7fff_ffff);
    p.and(x, k, t);
    p.loadi_bits(k, 0x3980_0000); // 2^-12
    p.if_(Cond::Less(t, k), |p| p.mov(x, q));
    p.loadi_bits(k, 0x7f80_0000);
    p.if_(Cond::LessEq(k, t), |p| p.loadi_bits(q, 0x7fc0_0000));
}

/// [`ATAN_FIT`]'s fit and evaluation together, relative, measured over every
/// 64th float of `[0, 1)` (0.11u and 1.41u).
pub const ATAN_FIT_BOUND: f64 = 1.6 / 16_777_216.0;
/// [`atan_program`]'s and [`atan2_program`]'s derived bound, relative.
pub const ATAN_BOUND: f64 = 6.2 / 16_777_216.0;
/// The same for `atan2`.
pub const ATAN2_BOUND: f64 = ATAN_BOUND;

/// `t G(t^2)` for `t` in `[0, sqrt piece.hi]` (in `L0`, raw bits; spilled at
/// `spill`) into `L7`, `G` `piece`'s fit by Clenshaw: `atan t` for
/// [`ATAN_FIT`], `asin t` for [`ASIN_FIT`]. Every register scratch.
fn odd_core(p: &mut Program, spill: u32, piece: Piece) {
    use LReg as R;
    p.store(R::L0, Format::Int32, spill);
    p.mul(R::L0, R::L0, R::L0);
    clenshaw(p, piece, R::L6);
    p.load(R::L1, Format::Int32, spill);
    p.mul(R::L1, R::L6, R::L7);
}

/// `asin t` for `t` in `[0, 0.7]` (in `L0`, raw bits; spilled at `spill`)
/// into `L7`: `t + t^3 K(t^2)`, `K` [`ASIN_FIT`] by Clenshaw, the last step
/// one multiply-add. Every register scratch.
fn asin_core(p: &mut Program, spill: u32) {
    use LReg as R;
    p.store(R::L0, Format::Int32, spill);
    p.mul(R::L0, R::L0, R::L0);
    clenshaw(p, ASIN_FIT, R::L6);
    p.load(R::L1, Format::Int32, spill);
    p.mul(R::L1, R::L0, R::L2);
    p.mad(R::L2, R::L6, R::L1, R::L7);
}

/// `v` (in `L7`) to `hi + lo - v`, `hi + lo` a constant split in two
/// (`pi/2`, `pi`): one rounding each. `L5` scratch.
fn from_constant(p: &mut Program, c: f64) {
    use LReg as R;
    let hi = c as f32;
    let lo = (c - hi as f64) as f32;
    p.loadi(R::L5, hi);
    p.sub(R::L5, R::L7, R::L7);
    p.loadi(R::L5, lo);
    p.add(R::L7, R::L5, R::L7);
}

/// `atan x` of `x` (in `L0`, raw bits) into `L7`: on `a = |x|`, [`odd_core`]
/// of `a` up to 1 and `pi/2 - atan(1/a)` beyond, then `x`'s sign. Spills at
/// `spill..spill + 128`.
///
/// Error, relative, in `u = 2^-24`: up to 1, `t = a` exact, `t^2` within
/// `u`, which `G` carries scaled by `|s G'/G| <= 0.18`, `G` within
/// [`ATAN_FIT_BOUND`] `= 1.6u`, the product `u`: `2.8u`. Beyond: `1/a` within
/// `2u` (`recip`), which `atan` carries as `t/((1 + t^2) atan t)` and the
/// result as `t/((1 + t^2) (pi/2 - atan t)) <= 0.64`: `1.27u`; the core's own
/// `2.8u` weighed by `atan t/(pi/2 - atan t) <= 1`; the subtraction and the
/// low part's add `2u` (the result is at least `pi/4`): under [`ATAN_BOUND`]
/// `= 6.2u`. `x` itself below `|x| = 2^-12` (`x^2/3 < 2^-25`), bits and all;
/// `±inf` give `±pi/2` (`1/inf = 0`), and so does every `|x|` from `2^126`,
/// where `1/a` flushes; a NaN stays one.
pub fn atan_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let sx = spill + 64;
    p.store(R::L0, Format::Int32, sx);
    p.loadi_bits(R::L7, 0x7fff_ffff);
    p.and(R::L0, R::L7, R::L1);
    p.if_(Cond::Less(R::ONE, R::L1), |p| {
        p.loadi_bits(R::L6, f32::MAX.to_bits());
        p.recip(R::L1, R::L0, R::L3, R::L4, R::L6);
    });
    p.if_(Cond::LessEq(R::L1, R::ONE), |p| p.mov(R::L1, R::L0));
    odd_core(p, spill, ATAN_FIT);
    p.load(R::L0, Format::Int32, sx);
    p.loadi_bits(R::L4, 0x7fff_ffff);
    p.and(R::L0, R::L4, R::L1);
    p.if_(Cond::Less(R::ONE, R::L1), |p| {
        from_constant(p, std::f64::consts::FRAC_PI_2)
    });
    p.copy_sign(R::L7, R::L0, R::L4);
    p.mov(R::L4, R::L7);
    p.loadi_bits(R::L5, 0x3980_0000); // 2^-12
    p.if_(Cond::Less(R::L1, R::L5), |p| p.mov(R::L0, R::L7));
    p.loadi_bits(R::L5, 0x7f80_0000);
    p.if_(Cond::Less(R::L5, R::L1), |p| {
        p.loadi_bits(R::L7, 0x7fc0_0000)
    });
}

/// `atan2(y, x)` of `y` (in `L0`) and `x` (in `L1`), raw bits, into `L7`, as
/// `f32::atan2`: [`odd_core`] of `t = min/max` of the magnitudes, `pi/2 - v`
/// where `|y| > |x|`, `pi - v` where `x`'s sign is set (`-0` included), then
/// `y`'s sign. Spills at `spill..spill + 192`.
///
/// Error, relative, in `u = 2^-24`: `t` within `2u` (`divide`, both operands
/// scaled by `2^-64` above `2^100`, exactly, as `DIV`'s), carried as
/// in [`atan_program`]'s large branch, so each of the three forms is within
/// its `6.2u`; `pi - w` with `w` within that adds its two roundings and
/// weighs `w`'s error by `w/(pi - w) <= 1` against a result above `pi/2`:
/// under [`ATAN2_BOUND`] `= 6.2u`. Special values as IEEE's: a zero `y`
/// gives `±0` or `±pi` by `x`'s sign; a zero `x`, `±pi/2`; infinities `±pi/4`
/// or `±3pi/4` together, an infinite `y` `±pi/2`, an infinite `x` `±0` or
/// `±pi`; a NaN in either NaN. A denormal operand is a zero of its sign, as
/// `SFPMAD` would read it (numerics row D): `atan2` of two denormals is a
/// zero's, where the host's is that of their ratio.
pub fn atan2_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let (sy, sx) = (spill + 64, spill + 128);
    p.store(R::L0, Format::Int32, sy);
    p.store(R::L1, Format::Int32, sx);
    // The magnitudes, a denormal as a zero.
    let mags = |p: &mut Program, ay: LReg, ax: LReg| {
        p.load(R::L0, Format::Int32, sy);
        p.load(R::L1, Format::Int32, sx);
        p.loadi_bits(R::L6, 0x7fff_ffff);
        p.and(R::L0, R::L6, ay);
        p.and(R::L1, R::L6, ax);
        p.loadi_bits(R::L6, 0x0080_0000);
        p.if_(Cond::Less(ay, R::L6), |p| p.mov(R::ZERO, ay));
        p.if_(Cond::Less(ax, R::L6), |p| p.mov(R::ZERO, ax));
    };
    let (ay, ax, num, den) = (R::L2, R::L3, R::L4, R::L5);
    mags(p, ay, ax);
    p.mov(ay, num);
    p.mov(ax, den);
    p.if_(Cond::Less(ax, ay), |p| {
        p.mov(ax, num);
        p.mov(ay, den);
    });
    // Both by `2^-64` above `2^100`: `1/den` would be denormal and flush.
    scale_large_divisor(p, num, den, R::L6, R::L7);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(den, R::L7, R::L0, R::L1, R::L6);
    p.loadi_bits(R::L6, 0x7f80_0000);
    divide(p, num, den, R::L7, R::L0, R::L1, R::L2, R::L6);
    // `0/0` (and `0/x`) is `0`; `inf/inf`, `1`.
    p.if_(Cond::Eq0(num), |p| p.mov(R::ZERO, R::L0));
    p.if_(Cond::LessEq(R::L6, num), |p| p.mov(R::ONE, R::L0));
    odd_core(p, spill, ATAN_FIT);
    mags(p, ay, ax);
    p.if_(Cond::Less(ax, ay), |p| {
        from_constant(p, std::f64::consts::FRAC_PI_2)
    });
    p.if_(Cond::Lt0(R::L1), |p| from_constant(p, std::f64::consts::PI));
    p.copy_sign(R::L7, R::L0, R::L4);
    p.mov(R::L4, R::L7);
    p.loadi_bits(R::L6, 0x7f80_0000);
    p.if_(Cond::Less(R::L6, ay), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
    p.if_(Cond::Less(R::L6, ax), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
}

/// [`ASIN_FIT`]'s fit and evaluation together, relative to `K`, measured over
/// every 64th float of `[0, 0.49)` (0.41u and 1.72u).
pub const ASIN_FIT_BOUND: f64 = 2.2 / 16_777_216.0;
/// [`asin_acos_program`]'s derived bound for `asin`, relative.
pub const ASIN_BOUND: f64 = 6.8 / 16_777_216.0;
/// The same for `acos`.
pub const ACOS_BOUND: f64 = 4.7 / 16_777_216.0;

/// `asin x` or `acos x` of `x` (in `L0`, raw bits) into `L7`, on `a = |x|`
/// and `z = sqrt((1 - a)/2)` -- `1 - a` exact from `a = 1/2` (Sterbenz), the
/// halving too -- through one [`asin_core`]: up to `a = 0.7` of `a` itself,
/// `asin x = sign(x) v` and `acos x = pi/2 - sign(x) v`; beyond, of `z`,
/// `asin x = sign(x) (pi/2 - 2v)`, and `acos x = 2v` or `pi - 2v` by `x`'s
/// sign. The switch at 0.7 rather than 1/2 keeps `2v` below the result it is
/// taken from (`2 asin(0.387) = 0.80 < pi/2 - 0.80`). Spills at `spill..spill
/// + 128`.
///
/// Error, relative, in `u = 2^-24`: the core of `t` exact: `t^3` within `2u`
/// (`t^2` and the product), `K` within [`ASIN_FIT_BOUND`] `= 2.2u`, the
/// multiply-add's product a further `u`: the correction within `5.2u` of
/// itself and at most `0.105` of the result, `0.55u`, and the last rounding
/// `u`: `1.55u`. `z` is within `3u` (`sqrt_program`, 1.5 ulps), which
/// `asin` carries by `z/(sqrt(1 - z^2) asin z) <= 1.03`: the core of `z`
/// within `4.64u`. `asin`: up to 0.7, `1.55u`; beyond, `2v`'s `4.64u` weighed
/// by `2v/(pi/2 - 2v) <= 1.03`, and two roundings: `6.78u`. `acos`: up to
/// 0.7, `v`'s `1.55u` weighed by `v/(pi/2 - |v|) <= 0.98`, and two roundings:
/// `3.52u`; beyond, `2v` itself (`4.64u`) or `pi - 2v` (weight `0.34`, two
/// roundings: `3.58u`). Under [`ASIN_BOUND`] `= 6.8u`, [`ACOS_BOUND`] `=
/// 4.7u`. `asin` is `x` itself below `|x| = 2^-12` (`x^2/6 < 2^-25`), bits
/// and all; beyond 1, `±inf` included, NaN by name; a NaN stays one.
pub fn asin_acos_program(p: &mut Program, spill: u32, acos: bool) {
    use LReg as R;
    let sx = spill + 64;
    let switch = 0.7f32;
    p.store(R::L0, Format::Int32, sx);
    p.loadi_bits(R::L7, 0x7fff_ffff);
    p.and(R::L0, R::L7, R::L1);
    p.sub(R::ONE, R::L1, R::L2);
    p.loadi(R::L3, 0.5);
    p.mul(R::L2, R::L3, R::L2);
    sqrt_program(p, R::L2, R::L0, false);
    p.load(R::L3, Format::Int32, sx);
    p.loadi_bits(R::L7, 0x7fff_ffff);
    p.and(R::L3, R::L7, R::L1);
    p.loadi(R::L7, switch);
    p.if_(Cond::LessEq(R::L1, R::L7), |p| p.mov(R::L1, R::L0));
    asin_core(p, spill);
    let (x, a, k) = (R::L0, R::L1, R::L2);
    p.load(x, Format::Int32, sx);
    p.loadi_bits(k, 0x7fff_ffff);
    p.and(x, k, a);
    p.loadi(k, switch);
    if acos {
        p.if_else(
            Cond::LessEq(a, k),
            |p| {
                // `pi/2 - sign(x) v`.
                p.if_(Cond::Lt0(x), |p| p.neg(R::L7, R::L7));
                from_constant(p, std::f64::consts::FRAC_PI_2);
            },
            |p| {
                p.add(R::L7, R::L7, R::L7);
                p.if_(Cond::Lt0(x), |p| from_constant(p, std::f64::consts::PI));
            },
        );
    } else {
        p.if_(Cond::Less(k, a), |p| {
            p.add(R::L7, R::L7, R::L7);
            from_constant(p, std::f64::consts::FRAC_PI_2);
        });
        p.copy_sign(R::L7, x, R::L4);
        p.mov(R::L4, R::L7);
        p.loadi_bits(k, 0x3980_0000); // 2^-12
        p.if_(Cond::Less(a, k), |p| p.mov(x, R::L7));
    }
    p.if_(Cond::Less(R::ONE, a), |p| p.loadi_bits(R::L7, 0x7fc0_0000));
}

/// The function a [`Piece`] fits, in `f64`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Fit {
    /// `erfcx(a) = e^(a^2) erfc(a)` ([`erfcx64`]).
    Erfcx,
    /// `atan(sqrt s) / sqrt s`, `1` at `s = 0` (10.2f): `atan t = t G(t^2)`.
    AtanSqrt,
    /// `(asin(sqrt s)/sqrt s - 1)/s`, `1/6` at `s = 0`: `asin t = t + t^3
    /// K(t^2)`, the fit's roundings only in the correction (`asin t/t` near
    /// 1.05 would cost an ulp of a binade it barely enters).
    AsinSqrt,
}

impl Fit {
    pub fn eval(self, x: f64) -> f64 {
        match self {
            Fit::Erfcx => erfcx64(x),
            Fit::AtanSqrt if x == 0.0 => 1.0,
            Fit::AtanSqrt => libm::atan(x.sqrt()) / x.sqrt(),
            // The series below `1e-3`, where the difference would cancel:
            // `asin y / y = sum (2n)!/(4^n n!^2 (2n + 1)) y^2n`, to `n = 6`.
            Fit::AsinSqrt if x < 1.0e-3 => [
                1.0 / 6.0,
                3.0 / 40.0,
                5.0 / 112.0,
                35.0 / 1152.0,
                63.0 / 2816.0,
                231.0 / 13312.0,
            ]
            .iter()
            .rev()
            .fold(0.0, |acc, c| acc * x + c),
            Fit::AsinSqrt => (libm::asin(x.sqrt()) / x.sqrt() - 1.0) / x,
        }
    }
}

/// A Chebyshev fit of `fit` over `[lo, hi)`, of degree `deg`: its error over
/// every float of the interval is measured by
/// `transcendental::every_fit_and_its_evaluation_are_within_their_parts`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Piece {
    pub lo: f64,
    pub hi: f64,
    pub deg: usize,
    pub fit: Fit,
}

/// The fit [`erfc_mid`] uses: where `erf` is `1 - erfc` without loss.
pub const ERFC_MID: Piece = Piece {
    lo: 0.5,
    hi: 3.92,
    deg: 16,
    fit: Fit::Erfcx,
};
/// The fit beyond it, for `erfc` itself (the normal CDF's far tail, `gelu`):
/// to 9.3, past which `erfc` is below `2^-126` relative to anything it
/// multiplies there.
pub const ERFC_TAIL: Piece = Piece {
    lo: 3.92,
    hi: 9.3,
    deg: 12,
    fit: Fit::Erfcx,
};
/// `atan`'s core on `[0, 1]` ([`atan_program`]): `atan(sqrt s)/sqrt s` is
/// analytic but for the branch point at `s = -1`, so the series falls as
/// `(3 + sqrt 8)^-k`.
pub const ATAN_FIT: Piece = Piece {
    lo: 0.0,
    hi: 1.0,
    deg: 13,
    fit: Fit::AtanSqrt,
};
/// `asin`'s core on `[0, 0.49]` ([`asin_acos_program`]): [`Fit::AsinSqrt`],
/// its branch point at `s = 1`.
pub const ASIN_FIT: Piece = Piece {
    lo: 0.0,
    hi: 0.49,
    deg: 16,
    fit: Fit::AsinSqrt,
};
/// Every piece a program uses, each fitted once.
pub const PIECES: [Piece; 4] = [ERFC_MID, ERFC_TAIL, ATAN_FIT, ASIN_FIT];
pub const ERFC_LO: f64 = ERFC_MID.lo;
pub const ERFC_HI: f64 = ERFC_MID.hi;
pub const ERFC_DEG: usize = ERFC_MID.deg;

/// `erfcx(x) = e^(x^2) erfc(x)` in `f64`, from `libm`'s `erfc`.
pub fn erfcx64(x: f64) -> f64 {
    libm::erfc(x) * (x * x).exp()
}

/// The Chebyshev coefficients of `erfcx` on `[ERFC_LO, ERFC_HI]` ([`piece_cheb`]
/// of [`ERFC_MID`]).
pub fn erfcx_cheb() -> &'static [f32] {
    piece_cheb(ERFC_MID)
}

/// The Chebyshev coefficients of `piece`'s function on its interval, `c_0`
/// halved (`f = c_0 + sum c_k T_k(t)`), as `f32`, computed once from
/// [`Fit::eval`] at 64 Chebyshev nodes -- in the builder, so no coefficient
/// is transcribed from anywhere. Each function is smooth on its interval, so
/// the series converges fast; the truncation and the rounding of each
/// coefficient to `f32` are both inside the measured fit error.
pub fn piece_cheb(piece: Piece) -> &'static [f32] {
    static CELLS: [std::sync::OnceLock<Vec<f32>>; PIECES.len()] =
        [const { std::sync::OnceLock::new() }; PIECES.len()];
    let i = PIECES
        .iter()
        .position(|q| *q == piece)
        .expect("a piece of `PIECES`");
    CELLS[i].get_or_init(|| {
        let m = 64usize;
        let (mid, half) = ((piece.hi + piece.lo) / 2.0, (piece.hi - piece.lo) / 2.0);
        let f: Vec<f64> = (0..m)
            .map(|j| {
                let th = std::f64::consts::PI * (j as f64 + 0.5) / m as f64;
                piece.fit.eval(mid + half * th.cos())
            })
            .collect();
        (0..=piece.deg)
            .map(|k| {
                let c: f64 = (0..m)
                    .map(|j| {
                        let th = std::f64::consts::PI * (j as f64 + 0.5) / m as f64;
                        f[j] * (k as f64 * th).cos()
                    })
                    .sum::<f64>()
                    * 2.0
                    / m as f64;
                (if k == 0 { c / 2.0 } else { c }) as f32
            })
            .collect()
    })
}

/// The Clenshaw sum of [`erfcx_cheb`] at `a` as the SFPU computes it, in
/// `f32` with `fma_bh`: what the program's `Q` is, for the tests to hold.
pub fn erfcx_clenshaw_f32(a: f32) -> f32 {
    piece_clenshaw_f32(ERFC_MID, a)
}

/// [`erfcx_clenshaw_f32`] for any [`Piece`].
pub fn piece_clenshaw_f32(piece: Piece, a: f32) -> f32 {
    use tt_isa::numerics::fma_bh;
    let (s1, s0) = clenshaw_map(piece);
    let f = |x: f32| x.to_bits();
    let g = f32::from_bits;
    let t = g(fma_bh(f(a), f(s1), f(s0)));
    let tt = g(fma_bh(f(t), f(1.0), f(t)));
    let c = piece_cheb(piece);
    let (mut b1, mut b2) = (0.0f32, 0.0f32);
    for k in (1..=piece.deg).rev() {
        let tmp = g(fma_bh(f(1.0), f(c[k]), f(-b2)));
        let b0 = g(fma_bh(f(tt), f(b1), f(tmp)));
        b2 = b1;
        b1 = b0;
    }
    let tmp = g(fma_bh(f(1.0), f(c[0]), f(-b2)));
    g(fma_bh(f(t), f(b1), f(tmp)))
}

/// `t = s1 a + s0` maps `piece`'s interval to `[-1, 1]`.
fn clenshaw_map(piece: Piece) -> (f32, f32) {
    let s1 = 2.0 / (piece.hi - piece.lo);
    let s0 = -(piece.hi + piece.lo) / (piece.hi - piece.lo);
    (s1 as f32, s0 as f32)
}

/// `erfc(a)` for `a` in `[ERFC_LO, ERFC_HI)` (`a` in `L0`, which is spilled at
/// `spill` and `spill + 64`) into `L7`: `e^(-a^2) erfcx(a)`, the square split
/// exactly (`a^2 = hi + lo`, `lo` by an fma) so the exponential's argument is
/// exact to first order -- `e^(-hi)(1 - lo)` -- and `erfcx` by the Clenshaw
/// recurrence over [`erfcx_cheb`] (stable where Horner on the monomial form
/// of a degree-16 fit would not be). Lanes outside the interval get a finite
/// value the caller replaces. Every register scratch.
pub fn erfc_mid(p: &mut Program, spill: u32) {
    use LReg as R;
    erfc_exp(p, spill, false);
    p.load(R::L0, Format::Int32, spill);
    clenshaw(p, ERFC_MID, R::L6);
    p.load(R::L0, Format::Int32, spill + 64);
    // `E (1 - lo)` as a product of normals: `E - E lo` would form `E lo`,
    // which is denormal from `E < 2^-110` and which `SFPMAD` then drops (found
    // in 10.2e, `gelu` near `x = -13`).
    p.sub(R::ONE, R::L0, R::L0);
    p.mul(R::L7, R::L0, R::L7);
    p.mul(R::L7, R::L6, R::L7);
}

/// Veltkamp's split of `a` into `hi + lo`, twelve significant bits each, `c`
/// holding `4097` (`2^12 + 1`): every product of two halves is exact even in
/// `SFPMAD`'s 27-bit product, which is what Dekker's exact product needs here
/// -- the SFPU's multiply-add is not fused (`Miscellaneous/FMA/README.md`: the
/// product keeps four extra bits, then a sticky one), so `fma(a, b, -fl(a b))`
/// is not the product's rounding error on this hardware (found in 10.2e).
/// `a` times 4097 must be finite.
fn split12(p: &mut Program, a: LReg, hi: LReg, lo: LReg, c: LReg) {
    p.mul(a, c, lo);
    p.sub(lo, a, hi);
    p.sub(lo, hi, hi);
    p.sub(a, hi, lo);
}

/// [`split12`] of an `f32` on the host, as the SFPU computes it.
fn split12_host(a: f32) -> (f32, f32) {
    let t = a * 4097.0;
    let hi = t - (t - a);
    (hi, a - hi)
}

/// `e^(-hi)` into `L7`, `a` in `L0` spilled at `spill` and `lo` -- with
/// `a^2 = hi + lo` exactly to the last term (Dekker: `a = ah + al` split,
/// `hi = fl(ah^2 + 2 ah al)`, `lo` its exact remainder plus `al^2`; each
/// product of halves exact), plus a correction already at `spill + 128` if
/// `extra` (`2 v_hi v_lo`, where `a` is the high part of a split argument) --
/// at `spill + 64`. `a < 2^115`. Every register scratch.
fn erfc_exp(p: &mut Program, spill: u32, extra: bool) {
    use LReg as R;
    let (sa, slo) = (spill, spill + 64);
    p.store(R::L0, Format::Int32, sa);
    p.loadi(R::L6, 4097.0);
    split12(p, R::L0, R::L1, R::L2, R::L6);
    // `p2 = 2 ah al` (exact); `hi = fl(ah^2 + p2)`; `d = hi - ah^2` (exact,
    // Fast2Sum, `ah^2 >= p2`); `lo = (p2 - d) + al^2`.
    p.mul(R::L1, R::L2, R::L3);
    p.add(R::L3, R::L3, R::L3);
    p.mad(R::L1, R::L1, R::L3, R::L4);
    p.nmad(R::L1, R::L1, R::L4, R::L5);
    p.sub(R::L3, R::L5, R::L3);
    p.mad(R::L2, R::L2, R::L3, R::L3);
    if extra {
        p.load(R::L5, Format::Int32, spill + 128);
        p.add(R::L3, R::L5, R::L3);
    }
    p.store(R::L3, Format::Int32, slo);
    p.neg(R::L4, R::L2);
    exp_program(p, R::L2, R::L7);
}

/// The Clenshaw sum of `piece`'s fit at `a` (`L0`, kept) into `out`;
/// `L1..L6` but `out` and `L7` scratch.
fn clenshaw(p: &mut Program, piece: Piece, out: LReg) {
    use LReg as R;
    let (s1, s0) = clenshaw_map(piece);
    p.loadi(R::L1, s1);
    p.loadi(R::L2, s0);
    p.mad(R::L0, R::L1, R::L2, R::L1);
    p.add(R::L1, R::L1, R::L2);
    let (t, tt) = (R::L1, R::L2);
    let free: Vec<LReg> = [R::L3, R::L4, R::L5, R::L6]
        .into_iter()
        .filter(|r| *r != out)
        .collect();
    let (mut b1, mut b2, tmp) = (free[0], free[1], free[2]);
    p.mov(R::ZERO, b1);
    p.mov(R::ZERO, b2);
    let c = piece_cheb(piece);
    for k in (1..=piece.deg).rev() {
        p.loadi(tmp, c[k]);
        p.sub(tmp, b2, tmp);
        // `b0` into `b2`'s register; then the names rotate.
        p.mad(tt, b1, tmp, b2);
        std::mem::swap(&mut b1, &mut b2);
    }
    p.loadi(tmp, c[0]);
    p.sub(tmp, b2, tmp);
    p.mad(t, b1, tmp, out);
}

/// `erf`'s Taylor series below [`ERFC_LO`]: `x P(x^2)`, `P` to `t^8` (the next
/// term under `2.5e-11` of the sum at `|x| = 1/2`), in Horner form. `x` in
/// `L0` (kept), the result in `d`, `L1..L5` scratch.
fn erf_taylor(p: &mut Program, d: LReg) {
    use LReg as R;
    p.mul(R::L0, R::L0, R::L1);
    let two_rtpi = 2.0 / std::f64::consts::PI.sqrt();
    let mut fact = 1.0f64;
    let coeff: Vec<f32> = (0..=8)
        .map(|n| {
            if n > 0 {
                fact *= n as f64;
            }
            (two_rtpi * if n % 2 == 0 { 1.0 } else { -1.0 } / (fact * (2 * n + 1) as f64)) as f32
        })
        .collect();
    p.loadi(d, coeff[8]);
    for n in (0..8).rev() {
        p.loadi(R::L2, coeff[n]);
        p.mad(d, R::L1, R::L2, d);
    }
    p.mul(d, R::L0, d);
}

/// [`erf_program`]'s derived bound, relative.
pub const ERF_BOUND: f64 = 10.5 / 16_777_216.0;

/// [`erfc_mid`]'s relative bound (and its tail piece's): `EXP_BOUND` (2.2u),
/// `lo`'s correction (u), the measured evaluation and fit together (under
/// 5.25u: 5.07u on [`ERFC_MID`], 3.48u on [`ERFC_TAIL`]), two roundings (2u).
pub const ERFC_BOUND: f64 = 10.5 / 16_777_216.0;

/// `erf x` of `x` (in `L0`, raw bits, spilled at `spill..spill + 192`) into `L7`.
///
/// Below `|x| = 1/2` the Taylor series ([`erf_taylor`]); from there to 3.92,
/// `1 - erfc(|x|)` with the sign ([`erfc_mid`]: `erfc <= 0.48`, so its
/// relative error reaches `erf` scaled by `erfc/erf <= 0.92`); from 3.92 `±1`
/// (`erfc(3.92) < 2^-24`). Error, in `u = 2^-24`: the series' Horner roundings
/// are damped by `t = x^2 <= 1/4` to under `1.2u`, the square and the final
/// product `u` each; the tail's `erfc` is within `EXP_BOUND` (2.2u, the
/// exponential of an exact argument), `u` (`lo`'s first-order correction drops
/// `lo^2/2 < 2^-46`), the Clenshaw sum's measured error (under 5u, mostly the
/// interval map `t = s1 a + s0` rounded) and the fit's (under 0.25u; both over
/// every float, `every_fit_and_its_evaluation_are_within_their_parts`), and
/// two roundings, `2u` -- [`ERFC_BOUND`] `= 10.5u` -- of which `erf` takes at
/// most `0.92`, plus the subtraction's `u`: under [`ERF_BOUND`] `= 10.5u` in
/// all. A denormal flushes
/// (its `erf` is `+0`/`-0`); a NaN stays one.
pub fn erf_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let sx = spill + 128;
    p.store(R::L0, Format::Int32, sx);
    p.loadi_bits(R::L1, 0x7fff_ffff);
    p.and(R::L0, R::L1, R::L0);
    erfc_mid(p, spill);
    // `1 - erfc`, magnitude, into `L7`.
    p.sub(R::ONE, R::L7, R::L7);
    p.load(R::L0, Format::Int32, sx);
    erf_taylor(p, R::L6);
    p.copy_sign(R::L7, R::L0, R::L5);
    p.mov(R::L5, R::L7);
    p.loadi_bits(R::L1, 0x7fff_ffff);
    p.and(R::L0, R::L1, R::L2);
    p.loadi(R::L1, ERFC_LO as f32);
    p.if_(Cond::Less(R::L2, R::L1), |p| p.mov(R::L6, R::L7));
    p.loadi(R::L1, ERFC_HI as f32);
    p.if_(Cond::LessEq(R::L1, R::L2), |p| {
        p.copy_sign(R::ONE, R::L0, R::L7)
    });
    p.loadi_bits(R::L1, 0x7f80_0000);
    p.if_(Cond::Less(R::L1, R::L2), |p| {
        p.loadi_bits(R::L7, 0x7fc0_0000)
    });
}

/// [`gelu_program`]'s derived bound for `gelu`, relative.
pub const GELU_BOUND: f64 = 12.5 / 16_777_216.0;

/// The bound on `gelu_backward(x, g)`'s error, absolute: the derivative `Phi +
/// x phi` can cancel (it is zero near `x = -0.75`), so its bound is on each
/// term's magnitude -- `GELU_BOUND` of `Phi`, `5.2u` of `|x| phi` (the
/// exponential of an exact argument, `lo`'s correction, the constant's and the
/// product's roundings), a rounding of the sum -- times `|g|`, and the final
/// product's rounding.
pub fn gelu_backward_bound(x: f32, g: f32) -> f64 {
    let u = 1.0 / 16_777_216.0;
    let (x, g) = (x as f64, (g as f64).abs());
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let d = cdf + x * phi;
    (GELU_BOUND * cdf + 5.2 * u * x.abs() * phi + u * d.abs()) * g + u * (g * d).abs()
}

/// Flex's `gelu` (`0.5 x (1 + erf(x/sqrt 2))`) of `x` (in `L0`, raw bits) into
/// `L7` -- or with `grad` (a `Dst` row) its `gelu_backward`, `g (Phi + x
/// phi)`. Spills at `spill..spill + 256`.
///
/// `2 Phi = 1 + erf(v)`, `v = x/sqrt 2`, each range by the branch that does
/// not cancel: below `|v| = 1/2` `1 + erf(v)` by the series; above it
/// `erfc(|v|)` (for `v < 0`) or `2 - erfc(v)`, `erfc` by [`ERFC_MID`] to 3.92
/// and [`ERFC_TAIL`] to 9.3 -- so the far negative side, where Flex's own `1 +
/// erf` cancels, is relatively accurate. `v` is split, `v_hi + v_lo` with the
/// constant's own error in `v_lo`, and `e^(-v^2)` takes `2 v_hi v_lo` in its
/// first-order correction: the argument is exact to `2^-47`, which the tail's
/// `e^(-v^2)` would otherwise amplify by `2 v^2`. Error of `gelu`, relative, in
/// `u`: `erfc` within `ERFC_BOUND` (10.5u; the fits evaluated at `v_hi`, `u`
/// more), its complement `2 - erfc` within a third of that, the series path
/// `4.5u`, the product `h 2Phi` (`h = x/2`, exact) `u`: under [`GELU_BOUND`] `=
/// 12.5u`. The derivative's density is `e^(-v^2)/sqrt(2 pi)`, from the same
/// exponential ([`gelu_backward_bound`]). `gelu(-inf)` is NaN as Flex's is (`-inf
/// * 0`), `+inf` itself, a NaN stays one.
pub fn gelu_exp_program(p: &mut Program, spill: u32) {
    use LReg as R;
    let (s_a, s_lo, s_x2, s_x) = (spill, spill + 64, spill + 128, spill + 192);
    let c = std::f32::consts::FRAC_1_SQRT_2;
    let c_tail = (std::f64::consts::FRAC_1_SQRT_2 - c as f64) as f32;
    let (c_hi, c_lo) = split12_host(c);
    p.store(R::L0, Format::Int32, s_x);
    // Dekker's product `x c = v_hi + err`, `x` split (`x_hi`, `x_lo`), `c`'s
    // halves from the host; `v_lo = err + x c_tail`, the constant's own error.
    p.loadi(R::L6, 4097.0);
    split12(p, R::L0, R::L1, R::L2, R::L6);
    p.loadi(R::L6, c);
    p.mul(R::L0, R::L6, R::L3);
    p.neg(R::L3, R::L4);
    p.loadi(R::L6, c_hi);
    p.mad(R::L1, R::L6, R::L4, R::L4);
    p.loadi(R::L5, c_lo);
    p.mad(R::L1, R::L5, R::L4, R::L4);
    p.mad(R::L2, R::L6, R::L4, R::L4);
    p.mad(R::L2, R::L5, R::L4, R::L4);
    p.loadi(R::L6, c_tail);
    p.mad(R::L0, R::L6, R::L4, R::L4);
    // `v_hi` in `L3`, `v_lo` in `L4`; the correction `2 v_hi v_lo`.
    p.mul(R::L3, R::L4, R::L5);
    p.add(R::L5, R::L5, R::L5);
    p.store(R::L5, Format::Int32, s_x2);
    p.loadi_bits(R::L5, 0x7fff_ffff);
    p.and(R::L3, R::L5, R::L0);
    // Beyond 9.3 (and where `a 4097` would overflow) `e^(-v^2)` is `0`: the
    // split's arithmetic is then not to be trusted, and nothing needs it.
    p.loadi(R::L5, ERFC_TAIL.hi as f32);
    p.if_(Cond::LessEq(R::L5, R::L0), |p| p.mov(R::ZERO, R::L0));
    erfc_exp(p, s_a, true);
    // `E = e^(-hi) (1 - lo)` into `L7`.
    p.load(R::L1, Format::Int32, s_lo);
    // `E (1 - lo)` as a product of normals: `E - E lo` would form `E lo`,
    // which is denormal from `E < 2^-110` and which `SFPMAD` then drops (found
    // in 10.2e, `gelu` near `x = -13`).
    p.sub(R::ONE, R::L1, R::L1);
    p.mul(R::L7, R::L1, R::L7);
    p.load(R::L0, Format::Int32, s_x);
    p.loadi(R::L1, std::f32::consts::FRAC_1_SQRT_2);
    p.mul(R::L0, R::L1, R::L1);
    p.loadi_bits(R::L2, 0x7fff_ffff);
    p.and(R::L1, R::L2, R::L1);
    p.loadi(R::L2, ERFC_TAIL.hi as f32);
    p.if_(Cond::LessEq(R::L2, R::L1), |p| p.mov(R::ZERO, R::L7));
    // A NaN sorts above 9.3 too: it stays one.
    p.loadi_bits(R::L2, 0x7f80_0000);
    p.if_(Cond::Less(R::L2, R::L1), |p| {
        p.loadi_bits(R::L7, 0x7fc0_0000)
    });
}

/// [`gelu_exp_program`]'s output, `e` in `L7` -- with `x` in `L0`, as it
/// leaves them -- to `gelu` into `L7`, or with `grad` (a `Dst` row) to
/// `gelu_backward`. Spills at `spill..spill + 256`, [`gelu_exp_program`]'s.
pub fn gelu_program(p: &mut Program, spill: u32, grad: Option<u32>) {
    use LReg as R;
    let (s_a, s_e, s_q, s_x) = (spill, spill + 64, spill + 128, spill + 192);
    let c = std::f32::consts::FRAC_1_SQRT_2;
    p.store(R::L0, Format::Int32, s_x);
    p.loadi(R::L1, c);
    p.mul(R::L0, R::L1, R::L1);
    p.loadi_bits(R::L3, 0x7fff_ffff);
    p.and(R::L1, R::L3, R::L0);
    p.store(R::L0, Format::Int32, s_a);
    p.store(R::L7, Format::Int32, s_e);
    // Both fits at `a = |v_hi|`; the first waits in a slot.
    clenshaw(p, ERFC_MID, R::L6);
    p.store(R::L6, Format::Int32, s_q);
    clenshaw(p, ERFC_TAIL, R::L6);
    p.loadi(R::L1, ERFC_MID.hi as f32);
    p.if_(Cond::Less(R::L0, R::L1), |p| {
        p.load(R::L6, Format::Int32, s_q)
    });
    // `Q` waits in its slot, `E` in `L7` (and its slot).
    p.store(R::L6, Format::Int32, s_q);
    // The series at `v_hi` (`L0`), into `L6`: `2 Phi` below `|v| = 1/2`.
    p.load(R::L1, Format::Int32, s_x);
    p.loadi(R::L2, c);
    p.mul(R::L1, R::L2, R::L0);
    erf_taylor(p, R::L6);
    p.add(R::L6, R::ONE, R::L6);
    // `v >= 1/2`: `2 - E Q`, and `2` from 3.92.
    p.load(R::L7, Format::Int32, s_e);
    p.load(R::L5, Format::Int32, s_q);
    p.loadi_bits(R::L3, 0x7fff_ffff);
    p.and(R::L0, R::L3, R::L4);
    p.loadi(R::L3, ERFC_MID.lo as f32);
    p.if_(Cond::LessEq(R::L3, R::L4), |p| {
        p.if_(Cond::Gte0(R::L0), |p| {
            p.mul(R::L7, R::L5, R::L2);
            p.loadi(R::L3, 2.0);
            p.sub(R::L3, R::L2, R::L6);
        })
    });
    p.loadi(R::L3, ERFC_MID.hi as f32);
    p.if_(Cond::LessEq(R::L3, R::L4), |p| {
        p.if_(Cond::Gte0(R::L0), |p| p.loadi(R::L6, 2.0))
    });
    // `x` into `L0`, the branches' flag `a >= 1/2 and v < 0` kept as `L4`
    // (`a`) and `L1` (`v`'s sign, from `x`).
    p.load(R::L0, Format::Int32, s_x);
    p.loadi(R::L1, 0.5);
    p.mul(R::L0, R::L1, R::L2);
    let negative_far = |p: &mut Program, then: &dyn Fn(&mut Program)| {
        p.loadi(R::L3, ERFC_MID.lo as f32);
        p.if_(Cond::LessEq(R::L3, R::L4), |p| {
            p.if_(Cond::Lt0(R::L0), then)
        });
    };
    let beyond = |p: &mut Program, then: &dyn Fn(&mut Program)| {
        p.loadi(R::L3, ERFC_TAIL.hi as f32);
        p.if_(Cond::LessEq(R::L3, R::L4), |p| {
            p.if_(Cond::Lt0(R::L0), then)
        });
    };
    match grad {
        None => {
            // `(x/2) 2Phi`; on the negative side `((x/2) E) Q`, which no
            // intermediate underflows (`E Q` alone would, from `x` near -13).
            p.mul(R::L2, R::L6, R::L7);
            negative_far(p, &|p| {
                p.load(R::L6, Format::Int32, s_e);
                p.mul(R::L2, R::L6, R::L6);
                p.mul(R::L6, R::L5, R::L7);
            });
            beyond(p, &|p| p.mul(R::L2, R::ZERO, R::L7));
        }
        Some(g) => {
            // `Phi + x phi`, `phi = E / sqrt(2 pi)`; on the negative side
            // `E (Q/2 + x / sqrt(2 pi))`, for the same reason.
            let k = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) as f32;
            p.mul(R::L6, R::L1, R::L6);
            p.loadi(R::L3, k);
            p.mul(R::L7, R::L3, R::L2);
            p.mad(R::L0, R::L2, R::L6, R::L6);
            negative_far(p, &|p| {
                p.mul(R::L5, R::L1, R::L5);
                p.loadi(R::L3, k);
                p.mad(R::L0, R::L3, R::L5, R::L5);
                p.load(R::L7, Format::Int32, s_e);
                p.mul(R::L7, R::L5, R::L6);
            });
            beyond(p, &|p| p.mov(R::ZERO, R::L6));
            // `±inf`: Flex's `x * pdf` is `inf * 0`, a NaN.
            p.loadi_bits(R::L3, 0x7fff_ffff);
            p.and(R::L0, R::L3, R::L4);
            p.loadi_bits(R::L3, 0x7f80_0000);
            p.if_(Cond::LessEq(R::L3, R::L4), |p| {
                p.loadi_bits(R::L6, 0x7fc0_0000)
            });
            p.load(R::L1, Format::Fp32, g);
            p.mul(R::L1, R::L6, R::L7);
        }
    }
}

/// `n`, a two's-complement `i32`, as FP32 in place, rounding to nearest even
/// as `as f32` does: the magnitude made sign-magnitude for `SFPCAST`, the sign
/// put back; `i32::MIN`, whose magnitude has no 31-bit form, by name. `t`, `c`
/// scratch.
fn i32_to_float(p: &mut Program, n: LReg, t: LReg, c: LReg) {
    p.mov(n, t);
    p.if_(Cond::Lt0(n), |p| p.isub_from(LReg::ZERO, t));
    p.sm32_to_float(t, t);
    p.if_(Cond::Lt0(n), |p| p.neg(t, t));
    p.loadi_bits(c, 0x8000_0000);
    p.xor(n, c);
    p.if_(Cond::Eq0(c), |p| p.loadi(t, -2_147_483_648.0));
    p.mov(t, n);
}

/// `pow`'s bound at `(x, y)`, relative: the chain computes `e^z`,
/// `z = y ln|x|`, and `ln|x|` (within `LOG_BOUND`) and the product's rounding
/// (`2^-24`) move `z` by `|z| (LOG_BOUND + 2^-24)` -- which `e^z` turns into
/// that relative error -- on top of `exp`'s own `EXP_BOUND`; the factor
/// `1.01` covers the products of the small terms. So a large `|z|` (a result
/// near the range's ends) carries a large bound: `4.3e-5` at `|z| = 88`.
pub fn pow_bound(x: f32, y: f32) -> f64 {
    // `0 ln 0` and the like: a special value, exact.
    let z = (y as f64 * (x.abs() as f64).ln()).abs();
    let z = if z.is_finite() { z } else { 0.0 };
    z * (LOG_BOUND + 1.0 / 16_777_216.0) * 1.01 + EXP_BOUND
}

/// Where [`pow_program`] reads `y`, for one row group.
#[derive(Copy, Clone, Debug)]
enum PowY {
    Row(u32),
    IntRow(u32),
    Scalar(f32),
}

impl PowY {
    fn at(self, o: u32) -> Self {
        match self {
            PowY::Row(r) => PowY::Row(r + o),
            PowY::IntRow(r) => PowY::IntRow(r + o),
            s => s,
        }
    }

    /// `y` into `d` as FP32 raw bits; `t`, `c` scratch.
    fn load(self, p: &mut Program, d: LReg, t: LReg, c: LReg) {
        match self {
            PowY::Row(r) => p.load(d, Format::Int32, r),
            PowY::IntRow(r) => {
                p.load(d, Format::Int32, r);
                i32_to_float(p, d, t, c);
            }
            PowY::Scalar(s) => p.loadi_bits(d, s.to_bits()),
        }
    }
}

/// `powf(x, y)` of `x` at `Dst` row `x_row` into `L7` (raw bits): `e^(y
/// ln|x|)` by [`log_program`], a multiply and [`exp_program`] -- one program
/// since the runner repeats a long row loop (X8) -- then [`pow_fix`]'s
/// special values over it; within [`pow_bound`].
fn pow_program(p: &mut Program, x_row: u32, y: PowY) {
    use LReg as R;
    p.load(R::L0, Format::Fp32, x_row);
    p.abs(R::L0, R::L0);
    log_program(p, R::L0, R::L7);
    // A denormal `y` is flushed as the multiply's operand (numerics row D).
    y.load(p, R::L1, R::L2, R::L3);
    p.mul(R::L7, R::L1, R::L0);
    exp_program(p, R::L0, R::L7);
    p.load(R::L0, Format::Int32, x_row);
    y.load(p, R::L1, R::L2, R::L3);
    pow_fix(p);
}

/// `powf`'s special values over `e = e^(y ln|x|)` ([`pow_program`]'s general
/// case): `x` in `L0`, `y` in `L1`, `e` in `L7` (raw bits), the result in `L7`.
///
/// Each scope over the last: a negative finite non-zero base gives NaN for a
/// non-integer `y` and is negated for an odd one (`-inf` and `-0` take only
/// the sign: their magnitudes, `inf` and `0` from `e^(±inf)`, are already
/// right); `y = ±inf` is `1` at `|x| = 1`, else `+inf` where `|x| > 1` meets
/// `y > 0` or `|x| < 1` meets `y < 0`, else `+0`; a NaN anywhere NaN; `x = 1`
/// gives `1`, and `y = ±0` gives `1`, whatever the other. `y` is an integer
/// where `|y| >= 2^23`, or where `|y| + 2^23` rounded back is `|y|`; odd where
/// the integer's last bit is (`|y| < 2^24`).
pub fn pow_fix(p: &mut Program) {
    use LReg as R;
    let (x, y, res, ax, ay, flags, t, c) = (R::L0, R::L1, R::L7, R::L2, R::L3, R::L4, R::L5, R::L6);
    p.loadi_bits(c, 0x7fff_ffff);
    p.and(x, c, ax);
    p.and(y, c, ay);
    // `flags`: 0 not an integer, 1 an even integer, 3 an odd one.
    p.mov(R::ZERO, flags);
    p.loadi_bits(c, 0x4b00_0000); // 2^23
    p.if_(Cond::Less(ay, c), |p| {
        p.add(ay, c, t);
        p.sub(t, c, c);
        p.if_(Cond::LessEq(c, ay), |p| {
            p.if_(Cond::LessEq(ay, c), |p| {
                p.loadi_bits(flags, 1);
                p.loadi_bits(c, 1);
                p.and(t, c, c);
                p.if_(Cond::Ne0(c), |p| p.loadi_bits(flags, 3));
            })
        });
    });
    p.loadi_bits(c, 0x4b00_0000);
    p.if_(Cond::LessEq(c, ay), |p| {
        p.loadi_bits(t, 0x7f80_0000);
        p.if_(Cond::Less(ay, t), |p| {
            p.loadi_bits(flags, 1);
            p.loadi_bits(t, 0x4b80_0000); // 2^24
            p.if_(Cond::Less(ay, t), |p| {
                p.loadi_bits(c, 1);
                p.and(ay, c, c);
                p.if_(Cond::Ne0(c), |p| p.loadi_bits(flags, 3));
            });
        });
    });
    // A negative base.
    p.loadi_bits(c, 0x7f80_0000);
    p.if_(Cond::Lt0(x), |p| {
        p.if_else(
            Cond::Eq0(flags),
            // Not an integer: NaN, unless the base is `-0` or `-inf`.
            |p| {
                p.if_(Cond::Ne0(ax), |p| {
                    p.if_(Cond::Less(ax, c), |p| p.loadi_bits(res, 0x7fc0_0000))
                })
            },
            |p| {
                p.loadi_bits(t, 2);
                p.and(flags, t, t);
                p.if_(Cond::Ne0(t), |p| p.neg(res, res));
            },
        )
    });
    // `y = ±inf`.
    p.if_(Cond::LessEq(c, ay), |p| {
        p.if_(Cond::LessEq(ay, c), |p| {
            p.mov(R::ZERO, res);
            p.if_(Cond::Lt0(y), |p| {
                p.if_(Cond::Less(ax, R::ONE), |p| p.mov(c, res))
            });
            p.if_(Cond::Gte0(y), |p| {
                p.if_(Cond::Less(R::ONE, ax), |p| p.mov(c, res))
            });
            p.if_(Cond::LessEq(ax, R::ONE), |p| {
                p.if_(Cond::LessEq(R::ONE, ax), |p| p.mov(R::ONE, res))
            });
        })
    });
    // NaNs.
    p.if_(Cond::Less(c, ax), |p| p.loadi_bits(res, 0x7fc0_0000));
    p.if_(Cond::Less(c, ay), |p| p.loadi_bits(res, 0x7fc0_0000));
    // `1^y` and `x^0`.
    p.if_(Cond::LessEq(x, R::ONE), |p| {
        p.if_(Cond::LessEq(R::ONE, x), |p| p.mov(R::ONE, res))
    });
    p.if_(Cond::Eq0(ay), |p| p.mov(R::ONE, res));
}

/// What operands `kind` takes, if the SFPU has it.
pub fn operands(kind: u32) -> Option<Operands> {
    if super::approx::is_approx(kind) {
        return Some(Operands::Unary);
    }
    if (kind_sfpu::HARDWARE_ROUND..kind_sfpu::HARDWARE_ROUND + 6).contains(&kind) {
        return Some(Operands::Unary);
    }
    if let Some((op, scalar)) = super::integer::operation(kind) {
        return Some(if scalar || op == 14 {
            Operands::Unary
        } else {
            Operands::Binary
        });
    }
    if matches!(kind, kind_sfpu::ROUND..=kind_sfpu::POW2_FLOOR_INV) {
        return Some(Operands::Unary);
    }
    Some(match kind {
        kind::ADD
        | kind_sfpu::ADD_RELU
        | kind::SUB
        | kind::MUL
        | kind::RELU_BACKWARD
        | kind_sfpu::DIV
        | kind_sfpu::BOOL_AND
        | kind_sfpu::BOOL_OR
        | kind_sfpu::BOOL_XOR
        | kind_sfpu::EQ..=kind_sfpu::LE
        | kind_sfpu::MASK_FILL
        | kind_sfpu::INT_MASK_FILL
        | kind_sfpu::BOOL_MASK_FILL
        | kind_sfpu::PRELU
        | kind_sfpu::POW
        | kind_sfpu::POW_I
        | kind_sfpu::SIGMOID_BACKWARD
        | kind_sfpu::GELU_BACKWARD
        | kind_sfpu::LOG_SIGMOID_BACKWARD
        | kind_sfpu::ATAN2 => Operands::Binary,
        kind_sfpu::MASK_WHERE | kind_sfpu::INT_MASK_WHERE | kind_sfpu::BOOL_MASK_WHERE => {
            Operands::Ternary
        }
        kind_sfpu::REM => Operands::Binary,
        kind_sfpu::REM_S => Operands::Unary,
        kind::MUL_SCALAR
        | kind::ADD_SCALAR
        | kind::RELU
        | kind_sfpu::RECIP
        | kind_sfpu::DIV_SCALAR
        | kind_sfpu::EXP
        | kind_sfpu::LOG
        | kind_sfpu::BOOL_NOT
        | kind_sfpu::NEG..=kind_sfpu::HARD_SIGMOID
        | kind_sfpu::EQ_S..=kind_sfpu::IS_INF
        | kind_sfpu::FILL..=kind_sfpu::LOG1P
        | kind_sfpu::POW_S
        | kind_sfpu::I32_TO_F32
        | kind_sfpu::INDEX_TO_I32
        | kind_sfpu::BOOL_TO_F32
        | kind_sfpu::BOOL_TO_I32
        | kind_sfpu::INT_ABS
        | kind_sfpu::EXPM1
        | kind_sfpu::SIGMOID
        | kind_sfpu::TANH
        | kind_sfpu::ERF
        | kind_sfpu::GELU
        | kind_sfpu::SINH..=kind_sfpu::LOG_SIGMOID
        | kind_sfpu::SIN..=kind_sfpu::ATAN
        | kind_sfpu::ASIN
        | kind_sfpu::ACOS => Operands::Unary,
        kind::ADD_ROW => Operands::RowBroadcast,
        _ => return None,
    })
}

/// `q = a / b` into `q`, from `a`, `b` and `y = 1/b` (`Program::recip`): the
/// product, then one correction `q += (a - b*q) * y` in the lanes where the
/// product is finite and not zero. The product is within `e_y + 2^-24`
/// (`< 1.8e-7`) of `a/b`; the remainder `a - b*q` is computed with one
/// rounding (`SFPMAD`) and is that error times `a`; adding `remainder * y`
/// leaves `a/b * (1 + O(e_y^2))` before the last rounding -- so within half an
/// ulp plus `~1e-13` relative, at most one ulp from the correct rounding.
/// `r`, `t` are scratch; `inf` holds `+inf`.
#[allow(clippy::too_many_arguments)]
fn divide(p: &mut Program, a: LReg, b: LReg, y: LReg, q: LReg, r: LReg, t: LReg, inf: LReg) {
    p.mul(a, y, q);
    p.abs(q, t);
    p.if_(Cond::Less(LReg::ZERO, t), |p| {
        p.if_(Cond::Less(t, inf), |p| {
            p.nmad(b, q, a, r);
            p.mad(r, y, q, q);
        })
    });
}

/// Where `|b| > 2^100`, `a` and `b` both times `2^-64` (exact; the quotient
/// unchanged): the reciprocal and the correction then stay in range. An `a`
/// that flushes doing so had a quotient under `2^-162`, which flushes anyway.
/// `t`, `c` scratch.
fn scale_large_divisor(p: &mut Program, a: LReg, b: LReg, t: LReg, c: LReg) {
    p.abs(b, t);
    p.loadi_bits(c, 0x7180_0000); // 2^100
    p.if_(Cond::Less(c, t), |p| {
        p.loadi_bits(c, 0x1f80_0000); // 2^-64
        p.mul(a, c, a);
        p.mul(b, c, b);
    });
}

/// How a binary op's second operand meets the first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Broadcast {
    /// The same shape.
    None,
    /// `[1, cols]`: its one row added (or subtracted, ...) to every row.
    Row,
    /// `[rows, 1]`: its one column to every column.
    Col,
}

/// The kinds that take a broadcast second operand.
pub fn broadcasts(kind: u32) -> bool {
    if matches!(kind, 0x170..=0x17d | 0x18e..=0x18f) {
        return true;
    }
    matches!(
        kind,
        kind::ADD
            | kind_sfpu::ADD_RELU
            | kind::SUB
            | kind::MUL
            | kind_sfpu::DIV
            | kind_sfpu::BOOL_AND
            | kind_sfpu::BOOL_OR
            | kind_sfpu::BOOL_XOR
            | kind_sfpu::EQ
            ..=kind_sfpu::LE
                | kind_sfpu::MASK_FILL
                | kind_sfpu::INT_MASK_FILL
                | kind_sfpu::BOOL_MASK_FILL
                | kind_sfpu::PRELU
    )
}

/// `out = a (kind) b` for one row group, `a` in `L0`, `b` from `Dst` at `b_at`,
/// the result stored at `out_at`. `DIV` needs `f32::MAX` in `L6` and `+inf`
/// in `L7` ([`binary_constants`]).
fn binary_body(p: &mut Program, kind: u32, a_at: u32, b_at: u32, out_at: u32) {
    // A `Bool` is `0`/`1` as an integer -- a denormal to FP32's load and
    // store, which would flush it -- so the logic ops move raw bits.
    // The exact ops of S2 move raw bits too: an FP32 store would flush a
    // denormal Flex keeps.
    let fmt = match kind {
        kind::ADD | kind_sfpu::ADD_RELU | kind::SUB | kind::MUL | kind_sfpu::DIV => Format::Fp32,
        _ => Format::Int32,
    };
    p.load(LReg::L0, fmt, a_at);
    p.load(LReg::L1, fmt, b_at);
    if let Some((op, _)) = super::integer::operation(kind) {
        super::integer::body(p, op);
        p.store(LReg::L2, Format::Int32, out_at);
        return;
    }
    let out = match kind {
        kind_sfpu::BOOL_AND => {
            p.and(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::BOOL_OR => {
            p.or(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::BOOL_XOR => {
            p.mov(LReg::L0, LReg::L2);
            p.xor(LReg::L1, LReg::L2);
            LReg::L2
        }
        k if compare_of(k).is_some() => {
            compare_body(p, k, true);
            LReg::L2
        }
        // As `LEAKY_RELU`, the slope `b`'s lane.
        kind_sfpu::PRELU => {
            p.mul(LReg::L0, LReg::L1, LReg::L2);
            p.and(LReg::L0, LReg::L4, LReg::L3);
            p.if_(Cond::Gte0(LReg::L0), |p| {
                p.if_(Cond::LessEq(LReg::L3, LReg::L5), |p| {
                    p.mov(LReg::L0, LReg::L2)
                })
            });
            p.if_(Cond::Eq0(LReg::L3), |p| p.mov(LReg::L0, LReg::L2));
            LReg::L2
        }
        // `mask ? value : a`, the value in `L3` (`binary_constants`).
        kind_sfpu::MASK_FILL | kind_sfpu::INT_MASK_FILL | kind_sfpu::BOOL_MASK_FILL => {
            p.mov(LReg::L0, LReg::L2);
            p.if_(Cond::Ne0(LReg::L1), |p| p.mov(LReg::L3, LReg::L2));
            LReg::L2
        }
        kind::ADD => {
            p.add(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::ADD_RELU => {
            p.add(LReg::L0, LReg::L1, LReg::L2);
            p.mov(LReg::ZERO, LReg::L3);
            p.if_(Cond::Less(LReg::ZERO, LReg::L2), |p| {
                p.if_(Cond::Less(LReg::L2, LReg::L5), |p| {
                    p.mov(LReg::L2, LReg::L3)
                })
            });
            LReg::L3
        }
        kind::SUB => {
            p.sub(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind::MUL => {
            p.mul(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::DIV => {
            scale_large_divisor(p, LReg::L0, LReg::L1, LReg::L3, LReg::L4);
            p.recip(LReg::L1, LReg::L2, LReg::L3, LReg::L4, LReg::L6);
            divide(
                p,
                LReg::L0,
                LReg::L1,
                LReg::L2,
                LReg::L5,
                LReg::L3,
                LReg::L4,
                LReg::L7,
            );
            LReg::L5
        }
        _ => unreachable!("kind {kind:#x} is not a broadcastable binary op"),
    };
    p.store(out, fmt, out_at);
}

/// The constants [`binary_body`] needs for `kind`, loaded once.
fn binary_constants(p: &mut Program, kind: u32, scalars: [f32; 2]) {
    if kind == kind_sfpu::ADD_RELU {
        p.loadi_bits(LReg::L5, FIRST_NAN);
    }
    if kind == kind_sfpu::DIV {
        p.loadi_bits(LReg::L6, f32::MAX.to_bits());
        p.loadi_bits(LReg::L7, 0x7f80_0000);
    }
    if compare_of(kind).is_some() || kind == kind_sfpu::PRELU {
        compare_constants(p);
    }
    if matches!(
        kind,
        kind_sfpu::MASK_FILL | kind_sfpu::INT_MASK_FILL | kind_sfpu::BOOL_MASK_FILL
    ) {
        p.loadi_bits(LReg::L3, scalars[0].to_bits());
    }
}

/// `L4 = 0x7fff_ffff` (the magnitude mask), `L5 = +inf`, `L6 = 1` (a `Bool`'s
/// true): what the IEEE comparisons use.
fn compare_constants(p: &mut Program) {
    p.loadi_bits(LReg::L4, 0x7fff_ffff);
    p.loadi_bits(LReg::L5, 0x7f80_0000);
    p.loadi_bits(LReg::L6, 1);
}

/// `L2 = x (cmp) y` as a `Bool`, by IEEE, `x` in `L0` and `y` in `L1` as raw
/// bits (clobbered), with [`compare_constants`] loaded. `SFPGT`/`SFPLE` order
/// sign-magnitude (`-0 < +0`, NaNs ranked), so both are first made canonical
/// -- a zero of either sign `+0` -- and a lane where either is a NaN takes the
/// unordered answer (false; true for `NE`). `y_may_be_nan` is false where the
/// caller has settled `y` (a scalar, on the host). Bit and flag operations
/// only: no arithmetic, so a denormal compares as itself, as on the host.
fn compare_body(p: &mut Program, kind: u32, y_may_be_nan: bool) {
    let cmp = compare_of(kind).expect("a comparison");
    let (x, y, out, ax, ay, one) = (LReg::L0, LReg::L1, LReg::L2, LReg::L3, LReg::L7, LReg::L6);
    let (mask, inf) = (LReg::L4, LReg::L5);
    p.and(x, mask, ax);
    p.if_(Cond::Eq0(ax), |p| p.mov(LReg::ZERO, x));
    p.and(y, mask, ay);
    p.if_(Cond::Eq0(ay), |p| p.mov(LReg::ZERO, y));
    let (unordered, holds) = if cmp == kind_sfpu::NE {
        (one, LReg::ZERO)
    } else {
        (LReg::ZERO, one)
    };
    p.mov(unordered, out);
    let ordered = move |p: &mut Program| match cmp {
        kind_sfpu::EQ | kind_sfpu::NE => p.if_(Cond::LessEq(x, y), |p| {
            p.if_(Cond::LessEq(y, x), |p| p.mov(holds, out))
        }),
        kind_sfpu::GT => p.if_(Cond::Less(y, x), |p| p.mov(holds, out)),
        kind_sfpu::GE => p.if_(Cond::LessEq(y, x), |p| p.mov(holds, out)),
        kind_sfpu::LT => p.if_(Cond::Less(x, y), |p| p.mov(holds, out)),
        _ => p.if_(Cond::LessEq(x, y), |p| p.mov(holds, out)),
    };
    p.if_(Cond::LessEq(ax, inf), |p| {
        if y_may_be_nan {
            p.if_(Cond::LessEq(ay, inf), ordered);
        } else {
            ordered(p);
        }
    });
}

/// `bits` with a zero of either sign made `+0`: how a scalar meets the
/// canonical lanes of [`compare_body`].
fn canonical(bits: u32) -> u32 {
    if bits & 0x7fff_ffff == 0 {
        0
    } else {
        bits
    }
}

/// The program for an exact S2 op (10.2c) over a tile, or `None` for a kind
/// that is not one -- or a `CLAMP` whose bounds `f32::clamp` would refuse.
fn exact_program(p: &mut Program, kind: u32, [s, s2]: [f32; 2]) -> Option<Operands> {
    use kind_sfpu::*;
    let (x, out, ax) = (LReg::L0, LReg::L2, LReg::L3);
    let (mask, inf) = (LReg::L4, LReg::L5);
    let (load, store) = (
        move |p: &mut Program, o: u32| p.load(x, Format::Int32, A_ROW + o),
        move |p: &mut Program, o: u32| p.store(out, Format::Int32, OUT_ROW + o),
    );
    match kind {
        // Two's complement `|x|`: negate the negative lanes, wrapping, so
        // `i32::MIN` stays itself (`i32::wrapping_abs`).
        INT_ABS => {
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mov(x, out);
                p.if_(Cond::Lt0(x), |p| p.isub_from(LReg::ZERO, out));
                store(p, o);
            });
        }
        NEG | ABS => {
            p.loadi_bits(mask, 0x7fff_ffff);
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                if kind == NEG {
                    p.neg(x, out);
                } else {
                    p.and(x, mask, out);
                }
                store(p, o);
            });
        }
        SIGN => {
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi(LReg::L6, 1.0);
            p.loadi(LReg::L7, -1.0);
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                // A NaN is itself.
                p.mov(x, out);
                p.and(x, mask, ax);
                p.if_(Cond::LessEq(ax, inf), |p| {
                    p.if_else(
                        Cond::Eq0(ax),
                        |p| p.mov(LReg::ZERO, out),
                        |p| {
                            p.if_else(
                                Cond::Lt0(x),
                                |p| p.mov(LReg::L7, out),
                                |p| p.mov(LReg::L6, out),
                            )
                        },
                    )
                });
                store(p, o);
            });
        }
        CLAMP_MIN | CLAMP_MAX => {
            // A NaN scalar: the other side, `x`, everywhere.
            if s.is_nan() {
                p.for_each_row_group(64, |p, o| {
                    load(p, o);
                    p.mov(x, out);
                    store(p, o);
                });
                return Some(Operands::Unary);
            }
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L1, s.to_bits());
            p.loadi_bits(LReg::L7, canonical(s.to_bits()));
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                // The scalar, unless `x` is not NaN and beyond it.
                p.mov(LReg::L1, out);
                p.and(x, mask, ax);
                p.mov(x, LReg::L6);
                p.if_(Cond::Eq0(ax), |p| p.mov(LReg::ZERO, LReg::L6));
                p.if_(Cond::LessEq(ax, inf), |p| {
                    let beyond = if kind == CLAMP_MIN {
                        Cond::Less(LReg::L7, LReg::L6)
                    } else {
                        Cond::Less(LReg::L6, LReg::L7)
                    };
                    p.if_(beyond, |p| p.mov(x, out));
                });
                store(p, o);
            });
        }
        CLAMP => {
            if s.is_nan() || s2.is_nan() || s > s2 {
                return None;
            }
            use super::ConfigLReg;
            // `f32::clamp`: below the low bound it, above the high one it,
            // else `x` -- a NaN, and a zero equal to a bound, included.
            p.constant(ConfigLReg::L11, s2.to_bits());
            p.constant(ConfigLReg::L12, canonical(s2.to_bits()));
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L1, s.to_bits());
            p.loadi_bits(LReg::L7, canonical(s.to_bits()));
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mov(x, out);
                p.and(x, mask, ax);
                p.mov(x, LReg::L6);
                p.if_(Cond::Eq0(ax), |p| p.mov(LReg::ZERO, LReg::L6));
                p.if_(Cond::LessEq(ax, inf), |p| {
                    p.if_(Cond::Less(LReg::L6, LReg::L7), |p| p.mov(LReg::L1, out));
                    p.if_(Cond::Less(ConfigLReg::L12.lreg(), LReg::L6), |p| {
                        p.mov(ConfigLReg::L11.lreg(), out)
                    });
                });
                store(p, o);
            });
        }
        LEAKY_RELU => {
            // `x >= 0` holds for `-0` and for every non-negative non-NaN; the
            // rest -- negatives and NaNs -- take the product.
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L6, s.to_bits());
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mul(x, LReg::L6, out);
                p.and(x, mask, ax);
                p.if_(Cond::Gte0(x), |p| {
                    p.if_(Cond::LessEq(ax, inf), |p| p.mov(x, out))
                });
                p.if_(Cond::Eq0(ax), |p| p.mov(x, out));
                store(p, o);
            });
        }
        HARD_SIGMOID => {
            // Flex's `alpha * x + beta`: two roundings, as two instructions.
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L6, s.to_bits());
            p.loadi_bits(LReg::L7, s2.to_bits());
            p.loadi(LReg::L1, 1.0);
            p.for_each_row_group(64, |p, o| {
                p.load(x, Format::Fp32, A_ROW + o);
                p.mul(LReg::L6, x, out);
                p.add(out, LReg::L7, out);
                // `.clamp(0, 1)`: a NaN stays, `-0` stays.
                p.and(out, mask, ax);
                p.if_(Cond::LessEq(ax, inf), |p| {
                    p.if_(Cond::Lt0(out), |p| {
                        p.if_(Cond::Ne0(ax), |p| p.mov(LReg::ZERO, out))
                    });
                    p.if_(Cond::Less(LReg::L1, out), |p| p.mov(LReg::L1, out));
                });
                store(p, o);
            });
        }
        k if (EQ_S..=LE_S).contains(&k) => {
            // Against a scalar: its zero made `+0` and its NaN settled here,
            // where every lane gets the unordered answer.
            compare_constants(p);
            if s.is_nan() {
                let v = if compare_of(k) == Some(NE) {
                    LReg::L6
                } else {
                    LReg::ZERO
                };
                p.for_each_row_group(64, |p, o| {
                    p.mov(v, out);
                    store(p, o);
                });
                return Some(Operands::Unary);
            }
            p.loadi_bits(LReg::L1, canonical(s.to_bits()));
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                compare_body(p, k, false);
                store(p, o);
            });
        }
        IS_NAN | IS_INF => {
            compare_constants(p);
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mov(LReg::ZERO, out);
                p.and(x, mask, ax);
                if kind == IS_NAN {
                    p.if_(Cond::Less(inf, ax), |p| p.mov(LReg::L6, out));
                } else {
                    p.if_(Cond::LessEq(inf, ax), |p| {
                        p.if_(Cond::LessEq(ax, inf), |p| p.mov(LReg::L6, out))
                    });
                }
                store(p, o);
            });
        }
        // A predicated move of raw words, whatever the element type: the
        // `Bool` and `I32` selects are this program under another signature.
        MASK_WHERE | INT_MASK_WHERE | BOOL_MASK_WHERE => {
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.load(LReg::L1, Format::Int32, super::kernel::B_ROW + o);
                p.load(LReg::L3, Format::Int32, super::kernel::C_ROW + o);
                p.mov(x, out);
                p.if_(Cond::Ne0(LReg::L1), |p| p.mov(LReg::L3, out));
                store(p, o);
            });
            return Some(Operands::Ternary);
        }
        _ => return None,
    }
    Some(Operands::Unary)
}

/// The program for `kind` with its second operand broadcast as `bcast`, and
/// what operands it takes; `None` where there is none. A row broadcast reads
/// the row's two faces from where the kernel lays them, which depends on the
/// row group, so its loop is unrolled; a column broadcast reads a whole tile
/// the mover has made of the column (`READ_BROADCAST_COL`), so its program is
/// the plain binary one.
pub fn program_for(
    kind: u32,
    scalars: [f32; 2],
    bcast: Broadcast,
) -> Option<(Operands, Vec<Instruction>)> {
    code_for(kind, scalars, bcast).map(|(o, c)| (o, c.expand()))
}

/// [`program_for`] as a role's slot holds it: its long row loops stored
/// once, the runner's block repeats beside them (`super::super::code`).
pub fn code_for(
    kind: u32,
    scalars: [f32; 2],
    bcast: Broadcast,
) -> Option<(Operands, super::super::code::Code)> {
    match bcast {
        Broadcast::None => code2(kind, scalars),
        _ if !broadcasts(kind) => None,
        Broadcast::Col => code2(kind, scalars).map(|(_, p)| (Operands::ColBroadcast, p)),
        Broadcast::Row => {
            let mut p = Program::with_policy(super::LoopPolicy::Unrolled);
            binary_constants(&mut p, kind, scalars);
            p.for_each_row_group(64, |p, o| {
                binary_body(p, kind, A_ROW + o, bias_row(o / 4) + (o & 2), OUT_ROW + o)
            });
            Some((Operands::RowBroadcast, p.finish_code()))
        }
    }
}

/// The program for `kind` (with its scalar), and what operands it takes, or
/// `None` for a kind with no SFPU program. `ADD_ROW` is `ADD` with a row
/// broadcast ([`program_for`]).
pub fn program(kind: u32, scalar: f32) -> Option<(Operands, Vec<Instruction>)> {
    program2(kind, [scalar, 0.0])
}

/// [`program`] for a kind with two scalars ([`kind_sfpu::CLAMP`],
/// [`kind_sfpu::HARD_SIGMOID`]).
pub fn program2(kind: u32, scalars: [f32; 2]) -> Option<(Operands, Vec<Instruction>)> {
    code2(kind, scalars).map(|(o, c)| (o, c.expand()))
}

/// [`program2`] as a role's slot holds it ([`code_for`]).
pub fn code2(kind: u32, scalars: [f32; 2]) -> Option<(Operands, super::super::code::Code)> {
    // The Approx twin of a transcendental (`approx.rs`).
    if let Some(code) = super::approx::code(kind) {
        return Some((Operands::Unary, code));
    }
    if (kind_sfpu::HARDWARE_ROUND..kind_sfpu::HARDWARE_ROUND + 6).contains(&kind) {
        use tt_isa::numerics::stochastic::{Precision, Rounding};
        let offset = kind - kind_sfpu::HARDWARE_ROUND;
        let precision = if offset < 3 {
            Precision::Bf16
        } else {
            Precision::Tf32
        };
        let rounding = match offset % 3 {
            0 => Rounding::Nearest,
            1 => Rounding::Stochastic,
            _ => Rounding::TowardZero,
        };
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Int32, A_ROW + o);
            p.hardware_round(LReg::L0, LReg::L2, precision, rounding);
            p.store(LReg::L2, Format::Int32, OUT_ROW + o);
        });
        return Some((Operands::Unary, p.finish_code()));
    }
    if matches!(kind, kind_sfpu::ROUND..=kind_sfpu::POW2_FLOOR_INV) {
        use super::round::{self, RoundOp};
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Int32, A_ROW + o);
            match kind {
                kind_sfpu::ROUND => round::body(p, RoundOp::Even),
                kind_sfpu::FLOOR => round::body(p, RoundOp::Floor),
                kind_sfpu::CEIL => round::body(p, RoundOp::Ceil),
                kind_sfpu::TRUNC => round::body(p, RoundOp::Trunc),
                kind_sfpu::POW2_FLOOR => {
                    p.loadi_bits(LReg::L3, 0x7f80_0000);
                    p.and(LReg::L0, LReg::L3, LReg::L2);
                }
                kind_sfpu::POW2_FLOOR_INV => {
                    p.loadi_bits(LReg::L3, 0x7f80_0000);
                    p.and(LReg::L0, LReg::L3, LReg::L1);
                    p.loadi_bits(LReg::L4, 0x7f00_0000);
                    p.mov(LReg::L1, LReg::L2);
                    p.isub_from(LReg::L4, LReg::L2); // 254 - e, in the exponent field
                                                     // `e >= 254` would make the field 0 or negative: 2^-126 instead.
                    p.if_(Cond::LessEq(LReg::L4, LReg::L1), |p| {
                        p.loadi_bits(LReg::L2, 0x0080_0000)
                    });
                }
                _ => round::to_i32(p),
            }
            p.store(LReg::L2, Format::Int32, OUT_ROW + o);
        });
        return Some((Operands::Unary, p.finish_code()));
    }
    // The exact remainder (`rem.rs`).
    match kind {
        kind_sfpu::REM => {
            let code = rem::code(rem::Variant::Exact, rem::Divisor::Tile);
            return Some((Operands::Binary, code));
        }
        kind_sfpu::REM_S => {
            let code = rem::code(rem::Variant::Exact, rem::Divisor::Scalar(scalars[0]));
            return Some((Operands::Unary, code));
        }
        _ => {}
    }
    if let Some((op, scalar)) = super::integer::operation(kind) {
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Int32, A_ROW + o);
            if scalar {
                p.loadi_bits(LReg::L1, scalars[0].to_bits());
            } else if op != 14 {
                p.load(LReg::L1, Format::Int32, B_ROW + o);
            }
            if op == 15 || op == 16 {
                p.loadi_bits(LReg::L3, 1);
                p.if_(Cond::Eq0(LReg::L1), |p| p.mov(LReg::ZERO, LReg::L3));
                p.store(LReg::L3, Format::Int32, super::kernel::C_ROW + o);
            }
            super::integer::body(p, op);
            p.store(LReg::L2, Format::Int32, OUT_ROW + o);
        });
        return Some((
            if scalar || op == 14 {
                Operands::Unary
            } else {
                Operands::Binary
            },
            p.finish_code(),
        ));
    }
    let scalar = scalars[0];
    let mut p = Program::new();
    if let Some(o) = exact_program(&mut p, kind, scalars) {
        return Some((o, p.finish_code()));
    }
    let mut p = Program::new();
    let operands = match kind {
        kind::ADD
        | kind_sfpu::ADD_RELU
        | kind::SUB
        | kind::MUL
        | kind_sfpu::DIV
        | kind_sfpu::BOOL_AND
        | kind_sfpu::BOOL_OR
        | kind_sfpu::BOOL_XOR
        | kind_sfpu::EQ..=kind_sfpu::LE
        | kind_sfpu::MASK_FILL
        | kind_sfpu::INT_MASK_FILL
        | kind_sfpu::BOOL_MASK_FILL
        | kind_sfpu::PRELU => {
            binary_constants(&mut p, kind, scalars);
            p.for_each_row_group(64, |p, o| {
                binary_body(p, kind, A_ROW + o, B_ROW + o, OUT_ROW + o)
            });
            Operands::Binary
        }
        kind::MUL_SCALAR | kind::ADD_SCALAR => {
            p.loadi(LReg::L3, scalar);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if kind == kind::MUL_SCALAR {
                    p.mul(LReg::L0, LReg::L3, LReg::L2);
                } else {
                    p.add(LReg::L0, LReg::L3, LReg::L2);
                }
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        // `max(x, 0)` as the mover has it: `x` where it is positive and not
        // NaN (`+0 < x <= +inf` in the total order), `+0` everywhere else --
        // and `RELU_BACKWARD`'s gradient on the same lanes.
        kind::RELU | kind::RELU_BACKWARD => {
            let backward = kind == kind::RELU_BACKWARD;
            p.loadi_bits(LReg::L5, FIRST_NAN);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if backward {
                    p.load(LReg::L1, Format::Fp32, B_ROW + o);
                }
                p.mov(LReg::ZERO, LReg::L2);
                p.if_(Cond::Less(LReg::ZERO, LReg::L0), |p| {
                    p.if_(Cond::Less(LReg::L0, LReg::L5), |p| {
                        p.mov(if backward { LReg::L1 } else { LReg::L0 }, LReg::L2)
                    })
                });
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            if backward {
                Operands::Binary
            } else {
                Operands::Unary
            }
        }
        // Each row group reads its column half of the broadcast row from the
        // rows the kernel laid it in, which depend on the group: unrolled.
        kind_sfpu::RECIP => {
            p.loadi_bits(LReg::L6, f32::MAX.to_bits());
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.recip(LReg::L0, LReg::L2, LReg::L3, LReg::L4, LReg::L6);
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::DIV_SCALAR => {
            p.loadi_bits(LReg::L6, f32::MAX.to_bits());
            p.loadi_bits(LReg::L7, 0x7f80_0000);
            // A divisor above `2^100`: both sides times `2^-64`, as
            // `scale_large_divisor` does, the scalar's here.
            let large = scalar.abs() > f32::from_bits(0x7180_0000);
            p.loadi(
                LReg::L1,
                if large {
                    scalar * f32::from_bits(0x1f80_0000)
                } else {
                    scalar
                },
            );
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if large {
                    p.loadi_bits(LReg::L3, 0x1f80_0000);
                    p.mul(LReg::L0, LReg::L3, LReg::L0);
                }
                p.recip(LReg::L1, LReg::L2, LReg::L3, LReg::L4, LReg::L6);
                divide(
                    p,
                    LReg::L0,
                    LReg::L1,
                    LReg::L2,
                    LReg::L5,
                    LReg::L3,
                    LReg::L4,
                    LReg::L7,
                );
                p.store(LReg::L5, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::EXP | kind_sfpu::LOG => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if kind == kind_sfpu::EXP {
                    exp_program(p, LReg::L0, LReg::L7);
                } else {
                    log_program(p, LReg::L0, LReg::L7);
                }
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        // `0`/`1` either way: `x ^ 1`, on raw bits.
        kind_sfpu::BOOL_NOT => {
            p.loadi_bits(LReg::L3, 1);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L2, Format::Int32, A_ROW + o);
                p.xor(LReg::L3, LReg::L2);
                p.store(LReg::L2, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::FILL => {
            p.loadi_bits(LReg::L2, scalar.to_bits());
            p.for_each_row_group(64, |p, o| p.store(LReg::L2, Format::Int32, OUT_ROW + o));
            Operands::Unary
        }
        kind_sfpu::SQRT | kind_sfpu::RSQRT => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                sqrt_program(p, LReg::L0, LReg::L7, kind == kind_sfpu::RSQRT);
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::LOG1P => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                log1p_program(p, LReg::L0, LReg::L7, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::POW | kind_sfpu::POW_S | kind_sfpu::POW_I => {
            let y = match kind {
                kind_sfpu::POW => PowY::Row(B_ROW),
                kind_sfpu::POW_I => PowY::IntRow(B_ROW),
                _ => PowY::Scalar(scalar),
            };
            p.for_each_row_group(64, |p, o| {
                pow_program(p, A_ROW + o, y.at(o));
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            if kind == kind_sfpu::POW_S {
                Operands::Unary
            } else {
                Operands::Binary
            }
        }
        kind_sfpu::EXPM1 => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                expm1_program(p, LReg::L0, LReg::L7);
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::SIGMOID => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                sigmoid_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        // Flex's `g * s * (1 - s)`, left to right: three roundings, as here.
        kind_sfpu::SIGMOID_BACKWARD => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.load(LReg::L1, Format::Fp32, B_ROW + o);
                p.sub(LReg::ONE, LReg::L0, LReg::L2);
                p.mul(LReg::L1, LReg::L0, LReg::L3);
                p.mul(LReg::L3, LReg::L2, LReg::L2);
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Binary
        }
        kind_sfpu::TANH => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                tanh_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::ERF => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                erf_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::GELU | kind_sfpu::GELU_BACKWARD => {
            let backward = kind == kind_sfpu::GELU_BACKWARD;
            p.for_each_row_group(64, |p, o| {
                let spill = super::kernel::SPILL_ROW + o;
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                gelu_exp_program(p, spill);
                gelu_program(p, spill, backward.then_some(B_ROW + o));
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            if backward {
                Operands::Binary
            } else {
                Operands::Unary
            }
        }
        kind_sfpu::SINH | kind_sfpu::COSH => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                sinh_cosh_program(p, super::kernel::SPILL_ROW + o, kind == kind_sfpu::COSH);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::ASINH | kind_sfpu::ACOSH | kind_sfpu::ATANH => {
            p.for_each_row_group(64, |p, o| {
                let spill = super::kernel::SPILL_ROW + o;
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                if kind == kind_sfpu::ATANH {
                    atanh_program(p, spill);
                } else {
                    asinh_acosh_program(p, spill, kind == kind_sfpu::ACOSH);
                }
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::ASIN | kind_sfpu::ACOS => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                asin_acos_program(p, super::kernel::SPILL_ROW + o, kind == kind_sfpu::ACOS);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::ATAN => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                atan_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::ATAN2 => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                p.load(LReg::L1, Format::Int32, B_ROW + o);
                atan2_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Binary
        }
        kind_sfpu::TAN => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                tan_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::SIN | kind_sfpu::COS => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                sin_cos_program(p, super::kernel::SPILL_ROW + o, kind == kind_sfpu::COS);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::LOG_SIGMOID => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                log_sigmoid_program(p, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        // Flex's `g * sigmoid(-x)`: the sigmoid of the negated input (exact),
        // then one product.
        kind_sfpu::LOG_SIGMOID_BACKWARD => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L1, Format::Fp32, A_ROW + o);
                p.neg(LReg::L1, LReg::L0);
                sigmoid_program(p, super::kernel::SPILL_ROW + o);
                p.load(LReg::L1, Format::Fp32, B_ROW + o);
                p.mul(LReg::L1, LReg::L7, LReg::L2);
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Binary
        }
        kind_sfpu::I32_TO_F32 | kind_sfpu::BOOL_TO_F32 => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L1, Format::Int32, A_ROW + o);
                i32_to_float(p, LReg::L1, LReg::L2, LReg::L3);
                p.store(LReg::L1, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::BOOL_TO_I32 => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                p.store(LReg::L0, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::INDEX_TO_I32 => {
            p.loadi(LReg::L3, 8_388_608.0);
            p.loadi_bits(LReg::L4, 0x4b00_0000);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.add(LReg::L0, LReg::L3, LReg::L2);
                p.isub_from(LReg::L2, LReg::L4);
                p.store(LReg::L4, Format::Int32, OUT_ROW + o);
                p.loadi_bits(LReg::L4, 0x4b00_0000);
            });
            Operands::Unary
        }
        kind::ADD_ROW => return code_for(kind::ADD, scalars, Broadcast::Row),
        _ => return None,
    };
    Some((operands, p.finish_code()))
}

/// What the device computes for `kind` over a whole `[rows, cols]` tensor:
/// the op's own program run by the interpreter (`super::interp`) over each
/// tile, laid in `Dst` as the kernel lays it -- so a gate can hold the device
/// to it bit for bit, and hold it to `burn-flex` within the op's derived
/// bound. `b` is `[rows, cols]`, or `[1, cols]` for a row broadcast. Padding
/// is zero, as an upload leaves it.
pub fn reference(
    kind: u32,
    scalar: f32,
    a: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let bcast = match (kind, b.map(<[f32]>::len)) {
        (kind::ADD_ROW, _) => Broadcast::Row,
        (_, Some(n)) if n == cols && rows > 1 && broadcasts(kind) => Broadcast::Row,
        (_, Some(n)) if n == rows && cols > 1 && broadcasts(kind) => Broadcast::Col,
        _ => Broadcast::None,
    };
    let kind = if kind == kind::ADD_ROW {
        kind::ADD
    } else {
        kind
    };
    reference_for(kind, scalar, bcast, a, b, rows, cols)
}

/// [`reference`] with the broadcast said rather than read from `b`'s length.
pub fn reference_for(
    kind: u32,
    scalar: f32,
    bcast: Broadcast,
    a: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let inputs: Vec<&[f32]> = std::iter::once(a).chain(b).collect();
    reference_op(kind, [scalar, 0.0], bcast, &inputs, rows, cols)
}

/// What [`crate::session::Session::pow`] computes: `POW` with `y` an `F32`
/// tensor of `x`'s shape, or (`y` `None`) `POW_S` with the scalar `s`.
pub fn pow_reference(x: &[f32], y: Option<&[f32]>, s: f32, rows: usize, cols: usize) -> Vec<f32> {
    let n = Broadcast::None;
    match y {
        Some(y) => reference_op(kind_sfpu::POW, [0.0; 2], n, &[x, y], rows, cols),
        None => reference_op(kind_sfpu::POW_S, [s, 0.0], n, &[x], rows, cols),
    }
}

/// What the device's `gelu` (`g` `None`) or `gelu_backward` computes.
pub fn gelu_reference(x: &[f32], g: Option<&[f32]>, rows: usize, cols: usize) -> Vec<f32> {
    let n = Broadcast::None;
    match g {
        None => reference_op(kind_sfpu::GELU, [0.0; 2], n, &[x], rows, cols),
        Some(g) => reference_op(kind_sfpu::GELU_BACKWARD, [0.0; 2], n, &[x, g], rows, cols),
    }
}

/// [`reference_for`] with both scalars and any number of operands: `B` as
/// `bcast` says, a third (`Operands::Ternary`) the same shape as `A`.
pub fn reference_op(
    kind: u32,
    scalars: [f32; 2],
    bcast: Broadcast,
    inputs: &[&[f32]],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (a, b, c) = (inputs[0], inputs.get(1).copied(), inputs.get(2).copied());
    let (operands, math) = program_for(kind, scalars, bcast).expect("an SFPU op");
    let (rt, ct) = (rows.div_ceil(32), cols.div_ceil(32));
    let mut out = vec![0.0f32; rows * cols];
    let tile = |x: &[f32], xr: usize, i: usize, j: usize| -> Vec<u32> {
        let mut t = vec![0u32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                let (gr, gc) = (32 * i + r, 32 * j + c);
                if gr < xr && gc < cols {
                    t[face_index(r, c)] = x[gr * cols + gc].to_bits();
                }
            }
        }
        t
    };
    for i in 0..rt {
        for j in 0..ct {
            let mut v = Vector::new();
            v.put_tile(A_ROW as usize, &tile(a, rows, i, j));
            match (operands, b) {
                (Operands::Binary, Some(b)) => v.put_tile(B_ROW as usize, &tile(b, rows, i, j)),
                (Operands::Ternary, Some(b)) => {
                    v.put_tile(B_ROW as usize, &tile(b, rows, i, j));
                    let c = c.expect("a ternary op's third operand");
                    v.put_tile(super::kernel::C_ROW as usize, &tile(c, rows, i, j));
                }
                (Operands::RowBroadcast, Some(b)) => {
                    let t = tile(b, 1, 0, j);
                    for r in 0..4 {
                        v.dst[(B_ROW + r) as usize] = std::array::from_fn(|c| t[c]);
                        v.dst[(B_ROW + 4 + r) as usize] = std::array::from_fn(|c| t[256 + c]);
                    }
                }
                (Operands::ColBroadcast, Some(b)) => {
                    // As `READ_BROADCAST_COL` makes it: row `r`'s value in
                    // every column.
                    let mut t = vec![0u32; 1024];
                    for r in 0..32 {
                        let gr = 32 * i + r;
                        let v = if gr < rows { b[gr].to_bits() } else { 0 };
                        for c in 0..32 {
                            t[face_index(r, c)] = v;
                        }
                    }
                    v.put_tile(B_ROW as usize, &t);
                }
                (Operands::Unary, _) => {}
                _ => panic!("kind {kind:#x} takes a second operand"),
            }
            v.run(&math)
                .unwrap_or_else(|e| panic!("kind {kind:#x}: {e}"));
            let t = v.tile(OUT_ROW as usize);
            for r in 0..32 {
                for c in 0..32 {
                    let (gr, gc) = (32 * i + r, 32 * j + c);
                    if gr < rows && gc < cols {
                        out[gr * cols + gc] = f32::from_bits(t[face_index(r, c)]);
                    }
                }
            }
        }
    }
    out
}

#[cfg(any())]
mod tests {
    use super::super::interp::Vector;
    use super::*;

    #[test]
    fn native_index_cast_is_exact_at_its_range_edges() {
        let values = [0.0, 1.0, 65_535.0, 8_388_607.0];
        let got = reference(kind_sfpu::INDEX_TO_I32, 0.0, &values, None, 1, values.len());
        assert_eq!(
            got.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            [0, 1, 65_535, 8_388_607]
        );
    }

    fn tile(seed: u32) -> Vec<u32> {
        let mut s = seed | 1;
        (0..1024)
            .map(|i| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match i % 13 {
                    0 => 0x7fc0_0000,
                    1 => 0x8000_0000,
                    2 => 0x7f80_0000,
                    3 => 0xff80_0000,
                    4 => 0xffc0_0001,
                    _ => (s & 0x8000_0000) | (((s >> 8) % 40 + 107) << 23) | (s & 0x7f_ffff),
                }
            })
            .collect()
    }

    /// The mover's per-datum arithmetic (`dm_b.rs`), in IEEE terms with its
    /// denormal flush -- the reference the programs must equal.
    fn mover(kind: u32, s: f32, a: u32, b: u32, bias: u32) -> u32 {
        let f = f32::from_bits;
        let positive = (a as i32) > 0 && a <= 0x7f80_0000;
        let r = match kind {
            kind::ADD => (f(a) + f(b)).to_bits(),
            kind_sfpu::ADD_RELU => {
                let sum = (f(a) + f(b)).to_bits();
                let sum_positive = (sum as i32) > 0 && sum <= 0x7f80_0000;
                if sum_positive {
                    sum
                } else {
                    0
                }
            }
            kind::SUB => (f(a) - f(b)).to_bits(),
            kind::MUL => (f(a) * f(b)).to_bits(),
            kind::MUL_SCALAR => (f(a) * s).to_bits(),
            kind::ADD_SCALAR => (f(a) + s).to_bits(),
            kind::RELU => {
                if positive {
                    a
                } else {
                    0
                }
            }
            kind::RELU_BACKWARD => {
                if positive {
                    b
                } else {
                    0
                }
            }
            kind::ADD_ROW => (f(a) + f(bias)).to_bits(),
            _ => unreachable!(),
        };
        if f(r).is_nan() && !matches!(kind, kind::RELU | kind::RELU_BACKWARD | kind_sfpu::ADD_RELU)
        {
            0x7fc0_0000
        } else {
            r
        }
    }

    #[test]
    fn every_program_computes_what_the_mover_does() {
        let (a, b) = (tile(3), tile(5));
        for (k, s) in [
            (kind::ADD, 0.0),
            (kind_sfpu::ADD_RELU, 0.0),
            (kind::SUB, 0.0),
            (kind::MUL, 0.0),
            (kind::MUL_SCALAR, -1.75),
            (kind::MUL_SCALAR, 3.0e-3),
            (kind::ADD_SCALAR, -0.0),
            (kind::ADD_SCALAR, 2.7),
            (kind::RELU, 0.0),
            (kind::RELU_BACKWARD, 0.0),
            (kind::ADD_ROW, 0.0),
        ] {
            let (operands, math) = program(k, s).unwrap();
            let mut v = Vector::new();
            v.put_tile(A_ROW as usize, &a);
            if operands == Operands::RowBroadcast {
                // As the kernel lays it: row 0's two faces, four times each.
                for r in 0..4 {
                    v.dst[(B_ROW + r) as usize] = std::array::from_fn(|c| b[c]);
                    v.dst[(B_ROW + 4 + r) as usize] = std::array::from_fn(|c| b[256 + c]);
                }
            } else {
                v.put_tile(B_ROW as usize, &b);
            }
            v.run(&math).unwrap_or_else(|e| panic!("kind {k}: {e}"));
            let got = v.tile(OUT_ROW as usize);
            for i in 0..1024 {
                // Datum i's column, for the broadcast row.
                let col = (i / 256 % 2) * 16 + i % 16;
                let bias = if col < 16 { b[col] } else { b[256 + col - 16] };
                assert_eq!(
                    got[i],
                    mover(k, s, a[i], b[i], bias),
                    "kind {k}: datum {i}: a {:#010x} b {:#010x}",
                    a[i],
                    b[i]
                );
            }
        }
    }
}

#[cfg(any())]
mod fit {
    use super::*;
    use crate::sfpu::kernel::{plan_layout, roles_code};

    /// Every op's longest run (`tensor::sfpu_eltwise`'s group) has role
    /// programs that fit a program slot, as the slots hold them (block
    /// repeats stored once), and is no shorter than it must be.
    #[test]
    fn every_op_s_longest_run_fits_a_program_slot() {
        for k in [
            kind::ADD,
            kind_sfpu::ADD_RELU,
            kind::SUB,
            kind::MUL,
            kind::MUL_SCALAR,
            kind::ADD_SCALAR,
            kind::RELU,
            kind::RELU_BACKWARD,
            kind::ADD_ROW,
        ]
        .into_iter()
        .chain(kind_sfpu::INT_ADD..=kind_sfpu::INT_NOT)
        .chain(kind_sfpu::INT_ADD_S..=kind_sfpu::INT_LE_S)
        .chain(kind_sfpu::ROUND..=kind_sfpu::POW2_FLOOR_INV)
        .chain([
            kind_sfpu::INT_MASK_FILL,
            kind_sfpu::BOOL_MASK_FILL,
            kind_sfpu::INT_ABS,
        ]) {
            let bcast = if k == kind::ADD_ROW {
                Broadcast::Row
            } else {
                Broadcast::None
            };
            let kk = if k == kind::ADD_ROW { kind::ADD } else { k };
            let (operands, math) = code_for(kk, [0.5, 0.0], bcast).unwrap();
            let n = crate::tensor::sfpu_group_for_tests(k, 0.5, operands);
            let fits = |n: usize| {
                let layout = plan_layout(n, operands).unwrap();
                roles_code(&layout, operands, &math)
                    .0
                    .iter()
                    .all(|p| p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize)
            };
            assert!(fits(n), "kind {k}: {n} tiles");
            assert!(
                n == 64 || !fits(n + 1),
                "kind {k}: {n} tiles, but {} would fit",
                n + 1
            );
        }
    }
}

#[cfg(any())]
mod division {
    use super::super::interp::Vector;
    use super::*;

    /// Distance in ulps between two finite floats of the same sign.
    fn ulps(a: f32, b: f32) -> u32 {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
    }

    fn normals(seed: u32) -> Vec<u32> {
        let mut s = seed | 1;
        (0..1024)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Exponents well inside the range, so quotients and
                // reciprocals stay normal.
                (s & 0x8000_0000) | (((s >> 8) % 100 + 77) << 23) | (s & 0x7f_ffff)
            })
            .collect()
    }

    fn run(k: u32, s: f32, a: &[u32], b: &[u32]) -> Vec<u32> {
        let (_, math) = program(k, s).unwrap();
        let mut v = Vector::new();
        v.put_tile(A_ROW as usize, a);
        v.put_tile(B_ROW as usize, b);
        v.run(&math).unwrap_or_else(|e| panic!("kind {k:#x}: {e}"));
        v.tile(OUT_ROW as usize)
    }

    /// Within one ulp of the correctly rounded result (the bound derived on
    /// `Program::recip` and `divide`), and the reported worst case.
    #[test]
    fn reciprocal_and_quotient_are_within_one_ulp() {
        let (a, b) = (normals(7), normals(11));
        let f = f32::from_bits;
        let recip = run(kind_sfpu::RECIP, 0.0, &a, &b);
        let div = run(kind_sfpu::DIV, 0.0, &a, &b);
        let div_s = run(kind_sfpu::DIV_SCALAR, -3.7, &a, &b);
        let (mut worst, mut exact) = (0, 0);
        for i in 0..1024 {
            for (got, want) in [
                (f(recip[i]), 1.0 / f(a[i])),
                (f(div[i]), f(a[i]) / f(b[i])),
                (f(div_s[i]), f(a[i]) / -3.7),
            ] {
                let d = ulps(got, want);
                assert!(
                    d <= 1,
                    "{got:e} vs {want:e}: {d} ulps (a {:#x} b {:#x})",
                    a[i],
                    b[i]
                );
                worst = worst.max(d);
                exact += usize::from(d == 0);
            }
        }
        println!("worst {worst} ulp; {exact} of 3072 correctly rounded");
    }

    /// The special cases, as IEEE has them (denormals aside: they flush).
    #[test]
    fn the_special_cases() {
        let f = f32::from_bits;
        let cases: [(f32, f32); 10] = [
            (1.0, 0.0),
            (1.0, -0.0),
            (-2.0, f32::INFINITY),
            (0.0, 0.0),
            (f32::INFINITY, f32::INFINITY),
            (f32::INFINITY, 2.0),
            (5.0, f32::NAN),
            (f32::NAN, 5.0),
            (0.0, 3.0),
            (1.0e-30, 1.0e20),
        ];
        let mut a = vec![1.0f32.to_bits(); 1024];
        let mut b = vec![1.0f32.to_bits(); 1024];
        for (i, (x, y)) in cases.iter().enumerate() {
            a[i] = x.to_bits();
            b[i] = y.to_bits();
        }
        let recip = run(kind_sfpu::RECIP, 0.0, &b, &a);
        let div = run(kind_sfpu::DIV, 0.0, &a, &b);
        for (i, (x, y)) in cases.iter().enumerate() {
            for (got, want, what) in [(f(recip[i]), 1.0 / y, "recip"), (f(div[i]), x / y, "div")] {
                // 1e-30 / 1e20 is denormal in IEEE and flushes here.
                let want = if want != 0.0 && want.abs() < f32::MIN_POSITIVE {
                    0.0
                } else {
                    want
                };
                if want.is_nan() {
                    assert!(got.is_nan(), "{what} {x} / {y}: {got}");
                } else {
                    assert_eq!(got, want, "{what} {x} / {y}");
                    assert_eq!(
                        got.is_sign_negative(),
                        want.is_sign_negative(),
                        "{what} {x} / {y}: sign"
                    );
                }
            }
        }
    }
}

#[cfg(any())]
mod transcendental {
    use super::*;

    /// Ulps between `got` and the exact value `want` (an f64), in units of
    /// `got`'s binade.
    fn ulps(got: f32, want: f64) -> f64 {
        let ulp = f32::from_bits((got.abs().to_bits() & 0x7f80_0000).max(0x0080_0000)) as f64
            * f64::powi(2.0, -23);
        (got as f64 - want).abs() / ulp
    }

    /// Every input's result within `bound` of the exact value, relative (the
    /// bound derived on the program); the worst case is reported in ulps.
    fn sweep(kind: u32, xs: impl Iterator<Item = f32>, exact: fn(f64) -> f64, bound: f64) {
        let xs: Vec<f32> = xs.collect();
        let mut worst = (0.0, 0.0f32);
        for chunk in xs.chunks(1024) {
            let mut a = chunk.to_vec();
            a.resize(1024, 1.0);
            let got = reference(kind, 0.0, &a, None, 32, 32);
            for (x, g) in chunk.iter().zip(&got) {
                let w = exact(*x as f64);
                if w.is_nan() {
                    assert!(g.is_nan(), "kind {kind:#x}({x}): {g}");
                } else if w.is_infinite()
                    || w.abs() < f32::MIN_POSITIVE as f64
                    || w.abs() > f32::MAX as f64
                {
                    let want = if w.abs() < f32::MIN_POSITIVE as f64 {
                        0.0
                    } else {
                        w.signum() * f64::INFINITY
                    };
                    assert_eq!(*g as f64, want, "kind {kind:#x}({x:e})");
                } else {
                    let u = ulps(*g, w);
                    if u > worst.0 {
                        worst = (u, *x);
                    }
                    let rel = (*g as f64 - w).abs() / w.abs();
                    assert!(
                        rel <= bound,
                        "kind {kind:#x}({x:e}) = {g:e}, exact {w:e}: {rel:e} ({u:.2} ulps)"
                    );
                }
            }
        }
        println!(
            "kind {kind:#x}: worst {:.3} ulps at {:e} over {} inputs",
            worst.0,
            worst.1,
            xs.len()
        );
    }

    fn grid(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..n).map(move |i| lo + (hi - lo) * (i as f32 / n as f32))
    }

    #[test]
    fn exp_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            88.7,
            88.73,
            f32::from_bits(0x42b1_7217),
            f32::from_bits(0x42b1_7218),
            f32::from_bits(0x42b1_7219),
            f32::from_bits(0xc2ae_ac4f),
            f32::from_bits(0xc2ae_ac50),
            f32::from_bits(0xc2ae_ac51),
            -87.3,
            -87.4,
            -100.0,
            100.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::EXP,
            grid(-87.3, 88.7, 60_000)
                .chain(grid(-0.01, 0.01, 4096))
                .chain(specials),
            f64::exp,
            EXP_BOUND,
        );
    }

    #[test]
    fn log_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            0.5,
            1.0e-40,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let ln = |x: f64| {
            if x == 0.0 || x.abs() < f32::MIN_POSITIVE as f64 {
                f64::NEG_INFINITY
            } else {
                x.ln()
            }
        };
        sweep(
            kind_sfpu::LOG,
            grid(1.0e-6, 3.0, 40_000)
                .chain(grid(0.7, 1.45, 20_000))
                .chain((0..20_000).map(|i| f32::from_bits(0x0080_0000 + i * 106_000)))
                .chain(specials),
            ln,
            LOG_BOUND,
        );
    }

    /// Every binade of positive normals, `n` mantissas each.
    fn binades(n: u32) -> impl Iterator<Item = f32> {
        (1..255u32).flat_map(move |e| {
            (0..n).map(move |k| f32::from_bits((e << 23) | (k * (0x7f_ffff / n.max(1)))))
        })
    }

    /// Every binade, so the large inputs whose Newton steps once flushed
    /// (`Program::recip`, `|x| > 2^100`) are held as the rest are.
    #[test]
    fn recip_is_within_one_ulp_in_every_binade() {
        let r = |x: f64| {
            if (1.0 / x).abs() < f32::MIN_POSITIVE as f64 {
                0.0
            } else {
                1.0 / x
            }
        };
        sweep(
            kind_sfpu::RECIP,
            binades(256).chain(binades(64).map(|x| -x)),
            r,
            1.5 / 8_388_608.0,
        );
    }

    /// Quotients across the whole range, the smallest normal ones included:
    /// `divide`'s correction `(a - b q) y` is the quotient's error times `a/b`,
    /// so it too could fall below `2^-126`.
    #[test]
    fn division_is_within_one_ulp_down_to_the_smallest_quotients() {
        let mut worst = 0.0f64;
        for (ai, bi) in [(0u32, 0u32), (0x3f80_0000, 0), (0x0100_0000, 0x3f80_0000)] {
            let b: Vec<f32> = binades(4).skip(bi as usize % 7).take(1024).collect();
            let mut b = b;
            b.resize(1024, 3.0);
            let a: Vec<f32> = (0..1024)
                .map(|i| f32::from_bits(ai.max(0x0080_0000) + (i as u32) * 0x0007_ffff))
                .collect();
            let got = reference(kind_sfpu::DIV, 0.0, &a, Some(&b), 32, 32);
            for i in 0..1024 {
                let q = a[i] as f64 / b[i] as f64;
                if q.abs() < f32::MIN_POSITIVE as f64 || q.abs() > f32::MAX as f64 {
                    continue;
                }
                let rel = (got[i] as f64 - q).abs() / q.abs();
                worst = worst.max(rel);
                assert!(
                    rel <= 1.5 / 8_388_608.0,
                    "{:e} / {:e} = {:e}, exact {q:e}: {rel:e}",
                    a[i],
                    b[i],
                    got[i]
                );
            }
        }
        println!("division: worst {:.3} ulps", worst * 8_388_608.0);
    }

    /// `powf`'s special values, every pairing of special bases and exponents,
    /// exactly as the host's `powf` has them (the sign of a zero or infinity
    /// included); finite results within [`pow_bound`] of the exact power.
    /// Random pairs over the ranges where the result is normal too. A denormal
    /// base is flushed first (numerics row D), as the device's arithmetic does.
    #[test]
    fn pow_has_powf_s_special_values_and_is_within_its_bound() {
        let ftz = |x: f32| {
            if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
                f32::from_bits(x.to_bits() & 0x8000_0000)
            } else {
                x
            }
        };
        let xs = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            3.0,
            -3.0,
            10.0,
            -10.0,
            1.0e-30,
            -1.0e-30,
            1.0e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN_POSITIVE,
            0.9999,
            1.0001,
        ];
        let ys = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            2.0,
            3.0,
            -2.0,
            -3.0,
            0.5,
            -0.5,
            2.5,
            -2.5,
            1.0e10,
            8_388_609.0,
            16_777_216.0,
            -8_388_609.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.1,
            100.0,
            -100.0,
        ];
        let mut pairs: Vec<(f32, f32)> = xs
            .iter()
            .flat_map(|&x| ys.iter().map(move |&y| (x, y)))
            .collect();
        let mut st = 0x2545_f491u32;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st as f32 / u32::MAX as f32
        };
        for _ in 0..20_000 {
            let x = rnd() * 20.0 - 10.0;
            let y = rnd() * 16.0 - 8.0;
            pairs.push((x, y));
            pairs.push((x, y.round()));
            pairs.push((rnd() * 1.0e3, rnd() * 25.0 - 12.5));
        }
        let mut worst = 0.0f64;
        for chunk in pairs.chunks(1024) {
            let mut x: Vec<f32> = chunk.iter().map(|p| p.0).collect();
            let mut y: Vec<f32> = chunk.iter().map(|p| p.1).collect();
            x.resize(1024, 1.0);
            y.resize(1024, 1.0);
            let got = pow_reference(&x, Some(&y), 0.0, 32, 32);
            for (i, &(xi, yi)) in chunk.iter().enumerate() {
                let host = ftz(xi).powf(yi);
                let g = got[i];
                let what = format!(
                    "pow({xi:e}, {yi:e}) = {g:e} ({:#010x}), powf {host:e}",
                    g.to_bits()
                );
                if host.is_nan() {
                    assert!(g.is_nan(), "{what}");
                } else if g.is_nan() {
                    panic!("{what}: NaN where the host has a value");
                } else if host.is_infinite() || host == 0.0 || host.abs() < f32::MIN_POSITIVE {
                    // The exact power is infinite or flushes: the device's
                    // must be too, with its sign -- except within the bound of
                    // the range's ends, where either side may round over.
                    let exact = (ftz(xi) as f64).powf(yi as f64);
                    let edge = exact.is_finite()
                        && exact != 0.0
                        && ((exact.abs() / f32::MAX as f64 - 1.0).abs() < 2.0 * pow_bound(xi, yi)
                            || (exact.abs() / f32::MIN_POSITIVE as f64 - 1.0).abs()
                                < 2.0 * pow_bound(xi, yi)
                            || exact.abs() < f32::MIN_POSITIVE as f64);
                    if !edge {
                        let want = if host.is_infinite() {
                            host
                        } else {
                            f32::from_bits(host.to_bits() & 0x8000_0000)
                        };
                        assert_eq!(g.to_bits(), want.to_bits(), "{what}");
                    }
                } else {
                    let exact = (ftz(xi) as f64).powf(yi as f64);
                    let rel = (g as f64 - exact).abs() / exact.abs();
                    // Within the bound of the range's ends either side may
                    // round over: `e^z` saturates at the threshold `exp`
                    // draws.
                    let b2 = 1.0 + 2.0 * pow_bound(xi, yi);
                    if exact.abs() * b2 > f32::MAX as f64
                        || exact.abs() < f32::MIN_POSITIVE as f64 * b2
                    {
                        continue;
                    }
                    worst = worst.max(rel / pow_bound(xi, yi));
                    assert!(
                        rel <= pow_bound(xi, yi),
                        "{what}: {rel:e} against {:e}",
                        pow_bound(xi, yi)
                    );
                    assert_eq!(
                        g.is_sign_negative(),
                        host.is_sign_negative(),
                        "{what}: sign"
                    );
                }
            }
        }
        println!("pow: worst {worst:.3} of the bound");
        // A scalar exponent, by the same stages.
        let x: Vec<f32> = (0..1024)
            .map(|i| xs[i % xs.len()] * (1.0 + i as f32 / 1024.0))
            .collect();
        for s in [2.5f32, -0.5, 3.0, 0.0, f32::NAN, f32::INFINITY] {
            let got = pow_reference(&x, None, s, 32, 32);
            let viat = pow_reference(&x, Some(&vec![s; 1024]), 0.0, 32, 32);
            let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(b(&got), b(&viat), "a scalar exponent {s} is a tensor of it");
        }
    }

    #[test]
    fn expm1_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0e-30,
            -1.0e-30,
            1.0e-7,
            0.3466,
            -0.3466,
            0.3467,
            1.0,
            -1.0,
            -17.9,
            -18.0,
            -18.1,
            88.38,
            88.7,
            f32::from_bits(0x42b1_7217),
            f32::from_bits(0x42b1_7218),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::EXPM1,
            grid(-18.0, 88.7, 60_000)
                .chain(grid(-1.0, 1.0, 20_000))
                .chain(binades(32).filter(|x| *x < 88.0 && *x > 1.0e-30))
                .chain(
                    binades(32)
                        .filter(|x| *x < 18.0 && *x > 1.0e-30)
                        .map(|x| -x),
                )
                .chain(specials),
            f64::exp_m1,
            EXPM1_BOUND,
        );
    }

    #[test]
    fn sigmoid_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            20.0,
            -20.0,
            87.0,
            -87.0,
            -88.0,
            -103.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let sig = |x: f64| {
            let v = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                x.exp() / (1.0 + x.exp())
            };
            // A value the device's exponential flushes.
            if (-x).exp() > f32::MAX as f64 * 1.000001 || x < -87.33 {
                0.0
            } else {
                v
            }
        };
        sweep(
            kind_sfpu::SIGMOID,
            grid(-87.0, 30.0, 60_000)
                .chain(grid(-1.0, 1.0, 20_000))
                .chain(specials),
            sig,
            SIGMOID_BOUND,
        );
    }

    #[test]
    fn tanh_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            2.4e-4,
            -2.4e-4,
            2.45e-4,
            0.5,
            9.0,
            9.01,
            9.02,
            20.0,
            -20.0,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::TANH,
            grid(-10.0, 10.0, 60_000)
                .chain(grid(-0.01, 0.01, 20_000))
                .chain(binades(64).filter(|x| *x < 10.0).flat_map(|x| [x, -x]))
                .chain(specials),
            f64::tanh,
            TANH_BOUND,
        );
    }

    /// The switch to `x` itself (`2^-12`), the large branch (88), `expm1`'s own
    /// overflow edge (88.72) and the result's (89.4159), every binade.
    fn hyperbolic_inputs() -> impl Iterator<Item = f32> {
        let specials = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(0x397f_ffff),
            f32::from_bits(0x3980_0000),
            0.5,
            1.0,
            9.0,
            87.99,
            88.0,
            88.38,
            88.72,
            f32::from_bits(0x42b1_7218),
            89.41,
            89.4159,
            89.416,
            89.42,
            100.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        grid(-90.0, 90.0, 60_000)
            .chain(grid(-1.0, 1.0, 20_000))
            .chain(binades(32).filter(|x| *x < 90.0))
            .chain(specials)
            .flat_map(|x| [x, -x])
    }

    #[test]
    fn sinh_is_within_its_derived_bound() {
        sweep(kind_sfpu::SINH, hyperbolic_inputs(), f64::sinh, SINH_BOUND);
    }

    #[test]
    fn cosh_is_within_its_derived_bound() {
        sweep(kind_sfpu::COSH, hyperbolic_inputs(), f64::cosh, COSH_BOUND);
    }

    /// The switches to `x` itself (`2^-12`) and to `ln(2a)` (`2^12`), every
    /// binade to `f32::MAX`, the edges of `acosh`'s and `atanh`'s domains.
    fn inverse_hyperbolic_inputs() -> impl Iterator<Item = f32> {
        let one = 1.0f32.to_bits();
        let specials = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(0x397f_ffff),
            f32::from_bits(0x3980_0000),
            0.5,
            f32::from_bits(one - 1),
            1.0,
            f32::from_bits(one + 1),
            2.0,
            f32::from_bits(0x457f_ffff),
            4096.0,
            f32::from_bits(0x4580_0001),
            16_777_216.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        grid(-4.0, 4.0, 40_000)
            .chain(grid(0.99, 1.01, 10_000))
            .chain(grid(-1.0, 1.0, 20_000))
            .chain(binades(32))
            .chain(specials)
            .flat_map(|x| [x, -x])
    }

    #[test]
    fn asinh_is_within_its_derived_bound() {
        sweep(
            kind_sfpu::ASINH,
            inverse_hyperbolic_inputs(),
            f64::asinh,
            ASINH_BOUND,
        );
    }

    #[test]
    fn acosh_is_within_its_derived_bound() {
        sweep(
            kind_sfpu::ACOSH,
            inverse_hyperbolic_inputs(),
            f64::acosh,
            ACOSH_BOUND,
        );
    }

    #[test]
    fn atanh_is_within_its_derived_bound() {
        sweep(
            kind_sfpu::ATANH,
            inverse_hyperbolic_inputs(),
            f64::atanh,
            ATANH_BOUND,
        );
    }

    #[test]
    fn log_sigmoid_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0e-30,
            -1.0e-30,
            1.0,
            -1.0,
            17.0,
            -17.0,
            87.0,
            87.3,
            87.4,
            -87.4,
            -100.0,
            1.0e30,
            -1.0e30,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let ls = |x: f64| {
            if x >= 0.0 {
                -(-x).exp().ln_1p()
            } else {
                x - x.exp().ln_1p()
            }
        };
        sweep(
            kind_sfpu::LOG_SIGMOID,
            grid(-90.0, 90.0, 60_000)
                .chain(grid(-1.0, 1.0, 20_000))
                .chain(binades(32).flat_map(|x| [x, -x]))
                .chain(specials),
            ls,
            LOG_SIGMOID_BOUND,
        );
    }

    /// The floats nearest a multiple of `pi/2`, in order (`|g|` from
    /// `2^-29.86`): the [`trig_reduce`] inputs whose `r` cancels most.
    const HARDEST_REDUCTIONS: [u32; 12] = [
        0x6f79_be45,
        0x50a3_e87f,
        0x6ff9_be45,
        0x5123_e87f,
        0x437c_e5f1,
        0x7079_be45,
        0x6a19_76f1,
        0x53b1_46a6,
        0x6589_8498,
        0x51a3_e87f,
        0x43fc_e5f1,
        0x7758_4625,
    ];

    /// The 100-bit window of `2/pi` from bit `E - 151`: `|g|` of `a = M
    /// 2^(E-150)` (`a (2/pi) = q + g`) is [`reduction_distance`]'s, exactly
    /// but for the bits past the window (below `2^-74`).
    fn reduction_window(e: i64) -> u128 {
        let bit = |k: i64| -> u128 {
            if k < 1 {
                0
            } else {
                let i = (k - 1) as usize;
                ((TWO_OVER_PI[i / 24] >> (23 - i % 24)) & 1) as u128
            }
        };
        (0..100).fold(0u128, |w, i| (w << 1) | bit(e - 151 + i))
    }

    fn reduction_distance(w: u128, bits: u32) -> f64 {
        let m = (bits & 0x7f_ffff | 0x80_0000) as u128;
        let one = 1u128 << 98;
        let fr = (m * w) & (one - 1);
        (fr.min(one - fr)) as f64 / one as f64
    }

    /// The premise of [`trig_reduce`]'s bound: no float from `pi/4` is nearer
    /// a multiple of `pi/2` than `16367173 2^72`, `|g| = 2^-29.86` -- every
    /// float scanned in a release build, every 1024th mantissa in a debug one
    /// (where the twelve hardest are checked by name).
    #[test]
    fn no_float_reduces_closer_than_the_hardest() {
        let step = if cfg!(debug_assertions) { 1024 } else { 1 };
        let start = std::f32::consts::FRAC_PI_4.to_bits();
        let mut worst = (f64::MAX, 0u32);
        for e in 126..255u32 {
            let w = reduction_window(e as i64);
            let named = HARDEST_REDUCTIONS.into_iter().filter(|b| b >> 23 == e);
            for bits in ((e << 23)..((e + 1) << 23)).step_by(step).chain(named) {
                if bits < start {
                    continue;
                }
                let g = reduction_distance(w, bits);
                if g < worst.0 {
                    worst = (g, bits);
                }
            }
        }
        assert_eq!(worst.1, HARDEST_REDUCTIONS[0], "{worst:?}");
        assert!(worst.0 > f64::powf(2.0, -29.87), "{worst:?}");
        let gs: Vec<f64> = HARDEST_REDUCTIONS
            .iter()
            .map(|b| reduction_distance(reduction_window((b >> 23) as i64), *b))
            .collect();
        assert!(gs.windows(2).all(|w| w[0] <= w[1]), "{gs:?}");
    }

    /// The switch to `x` itself (`2^-12`) and to the reduction (`pi/4`), the
    /// first windows, every binade to `f32::MAX`, and the hardest reductions
    /// with their neighbours.
    fn trig_inputs() -> impl Iterator<Item = f32> {
        let q = std::f32::consts::FRAC_PI_4.to_bits();
        let specials = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(0x397f_ffff),
            f32::from_bits(0x3980_0000),
            f32::from_bits(q - 1),
            f32::from_bits(q),
            f32::from_bits(q + 1),
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            std::f32::consts::TAU,
            1.0e5,
            1.0e10,
            16_777_216.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        let hardest = HARDEST_REDUCTIONS
            .into_iter()
            .flat_map(|b| (b - 8..=b + 8).map(f32::from_bits));
        grid(-12.6, 12.6, 60_000)
            .chain(grid(-1.0e4, 1.0e4, 20_000))
            .chain(binades(64))
            .chain(specials)
            .chain(hardest)
            .flat_map(|x| [x, -x])
    }

    #[test]
    fn sin_is_within_its_derived_bound() {
        sweep(kind_sfpu::SIN, trig_inputs(), libm::sin, SIN_BOUND);
    }

    #[test]
    fn cos_is_within_its_derived_bound() {
        sweep(kind_sfpu::COS, trig_inputs(), libm::cos, COS_BOUND);
    }

    #[test]
    fn tan_is_within_its_derived_bound() {
        sweep(kind_sfpu::TAN, trig_inputs(), libm::tan, TAN_BOUND);
    }

    #[test]
    fn atan_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(0x397f_ffff),
            f32::from_bits(0x3980_0000),
            f32::from_bits(0x3f7f_ffff),
            1.0,
            f32::from_bits(0x3f80_0001),
            16_777_216.0,
            f32::from_bits(0x7e7f_ffff),
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::ATAN,
            grid(-50.0, 50.0, 40_000)
                .chain(grid(-1.0, 1.0, 40_000))
                .chain(grid(0.98, 1.02, 10_000))
                .chain(binades(64))
                .chain(specials)
                .flat_map(|x| [x, -x]),
            libm::atan,
            ATAN_BOUND,
        );
    }

    /// The switches to `x` itself (`2^-12`) and to the root (0.7), the ends
    /// of the domain and beyond.
    fn arc_inputs() -> impl Iterator<Item = f32> {
        let (sw, one) = (0.7f32.to_bits(), 1.0f32.to_bits());
        let specials = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(0x397f_ffff),
            f32::from_bits(0x3980_0000),
            0.5,
            f32::from_bits(sw - 1),
            f32::from_bits(sw),
            f32::from_bits(sw + 1),
            f32::from_bits(one - 2),
            f32::from_bits(one - 1),
            1.0,
            f32::from_bits(one + 1),
            2.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        grid(-1.0, 1.0, 80_000)
            .chain(grid(0.999, 1.0, 10_000))
            .chain(grid(0.69, 0.71, 10_000))
            .chain(binades(64).filter(|x| *x < 1.0))
            .chain(specials)
            .flat_map(|x| [x, -x])
    }

    #[test]
    fn asin_is_within_its_derived_bound() {
        sweep(kind_sfpu::ASIN, arc_inputs(), libm::asin, ASIN_BOUND);
    }

    #[test]
    fn acos_is_within_its_derived_bound() {
        sweep(kind_sfpu::ACOS, arc_inputs(), libm::acos, ACOS_BOUND);
    }

    /// `atan2` over every pairing of signed specials -- as `f32::atan2` but for
    /// a denormal operand, which the device reads as a zero of its sign (and
    /// `f32::atan2` of the flushed pair is the answer) -- and over pairs across
    /// every binade and the four quadrants, within [`ATAN2_BOUND`] of
    /// `libm::atan2` in `f64`.
    #[test]
    fn atan2_has_ieee_s_special_values_and_is_within_its_bound() {
        let ftz = |x: f32| {
            if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
                f32::from_bits(x.to_bits() & 0x8000_0000)
            } else {
                x
            }
        };
        let mags = [
            0.0f32,
            1.0e-40,
            f32::MIN_POSITIVE,
            1.0e-30,
            0.5,
            1.0,
            2.0,
            1.0e30,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ];
        let signed: Vec<f32> = mags.iter().flat_map(|m| [*m, -*m]).collect();
        let mut pairs: Vec<(f32, f32)> = signed
            .iter()
            .flat_map(|&y| signed.iter().map(move |&x| (y, x)))
            .collect();
        let mut st = 0x2545_f491u32;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        for _ in 0..60_000 {
            let (a, b) = (rnd(), rnd());
            // Any finite normal, or within a few binades of each other.
            let y = f32::from_bits(a % 0x7f00_0000 + 0x0080_0000);
            let x = if b % 2 == 0 {
                f32::from_bits(b % 0x7f00_0000 + 0x0080_0000)
            } else {
                y * (1.0 + (b >> 8) as f32 / 16_777_216.0 * 8.0)
            };
            let (sy, sx) = (rnd() % 2 == 0, rnd() % 2 == 0);
            pairs.push((if sy { -y } else { y }, if sx { -x } else { x }));
        }
        let u = 1.0 / 16_777_216.0;
        let mut worst = 0.0f64;
        for chunk in pairs.chunks(1024) {
            let mut y: Vec<f32> = chunk.iter().map(|p| p.0).collect();
            let mut x: Vec<f32> = chunk.iter().map(|p| p.1).collect();
            y.resize(1024, 1.0);
            x.resize(1024, 1.0);
            let got = reference(kind_sfpu::ATAN2, 0.0, &y, Some(&x), 32, 32);
            for (i, &(y, x)) in chunk.iter().enumerate() {
                let (fy, fx) = (ftz(y), ftz(x));
                let g = got[i];
                let want = fy.atan2(fx);
                let what = format!("atan2({y:e}, {x:e}) = {g:e}, f32::atan2 {want:e}");
                if want.is_nan() {
                    assert!(g.is_nan(), "{what}");
                    continue;
                }
                let w = libm::atan2(fy as f64, fx as f64);
                if w == 0.0 || w.abs() < f32::MIN_POSITIVE as f64 {
                    // The zeros' signs, and a quotient below the normals
                    // flushed to one.
                    assert_eq!(g.to_bits() & 0x7fff_ffff, 0, "{what}");
                    assert_eq!(g.is_sign_negative(), want.is_sign_negative(), "{what}");
                    continue;
                }
                let rel = (g as f64 - w).abs() / w.abs();
                worst = worst.max(rel);
                assert!(rel <= ATAN2_BOUND, "{what}: {:.2}u", rel / u);
            }
        }
        println!("atan2: worst {:.3}u over {} pairs", worst / u, pairs.len());
    }

    /// A NaN of either sign and any payload comes out a NaN from every
    /// approximation: `SFPABS` leaves a negative NaN negative, which made
    /// `exp(-NaN)` `0` and `recip(-NaN)` `-inf` until 10.2e masked the sign.
    #[test]
    fn every_approximation_keeps_a_nan_of_either_sign() {
        let nans = [
            0x7fc0_0000u32,
            0xffc0_0000,
            0x7f80_0001,
            0xff80_0001,
            0xffff_ffff,
            0x7fff_ffff,
        ];
        let a: Vec<f32> = (0..1024)
            .map(|i| f32::from_bits(nans[i % nans.len()]))
            .collect();
        for kind in (0x100..=kind_sfpu::LAST).filter(|&k| {
            accuracy(k) == Accuracy::Approximate && operands(k) == Some(Operands::Unary)
        }) {
            let got = reference(kind, 0.5, &a, None, 32, 32);
            for (i, g) in got.iter().enumerate() {
                assert!(
                    g.is_nan(),
                    "kind {kind:#x}({:#010x}) = {g:e}",
                    a[i].to_bits()
                );
            }
        }
    }

    /// The two measured parts of each fit's error, over every float of its
    /// interval (every 64th in a debug build): the fit -- the `f32`
    /// coefficients' series, summed exactly, against the function -- and its
    /// evaluation -- the SFPU's Clenshaw sum (`fma_bh`) against that exact
    /// sum. [`ERF_BOUND`] allows the erfc pieces 5.25u together,
    /// [`ATAN_BOUND`] the atan piece [`ATAN_FIT_BOUND`].
    #[test]
    fn every_fit_and_its_evaluation_are_within_their_parts() {
        for piece in PIECES {
            let c = piece_cheb(piece);
            let (mid, half) = ((piece.hi + piece.lo) / 2.0, (piece.hi - piece.lo) / 2.0);
            let exact_sum = |a: f64| {
                let t = (a - mid) / half;
                let (mut b1, mut b2) = (0.0f64, 0.0f64);
                for k in (1..=piece.deg).rev() {
                    let b0 = c[k] as f64 + 2.0 * t * b1 - b2;
                    b2 = b1;
                    b1 = b0;
                }
                c[0] as f64 + t * b1 - b2
            };
            // `[0, 1)` holds 2^30 floats: every 64th of them.
            let stride = match (piece.fit, cfg!(debug_assertions)) {
                (Fit::Erfcx, false) => 1,
                (Fit::Erfcx, true) => 64,
                (_, false) => 64,
                (_, true) => 4096,
            };
            let (lo, hi) = ((piece.lo as f32).to_bits(), (piece.hi as f32).to_bits());
            let (mut fit, mut eval) = (0.0f64, 0.0f64);
            for b in (lo..hi).step_by(stride) {
                let a = f32::from_bits(b);
                let s = exact_sum(a as f64);
                let f = piece.fit.eval(a as f64);
                fit = fit.max((s - f).abs() / f);
                eval = eval.max((piece_clenshaw_f32(piece, a) as f64 - s).abs() / s);
            }
            let u = 1.0 / 16_777_216.0;
            println!(
                "{piece:?}: fit {:.3}u, evaluation {:.3}u",
                fit / u,
                eval / u
            );
            // The erfc pieces: the evaluation's map rounding dominates the
            // first, the coefficients' rounding to `f32` the second's fit.
            let budget = match piece.fit {
                Fit::Erfcx => 5.25 * u,
                Fit::AtanSqrt => ATAN_FIT_BOUND,
                Fit::AsinSqrt => ASIN_FIT_BOUND,
            };
            assert!(
                fit + eval < budget,
                "{piece:?}: fit {fit:e}, evaluation {eval:e}"
            );
        }
    }

    #[test]
    fn erf_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            0.4999,
            0.5,
            0.5001,
            1.0,
            -1.0,
            3.9199,
            3.92,
            3.9201,
            6.0,
            -6.0,
            1.0e-30,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::ERF,
            grid(-4.5, 4.5, 80_000)
                .chain(grid(-0.6, 0.6, 20_000))
                .chain(
                    binades(64)
                        .filter(|x| *x < 5.0 && *x > 1.0e-30)
                        .flat_map(|x| [x, -x]),
                )
                .chain(specials),
            libm::erf,
            ERF_BOUND,
        );
    }

    /// `gelu` against the exact `0.5 x erfc(-x/sqrt 2)` (Flex's formula's
    /// value), relative, over its whole range: the far negative side, where
    /// Flex's `1 + erf` cancels, included.
    #[test]
    fn gelu_is_within_its_derived_bound() {
        let mut xs: Vec<f32> = grid(-13.5, 6.0, 80_000).collect();
        xs.extend(grid(-1.5, 1.5, 20_000));
        xs.extend(
            binades(48)
                .filter(|x| *x < 13.0 && *x > 1.0e-30)
                .flat_map(|x| [x, -x]),
        );
        xs.extend([
            0.0,
            -0.0,
            0.707,
            -0.707,
            0.708,
            5.5437,
            -5.5437,
            -13.15,
            -13.2,
            1.0e-30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]);
        let exact = |x: f64| 0.5 * x * libm::erfc(-x / std::f64::consts::SQRT_2);
        let mut worst = 0.0f64;
        for chunk in xs.chunks(1024) {
            let mut a = chunk.to_vec();
            a.resize(1024, 1.0);
            let got = gelu_reference(&a, None, 32, 32);
            for (x, g) in chunk.iter().zip(&got) {
                let w = exact(*x as f64);
                if x.is_nan() || *x == f32::NEG_INFINITY {
                    assert!(g.is_nan(), "gelu({x}) = {g}");
                } else if w.is_infinite() {
                    assert_eq!(*g as f64, w, "gelu({x})");
                } else if w.abs() < f32::MIN_POSITIVE as f64 {
                    assert_eq!(*g, 0.0, "gelu({x:e}) = {g:e}: a denormal flushes");
                } else {
                    let rel = (*g as f64 - w).abs() / w.abs();
                    worst = worst.max(rel);
                    assert!(
                        rel <= GELU_BOUND,
                        "gelu({x:e}) = {g:e}, exact {w:e}: {rel:e}"
                    );
                }
            }
        }
        println!("gelu: worst {:.3} ulps", worst * 8_388_608.0);
    }

    #[test]
    fn gelu_backward_is_within_its_derived_bound() {
        let xs: Vec<f32> = grid(-13.0, 6.0, 60_000)
            .chain(grid(-1.0, 0.0, 20_000))
            .collect();
        let exact = |x: f64| {
            let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
            0.5 * libm::erfc(-x / std::f64::consts::SQRT_2) + x * phi
        };
        let mut worst = 0.0f64;
        for chunk in xs.chunks(1024) {
            let mut a = chunk.to_vec();
            a.resize(1024, 1.0);
            let g: Vec<f32> = (0..1024).map(|i| [1.0f32, -2.5, 0.125][i % 3]).collect();
            let got = gelu_reference(&a, Some(&g), 32, 32);
            for i in 0..chunk.len() {
                let w = g[i] as f64 * exact(a[i] as f64);
                let err = (got[i] as f64 - w).abs();
                let bound = gelu_backward_bound(a[i], g[i]);
                if w.abs() < f32::MIN_POSITIVE as f64 {
                    continue;
                }
                worst = worst.max(err / bound);
                assert!(
                    err <= bound,
                    "gelu'({:e}) * {} = {:e}, exact {w:e}: {err:e} > {bound:e}",
                    a[i],
                    g[i],
                    got[i]
                );
            }
        }
        println!("gelu_backward: worst {worst:.3} of the bound");
    }

    #[test]
    fn sqrt_is_within_one_ulp() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            4.0,
            -1.0,
            1.0e-40,
            -1.0e-40,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        // Flushed: a positive denormal's root is `+0`; a negative one is NaN.
        let root = |x: f64| {
            if x > 0.0 && x < f32::MIN_POSITIVE as f64 {
                0.0
            } else {
                x.sqrt()
            }
        };
        // One ulp of the correctly rounded root is at most 1.5 ulps of the
        // exact one: `1.5 * 2^-23` relative.
        sweep(
            kind_sfpu::SQRT,
            binades(256).chain(grid(0.0, 16.0, 20_000)).chain(specials),
            root,
            1.5 / 8_388_608.0,
        );
    }

    #[test]
    fn rsqrt_is_within_its_derived_bound() {
        let specials = [0.0, -0.0, 1.0, 4.0, f32::MAX, f32::INFINITY, f32::NAN, -2.0];
        let r = |x: f64| {
            if x == 0.0 || (x > 0.0 && x < f32::MIN_POSITIVE as f64) {
                f64::INFINITY.copysign(x)
            } else {
                1.0 / x.sqrt()
            }
        };
        sweep(
            kind_sfpu::RSQRT,
            binades(256).chain(grid(0.01, 16.0, 20_000)).chain(specials),
            r,
            RSQRT_BOUND,
        );
    }

    #[test]
    fn log1p_is_within_its_derived_bound() {
        // Denormal inputs come back as themselves (`log1p(x) = x` there, bits
        // and all), which the sweep's flush-to-zero rule would refuse; the
        // smallest normals are in.
        let specials = [
            0.0,
            -0.0,
            -1.0,
            -2.0,
            1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            5.9604645e-8,
            -5.9604645e-8,
            1.0e-7,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::LOG1P,
            grid(-0.999, 3.0, 40_000)
                .chain(grid(-1.0e-3, 1.0e-3, 20_000))
                .chain(binades(64).filter(|x| *x > 1.0e-30))
                .chain(
                    binades(64)
                        .filter(|x| *x > 1.0e-30 && *x < 0.99)
                        .map(|x| -x),
                )
                .chain(specials),
            f64::ln_1p,
            LOG1P_BOUND,
        );
    }
}

#[cfg(any())]
mod arity {
    use super::*;

    /// `operands` says what `program` builds, for every kind there is: the
    /// shape check and the kernel cannot disagree.
    #[test]
    fn operands_agrees_with_program() {
        for k in (1..=kind::LAST)
            .chain(0x100..=kind_sfpu::LAST)
            .chain([kind_sfpu::INDEX_TO_I32])
            .chain(0x170..=0x17e)
            .chain(0x180..=0x18d)
            .chain(0x190..=0x196)
            .chain(0x161..=0x165)
        {
            // Scalars any kind takes: `CLAMP`'s bounds uncrossed.
            let got = program2(k, [-0.5, 0.5]).map(|(o, _)| o);
            assert_eq!(operands(k), got, "kind {k:#x}");
        }
    }
}

#[cfg(any())]
mod s2 {
    use super::kind_sfpu::*;
    use super::*;

    /// Both zeros, both infinities, NaNs of either sign with payloads,
    /// denormals, the extremes, and ordinary values: every pairing of them
    /// lands in one 32x32 tile.
    const SPECIALS: [u32; 16] = [
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0xffc0_0001,
        0x0000_0001,
        0x8040_0000,
        0x7f7f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x4000_0000,
        0xbf00_0000,
        0x3e4c_cccd,
        0xc2c8_0000,
    ];

    fn pairs() -> (Vec<f32>, Vec<f32>) {
        let a = (0..1024)
            .map(|i| f32::from_bits(SPECIALS[i % 16]))
            .collect();
        let b = (0..1024)
            .map(|i| f32::from_bits(SPECIALS[(i / 16) % 16]))
            .collect();
        (a, b)
    }

    fn b(x: bool) -> f32 {
        f32::from_bits(u32::from(x))
    }

    /// The SFPU's arithmetic flushes a denormal operand to a signed zero
    /// before it computes (numerics row D): the host's arithmetic, so.
    fn ftz(x: f32) -> f32 {
        if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
            f32::from_bits(x.to_bits() & 0x8000_0000)
        } else {
            x
        }
    }

    /// The device's bits for the host's `want`: the same bits, or -- where
    /// `products` says the op multiplies -- a flushed denormal's signed zero,
    /// or any NaN for a NaN.
    fn same(got: f32, want: f32, products: bool) -> bool {
        let (g, w) = (got.to_bits(), want.to_bits());
        g == w
            || (products && want.is_nan() && got.is_nan())
            || (products && want != 0.0 && want.abs() < f32::MIN_POSITIVE && g == w & 0x8000_0000)
    }

    fn check(kind: u32, scalars: [f32; 2], host: impl Fn(f32, f32) -> f32, products: bool) {
        let (a, bv) = pairs();
        let inputs: Vec<&[f32]> = match operands(kind).unwrap() {
            Operands::Unary => vec![&a],
            _ => vec![&a, &bv],
        };
        let got = reference_op(kind, scalars, Broadcast::None, &inputs, 32, 32);
        for i in 0..1024 {
            let want = host(a[i], bv[i]);
            assert!(
                same(got[i], want, products),
                "kind {kind:#x} {scalars:?}: ({:#010x}, {:#010x}) gave {:#010x}, the host {:#010x}",
                a[i].to_bits(),
                bv[i].to_bits(),
                got[i].to_bits(),
                want.to_bits()
            );
        }
    }

    /// The power-of-two scale kinds the accurate matmul slices with, against the one-line host
    /// definitions, bit for bit on every special: zeros and denormals (`e = 0`), `f32::MAX`
    /// (`e = 254`), infinities and NaNs (`e = 255`), and ordinary values of both signs.
    #[test]
    fn the_power_of_two_scale_kinds_are_the_exponent_field() {
        check(
            POW2_FLOOR,
            [0.0; 2],
            |x, _| f32::from_bits(x.to_bits() & 0x7f80_0000),
            false,
        );
        check(
            POW2_FLOOR_INV,
            [0.0; 2],
            |x, _| {
                let e = (x.to_bits() >> 23) & 0xff;
                f32::from_bits(if e >= 254 {
                    0x0080_0000
                } else {
                    (254 - e) << 23
                })
            },
            false,
        );
        // The meaning, on normal values: `x * inverse` lands in [1, 2), and `floor * inverse` is 1.
        let (a, _) = pairs();
        let inv = reference_op(POW2_FLOOR_INV, [0.0; 2], Broadcast::None, &[&a], 32, 32);
        let floor = reference_op(POW2_FLOOR, [0.0; 2], Broadcast::None, &[&a], 32, 32);
        let mut normal = 0;
        for i in 0..1024 {
            let e = (a[i].to_bits() >> 23) & 0xff;
            if (1..254).contains(&e) {
                let m = f64::from(a[i].abs()) * f64::from(inv[i]);
                assert!((1.0..2.0).contains(&m), "{:#x}: {m}", a[i].to_bits());
                assert_eq!(f64::from(floor[i]) * f64::from(inv[i]), 1.0);
                normal += 1;
            }
        }
        assert!(normal > 256, "the specials hold too few normal values");
    }

    #[test]
    fn the_unary_ops_are_the_host_s_bit_for_bit() {
        check(NEG, [0.0; 2], |x, _| -x, false);
        check(ABS, [0.0; 2], |x, _| x.abs(), false);
        let sign = |x: f32| {
            if x.is_nan() {
                x
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        check(SIGN, [0.0; 2], move |x, _| sign(x), false);
        for (lo, hi) in [(-1.0f32, 1.0f32), (0.0, 0.5), (-0.0, 0.0), (-2.0, -2.0)] {
            check(CLAMP, [lo, hi], move |x, _| x.clamp(lo, hi), false);
        }
        // Flex's `x.max(s)`, measured: a NaN on either side gives the other,
        // and equal values the scalar.
        let max = |x: f32, s: f32| {
            if s.is_nan() || x > s {
                x
            } else {
                s
            }
        };
        let min = |x: f32, s: f32| {
            if s.is_nan() || x < s {
                x
            } else {
                s
            }
        };
        for s in [0.0f32, -0.0, 1.0, -0.5, f32::NAN, f32::INFINITY] {
            check(CLAMP_MIN, [s, 0.0], move |x, _| max(x, s), false);
            check(CLAMP_MAX, [s, 0.0], move |x, _| min(x, s), false);
        }
        for ns in [0.01f32, -3.0, 0.0] {
            // The product of a flushed operand; the pass-through keeps `x`.
            let leaky = move |x: f32, _| if x >= 0.0 { x } else { ns * ftz(x) };
            check(LEAKY_RELU, [ns, 0.0], leaky, true);
        }
        for (al, be) in [(0.2f32, 0.5f32), (1.0 / 6.0, 0.5), (-1.0, 0.0)] {
            check(
                HARD_SIGMOID,
                [al, be],
                move |x, _| ftz(ftz(al * ftz(x)) + be).clamp(0.0, 1.0),
                true,
            );
        }
        let prelu = |x: f32, a: f32| if x >= 0.0 { x } else { ftz(a) * ftz(x) };
        check(PRELU, [0.0; 2], prelu, true);
        check(IS_NAN, [0.0; 2], |x, _| b(x.is_nan()), false);
        check(IS_INF, [0.0; 2], |x, _| b(x.is_infinite()), false);
    }

    #[test]
    fn the_comparisons_are_ieee_s() {
        for (t, sk) in COMPARES {
            check(t, [0.0; 2], move |x, y| b(ieee_compare(t, x, y)), false);
            for s in [
                0.0f32,
                -0.0,
                1.0,
                -0.5,
                f32::NAN,
                f32::NEG_INFINITY,
                f32::from_bits(1),
            ] {
                check(sk, [s, 0.0], move |x, _| b(ieee_compare(t, x, s)), false);
            }
        }
    }

    #[test]
    fn the_masks_pass_bits_through() {
        let (a, v) = pairs();
        let m: Vec<f32> = (0..1024).map(|i| b((i * 7) % 3 == 0)).collect();
        for value in [2.5f32, -0.0, f32::NAN] {
            let got = reference_op(MASK_FILL, [value, 0.0], Broadcast::None, &[&a, &m], 32, 32);
            for i in 0..1024 {
                let want = if m[i].to_bits() != 0 { value } else { a[i] };
                assert_eq!(got[i].to_bits(), want.to_bits(), "fill {i}");
            }
        }
        let got = reference_op(MASK_WHERE, [0.0; 2], Broadcast::None, &[&a, &m, &v], 32, 32);
        for i in 0..1024 {
            let want = if m[i].to_bits() != 0 { v[i] } else { a[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "where {i}");
        }
    }
}

#[cfg(any())]
mod intbool {
    //! The integer and boolean selects and `int_abs`, held to host oracles through the
    //! interpreter. The bit patterns are the ones an `f32` path would damage: NaN payloads, the sign
    //! bit alone, the denormal one, and `2^24 + 1`.

    use super::super::interp::Vector;
    use super::kind_sfpu::*;
    use super::*;
    use crate::sfpu::kernel::C_ROW;

    const PATTERNS: [u32; 12] = [
        0x7fc0_0001,
        0x8000_0000,
        0x0000_0001,
        16_777_217,
        0x7f80_0000,
        0xffc0_0000,
        0x007f_ffff,
        0xffff_ffff,
        0,
        0x7fff_ffff,
        0x8000_0001,
        0x0100_0001,
    ];

    fn lanes(seed: u32) -> Vec<u32> {
        let mut s = seed | 1;
        (0..1024)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                if i % 3 == 0 {
                    PATTERNS[(i / 3 + seed as usize) % PATTERNS.len()]
                } else {
                    s
                }
            })
            .collect()
    }

    fn run(kind: u32, scalar: u32, rows: &[(u32, &[u32])]) -> Vec<u32> {
        let (_, math) = program(kind, f32::from_bits(scalar)).unwrap();
        let mut v = Vector::new();
        for (row, data) in rows {
            v.put_tile(*row as usize, data);
        }
        v.run(&math)
            .unwrap_or_else(|e| panic!("kind {kind:#x}: {e}"));
        v.tile(OUT_ROW as usize)
    }

    #[test]
    fn signatures_and_operands() {
        use crate::tensor::Elem::{Bool, I32};
        let sig = |k| (elems(k).inputs.to_vec(), elems(k).out);
        assert_eq!(sig(INT_MASK_FILL), (vec![I32, Bool], I32));
        assert_eq!(sig(BOOL_MASK_FILL), (vec![Bool, Bool], Bool));
        assert_eq!(sig(INT_MASK_WHERE), (vec![I32, Bool, I32], I32));
        assert_eq!(sig(BOOL_MASK_WHERE), (vec![Bool, Bool, Bool], Bool));
        assert_eq!(sig(INT_ABS), (vec![I32], I32));
        assert_eq!(operands(INT_ABS), Some(Operands::Unary));
        assert_eq!(operands(INT_MASK_FILL), Some(Operands::Binary));
        assert_eq!(operands(INT_MASK_WHERE), Some(Operands::Ternary));
        assert!(broadcasts(INT_MASK_FILL) && broadcasts(BOOL_MASK_FILL));
    }

    /// `|x|` wraps: `i32::MIN` stays itself, every other lane is `wrapping_abs`, whatever its bits
    /// look like as a float.
    #[test]
    fn int_abs_is_wrapping_abs_on_every_lane() {
        for seed in [3, 5, 7] {
            let a = lanes(seed);
            let got = run(INT_ABS, 0, &[(A_ROW, &a)]);
            for (i, (&x, &g)) in a.iter().zip(&got).enumerate() {
                assert_eq!(g, (x as i32).wrapping_abs() as u32, "lane {i}: {x:#010x}");
            }
        }
        let got = run(INT_ABS, 0, &[(A_ROW, &[i32::MIN as u32; 1024])]);
        assert!(got.iter().all(|&g| g == i32::MIN as u32));
    }

    /// The selects are predicated moves of raw words: each lane's output is exactly the chosen
    /// operand's word. The program is the float `MASK_WHERE`'s, instruction for instruction.
    #[test]
    fn selects_move_raw_words() {
        for kind in [INT_MASK_WHERE, BOOL_MASK_WHERE] {
            assert_eq!(
                program(kind, 0.0).unwrap().1,
                program(MASK_WHERE, 0.0).unwrap().1,
                "kind {kind:#x}"
            );
            let (a, c) = (lanes(11), lanes(13));
            let mask: Vec<u32> = lanes(17).iter().map(|w| w & 1).collect();
            let got = run(kind, 0, &[(A_ROW, &a), (B_ROW, &mask), (C_ROW, &c)]);
            for i in 0..1024 {
                assert_eq!(got[i], if mask[i] != 0 { c[i] } else { a[i] }, "lane {i}");
            }
        }
    }

    /// The fill's scalar is the exact 32-bit word, for every pattern, and only masked lanes take it.
    #[test]
    fn fills_carry_raw_scalar_bits() {
        let a = lanes(19);
        let mask: Vec<u32> = lanes(23).iter().map(|w| (w >> 5) & 1).collect();
        for kind in [INT_MASK_FILL, BOOL_MASK_FILL] {
            for scalar in PATTERNS {
                let got = run(kind, scalar, &[(A_ROW, &a), (B_ROW, &mask)]);
                for i in 0..1024 {
                    assert_eq!(
                        got[i],
                        if mask[i] != 0 { scalar } else { a[i] },
                        "kind {kind:#x}, scalar {scalar:#010x}, lane {i}"
                    );
                }
            }
        }
    }
}

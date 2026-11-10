//! `MathMode::{Precise, Approx}` (`hardware-coverage.md` S10): a fast,
//! opt-in approximate mode for the transcendentals.
//!
//! Every S3/S4 program in [`super::ops`] is built for a derived bound of a few
//! ulps (`EXP_BOUND`, `LOG_BOUND`, ...): degree-7 Taylor series, two or three
//! Newton steps, an exactness correction, every special value. Most training
//! does not need that. [`MathMode::Approx`] runs a *different program* for the
//! same op -- a hard-coded low-degree minimax polynomial, one Newton step on
//! the hardware's reciprocal seed, one reduction constant, a coarser range --
//! with its own, looser, **derived** bound (stated beside each program, held by
//! `step145_approx_math` and this module's sweeps), and special values handled
//! only where an ML input meets them: NaN of either sign, `±inf`, `±0` (and the
//! range's two ends).
//!
//! # How the mode reaches a program
//!
//! The mode is **carried in the op's kind**: every Approx program has a kind of
//! its own ([`kind`], `0x1d0..`), and [`MathMode::lower`] turns a Precise kind
//! into its Approx twin. `tensor::Eltwise::kind` is part of every program memo
//! and cache key (`sfpu_programs`, `sfpu_group`, the resident program cache
//! hashes the words), so a session that alternates modes can never be handed
//! the other mode's program -- there is no key that omits the mode, because the
//! mode *is* part of the key's kind. (An extra `Eltwise` field would have
//! touched every construction site in four crates for no added safety.)
//! `Session::set_math_mode` / `TT_MATH=approx` apply [`MathMode::lower`] at the
//! one entry every element-wise op takes (`Session::eltwise3`).
//!
//! This is **not** the retired exact mode (`TT_EXACT`, Flex's bits): that asked
//! "the host's bits or the device's"; this asks "how many ulps".
//!
//! Precise is the default and its programs are untouched: `step29_exp_log`,
//! `step44_algebraic`, `step45_exp_family` hold them unchanged.
//!
//! # Constants
//!
//! `LReg[11]` and `LReg[12]` hold `0x7fff_ffff` and `+inf` for every Approx
//! program, and `LReg[13]`, `LReg[14]` its two leading polynomial coefficients
//! (`Program::constant`, the F2 prologue), so the row loop reloads none of
//! them. Every program writes the constants it reads.
//!
//! # Instruction counts
//!
//! [`instructions_per_tile`] counts what the math thread executes for one tile
//! (a `REPLAY` counts its body). `step145_approx_math` prints the table; the
//! silicon-only `approx_per_tile_saving` in `silicon_perf` measures the time.

use super::super::code::Code;
use super::kernel::{A_ROW, OUT_ROW};
use super::ops::kind_sfpu;
use super::{Cond, ConfigLReg, Format, LReg, Program};
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::defs;

/// Whether a transcendental runs its full-accuracy program or the fast
/// approximate one. `Precise` is the default.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MathMode {
    /// The programs of `hardware-coverage.md` S3/S4, held to `EXP_BOUND` and
    /// the rest.
    #[default]
    Precise,
    /// The programs of this module, held to the `*_APPROX_BOUND`s.
    Approx,
}

/// The Approx programs' kinds (`0x1d0..=0x1df` is the Approx range).
pub mod kind {
    /// `e^a`, [`super::EXP_APPROX_BOUND`] on `[-87, 88)`.
    pub const EXP: u32 = 0x1d0;
    /// `ln a`, [`super::LOG_APPROX_BOUND`].
    pub const LOG: u32 = 0x1d1;
    /// `1 / a`, [`super::RECIP_APPROX_BOUND`].
    pub const RECIP: u32 = 0x1d2;
    /// The logistic function, [`super::SIGMOID_APPROX_BOUND`].
    pub const SIGMOID: u32 = 0x1d3;
    /// `tanh a`, [`super::TANH_APPROX_BOUND`].
    pub const TANH: u32 = 0x1d4;
    /// `0.5 a (1 + erf(a / sqrt 2))`, within [`super::gelu_approx_bound`].
    pub const GELU: u32 = 0x1d5;
    /// The first and last Approx kind.
    pub const FIRST: u32 = EXP;
    pub const LAST: u32 = GELU;
}

/// Whether `kind` is one of this module's programs.
pub const fn is_approx(kind: u32) -> bool {
    kind >= kind::FIRST && kind <= kind::LAST
}

/// `(precise, approx)` for every op with an Approx program.
pub const TWINS: [(u32, u32); 6] = [
    (kind_sfpu::EXP, kind::EXP),
    (kind_sfpu::LOG, kind::LOG),
    (kind_sfpu::RECIP, kind::RECIP),
    (kind_sfpu::SIGMOID, kind::SIGMOID),
    (kind_sfpu::TANH, kind::TANH),
    (kind_sfpu::GELU, kind::GELU),
];

impl MathMode {
    /// The mode `TT_MATH` names: `precise` (also unset) or `approx`. Anything
    /// else is refused, naming the variable, rather than silently running the
    /// mode the user did not ask for.
    pub fn parse(value: Option<&str>) -> Result<MathMode, String> {
        match value {
            None | Some("") | Some("precise") => Ok(MathMode::Precise),
            Some("approx") => Ok(MathMode::Approx),
            Some(v) => Err(format!("TT_MATH={v}: expected precise or approx")),
        }
    }

    /// [`MathMode::parse`] of the `TT_MATH` environment variable.
    pub fn from_env() -> Result<MathMode, String> {
        MathMode::parse(std::env::var("TT_MATH").ok().as_deref())
    }

    /// `kind` as this mode runs it: itself in `Precise`; in `Approx`, its
    /// Approx twin where it has one ([`TWINS`]) -- `exp`, `log`, `recip`,
    /// `sigmoid`, `tanh`, `gelu` -- and itself otherwise (an
    /// op with no Approx program stays Precise: `div`, `pow`, the trig ops).
    pub fn lower(self, kind: u32) -> u32 {
        match self {
            MathMode::Precise => precise_of(kind).unwrap_or(kind),
            MathMode::Approx => TWINS.iter().find(|(p, _)| *p == kind).map_or(kind, |(_, a)| *a),
        }
    }

    /// The mode `kind` runs in.
    pub fn of(kind: u32) -> MathMode {
        if is_approx(kind) {
            MathMode::Approx
        } else {
            MathMode::Precise
        }
    }
}

/// The Precise twin of an Approx kind.
pub fn precise_of(kind: u32) -> Option<u32> {
    TWINS.iter().find(|(_, a)| *a == kind).map(|(p, _)| *p)
}

// ---------------------------------------------------------------------------
// The fits. Computed by a weighted Remez exchange (`tests::remez`, which
// recomputes them and holds these to it), hard-coded here as the f64 values the
// programs round to f32.
// ---------------------------------------------------------------------------

/// `e^r` on `[-ln2/2, ln2/2]`, degree 3, minimising the *relative* error:
/// `a0 + a1 r + a2 r^2 + a3 r^3`. The minimax error is [`EXP_FIT_ERROR`].
pub const EXP_COEFFS: [f64; 4] = [
    0.9999280735393948,
    1.0001641857658377,
    0.5049632641799273,
    0.16566842342911733,
];
/// The relative error of [`EXP_COEFFS`] against `e^r`: the equioscillation
/// level of the exchange (five alternating extrema).
pub const EXP_FIT_ERROR: f64 = 7.4782e-5;

/// `(e^r - 1) / r` on `[-ln2/2, ln2/2]`, degree 3, relative error:
/// `b0 + b1 r + b2 r^2 + b3 r^3`; `tanh`'s `e^z - 1 = r g(r)` has `g`'s
/// relative error, with no cancellation near zero.
pub const EXPM1_COEFFS: [f64; 4] = [
    0.9999851073761886,
    0.5000125187282544,
    0.16766713192791918,
    0.04166658725395594,
];
pub const EXPM1_FIT_ERROR: f64 = 1.4993e-5;

/// `ln(1 + f) / f` on `[sqrt(2)/2 - 1, sqrt(2) - 1]`, degree 4, relative
/// error: `c0 + c1 f + ... + c4 f^4`.
pub const LOG_COEFFS: [f64; 5] = [
    0.9999661813597593,
    -0.49945064750654694,
    0.3363888423653176,
    -0.27094599437667555,
    0.17658054231910456,
];
pub const LOG_FIT_ERROR: f64 = 5.0192e-5;

/// `gelu`'s `Phi(x)` as `sigmoid(c1 x + c3 x^3)`: the two parameters of
/// Page's 1977 fit (also Bowling et al. 2009), polished by a local search for
/// the minimal supremum over `x >= 0`; [`GELU_FIT_ERROR`] is that supremum,
/// computed (`tests::gelu_fit_error_is_the_supremum`: grid maximum `1.4032e-4`, plus the derivative bound times half the grid step).
pub const GELU_C1: f64 = 1.5976029;
pub const GELU_C3: f64 = 0.0705641;
pub const GELU_FIT_ERROR: f64 = 1.41e-4;

/// `1 / (1 + 2^-24)`-style unit: `u = 2^-24`, half an ulp of 1.0.
const U: f64 = 1.0 / 16_777_216.0;

/// The hardware reciprocal seed's relative error (`SFPARECIP.md`: `0.9944 <
/// r x < 1.0054`), and one Newton step's result: `e1 = e0^2` plus the
/// roundings (the product `x y`, the correction, the final fma: under `3u`).
pub const RECIP_SEED: f64 = 0.0056;
/// `e1`, relative, after one Newton step on the seed: `0.0056^2 = 3.136e-5`,
/// `3u = 1.8e-7`.
pub const RECIP_STEP: f64 = RECIP_SEED * RECIP_SEED + 3.0 * U;

// Each derivation below is `sum of the parts`, written out; the sweeps in
// this module's tests and `step145_approx_math` hold the programs to them.

/// [`exp_body`]'s derived bound, relative to `e^x`, for `x` in `[-87, 88)`.
///
/// `e^x = 2^n e^r`, `n = round(x log2 e)`, `r = x - n ln2` (one constant: the
/// reduction is the program's whole range reduction), `|r| <= ln2/2` up to the
/// reduction's own error. Parts, relative to `e^x`:
///
/// * the polynomial: [`EXP_FIT_ERROR`] `= 7.478e-5` (Remez, relative);
/// * its `f32` coefficients: at most `u` each, weighted by `|a_j r^j|`, summing
///   below `(1 + 0.35 + 0.06 + 0.007) u = 8.5e-8`;
/// * Horner's three roundings: `3u = 1.8e-7`;
/// * the reduction: `n ln2` is rounded once (`SFPMAD` is not fused: take the
///   product rounded, `|n ln2| <= 127 * 0.6932 = 88.0`, so `88 u = 5.25e-6`) and
///   `ln2`'s own `f32` error is `1.9e-9 |n| <= 2.4e-7`; the error of `r` is
///   carried one for one by `e^r`: `5.5e-6`;
/// * the exponent add is an integer add: exact.
///
/// Total `7.478e-5 + 8.5e-8 + 1.8e-7 + 5.5e-6 = 8.05e-5`.
/// Below `-87` the result is `+0` (it would be within a factor 1.4 of the
/// normal range's end: the exponent add would leave it, so the range stops
/// there), from `88` `+inf` (the exponent field would reach 255 at `n = 128`),
/// a NaN of either sign stays one.
pub const EXP_APPROX_BOUND: f64 = 8.2e-5;

/// [`log_body`]'s derived bound, relative to `ln x`, for a positive normal `x`.
///
/// `x = 2^e m`, `m` in `[sqrt2/2, sqrt2)` (a halving where `m` comes out above
/// `sqrt 2`, as the Precise program does, so `e ln2` and `ln m` never cancel
/// to less than `|ln m|`); `f = m - 1` is exact (Sterbenz); `ln m = f q(f)`.
/// Parts, relative to `ln x`:
///
/// * the polynomial: [`LOG_FIT_ERROR`] `= 5.019e-5` (Remez, relative to
///   `ln(1+f)/f`, so relative to `ln m` too, and `|ln m| <= |ln x|`);
/// * the coefficients' `f32` rounding: `< 1.5 u = 9e-8`;
/// * Horner's four roundings and the product by `f`: `5u = 3e-7`;
/// * `e ln2 + ln m` in one fma (`u`), `ln2`'s `f32` error `2.7e-9` relative.
///
/// Total `5.019e-5 + 9e-8 + 3e-7 + 6e-8 + 3e-9 = 5.07e-5`. `ln(±0)` (and a
/// denormal, which flushes) is `-inf`, a negative number NaN, `ln(+inf) = +inf`,
/// a NaN stays one.
pub const LOG_APPROX_BOUND: f64 = 5.2e-5;

/// [`recip_body`]'s derived bound, relative to `1/x`, for `2^-126 <= |x| <=
/// 2^110`: the seed's `e0 < 0.0056`, one Newton step `y <- y + y (1 - x y)`
/// leaves `e1 = e0^2` (`3.136e-5`) plus its roundings (`3u`): `3.15e-5`.
/// From `2^110` the correction `y (1 - x y) < 2^-110 * 0.0056` meets the
/// denormal range and flushes, so the result is the seed alone:
/// [`RECIP_SEED_BOUND`] from `2^110` to `2^126`; from `2^126` the result is
/// `±0`. `1/±0 = ±inf`, `1/±inf = ±0`, a NaN stays one.
pub const RECIP_APPROX_BOUND: f64 = 3.2e-5;
/// [`recip_body`]'s bound where the Newton step flushes (`|x| > 2^110`).
pub const RECIP_SEED_BOUND: f64 = 5.6e-3;
/// `2^110` as `f32` bits: where [`RECIP_APPROX_BOUND`] ends.
pub const RECIP_RANGE_BITS: u32 = 0x7680_0000;

/// [`sigmoid_body`]'s derived bound, relative.
///
/// With `e = e^-|x|` ([`EXP_APPROX_BOUND`] `8.05e-5`, on `-|x|`, which is in
/// the range but for the underflow end, where `e` is made `0`), `den = 1 + e`
/// (one rounding, `u`), `y ~ 1/den` by the seed and one Newton step
/// ([`RECIP_STEP`] `3.15e-5`; `den` in `[1, 2]`), and `x >= 0 ? y : e y`:
/// the `x >= 0` branch carries `e`'s error scaled by `e/(1+e) <= 1/2`, the
/// other by `1/(1+e) <= 1`, plus one rounding more for the product. Worst
/// (`x < 0`): `8.05e-5 + 3.15e-5 + 3u (1.8e-7) = 1.12e-4`.
/// `±inf` give `1` and `0`, a NaN stays one; from `x = -87` the result is `0`.
pub const SIGMOID_APPROX_BOUND: f64 = 1.13e-4;

/// [`tanh_body`]'s derived bound, relative.
///
/// `tanh a = t / (t + 2)`, `t = e^(2a) - 1 = 2^n r g(r) + (2^n - 1)` with
/// `g = (e^r - 1)/r` fitted relatively: `t` carries [`EXPM1_FIT_ERROR`]
/// `1.5e-5`, its `f32` coefficients `< 1.5 u`, Horner's three roundings and the
/// product by `r` `4u`, and the reduction's error `delta` amplified by
/// `e^z/(e^z - 1) <= 3.41` at the smallest `z` that reduces (`n >= 1`) and by
/// 1 for large `z`: `<= 1.2e-6` (`n <= 26`: `26 (ln2 u + 1.9e-9) = 1.1e-6`).
/// The quotient scales `t`'s error by `2/(t + 2) <= 1`; `t + 2` rounds once
/// (`u`), the reciprocal ([`RECIP_STEP`]) `3.15e-5`, the product `u`.
/// Total `1.5e-5 + 9e-8 + 2.4e-7 + 1.2e-6 + 3.15e-5 + 3u (1.8e-7) = 4.8e-5`.
/// From `|x| = 9` (`2z = 18`) the result is `±1` (`1 - tanh 9 = 2.5e-8`);
/// `±0` stays `±0`; a denormal input flushes (`tanh` of it is `0`); a NaN stays one.
pub const TANH_APPROX_BOUND: f64 = 5.2e-5;

/// [`gelu_body`]'s derived bound at `x`, absolute: `gelu x = x Phi(x)`, and the
/// program computes `x s`, `s = sigmoid_approx(c1 x + c3 x^3)`:
///
/// * `|sigmoid(c1 x + c3 x^3) - Phi(x)| <= ` [`GELU_FIT_ERROR`] `= 1.404e-4`
///   (the fit's supremum, computed over `[0, 8]`; beyond 8 both are within
///   `1e-15` of 1; `Phi` is odd about `1/2` and so is the fit);
/// * the logistic's evaluation, [`SIGMOID_APPROX_BOUND`] of `s <= 1`;
/// * `z`'s three roundings move `s` by `sigma'(z) |z| 3u <= 0.225 * 3u = 4e-8`;
/// * the final product: `u |gelu|`.
///
/// So `|got - gelu x| <= |x| (1.404e-4 + 1.13e-4 + 4e-8) + u |gelu x|`.
pub fn gelu_approx_bound(x: f64, gelu: f64) -> f64 {
    x.abs() * (GELU_FIT_ERROR + SIGMOID_APPROX_BOUND + 4e-8) + U * gelu.abs()
}

// ---------------------------------------------------------------------------
// The programs.
// ---------------------------------------------------------------------------

const MASK: ConfigLReg = ConfigLReg::L11;
const INFINITY: ConfigLReg = ConfigLReg::L12;
const C0: ConfigLReg = ConfigLReg::L13;
const C1: ConfigLReg = ConfigLReg::L14;

fn mask() -> LReg {
    MASK.lreg()
}
fn inf() -> LReg {
    INFINITY.lreg()
}

/// Bits of an `f64` fit coefficient as the program's `f32` immediate.
fn bits(c: f64) -> u32 {
    (c as f32).to_bits()
}

/// A unary program: `L0` in (raw bits), `L7` out (FP32 store, which flushes a
/// denormal result as the Precise programs' does), the constants in `LReg[11..]`.
fn unary(c13: Option<u32>, c14: Option<u32>, body: impl Fn(&mut Program)) -> Code {
    let mut p = Program::new();
    p.constant(MASK, 0x7fff_ffff);
    p.constant(INFINITY, 0x7f80_0000);
    if let Some(b) = c13 {
        p.constant(C0, b);
    }
    if let Some(b) = c14 {
        p.constant(C1, b);
    }
    p.for_each_row_group(64, |p, o| {
        p.load(LReg::L0, Format::Int32, A_ROW + o);
        body(p);
        p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
    });
    p.finish_code()
}

/// If `x` is a NaN (of either sign), `d = x`: the sign-free magnitude `t`
/// above `+inf`.
fn keep_nan(p: &mut Program, x: LReg, t: LReg, d: LReg) {
    p.and(x, mask(), t);
    p.if_(Cond::Less(inf(), t), |p| p.mov(x, d));
}

/// `e^x` of `x` into `d`; `LReg[13]`, `LReg[14]` hold `a3`, `a2`. `s` scratch.
/// With `ends`, the range's upper end (`+inf`) and NaNs (see
/// [`EXP_APPROX_BOUND`]); without, only the underflow end, for an `x <= 0`.
fn exp_body(p: &mut Program, x: LReg, d: LReg, s: [LReg; 5], ends: bool) {
    let [k, magic, t, nf, r] = s;
    p.loadi(k, std::f32::consts::LOG2_E);
    p.loadi_bits(magic, 0x4b40_0000);
    // `t = x log2 e + 1.5 2^23`: the rounding to `n` in the low mantissa bits.
    p.mad(x, k, magic, t);
    p.sub(t, magic, nf);
    p.loadi(k, std::f32::consts::LN_2);
    p.nmad(nf, k, x, r);
    p.mad(C0.lreg(), r, C1.lreg(), d);
    p.loadi_bits(k, bits(EXP_COEFFS[1]));
    p.mad(d, r, k, d);
    p.loadi_bits(k, bits(EXP_COEFFS[0]));
    p.mad(d, r, k, d);
    // `n = bits(t) - bits(magic)`, into the exponent field.
    p.isub_from(t, magic);
    p.shl(magic, 23, magic);
    p.iadd(magic, d);
    p.loadi(k, -87.0);
    p.if_(Cond::Less(x, k), |p| p.mov(LReg::ZERO, d));
    if ends {
        p.loadi(k, 88.0);
        p.if_(Cond::LessEq(k, x), |p| p.mov(inf(), d));
        keep_nan(p, x, t, d);
    }
}

fn exp_code() -> Code {
    unary(Some(bits(EXP_COEFFS[3])), Some(bits(EXP_COEFFS[2])), |p| {
        use LReg as R;
        exp_body(p, R::L0, R::L7, [R::L1, R::L2, R::L3, R::L4, R::L5], true)
    })
}

/// `ln x` of `x` (`L0`) into `L7`; `LReg[13]`, `LReg[14]` hold `c4`, `c3`.
fn log_body(p: &mut Program) {
    use LReg as R;
    let (x, d, e, m, f, t) = (R::L0, R::L7, R::L1, R::L2, R::L3, R::L4);
    p.exponent(x, true, e);
    // Where `x` is negative `m` is too: the lanes are overwritten below.
    p.set_exponent(x, 127, m);
    p.loadi_bits(f, 0x3fb5_04f3); // sqrt(2)
    p.if_(Cond::Less(f, m), |p| {
        p.set_exponent(m, 126, m);
        p.iadd_imm(e, 1, e);
    });
    p.sub(m, R::ONE, f);
    p.mad(C0.lreg(), f, C1.lreg(), d);
    for c in [LOG_COEFFS[2], LOG_COEFFS[1], LOG_COEFFS[0]] {
        p.loadi_bits(t, bits(c));
        p.mad(d, f, t, d);
    }
    // `ln m = f q(f)`; `e` as a float; `ln x = e ln2 + ln m`.
    p.mul(d, f, d);
    p.loadi_bits(m, 0x4b40_0000);
    p.iadd(m, e);
    p.sub(e, m, e);
    p.loadi(m, std::f32::consts::LN_2);
    p.mad(e, m, d, d);
    // Every negative lane (`-0`, `-inf`, a negative NaN, a negative denormal
    // included) NaN first, then a zero or a denormal `-inf` (so `-0` is), then
    // `+inf` (itself) and a positive NaN (itself).
    p.if_(Cond::Lt0(x), |p| p.loadi_bits(d, 0x7fc0_0000));
    p.and(x, mask(), t);
    p.loadi_bits(f, 0x0080_0000);
    p.if_(Cond::Less(t, f), |p| p.loadi_bits(d, 0xff80_0000));
    // (`SFPLE` reads its first operand as `VD`, and `VD >= 12` would be a
    // template load: a config register may only be the second of a `LessEq`.)
    p.loadi_bits(f, 0x7f80_0000);
    p.if_(Cond::LessEq(f, x), |p| p.mov(x, d));
}

fn log_code() -> Code {
    unary(Some(bits(LOG_COEFFS[4])), Some(bits(LOG_COEFFS[3])), log_body)
}

/// `1/x` of `x` (`L0`) into `L7`: the hardware seed and one Newton step.
fn recip_body(p: &mut Program) {
    use LReg as R;
    let (x, y, t, ay, ax, d) = (R::L0, R::L1, R::L2, R::L3, R::L4, R::L7);
    p.approx_recip(x, y);
    p.nmad(x, y, R::ONE, t);
    p.mad(t, y, y, d);
    // The step met `0 * inf` (a zero or a denormal: the seed is `±inf`; an
    // infinity or a NaN: the seed is `±0`). A zero's result is the seed;
    // `1/±inf` is the seed (`±0`), and a NaN is itself.
    p.abs(y, ay);
    p.and(x, mask(), ax);
    p.loadi_bits(t, 0x7f80_0000);
    p.if_(Cond::LessEq(t, ay), |p| p.mov(y, d));
    p.if_(Cond::Eq0(ay), |p| {
        p.mov(y, d);
        p.if_(Cond::Less(inf(), ax), |p| p.mov(x, d));
    });
}

fn recip_code() -> Code {
    unary(None, None, recip_body)
}

/// `1/(1 + e)` or `e/(1 + e)` (by `x`'s sign) into `out`, `e = e^-|z|`
/// computed from `nz = -|z|` into `e`; `s` the exponential's five scratch
/// registers (three of them the quotient's again).
fn logistic(p: &mut Program, x: LReg, nz: LReg, e: LReg, s: [LReg; 5], out: LReg) {
    exp_body(p, nz, e, s, false);
    let (den, y, t) = (s[0], s[1], s[2]);
    p.add(e, LReg::ONE, den);
    p.approx_recip(den, y);
    p.nmad(den, y, LReg::ONE, t);
    p.mad(t, y, y, out);
    p.if_(Cond::Lt0(x), |p| p.mul(e, out, out));
}

fn sigmoid_body(p: &mut Program) {
    use LReg as R;
    let x = R::L0;
    p.abs(x, R::L1);
    p.neg(R::L1, R::L1);
    logistic(p, x, R::L1, R::L2, [R::L3, R::L4, R::L5, R::L6, R::L7], R::L7);
    keep_nan(p, x, R::L3, R::L7);
}

fn sigmoid_code() -> Code {
    unary(Some(bits(EXP_COEFFS[3])), Some(bits(EXP_COEFFS[2])), sigmoid_body)
}

/// `tanh x` of `x` (`L0`) into `L7`; `LReg[13]`, `LReg[14]` hold `b3`, `b2`.
fn tanh_body(p: &mut Program) {
    use LReg as R;
    let (x, d) = (R::L0, R::L7);
    let (z, k, magic, t, nf, r) = (R::L1, R::L2, R::L3, R::L4, R::L5, R::L6);
    // `z = 2|x|`; `n = round(z log2 e)`, `r = z - n ln2`.
    p.abs(x, z);
    p.add(z, z, z);
    p.loadi(k, std::f32::consts::LOG2_E);
    p.loadi_bits(magic, 0x4b40_0000);
    p.mad(z, k, magic, t);
    p.sub(t, magic, nf);
    p.loadi(k, std::f32::consts::LN_2);
    p.nmad(nf, k, z, r);
    // `p = r g(r)`, `g` by Horner.
    p.mad(C0.lreg(), r, C1.lreg(), d);
    p.loadi_bits(k, bits(EXPM1_COEFFS[1]));
    p.mad(d, r, k, d);
    p.loadi_bits(k, bits(EXPM1_COEFFS[0]));
    p.mad(d, r, k, d);
    p.mul(d, r, d);
    // `2^n` as a float: `(n + 127) << 23`; `t = 2^n p + (2^n - 1)`.
    p.isub_from(t, magic);
    p.iadd_imm(magic, 127, magic);
    p.shl(magic, 23, magic);
    p.sub(magic, R::ONE, t);
    p.mad(magic, d, t, d);
    // `q = t / (t + 2)` by the seed and one Newton step; `tanh = ±q`.
    p.loadi(k, 2.0);
    p.add(d, k, magic);
    p.approx_recip(magic, t);
    p.nmad(magic, t, R::ONE, nf);
    p.mad(nf, t, t, t);
    p.mul(d, t, nf);
    p.copy_sign(nf, x, d);
    // `2|x| >= 18`: `±1` (and an infinity); a NaN last.
    p.loadi(k, 18.0);
    p.if_(Cond::LessEq(k, z), |p| p.copy_sign(R::ONE, x, d));
    keep_nan(p, x, magic, d);
}

fn tanh_code() -> Code {
    unary(Some(bits(EXPM1_COEFFS[3])), Some(bits(EXPM1_COEFFS[2])), tanh_body)
}

/// `gelu x = x sigmoid(c1 x + c3 x^3)` of `x` (`L0`) into `L7`; `LReg[13]`,
/// `LReg[14]` hold the exponential's `a3`, `a2`.
fn gelu_body(p: &mut Program) {
    use LReg as R;
    let x = R::L0;
    // `z = x (c1 + c3 x^2)`, then `-|z|` (the sign is `x`'s: `c1`, `c3 > 0`).
    p.mul(x, x, R::L1);
    p.loadi_bits(R::L2, (GELU_C3 as f32).to_bits());
    p.loadi_bits(R::L3, (GELU_C1 as f32).to_bits());
    p.mad(R::L1, R::L2, R::L3, R::L1);
    p.mul(x, R::L1, R::L1);
    p.abs(R::L1, R::L1);
    p.neg(R::L1, R::L1);
    logistic(p, x, R::L1, R::L2, [R::L3, R::L4, R::L5, R::L6, R::L7], R::L6);
    p.mul(x, R::L6, R::L7);
    keep_nan(p, x, R::L3, R::L7);
}

fn gelu_code() -> Code {
    unary(Some(bits(EXP_COEFFS[3])), Some(bits(EXP_COEFFS[2])), gelu_body)
}

/// The Approx program for `kind`, if it is one of [`TWINS`]' second column.
pub fn code(kind: u32) -> Option<Code> {
    Some(match kind {
        kind::EXP => exp_code(),
        kind::LOG => log_code(),
        kind::RECIP => recip_code(),
        kind::SIGMOID => sigmoid_code(),
        kind::TANH => tanh_code(),
        kind::GELU => gelu_code(),
        _ => return None,
    })
}

/// How many instructions the math thread executes running `program` over one
/// tile: every instruction once, a `REPLAY` executing its recorded body each
/// time it is issued (`REPLAY.md`; `Vector::run` walks it the same way).
pub fn instructions_per_tile(program: &[Instruction]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < program.len() {
        let ins = program[i];
        if core::ptr::eq(ins.def(), &defs::REPLAY) {
            let op = |name| ins.operand(name).unwrap();
            let count = match op("Count") as usize {
                0 => 64,
                c => c,
            };
            if op("Load") != 0 {
                // Records `count` instructions that follow; `Exec` runs them.
                n += if op("Exec") != 0 { count } else { 0 };
                i += 1 + count;
            } else {
                n += count;
                i += 1;
            }
            continue;
        }
        n += 1;
        i += 1;
    }
    n
}

#[cfg(any())]
mod tests {
    //! `approx.rs`'s interpreter gates: the fits against a Remez derivation, every
    //! program against an f64 reference over its range, special values, the mode.
    #![allow(clippy::needless_range_loop, clippy::type_complexity)]

    use super::*;
    use crate::sfpu::ops::{self, kind_sfpu, reference};

    /// The program for `kind` over `a`, by the interpreter, `256` columns wide.
    fn run(kind: u32, a: &[f32]) -> Vec<f32> {
        let cols = 256;
        let rows = a.len().div_ceil(cols);
        let mut padded = a.to_vec();
        padded.resize(rows * cols, 1.0);
        let mut out = reference(kind, 0.0, &padded, None, rows, cols);
        out.truncate(a.len());
        out
    }

    /// A weighted Remez exchange, independent of the programs, so the hard-coded
    /// fits are checked against a derivation.
    mod remez {
        fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
            let n = b.len();
            for c in 0..n {
                let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
                a.swap(c, p);
                b.swap(c, p);
                for r in c + 1..n {
                    let f = a[r][c] / a[c][c];
                    for k in c..n {
                        a[r][k] -= f * a[c][k];
                    }
                    b[r] -= f * b[c];
                }
            }
            let mut x = vec![0.0; n];
            for c in (0..n).rev() {
                let mut s = b[c];
                for k in c + 1..n {
                    s -= a[c][k] * x[k];
                }
                x[c] = s / a[c][c];
            }
            x
        }

        fn horner(c: &[f64], x: f64) -> f64 {
            c.iter().rev().fold(0.0, |a, &k| a * x + k)
        }

        /// Minimise `max |(P(x) - f(x)) w(x)|` over `[lo, hi]`, `P` of degree `n`.
        pub fn fit(f: &dyn Fn(f64) -> f64, w: &dyn Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> (Vec<f64>, f64) {
            let m = n + 2;
            let mut xs: Vec<f64> = (0..m)
                .map(|i| (lo + hi) / 2.0 - (hi - lo) / 2.0 * (std::f64::consts::PI * i as f64 / (m - 1) as f64).cos())
                .collect();
            let (mut coef, mut level) = (vec![0.0; n + 1], 0.0);
            for _ in 0..60 {
                let mut a = vec![vec![0.0; m]; m];
                let mut b = vec![0.0; m];
                for i in 0..m {
                    let mut p = 1.0;
                    for j in 0..=n {
                        a[i][j] = p * w(xs[i]);
                        p *= xs[i];
                    }
                    a[i][n + 1] = if i % 2 == 0 { -1.0 } else { 1.0 };
                    b[i] = f(xs[i]) * w(xs[i]);
                }
                let sol = solve(a, b);
                coef = sol[..=n].to_vec();
                level = sol[n + 1].abs();
                let err = |x: f64| (horner(&coef, x) - f(x)) * w(x);
                let grid = 50_000;
                let g: Vec<f64> = (0..=grid).map(|i| lo + (hi - lo) * i as f64 / grid as f64).collect();
                let ev: Vec<f64> = g.iter().map(|&x| err(x)).collect();
                let mut new = Vec::new();
                let mut i = 0;
                while i <= grid {
                    let s = ev[i] >= 0.0;
                    let (mut j, mut best) = (i, i);
                    while j <= grid && (ev[j] >= 0.0) == s {
                        if ev[j].abs() > ev[best].abs() {
                            best = j;
                        }
                        j += 1;
                    }
                    let (mut a_, mut b_) = (g[best.saturating_sub(1)], g[(best + 1).min(grid)]);
                    for _ in 0..80 {
                        let c = b_ - 0.618_033_988_749_895 * (b_ - a_);
                        let d = a_ + 0.618_033_988_749_895 * (b_ - a_);
                        if err(c).abs() > err(d).abs() {
                            b_ = d;
                        } else {
                            a_ = c;
                        }
                    }
                    new.push((a_ + b_) / 2.0);
                    i = j;
                }
                while new.len() > m {
                    if err(new[0]).abs() < err(*new.last().unwrap()).abs() {
                        new.remove(0);
                    } else {
                        new.pop();
                    }
                }
                assert_eq!(new.len(), m, "the error did not alternate {m} times");
                let top = new.iter().map(|&x| err(x).abs()).fold(0.0, f64::max);
                xs = new;
                if (top - level).abs() / level < 1e-9 {
                    level = top;
                    break;
                }
            }
            (coef, level)
        }
    }

    #[test]
    fn the_fits_are_the_remez_fits() {
        let h = std::f64::consts::LN_2 / 2.0;
        let g = |r: f64| {
            if r.abs() < 1e-8 { 1.0 + r / 2.0 } else { r.exp_m1() / r }
        };
        let q = |f: f64| {
            if f.abs() < 1e-8 { 1.0 - f / 2.0 } else { f.ln_1p() / f }
        };
        let (lo, hi) = (std::f64::consts::FRAC_1_SQRT_2 - 1.0, std::f64::consts::SQRT_2 - 1.0);
        let cases: [(&str, (Vec<f64>, f64), &[f64], f64); 3] = [
            ("exp", remez::fit(&|r: f64| r.exp(), &|r: f64| (-r).exp(), -h, h, 3), &EXP_COEFFS, EXP_FIT_ERROR),
            ("expm1/r", remez::fit(&g, &|r: f64| 1.0 / g(r), -h, h, 3), &EXPM1_COEFFS, EXPM1_FIT_ERROR),
            ("ln(1+f)/f", remez::fit(&q, &|f: f64| 1.0 / q(f), lo, hi, 4), &LOG_COEFFS, LOG_FIT_ERROR),
        ];
        for (name, (coef, level), hard, stated) in cases {
            for (a, b) in coef.iter().zip(hard) {
                assert!((a - b).abs() < 1e-9, "{name}: {a} vs {b}");
            }
            // The stated error is the level, rounded up in the fifth digit.
            assert!(level <= stated && stated <= level * 1.0005, "{name}: {level:e} vs {stated:e}");
        }
    }

    /// `sigmoid(c1 x + c3 x^3)` against `Phi`, over `[0, 8]` on a 1e-6 grid: the
    /// derivative of the difference is below 0.8 (`sigma' z' <= 0.4`, `phi <=
    /// 0.399`), so the grid's maximum plus `0.4e-6` bounds the supremum.
    #[test]
    fn gelu_fit_error_is_the_supremum() {
        let (c1, c3) = (GELU_C1 as f32 as f64, GELU_C3 as f32 as f64);
        let mut worst = 0.0f64;
        for i in 0..=8_000_000u32 {
            let x = i as f64 * 1e-6;
            let fit = 1.0 / (1.0 + (-(c1 * x + c3 * x * x * x)).exp());
            let phi = 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
            worst = worst.max((fit - phi).abs());
        }
        println!("gelu fit supremum {worst:e}");
        assert!(worst + 0.4e-6 <= GELU_FIT_ERROR, "{worst:e}");
        assert!(worst > 1.3e-4, "{worst:e}: a better fit exists?");
    }

    fn rel(got: f32, want: f64) -> f64 {
        (got as f64 - want).abs() / want.abs()
    }

    /// Hold `kind` to `want` within `bound` (relative) on every input where `want`
    /// is a normal number, and print the worst case.
    fn sweep(name: &str, kind: u32, xs: &[f32], want: &dyn Fn(f64) -> f64, bound: f64) -> f64 {
        let got = run(kind, xs);
        let mut worst = (0.0f64, 0.0f32);
        for (&x, &g) in xs.iter().zip(&got) {
            let w = want(x as f64);
            if !(w.is_finite() && w.abs() >= f32::MIN_POSITIVE as f64) {
                continue;
            }
            let e = rel(g, w);
            assert!(e <= bound, "{name}({x:e} = {:#010x}) = {g:e} vs {w:e}: {e:e} > {bound:e}", x.to_bits());
            if e > worst.0 {
                worst = (e, x);
            }
        }
        println!("{name}: worst {:.3e} ({:.1}% of {bound:e}) at {:e}", worst.0, 100.0 * worst.0 / bound, worst.1);
        worst.0
    }

    fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f32> {
        (0..=n).map(|i| (lo + (hi - lo) * i as f64 / n as f64) as f32).collect()
    }

    /// `m` positions in every binade `lo..=hi`, each times every sign.
    fn binades(lo: i32, hi: i32, m: usize, signs: &[f32]) -> Vec<f32> {
        let mut v = Vec::new();
        for e in lo..=hi {
            for i in 0..m {
                let mant = 1.0 + (i as f64 + 0.37) / m as f64;
                for s in signs {
                    v.push(s * (mant * 2f64.powi(e)) as f32);
                }
            }
        }
        v
    }

    #[test]
    fn exp_is_within_its_bound_on_its_range() {
        let mut xs = linspace(-87.0, 88.0 - 1e-5, 80_000);
        xs.extend(linspace(-1.0, 1.0, 20_000));
        let worst = sweep("exp", kind::EXP, &xs, &|x| x.exp(), EXP_APPROX_BOUND);
        // The bound is not slack by an order of magnitude.
        assert!(worst > EXP_APPROX_BOUND / 4.0, "{worst:e}");
    }

    #[test]
    fn log_is_within_its_bound_for_every_positive_normal() {
        let mut xs = binades(-126, 127, 64, &[1.0]);
        xs.extend(linspace(0.5, 1.5, 20_000));
        xs.extend(linspace(1.4, 1.43, 3_000));
        xs.extend([f32::MIN_POSITIVE, f32::MAX, 1.0, 2.0, 0.5]);
        let worst = sweep("log", kind::LOG, &xs, &|x| x.ln(), LOG_APPROX_BOUND);
        assert!(worst > LOG_APPROX_BOUND / 4.0, "{worst:e}");
    }

    #[test]
    fn recip_is_within_its_bound_and_its_seed_beyond() {
        let xs = binades(-126, 109, 48, &[1.0, -1.0]);
        let worst = sweep("recip", kind::RECIP, &xs, &|x| 1.0 / x, RECIP_APPROX_BOUND);
        assert!(worst > RECIP_APPROX_BOUND / 4.0, "{worst:e}");
        // Where the Newton step flushes: the seed's.
        let far = binades(110, 125, 48, &[1.0, -1.0]);
        sweep("recip far", kind::RECIP, &far, &|x| 1.0 / x, RECIP_SEED_BOUND);
        // From `2^126` the seed is zero and so is the result.
        for x in [f32::from_bits(0x7e80_0000), 3.0e38, f32::MAX] {
            let got = run(kind::RECIP, &[x, -x]);
            assert_eq!([got[0].to_bits(), got[1].to_bits()], [0, 0x8000_0000]);
        }
    }

    #[test]
    fn sigmoid_is_within_its_bound() {
        let mut xs = linspace(-87.0, 87.0, 80_000);
        xs.extend(linspace(-1.0, 1.0, 20_000));
        let w = |x: f64| 1.0 / (1.0 + (-x).exp());
        let worst = sweep("sigmoid", kind::SIGMOID, &xs, &w, SIGMOID_APPROX_BOUND);
        assert!(worst > SIGMOID_APPROX_BOUND / 4.0, "{worst:e}");
    }

    #[test]
    fn tanh_is_within_its_bound_including_near_zero() {
        let mut xs = linspace(-9.5, 9.5, 80_000);
        xs.extend(linspace(-0.01, 0.01, 20_000));
        xs.extend(binades(-40, -1, 16, &[1.0, -1.0]));
        let worst = sweep("tanh", kind::TANH, &xs, &|x| x.tanh(), TANH_APPROX_BOUND);
        assert!(worst > TANH_APPROX_BOUND / 4.0, "{worst:e}");
    }

    #[test]
    fn gelu_is_within_its_absolute_bound() {
        let mut xs = linspace(-12.0, 12.0, 80_000);
        xs.extend(linspace(-1.0, 1.0, 20_000));
        xs.extend([1e-20, -1e-20, 1e10, -1e10, 3e38, -3e38]);
        let got = run(kind::GELU, &xs);
        let mut worst = 0.0f64;
        for (&x, &g) in xs.iter().zip(&got) {
            let x = x as f64;
            let want = 0.5 * x * libm::erfc(-x / std::f64::consts::SQRT_2);
            let bound = gelu_approx_bound(x, want);
            let e = (g as f64 - want).abs();
            assert!(e <= bound, "gelu({x:e}) = {g:e} vs {want:e}: {e:e} > {bound:e}");
            worst = worst.max(e / bound);
        }
        println!("gelu: worst {:.1}% of its bound", 100.0 * worst);
        assert!(worst > 0.25, "{worst}");
    }

    /// Every special value an ML input meets, per op.
    #[test]
    fn special_values() {
        let nans = [0x7fc0_0000u32, 0xffc0_0000, 0x7f80_0001, 0xff80_0001, 0xffff_ffff];
        let inf = f32::INFINITY;
        let denorm = f32::from_bits(0x0000_1234);
        let is = |kind: u32, x: f32| run(kind, &[x])[0];
        let bits = |kind: u32, x: f32| is(kind, x).to_bits();
        for k in kind::FIRST..=kind::LAST {
            for n in nans {
                assert!(is(k, f32::from_bits(n)).is_nan(), "kind {k:#x}({n:#x})");
            }
        }
        // exp.
        assert_eq!(is(kind::EXP, inf), inf);
        assert_eq!(bits(kind::EXP, -inf), 0);
        assert_eq!(is(kind::EXP, 88.0), inf);
        assert_eq!(is(kind::EXP, 1e30), inf);
        assert_eq!(bits(kind::EXP, -1e30), 0);
        assert_eq!(bits(kind::EXP, -87.5), 0);
        for z in [0.0f32, -0.0, denorm, -denorm] {
            assert!(rel(is(kind::EXP, z), 1.0) <= EXP_APPROX_BOUND, "exp({z:e})");
        }
        // log.
        for z in [0.0f32, -0.0, denorm, -denorm] {
            assert_eq!(is(kind::LOG, z), -inf, "log({z:e})");
        }
        assert_eq!(is(kind::LOG, inf), inf);
        assert!(is(kind::LOG, -inf).is_nan() && is(kind::LOG, -1.0).is_nan());
        assert_eq!(bits(kind::LOG, 1.0), 0);
        // recip.
        assert_eq!(is(kind::RECIP, 0.0), inf);
        assert_eq!(is(kind::RECIP, -0.0), -inf);
        assert_eq!(is(kind::RECIP, denorm), inf);
        assert_eq!(bits(kind::RECIP, inf), 0);
        assert_eq!(bits(kind::RECIP, -inf), 0x8000_0000);
        // sigmoid.
        // Saturates to within the bound of 1 (the Newton step on `1/1`), not to 1.
        for big in [inf, 1e30] {
            assert!(rel(is(kind::SIGMOID, big), 1.0) <= SIGMOID_APPROX_BOUND);
        }
        assert_eq!(bits(kind::SIGMOID, -inf), 0);
        assert_eq!(bits(kind::SIGMOID, -1e30), 0);
        for z in [0.0f32, -0.0] {
            assert!(rel(is(kind::SIGMOID, z), 0.5) <= SIGMOID_APPROX_BOUND);
        }
        // tanh.
        assert_eq!(is(kind::TANH, inf), 1.0);
        assert_eq!(is(kind::TANH, -inf), -1.0);
        assert_eq!(is(kind::TANH, 100.0), 1.0);
        assert_eq!(bits(kind::TANH, 0.0), 0);
        assert_eq!(bits(kind::TANH, -0.0), 0x8000_0000);
        // gelu.
        assert_eq!(is(kind::GELU, inf), inf);
        assert!(is(kind::GELU, -inf).is_nan(), "Flex's `-inf * 0`");
        assert_eq!(bits(kind::GELU, 0.0), 0);
        assert_eq!(bits(kind::GELU, -0.0), 0x8000_0000);
        assert!(rel(is(kind::GELU, 1e30), 1e30) <= SIGMOID_APPROX_BOUND);
    }

    /// The mode switches the program: some input is outside Precise's bound under
    /// Approx (a negative control: it fails if `lower` ever returned the Precise
    /// kind), and Precise is the default and its own kind.
    #[test]
    fn approx_breaks_the_precise_bound_somewhere() {
        assert_eq!(MathMode::default(), MathMode::Precise);
        let xs = linspace(-3.0, 3.0, 4000);
        let cases: [(u32, &dyn Fn(f64) -> f64, f64); 5] = [
            (kind_sfpu::EXP, &|x| x.exp(), ops::EXP_BOUND),
            (kind_sfpu::SIGMOID, &|x| 1.0 / (1.0 + (-x).exp()), ops::SIGMOID_BOUND),
            (kind_sfpu::TANH, &|x| x.tanh(), ops::TANH_BOUND),
            (kind_sfpu::LOG, &|x| x.ln(), ops::LOG_BOUND),
            (kind_sfpu::RECIP, &|x| 1.0 / x, 1.0 / 8_388_608.0),
        ];
        for (precise, want, bound) in cases {
            assert_eq!(MathMode::Precise.lower(precise), precise);
            let approx = MathMode::Approx.lower(precise);
            assert_ne!(approx, precise);
            assert_eq!(MathMode::of(approx), MathMode::Approx);
            assert_eq!(MathMode::of(precise), MathMode::Precise);
            assert_eq!(precise_of(approx), Some(precise));
            let xs: Vec<f32> = if precise == kind_sfpu::LOG || precise == kind_sfpu::RECIP {
                xs.iter().map(|x| x.abs() + 0.01).collect()
            } else {
                xs.clone()
            };
            let (p, a) = (run(precise, &xs), run(approx, &xs));
            let worst = |got: &[f32]| {
                xs.iter()
                    .zip(got)
                    .map(|(&x, &g)| {
                        let w = want(x as f64);
                        if w == 0.0 { 0.0 } else { rel(g, w) }
                    })
                    .fold(0.0, f64::max)
            };
            assert!(worst(&p) <= bound, "precise {precise:#x}: {:e}", worst(&p));
            assert!(worst(&a) > bound, "approx {approx:#x} stays within Precise's bound {bound:e}: {:e}", worst(&a));
            assert_ne!(p, a, "{precise:#x}: the modes gave the same bits");
        }
        // An op with no Approx twin is itself in either mode.
        for k in [kind_sfpu::DIV, kind_sfpu::POW, kind_sfpu::SIN, kind_sfpu::ERF] {
            assert_eq!(MathMode::Approx.lower(k), k);
        }
    }

    #[test]
    fn tt_math_is_parsed_and_refused() {
        assert_eq!(MathMode::parse(None), Ok(MathMode::Precise));
        assert_eq!(MathMode::parse(Some("precise")), Ok(MathMode::Precise));
        assert_eq!(MathMode::parse(Some("approx")), Ok(MathMode::Approx));
        let e = MathMode::parse(Some("fast")).unwrap_err();
        assert!(e.contains("TT_MATH=fast"), "{e}");
    }

    /// The instruction counts per tile, Precise against Approx.
    #[test]
    fn approx_programs_are_shorter() {
        println!("{:<8} {:>9} {:>9} {:>7}", "op", "precise", "approx", "ratio");
        for (precise, approx) in TWINS {
            let count = |k: u32| {
                let (_, code) = ops::program(k, 0.0).expect("a program");
                instructions_per_tile(&code)
            };
            let (p, a) = (count(precise), count(approx));
            println!("{precise:#06x}   {p:>9} {a:>9} {:>6.2}x", p as f64 / a as f64);
            assert!(a < p, "{precise:#x}: approx {a} >= precise {p}");
        }
    }
}

//! SFPU programs: a builder whose output is correct by construction, and an
//! interpreter that says what any program computes (`hardware-coverage.md` F2,
//! F5).
//!
//! The builder ([`Program`]) is where the Vector Unit's hazards live, as code:
//!
//! - registers are [`LReg`]s, so a write to a constant or to `LReg[11..15]`,
//!   which the hardware drops without a word, does not compile into a
//!   program -- it panics while building it;
//! - conditional execution is a scope ([`Program::if_`], [`Program::if_else`]):
//!   `SFPPUSHC`, `SFPSETCC`, `SFPCOMPC` and `SFPPOPC` are emitted balanced, the
//!   flag stack's depth is tracked, and a ninth level is refused, as is a
//!   complex `SFPPOPC` mode on a full stack (Tier 2; `SFPPOPC.md` says
//!   Blackhole fixed it, and in the same page's summary that it did not);
//! - an `SFPNOP` goes exactly where `SFPMAD`'s automatic stalling misses a
//!   dependency (`Instruction::stalls_automatically_after_mad`) and nowhere
//!   else;
//! - a loop over a tile's row groups ([`Program::for_each_row_group`]) is
//!   written once, recorded into the replay buffer and replayed (X1), the
//!   `Dst` row stepped by an address modifier on its last `Dst` access -- or,
//!   when the body does not fit the buffer, unrolled, and the program says
//!   which ([`Program::loops`]).
//!
//! The interpreter ([`interp`]) runs a program's instruction stream -- the
//! same words the device runs, `REPLAY` expanded by its own model of the
//! buffer -- over a model of the Vector Unit and `Dst` built from the
//! specification's functional models, so a kernel's expected output is
//! computed, never hand-derived. It is checked instruction by instruction
//! against ttsim and both cards (`step26_sfpu_isa`).

pub mod approx;
pub mod integer;
pub mod interp;
pub mod kernel;
pub mod ops;
pub mod round;

use tt_isa::frontend;
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::{defs, encode};
use tt_isa::sfpu::{self as enc, mad_mod1, mod0_fmt};
pub use tt_isa::sfpu::{ConfigLReg, LReg};

use super::datapath::thread_entry;

/// Depth of the per-lane flag stack (`SFPPUSHC.md`).
pub const FLAG_STACK: u8 = 8;

/// `Dst` row groups and column halves in one FP32 tile: sixty-four rows, four
/// to a group, two halves each.
pub const TILE_ITERATIONS: u32 = 32;

/// The address modifier the row loop steps `Dst` by: entry 0 is kept at no
/// increment for every other `SFPLOAD`/`SFPSTORE`, and this one adds two
/// rows' worth of address -- the next column half, or after the odd half the
/// next group of four rows.
pub const ROW_LOOP_ADDR_MOD: u32 = 7;

/// How a `Dst` datum is moved between `Dst` and an `LReg` (`SFPLOAD.md`'s
/// `MOD0_FMT_*`). Only the formats some kernel uses are offered.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Format {
    /// FP32 in `Dst`; a store flushes denormals to signed zero.
    Fp32,
    /// The 32 bits as they are, either way.
    Int32,
}

impl Format {
    fn mod0(self) -> u32 {
        match self {
            Format::Fp32 => mod0_fmt::FP32,
            Format::Int32 => mod0_fmt::INT32,
        }
    }
}

/// A lane condition, on an `LReg`'s 32 bits read as a signed integer
/// (`SFPSETCC.md`). For FP32 that is a test of the sign bit and of all-zero
/// bits: [`Cond::Lt0`] holds for `-0.0` and for a negative NaN, [`Cond::Eq0`]
/// only for `+0.0`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    Lt0(LReg),
    Ne0(LReg),
    Gte0(LReg),
    Eq0(LReg),
    /// `a < b`, the 32 bits of each read as sign-magnitude integers: for FP32
    /// the total order `-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN`
    /// (Blackhole's `SFPGT`, `SFPGT_MOD1_SET_CC`).
    Less(LReg, LReg),
    /// `a <= b` in the same order (Blackhole's `SFPLE`, `SFPLE_MOD1_SET_CC`).
    LessEq(LReg, LReg),
}

impl Cond {
    fn setcc(self) -> Instruction {
        // `SFPSETCC_MOD1_LREG_*`.
        let (vc, mod1) = match self {
            Cond::Lt0(r) => (r, 0),
            Cond::Ne0(r) => (r, 2),
            Cond::Gte0(r) => (r, 4),
            Cond::Eq0(r) => (r, 6),
            Cond::Less(a, b) => return encode::sfpgt(a.index(), b.index(), 1).unwrap(),
            // `SFPLE` holds where `VD <= VC`: `a` is `VD`, and a `VD` of 12 or
            // more is a template load (the interpreter refuses it).
            Cond::LessEq(a, b) => {
                assert!(
                    a.index() < 12,
                    "LessEq({a:?}, ..): SFPLE reads its first operand as VD, and VD >= 12 \
                     is an SFPLOADMACRO template load; put a config LReg second (or compare \
                     with Less)"
                );
                return encode::sfple(b.index(), a.index(), 1).unwrap();
            }
        };
        encode::sfpsetcc(0, vc.index(), 0, mod1).unwrap()
    }
}

/// How a program's row loops came out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopForm {
    /// Recorded once and replayed: one `REPLAY` per further iteration.
    Replayed { body: usize },
    /// Written out, every iteration: the program asked for it.
    Unrolled { body: usize },
    /// Stored once and pushed once per iteration by the role runner
    /// (`super::code`): the body did not fit the replay buffer.
    Repeated { body: usize },
}

/// Which form a row loop should take.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopPolicy {
    /// Replayed when the body fits, unrolled otherwise.
    Replay,
    /// Always unrolled: immediate addresses, no address modifiers, no replay.
    Unrolled,
}

/// An SFPU program for the math thread, under construction. See the module
/// documentation for what it guarantees.
#[derive(Clone, Debug)]
pub struct Program {
    ins: Vec<Instruction>,
    /// Flag-stack entries pushed and not popped.
    depth: u8,
    /// The last instruction was one whose result `SFPMAD`'s stalling covers
    /// (the MAD sub-unit's), so the next may need an `SFPNOP`.
    after_mad: bool,
    /// Inside a row loop's body: no nested loop, no `REPLAY`.
    in_loop: bool,
    loops: Vec<LoopForm>,
    policy: LoopPolicy,
    /// The runner's block repeats over `ins` (`super::code::Loop`).
    repeats: Vec<super::code::Loop>,
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    /// A program whose lanes are all enabled, driven by `LaneFlags` from the
    /// start, as `VectorUnit.md` recommends: `SFPENCC` setting both
    /// `UseLaneFlagsForLaneEnable` and `LaneFlags`.
    pub fn new() -> Self {
        Program::with_policy(LoopPolicy::Replay)
    }

    pub fn with_policy(policy: LoopPolicy) -> Self {
        let mut p = Program {
            ins: Vec::new(),
            depth: 0,
            after_mad: false,
            in_loop: false,
            loops: Vec::new(),
            policy,
            repeats: Vec::new(),
        };
        // `SFPENCC_MOD1_EI | SFPENCC_MOD1_RI`, both immediate bits set.
        p.push(encode::sfpencc(3, 0, 2 | 8).unwrap());
        // The thread's address modifier 0 steps nothing and its `Dst` row
        // counter is zero, so an `SFPLOAD`/`SFPSTORE` address is absolute --
        // whatever the kernel before left (a matmul steps the counter).
        p.ins.push(thread_entry(super::datapath::addr_mod_entry(0).dst_incr, 0));
        p.ins.push(clear_dst_rwc());
        p
    }

    /// Append `i`, first inserting the `SFPNOP` automatic stalling would miss.
    fn push(&mut self, i: Instruction) {
        if self.after_mad && !i.stalls_automatically_after_mad() {
            self.ins.push(enc::nop());
        }
        self.after_mad = matches!(
            i.def().mnemonic(),
            "SFPMAD" | "SFPMUL" | "SFPADD" | "SFPMULI" | "SFPADDI" | "SFPMUL24" | "SFPLUT" | "SFPLUTFP32"
        );
        self.ins.push(i);
    }

    /// An instruction the builder has no method for -- a measurement of a
    /// mode no op uses yet -- through the same `SFPNOP` bookkeeping. Nothing
    /// is checked: the caller owns what it does.
    pub fn raw(&mut self, i: Instruction) {
        self.push(i);
    }

    fn dst(d: LReg) -> u32 {
        assert!(d.writable(), "LReg[{}] cannot be written: the hardware would drop it", d.index());
        d.index()
    }

    /// `d = Dst[at]` for this lane's datum: `at` is a `Dst` address, its high
    /// eight bits a four-row group and bit 1 the odd columns
    /// (`SFPLOAD.md`).
    pub fn load(&mut self, d: LReg, fmt: Format, at: u32) {
        self.push(enc::load(Self::dst(d), fmt.mod0(), 0, at).unwrap());
    }

    /// `Dst[at] = s`, as [`Program::load`] addresses it.
    pub fn store(&mut self, s: LReg, fmt: Format, at: u32) {
        self.push(enc::store(s.index(), fmt.mod0(), 0, at).unwrap());
    }

    /// `d = v` in every enabled lane: one `SFPLOADI` when `v`'s low 16 bits
    /// are zero (a BF16 immediate), two otherwise.
    pub fn loadi(&mut self, d: LReg, v: f32) {
        self.loadi_bits(d, v.to_bits());
    }

    /// `d = bits` in every enabled lane.
    pub fn loadi_bits(&mut self, d: LReg, bits: u32) {
        let vd = Self::dst(d);
        if bits & 0xffff == 0 {
            self.push(enc::loadi(vd, enc::loadi_mode::FLOATB, bits >> 16).unwrap());
        } else {
            for i in enc::load_f32(vd, bits).unwrap() {
                self.push(i);
            }
        }
    }

    /// `d = a * b + c`, `fma_bh`'s rounding (`SFPMAD.md`).
    pub fn mad(&mut self, a: LReg, b: LReg, c: LReg, d: LReg) {
        self.push(enc::mad(a.index(), b.index(), c.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = a * b`, keeping the sign of a zero product (`-0` addend, numerics
    /// row C).
    pub fn mul(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(enc::mul(a.index(), b.index(), Self::dst(d)).unwrap());
    }

    /// `d = a + b`.
    pub fn add(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(enc::add(a.index(), b.index(), Self::dst(d)).unwrap());
    }

    /// `d = a - b`.
    pub fn sub(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(enc::sub(a.index(), b.index(), Self::dst(d)).unwrap());
    }

    /// `d = -(a * b) + c`.
    pub fn nmad(&mut self, a: LReg, b: LReg, c: LReg, d: LReg) {
        self.push(enc::mad(a.index(), b.index(), c.index(), Self::dst(d), mad_mod1::NEGATE_VA).unwrap());
    }

    /// `d = s`.
    pub fn mov(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpmov(s.index(), Self::dst(d), 0).unwrap());
    }

    /// Read and advance the hardware's per-lane PRNG. Its state is inherited
    /// from prior programs; this does not provide seedable application RNG.
    pub fn read_prng(&mut self, d: LReg) {
        self.push(encode::sfpmov(9, Self::dst(d), 8).unwrap());
    }

    /// Explicit hardware precision reduction, including the documented NaN,
    /// signed-zero and rounding defects. Ordinary BF16 casts use ties-even.
    pub fn hardware_round(
        &mut self,
        s: LReg,
        d: LReg,
        precision: tt_isa::numerics::stochastic::Precision,
        rounding: tt_isa::numerics::stochastic::Rounding,
    ) {
        self.push(
            encode::Sfpstochrnd::ZERO
                .rounding_mode(rounding as u32)
                .vc(s.index())
                .vd(Self::dst(d))
                .mod1(precision as u32)
                .encode()
                .unwrap(),
        );
    }

    /// `d = s` with its sign bit flipped: FP32 negation, NaNs included.
    pub fn neg(&mut self, s: LReg, d: LReg) {
        // `SFPMOV_MOD1_NEGATE`.
        self.push(encode::sfpmov(s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = |s|` as FP32: the sign cleared, except that a negative NaN stays
    /// one (`SFPABS.md`, `SFPABS_MOD1_FLOAT`).
    pub fn abs(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpabs(s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = s` with sign bit `negative` (`SFPSETSGN_MOD1_ARG_IMM`).
    pub fn set_sign(&mut self, s: LReg, negative: bool, d: LReg) {
        self.push(encode::sfpsetsgn(u32::from(negative), s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = s` with the sign bit of `sign` (`SFPSETSGN` taking it from `VD`,
    /// which is why `sign` is moved into `d` first).
    pub fn copy_sign(&mut self, s: LReg, sign: LReg, d: LReg) {
        // `d` takes `sign` first: an `s` in `d` would be gone before it is read.
        assert!(s != d || s == sign, "copy_sign into its own source");
        if sign != d {
            self.mov(sign, d);
        }
        self.push(encode::sfpsetsgn(0, s.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = c + d` as 32-bit two's-complement integers (`SFPIADD`), the lane
    /// flags left alone.
    pub fn iadd(&mut self, c: LReg, d: LReg) {
        // `SFPIADD_MOD1_ARG_LREG_DST | SFPIADD_MOD1_CC_NONE`.
        self.push(encode::sfpiadd(0, c.index(), Self::dst(d), 4).unwrap());
    }

    /// `d = c - d` as two's-complement integers (`SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`).
    pub fn isub_from(&mut self, c: LReg, d: LReg) {
        self.push(encode::sfpiadd(0, c.index(), Self::dst(d), 2 | 4).unwrap());
    }

    /// `d = c + imm`, `imm` a signed twelve-bit integer (`SFPIADD_MOD1_ARG_IMM`).
    pub fn iadd_imm(&mut self, c: LReg, imm: i32, d: LReg) {
        assert!((-2048..2048).contains(&imm), "{imm} is not a twelve-bit immediate");
        self.push(encode::sfpiadd(imm as u32 & 0xfff, c.index(), Self::dst(d), 1 | 4).unwrap());
    }

    /// `d = s << amount` (`SFPSHFT` with an immediate, the value from `VC`).
    pub fn shl(&mut self, s: LReg, amount: u32, d: LReg) {
        assert!(amount < 32);
        // `SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARG_IMM_USE_VC`.
        self.push(encode::sfpshft(amount, s.index(), Self::dst(d), 1 | 4).unwrap());
    }

    /// `d = s >> amount`, logical (`SFPSHFT` by `-amount` from a register
    /// holding it: the immediate is unsigned). `amount_reg` must hold
    /// `-amount`, `d` the value.
    pub fn shr_by(&mut self, neg_amount: LReg, d: LReg) {
        self.push(encode::sfpshft(0, neg_amount.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = d << amount`, `amount` a register holding `0..32` (the same
    /// `SFPSHFT` as [`Program::shr_by`]: the register's sign picks the
    /// direction).
    pub fn shl_by(&mut self, amount: LReg, d: LReg) {
        self.push(encode::sfpshft(0, amount.index(), Self::dst(d), 0).unwrap());
    }

    /// `d` = the exponent field of `s` as a two's-complement integer, minus
    /// 127 if `debias` (`SFPEXEXP`).
    pub fn exponent(&mut self, s: LReg, debias: bool, d: LReg) {
        self.push(encode::sfpexexp(s.index(), Self::dst(d), u32::from(!debias)).unwrap());
    }

    /// `d` = `s`'s mantissa bits with the hidden bit, `1 << 23` (`SFPEXMAN`).
    pub fn mantissa(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpexman(s.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = s` with its exponent field set to `exp` (`SFPSETEXP`, immediate).
    pub fn set_exponent(&mut self, s: LReg, exp: u32, d: LReg) {
        assert!(exp < 256);
        self.push(encode::sfpsetexp(exp, s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = s * 2^k` by adding `k` to the exponent field, an infinity or NaN
    /// unchanged, no check of the result's range (`SFPDIVP2`). `k` in 0..128:
    /// the page has the add wrap mod 256, so `-k` would be `256 - k`, but
    /// ttsim refuses an immediate from 128 up (divergence row 66); scale down
    /// with a multiply.
    pub fn scale_by_pow2(&mut self, s: LReg, k: u32, d: LReg) {
        assert!(k < 128, "SFPDIVP2 by {k}: ttsim refuses an immediate from 128 (row 66)");
        self.push(encode::sfpdivp2(k, s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = a & b`, bitwise (`SFPAND` with `SFPAND_MOD1_USE_VB`).
    pub fn and(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(encode::sfpand(b.index(), a.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = a | b`, bitwise (`SFPOR` with `SFPOR_MOD1_USE_VB`).
    pub fn or(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(encode::sfpor(b.index(), a.index(), Self::dst(d), 1).unwrap());
    }

    /// `d[lane] = s[lane - 1]` within each group of eight lanes -- one row of
    /// the 4x8 lane grid -- the first lane of each taking the last's
    /// (`SFPSHFT2_MOD1_SUBVEC_SHFLROR1`). Eight of them are the identity, so
    /// combining after each of seven reduces a row of eight into every lane.
    pub fn rotate_row(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpshft2(0, s.index(), Self::dst(d), 3).unwrap());
    }

    /// Transpose each lane column's 4x4 block of `L0..L4` (and of `L4..L8`):
    /// lane row `j` of `L[i]` swaps with lane row `i` of `L[j]` (`SFPTRANSP`).
    /// So row `i` of a four-row group held in `L0` ends in row 0 of `L[i]`.
    pub fn transpose4(&mut self) {
        self.push(encode::sfptransp(0).unwrap());
    }

    /// `lo, hi = min(lo, hi), max(lo, hi)` lanewise, in the sign-magnitude
    /// order of [`Cond::Less`] -- so `-0 < +0` and NaNs sort to the ends by
    /// sign: not IEEE's `min`/`max` (`SFPSWAP_MOD1_VEC_MIN_MAX`). Equal values
    /// are swapped where negative, which leaves the same bits.
    pub fn min_max(&mut self, lo: LReg, hi: LReg) {
        assert!(lo != hi);
        self.push(encode::sfpswap(hi.index(), Self::dst(lo), 1).unwrap());
    }

    /// `d = d * imm`, `imm` a BF16 constant, by `SFPMAD`'s arithmetic with a
    /// `+0` addend (`SFPMULI`) -- so a `-0` product comes out `+0`; use
    /// [`Program::mul`] where that sign matters.
    pub fn muli(&mut self, imm: f32, d: LReg) {
        self.push(encode::sfpmuli(bf16(imm), Self::dst(d), 0).unwrap());
    }

    /// `d = d + imm`, `imm` a BF16 constant (`SFPADDI`).
    pub fn addi(&mut self, imm: f32, d: LReg) {
        self.push(encode::sfpaddi(bf16(imm), Self::dst(d), 0).unwrap());
    }

    /// `d = d ^ c`, bitwise (`SFPXOR`, which takes `VD` as its other operand).
    pub fn xor(&mut self, c: LReg, d: LReg) {
        self.push(encode::sfpxor(c.index(), Self::dst(d)).unwrap());
    }

    /// `d = !s`, bitwise (`SFPNOT`).
    pub fn not(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpnot(s.index(), Self::dst(d)).unwrap());
    }

    /// `d` = the leading zeros of `s`'s 32 bits, 32 for zero; of its low 31
    /// if `ignore_sign` (`SFPLZ`, flags untouched).
    pub fn leading_zeros(&mut self, s: LReg, ignore_sign: bool, d: LReg) {
        let mod1 = if ignore_sign { 4 } else { 0 };
        self.push(encode::sfplz(s.index(), Self::dst(d), mod1).unwrap());
    }

    /// `d` = the product of `a` and `b`'s low 23 bits: its low 23 bits, or if
    /// `upper` the 23 above them (`SFPMUL24`, Blackhole only). `VC` is always
    /// the zero constant -- any other value adds the page's
    /// `NonContractualBehavior` shift-add, so it is not offered.
    pub fn mul24(&mut self, a: LReg, b: LReg, upper: bool, d: LReg) {
        let i = encode::Sfpmul24::ZERO
            .va(a.index())
            .vb(b.index())
            .vc(LReg::ZERO.index())
            .vd(Self::dst(d))
            .mod1(u32::from(upper))
            .encode()
            .unwrap();
        self.push(i);
    }

    /// `d` = the sign-magnitude integer in `s` as FP32, rounded to nearest
    /// even (`SFPCAST_MOD1_SM32_TO_FP32_RNE`). A two's-complement integer
    /// needs its magnitude and sign separated first.
    pub fn sm32_to_float(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpcast(s.index(), Self::dst(d), 0).unwrap());
    }

    /// `c = bits` in every lane, for the rest of the program: the F2 prologue
    /// for constants past the eight general registers (`SFPCONFIG` writing
    /// `LReg[11..15]` from `LReg[0]`). **Uses `L0`**, so it belongs before the
    /// program loads anything; and outside any scope, because `SFPCONFIG`
    /// takes both its value and its predication from lanes 0..8 alone. Every
    /// program writes the constants it reads: the interpreter starts these
    /// registers unknown, so a program relying on one an earlier program left
    /// (concepts review G11) is refused there.
    pub fn constant(&mut self, c: ConfigLReg, bits: u32) {
        assert_eq!(self.depth, 0, "SFPCONFIG inside a scope sees lanes 0..8's flags only");
        self.loadi_bits(LReg::L0, bits);
        self.push(encode::sfpconfig(0, c.lreg().index(), 0).unwrap());
    }

    /// `d = Lut8(c >> 8) * |L3| + Lut8(c)`, `c` being `L0` where `|L3| < 1`,
    /// `L1` where `< 2` and `L2` otherwise, each holding two
    /// [`tt_isa::numerics::sfpu::lut8_to_fp32`] codes; `L3`'s sign kept if
    /// `retain_sign` (`SFPLUT`). The four registers are implied operands.
    pub fn lut(&mut self, retain_sign: bool, d: LReg) {
        let mod0 = if retain_sign { 4 } else { 0 };
        self.push(encode::sfplut(Self::dst(d), mod0).unwrap());
    }

    /// `d = a * |L3| + c` with `(a, c)` from the table for `|L3|`'s range
    /// (`SFPLUTFP32`, `table` saying where and in what format). `L3`'s sign
    /// kept if `retain_sign`.
    ///
    /// The Tier 2 hazard is designed out here. `FP16_3ENTRY_TABLE` is
    /// `Mod1 = 10`, which includes `INDIRECT_VD`, so the hardware writes
    /// `LReg[LReg[7] & 15]` and not `VD`. For that table this method loads
    /// `d`'s index into `L7` first, so the two are the same register: **`L7`
    /// is clobbered**. It also sets the encoding's `Mod1Mirror` to match.
    /// Automatic stalling reads `Mod1Mirror`, and with its `INDIRECT_VD` bit
    /// clear it would assume `L7` is not read, so it would miss the `L7` just
    /// written.
    pub fn lut_fp32(&mut self, table: LutTable, retain_sign: bool, d: LReg) {
        use tt_isa::numerics::sfpu::lutfp32_mod1 as m;
        let vd = Self::dst(d);
        let mut mod1 = match table {
            LutTable::Fp32 => m::FP32_3ENTRY_TABLE,
            LutTable::Fp16Six { to_four } => {
                if to_four {
                    m::FP16_6ENTRY_TABLE2
                } else {
                    m::FP16_6ENTRY_TABLE1
                }
            }
            LutTable::Fp16Three => {
                self.push(enc::loadi(7, enc::loadi_mode::USHORT, vd).unwrap());
                m::FP16_3ENTRY_TABLE
            }
        };
        if retain_sign {
            mod1 |= m::SGN_RETAIN;
        }
        self.push(encode::sfplutfp32(mod1 & m::INDIRECT_VD, vd, mod1).unwrap());
    }

    /// `d = ApproxRecip(|x|)` with `x`'s sign (`SFPARECIP`, Blackhole only):
    /// within 0.56% of `1/x` for `2^-126 <= |x| < 2^126`, infinite below and
    /// zero above.
    pub fn approx_recip(&mut self, x: LReg, d: LReg) {
        let mod1 = tt_isa::numerics::sfpu::arecip_mod1::RECIP;
        self.push(encode::sfparecip(0, x.index(), Self::dst(d), mod1).unwrap());
    }

    /// `d = 1/x`, within one ulp of the correctly rounded reciprocal for
    /// every normal `x` whose reciprocal is normal; `1/±0 = ±inf`, `1/±inf =
    /// ±0`, a NaN stays one; denormals in or out flush to zero, as all of the
    /// SFPU's arithmetic does. Uses `t0`, `t1` as scratch, and needs `one`
    /// holding `1.0` and `max` holding `f32::MAX` (loaded once, outside a
    /// loop).
    ///
    /// The seed ([`Program::approx_recip`]) is within `e0 < 0.0056`; each
    /// Newton step `y += y * (1 - x*y)` squares the error and adds at most
    /// two roundings (`e1 <= e0^2 + 2^-23 < 3.2e-5`), and the last step's
    /// result is `1/x * (1 - e1^2)` before its single rounding, `e1^2 < 1.1e-9`
    /// -- so within half an ulp plus 0.02 of one of `1/x`, hence at most one
    /// ulp from its correct rounding. (`step28_recip` holds the device to
    /// this program bit for bit, and the program to the bound.)
    ///
    /// Where `|x| > 2^100` the step's product `y (1 - x y)` would fall below
    /// `2^-126` (`y` under `2^-100`, the correction under `2^-15` by the second
    /// step) and flush, losing the step; and from `2^126` the seed itself is
    /// zero though `1/x` may be normal -- the result was then only the seed,
    /// 0.56% off, for `|x|` from about `2^111` (found by 10.2d's `log1p`
    /// sweep). There the reciprocal is of `x' = x 2^-64`, seed and steps in
    /// range, and the result `y' 2^-64`: both scalings exact multiplies (a
    /// zero keeps its sign; a denormal result flushes, as documented). `max` is
    /// the scaling's register meanwhile, and holds `f32::MAX` again after.
    pub fn recip(&mut self, x: LReg, d: LReg, t0: LReg, t1: LReg, max: LReg) {
        assert!(d != x && t0 != x && t1 != x && d != t0 && d != t1 && t0 != t1);
        assert!(max != x && max != d && max != t0 && max != t1);
        self.abs(x, t1);
        self.loadi_bits(t0, 0x7180_0000); // 2^100
        self.if_else(
            Cond::Less(t0, t1),
            |p| {
                p.loadi_bits(max, 0x1f80_0000); // 2^-64
                p.mul(x, max, t1);
                p.approx_recip(t1, d);
                for _ in 0..2 {
                    p.nmad(t1, d, LReg::ONE, t0);
                    p.mad(t0, d, d, d);
                }
                p.mul(d, max, d);
                p.loadi_bits(max, f32::MAX.to_bits());
            },
            |p| {
                p.approx_recip(x, d);
                for _ in 0..2 {
                    p.nmad(x, d, LReg::ONE, t0);
                    p.mad(t0, d, d, d);
                }
            },
        );
        // `1/±0`, and a denormal, which the arithmetic flushes to a zero: the
        // seed's infinity met `0 * inf` in the steps. The magnitude by mask,
        // not `SFPABS`, which leaves a negative NaN negative -- it then passed
        // for a zero and came out `-inf` (found in 10.2e).
        self.loadi_bits(t0, 0x7fff_ffff);
        self.and(x, t0, t1);
        self.loadi_bits(t0, 0x0080_0000);
        self.if_(Cond::Less(t1, t0), |p| {
            p.loadi_bits(t0, 0x7f80_0000);
            p.copy_sign(t0, x, d);
        });
        // `1/±inf`: the seed's zero met `inf * 0`.
        self.if_(Cond::Less(max, t1), |p| {
            p.loadi_bits(t0, 0x7f80_0001);
            p.if_(Cond::Less(t1, t0), |p| {
                p.copy_sign(LReg::ZERO, x, d);
            });
        });
    }

    fn push_flags(&mut self) {
        assert!(self.depth < FLAG_STACK, "conditional execution nested deeper than the {FLAG_STACK}-entry flag stack");
        self.push(encode::sfppushc(0, 0).unwrap());
        self.depth += 1;
    }

    fn pop_flags(&mut self) {
        assert!(self.depth > 0, "a flag-stack pop with nothing pushed");
        self.push(encode::sfppopc(0, 0).unwrap());
        self.depth -= 1;
    }

    /// Run `then` only in the lanes where `cond` holds (among those enabled
    /// now). The lanes' predication is saved first and restored after, so a
    /// scope leaves the program as it found it.
    pub fn if_(&mut self, cond: Cond, then: impl FnOnce(&mut Self)) {
        self.push_flags();
        self.push(cond.setcc());
        then(self);
        self.pop_flags();
    }

    /// `then` where `cond` holds, `otherwise` in the other enabled lanes.
    /// `SFPCOMPC` inverts `LaneFlags` against the saved state, so a lane that
    /// was disabled before the scope runs neither branch.
    pub fn if_else(&mut self, cond: Cond, then: impl FnOnce(&mut Self), otherwise: impl FnOnce(&mut Self)) {
        self.push_flags();
        self.push(cond.setcc());
        then(self);
        self.push(encode::sfpcompc(0).unwrap());
        otherwise(self);
        self.pop_flags();
    }

    /// `body`, once per row group and column half of the `rows` `Dst` rows
    /// (64 per FP32 tile): `body` is handed the address offset to add to
    /// every `Dst` address it uses (`0` when replayed, where the row counter
    /// does the stepping instead).
    ///
    /// Replayed (X1) when the policy allows and the body fits the 32-entry
    /// buffer: the row counter `RWCs.Dst` is cleared, address modifier
    /// [`ROW_LOOP_ADDR_MOD`] set to step it by two and entry 0 to step it by
    /// nothing, the body recorded while it runs its first iteration -- its
    /// last `Dst` access carrying the stepping modifier -- and replayed for
    /// the rest; the counter is cleared again after, so later addresses are
    /// absolute. Unrolled otherwise, with the offset in each address.
    pub fn for_each_row_group(&mut self, rows: u32, body: impl Fn(&mut Self, u32)) {
        assert!(!self.in_loop, "a row loop inside a row loop");
        assert!(rows % 4 == 0 && rows > 0, "{rows} rows are not whole groups of four");
        let iterations = rows / 2;
        // The body once, at offset 0, to measure and patch.
        let mut b = Program {
            ins: Vec::new(),
            depth: self.depth,
            after_mad: self.after_mad,
            in_loop: true,
            loops: Vec::new(),
            policy: self.policy,
            repeats: Vec::new(),
        };
        body(&mut b, 0);
        assert_eq!(b.depth, self.depth, "a row loop's body leaves the flag stack as it found it");
        let last_dst = b
            .ins
            .iter()
            .rposition(|i| core::ptr::eq(i.def(), &defs::SFPLOAD) || core::ptr::eq(i.def(), &defs::SFPSTORE))
            .expect("a row loop's body reads or writes Dst");
        // The body was built following what precedes the loop, so a NOP it
        // needs at its start is already in it. Replayed, it also follows its
        // own end: a body ending in a multiply-add whose first instruction
        // stalling would miss needs one more, at its end.
        let wraps_nop = b.after_mad && b.ins.first().is_some_and(|i| !i.stalls_automatically_after_mad());
        let fits = b.ins.len() + usize::from(wraps_nop) <= frontend::REPLAY_BUFFER as usize;
        if self.policy == LoopPolicy::Unrolled {
            for k in 0..iterations {
                body(self, 2 * k);
            }
            self.loops.push(LoopForm::Unrolled { body: b.ins.len() });
            return;
        }
        let mut ins = b.ins;
        let step = ins[last_dst];
        ins[last_dst] = Instruction::new(step.word() | (ROW_LOOP_ADDR_MOD << 13), step.def());
        if wraps_nop {
            ins.push(enc::nop());
        }
        // Not SFPU instructions: they leave the multiply-add bookkeeping as
        // it was, and the body already allowed for it.
        let row = super::datapath::addr_mod_entry(0);
        let stepping = super::datapath::addr_mod_entry(ROW_LOOP_ADDR_MOD as usize);
        self.ins.push(thread_entry(row.dst_incr, 0));
        self.ins.push(thread_entry(stepping.dst_incr, 2));
        self.ins.push(clear_dst_rwc());
        if fits {
            frontend::record(0, &ins, true, &mut self.ins).expect("fits, and a body has no REPLAY");
            let r = frontend::replay(0, ins.len()).unwrap();
            for _ in 1..iterations {
                self.ins.push(r);
            }
            self.loops.push(LoopForm::Replayed { body: ins.len() });
        } else {
            // Too long to record: stored once, the runner pushes it once per
            // iteration -- the same words the replayed form's expansion is,
            // the row counter stepped the same way.
            self.repeats.push(super::code::Loop { start: self.ins.len() as u32, len: ins.len() as u32, count: iterations });
            self.ins.extend_from_slice(&ins);
            self.loops.push(LoopForm::Repeated { body: ins.len() });
        }
        self.ins.push(clear_dst_rwc());
        self.after_mad = ins.last().is_some_and(is_mad_unit);
    }

    /// The forms the program's row loops took, in order.
    pub fn loops(&self) -> &[LoopForm] {
        &self.loops
    }

    /// The instructions the math thread receives, block repeats expanded:
    /// what the interpreter runs and the device is held to. Refuses a program
    /// that leaves anything on the flag stack.
    pub fn finish(self) -> Vec<Instruction> {
        self.finish_code().expand()
    }

    /// The program as a role's slot holds it, its block repeats beside it
    /// (`super::code::Code`).
    pub fn finish_code(self) -> super::code::Code {
        assert_eq!(self.depth, 0, "a program ends with the flag stack empty");
        super::code::Code { ins: self.ins, loops: self.repeats }
    }
}

/// Where [`Program::lut_fp32`] finds its coefficients `(a, c)` for `|L3|`'s
/// range `i` (0: below 1.0, 1: below 2.0, 2: above).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LutTable {
    /// `a = L[i]`, `c = L[4 + i]`, FP32.
    Fp32,
    /// Six FP16-ish pairs: `L[i]` and `L[4 + i]` each hold two halves, the
    /// half chosen by a further split at 0.5, 1.5, and 3.0 (`to_four`: 4.0).
    Fp16Six { to_four: bool },
    /// Three FP16-ish pairs, `a` and `c` the halves of `L[i]` (clobbers `L7`).
    Fp16Three,
}

/// The BF16 immediate of an `f32` that is one, or a panic while building.
fn bf16(v: f32) -> u32 {
    let b = v.to_bits();
    assert_eq!(b & 0xffff, 0, "{v} is not a BF16 value");
    b >> 16
}

fn is_mad_unit(i: &Instruction) -> bool {
    matches!(i.def().mnemonic(), "SFPMAD" | "SFPMUL" | "SFPADD" | "SFPMULI" | "SFPADDI" | "SFPMUL24" | "SFPLUT" | "SFPLUTFP32")
}

/// `SETRWC` clearing the issuing thread's `Dst` row counter and nothing else.
fn clear_dst_rwc() -> Instruction {
    encode::Setrwc::ZERO.dst(1).encode().unwrap()
}

#[cfg(any())]
mod tests {
    use super::*;

    fn mnemonics(p: &[Instruction]) -> Vec<&'static str> {
        p.iter().map(|i| i.def().mnemonic()).collect()
    }

    #[test]
    #[should_panic(expected = "cannot be written")]
    fn a_write_to_a_constant_is_refused() {
        Program::new().mov(LReg::L0, LReg::ONE);
    }

    #[test]
    fn scopes_balance_and_nest() {
        let mut p = Program::new();
        p.if_else(
            Cond::Lt0(LReg::L0),
            |p| p.if_(Cond::Eq0(LReg::L1), |p| p.mov(LReg::ZERO, LReg::L0)),
            |p| p.neg(LReg::L0, LReg::L0),
        );
        let m = mnemonics(&p.finish());
        let push = m.iter().filter(|&&x| x == "SFPPUSHC").count();
        let pop = m.iter().filter(|&&x| x == "SFPPOPC").count();
        assert_eq!((push, pop), (2, 2));
        assert_eq!(m.iter().filter(|&&x| x == "SFPCOMPC").count(), 1);
    }

    #[test]
    #[should_panic(expected = "deeper than")]
    fn a_ninth_level_is_refused() {
        fn nest(p: &mut Program, n: u32) {
            if n > 0 {
                p.if_(Cond::Lt0(LReg::L0), |p| nest(p, n - 1));
            }
        }
        nest(&mut Program::new(), 9);
    }

    /// A NOP exactly where automatic stalling misses: after a multiply-add
    /// and before the cases `SFPMAD.md` lists, and nowhere else.
    #[test]
    fn nops_go_where_stalling_misses_and_nowhere_else() {
        let mut p = Program::new();
        p.mul(LReg::L0, LReg::L1, LReg::L2);
        p.store(LReg::L2, Format::Fp32, 0);
        p.add(LReg::L0, LReg::L1, LReg::L2);
        p.push(encode::sfpiadd(0, 2, 2, 0).unwrap());
        let m = mnemonics(&p.finish());
        let nops: Vec<usize> = m.iter().enumerate().filter(|(_, x)| **x == "SFPNOP").map(|(i, _)| i).collect();
        assert_eq!(nops.len(), 1, "{m:?}");
        assert_eq!(m[nops[0] + 1], "SFPIADD");
        assert!(matches!(m[nops[0] - 1], "SFPMAD" | "SFPADD"));
    }

    /// The FP16 three-entry table is `Mod1 = 10`, which includes
    /// `INDIRECT_VD`: the builder points `L7` at `VD` first and sets the
    /// mirror so stalling sees `L7` read. The other tables do neither.
    #[test]
    fn the_indirect_lut_points_l7_at_its_destination_and_says_so() {
        let mut p = Program::new();
        p.lut_fp32(LutTable::Fp16Three, false, LReg::L2);
        p.lut_fp32(LutTable::Fp32, true, LReg::L4);
        let ins = p.finish();
        let i = ins.iter().position(|i| i.def().mnemonic() == "SFPLUTFP32").unwrap();
        let loadi = ins[i - 1];
        assert_eq!(loadi.def().mnemonic(), "SFPLOADI");
        assert_eq!((loadi.operand("VD"), loadi.operand("Imm16")), (Some(7), Some(2)));
        assert_eq!(ins[i].operand("Mod1"), Some(10));
        assert_eq!(ins[i].operand("Mod1Mirror"), Some(8));
        let j = ins.iter().rposition(|i| i.def().mnemonic() == "SFPLUTFP32").unwrap();
        assert_eq!(ins[j].operand("Mod1"), Some(4));
        assert_eq!(ins[j].operand("Mod1Mirror"), Some(0));
        assert_ne!(ins[j - 1].def().mnemonic(), "SFPLOADI");
    }

    #[test]
    #[should_panic(expected = "lanes 0..8")]
    fn a_constant_inside_a_scope_is_refused() {
        let mut p = Program::new();
        p.if_(Cond::Lt0(LReg::L0), |p| p.constant(ConfigLReg::L12, 0));
    }

    #[test]
    #[should_panic(expected = "not a BF16 value")]
    fn an_immediate_that_is_not_bf16_is_refused() {
        Program::new().muli(1.1, LReg::L0);
    }

    #[test]
    fn a_row_loop_replays_when_it_fits_and_unrolls_when_it_does_not() {
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Fp32, o);
            p.load(LReg::L1, Format::Fp32, 64 + o);
            p.add(LReg::L0, LReg::L1, LReg::L0);
            p.store(LReg::L0, Format::Fp32, o);
        });
        assert_eq!(p.loops(), &[LoopForm::Replayed { body: 4 }]);
        let ins = p.finish();
        let m = mnemonics(&ins);
        assert_eq!(m.iter().filter(|&&x| x == "REPLAY").count(), 32, "one record, 31 replays");
        let store = ins.iter().find(|i| i.def().mnemonic() == "SFPSTORE").unwrap();
        assert_eq!(store.operand("AddrMod"), Some(ROW_LOOP_ADDR_MOD), "the last Dst access steps");
        let load = ins.iter().find(|i| i.def().mnemonic() == "SFPLOAD").unwrap();
        assert_eq!(load.operand("AddrMod"), Some(0));

        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            for k in 0..40 {
                p.load(LReg::L0, Format::Fp32, o + 4 * (k % 2));
            }
        });
        // Too long for the replay buffer: stored once, one runner repeat of
        // 32 over it (`super::code`), which expands to every iteration.
        assert_eq!(p.loops(), &[LoopForm::Repeated { body: 40 }]);
        let code = p.clone().finish_code();
        assert_eq!(code.ins.len(), 3 + 3 + 40 + 1, "the prologue, the setup, the body once, the clear");
        assert_eq!(code.loops, [super::code::Loop { start: 6, len: 40, count: 32 }]);
        assert_eq!(p.finish().len(), 3 + 3 + 32 * 40 + 1, "expanded: every iteration");

        let mut p = Program::with_policy(LoopPolicy::Unrolled);
        p.for_each_row_group(4, |p, o| p.store(LReg::L0, Format::Fp32, o));
        let ins = p.finish();
        let addrs: Vec<_> = ins.iter().skip(3).map(|i| i.operand("Imm10").unwrap()).collect();
        assert_eq!(addrs, [0, 2], "group 0, both halves");
    }
}

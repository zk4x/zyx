//! **Dead code (2026-11-02). Delete me.** The reduce programs now live in
//! `kernel/tenstorrent.rs` (`TtEmit::Reduce`) as unrolled row ops, and
//! `TTOp::ReduceTile` is gone; nothing outside this file references anything
//! here (the `#[cfg(any())]` test module below included). It is kept only so
//! this change stayed a pure "the variant is gone" diff. **`reduce.rs` is no
//! longer the source of truth for a reduce program** — read the emitter in
//! `kernel/tenstorrent.rs` and `NOTES.md`, not this file.
//!
//! Reductions over one dimension on the SFPU (`hardware-coverage.md` R1).
//!
//! A `[rows, cols]` tensor reduced over its columns (`Axis::Cols`, Burn's
//! `dim = 1`) gives `[rows, 1]`; over its rows (`Axis::Rows`, `dim = 0`),
//! `[1, cols]`. Each output tile is made from a line of input tiles -- a tile
//! row, or a tile column -- in three steps on the math thread:
//!
//! 1. **accumulate**: each input tile, unpacked to `Dst` rows 0..64, is
//!    combined lanewise into an accumulator tile at rows 128..192 (the first
//!    copied there). The last tile along the reduced dimension of a ragged
//!    tensor has its padding lanes replaced by the operation's identity first
//!    (`-inf` for a maximum, `0` for a sum), by the lane's column (or row)
//!    index from `LReg[15]` -- so padding never needs filling.
//! 2. **finish**: the accumulator folded within the tile. Over columns: each
//!    tile row's 32 values -- two faces, two column halves, eight lanes each --
//!    combined lanewise, then across the eight lanes by seven rotations
//!    (`SFPSHFT2`), and the row's result stored into every one of its
//!    columns: an output tile whose every column is the answer, ready to be
//!    broadcast. Over rows: each column's 32 values -- eight row groups --
//!    combined lanewise, then across the four lane rows by `SFPTRANSP`, the
//!    result in row 0.
//! 3. the packer writes the accumulator out.
//!
//! A maximum is exact whatever the order: the total order of `SFPGT`, in which
//! a positive NaN is above everything (so it propagates, and every NaN the
//! SFPU's arithmetic makes is one) and a negative NaN below everything. A sum
//! over columns is computed in the tree order above, not Flex's left to right:
//! its error is the standard `(n - 1) u sum |x|` bound, stated against Flex in
//! the gate.
//!
//! A **sum over rows** is Flex's exactly, so it is not a tree: each column
//! from `+0.0`, adding its rows one at a time, top to bottom, tile after tile
//! ([`accumulate_in_order`]). A load holds four rows of eight columns; one
//! `SFPTRANSP` puts each of the four rows into row 0 of its own register, and
//! they are added into the running sum in order. The sum lives in row 0 of
//! `L4`, which is on the transpose's diagonal and so survives it. Rows past
//! the data are not added at all -- adding `+0.0` would turn a `-0.0` sum
//! positive -- and there is no finishing fold.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::TILE_SLOT;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use super::super::datapath::{
    clear_unpacker0_adcs, config_program, pack_tile_from_dst, state_id, thread_config, tile_unpack_config, unpack_tile_to_dst,
};
use super::super::l1::{PlanError, Requirements};
use super::super::runtime::SemaphoreInit;
use super::kernel::{A_ROW, OUT_ROW};
use super::{Cond, Format, LReg, LoopPolicy, Program};

/// What a reduction computes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    /// Direct multiplication, with one final fold after all input chunks.
    Prod,
    /// Conjunction of canonical Boolean words.
    All,
    /// Disjunction of canonical Boolean words.
    Any,
    /// OR the raw bits of F32 storage, with zero-bit padding. Used after
    /// masking to one selected datum per row, preserving zeros/NaN payloads.
    /// This is a bit fold, not a floating-point arithmetic reduction.
    BitOr,
    SumI32,
    ProdI32,
    MinI32,
    MaxI32,
}

impl ReduceOp {
    /// The identity a padding lane is replaced by.
    fn identity(self) -> u32 {
        match self {
            ReduceOp::Sum => 0,
            ReduceOp::Max => 0xff80_0000,
            ReduceOp::Prod => 0x3f80_0000,
            ReduceOp::All => 1,
            ReduceOp::Any | ReduceOp::BitOr => 0,
            ReduceOp::SumI32 => 0,
            ReduceOp::ProdI32 => 1,
            ReduceOp::MinI32 => i32::MAX as u32,
            ReduceOp::MaxI32 => i32::MIN as u32,
        }
    }

    /// Physical element type consumed and produced by this reduction.
    pub fn elem(self) -> super::super::elem::Elem {
        match self {
            Self::All | Self::Any => super::super::elem::Elem::Bool,
            Self::SumI32 | Self::ProdI32 | Self::MinI32 | Self::MaxI32 => super::super::elem::Elem::I32,
            _ => super::super::elem::Elem::F32,
        }
    }

    fn format(self) -> Format {
        match self {
            Self::Max | Self::All | Self::Any | Self::BitOr | Self::SumI32 | Self::ProdI32 | Self::MinI32 | Self::MaxI32 => {
                Format::Int32
            }
            _ => Format::Fp32,
        }
    }
}

/// Which dimension is reduced.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Over rows (Burn's `dim = 0`): `[rows, cols] -> [1, cols]`.
    Rows,
    /// Over columns (`dim = 1`): `[rows, cols] -> [rows, 1]`.
    Cols,
}

/// `into = op(other, into)`, lanewise.
fn combine(p: &mut Program, op: ReduceOp, other: LReg, into: LReg) {
    match op {
        ReduceOp::Sum => p.add(other, into, into),
        ReduceOp::Max => p.if_(Cond::Less(into, other), |p| p.mov(other, into)),
        ReduceOp::Prod => p.mul(other, into, into),
        ReduceOp::All => p.and(other, into, into),
        ReduceOp::Any | ReduceOp::BitOr => p.or(other, into, into),
        ReduceOp::SumI32 => p.iadd(other, into),
        ReduceOp::ProdI32 => {
            // Preserve L0..L3: finishing a row fold still needs the other
            // three transposed row registers. Reconstruct a full low-32-bit
            // product from 16-bit limbs and the low/upper MUL24 modes.
            p.loadi_bits(LReg::L6, 0xffff);
            p.and(other, LReg::L6, LReg::L4);
            p.and(into, LReg::L6, LReg::L5);
            p.mov(into, LReg::L7);
            p.loadi_bits(LReg::L6, (-16i32) as u32);
            p.shr_by(LReg::L6, LReg::L7);
            p.mul24(LReg::L4, LReg::L7, false, LReg::L7);
            p.shl(LReg::L7, 16, LReg::L7);
            p.mul24(LReg::L4, LReg::L5, false, into);
            p.mul24(LReg::L4, LReg::L5, true, LReg::L6);
            p.shl(LReg::L6, 23, LReg::L6);
            p.or(LReg::L6, into, into);
            p.iadd(LReg::L7, into);
            p.mov(other, LReg::L4);
            p.loadi_bits(LReg::L6, (-16i32) as u32);
            p.shr_by(LReg::L6, LReg::L4);
            p.mul24(LReg::L4, LReg::L5, false, LReg::L7);
            p.shl(LReg::L7, 16, LReg::L7);
            p.iadd(LReg::L7, into);
        }
        ReduceOp::MinI32 | ReduceOp::MaxI32 => {
            p.loadi_bits(LReg::L4, 0x7fffffff);
            p.mov(other, LReg::L5);
            p.mov(into, LReg::L7);
            p.if_(Cond::Lt0(LReg::L5), |p| p.xor(LReg::L4, LReg::L5));
            p.if_(Cond::Lt0(LReg::L7), |p| p.xor(LReg::L4, LReg::L7));
            let cond = if op == ReduceOp::MinI32 {
                Cond::Less(LReg::L5, LReg::L7)
            } else {
                Cond::Less(LReg::L7, LReg::L5)
            };
            p.if_(cond, |p| p.mov(other, into));
        }
    }
}

/// The accumulate program for one input tile: `first` copies it to the
/// accumulator, otherwise it is combined in. `valid`: the input is the last
/// along the reduced dimension and only its first `valid` columns (or rows,
/// for [`Axis::Rows`]) are data -- the rest are replaced by the identity.
/// A sum over rows is [`accumulate_in_order`] instead.
pub fn accumulate(op: ReduceOp, axis: Axis, first: bool, valid: Option<u32>) -> Vec<Instruction> {
    if (op, axis) == (ReduceOp::Sum, Axis::Rows) {
        return accumulate_in_order(first, valid.unwrap_or(32));
    }
    let fmt = op.format();
    let masked = valid.is_some_and(|v| v < 32);
    let mut p = Program::with_policy(if masked { LoopPolicy::Unrolled } else { LoopPolicy::Replay });
    let mask_setup = |p: &mut Program| {
        // `L6` = this lane's column within its half (`2 * (lane & 7)`), or
        // its row within its group (`lane >> 3`), from `LReg[15] = 2 * lane`.
        match axis {
            Axis::Cols => {
                p.loadi_bits(LReg::L7, 14);
                p.and(LReg::LANE_X2, LReg::L7, LReg::L6);
            }
            Axis::Rows => {
                p.loadi_bits(LReg::L7, (-4i32) as u32);
                p.mov(LReg::LANE_X2, LReg::L6);
                p.shr_by(LReg::L7, LReg::L6);
            }
        }
        p.loadi_bits(LReg::L4, op.identity());
    };
    if masked {
        mask_setup(&mut p);
    }
    let v = valid.unwrap_or(32) as i32;
    p.for_each_row_group(64, |p, o| {
        if masked && matches!(op, ReduceOp::ProdI32 | ReduceOp::MinI32 | ReduceOp::MaxI32) {
            mask_setup(p);
        }
        p.load(LReg::L0, fmt, A_ROW + o);
        if masked {
            let g = o / 4;
            // What the lane index in `L6` must stay below for the lane to
            // hold data in this row group and half.
            let t = match axis {
                Axis::Cols => v - (o & 2 != 0) as i32 - 16 * ((g / 4) % 2) as i32,
                Axis::Rows => v - 4 * (g % 4) as i32 - 16 * (g / 8) as i32,
            };
            if t <= 0 {
                p.mov(LReg::L4, LReg::L0);
            } else if t < 16 {
                p.loadi_bits(LReg::L5, t as u32);
                p.if_else(Cond::Less(LReg::L6, LReg::L5), |_| {}, |p| p.mov(LReg::L4, LReg::L0));
            }
        }
        if first {
            p.store(LReg::L0, fmt, OUT_ROW + o);
        } else {
            p.load(LReg::L1, fmt, OUT_ROW + o);
            combine(p, op, LReg::L0, LReg::L1);
            p.store(LReg::L1, fmt, OUT_ROW + o);
        }
    });
    p.finish()
}

/// The column sums of one input tile's first `valid` rows, added in order into
/// the running sums in row 0 of the accumulator tile -- from `+0.0` for the
/// `first` tile, which also zeroes the accumulator's other rows (as the
/// result's padding). Flex's `sum` over `dim = 0`, bit for bit.
pub fn accumulate_in_order(first: bool, valid: u32) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    let acc = |r: u32| OUT_ROW + r;
    if first {
        p.for_each_row_group(64, |p, o| p.store(LReg::ZERO, Format::Fp32, acc(o)));
    }
    // Columns 0..16 are faces 0 (rows 0..16) and 2 (16..32), columns 16..32
    // faces 1 and 3; each column half of a face is eight lanes, a load four
    // rows of them.
    for (top, bottom) in [(0, 32), (16, 48)] {
        for half in [0, 2] {
            if first {
                p.mov(LReg::ZERO, LReg::L4);
            } else {
                p.load(LReg::L4, Format::Fp32, acc(top + half));
            }
            for g in 0..8u32 {
                let rows = valid.saturating_sub(4 * g).min(4);
                if rows == 0 {
                    break;
                }
                let r = if g < 4 { top + 4 * g } else { bottom + 4 * (g - 4) };
                p.load(LReg::L0, Format::Fp32, A_ROW + r + half);
                // Row `i` of the four into row 0 of `L[i]`; row 0 of `L4`
                // stays where it is.
                p.transpose4();
                for i in 0..rows {
                    p.add(LReg::L4, [LReg::L0, LReg::L1, LReg::L2, LReg::L3][i as usize], LReg::L4);
                }
            }
            // Row 0 the sums, rows 1..4 zero: `L5..L8` cleared and transposed
            // into them.
            for l in [LReg::L5, LReg::L6, LReg::L7] {
                p.mov(LReg::ZERO, l);
            }
            p.transpose4();
            p.store(LReg::L4, Format::Fp32, acc(top + half));
        }
    }
    p.finish()
}

/// The finishing program: the accumulator folded within the tile (see the
/// module documentation). Nothing for a sum over rows, already in row 0.
pub fn finish(op: ReduceOp, axis: Axis) -> Vec<Instruction> {
    let fmt = op.format();
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    if (op, axis) == (ReduceOp::Sum, Axis::Rows) {
        return p.finish();
    }
    let acc = |r: u32| OUT_ROW + r;
    match axis {
        Axis::Cols => {
            // Tensor rows 0..16 are in faces 0 (left) and 1 (right), 16..32 in
            // faces 2 and 3; a group of four rows in each.
            for lower in [0, 32] {
                for g in 0..4 {
                    let (left, right) = (lower + 4 * g, lower + 16 + 4 * g);
                    let at = [left, left | 2, right, right | 2];
                    p.load(LReg::L0, fmt, acc(at[0]));
                    for &a in &at[1..] {
                        p.load(LReg::L1, fmt, acc(a));
                        combine(&mut p, op, LReg::L1, LReg::L0);
                    }
                    // Across the eight lanes of each lane row.
                    p.mov(LReg::L0, LReg::L2);
                    for _ in 0..7 {
                        p.rotate_row(LReg::L2, LReg::L2);
                        combine(&mut p, op, LReg::L2, LReg::L0);
                    }
                    for &a in &at {
                        p.store(LReg::L0, fmt, acc(a));
                    }
                }
            }
        }
        Axis::Rows => {
            for (top, bottom) in [(0, 32), (16, 48)] {
                for half in [0, 2] {
                    p.load(LReg::L0, fmt, acc(top + half));
                    for g in 0..8 {
                        let r = if g < 4 { top + 4 * g } else { bottom + 4 * (g - 4) };
                        if r != top {
                            p.load(LReg::L1, fmt, acc(r + half));
                            combine(&mut p, op, LReg::L1, LReg::L0);
                        }
                    }
                    // Lane row `i` of `L0` into lane row 0 of `L[i]`.
                    p.transpose4();
                    combine(&mut p, op, LReg::L1, LReg::L0);
                    combine(&mut p, op, LReg::L3, LReg::L2);
                    combine(&mut p, op, LReg::L2, LReg::L0);
                    p.store(LReg::L0, fmt, acc(top + half));
                }
            }
        }
    }
    p.finish()
}

/// The reduce kernel's semaphores, numbered as a matmul's and an element-wise
/// kernel's are on the ones they share (zero, one, zero), so all three
/// alternate with no setup run.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceSemaphores {
    /// T0 -> T1: an input tile is in `Dst`.
    pub unpacked: Semaphore,
    /// T1 -> T0: the math has read it; the next may be unpacked over it.
    pub consumed: Semaphore,
    /// T1 -> T2: an output tile is finished.
    pub computed: Semaphore,
    /// T2 -> T1: the packer has read the accumulator.
    pub packed: Semaphore,
}

/// Where a run of `outputs` output tiles, each from `per` input tiles, lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub outputs: usize,
    pub per: usize,
    pub in_at: u64,
    pub out_at: u64,
    pub sems: ReduceSemaphores,
    pub init: Vec<SemaphoreInit>,
}

impl Layout {
    /// This layout with its slots `by` bytes on, its semaphores where they
    /// were: the second half of a double-buffered pair.
    pub fn shifted(&self, by: u64) -> Layout {
        Layout { in_at: self.in_at + by, out_at: self.out_at + by, ..self.clone() }
    }
}

pub fn plan_layout(outputs: usize, per: usize) -> Result<Layout, PlanError> {
    plan_layout_in(outputs, per, tt_isa::l1::DATA)
}

/// [`plan_layout`] in `arena`: half the data arena, for a run double-buffered
/// with the next (`super::super::matmul::half_arena`, then [`Layout::shifted`]).
pub fn plan_layout_in(outputs: usize, per: usize, arena: tt_isa::l1::Region) -> Result<Layout, PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let input = req.scratch("reduce inputs", (outputs * per) as u64 * TILE_SLOT, align, 0..1);
    let out = req.scratch("reduce outputs", outputs as u64 * TILE_SLOT, align, 0..1);
    let unpacked = req.semaphore("reduce unpacked", 0, 0..1);
    let consumed = req.semaphore("reduce consumed", 1, 0..1);
    let computed = req.semaphore("reduce computed", 0, 0..1);
    let packed = req.semaphore("reduce packed", 1, 0..1);
    let plan = req.plan(arena)?;
    Ok(Layout {
        outputs,
        per,
        in_at: plan.addr(input),
        out_at: plan.addr(out),
        sems: ReduceSemaphores {
            unpacked: plan.semaphore(unpacked),
            consumed: plan.semaphore(consumed),
            computed: plan.semaphore(computed),
            packed: plan.semaphore(packed),
        },
        init: plan.semaphore_init(),
    })
}

/// The math programs of one output tile: per input, in order, then the
/// finishing one.
pub fn math_programs(op: ReduceOp, axis: Axis, per: usize, last_valid: u32) -> (Vec<Vec<Instruction>>, Vec<Instruction>) {
    let inputs = (0..per)
        .map(|n| {
            let valid = (n + 1 == per && last_valid < 32).then_some(last_valid);
            accumulate(op, axis, n == 0, valid)
        })
        .collect();
    (inputs, finish(op, axis))
}

/// The three role programs of a run over `layout`.
pub fn roles(layout: &Layout, inputs: &[Vec<Instruction>], finish: &[Instruction]) -> [Vec<Instruction>; 3] {
    assert_eq!(inputs.len(), layout.per);
    roles_at(layout, inputs, finish, |k, n| layout.in_at + (k * layout.per + n) as u64 * TILE_SLOT)
}

/// [`roles`], with output `k`'s input `n` unpacked from `input(k, n)`.
pub fn roles_at(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
    input: impl Fn(usize, usize) -> u64,
) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;
    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.in_at);
    unpack.extend(config_program(&words));
    let mut math = vec![state_id()];
    let mut pack = vec![state_id()];
    for k in 0..layout.outputs {
        for (n, program) in inputs.iter().enumerate() {
            unpack.extend(sync::take(s.consumed, Before::UNPACKER));
            unpack.extend(unpack_tile_to_dst(input(k, n), A_ROW));
            unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));

            math.extend(sync::take(s.unpacked, Before::SFPU));
            if n == 0 {
                math.extend(sync::take(s.packed, Before::SFPU));
            }
            math.extend_from_slice(program);
            math.extend(sync::post_after(Unit::Sfpu, s.consumed));
        }
        math.extend_from_slice(finish);
        math.extend(sync::post_after(Unit::Sfpu, s.computed));

        pack.extend(sync::take(s.computed, Before::PACKER));
        pack.extend(pack_tile_from_dst(slot(layout.out_at, k) + tt_isa::dm::TILE_DATA, OUT_ROW));
        pack.extend(sync::post_after(Unit::Packer, s.packed));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

/// A sum over rows too long for one run, in chunks of [`ROW_CHUNK`] row
/// tiles: each chunk after the first starts from the last chunk's sums, read
/// back as an input of one valid row ahead of its own tiles. Exactly Flex's
/// order still -- from `+0.0`, rows in order -- because a running sum from
/// `+0.0` is never `-0.0`, so `+0.0` plus it is it.
pub const ROW_CHUNK: usize = 16;

/// Where a chunk of a long sum over rows lives: the last chunk's sums, one
/// tile per output at `prior_at`, then [`ROW_CHUNK`] input tiles per output at
/// `in_at`, and the outputs at `out_at`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLayout {
    pub layout: Layout,
    pub prior_at: u64,
}

pub fn plan_chunk_layout(outputs: usize) -> Result<ChunkLayout, PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let prior = req.scratch("sum prior", outputs as u64 * TILE_SLOT, align, 0..1);
    let input = req.scratch("sum inputs", (outputs * ROW_CHUNK) as u64 * TILE_SLOT, align, 0..1);
    let out = req.scratch("sum outputs", outputs as u64 * TILE_SLOT, align, 0..1);
    let unpacked = req.semaphore("reduce unpacked", 0, 0..1);
    let consumed = req.semaphore("reduce consumed", 1, 0..1);
    let computed = req.semaphore("reduce computed", 0, 0..1);
    let packed = req.semaphore("reduce packed", 1, 0..1);
    let plan = req.plan(tt_isa::l1::DATA)?;
    Ok(ChunkLayout {
        layout: Layout {
            outputs,
            per: ROW_CHUNK,
            in_at: plan.addr(input),
            out_at: plan.addr(out),
            sems: ReduceSemaphores {
                unpacked: plan.semaphore(unpacked),
                consumed: plan.semaphore(consumed),
                computed: plan.semaphore(computed),
                packed: plan.semaphore(packed),
            },
            init: plan.semaphore_init(),
        },
        prior_at: plan.addr(prior),
    })
}

/// The role programs of one chunk of a long sum over rows: `tiles` input
/// tiles per output (at most [`ROW_CHUNK`]), the last with `last_valid` rows,
/// after the prior chunk's sums when `prior`.
pub fn chunk_roles(c: &ChunkLayout, prior: bool, tiles: usize, last_valid: u32) -> [Vec<Instruction>; 3] {
    assert!((1..=ROW_CHUNK).contains(&tiles));
    let mut inputs = Vec::new();
    if prior {
        inputs.push(accumulate_in_order(true, 1));
    }
    for n in 0..tiles {
        let valid = if n + 1 == tiles { last_valid } else { 32 };
        inputs.push(accumulate_in_order(!prior && n == 0, valid));
    }
    let fin = finish(ReduceOp::Sum, Axis::Rows);
    let l = &c.layout;
    roles_at(l, &inputs, &fin, |k, n| match (prior, n) {
        (true, 0) => c.prior_at + k as u64 * TILE_SLOT,
        (true, n) => l.in_at + (k * ROW_CHUNK + n - 1) as u64 * TILE_SLOT,
        (false, n) => l.in_at + (k * ROW_CHUNK + n) as u64 * TILE_SLOT,
    })
}

/// Continuation of a bounded reduction, preserving the unfolded accumulator.
#[allow(clippy::too_many_arguments)]
pub fn continuation_roles(
    c: &ChunkLayout,
    op: ReduceOp,
    axis: Axis,
    prior: bool,
    tiles: usize,
    valid: u32,
    last: bool,
) -> [Vec<Instruction>; 3] {
    let mut inputs = Vec::new();
    if prior {
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Int32, A_ROW + o);
            p.store(LReg::L0, Format::Int32, OUT_ROW + o);
        });
        inputs.push(p.finish());
    }
    for n in 0..tiles {
        inputs.push(accumulate(op, axis, !prior && n == 0, (n + 1 == tiles && valid < 32).then_some(valid)));
    }
    let fin = if last { finish(op, axis) } else { Vec::new() };
    roles_at(&c.layout, &inputs, &fin, |k, n| match (prior, n) {
        (true, 0) => c.prior_at + k as u64 * TILE_SLOT,
        (true, n) => c.layout.in_at + (k * ROW_CHUNK + n - 1) as u64 * TILE_SLOT,
        (false, n) => c.layout.in_at + (k * ROW_CHUNK + n) as u64 * TILE_SLOT,
    })
}

/// What the device computes for a reduction of `a` (`[rows, cols]`): the
/// same programs run by the interpreter over each line of input tiles. The
/// result is `[rows, 1]` for [`Axis::Cols`], `[1, cols]` for [`Axis::Rows`].
pub fn reference(op: ReduceOp, axis: Axis, a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (rt, ct) = (rows.div_ceil(32), cols.div_ceil(32));
    let (outs, per, valid) = match axis {
        Axis::Cols => (rt, ct, (cols % 32) as u32),
        Axis::Rows => (ct, rt, (rows % 32) as u32),
    };
    let valid = if valid == 0 { 32 } else { valid };
    let (inputs, fin) = math_programs(op, axis, per, valid);
    let tile = |i: usize, j: usize| -> Vec<u32> {
        let mut t = vec![0u32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                let (gr, gc) = (32 * i + r, 32 * j + c);
                if gr < rows && gc < cols {
                    t[face_index(r, c)] = a[gr * cols + gc].to_bits();
                }
            }
        }
        t
    };
    let mut out = Vec::new();
    for k in 0..outs {
        let mut v = Vector::new();
        for (n, program) in inputs.iter().enumerate() {
            let (i, j) = match axis {
                Axis::Cols => (k, n),
                Axis::Rows => (n, k),
            };
            v.put_tile(A_ROW as usize, &tile(i, j));
            v.run(program).unwrap_or_else(|e| panic!("accumulate: {e}"));
        }
        v.run(&fin).unwrap_or_else(|e| panic!("finish: {e}"));
        let t = v.tile(OUT_ROW as usize);
        match axis {
            Axis::Cols => {
                for r in 0..32 {
                    if 32 * k + r < rows {
                        out.push(f32::from_bits(t[face_index(r, 0)]));
                    }
                }
            }
            Axis::Rows => {
                for c in 0..32 {
                    if 32 * k + c < cols {
                        out.push(f32::from_bits(t[face_index(0, c)]));
                    }
                }
            }
        }
    }
    out
}

#[cfg(any())]
mod tests {
    use super::*;

    fn data(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| ((i * 37 + 11) % 101) as f32 - 50.0).collect()
    }

    /// A sum over rows is Flex's to the bit, on values whose sums round at
    /// every step and whose order matters: `+0.0` then each row in turn,
    /// ragged rows left out (a `-0.0` column stays `-0.0`).
    #[test]
    fn a_sum_over_rows_is_flex_order_bit_for_bit() {
        for (rows, cols) in [(37, 70), (32, 32), (5, 3), (64, 100), (100, 33), (1, 40), (97, 1)] {
            let mut a: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32;
                    (x - 0.5) * 10f32.powi((i % 7) as i32 - 3)
                })
                .collect();
            // A column of negative zeros.
            for r in 0..rows {
                a[r * cols] = -0.0;
            }
            let flex: Vec<f32> = (0..cols).map(|c| (0..rows).fold(0.0f32, |s, r| s + a[r * cols + c])).collect();
            let got = reference(ReduceOp::Sum, Axis::Rows, &a, rows, cols);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&flex), "[{rows}, {cols}]");
        }
    }

    /// Small integers: every sum exact whatever the order, so the model must
    /// equal the plain reduction, ragged edges and all.
    #[test]
    fn the_programs_reduce_every_shape() {
        for (rows, cols) in [(37, 70), (32, 32), (5, 3), (64, 100), (100, 33)] {
            let a = data(rows, cols);
            for op in [ReduceOp::Sum, ReduceOp::Max] {
                let f = |x: f32, y: f32| if op == ReduceOp::Sum { x + y } else { x.max(y) };
                let init = if op == ReduceOp::Sum { 0.0 } else { f32::NEG_INFINITY };
                let by_row: Vec<f32> =
                    (0..rows).map(|r| a[r * cols..(r + 1) * cols].iter().fold(init, |s, &x| f(s, x))).collect();
                let by_col: Vec<f32> = (0..cols).map(|c| (0..rows).fold(init, |s, r| f(s, a[r * cols + c]))).collect();
                assert_eq!(reference(op, Axis::Cols, &a, rows, cols), by_row, "{op:?} over columns [{rows}, {cols}]");
                assert_eq!(reference(op, Axis::Rows, &a, rows, cols), by_col, "{op:?} over rows [{rows}, {cols}]");
            }
        }
    }
}

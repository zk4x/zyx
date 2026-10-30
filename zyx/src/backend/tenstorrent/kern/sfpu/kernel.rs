//! The SFPU tile kernel (`hardware-coverage.md` F3): a run of tiles through
//! `Dst`, an SFPU program over each.
//!
//! Three roles, as LLK splits them: T0 unpacks each tile's operands into
//! `Dst` -- `A` to rows 0..64, `B` to rows 64..128 -- T1 runs the op's program,
//! which writes its result to rows 128..192, and T2 packs those rows to the
//! tile's output slot. The data mover gathers the run's operands into L1
//! before the kernel and scatters the outputs after it
//! (`tt_isa::dm::record::{READ_RUN, WRITE_RUN}`), so the kernel sees only L1.
//!
//! The roles hand each tile on through three semaphores, declared through
//! `crate::l1` like every kernel's: `unpacked` (T0 -> T1), `computed` (T1 ->
//! T2) and `free` (T2 -> T0: the packer has read `Dst`, the next tile may be
//! unpacked over it). The consumer waits -- `STALLWAIT` blocks the
//! *consumer*'s unit (divergence row 46) -- and each producer posts only once
//! its unit has finished. A run takes every one it posts, so it leaves them as
//! it found them (`unpacked` and `computed` at zero, `free` at one) and the
//! mover can run kernels back to back (`Kernel::restores_semaphores`).
//!
//! One tile at a time: the three roles overlap across a tile's stages only in
//! that each starts its next tile's setup while waiting. Double-buffering
//! `Dst` is 9.8's.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::TILE_SLOT;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use super::super::datapath::{
    config_program, pack_tile_from_dst, state_id, thread_config, tile_unpack_config,
    unpack_datums_to_dst, unpack_tile_to_dst,
};
use super::super::l1::{PlanError, Requirements};
use super::super::runtime::SemaphoreInit;

/// Where an op's operands are in `Dst`, and its result.
pub const A_ROW: u32 = 0;
pub const B_ROW: u32 = 64;
pub const OUT_ROW: u32 = 128;
/// A ternary op's third operand (`Operands::Ternary`): inside 32-bit `Dst`'s
/// 512 rows (`Dst.md`).
pub const C_ROW: u32 = 192;
/// Rows a program may spill registers to (`log1p`, `pow`): no unpacker writes
/// them, no packer reads them.
pub const SPILL_ROW: u32 = 256;

/// What a tile's second operand is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operands {
    /// `A` only.
    Unary,
    /// `A` and `B`, the same shape.
    Binary,
    /// `A`, and `B`'s first row broadcast down every row: `B`'s row 0 is laid
    /// into `Dst` four times over, its first face's sixteen columns in rows
    /// `B_ROW..B_ROW + 4` and its second face's in `B_ROW + 4..B_ROW + 8`, so
    /// one `SFPLOAD` of either group gives every lane its column's value
    /// (`Program`s read them with [`bias_row`]).
    RowBroadcast,
    /// `A`, and `B`'s first column broadcast across every column: the mover
    /// makes `B`'s tile so (`tt_isa::dm::op::READ_BROADCAST_COL`), and the
    /// kernel unpacks it as [`Operands::Binary`] does.
    ColBroadcast,
    /// `A`, `B` and `C`, the same shape: `C` to rows `C_ROW..C_ROW + 64`.
    Ternary,
}

/// The `Dst` row of the broadcast row's group for row group `g` of a tile:
/// faces 0 and 2 (row groups 0..4 and 8..12) take columns 0..16, faces 1 and 3
/// columns 16..32.
pub const fn bias_row(g: u32) -> u32 {
    if (g / 4) % 2 == 0 {
        B_ROW
    } else {
        B_ROW + 4
    }
}

/// The kernel's three semaphores, as a plan placed them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SfpuSemaphores {
    pub unpacked: Semaphore,
    pub computed: Semaphore,
    pub free: Semaphore,
}

/// Where a run of `tiles` tiles lives in L1, and its semaphores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub tiles: usize,
    /// The first operand's slots, one per tile.
    pub a_at: u64,
    /// The second operand's, unless unary.
    pub b_at: Option<u64>,
    /// The third operand's, for a ternary op.
    pub c_at: Option<u64>,
    /// The packer's output slots: datums at `TILE_DATA` past each slot's start.
    pub out_at: u64,
    pub sems: SfpuSemaphores,
    pub init: Vec<SemaphoreInit>,
}

impl Layout {
    /// This layout with every slot `by` bytes on, its semaphores where they
    /// were: the second half of a double-buffered pair.
    pub fn shifted(&self, by: u64) -> Layout {
        Layout {
            a_at: self.a_at + by,
            b_at: self.b_at.map(|b| b + by),
            c_at: self.c_at.map(|c| c + by),
            out_at: self.out_at + by,
            ..self.clone()
        }
    }
}

/// Plan a run of `tiles` tiles in the data arena: slots for the operands and
/// the outputs, and the three semaphores.
pub fn plan_layout(tiles: usize, operands: Operands) -> Result<Layout, PlanError> {
    plan_layout_in(tiles, operands, tt_isa::l1::DATA)
}

/// [`plan_layout`] in `arena`: half the data arena, for a run double-buffered
/// with the next (`super::super::matmul::half_arena`, then [`Layout::shifted`]).
pub fn plan_layout_in(
    tiles: usize,
    operands: Operands,
    arena: tt_isa::l1::Region,
) -> Result<Layout, PlanError> {
    let mut req = Requirements::new(1);
    let bytes = tiles as u64 * TILE_SLOT;
    let align = tt_isa::dram::ALIGN;
    let a = req.scratch("sfpu A slots", bytes, align, 0..1);
    let b = (operands != Operands::Unary).then(|| req.scratch("sfpu B slots", bytes, align, 0..1));
    let c =
        (operands == Operands::Ternary).then(|| req.scratch("sfpu C slots", bytes, align, 0..1));
    let out = req.scratch("sfpu output slots", bytes, align, 0..1);
    // Declared in this order so the planner numbers them as a matmul's are --
    // the first starting at zero, the second at one -- and a tile that ran
    // one needs no setup run before the other (`runtime::Resident`).
    let unpacked = req.semaphore("sfpu unpacked", 0, 0..1);
    let free = req.semaphore("sfpu Dst free", 1, 0..1);
    let computed = req.semaphore("sfpu computed", 0, 0..1);
    let plan = req.plan(arena)?;
    Ok(Layout {
        tiles,
        a_at: plan.addr(a),
        b_at: b.map(|b| plan.addr(b)),
        c_at: c.map(|c| plan.addr(c)),
        out_at: plan.addr(out),
        sems: SfpuSemaphores {
            unpacked: plan.semaphore(unpacked),
            computed: plan.semaphore(computed),
            free: plan.semaphore(free),
        },
        init: plan.semaphore_init(),
    })
}

/// Most tiles a run may have: a slot per operand and one for the output, each,
/// in the data arena.
pub fn max_tiles(operands: Operands) -> usize {
    max_tiles_in(operands, tt_isa::l1::DATA.len())
}

/// [`max_tiles`] in an arena of `bytes`.
pub fn max_tiles_in(operands: Operands, bytes: u64) -> usize {
    let per = match operands {
        Operands::Unary => 2,
        Operands::Ternary => 4,
        _ => 3,
    };
    (bytes / (per * TILE_SLOT)) as usize
}

/// The three role programs of a run over `layout`'s tiles, `math` -- one
/// tile's SFPU program, reading `A_ROW` (and `B_ROW`) and writing `OUT_ROW`
/// -- run once per tile.
pub fn roles(layout: &Layout, operands: Operands, math: &[Instruction]) -> [Vec<Instruction>; 3] {
    let (code, loops) = roles_code(layout, operands, &super::super::code::Code::plain(math.to_vec()));
    std::array::from_fn(|t| {
        super::super::code::Code {
            ins: code[t].clone(),
            loops: loops[t].clone(),
        }
        .expand()
    })
}

/// [`roles`] as the role slots hold them, with each role's block repeats
/// (`super::super::code`): the math role's per-tile block -- the same for every tile
/// -- stored once and repeated once per tile, and `math`'s own row loops
/// inside it, so a run's math program is one tile's whatever its length.
pub fn roles_code(
    layout: &Layout,
    operands: Operands,
    math: &super::super::code::Code,
) -> ([Vec<Instruction>; 3], [Vec<super::super::code::Loop>; 3]) {
    roles_code_validated(layout, operands, math, false)
}

pub(crate) fn roles_code_validated(
    layout: &Layout,
    operands: Operands,
    math: &super::super::code::Code,
    checked: bool,
) -> ([Vec<Instruction>; 3], [Vec<super::super::code::Loop>; 3]) {
    use super::super::code::Loop;
    let s = layout.sems;
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;

    // Everything the unpacker's address generators read, set here: a kernel
    // inherits the thread's ADCs from whatever ran before (a matmul leaves Z
    // at its last face), and a stale counter moves where a tile is read from
    // (`tt-metal-concepts-review.md` G11).
    let mut unpack = thread_config();
    unpack.extend(super::super::datapath::clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.a_at);
    unpack.extend(config_program(&words));
    let mut m = vec![state_id()];
    let mut math_loops = Vec::new();
    let mut pack = vec![state_id()];
    for n in 0..layout.tiles {
        unpack.extend(sync::take(s.free, Before::UNPACKER));
        unpack.extend(unpack_tile_to_dst(slot(layout.a_at, n), A_ROW));
        match (operands, layout.b_at) {
            (Operands::Binary | Operands::ColBroadcast, Some(b)) => {
                unpack.extend(unpack_tile_to_dst(slot(b, n), B_ROW));
            }
            (Operands::Ternary, Some(b)) => {
                unpack.extend(unpack_tile_to_dst(slot(b, n), B_ROW));
                let c = layout.c_at.expect("a ternary layout has C slots");
                unpack.extend(unpack_tile_to_dst(slot(c, n), C_ROW));
            }
            (Operands::RowBroadcast, Some(b)) => {
                // Row 0 of faces 0 and 1: datums 0..16 and 256..272.
                for (face, first) in [(0, 0), (1, 256)] {
                    for r in 0..4 {
                        unpack.extend(unpack_datums_to_dst(
                            slot(b, n),
                            first,
                            16,
                            B_ROW + 4 * face + r,
                        ));
                    }
                }
            }
            (Operands::Unary, _) => {}
            _ => panic!("{operands:?} with no B slots"),
        }
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));

        if n == 0 {
            let block = m.len() as u32;
            m.extend(sync::take(s.unpacked, Before::SFPU));
            let at = m.len() as u32;
            m.extend_from_slice(&math.ins);
            m.extend(sync::post_after(Unit::Sfpu, s.computed));
            if layout.tiles > 1 {
                math_loops.push(Loop {
                    start: block,
                    len: m.len() as u32 - block,
                    count: layout.tiles as u32,
                });
            }
            math_loops.extend(math.loops.iter().map(|l| Loop {
                start: l.start + at,
                ..*l
            }));
        }

        pack.extend(sync::take(s.computed, Before::PACKER));
        pack.extend(pack_tile_from_dst(
            slot(layout.out_at, n) + tt_isa::dm::TILE_DATA,
            OUT_ROW,
        ));
        if checked {
            pack.extend(pack_tile_from_dst(
                slot(layout.c_at.expect("domain status slots"), n) + tt_isa::dm::TILE_DATA,
                C_ROW,
            ));
        }
        pack.extend(sync::post_after(Unit::Packer, s.free));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    ([unpack, m, pack], [Vec::new(), math_loops, Vec::new()])
}

#[cfg(any())]
mod tests {
    use super::*;

    /// The SFPU kernel's semaphores agree with a matmul's on every one they
    /// share, so the two alternate on a tile with no setup between them.
    #[test]
    fn the_semaphores_agree_with_a_matmul_s() {
        let (_, matmul) = super::super::matmul::MatmulSemaphores::alone();
        for operands in [
            Operands::Unary,
            Operands::Binary,
            Operands::RowBroadcast,
            Operands::ColBroadcast,
            Operands::Ternary,
        ] {
            let sfpu = plan_layout(8, operands).unwrap().init;
            for m in &matmul {
                assert!(
                    sfpu.iter().all(|s| s.0 != m.0 || s == m),
                    "{operands:?}: {sfpu:?} against {matmul:?}"
                );
            }
        }
    }
}

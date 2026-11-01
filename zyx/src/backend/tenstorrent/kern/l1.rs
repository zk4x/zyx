//! Placing a kernel's buffers in a tile's L1, and its semaphores in the
//! tile's eight.
//!
//! A kernel never names an L1 address. It declares [`Requirements`]: each
//! buffer's size, alignment and kind (scratch, or a circular buffer with a page
//! size, a page count and one producer and one consumer), the stages of the
//! kernel it is live in, and the semaphores it uses. [`Requirements::plan`]
//! places them in [`tt_isa::l1::DATA`] (or any arena), sharing bytes between
//! buffers that are never live at once and semaphores likewise, and [`check`]
//! verifies a plan independently of how it was made.
//!
//! Fusion is a merge of requirements, not of addresses: [`Requirements::fuse`]
//! concatenates two kernels' stages and unifies the circular buffer one
//! produces with the one the other consumes into one buffer, and the result is
//! planned as one. Neither kernel ever chose an address, so neither can
//! collide with the other; a fused kernel that does not fit is a planning
//! error at build time, never a silent overlap on the device.
//!
//! # Handles
//!
//! [`Buf`] and [`Sem`] carry the identity of the requirements that issued them.
//! Their fields are private, so one cannot be forged, and a plan refuses a
//! handle from any other set of requirements -- the bug fusion would otherwise
//! invite, a kernel's own handle used against the fused plan. (A lifetime brand
//! would catch that at compile time, but could not be stored with a memoised
//! program, nor survive a merge; the check is one comparison.) After a fusion
//! the old handles are mapped through [`Fused`].

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use tt_isa::l1::Region;
use tt_isa::sync::Semaphore;

use super::runtime::SemaphoreInit;

/// Who reads or writes a circular buffer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Endpoint {
    Reader,
    Writer,
    /// The data mover on RISCV B.
    Mover,
    /// The unpack role (thread 0).
    Unpack,
    /// The math role (thread 1).
    Math,
    /// The pack role (thread 2).
    Pack,
}

/// What a buffer is.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Bytes one stage or one core uses on its own.
    Scratch,
    /// A ring of `pages` pages of `page` bytes, written by `producer` and read
    /// by `consumer`, in order.
    Cb {
        page: u64,
        pages: u32,
        producer: Endpoint,
        consumer: Endpoint,
    },
}

/// One buffer a kernel needs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BufferSpec {
    pub name: &'static str,
    pub bytes: u64,
    /// The address is a multiple of this. Not necessarily a power of two: a
    /// tile slot is 4160 bytes.
    pub align: u64,
    /// The stages it holds data across, `[start, end)`.
    pub live: Range<u32>,
    pub kind: Kind,
}

/// One semaphore a kernel needs: the value it starts each run at, and the
/// stages it is in use.
///
/// The starting value is part of the semaphore, not of the kernel that sets
/// it: a concurrent run's setup initialises every semaphore of its plan
/// ([`Plan::semaphore_init`]), and a kernel that restores its semaphores
/// leaves each at its starting value. So two semaphores of different stages
/// may share one of the tile's eight only if they start at the same value --
/// otherwise the second would inherit what the first left.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemSpec {
    pub name: &'static str,
    pub initial: u8,
    pub live: Range<u32>,
}

/// A buffer of one set of [`Requirements`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Buf {
    set: u64,
    index: u32,
}

/// A semaphore of one set of [`Requirements`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Sem {
    set: u64,
    index: u32,
}

fn next_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The most pages a circular buffer can have: its page counters are 15 bits.
pub const MAX_CB_PAGES: u32 = 0x7FFF;

/// Why requirements could not be planned or fused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    /// No address in the arena holds this buffer beside everything live with it.
    DoesNotFit { name: &'static str, bytes: u64, arena: u64 },
    /// More semaphores live at once than the tile's eight.
    TooManySemaphores { name: &'static str },
    /// A circular buffer of more pages than its page counters hold
    /// ([`MAX_CB_PAGES`]): the kernel is too large for one buffer, not
    /// malformed, so a caller that can split the work does.
    TooManyPages { name: &'static str, pages: u32, max: u32 },
    /// A spec that cannot be planned at all, or a fusion edge that does not
    /// join a producer's buffer to a consumer's.
    Invalid(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::DoesNotFit { name, bytes, arena } => {
                write!(f, "{name} ({bytes} bytes) does not fit the {arena}-byte L1 arena beside what is live with it")
            }
            PlanError::TooManySemaphores { name } => {
                write!(f, "{name}: more than eight semaphores live at once")
            }
            PlanError::TooManyPages { name, pages, max } => {
                write!(f, "{name}: a circular buffer of {pages} pages; its page counters hold at most {max}")
            }
            PlanError::Invalid(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for PlanError {}

/// What a kernel needs of a tile: see the module documentation.
#[derive(Clone, Debug)]
pub struct Requirements {
    id: u64,
    stages: u32,
    buffers: Vec<BufferSpec>,
    sems: Vec<SemSpec>,
}

impl Requirements {
    /// A kernel of `stages` stages, needing nothing yet.
    pub fn new(stages: u32) -> Self {
        Requirements { id: next_id(), stages, buffers: Vec::new(), sems: Vec::new() }
    }

    pub fn stages(&self) -> u32 {
        self.stages
    }

    pub fn buffers(&self) -> &[BufferSpec] {
        &self.buffers
    }

    pub fn semaphores(&self) -> &[SemSpec] {
        &self.sems
    }

    /// Declare a buffer.
    pub fn buffer(&mut self, spec: BufferSpec) -> Buf {
        self.buffers.push(spec);
        Buf { set: self.id, index: self.buffers.len() as u32 - 1 }
    }

    /// Declare scratch of `bytes`, aligned to `align`, live in `live`.
    pub fn scratch(&mut self, name: &'static str, bytes: u64, align: u64, live: Range<u32>) -> Buf {
        self.buffer(BufferSpec { name, bytes, align, live, kind: Kind::Scratch })
    }

    /// Declare a circular buffer of `pages` pages of `page` bytes, each page
    /// aligned to `align` (so `page` is a multiple of it).
    #[allow(clippy::too_many_arguments)]
    pub fn cb(
        &mut self,
        name: &'static str,
        page: u64,
        pages: u32,
        align: u64,
        producer: Endpoint,
        consumer: Endpoint,
        live: Range<u32>,
    ) -> Buf {
        self.buffer(BufferSpec {
            name,
            bytes: page * pages as u64,
            align,
            live,
            kind: Kind::Cb { page, pages, producer, consumer },
        })
    }

    /// Declare a semaphore starting at `initial`, in use in `live`.
    pub fn semaphore(&mut self, name: &'static str, initial: u8, live: Range<u32>) -> Sem {
        self.sems.push(SemSpec { name, initial, live });
        Sem { set: self.id, index: self.sems.len() as u32 - 1 }
    }

    /// `first` then `second`, as one kernel: `second`'s stages follow
    /// `first`'s, and each `(out, input)` in `edges` -- a circular buffer of
    /// `first`'s and one of `second`'s -- becomes one circular buffer, written
    /// by `out`'s producer and read by `input`'s consumer, live from the
    /// first's start to the second's end. `out`'s consumer and `input`'s
    /// producer were the legs through GDDR the fusion removes.
    ///
    /// Refused: an edge that is not two circular buffers with the same page
    /// size, a buffer in two edges, or a handle of neither kernel.
    pub fn fuse(first: &Requirements, second: &Requirements, edges: &[(Buf, Buf)]) -> Result<Fused, PlanError> {
        let invalid = |s: String| Err(PlanError::Invalid(s));
        let mut req = Requirements::new(first.stages + second.stages);
        let shift = |r: &Range<u32>| r.start + first.stages..r.end + first.stages;
        let mut map_first = Vec::with_capacity(first.buffers.len());
        for b in &first.buffers {
            map_first.push(req.buffer(b.clone()));
        }
        let mut map_second = vec![None; second.buffers.len()];
        let mut joined = vec![false; first.buffers.len()];
        for &(out, input) in edges {
            if out.set != first.id || input.set != second.id {
                return invalid("a fusion edge names a buffer of neither kernel".into());
            }
            let (o, i) = (out.index as usize, input.index as usize);
            if std::mem::replace(&mut joined[o], true) || map_second[i].is_some() {
                return invalid(format!("{} or {} is in two fusion edges", first.buffers[o].name, second.buffers[i].name));
            }
            let (a, b) = (&first.buffers[o], &second.buffers[i]);
            let (Kind::Cb { page: pa, pages: na, producer, .. }, Kind::Cb { page: pb, pages: nb, consumer, .. }) =
                (a.kind, b.kind)
            else {
                return invalid(format!("{} -> {}: a fusion edge joins two circular buffers", a.name, b.name));
            };
            if pa != pb {
                return invalid(format!("{} -> {}: pages of {pa} and {pb} bytes", a.name, b.name));
            }
            let pages = na.max(nb);
            let merged = &mut req.buffers[o];
            merged.bytes = pa * pages as u64;
            merged.align = a.align.max(b.align);
            merged.live = a.live.start..shift(&b.live).end;
            merged.kind = Kind::Cb { page: pa, pages, producer, consumer };
            map_second[i] = Some(map_first[o]);
        }
        for (i, b) in second.buffers.iter().enumerate() {
            if map_second[i].is_none() {
                map_second[i] = Some(req.buffer(BufferSpec { live: shift(&b.live), ..b.clone() }));
            }
        }
        let sems_first = first.sems.iter().map(|s| req.semaphore(s.name, s.initial, s.live.clone())).collect();
        let sems_second = second.sems.iter().map(|s| req.semaphore(s.name, s.initial, shift(&s.live))).collect();
        Ok(Fused {
            first: (first.id, map_first, sems_first),
            second: (second.id, map_second.into_iter().map(Option::unwrap).collect(), sems_second),
            req,
        })
    }

    fn validate(&self) -> Result<(), PlanError> {
        let bad = |s: String| Err(PlanError::Invalid(s));
        for b in &self.buffers {
            if b.bytes == 0 || b.align == 0 {
                return bad(format!("{}: zero bytes or alignment", b.name));
            }
            if b.live.start >= b.live.end || b.live.end > self.stages {
                return bad(format!("{}: live {:?} outside the kernel's {} stages", b.name, b.live, self.stages));
            }
            if let Kind::Cb { page, pages, producer, consumer } = b.kind {
                if pages > MAX_CB_PAGES {
                    return Err(PlanError::TooManyPages { name: b.name, pages, max: MAX_CB_PAGES });
                }
                if pages == 0 || page % b.align != 0 || producer == consumer {
                    return bad(format!(
                        "{}: a circular buffer needs pages, aligned pages, and a producer \
                         and consumer that differ",
                        b.name
                    ));
                }
            }
        }
        for s in &self.sems {
            if s.live.start >= s.live.end || s.live.end > self.stages {
                return bad(format!("{}: live {:?} outside the kernel", s.name, s.live));
            }
            if s.initial > tt_isa::sync::MAX_VALUE {
                return bad(format!("{}: starts at {}, past the maximum", s.name, s.initial));
            }
        }
        Ok(())
    }

    /// Place every buffer in `arena` and every semaphore in the tile's eight.
    ///
    /// First fit, in declaration order: each buffer at the lowest aligned
    /// address that overlaps no buffer already placed and live in any of the
    /// same stages. Deterministic, so the same requirements always give the
    /// same addresses -- which is what lets programs built from a plan be
    /// memoised by shape.
    pub fn plan(&self, arena: Region) -> Result<Plan, PlanError> {
        self.validate()?;
        let mut addr: Vec<u64> = Vec::with_capacity(self.buffers.len());
        for (i, b) in self.buffers.iter().enumerate() {
            let live_with: Vec<(u64, u64)> = self.buffers[..i]
                .iter()
                .zip(&addr)
                .filter(|(p, _)| overlaps(&p.live, &b.live))
                .map(|(p, &at)| (at, at + p.bytes))
                .collect();
            let fits = |at: u64| at + b.bytes <= arena.end && live_with.iter().all(|&(s, e)| at + b.bytes <= s || at >= e);
            let mut candidates: Vec<u64> = std::iter::once(arena.base)
                .chain(live_with.iter().map(|&(_, e)| e))
                .map(|at| at.next_multiple_of(b.align))
                .collect();
            candidates.sort_unstable();
            let at = candidates.into_iter().find(|&at| fits(at)).ok_or(PlanError::DoesNotFit {
                name: b.name,
                bytes: b.bytes,
                arena: arena.len(),
            })?;
            addr.push(at);
        }
        let mut sems: Vec<u32> = Vec::with_capacity(self.sems.len());
        for (i, s) in self.sems.iter().enumerate() {
            let taken: Vec<u32> = self.sems[..i]
                .iter()
                .zip(&sems)
                // Taken by anything live with it, or by anything that starts
                // at another value (see `SemSpec`).
                .filter(|(p, _)| overlaps(&p.live, &s.live) || p.initial != s.initial)
                .map(|(_, &n)| n)
                .collect();
            let n = (0..8).find(|n| !taken.contains(n)).ok_or(PlanError::TooManySemaphores { name: s.name })?;
            sems.push(n);
        }
        let mut init: Vec<SemaphoreInit> = Vec::new();
        for (s, &n) in self.sems.iter().zip(&sems) {
            if !init.iter().any(|(sem, ..)| sem.index() as u32 == n) {
                let sem = Semaphore::new(n as u8).expect("one of eight");
                init.push((sem, s.initial, tt_isa::sync::MAX_VALUE));
            }
        }
        init.sort_by_key(|(sem, ..)| sem.index());
        Ok(Plan {
            set: self.id,
            init,
            addr,
            sems: sems.into_iter().map(|n| Semaphore::new(n as u8).expect("one of eight")).collect(),
        })
    }
}

fn overlaps(a: &Range<u32>, b: &Range<u32>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Two kernels' requirements merged ([`Requirements::fuse`]), and where each
/// kernel's own handles went.
#[derive(Clone, Debug)]
pub struct Fused {
    pub req: Requirements,
    first: (u64, Vec<Buf>, Vec<Sem>),
    second: (u64, Vec<Buf>, Vec<Sem>),
}

impl Fused {
    /// The fused kernel's handle for `b`, a handle of either kernel.
    pub fn buf(&self, b: Buf) -> Buf {
        let (_, map, _) = self.side(b.set);
        map[b.index as usize]
    }

    /// The fused kernel's handle for `s`, a handle of either kernel.
    pub fn sem(&self, s: Sem) -> Sem {
        let (_, _, map) = self.side(s.set);
        map[s.index as usize]
    }

    fn side(&self, set: u64) -> &(u64, Vec<Buf>, Vec<Sem>) {
        if set == self.first.0 {
            &self.first
        } else if set == self.second.0 {
            &self.second
        } else {
            panic!("a handle of neither fused kernel")
        }
    }
}

/// Where a set of [`Requirements`] was placed.
#[derive(Clone, Debug)]
pub struct Plan {
    set: u64,
    addr: Vec<u64>,
    sems: Vec<Semaphore>,
    init: Vec<SemaphoreInit>,
}

impl Plan {
    /// The L1 address of `b`.
    ///
    /// # Panics
    ///
    /// If `b` is not a handle of the requirements this plan was made from --
    /// a programming error, and the one the handles exist to catch.
    pub fn addr(&self, b: Buf) -> u64 {
        assert_eq!(b.set, self.set, "a buffer handle used against another kernel's plan");
        self.addr[b.index as usize]
    }

    /// The semaphore `s` was given.
    pub fn semaphore(&self, s: Sem) -> Semaphore {
        assert_eq!(s.set, self.set, "a semaphore handle used against another kernel's plan");
        self.sems[s.index as usize]
    }

    /// Every semaphore the plan uses, once, with the value a concurrent run's
    /// setup starts it at (`runtime::Schedule::Concurrent`), in semaphore order.
    pub fn semaphore_init(&self) -> Vec<SemaphoreInit> {
        self.init.clone()
    }
}

/// Check `plan` against `req` and `arena`, independently of how it was made:
/// every buffer aligned and inside the arena, no two buffers live in a common
/// stage sharing a byte, and no two semaphores live in a common stage sharing
/// a number.
pub fn check(req: &Requirements, plan: &Plan, arena: Region) -> Result<(), String> {
    if plan.set != req.id || plan.addr.len() != req.buffers.len() {
        return Err("the plan is not of these requirements".into());
    }
    for (i, (b, &at)) in req.buffers.iter().zip(&plan.addr).enumerate() {
        if at % b.align != 0 {
            return Err(format!("{} at {at:#x} is not {}-aligned", b.name, b.align));
        }
        if !arena.contains(at, b.bytes) {
            return Err(format!("{} at {at:#x}+{} leaves the arena", b.name, b.bytes));
        }
        for (c, &cat) in req.buffers[..i].iter().zip(&plan.addr) {
            if overlaps(&b.live, &c.live) && at < cat + c.bytes && cat < at + b.bytes {
                return Err(format!("{} and {} are live together and overlap at {at:#x}/{cat:#x}", b.name, c.name));
            }
        }
    }
    for (i, (s, n)) in req.sems.iter().zip(&plan.sems).enumerate() {
        for (t, m) in req.sems[..i].iter().zip(&plan.sems) {
            if overlaps(&s.live, &t.live) && n == m {
                return Err(format!("{} and {} share a semaphore while live", s.name, t.name));
            }
            if n == m && s.initial != t.initial {
                return Err(format!("{} and {} share a semaphore but start at {} and {}", s.name, t.name, s.initial, t.initial));
            }
        }
    }
    Ok(())
}

#[cfg(any())]
mod tests {
    use super::*;
    use tt_isa::l1::DATA;

    const SLOT: u64 = tt_isa::dm::TILE_SLOT;

    #[test]
    fn one_stage_buffers_are_packed_in_order() {
        let mut r = Requirements::new(1);
        let a = r.scratch("a", 3 * SLOT, 64, 0..1);
        let b = r.scratch("b", 2 * SLOT, SLOT, 0..1);
        let p = r.plan(DATA).unwrap();
        assert_eq!(p.addr(a), DATA.base);
        assert_eq!(p.addr(b), (DATA.base + 3 * SLOT).next_multiple_of(SLOT));
        check(&r, &p, DATA).unwrap();
    }

    #[test]
    fn buffers_never_live_together_share_bytes() {
        let mut r = Requirements::new(3);
        let a = r.scratch("a", 0x1000, 64, 0..1);
        let b = r.scratch("b", 0x1000, 64, 1..2);
        let c = r.scratch("c", 0x1000, 64, 0..3);
        let p = r.plan(DATA).unwrap();
        assert_eq!(p.addr(a), p.addr(b), "a and b are never live together");
        assert_ne!(p.addr(c), p.addr(a));
        check(&r, &p, DATA).unwrap();
    }

    #[test]
    fn what_does_not_fit_is_an_error_and_so_are_bad_specs() {
        let mut r = Requirements::new(1);
        r.scratch("big", DATA.len() + 1, 64, 0..1);
        assert!(matches!(r.plan(DATA), Err(PlanError::DoesNotFit { name: "big", .. })));
        let mut r = Requirements::new(1);
        r.scratch("dead", 64, 64, 1..2);
        assert!(matches!(r.plan(DATA), Err(PlanError::Invalid(_))));
        let mut r = Requirements::new(1);
        r.cb("loop", 64, 2, 64, Endpoint::Pack, Endpoint::Pack, 0..1);
        assert!(matches!(r.plan(DATA), Err(PlanError::Invalid(_))));
        let mut r = Requirements::new(1);
        for _ in 0..9 {
            r.semaphore("s", 0, 0..1);
        }
        assert!(matches!(r.plan(DATA), Err(PlanError::TooManySemaphores { .. })));
    }

    /// More pages than a circular buffer's counters hold is a kernel too large for one buffer, which
    /// a caller can split: a distinct error, not a malformed spec. The limit's edge: `MAX_CB_PAGES`
    /// pages is a legal buffer (it merely has to fit), one more is not.
    #[test]
    fn too_many_circular_buffer_pages_is_its_own_error() {
        let mut r = Requirements::new(1);
        r.cb("wide", 64, MAX_CB_PAGES + 1, 64, Endpoint::Pack, Endpoint::Unpack, 0..1);
        match r.plan(DATA) {
            Err(PlanError::TooManyPages { name, pages, max }) => {
                assert_eq!((name, pages, max), ("wide", MAX_CB_PAGES + 1, MAX_CB_PAGES));
            }
            other => panic!("expected TooManyPages, got {:?}", other.err()),
        }
        let mut r = Requirements::new(1);
        r.cb("wide", 64, MAX_CB_PAGES, 64, Endpoint::Pack, Endpoint::Unpack, 0..1);
        assert!(
            matches!(r.plan(DATA), Err(PlanError::DoesNotFit { name: "wide", .. })),
            "the largest legal count is validated, then asked to fit"
        );
    }

    #[test]
    fn semaphores_are_shared_only_across_stages() {
        let mut r = Requirements::new(2);
        let s = (0..8).map(|_| r.semaphore("first", 0, 0..1)).collect::<Vec<_>>();
        let t = r.semaphore("second", 0, 1..2);
        let p = r.plan(DATA).unwrap();
        let n: std::collections::BTreeSet<_> = s.iter().map(|&s| p.semaphore(s).index()).collect();
        assert_eq!(n.len(), 8);
        assert_eq!(p.semaphore(t).index(), 0);
        assert_eq!(p.semaphore_init().len(), 8);
        check(&r, &p, DATA).unwrap();
    }

    /// Never live together, but starting at different values: they may not
    /// share, or the second would start at whatever the first left.
    #[test]
    fn semaphores_share_only_with_the_same_starting_value() {
        let mut r = Requirements::new(2);
        let a = r.semaphore("a", 0, 0..1);
        let b = r.semaphore("b", 1, 1..2);
        let c = r.semaphore("c", 0, 1..2);
        let p = r.plan(DATA).unwrap();
        assert_ne!(p.semaphore(a), p.semaphore(b));
        assert_eq!(p.semaphore(a), p.semaphore(c));
        let init = p.semaphore_init();
        assert_eq!(init.len(), 2);
        assert!(init.contains(&(p.semaphore(b), 1, tt_isa::sync::MAX_VALUE)));
        check(&r, &p, DATA).unwrap();
        // And the checker refuses a plan that shares them anyway.
        let mut bad = p.clone();
        bad.sems[1] = bad.sems[0];
        assert!(check(&r, &bad, DATA).unwrap_err().contains("start at"));
    }

    #[test]
    #[should_panic(expected = "another kernel's plan")]
    fn a_handle_from_other_requirements_is_refused() {
        let mut r = Requirements::new(1);
        let mut other = Requirements::new(1);
        r.scratch("a", 64, 64, 0..1);
        let foreign = other.scratch("b", 64, 64, 0..1);
        r.plan(DATA).unwrap().addr(foreign);
    }

    /// Fusing a producer of `C` into a consumer of `C` makes one ring, live
    /// across both, and leaves everything else where liveness allows.
    #[test]
    fn fusion_joins_an_edge_and_plans_as_one() {
        let mut a = Requirements::new(1);
        let a_in = a.cb("a in", SLOT, 2, SLOT, Endpoint::Mover, Endpoint::Unpack, 0..1);
        let a_out = a.cb("a out", SLOT, 2, SLOT, Endpoint::Pack, Endpoint::Mover, 0..1);
        let a_sem = a.semaphore("a ready", 0, 0..1);
        let mut b = Requirements::new(1);
        let b_in = b.cb("b in", SLOT, 4, SLOT, Endpoint::Mover, Endpoint::Unpack, 0..1);
        let b_out = b.cb("b out", SLOT, 2, SLOT, Endpoint::Pack, Endpoint::Mover, 0..1);
        let b_sem = b.semaphore("b ready", 0, 0..1);
        let f = Requirements::fuse(&a, &b, &[(a_out, b_in)]).unwrap();
        assert_eq!(f.req.stages(), 2);
        assert_eq!(f.buf(a_out), f.buf(b_in), "one ring");
        let joined = &f.req.buffers()[2 - 1];
        assert_eq!(joined.live, 0..2);
        assert_eq!(joined.kind, Kind::Cb { page: SLOT, pages: 4, producer: Endpoint::Pack, consumer: Endpoint::Unpack });
        let p = f.req.plan(DATA).unwrap();
        check(&f.req, &p, DATA).unwrap();
        // `a in` is dead in stage 1, so `b out` may reuse it.
        assert_eq!(p.addr(f.buf(a_in)), p.addr(f.buf(b_out)));
        // The two kernels' semaphores are in different stages: one number.
        assert_eq!(p.semaphore(f.sem(a_sem)), p.semaphore(f.sem(b_sem)));
        // An old handle against the fused plan is refused, not misread.
        let r = std::panic::catch_unwind(|| p.addr(a_in));
        assert!(r.is_err());
    }

    #[test]
    fn fusion_refuses_edges_that_are_not_rings_of_one_page_size() {
        let mut a = Requirements::new(1);
        let s = a.scratch("s", SLOT, SLOT, 0..1);
        let o = a.cb("o", SLOT, 2, SLOT, Endpoint::Pack, Endpoint::Mover, 0..1);
        let mut b = Requirements::new(1);
        let i = b.cb("i", 2 * SLOT, 2, SLOT, Endpoint::Mover, Endpoint::Unpack, 0..1);
        let j = b.cb("j", SLOT, 2, SLOT, Endpoint::Mover, Endpoint::Unpack, 0..1);
        assert!(Requirements::fuse(&a, &b, &[(s, j)]).is_err());
        assert!(Requirements::fuse(&a, &b, &[(o, i)]).is_err());
        assert!(Requirements::fuse(&a, &b, &[(o, j), (o, j)]).is_err());
        assert!(Requirements::fuse(&b, &a, &[(o, j)]).is_err());
    }

    /// Random kernels and random fusions of them, planned and checked.
    #[test]
    fn random_requirements_plan_to_valid_layouts() {
        let mut seed = 0x5eed_u64;
        let mut next = |n: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        let ends = [Endpoint::Mover, Endpoint::Unpack, Endpoint::Math, Endpoint::Pack];
        let kernel = |next: &mut dyn FnMut(u64) -> u64| {
            let stages = 1 + next(4) as u32;
            let mut r = Requirements::new(stages);
            let mut cbs = Vec::new();
            for _ in 0..1 + next(8) {
                let s = next(stages as u64) as u32;
                let e = s + 1 + next((stages - s) as u64) as u32;
                let align = [16, 64, SLOT][next(3) as usize];
                if next(2) == 0 {
                    r.scratch("s", align * (1 + next(20)), align, s..e);
                } else {
                    let (p, c) = (next(4) as usize, 1 + next(3) as usize);
                    cbs.push(r.cb("c", align, 1 + next(8) as u32, align, ends[p], ends[(p + c) % 4], s..e));
                }
            }
            for _ in 0..next(6) {
                let s = next(stages as u64) as u32;
                r.semaphore("m", next(3) as u8, s..s + 1);
            }
            (r, cbs)
        };
        let mut planned = 0;
        for _ in 0..500 {
            let (a, a_cbs) = kernel(&mut next);
            let (b, b_cbs) = kernel(&mut next);
            for r in [&a, &b] {
                if let Ok(p) = r.plan(DATA) {
                    check(r, &p, DATA).unwrap_or_else(|e| panic!("{e}: {r:?}"));
                    planned += 1;
                }
            }
            // Fuse one ring of each with the same page size, if there is a pair.
            let edge = a_cbs.iter().find_map(|&o| {
                let po = a.buffers()[o.index as usize].kind;
                b_cbs
                    .iter()
                    .find(|&&i| match (po, b.buffers()[i.index as usize].kind) {
                        (Kind::Cb { page: x, .. }, Kind::Cb { page: y, .. }) => x == y,
                        _ => false,
                    })
                    .map(|&i| (o, i))
            });
            let f = Requirements::fuse(&a, &b, edge.as_slice()).unwrap();
            if let Ok(p) = f.req.plan(DATA) {
                check(&f.req, &p, DATA).unwrap_or_else(|e| panic!("fused: {e}"));
                planned += 1;
            }
        }
        assert!(planned > 900, "{planned}");
    }

    /// The checker is evidence only if it refuses a bad plan: two buffers live
    /// together at one address.
    #[test]
    fn the_checker_refuses_an_aliasing_plan() {
        let mut r = Requirements::new(1);
        r.scratch("a", 0x100, 64, 0..1);
        r.scratch("b", 0x100, 64, 0..1);
        let mut p = r.plan(DATA).unwrap();
        check(&r, &p, DATA).unwrap();
        p.addr[1] = p.addr[0];
        assert!(check(&r, &p, DATA).unwrap_err().contains("overlap"));
        let mut q = r.plan(DATA).unwrap();
        q.addr[1] = DATA.end - 0x80;
        assert!(check(&r, &q, DATA).is_err());
    }
}

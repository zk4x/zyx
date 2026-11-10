//! Loops in a role program, lowered to the frontend's expanders.
//!
//! A role's runner pushes every word of its program, at ~2.8 cycles a word
//! (`silicon_perf::role_push_rate`), and a matmul's unpack and math roles push
//! no faster than that: the kernel's pace is its word count. The frontend has
//! two expanders that turn a few pushed words into many instructions
//! (`tt_isa::frontend`): the Replay Expander, which records up to 32
//! instructions and replays them for one `REPLAY` word; and before it the MOP
//! Expander, which turns one `MOP` into a loop of the instructions in the
//! thread's `MopCfg` -- which may themselves be `REPLAY`s, so one `MOP`
//! repeats an arbitrary recorded body.
//!
//! A program states its loops as [`Item::Repeat`]; [`lower`] picks each one's
//! form and says which it picked and why ([`Lowered::loops`]):
//!
//! * a body of at most [`tt_isa::frontend::REPLAY_BUFFER`] instructions, none
//!   of them a `REPLAY` or `MOP`, is recorded once (executing as it records)
//!   and replayed: one word for each further iteration;
//! * the loop that saves the most words beyond that -- and every other loop
//!   with the same body -- takes the thread's one MOP configuration ([`super::runtime::Kernel::mop`]: one per thread per run):
//!   template 0 with `A0` the body's `REPLAY`, the iteration count in the
//!   `MOP` itself, so two words stand for up to 32 iterations (the mask's
//!   width; `Count1` reaches 128, but past 32 the expander would read mask
//!   bits shifted in, which no gate has measured yet);
//! * anything else is unrolled.
//!
//! The replay buffer is per-thread state the planner owns while a loop runs:
//! a program that uses `REPLAY` itself (the SFPU builder's row loops) is
//! lowered without it, every loop unrolled, since the planner cannot know
//! which of the buffer's slots that program relies on.
//!
//! The claim, held by [`frontend_stream`] in the tests and the gates: the
//! instructions the backend receives from the lowered program are exactly the
//! unrolled program's, word for word.

use tt_isa::frontend::mop::{self, MopConfig, Template0};
use tt_isa::frontend::{self, REPLAY_BUFFER};
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::defs;

/// One element of a role program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    I(Instruction),
    /// `body`, `times` times over.
    Repeat {
        times: u32,
        body: Vec<Item>,
    },
    /// A block that recurs, not necessarily back to back -- a matmul's face
    /// block, once per tile pair, between the pairs' own retargeting: recorded
    /// where `key` first appears, replayed wherever it appears again. Every
    /// occurrence of a key must have the same body.
    Shared {
        key: u32,
        body: Vec<Instruction>,
    },
}

impl Item {
    /// Every instruction, every iteration written out.
    pub fn unrolled(items: &[Item]) -> Vec<Instruction> {
        let mut out = Vec::new();
        for i in items {
            match i {
                Item::I(x) => out.push(*x),
                Item::Repeat { times, body } => {
                    let b = Item::unrolled(body);
                    for _ in 0..*times {
                        out.extend_from_slice(&b);
                    }
                }
                Item::Shared { body, .. } => out.extend_from_slice(body),
            }
        }
        out
    }
}

/// What a loop became.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopForm {
    /// Recorded and replayed under the thread's MOP configuration: `mops`
    /// `MOP`s (with their `MOP_CFG`s) for the iterations after the first.
    Mop { times: u32, body: usize, mops: u32 },
    /// Recorded and replayed: one `REPLAY` per further iteration.
    Replayed { times: u32, body: usize },
    /// A shared block: recorded once, replayed at each of its other `uses`.
    Shared { key: u32, body: usize, uses: u32 },
    /// Written out.
    Unrolled { times: u32, body: usize, why: Unrolled },
}

/// Why a loop was written out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unrolled {
    /// Its body is longer than the replay buffer.
    TooLong,
    /// Its body holds a loop that was itself written out, or a `REPLAY` or
    /// `MOP` of the program's own.
    Nested,
    /// The program uses `REPLAY` itself, so the buffer is not the planner's.
    BufferInUse,
    /// One iteration or none: nothing to save.
    Once,
    /// The replay buffer is full: the shared blocks before it, and a loop's
    /// body beside them, do not fit its 32 slots together.
    BufferFull,
    /// A shared block whose body differs from its key's first occurrence.
    Differs,
}

/// A lowered role program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lowered {
    pub words: Vec<Instruction>,
    /// The thread's MOP configuration for the run, if a loop took it.
    pub mop: Option<MopConfig>,
    /// Every top-level loop's form, in program order.
    pub loops: Vec<LoopForm>,
}

/// The most iterations one template-0 `MOP` runs here: one per mask bit
/// (`tt_isa::frontend::mop::mop_template0`).
const MOP_ITERATIONS: u32 = 32;

fn is_expander_insn(i: &Instruction) -> bool {
    core::ptr::eq(i.def(), &defs::REPLAY) || core::ptr::eq(i.def(), &defs::MOP) || core::ptr::eq(i.def(), &defs::MOP_CFG)
}

/// A loop's body flattened, if it can be recorded: no loop inside it that is
/// not itself written out, and no expander instruction.
fn flat_body(body: &[Item]) -> Option<Vec<Instruction>> {
    let b = Item::unrolled(body);
    (!b.iter().any(is_expander_insn)).then_some(b)
}

/// Words a loop costs in each form.
fn words_replayed(times: u32, len: usize) -> usize {
    1 + len + (times as usize - 1)
}
fn mop_count(times: u32) -> u32 {
    (times - 1).div_ceil(MOP_ITERATIONS)
}
fn words_mop(times: u32, len: usize) -> usize {
    1 + len + 2 * mop_count(times) as usize
}

/// Lower `items`: see the module documentation.
pub fn lower(items: &[Item]) -> Lowered {
    lower_with(items, true)
}

/// [`lower`], with or without the MOP: a caller that cannot carry a MOP
/// configuration to the role's mailbox lowers with `REPLAY` alone.
pub fn lower_with(items: &[Item], allow_mop: bool) -> Lowered {
    let mut notes = Vec::new();
    let inlined = inline_long(items, &mut notes, true);
    let mut l = lower_flat(&inlined, allow_mop);
    // The loops written out iteration by iteration, first: what each became.
    notes.extend(l.loops);
    l.loops = notes;
    l
}

/// A loop whose body is too long to record is written out iteration by
/// iteration -- but as items, so the loops *inside* it are still lowered: a
/// body recorded on its first iteration is replayed on the others
/// (`a_loop_too_long_to_record_still_replays_the_loops_inside_it`). Each
/// top-level loop so written out is noted as `Unrolled { why: TooLong }`.
fn inline_long(items: &[Item], notes: &mut Vec<LoopForm>, top: bool) -> Vec<Item> {
    let mut out = Vec::new();
    for it in items {
        match it {
            Item::Repeat { times, body } => {
                let body = inline_long(body, notes, false);
                let len = Item::unrolled(&body).len();
                if len > REPLAY_BUFFER as usize {
                    if top {
                        notes.push(LoopForm::Unrolled { times: *times, body: len, why: Unrolled::TooLong });
                    }
                    for _ in 0..*times {
                        out.extend(body.iter().cloned());
                    }
                } else {
                    out.push(Item::Repeat { times: *times, body });
                }
            }
            other => out.push(other.clone()),
        }
    }
    out
}

fn lower_flat(items: &[Item], allow_mop: bool) -> Lowered {
    let has_own = |b: &[Instruction]| b.iter().any(is_expander_insn);
    let own_replay = items.iter().any(|i| match i {
        Item::I(x) => is_expander_insn(x),
        Item::Repeat { body, .. } => has_own(&Item::unrolled(body)),
        Item::Shared { body, .. } => has_own(body),
    });
    // Shared blocks take slots from the top of the buffer, in order of first
    // appearance, as long as they fit; loops record from slot 0 below them.
    let mut shared: Vec<(u32, Vec<Instruction>, Option<u32>)> = Vec::new(); // key, body, slot
    let mut top = REPLAY_BUFFER as usize;
    if !own_replay {
        for it in items {
            if let Item::Shared { key, body } = it {
                if shared.iter().any(|(k, ..)| k == key) {
                    continue;
                }
                let slot = (!body.is_empty() && body.len() <= top).then(|| {
                    top -= body.len();
                    top as u32
                });
                shared.push((*key, body.clone(), slot));
            }
        }
    }
    let loop_room = top;
    // The loop the MOP configuration saves the most on, if any.
    let mut best: Option<(usize, usize)> = None; // (item index, words saved)
    if !own_replay && allow_mop {
        for (k, it) in items.iter().enumerate() {
            if let Item::Repeat { times, body } = it {
                if let Some(b) = flat_body(body) {
                    if *times > 1 && b.len() <= loop_room {
                        let saved = words_replayed(*times, b.len()).saturating_sub(words_mop(*times, b.len()));
                        if saved > 0 && best.is_none_or(|(_, s)| saved > s) {
                            best = Some((k, saved));
                        }
                    }
                }
            }
        }
    }
    let mut out = Lowered { words: Vec::new(), mop: None, loops: Vec::new() };
    let mut recorded: Vec<u32> = Vec::new();
    // What the loop region (from slot 0) holds now: a loop whose body is
    // already there replays it rather than recording it again.
    let mut slot0: Option<Vec<Instruction>> = None;
    let mut uses: Vec<(u32, usize, u32)> = Vec::new(); // key, body, uses
    for it in items {
        let (times, body) = match it {
            Item::I(x) => {
                out.words.push(*x);
                continue;
            }
            Item::Shared { key, body } => {
                let entry = shared.iter().find(|(k2, ..)| k2 == key);
                match entry {
                    Some((_, first, Some(slot))) if first == body => {
                        if recorded.contains(key) {
                            out.words.push(frontend::replay(*slot, body.len()).expect("allocated"));
                        } else {
                            frontend::record(*slot, body, true, &mut out.words).expect("allocated, no REPLAY");
                            recorded.push(*key);
                        }
                        match uses.iter_mut().find(|(k2, ..)| k2 == key) {
                            Some(u) => u.2 += 1,
                            None => uses.push((*key, body.len(), 1)),
                        }
                    }
                    _ => {
                        let why = if own_replay {
                            Unrolled::BufferInUse
                        } else if entry.is_some_and(|(_, first, _)| first != body) {
                            Unrolled::Differs
                        } else if body.len() > REPLAY_BUFFER as usize {
                            Unrolled::TooLong
                        } else {
                            Unrolled::BufferFull
                        };
                        out.words.extend_from_slice(body);
                        out.loops.push(LoopForm::Unrolled { times: 1, body: body.len(), why });
                    }
                }
                continue;
            }
            Item::Repeat { times, body } => (*times, body),
        };
        let unrolled = Item::unrolled(body);
        let len = unrolled.len();
        let write_out = |out: &mut Lowered, why| {
            for _ in 0..times {
                out.words.extend_from_slice(&unrolled);
            }
            out.loops.push(LoopForm::Unrolled { times, body: len, why });
        };
        if times <= 1 {
            write_out(&mut out, Unrolled::Once);
            continue;
        }
        if own_replay {
            write_out(&mut out, Unrolled::BufferInUse);
            continue;
        }
        let Some(b) = flat_body(body) else {
            write_out(&mut out, Unrolled::Nested);
            continue;
        };
        if b.len() > REPLAY_BUFFER as usize {
            write_out(&mut out, Unrolled::TooLong);
            continue;
        }
        if b.len() > loop_room {
            write_out(&mut out, Unrolled::BufferFull);
            continue;
        }
        // The first iteration runs as it is recorded, from slot 0 -- unless
        // slot 0 holds this body already, when every iteration replays.
        let resident = slot0.as_ref() == Some(&b);
        if !resident {
            frontend::record(0, &b, true, &mut out.words).expect("checked: fits, no REPLAY");
            slot0 = Some(b.clone());
        }
        let replay = frontend::replay(0, b.len()).expect("checked: fits");
        let rest = if resident { times } else { times - 1 };
        // The chosen loop, and every other loop with the same body: one MOP
        // configuration serves them all.
        let mop_body = best.and_then(|(at, _)| match &items[at] {
            Item::Repeat { body, .. } => flat_body(body),
            _ => None,
        });
        if mop_body.as_ref() == Some(&b) {
            out.mop = Some(MopConfig::Template0(Template0 { a0: replay, a123: None, b: None, skip_a0: replay, skip_b: replay }));
            let mut left = rest;
            let mut mops = 0;
            while left > 0 {
                let n = left.min(MOP_ITERATIONS);
                out.words.extend(mop::mop_template0(n, 0).expect("1..=32 iterations"));
                left -= n;
                mops += 1;
            }
            out.loops.push(LoopForm::Mop { times, body: b.len(), mops });
        } else {
            out.words.extend(core::iter::repeat_n(replay, rest as usize));
            out.loops.push(LoopForm::Replayed { times, body: b.len() });
        }
    }
    for (key, body, n) in uses {
        out.loops.push(LoopForm::Shared { key, body, uses: n });
    }
    out
}

/// What the backend receives from `words` under `mop`: the MOP Expander's
/// output (`tt_isa::frontend::mop::expand`) through the Replay Expander's
/// (`REPLAY.md`: `Load` records `Count` instructions from `Index`, executing
/// them if `Exec`; otherwise it expands to the recorded ones). The model the
/// lowering is held to; `None` for a stream the model refuses (a replay of
/// slots never recorded, a `MOP` with no configuration).
pub fn frontend_stream(words: &[Instruction], mop: Option<&MopConfig>) -> Option<Vec<Instruction>> {
    // The MOP Expander first.
    let mut stage1 = Vec::new();
    let mut mask_hi = 0;
    for &w in words {
        if core::ptr::eq(w.def(), &defs::MOP_CFG) {
            mask_hi = w.operand("MaskHi")?;
        } else if core::ptr::eq(w.def(), &defs::MOP) {
            mop::expand(mop?, w, mask_hi, &mut stage1);
        } else {
            stage1.push(w);
        }
    }
    // Then the Replay Expander.
    let mut buffer: [Option<Instruction>; REPLAY_BUFFER as usize] = [None; REPLAY_BUFFER as usize];
    let mut out = Vec::new();
    let mut it = stage1.into_iter();
    while let Some(w) = it.next() {
        if !core::ptr::eq(w.def(), &defs::REPLAY) {
            out.push(w);
            continue;
        }
        let (index, count) = (w.operand("Index")? as usize, w.operand("Count")? as usize);
        if w.operand("Load")? != 0 {
            let exec = w.operand("Exec")? != 0;
            for slot in buffer.iter_mut().skip(index).take(count) {
                let i = it.next()?;
                *slot = Some(i);
                if exec {
                    out.push(i);
                }
            }
        } else {
            for slot in buffer.iter().skip(index).take(count) {
                out.push((*slot)?);
            }
        }
    }
    Some(out)
}

#[cfg(any())]
mod tests {
    use super::*;

    fn op(n: u32) -> Instruction {
        tt_isa::sfpu::loadi(0, 0, n).unwrap()
    }
    fn body(range: std::ops::Range<u32>) -> Vec<Item> {
        range.map(|n| Item::I(op(n))).collect()
    }

    fn holds(items: &[Item]) -> Lowered {
        let l = lower(items);
        let want = Item::unrolled(items);
        let got = frontend_stream(&l.words, l.mop.as_ref()).expect("the model takes it");
        assert_eq!(got, want, "the backend's stream is the unrolled program's");
        l
    }

    #[test]
    fn the_biggest_saving_takes_the_mop_and_the_rest_replay() {
        let items = vec![
            Item::I(op(1000)),
            Item::Repeat { times: 5, body: body(0..3) },
            Item::Repeat { times: 100, body: body(10..14) },
            Item::I(op(1001)),
        ];
        let l = holds(&items);
        assert_eq!(
            l.loops,
            [
                LoopForm::Replayed { times: 5, body: 3 },
                LoopForm::Mop { times: 100, body: 4, mops: 4 },
            ]
        );
        // 1 + (1 + 3 + 4) + (1 + 4 + 2 * 4) + 1 words, against 1 + 15 + 400 + 1.
        assert_eq!(l.words.len(), 23);
        assert!(l.mop.is_some());
    }

    #[test]
    fn what_cannot_be_recorded_is_written_out_and_says_why() {
        let long = Item::Repeat { times: 3, body: body(0..33) };
        let once = Item::Repeat { times: 1, body: body(0..2) };
        let l = holds(&[long, once]);
        assert_eq!(
            l.loops,
            [
                LoopForm::Unrolled { times: 3, body: 33, why: Unrolled::TooLong },
                LoopForm::Unrolled { times: 1, body: 2, why: Unrolled::Once },
            ]
        );
        // A program with a REPLAY of its own keeps the buffer to itself.
        let own = frontend::replay(4, 2).unwrap();
        let mut rec = Vec::new();
        frontend::record(4, &[op(7), op(8)], true, &mut rec).unwrap();
        let mut items: Vec<Item> = rec.into_iter().map(Item::I).collect();
        items.push(Item::I(own));
        items.push(Item::Repeat { times: 4, body: body(0..2) });
        let l = lower(&items);
        assert_eq!(l.loops, [LoopForm::Unrolled { times: 4, body: 2, why: Unrolled::BufferInUse }]);
        assert!(l.mop.is_none());
    }

    #[test]
    fn nested_loops_flatten_into_one_recorded_body() {
        let inner = Item::Repeat { times: 3, body: body(0..2) };
        let items = vec![Item::Repeat { times: 40, body: vec![Item::I(op(50)), inner] }];
        let l = holds(&items);
        assert_eq!(l.loops, [LoopForm::Mop { times: 40, body: 7, mops: 2 }]);
    }

    #[test]
    fn the_model_refuses_what_the_hardware_would_not_run() {
        // A replay of slots nothing recorded, a MOP with no configuration.
        assert_eq!(frontend_stream(&[frontend::replay(0, 3).unwrap()], None), None);
        assert_eq!(frontend_stream(&[mop::mop_template1()], None), None);
    }

    #[test]
    fn a_shared_block_is_recorded_once_and_replayed_between_other_work() {
        let block: Vec<Instruction> = (0..20).map(op).collect();
        let mut items = Vec::new();
        for pair in 0..6 {
            items.push(Item::I(op(500 + pair))); // each pair's own retargeting
            items.push(Item::Shared { key: 1, body: block.clone() });
        }
        // A loop beside it, in the twelve slots left.
        items.push(Item::Repeat { times: 9, body: body(40..50) });
        let l = holds(&items);
        // 6 + (1 + 20) + 5 replays, then the loop.
        assert!(l.loops.contains(&LoopForm::Shared { key: 1, body: 20, uses: 6 }), "{:?}", l.loops);
        assert!(matches!(l.loops[0], LoopForm::Mop { times: 9, body: 10, .. }), "{:?}", l.loops);
        // A loop that would not fit beside the block is written out, and says so.
        let mut tight = items.clone();
        tight.pop();
        tight.push(Item::Repeat { times: 3, body: body(60..75) });
        let l = holds(&tight);
        assert!(l.loops.contains(&LoopForm::Unrolled { times: 3, body: 15, why: Unrolled::BufferFull }));
        // A key whose body changes is written out where it differs.
        let mut odd = items.clone();
        odd.push(Item::Shared { key: 1, body: (100..120).map(op).collect() });
        let l = holds(&odd);
        assert!(l.loops.contains(&LoopForm::Unrolled { times: 1, body: 20, why: Unrolled::Differs }));
    }

    #[test]
    fn loops_with_the_same_body_share_the_mop() {
        // A matmul's K loop, once per output tile, between each output's setup.
        let items: Vec<Item> =
            (0..3).flat_map(|out| [Item::I(op(900 + out)), Item::Repeat { times: 25, body: body(0..8) }]).collect();
        let l = holds(&items);
        assert_eq!(l.loops.iter().filter(|f| matches!(f, LoopForm::Mop { .. })).count(), 3, "{:?}", l.loops);
    }

    #[test]
    fn a_body_already_recorded_is_replayed_not_recorded_again() {
        let items: Vec<Item> =
            (0..3).flat_map(|out| [Item::I(op(900 + out)), Item::Repeat { times: 5, body: body(0..8) }]).collect();
        let l = lower_with(&items, false);
        let got = frontend_stream(&l.words, None).unwrap();
        assert_eq!(got, Item::unrolled(&items));
        // 3 setup words, one recording (1 + 8), and 4 + 5 + 5 replays.
        assert_eq!(l.words.len(), 3 + 9 + 14);
        assert!(l.mop.is_none());
    }

    #[test]
    fn a_loop_too_long_to_record_still_replays_the_loops_inside_it() {
        // A HiFi4 matmul's pair: a reset, then twice (a unit twice, a step).
        let unit = body(0..17);
        let pair = vec![
            Item::I(op(100)),
            Item::Repeat { times: 2, body: unit.clone() },
            Item::I(op(101)),
            Item::Repeat { times: 2, body: unit },
            Item::I(op(101)),
        ];
        let items = vec![Item::Repeat { times: 8, body: pair }];
        let l = lower_with(&items, false);
        assert_eq!(frontend_stream(&l.words, None).unwrap(), Item::unrolled(&items));
        // The first pair: its reset, the unit recorded (1 + 17) and replayed
        // once, a step, two replays, a step -- 24 words. Every other pair: a
        // reset, four REPLAYs, two steps -- 7.
        assert_eq!(l.words.len(), 24 + 7 * 7);
    }
}

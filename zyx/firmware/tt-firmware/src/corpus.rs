//! The body of the generic Tensix program runner, shared by the `corpus` (T1)
//! and `corpus_t0` (T0) images.
//!
//! Generic over the pushing core and its Tensix thread because which one is right
//! differs by target. ttsim models the RISC-V view of `Dst` only for T1
//! (divergence row 12). Silicon needs the program to run on thread 0, because
//! `UNPACR` in multi-context mode reads `ADCs[ContextADC]` -- thread 0 -- whichever
//! thread issues it, while the `SETADC*` instructions that program it write the
//! issuing thread's (divergence row 45). The core and thread stay type
//! parameters, so `PushesTo` still rules out the pairings that hang.

use crate::cfg::write_config_field;
use crate::tensix::{load_mop_config, push_word, read_dst32, wait_for_coprocessor};
use crate::{fail, finish, l1_read32, l1_write32, publish, spin};
use tt_isa::cfg::ConfigBank;
use tt_isa::mailbox::guard::{self, Guard};
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{self, panic_code, status};
use tt_isa::sfpu::dst32_address;
use tt_isa::tensix::{PushesTo, TensixThread};

/// The hooks the `.ttinsn` experiment image (`bin/ttinsn_t1.rs`) installs
/// (`docs/plans/ttinsn-push-experiment.md`). Single-threaded: one core runs
/// this, and the image sets them before the runner starts.
#[cfg(feature = "ttinsn-experiment")]
pub mod experiment {
    /// Push the staged program by other means, returning whether it did.
    /// Arguments: the program's address, its length in words and its block
    /// repeats, as the runner holds them.
    pub type TryPush = unsafe fn(code: u64, code_len: u32, loops: &[(u32, u32, u32)]) -> bool;
    /// Called once the coprocessor has retired the program, before the `Dst`
    /// dump.
    pub type AfterRetire = fn();

    pub static mut TRY_PUSH: Option<TryPush> = None;
    pub static mut AFTER_RETIRE: Option<AfterRetire> = None;
    /// Leave instruction gathering on, as the core boots, instead of turning it
    /// off as every shipped image does (`crate::disable_gathering`): what the
    /// experiment's gathering measurements compare against.
    pub static mut KEEP_GATHERING: bool = false;

    pub(super) fn try_push(code: u64, code_len: u32, loops: &[(u32, u32, u32)]) -> bool {
        // SAFETY: written once before the runner starts, by this core.
        match unsafe { TRY_PUSH } {
            // SAFETY: the hook's own contract; the arguments are the runner's
            // checked program.
            Some(f) => unsafe { f(code, code_len, loops) },
            None => false,
        }
    }

    pub(super) fn after_retire() {
        // SAFETY: as above.
        if let Some(f) = unsafe { AFTER_RETIRE } {
            f()
        }
    }
}

pub fn run<Riscv, Thread>() -> !
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    run_in::<Riscv, Thread>(Mailbox::single_core(), false)
}

/// The same runner, reporting through a role mailbox: one of three cores each
/// running its own part of a kernel (`tt_isa::mailbox::role`).
///
/// A role also translates a program with block repeats into looping code when
/// the exchange pays ([`translate`]): each role runs on the core whose Tensix
/// thread it pushes to, so the code goes in that role's own window and the core
/// invalidates its own instruction cache.
pub fn run_role<Riscv, Thread>() -> !
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    run_in::<Riscv, Thread>(Mailbox::of(Thread::INDEX), true)
}

/// Words of expanded program per stored word the translation must save before it
/// is worth doing: `expanded * 2.1 > stored * 22 + 330`, the measured break-even
/// for one launch with no translation cache (`docs/plans/ttinsn-push-experiment.md`).
/// Ten times each term, in integers: the runner saves about 2.1 cycles a pushed
/// word, and translating costs 13-22 a stored word plus ~330 fixed.
const TRANSLATE_GAIN: u64 = 21;
const TRANSLATE_COST: u64 = 220;
const TRANSLATE_FIXED: u64 = 3_300;

/// Where a role writes translated code: spare L1 in the image area's tail, above
/// NC's two-word reset stub (`0x1_2000`) and below the mover's list (`0x1_4000`).
/// The role images fill their slots (9.5-9.9 KiB of a 10 KiB slot), so a window
/// cannot live inside one.
const CODE_WINDOW: [u64; 3] = [0x1_2010, 0x1_2A10, 0x1_3410];
/// Words one window holds (2.5 KiB).
const CODE_WORDS: usize = 0x0A00 / 4;

/// Push the staged program at `code` by translating it into looping `.ttinsn`
/// code in this role's code window and calling it, and say whether it did: false
/// means the caller pushes it the ordinary way.
///
/// The code replaces the loop table with a real branch, which is what makes a
/// long-expanded, short-stored program (an SFPU row body repeated per tile)
/// cheap. It is left in the window: the next translated program overwrites it.
fn translate<Thread: TensixThread>(code: u64, code_len: u32, loops: &[(u32, u32, u32)]) -> bool {
    // A staged word that steps an address is the translator's own (it becomes
    // `sw`/`addi`, never an instruction of the machine), so a program holding
    // one is translated whether or not the exchange pays: pushing it as it is
    // would push a word the coprocessor cannot decode.
    let mut stepped = false;
    for i in 0..code_len {
        // SAFETY: inside the runner's checked program.
        if unsafe { crate::l1_read32(code + 4 * u64::from(i)) } & crate::ttinsn::STEP_MASK == crate::ttinsn::STEP_STORE {
            stepped = true;
            break;
        }
    }
    let stored = u64::from(code_len);
    let mut expanded = stored;
    for &(_, len, count) in loops {
        expanded += u64::from(len) * u64::from(count.saturating_sub(1));
    }
    if !stepped && expanded * TRANSLATE_GAIN <= stored * TRANSLATE_COST + TRANSLATE_FIXED {
        return false;
    }
    let window = CODE_WINDOW[Thread::INDEX as usize];
    // SAFETY: the runner's checked program, and a window this role owns.
    let Some(n) =
        (unsafe { crate::ttinsn::translate(code, code_len, loops, Some(&crate::ttinsn::OPCODES), window, CODE_WORDS) })
    else {
        return false;
    };
    // SAFETY: the last word just written; this core's own bit in the mask is
    // `1 << (INDEX + 1)` (B is bit 0, then T0, T1, T2, NC).
    unsafe { crate::ttinsn::invalidate_icache(window + 4 * (n as u64 - 1), 1 << (Thread::INDEX + 1)) };
    // SAFETY: valid code ending in a return, translated above, its lines
    // invalidated after it was written.
    unsafe { crate::ttinsn::call(window) };
    true
}

/// Stop, having said why in `mb` as well as in the single-core mailbox the
/// panic path always uses.
fn fail_in(mb: Mailbox, code: u32) -> ! {
    // SAFETY: fixed aligned mailbox locations inside L1.
    unsafe {
        l1_write32(mb.panic_code(), code);
        l1_write32(mb.status(), status::PANICKED);
    }
    fail(code)
}

/// Record `event` through the tile's timestamper, if the host asked for it.
///
/// One store per event, of a 128-bit event, which the timestamper writes to L1
/// whole -- so the three role cores can share one stream without their events
/// interleaving (`DebugTimestamper.md`). Off unless the mailbox says so:
/// whether ttsim models the timestamper is a question with its own probe, and
/// a register it does not model is fatal there.
fn trace(on: bool, thread: u32, event: u32) {
    if on {
        // SAFETY: a documented, aligned timestamper register; a store has no
        // effect beyond appending the event.
        unsafe {
            l1_write32(
                tt_isa::tensix::timestamper::TIMESTAMP,
                tt_isa::tensix::timestamper::event_128(mailbox::trace::token(thread, event)),
            )
        };
    }
}

/// Release whatever a failed kernel left a Tensix thread blocked on
/// (`mailbox::UNWEDGE`): post every semaphore that reads zero, in rounds, so
/// a `SEMWAIT` the backend reset did not clear completes -- and any later one
/// in what it had queued behind it. The reset program the runner then pushes
/// initialises every semaphore afresh, so the posts leave nothing behind.
fn unwedge() {
    use tt_isa::tensix::SEMAPHORE_ACCESS;
    // Until nothing has consumed a post for 64 rounds running: whatever the
    // thread had queued behind its first wait has run (each later wait taking
    // the next round's post), or it waits on something no post can release.
    // Bounded, so a tile wedged some other way still reaches the reset.
    let (mut quiet, mut rounds) = (0u32, 0u32);
    while quiet < 64 && rounds < 1 << 20 {
        rounds += 1;
        let mut posted = false;
        for i in 0..8u64 {
            let at = SEMAPHORE_ACCESS + 4 * i;
            // SAFETY: the documented semaphore window of this core; a load
            // reads a value, an even store posts.
            unsafe {
                if l1_read32(at) == 0 {
                    l1_write32(at, 0);
                    posted = true;
                }
            }
        }
        quiet = if posted { 0 } else { quiet + 1 };
        publish();
        for _ in 0..256 {
            // Not `spin_loop`: its `pause` is an encoding Blackhole lacks.
            // SAFETY: a no-op.
            unsafe { core::arch::asm!("nop") };
        }
    }
}

/// What a guarded run was armed with (`tt_isa::mailbox::guard`).
struct GuardRun {
    deadline: u32,
    grace: u32,
    complete: u32,
    mode: u32,
}

/// Record how far a guarded run got, where the host can read it
/// ([`Guard::stage`]). Fenced, so the breadcrumb is visible even if the very
/// next step hangs this core.
fn crumb(g: Guard, stage: u32) {
    // SAFETY: a fixed aligned word of this role's guard block.
    unsafe { l1_write32(g.stage(), stage) };
    publish();
}

/// One load of Tensix semaphore `i`, its result consumed (`andi`) before the
/// function returns, so no later load starts while this one is outstanding
/// (`ManualTTSync.md`: loads in `PC_BUF_BASE..+0xFFFF` must not overlap).
fn semaphore_load(i: u64) -> u32 {
    // SAFETY: the documented semaphore window of this core; a load reads the
    // value and has no side effect.
    let mut v = unsafe { l1_read32(tt_isa::tensix::SEMAPHORE_ACCESS + 4 * i) };
    // SAFETY: an ALU instruction on a register.
    unsafe { core::arch::asm!("andi {v}, {v}, 15", v = inout(reg) v, options(nomem, nostack)) };
    v
}

/// All eight semaphores as one packed word, one nibble each. `raw` loads them
/// back to back, as the first guarded design did; otherwise each is consumed
/// before the next. With `first`, a breadcrumb names the load about to start.
fn semaphore_snapshot(g: Guard, raw: bool, first: bool) -> u32 {
    let mut snapshot = 0u32;
    for i in 0..8u64 {
        if first {
            crumb(g, guard::stage::FIRST_LOAD + i as u32);
        }
        let v = if raw {
            // SAFETY: as `semaphore_load`, without consuming the result.
            unsafe { l1_read32(tt_isa::tensix::SEMAPHORE_ACCESS + 4 * i) & 0xf }
        } else {
            semaphore_load(i)
        };
        snapshot |= v << (4 * i);
    }
    snapshot
}

/// Whether the host armed this run as **guarded** (`tt_isa::mailbox::guard`),
/// consuming the arming word so a stale one cannot guard a later, unrelated run.
///
/// A guarded program may park its Tensix thread in a Wait Gate (a held mutex, a
/// retrying `ATCAS`, a full or empty `ATINCGETPTR` FIFO). The ordinary
/// end-of-program `wait_for_coprocessor` is a load that stalls inside the
/// memory subsystem until the thread drains, with no way for this core to time
/// it out -- so for these runs it is replaced by [`guard_wait`].
fn guard_armed(mb: Mailbox) -> bool {
    let g = Guard::of(mb);
    // SAFETY: fixed aligned words of the role mailbox's guard block.
    unsafe {
        if l1_read32(g.arm()) != guard::ARMED {
            return false;
        }
        l1_write32(g.arm(), 0);
    }
    crumb(g, guard::stage::ARM_SEEN);
    true
}

/// Read and check an armed run's parameters and clear what a previous run could
/// have left that would read as this program's completion.
fn guard_begin(
    mb: Mailbox,
    streamed: bool,
    looped: bool,
    dump_rows: u32,
    program_len: u32,
) -> GuardRun {
    let g = Guard::of(mb);
    // SAFETY: fixed aligned words of this role's guard block.
    let run = unsafe {
        GuardRun {
            deadline: l1_read32(g.deadline()),
            grace: l1_read32(g.grace()),
            complete: l1_read32(g.complete_semaphore()),
            mode: l1_read32(g.mode()),
        }
    };
    if streamed
        || looped
        || dump_rows != 0
        || program_len > guard::MAX_WORDS
        || run.deadline == 0
        || run.deadline > guard::MAX_POLLS
        || run.grace == 0
        || run.grace > guard::MAX_POLLS
        || run.complete >= 8
        || run.mode > guard::PollMode::L1Word as u32
    {
        fail_in(mb, guard::REFUSED);
    }
    crumb(g, guard::stage::PARAMS_OK);
    if run.mode == guard::PollMode::L1Word as u32 {
        // SAFETY: a fixed aligned word of the guard block.
        unsafe { l1_write32(g.complete_word(), 0) };
    } else {
        // A post left over from an earlier run would read as this program's
        // completion: take every one down first (an odd store is a `SEMGET`).
        let at = tt_isa::tensix::SEMAPHORE_ACCESS + 4 * run.complete as u64;
        for _ in 0..16 {
            if semaphore_load(run.complete as u64) == 0 {
                break;
            }
            // SAFETY: the semaphore window of this core.
            unsafe { l1_write32(at, 1) };
        }
    }
    crumb(g, guard::stage::CLEARED);
    run
}

/// Publish `snapshot` if it differs from the last one shown.
fn show(g: Guard, snapshot: u32, shown: &mut u32) {
    if snapshot != *shown {
        *shown = snapshot;
        // SAFETY: a fixed aligned word of the guard block.
        unsafe { l1_write32(g.snapshot(), snapshot) };
        publish();
    }
}

/// Wait for a guarded program's completion with a bounded poll.
///
/// How completion is seen depends on the armed [`guard::PollMode`]: the
/// program's last instruction posts a semaphore, which this core reads (every
/// poll all eight, publishing them as one packed word the host can read
/// ([`Guard::snapshot`]); or only the completion semaphore, with the full set
/// every [`guard::LIGHT_SNAPSHOT_EVERY`] polls), or stores a token to an L1 word
/// that this core polls. The semaphore is taken down again once seen (an odd
/// store is a `SEMGET`).
///
/// After `deadline` polls without it the role reports [`guard::BLOCKED`] in its
/// status word, where the host can see it, and keeps polling for up to `grace`
/// more. Every poll also honours the host's release request: the semaphores in
/// the low eight bits of [`Guard::release`] are posted from this core (a store
/// to the semaphore window, which queues behind nothing the blocked thread
/// holds), and the request is cleared and counted. Once the grace runs out the
/// role gives up with [`guard::ABANDONED`] -- *without* ever issuing the
/// coprocessor-drain load that would hang this core behind the stuck thread.
///
/// The count is polls rather than `mcycle`, which ttsim does not model
/// (divergence row 71). The running total is written to [`Guard::polls`] every
/// 16 polls, so a host that sees it frozen knows the loop is not turning.
fn guard_wait(mb: Mailbox, run: &GuardRun) {
    use guard::{stage, PollMode};
    use tt_isa::tensix::SEMAPHORE_ACCESS;
    let g = Guard::of(mb);
    crumb(g, stage::POLL_ENTERED);
    let complete = run.complete as u64;
    let (mut total, mut polls, mut blocked, mut released) = (0u32, 0u32, false, 0u32);
    let (mut shown, mut finishing) = (u32::MAX, false);
    loop {
        publish();
        let first = total == 0;
        if finishing {
            // One more full pass: the semaphores are read one after another, so
            // a post that preceded the completion may not be in the last one.
            let s = semaphore_snapshot(g, false, false);
            show(g, s, &mut shown);
            break;
        }
        let done = if run.mode == PollMode::L1Word as u32 {
            // SAFETY: a fixed aligned word of the guard block.
            unsafe { l1_read32(g.complete_word()) == guard::COMPLETE_TOKEN }
        } else if run.mode == PollMode::Light as u32 {
            if first {
                crumb(g, stage::FIRST_LOAD + run.complete);
            }
            let done = semaphore_load(complete) != 0;
            if done || first || total & (guard::LIGHT_SNAPSHOT_EVERY - 1) == 0 {
                let s = semaphore_snapshot(g, false, false);
                show(g, s, &mut shown);
            }
            done
        } else {
            let s = semaphore_snapshot(g, true, first);
            show(g, s, &mut shown);
            (s >> (4 * run.complete)) & 0xf != 0
        };
        if first {
            crumb(g, stage::FIRST_POLL_DONE);
        }
        if done {
            if run.mode == PollMode::L1Word as u32 {
                break;
            }
            // SAFETY: the semaphore window of this core; an odd store is a
            // SEMGET.
            unsafe { l1_write32(SEMAPHORE_ACCESS + 4 * complete, 1) };
            finishing = true;
            continue;
        }
        // SAFETY: a fixed aligned word of the guard block.
        let request = unsafe { l1_read32(g.release()) };
        if request & (guard::RELEASE_MASK | guard::POKE) != 0 {
            if request & guard::POKE != 0 {
                // A store by this core to an L1 word a parked `ATCAS` or
                // `ATINCGETPTR` polls: checked to be one aligned word of the
                // data arena, so a stale or corrupt request cannot write
                // firmware, a mailbox or a program.
                // SAFETY: fixed aligned words of the guard block.
                let (at, value) = unsafe { (l1_read32(g.poke_addr()), l1_read32(g.poke_value())) };
                if at % 4 != 0 || !tt_isa::l1::DATA.contains(at as u64, 4) {
                    fail_in(mb, guard::REFUSED);
                }
                // SAFETY: inside the data arena, checked above.
                unsafe { l1_write32(at as u64, value) };
                publish();
            }
            for i in 0..8u64 {
                if request & (1 << i) != 0 {
                    // SAFETY: the semaphore window; an even store posts.
                    unsafe { l1_write32(SEMAPHORE_ACCESS + 4 * i, 0) };
                }
            }
            released += 1;
            // SAFETY: as above; the count first, so a host that sees the
            // request cleared finds it counted.
            unsafe {
                l1_write32(g.released(), released);
                l1_write32(g.release(), 0);
            }
            publish();
        }
        total = total.saturating_add(1);
        polls = polls.saturating_add(1);
        if total & 0xf == 0 {
            // SAFETY: a fixed aligned word of the guard block.
            unsafe { l1_write32(g.polls(), total) };
            publish();
        }
        if !blocked {
            if polls >= run.deadline {
                blocked = true;
                polls = 0;
                // SAFETY: fixed aligned mailbox word.
                unsafe { l1_write32(mb.status(), guard::BLOCKED) };
                crumb(g, stage::BLOCKED);
            }
        } else if polls >= run.grace {
            // SAFETY: as above.
            unsafe { l1_write32(g.polls(), total) };
            fail_in(mb, guard::ABANDONED);
        }
    }
    // SAFETY: as above.
    unsafe { l1_write32(g.polls(), total) };
    crumb(g, stage::POLL_DONE);
    // Every blocking instruction has completed; what follows the post is a
    // program epilogue of instructions that cannot block (the builder in
    // `tt_kernels::atomics` puts the post last).
    wait_for_coprocessor();
    crumb(g, stage::DRAINED);
}

/// `translate_programs`: this core is a role's own core, so a program it pushes
/// may be translated into code in that role's window and called ([`translate`]).
fn run_in<Riscv, Thread>(mb: Mailbox, translate_programs: bool) -> !
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    // Gathering off before anything is pushed (`crate::disable_gathering`); the
    // `.ttinsn` experiment may ask to measure with it left on.
    // SAFETY: written once before the runner starts, by this core.
    #[cfg(feature = "ttinsn-experiment")]
    let keep_gathering = unsafe { experiment::KEEP_GATHERING };
    #[cfg(not(feature = "ttinsn-experiment"))]
    let keep_gathering = false;
    if !keep_gathering {
        crate::disable_gathering();
    }
    // SAFETY: fixed aligned mailbox locations inside L1.
    unsafe { l1_write32(mb.status(), status::RUNNING) };
    publish();
    // SAFETY: as above; written by the host before this core left reset.
    let mut generation = unsafe { l1_read32(mb.generation()) };
    loop {
        let program_len = run_once::<Riscv, Thread>(mb, translate_programs);
        if generation == 0 {
            // The dump is what carries the result; this only says it is complete.
            if mb == Mailbox::single_core() {
                finish(program_len);
            } else {
                // SAFETY: fixed aligned mailbox location inside L1.
                unsafe { l1_write32(mb.status(), status::DONE) };
                publish();
            }
            spin()
        }
        // Resident (`mailbox::GENERATION`): acknowledge, then wait for the next
        // one. The host writes the whole descriptor and program before the new
        // generation, and the poll goes through a fence, since nothing the NoC
        // writes invalidates the L0 data cache (`MemoryOrdering.md:59`).
        // SAFETY: fixed aligned mailbox locations inside L1.
        let tracing = || unsafe { l1_read32(mb.trace()) } != 0;
        trace(tracing(), Thread::INDEX, mailbox::trace::ACKED);
        // SAFETY: fixed aligned mailbox locations inside L1.
        unsafe {
            l1_write32(mb.status(), status::DONE);
            l1_write32(mb.ack(), generation);
        }
        publish();
        loop {
            publish();
            // SAFETY: as above.
            let next = unsafe { l1_read32(mb.generation()) };
            if next != 0 && next != generation {
                generation = next;
                break;
            }
        }
        trace(tracing(), Thread::INDEX, mailbox::trace::WOKE);
        // SAFETY: as above.
        unsafe { l1_write32(mb.status(), status::RUNNING) };
        publish();
    }
}

/// One run of the staged program: read the descriptor, push the program, wait
/// for it to retire, dump `Dst`. Returns the program's length.
fn run_once<Riscv, Thread>(mb: Mailbox, translate_programs: bool) -> u32
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    // SAFETY: fixed, aligned mailbox locations written by the host before this core
    // left reset (or, resident, before it wrote the new generation).
    let thread = unsafe { l1_read32(mb.thread_index()) };
    let fmt = unsafe { l1_read32(mb.dst_access_fmt()) };
    // A loop header leads the program where its length says so
    // (`mailbox::loops::LOOPED`).
    let program_len_word = unsafe { l1_read32(mb.program_len()) };
    let streamed = program_len_word & tt_isa::dataflow::STREAMED != 0;
    let looped = program_len_word & mailbox::loops::LOOPED != 0;
    let program_len = program_len_word & tt_isa::dataflow::LENGTH_MASK;
    let dump_first = unsafe { l1_read32(mb.dump_row_first()) };
    let dump_rows = unsafe { l1_read32(mb.dump_row_count()) };
    let tracing = unsafe { l1_read32(mb.trace()) } != 0;
    let mut push_window = unsafe { l1_read32(mb.push_window()) };
    let program_addr = unsafe { l1_read32(mb.program_addr()) } as u64;
    // SAFETY: as above.
    if unsafe { l1_read32(mb.unwedge()) } != 0 {
        unwedge();
    }

    // A guarded run (`guard_armed`): the program is short enough that its pushes
    // cannot fill a blocked thread's FIFO, is neither streamed, looped nor
    // dumping, and carries its own bounds. Checked before anything is pushed;
    // the window drain is off because it is a coprocessor-drain load too.
    let guarded = guard_armed(mb);
    let guard_run = if guarded {
        push_window = 0;
        Some(guard_begin(mb, streamed, looped, dump_rows, program_len))
    } else {
        None
    };

    // Bounds are checked here rather than trusted, because a runaway length would
    // push whatever happens to be in L1 into the coprocessor.
    // `thread` is a cross-check rather than a choice: which core this is, is fixed
    // at build time by `Riscv`/`Thread` above. A mismatch means the host started
    // this image somewhere it did not intend, which would otherwise show up as a
    // wrong result from a correctly-executed program.
    if thread != Thread::INDEX
        || program_len > mailbox::PROGRAM_MAX
        || dump_rows > mailbox::DUMP_MAX_ROWS
    {
        fail_in(mb, panic_code::EXPLICIT);
    }
    // A resident program (`mailbox::PROGRAM_ADDR`) must lie in the program
    // cache, so a stale or corrupt address cannot push arbitrary L1.
    let program = if program_addr == 0 {
        mb.program()
    } else {
        if program_addr % 16 != 0
            || !tt_isa::l1::PROGRAM_CACHE.contains(program_addr, program_len as u64 * 4)
        {
            fail_in(mb, panic_code::EXPLICIT);
        }
        program_addr
    };

    // Set the shape of the Dst mapping deliberately rather than inheriting whatever
    // reset left behind, exactly as the step 4 firmware does.
    //
    // SAFETY: the host released the Tensix backend before releasing this core, and
    // nothing has been pushed yet, so no Tensix instruction is in flight.
    unsafe {
        // Bank 0, not `cfg::active_bank(thread)`: ttsim maps the whole configuration
        // aperture to a flat `Config` array with the bank hardcoded to zero and no
        // `ThreadConfig` region, so reading the state ID there is fatal. Nothing
        // here changes it and its reset value is 0.
        let field = Thread::DST_ACCESS_FMT;
        if !field.fits(fmt) {
            fail_in(mb, panic_code::EXPLICIT);
        }
        write_config_field(field, ConfigBank::Bank0, fmt);
    }
    // Configuration is written by RISC-V but read by the coprocessor. Auto TTSync
    // covers the push-then-load direction, not this one, so drain the store queue
    // before pushing anything that depends on the new value.
    publish();

    // This run's MOP Expander configuration (`tt_isa::frontend::mop`), loaded
    // once the expander is idle and before anything is pushed that could use
    // it.
    // SAFETY: fixed mailbox words, written by the host before the generation.
    if unsafe { l1_read32(mb.mop_cfg_valid()) } != 0 {
        let mut cfg = [0u32; 9];
        for (k, w) in cfg.iter_mut().enumerate() {
            // SAFETY: as above.
            *w = unsafe { l1_read32(mb.mop_cfg(k as u32)) };
        }
        load_mop_config(&cfg);
    }

    if streamed {
        if looped || dump_rows != 0 {
            fail_in(mb, panic_code::EXPLICIT);
        }
        trace(tracing, Thread::INDEX, mailbox::trace::START);
        if let Err(code) = run_stream::<Riscv, Thread>(program, program_len, push_window, tracing) {
            crate::dataflow::abort_with(code);
            fail_in(mb, code);
        }
        trace(tracing, Thread::INDEX, mailbox::trace::RETIRED);
        return program_len;
    }

    // The block repeats (`mailbox::loops`): each inside the code, and any two
    // disjoint or one inside the other, at most two deep -- checked, since a
    // bad table would push whatever L1 holds.
    let (code, code_len, n) = if looped {
        // SAFETY: the program's first word, inside its checked extent.
        let n = if program_len == 0 {
            u32::MAX
        } else {
            unsafe { l1_read32(program) }
        };
        if n as usize > mailbox::loops::MAX || n + 1 > program_len {
            fail_in(mb, panic_code::EXPLICIT);
        }
        (
            program + 4 * (1 + n as u64),
            program_len - 1 - n,
            n as usize,
        )
    } else {
        (program, program_len, 0)
    };
    let mut loops = [(0u32, 0u32, 0u32); mailbox::loops::MAX];
    for (k, l) in loops.iter_mut().enumerate().take(n) {
        // SAFETY: inside the header the bound above checked.
        *l = mailbox::loops::decode(unsafe { l1_read32(program + 4 * (1 + k as u64)) });
        if l.0 + l.1 > code_len {
            fail_in(mb, panic_code::EXPLICIT);
        }
    }
    for a in 0..n {
        let mut depth = 0;
        for b in 0..n {
            let (x, y) = (loops[a], loops[b]);
            let (xe, ye) = (x.0 + x.1, y.0 + y.1);
            let disjoint = xe <= y.0 || ye <= x.0;
            let inside = y.0 <= x.0 && xe <= ye && (x.0, x.1) != (y.0, y.1);
            let around = x.0 <= y.0 && ye <= xe && (x.0, x.1) != (y.0, y.1);
            if a != b && !(disjoint || inside || around) {
                fail_in(mb, panic_code::EXPLICIT);
            }
            if a != b && inside {
                depth += 1;
            }
        }
        if depth > 1 {
            fail_in(mb, panic_code::EXPLICIT);
        }
    }

    trace(tracing, Thread::INDEX, mailbox::trace::START);
    // The `.ttinsn` experiment's only change to this path: the image may push a
    // program it was built with instead of reading it from L1
    // (`experiment::try_push`). Without the feature this is the line it was.
    #[cfg(feature = "ttinsn-experiment")]
    let fused = experiment::try_push(code, code_len, &loops[..n]);
    #[cfg(not(feature = "ttinsn-experiment"))]
    let fused = false;
    // A role's program with block repeats is translated into looping code when
    // the exchange pays: the loops become branches instead of the body written
    // out, which is what keeps a long expanded program out of the push stream.
    let fused = fused || (translate_programs && translate::<Thread>(code, code_len, &loops[..n]));
    if !fused {
        let mut push = Pusher::<Riscv, Thread> {
            program: code,
            push_window,
            until_drain: push_window,
            _p: core::marker::PhantomData,
        };
        push.span(push.start(code_len), code_len, &loops[..n]);
    }

    trace(tracing, Thread::INDEX, mailbox::trace::PUSHED);

    // The pushes above have only reached a FIFO. Wait for them to retire before
    // looking at Dst.
    if let Some(run) = &guard_run {
        crumb(Guard::of(mb), guard::stage::PUSHED);
        guard_wait(mb, run);
    } else {
        wait_for_coprocessor();
    }
    trace(tracing, Thread::INDEX, mailbox::trace::RETIRED);
    #[cfg(feature = "ttinsn-experiment")]
    experiment::after_retire();

    let mut row = 0;
    while row < dump_rows {
        let mut column = 0;
        while column < mailbox::DUMP_ROW_WORDS {
            // SAFETY: the coprocessor has retired the program, fmt is a 32-bit
            // shape, and the destination is inside the dump region.
            unsafe {
                let value = read_dst32(dst32_address(dump_first + row, column));
                l1_write32(mb.dump_offset(row, column), value);
            }
            column += 1;
        }
        row += 1;
    }

    program_len
}

fn push_body<Riscv, Thread>(
    address: u32,
    length: u32,
    push_window: u32,
    last_traced: bool,
) -> Result<(), u32>
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    use tt_isa::{dataflow, dm};
    if address == 0 && length == 0 {
        trace(last_traced, Thread::INDEX, mailbox::trace::PUSHED);
        return Ok(());
    }
    if !dataflow::program(address, length) {
        return Err(dm::error::PROGRAM);
    }
    let words = length & !mailbox::loops::LOOPED;
    let read = |offset| unsafe { l1_read32(address as u64 + offset * 4) };
    let count = if length & mailbox::loops::LOOPED != 0 {
        read(0)
    } else {
        0
    };
    if count as usize > mailbox::loops::MAX || (count != 0 && count + 1 > words) {
        return Err(dm::error::PROGRAM);
    }
    let prefix = if length & mailbox::loops::LOOPED != 0 {
        count + 1
    } else {
        0
    };
    if prefix > words {
        return Err(dm::error::PROGRAM);
    }
    let code_len = words - prefix;
    let mut loops = [(0u32, 0u32, 0u32); mailbox::loops::MAX];
    for (index, entry) in loops.iter_mut().enumerate().take(count as usize) {
        *entry = mailbox::loops::decode(read(1 + index as u64));
        if entry.0 + entry.1 > code_len {
            return Err(dm::error::PROGRAM);
        }
    }
    for (index, current) in loops[..count as usize].iter().enumerate() {
        let mut depth = 0;
        for (other_index, other) in loops[..count as usize].iter().enumerate() {
            if index == other_index {
                continue;
            }
            let end = current.0 + current.1;
            let other_end = other.0 + other.1;
            let inside = other.0 <= current.0
                && end <= other_end
                && (current.0, end) != (other.0, other_end);
            let outside = current.0 <= other.0
                && other_end <= end
                && (current.0, end) != (other.0, other_end);
            if !(end <= other.0 || other_end <= current.0 || inside || outside) {
                return Err(dm::error::PROGRAM);
            }
            if inside {
                depth += 1;
            }
        }
        if depth > 1 {
            return Err(dm::error::PROGRAM);
        }
    }
    let mut push = Pusher::<Riscv, Thread> {
        program: address as u64 + prefix as u64 * 4,
        push_window,
        until_drain: push_window,
        _p: core::marker::PhantomData,
    };
    push.span(push.start(code_len), code_len, &loops[..count as usize]);
    trace(last_traced, Thread::INDEX, mailbox::trace::PUSHED);
    wait_for_coprocessor();
    Ok(())
}

fn run_stream<Riscv, Thread>(
    program: u64,
    words: u32,
    push_window: u32,
    tracing: bool,
) -> Result<(), u32>
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    use dataflow::{Action, Endpoint, Stream};
    use tt_isa::{dataflow, dm};
    let read = |index| unsafe { l1_read32(program + index * 4) };
    if words < 4 || read(0) != dataflow::VERSION || read(3) != Thread::INDEX {
        return Err(dm::error::PROGRAM);
    }
    let count = read(1);
    let capacity = read(2);
    if count == 0
        || count > (words - 4) / dataflow::STEP_WORDS
        || words != 4 + count * dataflow::STEP_WORDS
        || capacity == 0
        || capacity >= 0x8000
    {
        return Err(dm::error::PROGRAM);
    }
    for index in 0..count {
        let offset = 4 + index as u64 * dataflow::STEP_WORDS as u64;
        let (address, length) = (read(offset), read(offset + 1));
        if !(address == 0 && length == 0 || dataflow::program(address, length))
            || read(offset + 2) != 0
            || read(offset + 3) != 0
        {
            return Err(dm::error::PROGRAM);
        }
    }
    let capacity = capacity as u16;
    for index in 0..count {
        crate::dataflow::check()?;
        if Thread::INDEX == 0 {
            crate::dataflow::buffer(Stream::Input, Action::Wait, capacity, Endpoint::Unpack, 1)?;
        }
        if Thread::INDEX == 2 {
            crate::dataflow::buffer(Stream::Output, Action::Reserve, capacity, Endpoint::Pack, 3)?;
        }
        let offset = 4 + index as u64 * dataflow::STEP_WORDS as u64;
        push_body::<Riscv, Thread>(
            read(offset),
            read(offset + 1),
            push_window,
            tracing && index + 1 == count,
        )?;
        if Thread::INDEX == 0 {
            crate::dataflow::buffer(Stream::Input, Action::Pop, capacity, Endpoint::Unpack, 1)?;
        }
        if Thread::INDEX == 2 {
            crate::dataflow::buffer(Stream::Output, Action::Push, capacity, Endpoint::Pack, 3)?;
        }
        // The roles share configuration and semaphores, so each batch starts
        // only once all three have retired the one before. After the last,
        // nothing follows in this region: the mover's `KERNEL_WAIT` joins.
        if index + 1 < count {
            crate::dataflow::retire_batch(Thread::INDEX, index + 1)?;
        } else {
            crate::dataflow::record_batch(Thread::INDEX, count);
        }
    }
    Ok(())
}

/// Pushes ranges of a staged program, its block repeats expanded.
struct Pusher<Riscv, Thread> {
    program: u64,
    push_window: u32,
    /// A countdown rather than `i % push_window`: T2 has no remainder
    /// instruction, and the instruction-set gate refuses one.
    until_drain: u32,
    _p: core::marker::PhantomData<(Riscv, Thread)>,
}

impl<Riscv, Thread> Pusher<Riscv, Thread>
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    /// Honour a seed directive (`tt_isa::dataflow::seed_directive`)
    /// at the head of the program, and return the first word to push (past it).
    /// The coprocessor is drained, the seed goes to the PRNG seed register by a
    /// full-width RISC-V store, a fence publishes it, and the settling interval
    /// elapses before anything that reads the generator is pushed: the restart
    /// procedure `step91_seeded_prng` validates on both cards (a `WRCFG` seed
    /// write does not restart silicon's stream).
    fn start(&self, code_len: u32) -> u32 {
        use tt_isa::dataflow::{SEED_DIRECTIVE, SEED_SETTLE_NOPS, SEED_WORDS};
        // SAFETY: the first words of the staged program, whose length the
        // caller checked.
        if code_len < SEED_WORDS || unsafe { l1_read32(self.program) } != SEED_DIRECTIVE {
            return 0;
        }
        // SAFETY: as above.
        let seed = unsafe { l1_read32(self.program + 4) };
        wait_for_coprocessor();
        // SAFETY: the coprocessor is drained; only the generated full-width
        // seed field is written, and the fence follows before anything
        // dependent on it is pushed.
        unsafe {
            core::ptr::write_volatile(
                tt_isa::cfg::generated::global::PRNG_SEED_Seed_Val.riscv_address(ConfigBank::Bank0)
                    as *mut u32,
                seed,
            );
        }
        publish();
        for _ in 0..SEED_SETTLE_NOPS {
            // SAFETY: an ordinary RISC-V NOP, which the instruction gate allows.
            unsafe { core::arch::asm!("nop", options(nomem, nostack)) };
        }
        SEED_WORDS
    }

    /// Words `[lo, hi)`, each block repeat inside them its `count` times --
    /// the next one being the first, not yet passed, that lies inside and is
    /// not the span itself; the ones inside it its own call's.
    fn span(&mut self, lo: u32, hi: u32, loops: &[(u32, u32, u32)]) {
        let mut at = lo;
        loop {
            let mut next: Option<(u32, u32, u32)> = None;
            for &l in loops {
                let inside = l.0 >= at && l.0 + l.1 <= hi && (l.0, l.0 + l.1) != (lo, hi);
                if inside && next.is_none_or(|n| l.0 < n.0 || (l.0 == n.0 && l.1 > n.1)) {
                    next = Some(l);
                }
            }
            let Some((start, len, count)) = next else {
                self.linear(at, hi);
                return;
            };
            self.linear(at, start);
            for _ in 0..count {
                self.span(start, start + len, loops);
            }
            at = start + len;
        }
    }

    /// Words `[lo, hi)` as they are.
    fn linear(&mut self, lo: u32, hi: u32) {
        let mut i = lo;
        // Silicon (no push window): sixteen words read, then sixteen pushed, so
        // the loads overlap rather than each push waiting on its own load. One
        // word at a time took ~7.6 cycles a word, eight at a time 3.5, sixteen
        // 2.8 (`silicon_perf::role_push_rate`, card 0) -- and a matmul's unpack
        // and math roles push as fast as their runner can. Sixteen is what fits
        // the registers.
        if self.push_window == 0 {
            while i + 16 <= hi {
                let at = self.program + (i as u64) * 4;
                // SAFETY: sixteen words inside the staged program, whose length
                // was checked by the caller.
                let w: [u32; 16] =
                    core::array::from_fn(|k| unsafe { l1_read32(at + 4 * k as u64) });
                for word in w {
                    // SAFETY: as the loop below.
                    unsafe { push_word::<Riscv, Thread>(word) }
                }
                i += 16;
            }
        }
        while i < hi {
            // SAFETY: the word is inside the staged program, whose length was
            // checked; `Riscv` may push to `Thread`, which the type system
            // checked; the backend is out of reset.
            unsafe { push_word::<Riscv, Thread>(l1_read32(self.program + (i as u64) * 4)) }
            i += 1;
            // Flow control for the simulator (`mailbox::PUSH_WINDOW`): silicon
            // stalls a push into a full FIFO, ttsim kills the process.
            if self.push_window != 0 {
                self.until_drain -= 1;
                if self.until_drain == 0 {
                    wait_for_coprocessor();
                    self.until_drain = self.push_window;
                }
            }
        }
    }
}

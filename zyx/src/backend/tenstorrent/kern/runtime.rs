//! Running a three-role kernel on one Tensix tile.
//!
//! Stage the operands, stage each role's program in its slot
//! (`tt_isa::mailbox::PROGRAM_REGION`), start the three role images, wait for
//! all three to report `DONE`, and read back what was asked for.
//!
//! # Two schedules
//!
//! [`Schedule::InOrder`] runs unpack, then math, then pack, each to completion.
//! It needs no synchronisation and is enough for a kernel whose unpacker never
//! gets more than the two `Src` banks ahead.
//!
//! [`Schedule::Concurrent`] is what everything bigger needs. A setup program on
//! thread 0 zeroes the `Dst` rows that will be dumped and initialises the
//! kernel's semaphores; *then* all three roles are released by one write to
//! `SOFT_RESET_0` (`Device::load_and_start_together`). Both halves of that are
//! load-bearing:
//!
//! * A semaphore outlives the program that last used it on silicon, so a
//!   consumer released alongside its own `SEMINIT` could read a stale value
//!   first.
//! * Released one by one, each after loading its image, the first roles of a
//!   short kernel finish before the last has started -- a sequential run by
//!   another name, which hid a missing hand-off on silicon that ttsim showed at
//!   once (divergence row 53).

use std::time::{Duration, Instant};
use super::super::dev::core_control::WaitError;
use super::super::dev::tlb::WindowKind;
use super::super::dev::trace::TraceEvent;

use super::super::dev::{Device, Traffic, Transport, TransportError};
use tt_isa::backend::{self, Before};
use tt_isa::isa::Instruction;
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{NocCoord, NocId};
use tt_isa::sfpu;
use tt_isa::sync::Semaphore;
use tt_isa::tensix::Core;

/// A semaphore's starting `Value` and `Max`, set before a concurrent run.
pub type SemaphoreInit = (Semaphore, u8, u8);

/// The three role firmware images, as `(core, image, load address)` for
/// threads 0, 1 and 2: unpack on T0, math on T1, pack on T2, each linked at its
/// core's default reset PC.
pub type RoleImages<'a> = [(Core, &'a [u8], u64); 3];

/// How the three roles are run; see the module documentation.
#[derive(Copy, Clone, Debug)]
pub enum Schedule<'a> {
    InOrder,
    Concurrent(&'a [SemaphoreInit]),
}

/// Pre-filled into every dumped `Dst` datum, so a datum the firmware never
/// wrote is distinguishable from a computed zero.
pub const DUMP_SENTINEL: u32 = 0xDEAD_BEEF;

/// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` 0: `float Dst32b[512][16]`.
pub const DST_FMT_FP32: u32 = 0;

/// A kernel, and what to stage and collect around it.
#[derive(Copy, Clone, Debug)]
pub struct Kernel<'a> {
    /// The programs of threads 0, 1 and 2.
    pub roles: [&'a [Instruction]; 3],
    pub schedule: Schedule<'a>,
    /// Bytes written to L1 first, as `(address, data)`.
    pub stage: &'a [(u64, &'a [u8])],
    /// L1 ranges read afterwards, as `(address, length)`.
    pub read_back: &'a [(u64, usize)],
    /// `Dst` rows the math role copies out through its mailbox, and which are
    /// zeroed before anything runs. At most `mailbox::DUMP_MAX_ROWS`.
    pub dump_rows: u32,
    /// Zero the dumped rows first. `Dst` has no reset value and survives
    /// everything the host can do to a tile from outside.
    pub clear_dst: bool,
    /// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` for the firmware's `Dst` view.
    pub dst_fmt: u32,
    /// Record each role's `START`, `PUSHED` and `RETIRED` through the tile's
    /// timestamper. Silicon only: ttsim does not model the event stream
    /// (divergence row 54).
    pub trace: bool,
    /// A completed run leaves every semaphore of its [`Schedule::Concurrent`]
    /// set as the set initialises it: every post is matched by a take. A
    /// [`Resident`] then skips the setup run of the next kernel with the same
    /// set (`matmul::MatmulSemaphores` says why a matmul qualifies).
    pub restores_semaphores: bool,
    /// Have each role release blocked semaphores before pushing
    /// (`mailbox::UNWEDGE`): a tile reset on silicon.
    pub unwedge: bool,
    /// Each role's MOP Expander configuration for this run, loaded by its
    /// runner before the program is pushed (`tt_isa::frontend::mop`,
    /// `mailbox::MOP_CFG`). `None` leaves the expander as it is.
    pub mop: [Option<tt_isa::frontend::mop::MopConfig>; 3],
    /// Each role's block repeats over its program (`super::code`,
    /// `mailbox::LOOPS`): the runner pushes each range `count` times.
    pub loops: [&'a [super::code::Loop]; 3],
}

impl<'a> Kernel<'a> {
    /// A kernel with nothing staged, nothing read back and nothing dumped.
    pub fn new(roles: [&'a [Instruction]; 3], schedule: Schedule<'a>) -> Self {
        Kernel {
            roles,
            schedule,
            stage: &[],
            read_back: &[],
            dump_rows: 0,
            clear_dst: true,
            dst_fmt: DST_FMT_FP32,
            trace: false,
            restores_semaphores: false,
            unwedge: false,
            mop: [None; 3],
            loops: [&[]; 3],
        }
    }

    /// Role `thread`'s `program` -- its own program, or that with a prefix the
    /// runtime put in front (`assemble`'s `Dst` clear) -- as its slot stores
    /// it, its block repeats moved past the prefix and led by their header
    /// (`super::code::Code::stored`), and the length word that names it.
    fn stored_program(
        &self,
        thread: usize,
        program: &[Instruction],
    ) -> Result<(Vec<u32>, u32), RunError> {
        // A resident list's descriptor, staged without its programs (their
        // `KERNEL` entries name them, length word and all).
        if program.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let prefix = program.len().saturating_sub(self.roles[thread].len()) as u32;
        let code = super::code::Code {
            ins: program.to_vec(),
            loops: self.loops[thread]
                .iter()
                .map(|l| super::code::Loop {
                    start: l.start + prefix,
                    ..*l
                })
                .collect(),
        };
        let (words, len) = code
            .stored()
            .map_err(|e| RunError::Loops { thread, reason: e })?;
        if words.len() > mailbox::PROGRAM_MAX as usize {
            return Err(RunError::ProgramTooLong {
                thread,
                len: words.len(),
            });
        }
        Ok((words, len))
    }

    /// Role `thread`'s MOP words, checked: a configuration the expander cannot
    /// run as the page describes it is refused before anything is staged.
    fn mop_words(&self, thread: usize) -> Result<Option<[u32; 9]>, RunError> {
        self.mop[thread]
            .map(|c| c.config_words())
            .transpose()
            .map_err(|e| RunError::Mop { thread, reason: e })
    }
}

/// What a run produced.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// The dumped `Dst` rows, flattened as `row * 16 + column`.
    pub dst: Vec<u32>,
    /// One buffer per [`Kernel::read_back`] range, in order.
    pub l1: Vec<Vec<u8>>,
    /// The timestamper events of a traced run, in order.
    pub trace: Vec<TraceEvent>,
    /// Where the run's host time and transport traffic went.
    pub profile: Profile,
}

/// The phases of one [`run`], each with its wall-clock time and what it moved
/// across the transport. Always collected: it costs one `Instant` per phase.
///
/// Host time only. On ttsim it measures the simulator, not a chip, and is no
/// performance number (divergence row 11); on silicon `Wait` is the one phase
/// the device, rather than PCIe, can dominate.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub phases: Vec<(Phase, Duration, Traffic)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Backend release and the kernel's L1 staging (operands).
    Stage,
    /// The semaphore-initialising setup program, loaded, run and waited for.
    Setup,
    /// Mailbox words and role programs written to their slots.
    Programs,
    /// Firmware images loaded and the role cores released.
    Launch,
    /// Polling the role mailboxes for `DONE`.
    Wait,
    /// `Dst` dump, trace and L1 read-back.
    ReadBack,
}

impl Profile {
    /// Total time and traffic of every phase named `phase`.
    pub fn of(&self, phase: Phase) -> (Duration, Traffic) {
        self.phases
            .iter()
            .filter(|p| p.0 == phase)
            .fold((Duration::ZERO, Traffic::default()), |(d, t), p| {
                (d + p.1, t + p.2)
            })
    }
}

/// Records consecutive phases against one device.
struct Stopwatch {
    at: Instant,
    traffic: Traffic,
    profile: Profile,
}

impl Stopwatch {
    fn start<T: Transport>(dev: &Device<T>) -> Self {
        Stopwatch {
            at: Instant::now(),
            traffic: dev.traffic(),
            profile: Profile::default(),
        }
    }

    /// Close the phase that has been running since the last lap.
    fn lap<T: Transport>(&mut self, dev: &Device<T>, phase: Phase) {
        let (now, traffic) = (Instant::now(), dev.traffic());
        self.profile
            .phases
            .push((phase, now - self.at, traffic - self.traffic));
        (self.at, self.traffic) = (now, traffic);
    }
}

impl Outcome {
    /// `Dst[row][column]`, by the firmware's dump indexing.
    pub fn dst_at(&self, row: usize, column: usize) -> u32 {
        self.dst[row * mailbox::DUMP_ROW_WORDS as usize + column]
    }
}

/// Why a run failed.
#[derive(Debug)]
pub enum RunError {
    Transport(TransportError),
    /// A role's program does not fit its slot.
    ProgramTooLong {
        thread: usize,
        len: usize,
    },
    /// More `Dst` rows asked for than the mailbox dump holds.
    TooManyDumpRows {
        rows: u32,
    },
    /// The concurrent schedule's setup program did not finish.
    Setup(WaitError),
    /// Roles that did not finish, as `(thread, core, why)`. The cores are held
    /// in reset again before this is returned.
    Roles(Vec<(usize, Core, WaitError)>),
    /// Something staged in L1 does not fit the region reserved for it.
    DoesNotFit {
        what: &'static str,
        bytes: u64,
        limit: u64,
    },
    /// Work queued earlier failed when the session waited for it
    /// (`crate::session::Session::sync`).
    Queued(String),
    /// Role `thread`'s MOP Expander configuration was refused.
    Mop {
        thread: usize,
        reason: tt_isa::frontend::mop::MopError,
    },
    /// Role `thread`'s block repeats were refused (`super::code::check`).
    Loops {
        thread: usize,
        reason: super::code::LoopError,
    },
    /// The tile's reset never finished: these roles' threads take no
    /// instruction even after the backend pulse, every semaphore released and
    /// their `Src` banks fed (`session::unwedge_tile`). What else holds them is
    /// unknown; a board reset clears it (`docs/completed-plans/hardware-coverage.md`, "Hazards
    /// and known bugs").
    Wedged {
        tile: (u8, u8),
        roles: Vec<Core>,
    },
}

impl From<TransportError> for RunError {
    fn from(e: TransportError) -> Self {
        RunError::Transport(e)
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Transport(e) => write!(f, "{e}"),
            RunError::ProgramTooLong { thread, len } => write!(
                f,
                "role {thread} program is {len} instructions; a program slot holds {}",
                mailbox::PROGRAM_MAX
            ),
            RunError::TooManyDumpRows { rows } => write!(
                f,
                "{rows} Dst rows asked for; the dump holds {}",
                mailbox::DUMP_MAX_ROWS
            ),
            RunError::Setup(e) => write!(f, "setup run: {e}"),
            RunError::Queued(e) => write!(f, "{e}"),
            RunError::Mop { thread, reason } => {
                write!(f, "role {thread}'s MOP configuration: {reason}")
            }
            RunError::Loops { thread, reason } => {
                write!(f, "role {thread}'s program: {reason}")
            }
            RunError::DoesNotFit { what, bytes, limit } => write!(
                f,
                "{what} need {bytes} bytes of L1; the region holds {limit}"
            ),
            RunError::Wedged {
                tile: (x, y),
                roles,
            } => {
                let names: Vec<&str> = roles.iter().map(|c| c.name()).collect();
                write!(
                    f,
                    "tile ({x},{y}) is wedged: {} took no instruction after the reset \
                     released every semaphore, nor after its `Src` banks were fed (on \
                     silicon; ttsim cannot finish that recovery). Software could not \
                     clear it; reset the board (`tt-smi -r`, or a power cycle), or \
                     choose another tile",
                    names.join(", ")
                )
            }
            RunError::Roles(stuck) => {
                let parts: Vec<String> = stuck
                    .iter()
                    .map(|(t, core, e)| format!("role {t} ({}): {e}", core.name()))
                    .collect();
                write!(f, "{}", parts.join("; "))
            }
        }
    }
}

impl std::error::Error for RunError {}

/// Zero `rows` of FP32 `Dst` through the SFPU, whose `Dst` writes read back
/// correctly on both targets, then wait for it.
fn dst_clear(rows: u32) -> Vec<Instruction> {
    let mut p = Vec::new();
    p.extend(sfpu::load_f32(0, 0).expect("LReg 0 is writable"));
    for group in 0..rows.div_ceil(4) {
        for cols in [0, sfpu::DST_ODD_COLUMNS] {
            p.push(
                sfpu::store(0, sfpu::store_format::FP32, 0, (group * 4) | cols)
                    .expect("dump rows are inside Dst"),
            );
        }
    }
    p.push(backend::wait_for_sfpu(Before::EVERYTHING).expect("named bits"));
    p
}

pub(crate) fn word_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The setup program (for a concurrent schedule) and thread 0's program, with
/// the `Dst` clear in front of whichever runs first.
#[allow(clippy::type_complexity)]
fn assemble(kernel: &Kernel<'_>) -> Result<(Option<Vec<Instruction>>, Vec<Instruction>), RunError> {
    if kernel.dump_rows > mailbox::DUMP_MAX_ROWS {
        return Err(RunError::TooManyDumpRows {
            rows: kernel.dump_rows,
        });
    }
    let clear = if kernel.clear_dst && kernel.dump_rows > 0 {
        dst_clear(kernel.dump_rows)
    } else {
        Vec::new()
    };
    let (setup, unpack) = match kernel.schedule {
        Schedule::InOrder => (None, [clear.as_slice(), kernel.roles[0]].concat()),
        Schedule::Concurrent(init) => {
            let mut setup = clear;
            for &(sem, value, max) in init {
                setup.push(tt_isa::sync::init(sem, value, max).map_err(|_| {
                    RunError::Transport(TransportError::Hazard {
                        address: 0,
                        reason: "a semaphore value or max does not fit in four bits",
                    })
                })?);
            }
            (Some(setup), kernel.roles[0].to_vec())
        }
    };
    Ok((setup, unpack))
}

/// Run `kernel` on the Tensix tile at `tile`, waiting up to `budget` simulated
/// cycles for each role (`Device::wait_for_mailbox`; on silicon a floor of one
/// second applies).
///
/// The tile's backend is released first; the three role cores are held in
/// reset again afterwards, success or not.
pub fn run<T: Transport, N: NocId>(
    dev: &mut Device<T>,
    tile: NocCoord<N>,
    images: &RoleImages<'_>,
    kernel: &Kernel<'_>,
    budget: u64,
) -> Result<Outcome, RunError> {
    let (setup, unpack) = assemble(kernel)?;
    let programs: [&[Instruction]; 3] = [&unpack, kernel.roles[1], kernel.roles[2]];
    for (thread, p) in programs.iter().enumerate() {
        if p.len() > mailbox::PROGRAM_MAX as usize {
            return Err(RunError::ProgramTooLong {
                thread,
                len: p.len(),
            });
        }
    }

    let mut clock = Stopwatch::start(dev);
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.release_tensix_backend(&w, tile)?;
    for (addr, data) in kernel.stage {
        dev.write(&w, tile, *addr, data)?;
    }
    clock.lap(dev, Phase::Stage);

    let push_window = if dev.transport().is_simulated() {
        mailbox::SIM_PUSH_WINDOW
    } else {
        0
    };
    let stage_role = |dev: &mut Device<T>,
                      thread: usize,
                      program: &[Instruction],
                      dump: u32,
                      traced: bool|
     -> Result<(), RunError> {
        let mb = Mailbox::of(thread as u32);
        dev.write32(&w, tile, mb.status(), 0)?;
        let (words, len) = kernel.stored_program(thread, program)?;
        let d = mailbox::Descriptor {
            thread_index: thread as u32,
            dst_access_fmt: kernel.dst_fmt,
            program_len: len,
            dump_row_count: dump,
            trace: u32::from(traced),
            push_window,
            unwedge: u32::from(kernel.unwedge),
            mop_cfg: kernel.mop_words(thread)?,
            ..Default::default()
        };
        for (at, v) in d.writes(mb) {
            dev.write32(&w, tile, at, v)?;
        }
        dev.write(&w, tile, mb.program(), &word_bytes(&words))?;
        for row in 0..dump {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                dev.write32(&w, tile, mb.dump_offset(row, col), DUMP_SENTINEL)?;
            }
        }
        Ok(())
    };
    let wait = |dev: &mut Device<T>, thread: usize| -> Result<Result<(), WaitError>, RunError> {
        let mb = Mailbox::of(thread as u32);
        Ok(dev
            .wait_for_mailbox(&w, tile, mb.status(), mb.panic_code(), budget, |s| {
                s == status::DONE
            })?
            .map(|_| ()))
    };
    let hold_all = |dev: &mut Device<T>| -> Result<(), RunError> {
        for (core, _, _) in images.iter() {
            dev.set_core_reset(&w, tile, *core, true)?;
        }
        Ok(())
    };

    if let Some(setup) = &setup {
        stage_role(dev, 0, setup, 0, false)?;
        let (core, image, at) = images[0];
        dev.load_and_start(&w, tile, core, image, at)?;
        if let Err(e) = wait(dev, 0)? {
            hold_all(dev)?;
            return Err(RunError::Setup(e));
        }
        clock.lap(dev, Phase::Setup);
    }
    for (thread, program) in programs.iter().enumerate() {
        let dump = if thread == 1 { kernel.dump_rows } else { 0 };
        stage_role(dev, thread, program, dump, kernel.trace)?;
    }
    if kernel.trace {
        dev.configure_trace(&w, tile, mailbox::TRACE_BUFFER, mailbox::TRACE_BUFFER_BYTES)?;
    }
    clock.lap(dev, Phase::Programs);

    let mut stuck = Vec::new();
    if setup.is_some() {
        // Every role released before any is waited for, by one write.
        dev.load_and_start_together(&w, tile, images)?;
        clock.lap(dev, Phase::Launch);
        for (thread, (core, _, _)) in images.iter().enumerate() {
            if let Err(e) = wait(dev, thread)? {
                stuck.push((thread, *core, e));
            }
        }
    } else {
        for (thread, &(core, image, at)) in images.iter().enumerate() {
            dev.load_and_start(&w, tile, core, image, at)?;
            clock.lap(dev, Phase::Launch);
            let waited = wait(dev, thread)?;
            clock.lap(dev, Phase::Wait);
            if let Err(e) = waited {
                stuck.push((thread, core, e));
                break;
            }
        }
    }
    if setup.is_some() {
        clock.lap(dev, Phase::Wait);
    }
    // Leave the cores the way they were found: held. They spin after `DONE`,
    // and whatever runs next may depend on a core being in reset.
    hold_all(dev)?;
    if !stuck.is_empty() {
        return Err(RunError::Roles(stuck));
    }

    let math = Mailbox::of(1);
    let mut dst = Vec::with_capacity((kernel.dump_rows * mailbox::DUMP_ROW_WORDS) as usize);
    for row in 0..kernel.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dst.push(dev.read32(&w, tile, math.dump_offset(row, col))?);
        }
    }
    let trace = if kernel.trace {
        dev.read_trace(&w, tile, mailbox::TRACE_BUFFER)?
    } else {
        Vec::new()
    };
    let mut l1 = Vec::with_capacity(kernel.read_back.len());
    for (addr, len) in kernel.read_back {
        let mut buf = vec![0u8; *len];
        dev.read(&w, tile, *addr, &mut buf)?;
        l1.push(buf);
    }
    clock.lap(dev, Phase::ReadBack);
    Ok(Outcome {
        dst,
        l1,
        trace,
        profile: clock.profile,
    })
}

/// The three role images, loaded once and left running between kernels.
///
/// [`run`] loads three images and releases three cores for every kernel; on
/// silicon that is most of a short kernel's cost, and a matmul is many short
/// kernels. A `Resident` does it once: [`Resident::start`] loads the images with
/// a non-zero `mailbox::GENERATION`, which makes each runner wait for the next
/// generation instead of stopping, and [`Resident::run`] then costs only the
/// descriptor words, the programs and the operands.
///
/// Leaves state between kernels exactly as consecutive kernels in one program
/// would: every kernel this crate builds sets the configuration it depends on
/// (`matmul`'s preludes, the datapath's `*_config` programs), so nothing it
/// reads is inherited. A kernel that fails leaves the cores in an unknown
/// state; `run` refuses to go on after one, and the caller resets the tile and
/// starts again.
pub struct Resident<N: NocId> {
    tile: NocCoord<N>,
    window: super::super::dev::Window,
    generation: u32,
    poisoned: bool,
    /// What each role's program slot holds, as last written: a slot already
    /// holding the program is not written again. Consecutive chunks of one
    /// matmul usually share their programs, which were most of what a
    /// DRAM-resident matmul still sent over PCIe.
    /// Held as encoded words: comparing `Instruction`s compares their
    /// definitions too, which cost more than the PCIe writes it saved
    /// (`MEASURED`: 14 us per descriptor write, on silicon).
    slots: std::cell::RefCell<[Vec<u32>; 3]>,
    /// The semaphores the tile is known to hold, as `(semaphore, value, max)`:
    /// set by a setup run, kept by kernels that restore theirs, dropped by
    /// anything else. A kernel touches only the semaphores it declares, so the
    /// others' entries stay true across it -- which is what lets a matmul and
    /// an SFPU kernel whose initial values agree alternate with no setup run
    /// between them.
    semaphores: Option<Vec<SemaphoreInit>>,
    /// What [`Resident::semaphores`] becomes once the kernel staged by
    /// [`Resident::begin`] completes and restores its own.
    semaphores_after: Option<Vec<SemaphoreInit>>,
    /// Each role's descriptor words as last written, but for the program's
    /// address and length (which a `KERNEL` entry rewrites on the tile): a
    /// word that already holds its value is not written again.
    descriptors: std::cell::RefCell<[Option<[u32; mailbox::DESCRIPTOR_WORDS]>; 3]>,
    /// A [`Resident::submit`]ted kernel's phases so far, until it is
    /// [`Resident::complete`]d.
    pending: Option<Stopwatch>,
    /// Every run's roles record their progress through the timestamper, into
    /// a stream the host configured and drains itself (`crate::profile`).
    profiling: bool,
    /// Whether a profile records the roles' events too, or the mover's
    /// alone (`Resident::set_profile_roles`).
    profile_roles: bool,
    /// The last setup run's generation and program, until taken: a trace's
    /// capture records it as a kernel of its own (`crate::trace`).
    last_setup: Option<(u32, Vec<Instruction>)>,
    /// Kernels [`Resident::reserve`]d and not yet closed, oldest first: their
    /// lists may be queued on the mover behind one another.
    reservations: std::collections::VecDeque<Stopwatch>,
}

impl<N: NocId> Resident<N> {
    /// Release the tile's backend, load the three role images and start them,
    /// and wait for each to acknowledge an empty first generation.
    pub fn start<T: Transport>(
        dev: &mut Device<T>,
        tile: NocCoord<N>,
        images: &RoleImages<'_>,
        budget: u64,
    ) -> Result<Self, RunError> {
        let window = dev.alloc_window(WindowKind::TwoMib)?;
        dev.release_tensix_backend(&window, tile)?;
        let r = Resident {
            tile,
            window,
            generation: 1,
            poisoned: false,
            slots: Default::default(),
            semaphores: None,
            semaphores_after: None,
            last_setup: None,
            descriptors: Default::default(),
            pending: None,
            profiling: false,
            profile_roles: true,
            reservations: Default::default(),
        };
        for thread in 0..3 {
            r.stage(
                dev,
                thread,
                0,
                false,
                DST_FMT_FP32,
                false,
                false,
                None,
                (Vec::new(), 0),
            )?;
            dev.write32(&r.window, tile, Mailbox::of(thread as u32).generation(), 1)?;
        }
        dev.load_and_start_together(&r.window, tile, images)?;
        let stuck = r.wait(dev, &[0, 1, 2], images, budget)?;
        if !stuck.is_empty() {
            r.hold(dev, images)?;
            return Err(RunError::Roles(stuck));
        }
        Ok(r)
    }

    pub fn tile(&self) -> NocCoord<N> {
        self.tile
    }

    /// Have every later run's roles record their progress through the tile's
    /// timestamper, whose stream the caller has configured and will read
    /// (`crate::profile`). A kernel that asks for its own trace
    /// (`Kernel::trace`) reconfigures the stream, so it is refused meanwhile.
    pub fn set_profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    /// Whether a profile's runs record the roles' events as well as the
    /// mover's (the default). Off, a profile is the mover's alone: with the
    /// mover moving while the roles compute (checklist 9.15) the two store to
    /// the timestamper at once, and on card 0 such streams lost events.
    pub fn set_profile_roles(&mut self, on: bool) {
        self.profile_roles = on;
    }

    fn roles_traced(&self) -> bool {
        self.profiling && self.profile_roles
    }

    /// The window this tile is reached through. A [`crate::dm::DataMover`] on
    /// the same tile shares it, so a session over every tile of a chip needs
    /// one window per tile, not two.
    pub fn window(&self) -> &super::super::dev::Window {
        &self.window
    }

    /// Run `kernel` on the resident roles. The same contract as [`run`], less
    /// the image loads and core releases.
    pub fn run<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
        kernel: &Kernel<'_>,
        budget: u64,
    ) -> Result<Outcome, RunError> {
        if let Schedule::Concurrent(_) = kernel.schedule {
            self.submit(dev, images, kernel, budget)?;
            return self.complete(dev, images, kernel, budget);
        }
        let mut clock = self.begin(dev, images, kernel, budget, false)?;
        // In order: each role runs to completion before the next moves,
        // exactly as `run` sequences them.
        let mut stuck = Vec::new();
        for thread in 0..3 {
            self.go(dev, thread)?;
            stuck = self.wait(dev, &[thread], images, budget)?;
            if !stuck.is_empty() {
                break;
            }
        }
        clock.lap(dev, Phase::Wait);
        self.finish(dev, kernel, clock, stuck, None)
    }

    /// The first half of [`Resident::run`] for a [`Schedule::Concurrent`]
    /// kernel: everything up to and including the three generation writes,
    /// with no wait for the roles to finish. [`Resident::complete`] is the
    /// other half, and must be called with the same kernel before anything
    /// else runs here. Between the two, the host is free to start work on
    /// other tiles, which is how a session computes on many at once.
    ///
    /// A setup run, when the kernel needs one, is still waited for here: it
    /// runs only when the tile's semaphores are not already the kernel's.
    pub fn submit<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
        kernel: &Kernel<'_>,
        budget: u64,
    ) -> Result<(), RunError> {
        if !matches!(kernel.schedule, Schedule::Concurrent(_)) {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "only a concurrent kernel can be submitted without waiting",
            }));
        }
        let mut clock = self.begin(dev, images, kernel, budget, false)?;
        for thread in 0..3 {
            self.go(dev, thread)?;
        }
        clock.lap(dev, Phase::Launch);
        self.pending = Some(clock);
        Ok(())
    }

    /// The second half of a [`Resident::submit`]: wait for the three roles,
    /// then read back what `kernel` asks for.
    pub fn complete<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
        kernel: &Kernel<'_>,
        budget: u64,
    ) -> Result<Outcome, RunError> {
        let Some(mut clock) = self.pending.take() else {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "complete without a submitted kernel",
            }));
        };
        let stuck = self.wait(dev, &[0, 1, 2], images, budget)?;
        clock.lap(dev, Phase::Wait);
        let init = match kernel.schedule {
            Schedule::Concurrent(init) => Some(init.to_vec()),
            Schedule::InOrder => None,
        };
        self.finish(dev, kernel, clock, stuck, init)
    }

    /// Stage `kernel` to be run `count` times by the tile's data mover rather
    /// than by the host (`tt_isa::dm::op::KERNEL`), and return the generations
    /// to post, in order: `first..first + count`. Nothing runs until the mover
    /// posts them; [`Resident::reserved_done`] closes the reservation once the
    /// mover's list has finished.
    ///
    /// Only for a kernel that restores its semaphores: the mover runs the
    /// kernel back to back with no setup between runs, which is right only if
    /// every run leaves them as the next expects (`Kernel::restores_semaphores`).
    /// Its setup, if the tile does not already hold them, runs here first.
    ///
    /// With `resident_programs`, `kernel`'s programs are not written to the
    /// fixed slots: every `KERNEL` entry names resident ones itself
    /// (`crate::program_cache`), and the kernels it runs may differ, as long as
    /// they share their semaphores.
    pub fn reserve<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
        kernel: &Kernel<'_>,
        budget: u64,
        count: u32,
        resident_programs: bool,
    ) -> Result<std::ops::Range<u32>, RunError> {
        if count == 0 || !kernel.restores_semaphores || kernel.dump_rows != 0 || kernel.trace {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "only a semaphore-restoring kernel with no dump or trace can run from \
                         the data mover",
            }));
        }
        let descriptors_only = Kernel {
            roles: [&[], &[], &[]],
            ..*kernel
        };
        let staged = if resident_programs {
            &descriptors_only
        } else {
            kernel
        };
        let clock = self.begin(dev, images, staged, budget, resident_programs)?;
        let first = self.generation;
        self.generation += count - 1;
        // What the tile will hold once these runs have finished, assumed now
        // so a kernel queued behind them sees it (`reserved_done` takes it
        // back if they fail).
        self.semaphores = self.semaphores_after.take();
        self.reservations.push_back(clock);
        Ok(first..first + count)
    }

    /// Would reserving `kernel` write anything the roles read while it waits
    /// behind kernels already queued -- a setup run of its semaphores, a fixed
    /// program slot, a descriptor word? Then the caller must let the tile
    /// drain first: those are read by the runs in flight.
    pub fn needs_idle<T: Transport>(
        &self,
        dev: &mut Device<T>,
        kernel: &Kernel<'_>,
        resident_programs: bool,
    ) -> bool {
        if !resident_programs || kernel.trace || kernel.dump_rows != 0 {
            return true;
        }
        if let Schedule::Concurrent(init) = kernel.schedule {
            let holds = self
                .semaphores
                .as_ref()
                .is_some_and(|k| init.iter().all(|e| k.contains(e)));
            if !holds {
                return true;
            }
        }
        let push_window = if dev.transport().is_simulated() {
            mailbox::SIM_PUSH_WINDOW
        } else {
            0
        };
        let descs = self.descriptors.borrow();
        (0..3).any(|thread| {
            let d = mailbox::Descriptor {
                thread_index: thread as u32,
                dst_access_fmt: kernel.dst_fmt,
                trace: u32::from(self.roles_traced()),
                push_window,
                // A MOP configuration different from the queued kernels'
                // would be rewritten under them.
                mop_cfg: kernel.mop_words(thread).ok().flatten(),
                ..Default::default()
            };
            let mb = Mailbox::of(thread as u32);
            let want = d.writes(mb);
            match descs[thread] {
                None => true,
                Some(have) => want.iter().enumerate().any(|(k, &(at, v))| {
                    at != mb.program_len() && at != mb.program_addr() && have[k] != v
                }),
            }
        })
    }

    /// Kernels reserved and not yet closed.
    /// `n` generations no kernel has had, for a trace's replay
    /// (`crate::trace`): the first of them, the next kernel's after the last.
    pub fn take_generations(&mut self, n: u32) -> u32 {
        let first = self.generation.wrapping_add(1);
        self.generation = self.generation.wrapping_add(n);
        first
    }

    /// The setup run [`Resident::reserve`] did last, if any since the last
    /// call: its generation and its thread-0 program.
    pub fn take_setup_run(&mut self) -> Option<(u32, Vec<Instruction>)> {
        self.last_setup.take()
    }

    /// The last generation handed out.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// A replay wrote the roles' descriptors and ran its kernels without the
    /// host (`crate::trace`): what this side remembers of the tile -- the
    /// descriptor words, the semaphores -- is no longer known, so the next
    /// kernel writes and sets up everything.
    pub fn forget_tile_state(&mut self) {
        *self.descriptors.borrow_mut() = [None; 3];
        self.semaphores = None;
        self.semaphores_after = None;
    }

    /// Role `thread`'s descriptor as a resident kernel runs it from a list:
    /// what [`Resident::reserve`] writes, but for the program's address and
    /// length, which the `KERNEL` entry writes on the tile. A trace stores it
    /// as `POKE` entries (`crate::trace`).
    pub fn queued_descriptor<T: Transport>(
        &self,
        dev: &mut Device<T>,
        kernel: &Kernel<'_>,
        thread: usize,
    ) -> Result<mailbox::Descriptor, RunError> {
        let push_window = if dev.transport().is_simulated() {
            mailbox::SIM_PUSH_WINDOW
        } else {
            0
        };
        Ok(mailbox::Descriptor {
            thread_index: thread as u32,
            dst_access_fmt: kernel.dst_fmt,
            trace: u32::from(kernel.trace || self.roles_traced()),
            push_window,
            mop_cfg: kernel.mop_words(thread)?,
            ..Default::default()
        })
    }

    pub fn reserved(&self) -> usize {
        self.reservations.len()
    }

    /// The mover has run every generation [`Resident::reserve`] handed out and
    /// reported each acknowledged; with `Ok(false)` from the mover -- a role
    /// panicked or the list failed -- the tile is poisoned, as a failed
    /// [`Resident::complete`] leaves it.
    pub fn reserved_done<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        ran: bool,
    ) -> Result<(), RunError> {
        let Some(mut clock) = self.reservations.pop_front() else {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "no reserved kernel to close",
            }));
        };
        clock.lap(dev, Phase::Wait);
        if !ran {
            self.poisoned = true;
            self.semaphores = None;
            self.reservations.clear();
        }
        Ok(())
    }

    /// Stage `kernel` and its programs, running its setup first if it needs
    /// one, and move to the next generation; nothing is released yet.
    fn begin<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
        kernel: &Kernel<'_>,
        budget: u64,
        resident_programs: bool,
    ) -> Result<Stopwatch, RunError> {
        if self.poisoned {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "a resident kernel failed; reset the tile and start again",
            }));
        }
        if self.pending.is_some() {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "a submitted kernel has not been completed",
            }));
        }
        // Anything but a reservation that `needs_idle` cleared writes what
        // queued runs read: only on an idle tile.
        if !self.reservations.is_empty() && !resident_programs {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason:
                    "a host run or fixed-slot kernel behind queued kernels; drain the tile first",
            }));
        }
        let (setup, unpack) = assemble(kernel)?;
        let programs: [&[Instruction]; 3] = [&unpack, kernel.roles[1], kernel.roles[2]];
        let mut clock = Stopwatch::start(dev);
        let tile = self.tile;
        for (addr, data) in kernel.stage {
            dev.l1_write(&self.window, tile, *addr, data)?;
        }
        clock.lap(dev, Phase::Stage);

        let init = match kernel.schedule {
            Schedule::Concurrent(init) => Some(init.to_vec()),
            Schedule::InOrder => None,
        };
        // The setup only initialises semaphores (nothing to clear), and they
        // already hold that: nothing to do.
        let clears = kernel.clear_dst && kernel.dump_rows > 0;
        let known = self.semaphores.take();
        let holds = match (&init, &known) {
            (Some(i), Some(k)) => i.iter().all(|e| k.contains(e)),
            _ => false,
        };
        let skip_setup = !clears && holds;
        // After the kernel: its own semaphores as it found them, everyone
        // else's untouched.
        self.semaphores_after = init.as_ref().map(|i| {
            let mut after: Vec<SemaphoreInit> = known
                .unwrap_or_default()
                .into_iter()
                .filter(|(s, ..)| !i.iter().any(|(t, ..)| t == s))
                .collect();
            after.extend(i.iter().copied());
            after
        });
        if let (Some(setup), false) = (&setup, skip_setup) {
            self.generation += 1;
            self.last_setup = Some((self.generation, setup.clone()));
            let stored = super::code::Code::plain(setup.to_vec())
                .stored()
                .map_err(|e| RunError::Loops {
                    thread: 0,
                    reason: e,
                })?;
            self.stage(dev, 0, 0, false, kernel.dst_fmt, true, false, None, stored)?;
            let stuck = self.wait(dev, &[0], images, budget)?;
            if let Some((_, _, e)) = stuck.into_iter().next() {
                self.poisoned = true;
                return Err(RunError::Setup(e));
            }
            clock.lap(dev, Phase::Setup);
        }
        if kernel.trace && self.profiling {
            return Err(RunError::Transport(TransportError::Hazard {
                address: mailbox::TRACE_BUFFER,
                reason: "a traced kernel would reset a profile's timestamper stream",
            }));
        }
        if kernel.trace {
            dev.configure_trace(
                &self.window,
                tile,
                mailbox::TRACE_BUFFER,
                mailbox::TRACE_BUFFER_BYTES,
            )?;
        }
        self.generation += 1;
        // Every role's program and descriptor is in place before any role's
        // generation moves, so the three start as close together as the host's
        // three final writes allow.
        for (thread, program) in programs.iter().enumerate() {
            let dump = if thread == 1 { kernel.dump_rows } else { 0 };
            self.stage(
                dev,
                thread,
                dump,
                kernel.trace || self.roles_traced(),
                kernel.dst_fmt,
                false,
                resident_programs,
                kernel.mop_words(thread)?,
                kernel.stored_program(thread, program)?,
            )?;
        }
        clock.lap(dev, Phase::Programs);
        Ok(clock)
    }

    /// Move role `thread` to the current generation, so it runs.
    fn go<T: Transport>(&self, dev: &mut Device<T>, thread: usize) -> Result<(), RunError> {
        let mb = Mailbox::of(thread as u32);
        dev.write32(&self.window, self.tile, mb.generation(), self.generation)?;
        Ok(())
    }

    /// After the roles have finished (or `stuck` says which did not): record
    /// the semaphores a restoring kernel left, and read back what it asks for.
    fn finish<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        kernel: &Kernel<'_>,
        mut clock: Stopwatch,
        stuck: Vec<(usize, Core, WaitError)>,
        init: Option<Vec<SemaphoreInit>>,
    ) -> Result<Outcome, RunError> {
        if !stuck.is_empty() {
            self.poisoned = true;
            return Err(RunError::Roles(stuck));
        }
        if kernel.restores_semaphores && init.is_some() {
            self.semaphores = self.semaphores_after.take();
        } else {
            self.semaphores_after = None;
        }
        let tile = self.tile;
        let math = Mailbox::of(1);
        let mut dst = Vec::with_capacity((kernel.dump_rows * mailbox::DUMP_ROW_WORDS) as usize);
        for row in 0..kernel.dump_rows {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                dst.push(dev.read32(&self.window, tile, math.dump_offset(row, col))?);
            }
        }
        let trace = if kernel.trace {
            dev.read_trace(&self.window, tile, mailbox::TRACE_BUFFER)?
        } else {
            Vec::new()
        };
        let mut l1 = Vec::with_capacity(kernel.read_back.len());
        for (addr, len) in kernel.read_back {
            let mut buf = vec![0u8; *len];
            dev.l1_read(&self.window, tile, *addr, &mut buf)?;
            l1.push(buf);
        }
        clock.lap(dev, Phase::ReadBack);
        Ok(Outcome {
            dst,
            l1,
            trace,
            profile: clock.profile,
        })
    }

    /// Hold the three role cores in reset again.
    pub fn stop<T: Transport>(
        self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
    ) -> Result<(), RunError> {
        self.hold(dev, images)
    }

    fn hold<T: Transport>(
        &self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
    ) -> Result<(), RunError> {
        for (core, _, _) in images.iter() {
            dev.set_core_reset(&self.window, self.tile, *core, true)?;
        }
        Ok(())
    }

    /// Write role `thread`'s descriptor and program; with `go`, also move its
    /// generation so it runs.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn stage<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
        dump: u32,
        traced: bool,
        dst_fmt: u32,
        go: bool,
        resident_programs: bool,
        mop_cfg: Option<[u32; 9]>,
        stored: (Vec<u32>, u32),
    ) -> Result<(), RunError> {
        let (w, tile) = (&self.window, self.tile);
        let mb = Mailbox::of(thread as u32);
        let (words, len) = stored;
        let push_window = if dev.transport().is_simulated() {
            mailbox::SIM_PUSH_WINDOW
        } else {
            0
        };
        // From the role's fixed slot (`program_addr` zero), unless a `KERNEL`
        // entry points it at a resident program (`mailbox::PROGRAM_ADDR`).
        let d = mailbox::Descriptor {
            thread_index: thread as u32,
            dst_access_fmt: dst_fmt,
            program_len: len,
            dump_row_count: dump,
            trace: u32::from(traced),
            push_window,
            mop_cfg,
            ..Default::default()
        };
        let mut descs = self.descriptors.borrow_mut();
        let last = descs[thread];
        // Forget it first: if a write fails, the cache claims nothing.
        descs[thread] = None;
        let writes = d.writes(mb);
        for (k, &(at, v)) in writes.iter().enumerate() {
            let rewritten_on_tile = at == mb.program_len() || at == mb.program_addr();
            // A resident kernel's program is named by its `KERNEL` entry,
            // which the mover writes before each run: the host writing it too
            // could land between the mover's write and the role's read.
            if rewritten_on_tile && resident_programs {
                continue;
            }
            if rewritten_on_tile || last.is_none_or(|l| l[k] != v) {
                dev.write32(w, tile, at, v)?;
            }
        }
        descs[thread] = Some(writes.map(|(_, v)| v));
        drop(descs);
        let mut slots = self.slots.borrow_mut();
        if !words.is_empty() && slots[thread] != words {
            // Forget the slot first: if the write fails, it holds neither.
            slots[thread].clear();
            dev.l1_write(w, tile, mb.program(), &word_bytes(&words))?;
            slots[thread] = words;
        }
        drop(slots);
        for row in 0..dump {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                dev.write32(w, tile, mb.dump_offset(row, col), DUMP_SENTINEL)?;
            }
        }
        if go {
            dev.write32(w, tile, mb.generation(), self.generation)?;
        }
        Ok(())
    }

    /// Wait for each of `threads` to acknowledge the current generation.
    fn wait<T: Transport>(
        &self,
        dev: &mut Device<T>,
        threads: &[usize],
        images: &RoleImages<'_>,
        budget: u64,
    ) -> Result<Vec<(usize, Core, WaitError)>, RunError> {
        let mut stuck = Vec::new();
        for &thread in threads {
            let mb = Mailbox::of(thread as u32);
            let generation = self.generation;
            let r = dev.wait_for_mailbox(
                &self.window,
                self.tile,
                mb.ack(),
                mb.panic_code(),
                budget,
                |v| v == generation,
            )?;
            if let Err(e) = r {
                stuck.push((thread, images[thread].0, e));
            }
        }
        Ok(stuck)
    }
}

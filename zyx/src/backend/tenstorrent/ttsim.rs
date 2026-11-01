// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
//! The native Tenstorrent runtime: one owner thread holding the device.
//!
//! `tt_ttsim::Simulator` is `!Send + !Sync` (the simulator library is
//! process-global and a transport borrows it), so the device can never leave
//! the thread that opened it. Everything in zyx parks on a channel instead:
//! this thread opens the simulator and the `Device` once, then answers
//! [`Cmd::Run`] with one run of a tile — the program slots are checked against
//! the residency ledger and loaded from the card's DRAM copy on a miss, the B
//! mover reads the operands into L1, [`tt_kernels::runtime::run`] runs the
//! three roles, the NC mover writes the results out. A launch is synchronous —
//! it returns only once every role has reported `DONE` and every moved byte has
//! landed — so the caller sees the same blocking semantics the old shim had.
//!
//! Nothing here decides what to run: the DRAM ranges and the L1 blocks they
//! land in come from the launch's [`NativeRun`], and which of them still have
//! to be moved comes from this thread's residency ledger (NOTES §2). What lives
//! here is the wire protocol: the movers' mailboxes and descriptor lists, the
//! program slots, and the host's own DRAM upload/readback for placement and
//! readback.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::thread;

use super::dev::Device;
use super::dev::tlb::WindowKind;
use super::kern::runtime::{self, Schedule, SemaphoreInit};
use super::sim::{self, Simulator};
use tt_isa::dm;
use tt_isa::dram::{Dram, DramRange};
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::Core;

/// Simulated cycles one run may take before it is declared stuck. A run that
/// does not reach `DONE` in this many clocks is a bug in the program, not
/// slowness.
const BUDGET: u64 = 400_000;

/// Simulated cycles a mover may take to boot or to finish a list.
const MOVER_BUDGET: u64 = 1_000_000;

/// One role program of a launch: the image it names and where its bytes go.
#[derive(Clone)]
pub(crate) struct RoleProgram {
    /// The image in the device's program cache.
    pub(crate) id: super::ImageId,
    /// The image's identity: a recycled id is a miss.
    pub(crate) generation: u32,
    /// The program length word the role's descriptor names.
    pub(crate) len: u32,
    /// DRAM -> L1 blocks, in page order: `(DRAM range, address in the slot)`.
    pub(crate) ranges: Vec<(DramRange, u32)>,
}

/// One mover firmware image: the DRAM copy of its bytes and where it runs.
#[derive(Clone)]
pub(crate) struct Firmware {
    pub(crate) id: super::ImageId,
    pub(crate) generation: u32,
    /// The core that runs it.
    pub(crate) core: Core,
    /// L1 address its bytes are written to.
    pub(crate) base: u64,
    /// The DRAM ranges holding the bytes, in order.
    pub(crate) ranges: Vec<DramRange>,
}

/// One launch of a native program: what the owner thread needs to move the
/// operands in, start the three roles, move the results out and collect what
/// the host still has to touch.
pub(crate) struct NativeRun {
    /// The Tensix tile this grid point runs on.
    pub(crate) tile: (u8, u8),
    /// T0/T1/T2 programs, by role thread; `None` for a leg the kernel does not
    /// have (a gather or scatter has none).
    pub(crate) programs: [Option<RoleProgram>; 3],
    /// The tile's two mover firmware images, B then NC.
    pub(crate) firmware: [Firmware; 2],
    /// DRAM -> L1 blocks the tile's B mover reads before any role starts.
    pub(crate) reads: Vec<(DramRange, u32)>,
    /// L1 -> DRAM blocks its NC mover writes after every role finished.
    pub(crate) writes: Vec<(DramRange, u32)>,
    /// L1 ranges read back after every role finished: `(address, bytes)`.
    pub(crate) read_back: Vec<(u64, usize)>,
    /// Semaphore initial states. Empty runs the roles in order without any
    /// synchronisation (`Schedule::InOrder`).
    pub(crate) sems: Vec<SemaphoreInit>,
}

/// What each core's program slot holds, per tile: `(image, generation)` when a
/// slot is resident, `None` when it must be loaded (NOTES §2).
pub(crate) type Ledger = [Option<(super::ImageId, u32)>; 5];

/// What a run produced.
pub(crate) struct NativeOutcome {
    /// One buffer per [`NativeRun::read_back`] range, in the same order.
    pub(crate) l1: Vec<Vec<u8>>,
    /// The `Dst` rows the math role dumped (`row * 16 + column`), empty when
    /// the program asked for none.
    pub(crate) dst: Vec<u32>,
}

enum Cmd {
    Run(Box<NativeRun>, Sender<Result<NativeOutcome, String>>),
    /// Write each range from its gathered bytes, in order.
    DramWrite(Vec<(DramRange, Vec<u8>)>, Sender<Result<(), String>>),
    /// Read each range, in order.
    DramRead(Vec<DramRange>, Sender<Result<Vec<Vec<u8>>, String>>),
}

static OWNER: OnceLock<Mutex<Sender<Cmd>>> = OnceLock::new();

/// Run one native program on the device, blocking until every role is done.
///
/// The first call opens the simulator and starts the owner thread.
pub(crate) fn run(run: NativeRun) -> Result<NativeOutcome, String> {
    let (reply, reply_rx) = channel();
    let sent = {
        let owner = owner().lock().unwrap_or_else(|p| p.into_inner());
        owner.send(Cmd::Run(Box::new(run), reply))
    };
    if sent.is_err() {
        return Err("the Tenstorrent owner thread is gone".to_string());
    }
    reply_rx.recv().map_err(|_| "the Tenstorrent owner thread dropped the reply".to_string())?
}

/// Write host bytes into the card's DRAM, one segment per channel run.
///
/// Each segment carries the range and the bytes that go there, already
/// gathered in channel-offset order. They go out in order on the owner
/// thread, which is the only holder of the device.
pub(crate) fn dram_write(segments: Vec<(DramRange, Vec<u8>)>) -> Result<(), String> {
    let (reply, reply_rx) = channel();
    let sent = {
        let owner = owner().lock().unwrap_or_else(|p| p.into_inner());
        owner.send(Cmd::DramWrite(segments, reply))
    };
    if sent.is_err() {
        return Err("the Tenstorrent owner thread is gone".to_string());
    }
    reply_rx.recv().map_err(|_| "the Tenstorrent owner thread dropped the reply".to_string())?
}

/// Read the card's DRAM into host bytes, one buffer per range, in order.
pub(crate) fn dram_read(segments: Vec<DramRange>) -> Result<Vec<Vec<u8>>, String> {
    let (reply, reply_rx) = channel();
    let sent = {
        let owner = owner().lock().unwrap_or_else(|p| p.into_inner());
        owner.send(Cmd::DramRead(segments, reply))
    };
    if sent.is_err() {
        return Err("the Tenstorrent owner thread is gone".to_string());
    }
    reply_rx.recv().map_err(|_| "the Tenstorrent owner thread dropped the reply".to_string())?
}

/// The owner thread's command channel, started on first use.
fn owner() -> &'static Mutex<Sender<Cmd>> {
    OWNER.get_or_init(|| {
        let (tx, rx) = channel::<Cmd>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        thread::Builder::new()
            .name("zyx-ttsim".to_string())
            .spawn(move || serve(rx, ready_tx))
            .expect("tenstorrent: cannot spawn the owner thread");
        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => panic!("tenstorrent: {e}"),
            Err(_) => panic!("tenstorrent: the owner thread died while opening the simulator"),
        }
        Mutex::new(tx)
    })
}

/// The owner thread: open the simulator once, then answer runs.
fn serve(rx: Receiver<Cmd>, ready: Sender<Result<(), String>>) {
    // `sim` must outlive `dev`: the transport borrows it.
    let mut sim = match Simulator::open() {
        Ok(sim) => sim,
        Err(e) => {
            let _ = ready.send(Err(format!("cannot open the simulator: {e}")));
            return;
        }
    };
    let mut dev = match Device::open(sim.transport()) {
        Ok(dev) => dev,
        Err(e) => {
            let _ = ready.send(Err(format!("cannot open the device: {e}")));
            return;
        }
    };
    // One 2 MiB TLB window for every host DRAM access; a transfer retargets it
    // as it goes (`Device::dram_read`/`dram_write`).
    let window = match dev.alloc_window(WindowKind::TwoMib) {
        Ok(window) => window,
        Err(e) => {
            let _ = ready.send(Err(format!("cannot allocate a DRAM window: {e}")));
            return;
        }
    };
    // Movers are booted per tile on the first launch that uses it.
    let mut movers: HashMap<(u8, u8), Movers> = HashMap::new();
    // What each tile's five program slots hold (NOTES §2).
    let mut resident: HashMap<(u8, u8), Ledger> = HashMap::new();
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Run(run, reply) => {
                let _ = reply.send(execute(&mut dev, &window, &mut movers, &mut resident, &run));
            }
            Cmd::DramWrite(segments, reply) => {
                let _ = reply.send(execute_dram_write(&mut dev, &window, &segments));
            }
            Cmd::DramRead(segments, reply) => {
                let _ = reply.send(execute_dram_read(&mut dev, &window, &segments));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The data movers: RISCV B reads DRAM into L1, RISCV NC writes L1 back.
//
// Each tile boots the two images once ([`tt_firmware_images::DM_B`],
// [`DM_NC`]) and keeps them resident; a launch then hands B its reads as one
// list before the roles run and NC its writes as one list after, so the host
// never carries a tensor byte. The protocol is `tt_isa::dm`'s: write the
// entries into the mover's ring, set `OP`/`LEN`, bump `SEQ`, poll `DONE` and
// read `ERROR`. One `Device` round trip per list, however many pages it holds.
// ---------------------------------------------------------------------------

/// The tile's two movers, booted on first use.
#[derive(Debug)]
struct Movers {
    /// Last sequence number submitted to each, in [`B`]/[`NC`] order. The
    /// protocol skips zero, so a descriptor always looks new to the mover.
    seq: [u32; 2],
}

/// Index of RISCV B's mover in [`Movers::seq`]: the reader.
const B: usize = 0;
/// Index of RISCV NC's mover: the writer.
const NC: usize = 1;

fn mover(which: usize) -> dm::Mover {
    match which {
        B => dm::Mover::B,
        NC => dm::Mover::NC,
        other => unreachable!("mover index {other} is neither B nor NC"),
    }
}

/// Boot one mover from its image: the bytes come back from the card's DRAM
/// copy over the NoC, the mailbox words go in, then the image starts and the
/// host waits for `RUNNING` and `CAPS`. This is the first residency miss on the
/// core's slot, not a separate step (NOTES §2).
///
/// The mailbox is written before the image starts because the mover reads
/// `MY_X`/`MY_Y`/`USABLE` once, as it comes up; everything the host resets
/// here is state L1 keeps across runs on silicon.
fn boot_image(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    tile: NocCoord<Noc0>,
    dram: &Dram,
    fw: &Firmware,
) -> Result<(), String> {
    let mut bytes: Vec<u8> = Vec::with_capacity(fw.ranges.iter().map(|r| r.len() as usize).sum());
    for range in &fw.ranges {
        let mut buf = vec![0u8; range.len() as usize];
        dev.dram_read(window, *range, &mut buf).map_err(|e| e.to_string())?;
        bytes.extend_from_slice(&buf);
    }
    let which = match fw.core {
        Core::B => B,
        Core::NC => NC,
        other => todo!("{other:?} has no mover firmware"),
    };
    let m = mover(which);
    let at = |word: u64| m.at(word);
    write32(dev, window, tile, at(dm::MY_X), tile.x() as u32)?;
    write32(dev, window, tile, at(dm::MY_Y), tile.y() as u32)?;
    write32(dev, window, tile, at(dm::USABLE), u32::from(dram.usable_mask()))?;
    // Nothing here uses host-encoded reads or tracing; zero is the off
    // state of both, and of every counter the mover writes.
    for word in [
        dm::SEQ,
        dm::DONE,
        dm::ERROR,
        dm::TRACE,
        dm::THROTTLE_STALLS,
        dm::THROTTLE_CYCLES,
        dm::WRITE_NOC,
        dm::READ_FAST,
        dm::CAPS,
        dm::QUEUE_HEAD,
        dm::QUEUE_DONE,
        dm::QUEUE_ERROR,
        dm::QUEUE_ERROR_AT,
        dm::TRACE_PROGRESS,
        dm::ENCODE_KEY,
    ] {
        write32(dev, window, tile, at(word), 0)?;
    }
    write32(dev, window, tile, at(dm::IN_FLIGHT_CAP), dm::TILE_IN_FLIGHT_CAP)?;
    let status_at = m.mailbox + mailbox::offset::STATUS;
    let panic_at = m.mailbox + mailbox::offset::PANIC_CODE;
    write32(dev, window, tile, status_at, 0)?;
    match fw.core {
        Core::NC => dev.load_and_start_nc(window, tile, &bytes, fw.base),
        core => dev.load_and_start(window, tile, core, &bytes, fw.base),
    }
    .map_err(|e| format!("cannot start the {core:?} mover: {e}", core = fw.core))?;
    dev.wait_for_mailbox(window, tile, status_at, panic_at, MOVER_BUDGET, |s| s == status::RUNNING)
        .map_err(|e| format!("the {:?} mover: {e}", fw.core))?
        .map_err(|e| format!("the {:?} mover: {e:?}", fw.core))?;
    // What the image says it can do, written after the status word.
    dev.wait_for_mailbox(window, tile, at(dm::CAPS), panic_at, MOVER_BUDGET, |c| c & dm::caps::VALID != 0)
        .map_err(|e| format!("the {:?} mover: {e}", fw.core))?
        .map_err(|e| format!("the {:?} mover: {e:?}", fw.core))?;
    Ok(())
}

fn write32(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    tile: NocCoord<Noc0>,
    at: u64,
    value: u32,
) -> Result<(), String> {
    dev.write32(window, tile, at, value).map_err(|e| e.to_string())
}

/// Run `segments` as one list on `which` mover: `(DRAM range, L1 address)`
/// pairs, read in or written out in order. Waits for the list to finish, so
/// returning means every byte moved (or the mover refused it).
fn mover_list(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    tile: NocCoord<Noc0>,
    movers: &mut Movers,
    which: usize,
    segments: &[(DramRange, u32)],
) -> Result<(), String> {
    if segments.is_empty() {
        return Ok(());
    }
    let m = mover(which);
    let op = match which {
        B => dm::op::READ,
        NC => dm::op::WRITE,
        other => unreachable!("mover index {other} is neither B nor NC"),
    };
    for list in segments.chunks(dm::LIST_MAX as usize) {
        let mut entries = Vec::with_capacity(list.len() * dm::ENTRY_BYTES as usize);
        for (range, l1) in list {
            // Port 0: the mover maps it onto a port its NoC owns, and reads
            // stay on NoC #0 whichever entry names.
            let entry: [u32; 8] = [
                op,
                u32::from(range.channel().index()),
                0,
                range.offset() as u32,
                *l1,
                range.len() as u32,
                0,
                0,
            ];
            for word in entry {
                entries.extend_from_slice(&word.to_le_bytes());
            }
        }
        dev.l1_write(window, tile, m.list, &entries)
            .map_err(|e| format!("cannot write the {:?} mover's list: {e}", m.core))?;
        write32(dev, window, tile, m.at(dm::OP), dm::op::LIST)?;
        write32(dev, window, tile, m.at(dm::LEN), list.len() as u32)?;
        let seq = movers.seq[which].wrapping_add(1).max(1);
        movers.seq[which] = seq;
        write32(dev, window, tile, m.at(dm::SEQ), seq)?;
        wait_done(dev, window, tile, m, seq)?;
    }
    Ok(())
}

/// Wait until `mover` has finished descriptor `seq`, and report a refusal.
fn wait_done(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    tile: NocCoord<Noc0>,
    m: dm::Mover,
    seq: u32,
) -> Result<(), String> {
    let mut ticks = 0u64;
    loop {
        let done = dev.read32(window, tile, m.at(dm::DONE)).map_err(|e| e.to_string())?;
        if done == seq {
            let error = dev.read32(window, tile, m.at(dm::ERROR)).map_err(|e| e.to_string())?;
            if error != dm::error::NONE {
                return Err(format!("the {:?} mover refused list {seq}: error {error}", m.core));
            }
            return Ok(());
        }
        if ticks >= MOVER_BUDGET {
            return Err(format!("the {:?} mover did not finish list {seq} (last done {done})", m.core));
        }
        dev.tick(super::dev::core_control::CYCLES_PER_POLL);
        ticks += u64::from(super::dev::core_control::CYCLES_PER_POLL);
    }
}

/// One DRAM upload: each channel run's bytes into its range.
fn execute_dram_write(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    segments: &[(DramRange, Vec<u8>)],
) -> Result<(), String> {
    for (range, bytes) in segments {
        dev.dram_write(window, *range, bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// One DRAM download: one buffer per range, in order.
fn execute_dram_read(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    segments: &[DramRange],
) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::with_capacity(segments.len());
    for range in segments {
        let mut buf = vec![0u8; range.len() as usize];
        dev.dram_read(window, *range, &mut buf).map_err(|e| e.to_string())?;
        out.push(buf);
    }
    Ok(out)
}

/// One run: the tile's program slots are checked against the ledger and
/// loaded from the device's cache on a miss, the B mover brings the operands
/// in, the roles compute, the NC mover writes the results out, and only what
/// the host must narrow is read back.
fn execute(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    movers: &mut HashMap<(u8, u8), Movers>,
    resident: &mut HashMap<(u8, u8), Ledger>,
    run: &NativeRun,
) -> Result<NativeOutcome, String> {
    if std::env::var_os("ZYX_TT_DUMP").is_some() {
        for (thread, program) in run.programs.iter().enumerate() {
            if let Some(p) = program {
                eprintln!("[tt] role T{thread}: {} bytes, {} entries, len {}", p.len * 4, p.ranges.len(), p.len);
            }
        }
        eprintln!(
            "[tt] moves: {} in (B), {} out (NC), {} read back",
            run.reads.len(),
            run.writes.len(),
            run.read_back.len()
        );
    }
    let tile =
        NocCoord::<Noc0>::new(run.tile.0, run.tile.1).ok_or_else(|| format!("{:?} is not a Tensix coordinate", run.tile))?;
    let state = movers.entry(run.tile).or_insert(Movers { seq: [0; 2] });
    // Boot a mover on its slot's first miss (NOTES §2: boot is not a step).
    {
        let ledger = resident.entry(run.tile).or_insert([None; 5]);
        for fw in run.firmware.iter() {
            let slot = super::slot(fw.core);
            if ledger[slot] == Some((fw.id, fw.generation)) {
                continue;
            }
            let dram = dev.dram_grid(window).map_err(|e| format!("cannot read the DRAM grid: {e}"))?;
            boot_image(dev, window, tile, &dram, fw)?;
            // A freshly booted mover starts its sequence over.
            state.seq = [0; 2];
            ledger[slot] = Some((fw.id, fw.generation));
        }
    }
    if !run.reads.is_empty() {
        mover_list(dev, window, tile, state, B, &run.reads)?;
    }
    // Which role slots already hold their program. The concurrent schedule's
    // setup run pushes its own words into slot 0, so that slot is a miss by
    // construction.
    let mut flags = [false; 3];
    {
        let ledger = resident.entry(run.tile).or_insert([None; 5]);
        if !run.sems.is_empty() {
            ledger[super::slot(Core::T0)] = None;
        }
        for (thread, program) in run.programs.iter().enumerate() {
            let Some(p) = program else { continue };
            let core = [Core::T0, Core::T1, Core::T2][thread];
            flags[thread] = ledger[super::slot(core)] == Some((p.id, p.generation));
        }
    }
    let mut loaded = [false; 3];
    let mut load = |dev: &mut Device<sim::LibTtsim<'_>>, thread: usize| -> Result<(), runtime::RunError> {
        let program = run.programs[thread].as_ref().expect("the loader is called only for a role with a program");
        mover_list(dev, window, tile, state, B, &program.ranges)
            .map_err(|reason| runtime::RunError::Program { thread, reason })?;
        loaded[thread] = true;
        Ok(())
    };
    let stage: Vec<(u64, &[u8])> = Vec::new();
    let schedule = if run.sems.is_empty() {
        Schedule::InOrder
    } else {
        Schedule::Concurrent(&run.sems)
    };
    let mut kernel = runtime::Kernel { schedule, stage: &stage, read_back: &run.read_back, ..runtime::Kernel::new(schedule) };
    for thread in 0..3 {
        if let Some(program) = &run.programs[thread] {
            kernel.programs[thread] = Some(program.len);
            kernel.resident[thread] = flags[thread];
        }
    }
    let out = runtime::run(dev, tile, &tt_firmware_images::ROLES, &kernel, BUDGET, &mut load);
    let ledger = resident.entry(run.tile).or_insert([None; 5]);
    match &out {
        Ok(_) => {
            for thread in 0..3 {
                if !loaded[thread] {
                    continue;
                }
                let program = run.programs[thread].as_ref().expect("loaded programs name their role");
                let core = [Core::T0, Core::T1, Core::T2][thread];
                ledger[super::slot(core)] = Some((program.id, program.generation));
            }
        }
        Err(_) => {
            // A failed tile holds nothing trustworthy: reload on next use.
            *ledger = [None; 5];
        }
    }
    let out = out.map_err(|e| e.to_string())?;
    if !run.writes.is_empty() {
        mover_list(dev, window, tile, state, NC, &run.writes)?;
    }
    Ok(NativeOutcome { l1: out.l1, dst: out.dst })
}

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
//! mover reads the operands into L1, [`crate::backend::tenstorrent::kern::runtime::run`] runs the
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
use super::firmware_images;
use super::runtime::{self, Schedule};
use super::sim::{self, Simulator};
use crate::codegen::tenstorrent::l1::SemaphoreInit;
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
    /// A pure transfer: one mover program and its launch arguments. When set,
    /// there are no role programs, no reads and no writes: the program itself
    /// moves DRAM through its L1 pages.
    pub(crate) transfer: Option<Transfer>,
}

/// One pure-transfer program: its words and the `u32` args array.
pub(crate) struct Transfer {
    pub(crate) words: Vec<u32>,
    pub(crate) args: Vec<u32>,
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
    #[allow(dead_code)] // populated for post-mortem dumps; no reader needs it yet
    pub(crate) dst: Vec<u32>,
}

/// A staged-program run on RISCV B: DRAM in, L1 blocks (program, indices,
/// control), and DRAM back out.
#[allow(dead_code)] // the pipeline movers are the first non-test caller

pub(crate) struct ExecRun {
    /// The tile that runs the program.
    pub(crate) tile: (u8, u8),
    /// DRAM segments to write before the run.
    pub(crate) dram: Vec<(DramRange, Vec<u8>)>,
    /// L1 blocks to write before the run: `(address, bytes)`.
    pub(crate) l1: Vec<(u32, Vec<u8>)>,
    /// DRAM segments to read after the run, in order.
    pub(crate) read_back: Vec<DramRange>,
    /// L1 ranges to read after the run: `(address, length)`.
    pub(crate) l1_read_back: Vec<(u32, u32)>,
}

/// What a staged-program run produced.
#[allow(dead_code)]
pub(crate) struct ExecResult {
    /// One buffer per [`ExecRun::read_back`] range.
    pub(crate) dram: Vec<Vec<u8>>,
    /// One buffer per [`ExecRun::l1_read_back`] range.
    pub(crate) l1: Vec<Vec<u8>>,
    /// The value the program left in `a0`.
    pub(crate) ret: u32,
    /// The executor's error word, zero on a clean run.
    pub(crate) error: u32,
}

enum Cmd {
    Run(Box<NativeRun>, Sender<Result<NativeOutcome, String>>),
    #[allow(dead_code)]
    Exec(Box<ExecRun>, Sender<Result<ExecResult, String>>),
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
        let owner = owner()?.lock().unwrap_or_else(|p| p.into_inner());
        owner.send(Cmd::Run(Box::new(run), reply))
    };
    if sent.is_err() {
        return Err("the Tenstorrent owner thread is gone".to_string());
    }
    reply_rx.recv().map_err(|_| "the Tenstorrent owner thread dropped the reply".to_string())?
}

/// Write host bytes into the card's DRAM, one segment per channel run.
pub(crate) fn dram_write(segments: Vec<(DramRange, Vec<u8>)>) -> Result<(), String> {
    let (reply, reply_rx) = channel();
    let sent = {
        let owner = owner()?.lock().unwrap_or_else(|p| p.into_inner());
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
        let owner = owner()?.lock().unwrap_or_else(|p| p.into_inner());
        owner.send(Cmd::DramRead(segments, reply))
    };
    if sent.is_err() {
        return Err("the Tenstorrent owner thread is gone".to_string());
    }
    reply_rx.recv().map_err(|_| "the Tenstorrent owner thread dropped the reply".to_string())?
}

/// Run a staged program on RISCV B and read DRAM back.
#[allow(dead_code)]
pub(crate) fn exec(run: ExecRun) -> Result<ExecResult, String> {
    let (reply, reply_rx) = channel();
    let sent = {
        let owner = owner()?.lock().unwrap_or_else(|p| p.into_inner());
        owner.send(Cmd::Exec(Box::new(run), reply))
    };
    if sent.is_err() {
        return Err("the Tenstorrent owner thread is gone".to_string());
    }
    reply_rx.recv().map_err(|_| "the Tenstorrent owner thread dropped the reply".to_string())?
}

/// Open the simulator and start the owner thread, once per successful open.
///
/// Errors are returned, never panicked: a machine without a usable simulator
/// has no Tenstorrent device, which is what the backend reports.
pub(crate) fn ensure_open() -> Result<(), String> {
    owner().map(|_| ())
}

/// The owner thread's command channel. The simulator is opened on the first
/// call; a failure leaves nothing cached, so the next call tries once more
/// (the simulator refuses a second open, which surfaces as its own error).
fn owner() -> Result<&'static Mutex<Sender<Cmd>>, String> {
    if let Some(owner) = OWNER.get() {
        return Ok(owner);
    }
    let (tx, rx) = channel::<Cmd>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();
    thread::Builder::new()
        .name("zyx-ttsim".to_string())
        .spawn(move || serve(rx, ready_tx))
        .map_err(|e| format!("cannot spawn the owner thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("the owner thread died while opening the simulator".to_string()),
    }
    Ok(OWNER.get_or_init(|| Mutex::new(tx)))
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
            Cmd::Exec(run, reply) => {
                let _ = reply.send(execute_exec(&mut dev, &window, &run));
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
// Each tile boots the two images once ([`firmware_images::DM_B`],
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
    broadcast: bool,
) -> Result<(), String> {
    if segments.is_empty() {
        return Ok(());
    }
    let m = mover(which);
    // One list per direction per launch: `tt_verify` rejects a longer stream
    // at compile time, so a second chunk here means that check was bypassed.
    // Never split silently: the entry count is the stream's shape.
    let mut batches = segments.chunks(dm::LIST_MAX as usize);
    let Some(list) = batches.next() else {
        return Ok(());
    };
    if batches.next().is_some() {
        panic!(
            "the {:?} mover's list holds {} entries, LIST_MAX is {}: tt_verify must reject this at compile time",
            m.core,
            segments.len(),
            dm::LIST_MAX
        );
    }
    // The entry's opcode is the ISA instruction that does the move, per entry:
    // one whole tile image (`TILE_SLOT`) is the mover's shaped read, whose
    // firmware copies the 16-byte header and then writes datum `(r, c)` from
    // `(r, 0)` of the tile.
    let mut entries = Vec::with_capacity(list.len() * dm::ENTRY_BYTES as usize);
    for (range, l1) in list {
        let op = match which {
            // A shaped read is a *data* read: one tile image into a CB. A role
            // program's bytes are code, whatever their length, so `broadcast`
            // is false there and a program of exactly one tile-slot is loaded
            // as it is written.
            B if broadcast && range.len() == dm::TILE_SLOT => dm::op::READ_BROADCAST_COL,
            B => dm::op::READ,
            NC => dm::op::WRITE,
            other => unreachable!("mover index {other} is neither B nor NC"),
        };
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
    dev.l1_write(window, tile, m.list, &entries).map_err(|e| format!("cannot write the {:?} mover's list: {e}", m.core))?;
    write32(dev, window, tile, m.at(dm::OP), dm::op::LIST)?;
    write32(dev, window, tile, m.at(dm::LEN), list.len() as u32)?;
    let seq = movers.seq[which].wrapping_add(1).max(1);
    movers.seq[which] = seq;
    write32(dev, window, tile, m.at(dm::SEQ), seq)?;
    wait_done(dev, window, tile, m, seq)?;
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
/// Write the run's DRAM and L1 blocks, boot the executor on RISCV B, run one
/// generation, then read DRAM back.
fn execute_exec(dev: &mut Device<sim::LibTtsim<'_>>, window: &super::dev::Window, run: &ExecRun) -> Result<ExecResult, String> {
    let tile =
        NocCoord::<Noc0>::new(run.tile.0, run.tile.1).ok_or_else(|| format!("{:?} is not a Tensix coordinate", run.tile))?;
    for (range, bytes) in &run.dram {
        dev.dram_write(window, *range, bytes).map_err(|e| e.to_string())?;
    }
    for (at, bytes) in &run.l1 {
        dev.l1_write(window, tile, u64::from(*at), bytes).map_err(|e| e.to_string())?;
    }
    let (core, image, base) = firmware_images::EXEC_B;
    assert_eq!(core, Core::B, "the staged-program executor runs on RISCV B");
    dev.load_and_start(window, tile, core, image, base).map_err(|e| format!("cannot start the executor: {e}"))?;
    // The image publishes RUNNING from its trampoline before it reads SEQ.
    let status_at = mailbox::MAILBOX_BASE + mailbox::offset::STATUS;
    let panic_at = mailbox::MAILBOX_BASE + mailbox::offset::PANIC_CODE;
    dev.wait_for_mailbox(window, tile, status_at, panic_at, BUDGET, |s| s == status::RUNNING)
        .map_err(|e| format!("the executor: {e}"))?
        .map_err(|e| format!("the executor: {e:?}"))?;
    // One generation; the protocol skips zero.
    dev.write32(window, tile, mailbox::exec::SEQ, 1).map_err(|e| e.to_string())?;
    dev.wait_for_mailbox(window, tile, mailbox::exec::DONE, panic_at, BUDGET, |d| d == 1)
        .map_err(|e| format!("the staged program: {e}"))?
        .map_err(|e| format!("the staged program: {e:?}"))?;
    let ret = dev.read32(window, tile, mailbox::exec::RET).map_err(|e| e.to_string())?;
    let error = dev.read32(window, tile, mailbox::exec::ERROR).map_err(|e| e.to_string())?;
    let mut dram = Vec::with_capacity(run.read_back.len());
    for range in &run.read_back {
        let mut buf = vec![0u8; range.len() as usize];
        dev.dram_read(window, *range, &mut buf).map_err(|e| e.to_string())?;
        dram.push(buf);
    }
    let mut l1 = Vec::with_capacity(run.l1_read_back.len());
    for (at, len) in &run.l1_read_back {
        let mut buf = vec![0u8; *len as usize];
        dev.l1_read(window, tile, u64::from(*at), &mut buf).map_err(|e| e.to_string())?;
        l1.push(buf);
    }
    Ok(ExecResult { dram, l1, ret, error })
}

fn execute(
    dev: &mut Device<sim::LibTtsim<'_>>,
    window: &super::dev::Window,
    movers: &mut HashMap<(u8, u8), Movers>,
    resident: &mut HashMap<(u8, u8), Ledger>,
    run: &NativeRun,
) -> Result<NativeOutcome, String> {
    if let Some(t) = &run.transfer {
        // One mover program moves the whole boundary: stage it and its args,
        // boot the executor and run one generation. The DRAM is the pool's, so
        // nothing is staged or read back here.
        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let args_at = program_at + 0x2000;
        let mut l1 = vec![
            (program_at, t.words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()),
            (args_at, t.args.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()),
        ];
        let mut control = Vec::new();
        for word in [program_at, t.words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));
        let out = execute_exec(
            dev,
            window,
            &ExecRun { tile: run.tile, dram: Vec::new(), l1, read_back: Vec::new(), l1_read_back: Vec::new() },
        )?;
        if out.error != 0 {
            return Err(format!("the transfer program's error word is {:#x}", out.error));
        }
        return Ok(NativeOutcome { l1: Vec::new(), dst: Vec::new() });
    }
    if std::env::var_os("ZYX_TT_DUMP").is_some() {
        for (thread, program) in run.programs.iter().enumerate() {
            if let Some(p) = program {
                // The length word also carries the STREAMED/LOOPED flags, so
                // the plain word count is the masked value.
                let words = p.len & tt_isa::dataflow::LENGTH_MASK;
                eprintln!("[tt] role T{thread}: {} bytes, {} entries, len {words}", words as u64 * 4, p.ranges.len());
            }
        }
        eprintln!("[tt] moves: {} in (B), {} out (NC), {} read back", run.reads.len(), run.writes.len(), run.read_back.len());
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
        mover_list(dev, window, tile, state, B, &run.reads, true)?;
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
        // A program's bytes are code: a plain read, never a tile image's shaped
        // one, whatever the range's length.
        mover_list(dev, window, tile, state, B, &program.ranges, false)
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
    if std::env::var_os("ZYX_DST_DUMP").is_some() {
        kernel.dump_rows = tt_isa::mailbox::DUMP_MAX_ROWS;
        kernel.clear_dst = false;
    }
    for thread in 0..3 {
        if let Some(program) = &run.programs[thread] {
            kernel.programs[thread] = Some(program.len);
            kernel.resident[thread] = flags[thread];
        }
    }
    let out = runtime::run(dev, tile, &firmware_images::ROLES, &kernel, BUDGET, &mut load);
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
    if std::env::var_os("ZYX_MOVE_DUMP").is_some() {
        for (r, at) in run.reads.iter() {
            eprintln!("READ dst_l1=0x{at:x} src={r:?}");
        }
        for (r, at) in run.writes.iter() {
            eprintln!("WRITE dst_l1=0x{at:x} src={r:?}");
        }
    }
    if std::env::var_os("ZYX_CODE_DUMP").is_some() {
        let mut buf = vec![0u8; 0x0A00];
        if dev.l1_read(&window, tile, 0x1_2010, &mut buf).is_ok() {
            for i in 0..0x0A00 / 4 {
                let w = u32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
                if w != 0 {
                    eprintln!("CODE0 {i:03}: {w:08x}");
                }
            }
        }
    }
    if std::env::var_os("ZYX_DST_DUMP").is_some()
        && let Ok(o) = &out
    {
        for row in 0..tt_isa::mailbox::DUMP_MAX_ROWS as usize {
            let vals: Vec<u32> = (0..tt_isa::mailbox::DUMP_ROW_WORDS as usize).map(|c| o.dst[row * 16 + c]).collect();
            eprintln!("DST row {row}: {vals:08x?}");
        }
    }
    let out = out.map_err(|e| e.to_string())?;
    if !run.writes.is_empty() {
        mover_list(dev, window, tile, state, NC, &run.writes, false)?;
    }
    Ok(NativeOutcome { l1: out.l1, dst: out.dst })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Map;
    use crate::codegen::tenstorrent::riscv::Gather;
    use crate::kernel::{Kernel, OpId};
    use tt_isa::noc::niu::{Endpoint, Niu, TxnId};

    /// The four CB syncs as emitted RISC-V, executed on RISCV B against a
    /// tt-metal `LocalCBInterface` in L1. One page, one push/pop: the counters
    /// end at 1/1 and both pointers wrap back to the page base.
    #[test]
    fn cb_sync_ops_run_on_riscv_b() {
        use crate::codegen::tenstorrent::riscv::{self, Assembler, cb, noc, reg};
        if ensure_open().is_err() {
            eprintln!("cb_sync_ops_run_on_riscv_b: no simulator, skipped");
            return;
        }
        const DST: u64 = 0x4000;
        const FIFO: u32 = 0x1_0000;
        const PAGE: u32 = 64;
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let dst = channel.range(DST, 32).expect("destination range");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("channel 0 endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let me_hi = Endpoint { x, y, addr: 0 }.hi();

        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let iface_at = program_at + 0x1000;
        let args_at = program_at + 0x2000;

        let mut a = Assembler::new();
        // t0 = the interface base (arg 0), t1 = the DRAM destination (arg 1).
        a.push(riscv::lw(reg::T0, reg::A0, 0));
        a.push(riscv::lw(reg::T1, reg::A0, 4));
        cb::reserve_back(&mut a, reg::T0, 0, 1, [reg::T2, reg::T3, reg::T4]);
        cb::push_back(&mut a, reg::T0, 0, 1, [reg::T2, reg::T3, reg::T4, reg::T5]);
        cb::wait_front(&mut a, reg::T0, 0, 1, [reg::T2, reg::T3, reg::T4]);
        cb::pop_front(&mut a, reg::T0, 0, 1, [reg::T2, reg::T3, reg::T4, reg::T5]);
        noc::open(&mut a, reg::T5, Niu::Noc0);
        noc::issue(&mut a, reg::T5, noc::Addr::Reg(reg::T0), me_hi, noc::Addr::Reg(reg::T1), hi, 0, 64, true, reg::T2);
        noc::drain(&mut a, reg::T5, TxnId::new(0).expect("transaction 0"), reg::T2);
        a.push(riscv::RET);
        let words = a.finish();

        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        let iface = cb::interface(FIFO, PAGE, PAGE, 1);
        let mut iface_bytes: Vec<u8> = iface.iter().flat_map(|w| w.to_le_bytes()).collect();
        iface_bytes.resize(64, 0);
        l1.push((iface_at, iface_bytes));
        let mut args = Vec::new();
        for word in [iface_at, DST as u32] {
            args.extend(word.to_le_bytes());
        }
        l1.push((args_at, args));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));

        let out = exec(ExecRun { tile: (x, y), dram: vec![], l1, read_back: vec![dst], l1_read_back: vec![] })
            .expect("the CB program runs");
        assert_eq!(out.error, 0, "executor error word");
        let got: Vec<u32> = out.dram[0].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, vec![PAGE, FIFO + PAGE, PAGE, 1, FIFO, FIFO, (1 << 16) | 1, 0], "CB interface after one page");
    }

    /// One NOC read into L1 and one NOC write back, at a size that is a whole
    /// number of 64-byte NOC words: the transfer path's shape, without a kernel
    /// in between. `N = 128` is two words — the case a page wider than one
    /// word exercises.
    #[test]
    fn a_128_byte_move_round_trips() {
        use crate::codegen::tenstorrent::riscv::{self, Assembler, noc, reg};
        if ensure_open().is_err() {
            eprintln!("a_128_byte_move_round_trips: no simulator, skipped");
            return;
        }
        const N: u32 = 128;
        const SRC: u64 = 0x1000;
        const DST: u64 = 0x4000;
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let src = channel.range(SRC, u64::from(N)).expect("source range");
        let dst = channel.range(DST, u64::from(N)).expect("destination range");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("channel 0 endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let me_hi = Endpoint { x, y, addr: 0 }.hi();

        let data: Vec<u8> = (0..N).map(|i| (i * 7) as u8).collect();
        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let page = program_at + 0x1000;
        let args_at = program_at + 0x2000;

        let mut a = Assembler::new();
        noc::open(&mut a, reg::A7, Niu::Noc0);
        noc::issue(&mut a, reg::A7, noc::Addr::Imm(SRC as u32), hi, noc::Addr::Imm(page), me_hi, 0, N, false, reg::T0);
        noc::drain(&mut a, reg::A7, TxnId::new(0).expect("transaction 0"), reg::T1);
        noc::issue(&mut a, reg::A7, noc::Addr::Imm(page), me_hi, noc::Addr::Imm(DST as u32), hi, 0, N, true, reg::T0);
        noc::drain(&mut a, reg::A7, TxnId::new(0).expect("transaction 0"), reg::T1);
        a.push(riscv::RET);
        let words = a.finish();

        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));

        let out = exec(ExecRun { tile: (x, y), dram: vec![(src, data.clone())], l1, read_back: vec![dst], l1_read_back: vec![] })
            .expect("the move program runs");
        assert_eq!(out.error, 0, "executor error word");
        assert_eq!(out.dram[0], data, "128 bytes out");
    }

    /// RV32M on the executor's core: `a0 = 3 * 5`.    #[test]
    fn rv32m_multiply_runs_on_riscv_b() {
        use crate::codegen::tenstorrent::riscv::{self, Assembler, reg};
        if ensure_open().is_err() {
            eprintln!("rv32m_multiply_runs_on_riscv_b: no simulator, skipped");
            return;
        }
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let mut a = Assembler::new();
        a.li(reg::T0, 3);
        a.li(reg::T1, 5);
        a.push(riscv::mul(reg::A0, reg::T0, reg::T1));
        a.li(reg::T0, 7);
        a.li(reg::T1, 3);
        a.push(riscv::divu(reg::A1, reg::T0, reg::T1));
        a.push(riscv::add(reg::A0, reg::A0, reg::A1));
        a.push(riscv::RET);
        let words = a.finish();
        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, 0, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));
        let out = exec(ExecRun { tile: (x, y), dram: vec![], l1, read_back: vec![], l1_read_back: vec![] })
            .expect("the RV32M program runs");
        assert_eq!(out.error, 0, "executor error word");
        assert_eq!(out.ret, 15 + 2, "3*5 + 7/3");
    }

    /// `issue_addr` with the high words from registers: a DRAM read to L1 and
    /// a write back, exactly as the boundary emitter uses it.
    #[test]
    fn noc_issue_with_register_high_words_runs_on_riscv_b() {
        use crate::codegen::tenstorrent::riscv::{self, Assembler, noc, reg};
        if ensure_open().is_err() {
            eprintln!("noc_issue_with_register_high_words_runs_on_riscv_b: no simulator, skipped");
            return;
        }
        const SRC: u64 = 0x1000;
        const DST: u64 = 0x4000;
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let src = channel.range(SRC, 64).expect("source");
        let dst = channel.range(DST, 64).expect("destination");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let me_hi = Endpoint { x, y, addr: 0 }.hi();
        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let args_at = program_at + 0x2000;
        let page = program_at + 0x1000;
        let mut a = Assembler::new();
        // a1 = source lo, a2 = hi, a3 = destination lo, a4 = page.
        a.push(riscv::lw(reg::A1, reg::A0, 0));
        a.push(riscv::lw(reg::A2, reg::A0, 4));
        a.push(riscv::lw(reg::A3, reg::A0, 8));
        a.li(reg::A4, page);
        a.li(reg::A5, me_hi);
        noc::open(&mut a, reg::A7, Niu::Noc0);
        noc::issue_addr(
            &mut a,
            reg::A7,
            noc::Addr::Reg(reg::A1),
            noc::Addr::Reg(reg::A2),
            noc::Addr::Reg(reg::A4),
            noc::Addr::Reg(reg::A5),
            0,
            64,
            false,
            reg::T0,
        );
        noc::drain(&mut a, reg::A7, TxnId::new(0).expect("transaction 0"), reg::T0);
        noc::issue_addr(
            &mut a,
            reg::A7,
            noc::Addr::Reg(reg::A4),
            noc::Addr::Reg(reg::A5),
            noc::Addr::Reg(reg::A3),
            noc::Addr::Reg(reg::A2),
            0,
            64,
            true,
            reg::T0,
        );
        noc::drain(&mut a, reg::A7, TxnId::new(0).expect("transaction 0"), reg::T0);
        a.push(riscv::RET);
        let words = a.finish();
        let data: Vec<u8> = (0..64u32).map(|i| (i * 3) as u8).collect();
        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        let mut args = Vec::new();
        for w in [SRC as u32, hi, DST as u32] {
            args.extend(w.to_le_bytes());
        }
        l1.push((args_at, args));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));
        let out = exec(ExecRun { tile: (x, y), dram: vec![(src, data.clone())], l1, read_back: vec![dst], l1_read_back: vec![] })
            .expect("the NOC program runs");
        assert_eq!(out.error, 0, "executor error word");
        assert_eq!(out.dram[0], data, "round trip");
    }

    /// The pure-transfer gather on ttsim: one B program, no CB, no compute, no
    /// writer. Reads the index span, picks the token in L1, reads the weight
    /// row, writes it to the output row (tt-metal's loopback with an index).
    #[test]
    fn gather_transfer_program_runs_on_riscv_b() {
        use crate::codegen::tenstorrent::boundary;
        use crate::kernel::{MemScope, Op};
        if ensure_open().is_err() {
            eprintln!("gather_transfer_program_runs_on_riscv_b: no simulator, skipped");
            return;
        }
        let (k, cbs, params) = gather_pipeline(false);
        assert!(cbs.is_empty(), "a pure transfer has no circular buffers");
        // Place the L1 pages, 64-byte aligned.
        let mut l1_at = 0x1_0000u32;
        let mut pages = Map::default();
        let mut id = k.head;
        while !id.is_null() {
            if let Op::Storage { scope: MemScope::Local, dtype, len, .. } = k.at(id) {
                pages.insert(id, l1_at);
                l1_at += u32::try_from(*len).unwrap() * (dtype.bit_size() as u32 / 8);
                l1_at = (l1_at + 63) & !63;
            }
            id = k.next_op(id);
        }
        let words = boundary::reader(&k, &cbs, &params, &pages);

        const WEIGHT: u64 = 0x1000;
        const INDICES: u64 = 0x2000;
        const OUT: u64 = 0x3000;
        const S: u32 = 3;
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let w_range = channel.range(WEIGHT, 4 * 64).expect("weight");
        let i_range = channel.range(INDICES, 4 * 8).expect("indices");
        let o_range = channel.range(OUT, 4 * 64).expect("output");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let me_hi = Endpoint { x, y, addr: 0 }.hi();
        let mut weight = Vec::new();
        for r in 0..4u32 {
            for _ in 0..16 {
                weight.extend((r as f32 + 1.0).to_le_bytes());
            }
        }
        let mut indices = Vec::new();
        for i in [1i64, 0, 3, 2] {
            indices.extend(i.to_le_bytes());
        }
        let mut args = Vec::new();
        for w in [S, 0, 0, me_hi, WEIGHT as u32, hi, INDICES as u32, hi, OUT as u32, hi] {
            args.extend(w.to_le_bytes());
        }
        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let args_at = program_at + 0x2000;
        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        l1.push((args_at, args));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));
        let run = exec(ExecRun {
            tile: (x, y),
            dram: vec![(w_range, weight), (i_range, indices)],
            l1,
            read_back: vec![o_range],
            l1_read_back: vec![],
        })
        .expect("the transfer program runs");
        assert_eq!(run.error, 0, "executor error word");
        // idx[3] = 2, so row 3 becomes weight row 2 (all 3.0).
        let mut expected = vec![0u8; 4 * 64];
        for v in 0..16 {
            expected[3 * 64 + v * 4..3 * 64 + v * 4 + 4].copy_from_slice(&3.0f32.to_le_bytes());
        }
        assert_eq!(run.dram[0], expected, "the gathered row");
    }

    /// The gather through the pass after the gate: `tt_dataflow_select` and the
    /// rest of the device epilogue. What the gather still needs shows up here.
    #[test]
    fn gather_passes_dataflow_select() {
        let (mut k, _, _) = gather_pipeline(false);
        k.tt_dataflow_select();
        k.tt_sfpu_fusion();
        k.tt_lreg_alloc();
        k.tt_dst_reuse();
        k.tt_reg_sync();
        k.tt_verify();
        if std::env::var_os("ZYX_ACCEPT_DUMP").is_some() {
            k.debug();
        }
    }

    /// A pure transfer renders as one mover program: no roles, no DmRead/
    /// DmWrite lists, no CB interface.
    #[test]
    fn gather_transfer_renders_as_one_program() {
        use crate::kernel::{Op, TTOp};
        let (k, _, _) = gather_pipeline(false);
        let rendered = k.render_tt().expect("a transfer renders");
        let (mut reader, mut dm, mut roles) = (None, 0usize, 0usize);
        let mut id = rendered.head;
        while !id.is_null() {
            match rendered.at(id) {
                Op::TT(TTOp::Reader(p)) => reader = Some(p.words.to_vec()),
                Op::TT(TTOp::DmRead(_) | TTOp::DmWrite(_)) => dm += 1,
                Op::TT(TTOp::Unpack(_) | TTOp::Math(_) | TTOp::Pack(_)) => roles += 1,
                _ => {}
            }
            id = rendered.next_op(id);
        }
        assert!(reader.is_some_and(|w| !w.is_empty()), "the transfer program exists");
        assert_eq!(dm, 0, "no boundary lists");
        assert_eq!(roles, 0, "no role programs");
    }

    /// End to end: the custom gather kernel compiles on `Dev::TT(0)`, the
    /// backend runs its transfer program, and the output tensor holds the
    /// gathered rows.
    #[test]
    fn gather_kernel_runs_through_the_backend() {
        use crate::kernel::{Dev, Kernel, Op};
        use crate::{DType, Tensor};
        if ensure_open().is_err() {
            eprintln!("gather_kernel_runs_through_the_backend: no simulator, skipped");
            return;
        }
        let dev = Dev::TT(0);
        if dev.info().is_err() {
            return;
        }
        let mut k = Kernel::new(dev);
        let weight = k.param(DType::F32);
        let idx = k.param(DType::I64);
        let out = k.param_mut(DType::F32);
        let rows = k.const_idx(4i64);
        let s = k.group_range(0, rows);
        let short = k.const_idx(16i64);
        let h = k.push_back(Op::Unroll { axis: 0, len: 16 });
        let token = k.load(idx, s);
        let row = k.mul(token, short);
        let at = k.add(row, h);
        let v = k.load(weight, at);
        let out_row = k.mul(s, short);
        let out_at = k.add(out_row, h);
        k.store(out, v, out_at);
        let compiled = k.compile().expect("the gather compiles");

        let mut w = Vec::new();
        for r in 0..4u32 {
            for _ in 0..16 {
                w.push(r as f32 + 1.0);
            }
        }
        let wt = Tensor::from(w).reshape([4i64, 16i64]).expect("weight").to(dev).expect("upload");
        let it = Tensor::from(vec![1i64, 0, 3, 2]).reshape([4i64]).expect("indices").to(dev).expect("upload");
        let y = compiled.forward(&[&wt, &it], vec![[4i64, 16i64]]).expect("the gather runs").remove(0);
        let got: Vec<f32> = y.to(Dev::C).expect("read back").try_into().expect("f32");
        let mut expected = Vec::new();
        for r in [1usize, 0, 3, 2] {
            for _ in 0..16 {
                expected.push(r as f32 + 1.0);
            }
        }
        assert_eq!(got, expected, "the gathered rows");
    }

    /// The writer program emitted from the gather's boundary IR: the CB page
    /// written to the output row the grid point names. Runs on NCRISC.
    #[test]
    fn gather_writer_program_runs_on_riscv_b() {
        use crate::codegen::tenstorrent::boundary;
        use crate::kernel::Op;
        if ensure_open().is_err() {
            eprintln!("gather_writer_program_runs_on_riscv_b: no simulator, skipped");
            return;
        }
        let (k, cbs, params) = gather_pipeline(true);
        let words = boundary::writer(&k, &cbs, &params, &Map::default());

        const OUT: u64 = 0x3000;
        const ROW0: u32 = 0x1_0000;
        let iface_at = ROW0 + 0x4000;
        let args_at = iface_at + 0x1000;
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let out = channel.range(OUT, 4 * 64).expect("output range");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let me_hi = Endpoint { x, y, addr: 0 }.hi();

        // Interfaces: cb0 (idx) 64 B, cb1 (row) 1024 B, cb2 (out) 64 B. The out
        // CB already holds one page.
        let mut iface = Vec::new();
        let mut fifo = ROW0;
        let mut out_fifo = 0;
        for cb in 0..cbs.len() as u32 {
            let st = cbs.iter().find(|(_, v)| **v == cb).map(|(k, _)| *k).expect("cb");
            let page = match k.at(st) {
                Op::Storage { dtype, len, .. } => u32::try_from(*len).expect("len") * (dtype.bit_size() as u32 / 8),
                _ => unreachable!(),
            };
            let mut words = crate::codegen::tenstorrent::riscv::cb::interface(fifo, page, page, 1);
            if cb == 2 {
                out_fifo = fifo;
                // One page received, none acked: the writer's wait passes.
                words[6] = 1 << 16;
            }
            for w in words {
                iface.extend(w.to_le_bytes());
            }
            fifo += page;
        }
        let mut args = Vec::new();
        for w in [2u32, 0, iface_at, me_hi, 0, hi, 0, hi, OUT as u32, hi] {
            args.extend(w.to_le_bytes());
        }
        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        l1.push((iface_at, iface));
        l1.push((args_at, args));
        let row: Vec<u8> = (0..16u32).flat_map(|i| (i as f32 + 10.0).to_le_bytes()).collect();
        l1.push((out_fifo, row.clone()));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));

        let run = exec(ExecRun { tile: (x, y), dram: vec![], l1, read_back: vec![out], l1_read_back: vec![] })
            .expect("the writer program runs");
        assert_eq!(run.error, 0, "executor error word");
        let mut expected = vec![0u8; 4 * 64];
        expected[2 * 64..3 * 64].copy_from_slice(&row);
        assert_eq!(run.dram[0], expected, "the output row");
    }

    /// The gather kernel brought to the boundary IR, with its parameters and
    /// circular buffers numbered.
    fn gather_pipeline(square: bool) -> (Kernel, Map<OpId, u32>, Map<OpId, u32>) {
        use crate::kernel::{Dev, Kernel, MemScope, Op, OpId};
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(std::sync::Arc::new(crate::backend::DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![i64::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [crate::backend::DTypeCapability::all(); crate::DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: true,
            tile: [i64::from(32), i64::from(32)],
            tile_sizes: vec![[i64::from(32), i64::from(32)]],
            mma_variants: vec![],
            num_circular_buffers: 8,
            has_openmp: false,
        }));
        let weight = k.param(crate::DType::F32);
        let idx = k.param(crate::DType::I64);
        let out = k.param_mut(crate::DType::F32);
        let rows = k.const_idx(4i64);
        let s = k.group_range(0, rows);
        let short = k.const_idx(16i64);
        let h = k.push_back(Op::Unroll { axis: 0, len: 16 });
        let token = k.load(idx, s);
        let row = k.mul(token, short);
        let at = k.add(row, h);
        let v = k.load(weight, at);
        // `square` makes the store's value a computed one, so the boundary is
        // the row/CB path (reader publishes into a CB, compute, writer drains)
        // instead of the pure transfer. `v * v` never folds to `v`.
        let v = if square { k.mul(v, v) } else { v };
        let out_row = k.mul(s, short);
        let out_at = k.add(out_row, h);
        k.store(out, v, out_at);
        k.tt_tile();
        k.tt_unfold();
        k.tt_reindex();
        k.default_epilogue();
        k.tt_add_movers();
        k.tt_sync_cbs();
        k.tt_dedup_pushes();
        k.tt_place_pops();
        k.tt_fifo_check();
        let mut params: Map<OpId, u32> = Map::default();
        let mut cbs: Map<OpId, u32> = Map::default();
        let mut id = k.head;
        while !id.is_null() {
            match k.at(id) {
                Op::Param { .. } => {
                    params.insert(id, params.len() as u32);
                }
                Op::Storage { scope: MemScope::Circular, .. } if !cbs.contains_key(&id) => {
                    cbs.insert(id, cbs.len() as u32);
                }
                _ => {}
            }
            id = k.next_op(id);
        }
        (k, cbs, params)
    }

    /// The reader program emitted from the gather's boundary IR: index span,
    /// token picked in L1, 16 per-lane row spans into the row CB page. Runs on
    /// RISCV B; the page is read back from L1.
    #[test]
    fn gather_reader_program_runs_on_riscv_b() {
        use crate::Map;
        use crate::codegen::tenstorrent::boundary;
        use crate::kernel::{Dev, Kernel, MemScope, Op, OpId};
        if ensure_open().is_err() {
            eprintln!("gather_reader_program_runs_on_riscv_b: no simulator, skipped");
            return;
        }
        // The row/CB fixture of `ir_contract`: out[s, h] = weight[idx[s], h],
        // squared so the store's value is computed and the boundary is the
        // reader/compute/writer path, not a pure transfer.
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(std::sync::Arc::new(crate::backend::DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![i64::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [crate::backend::DTypeCapability::all(); crate::DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: true,
            tile: [i64::from(32), i64::from(32)],
            tile_sizes: vec![[i64::from(32), i64::from(32)]],
            mma_variants: vec![],
            num_circular_buffers: 8,
            has_openmp: false,
        }));
        let weight = k.param(crate::DType::F32);
        let idx = k.param(crate::DType::I64);
        let out = k.param_mut(crate::DType::F32);
        let rows = k.const_idx(4i64);
        let s = k.group_range(0, rows);
        let short = k.const_idx(16i64);
        let h = k.push_back(Op::Unroll { axis: 0, len: 16 });
        let token = k.load(idx, s);
        let row = k.mul(token, short);
        let at = k.add(row, h);
        let v = k.load(weight, at);
        let v = k.mul(v, v);
        let out_row = k.mul(s, short);
        let out_at = k.add(out_row, h);
        k.store(out, v, out_at);
        k.tt_tile();
        k.tt_unfold();
        k.tt_reindex();
        k.default_epilogue();
        k.tt_add_movers();
        k.tt_sync_cbs();
        k.tt_dedup_pushes();
        k.tt_place_pops();
        k.tt_fifo_check();

        let mut params: Map<OpId, u32> = Map::default();
        let mut cbs: Map<OpId, u32> = Map::default();
        let mut id = k.head;
        while !id.is_null() {
            match k.at(id) {
                Op::Param { .. } => {
                    params.insert(id, params.len() as u32);
                }
                Op::Storage { scope: MemScope::Circular, .. } if !cbs.contains_key(&id) => {
                    cbs.insert(id, cbs.len() as u32);
                }
                _ => {}
            }
            id = k.next_op(id);
        }
        let words = boundary::reader(&k, &cbs, &params, &Map::default());
        if std::env::var_os("ZYX_TT_DUMP").is_some() {
            eprintln!("reader program: {} words", words.len());
            for (i, w) in words.iter().enumerate() {
                eprintln!("  {i:4}: {w:08x}");
            }
        }

        const WEIGHT: u64 = 0x1000;
        const INDICES: u64 = 0x2000;
        const ROW0: u32 = 0x1_0000;
        let page_of = |cb: u32| -> u32 {
            let st = cbs.iter().find(|(_, v)| **v == cb).map(|(k, _)| *k).expect("cb");
            match k.at(st) {
                Op::Storage { dtype, len, .. } => (u32::try_from(*len).expect("len")) * (dtype.bit_size() as u32 / 8),
                _ => unreachable!(),
            }
        };
        let iface_at = ROW0 + 0x4000;
        let args_at = iface_at + 0x1000;
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let w_range = channel.range(WEIGHT, 4 * 64).expect("weight");
        let i_range = channel.range(INDICES, 4 * 8).expect("indices");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let me_hi = Endpoint { x, y, addr: 0 }.hi();

        // weight rows: row r is 16 f32 all equal to r + 1.
        let mut weight_bytes = Vec::new();
        for r in 0..4u32 {
            for _ in 0..16 {
                weight_bytes.extend((r as f32 + 1.0).to_le_bytes());
            }
        }
        // indices: 1, 0, 3, 2.
        let mut idx_bytes = Vec::new();
        for i in [1i64, 0, 3, 2] {
            idx_bytes.extend(i.to_le_bytes());
        }
        // CB interfaces: one page each.
        let mut iface = Vec::new();
        let mut fifo = ROW0;
        for cb in 0..cbs.len() as u32 {
            let page = page_of(cb);
            let words = crate::codegen::tenstorrent::riscv::cb::interface(fifo, page, page, 1);
            for w in words {
                iface.extend(w.to_le_bytes());
            }
            fifo += page;
        }
        let row_cb = 1u32; // the second Circular storage: the row page
        let row_page = ROW0 + page_of(0);
        let mut args = Vec::new();
        // grid x, grid y, iface, me_hi, then each param's lo/hi.
        for w in [0u32, 0, iface_at, me_hi, WEIGHT as u32, hi, INDICES as u32, hi, 0, hi] {
            args.extend(w.to_le_bytes());
        }

        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        l1.push((iface_at, iface));
        l1.push((args_at, args));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, args_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));

        let out_run = exec(ExecRun {
            tile: (x, y),
            dram: vec![(w_range, weight_bytes), (i_range, idx_bytes)],
            l1,
            read_back: vec![],
            l1_read_back: vec![(row_page, page_of(row_cb))],
        })
        .expect("the reader program runs");
        assert_eq!(out_run.error, 0, "executor error word");
        let got: Vec<f32> = out_run.l1[0].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        // Token 1, so every 16-datum span of the page is row 1.
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, 2.0, "row page datum {i}");
        }
    }

    /// One row-per-64-bytes gather: `dst[i] = src[index[i]]`.
    ///
    /// The first staged-program run: RISCV B executes code zyx emitted, with no
    /// interpreter in between. The rows are 64 bytes so every read is congruent
    /// mod 64; the indices are out of order; nothing is tiled.
    #[test]
    fn gather_writeback_runs_on_riscv_b() {
        if ensure_open().is_err() {
            eprintln!("gather_writeback_runs_on_riscv_b: no simulator, skipped");
            return;
        }
        const ROW: u32 = 64;
        const ROWS: u32 = 4;
        const SRC: u64 = 0x1000;
        const DST: u64 = 0x4000;
        let channel = Dram::FULL.channel(0).expect("channel 0");
        let src = channel.range(SRC, u64::from(ROW * ROWS)).expect("source range");
        let dst = channel.range(DST, u64::from(ROW * ROWS)).expect("destination range");
        let endpoint = channel.endpoint(Niu::Noc0, 0).expect("channel 0 endpoint");
        let hi = Endpoint { x: endpoint.x(), y: endpoint.y(), addr: 0 }.hi();
        let tile = tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>().next().expect("a Tensix tile");
        let (x, y) = (tile.x(), tile.y());

        // Four distinguishable rows.
        let mut src_bytes = Vec::new();
        for row in 0..ROWS {
            src_bytes.extend((0..ROW).map(|i| (row * 16 + i / 4) as u8));
        }
        // Read them in this order: 2, 0, 3, 1.
        let order: [u32; ROWS as usize] = [2, 0, 3, 1];

        let program_at = tt_isa::l1::PROGRAM_CACHE.base as u32;
        let page = program_at + 0x1000;
        let idx_at = program_at + 0x2000;
        let gather = Gather {
            niu: Niu::Noc0,
            txn: TxnId::new(0).expect("transaction 0"),
            src: SRC as u32,
            dst: DST as u32,
            src_hi: hi,
            dst_hi: hi,
            me_hi: Endpoint { x, y, addr: 0 }.hi(),
            page,
            idx: idx_at,
            count: ROWS,
            stride_shift: ROW.trailing_zeros(),
            row_bytes: ROW,
            tag: 0,
        };
        let words = gather.program();

        let mut l1 = Vec::new();
        l1.push((program_at, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()));
        l1.push((idx_at, order.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>()));
        let mut control = Vec::new();
        for word in [program_at, words.len() as u32, idx_at, 0, 0, 0, 0] {
            control.extend(word.to_le_bytes());
        }
        l1.push((tt_isa::mailbox::exec::BASE as u32, control));

        let out =
            exec(ExecRun { tile: (x, y), dram: vec![(src, src_bytes.clone())], l1, read_back: vec![dst], l1_read_back: vec![] })
                .expect("the gather program runs");
        assert_eq!(out.error, 0, "executor error word");
        let mut expected = Vec::new();
        for row in order {
            expected.extend_from_slice(&src_bytes[row as usize * ROW as usize..(row as usize + 1) * ROW as usize]);
        }
        assert_eq!(out.dram[0], expected, "gathered rows");
    }
}

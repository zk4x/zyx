// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
//! The native Tenstorrent runtime: one owner thread holding the device.
//!
//! `tt_ttsim::Simulator` is `!Send + !Sync` (the simulator library is
//! process-global and a transport borrows it), so the device can never leave
//! the thread that opened it. Everything in zyx parks on a channel instead:
//! this thread opens the simulator and the `Device` once, then answers
//! [`Cmd::Run`] with one [`tt_kernels::runtime::run`]. A launch is
//! synchronous — it returns only once every role has reported `DONE` — so the
//! caller sees the same blocking semantics the old shim had.
//!
//! Nothing here decides what to run: the roles, the loops, the staged L1
//! blocks and the ranges to read back all come from the staged
//! [`crate::backend::tenstorrent::TTNativeProgram`]. This module is transport
//! and nothing else.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::thread;

use super::dev::Device;
use super::kern::code::Loop;
use super::kern::runtime::{self, Schedule, SemaphoreInit};
use super::sim::{self, Simulator};
use tt_isa::isa::Instruction;
use tt_isa::noc::{Noc0, NocCoord};

/// The Tensix tile every zyx kernel runs on.
///
/// One tile: `runtime::run` programs a single core, and grid dispatch is a
/// separate step (a launch is one tile's worth of work today).
const TILE: (u8, u8) = (3, 4);

/// Simulated cycles one run may take before it is declared stuck. A run that
/// does not reach `DONE` in this many clocks is a bug in the program, not
/// slowness.
const BUDGET: u64 = 400_000;

/// One launch of a native program: what the owner thread needs to start the
/// three roles and collect their results.
pub(crate) struct NativeRun {
    /// T0/T1/T2 role programs.
    pub(crate) roles: [Vec<Instruction>; 3],
    /// Host-side block repeats, per role.
    pub(crate) loops: [Vec<Loop>; 3],
    /// L1 blocks written before any role starts: `(address, bytes)`.
    pub(crate) stage: Vec<(u64, Vec<u8>)>,
    /// L1 ranges read after every role finished: `(address, bytes)`.
    pub(crate) read_back: Vec<(u64, usize)>,
    /// Semaphore initial states. Empty runs the roles in order without any
    /// synchronisation (`Schedule::InOrder`).
    pub(crate) sems: Vec<SemaphoreInit>,
}

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
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Cmd::Run(run, reply) => {
                let _ = reply.send(execute(&mut dev, &run));
            }
        }
    }
}

/// One run: hand the roles to `runtime::run` and collect what it read back.
fn execute(dev: &mut Device<sim::LibTtsim<'_>>, run: &NativeRun) -> Result<NativeOutcome, String> {
    if std::env::var_os("ZYX_TT_DUMP").is_some() {
        for (role, words) in run.roles.iter().enumerate() {
            eprintln!("[tt] role T{role}: {} words", words.len());
            for (pc, insn) in words.iter().enumerate() {
                eprintln!("  {pc:4} {:08x} {}", insn.word(), insn.def().mnemonic());
            }
        }
        for (at, bytes) in &run.stage {
            eprintln!("[tt] stage {at:#x}: {} bytes", bytes.len());
        }
    }
    let tile = NocCoord::<Noc0>::new(TILE.0, TILE.1).ok_or_else(|| format!("{TILE:?} is not a Tensix coordinate"))?;
    let roles: [&[Instruction]; 3] = [&run.roles[0], &run.roles[1], &run.roles[2]];
    let loops: [&[Loop]; 3] = [&run.loops[0], &run.loops[1], &run.loops[2]];
    let stage: Vec<(u64, &[u8])> = run.stage.iter().map(|(at, bytes)| (*at, bytes.as_slice())).collect();
    let schedule = if run.sems.is_empty() {
        Schedule::InOrder
    } else {
        Schedule::Concurrent(&run.sems)
    };
    let kernel = runtime::Kernel {
        roles,
        schedule,
        stage: &stage,
        read_back: &run.read_back,
        loops,
        ..runtime::Kernel::new(roles, schedule)
    };
    let out = runtime::run(dev, tile, &tt_firmware_images::ROLES, &kernel, BUDGET).map_err(|e| e.to_string())?;
    Ok(NativeOutcome { l1: out.l1, dst: out.dst })
}

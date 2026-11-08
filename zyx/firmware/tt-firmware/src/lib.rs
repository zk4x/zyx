//! Bare-metal runtime for the Blackhole baby RISC-V cores.
//!
//! # What this runtime does not have
//!
//! * **No `ebreak` on panic.** `ebreak` halts the core and requires an external
//!   agent to resume it, so it cannot back Rust's abort path — a panicking core
//!   would simply stop, indistinguishable from one that never started. Instead
//!   [`panic`] publishes [`status::PANICKED`] and a code to the L1 mailbox, then
//!   spins, so the host can report *why* rather than timing out.
//! * **No `fence.i`.** Zifencei is not implemented and `fence.i` executes as a
//!   `nop`. This runtime never needs it: code is written to L1 before reset is
//!   released, and leaving reset invalidates the instruction cache
//!   (`SoftReset.md:116`). Self-modifying code, or reloading a running core, would
//!   need `RISCV_IC_INVALIDATE_InvalidateAll` in Tensix backend config.
//! * **No core identity at run time.** `mhartid` reads zero on every core and
//!   `misa` lies, so which core this is comes from the link script.

#![no_std]

use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

use tt_isa::mailbox::{self, panic_code, status};

// Entry point.
//
// Sets up the stack, zeroes `.bss`, then calls `main`. Written in assembly
// because none of it is expressible in Rust: there is no stack yet.
core::arch::global_asm!(
    r#"
    .section .text.start, "ax"
    .global _start
    .type _start, @function
_start:
    // Stack lives in local data RAM. The core's own accesses stall automatically
    // while that RAM zeroes itself after reset, so no delay is needed here.
    la      sp, _stack_top

    // Zero .bss. The linker aligns both symbols to 4, so a word loop is exact.
    la      t0, __bss_start
    la      t1, __bss_end
1:
    bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
2:
    call    {main}

    // `main` is diverging, so this is unreachable. Spin rather than fall into
    // whatever follows in L1.
3:
    j       3b
    .size _start, . - _start
"#,
    main = sym crate::main_trampoline,
);

extern "Rust" {
    /// Provided by the binary. Diverging: there is nothing to return to.
    fn firmware_main() -> !;
}

/// Bridges the assembly prologue to the binary's `firmware_main`.
///
/// # Safety
///
/// Called exactly once, from `_start`, with a valid stack.
#[no_mangle]
unsafe extern "C" fn main_trampoline() -> ! {
    noc::reset();
    set_status(status::RUNNING);
    unsafe { firmware_main() }
}

/// Write a word to this tile's L1.
///
/// # Safety
///
/// `offset` must be within L1 and 4-byte aligned.
#[inline]
pub unsafe fn l1_write32(offset: u64, value: u32) {
    unsafe { write_volatile(offset as *mut u32, value) }
}

/// Read a word from this tile's L1.
///
/// # Safety
///
/// As [`l1_write32`].
#[inline]
pub unsafe fn l1_read32(offset: u64) -> u32 {
    unsafe { read_volatile(offset as *const u32) }
}

extern "C" {
    /// Where this image's mailbox is, set per binary by `build.rs` from
    /// `tt_isa::mailbox::MAILBOX_BASE` (Tensix) or `tt_isa::eth::MAILBOX_BASE`
    /// (Ethernet) -- generated, so the host and the image cannot disagree. Only its
    /// address means anything.
    static __mailbox: u8;
}

/// The address of mailbox word `offset` (one of `tt_isa::mailbox::offset`).
#[inline]
pub fn mailbox_word(offset: u64) -> u64 {
    // Taking the address of a linker-defined symbol reads nothing.
    (core::ptr::addr_of!(__mailbox) as u64) + offset
}

/// Publish a status word, ensuring it is visible to the host.
pub fn set_status(value: u32) {
    // SAFETY: the mailbox is a fixed, aligned location inside L1.
    unsafe { l1_write32(mailbox_word(mailbox::offset::STATUS), value) };
    publish();
}

/// `cfg0` (CSR `0x7c0`, `TensixTile/BabyRISCV/CSRs.md`): the baby core's chicken bits.
const CFG0_DIS_BP: u32 = 1 << 1;
/// `cfg0.DisTriscCache`: when clear, adjacent `.ttinsn` instructions are
/// macro-fused into one push of up to four Tensix instructions.
const CFG0_DIS_TRISC_CACHE: u32 = 1 << 18;

/// Turn instruction gathering off on this core, as tt-metal's firmware does on
/// every RISC (`configure_gathering`, tt-metal#16439: a `REPLAY` record window
/// can be corrupted while its instructions are gathered).
///
/// Blackhole boots with gathering on. It buys nothing here: this repo's
/// firmware pushes Tensix instructions with stores, not `.ttinsn`, and
/// measuring a `.ttinsn` push with gathering on or off gave identical cycles
/// (`docs/plans/ttinsn-push-experiment.md`). Off removes the hazard for free.
///
/// The sequence is tt-metal's: set `DisBp`, set `DisTriscCache`, clear `DisBp`,
/// with fences. `cfg0` is then read back, and a core where the bit did not
/// stick publishes [`panic_code::GATHERING_STILL_ON`]
/// instead of running with an unchecked assumption.
///
/// Tensix baby cores only (T0, T1, T2, B, NC): the CSR is documented for them,
/// not for the Ethernet tile's.
pub fn disable_gathering() {
    let cfg0: u32;
    // SAFETY: the custom CSR the vendored `CSRs.md` lists as `cfg0`; the same
    // sequence tt-metal runs at the start of every RISC's firmware.
    unsafe {
        core::arch::asm!(
            "csrrs zero, 0x7c0, {dis_bp}",
            "fence",
            "csrrs zero, 0x7c0, {dis_cache}",
            "csrrc zero, 0x7c0, {dis_bp}",
            "fence",
            "csrr  {cfg0}, 0x7c0",
            dis_bp = in(reg) CFG0_DIS_BP,
            dis_cache = in(reg) CFG0_DIS_TRISC_CACHE,
            cfg0 = out(reg) cfg0,
            options(nostack),
        );
    }
    if cfg0 & CFG0_DIS_TRISC_CACHE == 0 {
        fail(panic_code::GATHERING_STILL_ON);
    }
}

/// Make prior stores visible to observers outside this core.
///
/// The L0 data cache is not coherent — nothing written by the NoC, the unpackers,
/// the packers, or another baby invalidates it (`MemoryOrdering.md:59`) — and
/// store-then-load of non-overlapping ranges has no ordering guarantee in either
/// direction (`MemoryOrdering.md:54`). A `fence` is therefore not decoration: it is
/// what makes a heartbeat the host can see.
#[inline]
pub fn publish() {
    // SAFETY: `fence` has no operands and no memory-safety consequences.
    unsafe { core::arch::asm!("fence", options(nostack, preserves_flags)) };
}

/// Publish a result and mark the firmware finished.
pub fn finish(result: u32) {
    // SAFETY: fixed aligned mailbox locations.
    unsafe { l1_write32(mailbox_word(mailbox::offset::RESULT), result) };
    set_status(status::DONE);
}

/// Stop, having told the host why.
pub fn fail(code: u32) -> ! {
    // SAFETY: fixed aligned mailbox locations.
    unsafe {
        l1_write32(mailbox_word(mailbox::offset::PANIC_CODE), code);
        l1_write32(mailbox_word(mailbox::offset::STATUS), status::PANICKED);
    }
    publish();
    spin()
}

/// The low 32 bits of this core's own cycle counter (`mcycle`,
/// `TensixTile/BabyRISCV/CSRs.md:23`; every baby core has it, Ethernet's
/// included, `EthernetTile/BabyRISCV/README.md:18`).
/// For a core with no timestamper -- an Ethernet tile has none -- this is its
/// only clock. 32 bits wrap every ~3.2 s at 1.35 GHz, far longer than anything
/// it times: take differences with `wrapping_sub`. ttsim models none of the
/// counter CSRs and treats a read as fatal (divergence row 71), so only call
/// this where the host has said it is on silicon.
pub fn cycles() -> u32 {
    let lo: u32;
    // SAFETY: a CSR read, no side effects. By number, so the assembler needs
    // no Zicntr feature to accept it.
    unsafe { core::arch::asm!("csrr {lo}, 0xb00", lo = out(reg) lo, options(nomem, nostack)) };
    lo
}

/// Halt without halting the core.
///
/// A real halt would need `ebreak`, which cannot be resumed without an external
/// agent. Spinning keeps the core in a state the host can still inspect.
///
/// Deliberately *not* `core::hint::spin_loop()`. On RISC-V that emits `pause`,
/// which is Zihintpause — an extension `InstructionSet.md` does not list among
/// those Blackhole implements, putting it in `UndefinedBehavior` territory with no
/// illegal-instruction trap to catch it. ttsim rejects it outright
/// (`rv32_fence: fence_mode=0x10`), which is how this was found. An empty
/// assembly block keeps the loop from being optimised away without emitting
/// anything.
pub fn spin() -> ! {
    loop {
        // SAFETY: empty, with no operands and no memory effects.
        unsafe { core::arch::asm!("", options(nomem, nostack, preserves_flags)) };
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // The message is discarded: formatting it needs an allocator-free `Write`
    // implementation and several KiB of code, on a core with 4 KiB of stack. The
    // panic *location* is what debugging actually needs, and the code below stands
    // in for it until there is a reason to carry more.
    let code = if info.location().is_some() {
        panic_code::EXPLICIT
    } else {
        panic_code::ARITHMETIC
    };
    fail(code)
}

/// Pushing Tensix instructions from a baby RISC-V core.
pub mod corpus;
pub mod dataflow;
/// Run-time translation of a staged program into looping `.ttinsn` code
/// (`docs/plans/ttinsn-push-experiment.md`): the role runners call it when a
/// program's block repeats make the exchange pay, so a loop becomes a branch
/// instead of the body written out.
pub mod ttinsn;

pub mod tensix {
    use tt_isa::sfpu::Instruction;
    use tt_isa::tensix::{self, PushesTo, TensixThread};

    /// Push one Tensix instruction into this core's Tensix thread.
    ///
    /// A push is a plain `sw` of the instruction word to `INSTRN_BUF_BASE`
    /// (`PushTensixInstruction.md:5`). If the FIFO is full the core stalls
    /// automatically, so there is nothing to check.
    ///
    /// # Why `sw` rather than `.ttinsn`
    ///
    /// `.ttinsn IMM32` is the same push expressed as an instruction in the stream —
    /// the word is `IMM32` rotated left by two — which lets the T0/T1/T2
    /// instruction caches fuse up to four adjacent pushes into a single cycle.
    /// That is a throughput optimisation, and it costs something here: the fused
    /// words live in the compressed-instruction encoding space, so they disassemble
    /// as garbage, and the instruction-set gate in `crates/tt-tests/build.rs` would
    /// have to stop rejecting undecodable instructions to let them through. The
    /// encoding is implemented and tested in `tt_isa::sfpu`; using it belongs with
    /// the rest of the performance work, when there is a measurement to justify it.
    ///
    /// # Which core, and which thread
    ///
    /// `C` is the core this image runs on and `Th` is the Tensix thread to reach.
    /// Both are type parameters rather than arguments because the buffer address
    /// depends on the *pair*, and two of the nine combinations hang the RISC-V
    /// unrecoverably (`PushTensixInstruction.md:5-9`). `tt_isa::tensix::PushesTo`
    /// is implemented for exactly the six that do not, so a hanging push is a
    /// compile error rather than a lockup. `C` cannot be inferred -- core identity
    /// is not discoverable at run time -- so every call site names it, which is the
    /// point.
    ///
    /// # Safety
    ///
    /// The caller is responsible for the Tensix-side state the instruction assumes
    /// — in particular that the backend is out of soft reset, since instructions
    /// issued while it is held "might or might not be silently discarded" — and for
    /// `C` actually being the core executing this code. Nothing can check the
    /// latter: `mhartid` reads zero everywhere.
    #[inline]
    pub unsafe fn push<C, Th>(instruction: Instruction)
    where
        Th: TensixThread,
        C: PushesTo<Th>,
    {
        unsafe {
            core::ptr::write_volatile(
                <C as PushesTo<Th>>::INSTRN_BUF as *mut u32,
                instruction.word(),
            )
        }
    }

    /// Push a raw instruction word.
    ///
    /// The corpus firmware needs this: its program arrives as words in L1, encoded
    /// by the host, so there is no [`Instruction`] to push. Everything the word
    /// means was decided on the host side, where the generated table is.
    ///
    /// # Safety
    ///
    /// As [`push`], and additionally the word must be a valid encoding — nothing
    /// here checks it.
    #[inline]
    pub unsafe fn push_word<C, Th>(word: u32)
    where
        Th: TensixThread,
        C: PushesTo<Th>,
    {
        unsafe { core::ptr::write_volatile(<C as PushesTo<Th>>::INSTRN_BUF as *mut u32, word) }
    }

    /// Block until the Tensix coprocessor has retired every instruction this
    /// thread pushed.
    ///
    /// Required before reading `Dst`, and not optional: a push counts as processed
    /// once it reaches a FIFO rather than once it executes
    /// (`PushTensixInstruction.md:27`), and Auto TTSync does not cover `Dst`
    /// (`AutoTTSync.md:56-65`). Without this, a read of `Dst` races the store that
    /// was supposed to fill it.
    ///
    /// # Why this is one assembly block
    ///
    /// Two separate constraints, both from `ManualTTSync.md`, and neither
    /// expressible in Rust:
    ///
    /// * The store before the load is what orders the earlier instruction pushes
    ///   against the load. Its value is discarded; ordering is its only effect.
    /// * The load blocks *inside the memory subsystem*, which does not by itself
    ///   stop the core. The `andi` consuming the result is what actually waits.
    ///   It is also what satisfies the load-adjacency hazard: after starting this
    ///   load, the core must not start another load anywhere in
    ///   `PC_BUF_BASE ..= +0xFFFF` — all eight Tensix semaphores included — until
    ///   it finishes, on pain of wrong values or a hung core. The alternative to a
    ///   consuming ALU instruction is seven intervening instructions.
    ///
    /// Splitting these across statements would let the compiler schedule something
    /// between them, so the sequence is emitted as one block.
    #[inline]
    pub fn wait_for_coprocessor() {
        // SAFETY: the address is fixed; the block has no side effects beyond the
        // documented synchronisation, and clobbers only the scratch registers it
        // declares.
        unsafe {
            core::arch::asm!(
                "sw   zero, 0({addr})",
                "lw   {tmp}, 0({addr})",
                "andi {tmp}, {tmp}, 0",
                addr = in(reg) tensix::COPROCESSOR_DONE_CHECK as u32,
                tmp = out(reg) _,
                options(nostack),
            )
        }
    }

    /// Block until this thread's MOP Expander has no `MOP` queued or in
    /// expansion, so its configuration may change (`ManualTTSync.md:40-55`).
    /// The same store, load and consuming `andi` as [`wait_for_coprocessor`],
    /// for the same two reasons.
    #[inline]
    pub fn wait_for_mop_expander() {
        // SAFETY: as `wait_for_coprocessor`, on the expander's check word.
        unsafe {
            core::arch::asm!(
                "sw   zero, 0({addr})",
                "lw   {tmp}, 0({addr})",
                "andi {tmp}, {tmp}, 0",
                addr = in(reg) tensix::MOP_EXPANDER_DONE_CHECK as u32,
                tmp = out(reg) _,
                options(nostack),
            )
        }
    }

    /// Load this thread's `MopCfg` (`tt_isa::frontend::mop`): wait for the
    /// expander to be idle, write the nine words, and fence so they land before
    /// any later push reaches it.
    pub fn load_mop_config(words: &[u32; 9]) {
        wait_for_mop_expander();
        for (k, &w) in words.iter().enumerate() {
            // SAFETY: the write-only `MopCfg` window of this core's thread;
            // the expander is idle (above).
            unsafe {
                core::ptr::write_volatile((tensix::MOP_CFG_BASE + 4 * k as u64) as *mut u32, w)
            };
        }
        crate::publish();
    }

    /// Read a 32-bit `Dst` element through the RISCV mapping.
    ///
    /// Valid only when `RISC_DEST_ACCESS_CTRL_SEC[thread].fmt` selects one of the
    /// 32-bit shapes (0 or 1), which is the reset default. For the 16-bit shapes a
    /// word access misbehaves and `lhu` is required.
    ///
    /// # Safety
    ///
    /// `address` must come from [`tt_isa::sfpu::dst32_address`], and the
    /// coprocessor must have retired the instruction that wrote it — see
    /// [`wait_for_coprocessor`].
    #[inline]
    pub unsafe fn read_dst32(address: u64) -> u32 {
        unsafe { core::ptr::read_volatile(address as *const u32) }
    }
}

/// Reading and writing Tensix backend configuration from a baby RISC-V core.
pub mod cfg {
    use tt_isa::cfg::{ConfigBank, ConfigField, ThreadConfigField};

    /// Read the word containing `field`.
    ///
    /// # Safety
    ///
    /// The Tensix backend must be out of soft reset.
    #[inline]
    pub unsafe fn read_config_word(field: ConfigField, bank: ConfigBank) -> u32 {
        unsafe { core::ptr::read_volatile(field.riscv_address(bank) as *const u32) }
    }

    /// Set `field` to `value`, leaving the rest of its word alone.
    ///
    /// # Why this is a read-modify-write, and what that costs
    ///
    /// RISC-V can write `Config` **only with `sw`** — a whole 32-bit store — so a
    /// bitfield narrower than a word has to be merged in by hand. There is no
    /// atomic form, and the Configuration Unit enforces ordering across all three
    /// Tensix threads regardless of which issued a request, so heavy use from one
    /// thread starves the others. Neither matters for one-time setup before a
    /// kernel runs; both would matter in a loop.
    ///
    /// `WRCFG` and `RMWCIB` are the Tensix-side alternatives, and `RMWCIB` does
    /// the merge in hardware. They belong in a kernel's instruction stream rather
    /// than in a core's prologue, which is why this exists.
    ///
    /// # Safety
    ///
    /// The Tensix backend must be out of soft reset, and no Tensix instruction
    /// that reads this field may be in flight.
    #[inline]
    pub unsafe fn write_config_field(field: ConfigField, bank: ConfigBank, value: u32) {
        let address = field.riscv_address(bank) as *mut u32;
        // SAFETY: the caller guarantees the backend is out of reset; the address
        // comes from the generated field table.
        unsafe {
            let word = core::ptr::read_volatile(address);
            core::ptr::write_volatile(address, field.insert(word, value));
        }
    }

    /// Read a `ThreadConfig` field for `thread`.
    ///
    /// There is no writing counterpart: RISC-V cannot write `ThreadConfig` at all.
    /// `SETC16` is the only way, and it is a Tensix instruction.
    ///
    /// # Safety
    ///
    /// `thread` must be 0, 1 or 2, and the backend must be out of soft reset.
    #[inline]
    pub unsafe fn read_thread_config(field: ThreadConfigField, thread: u32) -> u16 {
        // The entry is 32 bits wide with the value in the low half; read the word
        // and narrow, rather than issuing a 16-bit load against an MMIO region.
        let word =
            unsafe { core::ptr::read_volatile(field.riscv_read_address(thread) as *const u32) };
        field.extract(word as u16)
    }

    /// Which `Config` bank the given Tensix thread is currently using.
    ///
    /// Selected by `ThreadConfig[thread].CFG_STATE_ID_StateID`. Reading it rather
    /// than assuming bank 0 costs one load and means configuration written here
    /// lands where the coprocessor will actually look for it.
    ///
    /// **Not usable against ttsim.** It maps the whole configuration aperture to a
    /// flat `Config` array with the bank hardcoded to zero, and models no
    /// `ThreadConfig` region, so this read is fatal there. Code that must run on
    /// the simulator has to assume bank 0 and say why.
    ///
    /// # Safety
    ///
    /// As [`read_thread_config`].
    #[inline]
    pub unsafe fn active_bank(thread: u32) -> ConfigBank {
        use tt_isa::cfg::generated::thread::CFG_STATE_ID_StateID;
        // SAFETY: delegated to the caller.
        if unsafe { read_thread_config(CFG_STATE_ID_StateID, thread) } == 0 {
            ConfigBank::Bank0
        } else {
            ConfigBank::Bank1
        }
    }
}

/// Issuing NoC requests from this core, through request initiator 0 of either
/// NIU: NoC #0's or NoC #1's, named per request.
pub mod noc {
    use core::ptr::{read_volatile, write_volatile};
    use tt_isa::noc::niu::{
        self, initiator, Batch, Command, InFlight, Niu, RequestError, TxnId,
    };

    const _: () = assert!(niu::NOC1_BASE <= u32::MAX as u64 && niu::NOC0_BASE < niu::NOC1_BASE);

    fn reg(niu: Niu, off: u64) -> *mut u32 {
        (niu.base() + off) as *mut u32
    }

    /// Each NIU's and transaction ID's requests in flight, kept under its cap
    /// so the 8-bit counter `wait` reads cannot wrap
    /// (`tt_isa::noc::niu::InFlight`), and what the cap has cost. Each NIU
    /// counts its own requests. In local data RAM (`sections.x`, `.local`):
    /// touched on every issue and every wait, and a load there is 2 cycles
    /// (`BabyRISCV/README.md:147`). Written by `reset` before `firmware_main`,
    /// since ttsim does not zero the RAM on reset release (divergence row 25).
    #[link_section = ".local"]
    static mut IN_FLIGHT: [[InFlight; 16]; 2] = [[InFlight::new(); 16]; 2];
    #[link_section = ".local"]
    static mut STALLS: Stalls = Stalls {
        count: 0,
        cycles: 0,
        clock: None,
    };
    /// Whether anything has been issued through NoC #1, so `wait` reads its
    /// counters too.
    #[link_section = ".local"]
    static mut NOC1_USED: bool = false;

    /// Requests that waited for room under their ID's cap, and the cycles
    /// they waited by the image's clock (`set_clock`; 0 without one).
    #[derive(Copy, Clone, Debug)]
    pub struct Stalls {
        pub count: u32,
        pub cycles: u32,
        clock: Option<fn() -> u32>,
    }

    fn in_flight(niu: Niu, txn: TxnId) -> &'static mut InFlight {
        // SAFETY: one core runs this firmware, with no interrupts, and every
        // reference made here is dropped before the next is made; the indices
        // are `< 2` by `Niu` and `< 16` by `TxnId`'s construction.
        unsafe { &mut *core::ptr::addr_of_mut!(IN_FLIGHT[niu.index()][txn.index()]) }
    }

    fn stalls_mut() -> &'static mut Stalls {
        // SAFETY: as `in_flight`.
        unsafe { &mut *core::ptr::addr_of_mut!(STALLS) }
    }

    fn noc1_used() -> &'static mut bool {
        // SAFETY: as `in_flight`.
        unsafe { &mut *core::ptr::addr_of_mut!(NOC1_USED) }
    }

    /// Every ID at no requests in flight and the default cap; nothing issued
    /// on NoC #1; no stalls, no clock. Called once, before `firmware_main`.
    pub(crate) fn reset() {
        // SAFETY: as `in_flight`; nothing else runs yet.
        unsafe { core::ptr::addr_of_mut!(IN_FLIGHT).write([[InFlight::new(); 16]; 2]) };
        *stalls_mut() = Stalls {
            count: 0,
            cycles: 0,
            clock: None,
        };
        *noc1_used() = false;
    }

    /// This tile's own coordinate as `niu` names it (`NOC_ID_LOGICAL`,
    /// `x | y << 6`): the return address of a read through it, as tt-metal
    /// reads it per NoC.
    pub fn me(niu: Niu) -> (u8, u8) {
        // SAFETY: a read-only NIU register.
        let v = unsafe { read_volatile(reg(niu, niu::NOC_ID_LOGICAL)) };
        ((v & 0x3F) as u8, ((v >> 6) & 0x3F) as u8)
    }

    /// Time waits for room with `clock` from now on: a core with a usable
    /// counter (RISCV B: the tile's `WALL_CLOCK_L`). Without one, waits are
    /// counted but not timed.
    pub fn set_clock(clock: fn() -> u32) {
        stalls_mut().clock = Some(clock);
    }

    /// What the in-flight caps have cost since `reset`, over every ID.
    pub fn stalls() -> Stalls {
        *stalls_mut()
    }

    fn outstanding(niu: Niu, txn: TxnId) -> u8 {
        // SAFETY: a read-only NIU counter.
        unsafe { read_volatile(reg(niu, niu::reqs_outstanding(txn))) as u8 }
    }

    /// Use in-flight cap `cap` for `txn` on both NIUs from now on: 0 for
    /// `niu::MAX_IN_FLIGHT`, otherwise clamped to `1..=MAX_IN_FLIGHT`.
    pub fn set_cap(txn: TxnId, cap: u32) {
        in_flight(Niu::Noc0, txn).set_cap(cap);
        in_flight(Niu::Noc1, txn).set_cap(cap);
    }

    /// `issue` at `txn`'s cap on `niu`: poll its counter until there is
    /// room, and count the wait if there was one. Out of line, so the issue
    /// path RISCV B's instruction cache (~4 KiB, `probe_icache`) holds is only the compare that decides
    /// to come here.
    #[cold]
    #[inline(never)]
    fn make_room(niu: Niu, txn: TxnId) {
        let st = stalls_mut();
        let t0 = st.clock.map_or(0, |c| c());
        if in_flight(niu, txn).before_issue(|| outstanding(niu, txn)) != 0 {
            st.count = st.count.wrapping_add(1);
            if let Some(c) = st.clock {
                st.cycles = st.cycles.wrapping_add(c().wrapping_sub(t0));
            }
        }
    }

    /// Issue `cmd` from this tile, at coordinate `me`, under `txn`, through
    /// `niu`. A GDDR request on a port `niu` does not own is refused
    /// (`Command::registers`: both NoCs on one endpoint is the SYS-1419 hang).
    ///
    /// First makes room under `txn`'s in-flight cap on `niu`: below it,
    /// nothing is read; at it, the counter is polled until a request completes
    /// (counted in `stalls`). Then waits for the initiator to be free
    /// (`MemoryMap.md`, `NOC_CMD_CTRL`: software must not touch it while the
    /// low bit reads 1), and reads `CMD_CTRL` back afterwards so a later
    /// counter read cannot overtake the issue (`Counters.md:42-43`).
    #[inline(always)]
    pub fn issue(niu: Niu, cmd: &Command, me: (u8, u8), txn: TxnId) -> Result<(), RequestError> {
        let regs = cmd.registers(me, txn, niu)?;
        match niu {
            Niu::Noc0 => issue_on::<false>(&regs, txn),
            Niu::Noc1 => issue_on::<true>(&regs, txn),
        }
        Ok(())
    }

    /// [`issue_on`] for one request of a `tt_isa::noc::niu::DramMove`, its
    /// values as arguments -- registers, on RISC-V -- written straight to the
    /// initiator: no register array built and read back per request. The two
    /// address-middle words and `AT_DATA` are written zero, as
    /// `Command::registers` lays them out.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn issue_dram_on<const NOC1: bool>(
        targ: u32,
        targ_hi: u32,
        ret: u32,
        ret_hi: u32,
        tag: u32,
        ctrl: u32,
        len: u32,
        txn: TxnId,
    ) {
        use initiator::*;
        let niu = if NOC1 { Niu::Noc1 } else { Niu::Noc0 };
        if NOC1 {
            *noc1_used() = true;
        }
        if in_flight(niu, txn).at_cap() {
            make_room(niu, txn);
        }
        // SAFETY: as `issue_on`.
        unsafe {
            while read_volatile(reg(niu, CMD_CTRL)) & 1 != 0 {}
            write_volatile(reg(niu, TARG_ADDR_LO), targ);
            write_volatile(reg(niu, TARG_ADDR_MID), 0);
            write_volatile(reg(niu, TARG_ADDR_HI), targ_hi);
            write_volatile(reg(niu, RET_ADDR_LO), ret);
            write_volatile(reg(niu, RET_ADDR_MID), 0);
            write_volatile(reg(niu, RET_ADDR_HI), ret_hi);
            write_volatile(reg(niu, PACKET_TAG), tag);
            write_volatile(reg(niu, CTRL), ctrl);
            write_volatile(reg(niu, AT_LEN_BE), len);
            write_volatile(reg(niu, AT_DATA), 0);
            write_volatile(reg(niu, CMD_CTRL), 1);
            let _ = read_volatile(reg(niu, CMD_CTRL));
        }
        in_flight(niu, txn).after_issue();
    }

    /// What request initiator 1 of NoC #0 holds for the fast read path
    /// ([`set_read_path`], [`issue_read`]): whether the path is on, and the
    /// two registers that change from request to request but usually not --
    /// the target coordinate and the length -- as one key, `targ_hi | len << 12`,
    /// as last written, or [`UNKNOWN`]. In local data RAM, as the rest of this
    /// module's state is; `set_read_path` writes it before any request reads
    /// it (the mover calls it at the start of every list), since ttsim does not
    /// zero the RAM (divergence row 25).
    #[derive(Copy, Clone)]
    struct ReadPath {
        fast: bool,
        key: u32,
    }

    /// An impossible key (`targ_hi` is 12 bits, `len` at most 16 KiB): "not
    /// known", so the next request writes every register of the initiator.
    const UNKNOWN: u32 = u32::MAX;

    #[link_section = ".local"]
    static mut READ_PATH: ReadPath = ReadPath {
        fast: false,
        key: UNKNOWN,
    };

    fn read_path() -> &'static mut ReadPath {
        // SAFETY: as `in_flight`.
        unsafe { &mut *core::ptr::addr_of_mut!(READ_PATH) }
    }

    /// Whether GDDR reads go through [`issue_read`].
    #[inline(always)]
    pub fn read_fast() -> bool {
        read_path().fast
    }

    /// Choose the read path for the lists from now on: `false` for the path that
    /// writes every register of initiator 0 per request ([`issue_dram_on`]),
    /// `true` for the fast one ([`issue_read`]). Either way the next request on
    /// the fast path finds the initiator's contents unknown and writes all of
    /// it, so nothing an earlier list, process or program left there is relied
    /// on: silicon keeps NIU state between programs.
    #[inline(always)]
    pub fn set_read_path(on: bool) {
        *read_path() = ReadPath {
            fast: on,
            key: UNKNOWN,
        };
    }

    /// Wait for room under `txn`'s in-flight cap on NoC #0, as
    /// [`issue_dram_on`] does before it writes anything: the check the caller
    /// of [`issue_read`] makes before each request.
    #[inline(always)]
    pub fn room_for_read(txn: TxnId) {
        if in_flight(Niu::Noc0, txn).at_cap() {
            make_room(Niu::Noc0, txn);
        }
    }

    /// Request initiator 1's offset from initiator 0.
    const INITIATOR_1: u64 = initiator::STRIDE;

    /// One request of a GDDR read through the fast path -- the arguments of
    /// [`issue_dram_on`] -- tt-metal's `noc_async_read_set_state` and
    /// `_with_state` on one initiator. Initiator 1 (`NIU_BASE + 0x800`) is this
    /// path's alone, so nothing else changes what it holds, and the page keeps
    /// every one of its words as software wrote it (`MemoryMap.md`,
    /// `NOC_CMD_CTRL`: Blackhole does not even write translated coordinates
    /// back) except `NOC_CTRL`'s reserved bits, which hardware may change.
    /// So: the first request after [`set_read_path`] writes every register (the
    /// set-state); a later one writes the target and return addresses, the
    /// target coordinate and length only when either differs from the last
    /// request's, `NOC_CTRL`, and the command. The return coordinate and tag
    /// are the same in every read of a move and of a list (`DramMove`), which
    /// `tt_isa::noc::niu` gates. The accounting is [`issue_dram_on`]'s, except
    /// that the caller makes room under `txn`'s cap first ([`room_for_read`]),
    /// so that this stays a leaf with no frame to save the request in: the
    /// issue is counted after, and `CMD_CTRL` read back.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn issue_read(
        targ: u32,
        targ_hi: u32,
        ret: u32,
        ret_hi: u32,
        tag: u32,
        ctrl: u32,
        len: u32,
        txn: TxnId,
    ) {
        use initiator::*;
        let niu = Niu::Noc0;
        let p = read_path();
        let key = targ_hi | len << 12;
        // SAFETY: initiator 1's registers, MMIO in every Tensix and Ethernet tile,
        // written once it reads free.
        unsafe {
            while read_volatile(reg(niu, INITIATOR_1 + CMD_CTRL)) & 1 != 0 {}
            if p.key != key {
                if p.key == UNKNOWN {
                    write_volatile(reg(niu, INITIATOR_1 + TARG_ADDR_MID), 0);
                    write_volatile(reg(niu, INITIATOR_1 + RET_ADDR_MID), 0);
                    write_volatile(reg(niu, INITIATOR_1 + RET_ADDR_HI), ret_hi);
                    write_volatile(reg(niu, INITIATOR_1 + PACKET_TAG), tag);
                    write_volatile(reg(niu, INITIATOR_1 + AT_DATA), 0);
                }
                write_volatile(reg(niu, INITIATOR_1 + TARG_ADDR_HI), targ_hi);
                write_volatile(reg(niu, INITIATOR_1 + AT_LEN_BE), len);
                p.key = key;
            }
            write_volatile(reg(niu, INITIATOR_1 + TARG_ADDR_LO), targ);
            write_volatile(reg(niu, INITIATOR_1 + RET_ADDR_LO), ret);
            write_volatile(reg(niu, INITIATOR_1 + CTRL), ctrl);
            write_volatile(reg(niu, INITIATOR_1 + CMD_CTRL), 1);
            let _ = read_volatile(reg(niu, INITIATOR_1 + CMD_CTRL));
        }
        in_flight(niu, txn).after_issue();
    }

    /// A run of GDDR reads through the fast path, [`issue_read`] without its per-request costs: the in-flight
    /// room and the initiator's last key live in registers for the run, and `CMD_CTRL` is read back once, when
    /// the run ends or must read a counter (`tt_isa::noc::niu::Batch`), not after each request. The next
    /// request's wait for the initiator to be free reads the same register behind the same store, so it
    /// already orders one request's writes before the next; only the counter reads need the fence.
    /// Every exit from a run calls [`ReadBatch::end`], or the requests it issued are not counted and the
    /// last one is not fenced.
    pub struct ReadBatch {
        batch: Batch,
        key: u32,
        txn: TxnId,
    }

    /// [`Batch::refill`]'s counter polling, with the stall accounting of [`make_room`]. Out of line: the
    /// issue path holds only the test that decides to come here.
    #[cold]
    #[inline(never)]
    fn refill_batch(batch: &mut Batch, txn: TxnId) {
        let st = stalls_mut();
        let t0 = st.clock.map_or(0, |c| c());
        let full = batch.refill(in_flight(Niu::Noc0, txn), fence_initiator_1, || {
            outstanding(Niu::Noc0, txn)
        });
        if full != 0 {
            st.count = st.count.wrapping_add(1);
            if let Some(c) = st.clock {
                st.cycles = st.cycles.wrapping_add(c().wrapping_sub(t0));
            }
        }
    }

    /// Read `CMD_CTRL` of initiator 1 back: the posted stores before it have reached the NIU.
    #[inline(always)]
    fn fence_initiator_1() {
        // SAFETY: a read of initiator 1's command register, as `issue_read` makes.
        unsafe {
            let _ = read_volatile(reg(Niu::Noc0, INITIATOR_1 + initiator::CMD_CTRL));
        }
    }

    impl ReadBatch {
        /// Start a run of reads under `txn`'s cap on NoC #0. The read path must be on ([`read_fast`]).
        #[inline(always)]
        pub fn begin(txn: TxnId) -> Self {
            ReadBatch {
                batch: Batch::begin(in_flight(Niu::Noc0, txn)),
                key: read_path().key,
                txn,
            }
        }

        /// One request: [`issue_read`]'s, with `tag` and `ctrl` known to the caller.
        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        pub fn issue(
            &mut self,
            targ: u32,
            targ_hi: u32,
            ret: u32,
            ret_hi: u32,
            tag: u32,
            ctrl: u32,
            len: u32,
        ) {
            use initiator::*;
            let niu = Niu::Noc0;
            if self.batch.full() {
                refill_batch(&mut self.batch, self.txn);
            }
            let key = targ_hi | len << 12;
            // SAFETY: initiator 1's registers, as `issue_read`.
            unsafe {
                while read_volatile(reg(niu, INITIATOR_1 + CMD_CTRL)) & 1 != 0 {}
                if self.key != key {
                    if self.key == UNKNOWN {
                        write_volatile(reg(niu, INITIATOR_1 + TARG_ADDR_MID), 0);
                        write_volatile(reg(niu, INITIATOR_1 + RET_ADDR_MID), 0);
                        write_volatile(reg(niu, INITIATOR_1 + RET_ADDR_HI), ret_hi);
                        write_volatile(reg(niu, INITIATOR_1 + PACKET_TAG), tag);
                        write_volatile(reg(niu, INITIATOR_1 + AT_DATA), 0);
                    }
                    write_volatile(reg(niu, INITIATOR_1 + TARG_ADDR_HI), targ_hi);
                    write_volatile(reg(niu, INITIATOR_1 + AT_LEN_BE), len);
                    self.key = key;
                }
                write_volatile(reg(niu, INITIATOR_1 + TARG_ADDR_LO), targ);
                write_volatile(reg(niu, INITIATOR_1 + RET_ADDR_LO), ret);
                write_volatile(reg(niu, INITIATOR_1 + CTRL), ctrl);
                write_volatile(reg(niu, INITIATOR_1 + CMD_CTRL), 1);
            }
            self.batch.issued();
        }

        /// End the run: charge what it issued, keep the initiator's key for the next, and fence the
        /// last request before anything reads a counter.
        #[inline(always)]
        pub fn end(self) {
            read_path().key = self.key;
            if self.batch.end(in_flight(Niu::Noc0, self.txn)) {
                fence_initiator_1();
            }
        }
    }

    /// One request of a `tt_isa::noc::niu::HostMove` through NoC #0, under
    /// `txn`: [`issue_dram_on`] with the high address words a host address
    /// needs. Cold: host moves are not the per-tile path.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub fn issue_host(r: tt_isa::noc::niu::HostRequest, txn: TxnId) {
        use initiator::*;
        let niu = Niu::Noc0;
        if in_flight(niu, txn).at_cap() {
            make_room(niu, txn);
        }
        // SAFETY: as `issue_on`.
        unsafe {
            while read_volatile(reg(niu, CMD_CTRL)) & 1 != 0 {}
            write_volatile(reg(niu, TARG_ADDR_LO), r.targ);
            write_volatile(reg(niu, TARG_ADDR_MID), r.targ_mid);
            write_volatile(reg(niu, TARG_ADDR_HI), r.targ_hi);
            write_volatile(reg(niu, RET_ADDR_LO), r.ret);
            write_volatile(reg(niu, RET_ADDR_MID), r.ret_mid);
            write_volatile(reg(niu, RET_ADDR_HI), r.ret_hi);
            write_volatile(reg(niu, PACKET_TAG), r.tag);
            write_volatile(reg(niu, CTRL), r.ctrl);
            write_volatile(reg(niu, AT_LEN_BE), r.len);
            write_volatile(reg(niu, AT_DATA), 0);
            write_volatile(reg(niu, CMD_CTRL), 1);
            let _ = read_volatile(reg(niu, CMD_CTRL));
        }
        in_flight(niu, txn).after_issue();
    }

    /// Issue one request already encoded -- by `Command::registers`, or by a
    /// `tt_isa::noc::niu::DramMove`, which checks a whole move once (the
    /// mover's per-entry path) -- under `txn`, through NoC #1 (`NOC1`) or
    /// NoC #0, fixed at compile time: the move path's copy for NoC #0 has a
    /// constant base and in-flight slot and none of NoC #1's bookkeeping.
    /// `sections.x` places both copies after `.text.hot`.
    ///
    /// First makes room under `txn`'s in-flight cap on the NIU: below it,
    /// nothing is read; at it, the counter is polled until a request completes
    /// (counted in `stalls`). Then waits for the initiator to be free
    /// (`MemoryMap.md`, `NOC_CMD_CTRL`: software must not touch it while the
    /// low bit reads 1), and reads `CMD_CTRL` back afterwards so a later
    /// counter read cannot overtake the issue (`Counters.md:42-43`).
    #[inline(never)]
    pub fn issue_on<const NOC1: bool>(regs: &[(u64, u32); 10], txn: TxnId) {
        let niu = if NOC1 { Niu::Noc1 } else { Niu::Noc0 };
        if NOC1 {
            *noc1_used() = true;
        }
        if in_flight(niu, txn).at_cap() {
            make_room(niu, txn);
        }
        // SAFETY: both NIUs' initiator registers are MMIO in every Tensix and
        // Ethernet tile, and these offsets are inside initiator 0.
        unsafe {
            while read_volatile(reg(niu, initiator::CMD_CTRL)) & 1 != 0 {}
            for &(off, value) in regs {
                write_volatile(reg(niu, off), value);
            }
            write_volatile(reg(niu, initiator::CMD_CTRL), 1);
            let _ = read_volatile(reg(niu, initiator::CMD_CTRL));
        }
        in_flight(niu, txn).after_issue();
    }

    /// Wait until every response-marked request issued under `txn` has
    /// completed, on NoC #0 and, once anything has gone out on it, NoC #1.
    /// Exact, because `issue` never lets more than the cap be in flight on
    /// either. It leaves each ID's room as it was: too small now, which costs
    /// the next issue at the cap one counter read, and keeps this loop the
    /// same as before there was a cap.
    ///
    /// One copy, in `.text.hot` beside the move path that calls it from every
    /// waiting entry kind, rather than inlined at each.
    #[inline(never)]
    #[link_section = ".text.hot"]
    pub fn wait(txn: TxnId) {
        drain(Niu::Noc0, txn);
        if *noc1_used() {
            drain_noc1(txn);
        }
    }

    /// NoC #1's half of [`wait`], out of line: the inlined wait stays one
    /// load and one branch longer than NoC #0's loop alone.
    #[inline(never)]
    fn drain_noc1(txn: TxnId) {
        drain(Niu::Noc1, txn);
    }

    #[inline(always)]
    fn drain(niu: Niu, txn: TxnId) {
        // SAFETY: a read-only NIU counter.
        unsafe { while read_volatile(reg(niu, niu::reqs_outstanding(txn))) & 0xFF != 0 {} }
    }
}

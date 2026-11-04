// The data mover, shared by RISCV B's image (`bin/dm_b.rs`) and RISCV NC's
// (`bin/dm_nc.rs`): each binary defines `M`, its `tt_isa::dm::Mover` -- the
// mailbox, list ring, scratch and trace chunk it uses -- and includes this.
// Protocol and layout are `tt_isa::dm`, which the host shares.

use tt_firmware::{l1_read32, l1_write32, mailbox_word, noc, publish};
use tt_isa::dm::{self, encoded, op, record, Descriptor, Entry, Transform};
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{offset, status};
use tt_isa::noc::niu::{Command, DramMove, Niu, TxnId};

const TXN: TxnId = match TxnId::new(if IS_NC { dm::TXN_NC } else { dm::TXN_B }) {
    Some(t) => t,
    None => panic!(),
};
/// The barrier's own requests, apart from the moves'.
const BARRIER_TXN: TxnId = match TxnId::new(3) {
    Some(t) => t,
    None => panic!(),
};

fn rd(addr: u64) -> u32 {
    // SAFETY: every address used is an aligned word of the mover's mailbox, its
    // list, its scratch slot, or a destination slot `Entry::decode` placed
    // inside L1.
    unsafe { l1_read32(addr) }
}

fn wr(addr: u64, v: u32) {
    // SAFETY: as `rd`, or the timestamper's `TIMESTAMP` register, where a
    // store only appends an event.
    unsafe { l1_write32(addr, v) }
}

/// Whether this image is RISCV NC's. B is a reader and NC the writer
/// (`dm::Mover::permits`): the write path, and with it every use of NoC #1
/// for writes, is compiled into NC's image alone.
const IS_NC: bool = matches!(M.core, tt_isa::tensix::Core::NC);

/// The entry the other direction would run, which this mover refuses
/// (`Mover::permits`, which the host checks lists against too): NC runs
/// writes, waits and credits only, B everything but GDDR writes. Const, so
/// each image drops the other's arms.
const fn reader_only() -> Result<(), u32> {
    if IS_NC {
        Err(dm::error::DIRECTION)
    } else {
        Ok(())
    }
}

/// Where writes go out, and this tile's coordinate as NoC #1 names it: set
/// per list by [`list_settings`], read per write. With `alternate`, `niu`
/// flips after each write entry (`dm::write_noc::ALTERNATE`). NC's only, as
/// its writes are. In local data RAM, as the `noc` module's state is.
#[derive(Copy, Clone)]
struct Writes {
    niu: Niu,
    alternate: bool,
    me1: (u8, u8),
}

#[link_section = ".local"]
static mut WRITES: Writes = Writes {
    niu: Niu::Noc0,
    alternate: false,
    me1: (0, 0),
};

fn writes() -> &'static mut Writes {
    // SAFETY: one core, no interrupts, and no reference outlives its use.
    unsafe { &mut *core::ptr::addr_of_mut!(WRITES) }
}

/// Issue one descriptor's bytes as NIU requests of at most 16 KiB, without
/// waiting for them. Every request's range is a sub-range of the checked
/// descriptor, so it is inside the channel and inside L1, with the congruence
/// preserved. Reads go out on NoC #0; writes on NoC #0 from B, and from NC on
/// the NIU the host chose (`M.at(dm::WRITE_NOC)`), each through the port of
/// its channel that NIU owns nearest the one asked for
/// (`DramChannel::port_for`).
#[link_section = ".text.hot"]
fn issue(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
    if !IS_NC && d.op == op::READ {
        let port = d.range.channel().port_for(Niu::Noc0, d.port);
        return issue_via::<false>(me, d, port);
    }
    // B only reads and NC only writes (`Mover::permits`): the write path is
    // NC's alone, and NC has no read path.
    if IS_NC && d.op != op::READ {
        issue_write(me, d)
    } else {
        Err(dm::error::DIRECTION)
    }
}

/// [`issue`]'s writes, NC's: through the NIU the host chose, out of the read
/// path (`.text.warm`: after the hot code and both NIUs' issue, `sections.x`).
#[link_section = ".text.warm"]
#[inline(never)]
fn issue_write(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
    let w = *writes();
    if w.alternate {
        writes().niu = if w.niu == Niu::Noc0 {
            Niu::Noc1
        } else {
            Niu::Noc0
        };
    }
    if w.niu == Niu::Noc1 {
        issue_via::<true>(w.me1, d, d.range.channel().port_for(Niu::Noc1, d.port))
    } else {
        issue_via::<false>(me, d, d.range.channel().port_for(Niu::Noc0, d.port))
    }
}

/// One descriptor's requests through NoC #1 (`NOC1`) or NoC #0, on `port`.
#[inline(always)]
fn issue_via<const NOC1: bool>(me: (u8, u8), d: Descriptor, port: u8) -> Result<(), u32> {
    let niu = if NOC1 { Niu::Noc1 } else { Niu::Noc0 };
    // The whole descriptor checked once; each request then only encoded.
    let mv = DramMove::new(d.range, port, d.l1, d.op != op::READ, me, TXN, niu)
        .map_err(|_| dm::error::ALIGNMENT)?;
    // B's reads on NoC #0 only, and only when the host chose the fast path
    // (`dm::READ_FAST`): each request then writes just the initiator words
    // that changed (`noc::issue_read`). One loop either way, so the image
    // holds one copy of the move's checks and encoding.
    let fast = !NOC1 && !IS_NC && noc::read_fast();
    for r in mv.words() {
        if fast {
            noc::room_for_read(TXN);
            noc::issue_read(
                r.targ, r.targ_hi, r.ret, r.ret_hi, r.tag, r.ctrl, r.len, TXN,
            );
        } else {
            noc::issue_dram_on::<NOC1>(
                r.targ, r.targ_hi, r.ret, r.ret_hi, r.tag, r.ctrl, r.len, TXN,
            );
        }
    }
    Ok(())
}

/// What the host may change between lists: the in-flight cap, on B the read
/// path (`dm::READ_FAST`) and, on NC, the NIU the writes go out on.
fn list_settings() {
    noc::set_cap(TXN, rd(M.at(dm::IN_FLIGHT_CAP)));
    if !IS_NC {
        noc::set_read_path(rd(M.at(dm::READ_FAST)) != 0);
        return;
    }
    let mode = rd(M.at(dm::WRITE_NOC));
    let w = writes();
    w.niu = if mode == dm::write_noc::NOC1 {
        Niu::Noc1
    } else {
        Niu::Noc0
    };
    w.alternate = mode == dm::write_noc::ALTERNATE;
}

/// The tile counter's low word, which ttsim models (divergence row 54). The
/// clock `noc` times waits for room under the in-flight cap with: read only
/// when a request has to wait.
fn wall_clock() -> u32 {
    rd(tt_isa::tensix::timestamper::WALL_CLOCK_L)
}

/// After a list: if any of its requests waited for room under the in-flight
/// cap, add them where the host reads them (`M.at(dm::THROTTLE_STALLS)`,
/// `THROTTLE_CYCLES`) and, in a traced list, record one `THROTTLE` event with
/// the cycles. `seen` is what was published last. One compare per list when
/// nothing waited.
fn publish_stalls(seen: &mut noc::Stalls, traced: bool) {
    let now = noc::stalls();
    if now.count != seen.count {
        stalled(*seen, now, traced);
        *seen = now;
    }
}

#[cold]
#[inline(never)]
fn stalled(seen: noc::Stalls, now: noc::Stalls, traced: bool) {
    use tt_isa::mailbox::trace as ev;
    let cycles = now.cycles.wrapping_sub(seen.cycles);
    wr(
        M.at(dm::THROTTLE_STALLS),
        rd(M.at(dm::THROTTLE_STALLS)).wrapping_add(now.count.wrapping_sub(seen.count)),
    );
    wr(
        M.at(dm::THROTTLE_CYCLES),
        rd(M.at(dm::THROTTLE_CYCLES)).wrapping_add(cycles),
    );
    trace(traced, ev::THROTTLE, cycles.min(ev::DETAIL_MAX));
}

/// Transpose the tile in the scratch slot into the slot at `dst`: header
/// copied, datum `(r, c)` from `(c, r)`, both in face order -- face `(fr, fc)`
/// of the result is face `(fc, fr)` of the source, transposed.
#[inline(never)]
fn transpose_from_scratch(dst: u64) {
    for w in 0..(dm::TILE_DATA / 4) {
        wr(dst + w * 4, rd(M.scratch + w * 4));
    }
    let src = (M.scratch + dm::TILE_DATA) as *const u32;
    let out = (dst + dm::TILE_DATA) as *mut u32;
    for fr in 0..2usize {
        for fc in 0..2usize {
            let (from, to) = ((fc * 2 + fr) * 256, (fr * 2 + fc) * 256);
            for i in 0..16usize {
                for j in 0..16usize {
                    // SAFETY: both slots are inside L1 (`Entry::decode`), and
                    // indices stay inside their 1024 datums. Plain accesses: the
                    // NoC's writes to the scratch were fenced before this, and
                    // the caller fences these stores before anything reads them.
                    unsafe { *out.add(to + i * 16 + j) = *src.add(from + j * 16 + i) };
                }
            }
        }
    }
}

/// The tile in the scratch slot into the slot at `dst`, its column 0 copied
/// into every column (`dm::op::READ_BROADCAST_COL`): header copied, datum
/// `(r, c)` from `(r, 0)`.
#[inline(never)]
fn broadcast_col0_from_scratch(dst: u64) {
    for w in 0..(dm::TILE_DATA / 4) {
        wr(dst + w * 4, rd(M.scratch + w * 4));
    }
    let (src, out) = (M.scratch + dm::TILE_DATA, dst + dm::TILE_DATA);
    for r in 0..32usize {
        let v = rd(src + dm::face_index(r, 0) as u64 * 4);
        for c in 0..32usize {
            wr(out + dm::face_index(r, c) as u64 * 4, v);
        }
    }
}

/// `dm::op::FILL`: `v` at every datum of the tile slot `dst` outside its
/// first `param & 0xff` rows and `param >> 8` columns (`0` meaning 32).
/// Stores only: the mover moves data and does no arithmetic.
#[inline(never)]
fn fill(v: u32, param: u32, dst: u64, halfwords: bool) {
    let dst = dst + dm::TILE_DATA;
    let rows = dm::fill::extent(param & 0xff) as usize;
    let cols = dm::fill::extent(param >> 8) as usize;
    for r in 0..32usize {
        for c in 0..32usize {
            if r >= rows || c >= cols {
                if halfwords {
                    // SAFETY: Entry::decode checks the whole slot in L1.
                    unsafe { core::ptr::write_volatile((dst + dm::face_index(r, c) as u64 * 2) as *mut u16, v as u16); }
                } else {
                    wr(dst + dm::face_index(r, c) as u64 * 4, v);
                }
            }
        }
    }
    // The stores must reach L1 before the mover's next NoC write reads it.
    publish();
}

/// The generation this mover last posted to the roles (`launch`), which a
/// `KERNEL_WAIT` must name. In local data RAM; zeroed at start (no
/// generation is 0).
#[link_section = ".local"]
static mut LAUNCHED: u32 = 0;

fn launched() -> &'static mut u32 {
    // SAFETY: one core, no interrupts, no reference outlives its use.
    unsafe { &mut *core::ptr::addr_of_mut!(LAUNCHED) }
}

/// Post `generation` to the three resident roles and wait for each to
/// acknowledge it (`dm::op::KERNEL`): [`launch`] then [`await_roles`].
#[inline(never)]
fn kernel(generation: u32, programs: [(u32, u32); 3]) -> Result<(), u32> {
    launch(generation, programs);
    await_roles(generation)
}

/// Point the roles at their programs, if named, and post `generation`
/// (`dm::op::LAUNCH`, and the first half of `KERNEL`). The roles' programs
/// and descriptors were staged by the host; everything this list moved
/// before is already in L1.
#[inline(never)]
fn launch(generation: u32, programs: [(u32, u32); 3]) {
    // Each role's resident program first, if named (checked by
    // `Entry::decode`), so the generation that starts the run finds it.
    for (t, (at, len)) in programs.into_iter().enumerate() {
        if at != 0 {
            let mb = Mailbox::of(t as u32);
            wr(mb.program_addr(), at);
            wr(mb.program_len(), len);
        }
    }
    publish();
    for t in 0..3 {
        wr(Mailbox::of(t).generation(), generation);
    }
    publish();
    *launched() = generation;
    let traced = rd(M.at(dm::TRACE)) != 0;
    trace(traced, tt_isa::mailbox::trace::KICK, generation);
}

/// Wait for each role to acknowledge `generation`, the last one launched
/// (`dm::op::KERNEL_WAIT`, and the second half of `KERNEL`), or report
/// `ROLE` if one panics. Any other generation is `GENERATION`: the roles
/// would never acknowledge it.
///
/// The roles' acknowledgements are stores by other cores, which do not
/// invalidate this core's L0 data cache (`MemoryOrdering.md:59`): every poll
/// goes through a fence.
#[inline(never)]
fn await_roles(generation: u32) -> Result<(), u32> {
    if generation != *launched() {
        return Err(dm::error::GENERATION);
    }
    for t in 0..3 {
        let mb = Mailbox::of(t);
        loop {
            publish();
            if rd(mb.ack()) == generation {
                break;
            }
            if rd(mb.status()) == status::PANICKED {
                return Err(dm::error::ROLE);
            }
        }
    }
    let traced = rd(M.at(dm::TRACE)) != 0;
    trace(traced, tt_isa::mailbox::trace::ROLES_DONE, generation);
    Ok(())
}

/// Record `event` through the tile's timestamper, if this list is traced
/// (`M.at(dm::TRACE)`). One 128-bit event per store, so the mover's events and the
/// role runners' share one stream without interleaving.
fn trace(on: bool, event: u32, detail: u32) {
    use tt_isa::mailbox::trace as ev;
    use tt_isa::tensix::timestamper as ts;
    if on {
        wr(
            ts::TIMESTAMP,
            ts::event_128(ev::token_with(ev::MOVER, event, detail)),
        );
    }
}

/// `dm::op::BARRIER`: one NoC atomic increment of the coordinator's counter
/// (unless `arrive` is false, the wait-only form), then NoC reads of it until
/// it reaches `target`. The reads land in this tile's L1, which the NoC writes
/// without invalidating the L0 cache: every check goes through a fence
/// (`publish`).
#[inline(never)]
fn barrier(me: (u8, u8), target: u32, x: u8, y: u8, arrive: bool) -> Result<(), u32> {
    use tt_isa::noc::niu::Endpoint;
    let counter = Endpoint {
        x,
        y,
        addr: dm::BARRIER_COUNTER as u32,
    };
    if arrive {
        noc::issue(
            Niu::Noc0,
            &Command::AtomicIncrement {
                to: counter,
                value: 1,
                ret_local: M.at(dm::BARRIER_RET) as u32,
            },
            me,
            BARRIER_TXN,
        )
        .map_err(|_| dm::error::ALIGNMENT)?;
        noc::wait(BARRIER_TXN);
    }
    loop {
        noc::issue(
            Niu::Noc0,
            &Command::Read {
                from: counter,
                to_local: M.at(dm::BARRIER_POLL) as u32,
                len: 4,
            },
            me,
            BARRIER_TXN,
        )
        .map_err(|_| dm::error::ALIGNMENT)?;
        noc::wait(BARRIER_TXN);
        publish();
        if (rd(M.at(dm::BARRIER_POLL)).wrapping_sub(target) as i32) >= 0 {
            return Ok(());
        }
    }
}

/// Run one descriptor to completion.
#[link_section = ".text.hot"]
fn run(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
    issue(me, d)?;
    noc::wait(TXN);
    Ok(())
}

/// Whether this image runs host-encoded reads ([`op::READ_ENCODED`]): the `fast-reads`
/// feature, B only. Without it (the default, "safe" mode) the card decodes and checks
/// every read itself, and a `READ_ENCODED` entry is an unknown op that reaches no NIU
/// (`error::OP`). A const, so a safe image carries none of the code below.
const FAST_READS: bool = cfg!(feature = "fast-reads") && !IS_NC;

/// Whether, besides the checksum, each encoded read is run through [`encoded::guard`]:
/// the `encoded-guard` feature (about 50 cycles an entry).
const ENCODED_GUARD: bool = FAST_READS && cfg!(feature = "encoded-guard");

/// What this image tells the host it can do ([`dm::CAPS`]).
const CAPS: u32 = dm::caps::VALID
    | if FAST_READS { dm::caps::FAST_READS } else { 0 }
    | if ENCODED_GUARD { dm::caps::GUARD } else { 0 };

/// `op::READ_ENCODED`: one GDDR read the host validated and encoded
/// (`tt_isa::dm::encoded`). The card verifies the entry's checksum -- damage in
/// transit, an overwritten list, or an entry the encoder did not make -- and issues it
/// as `issue_via` issues the words of a decoded one, with the tag and control word a B
/// read always has. With `encoded-guard` it also runs the semantic guard first.
#[link_section = ".text.hot"]
#[inline(never)]
fn read_encoded(me: (u8, u8), e: [u32; 8]) -> Result<(), u32> {
    reader_only()?;
    if !encoded::verify_with(&e, unsafe { ENCODE_SEED }) {
        return Err(dm::error::ENCODED);
    }
    if ENCODED_GUARD {
        let bit = |hi: u32| rd(M.at(encoded::ENDPOINTS) + (hi as u64 / 32) * 4) >> (hi % 32) & 1 != 0;
        if !encoded::guard(&e, encoded::endpoint_hi(me.0, me.1), bit) {
            return Err(dm::error::ENCODED);
        }
    }
    if noc::read_fast() {
        noc::room_for_read(TXN);
        noc::issue_read(e[1], e[2], e[3], e[4], encoded::TAG, encoded::CTRL, e[7], TXN);
    } else {
        noc::issue_dram_on::<false>(e[1], e[2], e[3], e[4], encoded::TAG, encoded::CTRL, e[7], TXN);
    }
    Ok(())
}

/// A run of host-encoded reads from the ring at `base`, at most `max` entries, stopping at the first
/// entry that is not one: `(entries issued, error code or 0)`. [`read_encoded`] for each, as a loop of
/// its own: the seed, the guard's tile word and the issue's state stay in registers for the run, one
/// `CMD_CTRL` read-back fences the run ([`noc::ReadBatch`]), and no entry is copied through an array.
/// Entries before a bad one are issued, as they would be one at a time; the run is always ended, so
/// they are counted. For the fast read path only, untraced.
#[link_section = ".text.hot"]
#[inline(never)]
fn run_encoded_reads(me: (u8, u8), base: u64, max: u32) -> (u32, u32) {
    let seed = unsafe { ENCODE_SEED };
    let me_hi = encoded::endpoint_hi(me.0, me.1);
    let mut batch = noc::ReadBatch::begin(TXN);
    let mut k = 0u32;
    let mut code = 0;
    while k < max {
        let e = entry_at(base + k as u64 * dm::ENTRY_BYTES);
        if e[0] != op::READ_ENCODED {
            break;
        }
        if !encoded::verify_with(&e, seed) {
            code = dm::error::ENCODED;
            break;
        }
        if ENCODED_GUARD {
            let bit = |hi: u32| rd(M.at(encoded::ENDPOINTS) + (hi as u64 / 32) * 4) >> (hi % 32) & 1 != 0;
            if !encoded::guard(&e, me_hi, bit) {
                code = dm::error::ENCODED;
                break;
            }
        }
        batch.issue(e[1], e[2], e[3], e[4], encoded::TAG, encoded::CTRL, e[7]);
        k += 1;
    }
    batch.end();
    (k, code)
}

/// Run one list entry: plain entries are issued without waiting; a transposed
/// read waits for its own tile before rearranging it; a kernel or a wait entry
/// first waits for everything before it.
#[link_section = ".text.hot"]
fn exec(me: (u8, u8), usable: u32, w: [u32; 8]) -> Result<(), u32> {
    if FAST_READS && w[0] == op::READ_ENCODED {
        return read_encoded(me, w);
    }
    match Entry::decode(usable, w)? {
        Entry::IndexPick { indices, source, output, bound, first, rows, bytes } => {
            reader_only()?;
            noc::wait(TXN);
            publish();
            // Validate every logical index before copying any datum. Padding
            // indices are never read, and a chunk miss performs no access.
            for r in 0..rows {
                let at = dm::face_index(r as usize, 0) as u64;
                let index = rd(indices as u64 + dm::TILE_DATA + at * 4);
                if index >= bound { return Err(dm::error::DOMAIN); }
            }
            for r in 0..rows {
                let dst = dm::face_index(r as usize, 0) as u64;
                let index = rd(indices as u64 + dm::TILE_DATA + dst * 4);
                if index < first || index - first >= 32 { continue; }
                let src = dm::face_index(r as usize, (index - first) as usize) as u64;
                let from = source as u64 + dm::TILE_DATA + src * bytes as u64;
                let to = output as u64 + dm::TILE_DATA + dst * bytes as u64;
                // SAFETY: decode bounds/alignment cover whole slots; r<32 and
                // the checked chunk-relative index<32 bound both face indices.
                unsafe {
                    if bytes == 2 {
                        core::ptr::write_volatile(to as *mut u16, core::ptr::read_volatile(from as *const u16));
                    } else {
                        core::ptr::write_volatile(to as *mut u32, core::ptr::read_volatile(from as *const u32));
                    }
                }
            }
            publish();
        }
        Entry::CopyWords { src, dst, count, src_stride, dst_stride, halfwords } => {
            reader_only()?;
            noc::wait(TXN);
            publish();
            for n in 0..count {
                // SAFETY: decode checks every word lies inside the data arena.
                unsafe {
                    if halfwords {
                        let value = core::ptr::read_volatile((src + n * src_stride) as *const u16);
                        core::ptr::write_volatile((dst + n * dst_stride) as *mut u16, value);
                    } else {
                        let value = core::ptr::read_volatile((src + n * src_stride) as *const u32);
                        core::ptr::write_volatile((dst + n * dst_stride) as *mut u32, value);
                    }
                }
            }
            publish();
        }
        Entry::Buffer {
            stream,
            action,
            capacity,
        } => {
            use tt_isa::dataflow::{Action, Endpoint};
            let (endpoint, actor) = if M == dm::Mover::B {
                (Endpoint::Reader, 0)
            } else {
                (Endpoint::Writer, 4)
            };
            if matches!(action, Action::Push | Action::Pop) {
                noc::wait(TXN);
                publish();
            }
            tt_firmware::dataflow::buffer(stream, action, capacity, endpoint, actor)?;
        }
        Entry::Released { target, which } => {
            reader_only()?;
            tt_firmware::dataflow::released(target, which)?;
        }
        Entry::PairCall {
            reader,
            writer,
            capacity,
        } => {
            reader_only()?;
            let generation_base = unsafe { CALL_BASE };
            pair_call(me, usable, reader, writer, capacity, generation_base)?;
        }
        Entry::Move {
            descriptor,
            transform: transform @ (Transform::Transpose | Transform::BroadcastCol0),
        } => {
            reader_only()?;
            // Everything before it has landed, and the scratch is free.
            noc::wait(TXN);
            run(
                me,
                Descriptor {
                    l1: M.scratch as u32,
                    ..descriptor
                },
            )?;
            publish();
            if transform == Transform::Transpose {
                transpose_from_scratch(descriptor.l1 as u64);
            } else {
                broadcast_col0_from_scratch(descriptor.l1 as u64);
            }
            // Visible in L1 before anything else reads the slot.
            publish();
        }
        Entry::Move { descriptor, .. } => issue(me, descriptor)?,
        Entry::Kernel {
            generation,
            programs,
        } => {
            reader_only()?;
            // The operands it computes on must have landed.
            noc::wait(TXN);
            publish();
            kernel(generation, programs)?;
        }
        Entry::Wait => {
            noc::wait(TXN);
            publish();
        }
        Entry::Barrier {
            target,
            x,
            y,
            arrive,
        } => {
            reader_only()?;
            // Everything this unit moved before it has landed.
            noc::wait(TXN);
            publish();
            barrier(me, target, x, y, arrive)?;
        }
        Entry::Poke { address, value } => {
            reader_only()?;
            // The roles read the word only once a later `KERNEL` posts their
            // generation.
            wr(address as u64, value);
            publish();
        }
        Entry::CheckFlags { slot, rows, cols } => {
            if !IS_NC { return Err(dm::error::DIRECTION); }
            // The packet's ready credit follows pack retirement. Fence the
            // SFPU-generated status tile before validating it; no tensor
            // arithmetic or host intermediate download is involved.
            publish();
            for r in 0..rows {
                for c in 0..cols {
                    let index = dm::face_index(r as usize, c as usize);
                    if rd(slot as u64 + dm::TILE_DATA + index as u64 * 4) != 1 {
                        return Err(dm::error::DOMAIN);
                    }
                }
            }
        }
        // Only as a list of its own, which `run_list_at` runs (B's).
        Entry::Call { .. } => {
            reader_only()?;
            return Err(IS_CALL);
        }
        Entry::Launch {
            generation,
            programs,
        } => {
            reader_only()?;
            // The operands it computes on must have landed, and anything
            // still being written out of the slots it computes into.
            noc::wait(TXN);
            publish();
            launch(generation, programs);
        }
        Entry::KernelWait { generation } => {
            reader_only()?;
            await_roles(generation)?
        }
        Entry::Host {
            write,
            host_lo,
            host_hi,
            l1,
            len,
        } => {
            reader_only()?;
            host_move(me, write, host_lo, host_hi, l1, len)
        }
        Entry::Tilize {
            tilize,
            block,
            stride,
            slot,
            valid,
        } => {
            reader_only()?;
            // The side it reads may still be arriving.
            noc::wait(TXN);
            publish();
            tile_layout(tilize, block as u64, stride as usize, slot as u64, valid);
            // Visible in L1 before anything else reads it.
            publish();
        }
        Entry::Fill { value, param, dst, halfwords } => {
            reader_only()?;
            // The tile it fills may still be arriving.
            noc::wait(TXN);
            publish();
            fill(value, param, dst as u64, halfwords);
        }
    }
    Ok(())
}

/// 16 words from `src` to `dst`: all the loads, then all the stores, so the
/// loads overlap (a call to `memcpy` cost more than the copy).
#[inline(always)]
unsafe fn copy16(src: *const u32, dst: *mut u32) {
    let w: [u32; 16] = [
        core::ptr::read_volatile(src),
        core::ptr::read_volatile(src.add(1)),
        core::ptr::read_volatile(src.add(2)),
        core::ptr::read_volatile(src.add(3)),
        core::ptr::read_volatile(src.add(4)),
        core::ptr::read_volatile(src.add(5)),
        core::ptr::read_volatile(src.add(6)),
        core::ptr::read_volatile(src.add(7)),
        core::ptr::read_volatile(src.add(8)),
        core::ptr::read_volatile(src.add(9)),
        core::ptr::read_volatile(src.add(10)),
        core::ptr::read_volatile(src.add(11)),
        core::ptr::read_volatile(src.add(12)),
        core::ptr::read_volatile(src.add(13)),
        core::ptr::read_volatile(src.add(14)),
        core::ptr::read_volatile(src.add(15)),
    ];
    for (i, v) in w.into_iter().enumerate() {
        core::ptr::write_volatile(dst.add(i), v);
    }
}

/// `dm::op::TILIZE` (`tilize`) or `UNTILIZE`: datum `(r, c)` of the
/// row-major block at `block` (rows `stride` bytes apart) and its place in the
/// tile's faces at `slot`, for the valid rows and columns; a tilize zeroes
/// the rest of the tile, an untilize leaves the rest of the block alone.
#[cold]
#[inline(never)]
fn tile_layout(tilize: bool, block: u64, stride: usize, slot: u64, valid: u32) {
    let rows = dm::fill::extent(valid & 0xff) as usize;
    let cols = dm::fill::extent(valid >> 8) as usize;
    let tile = slot as *mut u32;
    for face in 0..4usize {
        let (fr, fc) = (face / 2, face % 2);
        let n = cols.saturating_sub(16 * fc).min(16);
        for r in 0..16usize {
            let row = 16 * fr + r;
            let rm = (block as usize + row * stride + 64 * fc) as *mut u32;
            let at = face * 256 + r * 16;
            // SAFETY: the block and the slot are inside L1 (`Entry::decode`),
            // and every index stays inside one block row's 32 datums or the
            // tile's 1024. Plain accesses: the moves that filled the source
            // were fenced before this, and the caller fences these stores.
            unsafe {
                let live = if row < rows { n } else { 0 };
                let t = tile.add(at);
                match (tilize, live) {
                    // A whole face row, the common case: 16 words, unrolled.
                    (true, 16) => copy16(rm, t),
                    (false, 16) => copy16(t, rm),
                    (true, _) => {
                        for c in 0..live {
                            *t.add(c) = *rm.add(c);
                        }
                        for c in live..16 {
                            *t.add(c) = 0;
                        }
                    }
                    (false, _) => {
                        for c in 0..live {
                            *rm.add(c) = *t.add(c);
                        }
                    }
                }
            }
        }
    }
}

/// `dm::op::HOST_READ` / `HOST_WRITE`: the move's requests to the PCIe tile,
/// in flight like any other move's.
#[cold]
#[inline(never)]
fn host_move(me: (u8, u8), write: bool, host_lo: u32, host_hi: u32, l1: u32, len: u32) {
    let mv = tt_isa::noc::niu::HostMove {
        host_lo,
        host_hi,
        l1,
        len,
        write,
        me,
        txn: TXN,
    };
    for r in mv.words() {
        noc::issue_host(r, TXN);
    }
}

/// Read list entry `i` of the ring.
fn entry_at(at: u64) -> [u32; 8] {
    let mut w = [0u32; 8];
    for (k, word) in w.iter_mut().enumerate() {
        *word = rd(at + k as u64 * 4);
    }
    w
}

/// Run the list at `M.list`, entry by entry ([`exec`]). An op record
/// (`dm::record`) is expanded here, on the tile, into the entries the host
/// would otherwise have sent, and each runs exactly as a sent one would --
/// through `Entry::decode` and every refusal in it.
///
/// With `M.at(dm::TRACE)` set, the list, and each entry or record in it, is
/// bracketed by timestamper events (`tt_isa::mailbox::trace`); a record's
/// expanded moves are not, so the events per list stay bounded by its length.
fn run_list(me: (u8, u8), usable: u32, count: u32) -> Result<(), u32> {
    run_list_at(me, usable, 0, count)
}

/// [`run_list`] of the `count` entries from ring entry `first`.
fn run_list_at(me: (u8, u8), usable: u32, first: u32, count: u32) -> Result<(), u32> {
    if count > dm::LIST_MAX || first + count > dm::LIST_MAX {
        return Err(dm::error::LENGTH);
    }
    let base = M.list + first as u64 * dm::ENTRY_BYTES;
    // A `CALL` is a list of its own (a replay's), run here rather than from
    // the list loop: there it makes the loop recursive. `exec` checks it, as
    // it does every entry -- `Entry::decode` with a second caller is no longer
    // inlined into `exec`, which costs every entry ~0.04 us
    // (`silicon_perf::mover_read_shapes`).
    let head = entry_at(base);
    let result = if head[0] == op::PAIR {
        if M != dm::Mover::B || tt_isa::dataflow::packet_length(head)? != count as usize {
            return Err(dm::error::LENGTH);
        }
        pair_shared(me, usable, base + dm::ENTRY_BYTES, head[1], head[2])?;
        if head[4] != 0 {
            let trailer = base + (1 + head[1] as u64 + head[2] as u64) * dm::ENTRY_BYTES;
            if rd(trailer) != op::BARRIER {
                return Err(dm::error::OP);
            }
            run_entries(me, usable, trailer, head[4])
        } else {
            Ok(())
        }
    } else if count == 1 && head[0] == op::CALL {
        match exec(me, usable, head) {
            Err(IS_CALL) => call(me, usable, head[1], head[2], head[3], head[4], head[5]),
            r => r,
        }
    } else {
        run_entries(me, usable, base, count)
    };
    // One anywhere else.
    match result {
        Err(IS_CALL) => Err(dm::error::OP),
        r => r,
    }
}

/// What [`exec`] returns for a valid `CALL`, which only [`run_list_at`] runs:
/// no `dm::error` code.
const IS_CALL: u32 = u32::MAX;

static mut CALL_BASE: u32 = 0;
/// The seed of an encoded read's checksum (`encoded::context_seed`): this mover's key, tile,
/// chip and op, set once at boot; flipped by `encoded::TRACE_DOMAIN` while a trace replays,
/// so an encoded read inside a trace fails its check.
static mut ENCODE_SEED: u32 = 0;

/// Enter or leave a trace's domain of [`ENCODE_SEED`] (a safe image has none).
#[inline(always)]
fn flip_trace_domain() {
    if FAST_READS {
        unsafe {
            ENCODE_SEED ^= encoded::TRACE_DOMAIN;
        }
    }
}
static mut PAIR_ACTIVE: bool = false;
static mut TRACE_DEPTH: u32 = 0;

fn begin_pair() -> Result<u32, u32> {
    if unsafe { PAIR_ACTIVE }
        || rd(dm::Mover::NC.mailbox + offset::STATUS) != status::RUNNING
        || rd(dm::Mover::NC.at(dm::SEQ)) != rd(dm::Mover::NC.at(dm::DONE))
        || rd(dm::Mover::NC.at(dm::QUEUE_HEAD)) != rd(dm::Mover::NC.at(dm::QUEUE_DONE))
    {
        return Err(dm::error::PEER);
    }
    unsafe {
        PAIR_ACTIVE = true;
    }
    tt_firmware::dataflow::initialize();
    Ok(rd(dm::Mover::NC.at(dm::DONE)).wrapping_add(1).max(1))
}

fn finish_pair(sequence: u32, result: Result<(), u32>) -> Result<(), u32> {
    let mut poll = tt_firmware::dataflow::Poll::new();
    let result = result.and_then(|()| loop {
        poll.tick()?;
        if rd(dm::Mover::NC.at(dm::DONE)) == sequence {
            let error = rd(dm::Mover::NC.at(dm::ERROR));
            return if error == 0 { Ok(()) } else { Err(error) };
        }
    });
    if let Err(code) = result {
        tt_firmware::dataflow::abort_with(code);
        // NC reads this packet's writer entries out of B's ring and may still
        // be writing GDDR: B's slot is not reported done under it. NC stops at
        // its next credit wait, which sees the abort. Bounded (about 60 ms),
        // since NC may be what hung; the host resets the tile after any
        // failed region either way.
        let mut spins = 0u32;
        while rd(dm::Mover::NC.at(dm::DONE)) != sequence && spins < 1 << 22 {
            publish();
            spins += 1;
        }
    }
    unsafe {
        PAIR_ACTIVE = false;
    }
    result
}

#[cold]
fn pair_shared(
    me: (u8, u8),
    usable: u32,
    reader: u64,
    reader_count: u32,
    writer_count: u32,
) -> Result<(), u32> {
    let sequence = begin_pair()?;
    let peer = dm::Mover::NC;
    wr(peer.at(dm::OP), op::SHARED);
    wr(
        peer.at(dm::L1_ADDR),
        (reader + reader_count as u64 * dm::ENTRY_BYTES) as u32,
    );
    wr(peer.at(dm::LEN), writer_count);
    publish();
    wr(peer.at(dm::SEQ), sequence);
    publish();
    let result = run_entries(me, usable, reader, reader_count);
    finish_pair(sequence, result)
}

#[cold]
fn pair_call(
    me: (u8, u8),
    usable: u32,
    reader: [u32; 3],
    writer: [u32; 3],
    _capacity: u16,
    generation_base: u32,
) -> Result<(), u32> {
    let sequence = begin_pair()?;
    let peer = dm::Mover::NC;
    wr(peer.at(dm::OP), op::CALL);
    wr(peer.at(dm::CHANNEL), writer[0]);
    wr(peer.at(dm::DRAM_OFFSET), writer[1]);
    wr(peer.at(dm::LEN), writer[2]);
    publish();
    wr(peer.at(dm::SEQ), sequence);
    publish();
    let result = call(
        me,
        usable,
        reader[0],
        reader[1],
        reader[2],
        generation_base,
        0,
    );
    finish_pair(sequence, result)
}

/// The `count` entries at `base` in L1, in order: a list's, or a trace chunk's
/// ([`call`]).
#[link_section = ".text.hot"]
#[inline(never)]
fn run_entries(me: (u8, u8), usable: u32, base: u64, count: u32) -> Result<(), u32> {
    use tt_isa::mailbox::trace as ev;
    let mode = rd(M.at(dm::TRACE));
    let traced = mode != 0;
    let outer = traced && unsafe { TRACE_DEPTH == 0 };
    if traced {
        unsafe {
            TRACE_DEPTH += 1;
        }
    }
    trace(outer, ev::LIST_BEGIN, count);
    let result = (|| {
        let at = |i: u64| entry_at(base + i * dm::ENTRY_BYTES);
        let mut i = 0u64;
        while i < count as u64 {
            let head = at(i);
            let n = record::len(head[0]) as u64;
            // `trace_mode::LISTS` leaves the reads out, so they keep the untraced paths.
            let entry_traced = traced
                && head[0] != op::PAIR_CALL
                && !(mode == dm::trace_mode::LISTS && (head[0] == op::READ || head[0] == op::READ_ENCODED));
            trace(entry_traced, ev::ENTRY_BEGIN, head[0]);
            if n == 1 {
                // A run of host-encoded reads is one loop (`run_encoded_reads`), untraced, on the
                // fast read path; past it, each goes straight to its issue, past `exec`'s
                // prologue and decode (`exec` also takes it, for the other callers).
                if FAST_READS && !entry_traced && head[0] == op::READ_ENCODED && noc::read_fast() {
                    let (done, code) =
                        run_encoded_reads(me, base + i * dm::ENTRY_BYTES, (count as u64 - i) as u32);
                    if code != 0 {
                        wr(M.mailbox + offset::RESULT, head[0]);
                        return Err(code);
                    }
                    i += done as u64;
                    continue;
                }
                let r = if FAST_READS && head[0] == op::READ_ENCODED {
                    read_encoded(me, head)
                } else {
                    exec(me, usable, head)
                };
                if let Err(code) = r {
                    wr(M.mailbox + offset::RESULT, head[0]);
                    return Err(code);
                }
            } else {
                if i + n > count as u64 {
                    return Err(dm::error::LENGTH);
                }
                if let Err(code) = run_record(me, usable, base + i * dm::ENTRY_BYTES, n as usize) {
                    wr(M.mailbox + offset::RESULT, head[0]);
                    return Err(code);
                }
            }
            trace(entry_traced, ev::ENTRY_END, head[0]);
            i += n;
        }
        noc::wait(TXN);
        Ok(())
    })();
    if traced {
        unsafe {
            TRACE_DEPTH -= 1;
        }
    }
    if result.is_ok() {
        trace(outer, ev::LIST_END, count);
    }
    result
}

/// One entry of an expanded record: a plain read or write straight to the
/// issue, decoded as a descriptor (`Descriptor::decode`, every refusal
/// `exec` would make), without the general entry path; anything else
/// through [`exec`]. Beside [`issue`] in `.text.hot`: a record's tiles run
/// through here, and the general path was a third of a gathered tile's cost
/// (checklist 9.14).
#[link_section = ".text.hot"]
#[inline(never)]
fn record_entry(me: (u8, u8), usable: u32, e: [u32; 8]) -> Result<(), u32> {
    if e[0] == op::READ || e[0] == op::WRITE {
        issue(
            me,
            Descriptor::decode(usable, e[0], e[1], e[2], e[3], e[4], e[5])?,
        )
    } else {
        exec(me, usable, e)
    }
}

/// The `n`-entry op record at `at`, expanded ([`record::expand`]). Out of the
/// list loop, as the rest of the cold paths are, so the loop and [`exec`]'s
/// move path sit together in `.text.hot` (`sections.x`).
#[inline(never)]
fn run_record(me: (u8, u8), usable: u32, at: u64, n: usize) -> Result<(), u32> {
    let mut rec = [[0u32; 8]; 7];
    for (k, e) in rec[..n].iter_mut().enumerate() {
        *e = entry_at(at + k as u64 * dm::ENTRY_BYTES);
    }
    if rec[0][0] == record::POKE_RUN {
        return poke_run(&rec[..n]);
    }
    // Each mover expands its own direction's records, so the other's code is
    // not in its image.
    if IS_NC {
        record::expand_writes(&rec[..n], |e| record_entry(me, usable, e))
    } else {
        if rec[0][0] == record::SELECT_ROWS {
            // Its indices were read into L1 by entries before it: they have
            // landed, and are visible to this core, before the first is read
            // (as `INDEX_PICK`'s).
            noc::wait(TXN);
            publish();
        }
        record::expand_reads(&rec[..n], |at| rd(at.into()), |e| record_entry(me, usable, e))
    }
}

/// [`record::POKE_RUN`]: a run of consecutive role-mailbox words, written with one
/// fence for the run instead of [`dm::op::POKE`]'s one for each. What the record
/// expands to (`record::expand_reads`, which the host checks the list with) is exactly
/// these stores in this order; the loop is here so a descriptor costs a record rather
/// than seventeen entries.
///
/// The roles read these words only once a later `KERNEL` or `LAUNCH` posts their
/// generation, so one fence after the run is what orders them, as `op::POKE`'s is.
#[inline(never)]
fn poke_run(rec: &[[u32; 8]]) -> Result<(), u32> {
    reader_only()?;
    let count = rec[0][2];
    if count == 0 || count as usize > record::POKE_RUN_MAX {
        return Err(dm::error::RANGE);
    }
    // A value past the run would be a word the host's expansion never writes.
    let mut k = count as usize;
    while k < record::POKE_RUN_MAX {
        if record::poke_run_word(rec, k) != 0 {
            return Err(dm::error::OP);
        }
        k += 1;
    }
    let first = rec[0][1];
    // The run is at most 48 bytes and the mailboxes are 0x4000 apart, so first and
    // last inside a mailbox puts every word of it in that one.
    let last = match first.checked_add(4 * (count - 1)) {
        Some(last) => last,
        None => return Err(dm::error::RANGE),
    };
    if !dm::is_role_mailbox_word(first) || !dm::is_role_mailbox_word(last) {
        return Err(dm::error::OP);
    }
    for k in 0..count {
        wr((first + 4 * k) as u64, record::poke_run_word(rec, k as usize));
    }
    publish();
    Ok(())
}

/// [`dm::op::CALL`]: a trace's entries from GDDR, a chunk at a time into
/// `M.trace_chunk`, each chunk run by the list's own loop ([`run_entries`])
/// once it has landed. What a replay supplies is patched into the chunk first:
/// a top-level `KERNEL`'s generation plus `generation_base`, a `BARRIER`'s
/// target plus `barrier_base`. A `CALL` in a trace is refused, as is a record
/// the capture let cross a chunk.
#[cold]
#[inline(never)]
fn call(
    me: (u8, u8),
    usable: u32,
    channel: u32,
    offset: u32,
    count: u32,
    generation_base: u32,
    barrier_base: u32,
) -> Result<(), u32> {
    let mut done = 0u32;
    let chunk = if M == dm::Mover::B && !unsafe { PAIR_ACTIVE } {
        tt_isa::dataflow::OUTER_CHUNK
    } else {
        M.trace_chunk
    };
    while done < count {
        let n = (count - done).min(dm::TRACE_CHUNK_ENTRIES);
        let read = [
            op::READ,
            channel,
            0,
            offset + done * dm::ENTRY_BYTES as u32,
            chunk as u32,
            n * dm::ENTRY_BYTES as u32,
            0,
            0,
        ];
        if IS_NC {
            let descriptor =
                Descriptor::decode(usable, read[0], read[1], read[2], read[3], read[4], read[5])?;
            issue_via::<true>(
                noc::me(Niu::Noc1),
                descriptor,
                descriptor.range.channel().port_for(Niu::Noc1, 0),
            )?;
        } else {
            exec(me, usable, read)?;
        }
        noc::wait(TXN);
        // The chunk is in L1, written by the NoC past the L0 cache.
        publish();
        let mut i = 0u32;
        while i < n {
            let a = chunk + i as u64 * dm::ENTRY_BYTES;
            let head = rd(a);
            let base = match head {
                op::KERNEL | op::LAUNCH | op::KERNEL_WAIT => generation_base,
                op::BARRIER => barrier_base,
                op::CALL => return Err(dm::error::OP),
                _ => 0,
            };
            if base != 0 {
                wr(a + 4, rd(a + 4).wrapping_add(base));
            }
            i += record::len(head) as u32;
        }
        if i != n {
            return Err(dm::error::LENGTH);
        }
        publish();
        let previous = unsafe { CALL_BASE };
        unsafe {
            CALL_BASE = generation_base;
        }
        flip_trace_domain();
        let result = run_entries(me, usable, chunk, n);
        flip_trace_domain();
        unsafe {
            CALL_BASE = previous;
        }
        result?;
        done += n;
        wr(M.at(dm::TRACE_PROGRESS), rd(M.at(dm::TRACE_PROGRESS)).wrapping_add(1));
    }
    Ok(())
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // Gathering off, as on every role (`tt_firmware::disable_gathering`).
    tt_firmware::disable_gathering();
    // What this image was built to do, before the host's first list.
    wr(M.at(dm::CAPS), CAPS);
    let me = (rd(M.at(dm::MY_X)) as u8, rd(M.at(dm::MY_Y)) as u8);
    // Local data RAM is not zeroed on the simulator (divergence row 25).
    *launched() = 0;
    if IS_NC {
        *writes() = Writes {
            niu: Niu::Noc0,
            alternate: false,
            me1: noc::me(Niu::Noc1),
        };
    }
    let usable = rd(M.at(dm::USABLE));
    if FAST_READS {
        unsafe {
            ENCODE_SEED =
                encoded::context_seed(rd(M.at(dm::ENCODE_KEY)), me, usable, op::READ_ENCODED);
        }
    }
    let mut beat: u32 = 0;
    noc::set_clock(wall_clock);
    let mut stalls = noc::stalls();
    loop {
        beat = beat.wrapping_add(1);
        wr(mailbox_word(offset::HEARTBEAT), beat);
        // The host's descriptor arrives over the NoC, which does not invalidate
        // the L0 data cache (MemoryOrdering.md:59): every poll through a fence.
        publish();
        // Compared against DONE in L1 rather than a local copy, so the whole
        // of the mover's state is the mailbox, which the host can read and reset.
        // The queue first: lists the host enqueued, in order, until one
        // fails (which stops the queue until the host restarts the mover).
        let done = rd(M.at(dm::QUEUE_DONE));
        if done != rd(M.at(dm::QUEUE_HEAD)) && rd(M.at(dm::QUEUE_ERROR)) == dm::error::NONE {
            list_settings();
            let slot = rd(M.at(dm::QUEUE_SLOTS) + (done % dm::QUEUE_LEN) as u64 * 4);
            let result = run_list_at(me, usable, slot & 0xffff, slot >> 16);
            publish_stalls(&mut stalls, rd(M.at(dm::TRACE)) != 0);
            // Everything the list moved has landed before it is reported.
            publish();
            match result {
                Ok(()) => wr(M.at(dm::QUEUE_DONE), done.wrapping_add(1)),
                Err(code) => {
                    wr(M.at(dm::QUEUE_ERROR_AT), done.wrapping_add(1));
                    wr(M.at(dm::QUEUE_ERROR), code);
                }
            }
            publish();
            continue;
        }
        let seq = rd(M.at(dm::SEQ));
        if seq == 0 || seq == rd(M.at(dm::DONE)) {
            continue;
        }
        list_settings();
        let result = if rd(M.at(dm::OP)) == op::SHARED && M == dm::Mover::NC {
            let address = rd(M.at(dm::L1_ADDR)) as u64;
            let count = rd(M.at(dm::LEN));
            if count == 0
                || count > dm::LIST_MAX
                || address % dm::ENTRY_BYTES != 0
                || address < dm::LIST
                || address + count as u64 * dm::ENTRY_BYTES
                    > dm::LIST + dm::LIST_MAX as u64 * dm::ENTRY_BYTES
            {
                Err(dm::error::RANGE)
            } else {
                run_entries(me, usable, address, count)
            }
        } else if rd(M.at(dm::OP)) == op::CALL && M == dm::Mover::NC {
            call(
                me,
                usable,
                rd(M.at(dm::CHANNEL)),
                rd(M.at(dm::DRAM_OFFSET)),
                rd(M.at(dm::LEN)),
                0,
                0,
            )
        } else if rd(M.at(dm::OP)) == op::LIST {
            run_list(me, usable, rd(M.at(dm::LEN)))
        } else {
            Descriptor::decode(
                usable,
                rd(M.at(dm::OP)),
                rd(M.at(dm::CHANNEL)),
                rd(M.at(dm::PORT)),
                rd(M.at(dm::DRAM_OFFSET)),
                rd(M.at(dm::L1_ADDR)),
                rd(M.at(dm::LEN)),
            )
            .and_then(|d| run(me, d))
        };
        publish_stalls(&mut stalls, rd(M.at(dm::TRACE)) != 0);
        if let Err(code) = result {
            if matches!(rd(M.at(dm::OP)), op::SHARED | op::CALL) {
                tt_firmware::dataflow::abort_with(code);
            }
        }
        wr(M.at(dm::ERROR), result.err().unwrap_or(dm::error::NONE));
        // The data is in L1 (a read) or acknowledged by the DRAM tile (a write,
        // response-marked) before DONE is published.
        publish();
        wr(M.at(dm::DONE), seq);
        publish();
    }
}

# tt-firmware

Bare-metal Rust for the Blackhole baby RISC-V cores (`riscv32im-unknown-none-elf`).
A **separate workspace**, excluded from the host workspace so `cargo build` at the
root does not try to build it for the host. Normally built by
`crates/tt-firmware-images/build.rs`, which embeds the images; you rarely build it
by hand.

## Images (`src/bin/`)

| Binary | Core | Link script | Purpose |
|---|---|---|---|
| `role_t0`, `role_t1`, `role_t2` | T0, T1, T2 | `link_role_t0.x`, `link_role_t1.x`, `link_role_t2.x` | The unpack / math / pack role runners. Each in its 10 KiB slot (`tt_isa::l1::roles`), reached through its reset-PC override, so all three and B's mover sit in one tile's L1. Resident in a `Session`. |
| `dm_b` | B | `link_b.x` | The reader (`tt_isa::dm`): GDDR to L1, list entries, reading op records expanded on the tile, padding fills, tilizes, PCIe moves, barriers, `KERNEL` entries that point the resident roles at their programs and run them, and the shared packets that start NC. It never writes GDDR. No arithmetic: the image gate refuses every F-extension instruction, in every image. |
| `dm_nc` | NC | `link_nc.x` | The writer, loaded above the data arena via NC's reset stub: L1 to GDDR on NoC1, writing op records expanded on the tile, and the credits that order them. It runs no read, kernel launch, barrier or PCIe move. |

`dm_b` and `dm_nc` are one source (`src/mover.rs`) compiled twice with a const `M`;
`tt_isa::dm::Mover::permits` is the table of which entries each runs, refused with
`error::DIRECTION` by the host before a list is sent and by the tile if one gets
there anyway. Each image's other direction is compiled out. |
| `eth_e1` | E1 (Ethernet) | `link_e1.x` | The chip-to-chip data mover (`tt_isa::eth::mover`). |
| `corpus` / `corpus_t0` | T1 / T0 | `link.x` | Generic single-thread program runner (T1 for ttsim, T0 for silicon). |

The program runners push from their fixed program slot, or, when
`mailbox::PROGRAM_ADDR` is non-zero, from that address, which must lie in the
program cache region. The host writes a runner's whole descriptor
(`tt_isa::mailbox::Descriptor`) before every run: L1 survives between processes on
silicon, so a field left unwritten is the previous process's.
| `heartbeat`, `sfpu_mul` | T0, or the corpus core | `link.x` | The step 3 and step 4 bring-up gates. |
| `prng_seed` | T1 | `link.x` | Step91 diagnostic: direct configuration seed stores, settling interval and four lane snapshots. |

`src/lib.rs` is the runtime (entry, `.bss`, status/heartbeat/panic words in the L1
mailbox); `src/corpus.rs` is the shared body of the program runners. `build.rs`
picks each binary's link script and passes the mailbox base from `tt-isa` as a
`--defsym`.

`Session::enable_dram(b, nc)` enables GDDR and resident streaming ownership. B receives
one shared reader/writer packet, starts NC locally, and launches the three roles
once per compatible region (a standalone transfer is a packet with no roles: B
reads, NC writes). Tagged role scripts reuse the existing arithmetic
bodies and loops. Input/output credits are released after backend retirement and
DRAM acknowledgment, respectively; B completion joins the entire region. Traces
retain both mover streams and all referenced programs. See
[ownership and rollout](../../docs/learnings/streaming-dataflow-architecture.md).
This is the only GDDR compute scheduler. `set_pipeline(false)` serializes buffer
reuse without changing ownership. Downloads and control lists run on B alone;
low-level diagnostic movers remain available, each limited to its direction.

## Build by hand

```bash
cargo build --release --manifest-path crates/tt-firmware/Cargo.toml
cargo clippy --manifest-path crates/tt-firmware/Cargo.toml \
  --target riscv32im-unknown-none-elf --bins --lib -- -D warnings
```

`.cargo/config.toml` sets the target and `-C target-feature=-c`.

## Gotchas

- **No compressed instructions.** Their encoding space is reused by `.ttinsn`: a
  compressed instruction is a Tensix push, not a smaller instruction.
- **No `ebreak` on panic.** A panic publishes `status::PANICKED` and a code, then
  spins, so the host can say why.
- **No `fence.i`** (Zifencei is unimplemented). Code is written before reset is
  released; leaving reset invalidates the I-cache.
- **No core identity at run time** (`mhartid` reads 0): it comes from the link script.
- `tt-firmware-images/build.rs` disassembles every image and refuses instructions
  Blackhole cannot execute (compressed, `lr`/`sc`, `fdiv`, `fsqrt`, the `fmadd`
  family, `fence.i`, `pause`, unlisted `fence` orderings, undecodable words).
  Executing an unimplemented instruction is undefined behaviour, not a trap.

## Features

| Feature | What it does |
|---|---|
| (none) | The default "safe" images: the B mover decodes and checks every read itself and refuses `READ_ENCODED`. |
| `fast-reads` | The B mover runs host-encoded reads (`tt_isa::dm::encoded`), verifying their checksum; the host validates them in full. Reported to the host in `dm::CAPS`. |
| `encoded-guard` | With `fast-reads`: also `encoded::guard`, an independent semantic check of each encoded read (~40 cycles). |
| `ttinsn-experiment` | The `.ttinsn` push-fusion experiment image (`docs/plans/ttinsn-push-experiment.md`). |

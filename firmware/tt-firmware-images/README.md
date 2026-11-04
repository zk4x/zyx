# tt-firmware-images

The baby RISC-V firmware images, built from `crates/tt-firmware` and embedded as
byte arrays. Shippable, `no_std`, depends only on `tt-isa`. It exists so a backend
gets the role images from a crate it can depend on, not from the test harness.

## What it exports

| Item | Image |
|---|---|
| `ROLES: [Image; 3]` | Unpack (T0), math (T1), pack (T2), each in its slot of `tt_isa::l1::roles`. What `tt_kernels::runtime` and `Session` load. |
| `DM_B: Image` | The RISCV B data mover, loaded at `tt_isa::dm::IMAGE_BASE` (L1 offset 0). |
| `ETH_E1` | The Ethernet E1 mover, loaded at `tt_isa::eth::E1_IMAGE`. |
| `CORPUS`, `CORPUS_T0` | The single-thread program runner for T1 (ttsim) and T0 (silicon). |
| `HEARTBEAT`, `SFPU_MUL` | Bring-up gate images, loaded at `LOAD_ADDRESS`. |
| `PRNG_SEED` | Step91 direct RISC-V seed-store diagnostic on T1, loaded at `LOAD_ADDRESS`; includes a conservative settling interval and lane snapshots. |

`Image` is `(Core, &[u8], load address)`. The ELF entry points are recorded at build
time so each image can be checked against the core it is loaded for.

## Build

Automatic: `build.rs` runs a nested `cargo build --release --target
riscv32im-unknown-none-elf` in `crates/tt-firmware`, with its own target dir under
`OUT_DIR` and the parent's `CARGO_*`/`RUSTFLAGS` stripped. Then, with `llvm-objcopy`
and `llvm-objdump` from `llvm-tools`, it extracts each image and disassembles it.

Needs the `riscv32im-unknown-none-elf` target and the `llvm-tools` component (both
in `rust-toolchain.toml`).

## Gotchas

- **The instruction-set gate is a correctness gate.** `build.rs` fails the build on
  any instruction Blackhole lacks or gets wrong: compressed, `lr`/`sc`, `fdiv`,
  `fsqrt`, the `fmadd` family, `fence.i`, `pause` (`fence w, 0`), any `fence`
  ordering outside the allowed four, and anything that disassembles as `<unknown>`. There is no illegal-instruction
  trap on these cores.
- Rebuilds are driven by per-file `rerun-if-changed` over `tt-firmware` and `tt-isa`
  sources. A firmware edit that does not rebuild is a bug here.

## Features

`fast-reads` and `encoded-guard` build the B mover that runs host-encoded reads (see `tt-firmware`); `FAST_READS` says which
this build is, and the mover reports the same in `dm::CAPS`, which a host reads. Default: the safe image. `ttinsn-experiment`
adds the `.ttinsn` experiment image.

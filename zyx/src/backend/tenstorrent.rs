// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Tenstorrent backend for zyx.
//!
//! This backend compiles zyx kernel IR to tt-metal compute kernels that
//! execute on Tensix RISC-V cores. It uses the low-level C++ kernel API
//! (compute_kernel_api.h), not the high-level ttnn op API.
//!
//! # Architecture
//!
//! Each Tensix core runs 5 RISC-V processors in parallel:
//! - **BRISC** (boot RISC): data movement master, runs the reader kernel
//! - **NCRISC** (NOC RISC): data movement, runs the writer kernel
//! - **TRISC0/1/2** (triplicated compute RISC): unpack, math, pack pipeline
//!
//! A single zyx kernel is compiled to three coordinated tt-metal kernels:
//! 1. **Reader kernel** (BRISC): reads tiles from DRAM into circular buffers (CBs)
//!    via `noc_async_read`. Each input tensor gets a CB.
//! 2. **Compute kernel** (TRISC0/1/2): operates on tiles in DST register file.
//!    Unpack: `cb_wait_front` → copy_tile to DST.
//!    Math:   apply arithmetic (add_tiles, mul_tiles, etc.).
//!    Pack:   pack_tile from DST → `cb_push_back` to output CB.
//! 3. **Writer kernel** (NCRISC): reads output tiles from CB and writes to DRAM
//!    via `noc_async_write`.
//!
//! # Kernel IR Ops (post-lowering)
//!
//! These are the ops present in the `Kernel` after the kernelizer has lowered
//! `Reduce`, `Move`, `ConstView`, `LoadView`, `StoreView`. The CUDA and OpenCL
//! backends only handle these ops and `unreachable!()` on the pre-lowering ones:
//!
//! | Op | Description | tt-metal mapping |
//! |----|-------------|-----------------|
//! | `Op::Define { dtype, scope, ro, len }` | Variable declaration | `Scope::Global` → kernel arg pointer (DRAM addr)<br>`Scope::Local` → L1 buffer / CB slot<br>`Scope::Register` → DST register tile |
//! | `Op::Const(x)` | Constant value | Inlined as literal in compute kernel or pulled via reader |
//! | `Op::Load { src, index, vlen }` | Read from memory | Reader: `noc_async_read(src_addr + index * elem_size, cb_write_ptr, nbytes)` |
//! | `Op::Store { dst, x, index, vlen }` | Write to memory | Writer: `noc_async_write(cb_read_ptr, dst_addr + index * elem_size, nbytes)` |
//! | `Op::Cast { x, dtype }` | Type cast | `typecast_tile(dst_idst, src_idst)` (SFPU) |
//! | `Op::Unary { x, uop }` | Unary op | See unary mapping table |
//! | `Op::Binary { x, y, bop }` | Binary op | See binary mapping table |
//! | `Op::Mad { x, y, z }` | FMA | Sequence of `mul_tiles + add_tiles`, or use Math FMA pipe |
//! | `Op::Wmma { dims, ... }` | Tile matmul | `matmul_tiles` with appropriate block dimensions |
//! | `Op::Vectorize { ops }` | Vector pack | Pack scalar values into vector register (if vlen > 1) |
//! | `Op::Devectorize { vec, idx }` | Vector extract | Extract scalar from vector (for vlen > 1, handle per-component) |
//! | `Op::Index { len, scope, axis }` | Index variable | **NOT** mapped to thread index like CUDA.<br>Tensix has no SIMT threads. Instead, `Op::Index` is treated as a loop induction variable. Each kernel operates on one tile (32×32 elements) at a time. The index determines which tile in the tensor. |
//! | `Op::Loop { len }` | For loop | `for (uint32_t idx = 0; idx < len; idx++)` in the compute kernel |
//! | `Op::EndLoop` | Loop end | Closing brace |
//! | `Op::If { condition }` | Conditional | `if (condition)` |
//! | `Op::EndIf` | End conditional | Closing brace |
//! | `Op::Barrier { scope }` | Sync barrier | TRISC: barrier between unpack/math/pack stages<br>Global: NOC fence (`noc_async_read_barrier`, `noc_async_write_barrier`) |
//!
//! # Unary op mapping
//!
//! | `UOp` | tt-metal API |
//! |-------|-------------|
//! | `Neg` | `negative_tile(idst)` |
//! | `BitNot` | `bitwise_not_tile(idst)` |
//! | `Exp2` | `exp2_tile(idst)` (SFPU) |
//! | `Log2` | `log_tile(idst)` or `log2_tile(idst)` (SFPU) |
//! | `Reciprocal` | `recip_tile(idst)` (SFPU) |
//! | `Sqrt` | `sqrt_tile(idst)` (SFPU) |
//! | `Sin` | `sin_tile(idst)` (SFPU) |
//! | `Cos` | `cos_tile(idst)` (SFPU) |
//! | `Floor` | `floor_tile(idst)` |
//! | `Trunc` | `trunc_tile(idst)` |
//! | `Abs` | `abs_tile(idst)` or `max(copy, -copy)` |
//!
//! # Binary op mapping
//!
//! | `BOp` | tt-metal API |
//! |-------|-------------|
//! | `Add` | `add_tiles(cb_a, cb_b, i_a, i_b, idst)` |
//! | `Sub` | `sub_tiles(cb_a, cb_b, i_a, i_b, idst)` |
//! | `Mul` | `mul_tiles(cb_a, cb_b, i_a, i_b, idst)` |
//! | `Div` | `div_tiles(cb_a, cb_b, i_a, i_b, idst)` |
//! | `Pow` | `pow_tile(idst, exponent)` |
//! | `Mod` | `binary_mod_tile(...)` or SFPU |
//! | `Cmplt` | `binary_comp_tile(cb_a, cb_b, i_a, i_b, idst, COMPARE_LT)` |
//! | `Cmpgt` | `binary_comp_tile(cb_a, cb_b, i_a, i_b, idst, COMPARE_GT)` |
//! | `Max` | `binary_max_tile(cb_a, cb_b, i_a, i_b, idst)` |
//! | `Or` | `binary_bitwise_or_tile(cb_a, cb_b, i_a, i_b, idst)` |
//! | `And` | `binary_bitwise_and_tile(cb_a, cb_b, i_a, i_b, idst)` |
//! | `BitXor` | `binary_bitwise_xor_tile(cb_a, cb_b, i_a, i_b, idst)` |
//! | `BitOr` | `binary_bitwise_or_tile(cb_a, cb_b, i_a, i_b, idst)` |
//! | `BitAnd` | `binary_bitwise_and_tile(cb_a, cb_b, i_a, i_b, idst)` |
//! | `BitShiftLeft` | `binary_shift_tile(cb_a, cb_b, i_a, i_b, idst, SHIFT_LEFT)` |
//! | `BitShiftRight` | `binary_shift_tile(cb_a, cb_b, i_a, i_b, idst, SHIFT_RIGHT)` |
//! | `NotEq` | `binary_comp_tile(cb_a, cb_b, i_a, i_b, idst, COMPARE_NE)` |
//! | `Eq` | `binary_comp_tile(cb_a, cb_b, i_a, i_b, idst, COMPARE_EQ)` |
//!
//! # Compilation Pipeline
//!
//! 1. Walk kernel IR to extract `Op::Define { len: 1024 }` tiles and their interconnections.
//! 2. Generate reader kernel source (BRISC): for each input global buffer, loop over tiles
//!    and `noc_async_read(src_addr + tile_idx * tile_bytes, cb_write_ptr, tile_bytes)`.
//! 3. Generate compute kernel source (TRISC0/1/2):
//!    - Unpack: `cb_wait_front(cb_in, 1)` → `copy_tile(cb_in, 0, idst)`.
//!    - Math: apply `*_tile()` or `*_tiles()` on DST registers.
//!    - Pack: `pack_tile(idst, cb_out)` → `cb_push_back(cb_out, 1)`.
//!    - Pop inputs: `cb_pop_front(cb_in, 1)`.
//! 4. Generate writer kernel source (NCRISC): `cb_wait_front(cb_out, 1)` →
//!    `noc_async_write(cb_read_ptr, dst_addr + tile_idx * tile_bytes, tile_bytes)`.
//! 5. Write reader + compute + writer C++ sources to temp files.
//! 6. Invoke `riscv-tt-elf-g++ -mcpu=tt-bh` (SFPI cross-compiler) to compile each to ELF.
//! 7. Load ELF segments to device L1 memory.
//! 8. Configure launch_msg_t (kernel_config_base, kernel_text_offset, enables).
//! 9. Signal RUN_MSG_GO → BRISC firmware calls kernel as function pointer.
//!
//! # Memory Model
//!
//! - **DRAM** (global memory): accessed via NOC by reader/writer kernels.
//!   Each global `Op::Define` becomes a DRAM buffer pointer passed as kernel arg.
//! - **L1** (local memory): circular buffers for tile data between kernels.
//!   Each tensor tile in flight needs a CB slot. CB size = num_tiles_in_flight × tile_size.
//! - **DST registers**: 4 tile slots on the math processor. All compute ops
//!   read from DST and write to DST. Managed via `tile_regs_acquire()` /
//!   `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`.
//!
//! # Tiles in zyx
//!
//! In zyx, a tile is simply `Op::Define { dtype, scope: Register, len: 1024 }`.
//! zyx does not care whether the hardware lays this out as 32×32, 64×16, or
//! any other shape — it's a flat vector of 1024 elements. Elementwise ops
//! (`Op::Unary`, `Op::Binary`, `Op::Cast`, `Op::Mad`) operate tile-wide with
//! `vlen` tracking how many elements are processed per load/store.
//!
//! The backend maps zyx's 1024-element tile directly to a tt-metal 32×32 tile
//! (1024 == 32×32). This shape only matters for:
//! - **TMMA (matmul)**: the tile shape determines how fragments are laid out
//!   for matrix multiply-accumulate on the math engine.
//! - **Reader/writer addressing**: the NOC stride pattern when copying tile
//!   data between DRAM row-major and L1 tile layouts.
//!
//! For elementwise ops, the tile shape is irrelevant — the SFPU applies the
//! operation uniformly across all 1024 elements.
//!
//! The backend does NOT need to do explicit tiling/un-tiling. zyx already
//! generates vectorized loads of 1024 elements (`Load { vlen: 1024 }`),
//! executes the elementwise op, then vectorized stores of 1024 elements
//! (`Store { vlen: 1024 }`). The reader kernel copies DRAM rows into CB
//! tiles, the compute kernel runs tile ops on DST, and the writer kernel
//! copies CB tiles back to DRAM rows.
//!
//! # First Milestones
//!
//! 1. **Memory movement**: load a 1024-element tile from DRAM → CB → DST,
//!    then copy back to DRAM (identity passthrough). No compute. Verifies
//!    the reader/compute/writer pipeline, NOC addressing, and CB management.
//! 2. **exp2 kernel**: load tile → `exp2_tile(idst)` (SFPU) → store tile.
//!    Verifies SFPU compute on Tensix.
//! 3. **General unary/binary ops**: build out the full op mapping.
//! 4. **TMMA (matmul)**: add tile-aware matmul op for matrix multiplication.
//!
//! # Supported Data Types
//!
//! - F32: native Float32
//! - F16: native Float16 (half-precision)
//! - BF16: BFloat16 (truncated F32, should map to F16 for math, or stay as
//!   separate format with `bfloat16` dtype in CB)
//! - BFLOAT8: 8-bit block float (tt-metal specific, for memory bandwidth)
//! - I32/U32: 32-bit integer
//! - I16/U16: 16-bit integer
//! - I8/U8: 8-bit integer
//!
//! The `supported_dtypes` bitmask should expose everything tt-metal can handle.

// TODO: tenstorrent backend implementation

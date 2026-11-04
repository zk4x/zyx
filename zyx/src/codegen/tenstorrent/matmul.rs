// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
//! The matmul facts zyx emits: `Src` row placement, `ADDR_MOD` codes for the
//! Matrix Unit, the unpacker's tile descriptor, and the operand route.
//!
//! Ported from `metalium-rs` `tt-kernels::matmul` (Apache-2.0), trimmed to what
//! zyx's lowering and codegen consume. The block-float route is not ported:
//! zyx has no `Bfp` staging.

use tt_isa::tile::{L1Format, TileDescriptor};

/// Where `SrcA` operands sit: row 0, straight from the start of their L1 run.
pub const SRC_A_ROW: usize = 0;
/// Where `SrcB` operands sit; row 0, as `SrcA`.
pub const SRC_B_ROW: usize = 0;
/// `OutDataFormat` TF32: what FP32 in L1 becomes in `Src`.
pub const TF32_CODE: u32 = 4;
/// `OutDataFormat` BF16 (divergence row H).
pub const BF16_CODE: u32 = 5;

/// The Matrix Unit's `ADDR_MOD` codes, one per column-group boundary:
///
/// * [`MATH_AM_PHASE`] (1): the next fidelity phase;
/// * [`MATH_AM_HALF`] (2): the end of a group's first `SrcB` half — phase
///   back to 0, `SrcB` on eight rows;
/// * [`MATH_AM_GROUP`] (3): the end of a group, but not an `fi`'s last — phase
///   and `SrcB` back to 0;
/// * [`MATH_AM_FI`] (4): the end of an `fi` — phase and `SrcB` back to 0,
///   `Dst` on 32 rows (two faces), the next `fi`'s base.
///
/// 0 moves nothing.
pub const MATH_AM_PHASE: u32 = 1;
pub const MATH_AM_HALF: u32 = 2;
pub const MATH_AM_GROUP: u32 = 3;
pub const MATH_AM_FI: u32 = 4;

/// Datums in one face.
pub const FACE_DATUMS: u32 = 256;

/// The unpacker's view of a 32x32 tile in `format`: four 16x16 faces, one per
/// `Z` plane.
pub fn tile_descriptor(format: L1Format) -> TileDescriptor {
    TileDescriptor::zeroed()
        .with_x_dim(16)
        .with_y_dim(16)
        .with_z_dim(4)
        .with_is_uncompressed(true)
        .with_in_data_format_raw(format.code().expect("only measured format codes"))
}

/// How an operand reaches the `Src` banks, and so what the unpacker does to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrcRoute {
    /// FP32 in L1, truncated to TF32 by the unpacker.
    Tf32FromFp32,
    /// FP32 in L1, converted to BF16 by the unpacker.
    Bf16FromFp32,
    /// BF16 in L1, moved as is.
    Bf16FromBf16,
}

impl SrcRoute {
    /// The L1 format the operands are staged in, and the `OutDataFormat` code.
    pub const fn formats(self) -> (L1Format, u32) {
        match self {
            SrcRoute::Tf32FromFp32 => (L1Format::Fp32, TF32_CODE),
            SrcRoute::Bf16FromFp32 => (L1Format::Fp32, BF16_CODE),
            SrcRoute::Bf16FromBf16 => (L1Format::Bf16, BF16_CODE),
        }
    }
}

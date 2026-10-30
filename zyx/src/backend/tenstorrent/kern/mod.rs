// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
//! Tensix kernel builders zyx owns.
//!
//! Ported from `metalium-rs` `crates/tt-kernels` (Apache-2.0): the runner
//! (`runtime`), the data-mover/unpacker/packer word builders (`datapath`), the
//! SFPU program generator (`sfpu`), the L1 plan types (`l1`), and the matmul
//! constants zyx's lowering and codegen consume (`matmul`). Each module keeps
//! its upstream documentation.

pub mod code;
pub mod datapath;
pub mod elem;
pub mod kind;
pub mod l1;
pub mod loops;
pub mod matmul;
pub mod runtime;
pub mod sfpu;

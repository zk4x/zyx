// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
//! What an element-wise operation computes on (`metalium-rs`'s
//! `tt_kernels::tensor::Elem`, reduced to the enum itself).

/// The arithmetic an operand is interpreted as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Elem {
    F32,
    I32,
    Bool,
}

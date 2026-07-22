// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope},
    shape::Dim,
};

impl Kernel {
    /// Create local memory tiles
    pub(crate) fn tile_local(&mut self) {}
}

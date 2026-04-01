// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::kernel::Kernel;

impl Kernel {
    pub fn opt_fuse_mad(&self) -> (Optimization, usize) {
        (Optimization::FuseMad, 1)
    }
}

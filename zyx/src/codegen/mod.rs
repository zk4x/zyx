use crate::kernel::Kernel;

// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
mod c;
mod cuda;
mod opencl;
mod ptx;
mod spirv;
pub mod tenstorrent;

impl Kernel {
    /// Render to given source kind
    pub fn render(&self) -> Self {
        todo!()
    }
}

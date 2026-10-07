use crate::{
    backend::Dev,
    error::BackendError,
    kernel::{Kernel, Op},
};

// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
mod c;
mod cuda;
mod opencl;
mod ptx;
mod spirv;
pub mod tenstorrent;

impl Kernel {
    /// Render to a backend kernel: the original ops plus appended
    /// backend-specific `Source` and meta ops, selected by [`Dev`].
    /// A kernel that already holds a `Source` (hand-built, e.g. a custom
    /// compute section alongside compiled reader/writer) passes through
    /// untouched: render completes, never re-lowers.
    pub fn render(&self) -> Result<Kernel, BackendError> {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Source(_)) {
                return Ok(self.clone());
            }
            op_id = self.next_op(op_id);
        }
        match self.dev {
            Dev::C => self.render_c(),
            Dev::Cuda(_) => self.render_cuda(),
            Dev::OpenCL(_) => self.render_opencl(),
            Dev::Vulkan(_) => self.render_spirv(),
            #[cfg(feature = "wgpu")]
            Dev::WGPU(_) => self.render_spirv(),
            dev => todo!("render: backend {dev:?} not yet implemented"),
        }
    }
}

// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! PTX-specific device lowering.
//!
//! Runs in the device epilogue, after every generic pass, and only for the
//! PTX-direct renderer. Its job is decomposition: every op it produces is one
//! PTX instruction with its operands as IR values, so what the renderer used
//! to synthesise inline (`mov.u32 %r, __ldN`, `cvt.u32.u64`, `shl.b32`,
//! `add.u32`, `cvt.u64.u32` + `cvta.shared.u64`) is IR the passes own.
//!
//! [`Kernel::lower_shared_addr`] rewrites the shared accesses of the
//! IR — an element index into a `MemScope::Local` storage — into the address
//! the instruction takes:
//!
//! ```text
//! SharedBase  = mov.u32 %r, __ldN          // the storage's window offset
//! offset      = cvt.u32.u64 idx            // element index, 32-bit
//! offset      = shl.b32 offset, <shift>    // to bytes
//! addr        = add.u32 SharedBase, offset
//! access      = ld.shared / st.shared at addr
//! ```
//!
//! and `VLdMatrix` into the same address plus the two ops `ldmatrix` needs to
//! reach it (`cvt.u64.u32`, `cvta.shared.u64`) and the hardware instruction.

use crate::{
    dtype::Constant,
    kernel::{BOp, DType, Kernel, MemScope, Op, OpId, PTXOp},
};

impl Kernel {
    /// Decompose every shared access and every `ldmatrix` into the PTX
    /// instructions it renders as.
    pub fn lower_shared_addr(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op.clone() {
                Op::Load { src: buf, src_idx } if self.shared_storage(buf) => {
                    let addr = self.shared_byte_addr(op_id, buf, src_idx);
                    self.ops[op_id].op = Op::PTX(Box::new(PTXOp::LdShared { buf, addr }));
                }
                Op::Store { dst: buf, dst_idx, src } if self.shared_storage(buf) => {
                    let addr = self.shared_byte_addr(op_id, buf, dst_idx);
                    self.ops[op_id].op = Op::PTX(Box::new(PTXOp::StShared { buf, addr, src }));
                }
                Op::PTX(ref p) => {
                    // `ldmatrix` reads the shared window directly: its address
                    // is the same 32-bit byte offset a `ld.shared` takes, so the
                    // window base is one register per storage and each sub-tile
                    // offset folds into the instruction's immediate — the
                    // reference's `[R200+UR8+0x1000]`. A generic 64-bit address
                    // (a `cvta` per load) gives every load a register pair of
                    // its own and no immediates at all. Any other PTX op (the
                    // warp sync) has no shared address and is left alone; the
                    // walk must still advance past it.
                    if let PTXOp::VLdMatrix { frags, src, addr, trans } = **p {
                        let addr = self.shared_byte_addr(op_id, src, addr);
                        self.ops[op_id].op = Op::PTX(Box::new(PTXOp::LdMatrix { frags, addr, trans }));
                    }
                }
                _ => {}
            }
            op_id = next;
        }
    }

    /// `Op::Storage` of `buf` when it is shared memory.
    fn shared_storage(&self, buf: OpId) -> bool {
        matches!(self.at(buf), Op::Storage { scope: MemScope::Local, .. })
    }

    /// The byte address of element index `idx` inside shared storage `buf`:
    /// the storage's window offset plus the index scaled to bytes.
    fn shared_byte_addr(&mut self, before: OpId, buf: OpId, idx: OpId) -> OpId {
        let Op::Storage { dtype, .. } = self.at(buf) else {
            unreachable!("shared_byte_addr: {buf:?} is not a storage")
        };
        let shift = (dtype.bit_size() / 8).ilog2();
        let base = self.insert_before(before, Op::PTX(Box::new(PTXOp::SharedBase { buf })));
        let narrow = self.insert_before(before, Op::Cast { x: idx, dtype: DType::U32 });
        let amount = self.insert_before(before, Op::Const(Constant::new(shift)));
        let scaled = self.insert_before(before, Op::Binary { x: narrow, y: amount, bop: BOp::BitShiftLeft });
        self.insert_before(before, Op::Binary { x: base, y: scaled, bop: BOp::Add })
    }
}

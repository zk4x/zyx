// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! PTX assembly codegen from zyx kernel IR.

use crate::{
    DType, Map,
    backend::gws_from_kernel,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, UOp},
    scalar::bf16,
    shape::Dim,
};
use std::fmt::Write;

impl DType {
    fn ptx(&self) -> &'static str {
        match self {
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "s8",
            Self::I16 => "s16",
            Self::I32 => "s32",
            Self::I64 => "s64",
            Self::Bool => "pred",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }

    fn mem_ptx(&self) -> &'static str {
        match self {
            Self::BF16 | Self::F16 => "b16",
            Self::Bool => "u8",
            _ => self.ptx(),
        }
    }

    fn reg_ptx(&self) -> &'static str {
        match self {
            Self::BF16 | Self::F16 => "b16",
            Self::Bool => "pred",
            // mma.sync operands and bit-packed pairs require the untyped
            // 32-bit register type.
            Self::U32 | Self::I32 => "b32",
            _ => self.ptx(),
        }
    }
}

impl Constant {
    fn ptx(&self) -> String {
        fn format_precise(val: impl std::fmt::Display, decimals: usize) -> String {
            let s = format!("{:.*}", decimals, val);
            let s = s.trim_end_matches('0').trim_end_matches('.');
            if s.contains('.') { s.to_string() } else { format!("{s}.0") }
        }
        match self {
            Self::BF16(x) => {
                let val = f32::from(bf16::from_le_bytes(*x));
                if val.is_finite() {
                    format!("{}f", val)
                } else {
                    format!("0f{:08X}", val.to_bits())
                }
            }
            Self::F16(x) => format!("0x{:04X}", u16::from_le_bytes(*x)),
            Self::F32(x) => {
                let val = f32::from_le_bytes(*x);
                if val.is_finite() {
                    format_precise(val, 9)
                } else {
                    format!("0f{:08X}", val.to_bits())
                }
            }
            Self::F64(x) => {
                let val = f64::from_le_bytes(*x);
                if val.is_finite() {
                    format_precise(val, 18)
                } else {
                    format!("0d{:016X}", val.to_bits())
                }
            }
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}U"),
            Self::U64(x) => format!("{}", u64::from_le_bytes(*x)),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{}", i64::from_le_bytes(*x)),
            Self::Bool(x) => format!("{}", if *x { 1 } else { 0 }),
        }
    }
}

struct Compiler {
    var_map: Map<OpId, u16>,
    loops: Vec<(u16, u16, u16)>,
    if_labels: Vec<u16>,
    scopes: Map<OpId, MemScope>,
    header: String,
    body: String,
    indent: String,
    // One slot per op; a Vector(len) slot owns `len` consecutive physical
    // PTX registers, `var_map` holds the physical base register index.
    registers: Vec<((DType, MemLayout), u32, u8, u16)>,
    phys_regs: u16,
    loop_level: u8,
}

impl Compiler {
    fn bop_to_ptx(&self, bop: BOp, dtype: DType) -> &'static str {
        match bop {
            BOp::Add => "add",
            BOp::Sub => "sub",
            BOp::Mul => {
                if dtype.is_float() {
                    "mul"
                } else {
                    "mul.lo"
                }
            }
            BOp::Div => {
                if dtype == DType::F32 {
                    "div.approx"
                } else if dtype == DType::F64 {
                    "div.rn"
                } else {
                    "div"
                }
            }
            BOp::Pow => todo!(),
            BOp::Mod => "rem",
            BOp::Cmplt => "setp.lt",
            BOp::Cmpgt => "setp.gt",
            BOp::Cmpge => "setp.ge",
            BOp::Max => "max",
            BOp::Or => "or",
            BOp::And => "and",
            BOp::BitXor => "xor",
            BOp::BitOr => "or",
            BOp::BitAnd => "and",
            BOp::BitShiftLeft => "shl",
            BOp::BitShiftRight => "shr",
            BOp::NotEq => "setp.ne",
            BOp::Eq => "setp.eq",
        }
    }

    fn uop_to_ptx(&self, uop: UOp, dtype: DType) -> Result<&'static str, BackendError> {
        match uop {
            UOp::Neg => Ok("neg"),
            UOp::Not => Ok("not"),
            UOp::BitNot => Ok("not"),
            UOp::Exp => Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "PTX: UOp::Exp should be converted to Exp2 + mul by ln2(e) before reaching PTX backend".into(),
            }),
            UOp::Exp2 => match dtype {
                DType::F32 => Ok("ex2.approx"),
                DType::F16 => Ok("ex2.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: ex2.approx is only available for f32/f16, not {dtype:?}").into(),
                }),
            },
            UOp::Log2 => match dtype {
                DType::F32 => Ok("lg2.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: lg2.approx is only available for f32, not {dtype:?}").into(),
                }),
            },
            UOp::Reciprocal => Ok("rcp.ftz.approx"),
            UOp::Sqrt => match dtype {
                DType::F32 => Ok("sqrt.approx"),
                DType::F64 => Ok("sqrt.rn"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: sqrt not available for {dtype:?}").into(),
                }),
            },
            UOp::Rsqrt => match dtype {
                DType::F32 => Ok("rsqrt.approx"),
                DType::F16 => Ok("rsqrt.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: rsqrt not available for {dtype:?}").into(),
                }),
            },
            UOp::Sin => match dtype {
                DType::F32 => Ok("sin.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: sin.approx is only available for f32, not {dtype:?}").into(),
                }),
            },
            UOp::Cos => match dtype {
                DType::F32 => Ok("cos.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: cos.approx is only available for f32, not {dtype:?}").into(),
                }),
            },
            UOp::Floor => Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "PTX: UOp::Floor must use cvt.rmi, not a separate instruction".into(),
            }),
            UOp::Trunc => Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "PTX: UOp::Trunc must use cvt.rzi, not a separate instruction".into(),
            }),
            UOp::Abs => Ok("abs"),
        }
    }

    fn get_scope(&self, ops: &Kernel, var: OpId) -> MemScope {
        if let Some(&scope) = self.scopes.get(&var) {
            return scope;
        }
        // Kernel buffer arguments are Op::Param — they live in global memory.
        if matches!(ops.ops[var].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
            return MemScope::Global;
        }
        MemScope::Register
    }

    /// Number of physical PTX registers owned by one logical slot.
    fn slot_len(layout: MemLayout) -> u16 {
        match layout {
            MemLayout::Scalar => 1,
            MemLayout::Vector(len) => len,
            MemLayout::Tile { .. } => todo!("PTX: tile layout not implemented"),
        }
    }

    fn slot_of(&self, base: u16) -> usize {
        self.registers.iter().position(|r| r.3 == base).expect("PTX: register base without owning slot")
    }

    fn new_reg(&mut self, dtype: DType, layout: MemLayout, rc: u32) -> u16 {
        for reg in self.registers.iter_mut() {
            if reg.1 == 0 && reg.0 == (dtype, layout) && self.loop_level <= reg.2 {
                // Freed slot: reuse it (its physical base is unchanged).
                reg.1 = rc;
                reg.2 = self.loop_level;
                return reg.3;
            }
        }
        let base = self.phys_regs;
        self.phys_regs += Self::slot_len(layout);
        self.registers.push(((dtype, layout), rc, self.loop_level, base));
        base
    }

    fn new_var(&mut self, op_id: OpId, dtype: DType, layout: MemLayout, rc: u32) -> u16 {
        let base = self.new_reg(dtype, layout, rc);
        self.var_map.insert(op_id, base);
        base
    }

    fn get_var(&mut self, x: OpId) -> u16 {
        let r = self.var_map[&x];
        let slot = self.slot_of(r);
        if self.loop_level == self.registers[slot].2 {
            self.registers[slot].1 -= 1;
        }
        r
    }

    /// Physical register indices of an operand's components. A Stack groups
    /// already-mapped scalar registers; a Vector-layout op owns consecutive
    /// registers from its base.
    fn components_of(
        &mut self,
        kernel: &Kernel,
        op: OpId,
        dtypes: &Map<OpId, (DType, MemLayout)>,
    ) -> Result<Vec<u16>, BackendError> {
        match &kernel.ops[op].op {
            Op::Stack { ops } => Ok(ops.iter().map(|&x| self.var_map[&x]).collect()),
            _ => {
                let (_, layout) = dtypes[&op];
                let len = Self::slot_len(layout);
                let base = self.var_map[&op];
                Ok((0..len).map(|i| base + i).collect())
            }
        }
    }

    fn release_reg(&mut self, x: u16) {
        let slot = self.slot_of(x);
        self.registers[slot].1 -= 1;
    }

    /// Emit a load from `[ %address ]` into the register(s) at `reg`.
    /// `space` is one of "global", "shared", "local".
    fn emit_load(&mut self, layout: MemLayout, dtype: DType, reg: u16, space: &str) -> Result<(), BackendError> {
        match layout {
            MemLayout::Scalar => {
                if matches!(dtype, DType::F16 | DType::BF16) {
                    let tmp = self.new_reg(DType::U16, MemLayout::Scalar, 1);
                    _ = writeln!(self.body, "{}ld.{}.b16 %r{tmp}, [%address];", self.indent, space);
                    _ = writeln!(self.body, "{}mov.b16 %r{reg}, %r{tmp};", self.indent);
                    self.release_reg(tmp);
                } else {
                    _ = writeln!(self.body, "{}ld.{}.{} %r{reg}, [%address];", self.indent, space, dtype.mem_ptx());
                }
            }
            MemLayout::Vector(4) if dtype == DType::F32 => {
                _ = writeln!(
                    self.body,
                    "{}ld.{}.v4.f32 {{%r{reg}, %r{}, %r{}, %r{}}}, [%address];",
                    self.indent,
                    space,
                    reg + 1,
                    reg + 2,
                    reg + 3
                );
            }
            MemLayout::Vector(_) => {
                todo!("PTX: vector load of {layout:?} {dtype:?} not implemented")
            }
            MemLayout::Tile { .. } => todo!("PTX: tile load not implemented"),
        }
        Ok(())
    }

    /// Emit a store of the register(s) at `x` to `[ %address ]` in global space.
    fn emit_store(&mut self, layout: MemLayout, dtype: DType, x: u16) -> Result<(), BackendError> {
        if matches!(dtype, DType::F16 | DType::BF16) && layout == MemLayout::Scalar {
            let tmp = self.new_reg(DType::U16, MemLayout::Scalar, 1);
            _ = writeln!(self.body, "{}mov.b16 %r{tmp}, %r{x};", self.indent);
            _ = writeln!(self.body, "{}st.global.b16 [%address], %r{tmp};", self.indent);
            self.release_reg(tmp);
        } else {
            self.emit_store_space(layout, dtype, x, "global")?;
        }
        Ok(())
    }

    /// Emit a store of the register(s) at `x` to `[ %address ]`.
    /// `space` is one of "global", "shared", "local".
    fn emit_store_space(&mut self, layout: MemLayout, dtype: DType, x: u16, space: &str) -> Result<(), BackendError> {
        match layout {
            MemLayout::Scalar => {
                _ = writeln!(self.body, "{}st.{}.{} [%address], %r{x};", self.indent, space, dtype.mem_ptx());
            }
            MemLayout::Vector(4) if dtype == DType::F32 => {
                _ = writeln!(
                    self.body,
                    "{}st.{}.v4.f32 [%address], {{%r{x}, %r{}, %r{}, %r{}}};",
                    self.indent,
                    space,
                    x + 1,
                    x + 2,
                    x + 3
                );
            }
            MemLayout::Vector(_) => {
                todo!("PTX: vector store of {layout:?} {dtype:?} not implemented")
            }
            MemLayout::Tile { .. } => todo!("PTX: tile store not implemented"),
        }
        Ok(())
    }
}

impl Kernel {
    /// Compile kernel to PTX assembly.
    #[allow(clippy::type_complexity)] // complex return type inherent to PTX backend API
    pub fn generate_ptx(&self, name: &str) -> Result<(Vec<u8>, Vec<Dim>), BackendError> {
        // Reject group lengths that are constant and exceed the device grid limits.
        gws_from_kernel(self, &self.dev_info().max_global_work_dims)?;
        let mut comp = Compiler {
            var_map: Map::default(),
            loops: Vec::new(),
            if_labels: Vec::new(),
            header: String::new(),
            body: String::new(),
            indent: "  ".to_string(),
            registers: Vec::new(),
            phys_regs: 0,
            loop_level: 0,
            scopes: Map::default(),
        };

        let mut lws = vec![1; 3];
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_ptx did not finish in 10000 steps");
            }
            if let Op::Range { axis, kind: scope } = self.ops[op_id].op {
                match scope {
                    RangeKind::Group(_) => {}
                    RangeKind::Local(len) => lws[axis as usize] = Dim::from(len),
                    // A warp is a view over a local range — adds no threads.
                    RangeKind::Warp(_) => {}
                }
            }
            op_id = self.next_op(op_id);
        }
        if lws.iter().product::<Dim>() > self.dev_info().max_local_threads as Dim {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "Invalid local work size.".into() });
        }
        let cc = self.dev_info().cc;
        _ = writeln!(comp.header, ".version {0}.{1}\n.target sm_{0}{1}\n.address_size 64\n.visible .entry {name}(", cc[0], cc[1]);
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_ptx did not finish in 10000 steps");
            }
            // Kernel arguments are Op::Param defines, flat head order —
            // the same order the backend passes `args` at launch.
            if let Op::Param { dtype, kind, .. } = self.ops[op_id].op {
                match kind {
                    ParamKind::Variable => {
                        writeln!(comp.header, "{}.param .{} g{op_id},", comp.indent, dtype.ptx()).unwrap();
                    }
                    ParamKind::Global | ParamKind::GlobalMut => {
                        writeln!(comp.header, "{}.param .u64 g{op_id},", comp.indent).unwrap();
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        comp.header.pop();
        comp.header.pop();
        _ = writeln!(comp.header, "\n) {{");

        let mut loop_id_label_map: Map<u8, u32> = Map::default();
        let mut label = 0;

        let (dtypes, rcs) = self.compute_dtypes_and_rcs();
        let mut loop_id: u8 = 0;
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_ptx did not finish in 10000 steps");
            }
            match self.ops[op_id].op {
                Op::Param { kind, .. } => match kind {
                    ParamKind::Variable => {
                        // Runtime scalar (IDX_T dims etc.): load into a typed register.
                        let dtype = dtypes[&op_id].0;
                        let rc = rcs.get(&op_id).copied().unwrap_or(0);
                        let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rc);
                        _ = writeln!(comp.body, "{}ld.param.{} %r{reg}, [g{op_id}];", comp.indent, dtype.ptx())
                    }
                    ParamKind::Global | ParamKind::GlobalMut => {
                        _ = writeln!(comp.body, "{}ld.param.u64 %p{op_id}, [g{op_id}];", comp.indent)
                    }
                },
                Op::Storage { dtype, scope, len, .. } => {
                    comp.scopes.insert(op_id, scope);
                    match scope {
                        MemScope::Local => {
                            // 16-byte alignment so vectorized (v4) accesses are legal.
                            _ = writeln!(comp.body, "{}.shared .align 16 .{} __ld{op_id}[{len}];", comp.indent, dtype.ptx());
                        }
                        MemScope::Register => {
                            _ = writeln!(comp.body, "{}.local .align 16 .{} __ld{op_id}[{len}];", comp.indent, dtype.ptx());
                        }
                        MemScope::Circular | MemScope::Global => {
                            unreachable!("ptx only supports local or register storage")
                        }
                    }
                }
                Op::Range { axis, kind: scope, .. } => {
                    let rc = rcs.get(&op_id).copied().unwrap_or(0);
                    let reg = comp.new_var(op_id, IDX_T, MemLayout::Scalar, rc);
                    let axis_letter = ["x", "y", "z"][axis as usize];
                    // Special registers are 32-bit: read into a u32 temp,
                    // then convert to IDX_T.
                    let t = comp.new_reg(DType::U32, MemLayout::Scalar, 1);
                    let src = match scope {
                        RangeKind::Group(_) => "%ctaid",
                        RangeKind::Local(_) => "%tid",
                        RangeKind::Warp(_) => "",
                    };
                    match scope {
                        RangeKind::Group(_) | RangeKind::Local(_) => {
                            _ = writeln!(comp.body, "{}mov.u32 %r{t}, {src}.{};", comp.indent, axis_letter);
                        }
                        // Lane id within the warp: local thread id mod warp size.
                        RangeKind::Warp(local_id) => {
                            let local_reg = comp.get_var(local_id);
                            let warp_size = self.dev_info().warp_size;
                            _ = writeln!(comp.body, "{}cvt.u32.s64 %r{t}, %r{local_reg};", comp.indent);
                            _ = writeln!(comp.body, "{}rem.u32 %r{t}, %r{t}, {warp_size};", comp.indent);
                        }
                    }
                    match IDX_T {
                        DType::U32 => _ = writeln!(comp.body, "{}mov.u32 %r{reg}, %r{t};", comp.indent),
                        DType::I64 => _ = writeln!(comp.body, "{}cvt.s64.u32 %r{reg}, %r{t};", comp.indent),
                        DType::U64 => _ = writeln!(comp.body, "{}cvt.u64.u32 %r{reg}, %r{t};", comp.indent),
                        _ => unreachable!("PTX: unexpected IDX_T {IDX_T:?}"),
                    }
                    comp.release_reg(t);
                }
                Op::Const(ref constant) => {
                    let reg = comp.new_var(op_id, constant.dtype(), MemLayout::Scalar, u32::MAX);
                    let ptx_dtype = if constant.dtype() == DType::F16 || constant.dtype() == DType::BF16 {
                        "b16"
                    } else {
                        constant.dtype().ptx()
                    };
                    _ = writeln!(comp.body, "{}mov.{ptx_dtype} %r{reg}, {};", comp.indent, constant.ptx());
                }
                Op::Load { src, index, layout, .. } => {
                    let dtype = dtypes[&src].0;
                    match comp.get_scope(self, src) {
                        MemScope::Circular => unreachable!(),
                        MemScope::Global => {
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            let idx = comp.get_var(index);
                            let offset = comp.new_reg(DType::U64, MemLayout::Scalar, 1);
                            let reg = comp.new_var(op_id, dtype, layout, rcs[&op_id]);
                            if IDX_T == DType::U64 {
                                if offset != idx {
                                    _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                }
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u64 %address, %p{src}, %r{offset};", comp.indent);
                            comp.release_reg(offset);
                            comp.emit_load(layout, dtype, reg, "global")?;
                        }
                        MemScope::Local => {
                            let idx = comp.get_var(index);
                            let reg = comp.new_var(op_id, dtype, layout, rcs[&op_id]);
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            _ = writeln!(comp.body, "{}mov.u64 %address, __ld{src};", comp.indent);
                            let t = comp.new_reg(DType::U64, MemLayout::Scalar, 1);
                            if IDX_T == DType::U64 {
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{idx}, {byte_shift};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{t}, %r{idx};", comp.indent);
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}add.u64 %address, %address, %r{t};", comp.indent);
                            comp.release_reg(t);
                            comp.emit_load(layout, dtype, reg, "shared")?;
                        }
                        MemScope::Register => {
                            let idx = comp.get_var(index);
                            let reg = comp.new_var(op_id, dtype, layout, rcs[&op_id]);
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            _ = writeln!(comp.body, "{}mov.u64 %address, __ld{src};", comp.indent);
                            let t = comp.new_reg(DType::U64, MemLayout::Scalar, 1);
                            if IDX_T == DType::U64 {
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{idx}, {byte_shift};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{t}, %r{idx};", comp.indent);
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}add.u64 %address, %address, %r{t};", comp.indent);
                            comp.release_reg(t);
                            comp.emit_load(layout, dtype, reg, "local")?;
                        }
                    }
                }
                Op::Store { dst, src: x, index, layout, .. } => {
                    let dtype = dtypes[&x].0;
                    let byte_shift = (dtype.bit_size() / 8).ilog2();
                    let offset = comp.new_reg(DType::U64, MemLayout::Scalar, 1);
                    match comp.get_scope(self, dst) {
                        MemScope::Circular => unreachable!(),
                        MemScope::Global => {
                            if dtype == DType::Bool {
                                let gstu = comp.new_reg(DType::U32, MemLayout::Scalar, 1);
                                let idx = comp.get_var(index);
                                let x = comp.get_var(x);
                                _ = writeln!(comp.body, "{}selp.u32 %r{gstu}, 1, 0, %r{x};", comp.indent);
                                if IDX_T == DType::U64 {
                                    if offset != idx {
                                        _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                    }
                                } else {
                                    _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                                }
                                _ = writeln!(comp.body, "{}add.u64 %address, %p{dst}, %r{offset};", comp.indent);
                                _ = writeln!(comp.body, "{}st.global.u8 [%address], %r{gstu};", comp.indent);
                                comp.release_reg(gstu);
                            } else {
                                let idx = comp.get_var(index);
                                let x = comp.get_var(x);
                                if IDX_T == DType::U64 {
                                    if offset != idx {
                                        _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                    }
                                } else {
                                    _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                                }
                                _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                                _ = writeln!(comp.body, "{}add.u64 %address, %p{dst}, %r{offset};", comp.indent);
                                comp.emit_store(layout, dtype, x)?;
                            }
                        }
                        MemScope::Local => {
                            let idx = comp.get_var(index);
                            let x = comp.get_var(x);
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            _ = writeln!(comp.body, "{}mov.u64 %address, __ld{dst};", comp.indent);
                            let t = comp.new_reg(DType::U64, MemLayout::Scalar, 1);
                            if IDX_T == DType::U64 {
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{idx}, {byte_shift};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{t}, %r{idx};", comp.indent);
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}add.u64 %address, %address, %r{t};", comp.indent);
                            comp.release_reg(t);
                            comp.emit_store_space(layout, dtype, x, "shared")?;
                        }
                        MemScope::Register => {
                            let idx = comp.get_var(index);
                            let x = comp.get_var(x);
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            _ = writeln!(comp.body, "{}mov.u64 %address, __ld{dst};", comp.indent);
                            let t = comp.new_reg(DType::U64, MemLayout::Scalar, 1);
                            if IDX_T == DType::U64 {
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{idx}, {byte_shift};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{t}, %r{idx};", comp.indent);
                                _ = writeln!(comp.body, "{}shl.b64 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}add.u64 %address, %address, %r{t};", comp.indent);
                            comp.release_reg(t);
                            comp.emit_store_space(layout, dtype, x, "local")?;
                        }
                    }
                    comp.release_reg(offset);
                }
                Op::Cast { x, dtype } => {
                    let xdtype = dtypes[&x].0;
                    debug_assert_eq!(dtypes[&x].1, MemLayout::Scalar, "PTX: vector cast not implemented");
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rcs[&op_id]);
                    match (dtype, xdtype) {
                        (DType::Bool, _) => {
                            if dtype.is_float() {
                                _ = writeln!(comp.body, "{}setp.ne.{} %r{reg}, %r{x}, 0.0;", comp.indent, xdtype.ptx());
                            } else {
                                _ = writeln!(comp.body, "{}setp.ne.{} %r{reg}, %r{x}, 0;", comp.indent, xdtype.ptx());
                            }
                        }
                        (_, DType::Bool) => {
                            if dtype == DType::F64 || dtype == DType::F32 {
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, 1.0, 0.0, %r{x};", comp.indent, dtype.ptx());
                            } else if dtype == DType::F16 {
                                _ = writeln!(comp.body, "{}selp.b16 %r{reg}, 0x3C00, 0, %r{x};", comp.indent);
                            } else if dtype == DType::BF16 {
                                _ = writeln!(comp.body, "{}selp.b16 %r{reg}, 0x3F80, 0, %r{x};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, 1, 0, %r{x};", comp.indent, dtype.ptx());
                            }
                        }
                        (DType::I32, DType::F32) => {
                            _ = writeln!(comp.body, "{}cvt.rni.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        _ if dtype == xdtype => {
                            if reg != x {
                                _ = writeln!(comp.body, "{}mov.{} %r{reg}, %r{x};", comp.indent, dtype.ptx());
                            }
                        }
                        (_, _) if xdtype.is_float() && dtype.is_float() && dtype.bit_size() > xdtype.bit_size() => {
                            _ = writeln!(comp.body, "{}cvt.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        (_, _) if xdtype.is_float() && !dtype.is_float() => {
                            _ = writeln!(comp.body, "{}cvt.rni.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        (_, _) if !xdtype.is_float() && dtype.is_float() => {
                            _ = writeln!(comp.body, "{}cvt.rn.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        (_, _) if xdtype.is_float() && dtype.is_float() => {
                            _ = writeln!(comp.body, "{}cvt.rn.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        (_, _) => {
                            _ = writeln!(comp.body, "{}cvt.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                    }
                }
                Op::Bitcast { x, dtype } => {
                    // Equal bit widths (asserted upstream): a raw bit move.
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rcs[&op_id]);
                    _ = writeln!(comp.body, "{}mov.b{} %r{reg}, %r{x};", comp.indent, dtype.bit_size());
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x].0;
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rcs[&op_id]);
                    match uop {
                        UOp::Floor => _ = writeln!(comp.body, "{}cvt.rmi.{t}.{t} %r{reg}, %r{x};", comp.indent, t = dtype.ptx()),
                        UOp::Trunc => _ = writeln!(comp.body, "{}cvt.rzi.{t}.{t} %r{reg}, %r{x};", comp.indent, t = dtype.ptx()),
                        _ => {
                            _ = writeln!(
                                comp.body,
                                "{}{}.{} %r{reg}, %r{x};",
                                comp.indent,
                                comp.uop_to_ptx(uop, dtype)?,
                                dtype.ptx()
                            )
                        }
                    }
                }
                Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id].0;
                    let xr = comp.get_var(x);
                    let yr = comp.get_var(y);
                    let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rcs[&op_id]);
                    let type_ext = if matches!(bop, BOp::BitShiftLeft | BOp::BitShiftRight) {
                        match dtypes[&x].0.bit_size() {
                            32 => "b32",
                            64 => "b64",
                            _ => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("PTX: unsupported shift bit size {}", dtypes[&x].0.bit_size()).into(),
                                });
                            }
                        }
                    } else {
                        dtypes[&x].0.ptx()
                    };
                    _ = writeln!(
                        comp.body,
                        "{}{}.{} %r{reg}, %r{xr}, %r{yr};",
                        comp.indent,
                        comp.bop_to_ptx(bop, dtype),
                        type_ext,
                    );
                }
                Op::Mad { x, y, z, .. } => {
                    let dtype = dtypes[&op_id].0;
                    let xr = comp.get_var(x);
                    let yr = comp.get_var(y);
                    let zr = comp.get_var(z);
                    let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rcs[&op_id]);
                    let mul = comp.new_reg(dtype, MemLayout::Scalar, 1);
                    _ = writeln!(
                        comp.body,
                        "{}{}.{} %r{mul}, %r{xr}, %r{yr};",
                        comp.indent,
                        comp.bop_to_ptx(BOp::Mul, dtype),
                        dtype.ptx(),
                    );
                    _ = writeln!(
                        comp.body,
                        "{}{}.{} %r{reg}, %r{mul}, %r{zr};",
                        comp.indent,
                        comp.bop_to_ptx(BOp::Add, dtype),
                        dtype.ptx(),
                    );
                    comp.release_reg(mul);
                }
                Op::Loop { len } => {
                    comp.loop_level += 1;
                    let len = comp.get_var(len);
                    let loop_idx = comp.new_var(op_id, IDX_T, MemLayout::Scalar, rcs.get(&op_id).copied().unwrap_or(0) + 1);
                    let loop_pred = comp.new_reg(DType::Bool, MemLayout::Scalar, 2);
                    comp.loops.push((len, loop_pred, loop_idx));
                    _ = writeln!(comp.body, "{}mov.{} %r{loop_idx}, 0;", comp.indent, IDX_T.ptx());
                    _ = writeln!(comp.body, "{}LOOP_{label}:", comp.indent);
                    loop_id_label_map.insert(loop_id, label);
                    label += 1;
                    comp.indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    loop_id -= 1;
                    if let Some((len, loop_pred, loop_idx)) = comp.loops.pop() {
                        _ = writeln!(comp.body, "{}add.{} %r{loop_idx}, %r{loop_idx}, 1;", comp.indent, IDX_T.ptx());
                        writeln!(comp.body, "{}setp.lt.{} %r{loop_pred}, %r{loop_idx}, %r{len};", comp.indent, IDX_T.ptx(),)
                            .unwrap();
                        _ = writeln!(comp.body, "{}@%r{loop_pred} bra LOOP_{};", comp.indent, loop_id_label_map[&loop_id]);
                        comp.indent.pop();
                        comp.indent.pop();
                    }
                    comp.loop_level -= 1;
                }
                Op::If { condition } => {
                    let cond = comp.get_var(condition);
                    let endif_label = label as u16;
                    label += 1;
                    comp.if_labels.push(endif_label);
                    _ = writeln!(comp.body, "{}@!%r{cond} bra ENDIF_{endif_label};", comp.indent);
                    comp.indent += "  ";
                }
                Op::EndIf => {
                    comp.indent.pop();
                    comp.indent.pop();
                    if let Some(endif_label) = comp.if_labels.pop() {
                        _ = writeln!(comp.body, "{}ENDIF_{endif_label}:", comp.indent);
                    }
                }
                Op::Barrier => {
                    _ = writeln!(comp.body, "{}bar.sync 1;", comp.indent);
                }
                // A Stack is a pure register grouping (e.g. an mma operand
                // fragment): its components are already mapped registers and
                // consumers read them via `components_of`. Nothing to emit.
                Op::Stack { .. } => {}
                Op::Index { vec, idx } => {
                    // Component extract: with a constant index the component
                    // is a fixed register — emit a move.
                    let (dtype, layout) = dtypes[&vec];
                    debug_assert!(layout != MemLayout::Scalar, "PTX: Op::Index on a scalar");
                    let len = Compiler::slot_len(layout);
                    debug_assert!(idx < len as usize, "PTX: Op::Index {idx} out of bounds for len {len}");
                    let vb = comp.get_var(vec);
                    let reg = comp.new_var(op_id, dtype, MemLayout::Scalar, rcs[&op_id]);
                    _ = writeln!(comp.body, "{}mov.{} %r{reg}, %r{};", comp.indent, dtype.ptx(), vb + idx as u16);
                }
                Op::Wmma { dims, layout, dtype, c, a, b } => {
                    // Only the m16n8k8 f16->f32 path is implemented; other
                    // combos must be added with their correct operand packing.
                    if dims != MMADims::m16n8k8 || layout != MMALayout::row_col || dtype != MMADType::f16_f16_f16_f32 {
                        todo!("PTX: wmma combo dims={dims:?} layout={layout:?} dtype={dtype:?} not implemented");
                    }
                    let a_comps = comp.components_of(self, a, &dtypes)?;
                    let b_comps = comp.components_of(self, b, &dtypes)?;
                    let c_comps = comp.components_of(self, c, &dtypes)?;
                    debug_assert_eq!(a_comps.len(), 4, "PTX: mma A fragment must hold 4 f16 values");
                    debug_assert_eq!(b_comps.len(), 2, "PTX: mma B fragment must hold 2 f16 values");
                    debug_assert_eq!(c_comps.len(), 4, "PTX: mma C fragment must hold 4 f32 values");
                    // Pack the f16 pairs into .b32 registers: a = {a01, a23},
                    // b = {b01}.
                    let a01 = comp.new_reg(DType::U32, MemLayout::Scalar, 1);
                    let a23 = comp.new_reg(DType::U32, MemLayout::Scalar, 1);
                    let b01 = comp.new_reg(DType::U32, MemLayout::Scalar, 1);
                    _ = writeln!(comp.body, "{}mov.b32 %r{a01}, {{%r{}, %r{}}};", comp.indent, a_comps[0], a_comps[1]);
                    _ = writeln!(comp.body, "{}mov.b32 %r{a23}, {{%r{}, %r{}}};", comp.indent, a_comps[2], a_comps[3]);
                    _ = writeln!(comp.body, "{}mov.b32 %r{b01}, {{%r{}, %r{}}};", comp.indent, b_comps[0], b_comps[1]);
                    // Result: fresh 4-wide f32 vector, mma.sync writes D in place.
                    let rc = rcs[&op_id];
                    let reg = comp.new_var(op_id, DType::F32, MemLayout::Vector(4), rc);
                    _ = writeln!(
                        comp.body,
                        "{}mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {{%r{reg}, %r{}, %r{}, %r{}}}, {{%r{a01}, %r{a23}}}, {{%r{b01}}}, {{%r{}, %r{}, %r{}, %r{}}};",
                        comp.indent,
                        reg + 1,
                        reg + 2,
                        reg + 3,
                        c_comps[0],
                        c_comps[1],
                        c_comps[2],
                        c_comps[3],
                    );
                    comp.release_reg(a01);
                    comp.release_reg(a23);
                    comp.release_reg(b01);
                }
                Op::Asm { .. } => todo!("PTX: inline asm not implemented"),
                ref op => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("PTX: unexpected kernel op={op:?} (should be unfolded)").into(),
                    });
                }
            }
            op_id = self.next_op(op_id);
        }

        _ = writeln!(comp.body, "{}ret;\n}}", comp.indent);

        // Pointer registers for global params (filled by ld.param above).
        let mut op_id = self.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("generate_ptx did not finish in 10000 steps");
            }
            if matches!(self.ops[op_id].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                _ = writeln!(comp.header, "{}.reg .u64 %p{op_id};", comp.indent);
            }
            op_id = self.next_op(op_id);
        }

        _ = writeln!(comp.header, "{}.reg .u64 %address;", comp.indent);
        for (dtype_layout, _, _, base) in comp.registers.iter() {
            let len = Compiler::slot_len(dtype_layout.1);
            for i in 0..len {
                _ = writeln!(comp.header, "{}.reg .{} %r{};", comp.indent, dtype_layout.0.reg_ptx(), base + i);
            }
        }

        comp.header.push_str(&comp.body);

        Ok((comp.header.into_bytes(), lws))
    }
}

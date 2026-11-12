// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! PTX assembly codegen from zyx kernel IR.

use crate::{
    DType, Map, Set,
    backend::gws_from_kernel,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MMADims, MMALayout, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind, UOp},
    scalar::bf16,
    shape::Dim,
};
use std::fmt::Write;

/// One PTX register slot: the value's `(dtype, lane shape)`, its live
/// reference count, the loop level it was assigned at, and its physical
/// base register.
type PhysRegister = ((DType, Vec<Dim>), u32, u8, u16);

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
            Self::F8E4M3 => "e4m3",
            Self::F8E5M2 => "e5m2",
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
            Self::F8E4M3(_) | Self::F8E5M2(_) => todo!("fp8 immediates not yet supported in PTX"),
        }
    }
}

/// The element offset and width of a register-scope access. A register file
/// has no address, so the index must name registers statically: a constant, or
/// a constant base plus an `Unroll(n)` lane — the index vector `{0, .., n-1}`,
/// which is a run of `n` consecutive registers.
fn slot_span(kernel: &Kernel, idx: OpId) -> (u16, u16) {
    let as_dim = |o: OpId| kernel.resolve_const(o).and_then(Constant::as_dim);
    match kernel.at(idx) {
        Op::Binary { x, y, bop: BOp::Add } => match (kernel.at(*x), kernel.at(*y)) {
            (_, Op::Unroll { len, .. }) | (Op::Unroll { len, .. }, _) => {
                let base = as_dim(*x).or_else(|| as_dim(*y)).expect("PTX: register base is not constant");
                (u16::try_from(base).expect("PTX: register index out of range"), *len)
            }
            _ => match as_dim(idx) {
                Some(d) => (u16::try_from(d).expect("PTX: register index out of range"), 1),
                None => todo!("PTX: register-scope access with a computed index"),
            },
        },
        Op::Unroll { len, .. } => (0, *len),
        _ => match as_dim(idx) {
            Some(d) => (u16::try_from(d).expect("PTX: register index out of range"), 1),
            None => todo!("PTX: register-scope access with a runtime index"),
        },
    }
}

struct Compiler {
    var_map: Map<OpId, u16>,
    /// Ops that name an existing register instead of owning one (a component
    /// extract): their value is one register of another op, so reading them
    /// must not consume that op's slot.
    aliases: Set<OpId>,
    loops: Vec<(u16, u16, u16)>,
    if_labels: Vec<u16>,
    // Parallels every open Loop: true = boolean (conditional) close,
    // false = counted-loop close. The closes emit different code
    // (ENDIF label vs loop back-edge), unlike the C-like backends.
    scope_is_bool: Vec<bool>,
    scopes: Map<OpId, MemScope>,
    header: String,
    body: String,
    indent: String,
    // One slot per op; a Vector(len) slot owns `len` consecutive physical
    // PTX registers, `var_map` holds the physical base register index.
    registers: Vec<PhysRegister>,
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
            // Not/BitNot need dedicated emission (setp+selp / .bN suffix
            // with b-typed registers); a bare name here would miscompile.
            UOp::Not | UOp::BitNot => Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "PTX: Not/BitNot must use the dedicated Unary emission arms".into(),
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

    /// Number of physical PTX registers owned by one logical slot:
    /// the shape's element count. Widths are always concrete here — a
    /// symbolic width cannot lower, and the conversion panics loudly.
    fn slot_len(shape: &[Dim]) -> u16 {
        match shape {
            [] => 1,
            [n] => u16::try_from(*n).expect("PTX: vector width is not a u16"),
            [_, _] => todo!("PTX: tile shape not implemented"),
            _ => todo!("PTX: rank-3+ shape not implemented"),
        }
    }

    fn slot_of(&self, base: u16) -> usize {
        self.registers.iter().position(|r| r.3 == base).expect("PTX: register base without owning slot")
    }

    fn new_reg(&mut self, dtype: DType, shape: Vec<Dim>, rc: u32) -> u16 {
        for reg in self.registers.iter_mut() {
            if reg.1 == 0 && reg.0.0 == dtype && reg.0.1 == shape && self.loop_level <= reg.2 {
                // Freed slot: reuse it (its physical base is unchanged).
                reg.1 = rc;
                reg.2 = self.loop_level;
                return reg.3;
            }
        }
        let base = self.phys_regs;
        self.phys_regs += Self::slot_len(&shape);
        self.registers.push(((dtype, shape), rc, self.loop_level, base));
        base
    }

    fn new_var(&mut self, op_id: OpId, dtype: DType, shape: Vec<Dim>, rc: u32) -> u16 {
        let base = self.new_reg(dtype, shape, rc);
        self.var_map.insert(op_id, base);
        base
    }

    fn get_var(&mut self, x: OpId) -> u16 {
        let r = self.var_map[&x];
        if self.aliases.contains(&x) {
            return r;
        }
        let slot = self.slot_of(r);
        if self.loop_level == self.registers[slot].2 {
            self.registers[slot].1 -= 1;
        }
        r
    }

    /// Like [`Compiler::get_var`], but for address composition only (index
    /// operands at load/store sites): the returned register indexes the inline
    /// address expression and the index op's register must NOT be decremented.
    /// One address expression is routinely shared by several load/store users
    /// (accumulator pattern), and a "consumed per read" account would underflow
    /// on the second user. The index op's register instead stays live for the
    /// whole kernel.
    fn get_var_addr(&mut self, x: OpId) -> u16 {
        self.var_map[&x]
    }

    /// Physical register indices of an operand's components. A Stack groups
    /// already-mapped scalar registers; a Vector-layout op owns consecutive
    /// registers from its base.
    fn components_of(
        &mut self,
        kernel: &Kernel,
        op: OpId,
        dtypes: &Map<OpId, (DType, Vec<Dim>)>,
    ) -> Result<Vec<u16>, BackendError> {
        match &kernel.ops[op].op {
            Op::Stack { ops } => Ok(ops.iter().map(|&x| self.var_map[&x]).collect()),
            _ => {
                let (_, shape) = dtypes[&op].clone();
                let len = Self::slot_len(&shape);
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
    fn emit_load(&mut self, shape: &[Dim], dtype: DType, reg: u16, space: &str) -> Result<(), BackendError> {
        // A shared variable's address is a 32-bit offset in the shared
        // window (`%saddress`); global and local use the generic `%address`.
        let addr = if space == "shared" { "%saddress" } else { "%address" };
        match shape {
            [] => {
                if matches!(dtype, DType::F16 | DType::BF16) {
                    let tmp = self.new_reg(DType::U16, Vec::new(), 1);
                    _ = writeln!(self.body, "{}ld.{}.b16 %r{tmp}, [{addr}];", self.indent, space);
                    _ = writeln!(self.body, "{}mov.b16 %r{reg}, %r{tmp};", self.indent);
                    self.release_reg(tmp);
                } else {
                    _ = writeln!(self.body, "{}ld.{}.{} %r{reg}, [{addr}];", self.indent, space, dtype.mem_ptx());
                }
            }
            [4] if dtype == DType::F32 => {
                _ = writeln!(
                    self.body,
                    "{}ld.{}.v4.f32 {{%r{reg}, %r{}, %r{}, %r{}}}, [{addr}];",
                    self.indent,
                    space,
                    reg + 1,
                    reg + 2,
                    reg + 3
                );
            }
            // A vector of f16 pairs: one 32-bit word per pair, unpacked into
            // one 16-bit slot per element.
            [n] if matches!(dtype, DType::F16 | DType::BF16) && *n % 2 == 0 && *n <= 8 => {
                let words = usize::try_from(n / 2).expect("pack word count");
                let t = self.new_reg(DType::U32, vec![Dim::from(words as i64)], words as u32);
                match words {
                    1 => _ = writeln!(self.body, "{}ld.{}.b32 %r{t}, [{addr}];", self.indent, space),
                    2 => _ = writeln!(
                        self.body,
                        "{}ld.{}.v2.b32 {{%r{t}, %r{}}}, [{addr}];",
                        self.indent,
                        space,
                        t + 1
                    ),
                    _ => _ = writeln!(
                        self.body,
                        "{}ld.{}.v4.b32 {{%r{t}, %r{}, %r{}, %r{}}}, [{addr}];",
                        self.indent,
                        space,
                        t + 1,
                        t + 2,
                        t + 3
                    ),
                }
                for w in 0..words {
                    let (lo, hi) = (reg + 2 * w as u16, reg + 2 * w as u16 + 1);
                    _ = writeln!(self.body, "{}mov.b32 {{%r{lo}, %r{hi}}}, %r{};", self.indent, t + w as u16);
                }
                self.release_reg(t);
            }
            [_] => {
                todo!("PTX: vector load of {shape:?} {dtype:?} not implemented")
            }
            [_, _] => todo!("PTX: tile load not implemented"),
            _ => todo!("PTX: rank-3+ load not implemented"),
        }
        Ok(())
    }

    /// Emit a store of the register(s) at `x` to `[ %address ]` in global space.
    fn emit_store(&mut self, shape: &[Dim], dtype: DType, regs: &[u16]) -> Result<(), BackendError> {
        if matches!(dtype, DType::F16 | DType::BF16) && shape.is_empty() {
            let tmp = self.new_reg(DType::U16, Vec::new(), 1);
            _ = writeln!(self.body, "{}mov.b16 %r{tmp}, %r{};", self.indent, regs[0]);
            _ = writeln!(self.body, "{}st.global.b16 [%address], %r{tmp};", self.indent);
            self.release_reg(tmp);
        } else {
            self.emit_store_space(shape, dtype, regs, "global")?;
        }
        Ok(())
    }

    /// Emit a store of the operand's registers (in component order) to
    /// `[ %address ]`. `space` is one of "global", "shared", "local".
    fn emit_store_space(&mut self, shape: &[Dim], dtype: DType, regs: &[u16], space: &str) -> Result<(), BackendError> {
        // A shared variable's address is a 32-bit offset in the shared
        // window (`%saddress`); global and local use the generic `%address`.
        let addr = if space == "shared" { "%saddress" } else { "%address" };
        let list = |regs: &[u16]| regs.iter().map(|r| format!("%r{r}")).collect::<Vec<_>>().join(", ");
        match shape {
            [] => {
                _ = writeln!(self.body, "{}st.{}.{} [{addr}], %r{};", self.indent, space, dtype.mem_ptx(), regs[0]);
            }
            // A vector of f32 (or any non-f16 element): one `v<n>` store of
            // the operand's registers, in component order — a `Stack` of
            // accumulator slots is exactly this.
            [n] if dtype != DType::F16 && dtype != DType::BF16 && *n <= 4 => {
                let v = usize::try_from(*n).expect("vector width");
                let (op, list) = if v == 1 {
                    (format!("st.{}.{}", space, dtype.mem_ptx()), list(regs))
                } else {
                    (format!("st.{}.v{v}.{}", space, dtype.mem_ptx()), list(regs))
                };
                _ = writeln!(self.body, "{}{op} [{addr}], {{{list}}};", self.indent);
            }
            // A vector of f16 pairs: pack the 16-bit slots into 32-bit words.
            [n] if matches!(dtype, DType::F16 | DType::BF16) && *n % 2 == 0 && *n <= 8 => {
                let words = usize::try_from(n / 2).expect("pack word count");
                let t = self.new_reg(DType::U32, vec![Dim::from(words as i64)], words as u32);
                for w in 0..words {
                    let (lo, hi) = (regs[2 * w], regs[2 * w + 1]);
                    _ = writeln!(self.body, "{}mov.b32 %r{}, {{%r{lo}, %r{hi}}};", self.indent, t + w as u16);
                }
                match words {
                    1 => _ = writeln!(self.body, "{}st.{}.b32 [{addr}], %r{t};", self.indent, space),
                    2 => _ = writeln!(
                        self.body,
                        "{}st.{}.v2.b32 [{addr}], {{%r{t}, %r{}}};",
                        self.indent,
                        space,
                        t + 1
                    ),
                    _ => _ = writeln!(
                        self.body,
                        "{}st.{}.v4.b32 [{addr}], {{%r{t}, %r{}, %r{}, %r{}}};",
                        self.indent,
                        space,
                        t + 1,
                        t + 2,
                        t + 3
                    ),
                }
                self.release_reg(t);
            }
            [_] => {
                todo!("PTX: vector store of {shape:?} {dtype:?} not implemented")
            }
            [_, _] => todo!("PTX: tile store not implemented"),
            _ => todo!("PTX: rank-3+ store not implemented"),
        }
        Ok(())
    }
}

impl Kernel {
    /// Compile kernel to PTX assembly.
    #[allow(clippy::type_complexity)] // complex return type inherent to PTX backend API
    pub(crate) fn generate_ptx(&self, name: &str) -> Result<(Vec<u8>, Vec<Dim>), BackendError> {
        // Reject group lengths that are constant and exceed the device grid limits.
        gws_from_kernel(self, &self.dev_info().max_global_work_dims)?;
        let mut comp = Compiler {
            var_map: Map::default(),
            aliases: Set::default(),
            loops: Vec::new(),
            if_labels: Vec::new(),
            scope_is_bool: Vec::new(),
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
            if steps_op_id > crate::kernel::MAX_OPS as usize {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "generate_ptx: kernel exceeds the op render limit".into(),
                });
            }
            if let Op::Range { axis, kind: scope } = self.ops[op_id].op {
                match scope {
                    RangeKind::Group(_) => {}
                    RangeKind::Local(len) => lws[axis as usize] = Dim::from(len),
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
            if steps_op_id > crate::kernel::MAX_OPS as usize {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "generate_ptx: kernel exceeds the op render limit".into(),
                });
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
            if steps_op_id > crate::kernel::MAX_OPS as usize {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "generate_ptx: kernel exceeds the op render limit".into(),
                });
            }
            match self.ops[op_id].op {
                Op::Param { kind, .. } => match kind {
                    ParamKind::Variable => {
                        // Runtime scalar (IDX_T dims etc.): load into a typed register.
                        let dtype = dtypes[&op_id].0;
                        let rc = rcs.get(&op_id).copied().unwrap_or(0);
                        let reg = comp.new_var(op_id, dtype, Vec::new(), rc);
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
                            // A register-scope storage is a register file, not
                            // memory: reserve `len` registers and index them
                            // statically (the CUDA backend gets this from the
                            // C compiler promoting `T p[len]`). A `.local`
                            // array would put every accumulator access through
                            // local memory.
                            _ = comp.new_var(op_id, dtype, vec![len], u32::MAX);
                        }
                        MemScope::Circular | MemScope::Global => {
                            unreachable!("ptx only supports local or register storage")
                        }
                    }
                }
                Op::Range { axis, kind: scope, .. } => {
                    let rc = rcs.get(&op_id).copied().unwrap_or(0);
                    let reg = comp.new_var(op_id, IDX_T, Vec::new(), rc);
                    let axis_letter = ["x", "y", "z"][axis as usize];
                    // Special registers are 32-bit: read into a u32 temp,
                    // then convert to IDX_T.
                    let t = comp.new_reg(DType::U32, Vec::new(), 1);
                    let src = match scope {
                        RangeKind::Group(_) => "%ctaid",
                        RangeKind::Local(_) => "%tid",
                    };
                    match scope {
                        RangeKind::Group(_) | RangeKind::Local(_) => {
                            _ = writeln!(comp.body, "{}mov.u32 %r{t}, {src}.{};", comp.indent, axis_letter);
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
                    let reg = comp.new_var(op_id, constant.dtype(), Vec::new(), u32::MAX);
                    let ptx_dtype = if constant.dtype() == DType::F16 || constant.dtype() == DType::BF16 {
                        "b16"
                    } else {
                        constant.dtype().ptx()
                    };
                    _ = writeln!(comp.body, "{}mov.{ptx_dtype} %r{reg}, {};", comp.indent, constant.ptx());
                }
                Op::Load { src: buf, src_idx } => {
                    let access_shape = self.shape(op_id);
                    let dtype = dtypes[&buf].0;
                    match comp.get_scope(self, buf) {
                        MemScope::Circular => unreachable!(),
                        MemScope::Global => {
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            let idx = comp.get_var_addr(src_idx);
                            let offset = comp.new_reg(DType::U64, Vec::new(), 1);
                            let reg = comp.new_var(op_id, dtype, access_shape.clone(), rcs[&op_id]);
                            if IDX_T.bit_size() == 64 {
                                if offset != idx {
                                    _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                }
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u64 %address, %p{buf}, %r{offset};", comp.indent);
                            comp.release_reg(offset);
                            comp.emit_load(&access_shape, dtype, reg, "global")?;
                        }
                        MemScope::Local => {
                            let idx = comp.get_var_addr(src_idx);
                            let reg = comp.new_var(op_id, dtype, access_shape.clone(), rcs[&op_id]);
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            // A shared variable's address is a 32-bit offset in
                            // the shared window.
                            _ = writeln!(comp.body, "{}mov.u32 %saddress, __ld{buf};", comp.indent);
                            let t = comp.new_reg(DType::U32, Vec::new(), 1);
                            if IDX_T.bit_size() == 64 {
                                _ = writeln!(comp.body, "{}cvt.u32.u64 %r{t}, %r{idx};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}mov.b32 %r{t}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b32 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u32 %saddress, %saddress, %r{t};", comp.indent);
                            comp.release_reg(t);
                            comp.emit_load(&access_shape, dtype, reg, "shared")?;
                        }
                        MemScope::Register => {
                            // Static component of a register-scope storage: a
                            // register move, not a memory access.
                            let (at, n) = slot_span(self, src_idx);
                            let base = comp.var_map[&buf];
                            let reg = comp.new_var(op_id, dtype, access_shape.clone(), rcs[&op_id]);
                            let len = n.max(u16::try_from(Compiler::slot_len(&access_shape)).expect("slot count"));
                            for i in 0..len {
                                _ = writeln!(
                                    comp.body,
                                    "{}mov.b{} %r{}, %r{};",
                                    comp.indent,
                                    dtype.bit_size(),
                                    reg + i as u16,
                                    base + at + i as u16
                                );
                            }
                        }
                    }
                }
                Op::Store { dst: dst_buf, dst_idx, src: x } => {
                    let access_shape = self.shape(op_id);
                    let dtype = dtypes[&x].0;
                    let byte_shift = (dtype.bit_size() / 8).ilog2();
                    let offset = comp.new_reg(DType::U64, Vec::new(), 1);
                    match comp.get_scope(self, dst_buf) {
                        MemScope::Circular => unreachable!(),
                        MemScope::Global => {
                            if dtype == DType::Bool {
                                let gstu = comp.new_reg(DType::U32, Vec::new(), 1);
                                let idx = comp.get_var_addr(dst_idx);
                                let x = comp.get_var(x);
                                _ = writeln!(comp.body, "{}selp.u32 %r{gstu}, 1, 0, %r{x};", comp.indent);
                                if IDX_T.bit_size() == 64 {
                                    if offset != idx {
                                        _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                    }
                                } else {
                                    _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                                }
                                _ = writeln!(comp.body, "{}add.u64 %address, %p{dst_buf}, %r{offset};", comp.indent);
                                _ = writeln!(comp.body, "{}st.global.u8 [%address], %r{gstu};", comp.indent);
                                comp.release_reg(gstu);
                            } else {
                                let idx = comp.get_var_addr(dst_idx);
                                let regs = comp.components_of(self, x, &dtypes)?;
                                if IDX_T.bit_size() == 64 {
                                    if offset != idx {
                                        _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                    }
                                } else {
                                    _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                                }
                                _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                                _ = writeln!(comp.body, "{}add.u64 %address, %p{dst_buf}, %r{offset};", comp.indent);
                                comp.emit_store(&access_shape, dtype, &regs)?;
                            }
                        }
                        MemScope::Local => {
                            let idx = comp.get_var_addr(dst_idx);
                            let regs = comp.components_of(self, x, &dtypes)?;
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            // A shared variable's address is a 32-bit offset in
                            // the shared window.
                            _ = writeln!(comp.body, "{}mov.u32 %saddress, __ld{dst_buf};", comp.indent);
                            let t = comp.new_reg(DType::U32, Vec::new(), 1);
                            if IDX_T.bit_size() == 64 {
                                _ = writeln!(comp.body, "{}cvt.u32.u64 %r{t}, %r{idx};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}mov.b32 %r{t}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b32 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u32 %saddress, %saddress, %r{t};", comp.indent);
                            comp.release_reg(t);
                            comp.emit_store_space(&access_shape, dtype, &regs, "shared")?;
                        }
                        MemScope::Register => {
                            let (at, _) = slot_span(self, dst_idx);
                            let regs = comp.components_of(self, x, &dtypes)?;
                            let base = comp.var_map[&dst_buf];
                            for (i, &r) in regs.iter().enumerate() {
                                _ = writeln!(
                                    comp.body,
                                    "{}mov.b{} %r{}, %r{r};",
                                    comp.indent,
                                    dtype.bit_size(),
                                    base + at + i as u16
                                );
                            }
                        }
                    }
                    comp.release_reg(offset);
                }
                Op::Cast { x, dtype } => {
                    let xdtype = dtypes[&x].0;
                    debug_assert!(dtypes[&x].1.is_empty(), "PTX: vector cast not implemented");
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
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
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    _ = writeln!(comp.body, "{}mov.b{} %r{reg}, %r{x};", comp.indent, dtype.bit_size());
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x].0;
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    match uop {
                        UOp::Floor => _ = writeln!(comp.body, "{}cvt.rmi.{t}.{t} %r{reg}, %r{x};", comp.indent, t = dtype.ptx()),
                        UOp::Trunc => _ = writeln!(comp.body, "{}cvt.rzi.{t}.{t} %r{reg}, %r{x};", comp.indent, t = dtype.ptx()),
                        UOp::Not => {
                            // Logical not yields 0/1 in the input dtype (C
                            // `!x` semantics). PTX `not` is bitwise-only and
                            // `cnot` writes int bits (0x1 = 1.4e-45 as f32,
                            // not 1.0), so: compare-then-select. The b16
                            // path compares bits (-0.0 reads nonzero);
                            // float paths compare by value.
                            let pred = comp.new_reg(DType::Bool, Vec::new(), 1);
                            let (t, one, zero) = match dtype {
                                DType::F32 => ("f32", "0f3F800000", "0f00000000"),
                                DType::F64 => ("f64", "0d3FF0000000000000", "0d0000000000000000"),
                                DType::F16 => ("b16", "0x3C00", "0x0"),
                                DType::BF16 => ("b16", "0x3F80", "0x0"),
                                DType::U32 | DType::I32 => ("b32", "1", "0"),
                                _ => {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("PTX: logical not unsupported for {dtype:?}").into(),
                                    });
                                }
                            };
                            _ = writeln!(comp.body, "{indent}setp.eq.{t} %r{pred}, %r{x}, {zero};", indent = comp.indent);
                            _ = writeln!(comp.body, "{indent}selp.{t} %r{reg}, {one}, {zero}, %r{pred};", indent = comp.indent);
                            comp.release_reg(pred);
                        }
                        UOp::BitNot => {
                            // PTX `not` exists only as .b16/.b32/.b64 and the
                            // datum must already live in a b-typed register.
                            // No float form exists: loud error, never a
                            // wrong op.
                            let t = match dtype {
                                DType::U32 | DType::I32 => "b32",
                                _ => {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("PTX: bitwise not unsupported for {dtype:?}").into(),
                                    });
                                }
                            };
                            _ = writeln!(comp.body, "{}not.{t} %r{reg}, %r{x};", comp.indent);
                        }
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
                    // A constant operand is written as the literal, not a
                    // register: ptxas folds `x / 8192` and `x % 8192` into a
                    // shift and a mask only when it sees the value, and a
                    // register divisor makes it call the 64-bit integer
                    // division helper instead (which spills the kernel).
                    let imm = |kernel: &Kernel, o: OpId| match kernel.ops[o].op {
                        Op::Const(ref c) => Some(c.ptx()),
                        _ => None,
                    };
                    let xs = imm(self, x).unwrap_or_else(|| format!("%r{}", comp.get_var(x)));
                    let ys = imm(self, y).unwrap_or_else(|| format!("%r{}", comp.get_var(y)));
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
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
                        "{}{}.{} %r{reg}, {xs}, {ys};",
                        comp.indent,
                        comp.bop_to_ptx(bop, dtype),
                        type_ext,
                    );
                }
                // Branchless select as a predicated mov-pair (tinygrad's
                // WHERE lowers the same way): the winning arm's bits are
                // copied, so ±inf on the dead side stays dead. The mov
                // suffix is the register type (reg_ptx), matching the
                // declaration; predication keeps it branchless.
                Op::Select { cond, if_true, if_false } => {
                    let dtype = dtypes[&op_id].0;
                    let cond = comp.get_var(cond);
                    let t = comp.get_var(if_true);
                    let f = comp.get_var(if_false);
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    let mt = dtype.reg_ptx();
                    _ = writeln!(comp.body, "{}@%r{cond} mov.{mt} %r{reg}, %r{t};", comp.indent);
                    _ = writeln!(comp.body, "{}@!%r{cond} mov.{mt} %r{reg}, %r{f};", comp.indent);
                }
                Op::Loop { len } => {
                    comp.loop_level += 1;
                    // Boolean length = conditional (the old `If`); the
                    // length dtype comes from the global dtypes map.
                    if dtypes[&len].0 == DType::Bool {
                        let cond = comp.get_var(len);
                        let endif_label = label as u16;
                        label += 1;
                        comp.if_labels.push(endif_label);
                        comp.scope_is_bool.push(true);
                        _ = writeln!(comp.body, "{}@!%r{cond} bra ENDIF_{endif_label};", comp.indent);
                        comp.indent += "  ";
                        loop_id += 1;
                    } else {
                        let len = comp.get_var(len);
                        let loop_idx = comp.new_var(op_id, IDX_T, Vec::new(), rcs.get(&op_id).copied().unwrap_or(0) + 1);
                        let loop_pred = comp.new_reg(DType::Bool, Vec::new(), 2);
                        comp.loops.push((len, loop_pred, loop_idx));
                        comp.scope_is_bool.push(false);
                        _ = writeln!(comp.body, "{}mov.{} %r{loop_idx}, 0;", comp.indent, IDX_T.ptx());
                        _ = writeln!(comp.body, "{}LOOP_{label}:", comp.indent);
                        loop_id_label_map.insert(loop_id, label);
                        label += 1;
                        comp.indent += "  ";
                        loop_id += 1;
                    }
                }
                Op::EndLoop => {
                    loop_id -= 1;
                    if comp.scope_is_bool.pop() == Some(true) {
                        comp.indent.pop();
                        comp.indent.pop();
                        if let Some(endif_label) = comp.if_labels.pop() {
                            _ = writeln!(comp.body, "{}ENDIF_{endif_label}:", comp.indent);
                        }
                    } else if let Some((len, loop_pred, loop_idx)) = comp.loops.pop() {
                        _ = writeln!(comp.body, "{}add.{} %r{loop_idx}, %r{loop_idx}, 1;", comp.indent, IDX_T.ptx());
                        writeln!(comp.body, "{}setp.lt.{} %r{loop_pred}, %r{loop_idx}, %r{len};", comp.indent, IDX_T.ptx(),)
                            .unwrap();
                        _ = writeln!(comp.body, "{}@%r{loop_pred} bra LOOP_{};", comp.indent, loop_id_label_map[&loop_id]);
                        comp.indent.pop();
                        comp.indent.pop();
                    }
                    comp.loop_level -= 1;
                }
                Op::Barrier => {
                    _ = writeln!(comp.body, "{}bar.sync 1;", comp.indent);
                }
                // A kept lane is the vector width: the access covers its
                // lanes, so the address is the lane-0 base and the value's
                // shape (from `Op::Unroll` in the cone) is the access width.
                Op::Unroll { .. } => {
                    // A lane marker read by every access of the vector: its
                    // value is always zero and it is never released, so it
                    // must not be refcounted (like a constant).
                    let reg = comp.new_var(op_id, DType::I64, Vec::new(), u32::MAX);
                    _ = writeln!(comp.body, "{}mov.s64 %r{reg}, 0;", comp.indent);
                }
                // A Stack is a pure register grouping (e.g. an mma operand
                // fragment): its components are already mapped registers and
                // consumers read them via `components_of`. Nothing to emit.
                Op::Stack { .. } => {}
                Op::Index { vec, idx } => {
                    // Component extract: the component *is* a register of the
                    // vector, so the extraction is a rename, not an
                    // instruction. Consumers then read the vector's own
                    // registers (`LDSM` output straight into `HMMA`, like the
                    // reference), instead of moving through a copy.
                    let (dtype, shape) = dtypes[&vec].clone();
                    debug_assert!(!shape.is_empty(), "PTX: Op::Index on a scalar");
                    let len = Compiler::slot_len(&shape);
                    debug_assert!(idx < len as usize, "PTX: Op::Index {idx} out of bounds for len {len}");
                    let _ = dtype;
                    let vb = comp.components_of(self, vec, &dtypes)?;
                    comp.var_map.insert(op_id, vb[idx]);
                    comp.aliases.insert(op_id);
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
                    debug_assert_eq!(
                        a_comps.len(),
                        4,
                        "PTX: mma A fragment must hold 4 f16 values: a={a:?} {:?}",
                        self.ops[a].op
                    );
                    debug_assert_eq!(b_comps.len(), 2, "PTX: mma B fragment must hold 2 f16 values");
                    debug_assert_eq!(c_comps.len(), 4, "PTX: mma C fragment must hold 4 f32 values");
                    // Pack the f16 pairs into .b32 registers: a = {a01, a23},
                    // b = {b01}.
                    let a01 = comp.new_reg(DType::U32, Vec::new(), 1);
                    let a23 = comp.new_reg(DType::U32, Vec::new(), 1);
                    let b01 = comp.new_reg(DType::U32, Vec::new(), 1);
                    _ = writeln!(comp.body, "{}mov.b32 %r{a01}, {{%r{}, %r{}}};", comp.indent, a_comps[0], a_comps[1]);
                    _ = writeln!(comp.body, "{}mov.b32 %r{a23}, {{%r{}, %r{}}};", comp.indent, a_comps[2], a_comps[3]);
                    _ = writeln!(comp.body, "{}mov.b32 %r{b01}, {{%r{}, %r{}}};", comp.indent, b_comps[0], b_comps[1]);
                    // Result: fresh 4-wide f32 vector, mma.sync writes D in place.
                    let rc = rcs[&op_id];
                    let reg = comp.new_var(op_id, DType::F32, vec![4], rc);
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
                // One 32-bit word read from another lane of the warp, every
                // lane naming its own source.
                Op::PTX(ref p) => match p.as_ref() {
                    PTXOp::Shfl { value, lane } => {
                        let vb = comp.get_var(*value);
                        let l = comp.get_var(*lane);
                        let reg = comp.new_var(op_id, DType::F16, vec![Dim::from(2)], rcs[&op_id]);
                        _ = writeln!(
                            comp.body,
                            "{}shfl.sync.idx.b32 %r{reg}, %r{vb}, %r{l}, 0x1f, 0xffffffff;",
                            comp.indent
                        );
                    }
                    PTXOp::Bytes(_) => todo!("PTX: assembled payload in kernel IR"),
                    // `ldmatrix` reads one 8x8 b16 tile per group of eight
                    // lanes into one register per tile, in lane order.
                    PTXOp::LdMatrix { frags, src, addr, trans } => {
                        let idx = comp.get_var_addr(*addr);
                        let reg = comp.new_var(op_id, DType::F16, vec![Dim::from(i64::from(*frags) * 2)], rcs[&op_id]);
                        // A shared variable's address is a 32-bit offset in
                        // the shared window; the index is in f16 elements.
                        _ = writeln!(comp.body, "{}mov.u32 %saddress, __ld{src};", comp.indent);
                        let off = comp.new_reg(DType::U32, Vec::new(), 1);
                        if IDX_T.bit_size() == 64 {
                            _ = writeln!(comp.body, "{}cvt.u32.u64 %r{off}, %r{idx};", comp.indent);
                        } else {
                            _ = writeln!(comp.body, "{}mov.b32 %r{off}, %r{idx};", comp.indent);
                        }
                        _ = writeln!(comp.body, "{}shl.b32 %r{off}, %r{off}, 1;", comp.indent);
                        _ = writeln!(comp.body, "{}add.u32 %saddress, %saddress, %r{off};", comp.indent);
                        comp.release_reg(off);
                        let t = comp.new_reg(DType::U32, vec![Dim::from(i64::from(*frags))], u32::from(*frags));
                        let t_suffix = if *trans { ".trans" } else { "" };
                        // `ldmatrix` writes one 32-bit word per 8x8 tile, so
                        // the words land in temporaries and are split into the
                        // fragment's two 16-bit halves (`ptxas` coalesces the
                        // moves into the fragment's registers).
                        let pattern =
                            (0..*frags).map(|i| format!("%r{}", t + u16::from(i))).collect::<Vec<_>>().join(", ");
                        _ = writeln!(
                            comp.body,
                            "{}ldmatrix.sync.aligned.m8n8.x{frags}{t_suffix}.b16 {{{pattern}}}, [%saddress];",
                            comp.indent
                        );
                        for i in 0..u16::from(*frags) {
                            _ = writeln!(
                                comp.body,
                                "{}mov.b32 {{%r{}, %r{}}}, %r{};",
                                comp.indent,
                                reg + 2 * i,
                                reg + 2 * i + 1,
                                t + i
                            );
                        }
                        comp.release_reg(t);
                    }
                    // Full-mask warp barrier: the lanes of one warp wait for
                    // each other. A per-warp shared tile needs no more.
                    PTXOp::WarpSync => _ = writeln!(comp.body, "{}bar.warp.sync 0xffffffff;", comp.indent),
                },
                Op::Copy { .. } => {
                    todo!("Copy lowering for PTX (tenstorrent phase)");
                }
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
            if steps_op_id > crate::kernel::MAX_OPS as usize {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "generate_ptx: kernel exceeds the op render limit".into(),
                });
            }
            if matches!(self.ops[op_id].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                _ = writeln!(comp.header, "{}.reg .u64 %p{op_id};", comp.indent);
            }
            op_id = self.next_op(op_id);
        }

        _ = writeln!(comp.header, "{}.reg .u64 %address;", comp.indent);
        _ = writeln!(comp.header, "{}.reg .b32 %saddress;", comp.indent);
        for (dtype_layout, _, _, base) in comp.registers.iter() {
            let len = Compiler::slot_len(&dtype_layout.1);
            for i in 0..len {
                _ = writeln!(comp.header, "{}.reg .{} %r{};", comp.indent, dtype_layout.0.reg_ptx(), base + i);
            }
        }

        comp.header.push_str(&comp.body);
        // cuModuleLoadDataEx reads a nul-terminated image.
        comp.header.push('\0');

        Ok((comp.header.into_bytes(), lws))
    }
}

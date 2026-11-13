// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! PTX assembly codegen from zyx kernel IR.

use crate::{
    DType, Map, Set,
    backend::gws_from_kernel,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MMADims, MMALayout, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind, UOp},
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
            // bf16 lives in a b16 register and every use is a raw bit pattern
            // (`ld/st.b16`, the f32 pack around arithmetic), so the immediate
            // is the 16-bit encoding, exactly like f16.
            Self::BF16(x) => format!("0x{:04X}", u16::from_le_bytes(*x)),
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
/// The slot a register-storage access starts at: lane 0 of its index.
///
/// A compile-time constant, or `base + lane` where the lane is the access
/// width (its value at lane 0 is 0) — the same convention every access uses.
/// A register file has no runtime index, so anything else is a loud error:
/// the IR has to unroll that lane before codegen.
fn slot_base(kernel: &Kernel, idx: OpId) -> u16 {
    let c = |o: OpId| match kernel.at(o) {
        // A loop op *is* the loop index: its value is runtime, never its
        // trip count.
        Op::Loop { .. } => None,
        Op::Unroll { .. } => Some(0),
        _ => kernel.resolve_const(o).and_then(Constant::as_dim),
    };
    let at = match kernel.at(idx) {
        Op::Unroll { .. } => Some(0),
        Op::Binary { x, y, bop: BOp::Add } => match (c(*x), c(*y)) {
            (Some(a), Some(b)) => Some(a + b),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        },
        _ => c(idx),
    };
    at.map(|d| u16::try_from(d).expect("PTX: register slot out of range"))
        .unwrap_or_else(|| todo!("PTX: runtime index into a register storage"))
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
            // bf16 arithmetic rounds at each step: `add.bf16` is not a valid
            // form, `add.rn.bf16` is.
            BOp::Add if dtype == DType::BF16 => "add.rn",
            BOp::Sub if dtype == DType::BF16 => "sub.rn",
            BOp::Add => "add",
            BOp::Sub => "sub",
            BOp::Mul => {
                if dtype == DType::BF16 {
                    "mul.rn"
                } else if dtype.is_float() {
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
                DType::F32 | DType::F64 => Ok("ex2.approx"),
                DType::F16 => Ok("ex2.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: ex2.approx is only available for f32/f64/f16, not {dtype:?}").into(),
                }),
            },
            UOp::Log2 => match dtype {
                DType::F32 | DType::F64 => Ok("lg2.approx"),
                _ => Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("PTX: lg2.approx is only available for f32/f64, not {dtype:?}").into(),
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
    fn emit_load(&mut self, shape: &[Dim], dtype: DType, reg: u16, space: &str, addr: &str) -> Result<(), BackendError> {
        match shape {
            [] if dtype == DType::Bool => {
                // A predicate has no memory form: read the byte, then compare.
                let tmp = self.new_reg(DType::U32, Vec::new(), 1);
                _ = writeln!(self.body, "{}ld.{}.u8 %r{tmp}, [{addr}];", self.indent, space);
                _ = writeln!(self.body, "{}setp.ne.u32 %r{reg}, %r{tmp}, 0;", self.indent);
                self.release_reg(tmp);
            }
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
                    2 => _ = writeln!(self.body, "{}ld.{}.v2.b32 {{%r{t}, %r{}}}, [{addr}];", self.indent, space, t + 1),
                    _ => {
                        _ = writeln!(
                            self.body,
                            "{}ld.{}.v4.b32 {{%r{t}, %r{}, %r{}, %r{}}}, [{addr}];",
                            self.indent,
                            space,
                            t + 1,
                            t + 2,
                            t + 3
                        )
                    }
                }
                for w in 0..words {
                    let (lo, hi) = (reg + 2 * w as u16, reg + 2 * w as u16 + 1);
                    _ = writeln!(self.body, "{}mov.b32 {{%r{lo}, %r{hi}}}, %r{};", self.indent, t + w as u16);
                }
                self.release_reg(t);
            }
            [_] => {
                // PTX vectors hold at most four 32-bit elements, and the
                // register file has no address: a wider (or odd) access is a
                // run of scalar loads at ascending byte offsets. A one-element
                // value broadcasts.
                let n = usize::try_from(shape[0]).expect("PTX: vector width");
                let elem = usize::try_from(dtype.bit_size() / 8).expect("PTX: element bytes");
                for i in 0..n {
                    let off = i * elem;
                    let a = if off == 0 { addr.to_string() } else { format!("{addr}+{off}") };
                    if matches!(dtype, DType::F16 | DType::BF16) {
                        let tmp = self.new_reg(DType::U16, Vec::new(), 1);
                        _ = writeln!(self.body, "{}ld.{}.b16 %r{tmp}, [{a}];", self.indent, space);
                        _ = writeln!(self.body, "{}mov.b16 %r{}, %r{tmp};", self.indent, reg + i as u16);
                        self.release_reg(tmp);
                    } else {
                        _ = writeln!(self.body, "{}ld.{}.{} %r{}, [{a}];", self.indent, space, dtype.mem_ptx(), reg + i as u16);
                    }
                }
            }
            [_, _] => todo!("PTX: tile load not implemented"),
            _ => todo!("PTX: rank-3+ load not implemented"),
        }
        Ok(())
    }

    /// Emit a store of the register(s) at `x` to `[addr]` in global space.
    fn emit_store(&mut self, shape: &[Dim], dtype: DType, regs: &[u16], addr: &str) -> Result<(), BackendError> {
        if matches!(dtype, DType::F16 | DType::BF16) && shape.is_empty() {
            let tmp = self.new_reg(DType::U16, Vec::new(), 1);
            _ = writeln!(self.body, "{}mov.b16 %r{tmp}, %r{};", self.indent, regs[0]);
            _ = writeln!(self.body, "{}st.global.b16 [{addr}], %r{tmp};", self.indent);
            self.release_reg(tmp);
        } else {
            self.emit_store_space(shape, dtype, regs, "global", addr)?;
        }
        Ok(())
    }

    /// Emit a store of the operand's registers (in component order) to
    /// `[addr]`. `space` is one of "global", "shared", "local".
    fn emit_store_space(
        &mut self,
        shape: &[Dim],
        dtype: DType,
        regs: &[u16],
        space: &str,
        addr: &str,
    ) -> Result<(), BackendError> {
        let list = |regs: &[u16]| regs.iter().map(|r| format!("%r{r}")).collect::<Vec<_>>().join(", ");
        match shape {
            [] => {
                _ = writeln!(self.body, "{}st.{}.{} [{addr}], %r{};", self.indent, space, dtype.mem_ptx(), regs[0]);
            }
            // A vector of f32 (or any non-f16 element): one `v<n>` store of
            // the operand's registers, in component order — a `Stack` of
            // accumulator slots is exactly this. A value with a single
            // component broadcasts into every lane.
            [n] if dtype != DType::F16 && dtype != DType::BF16 && *n <= 4 => {
                let v = usize::try_from(*n).expect("vector width");
                let regs: Vec<u16> = if regs.len() == 1 { vec![regs[0]; v] } else { regs.to_vec() };
                let (op, list) = if v == 1 {
                    (format!("st.{}.{}", space, dtype.mem_ptx()), list(&regs))
                } else {
                    (format!("st.{}.v{v}.{}", space, dtype.mem_ptx()), list(&regs))
                };
                _ = writeln!(self.body, "{}{op} [{addr}], {{{list}}};", self.indent);
            }
            // A vector of f16 pairs: pack the 16-bit slots into 32-bit words.
            [n] if matches!(dtype, DType::F16 | DType::BF16) && *n % 2 == 0 && *n <= 8 => {
                let words = usize::try_from(n / 2).expect("pack word count");
                let t = self.new_reg(DType::U32, vec![Dim::from(words as i64)], words as u32);
                for w in 0..words {
                    // A one-component value broadcasts into every lane.
                    let (lo, hi) = if regs.len() == 1 {
                        (regs[0], regs[0])
                    } else {
                        (regs[2 * w], regs[2 * w + 1])
                    };
                    _ = writeln!(self.body, "{}mov.b32 %r{}, {{%r{lo}, %r{hi}}};", self.indent, t + w as u16);
                }
                match words {
                    1 => _ = writeln!(self.body, "{}st.{}.b32 [{addr}], %r{t};", self.indent, space),
                    2 => _ = writeln!(self.body, "{}st.{}.v2.b32 [{addr}], {{%r{t}, %r{}}};", self.indent, space, t + 1),
                    _ => {
                        _ = writeln!(
                            self.body,
                            "{}st.{}.v4.b32 [{addr}], {{%r{t}, %r{}, %r{}, %r{}}};",
                            self.indent,
                            space,
                            t + 1,
                            t + 2,
                            t + 3
                        )
                    }
                }
                self.release_reg(t);
            }
            [_] => {
                // PTX vectors hold at most four 32-bit elements, and the
                // register file has no address: a wider (or odd) access is a
                // run of scalar stores at ascending byte offsets. A value with
                // a single component broadcasts into every lane.
                let n = usize::try_from(shape[0]).expect("PTX: vector width");
                let elem = usize::try_from(dtype.bit_size() / 8).expect("PTX: element bytes");
                for i in 0..n {
                    let off = i * elem;
                    let a = if off == 0 { addr.to_string() } else { format!("{addr}+{off}") };
                    let r = if regs.len() == 1 { regs[0] } else { regs[i] };
                    if matches!(dtype, DType::F16 | DType::BF16) {
                        let tmp = self.new_reg(DType::U16, Vec::new(), 1);
                        _ = writeln!(self.body, "{}mov.b16 %r{tmp}, %r{r};", self.indent);
                        _ = writeln!(self.body, "{}st.{}.b16 [{a}], %r{tmp};", self.indent, space);
                        self.release_reg(tmp);
                    } else {
                        _ = writeln!(self.body, "{}st.{}.{} [{a}], %r{r};", self.indent, space, dtype.mem_ptx());
                    }
                }
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
        // bf16 is its own 16-bit type with its own instructions: `cvt` and
        // `min`/`max`/`neg`/`abs` from sm_80, `add`/`sub`/`mul` from sm_90
        // (Ampere has bf16 tensor cores, not scalar bf16 arithmetic). Nothing
        // is emulated here: a kernel that needs an instruction the target does
        // not have is rejected, so any emulation is the caller's.
        let bf16_cvt = cc[0] >= 8;
        let bf16_arith = cc[0] >= 9;
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
                            // A shared buffer of f16/bf16 is an array of 16-bit
                            // words; `.bf16` does not exist below sm_80.
                            let elem = if dtype == DType::BF16 { "b16" } else { dtype.ptx() };
                            _ = writeln!(comp.body, "{}.shared .align 16 .{elem} __ld{op_id}[{len}];", comp.indent);
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
                    if constant.dtype() == DType::U8 || constant.dtype() == DType::I8 {
                        // No `mov.u8`: materialize the byte through a 16-bit
                        // register and narrow, which keeps the register a
                        // byte wide for the `ld/st.u8` it feeds.
                        let (t, outcvt) = if constant.dtype() == DType::I8 {
                            (DType::I16, "cvt.s8.s16")
                        } else {
                            (DType::U16, "cvt.u8.u16")
                        };
                        let tmp = comp.new_reg(t, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}mov.{} %r{tmp}, {};", comp.indent, t.ptx(), constant.ptx());
                        _ = writeln!(comp.body, "{}{outcvt} %r{reg}, %r{tmp};", comp.indent);
                        comp.release_reg(tmp);
                    } else {
                        let ptx_dtype = if constant.dtype() == DType::F16 || constant.dtype() == DType::BF16 {
                            "b16"
                        } else {
                            constant.dtype().ptx()
                        };
                        _ = writeln!(comp.body, "{}mov.{ptx_dtype} %r{reg}, {};", comp.indent, constant.ptx());
                    }
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
                            // One address register per access: a shared
                            // `%address` scratch makes every load war-write it,
                            // so the address math of consecutive accesses cannot
                            // overlap (and ptxas must serialize them).
                            let address = comp.new_reg(DType::U64, Vec::new(), 1);
                            if IDX_T.bit_size() == 64 {
                                if offset != idx {
                                    _ = writeln!(comp.body, "{}mov.u64 %r{offset}, %r{idx};", comp.indent);
                                }
                            } else {
                                _ = writeln!(comp.body, "{}cvt.u64.u32 %r{offset}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b64 %r{offset}, %r{offset}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u64 %r{address}, %p{buf}, %r{offset};", comp.indent);
                            comp.release_reg(offset);
                            let addr = format!("%r{address}");
                            // A read-only parameter is never written by the
                            // kernel, so its loads may take the non-coherent
                            // path: `ld.global.nc` renders as
                            // `LDG.E...CONSTANT` (the read-only cache) and
                            // frees ptxas to hoist the load, which is what the
                            // reference's `LDG.E.LTC128B.128.CONSTANT.GPU` is.
                            // `.L2::128B` is the 128-byte L2 sector promotion
                            // that gives it the `.LTC128B` modifier: a hint
                            // only, so it changes no value, and the reads are
                            // streaming tiles whose neighbours are used next.
                            let space = match self.ops[buf].op {
                                Op::Param { kind: ParamKind::Global, .. } => "global.nc.L2::128B",
                                _ => "global",
                            };
                            comp.emit_load(&access_shape, dtype, reg, space, &addr)?;
                            comp.release_reg(address);
                        }
                        MemScope::Local => {
                            let idx = comp.get_var_addr(src_idx);
                            let reg = comp.new_var(op_id, dtype, access_shape.clone(), rcs[&op_id]);
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            // A shared variable's address is a 32-bit offset in
                            // the shared window, in a register of its own.
                            let address = comp.new_reg(DType::U32, Vec::new(), 1);
                            _ = writeln!(comp.body, "{}mov.u32 %r{address}, __ld{buf};", comp.indent);
                            let t = comp.new_reg(DType::U32, Vec::new(), 1);
                            if IDX_T.bit_size() == 64 {
                                _ = writeln!(comp.body, "{}cvt.u32.u64 %r{t}, %r{idx};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}mov.b32 %r{t}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b32 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u32 %r{address}, %r{address}, %r{t};", comp.indent);
                            comp.release_reg(t);
                            let addr = format!("%r{address}");
                            comp.emit_load(&access_shape, dtype, reg, "shared", &addr)?;
                            comp.release_reg(address);
                        }
                        MemScope::Register => {
                            // Static component of a register-scope storage: a
                            // register move, not a memory access. A register
                            // file has no runtime index — the IR unrolls the
                            // lane before codegen, so the slot is a constant.
                            let at = slot_base(self, src_idx);
                            let base = comp.var_map[&buf];
                            let reg = comp.new_var(op_id, dtype, access_shape.clone(), rcs[&op_id]);
                            let len = u16::try_from(Compiler::slot_len(&access_shape)).expect("slot count");
                            for i in 0..len {
                                let dst = reg + i as u16;
                                let src = base + at + i as u16;
                                // A predicate register moves as `mov.pred`:
                                // `mov.b8` is not an instruction ptxas takes.
                                if dtype == DType::Bool {
                                    _ = writeln!(comp.body, "{}mov.pred %r{dst}, %r{src};", comp.indent);
                                } else {
                                    _ = writeln!(comp.body, "{}mov.b{} %r{dst}, %r{src};", comp.indent, dtype.bit_size());
                                }
                            }
                        }
                    }
                }
                Op::Store { dst: dst_buf, dst_idx, src: x } => {
                    // The access shape rides on the index: a store's value may
                    // be narrower (a broadcast) or an instruction result with
                    // no lane shape of its own.
                    let access_shape = self.shape(dst_idx);
                    let dtype = dtypes[&x].0;
                    let byte_shift = (dtype.bit_size() / 8).ilog2();
                    let offset = comp.new_reg(DType::U64, Vec::new(), 1);
                    match comp.get_scope(self, dst_buf) {
                        MemScope::Circular => unreachable!(),
                        MemScope::Global => {
                            // One address register per access: a shared
                            // `%address` scratch makes every store war-write it,
                            // so consecutive accesses cannot overlap their
                            // address math and ptxas must serialize them.
                            let address = comp.new_reg(DType::U64, Vec::new(), 1);
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
                                _ = writeln!(comp.body, "{}add.u64 %r{address}, %p{dst_buf}, %r{offset};", comp.indent);
                                _ = writeln!(comp.body, "{}st.global.u8 [%r{address}], %r{gstu};", comp.indent);
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
                                _ = writeln!(comp.body, "{}add.u64 %r{address}, %p{dst_buf}, %r{offset};", comp.indent);
                                let addr = format!("%r{address}");
                                comp.emit_store(&access_shape, dtype, &regs, &addr)?;
                            }
                            comp.release_reg(address);
                        }
                        MemScope::Local => {
                            let idx = comp.get_var_addr(dst_idx);
                            let regs = comp.components_of(self, x, &dtypes)?;
                            let byte_shift = (dtype.bit_size() / 8).ilog2();
                            // A shared variable's address is a 32-bit offset in
                            // the shared window, in a register of its own.
                            let address = comp.new_reg(DType::U32, Vec::new(), 1);
                            _ = writeln!(comp.body, "{}mov.u32 %r{address}, __ld{dst_buf};", comp.indent);
                            let t = comp.new_reg(DType::U32, Vec::new(), 1);
                            if IDX_T.bit_size() == 64 {
                                _ = writeln!(comp.body, "{}cvt.u32.u64 %r{t}, %r{idx};", comp.indent);
                            } else {
                                _ = writeln!(comp.body, "{}mov.b32 %r{t}, %r{idx};", comp.indent);
                            }
                            _ = writeln!(comp.body, "{}shl.b32 %r{t}, %r{t}, {byte_shift};", comp.indent);
                            _ = writeln!(comp.body, "{}add.u32 %r{address}, %r{address}, %r{t};", comp.indent);
                            comp.release_reg(t);
                            let addr = format!("%r{address}");
                            comp.emit_store_space(&access_shape, dtype, &regs, "shared", &addr)?;
                            comp.release_reg(address);
                        }
                        MemScope::Register => {
                            let at = slot_base(self, dst_idx);
                            let regs = comp.components_of(self, x, &dtypes)?;
                            let base = comp.var_map[&dst_buf];
                            for (i, &r) in regs.iter().enumerate() {
                                let dst = base + at + i as u16;
                                // A predicate register moves as `mov.pred`:
                                // `mov.b8` is not an instruction ptxas takes.
                                if dtype == DType::Bool {
                                    _ = writeln!(comp.body, "{}mov.pred %r{dst}, %r{r};", comp.indent);
                                } else {
                                    _ = writeln!(comp.body, "{}mov.b{} %r{dst}, %r{r};", comp.indent, dtype.bit_size());
                                }
                            }
                        }
                    }
                    comp.release_reg(offset);
                }
                Op::Cast { x, dtype } => {
                    let xdtype = dtypes[&x].0;
                    // A lane here is a kept vector width, not an element-wise
                    // vector value: `Op::Unroll` renders as zero, so this
                    // casts the lane-0 base and the access covers the rest
                    // (same convention as `Binary` below). A real per-element
                    // cast never reaches codegen: `devectorize` unfolds it.
                    debug_assert!(dtypes[&x].1.len() <= 1, "PTX: element-wise cast of a nested vector is not implemented");
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    // bf16 conversions are `cvt` instructions (sm_80+). There is
                    // no emulation: below sm_80 the kernel is rejected.
                    if (dtype == DType::BF16 || xdtype == DType::BF16) && dtype != xdtype {
                        if !bf16_cvt {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("PTX: no bf16 conversion for sm_{}{} (cvt.bf16 needs sm_80)", cc[0], cc[1])
                                    .into(),
                            });
                        }
                        let f = comp.new_reg(DType::F32, Vec::new(), 1);
                        match xdtype {
                            DType::BF16 => _ = writeln!(comp.body, "{}cvt.f32.bf16 %r{f}, %r{x};", comp.indent),
                            DType::F16 => _ = writeln!(comp.body, "{}cvt.f32.f16 %r{f}, %r{x};", comp.indent),
                            DType::F32 => _ = writeln!(comp.body, "{}mov.f32 %r{f}, %r{x};", comp.indent),
                            DType::Bool => _ = writeln!(comp.body, "{}selp.f32 %r{f}, 1.0, 0.0, %r{x};", comp.indent),
                            _ => _ = writeln!(comp.body, "{}cvt.rn.f32.{} %r{f}, %r{x};", comp.indent, xdtype.ptx()),
                        }
                        match dtype {
                            DType::F32 => _ = writeln!(comp.body, "{}mov.f32 %r{reg}, %r{f};", comp.indent),
                            DType::F16 => _ = writeln!(comp.body, "{}cvt.rn.f16.f32 %r{reg}, %r{f};", comp.indent),
                            DType::BF16 => _ = writeln!(comp.body, "{}cvt.rn.bf16.f32 %r{reg}, %r{f};", comp.indent),
                            _ => _ = writeln!(comp.body, "{}cvt.rni.{}.f32 %r{reg}, %r{f};", comp.indent, dtype.ptx()),
                        }
                        comp.release_reg(f);
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    match (dtype, xdtype) {
                        (DType::Bool, _) => {
                            // bf16 has no `setp`: nonzero bits are a nonzero
                            // value (NaN included), which is what C yields.
                            let t = if xdtype == DType::BF16 { "u16" } else { xdtype.ptx() };
                            if dtype.is_float() {
                                _ = writeln!(comp.body, "{}setp.ne.{} %r{reg}, %r{x}, 0.0;", comp.indent, t);
                            } else {
                                _ = writeln!(comp.body, "{}setp.ne.{} %r{reg}, %r{x}, 0;", comp.indent, t);
                            }
                        }
                        (_, DType::Bool) => {
                            if dtype == DType::F64 || dtype == DType::F32 {
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, 1.0, 0.0, %r{x};", comp.indent, dtype.ptx());
                            } else if dtype == DType::F16 {
                                _ = writeln!(comp.body, "{}selp.b16 %r{reg}, 0x3C00, 0, %r{x};", comp.indent);
                            } else if dtype == DType::BF16 {
                                _ = writeln!(comp.body, "{}selp.b16 %r{reg}, 0x3F80, 0, %r{x};", comp.indent);
                            } else if dtype == DType::U8 || dtype == DType::I8 {
                                // `selp.u8` does not exist: pick in 32 bits and
                                // truncate into the byte register.
                                let (wide, outcvt) = if dtype == DType::I8 {
                                    (DType::I32, "cvt.s8.s32")
                                } else {
                                    (DType::U32, "cvt.u8.u32")
                                };
                                let t = comp.new_reg(wide, Vec::new(), 1);
                                _ = writeln!(comp.body, "{}selp.{} %r{t}, 1, 0, %r{x};", comp.indent, wide.ptx());
                                _ = writeln!(comp.body, "{}{outcvt} %r{reg}, %r{t};", comp.indent);
                                comp.release_reg(t);
                            } else {
                                _ = writeln!(comp.body, "{}selp.{} %r{reg}, 1, 0, %r{x};", comp.indent, dtype.ptx());
                            }
                        }
                        (DType::I32, DType::F32) => {
                            _ = writeln!(comp.body, "{}cvt.rni.{}.{} %r{reg}, %r{x};", comp.indent, dtype.ptx(), xdtype.ptx());
                        }
                        _ if dtype == xdtype => {
                            if reg != x {
                                _ = writeln!(comp.body, "{}mov.{} %r{reg}, %r{x};", comp.indent, dtype.reg_ptx());
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
                    let xdtype = dtypes[&x].0;
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    if (dtype == DType::U8 || dtype == DType::I8) && (xdtype == DType::U8 || xdtype == DType::I8) {
                        // `mov.b8` does not exist; widen through s32 and take
                        // the low byte back, which preserves the two's-
                        // complement bits.
                        let t = comp.new_reg(DType::I32, Vec::new(), 1);
                        let incvt = if xdtype == DType::I8 { "cvt.s32.s8" } else { "cvt.s32.u8" };
                        let outcvt = if dtype == DType::I8 { "cvt.s8.s32" } else { "cvt.u8.s32" };
                        _ = writeln!(comp.body, "{}{incvt} %r{t}, %r{x};", comp.indent);
                        _ = writeln!(comp.body, "{}{outcvt} %r{reg}, %r{t};", comp.indent);
                        comp.release_reg(t);
                    } else {
                        _ = writeln!(comp.body, "{}mov.b{} %r{reg}, %r{x};", comp.indent, dtype.bit_size());
                    }
                }
                Op::Unary { x, uop } => {
                    let dtype = dtypes[&x].0;
                    let x = comp.get_var(x);
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    // sm_75 has no f16 `sqrt`/`rsqrt`/`lg2`/`rcp`/`sin`/`cos`:
                    // those run in f32 and the result narrows back. bf16 is
                    // never emulated: `neg`/`abs` need their own instruction
                    // (sm_80), everything else has none at any target.
                    if dtype == DType::BF16 && !matches!(uop, UOp::Not | UOp::BitNot) {
                        let native = bf16_cvt && matches!(uop, UOp::Neg | UOp::Abs);
                        if !native {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("PTX: no bf16 {uop:?} instruction for sm_{}{}", cc[0], cc[1]).into(),
                            });
                        }
                    }
                    let widen = match uop {
                        UOp::Not | UOp::BitNot => false,
                        UOp::Log2 | UOp::Sqrt | UOp::Rsqrt | UOp::Reciprocal | UOp::Sin | UOp::Cos if dtype == DType::F16 => true,
                        _ => false,
                    };
                    if widen {
                        let f = comp.new_reg(DType::F32, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}cvt.f32.f16 %r{f}, %r{x};", comp.indent);
                        match uop {
                            UOp::Floor => _ = writeln!(comp.body, "{}cvt.rmi.f32.f32 %r{f}, %r{f};", comp.indent),
                            UOp::Trunc => _ = writeln!(comp.body, "{}cvt.rzi.f32.f32 %r{f}, %r{f};", comp.indent),
                            _ => {
                                _ = writeln!(comp.body, "{}{}.f32 %r{f}, %r{f};", comp.indent, comp.uop_to_ptx(uop, DType::F32)?)
                            }
                        }
                        _ = writeln!(comp.body, "{}cvt.rn.f16.f32 %r{reg}, %r{f};", comp.indent);
                        comp.release_reg(f);
                        op_id = self.next_op(op_id);
                        continue;
                    }
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
                                DType::F16 => ("u16", "0x3C00", "0x0"),
                                DType::BF16 => ("u16", "0x3F80", "0x0"),
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
                                DType::U16 | DType::I16 | DType::F16 | DType::BF16 => "b16",
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
                    // f16/bf16 immediates are not accepted by `setp` below
                    // sm_80, so those stay in the register the `Const` op owns.
                    let imm = |kernel: &Kernel, o: OpId| match kernel.ops[o].op {
                        Op::Const(ref c) if !matches!(c.dtype(), DType::F16 | DType::BF16) => Some(c.ptx()),
                        _ => None,
                    };
                    let xs = imm(self, x).unwrap_or_else(|| format!("%r{}", comp.get_var(x)));
                    let ys = imm(self, y).unwrap_or_else(|| format!("%r{}", comp.get_var(y)));
                    let reg = comp.new_var(op_id, dtype, Vec::new(), rcs[&op_id]);
                    // `setp` has no predicate form: compare the two predicates
                    // as 0/1 words and write the predicate. `and`/`or` do have
                    // one, so only the comparisons take this path.
                    let predicate_compare = matches!(bop, BOp::Eq | BOp::NotEq | BOp::Cmplt | BOp::Cmpgt | BOp::Cmpge);
                    if bop.returns_bool() && dtypes[&x].0 == DType::Bool && predicate_compare {
                        // `setp` has no predicate form: compare the two
                        // predicates as 0/1 words and write the predicate.
                        let t1 = comp.new_reg(DType::U32, Vec::new(), 1);
                        let t2 = comp.new_reg(DType::U32, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}selp.u32 %r{t1}, 1, 0, {xs};", comp.indent);
                        _ = writeln!(comp.body, "{}selp.u32 %r{t2}, 1, 0, {ys};", comp.indent);
                        _ = writeln!(comp.body, "{}{}.u32 %r{reg}, %r{t1}, %r{t2};", comp.indent, comp.bop_to_ptx(bop, dtype));
                        comp.release_reg(t1);
                        comp.release_reg(t2);
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    let xdtype = dtypes[&x].0;
                    // bf16 has its own instructions, and only some of them: a
                    // target that lacks the one this op needs rejects the
                    // kernel (nothing is emulated here).
                    if xdtype == DType::BF16 {
                        let native = match bop {
                            BOp::Add | BOp::Sub | BOp::Mul => bf16_arith,
                            BOp::Max => bf16_cvt,
                            BOp::And | BOp::Or | BOp::BitAnd | BOp::BitOr | BOp::BitXor => true,
                            _ => false,
                        };
                        if !native {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("PTX: no bf16 {bop:?} instruction for sm_{}{}", cc[0], cc[1]).into(),
                            });
                        }
                    }
                    let bitwise = matches!(
                        bop,
                        BOp::BitShiftLeft | BOp::BitShiftRight | BOp::BitAnd | BOp::BitOr | BOp::BitXor | BOp::And | BOp::Or
                    );
                    // PTX has no 8-bit `mov`, logic, arithmetic or shift
                    // (only `ld/st/cvt` take `.u8`), and its `shl/shr` stop at
                    // `.b32`: those run in 32 bits, truncated back afterwards.
                    // The value itself stays a byte/half in its own register so
                    // `ld/st.u8` remains a byte access.
                    let shift = matches!(bop, BOp::BitShiftLeft | BOp::BitShiftRight);
                    let byte = xdtype == DType::U8 || xdtype == DType::I8;
                    let shift16 = shift && (xdtype == DType::U16 || xdtype == DType::I16);
                    if byte || shift16 {
                        let signed = xdtype == DType::I8 || xdtype == DType::I16;
                        let wide = if signed { DType::I32 } else { DType::U32 };
                        let (cvt_in, cvt_out) = if xdtype == DType::I8 {
                            ("cvt.s32.s8", "cvt.s8.s32")
                        } else if xdtype == DType::U8 {
                            ("cvt.u32.u8", "cvt.u8.u32")
                        } else if xdtype == DType::I16 {
                            ("cvt.s32.s16", "cvt.s16.s32")
                        } else {
                            ("cvt.u32.u16", "cvt.u16.u32")
                        };
                        let ta = comp.new_reg(wide, Vec::new(), 1);
                        if let Some(lit) = imm(self, x) {
                            _ = writeln!(comp.body, "{}mov.{} %r{ta}, {lit};", comp.indent, wide.ptx());
                        } else {
                            _ = writeln!(comp.body, "{}{cvt_in} %r{ta}, {xs};", comp.indent);
                        }
                        // The shift amount has its own dtype (often a u16
                        // constant), so it widens by what it is, not by what
                        // the left operand is.
                        let yt = dtypes[&y].0;
                        let ycvt = match yt {
                            DType::U8 => "cvt.u32.u8",
                            DType::I8 => "cvt.s32.s8",
                            DType::U16 => "cvt.u32.u16",
                            DType::I16 => "cvt.s32.s16",
                            DType::U64 => "cvt.u32.u64",
                            DType::I64 => "cvt.s32.s64",
                            _ => "",
                        };
                        let tb = comp.new_reg(wide, Vec::new(), 1);
                        if let Some(lit) = imm(self, y) {
                            _ = writeln!(comp.body, "{}mov.{} %r{tb}, {lit};", comp.indent, wide.ptx());
                        } else if ycvt.is_empty() {
                            _ = writeln!(comp.body, "{}mov.b32 %r{tb}, {ys};", comp.indent);
                        } else {
                            _ = writeln!(comp.body, "{}{ycvt} %r{tb}, {ys};", comp.indent);
                        }
                        let suffix = match bop {
                            // A signed right shift is arithmetic.
                            BOp::BitShiftRight if signed => "s32",
                            _ if bitwise => "b32",
                            _ => wide.ptx(),
                        };
                        if bop.returns_bool() {
                            _ = writeln!(
                                comp.body,
                                "{}{}.{suffix} %r{reg}, %r{ta}, %r{tb};",
                                comp.indent,
                                comp.bop_to_ptx(bop, wide)
                            );
                        } else {
                            let f = comp.new_reg(wide, Vec::new(), 1);
                            _ = writeln!(
                                comp.body,
                                "{}{}.{suffix} %r{f}, %r{ta}, %r{tb};",
                                comp.indent,
                                comp.bop_to_ptx(bop, wide)
                            );
                            _ = writeln!(comp.body, "{}{cvt_out} %r{reg}, %r{f};", comp.indent);
                            comp.release_reg(f);
                        }
                        comp.release_reg(ta);
                        comp.release_reg(tb);
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    // sm_75 has no f16 `div`/`max`: those run in f32 and narrow
                    // back. bf16 is never widened here — its own instructions
                    // are used, and a target without them rejects the kernel.
                    let widen = !bitwise && xdtype == DType::F16 && matches!(bop, BOp::Div | BOp::Max);
                    if widen {
                        let fa = comp.new_reg(DType::F32, Vec::new(), 1);
                        let fb = comp.new_reg(DType::F32, Vec::new(), 1);
                        for (dst, s, o) in [(fa, &xs, x), (fb, &ys, y)] {
                            if let Some(lit) = imm(self, o) {
                                let t = comp.new_reg(DType::U16, Vec::new(), 1);
                                _ = writeln!(comp.body, "{}mov.u16 %r{t}, {lit};", comp.indent);
                                _ = writeln!(comp.body, "{}cvt.f32.f16 %r{dst}, %r{t};", comp.indent);
                                comp.release_reg(t);
                            } else {
                                _ = writeln!(comp.body, "{}cvt.f32.f16 %r{dst}, {s};", comp.indent);
                            }
                        }
                        let f = comp.new_reg(DType::F32, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}{}.f32 %r{f}, %r{fa}, %r{fb};", comp.indent, comp.bop_to_ptx(bop, DType::F32));
                        _ = writeln!(comp.body, "{}cvt.rn.f16.f32 %r{reg}, %r{f};", comp.indent);
                        comp.release_reg(f);
                        comp.release_reg(fa);
                        comp.release_reg(fb);
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    let type_ext = match bop {
                        // Bit-size instructions: PTX has no signed or unsigned
                        // form for them, and `.s64`/`.u64` are rejected by
                        // ptxas. Booleans keep the predicate form.
                        BOp::BitShiftLeft | BOp::BitShiftRight | BOp::BitAnd | BOp::BitOr | BOp::BitXor | BOp::And | BOp::Or => {
                            match dtypes[&x].0 {
                                DType::Bool => "pred",
                                dtype => match dtype.bit_size() {
                                    16 => "b16",
                                    32 => "b32",
                                    64 => "b64",
                                    n => {
                                        return Err(BackendError {
                                            status: ErrorStatus::KernelCompilation,
                                            context: format!("PTX: unsupported bitwise bit size {n}").into(),
                                        });
                                    }
                                },
                            }
                        }
                        _ => dtypes[&x].0.ptx(),
                    };
                    _ = writeln!(comp.body, "{}{}.{} %r{reg}, {xs}, {ys};", comp.indent, comp.bop_to_ptx(bop, dtype), type_ext,);
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
                    if dtype == DType::U8 || dtype == DType::I8 {
                        // No `mov.u8`: pick in 32 bits and truncate back.
                        let (wide, incvt, outcvt) = if dtype == DType::I8 {
                            (DType::I32, "cvt.s32.s8", "cvt.s8.s32")
                        } else {
                            (DType::U32, "cvt.u32.u8", "cvt.u8.u32")
                        };
                        let wt = comp.new_reg(wide, Vec::new(), 1);
                        let wf = comp.new_reg(wide, Vec::new(), 1);
                        let w = comp.new_reg(wide, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}{incvt} %r{wt}, %r{t};", comp.indent);
                        _ = writeln!(comp.body, "{}{incvt} %r{wf}, %r{f};", comp.indent);
                        _ = writeln!(comp.body, "{}selp.{} %r{w}, %r{wt}, %r{wf}, %r{cond};", comp.indent, wide.ptx());
                        _ = writeln!(comp.body, "{}{outcvt} %r{reg}, %r{w};", comp.indent);
                        comp.release_reg(wt);
                        comp.release_reg(wf);
                        comp.release_reg(w);
                    } else {
                        let mt = dtype.reg_ptx();
                        _ = writeln!(comp.body, "{}@%r{cond} mov.{mt} %r{reg}, %r{t};", comp.indent);
                        _ = writeln!(comp.body, "{}@!%r{cond} mov.{mt} %r{reg}, %r{f};", comp.indent);
                    }
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
                    // A scalar operand broadcasts into every slot: `a`, `b`
                    // and `c` are all allowed to be one value, so a
                    // single-component operand names its register once per slot
                    // the instruction reads. Any other width is a fragment that
                    // does not match the instruction.
                    let fragment = |mut comps: Vec<u16>, slots: usize| -> Vec<u16> {
                        if comps.len() == 1 {
                            comps.resize(slots, comps[0]);
                        }
                        comps
                    };
                    let a_comps = fragment(comp.components_of(self, a, &dtypes)?, 4);
                    let b_comps = fragment(comp.components_of(self, b, &dtypes)?, 2);
                    let c_comps = fragment(comp.components_of(self, c, &dtypes)?, 4);
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
                        _ = writeln!(comp.body, "{}shfl.sync.idx.b32 %r{reg}, %r{vb}, %r{l}, 0x1f, 0xffffffff;", comp.indent);
                    }
                    PTXOp::Bytes(_) => todo!("PTX: assembled payload in kernel IR"),
                    // `ldmatrix` reads one 8x8 b16 tile per group of eight
                    // lanes into one register per tile, in lane order.
                    PTXOp::VLdMatrix { frags, src, addr, trans } => {
                        let idx = comp.get_var_addr(*addr);
                        let reg = comp.new_var(op_id, DType::F16, vec![Dim::from(i64::from(*frags) * 2)], rcs[&op_id]);
                        // A shared variable's address is a 32-bit offset in the
                        // shared window; the index is in f16 elements. One
                        // register per load: the reference keeps a base per
                        // tile row and reaches its sub-tiles with immediates, so
                        // the address math is not a serial chain through one
                        // scratch register.
                        let saddr = comp.new_reg(DType::U32, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}mov.u32 %r{saddr}, __ld{src};", comp.indent);
                        let off = comp.new_reg(DType::U32, Vec::new(), 1);
                        if IDX_T.bit_size() == 64 {
                            _ = writeln!(comp.body, "{}cvt.u32.u64 %r{off}, %r{idx};", comp.indent);
                        } else {
                            _ = writeln!(comp.body, "{}mov.b32 %r{off}, %r{idx};", comp.indent);
                        }
                        _ = writeln!(comp.body, "{}shl.b32 %r{off}, %r{off}, 1;", comp.indent);
                        _ = writeln!(comp.body, "{}add.u32 %r{saddr}, %r{saddr}, %r{off};", comp.indent);
                        comp.release_reg(off);
                        // The instruction takes the generic address of the
                        // shared slot (`mov.u32` yields the window address; a
                        // 32-bit operand is left for the driver's JIT to
                        // resolve and it does not always convert it).
                        let rd = comp.new_reg(DType::U64, Vec::new(), 1);
                        _ = writeln!(comp.body, "{}cvt.u64.u32 %r{rd}, %r{saddr};", comp.indent);
                        _ = writeln!(comp.body, "{}cvta.shared.u64 %r{rd}, %r{rd};", comp.indent);
                        comp.release_reg(saddr);
                        let t = comp.new_reg(DType::U32, vec![Dim::from(i64::from(*frags))], u32::from(*frags));
                        let t_suffix = if *trans { ".trans" } else { "" };
                        // `ldmatrix` writes one 32-bit word per 8x8 tile, so
                        // the words land in temporaries and are split into the
                        // fragment's two 16-bit halves (`ptxas` coalesces the
                        // moves into the fragment's registers).
                        let pattern = (0..*frags).map(|i| format!("%r{}", t + u16::from(i))).collect::<Vec<_>>().join(", ");
                        _ = writeln!(
                            comp.body,
                            "{}ldmatrix.sync.aligned.m8n8.x{frags}{t_suffix}.b16 {{{pattern}}}, [%r{rd}];",
                            comp.indent
                        );
                        comp.release_reg(rd);
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
                    // The hardware `ldmatrix` at an address the IR already
                    // built: no window base, no `<<1`, no offset arithmetic.
                    PTXOp::LdMatrix { frags, addr, trans } => {
                        let a = comp.get_var_addr(*addr);
                        let reg = comp.new_var(op_id, DType::F16, vec![Dim::from(i64::from(*frags) * 2)], rcs[&op_id]);
                        let t = comp.new_reg(DType::U32, vec![Dim::from(i64::from(*frags))], u32::from(*frags));
                        let t_suffix = if *trans { ".trans" } else { "" };
                        let pattern = (0..*frags).map(|i| format!("%r{}", t + u16::from(i))).collect::<Vec<_>>().join(", ");
                        _ = writeln!(
                            comp.body,
                            "{}ldmatrix.sync.aligned.m8n8.x{frags}{t_suffix}.shared.b16 {{{pattern}}}, [%r{a}];",
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
                    // `cvta.shared.u64 %rd, %rs`.
                    PTXOp::CvtaShared { addr } => {
                        let a = comp.get_var_addr(*addr);
                        let reg = comp.new_var(op_id, DType::U64, Vec::new(), rcs[&op_id]);
                        _ = writeln!(comp.body, "{}cvta.shared.u64 %r{reg}, %r{a};", comp.indent);
                    }
                    // `mov.u32 %r, __ldN`: the storage's window offset.
                    PTXOp::SharedBase { buf } => {
                        let reg = comp.new_var(op_id, DType::U32, Vec::new(), rcs[&op_id]);
                        let b = buf.0;
                        _ = writeln!(comp.body, "{}mov.u32 %r{reg}, __ld{b};", comp.indent);
                    }
                    // `ld.shared` at a byte offset the IR computed.
                    PTXOp::LdShared { buf, addr } => {
                        let access_shape = self.shape(op_id);
                        let dtype = dtypes[buf].0;
                        let a = comp.get_var_addr(*addr);
                        let reg = comp.new_var(op_id, dtype, access_shape.clone(), rcs[&op_id]);
                        let addr = format!("%r{a}");
                        comp.emit_load(&access_shape, dtype, reg, "shared", &addr)?;
                    }
                    // `st.shared` at a byte offset the IR computed.
                    PTXOp::StShared { addr, src, .. } => {
                        let access_shape = self.shape(*addr);
                        let dtype = dtypes[src].0;
                        let a = comp.get_var_addr(*addr);
                        let regs = comp.components_of(self, *src, &dtypes)?;
                        let addr = format!("%r{a}");
                        comp.emit_store_space(&access_shape, dtype, &regs, "shared", &addr)?;
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
        _ = writeln!(comp.header, "{}.reg .u64 %rd_saddress;", comp.indent);
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

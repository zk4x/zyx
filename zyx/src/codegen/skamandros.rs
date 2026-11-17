// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! skamandros program text codegen.
//!
//! zyx's tiled kernel IR (`tt_tile` and the generic passes) is written out in
//! the text form `libskamandros.so` compiles: SSA values over DRAM parameters,
//! L1 buffers, accumulators, loops and the 2-D grid. The machine lowering —
//! lane registers, `Dst` slots, circular buffers, the movers, the encoders —
//! is skamandros's, not this file's.
//!
//! One unit rule: a zyx index counts elements, a skamandros index counts the
//! access unit named before it (`1,1` a datum, `1,32` a row, `32,1` a column,
//! `32,32` a tile). An index expression is emitted with its `Unroll` lanes at
//! zero and then divided by the unit's element count, which is exact for the
//! affine tile indices `tt_tile` produces.

use crate::{
    DType, Map,
    backend::{GwsDim, gws_from_kernel},
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, GPUOp, Kernel, Op, OpId, ParamKind, RangeKind, SourceBlock, UOp},
    shape::Dim,
};
use std::fmt::Write;

impl DType {
    /// The text-form dtype. skamandros has four: `f32`, `bf16`, `i32`, `bool`.
    fn skamandros(self) -> Result<&'static str, BackendError> {
        Ok(match self {
            Self::F32 => "f32",
            Self::BF16 => "bf16",
            Self::I32 | Self::I64 | Self::U32 | Self::U64 | Self::I16 | Self::U16 | Self::I8 | Self::U8 => "i32",
            Self::Bool => "bool",
            other => return Err(tt_err(format!("skamandros: dtype {other:?} has no tile form"))),
        })
    }
}

impl Constant {
    /// The literal token a `const` line carries.
    fn skamandros(self, dtype: DType) -> Result<String, BackendError> {
        Ok(match dtype {
            DType::F32 => {
                let value = self.as_float().ok_or_else(|| tt_err("skamandros: non-float constant in an f32 slot".into()))?;
                format!("{value}")
            }
            DType::BF16 => {
                let value = self.as_float().ok_or_else(|| tt_err("skamandros: non-float constant in a bf16 slot".into()))?;
                format!("{value}")
            }
            DType::I32
            | DType::I64
            | DType::U32
            | DType::U64
            | DType::I16
            | DType::U16
            | DType::I8
            | DType::U8 => {
                let value = self.as_integer().ok_or_else(|| tt_err("skamandros: non-integer constant in an i32 slot".into()))?;
                format!("{value}")
            }
            DType::Bool => return Err(tt_err("skamandros: bool constants are not text form".into())),
            other => return Err(tt_err(format!("skamandros: constant dtype {other:?} has no tile form"))),
        })
    }
}

fn tt_err(context: String) -> BackendError {
    BackendError { status: ErrorStatus::KernelCompilation, context: context.into_boxed_str() }
}

fn bop(b: BOp) -> Result<&'static str, BackendError> {
    Ok(match b {
        BOp::Add => "add",
        BOp::Sub => "sub",
        BOp::Mul => "mul",
        BOp::Div => "div",
        BOp::Pow => "pow",
        BOp::Mod => "mod",
        BOp::Cmplt => "cmplt",
        BOp::Cmpgt => "cmpgt",
        BOp::Cmpge => "cmpge",
        BOp::Max => "max",
        BOp::Or => "or",
        BOp::And => "and",
        BOp::BitXor => "bitxor",
        BOp::BitOr => "bitor",
        BOp::BitAnd => "bitand",
        BOp::BitShiftLeft => "bitshiftleft",
        BOp::BitShiftRight => "bitshiftright",
        BOp::NotEq => "noteq",
        BOp::Eq => "eq",
    })
}

fn uop(u: UOp) -> &'static str {
    match u {
        UOp::Neg => "neg",
        UOp::Not => "not",
        UOp::BitNot => "bitnot",
        UOp::Exp2 => "exp2",
        UOp::Log2 => "log2",
        UOp::Reciprocal => "reciprocal",
        UOp::Sqrt => "sqrt",
        UOp::Rsqrt => "rsqrt",
        UOp::Sin => "sin",
        UOp::Cos => "cos",
        UOp::Floor => "floor",
        UOp::Trunc => "trunc",
        UOp::Abs => "abs",
    }
}

struct Compiler<'a> {
    k: &'a Kernel,
    text: String,
    /// Appended after the body: skamandros's `grid` line, with the value
    /// definitions its terms need in front of it.
    tail: String,
    /// Parameter names in head order (the launch argument order).
    params: Vec<(OpId, String)>,
    /// One ones page per dtype, for `sum`'s scaler.
    ones: Map<DType, String>,
    /// One zero/`low` literal per dtype, for accumulator initialization.
    zeroes: Map<DType, String>,
    /// Emitted unit index per element index, so a load and its store reuse
    /// the one `div`.
    indices: Map<OpId, String>,
}

impl Compiler<'_> {
    fn name(&self, id: OpId) -> String {
        format!("%v{id}")
    }

    /// The access unit for an index expression, as skamandros's `a,b` shape,
    /// plus the element stride one unit step moves. A tile/row/column is 32
    /// lanes wide on its varying axis; a scalar is one datum.
    fn unit(&self, idx: OpId) -> Result<([Dim; 2], u64), BackendError> {
        let shape = self.k.shape(idx);
        let unit: [Dim; 2] = match shape.len() {
            0 => [1, 1],
            1 => {
                // A vector is a row or a column depending on which axis its
                // `Unroll` was split from: axis 0 runs down rows, axis 1 runs
                // across columns.
                match self.vector_axis(idx) {
                    Some(0) => [32, 1],
                    _ => [1, 32],
                }
            }
            _ => [32, 32],
        };
        let stride = if unit == [1, 1] { 1 } else { 32 };
        Ok((unit, stride))
    }

    /// The axis of the first `Unroll` feeding `root`'s index expression.
    fn vector_axis(&self, root: OpId) -> Option<u32> {
        let mut stack = vec![root];
        let mut seen = crate::Set::default();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            match &self.k.ops[id].op {
                Op::Unroll { axis, .. } => return Some(*axis),
                op => stack.extend(op.parameters()),
            }
        }
        None
    }

    /// The base of an element index in access-unit units.
    fn unit_index(&mut self, idx: OpId, unit: u64) -> String {
        if let Some(name) = self.indices.get(&idx) {
            return name.clone();
        }
        let base = self.name(idx);
        if unit <= 1 {
            return base;
        }
        let out = format!("%u{}", usize::from(idx));
        let _ = writeln!(self.text, "{out} = div {base} {unit}");
        self.indices.insert(idx, out.clone());
        out
    }

    fn zero(&mut self, dtype: DType) -> Result<String, BackendError> {
        if let Some(name) = self.zeroes.get(&dtype) {
            return Ok(name.clone());
        }
        let name = format!("%zero{}", dtype.skamandros()?);
        let _ = writeln!(self.text, "{name} = const {} 0", dtype.skamandros()?);
        self.zeroes.insert(dtype, name.clone());
        Ok(name)
    }

    /// A loaded page of ones, the scaler `sum` takes.
    fn ones(&mut self, dtype: DType) -> Result<String, BackendError> {
        if let Some(name) = self.ones.get(&dtype) {
            return Ok(name.clone());
        }
        let dt = dtype.skamandros()?;
        let one = format!("%one{dt}");
        let buf = format!("%onebuf{dt}");
        let page = format!("%ones{dt}");
        let _ = writeln!(self.text, "{one} = const {dt} 1");
        let _ = writeln!(self.text, "{buf} = buffer {dt} 32,32");
        let _ = writeln!(self.text, "store 32,32 {buf} = {one}");
        let _ = writeln!(self.text, "load 32,32 {page} = {buf} [0]");
        self.ones.insert(dtype, page.clone());
        Ok(page)
    }
}

impl Kernel {
    /// Render a tiled kernel into skamandros program text.
    ///
    /// The returned kernel carries the text as its `Op::Source` payload plus
    /// the launch parameter list, exactly like the CUDA renderers carry PTX or
    /// CUDA C.
    pub(crate) fn render_skamandros(&self) -> Result<Kernel, BackendError> {
        let mut c = Compiler {
            k: self,
            text: String::new(),
            tail: String::new(),
            params: Vec::new(),
            ones: Map::default(),
            zeroes: Map::default(),
            indices: Map::default(),
        };
        let _ = writeln!(c.text, "version 9");

        // Prologue: every definition a later line can name, in head order.
        let mut params: Vec<ParamKind> = Vec::new();
        let mut op_id = self.head;
        let mut steps = 0usize;
        while !op_id.is_null() {
            steps += 1;
            if steps > crate::kernel::MAX_OPS as usize {
                return Err(tt_err("skamandros: kernel exceeds the op render limit".into()));
            }
            match &self.ops[op_id].op {
                Op::Const(value) => {
                    let dtype = self.op_dtype(op_id).unwrap_or(DType::I32);
                    let dt = dtype.skamandros()?;
                    let lit = value.skamandros(dtype)?;
                    let _ = writeln!(c.text, "{} = const {dt} {lit}", c.name(op_id));
                }
                Op::Param { kind, .. } => {
                    let name = c.name(op_id);
                    c.params.push((op_id, name.clone()));
                    params.push(*kind);
                    let dtype = self.op_dtype(op_id).unwrap_or(DType::F32);
                    let dt = dtype.skamandros()?;
                    if matches!(kind, ParamKind::Variable) {
                        let _ = writeln!(c.text, "{name} = param {dt} 0");
                    } else {
                        let shape = self.shape(op_id);
                        if shape.is_empty() {
                            return Err(tt_err(format!("skamandros: parameter {op_id:?} has no shape")));
                        }
                        let mut terms = String::new();
                        for (at, dim) in shape.iter().enumerate() {
                            if at != 0 {
                                terms.push(',');
                            }
                            if *dim <= 0 {
                                return Err(tt_err(format!("skamandros: parameter {op_id:?} has non-positive dim {dim}")));
                            }
                            let _ = write!(terms, "{dim}");
                        }
                        let _ = writeln!(c.text, "{name} = param {dt} {terms}");
                    }
                }
                Op::Storage { dtype, scope, len, .. } => {
                    let dt = dtype.skamandros()?;
                    // A pre-mover storage is a page ring: a tile for a tile,
                    // a row/column for a smaller one.
                    let _ = scope;
                    let page = if *len >= 1024 {
                        "32,32"
                    } else if *len >= 32 {
                        "1,32"
                    } else {
                        "1,1"
                    };
                    let _ = writeln!(c.text, "{} = buffer {dt} {page}", c.name(op_id));
                }
                Op::Range { axis, kind: RangeKind::Group(_) } => {
                    let coord = match axis {
                        0 => "grid.x",
                        1 => "grid.y",
                        other => {
                            return Err(tt_err(format!("skamandros: grid axis {other} is outside the 2-D core grid")));
                        }
                    };
                    let _ = writeln!(c.text, "{} = {coord}", c.name(op_id));
                }
                Op::Range { axis, kind: RangeKind::Local(_) } => {
                    // A local axis is a thread id on a GPU; Tensix has none.
                    // The tiling that produces one is a search candidate, not
                    // a program shape, so it reads as lane 0.
                    let _ = axis;
                    let _ = writeln!(c.text, "{} = const i32 0", c.name(op_id));
                }
                Op::Unroll { .. } => {
                    // A lane is the access unit's own dimension; the emitted
                    // index is its base (lane 0) and the unit names the rest.
                    let dt = "i32";
                    let _ = writeln!(c.text, "{} = const {dt} 0", c.name(op_id));
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // Body, in program order.
        let mut op_id = self.head;
        let mut steps = 0usize;
        while !op_id.is_null() {
            steps += 1;
            if steps > crate::kernel::MAX_OPS as usize {
                return Err(tt_err("skamandros: kernel exceeds the op render limit".into()));
            }
            let name = c.name(op_id);
            match &self.ops[op_id].op {
                Op::Const(_) | Op::Param { .. } | Op::Storage { .. } | Op::Range { .. } | Op::Unroll { .. } => {}
                // A shape is metadata; every use reads it through the shape
                // descriptors, never as a value.
                Op::Stack { .. } => {}
                Op::Index { vec, idx } => {
                    // A lane/member selection is the member's own value.
                    let member = match &self.ops[*vec].op {
                        Op::Stack { ops } => ops.get(*idx).copied(),
                        _ => None,
                    };
                    let Some(member) = member else {
                        return Err(tt_err(format!("skamandros: index {op_id:?} names no stack member")));
                    };
                    let dtype = self.op_dtype(op_id).unwrap_or(DType::I32);
                    let zero = c.zero(dtype)?;
                    let _ = writeln!(c.text, "{name} = add {} {zero}", c.name(member));
                }
                Op::Loop { len } => {
                    let _ = writeln!(c.text, "{name} = loop {}", c.name(*len));
                }
                Op::EndLoop => {
                    let _ = writeln!(c.text, "endloop");
                }
                Op::Barrier => {}
                Op::Cast { x, dtype } => {
                    let _ = writeln!(c.text, "{name} = cast {} to {}", c.name(*x), dtype.skamandros()?);
                }
                Op::Bitcast { x, dtype } => {
                    let _ = writeln!(c.text, "{name} = bitcast {} to {}", c.name(*x), dtype.skamandros()?);
                }
                Op::Unary { x, uop: u } => {
                    let _ = writeln!(c.text, "{name} = {} {}", uop(*u), c.name(*x));
                }
                Op::Binary { x, y, bop: b } => {
                    let _ = writeln!(c.text, "{name} = {} {} {}", bop(*b)?, c.name(*x), c.name(*y));
                }
                Op::Select { cond, if_true, if_false } => {
                    let _ = writeln!(c.text, "{name} = sel {} {} {}", c.name(*cond), c.name(*if_true), c.name(*if_false));
                }
                Op::Load { src, src_idx } => {
                    let (unit, elems) = c.unit(*src_idx)?;
                    let idx = c.unit_index(*src_idx, elems);
                    let _ = writeln!(c.text, "load {},{} {name} = {} [{idx}]", unit[0], unit[1], c.name(*src));
                }
                Op::Store { dst, dst_idx, src } => {
                    let (unit, elems) = c.unit(*dst_idx)?;
                    let idx = c.unit_index(*dst_idx, elems);
                    let _ = writeln!(c.text, "store {},{} {} [{idx}] = {}", unit[0], unit[1], c.name(*dst), c.name(*src));
                }
                Op::Copy { src, src_idx, dst, dst_idx } => {
                    let (su, se) = c.unit(*src_idx)?;
                    let (du, de) = c.unit(*dst_idx)?;
                    if su != du {
                        return Err(tt_err(format!("skamandros: copy units {su:?} and {du:?} disagree")));
                    }
                    let sidx = c.unit_index(*src_idx, se);
                    let didx = c.unit_index(*dst_idx, de);
                    let _ = writeln!(c.text, "copy {},{} {} [{didx}] = {} [{sidx}]", su[0], su[1], c.name(*dst), c.name(*src));
                }
                Op::Reduce { x, rop, reduce_axis } => {
                    c.reduce(op_id, *x, *rop, *reduce_axis)?;
                }
                Op::Wmma { a, b, c: acc, .. } => {
                    // `d = a @ b + c`: skamandros accumulates into an `acc`,
                    // so the chain's existing accumulator is named directly.
                    let _ = writeln!(c.text, "matmul {} {} {}", c.name(*acc), c.name(*a), c.name(*b));
                    let _ = writeln!(c.text, "{} = add {} {}", name, c.name(*acc), c.name(*acc));
                }
                Op::Source(_) => {}
                other => {
                    return Err(tt_err(format!("skamandros: op {other:?} has no text form")));
                }
            }
            op_id = self.next_op(op_id);
        }

        // Grid, with any dim expressions its terms need.
        let gws = gws_from_kernel(self, &self.dev_info().max_global_work_dims)?;
        let mut terms = Vec::with_capacity(2);
        for dim in gws.iter().take(2) {
            terms.push(c.gws(dim)?);
        }
        while terms.len() < 2 {
            terms.push("1".to_string());
        }
        let _ = writeln!(c.tail, "grid {},{}", terms[0], terms[1]);
        c.text.push_str(&c.tail);

        let mut rendered = Kernel {
            ops: crate::slab::Slab::new(),
            head: OpId::NULL,
            tail: OpId::NULL,
            dev: self.dev,
            dev_info: self.dev_info.clone(),
        };
        rendered.push_back(Op::GPU(Box::new(GPUOp::Params(params.into_boxed_slice()))));
        rendered.push_back(Op::Source(SourceBlock(c.text.into_boxed_str())));
        Ok(rendered)
    }
}

impl Compiler<'_> {
    /// Emit a grid term, defining any value it needs in the tail.
    fn gws(&mut self, dim: &GwsDim) -> Result<String, BackendError> {
        Ok(match dim {
            GwsDim::Const(c) => c.to_string(),
            GwsDim::Param(ordinal) => self
                .params
                .get(*ordinal)
                .map(|(_, name)| name.clone())
                .ok_or_else(|| tt_err(format!("skamandros: grid names parameter {ordinal}, which does not exist")))?,
            GwsDim::Unary { x, uop: u } => {
                let x = self.gws(x)?;
                let name = format!("%gu{}", self.tail.len());
                let _ = writeln!(self.tail, "{name} = {} {x}", uop(*u));
                name
            }
            GwsDim::Binary { x, y, bop: b } => {
                let x = self.gws(x)?;
                let y = self.gws(y)?;
                let name = format!("%gb{}", self.tail.len());
                let _ = writeln!(self.tail, "{name} = {} {x} {y}", bop(*b)?);
                name
            }
            GwsDim::Cast { x, dtype } => {
                let x = self.gws(x)?;
                let name = format!("%gc{}", self.tail.len());
                let _ = writeln!(self.tail, "{name} = cast {x} to {}", dtype.skamandros()?);
                name
            }
        })
    }

    /// A reduce in skamandros is an accumulator plus a fold. The accumulator
    /// shape names the fold: `1,32` folds rows into row 0, `32,1` folds
    /// columns into column 0, `1,1` folds the whole tile.
    fn reduce(&mut self, id: OpId, x: OpId, rop: BOp, axis: OpId) -> Result<(), BackendError> {
        let dtype = self.k.op_dtype(x).ok_or_else(|| tt_err(format!("skamandros: reduce operand {x:?} has no dtype")))?;
        let dt = dtype.skamandros()?;
        let xshape = self.k.shape(x);
        let rshape = self.k.shape(id);
        let acc = format!("%acc{id}");
        // The fold's home. A reduce over a lane leaves a vector: which lane
        // went away names the row/column of the result.
        let acc_shape = match rshape.len() {
            0 => "1,1",
            1 => match self.vector_axis(axis) {
                Some(0) => "1,32",
                Some(_) => "32,1",
                None => "32,1",
            },
            _ => return Err(tt_err(format!("skamandros: reduce {id:?} leaves shape {rshape:?}, which has no accumulator"))),
        };
        let _ = xshape;
        let _ = writeln!(self.text, "{acc} = acc {dt} {acc_shape}");
        match rop {
            BOp::Add => {
                let zero = self.zero(dtype)?;
                let _ = writeln!(self.text, "store 32,32 {acc} = {zero}");
                let ones = self.ones(dtype)?;
                // `x` is a register value; `sum` takes the scaler page.
                let _ = xshape;
                let _ = writeln!(self.text, "sum {acc} {} {ones}", self.name(x));
            }
            BOp::Max => {
                let low = format!("%low{dt}");
                let _ = writeln!(self.text, "{low} = const {dt} {}", low_literal(dt));
                let _ = writeln!(self.text, "store 32,32 {acc} = {low}");
                let _ = writeln!(self.text, "max {acc} {}", self.name(x));
            }
            other => return Err(tt_err(format!("skamandros: reduce op {other:?} has no pool form"))),
        }
        Ok(())
    }
}

fn low_literal(dt: &str) -> &'static str {
    match dt {
        "bf16" => "-3.38953139e38",
        _ => "-3.40282347e38",
    }
}

impl Kernel {
    fn op_dtype(&self, id: OpId) -> Option<DType> {
        use crate::kernel::Op as KOp;
        match &self.ops[id].op {
            KOp::Const(c) => Some(match c {
                Constant::BF16(_) => DType::BF16,
                Constant::F16(_) => DType::F16,
                Constant::F32(_) => DType::F32,
                Constant::F64(_) => DType::F64,
                Constant::F8E4M3(_) => DType::F8E4M3,
                Constant::F8E5M2(_) => DType::F8E5M2,
                Constant::U8(_) => DType::U8,
                Constant::U16(_) => DType::U16,
                Constant::U32(_) => DType::U32,
                Constant::U64(_) => DType::U64,
                Constant::I8(_) => DType::I8,
                Constant::I16(_) => DType::I16,
                Constant::I32(_) => DType::I32,
                Constant::I64(_) => DType::I64,
                Constant::Bool(_) => DType::Bool,
            }),
            KOp::Param { dtype, .. } | KOp::Storage { dtype, .. } => Some(*dtype),
            KOp::Load { src, .. } => self.op_dtype(*src),
            KOp::Cast { dtype, .. } | KOp::Bitcast { dtype, .. } => Some(*dtype),
            KOp::Unary { x, .. } => self.op_dtype(*x),
            KOp::Binary { x, .. } => self.op_dtype(*x),
            KOp::Select { if_true, .. } => self.op_dtype(*if_true),
            KOp::Reduce { x, .. } => self.op_dtype(*x),
            KOp::Wmma { a, .. } => self.op_dtype(*a),
            _ => None,
        }
    }
}

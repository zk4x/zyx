// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Host interpreter for the ordered kernel IR.
//!
//! [`Kernel::interpret`] runs a kernel on the host: one op at a time, with
//! every thread of a block evaluated side by side. It is a *debugging view* of
//! the IR, not a backend — no optimizations, no lowering, no machine
//! decisions. It executes exactly the ops the kernel holds, lanes included:
//! [`Op::Unroll`] is a vector of indices, so a lane-shaped value is a
//! `Vec<Constant>` with that shape.
//!
//! The threads of a block execute in lockstep op by op (SIMT), which is what
//! makes the cross-lane instructions exact: `ldmatrix` takes the eight row
//! addresses of each matrix from the lanes of its group, `shfl` reads another
//! lane's register, the mma gathers the warp's fragments, multiplies and
//! scatters the accumulator back. A block-wide [`Op::Barrier`] needs no work
//! in this execution model.
//!
//! A `Shared` storage is one block-wide buffer, a `Register` storage is one
//! buffer per thread, and a global `Param` outlives the block loop.

use crate::{
    DType, Map, ZyxError,
    dtype::Constant,
    f16,
    kernel::{BOp, Kernel, MemScope, Op, OpId, PTXOp, ParamKind, RangeKind, UOp},
    shape::Dim,
};

/// One op's value for one thread.
#[derive(Clone)]
struct Value {
    dtype: DType,
    shape: Vec<Dim>,
    /// Elements in lane order: the product of `shape` of them.
    data: Vec<Constant>,
}

/// A storage during the interpreted launch.
enum Mem {
    /// One buffer for the whole block (shared memory).
    Shared(Vec<Constant>),
    /// One buffer per thread (registers).
    Register(Vec<Vec<Constant>>),
}

impl Kernel {
    /// Run the kernel on the host and return the global buffers, in `Param`
    /// order, after the launch.
    ///
    /// `params` are the global buffers in IR order, little-endian in the
    /// param's dtype. `print` prints every executed op with its first thread's
    /// value; `stop` ends the launch right after that op, so one window of the
    /// mainloop can be inspected instead of all of them.
    pub fn interpret(
        &self,
        params: &[&[u8]],
        print: bool,
        stop: Option<OpId>,
    ) -> Result<Vec<Vec<u8>>, ZyxError> {
        // A lane bundle is many instructions (`a` of `wmma` is `[tile][slot]`),
        // so the interpreter runs the device-lowered form: devectorize a copy
        // first. `stop` therefore names an op of the *lowered* IR.
        let mut k = self.clone();
        k.devectorize()?;
        k.interpret_flat(params, print, stop)
    }

    /// The evaluator: `self` is already in scalar form.
    fn interpret_flat(
        &self,
        params: &[&[u8]],
        print: bool,
        stop: Option<OpId>,
    ) -> Result<Vec<Vec<u8>>, ZyxError> {
        let mut op_ids: Vec<OpId> = Vec::new();
        let mut groups: Vec<usize> = Vec::new();
        let mut locals: Vec<usize> = Vec::new();
        let mut globals: Map<OpId, Vec<Constant>> = Map::default();
        let mut mem: Map<OpId, Mem> = Map::default();
        let mut buffers = 0usize;

        let mut path = self.head;
        while !path.is_null() {
            op_ids.push(path);
            match self.at(path) {
                Op::Range { axis, kind } => {
                    let axis = *axis as usize;
                    let extent = match kind {
                        RangeKind::Group(len) => {
                            let len = self
                                .resolve_const(*len)
                                .and_then(Constant::as_dim)
                                .ok_or_else(|| ZyxError::kernel_error("interpret: group extent is not const".into()))?;
                            usize::try_from(len).expect("group extent fits usize")
                        }
                        RangeKind::Local(len) => usize::try_from(*len).expect("local extent fits usize"),
                    };
                    let axes = if let RangeKind::Group(_) = kind { &mut groups } else { &mut locals };
                    if axes.len() <= axis {
                        axes.resize(axis + 1, 1);
                    }
                    axes[axis] = extent;
                }
                Op::Param { dtype, kind, .. } if !matches!(kind, ParamKind::Variable) => {
                    let bytes = params.get(buffers).ok_or_else(|| {
                        ZyxError::kernel_error(format!("interpret: no buffer for {path:?}").into())
                    })?;
                    buffers += 1;
                    let len = bytes.len() / (dtype.bit_size() as usize / 8);
                    globals.insert(path, decode(*dtype, bytes, len));
                }
                Op::Storage { scope, len, .. } => {
                    let len = usize::try_from(*len).expect("storage length fits usize");
                    match scope {
                        MemScope::Register => {
                            mem.insert(path, Mem::Register(Vec::new()));
                        }
                        MemScope::Local => {
                            mem.insert(path, Mem::Shared(vec![Constant::new(0u32); len]));
                        }
                        MemScope::Global => todo!("interpret: global-scope storage"),
                        MemScope::Circular => todo!("interpret: circular storage"),
                    }
                }
                _ => {}
            }
            path = self.next_op(path);
        }

        let nblocks = groups.iter().product::<usize>().max(1);
        let nthreads = locals.iter().product::<usize>().max(1);
        let warp = self.device_info().warp_size.max(1) as usize;
        for (id, m) in mem.iter_mut() {
            if let Mem::Register(regs) = m {
                let len = match self.at(*id) {
                    Op::Storage { len, .. } => usize::try_from(*len).expect("storage length fits usize"),
                    _ => unreachable!("register memory without a storage op"),
                };
                *regs = vec![vec![Constant::new(0u16); len]; nthreads];
            }
        }

        for block in 0..nblocks {
            let group_at = decompose(block, &groups);
            let mut values: Map<OpId, Vec<Value>> = Map::default();
            let mut loops: Vec<(OpId, usize, usize)> = Vec::new();
            let mut pc = 0usize;
            let mut steps = 0usize;
            while pc < op_ids.len() {
                steps += 1;
                if steps > 100_000_000 {
                    return Err(ZyxError::kernel_error("interpret: launch exceeded 1e8 ops".into()));
                }
                let id = op_ids[pc];
                match self.at(id).clone() {
                    Op::EndLoop => {
                        let (loop_id, len, trip) = loops.pop().expect("interpret: EndLoop without Loop");
                        if trip + 1 < len {
                            loops.push((loop_id, len, trip + 1));
                            values.insert(loop_id, vec![Value::scalar(DType::I64, Constant::new(trip as i64 + 1)); nthreads]);
                            pc = op_ids.iter().position(|&x| x == loop_id).expect("loop op is in the op list") + 1;
                        } else {
                            pc += 1;
                        }
                        continue;
                    }
                    Op::Loop { len } => {
                        let len = self
                            .resolve_const(len)
                            .and_then(Constant::as_dim)
                            .ok_or_else(|| ZyxError::kernel_error("interpret: loop length is not const".into()))?;
                        let len = usize::try_from(len).expect("loop length fits usize");
                        if len == 0 {
                            let mut depth = 1;
                            while depth > 0 {
                                pc += 1;
                                match self.at(op_ids[pc]) {
                                    Op::Loop { .. } => depth += 1,
                                    Op::EndLoop => depth -= 1,
                                    _ => {}
                                }
                            }
                            pc += 1;
                            continue;
                        }
                        loops.push((id, len, 0));
                        values.insert(id, vec![Value::scalar(DType::I64, Constant::new(0i64)); nthreads]);
                        pc += 1;
                        continue;
                    }
                    op => {
                        self.interpret_one(id, &op, &mut values, &mut mem, &mut globals, &group_at, nthreads, warp);
                        if print {
                            let shown = values.get(&id).and_then(|v| v.first()).map(|v| (v.dtype, v.shape.clone(), v.data.clone()));
                            println!("interpret {id:?}: {op:?} => {shown:?}");
                        }
                    }
                }
                if stop == Some(id) {
                    return Ok(self.encode_globals(&globals, &op_ids));
                }
                pc += 1;
            }
        }
        Ok(self.encode_globals(&globals, &op_ids))
    }

    /// Evaluate one op for every thread of the block.
    #[allow(clippy::too_many_arguments)]
    fn interpret_one(
        &self,
        id: OpId,
        op: &Op,
        values: &mut Map<OpId, Vec<Value>>,
        mem: &mut Map<OpId, Mem>,
        globals: &mut Map<OpId, Vec<Constant>>,
        group_at: &[usize],
        nthreads: usize,
        warp: usize,
    ) {
        let val = |x: OpId, t: usize| -> Value { values[&x][t].clone() };
        match op {
            Op::Const(c) => {
                values.insert(id, vec![Value::scalar(c.dtype(), *c); nthreads]);
            }
            Op::Range { axis, kind } => {
                let axis = *axis as usize;
                let local_extents = self.local_extents();
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let x = match kind {
                            RangeKind::Group(_) => group_at.get(axis).copied().unwrap_or(0),
                            RangeKind::Local(_) => axis_at(t, &local_extents, axis),
                        };
                        Value::scalar(DType::I64, Constant::new(x as i64))
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Unroll { axis, len } => {
                let mut shape = vec![0; *axis as usize + 1];
                shape[*axis as usize] = Dim::from(*len);
                let data = (0..*len).map(|i| Constant::new(i as i64)).collect();
                values.insert(id, vec![Value { dtype: DType::I64, shape, data }; nthreads]);
            }
            Op::Binary { x, y, bop } => {
                let shape = self.shape(id);
                let n = product(&shape);
                let dtype = if bop.returns_bool() { DType::Bool } else { values[&x][0].dtype };
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let (a, b) = (val(*x, t), val(*y, t));
                        let data = (0..n).map(|j| binary(*bop, dtype, a.data[j % a.data.len()], b.data[j % b.data.len()])).collect();
                        Value { dtype, shape: shape.clone(), data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Unary { x, uop } => {
                let shape = self.shape(id);
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let a = val(*x, t);
                        let data = a.data.iter().map(|&c| unary(*uop, a.dtype, c)).collect();
                        Value { dtype: a.dtype, shape: shape.clone(), data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Cast { x, dtype } => {
                let shape = self.shape(id);
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let a = val(*x, t);
                        let data = a.data.iter().map(|&c| cast_to(c, *dtype)).collect();
                        Value { dtype: *dtype, shape: shape.clone(), data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Bitcast { x, dtype } => {
                let shape = self.shape(id);
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let a = val(*x, t);
                        let data = a.data.iter().map(|&c| decode(*dtype, &c.to_le_bytes(), 1)[0]).collect();
                        Value { dtype: *dtype, shape: shape.clone(), data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Select { cond, if_true, if_false } => {
                let shape = self.shape(id);
                let n = product(&shape);
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let (c, a, b) = (val(*cond, t), val(*if_true, t), val(*if_false, t));
                        let data = (0..n)
                            .map(|j| {
                                if truth(c.data[j % c.data.len()]) {
                                    a.data[j % a.data.len()]
                                } else {
                                    b.data[j % b.data.len()]
                                }
                            })
                            .collect();
                        Value { dtype: a.dtype, shape: shape.clone(), data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            // A component extract: `vec[idx]` names a run of `stride` elements
            // of the vector (one element for a lane vector).
            Op::Index { vec, idx } => {
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let a = val(*vec, t);
                        let mut shape = a.shape.clone();
                        if !shape.is_empty() {
                            shape.remove(0);
                        }
                        let stride = product(&shape);
                        let base = *idx * stride;
                        Value { dtype: a.dtype, shape, data: a.data[base..base + stride].to_vec() }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Stack { ops } => {
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let mut data = Vec::new();
                        for &o in ops.iter() {
                            data.extend_from_slice(&val(o, t).data);
                        }
                        let dtype = val(ops[0], t).dtype;
                        Value { dtype, shape: vec![Dim::from(data.len() as i64)], data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Load { src, src_idx } => {
                let shape = self.shape(id);
                let dtype = self.mem_dtype(*src);
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let idx = val(*src_idx, t);
                        let data = idx
                            .data
                            .iter()
                            .map(|&i| read(mem, globals, *src, t, usize::try_from(to_i64(i)).expect("index fits usize")))
                            .collect();
                        Value { dtype, shape: shape.clone(), data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            Op::Store { dst, dst_idx, src } => {
                for t in 0..nthreads {
                    let idx = val(*dst_idx, t);
                    let x = val(*src, t);
                    for j in 0..idx.data.len() {
                        let at = usize::try_from(to_i64(idx.data[j])).expect("index fits usize");
                        write(mem, globals, *dst, t, at, x.data[j % x.data.len()]);
                    }
                }
            }
            Op::PTX(p) => match p.as_ref() {
                PTXOp::LdMatrix { frags, src, addr, trans } => {
                    assert!(!*trans, "interpret: transposed ldmatrix");
                    let frags = usize::from(*frags);
                    let dtype = self.mem_dtype(*src);
                    let per_thread = (0..nthreads)
                        .map(|t| {
                            let base = t / warp * warp;
                            let lane = t % warp;
                            let mut data = Vec::with_capacity(frags * 2);
                            for m in 0..frags {
                                let row = to_i64(val(*addr, base + m * 8 + lane / 4).data[0]);
                                for j in 0..2 {
                                    let at = usize::try_from(row + (2 * (lane % 4) + j) as i64).expect("shared index");
                                    data.push(read(mem, globals, *src, t, at));
                                }
                            }
                            Value { dtype, shape: vec![Dim::from(data.len() as i64)], data }
                        })
                        .collect();
                    values.insert(id, per_thread);
                }
                PTXOp::Shfl { value, lane } => {
                    let per_thread = (0..nthreads)
                        .map(|t| {
                            let base = t / warp * warp;
                            let from = usize::try_from(to_i64(val(*lane, t).data[0])).expect("lane fits usize") % warp;
                            val(*value, base + from)
                        })
                        .collect();
                    values.insert(id, per_thread);
                }
                PTXOp::WarpSync => {
                    values.insert(id, vec![Value::scalar(DType::U32, Constant::new(0u32)); nthreads]);
                }
                PTXOp::Bytes(_) => todo!("interpret: assembled payload"),
            },
            Op::Wmma { dims, layout, dtype, c, a, b } => {
                assert_eq!(*dims, crate::kernel::MMADims::m16n8k8, "interpret: only m16n8k8");
                assert_eq!(*layout, crate::kernel::MMALayout::row_col, "interpret: only row_col");
                assert_eq!(*dtype, crate::kernel::MMADType::f16_f16_f16_f32, "interpret: only f16 x f16 -> f32");
                let per_thread = (0..nthreads)
                    .map(|t| {
                        let base = t / warp * warp;
                        let lane = t % warp;
                        // Gather the warp's fragments: A is 16x8, B is 8x8, C is 16x8.
                        let mut af = [[0f32; 8]; 16];
                        let mut bf = [[0f32; 8]; 8];
                        for l in 0..warp {
                            let (av, bv) = (val(*a, base + l), val(*b, base + l));
                            let (row, k) = (l / 4, 2 * (l % 4));
                            for j in 0..2 {
                                af[row][k + j] = to_f64(av.data[j]) as f32;
                                af[row + 8][k + j] = to_f64(av.data[2 + j]) as f32;
                                bf[k + j][l / 4] = to_f64(bv.data[j]) as f32;
                            }
                        }
                        let cv = val(*c, t);
                        let (row, col) = (lane / 4, 2 * (lane % 4));
                        let mut data = Vec::with_capacity(4);
                        for (i, (r, c)) in [(row, col), (row, col + 1), (row + 8, col), (row + 8, col + 1)].into_iter().enumerate() {
                            let mut sum = to_f64(cv.data[i]) as f32;
                            for k in 0..8 {
                                sum += af[r][k] * bf[k][c];
                            }
                            data.push(Constant::new(sum));
                        }
                        Value { dtype: DType::F32, shape: vec![Dim::from(4)], data }
                    })
                    .collect();
                values.insert(id, per_thread);
            }
            // Declarations: a buffer or an allocation holds no value.
            Op::Param { .. } | Op::Storage { .. } => {}
            Op::Barrier => {
                values.insert(id, vec![Value::scalar(DType::U32, Constant::new(0u32)); nthreads]);
            }
            ref other => todo!("interpret: op {other:?}"),
        }
    }

    /// The dtype of the elements a load from or a store to `buffer` moves.
    fn mem_dtype(&self, buffer: OpId) -> DType {
        match self.at(buffer) {
            Op::Param { dtype, .. } | Op::Storage { dtype, .. } => *dtype,
            ref other => todo!("interpret: buffer op {other:?}"),
        }
    }

    fn local_extents(&self) -> Vec<usize> {
        let mut extents: Vec<usize> = Vec::new();
        for (_, op) in self.iter_unordered() {
            if let Op::Range { axis, kind: RangeKind::Local(len) } = op {
                let axis = *axis as usize;
                if extents.len() <= axis {
                    extents.resize(axis + 1, 1);
                }
                extents[axis] = *len as usize;
            }
        }
        extents
    }

    /// The launch's global buffers, in `Param` order, little-endian.
    fn encode_globals(&self, globals: &Map<OpId, Vec<Constant>>, op_ids: &[OpId]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for id in op_ids {
            if let Op::Param { kind, .. } = self.at(*id)
                && !matches!(kind, ParamKind::Variable)
            {
                let mut bytes = Vec::new();
                for &v in &globals[id] {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                out.push(bytes);
            }
        }
        out
    }
}

impl Value {
    fn scalar(dtype: DType, x: Constant) -> Self {
        Self { dtype, shape: Vec::new(), data: vec![x] }
    }
}

fn product(dims: &[Dim]) -> usize {
    dims.iter().map(|&d| usize::try_from(d.max(1)).unwrap_or(1)).product::<usize>().max(1)
}

/// Split a flat thread id into per-axis local indices (axis 0 innermost).
fn axis_at(t: usize, extents: &[usize], axis: usize) -> usize {
    let mut rest = t;
    let mut at = 0;
    for a in 0..=axis {
        let extent = extents.get(a).copied().unwrap_or(1).max(1);
        at = rest % extent;
        rest /= extent;
    }
    at
}

/// Split a flat block id into per-axis group indices (axis 0 innermost).
fn decompose(flat: usize, extents: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(extents.len());
    let mut rest = flat;
    for &e in extents {
        let e = e.max(1);
        out.push(rest % e);
        rest /= e;
    }
    out
}

fn decode(dtype: DType, bytes: &[u8], len: usize) -> Vec<Constant> {
    let size = dtype.bit_size() as usize / 8;
    (0..len)
        .map(|i| {
            let b = &bytes[i * size..(i + 1) * size];
            match dtype {
                DType::F16 => Constant::F16([b[0], b[1]]),
                DType::F32 => Constant::F32([b[0], b[1], b[2], b[3]]),
                DType::F64 => Constant::F64(b.try_into().expect("8 bytes")),
                DType::U8 => Constant::U8(b[0]),
                DType::U16 => Constant::U16(u16::from_le_bytes([b[0], b[1]])),
                DType::U32 => Constant::U32(u32::from_le_bytes(b.try_into().expect("4 bytes"))),
                DType::U64 => Constant::U64(b.try_into().expect("8 bytes")),
                DType::I8 => Constant::I8(b[0] as i8),
                DType::I16 => Constant::I16(i16::from_le_bytes([b[0], b[1]])),
                DType::I32 => Constant::I32(i32::from_le_bytes(b.try_into().expect("4 bytes"))),
                DType::I64 => Constant::I64(b.try_into().expect("8 bytes")),
                DType::Bool => Constant::Bool(b[0] != 0),
                other => todo!("interpret: decode {other:?}"),
            }
        })
        .collect()
}

fn read(mem: &Map<OpId, Mem>, globals: &Map<OpId, Vec<Constant>>, buffer: OpId, thread: usize, at: usize) -> Constant {
    match mem.get(&buffer) {
        Some(Mem::Shared(buf)) => buf[at],
        Some(Mem::Register(regs)) => regs[thread][at],
        None => globals[&buffer][at],
    }
}

fn write(
    mem: &mut Map<OpId, Mem>,
    globals: &mut Map<OpId, Vec<Constant>>,
    buffer: OpId,
    thread: usize,
    at: usize,
    value: Constant,
) {
    match mem.get_mut(&buffer) {
        Some(Mem::Shared(buf)) => buf[at] = value,
        Some(Mem::Register(regs)) => regs[thread][at] = value,
        None => globals.get_mut(&buffer).expect("buffer is a param")[at] = value,
    }
}

fn truth(c: Constant) -> bool {
    match c {
        Constant::Bool(b) => b,
        other => to_f64(other) != 0.0,
    }
}

fn to_f64(c: Constant) -> f64 {
    match c {
        Constant::F16(b) => f16(u16::from_le_bytes(b)).to_f64(),
        Constant::F32(b) => f32::from_le_bytes(b) as f64,
        Constant::F64(b) => f64::from_le_bytes(b),
        other => to_i64(other) as f64,
    }
}

fn to_i64(c: Constant) -> i64 {
    match c {
        Constant::F16(_) | Constant::F32(_) | Constant::F64(_) => to_f64(c) as i64,
        other => other.as_integer().expect("integer or float"),
    }
}

fn from_f64(dtype: DType, x: f64) -> Constant {
    match dtype {
        DType::F16 => Constant::F16(f16::from_f32(x as f32).0.to_le_bytes()),
        DType::F32 => Constant::F32((x as f32).to_le_bytes()),
        DType::F64 => Constant::F64(x.to_le_bytes()),
        DType::U8 => Constant::U8(x as u8),
        DType::U16 => Constant::U16(x as u16),
        DType::U32 => Constant::U32(x as u32),
        DType::U64 => Constant::U64((x as u64).to_le_bytes()),
        DType::I8 => Constant::I8(x as i8),
        DType::I16 => Constant::I16(x as i16),
        DType::I32 => Constant::I32(x as i32),
        DType::I64 => Constant::I64((x as i64).to_le_bytes()),
        DType::Bool => Constant::Bool(x != 0.0),
        other => todo!("interpret: encode {other:?}"),
    }
}

fn from_i64(dtype: DType, x: i64) -> Constant {
    match dtype {
        DType::U8 => Constant::U8(x as u8),
        DType::U16 => Constant::U16(x as u16),
        DType::U32 => Constant::U32(x as u32),
        DType::U64 => Constant::U64((x as u64).to_le_bytes()),
        DType::I8 => Constant::I8(x as i8),
        DType::I16 => Constant::I16(x as i16),
        DType::I32 => Constant::I32(x as i32),
        DType::I64 => Constant::I64(x.to_le_bytes()),
        DType::Bool => Constant::Bool(x != 0),
        other => from_f64(other, x as f64),
    }
}

fn cast_to(c: Constant, dtype: DType) -> Constant {
    if c.dtype() == dtype {
        return c;
    }
    if dtype.is_float() {
        from_f64(dtype, to_f64(c))
    } else if dtype == DType::Bool {
        Constant::Bool(truth(c))
    } else {
        from_i64(dtype, to_i64(c))
    }
}

fn binary(bop: BOp, dtype: DType, x: Constant, y: Constant) -> Constant {
    if dtype.is_float() {
        let (a, b) = (to_f64(x), to_f64(y));
        let r = match bop {
            BOp::Add => a + b,
            BOp::Sub => a - b,
            BOp::Mul => a * b,
            BOp::Div => a / b,
            BOp::Mod => a % b,
            BOp::Pow => a.powf(b),
            BOp::Max => a.max(b),
            BOp::Cmplt => return Constant::Bool(a < b),
            BOp::Cmpgt => return Constant::Bool(a > b),
            BOp::Cmpge => return Constant::Bool(a >= b),
            ref other => todo!("interpret: float {other:?}"),
        };
        from_f64(dtype, r)
    } else {
        let (a, b) = (to_i64(x), to_i64(y));
        let r = match bop {
            BOp::Add => a + b,
            BOp::Sub => a - b,
            BOp::Mul => a * b,
            BOp::Div => a / b,
            BOp::Mod => a % b,
            BOp::Max => a.max(b),
            BOp::And | BOp::BitAnd => a & b,
            BOp::Or | BOp::BitOr => a | b,
            BOp::BitXor => a ^ b,
            BOp::Cmplt => return Constant::Bool(a < b),
            BOp::Cmpgt => return Constant::Bool(a > b),
            BOp::Cmpge => return Constant::Bool(a >= b),
            ref other => todo!("interpret: integer {other:?}"),
        };
        from_i64(dtype, r)
    }
}

fn unary(uop: UOp, dtype: DType, x: Constant) -> Constant {
    if dtype.is_float() {
        let a = to_f64(x);
        let r = match uop {
            UOp::Neg => -a,
            UOp::Reciprocal => 1.0 / a,
            UOp::Sqrt => a.sqrt(),
            UOp::Rsqrt => 1.0 / a.sqrt(),
            UOp::Exp2 => a.exp2(),
            UOp::Log2 => a.log2(),
            UOp::Sin => a.sin(),
            UOp::Cos => a.cos(),
            UOp::Floor => a.floor(),
            UOp::Trunc => a.trunc(),
            ref other => todo!("interpret: float unary {other:?}"),
        };
        from_f64(dtype, r)
    } else {
        let a = to_i64(x);
        let r = match uop {
            UOp::Neg => -a,
            UOp::BitNot => !a,
            UOp::Not => {
                return Constant::Bool(a == 0);
            }
            ref other => todo!("interpret: integer unary {other:?}"),
        };
        from_i64(dtype, r)
    }
}

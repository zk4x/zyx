// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map, Set,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, UOp},
};
use std::fmt::Write;

/// Bit pattern for `fill_tile_bitcast` in the kernel's DST data format.
/// 16-bit formats (bf16, f16) pack the value into both halves of each 32-bit
/// DST register, mirroring how the SFPU packs two 16-bit elements per register.
fn tt_fill_bits(val: Constant) -> u32 {
    match val {
        Constant::BF16(b) | Constant::F16(b) => {
            let v = u16::from_le_bytes(b) as u32;
            (v << 16) | v
        }
        Constant::F32(b) => u32::from_le_bytes(b),
        other => todo!("fill_tile_bitcast for {other}"),
    }
}

/// TT Metalium `tt::DataFormat` constant for a zyx dtype, as used by
/// `typecast_tile`'s template parameters. Only dtypes supported by the
/// tenstorrent tile path are valid.
fn tt_dtype_format(dt: DType) -> u32 {
    match dt {
        DType::F32 => 0,  // Float32
        DType::F16 => 1,  // Float16
        DType::BF16 => 5, // Float16_b
        other => unreachable!("unsupported dtype {other:?} for tenstorrent tile op"),
    }
}

/// Init call required before using `uop`'s tile op, if any.
fn tt_unary_init(uop: UOp) -> Option<&'static str> {
    match uop {
        UOp::Neg => Some("negative_tile_init();"),
        UOp::Not => todo!("logical not tile"),
        UOp::BitNot => Some("bitwise_not_tile_init();"),
        UOp::Exp => Some("exp_tile_init();"),
        UOp::Exp2 => Some("exp2_tile_init();"),
        UOp::Ln => unreachable!("should've been changed to log2"),
        UOp::Log2 => Some("log_tile_init();"),
        UOp::Reciprocal => Some("recip_tile_init();"),
        UOp::Sqrt => Some("sqrt_tile_init();"),
        UOp::Sin => Some("sin_tile_init();"),
        UOp::Cos => Some("cos_tile_init();"),
        UOp::Floor | UOp::Trunc => Some("rounding_op_tile_init();"),
        UOp::Abs => Some("abs_tile_init();"),
    }
}

/// Init call required before using `bop`'s tile op, if any.
fn tt_binary_init(bop: BOp) -> Option<&'static str> {
    match bop {
        BOp::Add => Some("add_binary_tile_init();"),
        BOp::Sub => Some("sub_binary_tile_init();"),
        BOp::Mul => Some("mul_binary_tile_init();"),
        BOp::Div => Some("div_binary_tile_init();"),
        // Unsupported binary ops are todo!() in the tile pass anyway
        _ => None,
    }
}

impl Kernel {
    /// Generate TT Metalium reader, compute, and writer C++ kernel sources from zyx IR.
    ///
    /// Walks the kernel three times — once per section (reader, compute, writer) —
    /// emitting TT Metalium dataflow and compute API calls for each op.
    ///
    /// # Parameters
    /// - `kernel` — zyx kernel IR (after tiling passes: pad, local, group, loop_local)
    /// - `debug_asm` — if true, print each generated source to stdout
    /// - `n_inputs` — number of Global params (GlobalMut params occupy the tail of the
    ///   head-order param list after the Global + Variable params)
    /// - `n_outputs` — number of writable global tensors
    /// - `reader_params` / `compute_params` / `writer_params` — per-section lists of
    ///   param ordinals (head order) that section needs. **Runtime-arg convention**
    ///   (identical for every section): the section's args are exactly
    ///   `[its Global + Variable params in head order] + [its GlobalMut params] +
    ///   [gidx0, gidx1]` — gidx0/gidx1 are the core's coordinates in the tensix grid,
    ///   different in each core. Tenstorrent has no SIMT threads, so there are no
    ///   local ranges — local indices must have been converted to loops by the
    ///   opt_tenstorrent_tile pass before codegen.
    /// - `cb_map` — maps each Circular storage op to its single CB index,
    ///   shared by reader, compute, and writer sections
    ///
    /// # Returns
    /// `(reader_source, compute_source, writer_source)` as C++ strings ready
    /// for the tt-metal JIT compiler.
    #[allow(unused_must_use)]
    pub(crate) fn generate_tenstorrent(
        &self,
        debug_asm: bool,
        cb_map: &Map<OpId, u32>,
        reader_params: &[u32],
        compute_params: &[u32],
        writer_params: &[u32],
    ) -> Result<(String, String, String), BackendError> {
        // f64 is not supported by the tenstorrent tile path. Reject it up front
        // so downstream codegen never has to reason about f64 tiles.
        {
            let mut scan = self.head;
            let mut steps_scan = 0usize;
            while !scan.is_null() {
                steps_scan += 1;
                if steps_scan > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                if let Op::Load { src, layout: MemLayout::Tile { .. }, .. } = &self.ops[scan].op {
                    if self.dtype(*src) == DType::F64 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent has no f64 compute units -- f64 is unsupported, use f32 or bf16".into(),
                        });
                    }
                }
                scan = self.next_op(scan);
            }
        }
        // CB sync balance check. A push/wait/pop mismatch deadlocks the
        // tensix cores and wedges the board (only an external reset
        // recovers), so refuse to emit sources unless every CB balances.
        // Sections split at barriers (exactly 2: reader | compute |
        // writer). Const-trip loops expand; anything else countable
        // panics instead of risking a hang. Evaluates to the DST slot
        // limit for the compute section (docs: 16 sixteen-bit tiles,
        // 8 thirty-two-bit).
        let compute_slot_limit: usize = {
            struct Bal {
                cb: u32,
                depth_bytes: i64,
                is_f32: bool,
                reader_store: bool,
                compute_load: bool,
                compute_stores: i64,
                writer_bytes: i64,
            }
            let mut bals: Vec<Bal> = Vec::new();
            for (&storage, &cb) in cb_map.iter() {
                let Op::Storage { dtype, len, .. } = self.ops[storage].op else {
                    panic!("tenstorrent cb_map entry {storage} is not a storage op");
                };
                if cb >= 32 {
                    panic!("tenstorrent CB{cb} out of range, hardware has CB0-CB31");
                }
                if bals.iter().any(|b: &Bal| b.cb == cb) {
                    panic!("tenstorrent CB{cb} assigned to two storages, sharing is unsupported");
                }
                let elem = dtype.bit_size() as i64 / 8;
                let bytes = len * elem;
                if bytes % 2048 != 0 {
                    panic!("tenstorrent CB{cb} holds {bytes} bytes, not whole 2048B pages");
                }
                if len / 1024 != bytes / 2048 {
                    panic!("tenstorrent CB{cb} dtype {dtype:?} breaks the 2-byte-element depth math in codegen");
                }
                if bytes > 4096 {
                    panic!("tenstorrent CB{cb} needs {bytes} bytes, CB capacity is 4096 (2 pages)");
                }
                bals.push(Bal {
                    cb,
                    depth_bytes: bytes,
                    is_f32: dtype == DType::F32,
                    reader_store: false,
                    compute_load: false,
                    compute_stores: 0,
                    writer_bytes: 0,
                });
            }
            let mut matmul_seen = false;
            let mut compute_out_cbs: Vec<u32> = Vec::new();
            let mut section = 0u32;
            let mut trips: Vec<i64> = Vec::new();
            let mut scan = self.head;
            let mut steps = 0usize;
            while !scan.is_null() {
                steps += 1;
                if steps > 10_000 {
                    panic!("tenstorrent CB balance scan did not finish in 10000 steps");
                }
                match self.ops[scan].op {
                    Op::Barrier => section += 1,
                    Op::Loop { len } => {
                        let Op::Const(c) = self.ops[len].op else {
                            panic!("tenstorrent CB balance needs a const loop trip count, op {scan} is dynamic");
                        };
                        trips.push(c.as_dim().expect("tenstorrent loop trip count must be a concrete dim"));
                    }
                    Op::EndLoop => {
                        trips.pop().expect("tenstorrent EndLoop without Loop");
                    }
                    Op::Store { dst, src, layout, .. } => {
                        let trip: i64 = trips.iter().product();
                        if let MemLayout::Tile { x, y, .. } = layout {
                            if x != 32 || y != 32 {
                                panic!("tenstorrent supports only 32x32 tiles, op {scan} is {x}x{y}");
                            }
                        }
                        if section == 0 {
                            let Some(&cb) = cb_map.get(&dst) else {
                                panic!("tenstorrent reader stores must target circular buffers, op {scan} targets {dst}");
                            };
                            bals.iter_mut().find(|b| b.cb == cb).expect("tenstorrent op hit a CB missing from cb_map").reader_store = true;
                        } else if section == 1 {
                            if !matches!(layout, MemLayout::Tile { .. }) {
                                panic!("tenstorrent compute stores must be tiles, op {scan} is {layout:?}");
                            }
                            let Some(&cb) = cb_map.get(&dst) else {
                                panic!("tenstorrent compute stores must target circular buffers, op {scan} targets {dst}");
                            };
                            bals.iter_mut().find(|b| b.cb == cb).expect("tenstorrent op hit a CB missing from cb_map").compute_stores += trip;
                            if !compute_out_cbs.contains(&cb) {
                                compute_out_cbs.push(cb);
                            }
                        } else if section == 2 {
                            if let Op::Load { src: cb_src, .. } = self.ops[src].op {
                                if let Some(&cb) = cb_map.get(&cb_src) {
                                    let Op::Param { dtype, kind: ParamKind::GlobalMut, .. } = self.ops[dst].op else {
                                        panic!("tenstorrent writer store dst must be GlobalMut");
                                    };
                                    let elem = dtype.bit_size() as i64 / 8;
                                    let bytes = match layout {
                                        MemLayout::Scalar => elem,
                                        MemLayout::Tile { x, y, .. } => x as i64 * y as i64 * elem,
                                        _ => panic!("tenstorrent CB balance: unsupported writer layout {layout:?}"),
                                    };
                                    bals.iter_mut().find(|b| b.cb == cb).expect("tenstorrent op hit a CB missing from cb_map").writer_bytes += trip * bytes;
                                }
                            }
                        } else {
                            panic!("tenstorrent kernels have exactly 3 sections (2 barriers), found section {section}");
                        }
                    }
                    Op::Load { src, layout: MemLayout::Tile { x, y, .. }, .. } => {
                        if x != 32 || y != 32 {
                            panic!("tenstorrent supports only 32x32 tiles, op {scan} is {x}x{y}");
                        }
                        if section == 1 {
                            if let Some(&cb) = cb_map.get(&src) {
                                bals
                                    .iter_mut()
                                    .find(|b| b.cb == cb)
                                    .expect("tenstorrent op hit a CB missing from cb_map")
                                    .compute_load = true;
                            }
                        }
                    }
                    Op::MatmulTile { .. } => {
                        if section == 1 {
                            matmul_seen = true;
                        }
                    }
                    _ => {}
                }
                scan = self.next_op(scan);
            }
            if section != 2 {
                panic!("tenstorrent kernels need exactly 2 barriers (3 sections), found {section}");
            }
            // Codegen emits one mm_init triple from the first circular
            // store, so every matmul in the kernel must pack into the same
            // output CB.
            if matmul_seen && compute_out_cbs.len() != 1 {
                panic!("tenstorrent matmul kernels need exactly one output CB, found {:?}", compute_out_cbs);
            }
            for b in &bals {
                if b.reader_store != b.compute_load {
                    panic!(
                        "tenstorrent CB{} imbalance: reader_store={} compute_load={} (depth {} bytes)",
                        b.cb, b.reader_store, b.compute_load, b.depth_bytes
                    );
                }
                if b.compute_stores * 2048 != b.writer_bytes {
                    panic!(
                        "tenstorrent CB{} imbalance: compute pushes {} pages but writer consumes {} bytes",
                        b.cb, b.compute_stores, b.writer_bytes
                    );
                }
            }
            // DST capacity (docs: 16 sixteen-bit tiles, 8 thirty-two-bit).
            if bals.iter().any(|b| b.is_f32) { 8 } else { 16 }
        };
        // Per-section runtime-arg position of a param ordinal: the section's args are
        // exactly its needed params (head order; GlobalMut already trails Global +
        // Variable because GlobalMut occupies the tail of the head-order param list),
        // followed by gidx0/gidx1 at len(params) + axis.
        let reader_pos: Map<u32, u32> = reader_params.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
        let compute_pos: Map<u32, u32> = compute_params.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
        let writer_pos: Map<u32, u32> = writer_params.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
        // Generate reader kernel source
        let mut reader = String::new();
        writeln!(reader, "#include <cstdint>");
        writeln!(reader, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(reader, "#include \"api/dataflow/noc.h\"");
        writeln!(reader, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(reader, "#include \"api/tensor/noc_traits.h\"");
        writeln!(reader, "#include \"api/debug/device_print.h\"");
        writeln!(reader, "void kernel_main() {{");
        let mut indent = String::from("  ");

        let mut op_id = self.head;
        {
            const PAGE_SIZE: u32 = 4096;
            let mut input_arg_idx = 0u32;
            // Head-order param ordinal: Global + Variable interleaved, GlobalMut after.
            let mut param_idx = 0u32;
            let mut loop_depth = 0u32;
            // CBs this section fills, with depth in tiles (from storage
            // len): push the full depth at the end, not the whole cb_map.
            let mut filled_cbs: Vec<(u32, i64)> = Vec::new();
            let mut steps_op_id = 0usize;
            while !op_id.is_null() {
                steps_op_id += 1;
                if steps_op_id > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                match self.ops[op_id].op {
                    Op::Param { dtype: _, kind: ParamKind::Global, .. } => {
                        if reader_params.contains(&param_idx) {
                            let arg = reader_pos[&param_idx];
                            writeln!(reader, "{indent}uint32_t src{op_id} = get_arg_val<uint32_t>({arg});");
                            writeln!(reader, "{indent}auto args{op_id} = TensorAccessorArgs<{}>({arg});", input_arg_idx * 2);
                            writeln!(reader, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, src{op_id}, {PAGE_SIZE});");
                            input_arg_idx += 1;
                        }
                        param_idx += 1;
                    }
                    Op::Param { dtype, kind: ParamKind::Variable, .. } => {
                        if reader_params.contains(&param_idx) {
                            // Scalar runtime arg (I64 values must fit u32; larger
                            // values are unsupported on this path).
                            let arg = reader_pos[&param_idx];
                            writeln!(
                                reader,
                                "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});",
                                dtype.c_type(),
                                dtype.c_type()
                            );
                        }
                        param_idx += 1;
                    }
                    Op::Param { kind: ParamKind::GlobalMut, .. } => {
                        if reader_params.contains(&param_idx) {
                            let arg = reader_pos[&param_idx];
                            writeln!(reader, "{indent}uint32_t dst{op_id} = get_arg_val<uint32_t>({arg});");
                            writeln!(reader, "{indent}auto args{op_id} = TensorAccessorArgs<{}>({arg});", input_arg_idx * 2);
                            writeln!(reader, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, dst{op_id}, {PAGE_SIZE});");
                            input_arg_idx += 1;
                        }
                        param_idx += 1;
                    }
                    Op::Storage { dtype: _, scope: MemScope::Circular, .. } => {
                        if let Some(cb_id) = cb_map.get(&op_id) {
                            writeln!(reader, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
                        }
                    }
                    Op::Storage { scope: MemScope::Local, .. } => unreachable!(),
                    Op::Storage { scope: MemScope::Register, .. } => todo!(),
                    Op::Load { .. } => {}
                    Op::Store { dst, src, index: st_idx, layout: st_layout } => {
                        let Op::Load { src: ld_src, index: ld_idx, layout: ld_layout } = self.ops[src].op else {
                            panic!("tenstorrent supports only global to local loads in reader kernels with no ops inbetween")
                        };
                        let Op::Param { kind: ParamKind::Global, .. } = self.ops[ld_src].op else {
                            unreachable!()
                        };
                        let Op::Storage { dtype, scope: MemScope::Circular, .. } = self.ops[dst].op else {
                            unreachable!()
                        };

                        let elem_size = dtype.bit_size() as u32 / 8;
                        if let Some(cb_id) = cb_map.get(&dst) {
                            if !filled_cbs.iter().any(|(id, _)| *id == *cb_id) {
                                if let Op::Storage { len, .. } = self.ops[dst].op {
                                    filled_cbs.push((*cb_id, len / 1024));
                                }
                            }
                            match (ld_layout, st_layout) {
                                (MemLayout::Scalar, MemLayout::Scalar) => {
                                    if loop_depth == 0 {
                                        writeln!(reader, "{indent}cb{cb_id}.reserve_back(1);");
                                        writeln!(reader, "{indent}uint32_t rbase{cb_id} = cb{cb_id}.get_write_ptr();");
                                        writeln!(reader, "{indent}DEVICE_PRINT(\"rbase{cb_id}={{}}\\n\", rbase{cb_id});");
                                    }
                                    // Old dataflow API with raw L1 addresses
                                    // (mirrors TT's own readers): the Noc-class
                                    // CB-endpoint forms misaddress sub-tile
                                    // offsets (bit9 := bit5 substitution).
                                    writeln!(
                                        reader,
                                        "{indent}uint64_t rnoc{op_id} = p{ld_src}.get_noc_addr((uint32_t)((r{ld_idx}*{elem_size})/{PAGE_SIZE}), (uint32_t)((r{ld_idx}*{elem_size})%{PAGE_SIZE}));"
                                    );
                                    writeln!(
                                        reader,
                                        "{indent}noc_async_read(rnoc{op_id}, rbase{cb_id} + (uint32_t)(r{st_idx}*{elem_size}), {elem_size});"
                                    );
                                }
                                (
                                    MemLayout::Tile { x, y, .. },
                                    MemLayout::Tile { .. },
                                ) => {
                                    // Whole-tile DRAM -> CB transfer (tile-layout
                                    // DRAM): a single sequential NOC read.
                                    let tile_bytes =
                                        x as u32 * y as u32 * elem_size;
                                    writeln!(
                                        reader,
                                        "{indent}cb{cb_id}.reserve_back(1);"
                                    );
                                    writeln!(
                                        reader,
                                        "{indent}uint64_t rnoc{op_id} = p{ld_src}.get_noc_addr((uint32_t)((r{ld_idx}*{elem_size})/{PAGE_SIZE}), (uint32_t)((r{ld_idx}*{elem_size})%{PAGE_SIZE}));"
                                    );
                                    writeln!(
                                        reader,
                                        "{indent}noc_async_read(rnoc{op_id}, cb{cb_id}.get_write_ptr(), {tile_bytes});"
                                    );
                                }
                                _ => todo!(),
                            }
                        }
                    }
                    Op::Binary { x, y, bop } => {
                        let dt = self.dtype(op_id);
                        let _ = match bop {
                            BOp::Add => writeln!(reader, "{indent}{} r{op_id} = r{x} + r{y};", dt.c_type()),
                            BOp::Sub => writeln!(reader, "{indent}{} r{op_id} = r{x} - r{y};", dt.c_type()),
                            BOp::Mul => writeln!(reader, "{indent}{} r{op_id} = r{x} * r{y};", dt.c_type()),
                            BOp::Div => writeln!(reader, "{indent}{} r{op_id} = r{x} / r{y};", dt.c_type()),
                            BOp::Mod => writeln!(reader, "{indent}{} r{op_id} = r{x} % r{y};", dt.c_type()),
                            BOp::Max => writeln!(reader, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                            BOp::BitShiftLeft => writeln!(reader, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                            BOp::Cmplt => writeln!(reader, "{indent}{} r{op_id} = r{x} < r{y};", dt.c_type()),
                            _ => unreachable!("{bop:?}"),
                        };
                    }
                    Op::Mad { x, y, z } => {
                        let dt = self.dtype(op_id);
                        writeln!(reader, "{indent}{} r{op_id} = r{x} * r{y} + r{z};", dt.c_type());
                    }
                    Op::Loop { len } => {
                        if loop_depth == 0 {
                            for cb_id in cb_map.values() {
                                writeln!(reader, "{indent}cb{cb_id}.reserve_back(1);");
                                writeln!(reader, "{indent}uint32_t rbase{cb_id} = cb{cb_id}.get_write_ptr();");
                                writeln!(reader, "{indent}DEVICE_PRINT(\"rbase{cb_id}={{}}\\n\", rbase{cb_id});");
                            }
                        }
                        writeln!(reader, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                        indent += "  ";
                        loop_depth += 1;
                    }
                    Op::EndLoop => {
                        indent.pop();
                        indent.pop();
                        writeln!(reader, "{indent}}}");
                        loop_depth -= 1;
                    }
                    Op::Const(val) => {
                        writeln!(reader, "{indent}{} r{op_id} = {};", val.dtype().c_type(), val.c_code());
                    }
                    Op::Range { axis, kind: RangeKind::Group(_), .. } => {
                        writeln!(
                            reader,
                            "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({});",
                            reader_params.len() + axis as usize
                        );
                        writeln!(reader, "{indent}DEVICE_PRINT(\"r{op_id}=gidx{axis}={{}}\\n\", r{op_id});");
                    }
                    Op::Barrier => {
                        break;
                    }
                    Op::Cast { x, dtype } => {
                        writeln!(reader, "{indent}{} r{op_id} = ({})r{x};", dtype.c_type(), dtype.c_type());
                    }
                    Op::Bitcast { .. } => todo!("tenstorrent: bitcast not implemented"),
                    Op::Range { kind: RangeKind::Local(_), .. } => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    ref op => todo!("{op:?}"),
                }
                op_id = self.next_op(op_id);
            }
            writeln!(reader, "{indent}noc_async_read_barrier();");
            for &(cb_id, depth) in &filled_cbs {
                writeln!(reader, "{indent}cb{cb_id}.push_back({depth});");
            }
            writeln!(reader, "}}");
        }
        op_id = self.next_op(op_id);

        if debug_asm {
            println!("[tenstorrent] reader:\n{reader}");
        }

        // Generate compute kernel source
        let mut compute = String::new();
        writeln!(compute, "#include <cstdint>");
        writeln!(compute, "#include \"api/compute/common.h\"");
        writeln!(compute, "#include \"api/compute/compute_kernel_api.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_binary_sfpu.h\"");
        writeln!(compute, "#include \"api/compute/tile_move_copy.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/eltwise_unary.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/trigonometry.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/exp.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/recip.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/sqrt.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/rounding.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/negative.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/bitwise_not.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/typecast.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/fill.h\"");
        writeln!(compute, "#include \"api/compute/matmul.h\"");
        writeln!(compute, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(compute, "#include \"api/debug/device_print.h\"");
        writeln!(compute, "void kernel_main() {{");
        let mut indent = String::from("  ");
        {
            let mut cb_ids: Vec<u32> = cb_map.values().copied().collect();
            for cb_id in cb_map.values() {
                if !cb_ids.contains(cb_id) {
                    cb_ids.push(*cb_id);
                }
            }
            cb_ids.sort();
            for cb_id in &cb_ids {
                writeln!(compute, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
            }

            let input_ids: Vec<u32> = cb_map.values().copied().collect();
            let output_ids: Vec<u32> = cb_map.values().copied().collect();
            if !input_ids.is_empty() && !output_ids.is_empty() {
                let in0 = input_ids[0];
                let _in1 = input_ids.get(1).copied().unwrap_or(in0);
                let out0 = output_ids[0];
                writeln!(compute, "{indent}init_sfpu({in0}, {out0});");
            }

            let mut unary_inits: Set<&'static str> = Set::default();
            let mut binary_inits: Set<&'static str> = Set::default();
            let mut typecast_inits: Set<(u32, u32)> = Set::default();
            // (in0_cb, in1_cb) pairs used by MatmulTile ops; mm_init needs
            // them plus the section's output CB (for packer config).
            let mut mm_inits: Vec<(u32, u32)> = Vec::new();
            let mut mm_out_cb: Option<u32> = None;
            let mut has_fill = false;
            let (_dtypes, rcs) = self.compute_dtypes_and_rcs();
            let mut dst_slots: Map<OpId, Vec<u32>> = Map::default();
            let mut consumer_count: Map<OpId, u32> = Map::default();
            let mut next_slot = 0u32;
            let mut output_stores: Vec<(u32, u32)> = Vec::new();

            // Emit scalar deps of compute stores in kernel order
            let compute_stores: Vec<OpId> = {
                let mut stores = Vec::new();
                let mut scan = op_id;
                let mut steps_scan = 0usize;
                while !scan.is_null() {
                    steps_scan += 1;
                    if steps_scan > 10_000 {
                        panic!("tt_binary_init did not finish in 10000 steps");
                    }
                    if let Op::Barrier = self.ops[scan].op {
                        break;
                    }
                    if let Op::Store { .. } = self.ops[scan].op {
                        stores.push(scan);
                    }
                    scan = self.next_op(scan);
                }
                stores
            };
            let compute_deps = {
                let mut deps = Set::default();
                let mut stack: Vec<OpId> = compute_stores.iter().copied().collect();
                while let Some(id) = stack.pop() {
                    if !deps.insert(id) {
                        continue;
                    }
                    stack.extend(self.ops[id].op.parameters());
                }
                deps
            };
            {
                let mut scan = self.head;
                let mut param_idx = 0u32;
                let mut steps_scan = 0usize;
                while scan != op_id {
                    steps_scan += 1;
                    if steps_scan > 10_000 {
                        panic!("tt_codegen did not finish in 10000 steps");
                    }
                    if compute_deps.contains(&scan) {
                        match &self.ops[scan].op {
                            Op::Range { kind: RangeKind::Local(_), .. } => unreachable!(
                                "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                            ),
                            Op::Range { axis, kind: RangeKind::Group(_), .. } => {
                                writeln!(
                                    compute,
                                    "{indent}uint32_t r{scan} = get_arg_val<uint32_t>({});",
                                    compute_params.len() + *axis as usize
                                );
                            }
                            Op::Param { dtype, kind: ParamKind::Variable, .. } => {
                                if compute_params.contains(&param_idx) {
                                    let arg = compute_pos[&param_idx];
                                    writeln!(
                                        compute,
                                        "{indent}{} r{scan} = ({})get_arg_val<uint32_t>({arg});",
                                        dtype.c_type(),
                                        dtype.c_type()
                                    );
                                }
                            }
                            Op::Const(val) => {
                                writeln!(compute, "{indent}{} r{scan} = {};", val.dtype().c_type(), val.c_code());
                            }
                            Op::Binary { x, y, bop } => {
                                let dt = self.dtype(scan);
                                let _ = match bop {
                                    BOp::Add => writeln!(compute, "{indent}{} r{scan} = r{x} + r{y};", dt.c_type()),
                                    BOp::Sub => writeln!(compute, "{indent}{} r{scan} = r{x} - r{y};", dt.c_type()),
                                    BOp::Mul => writeln!(compute, "{indent}{} r{scan} = r{x} * r{y};", dt.c_type()),
                                    BOp::Div => writeln!(compute, "{indent}{} r{scan} = r{x} / r{y};", dt.c_type()),
                                    BOp::Mod => writeln!(compute, "{indent}{} r{scan} = r{x} % r{y};", dt.c_type()),
                                    BOp::BitShiftLeft => writeln!(compute, "{indent}{} r{scan} = r{x} << r{y};", dt.c_type()),
                                    BOp::Cmplt => writeln!(compute, "{indent}{} r{scan} = r{x} < r{y};", dt.c_type()),
                                    _ => unreachable!("{bop:?}"),
                                };
                            }
                            Op::Cast { x, dtype } => {
                                writeln!(compute, "{indent}{} r{scan} = r{x};", dtype.c_type());
                            }
                            Op::Bitcast { .. } => todo!("tenstorrent: bitcast not implemented"),
                            _ => {}
                        }
                    }
                    if matches!(&self.ops[scan].op, Op::Param { .. }) {
                        param_idx += 1;
                    }
                    scan = self.next_op(scan);
                }
            }

            // First pass: collect init headers from ops
            let mut scan = op_id;
            let mut steps_scan = 0usize;
            while !scan.is_null() {
                steps_scan += 1;
                if steps_scan > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                match self.ops[scan].op {
                    Op::Cast { x, .. } => {
                        if matches!(self.ops[x].op, Op::Const(_)) {
                            has_fill = true;
                        } else if matches!(self.ops[x].op, Op::Load { layout: MemLayout::Tile { .. }, .. }) {
                            let in_fmt = tt_dtype_format(self.dtype(x));
                            let out_fmt = tt_dtype_format(self.dtype(op_id));
                            if in_fmt != out_fmt {
                                typecast_inits.insert((in_fmt, out_fmt));
                            }
                        }
                    }
                    Op::Unary { x, uop } => {
                        if matches!(self.ops[x].op, Op::Const(_)) {
                            has_fill = true;
                        }
                        if let Some(init) = tt_unary_init(uop) {
                            unary_inits.insert(init);
                        }
                    }
                    Op::Binary { x, y, bop } => {
                        if matches!(self.ops[x].op, Op::Const(_)) || matches!(self.ops[y].op, Op::Const(_)) {
                            has_fill = true;
                        }
                        if let Some(init) = tt_binary_init(bop) {
                            binary_inits.insert(init);
                        }
                    }
                    Op::Store { dst, src, .. } => {
                        if matches!(self.ops[src].op, Op::Const(_)) {
                            has_fill = true;
                        }
                        if mm_out_cb.is_none() {
                            if let Op::Storage { scope: MemScope::Circular, .. } = self.ops[dst].op {
                                if let Some(&cb_id) = cb_map.get(&dst) {
                                    mm_out_cb = Some(cb_id);
                                }
                            }
                        }
                    }
                    Op::MatmulTile { x, y } => {
                        let Op::Load { src: cb_a, layout: MemLayout::Tile { .. }, .. } = self.ops[x].op else {
                            panic!("tenstorrent matmul_tile x must be a tile load from a circular buffer")
                        };
                        let Op::Load { src: cb_b, layout: MemLayout::Tile { .. }, .. } = self.ops[y].op else {
                            panic!("tenstorrent matmul_tile y must be a tile load from a circular buffer")
                        };
                        let (Some(&cb_a_id), Some(&cb_b_id)) = (cb_map.get(&cb_a), cb_map.get(&cb_b)) else {
                            panic!("tenstorrent matmul_tile inputs must be circular buffers")
                        };
                        if !mm_inits.contains(&(cb_a_id, cb_b_id)) {
                            mm_inits.push((cb_a_id, cb_b_id));
                        }
                    }
                    Op::Const(_) => has_fill = true,
                    Op::Barrier => break,
                    _ => {}
                }
                scan = self.next_op(scan);
            }

            if has_fill {
                writeln!(compute, "{indent}fill_tile_init();");
            }
            for init in unary_inits {
                writeln!(compute, "{indent}{init}");
            }
            for init in binary_inits {
                writeln!(compute, "{indent}{init}");
            }
            if !mm_inits.is_empty() {
                let out_cb = mm_out_cb.expect("tenstorrent matmul kernel needs a tile store to a circular buffer");
                // One-time HW setup (like the official matmul example):
                // all matmuls in the kernel must share this triple.
                let (first_a, first_b) = mm_inits[0];
                writeln!(compute, "{indent}compute_kernel_hw_startup({first_a}, {first_b}, {out_cb});");
                for (cb_a, cb_b) in &mm_inits {
                    writeln!(compute, "{indent}mm_init({cb_a}, {cb_b}, {out_cb});");
                }
            }
            for (in_fmt, out_fmt) in &typecast_inits {
                writeln!(compute, "{indent}typecast_tile_init<{in_fmt}, {out_fmt}>();");
            }

            // Input CBs with their depth in tiles (from storage len).
            // CBs are sized exactly, so wait/pop the full depth: the
            // reader pushes everything up front (sections are sequential).
            let mut load_input_cbs: Vec<(u32, i64)> = Vec::new();
            let mut pre_scan = op_id;
            let mut steps_pre_scan = 0usize;
            while !pre_scan.is_null() {
                steps_pre_scan += 1;
                if steps_pre_scan > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                match self.ops[pre_scan].op {
                    Op::Load { src, layout: MemLayout::Tile { .. }, .. } => {
                        if let Some(&cb_id) = cb_map.get(&src) {
                            if !load_input_cbs.iter().any(|(id, _)| *id == cb_id) {
                                let depth = if let Op::Storage { len, .. } = self.ops[src].op {
                                    len / 1024
                                } else {
                                    1
                                };
                                load_input_cbs.push((cb_id, depth));
                            }
                        }
                    }
                    Op::Barrier => break,
                    _ => {}
                }
                pre_scan = self.next_op(pre_scan);
            }
            for &(cb_id, depth) in &load_input_cbs {
                writeln!(compute, "{indent}cb{cb_id}.wait_front({depth});");
            }
            writeln!(compute, "{indent}tile_regs_acquire();");

            // Materialize constants used as tile operands (defined outside the
            // compute range) into DST slots before the tile op loop runs
            {
                let mut materialize_const = |target: OpId| {
                    if dst_slots.contains_key(&target) {
                        return;
                    }
                    let Op::Const(val) = self.ops[target].op else {
                        return;
                    };
                    let n = rcs.get(&target).copied().unwrap_or(1).max(1) as usize;
                    let mut slots = Vec::with_capacity(n);
                    for _ in 0..n {
                        let slot = next_slot;
                        next_slot += 1;
                        slots.push(slot);
                        writeln!(compute, "{indent}fill_tile_bitcast({slot}, {});", tt_fill_bits(val));
                    }
                    dst_slots.insert(target, slots);
                };
                let mut scan = op_id;
                let mut steps_scan = 0usize;
                while !scan.is_null() {
                    steps_scan += 1;
                    if steps_scan > 10_000 {
                        panic!("tt_binary_init did not finish in 10000 steps");
                    }
                    match self.ops[scan].op {
                        Op::Cast { x, .. } | Op::Unary { x, .. } | Op::Store { src: x, .. } => materialize_const(x),
                        Op::Binary { x, y, .. } => {
                            materialize_const(x);
                            materialize_const(y);
                        }
                        Op::Barrier => break,
                        _ => {}
                    }
                    scan = self.next_op(scan);
                }
            }

            let mut steps_op_id = 0usize;
            while !op_id.is_null() {
                steps_op_id += 1;
                if steps_op_id > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                match self.ops[op_id].op {
                    Op::Load { src, index: _, layout: MemLayout::Tile { .. } } => {
                        if self.dtype(src) == DType::F64 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent has no f64 compute units -- f64 is unsupported, use f32 or bf16".into(),
                            });
                        }
                        if let Some(&cb_id) = cb_map.get(&src) {
                            let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                            let mut slots = Vec::with_capacity(n);
                            for _ in 0..n {
                                let slot = next_slot;
                                next_slot += 1;
                                slots.push(slot);
                                writeln!(compute, "{indent}copy_tile({cb_id}, 0, {slot});");
                            }
                            dst_slots.insert(op_id, slots);
                        }
                    }
                    Op::Const(val) => {
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        let mut slots = Vec::with_capacity(n);
                        for _ in 0..n {
                            let slot = next_slot;
                            next_slot += 1;
                            slots.push(slot);
                            writeln!(compute, "{indent}fill_tile_bitcast({slot}, {});", tt_fill_bits(val));
                        }
                        dst_slots.insert(op_id, slots);
                    }
                    Op::Cast { x, dtype } if matches!(dtype, DType::BF16 | DType::F16 | DType::F32) => {
                        let idx = consumer_count.entry(x).or_insert(0);
                        let slot = dst_slots[&x][*idx as usize];
                        *idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot; n]);
                        let in_fmt = tt_dtype_format(self.dtype(x));
                        let out_fmt = tt_dtype_format(dtype);
                        if in_fmt != out_fmt {
                            writeln!(compute, "{indent}typecast_tile<{in_fmt}, {out_fmt}>({slot});");
                        }
                    }
                    Op::Unary { x, uop } => {
                        let idx = consumer_count.entry(x).or_insert(0);
                        let slot = dst_slots[&x][*idx as usize];
                        *idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot; n]);
                        match uop {
                            UOp::Neg => writeln!(compute, "{indent}negative_tile({slot});"),
                            UOp::Not => todo!("logical not tile"),
                            UOp::BitNot => writeln!(compute, "{indent}bitwise_not_tile({slot});"),
                            UOp::Exp => writeln!(compute, "{indent}exp_tile({slot});"),
                            UOp::Exp2 => writeln!(compute, "{indent}exp2_tile({slot});"),
                            UOp::Ln => unreachable!("should've been changed to log2"),
                            UOp::Log2 => writeln!(compute, "{indent}log_tile({slot});"),
                            UOp::Reciprocal => writeln!(compute, "{indent}recip_tile({slot});"),
                            UOp::Sqrt => writeln!(compute, "{indent}sqrt_tile({slot});"),
                            UOp::Sin => writeln!(compute, "{indent}sin_tile({slot});"),
                            UOp::Cos => writeln!(compute, "{indent}cos_tile({slot});"),
                            UOp::Floor => writeln!(compute, "{indent}floor_tile({slot});"),
                            UOp::Trunc => writeln!(compute, "{indent}trunc_tile({slot});"),
                            UOp::Abs => writeln!(compute, "{indent}abs_tile({slot});"),
                        };
                    }
                    Op::Binary { x, y, bop } => {
                        let x_idx = consumer_count.entry(x).or_insert(0);
                        let slot_x = dst_slots[&x][*x_idx as usize];
                        *x_idx += 1;
                        let y_idx = consumer_count.entry(y).or_insert(0);
                        let slot_y = dst_slots[&y][*y_idx as usize];
                        *y_idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot_x; n]);
                        match bop {
                            BOp::Add => writeln!(compute, "{indent}add_binary_tile({slot_x}, {slot_y}, {slot_x});"),
                            BOp::Sub => writeln!(compute, "{indent}sub_binary_tile({slot_x}, {slot_y}, {slot_x});"),
                            BOp::Mul => writeln!(compute, "{indent}mul_binary_tile({slot_x}, {slot_y}, {slot_x});"),
                            BOp::Div => writeln!(compute, "{indent}div_binary_tile({slot_x}, {slot_y}, {slot_x});"),
                            BOp::Pow => todo!(),
                            BOp::Mod => todo!(),
                            BOp::Cmplt => todo!(),
                            BOp::Cmpgt => todo!(),
                            BOp::Max => todo!(),
                            BOp::Or => todo!(),
                            BOp::And => todo!(),
                            BOp::BitXor => todo!(),
                            BOp::BitOr => todo!(),
                            BOp::BitAnd => todo!(),
                            BOp::BitShiftLeft => todo!(),
                            BOp::BitShiftRight => todo!(),
                            BOp::NotEq => todo!(),
                            BOp::Eq => todo!(),
                            BOp::Cmpge => todo!(),
                        };
                    }
                    Op::MatmulTile { x, y } => {
                        let Op::Load { src: cb_a, layout: MemLayout::Tile { .. }, .. } = self.ops[x].op else {
                            panic!("tenstorrent matmul_tile x must be a tile load from a circular buffer")
                        };
                        let Op::Load { src: cb_b, layout: MemLayout::Tile { .. }, .. } = self.ops[y].op else {
                            panic!("tenstorrent matmul_tile y must be a tile load from a circular buffer")
                        };
                        let (Some(&cb_a_id), Some(&cb_b_id)) = (cb_map.get(&cb_a), cb_map.get(&cb_b)) else {
                            panic!("tenstorrent matmul_tile inputs must be circular buffers")
                        };
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        let slot = next_slot;
                        next_slot += 1;
                        dst_slots.insert(op_id, vec![slot; n]);
                        // matmul_tiles accumulates into DST, which
                        // tile_regs_acquire() zeroes up front: each call
                        // computes slot = previous + A@B. K-loop
                        // accumulation is expressed in the IR with add,
                        // never in codegen.
                        writeln!(compute, "{indent}matmul_tiles({cb_a_id}, {cb_b_id}, 0, 0, {slot});");
                    }
                    Op::Store { dst, src, index: _, layout: MemLayout::Tile { .. } } => {
                        if let Some(&cb_id) = cb_map.get(&dst) {
                            let idx = consumer_count.entry(src).or_insert(0);
                            let slot = dst_slots.get(&src).expect("dst slot must exist")[*idx as usize];
                            *idx += 1;
                            output_stores.push((slot, cb_id));
                        }
                    }
                    Op::Barrier => break,
                    ref op => todo!("{op:?}"),
                }
                op_id = self.next_op(op_id);
            }
            writeln!(compute, "{indent}tile_regs_commit();");
            writeln!(compute, "{indent}tile_regs_wait();");
            for &(slot, cb_id) in &output_stores {
                writeln!(compute, "{indent}cb{cb_id}.reserve_back(1);");
                writeln!(compute, "{indent}pack_tile({slot}, {cb_id});");
            }
            for &(loaded_cb, depth) in &load_input_cbs {
                writeln!(compute, "{indent}cb{loaded_cb}.pop_front({depth});");
            }
            writeln!(compute, "{indent}tile_regs_release();");
            for &(_, cb_id) in &output_stores {
                writeln!(compute, "{indent}cb{cb_id}.push_back(1);");
            }
            if (next_slot as usize) > compute_slot_limit {
                panic!("tenstorrent compute uses {} DST slots, hardware holds {}", next_slot, compute_slot_limit);
            }
            writeln!(compute, "}}");
        }

        if debug_asm {
            println!("[tenstorrent] compute:\n{compute}");
        }

        // Generate writer kernel source
        let mut writer = String::new();
        op_id = self.next_op(op_id);

        const PAGE_SIZE: u32 = 4096;
        writeln!(writer, "#include <cstdint>");
        writeln!(writer, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(writer, "#include \"api/dataflow/noc.h\"");
        writeln!(writer, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(writer, "#include \"api/tensor/noc_traits.h\"");
        writeln!(writer, "#include \"api/debug/dprint.h\"");
        writeln!(writer, "void kernel_main() {{");

        for cb_id in cb_map.values() {
            writeln!(writer, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
        }

        // Emit accessors only for the GlobalMut params this section needs.
        // `param_idx` is the head-order ordinal over ALL Param kinds; the runtime
        // arg index is the param's position in the writer's section arg list.
        let mut out_accessor_idx = 0u32;
        {
            let mut param_idx = 0u32;
            let mut scan = self.head;
            let mut steps_scan = 0usize;
            while !scan.is_null() {
                steps_scan += 1;
                if steps_scan > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                if let Op::Param { kind: ParamKind::GlobalMut, .. } = self.ops[scan].op {
                    if writer_params.contains(&param_idx) {
                        let arg = writer_pos[&param_idx];
                        writeln!(writer, "{indent}uint32_t out{scan} = get_arg_val<uint32_t>({arg});");
                        writeln!(writer, "{indent}auto args_out{scan} = TensorAccessorArgs<{}>({arg});", out_accessor_idx * 2);
                        writeln!(writer, "{indent}auto p_out{scan} = TensorAccessor(args_out{scan}, out{scan}, {PAGE_SIZE});");
                        out_accessor_idx += 1;
                    }
                }
                if matches!(self.ops[scan].op, Op::Param { .. }) {
                    param_idx += 1;
                }
                scan = self.next_op(scan);
            }
        }

        // CBs the writer section reads (any depth): wait/push/pop these at
        // loop boundaries. Collected here because cb_map now spans sections.
        let mut writer_loop_cbs: Vec<u32> = Vec::new();
        {
            let mut scan = op_id;
            let mut steps_scan = 0usize;
            while !scan.is_null() {
                steps_scan += 1;
                if steps_scan > 10_000 {
                    panic!("tt_binary_init did not finish in 10000 steps");
                }
                match self.ops[scan].op {
                    Op::Store { src, .. } => {
                        if let Op::Load { src: cb_src, .. } = self.ops[src].op {
                            if let Some(&cb_id) = cb_map.get(&cb_src) {
                                if !writer_loop_cbs.contains(&cb_id) {
                                    writer_loop_cbs.push(cb_id);
                                }
                            }
                        }
                    }
                    Op::Barrier => break,
                    _ => {}
                }
                scan = self.next_op(scan);
            }
            writer_loop_cbs.sort();
        }

        // Gather transitive deps of all writer stores and emit them in kernel order
        {
            let writer_stores: Vec<OpId> = {
                let mut stores = Vec::new();
                let mut scan = op_id;
                let mut steps_scan = 0usize;
                while !scan.is_null() {
                    steps_scan += 1;
                    if steps_scan > 10_000 {
                        panic!("tt_binary_init did not finish in 10000 steps");
                    }
                    if let Op::Barrier = self.ops[scan].op {
                        break;
                    }
                    if let Op::Store { .. } = self.ops[scan].op {
                        stores.push(scan);
                    }
                    scan = self.next_op(scan);
                }
                stores
            };
            let writer_deps = {
                let mut deps = Set::default();
                let mut stack: Vec<OpId> = writer_stores.iter().copied().collect();
                while let Some(id) = stack.pop() {
                    if !deps.insert(id) {
                        continue;
                    }
                    stack.extend(self.ops[id].op.parameters());
                }
                deps
            };

            let mut scan = self.head;
            let mut param_idx = 0u32;
            let mut steps_scan = 0usize;
            while scan != op_id {
                steps_scan += 1;
                if steps_scan > 10_000 {
                    panic!("tt_codegen did not finish in 10000 steps");
                }
                if writer_deps.contains(&scan) {
                    match &self.ops[scan].op {
                        Op::Range { axis, kind: RangeKind::Group(_), .. } => {
                            writeln!(
                                writer,
                                "{indent}uint32_t r{scan} = get_arg_val<uint32_t>({});",
                                writer_params.len() + *axis as usize
                            );
                        }
                        Op::Param { dtype, kind: ParamKind::Variable, .. } => {
                            if writer_params.contains(&param_idx) {
                                let arg = writer_pos[&param_idx];
                                writeln!(
                                    writer,
                                    "{indent}{} r{scan} = ({})get_arg_val<uint32_t>({arg});",
                                    dtype.c_type(),
                                    dtype.c_type()
                                );
                            }
                        }
                        Op::Const(val) => {
                            writeln!(writer, "{indent}{} r{scan} = {};", val.dtype().c_type(), val.c_code());
                        }
                        Op::Binary { x, y, bop } => {
                            let dt = self.dtype(scan);
                            let _ = match bop {
                                BOp::Add => writeln!(writer, "{indent}{} r{scan} = r{x} + r{y};", dt.c_type()),
                                BOp::Sub => writeln!(writer, "{indent}{} r{scan} = r{x} - r{y};", dt.c_type()),
                                BOp::Mul => writeln!(writer, "{indent}{} r{scan} = r{x} * r{y};", dt.c_type()),
                                BOp::Div => writeln!(writer, "{indent}{} r{scan} = r{x} / r{y};", dt.c_type()),
                                BOp::Mod => writeln!(writer, "{indent}{} r{scan} = r{x} % r{y};", dt.c_type()),
                                BOp::Max => writeln!(writer, "{indent}{} r{scan} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                                BOp::BitShiftLeft => writeln!(writer, "{indent}{} r{scan} = r{x} << r{y};", dt.c_type()),
                                BOp::Cmplt => writeln!(writer, "{indent}{} r{scan} = r{x} < r{y};", dt.c_type()),
                                _ => unreachable!("{bop:?}"),
                            };
                        }
                        Op::Mad { x, y, z } => {
                            let dt = self.dtype(scan);
                            writeln!(writer, "{indent}{} r{scan} = r{x} * r{y} + r{z};", dt.c_type());
                        }
                        Op::Cast { x, dtype } => {
                            writeln!(writer, "{indent}{} r{scan} = r{x};", dtype.c_type());
                        }
                        Op::Bitcast { .. } => todo!("tenstorrent: bitcast not implemented"),
                        _ => {}
                    }
                }
                if matches!(&self.ops[scan].op, Op::Param { .. }) {
                    param_idx += 1;
                }
                scan = self.next_op(scan);
            }
        }

        let mut loop_depth = 0u32;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("tt_binary_init did not finish in 10000 steps");
            }
            match self.ops[op_id].op {
                Op::Store { dst, src, index: st_idx, layout } => {
                    if let Op::Load { src: cb_src, index: ld_idx, layout: ld_layout } = self.ops[src].op {
                        if let Some(&cb_id) = cb_map.get(&cb_src) {
                            let Op::Param { dtype, kind: ParamKind::GlobalMut, .. } = self.ops[dst].op else {
                                panic!("tt writer store dst must be a GlobalMut Param, got {:?}", self.ops[dst].op)
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            match (ld_layout, layout) {
                                (MemLayout::Scalar, MemLayout::Scalar) => {
                                    if loop_depth == 0 {
                                        writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
                                        writeln!(writer, "{indent}uint32_t wbase{cb_id} = cb{cb_id}.get_read_ptr();");
                                    }
                                    // Old dataflow API with raw L1 addresses (mirrors TT's own
                                    // writers): the Noc-class READ_PTR-selector source path
                                    // misaddresses CB offsets with bit9 != bit5.
                                    writeln!(
                                        writer,
                                        "{indent}uint64_t wnoc{dst} = p_out{dst}.get_noc_addr((uint32_t)((r{st_idx}*{elem_size})/{PAGE_SIZE}), (uint32_t)((r{st_idx}*{elem_size})%{PAGE_SIZE}));"
                                    );
                                    writeln!(
                                        writer,
                                        "{indent}noc_async_write(wbase{cb_id} + (uint32_t)(r{ld_idx}*{elem_size}), wnoc{dst}, {elem_size});"
                                    );
                                    if loop_depth == 0 {
                                        writeln!(writer, "{indent}cb{cb_id}.pop_front(1);");
                                    }
                                }
                                (MemLayout::Tile { x, y, .. }, MemLayout::Tile { .. }) => {
                                    // Whole-tile CB -> DRAM transfer (tile-layout
                                    // DRAM): a single sequential NOC write.
                                    let tile_bytes = x as u32 * y as u32 * elem_size;
                                    writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
                                    writeln!(
                                        writer,
                                        "{indent}uint64_t wnoc{dst} = p_out{dst}.get_noc_addr((uint32_t)((r{st_idx}*{elem_size})/{PAGE_SIZE}), (uint32_t)((r{st_idx}*{elem_size})%{PAGE_SIZE}));"
                                    );
                                    writeln!(
                                        writer,
                                        "{indent}noc_async_write(cb{cb_id}.get_read_ptr(), wnoc{dst}, {tile_bytes});"
                                    );
                                    writeln!(writer, "{indent}cb{cb_id}.pop_front(1);");
                                }
                                _ => todo!("add support for non-scalar stores back to DRAM"),
                            }
                        }
                    }
                }
                Op::Load { .. } => {}
                Op::Const(val) => {
                    writeln!(writer, "{indent}{} r{op_id} = {};", val.dtype().c_type(), val.c_code());
                }
                Op::Range { axis, kind: RangeKind::Group(_), .. } => {
                    writeln!(
                        writer,
                        "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({});",
                        writer_params.len() + axis as usize
                    );
                }
                Op::Cast { x, dtype } => {
                    writeln!(writer, "{indent}{} r{op_id} = r{x};", dtype.c_type());
                }
                Op::Bitcast { .. } => todo!("tenstorrent: bitcast not implemented"),
                Op::Binary { x, y, bop } => {
                    let dt = self.dtype(op_id);
                    let _ = match bop {
                        BOp::Add => writeln!(writer, "{indent}{} r{op_id} = r{x} + r{y};", dt.c_type()),
                        BOp::Sub => writeln!(writer, "{indent}{} r{op_id} = r{x} - r{y};", dt.c_type()),
                        BOp::Mul => writeln!(writer, "{indent}{} r{op_id} = r{x} * r{y};", dt.c_type()),
                        BOp::Div => writeln!(writer, "{indent}{} r{op_id} = r{x} / r{y};", dt.c_type()),
                        BOp::Mod => writeln!(writer, "{indent}{} r{op_id} = r{x} % r{y};", dt.c_type()),
                        BOp::Max => writeln!(writer, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                        BOp::BitShiftLeft => writeln!(writer, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                        BOp::Cmplt => writeln!(writer, "{indent}{} r{op_id} = r{x} < r{y};", dt.c_type()),
                        _ => unreachable!("{bop:?}"),
                    };
                }
                Op::Mad { x, y, z } => {
                    let dt = self.dtype(op_id);
                    writeln!(writer, "{indent}{} r{op_id} = r{x} * r{y} + r{z};", dt.c_type());
                }
                Op::Loop { len } => {
                    if loop_depth == 0 {
                        for cb_id in &writer_loop_cbs {
                            writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
                            writeln!(writer, "{indent}uint32_t wbase{cb_id} = cb{cb_id}.get_read_ptr();");
                        }
                    }
                    writeln!(writer, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                    indent += "  ";
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    writeln!(writer, "{indent}}}");
                    if loop_depth == 1 {
                        writeln!(writer, "{indent}noc_async_write_barrier();");
                        for cb_id in &writer_loop_cbs {
                            writeln!(writer, "{indent}cb{cb_id}.pop_front(1);");
                        }
                    }
                    loop_depth -= 1;
                }
                Op::Barrier => break,
                ref op => todo!("{op:?}"),
            }
            op_id = self.next_op(op_id);
        }
        writeln!(writer, "}}");

        if debug_asm {
            println!("[tenstorrent] writer:\n{writer}");
        }

        Ok((reader, compute, writer))
    }
}

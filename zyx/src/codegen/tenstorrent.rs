// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map, Set,
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope, UOp},
};
use std::fmt::Write;

impl Kernel {
    /// Generate TT Metalium reader, compute, and writer C++ kernel sources from zyx IR.
    ///
    /// Walks the kernel three times — once per section (reader, compute, writer) —
    /// emitting TT Metalium dataflow and compute API calls for each op.
    ///
    /// # Parameters
    /// - `kernel` — zyx kernel IR (after tiling passes: pad, local, group, loop_local)
    /// - `debug_asm` — if true, print each generated source to stdout
    /// - `n_inputs` — number of readonly global tensors (used as arg offset for `get_arg_val`)
    /// - `n_outputs` — number of writable global tensors (used as arg offset for `get_arg_val`)
    /// - `input_cb_map` — maps local Defines → input CB index (reader pushes to these)
    /// - `output_cb_map` — maps local Defines → output CB index (writer pulls from these)
    ///
    /// # Returns
    /// `(reader_source, compute_source, writer_source)` as C++ strings ready
    /// for the tt-metal JIT compiler.
    #[allow(unused_must_use)]
    pub(crate) fn generate_tenstorrent(
        &self,
        debug_asm: bool,
        n_inputs: usize,
        n_outputs: usize,
        input_cb_map: &Map<OpId, u32>,
        output_cb_map: &Map<OpId, u32>,
    ) -> (String, String, String) {
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
        writeln!(reader, "{indent}Noc noc;");

        let mut op_id = self.head;
        {
            const PAGE_SIZE: u32 = 4096;
            let mut input_arg_idx = 0u32;
            let mut loop_depth = 0u32;
            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Define { dtype: _, scope, ro, .. } => match scope {
                        Scope::Global => {
                            if ro {
                                writeln!(reader, "{indent}uint32_t src{op_id} = get_arg_val<uint32_t>({input_arg_idx});");
                                writeln!(
                                    reader,
                                    "{indent}auto args{op_id} = TensorAccessorArgs<{}>({input_arg_idx});",
                                    input_arg_idx * 2
                                );
                                writeln!(reader, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, src{op_id}, {PAGE_SIZE});");
                                input_arg_idx += 1;
                            }
                        }
                        Scope::Local => {
                            if let Some(cb_id) = input_cb_map.get(&op_id) {
                                writeln!(reader, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
                            }
                        }
                        Scope::Register => todo!(),
                    },
                    Op::Load { .. } => {}
                    Op::Store { dst, x, index: st_idx, layout: st_layout } => {
                        let Op::Load { src, index: ld_idx, layout: ld_layout } = self.ops[x].op else {
                            panic!("tenstorrent supports only global to local loads in reader kernels with no ops inbetween")
                        };
                        let Op::Define { scope: Scope::Global, ro, .. } = self.ops[src].op else {
                            unreachable!()
                        };
                        if !ro {
                            continue;
                        }
                        let Op::Define { dtype, scope: Scope::Local, .. } = self.ops[dst].op else {
                            unreachable!()
                        };

                        let elem_size = dtype.bit_size() as u32 / 8;
                        if let Some(cb_id) = input_cb_map.get(&dst) {
                            match (ld_layout, st_layout) {
                                (MemLayout::Scalar, MemLayout::Scalar) => {
                                    if loop_depth == 0 {
                                        writeln!(reader, "{indent}cb{cb_id}.reserve_back(1);");
                                    }
                                    writeln!(
                                        reader,
                                        "{indent}noc.async_read(p{src}, cb{cb_id}, {elem_size},\n{indent}  {{ .page_id = (r{ld_idx}*{elem_size})/{PAGE_SIZE}, .offset_bytes = (r{ld_idx}*{elem_size})%{PAGE_SIZE} }},\n{indent}  {{ .offset_bytes = r{st_idx}*{elem_size} }});"
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
                            BOp::Max => writeln!(reader, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                            BOp::BitShiftLeft => writeln!(reader, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                            BOp::Cmplt => writeln!(reader, "{indent}{} r{op_id} = r{x} < r{y};", dt.c_type()),
                            _ => unreachable!("{bop:?}"),
                        };
                    }
                    Op::Loop { len } => {
                        if loop_depth == 0 {
                            for cb_id in input_cb_map.values() {
                                writeln!(reader, "{indent}cb{cb_id}.reserve_back(1);");
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
                    Op::GroupIndex { axis, .. } => {
                        writeln!(reader, "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({});", n_inputs + axis as usize);
                        writeln!(reader, "{indent}DEVICE_PRINT(\"r{op_id}=gidx{axis}={{}}\\n\", r{op_id});");
                    }
                    Op::Barrier => {
                        break;
                    }
                    Op::Cast { x, dtype } => {
                        writeln!(reader, "{indent}{} r{op_id} = ({})r{x};", dtype.c_type(), dtype.c_type());
                    }
                    Op::LocalIndex { .. } => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    ref op => todo!("{op:?}"),
                }
                op_id = self.next_op(op_id);
            }
            writeln!(reader, "{indent}noc.async_read_barrier();");
            for cb_id in input_cb_map.values() {
                writeln!(reader, "{indent}cb{cb_id}.push_back(1);");
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
        writeln!(compute, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(compute, "#include \"api/debug/device_print.h\"");
        writeln!(compute, "void kernel_main() {{");
        let mut indent = String::from("  ");
        {
            let mut cb_ids: Vec<u32> = input_cb_map.values().copied().collect();
            for cb_id in output_cb_map.values() {
                if !cb_ids.contains(cb_id) {
                    cb_ids.push(*cb_id);
                }
            }
            cb_ids.sort();
            for cb_id in &cb_ids {
                writeln!(compute, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
            }

            let input_ids: Vec<u32> = input_cb_map.values().copied().collect();
            let output_ids: Vec<u32> = output_cb_map.values().copied().collect();
            if !input_ids.is_empty() && !output_ids.is_empty() {
                let in0 = input_ids[0];
                let _in1 = input_ids.get(1).copied().unwrap_or(in0);
                let out0 = output_ids[0];
                writeln!(compute, "{indent}init_sfpu({in0}, {out0});");
            }

            let mut has_sin = false;
            let mut has_binary = false;
            let (_dtypes, rcs) = self.compute_dtypes_and_rcs();
            let mut dst_slots: Map<OpId, Vec<u32>> = Map::default();
            let mut consumer_count: Map<OpId, u32> = Map::default();
            let mut next_slot = 0u32;
            let mut output_stores: Vec<(u32, u32)> = Vec::new();

            // Emit scalar deps of compute stores in kernel order
            {
                let compute_stores: Vec<OpId> = {
                    let mut stores = Vec::new();
                    let mut scan = op_id;
                    while !scan.is_null() {
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
                let mut scan = self.head;
                while scan != op_id {
                    if compute_deps.contains(&scan) {
                        match &self.ops[scan].op {
                            Op::GroupIndex { axis, .. } => {
                                writeln!(
                                    compute,
                                    "{indent}uint32_t r{scan} = get_arg_val<uint32_t>({});",
                                    n_outputs + *axis as usize
                                );
                                writeln!(compute, "{indent}DPRINT << \"compute r{scan}=gidx{axis}=\" << r{scan} << ENDL();");
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
                                    BOp::BitShiftLeft => writeln!(compute, "{indent}{} r{scan} = r{x} << r{y};", dt.c_type()),
                                    BOp::Cmplt => writeln!(compute, "{indent}{} r{scan} = r{x} < r{y};", dt.c_type()),
                                    _ => unreachable!("{bop:?}"),
                                };
                            }
                            Op::Cast { x, dtype } => {
                                writeln!(compute, "{indent}{} r{scan} = r{x};", dtype.c_type());
                            }
                            _ => {}
                        }
                    }
                    scan = self.next_op(scan);
                }
            }

            // First pass: collect init headers from ops
            let mut scan = op_id;
            while !scan.is_null() {
                match self.ops[scan].op {
                    Op::Cast { .. } => {}
                    Op::Unary { uop: UOp::Sin, .. } => has_sin = true,
                    Op::Binary { bop: BOp::Add, .. } => has_binary = true,
                    Op::Barrier => break,
                    _ => {}
                }
                scan = self.next_op(scan);
            }

            if has_sin {
                writeln!(compute, "{indent}sin_tile_init();");
            }
            if has_binary {
                writeln!(compute, "{indent}add_binary_tile_init();");
            }

            let mut load_input_cbs: Vec<u32> = Vec::new();
            let mut pre_scan = op_id;
            while !pre_scan.is_null() {
                match self.ops[pre_scan].op {
                    Op::Load { src, layout: MemLayout::Tile { .. }, .. } => {
                        if let Some(&cb_id) = input_cb_map.get(&src) {
                            if !load_input_cbs.contains(&cb_id) {
                                load_input_cbs.push(cb_id);
                            }
                        }
                    }
                    Op::Barrier => break,
                    _ => {}
                }
                pre_scan = self.next_op(pre_scan);
            }
            for &cb_id in &load_input_cbs {
                writeln!(compute, "{indent}cb{cb_id}.wait_front(1);");
            }
            writeln!(compute, "{indent}tile_regs_acquire();");

            while !op_id.is_null() {
                match self.ops[op_id].op {
                    Op::Load { src, index: _, layout: MemLayout::Tile { .. } } => {
                        if let Some(&cb_id) = input_cb_map.get(&src) {
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
                    Op::Cast { x, dtype: DType::BF16 | DType::F16 | DType::F32 } => {
                        let idx = consumer_count.entry(x).or_insert(0);
                        let slot = dst_slots[&x][*idx as usize];
                        *idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot; n]);
                    }
                    Op::Unary { x, uop: UOp::Sin } => {
                        let idx = consumer_count.entry(x).or_insert(0);
                        let slot = dst_slots[&x][*idx as usize];
                        *idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot; n]);
                        writeln!(compute, "{indent}sin_tile({slot});");
                    }
                    Op::Binary { x, y, bop: BOp::Add } => {
                        let x_idx = consumer_count.entry(x).or_insert(0);
                        let slot_x = dst_slots[&x][*x_idx as usize];
                        *x_idx += 1;
                        let y_idx = consumer_count.entry(y).or_insert(0);
                        let slot_y = dst_slots[&y][*y_idx as usize];
                        *y_idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot_x; n]);
                        writeln!(compute, "{indent}add_binary_tile({slot_x}, {slot_y}, {slot_x});");
                    }
                    Op::Store { dst, x, index: _, layout: MemLayout::Tile { .. } } => {
                        if let Some(&cb_id) = output_cb_map.get(&dst) {
                            let idx = consumer_count.entry(x).or_insert(0);
                            let slot = dst_slots[&x][*idx as usize];
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
            for &loaded_cb in &load_input_cbs {
                writeln!(compute, "{indent}cb{loaded_cb}.pop_front(1);");
            }
            writeln!(compute, "{indent}tile_regs_release();");
            for &(_, cb_id) in &output_stores {
                writeln!(compute, "{indent}cb{cb_id}.push_back(1);");
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
        writeln!(writer, "{indent}Noc noc(1);");

        for cb_id in output_cb_map.values() {
            writeln!(writer, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
        }

        let mut out_global_count = 0u32;
        {
            let mut scan = self.head;
            while !scan.is_null() {
                if let Op::Define { scope: Scope::Global, ro: false, .. } = self.ops[scan].op {
                    writeln!(writer, "{indent}uint32_t out{scan} = get_arg_val<uint32_t>({out_global_count});");
                    writeln!(
                        writer,
                        "{indent}auto args_out{scan} = TensorAccessorArgs<{}>({out_global_count});",
                        out_global_count * 2
                    );
                    writeln!(writer, "{indent}auto p_out{scan} = TensorAccessor(args_out{scan}, out{scan}, {PAGE_SIZE});");
                    out_global_count += 1;
                }
                scan = self.next_op(scan);
            }
        }

        let mut writer_loop_cbs: Vec<u32> = output_cb_map.values().copied().collect();
        writer_loop_cbs.sort();
        {
            let mut scan = op_id;
            let mut depth = 0u32;
            let mut in_loop_cbs: Vec<u32> = Vec::new();
            while !scan.is_null() {
                match self.ops[scan].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => depth -= 1,
                    Op::Store { x, .. } if depth > 0 => {
                        if let Op::Load { src, .. } = self.ops[x].op {
                            if let Some(&cb_id) = output_cb_map.get(&src) {
                                if !in_loop_cbs.contains(&cb_id) {
                                    in_loop_cbs.push(cb_id);
                                }
                            }
                        }
                    }
                    Op::Barrier if depth == 0 => break,
                    _ => {}
                }
                scan = self.next_op(scan);
            }
            if !in_loop_cbs.is_empty() {
                writer_loop_cbs = in_loop_cbs;
                writer_loop_cbs.sort();
            }
        }

        // Gather transitive deps of all writer stores and emit them in kernel order
        {
            let writer_stores: Vec<OpId> = {
                let mut stores = Vec::new();
                let mut scan = op_id;
                while !scan.is_null() {
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
            while scan != op_id {
                if writer_deps.contains(&scan) {
                    match &self.ops[scan].op {
                        Op::GroupIndex { axis, .. } => {
                            writeln!(writer, "{indent}uint32_t r{scan} = get_arg_val<uint32_t>({});", n_outputs + *axis as usize);
                            writeln!(writer, "{indent}DPRINT << \"writer r{scan}=gidx{axis}=\" << r{scan} << ENDL();");
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
                                BOp::Max => writeln!(writer, "{indent}{} r{scan} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                                BOp::BitShiftLeft => writeln!(writer, "{indent}{} r{scan} = r{x} << r{y};", dt.c_type()),
                                BOp::Cmplt => writeln!(writer, "{indent}{} r{scan} = r{x} < r{y};", dt.c_type()),
                                _ => unreachable!("{bop:?}"),
                            };
                        }
                        Op::Cast { x, dtype } => {
                            writeln!(writer, "{indent}{} r{scan} = r{x};", dtype.c_type());
                        }
                        _ => {}
                    }
                }
                scan = self.next_op(scan);
            }
        }

        let mut loop_depth = 0u32;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Store { dst, x, index: st_idx, layout } => {
                    if layout != MemLayout::Scalar {
                        todo!("add support for non-scalar stores back to DRAM")
                    }
                    if let Op::Load { src, index: ld_idx, .. } = self.ops[x].op {
                        if let Some(&cb_id) = output_cb_map.get(&src) {
                            let Op::Define { dtype, .. } = self.ops[dst].op else {
                                unreachable!()
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            if loop_depth == 0 {
                                writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
                            }
                            writeln!(
                                writer,
                                "{indent}noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(cb{cb_id}),\n{indent}  p_out{dst}, {elem_size}, {{ .offset_bytes = r{ld_idx}*{elem_size} }},\n{indent}  {{ .page_id = (r{st_idx}*{elem_size})/{PAGE_SIZE}, .offset_bytes = (r{st_idx}*{elem_size})%{PAGE_SIZE} }});"
                            );
                            if loop_depth == 0 {
                                writeln!(writer, "{indent}cb{cb_id}.pop_front(1);");
                            }
                        }
                    }
                }
                Op::Load { .. } => {}
                Op::Const(val) => {
                    writeln!(writer, "{indent}{} r{op_id} = {};", val.dtype().c_type(), val.c_code());
                }
                Op::GroupIndex { axis, .. } => {
                    writeln!(writer, "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({});", n_outputs + axis as usize);
                }
                Op::Cast { x, dtype } => {
                    writeln!(writer, "{indent}{} r{op_id} = r{x};", dtype.c_type());
                }
                Op::Binary { x, y, bop } => {
                    let dt = self.dtype(op_id);
                    let _ = match bop {
                        BOp::Add => writeln!(writer, "{indent}{} r{op_id} = r{x} + r{y};", dt.c_type()),
                        BOp::Sub => writeln!(writer, "{indent}{} r{op_id} = r{x} - r{y};", dt.c_type()),
                        BOp::Mul => writeln!(writer, "{indent}{} r{op_id} = r{x} * r{y};", dt.c_type()),
                        BOp::Max => writeln!(writer, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                        BOp::BitShiftLeft => writeln!(writer, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                        BOp::Cmplt => writeln!(writer, "{indent}{} r{op_id} = r{x} < r{y};", dt.c_type()),
                        _ => unreachable!("{bop:?}"),
                    };
                }
                Op::Loop { len } => {
                    if loop_depth == 0 {
                        for cb_id in &writer_loop_cbs {
                            writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
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
                        writeln!(writer, "{indent}noc.async_write_barrier();");
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

        (reader, compute, writer)
    }
}

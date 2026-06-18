// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! C/Clang CPU backend — compiles zyx kernel IR to C, compiles with clang, loads via dlopen

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

use super::{Device, DeviceId, DeviceInfo, DeviceProgramId, Event, MemoryPool, OpCapability, PoolBufferId, PoolId, host::HostMemoryPool};
use crate::{
    DType, Map, Set,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, Kernel, MemLayout, Op, OpId, Scope, UOp},
    shape::Dim,
    slab::Slab,
};
use libloading::{Library, Symbol};
use nanoserde::DeJson;
use std::{ffi::CString, fmt::Write, hash::BuildHasherDefault, path::PathBuf, process::Command};

#[derive(Debug, DeJson)]
#[nserde(default)]
pub struct CConfig {
    /// Enable this backend
    pub enabled: bool,
}

impl Default for CConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug)]
pub struct CProgram {
    lib: Library,
    name: String,
}

#[derive(Debug)]
pub struct CDevice {
    device_info: DeviceInfo,
    memory_pool_id: PoolId,
    programs: Slab<DeviceProgramId, CProgram>,
}

pub(super) fn initialize_device(
    config: &CConfig,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        return Err(BackendError { status: ErrorStatus::Initialization, context: "[C] backend configured out.".into() });
    }
    if debug_dev {
        println!("[C] initialized");
    }
    // C backend reuses HostMemoryPool — doesn't create its own pool
    // Just register the device with the host pool
    if memory_pools.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "C backend requires HostMemoryPool to be initialized first.".into(),
        });
    }
    let pool_id = PoolId::from(0); // use the first (host) pool
    if debug_dev {
        println!("[C] device total memory: {} MB", 10_485_760u64);
    }
    devices.push(Device::C(CDevice {
        device_info: DeviceInfo {
            compute: 10 * 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![Dim::from(1_000_000_000u64); 3],
            max_local_threads: 1,
            max_local_work_dims: vec![1, 1, 1],
            preferred_vector_size: 8,
            local_mem_size: 0,
            max_register_bytes: 1000,
            tensor_cores: false,
            warp_size: 1,
            supported_dtype_ops: [OpCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
        },
        memory_pool_id: pool_id,
        programs: Slab::new(),
    }));
    Ok(())
}

impl CDevice {
    pub const fn deinitialize(&mut self) {}

    pub const fn info(&self) -> &DeviceInfo {
        &self.device_info
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.programs.remove(program_id);
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // --- Phase 0: Compute kernel hash and check disk cache ---
        let hash = kernel.get_hash();
        let name = format!("k_{hash:016x}");

        let cache_dir = std::env::var_os("XDG_CONFIG_HOME")
            .and_then(|p| {
                let p = PathBuf::from(p);
                if p.is_absolute() { Some(p) } else { None }
            })
            .or_else(|| std::env::home_dir().map(|h| h.join(".config")))
            .map(|p| p.join("zyx/cache/c"));

        if let Some(ref cache_dir) = cache_dir {
            let cached_so = cache_dir.join(format!("{hash:016x}.so"));
            if cached_so.is_file() {
                if debug_asm {
                    println!("[C] loading cached kernel {name} from {}", cached_so.display());
                }
                if let Ok(lib) = unsafe { Library::new(&cached_so) } {
                    let program_id = self.programs.push(CProgram { lib, name });
                    return Ok(program_id);
                }
                if debug_asm {
                    println!("[C] failed to load cached kernel, recompiling...");
                }
            }
        }

        // --- Phase 1: collect global args, gws/lws ---
        let mut gws = [1u64; 3];
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            if let &Op::Index { len: dim, scope, axis } = op {
                if scope == Scope::Global {
                    gws[axis as usize] = dim.max(1u64);
                }
            }
            op_id = kernel.next_op(op_id);
        }

        // --- Phase 2: RC and dtype analysis ---
        let (dtypes, rcs) = kernel.compute_dtypes_and_rcs();

        // --- Phase 3: Codegen ---
        let mut reg_map: Map<OpId, usize> = Map::with_capacity_and_hasher(kernel.ops.len().into(), BuildHasherDefault::new());
        let mut registers: Vec<((DType, MemLayout), u32, u8)> = Vec::new();
        let mut constants: Map<OpId, Constant> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut indices: Map<OpId, u8> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());

        // Collect all Index ops and assign loop IDs.
        // Index ops with Global or Local scope become nested for-loops.
        // Register-scope indices are just variable declarations (no loop).
        let mut loop_id: u8 = 0;
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if matches!(kernel.at(op_id), Op::Index { .. }) {
                indices.insert(op_id, loop_id);
                loop_id += 1;
            }
            op_id = kernel.next_op(op_id);
        }

        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        let mut global_cast = String::new();
        let mut index: usize = 0;
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            if let &Op::Define { dtype, scope, .. } = op {
                if scope == Scope::Global {
                    if matches!(dtype, DType::F16 | DType::BF16) {
                        _ = writeln!(global_cast, "  unsigned short* p{op_id} = (unsigned short*)args[{index}];");
                    } else {
                        let ct = dtype.c_type();
                        _ = writeln!(global_cast, "  {ct}* p{op_id} = ({ct}*)args[{index}];");
                    }
                    index += 1;
                }
            } else {
                break;
            }
            op_id = kernel.next_op(op_id);
        }

        // Emit function header with void** args
        _ = writeln!(source, "void {name}(void** args, unsigned long nargs) {{");
        _ = writeln!(source, "  (void)nargs;");
        // Emit pointer casts from args array
        _ = write!(source, "{global_cast}");

        // Emit all Register/Local defines (may appear anywhere in the IR,
        // not just at the beginning — e.g. after Global defines and Index ops)
        let mut emitted_defines: Set<OpId> = Set::with_capacity_and_hasher(8, BuildHasherDefault::new());
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let &Op::Define { dtype, scope, ro, len } = kernel.at(op_id) {
                if matches!(scope, Scope::Register | Scope::Local) && !emitted_defines.contains(&op_id) {
                    emitted_defines.insert(op_id);
                    _ = writeln!(
                        source,
                        "{indent}{}{} p{op_id}[{len}] __attribute__((aligned));",
                        if ro { "const " } else { "" },
                        dtype.c_type(),
                    );
                }
            }
            op_id = kernel.next_op(op_id);
        }

        // --- Process all ops in order ---
        // For Index (Global/Local): emit for-loop header
        // For Index (Register): emit idx = 0 declaration
        // For Loop/EndLoop: emit for-loop / close brace
        // For body ops: emit computation code
        // Track how many Index loops are opened so we can close them at the end.
        let mut index_loop_depth: u8 = 0;
        loop_id = 0;
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            match op {
                &Op::Index { len, scope, .. } => {
                    match scope {
                        Scope::Global | Scope::Local => {
                            if index_loop_depth == 0 && scope == Scope::Global && gws[0] > 1 {
                                _ = writeln!(source, "{indent}#pragma omp parallel for");
                            }
                            _ = writeln!(
                                source,
                                "{indent}for (unsigned int idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{"
                            );
                            indent += "  ";
                            index_loop_depth += 1;
                        }
                        Scope::Register => {
                            // Register-scope index: just declare the variable for completeness
                            _ = writeln!(source, "{indent}unsigned int idx{loop_id} = 0;");
                        }
                    }
                    loop_id += 1;
                }
                &Op::Loop { len, .. } => {
                    indices.insert(op_id, loop_id);
                    _ = writeln!(
                        source,
                        "{indent}for (unsigned int idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{"
                    );
                    indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    if indent.len() < 2 {
                        indent = String::from("  ");
                    }
                    _ = writeln!(source, "{indent}}}");
                    loop_id -= 1;
                }
                &Op::Const(x) => {
                    constants.insert(op_id, x);
                }
                &Op::Load { src, index, layout } => {
                    if let Some(&rc) = rcs.get(&op_id) {
                        let dtype = dtypes[&op_id];
                        let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id);
                        let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rc, loop_id);
                        match layout {
                            MemLayout::Scalar => match dtypes[&src].0 {
                                DType::F16 => {
                                    _ = writeln!(source, "{indent}r{reg} = f16tof32(p{src}[{idx}]);");
                                }
                                DType::BF16 => {
                                    _ = writeln!(source, "{indent}r{reg} = bf16tof32(p{src}[{idx}]);");
                                }
                                _ => {
                                    _ = writeln!(source, "{indent}r{reg} = p{src}[{idx}];");
                                }
                            },
                            MemLayout::Vector(len) => match dtypes[&src].0 {
                                DType::F16 => {
                                    for i in 0..len {
                                        _ = writeln!(source, "{indent}r{reg}.s{i} = f16tof32(p{src}[{idx} + {i}]);");
                                    }
                                }
                                DType::BF16 => {
                                    for i in 0..len {
                                        _ = writeln!(source, "{indent}r{reg}.s{i} = bf16tof32(p{src}[{idx} + {i}]);");
                                    }
                                }
                                _ => {
                                    for i in 0..len {
                                        _ = writeln!(source, "{indent}r{reg}.s{i} = p{src}[{idx} + {i}];");
                                    }
                                }
                            },
                            MemLayout::Tile { .. } => todo!(),
                        }
                    }
                }
                &Op::Store { dst, x: src, index, layout } => {
                    let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let x = get_var(src, &constants, &indices, &reg_map, &mut registers, loop_id);
                    match layout {
                        MemLayout::Scalar => match dtypes[&dst].0 {
                            DType::F16 => {
                                _ = writeln!(source, "{indent}p{dst}[{idx}] = f32tof16({x});");
                            }
                            DType::BF16 => {
                                _ = writeln!(source, "{indent}p{dst}[{idx}] = f32tobf16({x});");
                            }
                            _ => {
                                _ = writeln!(source, "{indent}p{dst}[{idx}] = {x};");
                            }
                        },
                        MemLayout::Vector(len) => match dtypes[&dst].0 {
                            DType::F16 => {
                                for i in 0..len {
                                    _ = writeln!(source, "{indent}p{dst}[{idx} + {i}] = f32tof16({x}.s{i});");
                                }
                            }
                            DType::BF16 => {
                                for i in 0..len {
                                    _ = writeln!(source, "{indent}p{dst}[{idx} + {i}] = f32tobf16({x}.s{i});");
                                }
                            }
                            _ => {
                                for i in 0..len {
                                    _ = writeln!(source, "{indent}p{dst}[{idx} + {i}] = {x}.s{i};");
                                }
                            }
                        },
                        MemLayout::Tile { .. } => todo!(),
                    }
                }
                &Op::Cast { x, dtype } => {
                    let vlen = dtypes[&x].1;
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, vlen), rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = ({}){x};", dtype.c_type());
                }
                &Op::Unary { x, uop } => {
                    let dtype = dtypes[&x];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match uop {
                        UOp::BitNot => _ = writeln!(source, "{indent}r{reg} = ~{x};"),
                        UOp::Neg => _ = writeln!(source, "{indent}r{reg} = -{x};"),
                        UOp::Exp => _ = writeln!(source, "{indent}r{reg} = exp({x});"),
                        UOp::Exp2 => _ = writeln!(source, "{indent}r{reg} = exp2({x});"),
                        UOp::Ln => _ = writeln!(source, "{indent}r{reg} = log({x});"),
                        UOp::Log2 => _ = writeln!(source, "{indent}r{reg} = log2({x});"),
                        UOp::Reciprocal => {
                            _ = writeln!(source, "{indent}r{reg} = {}/{x};", dtype.0.one_constant().c_code());
                        }
                        UOp::Sqrt => _ = writeln!(source, "{indent}r{reg} = sqrt({x});"),
                        UOp::Sin => _ = writeln!(source, "{indent}r{reg} = sin({x});"),
                        UOp::Cos => _ = writeln!(source, "{indent}r{reg} = cos({x});"),
                        UOp::Floor => _ = writeln!(source, "{indent}r{reg} = floor({x});"),
                        UOp::Trunc => _ = writeln!(source, "{indent}r{reg} = trunc({x});"),
                        UOp::Abs => _ = writeln!(source, "{indent}r{reg} = fabs({x});"),
                    }
                }
                Op::Vectorize { ops } => {
                    let dtype = dtypes[&op_id];
                    let mut vars = String::new();
                    for &x in ops {
                        let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                        _ = write!(vars, "{x}, ");
                    }
                    vars.pop();
                    vars.pop();
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    let dtype = dtypes[&op_id];
                    let vlen = match dtype.1 {
                        MemLayout::Vector(n) => n,
                        _ => unreachable!(),
                    };
                    _ = writeln!(source, "{indent}r{reg} = ({}{}){{{}}};", dtype.0.c_type(), vlen, vars);
                }
                &Op::Wmma { .. } => {
                    todo!("C needs higher level of abstraction than WMMA, as WMMA requires cross-thread sharing")
                }
                &Op::Devectorize { vec, idx } => {
                    let dtype = dtypes[&op_id];
                    let vec = get_var(vec, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {vec}.s{idx};");
                }
                &Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = match bop {
                        BOp::Add => writeln!(source, "{indent}r{reg} = {x} + {y};"),
                        BOp::Sub => writeln!(source, "{indent}r{reg} = {x} - {y};"),
                        BOp::Mul => writeln!(source, "{indent}r{reg} = {x} * {y};"),
                        BOp::Div => writeln!(source, "{indent}r{reg} = {x} / {y};"),
                        BOp::Pow => writeln!(source, "{indent}r{reg} = pow({x}, {y});"),
                        BOp::Mod => writeln!(source, "{indent}r{reg} = (int){x} % (int){y};"),
                        BOp::Cmplt => writeln!(source, "{indent}r{reg} = {x} < {y};"),
                        BOp::Cmpgt => writeln!(source, "{indent}r{reg} = {x} > {y};"),
                        BOp::Max => writeln!(source, "{indent}r{reg} = fmax({x}, {y});"),
                        BOp::Or => writeln!(source, "{indent}r{reg} = {x} || {y};"),
                        BOp::And => writeln!(source, "{indent}r{reg} = {x} && {y};"),
                        BOp::BitXor => writeln!(source, "{indent}r{reg} = {x} ^ {y};"),
                        BOp::BitOr => writeln!(source, "{indent}r{reg} = {x} | {y};"),
                        BOp::BitAnd => writeln!(source, "{indent}r{reg} = {x} & {y};"),
                        BOp::BitShiftLeft => writeln!(source, "{indent}r{reg} = {x} << {y};"),
                        BOp::BitShiftRight => writeln!(source, "{indent}r{reg} = {x} >> {y};"),
                        BOp::NotEq => writeln!(source, "{indent}r{reg} = {x} != {y};"),
                        BOp::Eq => writeln!(source, "{indent}r{reg} = {x} == {y};"),
                    };
                }
                &Op::Mad { x, y, z } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let z = get_var(z, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {x} * {y} + {z};");
                }
                &Op::If { condition } => {
                    let condition = get_var(condition, &constants, &indices, &reg_map, &mut registers, loop_id);
                    _ = writeln!(source, "{indent}if ({condition}) {{");
                    indent += "  ";
                }
                Op::EndIf => {
                    indent.pop();
                    indent.pop();
                    if indent.len() < 2 {
                        indent = String::from("  ");
                    }
                    _ = writeln!(source, "{indent}}}");
                }
                Op::Define { .. } | Op::Barrier { .. } => {}
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Move { .. } | Op::Reduce { .. } => {
                    unreachable!()
                }
            }
            op_id = kernel.next_op(op_id);
        }

        // Close all Index for-loops that are still open
        for _ in 0..index_loop_depth {
            indent.pop();
            indent.pop();
            if indent.len() < 2 {
                indent = String::from("  ");
            }
            _ = writeln!(source, "{indent}}}");
        }

        // Close function
        indent.pop();
        indent.pop();
        _ = writeln!(source, "}}");

        // Build register declarations string
        let mut reg_str = String::new();
        if !registers.is_empty() {
            let (dt, _, _) = registers[0];
            let mut prev_dt = dt;
            let prefix = "  ";
            _ = write!(
                reg_str,
                "{prefix}{}{} r0",
                dt.0.c_type(),
                match dt.1 {
                    MemLayout::Scalar => "".into(),
                    MemLayout::Vector(len) => len.to_string(),
                    MemLayout::Tile { .. } => unreachable!(),
                }
            );
            let mut i = 1;
            for (dt, _, _) in &registers[1..] {
                if *dt == prev_dt {
                    _ = write!(reg_str, ", r{i}");
                } else {
                    _ = write!(
                        reg_str,
                        ";\n{prefix}{}{} r{i}",
                        dt.0.c_type(),
                        match dt.1 {
                            MemLayout::Scalar => "".into(),
                            MemLayout::Vector(len) => len.to_string(),
                            MemLayout::Tile { .. } => unreachable!(),
                        }
                    );
                }
                prev_dt = *dt;
                i += 1;
            }
            _ = writeln!(reg_str, ";");
        }

        // Insert register declarations after function opening brace
        // Find the position after the opening brace line
        if let Some(pos) = source.find("{\n") {
            source.insert_str(pos + 2, &reg_str);
        } else {
            _ = writeln!(source, "{reg_str}");
        }

        if debug_asm {
            println!();
            println!("{source}");
        }

        // --- Phase 4: Compile with clang ---
        let tmp_dir = std::env::temp_dir().join(format!("zyx_c_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let c_path = tmp_dir.join(format!("{name}.c"));
        let so_path = tmp_dir.join(format!("{name}.so"));

        // Add conversion helpers if F16/BF16 values are used
        let f16_helpers = if !dtypes.values().any(|(dt, _)| matches!(dt, DType::F16 | DType::BF16)) {
            String::new()
        } else {
            r"static inline float f16tof32(unsigned short h) {
  unsigned int sign = (unsigned int)(h & 0x8000) << 16;
  unsigned int mantissa = (unsigned int)(h & 0x03FF);
  unsigned int exp = (unsigned int)((h >> 10) & 0x1F);
  unsigned int f;
  if (exp == 0) {
    if (mantissa == 0) { f = sign; }
    else {
      int e = -1; unsigned int m = mantissa;
      while ((m & 0x0400) == 0) { m <<= 1; e--; }
      f = sign | ((127 + e) << 23) | ((m & 0x03FF) << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000 | (mantissa << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (mantissa << 13);
  }
  float r; memcpy(&r, &f, sizeof(r)); return r;
}
static inline unsigned short f32tof16(float v) {
  unsigned int f; memcpy(&f, &v, sizeof(f));
  unsigned int sign = (f >> 16) & 0x8000;
  unsigned int exp = (f >> 23) & 0xFF;
  unsigned int mantissa = f & 0x007FFFFF;
  unsigned short h;
  if (exp == 0) { h = (unsigned short)sign; }
  else if (exp == 255) { h = (unsigned short)(sign | 0x7C00 | (mantissa >> 13)); }
  else {
    int new_exp = (int)exp - 127 + 15;
    if (new_exp >= 31) { h = (unsigned short)(sign | 0x7C00); }
    else if (new_exp <= 0) { h = (unsigned short)sign; }
    else { h = (unsigned short)(sign | (new_exp << 10) | (mantissa >> 13)); }
  }
  return h;
}
static inline float bf16tof32(unsigned short h) {
  unsigned int b = (unsigned int)h << 16; float r; memcpy(&r, &b, sizeof(r)); return r;
}
static inline unsigned short f32tobf16(float v) {
  unsigned int b; memcpy(&b, &v, sizeof(b)); return (unsigned short)(b >> 16);
}
"
            .to_string()
        };
        // Add #include for math functions and optional OpenMP header
        let omp_include = if gws[0] > 1 { "#include <omp.h>\n" } else { "" };
        let mut vec_types = String::new();
        for (dt, _, _) in &registers {
            if let MemLayout::Vector(len) = dt.1 {
                let base = dt.0.c_type();
                let name = format!("{base}{len}");
                if !vec_types.contains(&format!("\ntypedef {base} {name}")) {
                    _ = writeln!(vec_types, "typedef {base} {name} __attribute__((ext_vector_type({len})));");
                }
            }
        }
        let full_source =
            format!("#include <math.h>\n#include <stdint.h>\n#include <string.h>\n{omp_include}{vec_types}{f16_helpers}{source}");
        std::fs::write(&c_path, &full_source).map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("Failed to write C source: {e}").into(),
        })?;

        // Try clang-11, clang, gcc, cc in order
        let compilers = ["clang-11", "clang", "gcc", "cc"];
        let compiler = compilers
            .iter()
            .find(|c| Command::new(c).arg("--version").output().is_ok())
            .copied()
            .unwrap_or("cc");
        let is_clang = compiler.contains("clang");

        // Try compiling with OpenMP first (best-effort parallelism)
        let has_openmp = gws[0] > 1;
        let openmp_success = if has_openmp {
            let openmp_flag = if is_clang { "-fopenmp=libgomp" } else { "-fopenmp" };
            let output = Command::new(compiler)
                .args(["-shared", "-O3", "-ffast-math", "-fPIC", "-o"])
                .arg(&so_path)
                .arg(&c_path)
                .arg("-lm")
                .arg(openmp_flag)
                .output();
            matches!(output, Ok(o) if o.status.success())
        } else {
            false
        };

        if !openmp_success {
            // Fall back to sequential: strip OpenMP pragma and include, recompile without -fopenmp
            let seq_source = full_source
                .replace("#pragma omp parallel for\n", "")
                .replace("#include <omp.h>\n", "");
            std::fs::write(&c_path, &seq_source).map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("Failed to write C source: {e}").into(),
            })?;
            let output = Command::new(compiler)
                .args(["-shared", "-O3", "-ffast-math", "-fPIC", "-o"])
                .arg(&so_path)
                .arg(&c_path)
                .arg("-lm")
                .output()
                .map_err(|e| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("Failed to run compiler '{compiler}': {e}. Is a C compiler installed?").into(),
                })?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if debug_asm {
                    println!("[C] compiler stderr:\n{stderr}");
                }
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("Compiler '{compiler}' compilation failed:\n{stderr}").into(),
                });
            }
        }

        // Cache the compiled .so for future runs
        if let Some(ref cache_dir) = cache_dir {
            let _ = std::fs::create_dir_all(cache_dir);
            let cached_so = cache_dir.join(format!("{hash:016x}.so"));
            let _ = std::fs::copy(&so_path, &cached_so);
        }

        // Load the shared library
        let lib = unsafe { Library::new(&so_path) }.map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("Failed to dlopen compiled kernel: {e}").into(),
        })?;

        let program_id = self.programs.push(CProgram { lib, name });
        Ok(program_id)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut HostMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list; // sync not needed for sequential CPU

        let program = &self.programs[program_id];

        // Get buffer pointers
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(args.len());
        for &arg in args {
            let ptr = memory_pool.buffer_ptr(arg).ok_or_else(|| BackendError {
                status: ErrorStatus::MemoryCopyH2P,
                context: "Invalid buffer id in kernel launch".into(),
            })?;
            ptrs.push(ptr);
        }

        let func_name = CString::new(program.name.as_str()).unwrap();
        unsafe {
            let func: Symbol<unsafe extern "C" fn(*const *mut std::ffi::c_void, usize)> =
                program.lib.get(func_name.as_bytes()).map_err(|e| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("Failed to find kernel symbol: {e}").into(),
                })?;
            let ptrs_raw: Vec<*mut std::ffi::c_void> = ptrs.iter().map(|p| (*p).cast::<std::ffi::c_void>()).collect();
            func(ptrs_raw.as_ptr(), ptrs_raw.len());
        }

        Ok(Event::Host(super::host::HostEvent))
    }
}

// --- New reg / get_var helpers ---

fn new_reg(
    op_id: OpId,
    reg_map: &mut Map<OpId, usize>,
    registers: &mut Vec<((DType, MemLayout), u32, u8)>,
    dtype: (DType, MemLayout),
    rc: u32,
    current_loop_level: u8,
) -> usize {
    for (i, (dt, nrc, loop_level)) in registers.iter_mut().enumerate() {
        if *nrc == 0 && *dt == dtype && current_loop_level <= *loop_level {
            reg_map.insert(op_id, i);
            *nrc = rc;
            *loop_level = current_loop_level;
            return i;
        }
    }
    let i = registers.len();
    registers.push((dtype, rc, current_loop_level));
    reg_map.insert(op_id, i);
    i
}

fn get_var(
    op_id: OpId,
    constants: &Map<OpId, Constant>,
    indices: &Map<OpId, u8>,
    reg_map: &Map<OpId, usize>,
    registers: &mut [((DType, MemLayout), u32, u8)],
    loop_level: u8,
) -> String {
    if let Some(c) = constants.get(&op_id) {
        c.c_code()
    } else if let Some(&id) = indices.get(&op_id) {
        format!("idx{id}")
    } else if let Some(&reg) = reg_map.get(&op_id) {
        if registers[reg].2 == loop_level {
            registers[reg].1 -= 1;
        }
        format!("r{reg}")
    } else {
        unreachable!()
    }
}

// --- C type codegen helpers ---

impl DType {
    const fn c_type(self) -> &'static str {
        match self {
            Self::F64 => "double",
            Self::U8 | Self::Bool => "unsigned char",
            Self::U16 => "unsigned short",
            Self::U32 => "unsigned int",
            Self::U64 => "unsigned long",
            Self::I8 => "signed char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::F32 | Self::F16 | Self::BF16 => "float", // fallback to float
        }
    }
}

impl Constant {
    fn c_code(self) -> String {
        match self {
            Self::F32(x) => format!("{:.16}f", f32::from_le_bytes(x)),
            Self::F64(x) => format!("{:.16}", f64::from_le_bytes(x)),
            Self::U8(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}u"),
            Self::U64(x) => format!("{}ul", u64::from_le_bytes(x)),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{}l", i64::from_le_bytes(x)),
            Self::Bool(x) => format!("{x}"),
            Self::F16(x) => format!("{:.16}f", half::f16::from_le_bytes(x).to_f32()),
            Self::BF16(x) => format!("{:.16}f", half::bf16::from_le_bytes(x).to_f32()),
        }
    }
}

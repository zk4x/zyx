// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! SPIR-V binary codegen from zyx kernel IR.
//! Translates kernel IR ops to SPIR-V machine code (Vec<u32>).

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId, Scope, UOp},
};
use crate::error::{BackendError, ErrorStatus};
use std::hash::BuildHasherDefault;

// SPIR-V magic and version
const MAGIC: u32 = 0x0723_0203;
const VERSION: u32 = 0x0001_0500;
const GENERATOR: u32 = 0;
const SCHEMA: u32 = 0;

// Storage classes
const SC_FUNCTION: u32 = 7;
const SC_INPUT: u32 = 1;
const SC_STORAGE_BUFFER: u32 = 12;
const SC_WORKGROUP: u32 = 4;

// Decorations
const DEC_BUILT_IN: u32 = 11;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_BINDING: u32 = 33;
const DEC_NON_WRITABLE: u32 = 24;

// BuiltIns
const BI_WORKGROUP_ID: u32 = 26;
const BI_LOCAL_INVOCATION_ID: u32 = 27;


// Execution model
const EXEC_GL_COMPUTE: u32 = 5;

// Execution modes
const MODE_LOCAL_SIZE: u32 = 17;

const LOOP_CTRL_NONE: u32 = 0;
const SELECT_CTRL_NONE: u32 = 0;
const FN_CTRL_NONE: u32 = 0;

// Barrier scopes/semantics
const SCOPE_WORKGROUP: u32 = 2;
const SEM_ACQUIRE_RELEASE: u32 = 0x8;
const SEM_WORKGROUP_MEMORY: u32 = 0x100;

macro_rules! op {
    ($($id:ident = $val:expr),+ $(,)?) => {
        $(const $id: u16 = $val;)+
    };
}

op! {
    OP_CAPABILITY = 17,
    OP_EXT_INST_IMPORT = 11,
    OP_MEMORY_MODEL = 14,
    OP_ENTRY_POINT = 15,
    OP_EXECUTION_MODE = 16,
    OP_DECORATE = 71,
    OP_TYPE_VOID = 19,
    OP_TYPE_BOOL = 20,
    OP_TYPE_INT = 21,
    OP_TYPE_FLOAT = 22,
    OP_TYPE_VECTOR = 23,
    OP_TYPE_ARRAY = 28,
    OP_TYPE_RUNTIME_ARRAY = 29,
    OP_TYPE_POINTER = 32,
    OP_TYPE_FUNCTION = 33,
    OP_CONSTANT = 43,
    OP_VARIABLE = 59,
    OP_FUNCTION = 54,
    OP_FUNCTION_END = 56,
    OP_LABEL = 248,
    OP_BRANCH = 249,
    OP_BRANCH_CONDITIONAL = 250,
    OP_LOOP_MERGE = 246,
    OP_SELECTION_MERGE = 247,
    OP_RETURN = 253,
    OP_LOAD = 61,
    OP_STORE = 62,
    OP_ACCESS_CHAIN = 65,
    OP_FADD = 129,
    OP_FSUB = 131,
    OP_FMUL = 133,
    OP_FDIV = 136,
    OP_FNEGATE = 127,
    OP_FMOD = 141,
    OP_IADD = 128,
    OP_ISUB = 130,
    OP_IMUL = 132,
    OP_SDIV = 135,
    OP_UDIV = 134,
    OP_SREM = 138,
    OP_SNEGATE = 126,
    OP_NOT = 200,
    OP_SHIFT_LEFT_LOGICAL = 196,
    OP_SHIFT_RIGHT_LOGICAL = 194,
    OP_BITWISE_AND = 199,
    OP_BITWISE_OR = 197,
    OP_BITWISE_XOR = 198,
    OP_I_EQUAL = 170,
    OP_I_NOT_EQUAL = 171,
    OP_U_LESS_THAN = 176,
    OP_U_GREATER_THAN = 172,
    OP_S_LESS_THAN = 177,
    OP_S_GREATER_THAN = 173,
    OP_F_ORD_LESS_THAN = 184,
    OP_F_ORD_GREATER_THAN = 186,
    OP_F_ORD_EQUAL = 180,
    OP_F_ORD_NOT_EQUAL = 182,
    OP_CONVERT_F_TO_U = 109,
    OP_CONVERT_F_TO_S = 110,
    OP_CONVERT_S_TO_F = 111,
    OP_CONVERT_U_TO_F = 112,
    OP_F_CONVERT = 115,
    OP_S_CONVERT = 114,
    OP_U_CONVERT = 113,
    OP_SELECT = 169,
    OP_EXT_INST = 12,
    OP_CONTROL_BARRIER = 224,
    OP_COMPOSITE_EXTRACT = 81,
    OP_BITCAST = 124,
}

// GLSL.std.450 extended instructions used
#[allow(non_upper_case_globals)]
mod glsl {
    pub const Trunc: u32 = 3;
    pub const FAbs: u32 = 4;
    pub const Floor: u32 = 8;
    pub const FMax: u32 = 40;
    pub const Sin: u32 = 13;
    pub const Cos: u32 = 14;
    pub const Exp2: u32 = 29;
    pub const Log2: u32 = 30;
    pub const Sqrt: u32 = 31;
    pub const Pow: u32 = 26;
}

// ---------- SPIR-V binary assembler ----------

struct Asm {
    words: Vec<u32>,
    next_id: u32,
}

impl Asm {
    fn new() -> Self {
        Self { words: Vec::new(), next_id: 1 }
    }

    fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn set_bound(&mut self) {
        self.words[3] = self.next_id;
    }

    // Emit instruction without result (opcode + operands)
    fn emit(&mut self, op: u16, operands: &[u32]) {
        let wc = 1u16 + operands.len() as u16;
        self.words.push((wc as u32) << 16 | op as u32);
        self.words.extend_from_slice(operands);
    }

    // Emit instruction with result type + id (type, id, opcode, operands...)
    fn emit_typed(&mut self, op: u16, type_id: u32, result_id: u32, operands: &[u32]) {
        let wc = 3u16 + operands.len() as u16;
        self.words.push((wc as u32) << 16 | op as u32);
        self.words.push(type_id);
        self.words.push(result_id);
        self.words.extend_from_slice(operands);
    }

    // Emit type declaration (id, opcode, operands...)
    fn emit_type(&mut self, op: u16, result_id: u32, operands: &[u32]) {
        let wc = 2u16 + operands.len() as u16;
        self.words.push((wc as u32) << 16 | op as u32);
        self.words.push(result_id);
        self.words.extend_from_slice(operands);
    }
}

// ---------- Type helpers ----------

fn emit_type(asm: &mut Asm, cache: &mut Map<DType, u32>, dt: DType) -> u32 {
    if let Some(&id) = cache.get(&dt) { return id; }
    let id = match dt {
        DType::Bool => { let i = asm.id(); asm.emit_type(OP_TYPE_BOOL, i, &[]); i }
        DType::U8 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[8, 0]); i }
        DType::U16 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[16, 0]); i }
        DType::U32 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[32, 0]); i }
        DType::U64 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[64, 0]); i }
        DType::I8 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[8, 1]); i }
        DType::I16 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[16, 1]); i }
        DType::I32 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[32, 1]); i }
        DType::I64 => { let i = asm.id(); asm.emit_type(OP_TYPE_INT, i, &[64, 1]); i }
        DType::F16 | DType::BF16 => { let i = asm.id(); asm.emit_type(OP_TYPE_FLOAT, i, &[16]); i }
        DType::F32 => { let i = asm.id(); asm.emit_type(OP_TYPE_FLOAT, i, &[32]); i }
        DType::F64 => { let i = asm.id(); asm.emit_type(OP_TYPE_FLOAT, i, &[64]); i }
    };
    cache.insert(dt, id);
    id
}

// ---------- Compute dtypes for all ops ----------

fn compute_dtypes(kernel: &Kernel) -> Map<OpId, DType> {
    let mut dt: Map<OpId, DType> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
    let mut op_id = kernel.head;
    while !op_id.is_null() {
        match kernel.at(op_id) {
            Op::Const(x) => { dt.insert(op_id, x.dtype()); }
            &Op::Define { dtype, .. } => { dt.insert(op_id, dtype); }
            &Op::Load { src, .. } => { let d = dt[&src]; dt.insert(op_id, d); }
            &Op::Store { x, .. } => { let d = dt[&x]; dt.insert(op_id, d); }
            &Op::Cast { x, dtype } => { dt.insert(op_id, dtype); let _ = x; }
            &Op::Unary { x, .. } => { let d = dt[&x]; dt.insert(op_id, d); }
            &Op::Binary { x, y, bop } => {
                if matches!(bop, BOp::Cmpgt | BOp::Cmplt | BOp::NotEq | BOp::Eq | BOp::And | BOp::Or) {
                    dt.insert(op_id, DType::Bool);
                } else {
                    dt.insert(op_id, dt[&x]);
                }
                let _ = y;
            }
            &Op::Mad { x, .. } => { dt.insert(op_id, dt[&x]); }
            &Op::Index { .. } | &Op::Loop { .. } => { dt.insert(op_id, IDX_T); }
            Op::If { condition } => { dt.insert(op_id, dt[condition]); }
            _ => {}
        }
        op_id = kernel.next_op(op_id);
    }
    dt
}

// ---------- Public compile function ----------

pub fn compile(kernel: &Kernel, debug_asm: bool) -> Result<Vec<u32>, BackendError> {
    let dtypes = compute_dtypes(kernel);
    let mut asm = Asm::new();
    
    // === SPIR-V Header ===
    asm.words.push(MAGIC);
    asm.words.push(VERSION);
    asm.words.push(GENERATOR);
    asm.words.push(0); // Bound will be set later
    asm.words.push(SCHEMA);
    
    // Required SPIR-V instructions
    asm.emit(OP_CAPABILITY, &[1]); // Shader capability
    let glsl_id = asm.id();
    let glsl_name = b"GLSL.std.450\x00";
    let mut glsl_words = Vec::new();
    for chunk in glsl_name.chunks(4) {
        let mut w = 0u32;
        for (i, &b) in chunk.iter().enumerate() { w |= (b as u32) << (i * 8); }
        glsl_words.push(w);
    }
    let void_id = asm.id(); // reserved for OpTypeVoid (emitted later in types section)
    asm.emit_type(OP_EXT_INST_IMPORT, glsl_id, &glsl_words);
    asm.emit(OP_MEMORY_MODEL, &[0, 1]); // Logical GLSL450

    // === Type helpers (closures) ===
    let push_dtype = |asm: &mut Asm, cache: &mut Map<DType, u32>, entries: &mut Vec<(u16, u32, Vec<u32>)>, dt: DType| {
        if let Some(&id) = cache.get(&dt) { return id; }
        let id = match dt {
            DType::Bool => { let i = asm.id(); entries.push((OP_TYPE_BOOL, i, vec![])); i }
            DType::U8 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![8, 0])); i }
            DType::U16 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![16, 0])); i }
            DType::U32 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![32, 0])); i }
            DType::U64 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![64, 0])); i }
            DType::I8 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![8, 1])); i }
            DType::I16 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![16, 1])); i }
            DType::I32 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![32, 1])); i }
            DType::I64 => { let i = asm.id(); entries.push((OP_TYPE_INT, i, vec![64, 1])); i }
            DType::F16 | DType::BF16 => { let i = asm.id(); entries.push((OP_TYPE_FLOAT, i, vec![16])); i }
            DType::F32 => { let i = asm.id(); entries.push((OP_TYPE_FLOAT, i, vec![32])); i }
            DType::F64 => { let i = asm.id(); entries.push((OP_TYPE_FLOAT, i, vec![64])); i }
        };
        cache.insert(dt, id);
        id
    };
    
    let push_ptr_type = |asm: &mut Asm, cache: &mut Map<(u32, u32), u32>, entries: &mut Vec<(u16, u32, Vec<u32>)>, class: u32, elem_type: u32| {
        if let Some(&id) = cache.get(&(class, elem_type)) { return id; }
        let id = asm.id();
        entries.push((OP_TYPE_POINTER, id, vec![class, elem_type]));
        cache.insert((class, elem_type), id);
        id
    };

    // === SPIR-V state ===
    let mut type_cache: Map<DType, u32> = Map::with_capacity_and_hasher(16, BuildHasherDefault::new());
    let mut type_entries: Vec<(u16, u32, Vec<u32>)> = Vec::with_capacity(32);
    let mut const_entries: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(16);
    let mut len_const_ids: std::collections::HashSet<u32> = std::collections::HashSet::new(); // constant IDs used as array lengths
    let mut spv_values: Map<OpId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
    let mut var_entries: Vec<(u32, u32, u32, bool)> = Vec::with_capacity(16);
    let mut global_var_ids: Vec<u32> = Vec::with_capacity(4);
    let mut decorations: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(8);
    let mut binding: u32 = 0;
    let mut spv_variables: Map<OpId, u32> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());
    let mut ptr_cache: Map<(u32, u32), u32> = Map::with_capacity_and_hasher(16, BuildHasherDefault::new());
    let mut reg_arrays: Map<OpId, (u32, u32)> = Map::with_capacity_and_hasher(8, BuildHasherDefault::new());
    let mut cast_u32_consts: Map<OpId, (u32, u32)> = Map::with_capacity_and_hasher(4, BuildHasherDefault::new());
    let mut const_pool: Map<(DType, u32), u32> = Map::with_capacity_and_hasher(16, BuildHasherDefault::new());
    
    // Pre-define common types
    type_entries.push((OP_TYPE_VOID, void_id, vec![]));
    let u32_id = push_dtype(&mut asm, &mut type_cache, &mut type_entries, DType::U32);
    let vec3_id = asm.id();
    type_entries.push((OP_TYPE_VECTOR, vec3_id, vec![u32_id, 3]));
    
    // GLSL extension set
    let glsl_set = 1; // GLSL extension set number

    // === Pass 1: scan for work sizes, collect info ===
    let mut gws: Vec<u64> = vec![1; 3];
    let mut lws: Vec<u64> = vec![1; 3];
    {
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            match kernel.at(op_id) {
                Op::Const(c) => {
                    let dt = c.dtype();
                    let st = push_dtype(&mut asm, &mut type_cache, &mut type_entries, dt);
                    let words = const_to_words(c);
                    let cid = asm.id();
                    const_entries.push((st, cid, words));
                    spv_values.insert(op_id, cid);
                }
                &Op::Define { dtype, scope, ro, len } => {
                    let st = push_dtype(&mut asm, &mut type_cache, &mut type_entries, dtype);
                    match scope {
                        Scope::Global => {
                            let arr = asm.id();
                            type_entries.push((OP_TYPE_RUNTIME_ARRAY, arr, vec![st]));
                            let ptr = asm.id();
                            type_entries.push((OP_TYPE_POINTER, ptr, vec![SC_STORAGE_BUFFER, arr]));
                            let var = asm.id();
                            var_entries.push((ptr, var, SC_STORAGE_BUFFER, true));
                            global_var_ids.push(var);
                            decorations.push((var, DEC_DESCRIPTOR_SET, vec![0]));
                            decorations.push((var, DEC_BINDING, vec![binding]));
                            if ro { decorations.push((var, DEC_NON_WRITABLE, vec![])); }
                            binding += 1;
                            spv_variables.insert(op_id, var);
                            // Pre-cache element pointer type
                            push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_STORAGE_BUFFER, st);
                        }
                        Scope::Local => {
                            let len_cid = asm.id();
                            const_entries.push((u32_id, len_cid, vec![len as u32]));
                            len_const_ids.insert(len_cid);
                            let arr = asm.id();
                            type_entries.push((OP_TYPE_ARRAY, arr, vec![st, len_cid]));
                            let ptr = asm.id();
                            type_entries.push((OP_TYPE_POINTER, ptr, vec![SC_WORKGROUP, arr]));
                            let var = asm.id();
                            var_entries.push((ptr, var, SC_WORKGROUP, false));
                            spv_variables.insert(op_id, var);
                            push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_WORKGROUP, st);
                        }
                        Scope::Register => {
                            let len_cid = asm.id();
                            const_entries.push((u32_id, len_cid, vec![len as u32]));
                            len_const_ids.insert(len_cid);
                            let arr = asm.id();
                            type_entries.push((OP_TYPE_ARRAY, arr, vec![st, len_cid]));
                            let ptr = asm.id();
                            type_entries.push((OP_TYPE_POINTER, ptr, vec![SC_FUNCTION, arr]));
                            reg_arrays.insert(op_id, (ptr, st));
                            push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, st);
                        }
                    };
                }
                &Op::Cast { x, dtype } => {
                    let src = dtypes[&x];
                    let dst = dtype;
                    if src == DType::Bool && dst.is_float() {
                        let u32_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, DType::U32);
                        let one_cid = asm.id();
                        let zero_cid = asm.id();
                        const_entries.push((u32_type, one_cid, vec![1u32]));
                        const_entries.push((u32_type, zero_cid, vec![0u32]));
                        cast_u32_consts.insert(op_id, (one_cid, zero_cid));
                    } else if src == DType::Bool && (dst == DType::U32 || dst == DType::I32) {
                        let int_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, dst);
                        let one_cid = asm.id();
                        let zero_cid = asm.id();
                        const_entries.push((int_type, one_cid, vec![1u32]));
                        const_entries.push((int_type, zero_cid, vec![0u32]));
                        cast_u32_consts.insert(op_id, (one_cid, zero_cid));
                    } else if (src == DType::U32 || src == DType::I32) && dst == DType::Bool {
                        let src_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, src);
                        let zero_cid = asm.id();
                        const_entries.push((src_type, zero_cid, vec![0u32]));
                        cast_u32_consts.insert(op_id, (zero_cid, 0));
                    }
                }
                _ => {}
            }
            // Pre-allocate constants for Recip, Abs, and Loop
            match kernel.at(op_id) {
                &Op::Unary { uop, x } if uop == UOp::Reciprocal => {
                    let dt = dtypes[&x];
                    let val = float_one(dt);
                    if !const_pool.contains_key(&(dt, val)) {
                        let tid = type_cache[&dt];
                        let cid = asm.id();
                        const_entries.push((tid, cid, vec![val]));
                        const_pool.insert((dt, val), cid);
                    }
                }
                &Op::Unary { uop, x } if uop == UOp::Abs && dtypes[&x].is_int() && !dtypes[&x].is_uint() => {
                    let dt = dtypes[&x];
                    let tid = type_cache[&dt];
                    if !const_pool.contains_key(&(dt, 0u32)) {
                        let cid = asm.id();
                        const_entries.push((tid, cid, vec![0u32]));
                        const_pool.insert((dt, 0u32), cid);
                    }
                }
                &Op::Loop { len } => {
                    for &val in &[0u32, 1, len as u32] {
                        if !const_pool.contains_key(&(IDX_T, val)) {
                            let tid = type_cache[&IDX_T];
                            let cid = asm.id();
                            const_entries.push((tid, cid, vec![val]));
                            const_pool.insert((IDX_T, val), cid);
                        }
                    }
                }
                &Op::Barrier { .. } => {
                    for &val in &[SCOPE_WORKGROUP, SEM_ACQUIRE_RELEASE | SEM_WORKGROUP_MEMORY] {
                        if !const_pool.contains_key(&(DType::U32, val)) {
                            let tid = type_cache[&DType::U32];
                            let cid = asm.id();
                            const_entries.push((tid, cid, vec![val]));
                            const_pool.insert((DType::U32, val), cid);
                        }
                    }
                }
                _ => {}
            }
            // Track work sizes from Index ops
            if let &Op::Index { len, scope, axis } = kernel.at(op_id) {
                match scope {
                    Scope::Global if axis < 3 => gws[axis as usize] = gws[axis as usize].max(len),
                    Scope::Local if axis < 3 => lws[axis as usize] = lws[axis as usize].max(len),
                    _ => {}
                }
            }
            op_id = kernel.next_op(op_id);
        }
    }

    // Pre-populate all dtypes (kernel dtypes + internal ones like Bool for loop conditions)
    push_dtype(&mut asm, &mut type_cache, &mut type_entries, DType::Bool);
    for &dt in dtypes.values() {
        push_dtype(&mut asm, &mut type_cache, &mut type_entries, dt);
    }
    // Pre-populate pointer types needed during body processing
    let idx_type = type_cache[&IDX_T];
    push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, idx_type);

    // === Builtin variables for Index ops ===
    let needs_global = {
        let mut op_id = kernel.head;
        let mut found = false;
        while !op_id.is_null() {
            if let &Op::Index { scope, .. } = kernel.at(op_id) {
                if matches!(scope, Scope::Global | Scope::Local) { found = true; break; }
            }
            op_id = kernel.next_op(op_id);
        }
        found
    };

    let (wg_id_var, local_inv_var) = if needs_global {
        let wiv = asm.id();
        let liv = asm.id();
        let inp_ptr = asm.id();
        type_entries.push((OP_TYPE_POINTER, inp_ptr, vec![SC_INPUT, vec3_id]));
        var_entries.push((inp_ptr, wiv, SC_INPUT, true));
        var_entries.push((inp_ptr, liv, SC_INPUT, true));
        decorations.push((wiv, DEC_BUILT_IN, vec![BI_WORKGROUP_ID]));
        decorations.push((liv, DEC_BUILT_IN, vec![BI_LOCAL_INVOCATION_ID]));
        global_var_ids.push(wiv);
        global_var_ids.push(liv);
        (wiv, liv)
    } else {
        (0, 0)
    };

    // === Phase 2: emit in SPIR-V binary order ===

    // Allocate function ID
    let func_id = asm.id();

    // Entry point: GLCompute %func_id "name" %interfaces...
    let ep_name = format!(
        "k_{}__{}",
        gws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
        lws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
    );
    {
        let mut ep_words = vec![EXEC_GL_COMPUTE, func_id];
        let name_bytes: Vec<u8> = ep_name.bytes().chain(std::iter::once(0)).collect();
        for chunk in name_bytes.chunks(4) {
            let mut w = 0u32;
            for (i, &b) in chunk.iter().enumerate() { w |= (b as u32) << (i * 8); }
            ep_words.push(w);
        }
        ep_words.extend_from_slice(&global_var_ids);
        let wc = 1 + ep_words.len() as u16;
        asm.words.push((wc as u32) << 16 | OP_ENTRY_POINT as u32);
        asm.words.extend_from_slice(&ep_words);
    }

    // Execution mode
    asm.emit(OP_EXECUTION_MODE, &[func_id, MODE_LOCAL_SIZE, lws[0] as u32, lws[1] as u32, lws[2] as u32]);

    // Annotations
    for (var_id, dec, operands) in &decorations {
        let mut args = vec![*var_id, *dec];
        args.extend_from_slice(operands);
        asm.emit(OP_DECORATE, &args);
    }

    // Types (emit array-length constants inline before OpTypeArray that references them)
    let mut emitted_consts: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (op, id, operands) in &type_entries {
        if *op == OP_TYPE_ARRAY {
            // OpTypeArray [result_id, element_type, length_const_id]
            if let Some(&len_cid) = operands.get(1) {
                if emitted_consts.insert(len_cid) {
                    // Find and emit the constant before the array type that references it
                    for &(ct, cr, ref cw) in &const_entries {
                        if cr == len_cid {
                            asm.emit_typed(OP_CONSTANT, ct, cr, cw);
                            break;
                        }
                    }
                }
            }
        }
        asm.emit_type(*op, *id, operands);
    }

    // Remaining constants (not yet emitted inline)
    for &(type_id, result_id, ref words) in &const_entries {
        if emitted_consts.insert(result_id) {
            asm.emit_typed(OP_CONSTANT, type_id, result_id, words);
        }
    }

    // Global variables (non-Function storage class)
    for &(ptr_type_id, var_id, storage, _is_global) in &var_entries {
        asm.emit(OP_VARIABLE, &[ptr_type_id, var_id, storage]);
    }

    // Function type: void()
    let func_type_id = asm.id();
    asm.emit_type(OP_TYPE_FUNCTION, func_type_id, &[void_id]);

    // Function definition
    asm.emit_typed(OP_FUNCTION, void_id, func_id, &[FN_CTRL_NONE, func_type_id]);

    // Entry block label
    let entry_label = asm.id();
    asm.emit(OP_LABEL, &[entry_label]);

    // Function-scope variables (register arrays)
    let mut reg_vars: Map<OpId, u32> = Map::with_capacity_and_hasher(8, BuildHasherDefault::new());
    for (&op_id, &(ptr_type, _elem_type)) in reg_arrays.iter() {
        let var_id = asm.id();
        asm.emit(OP_VARIABLE, &[ptr_type, var_id, SC_FUNCTION]);
        reg_vars.insert(op_id, var_id);
    }

    // === Function body: walk kernel ops ===
    
    // Loop stack: (header_label, merge_label, continue_label, counter_var, len)
    let mut loop_stack: Vec<(u32, u32, u32, u32, u64)> = Vec::new();
    let mut if_stack: Vec<u32> = Vec::new(); // merge_label

    {
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            
            match kernel.at(op_id) {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Move { .. } | Op::Reduce { .. } | Op::Wmma { .. } | Op::Vectorize { .. } | Op::Devectorize { .. } => {
                    return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "SPIR-V: unexpected kernel op (should be unfolded)".into() });
                }

                Op::Const(_) => {
                    // Already emitted in pass 1
                }

                &Op::Define { scope, .. } => {
                    match scope {
                        Scope::Global | Scope::Local => {
                            // Already declared as module-level variable
                        }
                        Scope::Register => {
                            // Variable was emitted at function entry
                        }
                    }
                }

                &Op::Load { src, index, layout: _ } => {
                    let result_type = emit_type(&mut asm, &mut type_cache, dtypes[&op_id]);
                    let index_id = spv_values[&index];

                    let (base_ptr, element_ptr_type) = if let Some(&var_id) = spv_variables.get(&src) {
                        let sc = if let &Op::Define { scope: Scope::Local, .. } = kernel.at(src) { SC_WORKGROUP } else { SC_STORAGE_BUFFER };
                        let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, sc, result_type);
                        (var_id, elem_ptr)
                    } else if let Some(&var_id) = reg_vars.get(&src) {
                        // Register array
                        let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, result_type);
                        (var_id, elem_ptr)
                    } else {
                        return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "SPIR-V: Load from unknown variable".into() });
                    };

                    let access = asm.id();
                    asm.emit_typed(OP_ACCESS_CHAIN, element_ptr_type, access, &[base_ptr, index_id]);
                    let loaded = asm.id();
                    asm.emit_typed(OP_LOAD, result_type, loaded, &[access]);
                    spv_values.insert(op_id, loaded);
                }

                &Op::Store { dst, x, index, layout: _ } => {
                    let val_type = emit_type(&mut asm, &mut type_cache, dtypes[&x]);
                    let val_id = spv_values[&x];
                    let index_id = spv_values[&index];

                    let (base_ptr, element_ptr_type) = if let Some(&var_id) = spv_variables.get(&dst) {
                        let sc = if let &Op::Define { scope: Scope::Local, .. } = kernel.at(dst) { SC_WORKGROUP } else { SC_STORAGE_BUFFER };
                        let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, sc, val_type);
                        (var_id, elem_ptr)
                    } else if let Some(&var_id) = reg_vars.get(&dst) {
                        let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, val_type);
                        (var_id, elem_ptr)
                    } else {
                        return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "SPIR-V: Store to unknown variable".into() });
                    };

                    let access = asm.id();
                    asm.emit_typed(OP_ACCESS_CHAIN, element_ptr_type, access, &[base_ptr, index_id]);
                    asm.emit(OP_STORE, &[access, val_id]);
                }

                &Op::Cast { x, dtype } => {
                    let src_type = dtypes[&x];
                    let src_id = spv_values[&x];
                    let dst_type = dtype;
                    let result_type = emit_type(&mut asm, &mut type_cache, dst_type);

                    let rid = asm.id();
                    if src_type == DType::Bool && dst_type.is_float() {
                        let u32_type = emit_type(&mut asm, &mut type_cache, DType::U32);
                        let (u32_one, u32_zero) = cast_u32_consts[&op_id];
                        let int_tmp = asm.id();
                        asm.emit_typed(OP_SELECT, u32_type, int_tmp, &[src_id, u32_one, u32_zero]);
                        asm.emit_typed(OP_CONVERT_U_TO_F, result_type, rid, &[int_tmp]);
                    } else if src_type == DType::Bool && (dst_type == DType::U32 || dst_type == DType::I32) {
                        let (one_cid, zero_cid) = cast_u32_consts[&op_id];
                        asm.emit_typed(OP_SELECT, result_type, rid, &[src_id, one_cid, zero_cid]);
                    } else if dst_type == DType::Bool && (src_type == DType::U32 || src_type == DType::I32) {
                        let (zero_cid, _) = cast_u32_consts[&op_id];
                        asm.emit_typed(OP_I_NOT_EQUAL, result_type, rid, &[src_id, zero_cid]);
                    } else {
                        let op = cast_op(src_type, dst_type);
                        asm.emit_typed(op, result_type, rid, &[src_id]);
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Unary { x, uop } => {
                    let src_id = spv_values[&x];
                    let dt = dtypes[&x];
                    let result_type = emit_type(&mut asm, &mut type_cache, dt);
                    let rid = asm.id();

                    match uop {
                        UOp::Neg => {
                            if dt.is_float() {
                                asm.emit_typed(OP_FNEGATE, result_type, rid, &[src_id]);
                            } else {
                                asm.emit_typed(OP_SNEGATE, result_type, rid, &[src_id]);
                            }
                        }
                        UOp::BitNot => {
                            asm.emit_typed(OP_NOT, result_type, rid, &[src_id]);
                        }
                        UOp::Exp | UOp::Exp2 => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Exp2, src_id]);
                        }
                        UOp::Ln | UOp::Log2 => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Log2, src_id]);
                        }
                        UOp::Reciprocal => {
                            let one = const_pool[&(dt, float_one(dt))];
                            asm.emit_typed(OP_FDIV, result_type, rid, &[one, src_id]);
                        }
                        UOp::Sqrt => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Sqrt, src_id]);
                        }
                        UOp::Sin => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Sin, src_id]);
                        }
                        UOp::Cos => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Cos, src_id]);
                        }
                        UOp::Floor => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Floor, src_id]);
                        }
                        UOp::Trunc => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Trunc, src_id]);
                        }
                        UOp::Abs => {
                            if dt.is_float() {
                                asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::FAbs, src_id]);
                            } else if dt.is_uint() {
                                spv_values.insert(op_id, src_id);
                                continue;
                            } else {
                                // Signed int abs: (x < 0) ? -x : x
                                let zero = const_pool[&(dt, 0u32)];
                                let bool_type = emit_type(&mut asm, &mut type_cache, DType::Bool);
                                let cmp = asm.id();
                                asm.emit_typed(OP_S_LESS_THAN, bool_type, cmp, &[src_id, zero]);
                                let neg = asm.id();
                                asm.emit_typed(OP_SNEGATE, result_type, neg, &[src_id]);
                                asm.emit_typed(OP_SELECT, result_type, rid, &[cmp, neg, src_id]);
                            }
                        }
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Binary { x, y, bop } => {
                    let x_id = spv_values[&x];
                    let y_id = spv_values[&y];
                    let dt = dtypes[&x];
                    let result_type = emit_type(&mut asm, &mut type_cache, dtypes[&op_id]);
                    let rid = asm.id();

                    let (float_op, int_op, _): (Option<u16>, Option<u16>, Option<u16>) = match bop {
                        BOp::Add => (Some(OP_FADD), Some(OP_IADD), None),
                        BOp::Sub => (Some(OP_FSUB), Some(OP_ISUB), None),
                        BOp::Mul => (Some(OP_FMUL), Some(OP_IMUL), None),
                        BOp::Div => (Some(OP_FDIV), if dt.is_float() { None } else if dt.is_int() { Some(OP_SDIV) } else { Some(OP_UDIV) }, None),
                        BOp::Pow => {
                            asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::Pow, x_id, y_id]);
                            (None, None, None)
                        }
                        BOp::Mod => (Some(OP_FMOD), Some(OP_SREM), None), // SPIR-V uses SRem for C-style %
                        BOp::Cmplt => (Some(OP_F_ORD_LESS_THAN), if dt.is_float() { None } else if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) { Some(OP_S_LESS_THAN) } else { Some(OP_U_LESS_THAN) }, None),
                        BOp::Cmpgt => (Some(OP_F_ORD_GREATER_THAN), if dt.is_float() { None } else if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) { Some(OP_S_GREATER_THAN) } else { Some(OP_U_GREATER_THAN) }, None),
                        BOp::Max => {
                            if dt.is_float() {
                                asm.emit_typed(OP_EXT_INST, result_type, rid, &[glsl_set, glsl::FMax, x_id, y_id]);
                            } else {
                                let cmp = asm.id();
                                let cmp_type = emit_type(&mut asm, &mut type_cache, DType::Bool);
                                if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
                                    asm.emit_typed(OP_S_GREATER_THAN, cmp_type, cmp, &[x_id, y_id]);
                                } else {
                                    asm.emit_typed(OP_U_GREATER_THAN, cmp_type, cmp, &[x_id, y_id]);
                                }
                                asm.emit_typed(OP_SELECT, result_type, rid, &[cmp, x_id, y_id]);
                            }
                            (None, None, None)
                        }
                        BOp::Or => (None, None, Some(OP_NOT)), // handled differently
                        BOp::And => (None, None, Some(OP_NOT)),
                        BOp::BitXor => (None, Some(OP_BITWISE_XOR), None),
                        BOp::BitOr => (None, Some(OP_BITWISE_OR), None),
                        BOp::BitAnd => (None, Some(OP_BITWISE_AND), None),
                        BOp::BitShiftLeft => (None, Some(OP_SHIFT_LEFT_LOGICAL), None),
                        BOp::BitShiftRight => (None, Some(OP_SHIFT_RIGHT_LOGICAL), None),
                        BOp::NotEq => (Some(OP_F_ORD_NOT_EQUAL), Some(OP_I_NOT_EQUAL), None),
                        BOp::Eq => (Some(OP_F_ORD_EQUAL), Some(OP_I_EQUAL), None),
                    };

                    if dt == DType::Bool {
                        if matches!(bop, BOp::Or) {
                            asm.emit_typed(OP_BITWISE_OR, result_type, rid, &[x_id, y_id]);
                        } else if matches!(bop, BOp::And) {
                            asm.emit_typed(OP_BITWISE_AND, result_type, rid, &[x_id, y_id]);
                        }
                    } else if dt.is_float() {
                        if let Some(op) = float_op {
                            asm.emit_typed(op, result_type, rid, &[x_id, y_id]);
                        }
                    } else if let Some(op) = int_op {
                        asm.emit_typed(op, result_type, rid, &[x_id, y_id]);
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Mad { x, y, z } => {
                    let x_id = spv_values[&x];
                    let y_id = spv_values[&y];
                    let z_id = spv_values[&z];
                    let dt = dtypes[&x];
                    let result_type = emit_type(&mut asm, &mut type_cache, dt);
                    let rid = asm.id();

                    // FMad not available in spirv crate, decompose to FMul + FAdd
                    if dt.is_float() {
                        let mul = asm.id();
                        asm.emit_typed(OP_FMUL, result_type, mul, &[x_id, y_id]);
                        asm.emit_typed(OP_FADD, result_type, rid, &[mul, z_id]);
                    } else {
                        let mul = asm.id();
                        asm.emit_typed(OP_IMUL, result_type, mul, &[x_id, y_id]);
                        asm.emit_typed(OP_IADD, result_type, rid, &[mul, z_id]);
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Index { len: _, scope, axis } => {
                    let result_type = emit_type(&mut asm, &mut type_cache, IDX_T);
                    match scope {
                        Scope::Global => {
                            let loaded = asm.id();
                            asm.emit_typed(OP_LOAD, vec3_id, loaded, &[wg_id_var]);
                            let elem = asm.id();
                            asm.emit_typed(OP_COMPOSITE_EXTRACT, u32_id, elem, &[loaded, axis]);
                            if IDX_T == DType::U32 {
                                spv_values.insert(op_id, elem);
                            } else {
                                let widened = asm.id();
                                let op = match IDX_T {
                                    DType::U64 => OP_U_CONVERT,
                                    _ => unreachable!(),
                                };
                                asm.emit_typed(op, result_type, widened, &[elem]);
                                spv_values.insert(op_id, widened);
                            }
                        }
                        Scope::Local => {
                            let loaded = asm.id();
                            asm.emit_typed(OP_LOAD, vec3_id, loaded, &[local_inv_var]);
                            let elem = asm.id();
                            asm.emit_typed(OP_COMPOSITE_EXTRACT, u32_id, elem, &[loaded, axis]);
                            if IDX_T == DType::U32 {
                                spv_values.insert(op_id, elem);
                            } else {
                                let widened = asm.id();
                                let op = match IDX_T {
                                    DType::U64 => OP_U_CONVERT,
                                    _ => unreachable!(),
                                };
                                asm.emit_typed(op, result_type, widened, &[elem]);
                                spv_values.insert(op_id, widened);
                            }
                        }
                        Scope::Register => {
                            // Should not happen as register indices come from loops
                        }
                    }
                }

                &Op::Loop { len } => {
                    let header = asm.id();
                    let body = asm.id();
                    let continue_lbl = asm.id();
                    let merge = asm.id();
                    let idx_type = emit_type(&mut asm, &mut type_cache, IDX_T);

                    // Pre-header: allocate counter var and store 0, then branch to header
                    let counter_ptr_type = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, idx_type);
                    let counter_var = asm.id();
                    asm.emit(OP_VARIABLE, &[counter_ptr_type, counter_var, SC_FUNCTION]);
                    let zero = const_pool[&(IDX_T, 0u32)];
                    asm.emit(OP_STORE, &[counter_var, zero]);
                    asm.emit(OP_BRANCH, &[header]);

                    // Header (loop continue target)
                    asm.emit(OP_LABEL, &[header]);
                    asm.emit(OP_LOOP_MERGE, &[merge, continue_lbl, LOOP_CTRL_NONE]);
                    asm.emit(OP_BRANCH, &[body]);

                    // Body block
                    asm.emit(OP_LABEL, &[body]);

                    // Load current counter value (this is the Loop op's SSA value)
                    let counter_val = asm.id();
                    asm.emit_typed(OP_LOAD, idx_type, counter_val, &[counter_var]);
                    spv_values.insert(op_id, counter_val);

                    loop_stack.push((header, merge, continue_lbl, counter_var, len));
                }

                Op::EndLoop => {
                    let (header, merge, continue_lbl, counter_var, len) = loop_stack.pop().unwrap();
                    let idx_type = emit_type(&mut asm, &mut type_cache, IDX_T);

                    // Branch to continue block
                    asm.emit(OP_BRANCH, &[continue_lbl]);

                    // Continue block: load, increment, store, check
                    asm.emit(OP_LABEL, &[continue_lbl]);
                    let old = asm.id();
                    asm.emit_typed(OP_LOAD, idx_type, old, &[counter_var]);
                    let one = const_pool[&(IDX_T, 1u32)];
                    let inc = asm.id();
                    asm.emit_typed(OP_IADD, idx_type, inc, &[old, one]);
                    asm.emit(OP_STORE, &[counter_var, inc]);

                    // Check if counter < len
                    let len_cid = const_pool[&(IDX_T, len as u32)];
                    let cmp_type = emit_type(&mut asm, &mut type_cache, DType::Bool);
                    let cmp = asm.id();
                    asm.emit_typed(OP_U_LESS_THAN, cmp_type, cmp, &[inc, len_cid]);
                    asm.emit(OP_BRANCH_CONDITIONAL, &[cmp, header, merge]);

                    // Merge block
                    asm.emit(OP_LABEL, &[merge]);
                }

                &Op::If { condition } => {
                    let cond_id = spv_values[&condition];
                    let true_block = asm.id();
                    let merge = asm.id();

                    asm.emit(OP_SELECTION_MERGE, &[merge, SELECT_CTRL_NONE]);
                    asm.emit(OP_BRANCH_CONDITIONAL, &[cond_id, true_block, merge]);

                    // True block
                    asm.emit(OP_LABEL, &[true_block]);
                    if_stack.push(merge);
                }

                Op::EndIf => {
                    let merge = if_stack.pop().unwrap();
                    asm.emit(OP_BRANCH, &[merge]);
                    asm.emit(OP_LABEL, &[merge]);
                }

                &Op::Barrier { scope } => {
                    match scope {
                        Scope::Local => {
                            let scope_id = const_pool[&(DType::U32, SCOPE_WORKGROUP)];
                            let sem_id = const_pool[&(DType::U32, SEM_ACQUIRE_RELEASE | SEM_WORKGROUP_MEMORY)];
                            asm.emit(OP_CONTROL_BARRIER, &[scope_id, scope_id, sem_id]);
                        }
                        _ => {
                            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "SPIR-V: unsupported barrier scope".into() });
                        }
                    }
                }
            }
            op_id = kernel.next_op(op_id);
        }
    }

    // Return and end function
    asm.emit(OP_RETURN, &[]);
    asm.emit(OP_FUNCTION_END, &[]);

    // Set bound
    asm.set_bound();
    
    if debug_asm {
        println!("SPIR-V: {} words, {} ops", asm.words.len(), kernel.ops.len().0);
        println!("SPIR-V: first 10 words: {:?}", &asm.words[..10.min(asm.words.len())]);
        println!("SPIR-V: full assembly:");
        debug_print(&asm.words);
    }

    Ok(asm.words)
}

fn cast_op(src: DType, dst: DType) -> u16 {
    use DType::*;
    match (src, dst) {
        (BF16, F32) | (F16, F32) | (F32, F64) | (F16, F64) | (BF16, F64) | (F32, F16) | (F64, F16) | (F64, F32) => OP_F_CONVERT,
        (I8, I32) | (I16, I32) | (I32, I64) | (I8, I64) | (I16, I64) | (I32, I8) | (I64, I8) | (I32, I16) | (I64, I16) | (I64, I32) => OP_S_CONVERT,
        (U8, U32) | (U16, U32) | (U32, U64) | (U8, U64) | (U16, U64) | (U32, U8) | (U64, U8) | (U32, U16) | (U64, U16) | (U64, U32) => OP_U_CONVERT,
        (F32, I32) | (F64, I32) | (F32, I64) | (F64, I64) | (F16, I32) | (F16, I64) => OP_CONVERT_F_TO_S,
        (F32, U32) | (F64, U32) | (F32, U64) | (F64, U64) | (F16, U32) | (F16, U64) => OP_CONVERT_F_TO_U,
        (I32, F32) | (I64, F32) | (I32, F64) | (I64, F64) | (I32, F16) | (I64, F16) => OP_CONVERT_S_TO_F,
        (U32, F32) | (U64, F32) | (U32, F64) | (U64, F64) | (U32, F16) | (U64, F16) => OP_CONVERT_U_TO_F,
        (Bool, I32) | (Bool, U32) | (I32, Bool) | (U32, Bool) => OP_BITCAST,
        _ => {
            // Fallback: if same bit width, use bitcast
            if bit_size(src) == bit_size(dst) {
                OP_BITCAST
            } else {
                // Try via FConvert/SConvert/UConvert
                if dst.is_float() { OP_F_CONVERT }
                else if dst.is_int() { OP_S_CONVERT }
                else { OP_U_CONVERT }
            }
        }
    }
}

fn const_to_words(c: &Constant) -> Vec<u32> {
    match *c {
        Constant::U8(x) => vec![x as u32],
        Constant::U16(x) => vec![x as u32],
        Constant::U32(x) => vec![x],
        Constant::U64(x) => { let v = u64::from_le_bytes(x); vec![v as u32, (v >> 32) as u32] }
        Constant::I8(x) => vec![x as u32],
        Constant::I16(x) => vec![x as u32],
        Constant::I32(x) => vec![x as u32],
        Constant::I64(x) => { let v = i64::from_le_bytes(x); vec![v as u32, (v >> 32) as u32] }
        Constant::F16(x) => vec![u16::from_le_bytes(x) as u32],
        Constant::BF16(x) => vec![u16::from_le_bytes(x) as u32],
        Constant::F32(x) => vec![u32::from_le_bytes(x)],
        Constant::F64(x) => vec![u32::from_le_bytes(x[..4].try_into().unwrap()), u32::from_le_bytes(x[4..].try_into().unwrap())],
        Constant::Bool(x) => vec![x as u32],
    }
}

fn float_one(dt: DType) -> u32 {
    match dt {
        DType::F16 | DType::BF16 => 0x3C00,
        DType::F32 => 0x3F80_0000,
        DType::F64 => 0x0000_0000, // Lower word; F64 not yet supported
        _ => 1,
    }
}

fn bit_size(dt: DType) -> u32 {
    match dt {
        DType::Bool => 8,
        DType::I8 | DType::U8 => 8,
        DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 16,
        DType::I32 | DType::U32 | DType::F32 => 32,
        DType::I64 | DType::U64 | DType::F64 => 64,
    }
}

// ---------- Debug disassembly ----------

fn opcode_name(op: u16) -> &'static str {
    match op {
        17 => "OpCapability",
        11 => "OpExtInstImport",
        14 => "OpMemoryModel",
        15 => "OpEntryPoint",
        16 => "OpExecutionMode",
        71 => "OpDecorate",
        19 => "OpTypeVoid",
        20 => "OpTypeBool",
        21 => "OpTypeInt",
        22 => "OpTypeFloat",
        23 => "OpTypeVector",
        28 => "OpTypeArray",
        29 => "OpTypeRuntimeArray",
        32 => "OpTypePointer",
        33 => "OpTypeFunction",
        43 => "OpConstant",
        42 => "OpConstantFalse",
        41 => "OpConstantTrue",
        59 => "OpVariable",
        54 => "OpFunction",
        56 => "OpFunctionEnd",
         248 => "OpLabel",
         249 => "OpBranch",
         250 => "OpBranchConditional",
         246 => "OpLoopMerge",
         247 => "OpSelectionMerge",
         253 => "OpReturn",
         61 => "OpLoad",
         62 => "OpStore",
         65 => "OpAccessChain",
         129 => "OpFAdd",
         131 => "OpFSub",
         133 => "OpFMul",
         136 => "OpFDiv",
         127 => "OpFNegate",
         141 => "OpFMod",
         128 => "OpIAdd",
         130 => "OpISub",
         132 => "OpIMul",
         135 => "OpSDiv",
         134 => "OpUDiv",
         138 => "OpSRem",
         139 => "OpSMod",
         137 => "OpUMod",
         126 => "OpSNegate",
         200 => "OpNot",
         196 => "OpShiftLeftLogical",
         194 => "OpShiftRightLogical",
         199 => "OpBitwiseAnd",
         197 => "OpBitwiseOr",
         198 => "OpBitwiseXor",
         170 => "OpIEqual",
         171 => "OpINotEqual",
         176 => "OpULessThan",
         172 => "OpUGreaterThan",
         177 => "OpSLessThan",
         173 => "OpSGreaterThan",
         184 => "OpFOrdLessThan",
         186 => "OpFOrdGreaterThan",
         180 => "OpFOrdEqual",
         182 => "OpFOrdNotEqual",
         109 => "OpConvertFToU",
         110 => "OpConvertFToS",
         111 => "OpConvertSToF",
         112 => "OpConvertUToF",
         115 => "OpFConvert",
         114 => "OpSConvert",
         113 => "OpUConvert",
         169 => "OpSelect",
        12 => "OpExtInst",
        224 => "OpControlBarrier",
        81 => "OpCompositeExtract",
        124 => "OpBitcast",
        _ => "??",
    }
}

fn storage_class_name(sc: u32) -> &'static str {
    match sc {
        0 => "UniformConstant",
        1 => "Input",
        2 => "Uniform",
        3 => "Output",
        4 => "Workgroup",
        5 => "CrossWorkgroup",
        6 => "Private",
        7 => "Function",
        8 => "Generic",
        9 => "PushConstant",
        12 => "StorageBuffer",
        _ => "??",
    }
}

fn decoration_name(d: u32) -> &'static str {
    match d {
        11 => "BuiltIn",
        33 => "Binding",
        34 => "DescriptorSet",
        24 => "NonWritable",
        _ => "??",
    }
}

fn builtin_name(b: u32) -> &'static str {
    match b {
        28 => "GlobalInvocationId",
        27 => "LocalInvocationId",
        24 => "NumWorkgroups",
        26 => "WorkgroupId",
        _ => "??",
    }
}

fn capability_name(c: u32) -> &'static str {
    match c {
        1 => "Shader",
        9 => "Float16",
        12 => "Int64",
        _ => "??",
    }
}

pub fn debug_print(spv: &[u32]) {
    if spv.len() < 5 { return; }
    let bound = spv[3];
    println!("; SPIR-V disassembly (bound={bound})");
    println!("; {} words", spv.len());

    let mut i = 5; // skip header
    while i < spv.len() {
        let w = spv[i];
        let word_count = (w >> 16) as u16;
        let op = (w & 0xffff) as u16;
        if word_count == 0 { break; }

        let name = opcode_name(op);
        print!("  {name}");

        let operands = if i + word_count as usize <= spv.len() {
            &spv[i+1..i + word_count as usize]
        } else {
            // Malformed SPIR-V, slice to end
            &spv[i+1..]
        };

        // Format known instructions
        match op {
            17 => { // Capability
                if !operands.is_empty() { print!(" {}", capability_name(operands[0])); }
            }
            11 => { // ExtInstImport
                if operands.len() >= 2 {
                    print!(" %{}", operands[0]);
                    let name_bytes: Vec<u8> = operands[1..].iter().flat_map(|w| w.to_le_bytes()).collect();
                    let name_str = String::from_utf8_lossy(&name_bytes).trim_end_matches('\0').to_string();
                    print!(" \"{name_str}\"");
                }
            }
            14 => { // MemoryModel
                if operands.len() >= 2 {
                    print!(" {}", if operands[0] == 0 { "Logical" } else { "??" });
                    print!(" {}", if operands[1] == 1 { "GLSL450" } else { "??" });
                }
            }
            15 => { // EntryPoint
                if operands.len() >= 3 {
                    let model = match operands[0] { 5 => "GLCompute", _ => "??" };
                    print!(" {} %{}", model, operands[1]);
                    let name_bytes: Vec<u8> = operands[2..].iter().flat_map(|w| w.to_le_bytes()).collect();
                    let name_end = name_bytes.iter().position(|&b| b == 0).unwrap_or(name_bytes.len());
                    let name_str = String::from_utf8_lossy(&name_bytes[..name_end]);
                    print!(" \"{name_str}\"");
                    let name_word_len = (name_end + 4) / 4;
                    for &id in &operands[2 + name_word_len..] {
                        print!(" %{id}");
                    }
                }
            }
            16 => { // ExecutionMode
                if operands.len() >= 2 {
                    print!(" %{}", operands[0]);
                    let mode = match operands[1] {
                        17 => format!("LocalSize {} {} {}", operands.get(2).unwrap_or(&0), operands.get(3).unwrap_or(&0), operands.get(4).unwrap_or(&0)),
                        _ => format!("??({})", operands[1]),
                    };
                    print!(" {mode}");
                }
            }
            71 => { // Decorate
                if operands.len() >= 2 {
                    print!(" %{} {}", operands[0], decoration_name(operands[1]));
                    if operands.len() > 2 {
                        if operands[1] == 11 { // BuiltIn
                            print!(" {}", builtin_name(operands[2]));
                        } else {
                            for &v in &operands[2..] { print!(" {v}"); }
                        }
                    }
                }
            }
            21 => { // TypeInt
                if operands.len() >= 2 {
                    print!(" %{} {} {}", operands[0], operands[1], if operands[2] == 0 { "u" } else { "i" });
                }
            }
            22 => { // TypeFloat
                if operands.len() >= 1 { print!(" %{} {}", operands[0], operands[1]); }
            }
            19 | 20 => { // TypeVoid, TypeBool
                if !operands.is_empty() { print!(" %{}", operands[0]); }
            }
            23 | 28 | 29 | 33 => { // TypeVector, TypeArray, TypeRuntimeArray, TypeFunction
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                    for &v in &operands[1..] { print!(" %{v}"); }
                }
            }
            32 => { // TypePointer
                if operands.len() >= 2 {
                    print!(" %{} {} %{}", operands[0], storage_class_name(operands[1]), operands[2]);
                }
            }
            43 | 61 | 65 | 81 | 87 | 12 => { // Constant, Load, AccessChain, CompositeExtract, Select, ExtInst
                if operands.len() >= 2 {
                    print!(" %{} %{}", operands[0], operands[1]);
                    for &v in &operands[2..] { print!(" %{v}"); }
                }
            }
            59 => { // Variable
                if operands.len() >= 2 {
                    print!(" %{} %{} {}", operands[0], operands[1], storage_class_name(operands[2]));
                }
            }
            54 => { // Function
                if operands.len() >= 3 {
                    print!(" %{} %{}", operands[0], operands[1]);
                    let ctrl = match operands[2] { 0 => "None", _ => "??" };
                    print!(" {ctrl}");
                    for &v in &operands[3..] { print!(" %{v}"); }
                }
            }
            248 => { // Label
                if !operands.is_empty() { print!(" %{}", operands[0]); }
            }
            49 => { // Branch
                if !operands.is_empty() { print!(" %{}", operands[0]); }
            }
            50 => { // BranchConditional
                if operands.len() >= 3 {
                    print!(" %{} %{} %{}", operands[0], operands[1], operands[2]);
                }
            }
            246 | 247 => { // LoopMerge, SelectionMerge
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                    if operands.len() > 1 { print!(" %{}", operands[1]); }
                }
            }
            62 => { // Store
                if operands.len() >= 2 {
                    print!(" %{} %{}", operands[0], operands[1]);
                }
            }
            224 => { // ControlBarrier
                if operands.len() >= 3 {
                    print!(" {} {} {}", operands[0], operands[1], operands[2]);
                }
            }
            _ => {
                // Generic: print all operands as numbers/IDs
                for (j, &v) in operands.iter().enumerate() {
                    if j == 0 && !matches!(op, 56 | 63) { print!(" %{v}"); }
                    else { print!(" {v}"); }
                }
            }
        }
        println!();
        i += word_count as usize;
    }
}

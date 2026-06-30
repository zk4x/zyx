// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! SPIR-V binary codegen from zyx kernel IR.
//! Translates kernel IR ops to SPIR-V machine code (`Vec<u32>`).

use crate::error::{BackendError, ErrorStatus};
use crate::shape::Dim;
use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MemLayout, Op, OpId, Scope, UOp},
};
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

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoration {
    DecBlock = 2,
    DecArrayStride = 6,
    DecBuiltIn = 11,
    DecNonWritable = 24,
    DecBinding = 33,
    DecDescriptorSet = 34,
    DecOffset = 35,
}

impl TryFrom<u32> for Decoration {
    type Error = ();
    fn try_from(v: u32) -> Result<Self, ()> {
        match v {
            2 => Ok(Self::DecBlock),
            6 => Ok(Self::DecArrayStride),
            11 => Ok(Self::DecBuiltIn),
            24 => Ok(Self::DecNonWritable),
            33 => Ok(Self::DecBinding),
            34 => Ok(Self::DecDescriptorSet),
            35 => Ok(Self::DecOffset),
            _ => Err(()),
        }
    }
}

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

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    OpCapability = 17,
    OpExtInstImport = 11,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstant = 43,
    OpVariable = 59,
    OpFunction = 54,
    OpFunctionEnd = 56,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpLoopMerge = 246,
    OpSelectionMerge = 247,
    OpReturn = 253,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpFAdd = 129,
    OpFSub = 131,
    OpFMul = 133,
    OpFDiv = 136,
    OpFNegate = 127,
    OpFMod = 141,
    OpIAdd = 128,
    OpISub = 130,
    OpIMul = 132,
    OpSDiv = 135,
    OpUDiv = 134,
    OpSRem = 138,
    OpSNegate = 126,
    OpNot = 200,
    OpShiftLeftLogical = 196,
    OpShiftRightLogical = 194,
    OpBitwiseAnd = 199,
    OpBitwiseOr = 197,
    OpBitwiseXOr = 198,
    OpIEqual = 170,
    OpINotEqual = 171,
    OpULessThan = 176,
    OpUGreaterThan = 172,
    OpSLessThan = 177,
    OpSGreaterThan = 173,
    OpFOrdLessThan = 184,
    OpFOrdGreaterThan = 186,
    OpFOrdEqual = 180,
    OpFOrdNotEqual = 182,
    OpConvertFToU = 109,
    OpConvertFToS = 110,
    OpConvertSToF = 111,
    OpConvertUToF = 112,
    OpFConvert = 115,
    OpSConvert = 114,
    OpUConvert = 113,
    OpSelect = 169,
    OpExtInst = 12,
    OpControlBarrier = 224,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpBitcast = 124,
}

impl TryFrom<u16> for OpCode {
    type Error = ();
    fn try_from(v: u16) -> Result<Self, ()> {
        match v {
            17 => Ok(Self::OpCapability),
            11 => Ok(Self::OpExtInstImport),
            14 => Ok(Self::OpMemoryModel),
            15 => Ok(Self::OpEntryPoint),
            16 => Ok(Self::OpExecutionMode),
            71 => Ok(Self::OpDecorate),
            72 => Ok(Self::OpMemberDecorate),
            19 => Ok(Self::OpTypeVoid),
            20 => Ok(Self::OpTypeBool),
            21 => Ok(Self::OpTypeInt),
            22 => Ok(Self::OpTypeFloat),
            23 => Ok(Self::OpTypeVector),
            28 => Ok(Self::OpTypeArray),
            29 => Ok(Self::OpTypeRuntimeArray),
            30 => Ok(Self::OpTypeStruct),
            32 => Ok(Self::OpTypePointer),
            33 => Ok(Self::OpTypeFunction),
            43 => Ok(Self::OpConstant),
            59 => Ok(Self::OpVariable),
            54 => Ok(Self::OpFunction),
            56 => Ok(Self::OpFunctionEnd),
            248 => Ok(Self::OpLabel),
            249 => Ok(Self::OpBranch),
            250 => Ok(Self::OpBranchConditional),
            246 => Ok(Self::OpLoopMerge),
            247 => Ok(Self::OpSelectionMerge),
            253 => Ok(Self::OpReturn),
            61 => Ok(Self::OpLoad),
            62 => Ok(Self::OpStore),
            65 => Ok(Self::OpAccessChain),
            129 => Ok(Self::OpFAdd),
            131 => Ok(Self::OpFSub),
            133 => Ok(Self::OpFMul),
            136 => Ok(Self::OpFDiv),
            127 => Ok(Self::OpFNegate),
            141 => Ok(Self::OpFMod),
            128 => Ok(Self::OpIAdd),
            130 => Ok(Self::OpISub),
            132 => Ok(Self::OpIMul),
            135 => Ok(Self::OpSDiv),
            134 => Ok(Self::OpUDiv),
            138 => Ok(Self::OpSRem),
            126 => Ok(Self::OpSNegate),
            200 => Ok(Self::OpNot),
            196 => Ok(Self::OpShiftLeftLogical),
            194 => Ok(Self::OpShiftRightLogical),
            199 => Ok(Self::OpBitwiseAnd),
            197 => Ok(Self::OpBitwiseOr),
            198 => Ok(Self::OpBitwiseXOr),
            170 => Ok(Self::OpIEqual),
            171 => Ok(Self::OpINotEqual),
            176 => Ok(Self::OpULessThan),
            172 => Ok(Self::OpUGreaterThan),
            177 => Ok(Self::OpSLessThan),
            173 => Ok(Self::OpSGreaterThan),
            184 => Ok(Self::OpFOrdLessThan),
            186 => Ok(Self::OpFOrdGreaterThan),
            180 => Ok(Self::OpFOrdEqual),
            182 => Ok(Self::OpFOrdNotEqual),
            109 => Ok(Self::OpConvertFToU),
            110 => Ok(Self::OpConvertFToS),
            111 => Ok(Self::OpConvertSToF),
            112 => Ok(Self::OpConvertUToF),
            115 => Ok(Self::OpFConvert),
            114 => Ok(Self::OpSConvert),
            113 => Ok(Self::OpUConvert),
            169 => Ok(Self::OpSelect),
            12 => Ok(Self::OpExtInst),
            224 => Ok(Self::OpControlBarrier),
            79 => Ok(Self::OpCompositeConstruct),
            81 => Ok(Self::OpCompositeExtract),
            124 => Ok(Self::OpBitcast),
            _ => Err(()),
        }
    }
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
    pub const Exp: u32 = 27;
    pub const Log: u32 = 28;
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
        Self {
            words: Vec::new(),
            next_id: 1,
        }
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
    fn emit(&mut self, op: OpCode, operands: &[u32]) {
        let wc = 1u16 + operands.len() as u16;
        self.words.push((wc as u32) << 16 | op as u32);
        self.words.extend_from_slice(operands);
    }

    // Emit instruction with result type + id (type, id, opcode, operands...)
    fn emit_typed(&mut self, op: OpCode, type_id: u32, result_id: u32, operands: &[u32]) {
        let wc = 3u16 + operands.len() as u16;
        self.words.push((wc as u32) << 16 | op as u32);
        self.words.push(type_id);
        self.words.push(result_id);
        self.words.extend_from_slice(operands);
    }

    // Emit type declaration (id, opcode, operands...)
    fn emit_type(&mut self, op: OpCode, result_id: u32, operands: &[u32]) {
        let wc = 2u16 + operands.len() as u16;
        self.words.push((wc as u32) << 16 | op as u32);
        self.words.push(result_id);
        self.words.extend_from_slice(operands);
    }
}

// ---------- Type helpers ----------

fn emit_type(asm: &mut Asm, cache: &mut Map<DType, u32>, dt: DType) -> u32 {
    use OpCode::*;
    if let Some(&id) = cache.get(&dt) {
        return id;
    }
    let id = match dt {
        DType::Bool => {
            let i = asm.id();
            asm.emit_type(OpTypeBool, i, &[]);
            i
        }
        DType::U8 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[8, 0]);
            i
        }
        DType::U16 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[16, 0]);
            i
        }
        DType::U32 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[32, 0]);
            i
        }
        DType::U64 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[64, 0]);
            i
        }
        DType::I8 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[8, 1]);
            i
        }
        DType::I16 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[16, 1]);
            i
        }
        DType::I32 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[32, 1]);
            i
        }
        DType::I64 => {
            let i = asm.id();
            asm.emit_type(OpTypeInt, i, &[64, 1]);
            i
        }
        DType::F16 | DType::BF16 => {
            let i = asm.id();
            asm.emit_type(OpTypeFloat, i, &[16]);
            i
        }
        DType::F32 => {
            let i = asm.id();
            asm.emit_type(OpTypeFloat, i, &[32]);
            i
        }
        DType::F64 => {
            let i = asm.id();
            asm.emit_type(OpTypeFloat, i, &[64]);
            i
        }
    };
    cache.insert(dt, id);
    id
}

// ---------- Compute dtypes for all ops ----------

fn compute_dtypes(kernel: &Kernel) -> Map<OpId, (DType, MemLayout)> {
    let (dt, _) = kernel.compute_dtypes_and_rcs();
    dt
}

// ---------- Public compile function ----------

fn elem_stride(dt: DType) -> usize {
    match dt {
        DType::Bool | DType::I8 | DType::U8 => 1,
        DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 => 8,
    }
}

pub fn compile(kernel: &Kernel, debug_asm: bool) -> Result<(Vec<u32>, Vec<Dim>, Vec<Dim>), BackendError> {
    use OpCode::*;
    let dtypes = compute_dtypes(kernel);
    let mut asm = Asm::new();

    // === SPIR-V Header ===
    asm.words.push(MAGIC);
    asm.words.push(VERSION);
    asm.words.push(GENERATOR);
    asm.words.push(0); // Bound will be set later
    asm.words.push(SCHEMA);

    // Pre-scan: does this kernel have global bool buffers?
    let needs_u8 = {
        let mut op_id = kernel.head;
        let mut found = false;
        while !op_id.is_null() {
            if let Op::Define {
                dtype: DType::Bool,
                scope: Scope::Global,
                ..
            } = kernel.at(op_id)
            {
                found = true;
                break;
            }
            op_id = kernel.next_op(op_id);
        }
        found
    };

    // Required SPIR-V instructions
    asm.emit(OpCapability, &[1]); // Shader capability
    if needs_u8 {
        asm.emit(OpCapability, &[44]); // StorageUniform8BitAccess (for bool buffers)
    }
    let glsl_id = asm.id();
    let glsl_name = b"GLSL.std.450\x00";
    let mut glsl_words = Vec::new();
    for chunk in glsl_name.chunks(4) {
        let mut w = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            w |= (b as u32) << (i * 8);
        }
        glsl_words.push(w);
    }
    let void_id = asm.id(); // reserved for OpTypeVoid (emitted later in types section)
    asm.emit_type(OpExtInstImport, glsl_id, &glsl_words);
    asm.emit(OpMemoryModel, &[0, 1]); // Logical GLSL450

    // === Type helpers (closures) ===
    let push_dtype = |asm: &mut Asm, cache: &mut Map<DType, u32>, entries: &mut Vec<(OpCode, u32, Vec<u32>)>, dt: DType| {
        if let Some(&id) = cache.get(&dt) {
            return id;
        }
        let id = match dt {
            DType::Bool => {
                let i = asm.id();
                entries.push((OpTypeBool, i, vec![]));
                i
            }
            DType::U8 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![8, 0]));
                i
            }
            DType::U16 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![16, 0]));
                i
            }
            DType::U32 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![32, 0]));
                i
            }
            DType::U64 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![64, 0]));
                i
            }
            DType::I8 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![8, 1]));
                i
            }
            DType::I16 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![16, 1]));
                i
            }
            DType::I32 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![32, 1]));
                i
            }
            DType::I64 => {
                let i = asm.id();
                entries.push((OpTypeInt, i, vec![64, 1]));
                i
            }
            DType::F16 | DType::BF16 => {
                let i = asm.id();
                entries.push((OpTypeFloat, i, vec![16]));
                i
            }
            DType::F32 => {
                let i = asm.id();
                entries.push((OpTypeFloat, i, vec![32]));
                i
            }
            DType::F64 => {
                let i = asm.id();
                entries.push((OpTypeFloat, i, vec![64]));
                i
            }
        };
        cache.insert(dt, id);
        id
    };

    let push_ptr_type = |asm: &mut Asm,
                         cache: &mut Map<(u32, u32), u32>,
                         entries: &mut Vec<(OpCode, u32, Vec<u32>)>,
                         class: u32,
                         elem_type: u32| {
        if let Some(&id) = cache.get(&(class, elem_type)) {
            return id;
        }
        let id = asm.id();
        entries.push((OpTypePointer, id, vec![class, elem_type]));
        cache.insert((class, elem_type), id);
        id
    };

    let push_vec_type = |asm: &mut Asm,
                         cache: &mut Map<(u32, u16), u32>,
                         entries: &mut Vec<(OpCode, u32, Vec<u32>)>,
                         scalar_type: u32,
                         len: u16| {
        if let Some(&id) = cache.get(&(scalar_type, len)) {
            return id;
        }
        let id = asm.id();
        entries.push((OpTypeVector, id, vec![scalar_type, len as u32]));
        cache.insert((scalar_type, len), id);
        id
    };

    let layout_type_id = |asm: &mut Asm,
                          type_cache: &mut Map<DType, u32>,
                          vec_type_cache: &mut Map<(u32, u16), u32>,
                          entries: &mut Vec<(OpCode, u32, Vec<u32>)>,
                          dt: DType,
                          layout: MemLayout|
     -> u32 {
        let scalar_id = push_dtype(asm, type_cache, entries, dt);
        match layout {
            MemLayout::Scalar => scalar_id,
            MemLayout::Vector(len) => push_vec_type(asm, vec_type_cache, entries, scalar_id, len),
            MemLayout::Tile { .. } => scalar_id,
        }
    };

    // === SPIR-V state ===
    let mut type_cache: Map<DType, u32> = Map::with_capacity_and_hasher(16, BuildHasherDefault::new());
    let mut vec_type_cache: Map<(u32, u16), u32> = Map::with_capacity_and_hasher(4, BuildHasherDefault::new());
    let mut type_entries: Vec<(OpCode, u32, Vec<u32>)> = Vec::with_capacity(32);
    let mut const_entries: Vec<(u32, u32, Vec<u32>)> = Vec::with_capacity(16);
    let mut len_const_ids: std::collections::HashSet<u32> = std::collections::HashSet::new(); // constant IDs used as array lengths
    let mut spv_values: Map<OpId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
    let mut var_entries: Vec<(u32, u32, u32, bool)> = Vec::with_capacity(16);
    let mut global_var_ids: Vec<u32> = Vec::with_capacity(4);
    let mut decorations: Vec<(u32, Decoration, Vec<u32>)> = Vec::with_capacity(8);
    let mut member_decorations: Vec<(u32, u32, Decoration, Vec<u32>)> = Vec::with_capacity(4);
    let mut binding: u32 = 0;
    let mut spv_variables: Map<OpId, u32> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());
    let mut bool_buffers: std::collections::HashSet<OpId> = std::collections::HashSet::new(); // global buffers storing bool as u32
    let mut ptr_cache: Map<(u32, u32), u32> = Map::with_capacity_and_hasher(16, BuildHasherDefault::new());
    let mut reg_arrays: Map<OpId, (u32, u32)> = Map::with_capacity_and_hasher(8, BuildHasherDefault::new());
    let mut cast_u32_consts: Map<OpId, (u32, u32)> = Map::with_capacity_and_hasher(4, BuildHasherDefault::new());
    let mut const_pool: Map<Constant, u32> = Map::with_capacity_and_hasher(16, BuildHasherDefault::new());

    // Pre-define common types
    type_entries.push((OpTypeVoid, void_id, vec![]));
    let u32_id = push_dtype(&mut asm, &mut type_cache, &mut type_entries, DType::U32);
    let const_u32_0 = asm.id();
    const_entries.push((u32_id, const_u32_0, vec![0]));
    const_pool.insert(Constant::U32(0), const_u32_0);

    // Pre-populate all vector types needed by dtypes, so they're emitted before function body
    for (_op_id, (dt, layout)) in dtypes.iter() {
        if let MemLayout::Vector(len) = layout {
            let scalar_id = push_dtype(&mut asm, &mut type_cache, &mut type_entries, *dt);
            push_vec_type(&mut asm, &mut vec_type_cache, &mut type_entries, scalar_id, *len);
        }
    }
    let const_u32_1 = asm.id();
    const_entries.push((u32_id, const_u32_1, vec![1]));
    const_pool.insert(Constant::U32(1), const_u32_1);
    for i in 2u32..8 {
        let c = asm.id();
        const_entries.push((u32_id, c, vec![i]));
        const_pool.insert(Constant::U32(i), c);
    }
    // u8 type for bool storage in Vulkan buffers (StorageUniform8BitAccess)
    let (u8_id, const_u8_0, const_u8_1) = if needs_u8 {
        let id = push_dtype(&mut asm, &mut type_cache, &mut type_entries, DType::U8);
        let c0 = asm.id();
        const_entries.push((id, c0, vec![0]));
        let c1 = asm.id();
        const_entries.push((id, c1, vec![1]));
        (Some(id), Some(c0), Some(c1))
    } else {
        (None, None, None)
    };
    let vec3_id = asm.id();
    type_entries.push((OpTypeVector, vec3_id, vec![u32_id, 3]));

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
                            let is_bool = dtype == DType::Bool;
                            if is_bool {
                                bool_buffers.insert(op_id);
                            }
                            // Use u8 as storage type for bool buffers (Vulkan can't store bool in StorageBuffer)
                            let storage_st = if is_bool { u8_id.unwrap() } else { st };
                            let arr = asm.id();
                            type_entries.push((OpTypeRuntimeArray, arr, vec![storage_st]));
                            let stride = if is_bool { 1 } else { elem_stride(dtype) as u32 };
                            decorations.push((arr, Decoration::DecArrayStride, vec![stride]));
                            let struct_id = asm.id();
                            type_entries.push((OpTypeStruct, struct_id, vec![arr]));
                            decorations.push((struct_id, Decoration::DecBlock, vec![]));
                            member_decorations.push((struct_id, 0, Decoration::DecOffset, vec![0]));
                            let ptr = asm.id();
                            type_entries.push((OpTypePointer, ptr, vec![SC_STORAGE_BUFFER, struct_id]));
                            let var = asm.id();
                            var_entries.push((ptr, var, SC_STORAGE_BUFFER, true));
                            global_var_ids.push(var);
                            decorations.push((var, Decoration::DecDescriptorSet, vec![0]));
                            decorations.push((var, Decoration::DecBinding, vec![binding]));
                            if ro {
                                decorations.push((var, Decoration::DecNonWritable, vec![]));
                            }
                            binding += 1;
                            spv_variables.insert(op_id, var);
                            // Pre-cache element pointer type using the actual storage type (u8 for bool, logical type otherwise)
                            push_ptr_type(
                                &mut asm,
                                &mut ptr_cache,
                                &mut type_entries,
                                SC_STORAGE_BUFFER,
                                if is_bool { u8_id.unwrap() } else { st },
                            );
                        }
                        Scope::Local => {
                            let len_cid = asm.id();
                            const_entries.push((u32_id, len_cid, vec![len as u32]));
                            len_const_ids.insert(len_cid);
                            let arr = asm.id();
                            type_entries.push((OpTypeArray, arr, vec![st, len_cid]));
                            let ptr = asm.id();
                            type_entries.push((OpTypePointer, ptr, vec![SC_WORKGROUP, arr]));
                            let var = asm.id();
                            var_entries.push((ptr, var, SC_WORKGROUP, false));
                            global_var_ids.push(var);
                            spv_variables.insert(op_id, var);
                            push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_WORKGROUP, st);
                        }
                        Scope::Register => {
                            let len_cid = asm.id();
                            const_entries.push((u32_id, len_cid, vec![len as u32]));
                            len_const_ids.insert(len_cid);
                            let arr = asm.id();
                            type_entries.push((OpTypeArray, arr, vec![st, len_cid]));
                            let ptr = asm.id();
                            type_entries.push((OpTypePointer, ptr, vec![SC_FUNCTION, arr]));
                            reg_arrays.insert(op_id, (ptr, st));
                            push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, st);
                        }
                    };
                }
                &Op::Cast { x, dtype } => {
                    let src = dtypes[&x].0;
                    let dst = dtype;
                    if src == DType::Bool && dst.is_float() {
                        let u32_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, DType::U32);
                        let one_cid = asm.id();
                        let zero_cid = asm.id();
                        const_entries.push((u32_type, one_cid, vec![1u32]));
                        const_entries.push((u32_type, zero_cid, vec![0u32]));
                        cast_u32_consts.insert(op_id, (one_cid, zero_cid));
                    } else if src == DType::Bool
                        && (dst == DType::U32
                            || dst == DType::I32
                            || dst == DType::U64
                            || dst == DType::I64
                            || dst == DType::U16
                            || dst == DType::I16
                            || dst == DType::U8
                            || dst == DType::I8)
                    {
                        let int_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, dst);
                        let one_cid = asm.id();
                        let zero_cid = asm.id();
                        let one_words = match dst.bit_size() {
                            64 => vec![1u32, 0u32],
                            _ => vec![1u32],
                        };
                        let zero_words = match dst.bit_size() {
                            64 => vec![0u32, 0u32],
                            _ => vec![0u32],
                        };
                        const_entries.push((int_type, one_cid, one_words));
                        const_entries.push((int_type, zero_cid, zero_words));
                        cast_u32_consts.insert(op_id, (one_cid, zero_cid));
                    } else if (src == DType::U32
                        || src == DType::I32
                        || src == DType::U64
                        || src == DType::I64
                        || src == DType::U16
                        || src == DType::I16
                        || src == DType::U8
                        || src == DType::I8)
                        && dst == DType::Bool
                    {
                        let src_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, src);
                        let zero_cid = asm.id();
                        let zero_words = match src.bit_size() {
                            64 => vec![0u32, 0u32],
                            _ => vec![0u32],
                        };
                        const_entries.push((src_type, zero_cid, zero_words));
                        cast_u32_consts.insert(op_id, (zero_cid, 0));
                    }
                }
                _ => {}
            }
            // Pre-allocate constants for Recip, Abs, and Loop
            match kernel.at(op_id) {
                &Op::Unary { uop, x } if uop == UOp::Reciprocal => {
                    let dt = dtypes[&x].0;
                    let one_c = float_one(dt);
                    if !const_pool.contains_key(&one_c) {
                        let tid = type_cache[&dt];
                        let cid = asm.id();
                        let words = const_to_words(&one_c);
                        const_entries.push((tid, cid, words));
                        const_pool.insert(one_c, cid);
                    }
                }
                &Op::Unary { uop, x } if uop == UOp::Abs && dtypes[&x].0.is_int() && !dtypes[&x].0.is_uint() => {
                    let dt = dtypes[&x].0;
                    let tid = type_cache[&dt];
                    let zero = dt.zero_constant();
                    if !const_pool.contains_key(&zero) {
                        let cid = asm.id();
                        let words = const_to_words(&zero);
                        const_entries.push((tid, cid, words));
                        const_pool.insert(zero, cid);
                    }
                }
                &Op::Loop { len } => {
                    for &val in &[0u32, 1, len as u32] {
                        let key = match IDX_T {
                            DType::U32 => Constant::U32(val),
                            DType::I32 => Constant::I32(val as i32),
                            DType::U64 => Constant::U64((val as u64).to_le_bytes()),
                            _ => unreachable!(),
                        };
                        if !const_pool.contains_key(&key) {
                            let tid = type_cache[&IDX_T];
                            let cid = asm.id();
                            let words = const_to_words(&key);
                            const_entries.push((tid, cid, words));
                            const_pool.insert(key, cid);
                        }
                    }
                }
                &Op::Barrier { .. } => {
                    for &val in &[SCOPE_WORKGROUP, SEM_ACQUIRE_RELEASE | SEM_WORKGROUP_MEMORY] {
                        let key = Constant::U32(val);
                        if !const_pool.contains_key(&key) {
                            let tid = type_cache[&DType::U32];
                            let cid = asm.id();
                            const_entries.push((tid, cid, vec![val]));
                            const_pool.insert(key, cid);
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
    for &(dt, _) in dtypes.values() {
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
                if matches!(scope, Scope::Global | Scope::Local) {
                    found = true;
                    break;
                }
            }
            op_id = kernel.next_op(op_id);
        }
        found
    };

    let (wg_id_var, local_inv_var) = if needs_global {
        let wiv = asm.id();
        let liv = asm.id();
        let inp_ptr = asm.id();
        type_entries.push((OpTypePointer, inp_ptr, vec![SC_INPUT, vec3_id]));
        var_entries.push((inp_ptr, wiv, SC_INPUT, true));
        var_entries.push((inp_ptr, liv, SC_INPUT, true));
        decorations.push((wiv, Decoration::DecBuiltIn, vec![BI_WORKGROUP_ID]));
        decorations.push((liv, Decoration::DecBuiltIn, vec![BI_LOCAL_INVOCATION_ID]));
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
            for (i, &b) in chunk.iter().enumerate() {
                w |= (b as u32) << (i * 8);
            }
            ep_words.push(w);
        }
        ep_words.extend_from_slice(&global_var_ids);
        let wc = 1 + ep_words.len() as u16;
        asm.words.push((wc as u32) << 16 | OpEntryPoint as u32);
        asm.words.extend_from_slice(&ep_words);
    }

    // Execution mode
    asm.emit(
        OpExecutionMode,
        &[func_id, MODE_LOCAL_SIZE, lws[0] as u32, lws[1] as u32, lws[2] as u32],
    );

    // Annotations
    for (var_id, dec, operands) in &decorations {
        let mut args = vec![*var_id, *dec as u32];
        args.extend_from_slice(operands);
        asm.emit(OpDecorate, &args);
    }
    for (struct_id, member, dec, operands) in &member_decorations {
        let mut args = vec![*struct_id, *member, *dec as u32];
        args.extend_from_slice(operands);
        asm.emit(OpMemberDecorate, &args);
    }

    // Types (emit array-length constants inline before OpTypeArray that references them)
    let mut emitted_consts: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (op, id, operands) in &type_entries {
        if *op == OpTypeArray {
            // OpTypeArray [result_id, element_type, length_const_id]
            if let Some(&len_cid) = operands.get(1) {
                if emitted_consts.insert(len_cid) {
                    // Find and emit the constant before the array type that references it
                    for &(ct, cr, ref cw) in &const_entries {
                        if cr == len_cid {
                            asm.emit_typed(OpConstant, ct, cr, cw);
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
            asm.emit_typed(OpConstant, type_id, result_id, words);
        }
    }

    // Global variables (non-Function storage class)
    for &(ptr_type_id, var_id, storage, _is_global) in &var_entries {
        asm.emit(OpVariable, &[ptr_type_id, var_id, storage]);
    }

    // Function type: void()
    let func_type_id = asm.id();
    asm.emit_type(OpTypeFunction, func_type_id, &[void_id]);

    // Function definition
    asm.emit_typed(OpFunction, void_id, func_id, &[FN_CTRL_NONE, func_type_id]);

    // Entry block label
    let entry_label = asm.id();
    asm.emit(OpLabel, &[entry_label]);

    // Function-scope variables (register arrays)
    let mut reg_vars: Map<OpId, u32> = Map::with_capacity_and_hasher(8, BuildHasherDefault::new());
    for (&op_id, &(ptr_type, _elem_type)) in reg_arrays.iter() {
        let var_id = asm.id();
        asm.emit(OpVariable, &[ptr_type, var_id, SC_FUNCTION]);
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
                Op::ConstView { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Move { .. }
                | Op::Reduce { .. }
                | Op::Wmma { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "SPIR-V: unexpected kernel op (should be unfolded)".into(),
                    });
                }

                Op::Vectorize { ops } => {
                    let (dt, layout) = dtypes[&op_id];
                    let result_type =
                        layout_type_id(&mut asm, &mut type_cache, &mut vec_type_cache, &mut type_entries, dt, layout);
                    let rid = asm.id();
                    let operands: Vec<u32> = ops.iter().map(|x| spv_values[x]).collect();
                    asm.emit_typed(OpCompositeConstruct, result_type, rid, &operands);
                    spv_values.insert(op_id, rid);
                }

                Op::Devectorize { vec, idx } => {
                    let scalar_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, dtypes[&op_id].0);
                    let rid = asm.id();
                    asm.emit_typed(OpCompositeExtract, scalar_type, rid, &[spv_values[&vec], *idx as u32]);
                    spv_values.insert(op_id, rid);
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

                &Op::Load { src, index, layout } => {
                    let (load_dt, _) = dtypes[&op_id];
                    let result_type = layout_type_id(
                        &mut asm,
                        &mut type_cache,
                        &mut vec_type_cache,
                        &mut type_entries,
                        load_dt,
                        layout,
                    );
                    let index_id = spv_values[&index];

                    // For vector loads, use scalar pointer type for OpAccessChain,
                    // then decompose into scalar loads + OpCompositeConstruct.
                    // SPIR-V does not allow loading a vector through a scalar pointer.
                    let is_vec = matches!(layout, MemLayout::Vector(_));
                    let (base_ptr, element_ptr_type, is_storage_buffer, is_bool_src) =
                        if let Some(&var_id) = spv_variables.get(&src) {
                            let is_local = matches!(kernel.at(src), &Op::Define { scope: Scope::Local, .. });
                            let sc = if is_local { SC_WORKGROUP } else { SC_STORAGE_BUFFER };
                            let is_bool_buf = bool_buffers.contains(&src) && !is_local;
                            // For bool storage buffers or vector loads, use scalar storage type
                            let storage_type = if is_bool_buf {
                                u8_id.unwrap()
                            } else {
                                push_dtype(&mut asm, &mut type_cache, &mut type_entries, load_dt)
                            };
                            let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, sc, storage_type);
                            (var_id, elem_ptr, !is_local, is_bool_buf)
                        } else if let Some(&var_id) = reg_vars.get(&src) {
                            let scalar_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, load_dt);
                            let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, scalar_type);
                            (var_id, elem_ptr, false, false)
                        } else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "SPIR-V: Load from unknown variable".into(),
                            });
                        };

                    if is_vec && !is_bool_src {
                        // Vector load: access into buffer at index, then do N scalar loads
                        let mut scalars = Vec::new();
                        let vec_len = match layout {
                            MemLayout::Vector(n) => n as usize,
                            _ => unreachable!(),
                        };
                        for i in 0..vec_len {
                            let off_const = const_pool[&Constant::U32(i as u32)];
                            let addr = if i == 0 {
                                index_id
                            } else {
                                let a = asm.id();
                                asm.emit_typed(OpIAdd, u32_id, a, &[index_id, off_const]);
                                a
                            };
                            let access = asm.id();
                            if is_storage_buffer {
                                asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, const_u32_0, addr]);
                            } else {
                                asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, addr]);
                            }
                            let loaded = asm.id();
                            let scalar_type = push_dtype(&mut asm, &mut type_cache, &mut type_entries, load_dt);
                            asm.emit_typed(OpLoad, scalar_type, loaded, &[access]);
                            scalars.push(loaded);
                        }
                        let loaded = asm.id();
                        asm.emit_typed(OpCompositeConstruct, result_type, loaded, &scalars);
                        spv_values.insert(op_id, loaded);
                    } else {
                        let access = asm.id();
                        if is_storage_buffer {
                            asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, const_u32_0, index_id]);
                        } else {
                            asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, index_id]);
                        }
                        let loaded = asm.id();
                        let load_type = if is_bool_src { u8_id.unwrap() } else { result_type };
                        asm.emit_typed(OpLoad, load_type, loaded, &[access]);
                        if is_bool_src {
                            let bool_val = asm.id();
                            asm.emit_typed(OpINotEqual, result_type, bool_val, &[loaded, const_u8_0.unwrap()]);
                            spv_values.insert(op_id, bool_val);
                        } else {
                            spv_values.insert(op_id, loaded);
                        }
                    }
                }

                &Op::Store { dst, x, index, layout } => {
                    let val_id = spv_values[&x];
                    let index_id = spv_values[&index];

                    let (base_ptr, element_ptr_type, is_storage_buffer, is_bool_dst) =
                        if let Some(&var_id) = spv_variables.get(&dst) {
                            let is_local = matches!(kernel.at(dst), &Op::Define { scope: Scope::Local, .. });
                            let sc = if is_local { SC_WORKGROUP } else { SC_STORAGE_BUFFER };
                            let is_bool_buf = bool_buffers.contains(&dst) && !is_local;
                            let val_type = emit_type(&mut asm, &mut type_cache, dtypes[&x].0);
                            let store_type = if is_bool_buf { u8_id.unwrap() } else { val_type };
                            let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, sc, store_type);
                            (var_id, elem_ptr, !is_local, is_bool_buf)
                        } else if let Some(&var_id) = reg_vars.get(&dst) {
                            let val_type = emit_type(&mut asm, &mut type_cache, dtypes[&x].0);
                            let elem_ptr = push_ptr_type(&mut asm, &mut ptr_cache, &mut type_entries, SC_FUNCTION, val_type);
                            (var_id, elem_ptr, false, false)
                        } else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "SPIR-V: Store to unknown variable".into(),
                            });
                        };

                    let val_type = emit_type(&mut asm, &mut type_cache, dtypes[&x].0);
                    match layout {
                        MemLayout::Vector(len) => {
                            for i in 0..len {
                                let lane = asm.id();
                                asm.emit_typed(OpCompositeExtract, val_type, lane, &[val_id, i as u32]);
                                let lane_val = if is_bool_dst {
                                    let u8_tmp = asm.id();
                                    asm.emit_typed(
                                        OpSelect,
                                        u8_id.unwrap(),
                                        u8_tmp,
                                        &[lane, const_u8_1.unwrap(), const_u8_0.unwrap()],
                                    );
                                    u8_tmp
                                } else {
                                    lane
                                };
                                let access = asm.id();
                                let addr = if i == 0 {
                                    index_id
                                } else {
                                    let off = asm.id();
                                    let off_const = const_pool[&Constant::U32(i as u32)];
                                    asm.emit_typed(OpIAdd, u32_id, off, &[index_id, off_const]);
                                    off
                                };
                                if is_storage_buffer {
                                    asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, const_u32_0, addr]);
                                } else {
                                    asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, addr]);
                                }
                                asm.emit(OpStore, &[access, lane_val]);
                            }
                        }
                        _ => {
                            let store_val = if is_bool_dst {
                                let u8_tmp = asm.id();
                                asm.emit_typed(
                                    OpSelect,
                                    u8_id.unwrap(),
                                    u8_tmp,
                                    &[val_id, const_u8_1.unwrap(), const_u8_0.unwrap()],
                                );
                                u8_tmp
                            } else {
                                val_id
                            };
                            let access = asm.id();
                            if is_storage_buffer {
                                asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, const_u32_0, index_id]);
                            } else {
                                asm.emit_typed(OpAccessChain, element_ptr_type, access, &[base_ptr, index_id]);
                            }
                            asm.emit(OpStore, &[access, store_val]);
                        }
                    }
                }

                &Op::Cast { x, dtype } => {
                    let src_type = dtypes[&x].0;
                    let src_id = spv_values[&x];
                    let dst_type = dtype;
                    let (_, layout) = dtypes[&op_id];
                    let result_type = layout_type_id(
                        &mut asm,
                        &mut type_cache,
                        &mut vec_type_cache,
                        &mut type_entries,
                        dst_type,
                        layout,
                    );

                    let rid = asm.id();
                    if src_type == DType::Bool && dst_type.is_float() {
                        let u32_type = emit_type(&mut asm, &mut type_cache, DType::U32);
                        let (u32_one, u32_zero) = cast_u32_consts[&op_id];
                        let int_tmp = asm.id();
                        asm.emit_typed(OpSelect, u32_type, int_tmp, &[src_id, u32_one, u32_zero]);
                        asm.emit_typed(OpConvertUToF, result_type, rid, &[int_tmp]);
                    } else if src_type == DType::Bool
                        && (dst_type == DType::U32
                            || dst_type == DType::I32
                            || dst_type == DType::U64
                            || dst_type == DType::I64
                            || dst_type == DType::U16
                            || dst_type == DType::I16
                            || dst_type == DType::U8
                            || dst_type == DType::I8)
                    {
                        let (one_cid, zero_cid) = cast_u32_consts[&op_id];
                        let (true_val, false_val) = match layout {
                            MemLayout::Vector(n) => {
                                let mut broadcast = |val: u32| -> u32 {
                                    let components = vec![val; n as usize];
                                    let b = asm.id();
                                    asm.emit_typed(OpCompositeConstruct, result_type, b, &components);
                                    b
                                };
                                (broadcast(one_cid), broadcast(zero_cid))
                            }
                            _ => (one_cid, zero_cid),
                        };
                        asm.emit_typed(OpSelect, result_type, rid, &[src_id, true_val, false_val]);
                    } else if dst_type == DType::Bool
                        && (src_type == DType::U32
                            || src_type == DType::I32
                            || src_type == DType::U64
                            || src_type == DType::I64
                            || src_type == DType::U16
                            || src_type == DType::I16
                            || src_type == DType::U8
                            || src_type == DType::I8)
                    {
                        let (zero_cid, _) = cast_u32_consts[&op_id];
                        asm.emit_typed(OpINotEqual, result_type, rid, &[src_id, zero_cid]);
                    } else {
                        let op = cast_op(src_type, dst_type);
                        asm.emit_typed(op, result_type, rid, &[src_id]);
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Unary { x, uop } => {
                    let src_id = spv_values[&x];
                    let (dt, layout) = dtypes[&x];
                    let result_type =
                        layout_type_id(&mut asm, &mut type_cache, &mut vec_type_cache, &mut type_entries, dt, layout);
                    let rid = asm.id();

                    match uop {
                        UOp::Neg => {
                            if dt.is_float() {
                                asm.emit_typed(OpFNegate, result_type, rid, &[src_id]);
                            } else {
                                asm.emit_typed(OpSNegate, result_type, rid, &[src_id]);
                            }
                        }
                        UOp::BitNot => {
                            asm.emit_typed(OpNot, result_type, rid, &[src_id]);
                        }
                        UOp::Exp => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Exp, src_id]);
                        }
                        UOp::Exp2 => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Exp2, src_id]);
                        }
                        UOp::Ln => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Log, src_id]);
                        }
                        UOp::Log2 => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Log2, src_id]);
                        }
                        UOp::Reciprocal => {
                            let one = const_pool[&float_one(dt)];
                            asm.emit_typed(OpFDiv, result_type, rid, &[one, src_id]);
                        }
                        UOp::Sqrt => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Sqrt, src_id]);
                        }
                        UOp::Sin => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Sin, src_id]);
                        }
                        UOp::Cos => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Cos, src_id]);
                        }
                        UOp::Floor => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Floor, src_id]);
                        }
                        UOp::Trunc => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Trunc, src_id]);
                        }
                        UOp::Abs => {
                            if dt.is_float() {
                                asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::FAbs, src_id]);
                            } else if dt.is_uint() {
                                spv_values.insert(op_id, src_id);
                                continue;
                            } else {
                                // Signed int abs: (x < 0) ? -x : x
                                let zero = const_pool[&dt.zero_constant()];
                                let bool_type = emit_type(&mut asm, &mut type_cache, DType::Bool);
                                let cmp = asm.id();
                                asm.emit_typed(OpSLessThan, bool_type, cmp, &[src_id, zero]);
                                let neg = asm.id();
                                asm.emit_typed(OpSNegate, result_type, neg, &[src_id]);
                                asm.emit_typed(OpSelect, result_type, rid, &[cmp, neg, src_id]);
                            }
                        }
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Binary { x, y, bop } => {
                    let x_id = spv_values[&x];
                    let y_id = spv_values[&y];
                    let dt = dtypes[&x].0;
                    let (res_dt, res_layout) = dtypes[&op_id];
                    let result_type = layout_type_id(
                        &mut asm,
                        &mut type_cache,
                        &mut vec_type_cache,
                        &mut type_entries,
                        res_dt,
                        res_layout,
                    );
                    let rid = asm.id();

                    let (float_op, int_op, _): (Option<OpCode>, Option<OpCode>, Option<OpCode>) = match bop {
                        BOp::Add => (Some(OpFAdd), Some(OpIAdd), None),
                        BOp::Sub => (Some(OpFSub), Some(OpISub), None),
                        BOp::Mul => (Some(OpFMul), Some(OpIMul), None),
                        BOp::Div => (
                            Some(OpFDiv),
                            if dt.is_float() {
                                None
                            } else if dt.is_int() {
                                Some(OpSDiv)
                            } else {
                                Some(OpUDiv)
                            },
                            None,
                        ),
                        BOp::Pow => {
                            asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::Pow, x_id, y_id]);
                            (None, None, None)
                        }
                        BOp::Mod => (Some(OpFMod), Some(OpSRem), None), // SPIR-V uses SRem for C-style %
                        BOp::Cmplt => (
                            Some(OpFOrdLessThan),
                            if dt.is_float() {
                                None
                            } else if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
                                Some(OpSLessThan)
                            } else {
                                Some(OpULessThan)
                            },
                            None,
                        ),
                        BOp::Cmpgt => (
                            Some(OpFOrdGreaterThan),
                            if dt.is_float() {
                                None
                            } else if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
                                Some(OpSGreaterThan)
                            } else {
                                Some(OpUGreaterThan)
                            },
                            None,
                        ),
                        BOp::Max => {
                            if dt.is_float() {
                                asm.emit_typed(OpExtInst, result_type, rid, &[glsl_set, glsl::FMax, x_id, y_id]);
                            } else {
                                let cmp = asm.id();
                                let cmp_type = emit_type(&mut asm, &mut type_cache, DType::Bool);
                                if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
                                    asm.emit_typed(OpSGreaterThan, cmp_type, cmp, &[x_id, y_id]);
                                } else {
                                    asm.emit_typed(OpUGreaterThan, cmp_type, cmp, &[x_id, y_id]);
                                }
                                asm.emit_typed(OpSelect, result_type, rid, &[cmp, x_id, y_id]);
                            }
                            (None, None, None)
                        }
                        BOp::Or => (None, None, Some(OpNot)), // handled differently
                        BOp::And => (None, None, Some(OpNot)),
                        BOp::BitXor => (None, Some(OpBitwiseXOr), None),
                        BOp::BitOr => (None, Some(OpBitwiseOr), None),
                        BOp::BitAnd => (None, Some(OpBitwiseAnd), None),
                        BOp::BitShiftLeft => (None, Some(OpShiftLeftLogical), None),
                        BOp::BitShiftRight => (None, Some(OpShiftRightLogical), None),
                        BOp::NotEq => (Some(OpFOrdNotEqual), Some(OpINotEqual), None),
                        BOp::Eq => (Some(OpFOrdEqual), Some(OpIEqual), None),
                    };

                    if dt == DType::Bool {
                        if matches!(bop, BOp::Or) {
                            asm.emit_typed(OpBitwiseOr, result_type, rid, &[x_id, y_id]);
                        } else if matches!(bop, BOp::And) {
                            asm.emit_typed(OpBitwiseAnd, result_type, rid, &[x_id, y_id]);
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
                    let dt = dtypes[&x].0;
                    let (_, layout) = dtypes[&op_id];
                    let result_type =
                        layout_type_id(&mut asm, &mut type_cache, &mut vec_type_cache, &mut type_entries, dt, layout);
                    let rid = asm.id();

                    // FMad not available in spirv crate, decompose to FMul + FAdd
                    if dt.is_float() {
                        let mul = asm.id();
                        asm.emit_typed(OpFMul, result_type, mul, &[x_id, y_id]);
                        asm.emit_typed(OpFAdd, result_type, rid, &[mul, z_id]);
                    } else {
                        let mul = asm.id();
                        asm.emit_typed(OpIMul, result_type, mul, &[x_id, y_id]);
                        asm.emit_typed(OpIAdd, result_type, rid, &[mul, z_id]);
                    }
                    spv_values.insert(op_id, rid);
                }

                &Op::Index { len: _, scope, axis } => {
                    let result_type = emit_type(&mut asm, &mut type_cache, IDX_T);
                    match scope {
                        Scope::Global => {
                            let loaded = asm.id();
                            asm.emit_typed(OpLoad, vec3_id, loaded, &[wg_id_var]);
                            let elem = asm.id();
                            asm.emit_typed(OpCompositeExtract, u32_id, elem, &[loaded, axis]);
                            if IDX_T == DType::U32 {
                                spv_values.insert(op_id, elem);
                            } else {
                                let widened = asm.id();
                                let op = match IDX_T {
                                    DType::U64 => OpUConvert,
                                    _ => unreachable!(),
                                };
                                asm.emit_typed(op, result_type, widened, &[elem]);
                                spv_values.insert(op_id, widened);
                            }
                        }
                        Scope::Local => {
                            let loaded = asm.id();
                            asm.emit_typed(OpLoad, vec3_id, loaded, &[local_inv_var]);
                            let elem = asm.id();
                            asm.emit_typed(OpCompositeExtract, u32_id, elem, &[loaded, axis]);
                            if IDX_T == DType::U32 {
                                spv_values.insert(op_id, elem);
                            } else {
                                let widened = asm.id();
                                let op = match IDX_T {
                                    DType::U64 => OpUConvert,
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
                    asm.emit(OpVariable, &[counter_ptr_type, counter_var, SC_FUNCTION]);
                    let zero = const_pool[&Constant::U32(0)];
                    asm.emit(OpStore, &[counter_var, zero]);
                    asm.emit(OpBranch, &[header]);

                    // Header (loop continue target)
                    asm.emit(OpLabel, &[header]);
                    asm.emit(OpLoopMerge, &[merge, continue_lbl, LOOP_CTRL_NONE]);
                    asm.emit(OpBranch, &[body]);

                    // Body block
                    asm.emit(OpLabel, &[body]);

                    // Load current counter value (this is the Loop op's SSA value)
                    let counter_val = asm.id();
                    asm.emit_typed(OpLoad, idx_type, counter_val, &[counter_var]);
                    spv_values.insert(op_id, counter_val);

                    loop_stack.push((header, merge, continue_lbl, counter_var, len));
                }

                Op::EndLoop => {
                    let (header, merge, continue_lbl, counter_var, len) = loop_stack.pop().unwrap();
                    let idx_type = emit_type(&mut asm, &mut type_cache, IDX_T);

                    // Branch to continue block
                    asm.emit(OpBranch, &[continue_lbl]);

                    // Continue block: load, increment, store, check
                    asm.emit(OpLabel, &[continue_lbl]);
                    let old = asm.id();
                    asm.emit_typed(OpLoad, idx_type, old, &[counter_var]);
                    let one = const_pool[&Constant::U32(1)];
                    let inc = asm.id();
                    asm.emit_typed(OpIAdd, idx_type, inc, &[old, one]);
                    asm.emit(OpStore, &[counter_var, inc]);

                    // Check if counter < len
                    let len_cid = const_pool[&Constant::U32(len as u32)];
                    let cmp_type = emit_type(&mut asm, &mut type_cache, DType::Bool);
                    let cmp = asm.id();
                    asm.emit_typed(OpULessThan, cmp_type, cmp, &[inc, len_cid]);
                    asm.emit(OpBranchConditional, &[cmp, header, merge]);

                    // Merge block
                    asm.emit(OpLabel, &[merge]);
                }

                &Op::If { condition } => {
                    let cond_id = spv_values[&condition];
                    let true_block = asm.id();
                    let merge = asm.id();

                    asm.emit(OpSelectionMerge, &[merge, SELECT_CTRL_NONE]);
                    asm.emit(OpBranchConditional, &[cond_id, true_block, merge]);

                    // True block
                    asm.emit(OpLabel, &[true_block]);
                    if_stack.push(merge);
                }

                Op::EndIf => {
                    let merge = if_stack.pop().unwrap();
                    asm.emit(OpBranch, &[merge]);
                    asm.emit(OpLabel, &[merge]);
                }

                &Op::Barrier { scope } => match scope {
                    Scope::Local => {
                        let scope_id = const_pool[&Constant::U32(SCOPE_WORKGROUP)];
                        let sem_id = const_pool[&Constant::U32(SEM_ACQUIRE_RELEASE | SEM_WORKGROUP_MEMORY)];
                        asm.emit(OpControlBarrier, &[scope_id, scope_id, sem_id]);
                    }
                    _ => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "SPIR-V: unsupported barrier scope".into(),
                        });
                    }
                },
            }
            op_id = kernel.next_op(op_id);
        }
    }

    // Return and end function
    asm.emit(OpReturn, &[]);
    asm.emit(OpFunctionEnd, &[]);

    // Set bound
    asm.set_bound();

    if debug_asm {
        debug_print(&asm.words);
    }

    if let Ok(path) = std::env::var("SPIRV_DUMP") {
        let bytes: Vec<u8> = asm.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(&path, &bytes);
    }

    Ok((asm.words, gws, lws))
}

fn cast_op(src: DType, dst: DType) -> OpCode {
    use DType::*;
    use OpCode::*;
    match (src, dst) {
        (BF16, F32) | (F16, F32) | (F32, F64) | (F16, F64) | (BF16, F64) | (F32, F16) | (F64, F16) | (F64, F32) => OpFConvert,
        (I8, I32)
        | (I16, I32)
        | (I32, I64)
        | (I8, I64)
        | (I16, I64)
        | (I32, I8)
        | (I64, I8)
        | (I32, I16)
        | (I64, I16)
        | (I64, I32) => OpSConvert,
        (U8, U32)
        | (U16, U32)
        | (U32, U64)
        | (U8, U64)
        | (U16, U64)
        | (U32, U8)
        | (U64, U8)
        | (U32, U16)
        | (U64, U16)
        | (U64, U32) => OpUConvert,
        (F32, I32) | (F64, I32) | (F32, I64) | (F64, I64) | (F16, I32) | (F16, I64) => OpConvertFToS,
        (F32, U32) | (F64, U32) | (F32, U64) | (F64, U64) | (F16, U32) | (F16, U64) => OpConvertFToU,
        (I32, F32) | (I64, F32) | (I32, F64) | (I64, F64) | (I32, F16) | (I64, F16) => OpConvertSToF,
        (U32, F32) | (U64, F32) | (U32, F64) | (U64, F64) | (U32, F16) | (U64, F16) => OpConvertUToF,
        (Bool, I32) | (Bool, U32) | (I32, Bool) | (U32, Bool) => OpBitcast,
        _ if src == DType::Bool || dst == DType::Bool => {
            // Bool conversions should be handled via OpSelect/OpINotEqual upstream
            unreachable!("Bool cast not handled upstream: {src:?} -> {dst:?}")
        }
        _ => {
            // Fallback: if same bit width, use bitcast
            if bit_size(src) == bit_size(dst) {
                OpBitcast
            } else {
                // Try via FConvert/SConvert/UConvert
                if dst.is_float() {
                    OpFConvert
                } else if dst.is_int() {
                    OpSConvert
                } else {
                    OpUConvert
                }
            }
        }
    }
}

fn const_to_words(c: &Constant) -> Vec<u32> {
    match *c {
        Constant::U8(x) => vec![x as u32],
        Constant::U16(x) => vec![x as u32],
        Constant::U32(x) => vec![x],
        Constant::U64(x) => {
            let v = u64::from_le_bytes(x);
            vec![v as u32, (v >> 32) as u32]
        }
        Constant::I8(x) => vec![x as u32],
        Constant::I16(x) => vec![x as u32],
        Constant::I32(x) => vec![x as u32],
        Constant::I64(x) => {
            let v = i64::from_le_bytes(x);
            vec![v as u32, (v >> 32) as u32]
        }
        Constant::F16(x) => vec![u16::from_le_bytes(x) as u32],
        Constant::BF16(x) => vec![u16::from_le_bytes(x) as u32],
        Constant::F32(x) => vec![u32::from_le_bytes(x)],
        Constant::F64(x) => vec![
            u32::from_le_bytes(x[..4].try_into().unwrap()),
            u32::from_le_bytes(x[4..].try_into().unwrap()),
        ],
        Constant::Bool(x) => vec![x as u32],
    }
}

fn float_one(dt: DType) -> Constant {
    dt.one_constant()
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
    if spv.len() < 5 {
        return;
    }
    let bound = spv[3];
    println!("; SPIR-V disassembly (bound={bound})");
    println!("; {} words", spv.len());

    let mut i = 5; // skip header
    while i < spv.len() {
        let w = spv[i];
        let word_count = (w >> 16) as u16;
        let op = (w & 0xffff) as u16;
        if word_count == 0 {
            break;
        }

        if let Ok(op) = OpCode::try_from(op) {
            print!("  {op:?}");
        } else {
            print!("  ??");
        }

        let operands = if i + word_count as usize <= spv.len() {
            &spv[i + 1..i + word_count as usize]
        } else {
            // Malformed SPIR-V, slice to end
            &spv[i + 1..]
        };

        // Format known instructions
        match op {
            17 => {
                // Capability
                if !operands.is_empty() {
                    print!(" {}", capability_name(operands[0]));
                }
            }
            11 => {
                // ExtInstImport
                if operands.len() >= 2 {
                    print!(" %{}", operands[0]);
                    let name_bytes: Vec<u8> = operands[1..].iter().flat_map(|w| w.to_le_bytes()).collect();
                    let name_str = String::from_utf8_lossy(&name_bytes).trim_end_matches('\0').to_string();
                    print!(" \"{name_str}\"");
                }
            }
            14 => {
                // MemoryModel
                if operands.len() >= 2 {
                    print!(" {}", if operands[0] == 0 { "Logical" } else { "??" });
                    print!(" {}", if operands[1] == 1 { "GLSL450" } else { "??" });
                }
            }
            15 => {
                // EntryPoint
                if operands.len() >= 3 {
                    let model = match operands[0] {
                        5 => "GLCompute",
                        _ => "??",
                    };
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
            16 => {
                // ExecutionMode
                if operands.len() >= 2 {
                    print!(" %{}", operands[0]);
                    let mode = match operands[1] {
                        17 => format!(
                            "LocalSize {} {} {}",
                            operands.get(2).unwrap_or(&0),
                            operands.get(3).unwrap_or(&0),
                            operands.get(4).unwrap_or(&0)
                        ),
                        _ => format!("??({})", operands[1]),
                    };
                    print!(" {mode}");
                }
            }
            71 => {
                // Decorate
                if operands.len() >= 2 {
                    print!(" %{} {:?}", operands[0], Decoration::try_from(operands[1]).unwrap());
                    if operands.len() > 2 {
                        if operands[1] == 11 {
                            // BuiltIn
                            print!(" {}", builtin_name(operands[2]));
                        } else {
                            for &v in &operands[2..] {
                                print!(" {v}");
                            }
                        }
                    }
                }
            }
            21 => {
                // TypeInt
                if operands.len() >= 2 {
                    print!(
                        " %{} {} {}",
                        operands[0],
                        operands[1],
                        if operands[2] == 0 { "u" } else { "i" }
                    );
                }
            }
            22 => {
                // TypeFloat
                if operands.len() >= 1 {
                    print!(" %{} {}", operands[0], operands[1]);
                }
            }
            19 | 20 => {
                // TypeVoid, TypeBool
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                }
            }
            23 | 28 | 29 | 33 => {
                // TypeVector, TypeArray, TypeRuntimeArray, TypeFunction
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                    for &v in &operands[1..] {
                        print!(" %{v}");
                    }
                }
            }
            32 => {
                // TypePointer
                if operands.len() >= 2 {
                    print!(" %{} {} %{}", operands[0], storage_class_name(operands[1]), operands[2]);
                }
            }
            43 | 61 | 65 | 81 | 87 | 12 => {
                // Constant, Load, AccessChain, CompositeExtract, Select, ExtInst
                if operands.len() >= 2 {
                    print!(" %{} %{}", operands[0], operands[1]);
                    for &v in &operands[2..] {
                        print!(" %{v}");
                    }
                }
            }
            59 => {
                // Variable
                if operands.len() >= 2 {
                    print!(" %{} %{} {}", operands[0], operands[1], storage_class_name(operands[2]));
                }
            }
            54 => {
                // Function
                if operands.len() >= 3 {
                    print!(" %{} %{}", operands[0], operands[1]);
                    let ctrl = match operands[2] {
                        0 => "None".into(),
                        x => format!("?[{x}]?"),
                    };
                    print!(" {ctrl}");
                    for &v in &operands[3..] {
                        print!(" %{v}");
                    }
                }
            }
            248 => {
                // Label
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                }
            }
            49 => {
                // Branch
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                }
            }
            50 => {
                // BranchConditional
                if operands.len() >= 3 {
                    print!(" %{} %{} %{}", operands[0], operands[1], operands[2]);
                }
            }
            246 | 247 => {
                // LoopMerge, SelectionMerge
                if !operands.is_empty() {
                    print!(" %{}", operands[0]);
                    if operands.len() > 1 {
                        print!(" %{}", operands[1]);
                    }
                }
            }
            62 => {
                // Store
                if operands.len() >= 2 {
                    print!(" %{} %{}", operands[0], operands[1]);
                }
            }
            224 => {
                // ControlBarrier
                if operands.len() >= 3 {
                    print!(" {} {} {}", operands[0], operands[1], operands[2]);
                }
            }
            _ => {
                // Generic: print all operands as numbers/IDs
                for (j, &v) in operands.iter().enumerate() {
                    if j == 0 && !matches!(op, 56 | 63) {
                        print!(" %{v}");
                    } else {
                        print!(" {v}");
                    }
                }
            }
        }
        println!();
        i += word_count as usize;
    }
}

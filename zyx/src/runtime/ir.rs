use crate::{
    dtype::Constant,
    runtime::{
        node::{BOp, ROp, UOp},
    },
    tensor::TensorId,
    DType,
};
use std::collections::BTreeMap;

use super::scheduler::{Kernel, VOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Var {
    Id(u8, Scope),
    Const(Constant),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, bitcode::Encode, bitcode::Decode)]
pub(super) enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IROp {
    Set {
        z: u8,
        len: usize,
        value: Constant,
    },
    Load {
        z: Var,
        x: Var,
        at: Var,
        dtype: IRDType,
    },
    Store {
        z: Var,
        x: Var,
        at: Var,
        dtype: IRDType,
    },
    Unary {
        z: Var,
        x: Var,
        uop: UOp,
        // For cast this is dtype before cast
        dtype: IRDType,
    },
    Binary {
        z: Var,
        x: Var,
        y: Var,
        bop: BOp,
        dtype: IRDType,
    },
    // z = a * b + c
    MAdd {
        z: Var,
        a: Var,
        b: Var,
        c: Var,
        dtype: IRDType,
    },
    Loop {
        id: u8,
        len: usize,
    },
    EndLoop {
        id: u8,
        len: usize,
    },
    Barrier {
        scope: Scope,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum IRDType {
    #[cfg(feature = "half")]
    BF16,
    #[cfg(feature = "half")]
    F16,
    F32,
    F64,
    #[cfg(feature = "complex")]
    CF32,
    #[cfg(feature = "complex")]
    CF64,
    U8,
    I8,
    I16,
    I32,
    I64,
    Bool,
    // For indexing, usually u32
    U32,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct IRKernel {
    // Index of var is it's Id
    // len, dtype, read_only
    pub(super) addressables: Vec<(usize, IRDType, bool)>,
    // dtype, read_only
    pub(super) registers: Vec<(IRDType, bool)>,
    pub(super) ops: Vec<IROp>,
}

impl IRDType {
    pub(super) fn byte_size(&self) -> usize {
        match self {
            #[cfg(feature = "half")]
            IRDType::F16 => 2,
            #[cfg(feature = "half")]
            IRDType::BF16 => 2,
            IRDType::F32 => 4,
            IRDType::F64 => 8,
            #[cfg(feature = "complex")]
            IRDType::CF32 => 8,
            #[cfg(feature = "complex")]
            IRDType::CF64 => 16,
            IRDType::U8 => 1,
            IRDType::I8 => 1,
            IRDType::I16 => 2,
            IRDType::I32 => 4,
            IRDType::I64 => 8,
            IRDType::Bool => 1,
            IRDType::U32 => 4,
        }
    }

    pub(super) fn dtype(&self) -> DType {
        match self {
            #[cfg(feature = "half")]
            IRDType::BF16 => DType::BF16,
            #[cfg(feature = "half")]
            IRDType::F16 => DType::F16,
            IRDType::F32 => DType::F32,
            IRDType::F64 => DType::F64,
            #[cfg(feature = "complex")]
            IRDType::CF32 => DType::CF32,
            #[cfg(feature = "complex")]
            IRDType::CF64 => DType::CF64,
            IRDType::U8 => DType::U8,
            IRDType::I8 => DType::I8,
            IRDType::I16 => DType::I16,
            IRDType::I32 => DType::I32,
            IRDType::I64 => DType::I64,
            IRDType::Bool => DType::Bool,
            IRDType::U32 => panic!(),
        }
    }
}

impl From<DType> for IRDType {
    fn from(value: DType) -> Self {
        match value {
            #[cfg(feature = "half")]
            DType::BF16 => IRDType::BF16,
            #[cfg(feature = "half")]
            DType::F16 => IRDType::F16,
            DType::F32 => IRDType::F32,
            DType::F64 => IRDType::F64,
            #[cfg(feature = "complex")]
            DType::CF32 => IRDType::CF32,
            #[cfg(feature = "complex")]
            DType::CF64 => IRDType::CF64,
            DType::U8 => IRDType::U8,
            DType::I8 => IRDType::I8,
            DType::I16 => IRDType::I16,
            DType::I32 => IRDType::I32,
            DType::I64 => IRDType::I64,
            DType::Bool => IRDType::Bool,
        }
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Global => "g",
            Self::Local => "l",
            Self::Register => "r",
        })
    }
}

// Data structures will stay the same, but we will rewrite the implementation of to_ir function.
// The current version just does not work, particularly with padding. We also want to support vector datatypes,
// and local and register tiling. Neither of which currently works.
// Indexing also needs to be rewritten so that as much of it happens outside of the loops
// and so that it does work properly

// Returns IRKernel and order in which tensors are passed to it
pub(super) fn to_ir(ops: &[VOp]) -> (IRKernel, Vec<TensorId>) {
    for op in ops {
        println!("{op}");
    }
    todo!()
}

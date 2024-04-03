// This is the final compilation step. Looped get rewritten into IR that is easy to compile.
// Tiles are flattened into single dimension, common indices are extracted out of register loops
// (to maximize reuse), some final optimizations are applied (like log2 instead of ln, fast sqrt,
// remove tanh and such).
// IR kernel gets send into device for direct translation and compilation. Compiled programs
// are stored in CompiledGraph and efficiently executed. For OpenCl, we can technically compile
// all of those kernels into one program, but it does not really give us huge advantages.

use alloc::vec::Vec;
use alloc::string::String;
use zyx_core::dtype::DType;

enum IROp {
    Loop {
        iterations: usize,
        scope: u8,
    },
    DeclareIndex {
        id: u8,
    },
    AssignIndex {
        id: u8,
        value: String,
    },
    DeclareMem {
        id: u8,
        dtype: DType,
        size: Option<usize>,
    },
    Neg {
        x: usize,
    },
    Sin {
        x: usize,
    },
    Cos {
        x: usize,
    },
    Exp2 {
        x: usize,
    },
    Ln2 {
        x: usize,
    },
    Sqrt {
        x: usize,
    },
    Add {
        x: usize,
        y: usize,
    },
    Sub {
        x: usize,
        y: usize,
    },
    Mul {
        x: usize,
        y: usize,
    },
    Div {
        x: usize,
        y: usize,
    },
    Pow {
        x: usize,
        y: usize,
    },
    Cmplt {
        x: usize,
        y: usize,
    },
    Max {
        x: usize,
        y: usize,
    },
}

struct IRKernel {
    ops: Vec<IROp>,
}

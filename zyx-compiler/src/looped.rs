// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use zyx_core::dtype::DType;

// Includes Noop for copying between tiles of various scopes
enum UOp {
    Noop,
    Neg,
    Sin,
    Cos,
    Exp,
    Ln,
    Tanh,
    Sqrt,
}

enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Max, // for ReLU and max reduce
}

enum Op {
    Tile,
    Cast {
        x: usize,
        dtype: DType,
    },
    Unary {
        x: usize,
        op: UOp,
    },
    Binary {
        x: usize,
        y: usize,
        op: BOp,
    },
    Loop {
        iters: usize,
        scope: u8,
    },
    EndLoop,
}
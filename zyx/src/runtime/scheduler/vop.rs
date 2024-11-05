use crate::{
    dtype::{Constant, DType},
    runtime::{
        ir::Scope,
        node::{BOp, ROp, UOp},
        view::View,
    },
    shape::{Axis, Dimension},
    tensor::TensorId,
};

// Should be just Unary, Binary, Const, Copy, Loop, Reduce
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum VOp {
    Loop {
        axis: Axis,
        len: Dimension,
    },
    // End the latest loop
    EndLoop,
    Const {
        z: TensorId,
        value: Constant,
        view: View,
    },
    Load {
        z: TensorId,
        zscope: Scope,
        zview: View,
        x: TensorId,
        xscope: Scope,
        xview: View,
        xdtype: DType,
    },
    Store {
        z: TensorId,
        zscope: Scope,
        zview: View,
        zdtype: DType,
        xscope: Scope,
        xview: View,
    },
    Accumulator {
        z: TensorId,
        rop: ROp,
        view: View,
        dtype: DType,
    },
    // Move is noop, just a marker for easy debugging
    // and to keep track of tensor ids
    Move {
        z: TensorId,
        x: TensorId,
        mop: MOp,
    },
    Unary {
        z: TensorId,
        x: TensorId,
        uop: UOp,
    },
    Binary {
        z: TensorId,
        x: TensorId,
        y: TensorId,
        bop: BOp,
    },
    // Synchronization for local and global memory
    Barrier {
        scope: Scope,
    },
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MOp {
    Expa,
    Perm,
    Resh,
    Padd,
}

impl std::fmt::Display for VOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const C_BLUE: &str = "\x1B[34m";
        const C_GREEN: &str = "\x1B[32m";
        const C_MAGENTA: &str = "\x1B[35m";
        const C_RED: &str = "\x1B[31m";
        const C_WHITE: &str = "\x1B[37m";
        const C_YELLOW: &str = "\x1B[33m";
        const C_RESET: &str = "\x1B[39m";
        match self {
            VOp::Const { z, value, view } => f.write_fmt(format_args!(
                "{C_WHITE}Const{C_RESET}       {z} <- value: {value}, {view}"
            )),
            VOp::Load {
                z,
                zscope,
                zview: _,
                x,
                xscope,
                xview,
                xdtype,
            } => f.write_fmt(format_args!(
                "{C_YELLOW}Load{C_RESET}        {z}[{zscope:?}] <- {x}[{xscope:?}, {xdtype}], {xview}"
            )),
            VOp::Store {
                z,
                zview,
                zscope,
                zdtype,
                xscope,
                xview: _,
            } => f.write_fmt(format_args!(
                "{C_RED}Store{C_RESET}        {z}[{zscope:?}] <- {xscope:?}, {zview}, {zdtype}"
            )),
            VOp::Loop {
                axis,
                len: dimension,
            } => f.write_fmt(format_args!(
                "{C_GREEN}Loop{C_RESET}        axis: {axis}, dimension: {dimension}"
            )),
            VOp::Accumulator { z, rop, view, dtype } => f.write_fmt(format_args!(
                "{C_BLUE}Accum{C_RESET}.{rop:?}   {z}, shape: {:?}, {dtype}",
                view.shape()
            )),
            VOp::EndLoop => f.write_fmt(format_args!("{C_BLUE}EndLoop{C_RESET} ")),
            VOp::Move { z, x, mop } => f.write_fmt(format_args!(
                "{C_WHITE}Move{C_RESET}.{mop:?}   {z} <- {x}"
            )),
            VOp::Unary { z, x, uop } => {
                let mut len = format!("{uop:?}").len();
                if len > 5 {
                    len = 5;
                }
                f.write_fmt(format_args!(
                    "{C_WHITE}Unary{C_RESET}.{uop:?}{} {z} <- {x}",
                    " ".repeat(5 - len)
                ))
            }
            VOp::Binary { z, x, y, bop } => f.write_fmt(format_args!(
                "{C_WHITE}Binary{C_RESET}.{bop:?}  {z} <- {x}, {y}"
            )),
            VOp::Barrier { scope } => f.write_fmt(format_args!("{C_MAGENTA}Barrier{C_RESET}({scope})")),
        }
    }
}

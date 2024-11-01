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
pub(crate) enum VOp {
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
pub(crate) enum MOp {
    Expa,
    Perm,
    Resh,
    Padd,
}

impl std::fmt::Display for VOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use inline_colorization::*;
        match self {
            VOp::Const { z, value, view } => f.write_fmt(format_args!(
                "{color_white}Const{color_reset}       {z} <- value: {value}, {view}"
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
                "{color_yellow}Load{color_reset}        {z}[{zscope:?}] <- {x}[{xscope:?}, {xdtype}], {xview}"
            )),
            VOp::Store {
                z,
                zview,
                zscope,
                zdtype,
                xscope,
                xview: _,
            } => f.write_fmt(format_args!(
                "{color_red}Store{color_reset}        {z}[{zscope:?}] <- {xscope:?}, {zview}, {zdtype}"
            )),
            VOp::Loop {
                axis,
                len: dimension,
            } => f.write_fmt(format_args!(
                "{color_green}Loop{color_reset}        axis: {axis}, dimension: {dimension}"
            )),
            VOp::Accumulator { z, rop, view, dtype } => f.write_fmt(format_args!(
                "{color_blue}Accum{color_reset}.{rop:?}   {z}, shape: {:?}, {dtype}",
                view.shape()
            )),
            VOp::EndLoop => f.write_fmt(format_args!("{color_blue}EndLoop{color_reset} ")),
            VOp::Move { z, x, mop } => f.write_fmt(format_args!(
                "{color_white}Move{color_reset}.{mop:?}   {z} <- {x}"
            )),
            VOp::Unary { z, x, uop } => {
                let mut len = format!("{uop:?}").len();
                if len > 5 {
                    len = 5;
                }
                f.write_fmt(format_args!(
                    "{color_white}Unary{color_reset}.{uop:?}{} {z} <- {x}",
                    " ".repeat(5 - len)
                ))
            }
            VOp::Binary { z, x, y, bop } => f.write_fmt(format_args!(
                "{color_white}Binary{color_reset}.{bop:?}  {z} <- {x}, {y}"
            )),
            VOp::Barrier { scope } => f.write_fmt(format_args!("{color_magenta}Barrier{color_reset}({scope})")),
        }
    }
}

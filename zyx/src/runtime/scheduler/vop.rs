use crate::{
    dtype::Constant,
    runtime::{
        ir::Scope,
        node::{BOp, ROp, UOp},
        view::View,
    },
    shape::{Axis, Dimension},
    tensor::TensorId,
};

// Should be just Unary, Binary, Const, Copy, Loop, Reduce
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, bitcode::Encode, bitcode::Decode)]
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
        x: TensorId,
        xscope: Scope,
        view: View,
    },
    Store {
        z: TensorId,
        zscope: Scope,
        xscope: Scope,
        view: View,
    },
    // TODO remove accumulator and use const + load
    // instead to create register tile
    Accumulator {
        z: TensorId,
        rop: ROp,
        view: View,
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
        view: View,
    },
    Binary {
        z: TensorId,
        zview: View,
        x: TensorId,
        xview: View,
        y: TensorId,
        yview: View,
        bop: BOp,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, bitcode::Encode, bitcode::Decode)]
pub(super) enum MOp {
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
                x,
                xscope,
                view,
            } => f.write_fmt(format_args!(
                "{color_yellow}Load{color_reset}        {z}[{zscope:?}] <- {x}[{xscope:?}], {view}"
            )),
            VOp::Store {
                z,
                zscope,
                xscope,
                view,
            } => f.write_fmt(format_args!(
                "{color_red}Store{color_reset}        {z}[{zscope:?}] <- [{xscope:?}], {view}"
            )),
            VOp::Loop {
                axis,
                len: dimension,
            } => f.write_fmt(format_args!(
                "{color_green}Loop{color_reset}        axis: {axis}, dimension: {dimension}"
            )),
            VOp::Accumulator { z, rop, view } => f.write_fmt(format_args!(
                "{color_blue}Accum{color_reset}.{rop:?}   {z}, shape: {:?}",
                view.shape()
            )),
            VOp::EndLoop => f.write_fmt(format_args!("{color_blue}EndLoop{color_reset} ")),
            /*VOp::Reduce {
                z,
                x,
                num_axes,
                rop,
            } => f.write_fmt(format_args!(
                "{color_magenta}Reduce{color_reset}.{rop:?}  {z} <- {x}, num_axes: {num_axes}"
            )),*/
            VOp::Move { z, x, mop } => f.write_fmt(format_args!(
                "{color_white}Move{color_reset}.{mop:?}   {z} <- {x}"
            )),
            VOp::Unary { z, x, uop, view } => {
                let mut len = format!("{uop:?}").len();
                if len > 5 {
                    len = 5;
                }
                f.write_fmt(format_args!(
                    "{color_white}Unary{color_reset}.{uop:?}{} {z} <- {x}, {view}",
                    core::iter::repeat(" ").take(5 - len).collect::<String>()
                ))
            }
            VOp::Binary { z, zview, x, xview, y, yview, bop } => f.write_fmt(format_args!(
                "{color_white}Binary{color_reset}.{bop:?}  {z}[{zview}] <- {x}[{xview}], {y}[{yview}]"
            )),
        }
    }
}

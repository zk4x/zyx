use crate::{
    dtype::Constant, graph::{BOp, ROp, UOp}, optimizer::Optimization, shape::{Axis, Dim}, DType
};

use super::view::View;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpKind {
    Const { value: Constant, view: View },
    Load { view: View, dtype: DType },
    Store { x: Op, view: View },
    Cast { x: Op, dtype: DType },
    Unary { x: Op, uop: UOp },
    Binary { x: Op, y: Op, bop: BOp },
    Reduce { x: Op, rop: ROp, num_loops: u32 },
    // Sink { ops: Set<Op> } - will be just a way to put multiple ops/stores into one kernel
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Op(Box<OpKind>);

impl Op {
    pub fn constant(value: Constant) -> Self {
        let view = View::contiguous(&[1]);
        Self(Box::new(OpKind::Const { value, view }))
    }

    pub fn load(shape: &[Dim], dtype: DType) -> Self {
        let view = View::contiguous(shape);
        Self(Box::new(OpKind::Load { view, dtype }))
    }

    pub fn unary(x: Self, uop: UOp) -> Self {
        Self(Box::new(OpKind::Unary { x, uop }))
    }

    pub fn cast(x: Self, dtype: DType) -> Self {
        Self(Box::new(OpKind::Cast { x, dtype }))
    }

    pub fn store(x: Self, shape: &[Dim]) -> Self {
        let view = View::contiguous(shape);
        Self(Box::new(OpKind::Store { x, view }))
    }

    pub fn reduce(mut x: Self, rop: ROp, axes: &[Axis], n: usize) -> Self {
        // Permute so that reduce axes are last
        let permute_axes: Vec<Axis> =
            (0..n).filter(|a| !axes.contains(a)).chain(axes.iter().copied()).collect();
        x.movement(|view| view.permute(&permute_axes));
        let num_loops = axes.len() as u32;
        Self(Box::new(OpKind::Reduce { x, rop, num_loops }))
    }

    pub fn binary(x: Self, y: Self, bop: BOp) -> Self {
        Self(Box::new(OpKind::Binary { x, y, bop }))
    }

    pub fn movement(&mut self, func: impl Fn(&mut View) + Clone) {
        match self.0.as_mut() {
            OpKind::Load { view, .. } => func(view),
            OpKind::Const { view, .. } => func(view),
            OpKind::Store { view, .. } => func(view),
            OpKind::Cast { x, .. } => x.movement(func),
            OpKind::Unary { x, .. } => x.movement(func),
            OpKind::Binary { x, y, .. } => {
                x.movement(func.clone());
                y.movement(func);
            }
            OpKind::Reduce { x, .. } => x.movement(func),
        }
    }
    
    fn shape_numel(&self) -> Dim {
        match self.0.as_ref() {
            OpKind::Const { view, .. } |
            OpKind::Load { view, .. } |
            OpKind::Store { view, .. } => view.numel(),
            OpKind::Cast { x, .. } |
            OpKind::Unary { x, .. } |
            OpKind::Binary { x, .. } |
            OpKind::Reduce { x, .. } => x.shape_numel(),
        }
    }

    pub fn flop_mem_rw(&self) -> (u128, u128, u128) {
        match self.0.as_ref() {
            OpKind::Const { .. } => (0, 0, 0),
            OpKind::Load { view, .. } => (0, view.original_numel() as u128, 0),
            OpKind::Store { x, view } => {
                let (f, mr, mw) = x.flop_mem_rw();
                (f, mr, mw + view.original_numel() as u128)
            }
            OpKind::Cast { x, .. } |
            OpKind::Unary { x, .. } => {
                let (f, mr, mw) = x.flop_mem_rw();
                (f + self.shape_numel() as u128, mr, mw)
            }
            OpKind::Binary { x, y, .. } => {
                let (xf, xmr, xmw) = x.flop_mem_rw();
                let (yf, ymr, ymw) = y.flop_mem_rw();
                (xf + yf + self.shape_numel() as u128, xmr + ymr, xmw + ymw)
            }
            OpKind::Reduce { x, .. } => {
                let (f, mr, mw) = x.flop_mem_rw();
                (f + self.shape_numel() as u128 - 1, mr, mw)
            }
        }
    }
    
    pub fn apply_optimization(&mut self, opt: &Optimization) {
        // TODO
    }
}

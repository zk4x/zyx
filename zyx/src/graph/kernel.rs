use crate::{
    DType, dtype::Constant, shape::Dim,
};

use super::{BOp, ROp, UOp, view::View};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpKind {
    Const { value: Constant, view: View },
    Load { view: View, dtype: DType },
    Store { x: Op, view: View, dtype: DType },
    Cast { x: Op, dtype: DType },
    Unary { x: Op, uop: UOp },
    Binary { x: Op, y: Op, bop: BOp },
    Reduce { x: Op, rop: ROp, num_loops: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
}

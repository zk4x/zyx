use crate::{
    DType,
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    optimizer::Optimization,
    shape::{Axis, Dim},
};

use super::view::View;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpKind {
    ConstView { value: Constant, view: View },
    Const { value: Constant },
    LoopIndex { i: u32 },
    LoadView { view: View, dtype: DType },
    Load { dtype: DType, index: Op },
    StoreView { x: Op, view: View },
    Store { x: Op, index: Op },
    Cast { x: Op, dtype: DType },
    Unary { x: Op, uop: UOp },
    Binary { x: Op, y: Op, bop: BOp },
    Reduce { x: Op, rop: ROp, num_loops: u32 },
    // Sink { ops: Set<Op> } - will be just a way to put multiple ops/stores into one kernel
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Op(Box<OpKind>);

impl Op {
    pub fn constant_view(value: Constant) -> Self {
        let view = View::contiguous(&[1]);
        Self(Box::new(OpKind::ConstView { value, view }))
    }

    pub fn constant(value: Constant) -> Self {
        Self(Box::new(OpKind::Const { value }))
    }

    pub fn loop_index(i: u32) -> Self {
        Self(Box::new(OpKind::LoopIndex { i }))
    }

    pub fn load(shape: &[Dim], dtype: DType) -> Self {
        let view = View::contiguous(shape);
        Self(Box::new(OpKind::LoadView { view, dtype }))
    }

    pub fn unary(x: Self, uop: UOp) -> Self {
        Self(Box::new(OpKind::Unary { x, uop }))
    }

    pub fn cast(x: Self, dtype: DType) -> Self {
        Self(Box::new(OpKind::Cast { x, dtype }))
    }

    pub fn store(x: Self, shape: &[Dim]) -> Self {
        let view = View::contiguous(shape);
        Self(Box::new(OpKind::StoreView { x, view }))
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
            OpKind::LoadView { view, .. } => func(view),
            OpKind::ConstView { view, .. } => func(view),
            OpKind::StoreView { view, .. } => func(view),
            OpKind::Cast { x, .. } => x.movement(func),
            OpKind::Unary { x, .. } => x.movement(func),
            OpKind::Binary { x, y, .. } => {
                x.movement(func.clone());
                y.movement(func);
            }
            OpKind::LoopIndex { .. } => unreachable!(),
            OpKind::Reduce { x, .. } => x.movement(func),
            OpKind::Load { .. } | OpKind::Store { .. } => unreachable!(),
            OpKind::Const { .. } => unreachable!(),
        }
    }

    fn shape_numel(&self) -> Dim {
        match self.0.as_ref() {
            OpKind::ConstView { view, .. }
            | OpKind::LoadView { view, .. }
            | OpKind::StoreView { view, .. } => view.numel(),
            OpKind::Cast { x, .. }
            | OpKind::Unary { x, .. }
            | OpKind::Binary { x, .. }
            | OpKind::Reduce { x, .. } => x.shape_numel(),
            OpKind::Const { .. }
            | OpKind::LoopIndex { .. }
            | OpKind::Load { .. }
            | OpKind::Store { .. } => unreachable!(),
        }
    }

    pub fn flop_mem_rw(&self) -> (u128, u128, u128) {
        match self.0.as_ref() {
            OpKind::ConstView { .. } => (0, 0, 0),
            OpKind::LoadView { view, .. } => (0, view.original_numel() as u128, 0),
            OpKind::StoreView { x, view } => {
                let (f, mr, mw) = x.flop_mem_rw();
                (f, mr, mw + view.original_numel() as u128)
            }
            OpKind::Cast { x, .. } | OpKind::Unary { x, .. } => {
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
            OpKind::Const { .. }
            | OpKind::LoopIndex { .. }
            | OpKind::Load { .. }
            | OpKind::Store { .. } => unreachable!(),
        }
    }

    pub fn debug(&self) {
        fn debug_op(op: &Op, indent: u8) {
            match op.0.as_ref() {
                OpKind::ConstView { value, view } => {
                    println!("{}CONST VIEW {value} {view}", " ".repeat(indent as usize))
                }
                OpKind::LoadView { view, dtype } => {
                    println!("{}LOAD VIEW {dtype} {view}", " ".repeat(indent as usize))
                }
                OpKind::StoreView { x, view } => {
                    println!("{}STORE VIEW {view}", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                }
                OpKind::Cast { x, dtype } => {
                    println!("{}CAST {dtype}", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                }
                OpKind::Unary { x, uop } => {
                    println!("{}{uop:?}", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                }
                OpKind::Binary { x, y, bop } => {
                    println!("{}{bop:?}", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                    debug_op(y, indent + 2);
                }
                OpKind::Reduce { x, rop, num_loops } => {
                    println!("{}{rop:?}, {num_loops}", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                }
                OpKind::LoopIndex { i } => {
                    println!("{}LOOP {i}", " ".repeat(indent as usize));
                }
                OpKind::Load { dtype, index } => {
                    println!("{}LOAD {dtype} with index:", " ".repeat(indent as usize));
                    debug_op(index, indent + 2);
                }
                OpKind::Store { x, index } => {
                    println!("{}LOAD x with index:", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                    debug_op(index, indent + 2);
                }
                OpKind::Const { value } => {
                    println!("{}CONST {value}", " ".repeat(indent as usize))
                }
            }
        }
        println!();
        debug_op(self, 0);
        println!();
    }

    pub fn apply_optimization(&mut self, opt: &Optimization) {
        // TODO
        self.resolve_views();
    }

    pub fn resolve_views(&mut self) {
        match self.0.as_mut() {
            OpKind::ConstView { value, view } => todo!(),
            OpKind::LoadView { view, dtype } => {
                let index = view_index(view);
                self.0 = Box::new(OpKind::Load { dtype: *dtype, index });
            }
            OpKind::StoreView { x, view } => {
                let index = view_index(view);
                x.resolve_views();
                self.0 = Box::new(OpKind::Store { x: x.clone(), index });
            }
            OpKind::Cast { x, .. } | OpKind::Unary { x, .. } | OpKind::Reduce { x, .. } => {
                x.resolve_views();
            }
            OpKind::Binary { x, y, .. } => {
                x.resolve_views();
                y.resolve_views();
            }
            OpKind::Const { .. }
            | OpKind::LoopIndex { .. }
            | OpKind::Load { .. }
            | OpKind::Store { .. } => unreachable!(),
        }

        fn view_index(view: &View) -> Op {
            let mut index = Op::constant(Constant::U32(0));
            for (i, dim) in view.0[0].iter().enumerate() {
                let x = Op::loop_index(i as u32);
                let y = Op::constant(Constant::U32(dim.st as u32));
                let y = Op::binary(x, y, BOp::Mul);
                index = Op::binary(index, y, BOp::Add);
            }
            index
        }
    }
}

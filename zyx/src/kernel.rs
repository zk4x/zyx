use std::hash::BuildHasherDefault;

use crate::{
    DType, Set,
    cache::Optimization,
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    shape::{Axis, Dim},
    view::View,
};

pub struct Kernel {
    pub shape: [Dim; 6],
    pub op: Op,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpKind {
    //Sink { stores: Vec<Op> }, // A way to put multiple stores in one kernel
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
    Block { op: Op, vars: Vec<Op> }, // deduplication block
    Ref { id: usize },               // Reference to variable in deduplication block
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Op(pub(crate) Box<OpKind>);

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
        println!("n = {n}, axes: {axes:?}, permute axes: {:?}", permute_axes);
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
            _ => unreachable!(),
        }
    }

    pub fn shape(&self) -> Vec<Dim> {
        match self.0.as_ref() {
            OpKind::ConstView { view, .. }
            | OpKind::LoadView { view, .. }
            | OpKind::StoreView { view, .. } => view.shape(),
            OpKind::Cast { x, .. }
            | OpKind::Unary { x, .. }
            | OpKind::Binary { x, .. }
            | OpKind::Reduce { x, .. } => x.shape(),
            OpKind::Const { .. }
            | OpKind::LoopIndex { .. }
            | OpKind::Load { .. }
            | OpKind::Store { .. } => unreachable!(),
            _ => unreachable!(),
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
            _ => unreachable!(),
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
            _ => unreachable!(),
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
                    println!("{}STORE x with index:", " ".repeat(indent as usize));
                    debug_op(x, indent + 2);
                    debug_op(index, indent + 2);
                }
                OpKind::Const { value } => {
                    println!("{}CONST {value}", " ".repeat(indent as usize));
                }
                OpKind::Block { op, vars } => {
                    println!("{}BLOCK VARS", " ".repeat(indent as usize));
                    for x in vars {
                        debug_op(x, indent + 2);
                    }
                    println!("{}BLOCK OP", " ".repeat(indent as usize));
                    debug_op(op, indent + 2);
                }
                OpKind::Ref { id } => {
                    println!("{}REF {id}", " ".repeat(indent as usize));
                }
            }
        }
        println!();
        debug_op(self, 0);
        println!();
    }

    fn is_constant(&self) -> bool {
        match self.0.as_ref() {
            OpKind::Const { .. } => true,
            _ => false,
        }
    }

    pub fn apply_optimization(mut self, opt: &Optimization) -> Kernel {
        let n = self.shape().len();
        match opt {
            Optimization::Basic { shape } => {
                self.movement(|view| view.reshape(0..n, &shape));
                self.resolve_views();
                self.constant_fold();
                self.deduplicate();
                Kernel {
                    shape: [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]],
                    op: self,
                }
            }
        }
    }

    pub fn deduplicate(&mut self) {
        // deduplication block is created in every scope, that is kernel scope and within every reduce op
        // this function also does loop invariant code motion
        fn dedup(cache: &mut Set<Op>, op: &mut Op) {
            cache.insert(op.clone());
            match op.0.as_mut() {
                OpKind::ConstView { value, view } => {}
                OpKind::Const { value } => todo!(),
                OpKind::LoopIndex { i } => todo!(),
                OpKind::LoadView { view, dtype } => todo!(),
                OpKind::Load { dtype, index } => todo!(),
                OpKind::StoreView { x, view } => todo!(),
                OpKind::Store { x, index } => todo!(),
                OpKind::Cast { x, dtype } => todo!(),
                OpKind::Unary { x, uop } => todo!(),
                OpKind::Binary { x, y, bop } => todo!(),
                OpKind::Reduce { x, rop, num_loops } => todo!(),
                OpKind::Block { .. } => unreachable!(),
                OpKind::Ref { .. } => unreachable!(),
            }
        }
        let mut cache = Set::with_hasher(BuildHasherDefault::default());
        dedup(&mut cache, self);
    }

    pub fn constant_fold(&mut self) {
        match self.0.as_mut() {
            OpKind::Const { .. } | OpKind::LoopIndex { .. } => {}
            OpKind::Load { index, .. } => {
                index.constant_fold();
            }
            OpKind::Store { x, index } => {
                x.constant_fold();
                index.constant_fold();
            }
            OpKind::Cast { x, dtype } => {
                x.constant_fold();
                if let OpKind::Const { value } = x.0.as_mut() {
                    *self.0 = OpKind::Const { value: value.cast(*dtype) };
                }
            }
            OpKind::Unary { x, uop } => {
                x.constant_fold();
                if let OpKind::Const { value } = x.0.as_mut() {
                    *self.0 = OpKind::Const { value: value.unary(*uop) };
                }
            }
            OpKind::Binary { x, y, bop } => {
                x.constant_fold();
                y.constant_fold();
                if let OpKind::Const { value: x_value } = *x.0
                    && let OpKind::Const { value: y_value } = *y.0
                {
                    *self.0 = OpKind::Const { value: Constant::binary(x_value, y_value, *bop) };
                    return;
                }
                match bop {
                    BOp::Add => {
                        if let OpKind::Const { value: x_value } = *x.0 {
                            if x_value.is_zero() {
                                *self.0 = (*y.0).clone();
                                return;
                            }
                        }
                        if let OpKind::Const { value: y_value } = *y.0 {
                            if y_value.is_zero() {
                                *self.0 = (*x.0).clone();
                                return;
                            }
                        }
                    }
                    BOp::Sub => {
                        if let OpKind::Const { value: x_value } = *x.0 {
                            if x_value.is_zero() {
                                *self.0 = OpKind::Unary { x: y.clone(), uop: UOp::Not };
                                return;
                            }
                        }
                        if let OpKind::Const { value: x_value } = *x.0 {
                            if x_value.is_zero() {
                                *self.0 = (*x.0).clone();
                                return;
                            }
                        }
                    }
                    BOp::Mul => todo!(),
                    BOp::Div => todo!(),
                    BOp::Pow => todo!(),
                    BOp::Mod => todo!(),
                    BOp::Cmplt => todo!(),
                    BOp::Cmpgt => todo!(),
                    BOp::Max => todo!(),
                    BOp::Or => todo!(),
                    BOp::And => todo!(),
                    BOp::BitXor => todo!(),
                    BOp::BitOr => todo!(),
                    BOp::BitAnd => todo!(),
                    BOp::BitShiftLeft => todo!(),
                    BOp::BitShiftRight => todo!(),
                    BOp::NotEq => todo!(),
                }
            }
            OpKind::Reduce { x, rop, num_loops } => {
                // TODO if num_loops is small enough
            }
            OpKind::Block { .. }
            | OpKind::Ref { .. }
            | OpKind::ConstView { .. }
            | OpKind::LoadView { .. }
            | OpKind::StoreView { .. } => {
                unreachable!()
            }
        }
    }

    pub fn resolve_views(&mut self) {
        match self.0.as_mut() {
            OpKind::ConstView { value, view } => todo!(),
            OpKind::LoadView { view, dtype } => {
                let index = view_index(view);
                *self.0 = OpKind::Load { dtype: *dtype, index };
            }
            OpKind::StoreView { x, view } => {
                let index = view_index(view);
                x.resolve_views();
                *self.0 = OpKind::Store { x: x.clone(), index };
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
            _ => unreachable!(),
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

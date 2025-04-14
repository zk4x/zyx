use crate::{dtype::Constant, node::{BOp, UOp}};

use super::{IRCompiler, IROp, Reg};

impl IRCompiler {
    /// This includes elimination of useless ops, i.e. y = x*1
    /// and includes peephole optimizations
    #[allow(clippy::match_on_vec_items)]
    #[allow(clippy::single_match)]
    pub fn constant_folding_and_propagation(&mut self) {
        let mut i = 0;
        while i < self.ops.len() {
            #[allow(clippy::match_same_arms)]
            match self.ops[i] {
                IROp::Binary { z, x, y, bop } => match (x, y) {
                    (Reg::Var(_), Reg::Var(_)) => {}
                    (Reg::Var(xv), Reg::Const(yv)) => {
                        if yv.is_zero() {
                            match bop {
                                BOp::Mul | BOp::And | BOp::BitAnd => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Const(yv), 0);
                                }
                                BOp::Div | BOp::Mod => panic!("Division by zero constant"),
                                BOp::Pow | BOp::Or => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Const(yv.dtype().one_constant()), 0);
                                }
                                BOp::Add | BOp::Sub | BOp::BitXor | BOp::BitOr => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Var(xv), 0);
                                }
                                BOp::Max => self.ops[i] = IROp::Unary { z, x: xv, uop: UOp::ReLU },
                                BOp::NotEq
                                | BOp::Cmpgt
                                | BOp::Cmplt
                                | BOp::BitShiftLeft
                                | BOp::BitShiftRight => {}
                            }
                        } else if yv.is_one() {
                            match bop {
                                BOp::Mul | BOp::Div | BOp::Pow => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Var(xv), 0);
                                }
                                BOp::Mod => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Const(yv.dtype().zero_constant()), 0);
                                }
                                BOp::BitOr => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Const(yv), 0);
                                }
                                BOp::BitXor
                                | BOp::BitAnd
                                | BOp::Cmplt
                                | BOp::And
                                | BOp::Or
                                | BOp::Cmpgt
                                | BOp::Max
                                | BOp::Add
                                | BOp::Sub
                                | BOp::NotEq
                                | BOp::BitShiftLeft
                                | BOp::BitShiftRight => {}
                            }
                        } else if yv.is_two() {
                            match bop {
                                BOp::Mul => {
                                    if yv.dtype().is_shiftable() {
                                        self.ops[i] = IROp::Binary {
                                            z,
                                            x: Reg::Var(xv),
                                            y: Reg::Const(yv.dtype().one_constant()),
                                            bop: BOp::BitShiftLeft,
                                        };
                                    }
                                }
                                BOp::Div => {
                                    if yv.dtype().is_shiftable() {
                                        self.ops[i] = IROp::Binary {
                                            z,
                                            x: Reg::Var(xv),
                                            y: Reg::Const(yv.dtype().one_constant()),
                                            bop: BOp::BitShiftRight,
                                        };
                                    }
                                }
                                BOp::Pow => {
                                    self.ops[i] = IROp::Binary {
                                        z,
                                        x: Reg::Var(xv),
                                        y: Reg::Var(xv),
                                        bop: BOp::Mul,
                                    };
                                }
                                BOp::Mod => {
                                    if yv.dtype().is_shiftable() {
                                        self.ops[i] = IROp::Binary {
                                            z,
                                            x: Reg::Var(xv),
                                            y: Reg::Const(yv.dtype().one_constant()),
                                            bop: BOp::BitAnd,
                                        };
                                    }
                                }
                                BOp::Add => {}
                                BOp::Sub => {}
                                BOp::Cmplt => {}
                                BOp::Cmpgt => {}
                                BOp::Max => todo!(),
                                BOp::Or => todo!(),
                                BOp::And => todo!(),
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                                BOp::NotEq => todo!(),
                                BOp::BitShiftLeft => {}
                                BOp::BitShiftRight => {}
                            }
                        }
                    }
                    (Reg::Const(xv), Reg::Var(yv)) => {
                        if xv.is_zero() {
                            match bop {
                                BOp::Add => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Var(yv), 0);
                                }
                                BOp::Sub => self.ops[i] = IROp::Unary { z, x: yv, uop: UOp::Neg },
                                BOp::Mul | BOp::Div | BOp::Pow | BOp::Mod | BOp::And => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Const(xv), 0);
                                }
                                BOp::Cmplt => {}
                                BOp::Cmpgt => {}
                                BOp::Max => self.ops[i] = IROp::Unary { z, x: yv, uop: UOp::ReLU },
                                BOp::Or => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Var(yv), 0);
                                }
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                                BOp::NotEq => todo!(),
                                BOp::BitShiftLeft => todo!(),
                                BOp::BitShiftRight => todo!(),
                            }
                        } else if xv.is_one() {
                            match bop {
                                BOp::Add => {}
                                BOp::Sub => {}
                                BOp::Mul => {
                                    self.ops.remove(i);
                                    i -= 1;
                                    self.replace(z, Reg::Var(yv), 0);
                                }
                                BOp::Div => {
                                    self.ops[i] = IROp::Unary { z, x: yv, uop: UOp::Reciprocal }
                                }
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
                                BOp::NotEq => todo!(),
                                BOp::BitShiftLeft => todo!(),
                                BOp::BitShiftRight => todo!(),
                            }
                        } else if xv.is_two() {
                            match bop {
                                BOp::Add => {}
                                BOp::Sub => todo!(),
                                BOp::Mul => {
                                    self.ops[i] = IROp::Binary { z, x: y, y, bop: BOp::Add }
                                }
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
                    }
                    (Reg::Const(x), Reg::Const(y)) => {
                        self.ops.remove(i);
                        i -= 1;
                        self.replace(z, Reg::Const(Constant::binary(x, y, bop)), 0);
                    }
                },
                IROp::MAdd { .. } => {}
                IROp::SetLocal { .. }
                | IROp::Set { .. }
                | IROp::Cast { .. }
                | IROp::Unary { .. }
                | IROp::Loop { .. }
                | IROp::EndLoop { .. }
                | IROp::Load { .. }
                | IROp::Store { .. }
                | IROp::Barrier { .. } => {}
            }
            i += 1;
        }
    }
}
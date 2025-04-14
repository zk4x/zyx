use crate::dtype::Constant;
use super::{IRCompiler, IROp, Reg};

impl IRCompiler {
    pub fn global_loop_unrolling(&mut self) {
        let mut op_i = 6;
        while op_i > 0 {
            op_i -= 1;
            if let IROp::Loop { id, len } = self.ops[op_i] {
                if len == 1 {
                    self.replace(id, Reg::Const(Constant::U64(0)), 0);
                }
            }
        }
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn loop_unrolling(&mut self) {
        // TODO after unroll of a loop, constant propagate the accumulator
        let mut op_i = self.ops.len();
        let mut last_end_loop = Vec::new();
        while op_i > 6 {
            op_i -= 1;
            if let IROp::EndLoop { .. } = self.ops[op_i] {
                last_end_loop.push(op_i);
            }
            if let IROp::Loop { id, len } = self.ops[op_i] {
                if len < 32 {
                    let end = last_end_loop.pop().unwrap();
                    self.ops.remove(end);
                    self.ops.remove(op_i);
                    let ops2: Vec<IROp> = self.ops[op_i..end - 1].into();

                    self.replace(id, Reg::Const(Constant::U64(len as u64 - 1)), op_i);
                    /*println!();
                    for op in &ops {
                        println!("{op:?}");
                    }
                    println!();*/
                    if let Some(tc) = self.ops[op_i..].iter().find_map(|op| match op {
                        IROp::Set { z, .. }
                        | IROp::Cast { z, .. }
                        | IROp::Unary { z, .. }
                        | IROp::Binary { z, .. }
                        | IROp::MAdd { z, .. } => Some(z - 1),
                        IROp::SetLocal { .. }
                        | IROp::Loop { .. }
                        | IROp::EndLoop { .. }
                        | IROp::Barrier { .. }
                        | IROp::Store { .. }
                        | IROp::Load { .. } => None,
                    }) {
                        let ta: u16 = ops2.len().try_into().unwrap();
                        for i in (0..len - 1).rev() {
                            // First we need to increase ids of all variables past loop declaration
                            for i in op_i..self.ops.len() - 6 {
                                #[allow(clippy::match_on_vec_items)]
                                match self.ops[i] {
                                    IROp::Load { ref mut z, ref mut offset, .. } => {
                                        if let Reg::Var(offset) = offset {
                                            if *offset > tc {
                                                *offset += ta;
                                            }
                                        }
                                        if *z > tc {
                                            *z += ta;
                                        }
                                    }
                                    IROp::Store { ref mut offset, ref mut x, .. } => {
                                        if let Reg::Var(offset) = offset {
                                            if *offset > tc {
                                                *offset += ta;
                                            }
                                        }
                                        if let Reg::Var(x) = x {
                                            if *x > tc {
                                                *x += ta;
                                            }
                                        }
                                    }
                                    IROp::Set { ref mut z, .. } => {
                                        if *z > tc {
                                            *z += ta;
                                        }
                                    }
                                    IROp::Cast { ref mut z, ref mut x, .. }
                                    | IROp::Unary { ref mut z, ref mut x, .. } => {
                                        if *x > tc {
                                            *x += ta;
                                        }
                                        if *z > tc {
                                            *z += ta;
                                        }
                                    }
                                    IROp::Binary { ref mut z, ref mut x, ref mut y, .. } => {
                                        if let Reg::Var(x) = x {
                                            if *x > tc {
                                                *x += ta;
                                            }
                                        }
                                        if let Reg::Var(y) = y {
                                            if *y > tc {
                                                *y += ta;
                                            }
                                        }
                                        if *z > tc {
                                            *z += ta;
                                        }
                                    }
                                    IROp::MAdd { ref mut z, ref mut a, ref mut b, ref mut c } => {
                                        if let Reg::Var(a) = a {
                                            if *a > tc {
                                                *a += ta;
                                            }
                                        }
                                        if let Reg::Var(b) = b {
                                            if *b > tc {
                                                *b += ta;
                                            }
                                        }
                                        if let Reg::Var(c) = c {
                                            if *c > tc {
                                                *c += ta;
                                            }
                                        }
                                        if *z > tc {
                                            *z += ta;
                                        }
                                    }
                                    IROp::Loop { ref mut id, .. }
                                    | IROp::EndLoop { ref mut id, .. } => {
                                        if *id > tc {
                                            *id += ta;
                                        }
                                    }
                                    IROp::SetLocal { .. } | IROp::Barrier { .. } => {}
                                }
                            }
                            // Copy ops
                            let mut ops3 = ops2.clone();
                            while let Some(op) = ops3.pop() {
                                self.ops.insert(op_i, op);
                            }
                            // Replace loop variable
                            self.replace(id, Reg::Const(Constant::U64(i as u64)), op_i);
                        }
                    }
                    let x = isize::try_from(ops2.len()).unwrap()
                        * (isize::try_from(len).unwrap() - 1)
                        - 2;
                    for end in &mut last_end_loop {
                        *end = usize::try_from(isize::try_from(*end).unwrap() + x).unwrap();
                    }
                } else {
                    let _ = last_end_loop.pop().unwrap();
                }
            }
        }
    }
}
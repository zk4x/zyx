use std::collections::BTreeSet;

use super::{IRCompiler, IROp, Reg};

impl IRCompiler {
    // Loop invariant code motion and dependence analysis
    pub fn loop_invariant_code_motion(&mut self) {
        // Make a list of accumulators. These cannot be moved.
        let accs: BTreeSet<u16> = self
            .ops
            .iter()
            .filter_map(|op| {
                if let IROp::Set { z, .. } = op {
                    Some(*z)
                } else {
                    None
                }
            })
            .collect();
        // Go from innermost loop to outermost loop. If there are multiple innermost loops,
        // they can be processed in parallel.
        for op_id in (6..self.ops.len()).rev() {
            if let IROp::Loop { id, .. } = self.ops[op_id] {
                let mut loop_id = op_id;
                // which variables can't be eliminated
                let mut dependents: BTreeSet<u16> = BTreeSet::from([id]);
                let mut inner_loop_counter = 0;
                let mut op_id = loop_id + 1;
                'a: loop {
                    // if operands are not in dependents, move operation before loop
                    #[allow(clippy::match_on_vec_items)]
                    let move_possible: bool = match self.ops[op_id] {
                        IROp::Load { z, offset, .. } => {
                            if let Reg::Var(offset) = offset {
                                if dependents.contains(&offset) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            }
                        }
                        IROp::Store { offset, x, .. } => {
                            let a = if let Reg::Var(offset) = offset {
                                !dependents.contains(&offset)
                            } else {
                                true
                            };
                            let b = if let Reg::Var(x) = x {
                                !dependents.contains(&x)
                            } else {
                                true
                            };
                            a && b
                        }
                        IROp::Set { z, .. } => {
                            dependents.insert(z);
                            false
                        }
                        IROp::Barrier { .. } | IROp::SetLocal { .. } => false,
                        IROp::Cast { z, x, .. } | IROp::Unary { z, x, .. } => {
                            if dependents.contains(&x) {
                                dependents.insert(z);
                                false
                            } else {
                                true
                            }
                        }
                        IROp::Binary { z, x, y, .. } => {
                            let a = if let Reg::Var(x) = x {
                                if dependents.contains(&x) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            let b = if let Reg::Var(y) = y {
                                if dependents.contains(&y) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            let c = !accs.contains(&z);
                            a && b && c
                        }
                        IROp::MAdd { z, a, b, c } => {
                            let a = if let Reg::Var(x) = a {
                                if dependents.contains(&x) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            let b = if let Reg::Var(x) = b {
                                if dependents.contains(&x) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            let c = if let Reg::Var(x) = c {
                                if dependents.contains(&x) {
                                    dependents.insert(z);
                                    false
                                } else {
                                    true
                                }
                            } else {
                                true
                            };
                            let z = !accs.contains(&z);
                            a && b && c && z
                        }
                        IROp::Loop { .. } => {
                            inner_loop_counter += 1;
                            // This is a bit more complicated. We have to check all values
                            // in this loop block and move the loop as a whole.
                            // This is however rarely needed due to way we construct loops,
                            // so we do not need to hurry implementing this.
                            false
                        }
                        IROp::EndLoop { .. } => {
                            if inner_loop_counter == 0 {
                                break 'a;
                            }
                            inner_loop_counter -= 1;
                            false
                        }
                    };
                    //println!("Move possible: {move_possible}");
                    if move_possible && inner_loop_counter == 0 {
                        let op = self.ops.remove(op_id);
                        self.ops.insert(loop_id, op);
                        loop_id += 1;
                    }
                    op_id += 1;
                }
            }
        }
    }
}
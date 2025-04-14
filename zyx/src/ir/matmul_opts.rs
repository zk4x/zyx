use super::{IRCompiler, IROp};

impl IRCompiler {
    pub fn matmul_register_tiling(&mut self) {
        // Move accumulator before register loops, repeat it for all register loops
        {
            // Get number of repetitions
            let num_accs: usize = self.ops[6..9]
                .iter()
                .map(|op| {
                    let IROp::Loop { len, .. } = op else { unreachable!() };
                    len
                })
                .product();
            let IROp::Set { z, value } = self.ops.remove(9) else { unreachable!() };
            // Insert acc back, before reduce loops
            for i in 0..u16::try_from(num_accs).unwrap() {
                self.ops.insert(9, IROp::Set { z: z + i, value });
            }
            // Increase ids of all following variables
        }

        // Move register loops into reduce loop
        {
            // Finish register loops before reduce loop

            // Begin register loops after reduce loop

            // Insert register loops in reduce loop

            // Resulve accumulator accumulation in reduce loop
        }
        
        self.debug();
    }
}
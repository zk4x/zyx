use crate::kernel::{BOp, Kernel, OpId, UOp};

enum TTOp {
    Unary { z: OpId, x: OpId, uop: UOp },
    Binary { z: OpId, x: OpId, y: OpId, bop: BOp },
}

impl Kernel {
    fn generate_tenstorrent2() {
        todo!()
    }
}

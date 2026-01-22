use crate::kernel::Kernel;

impl Kernel {
    fn vectorize(&self) {
        todo!()
    }

    /// This function will unroll define[N] into N x define[1] ops,
    /// so that we can use scalars instead of arrays in registers.
    /// But it can also unrool define[16] into 4 x define[4] for float4 vectors, etc.
    /// May be better to put this in optimizer.
    pub fn unroll_defines(&mut self) {
        // TODO
    }
}

// For WMMA
// We need to have specific size of inner and local work sizes such
//
// 1. work_size.rs, the loads are permuted such that we can use tensor cores by putting
// the local and register loops after the inner reduce loop (in strides).
//
// 2. now we can replace the inner register loops with the tensor core instructions directly,
// nothing else to do, just verify the strides :D
//

use crate::kernel::Kernel;

impl Kernel {
    /// Turns inner loops into tensor core instructions if possible
    fn apply_wmma() {}
}

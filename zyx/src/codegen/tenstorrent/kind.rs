//! What an element-wise op computes: the op IDs `tensor::Eltwise` carries,
//! computed on the SFPU (`crate::sfpu::ops`), bit for bit `burn-flex`'s.
//! `kind_sfpu` there numbers the ops beyond these.
//!
//! These were once the data mover's own FP32 kinds (`tt_isa::dm`); the mover
//! now only moves data, so they live here, with the numbers they had. 8, 9 and
//! 10 -- the column sum, the padding fill and the copy -- are not element-wise
//! ops any more: the sum is `sfpu::reduce`'s, the fill `dm::op::FILL`, the copy
//! `Session::copy`.

/// `a + b`.
pub const ADD: u32 = 1;
/// `a - b`.
pub const SUB: u32 = 2;
/// `a * b`.
pub const MUL: u32 = 3;
/// `a * s`.
pub const MUL_SCALAR: u32 = 4;
/// `max(a, 0)`, as `burn-flex` has it.
pub const RELU: u32 = 5;
/// `a > 0 ? b : 0`: `a` the forward output, `b` the gradient.
pub const RELU_BACKWARD: u32 = 6;
/// `a[r, c] + b[0, c]`: [`ADD`] with `b` one row, broadcast down `a`.
pub const ADD_ROW: u32 = 7;
/// `a + s`. `a - s` is this with `-s`, bit for bit in IEEE arithmetic.
pub const ADD_SCALAR: u32 = 11;
/// The last of these; `crate::sfpu::ops::kind_sfpu` numbers from above it.
pub const LAST: u32 = ADD_SCALAR;

// This module contains tensor implementations of operations defined in ops module.

pub(super) mod add;
pub(super) mod conv;
pub(super) mod div;
pub(super) mod exp;
pub(super) mod expand;
pub(super) mod has_device;
pub(super) mod has_dtype;
pub(super) mod has_shape;
pub(super) mod into_vec;
pub(super) mod ln;
pub(super) mod matmul;
pub(super) mod max;
pub(super) mod min;
pub(super) mod mul;
pub(super) mod neg;
pub(super) mod permute;
pub(super) mod pow;
pub(super) mod relu;
pub(super) mod reshape;
pub(super) mod sub;
pub(super) mod sum;
pub(super) mod tanh;

// Naming scheme for backward function is FunctionName + Backward + letters of the tensor types
// which are parameters:
// S - Storage = DType/Buffer
// V - Variable
// T - Tensor

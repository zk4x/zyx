// This module contains tensor implementations of operations defined in ops module.

pub(super) mod into_vec;
pub(super) mod has_dtype;
pub(super) mod has_device;
pub(super) mod has_shape;
pub(super) mod relu;
pub(super) mod exp;
pub(super) mod ln;
pub(super) mod tanh;
pub(super) mod neg;
pub(super) mod sum;
pub(super) mod max;
pub(super) mod min;
pub(super) mod reshape;
pub(super) mod expand;
pub(super) mod permute;
pub(super) mod add;
pub(super) mod sub;
pub(super) mod mul;
pub(super) mod div;
pub(super) mod pow;
pub(super) mod matmul;
pub(super) mod conv;

// Naming scheme for backward function is FunctionName + Backward + letters of the tensor types
// which are parameters:
// S - Storage = DType/Buffer
// V - Variable
// T - Tensor

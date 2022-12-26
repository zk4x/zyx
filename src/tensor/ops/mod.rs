// This module contains tensor implementations of operations defined in ops module.

mod into_vec;
mod has_dtype;
mod has_device;
mod has_shape;
mod relu;
mod exp;
mod ln;
mod tanh;
mod neg;
mod sum;
mod max;
mod min;
mod reshape;
mod expand;
mod permute;
mod add;
mod sub;
mod mul;
mod div;
mod pow;
mod matmul;
mod conv;

// Naming scheme for backward function is FunctionName + Backward + letters of the tensor types
// which are parameters:
// S - Storage = DType/Buffer
// V - Variable
// T - Tensor

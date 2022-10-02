//! Various implementations of accelerators.
//! The default is zyx::accel::Cpu.
//! 
//! Every accelerator must implement following traits:
//! 
//! std::default::Default
//! std::fmt::Display
//! zyx::ops::*
//! std::ops::{Neg, Add, Sub, Mul, Div}
//! 
//! The zyx::ops module documents (with examples) how these operations
//! should work.
//! 

pub mod cpu;
//pub mod opencl;

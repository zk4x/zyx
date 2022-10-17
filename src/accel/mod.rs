//! Various implementations of accelerators.
//! The default is zyx::accel::cpu::Buffer.
//! 
//! Every accelerator can implement following traits in order to be fully compatible with tensors:
//! 
//! Clone
//! std::default::Default
//! std::ops::{Neg, Add, Sub, Mul, Div}
//! std::ops::Mul<f32>
//! zyx::ops::*
//! 
//! The zyx::ops module documents (with examples) how these operations should work.
//! 
//! All operations take buffer by value. Cloning can be implemented as shallow copying,
//! but you will need to do the necessary reference counting.
//! 

pub mod cpu;
#[cfg(feature = "ndarray")]
pub mod ndarray;

//#[cfg(features = "opencl")]
//pub mod opencl;

/*
An idea about lazy::Buffer

Operations on lazy buffer will be simply saved. If there are unary operations, they can be resolved all at once.
Binary operations add, sub, mul, div, pow can be also resolved at once.
MatMul and Conv are quite complex and probably it is not possible to just store them for later resolution, respectively
there is no advantage in doing so.
There needs to be a function called something like evaluate that will evaluate the buffer.
Beware the cloning, thou. All operations take inputs by value, and cloning is used to generate new buffers, when you put
it on the graph, clones are splits of the graph. So we need to make sure, that upon cloning, both branches get the correct
input and this input is evaluated only once, so maybe cloning should call evaluate function.
Still, clones should be shallow copies, unlike cpu::Buffer, where clones are hard copies. Hard copies are bad for performance.
With shallow copies, and all this lazy evaluation graph, there needs to be reference counting system.
Without cloning, there would be no need for reference counting, because all inputs could be moved to the last non evaluated lazy buffer.
So lazy buffer should employ type state to mark whether it is in evaluated or non evaluated state.

I believe that lazy evaluation is very good way to improve performance, since it highly increases cache locality, especially if there
is lot of unary operations applicable in sequence.

Important note: There are ZERO regressions when using lazy evaluation in this proposed way as compared to using eager evaluation.
In other approaches, there can be some regressions in performance, but not in this approach, though if you can find a possible performance
regression, let me know.
*/

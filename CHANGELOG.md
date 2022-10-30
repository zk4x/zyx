# 0.6.1

- More operations now work with different input and output types (previously it was required that input and output to operations has the same type). After this is done, we can provide full support for ndarray and possibly other libraries that use rank or shape as part of type state. And it will allow us to support const generic shape once support for user defined const generics lands.
- Added basically all clippy lints, with two exeptions: we do not provide examples for everything yet (this is TODO) and we use two unsafe blocks for matrixmultiply crate f32 and f64 operations. With default features there is zero unsafe code.
- Performance improvements by adjusting the requirements notably for tanh backward operation.

# 0.6.0

- Updated documentation
- accel::cpu::Buffer now uses Arc to mitigate the cost of cloning (lowers RAM usage and improves performance)
- Removed tensor::B placeholder, as this was never an intuitive thing from user perspective. This is a change in the public API, so we have to make it a major version bump.

# 0.5.0

- Created Parameters trait
- Some performance improvements for Buffer
- Started adding support for different input-output types in tensor operations

# 0.4.2

- Added more documentation
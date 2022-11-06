# 0.7.0
- BREAKING removed RefCells from Variable implementation. Variable is now fully zero cost abstraction.
- BREAKING rewrote optimizer API due to the change in the Variable implementation. This API however makes a lot of sense. The Variables are stored in network and accessed by calling .parameters(). Optimizers no longer store references to parameters. Instead gradients are zeroed using network.parameters().zero_grad() and optimizer is used like this: network.parameters().step(&optimizer). I would prefer optimizer.step(network.parameters()), but under the hood that seemed to require more work, so we went with the former way. So parameters are mutably borrowed during those two function calls instead of using RefCell and hiding it from the user. From this is seems that PyTorch way of doing this is more of a backward compatibility thing than a good way of doing this, but correct me if I am wrong.
- TL;DR with this release, the library is actual zero cost abstraction

# 0.6.3

- Finished rewrite of tensor::ops to support outputs different from inputs and different inputs in binary operations.
- Performance improvements

# 0.6.2

- Found some incompatibility of some lints with stable channel, so we removed them.
- Continuing to support more operations with different input and output values.

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
# 0.9.0
- BREAKING Rewritten shape to now be stack only, since until now we used heap allocated shape. On description how shape works now, look at the documentation.
- BREAKING Rewritten the way optimizers are handled. There are no more general requirements for all optimizers together, now every optimizers defines it's own requirements and is implemented separately. This gets rid of redundant requirements, but requires more code to be written.
- BREAKING Rewritten the way parameters are handled. This is together with optimizers change. Not only allows us to use different optimizers with separate requirements, it also allows us to define ways to read and mutate parameters. This can be used for loading and saving to IO as well as initializing from different random distributions and so on.
- Although all of the changes above are technically breaking, they do not require you to change the way you write your programs. For example shape is now part of cpu::Buffer generics (previously cpu::Buffer<T>, now cpu::Buffer<T, Sh>), but shape can always be inferred, so you will at most need to write cpu::Buffer<f32, _> instead of cpu::Buffer<f32>.
- Removed no_std claim, because it is not true, since we rely on rayon which requires std. In future, we will add std feature enabled by default and make rayon optinal dependency. Shouldn't be too difficult to do, since rayon is required only for cpu::Buffer. For no_std support we also need to make rand optional or create our own random seed. Rand is required by RandInit and UniformInit, so again it is not widely used.
- Overall the way we rewritten shape and optimizer means that they are both now more modular, but require more code to be written. It should be now easier to add features as non-breaking changes.
- ndarray has not been yet rewritten to be compatible with the changes

# 0.8.0
- BREAKING Change of optimizer API, look at the docs to see how it works now
- BREAKING Variable's gradient must now have the same type as it's data.
- BREAKING Variable's .grad() function returns &Gradient<G>, which is new unit type to encapsulate stored values.
- BREAKING .grad_fn() can now longer be called on Tensor
- added support for no_std environments, we still require alloc crate

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
- device::cpu::Buffer now uses Arc to mitigate the cost of cloning (lowers RAM usage and improves performance)
- Removed tensor::B placeholder, as this was never an intuitive thing from user perspective. This is a change in the public API, so we have to make it a major version bump.

# 0.5.0

- Created Parameters trait
- Some performance improvements for Buffer
- Started adding support for different input-output types in tensor operations

# 0.4.2

- Added more documentation

# This is the manual on adding new backends to zyx

## Initialization

In initialization functions backends needs to return any number of memory pools and devices.
Each device can have associated any number of queues.

## Memory pools

Each memory pool needs to be able to load data from cpu `&[u8]`, store them into cpu `&mut [u8]`
as well as be capable of copying from one memory pool to another.

For example CUDA driver returns one memory pool per each gpu. Each gpu can copy data from and to cpu
as well as from and to any other CUDA gpu.

When dealing with transfers between different backends, for example from CUDA gpu to Intel GPU,
zyx moves data through CPU. Afaik there is no faster way.

## Devices

Each device has a number of associated queues. Devices compile IR kernels into machine code.

## Queues

Queues are responsible for dispatching compiled kernels and they need to have the ability to synchronize
- block all running programs till completion.

## Programs

Programs (kernels) are blocks of machine code running on devices. Programs are compiled from IR.
IR contains very small number of very simple operations. These are load, store, unary, binary, barrier and loop.
There are no branching operations other than loops. Load and store access device memory accessible by pointers.
These are global variables, local variables and register variables with more than one value (vectorized datatypes
are exempt).

## Datatypes

Devices can support any number of datatypes. If kernel includes unsupported datatype, backend should return
compilation error.

## Conclusion

This is pretty much all that is needed to add new backends to zyx. If you have any problems adding support
for your device, please do not hesitate to create an issue on (github)[github.com/zk4x/zyx], we're happy to assist you.
Hardware support is the second primary goal of zyx (first one is correctness, obviously).

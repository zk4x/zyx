# Architecture

This document describes limitations and design choices behind zyx.

## Hardware support

Zyx supports many hardware backends through singular intermediate representation, which is very close to assembly
and simply uses enum as an op.

## Error handling

Zyx uses simple method to differentiate when to panic and when to return a result. If a user provides incorrect
input, function should return result. Everything else should be panic. That is if zyx panics, it is a bug.
All panics and assertions (there are many and we will add more) check for consistency of invariants that cannot
be broken. Breaking any of these invariants puts zyx into irrecoverable state, thus the only right thing to do
is to immediately stop execution. One other option when panic happens is in case of hardware failure.
Zyx already detects hardware devices at runtime and disallows explicit programming for cpu or gpu only.
However zyx currently assumes that hardware configuration stays constant as long as at least one tensor exists.

# Single global mutable state

All tensors are part of single global mutable struct. It is preferable to have decentralized systems, but zyx ultimately
has to run the graph on physical devices, which are global state. Zyx tracks their performance, compute load
and memory usage, so having it all in single global mutable struct is the obvious choice.

## How does it actually work?

Zyx creates graph of nodes at runtime. No calculations are performend until explicit realization. This ensures
zero unnecessary allocations and is particularly important for backpropagation with implicit tracing of all
tensors.

Graph realization consits of these steps:

### 1. Graph tracing

Depth first search is used to find all tensors that are needed to evaluate tensors requested by Tensor::realize
function. Dead tensors are optimized away using constant folding.

### 2. Kernel generation

In this stage tensor ops are fused into large kernels.

### 3. Scheduling kernel to devices

Once kernels are generated, a device or a set of devices are picked that will run it. Currently the heurestic
is very primitive, but in the future this can contain complex decision process including automatic sharding.

### 4. Kernel optimization

An optimizer then takes the kernel and runs tree search trying different optimizations possible with the kernel
and the device it was scheduled for. There can be thousands of different work sizes and other optimizations
that can be applied for each kernel. This search tries it's best to find the best kernel version with as few
tries as possible, but if you want to really want to push your hardware from 80% to 90%, it will take some time,
especially for large kernels.

### 5. Kernel compilation

Finally kernels are compiled into respective backends - CUDA, OpenCL, WGPU, ...
Compilation from IR into backends is straightforward and usually the whole mapping function is about 100 lines of code
per backend.

## Conclusion

As you can see, zyx uses JIT kernel fusion, search and compilation with fully dynamic graph. Optimized kernels
are cached, but if you want to achieve maximum speed, static graph can be beneficial, because then kernel
generator won't need to run during every iteration of training loop. Also more constant folding may be possible
with static graph.

# How to optimize GPU compute kernels in zyx?

We can classify kernels into three categories:

## 1. kernels without loops (only global and local work size) - these are rare (due to kernelizer being good at fusion) and simple to optimize. For now just having work size search is enough.

## 2. kernels with inner loop and large global work size (roughly >32)

This includes matmul and convolution

Possible optimizations:
- local memory tiling
- register memory tiling
- combination of local and register memory tiling
- tensor cores/WMMA
- other ...


## 3. kernels with inner loop and small global work size (about <32)

These are pure reduce kernels

Possible optimizations

- local memory accumulator
- register memory accumulator
- other ...

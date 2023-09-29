# Limits

This file describes limitations of Zyx.

## Hardware

Zyx runs on all platforms supported by rust. Zyx is no-std, but requires alloc.

## Library limitations

There can be at most u32::MAX-1 nodes. Each tensor has one node for data and optionally another node for gradient.
Each node can have at most u8::MAX references to it. For example cloning tensor increses ref count by one.

## Allocation sizes

There is small fixed allocation of some 200 bytes per context.

As for variable allocation, each node takes 48 bytes. Tensor takes 24 bytes.

As for actual values, allocation is as lazy as evaluation and thus all allocations are explicitly made only in functions that could return OOM error.
Currently this is realize function and optim.step function.

There is an exception to this rule when loading from disk. This immediatelly loads all data into Box (as if you called tensor_from_iter and passed Box<[f32]> or Vec<f32> as iterator).
Such data is loaded in RAM and not on the actual device (i. e. no VRAM allocation).

Since Zyx evaluates and allocates lazily and explicitly when you call realize, it can theoretically optimize out all intermediate allocations and only allocate actual data for parameters.
In practice it is good to have many intermediate buffers to speed up calculations. This depends on the actual backend, but currently all Zyx backends drop intermediate buffers as soon as they are not needed.
This gives small memory footprint especially during training (backpropagation), since Zyx will wait till backward pass is defined and not store buffers for later use during backpropagation like other ML libraries.

## OpenCL

Zyx supports only one opencl version - 1.2
This is deliberate decision as most hardware supports this version.
Primary drawback is that we can not use shared buffers (SVM). This currently does not seem that important, as gpu will simply use RAM when it runs low on VRAM.

Zyx works within single platform. In the future we probably will add support for multiple platforms, but this will be implemented as different
devices, therefore you will need to manually move tensors between OpenCL platforms (similar to moving tensors between CPU and GPU).

Zyx will create 8 queues on each OpenCL device. If your device supports fewer queues, this can be manually changed. Please contact the developer or change value queues_per_device in OpenCLDev::new function.
Runtime setting of this value is only available in OpenCL 2.0.

# zyx-opencl

OpenCL backend for zyx machine learning library.

This backend implements:
- shape specialization
- ops fusion

If wondering which zyx backend to use, try using this backend first.

For README and source code, please visit [github](https://www.github.com/zk4x/zyx).

For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).

# Cargo features

- std - enables zyx-core/std
- debug1 - enables printing of information about devices and prints compiled opencl kernels
- CL_VERSION_1_1 - enables opencl v1.1
- CL_VERSION_1_2 - enables opencl v1.2
- CL_VERSION_2_1 - enables opencl v2.1

# zyx-cpu

Cpu backend of zyx machine learning library.

This backend is written in pure rust and runs only on the cpu. It is not currently very fast, because it does not fuse
operations together. It should be used mostly as reference implementation for other backends.

For README, quick tutorial and source code, please visit [https://www.github.com/zk4x/zyx].

For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).

# Cargo features

std - enables multithreading
    - enables zyx-core std

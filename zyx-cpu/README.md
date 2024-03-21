# zyx-cpu

Cpu backend of zyx machine learning library.

This backend is written in pure rust and runs only on the cpu. It is not currently very fast, because it does not fuse
operations together. It should be used mostly as reference implementation for other backends.

For README and source code, please visit [github](https://www.github.com/zk4x/zyx).

For more details, there is a [book](https://zk4x.github.io/zyx).

# Cargo features

- std - enables multithreading and zyx-core/std

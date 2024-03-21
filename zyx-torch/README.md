# zyx-torch

Libtorch backend for zyx machine learning library.

Libtorch backend uses libtorch c++ library for executing graph.

This backend needs to have access to libtorch c++ library. zyx-torch uses tch-rs library
as it's backend, therefore following methods can be used to install libtorch:

LibTorch can be downloaded directly from pytorch [website](https://pytorch.org/get-started/locally/). Select stable, your OS, LibTorch and your preferred compute platform.
Download cxx11 ABI. Once downloaded, extract the archive to some folder. Then run following commands and replace
/path/to/libtorch with folder where you extracted libtorch. Use absolute path. This sets necessary environment variables for libtorch.
Note that environment variables get reset when you reboot your computer, so you need to reexport them again,
or put them into some script that runs automatically.
```shell
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

Please read pytorch documentation for all prerequisities, you may need to install c++ compiler or gpu drivers.

If you have pytorch installed, the above environment variables will interfere with it. In that case, you may prefer
to use pytorch's libtorch,
```shell
export LIBTORCH_USE_PYTORCH=1
```

If neither of those options works for you, zyx-torch can install libtorch automatically, by enabling download-libtorch feature.

For README and source code, please visit [github](https://www.github.com/zk4x/zyx).

For more details, there is a [book](https://zk4x.github.io/zyx).

# Cargo features

- std - enables zyx-core/std
- download-libtorch - installs libtorch

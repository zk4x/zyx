# zyx-torch

Libtorch backend for zyx machine learning library.

Libtorch backend uses libtorch c++ library for executing graph. You may find that when you use libtorch with zyx,
you may get better performance than using libtorch directly. This is because zyx optimizes graph of operations before
sending it to libtorch. There may also be particularly noticeable reduction in memory usage.

This backend needs to have access to libtorch c++ library. LibTorch can be downloaded directly from pytorch
[website](https://pytorch.org/get-started/locally/). Select stable, your OS, LibTorch and your preferred compute platform.
Download cxx11 ABI. Once downloaded, extract the archive to some folder. Then run following commands and replace
/path/to/libtorch with folder where you extracted libtorch. Use absolute path. This sets necessary environment variables for libtorch.
Note that environment variables get reset when you reboot your computer, so you need to reexport them again,
or put them into some script that runs automatically.
```shell
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

Please read pytorch documentation for all prerequisities, you may need to install c++ compiler or gpu drivers.

# Cargo features

std - enables zyx-core/std

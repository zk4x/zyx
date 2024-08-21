# How to install zyx

Zyx supports multiple accelerators. Each comes with it's own installation
instructions.

To install zyx by itself, which gives you cpu support, just run:
```shell
cargo add zyx
```

## CUDA

CUDA is toolkit for running code on NVIDIA gpus.

To install cuda toolkit, please refer to Nvidia's official [website](https://developer.nvidia.com/cuda-downloads).

Zyx uses only CUDA's driver API, so you don't need to install runtime API.
Zyx also needs NVRTC, so it runs on all devices with CUDA compute capability 2.0
and above (GTX 465 and above).

## HSA

HSA is runtime that for running code on AMD gpus.

To install HSA, you should install [ROCM software](https://rocm.docs.amd.com/en/latest/) from AMD.

Here is also a list of commands to install ROCM on some linux distributions.
This is sufficient to get zyx running, but you may also want to install some other packages if you want to work with ROCM outside of zyx.
Debian
```sh
sudo apt install rocm-device-libs
```
Fedora
```sh
sudo dnf install rocm-runtime
```
Arch
```sh
sudo pacman -S rocm-language-runtime
```

## OpenCL

OpenCL is stable programming language that runs on large number of devices,

You can use different runtimes for OpenCL.

Clover runtime (mostly for GPU's before RDNA, e.g. RX 580)
Debian
```sh
sudo apt install mesa-opencl-icd opencl-headers
```
Fedora
```sh
sudo dnf install mesa-libOpenCL opencl-headers
```
Arch
```sh
sudo pacman -S openc-clover-mesa opencl-headers
```

AMD's ROCM runtime:
Fedora
```sh
sudo dnf install rocm-opencl opencl-headers
```
Arch
```sh
sudo pacman -S rocm-opencl-sdk opencl-headers
```

POCL runtime that compiles to LLVM and uses CPU (very stable, good for testing).
Debian
```sh
sudo apt install pocl-opencl-icd opencl-headers
```
Fedora
```sh
sudo dnf install pocl opencl-headers
```
Arch
```sh
sudo pacman -S pocl opencl-headers
```

OpenCL also runs on FPGAs and many custom accelerators. Please raise an issue on github to add installation instructions here for your particular platform.

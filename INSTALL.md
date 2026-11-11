# Installing zyx

## From crates.io

```bash
cargo add zyx
# optional companion crates
cargo add zyx-nn zyx-optim
```

zyx needs Rust **1.88** or newer. Linux is the primary target; other `*nix`
systems should work.

## From PyPI

```bash
pip install zyx-py
```

The package is named `zyx-py` — **`pip install zyx` installs an unrelated
project**. The import name is still `zyx`. `numpy` is optional and only needed
for `.numpy()`.

See [zyx-py/README.md](zyx-py/README.md) for prebuilt-wheel platforms and for
building the bindings with `maturin`.

## Backend prerequisites

zyx itself is pure Rust: backends load their runtime libraries at load time, so
nothing is linked at build time. The optional Tenstorrent backend is the one
build-time exception: it compiles RISC-V firmware and needs the target and LLVM
tools listed below.

| Backend | Needs | Debian/Ubuntu | Notes |
|---------|-------|---------------|-------|
| C | a C compiler: `clang-11`, `clang`, `gcc`, or `cc` (first found) | `sudo apt install clang` or `sudo apt install build-essential` | always available if one exists |
| CUDA | NVIDIA driver (`libcuda.so.1`) and NVRTC (`libnvrtc.so`) | driver from NVIDIA; `sudo apt install nvidia-cuda-toolkit` (or NVIDIA's toolkit) | `ZYX_PTX=1` emits PTX directly and skips NVRTC; cuDNN is optional (`"cuda": { "cudnn": true }`) |
| OpenCL | ICD loader (`libOpenCL.so`) plus a vendor ICD | `sudo apt install ocl-icd-libopencl1`; CPU-only: `sudo apt install pocl-opencl-icd` | |
| Vulkan | `libvulkan.so.1` plus a Vulkan driver | `sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools` | |
| WGPU | cargo feature `wgpu` plus a wgpu backend | build with `--features wgpu` | uses Vulkan/Metal/DX12/GLES; large optional dependency tree |
| Tenstorrent | cargo feature `tenstorrent`, the `libttsim_bh.so` simulator (obtain it from Tenstorrent), the `riscv32im-unknown-none-elf` target, and `llvm-tools` | `rustup target add riscv32im-unknown-none-elf`; `rustup component add llvm-tools` | build with `--features tenstorrent`; set `TT_TTSIM_LIB=/path/to/libttsim_bh.so`; default is `<parent-of-repo>/vendor/libttsim_bh.so`; multi-chip builds use `TT_TTSIM_LIB_X2` / `TT_TTSIM_LIB_X4` |

With no C compiler and no GPU runtime, zyx has no devices. To see what was
found:

```bash
ZYX_DEBUG=1 cargo run      # devices + configuration
```

Device selection and autotune live in [CONFIG.md](CONFIG.md); other diagnostics
are in [ENV_VARS.md](ENV_VARS.md).

## Building from source

The Tenstorrent firmware lives in `zyx/firmware/tt-firmware` and is built by
`zyx/build.rs` only when the `tenstorrent` feature is enabled; the ISA tables
come from the [`tt-isa`](https://github.com/zk4x/tt-isa) crate. No sibling
checkout is required:

```bash
git clone https://github.com/zk4x/zyx
cd zyx/zyx
cargo build
```

The Tenstorrent backend additionally needs the RISC-V target and LLVM tools
that `build.rs` uses for the firmware images:

```bash
rustup target add riscv32im-unknown-none-elf
rustup component add llvm-tools
cargo build --features tenstorrent
```

Run the tests (no GPU needed):

```bash
cd zyx/zyx
AGENT=1 ZYX_DEV=cpu cargo test --lib --tests -- --skip mnist
AGENT=1 cargo test --doc
```

`--skip mnist` leaves out the integration test that needs
`examples/data/mnist_dataset.safetensors`; `run_ci.sh` uses the same filter.

Run the full local gate (unit and integration tests, doctests, private-item
docs, formatting, examples, Python tests when available, and the architecture
book) from the repository root:

```bash
./run_ci.sh
```

Try the examples:

```bash
cd examples
./tiny-net/run.sh
```

## Python from source

If there is no prebuilt wheel for your platform, build the extension with
`maturin`:

```bash
git clone https://github.com/zk4x/zyx
pip install maturin
maturin build --release -m zyx/zyx-py/Cargo.toml
```

See [zyx-py/README.md](zyx-py/README.md) for details.

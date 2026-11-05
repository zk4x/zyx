# zyx Examples

Demo models implemented with zyx, roughly simplest → hardest. Every directory
listed here except `qwen3.8-27b` is a member of the `examples/` cargo workspace
and has a `run.sh`.

| # | Example | Package | Data / model | What it shows |
|---|---------|---------|--------------|---------------|
| 1 | `tiny-net` | `tiny-net` | — | manual tensors + SGD |
| 2 | `tiny-net2` | `tiny-net2` | — | `zyx_nn::Linear` module + SGD |
| 3 | `readme-test` | `readme-test` | — | transformer block (attention, LayerNorm, AdamW) |
| 4 | `rnn` | `rnn` | — | `RNNCell` training loop with `Tape` autograd |
| 5 | `mnist` | `mnist` | `data/mnist_dataset.safetensors` | MLP training on MNIST |
| 6 | `mnist-snn` | `mnist-snn` | `data/mnist_dataset.safetensors` | spiking neural net on MNIST |
| 7 | `nanogpt` | `nanogpt` | `nanogpt/data/input.txt` | character-level GPT (tiny shakespeare) |
| 8 | `phi` | `phi` | `phi/phi1_5-model.safetensors` | phi-1.5 inference |
| 9 | `llama` | `llama` | `models/llama-3.2-3b/` | Llama 3.2 3B Instruct GGUF inference |
| 10 | `tt_gemm` | `tt_gemm` | — | hand-tiled BF16 GEMM for Tenstorrent; see [`tt_gemm/README.md`](tt_gemm/README.md) |
| 11 | `qwen3.8-27b` | `qwen3-8-27b` | 17 GB GGUF (see `plan.md`) | Qwen3.8-27B quantized inference — **work in progress, no runner** |

## Running

Each example's `run.sh`:

1. downloads its data into `examples/data/` and models into `examples/models/`
   (or the example's own directory when its code reads from there),
2. skips the download when the file already exists (idempotent),
3. runs the example in release mode (`cargo run --release`).

```bash
cd examples/tiny-net && ./run.sh
cd examples/llama && ./run.sh --prompt "Hello world"   # args are passed through
```

From the workspace you can also run an example by package name:

```bash
cd examples
cargo run --release -p tiny-net
```

Exceptions:

- **`tt_gemm`** needs a Tenstorrent runtime/board (`TT_METAL_RUNTIME_ROOT` for its
  build); `ZYX_DRY_RUN=1 cargo run -p tt_gemm` compiles without launching.
- **`qwen3.8-27b`** has no `run.sh` and needs its 17 GB quantized GGUF; the plan
  and current status are in [`qwen3.8-27b/plan.md`](qwen3.8-27b/plan.md) and
  [`qwen3.8-27b/tenstorrent_debugging.md`](qwen3.8-27b/tenstorrent_debugging.md).

## Layout

- `examples/data/` — downloaded datasets (`mnist_dataset.safetensors`, ...)
- `examples/models/` — downloaded model weights (`llama-3.2-3b/`, ...)
- `examples/Cargo.toml` — workspace over the runnable examples

Downloaded data and models are ignored by git.

## Dependencies

- Rust and the zyx workspace prerequisites — see
  [INSTALL.md](../INSTALL.md)
- `python3.12` with `torch`, `torchvision`, `safetensors`, `numpy` — MNIST
  download/conversion (`mnist`, `mnist-snn`)
- `huggingface_hub` — `phi`, `llama`, `qwen3.8-27b`
- `curl` — `nanogpt`
- a Tenstorrent runtime/board — `tt_gemm`

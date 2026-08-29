# zyx Examples

Demo models implemented with zyx, ordered **simplest → hardest**:

| # | Example | Data / Model | Notes |
|---|---------|--------------|-------|
| 1 | `probe` | — | smoke test, 7 lines |
| 2 | `tiny-net` | — | manual tensors + SGD |
| 3 | `tiny-net2` | — | `Linear` module + SGD |
| 4 | `readme-test` | — | transformer block (attention, LayerNorm, AdamW) |
| 5 | `rnn` | — | RNNCell training loop (`Tape` autograd) |
| 6 | `mnist` | `data/mnist_dataset.safetensors` | MLP training on MNIST |
| 7 | `mnist-snn` | `data/mnist_dataset.safetensors` | spiking neural net on MNIST |
| 8 | `nanogpt` | `nanogpt/data/input.txt` | character-level GPT training (tiny shakespeare) |
| 9 | `phi` | `phi/phi1_5-model.safetensors` | phi-1.5 inference |
| 10 | `llama` | `models/llama-3.2-3b/` | Llama 3.2 3B GGUF inference |

## Running

Every example has a `run.sh` in its directory. It:

1. downloads any data into `examples/data/` and models into `examples/models/`
   (or the example's own directory when its code reads from there),
2. **skips the download when the file already exists** (idempotent),
3. runs the example in **release mode** (`cargo run --release`).

```bash
cd examples/llama && ./run.sh          # single example
cd examples && ./run_all.sh            # all of them, simplest to hardest
```

`llama/run.sh` also passes through extra CLI args:

```bash
cd examples/llama && ./run.sh --prompt "Hello world"
```

## Layout

- `examples/data/` — datasets (`mnist_dataset.safetensors`, ...)
- `examples/models/` — model weights (`llama-3.2-3b/`, ...)

## Dependencies

- `python3.12` + `torch`, `torchvision`, `safetensors`, `numpy` (mnist download/conversion)
- `huggingface_hub` (phi, llama)
- `curl` (nanogpt)

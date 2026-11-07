# Benchmarks

## llama-3.2-3B GEMMs: prefill and decode

One decoder block's seven projections, both phases: **prefill** is sequence 128,
**decode** is sequence 1 plus `lm_head` (`3072 -> 128256`) — the same set
`zyx-bench llama prefill_gemm` and `zyx-bench llama decode_gemm` run. f16
weights and activations, f32 accumulation, no bias.

Device: NVIDIA GeForce RTX 2060 12GB, compute capability 7.5, driver 13.0.
Phase total is the sum of the projection times, i.e. what one block of the
phase costs in its matmuls.

### Phase totals (end to end)

| phase | framework | total | TFLOP/s |
|---|---|---|---|
| prefill (seq 128) | llama.cpp `c6824a9e4` (ggml CUDA, f16) | 1.2288 ms | 20.97 |
| prefill (seq 128) | torch 2.11.0+cu130 (cuBLAS) | 1.3037 ms | 19.77 |
| prefill (seq 128) | tinygrad 0.11.0 (CUDA, `TinyJit`) | 3.7941 ms | 6.79 |
| decode (seq 1 + lm_head) | torch 2.11.0+cu130 (cuBLAS) | 3.2532 ms | 0.30 |
| decode (seq 1 + lm_head) | llama.cpp `c6824a9e4` (ggml CUDA, f16) | 3.3703 ms | 0.29 |
| decode (seq 1 + lm_head) | tinygrad 0.11.0 (CUDA, `TinyJit`) | 39.6068 ms | 0.02 |

Decode is memory-bound (one row of activations against the whole weight
matrix), so TFLOP/s is small by construction: the useful decode number is the
time. tinygrad's decode total is dominated by `lm_head` (37.5 ms), where it is
~14x slower than the other two.

### Prefill, per projection

| projection | shape | llama.cpp | torch | tinygrad |
|---|---|---|---|---|
| `q_proj` | [128x3072]x[3072x3072] | 0.1205 ms | 0.1485 ms | 0.3557 ms |
| `k_proj` | [128x3072]x[3072x1024] | 0.0560 ms | 0.0552 ms | 0.1349 ms |
| `v_proj` | [128x3072]x[3072x1024] | 0.0568 ms | 0.0547 ms | 0.1355 ms |
| `o_proj` | [128x3072]x[3072x3072] | 0.1200 ms | 0.1507 ms | 0.3525 ms |
| `gate_proj` | [128x3072]x[3072x8192] | 0.3185 ms | 0.2927 ms | 1.0439 ms |
| `up_proj` | [128x3072]x[3072x8192] | 0.3163 ms | 0.2936 ms | 1.0353 ms |
| `down_proj` | [128x8192]x[8192x3072] | 0.2407 ms | 0.3083 ms | 0.7364 ms |

### Decode, per projection

| projection | shape | torch | llama.cpp | tinygrad |
|---|---|---|---|---|
| `q_proj` | [1x3072]x[3072x3072] | 0.0656 ms | 0.0746 ms | 0.1917 ms |
| `k_proj` | [1x3072]x[3072x1024] | 0.0241 ms | 0.0327 ms | 0.2011 ms |
| `v_proj` | [1x3072]x[3072x1024] | 0.0245 ms | 0.0328 ms | 0.2009 ms |
| `o_proj` | [1x3072]x[3072x3072] | 0.0649 ms | 0.0743 ms | 0.1921 ms |
| `gate_proj` | [1x3072]x[3072x8192] | 0.1672 ms | 0.1887 ms | 0.4576 ms |
| `up_proj` | [1x3072]x[3072x8192] | 0.1679 ms | 0.1885 ms | 0.4670 ms |
| `down_proj` | [1x8192]x[8192x3072] | 0.1683 ms | 0.1807 ms | 0.3490 ms |
| `lm_head` | [1x3072]x[3072x128256] | 2.5707 ms | 2.5979 ms | 37.5475 ms |

### Reproduce

```bash
# llama.cpp (f16, per-GEMM): build once against a llama.cpp checkout
LLAMA_CPP=~/Dev/cpp/llama_cpp/llama.cpp
g++ -O2 examples/llama/ggml_gemm.cpp -o /tmp/opencode/ggml_gemm \
  -I$LLAMA_CPP/include -I$LLAMA_CPP/ggml/include \
  -L$LLAMA_CPP/build/bin -lllama -lggml -lggml-base -lggml-cuda \
  -Wl,-rpath,$LLAMA_CPP/build/bin
/tmp/opencode/ggml_gemm

python3.12 examples/llama/gemm_torch.py
python3.12 examples/llama/gemm_tinygrad.py
```

Each script measures one launch per iteration (torch: a batch of 50 launches
and one sync; tinygrad: `TinyJit` + device sync), best of 5 batches, after 20
warmup launches. llama.cpp's numbers are one `ggml_mul_mat` graph per
projection on the CUDA backend, which is the path its f16 prefill takes.

### zyx

Pending a representative budget: `zyx-bench llama prefill_gemm` and `llama
decode_gemm` write per-candidate device times to `zyx-bench/bench.db`
(`llama/prefill_gemm`, `llama/decode_gemm`, with `n_launches`, `grid` and
`code_version` per row). At the default `n_launches=20` the search stops too
early to reach the tensor-core recipes and the numbers are not comparable; at
`n_launches=1000` the run currently hangs in the optimizer (TODO item 1). The
zyx rows go into the tables above as soon as that item is closed.

## Training Step Time

| Model | CPU | GPU | Backend | Step Time |
|-------|-----|-----|---------|-----------|
| MnistNet | i3-12100F / 64GB | RTX 2060 / 12GB | zyx-0.17 (CUDA PTX, frozen tape, 1000 launches/kernel) | 0.216ms |
| MnistNet | i3-12100F / 64GB | RTX 2060 / 12GB | zyx-0.17 (CUDA PTX, frozen tape, 50 launches/kernel) | 0.242ms |
| MnistNet | i3-12100F / 64GB | RTX 2060 / 12GB | torch-2.11.0 (CUDA, `compile(mode="reduce-overhead")`) | 0.745ms |
| MnistNet | i3-12100F / 64GB | RTX 2060 / 12GB | tinygrad-0.11.0 pip (CUDA, `TinyJit`, `BEAM=1`) | 2.913ms |
| MnistNet | i3-12100F / 64GB | RTX 2060 / 12GB | tinygrad main `31aeb20be` (CUDA, `TinyJit`, `BEAM=10`) | 2.943ms |

## Harness

All three run the same loop so the numbers compare directly:

- 200 steps, batch 128, indices drawn on the host and uploaded per step
  (`Tensor::randint(..).to(dev)` / `train_x[samples]`),
- one-hot cross entropy, SGD `lr=0.01, momentum=0.9`, `nesterov=False`,
- a warm-up step before timing (optimizer state must exist; it is also what
  triggers PT2's cudagraph capture and zyx's `Tape::freeze`),
- `loss.item()` only on the steps that print, and *outside* the timed window,
- average reported over steps 100-200, three decimals.

| script | framework |
|---|---|
| `examples/mnist/src/main.rs` | zyx |
| `examples/mnist/mnist_torch.py` | torch |
| `examples/mnist/mnist_tinygrad.py` | tinygrad |

zyx traces one step, freezes it (`Tape::freeze`, `samples` as the only replay
input, dataset and optimizer state captured), then replays. Host-side replay
cost is 0.037ms/step measured with `ZYX_DRY_RUN=1`; the remaining time is CUDA
graph submission plus device execution.

## Kernel-level (MnistNet, batch 128)

Device times from the tuners with a clean cache (`ZYX_DEBUG=1` for zyx,
`DEBUG=2` for tinygrad):

| kernel | zyx | tinygrad |
|---|---|---|
| layer-1 matmul + bias + relu (784-deep reduction, 12.8M MAC) | 104.4us best / 455.5us median | ~149us |
| other kernels (elementwise, reduces, gather) | 4-7us (timed-launch floor) | 15-40us |

Both matmuls are ~25x off the card's fp32 peak (~4us for this reduction), so
the reduction shape (16 lanes per output + barrier tree-reduce over 784) is the
shared weak spot, not the framework.

## Search budgets

| framework | candidates |
|---|---|
| zyx, `n_launches=50` | 22 kernels x 50 = 1,100 timed launches |
| zyx, `n_launches=1000` | 22 kernels x 1000 = 22,000 timed launches |
| tinygrad `BEAM=1` (clean cache) | 1,843 candidate timings over 47 programs |
| tinygrad `BEAM=10` (clean cache) | 13,467 candidate timings over 47 programs |

tinygrad's 7.3x larger search (BEAM=1 -> BEAM=10) produced no step-time change
(2.913 -> 2.943ms); zyx's 20x larger search (50 -> 1000 launches) gained 11%
(0.242 -> 0.216ms).

## Notes

- zyx's frozen plan re-records its CUDA graph every step: the fast tier
  requires identical node pointers, and 36 of 85 pointers move between replays
  (fresh sample upload + fresh output placements). That is ~0.1ms/step of host
  submission that a stable-buffer plan would remove.
- `loss.item()` inside the timed window costs ~36us on that step; the harness
  above keeps it outside.
- tinygrad's step is jitted end to end (`JIT=0` is 91ms/step vs 2.9ms with
  `JIT=1`), so its 2.9ms is not Python-scheduling overhead: ~0.9ms of it is
  device kernels (0.78ms kernels + 0.14ms sample upload) and the rest is
  replay/upload bookkeeping.

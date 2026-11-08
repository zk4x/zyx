# Benchmarks

## llama-3.2-3B GEMMs: prefill and decode

One decoder block's seven projections, both phases: **prefill** is sequence 128,
**decode** is sequence 1 plus `lm_head` (`3072 -> 128256`) — the same set
`zyx-bench llama prefill_gemm` and `zyx-bench llama decode_gemm` run. f16
weights and activations, f32 accumulation, no bias.

Device: NVIDIA GeForce RTX 2060 12GB, compute capability 7.5, driver 13.0,
336 GB/s HBM. Phase total is the sum of the projection times, i.e. what one
block of the phase costs in its matmuls.

Two numbers per row, because the two phases stress different limits:

- **TFLOP/s** = `2*m*k*n / time` — useful for prefill, which is compute-bound.
- **GB/s** = `(m*k + n*k + m*n) * 2 / time` — the *effective* bandwidth: the
  minimal traffic (activations in, weights in, result out, f16 elements), not
  a measured counter. Useful for decode, which is bandwidth-bound: at
  `m = 1` the whole weight matrix has to be read for a single row of output.
- Phase totals use summed bytes and summed time, so they are the aggregate,
  not an average of the per-projection ratios.

### Phase totals (end to end)

| phase | framework | total | TFLOP/s | GB/s |
|---|---|---|---|---|
| prefill (seq 128) | llama.cpp `c6824a9e4` (ggml CUDA, f16) | 1.2288 ms | 20.97 | 175 |
| prefill (seq 128) | torch 2.11.0+cu130 (cuBLAS) | 1.3037 ms | 19.77 | 165 |
| prefill (seq 128) | tinygrad 0.11.0 (CUDA, `BEAM=1`, `TinyJit`) | 2.5213 ms | 10.22 | 85 |
| prefill (seq 128) | zyx 0.18-dev (CUDA, 20 launches) | 2.7662 ms | 9.32 | 78 |
| decode (seq 1 + lm_head) | torch 2.11.0+cu130 (cuBLAS) | 3.2532 ms | 0.30 | 304 |
| decode (seq 1 + lm_head) | llama.cpp `c6824a9e4` (ggml CUDA, f16) | 3.3702 ms | 0.29 | 294 |
| decode (seq 1 + lm_head) | tinygrad 0.11.0 (CUDA, `BEAM=1`, `TinyJit`) | 3.2803 ms | 0.30 | 302 |
| decode (seq 1 + lm_head) | zyx 0.18-dev (CUDA, 20 launches) | 3.2611 ms | 0.30 | 303 |

Decode is bandwidth-bound: all four reach ~90% of the card's 336 GB/s (torch
304, zyx 303, tinygrad 302, llama.cpp 294), and TFLOP/s is small by
construction. Prefill is compute-bound: llama.cpp and torch sit at ~20
TFLOP/s, tinygrad at 10.2 and zyx at 9.3 — none of them is limited by a wall.

tinygrad is measured with `BEAM=1`, its own recommended kernel search, not at
the `BEAM=0` default: at `BEAM=0` the same script reports 3.7942 ms prefill
(6.79 TFLOP/s) and 39.6069 ms decode, the latter dominated by `lm_head` at
21 GB/s (37.5 ms) — a kernel `BEAM=1` replaces with a 2.5684 ms one.

### Prefill, per projection

| projection | shape | framework | ms | TFLOP/s | GB/s |
|---|---|---|---|---|---|
| `q_proj` | [128x3072]x[3072x3072] | llama.cpp | 0.1205 | 20.05 | 170 |
| `q_proj` | [128x3072]x[3072x3072] | torch | 0.1485 | 16.27 | 138 |
| `q_proj` | [128x3072]x[3072x3072] | tinygrad | 0.3158 | 7.65 | 65 |
| `q_proj` | [128x3072]x[3072x3072] | zyx | 0.2786 | 8.67 | 73 |
| `k_proj` | [128x3072]x[3072x1024] | llama.cpp | 0.0560 | 14.38 | 131 |
| `k_proj` | [128x3072]x[3072x1024] | torch | 0.0552 | 14.59 | 133 |
| `k_proj` | [128x3072]x[3072x1024] | tinygrad | 0.1363 | 5.91 | 54 |
| `k_proj` | [128x3072]x[3072x1024] | zyx | 0.1561 | 5.16 | 47 |
| `v_proj` | [128x3072]x[3072x1024] | llama.cpp | 0.0568 | 14.18 | 129 |
| `v_proj` | [128x3072]x[3072x1024] | torch | 0.0547 | 14.72 | 134 |
| `v_proj` | [128x3072]x[3072x1024] | tinygrad | 0.1364 | 5.90 | 54 |
| `v_proj` | [128x3072]x[3072x1024] | zyx | 0.1561 | 5.16 | 47 |
| `o_proj` | [128x3072]x[3072x3072] | llama.cpp | 0.1200 | 20.13 | 170 |
| `o_proj` | [128x3072]x[3072x3072] | torch | 0.1507 | 16.03 | 136 |
| `o_proj` | [128x3072]x[3072x3072] | tinygrad | 0.2828 | 8.54 | 72 |
| `o_proj` | [128x3072]x[3072x3072] | zyx | 0.2786 | 8.67 | 73 |
| `gate_proj` | [128x3072]x[3072x8192] | llama.cpp | 0.3185 | 20.23 | 167 |
| `gate_proj` | [128x3072]x[3072x8192] | torch | 0.2927 | 22.01 | 182 |
| `gate_proj` | [128x3072]x[3072x8192] | tinygrad | 0.5385 | 11.96 | 99 |
| `gate_proj` | [128x3072]x[3072x8192] | zyx | 0.6289 | 10.24 | 85 |
| `up_proj` | [128x3072]x[3072x8192] | llama.cpp | 0.3163 | 20.37 | 168 |
| `up_proj` | [128x3072]x[3072x8192] | torch | 0.2936 | 21.94 | 181 |
| `up_proj` | [128x3072]x[3072x8192] | tinygrad | 0.5365 | 12.01 | 99 |
| `up_proj` | [128x3072]x[3072x8192] | zyx | 0.6289 | 10.24 | 85 |
| `down_proj` | [128x8192]x[8192x3072] | llama.cpp | 0.2407 | 26.77 | 221 |
| `down_proj` | [128x8192]x[8192x3072] | torch | 0.3083 | 20.90 | 173 |
| `down_proj` | [128x8192]x[8192x3072] | tinygrad | 0.5749 | 11.21 | 93 |
| `down_proj` | [128x8192]x[8192x3072] | zyx | 0.6390 | 10.08 | 83 |

### Decode, per projection

| projection | shape | framework | ms | TFLOP/s | GB/s |
|---|---|---|---|---|---|
| `q_proj` | [1x3072]x[3072x3072] | torch | 0.0656 | 0.29 | 288 |
| `q_proj` | [1x3072]x[3072x3072] | llama.cpp | 0.0746 | 0.25 | 253 |
| `q_proj` | [1x3072]x[3072x3072] | tinygrad | 0.0644 | 0.29 | 293 |
| `q_proj` | [1x3072]x[3072x3072] | zyx | 0.0655 | 0.29 | 288 |
| `k_proj` | [1x3072]x[3072x1024] | torch | 0.0241 | 0.26 | 261 |
| `k_proj` | [1x3072]x[3072x1024] | llama.cpp | 0.0327 | 0.19 | 193 |
| `k_proj` | [1x3072]x[3072x1024] | tinygrad | 0.0401 | 0.16 | 157 |
| `k_proj` | [1x3072]x[3072x1024] | zyx | 0.0254 | 0.25 | 248 |
| `v_proj` | [1x3072]x[3072x1024] | torch | 0.0245 | 0.26 | 257 |
| `v_proj` | [1x3072]x[3072x1024] | llama.cpp | 0.0328 | 0.19 | 192 |
| `v_proj` | [1x3072]x[3072x1024] | tinygrad | 0.0405 | 0.16 | 156 |
| `v_proj` | [1x3072]x[3072x1024] | zyx | 0.0254 | 0.25 | 248 |
| `o_proj` | [1x3072]x[3072x3072] | torch | 0.0649 | 0.29 | 291 |
| `o_proj` | [1x3072]x[3072x3072] | llama.cpp | 0.0743 | 0.25 | 254 |
| `o_proj` | [1x3072]x[3072x3072] | tinygrad | 0.0646 | 0.29 | 292 |
| `o_proj` | [1x3072]x[3072x3072] | zyx | 0.0655 | 0.29 | 288 |
| `gate_proj` | [1x3072]x[3072x8192] | torch | 0.1672 | 0.30 | 301 |
| `gate_proj` | [1x3072]x[3072x8192] | llama.cpp | 0.1887 | 0.27 | 267 |
| `gate_proj` | [1x3072]x[3072x8192] | tinygrad | 0.1673 | 0.30 | 301 |
| `gate_proj` | [1x3072]x[3072x8192] | zyx | 0.1687 | 0.30 | 298 |
| `up_proj` | [1x3072]x[3072x8192] | torch | 0.1679 | 0.30 | 300 |
| `up_proj` | [1x3072]x[3072x8192] | llama.cpp | 0.1885 | 0.27 | 267 |
| `up_proj` | [1x3072]x[3072x8192] | tinygrad | 0.1673 | 0.30 | 301 |
| `up_proj` | [1x3072]x[3072x8192] | zyx | 0.1687 | 0.30 | 298 |
| `down_proj` | [1x8192]x[8192x3072] | torch | 0.1683 | 0.30 | 299 |
| `down_proj` | [1x8192]x[8192x3072] | llama.cpp | 0.1807 | 0.28 | 279 |
| `down_proj` | [1x8192]x[8192x3072] | tinygrad | 0.1676 | 0.30 | 300 |
| `down_proj` | [1x8192]x[8192x3072] | zyx | 0.1694 | 0.30 | 297 |
| `lm_head` | [1x3072]x[3072x128256] | torch | 2.5707 | 0.31 | 307 |
| `lm_head` | [1x3072]x[3072x128256] | llama.cpp | 2.5979 | 0.30 | 303 |
| `lm_head` | [1x3072]x[3072x128256] | tinygrad | 2.5684 | 0.31 | 307 |
| `lm_head` | [1x3072]x[3072x128256] | zyx | 2.5725 | 0.31 | 306 |

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
BEAM=1 python3.12 examples/llama/gemm_tinygrad.py
```

Each script measures one launch per iteration (torch: a batch of 50 launches
and one sync; tinygrad: `TinyJit` + device sync), best of 5 batches, after 20
warmup launches. tinygrad is run with `BEAM=1` (its recommended kernel search,
2,197 candidates over ~4 s for the whole set); without it the script's
`BEAM=0` default lands at 3.7942 ms prefill and 39.6069 ms decode. llama.cpp's
numbers are one `ggml_mul_mat` graph per projection on the CUDA backend, which
is the path its f16 prefill takes.

### zyx

Budget: **20 timed launches per kernel** (`autotune.n_launches = 20` in
`~/.config/zyx/config.json`) — the default a user gets, and roughly the point
where tuning cost stops being noticeable: the whole prefill set tuned in
~2 minutes, decode in under 1. It is a cap, not a fixed cost: the search stops
earlier when candidates collapse onto kernels it has already timed. A seed
shared by two projections is tuned once per projection, so the run above
logged 41 launches each for `q`/`o` and `gate`/`up`, 20 for `k`/`v` and
`down`, and 20 for every decode kernel. The table entry is the fastest
candidate measured within that budget, on the device.

For scale, the same search at 1000 launches reached 2.687 ms / 9.59 TFLOP/s
for prefill and 3.260 ms for decode on the earlier code; that run has not been
repeated since. At 20 launches the phase total above is 2.7662 ms / 9.32
TFLOP/s, i.e. within 3% of it, so the budget is no longer what limits the
result — the numbers above are the ones a user actually gets.

Reproduce:

```bash
cd zyx-bench
AGENT=1 ZYX_DEV=cuda ZYX_QUIET_INPUT_COPIES=1 ./target/release/zyx-bench llama prefill_gemm
AGENT=1 ZYX_DEV=cuda ZYX_QUIET_INPUT_COPIES=1 ./target/release/zyx-bench llama decode_gemm
```

Every candidate is written to `zyx-bench/bench.db` (`llama/prefill_gemm`,
`llama/decode_gemm`) with its device time, `grid`, `n_launches`, kernel IR and
`code_version`; the rows above are the minimum `nanos` per seed at
`n_launches=20`, `code_version=a13f5338+dirty` (tensor recipe, packed mma fragments, dense
staging decode, `std` runtime lock, and the cost model retrained on 19,321
rows by `regression.py`). The tree was dirty because this table's own changes
were not committed yet; the run is reproducible from it once they land.

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

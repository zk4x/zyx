# Benchmarks

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

# Qwen3.8-27B UD-Q4_K_XL Plan

## Phase 1 — Nvidia CUDA reference (zyx custom IR)

- Model: `examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf` 17GB, 65 blocks, 49 SSM + 16 full attn (interval 4), 5120 hidden, 17408 ffn, 248320 vocab, 48 rank, 32K context.
- Quant: UD-Q4_K_XL mixed Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS (0.43-1.06 BPE) + 8b KV cache Q8_0 for 100k (3.2GB KV → 20.8GB < 28GB).
- Kernels: zyx `Kernel::new(Dev::Cuda(0))` custom IR only, no Tensor ops. `gemm_cuda_q4_k` F16×Q4→F32, `mlp`, `rope`, `attention`, `delta_core`, `rmsnorm`, `conv_silu`, `embed`. `dequant_q4_k` U32 8×4b → F16 scale/min per 32, fused in `global→local` stage, `Local 16×32/8×32` + `barrier` + `mma 16×8`.
- Verify: `tests/*_q4k.rs` vs torch dequant+matmul, `cargo test --test gemm_q4k`.
- Bench: `tests/bench_all.rs` single `bench_all` `builder+compile 50ms` / `forward+sync 1ms` per kernel, `flop_mem_rw` TFLOPS GB/s, `forward+sync × inv` → `tok/s` (seq 6). Target `F16 3.7 tok/s` now, `Q4 38ms→6ms` → `19 tok/s` 2060 roof `17GB/330GB/s`. Pass when `bench_all` `tok/s` `>5` (roof `19` `→` `26%`, `P100A` `28` `roof` `→` `18%`).

## Phase 2 — Tenstorrent P100A port (once CUDA runs fast)

- PTX/NVRTC only for CUDA. P100A Blackhole 28GB 480GB/s 320 TFLOPS BF16 uses TT-Metal `32×32` CB, `tilize`/`untilize`, `L1` `CB`, `matmul_tile` + `barrier`, `BF16` scales → `dequant BF16` → `F32` DST `fp32_dest_acc_en`.
- Port: `gemm_tt_q4_k` BF16 `32×32` local, `rope/attention/mlp` 32×32, same `dequant` logic BF16, `zyx/src/backend/tenstorrent.rs` `tile_sizes [32,32]`.
- Verify on P100A, bench same `bench_all` `→ 28 tok/s` roof `17GB/480GB/s`, `100k` 8b KV `20.8GB`.

CUDA PTX never runs on P100A. TT kernels separate.

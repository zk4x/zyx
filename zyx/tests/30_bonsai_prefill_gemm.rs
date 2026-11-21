// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B (`qwen35`, PQ2_0) prefill GEMM: the ten projections of
//! the block at the padded prefill width `M = 128`, and the tensor-core shape
//! llama.cpp's `mmq` runs for PQ2_0 on this device.
//!
//! Four things are measured and checked here, in the order the kernel work
//! needs them:
//!
//! 1. `qwen_prefill_gemm_ms` measures every projection's kernel as the
//!    optimization passes receive it — the naive fused-dequant IR — and names
//!    the slowest prefill shape by measurement, not by FLOPs;
//! 2. `qwen_prefill_mmq` writes that shape's kernel in zyx custom-kernel IR in
//!    llama.cpp's `mmq` block (`I = 128` weight rows, `J = 128` activation rows,
//!    256 threads as 4x2 warps, a 128-K chunk, `ldmatrix.x4` + `mma.m8n8k16.s8`)
//!    and checks it against a host GEMM on the same int8 operands;
//! 3. `qwen_prefill_tensorize_shape` checks that `opt_tensorize` enumerates
//!    that exact block — the same `128x128` tile grid, a `128`-wide K chunk,
//!    staged and swizzled — for every one of the ten projections, from the
//!    naive IR the linearizer emits, and that applying the config produces the
//!    kernel whose shared tiles are the `mmq` ones (A: 128 rows x 40 words, B:
//!    128 rows x 36 words);
//! 4. `qwen_prefill_naive_ir` prints that naive IR, and `qwen_prefill_graph_ms`
//!    / `qwen_prefill_probe` hold the graph-form measurement and its blocker
//!    (a tensorize candidate on the graph's dequantize kernel faults on the
//!    device; see their docs).
//!
//! The naive kernel is the one test 26 established as what the linearizer emits
//! for this fused dequant-matmul: two group ranges over the output, one K loop,
//! one register accumulator, the PQ2_0 super-block dequantized inline and the
//! q8_1 activation dequantized inline, `acc += f32(f16(w) * f16(a))`.

use std::time::Instant;

use zyx::kernel::autotune::BeamSearch;
use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, QDType, Tape, Tensor, ZyxError, f16};

/// Prefill width: the padded sequence length of a prefill chunk.
const M: i64 = 128;
/// Weights per PQ2_0 super-block.
const BLOCK: i64 = 128;
/// Bytes per PQ2_0 super-block: 2 scale bytes, then 32 code bytes.
const ROW: i64 = 34;
/// One q8_1 activation scale per this many elements.
const QG: i64 = 32;

/// The ten projections of the two block types and the output projection, as
/// `zyx-bench qwen prefill_gemm` runs them (`[M x k] x [k x n]`).
const PROJS: [(&str, i64, i64); 10] = [
    ("attn_q", 5120, 12288),
    ("attn_k", 5120, 1024),
    ("attn_v", 5120, 1024),
    ("attn_output", 6144, 5120),
    ("attn_qkv", 5120, 10240),
    ("attn_gate", 5120, 6144),
    ("ssm_out", 6144, 5120),
    ("ffn_gate", 5120, 17408),
    ("ffn_up", 5120, 17408),
    ("ffn_down", 17408, 5120),
];

/// The config index `opt_tensorize` assigns to the `mmq` block, in the pass's
/// own enumeration order (read off the pass with `ZYX_PLANS=1`). The plan table
/// is the same for all ten projections — every `M`, `N` and `K` here is a whole
/// number of `128x128` tiles at a `128`-wide K chunk, so every plan the smaller
/// shapes admit the larger ones admit too — so the one index names the block in
/// each. `qwen_prefill_tensorize_shape` re-derives the shape it builds and
/// fails if the index stops naming it.
const TILE_CFG: u64 = 717;

fn kgroup(k: i64) -> i64 {
    k / QG
}

// ===========================================================================
// The naive kernel the linearizer emits
// ===========================================================================

/// The scalar fused-dequant matmul the linearizer emits for
/// `x.dot_dtype_output(w.dequantize(PQ2_0))`: `[M x K] x [N x K]^T`, one output
/// element per thread, the PQ2_0 super-block and the q8_1 group dequantized
/// inline, an f32 register accumulator (`K`, `N` are the projection's own).
fn naive_gemm(dev: Dev, kd: i64, nd: i64) -> Kernel {
    let mut k = Kernel::new(dev);
    // The operands in the width the kernel reads them: the q8_1 payload as u32
    // words (four int8 each), the PQ2_0 super-block as its 17 u16 words.
    let a8 = k.param(DType::U32);
    // The q8_1 scale is the f16 `d` llama.cpp's quantizer stores.
    let sa = k.param(DType::F16);
    let w = k.param(DType::U16);
    let c = k.param_mut(DType::F32);
    let m = k.group_range(0, M);
    let n = k.group_range(1, nd);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(kd, |k, k_loop| {
        // The weight's element index in the [N, K] row-major projection.
        let elem = k.mad(n, kd, k_loop);
        let blk = k.div(elem, BLOCK);
        let off = k.mod_(elem, BLOCK);
        // The 34-byte super-block as its 17 u16 words: word 0 is the f16
        // scale, words 1..17 the 32 code bytes, eight 2-bit codes per word.
        let c17 = k.const_val(17i64);
        let wrow = k.mul(blk, c17);
        let two = k.const_val(2i64);
        let eight_i = k.const_val(8i64);
        let off_word = k.div(off, eight_i);
        let word_in_row = k.add(1i64, off_word);
        let word_idx = k.add(wrow, word_in_row);
        let word = k.load(w, word_idx);
        let off_mod = k.mod_(off, eight_i);
        let shift = k.mul(off_mod, two);
        // The shift amount is an operand of the word's own dtype (u16).
        let shift = k.cast(shift, DType::U16);
        let shifted = k.bit_shift_right(word, shift);
        let mask = k.const_val(3u16);
        let code = k.bit_and(shifted, mask);
        // The block's scale: the little-endian f16 in the super-block's first
        // two bytes, one u16 load.
        let bits = k.load(w, wrow);
        let scale_f16 = k.bitcast(bits, DType::F16);
        let scale = k.cast(scale_f16, DType::F32);
        let code_f32 = k.cast(code, DType::F32);
        let centered = k.sub(code_f32, 1.0f32);
        let val = k.mul(centered, scale);
        // The projection is an f16 x f16 multiply with an f32 accumulate.
        let val = k.cast(val, DType::F16);
        // The activation's q8_1 dequant: one int8 value and the scale of its
        // 32-element group, the payload read as u32 words (four int8 each).
        let act_idx = k.mad(m, kd, k_loop);
        let four = k.const_val(4i64);
        let word_i = k.div(act_idx, four);
        let aword = k.load(a8, word_i);
        let byte_off = k.mod_(act_idx, four);
        let bo32 = k.cast(byte_off, DType::U32);
        let eight_u = k.const_val(8u32);
        let sh = k.mul(bo32, eight_u);
        let bword = k.bit_shift_right(aword, sh);
        let bmask = k.const_val(0xffu32);
        let byte_bits = k.bit_and(bword, bmask);
        let byte_u = k.cast(byte_bits, DType::U8);
        let byte = k.bitcast(byte_u, DType::I8);
        let byte_f = k.cast(byte, DType::F32);
        let group = k.div(k_loop, kgroup(kd));
        let kg = k.const_val(kgroup(kd));
        let sa_idx = k.mad(m, kg, group);
        // The stored f16 scale, widened to the f32 the product multiplies.
        let sa_v = k.load(sa, sa_idx);
        let sa_f = k.cast(sa_v, DType::F32);
        let act = k.mul(byte_f, sa_f);
        let act = k.cast(act, DType::F16);
        let prod = k.mul(val, act);
        let prod = k.cast(prod, DType::F32);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prod, prev);
        k.store(acc, sum, 0i64);
    });
    let res = k.load(acc, 0i64);
    let c_idx = k.mad(m, nd, n);
    k.store(c, res, c_idx);
    k.verify();
    k
}

// ===========================================================================
// 1. the current prefill kernel, per projection
// ===========================================================================

/// The kernel the graph hands to the optimization passes for every prefill
/// projection — the naive fused-dequant matmul above — measured on the device
/// by the search's own single launch (the seed, `ZYX_DEBUG=256`'s NoSearch IR).
///
/// Long by nature: the naive form re-reads the weight once per activation row,
/// so one launch is ~9.9 s for the FFN projections and the search launches it
/// twice (warmup + timed). Run it with `--ignored`; the whole ten-projection
/// table is ~111 s on this device. Measured 2026-11-21 (RTX 2060, sm_75):
///
/// ```text
/// attn_q      6827.6 ms   attn_qkv    5731.0 ms
/// attn_k       569.5 ms   attn_gate   3438.9 ms
/// attn_v       570.4 ms   ssm_out     3469.2 ms
/// attn_output 3413.6 ms   ffn_gate    9948.3 ms   <- slowest
///                         ffn_up      9851.7 ms
///                         ffn_down    9873.2 ms
/// ```
///
/// The graph form of the same projection is *not* measured here: on this device
/// its PQ2_0 dequantize stage panics in the autotune
/// (`src/kernel/mod.rs:796`, a `Binary { x: Loop, y: Const }` whose `Loop`
/// operand has no dtype entry), so the `qwen prefill_gemm` bench entry cannot
/// produce a prefill number at all. `qwen_prefill_probe` is the minimal
/// reproducer.
#[test]
#[ignore = "one launch is 9.9 s at the FFN shapes; run with --ignored"]
fn qwen_prefill_gemm_ms() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    for (name, kk, nn) in PROJS {
        let naive = naive_gemm(dev, kk, nn);
        let (a32, sa, w16, out) = operands(dev, kk, nn)?;
        let flop = 2.0 * (M * kk * nn) as f64;
        let values = naive.clone();
        let (_winner, nanos) = BeamSearch::new().run([naive], &[&a32, &sa, &w16, &out], &[], |_| {}, |_| 0)?;
        let ms = nanos as f64 * 1e-6;
        println!("qwen prefill {name} [{M}x{kk}]x[{kk}x{nn}]: {ms:.4} ms, {:.2} TFLOP/s (naive)", flop / (ms * 1e-3) * 1e-12);
        let _ = values;
    }
    Ok(())
}

/// The four buffers a prefill projection's kernel binds: the q8_1 payload as
/// u32 words, its per-32 scales, the PQ2_0 super-block as its 17 u16 words,
/// and the f32 output.
fn operands(dev: Dev, kk: i64, nn: i64) -> Result<(Tensor, Tensor, Tensor, Tensor), ZyxError> {
    let a32: Tensor = Tensor::rand([M * kk / 4], DType::U32)?.to(dev)?;
    let sa: Tensor = Tensor::rand([M * kgroup(kk)], DType::F16)?.to(dev)?;
    let w16: Tensor = Tensor::rand([nn * kk / BLOCK * 17], DType::U16)?.to(dev)?;
    let out: Tensor = Tensor::zeros([M, nn], DType::F32).to(dev)?;
    Ok((a32, sa, w16, out))
}

// ===========================================================================
// 2. the tensorization space, per projection
// ===========================================================================

/// `opt_tensorize` must offer the `mmq` block for every prefill projection:
/// a `128x128` tile grid (16 eight-row tiles each way), a `128`-element K chunk
/// (8 `fk = 16` steps, so 4 q8_1 groups), staged and swizzled, at the 4x2 warp
/// grid. Its shape fingerprint in the emitted IR is the two shared tiles, the
/// 8 `ldmatrix.x4` fragment loads (one per weight sub-tile and K group, each
/// over the 8 column steps) and the 256 in-place `m8n8k16.s8` instructions
/// (4 groups x 2 sub-tiles x 8 column steps x 4 per pair, one mma per warp per
/// 8x8 output tile and 16-wide K step of the 128x128x128 block).
#[test]
fn qwen_prefill_tensorize_shape() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    for (name, kk, nn) in PROJS {
        let naive = naive_gemm(dev, kk, nn);
        let recipe = naive.opt_tensorize();
        let n = recipe.nconfigs();
        let cfg = TILE_CFG;
        assert!(cfg < n, "{name}: the tensorization space must contain the mmq block: {n} configs, wanted {cfg}");
        let mut tiled = naive.clone();
        recipe.apply(&mut tiled, cfg);
        tiled.default_epilogue();
        tiled.verify();
        let (mut local, mut ldmatrix, mut mma) = (0usize, 0usize, 0usize);
        let mut op_id = tiled.head;
        while !op_id.is_null() {
            match tiled.at(op_id) {
                zyx::kernel::Op::Storage { scope: MemScope::Local, .. } => local += 1,
                zyx::kernel::Op::PTX(p) if matches!(**p, zyx::kernel::PTXOp::VLdMatrix { .. }) => ldmatrix += 1,
                zyx::kernel::Op::Wmma { dtype: zyx::kernel::MMADType::s8_s8_s32_s32, .. } => mma += 1,
                _ => {}
            }
            op_id = tiled.next_op(op_id);
        }
        println!("qwen prefill {name}: {n} configs, mmq block at {cfg}: {local} shared tiles, {ldmatrix} ldmatrix, {mma} mma");
        assert_eq!((local, ldmatrix, mma), (2, 8, 256), "{name}: config {cfg} must build the mmq block (I=J=128, K chunk 128)");
    }
    Ok(())
}

/// Every prefill projection through the graph form the model and
/// `zyx-bench qwen prefill_gemm` run: the PQ2_0 weight dequantized to f16 in
/// the traced step, then the f16/f32 dot. The second `realize` of the same
/// shape is timed (the first warms the kernel cache), so the number is the
/// kernel the model runs.
///
/// This is the graph-form counterpart of the naive measurement above, and the
/// same path `zyx-bench qwen prefill_gemm` drives. It is ignored only because a
/// full pass over the ten shapes autotunes for minutes; `opt_tensorize`'s
/// element form on a fused-dequant operand is no longer offered (its staged
/// fill packed the tile slot from the chain's global row), so no candidate
/// faults and the run completes.
#[test]
#[ignore = "a full graph-form pass over the ten shapes autotunes for minutes; see the docs"]
fn qwen_prefill_graph_ms() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    for (name, kk, nn) in PROJS {
        let x = Tensor::rand([M, kk], DType::F16)?.to(dev)?;
        let w = Tensor::rand([nn * kk / BLOCK, ROW], DType::U8)?.to(dev)?;
        let run = || -> Result<Tensor, ZyxError> {
            let tape = Tape::new([&x, &w])?;
            let w = w.dequantize(QDType::PQ2_0, DType::F16)?.reshape([nn, kk])?;
            let out = x.dot_dtype_output(w.t(), DType::F32, DType::F32)?;
            tape.realize([&out])?;
            Ok(out)
        };
        let _ = run()?;
        let t0 = Instant::now();
        let out = run()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        // Reading the value bounds the measurement to a completed launch.
        let _ = out.item::<f32>();
        let flop = 2.0 * (M * kk * nn) as f64;
        println!("qwen prefill graph {name} [{M}x{kk}]x[{kk}x{nn}]: {ms:.4} ms, {:.2} TFLOP/s", flop / (ms * 1e-3) * 1e-12);
    }
    Ok(())
}

/// The naive IR the optimization passes are handed for the slowest projection,
/// printed as the passes see it (the `ZYX_DEBUG=256` NoSearch seed form).
#[test]
fn qwen_prefill_naive_ir() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    let (name, kk, nn) = WORST;
    let naive = naive_gemm(dev, kk, nn);
    println!("naive IR {name} [{M}x{kk}]x[{kk}x{nn}]:");
    naive.debug();
    Ok(())
}

/// Reproducer for the f16 tensorization candidates' failures, outside the
/// graph: apply one `opt_tensorize` config (`ZYX_CFG`, default 52) to the naive
/// prefill kernel — after the epilogue the search applies between steps — and
/// launch it once. The s8 `mmq` block (1279) and the f16 *region* forms are not
/// what this shows: on this kernel the f16 staged *grid* plans (36, 46, 50, 52)
/// fault at launch (`CUDA_ERROR_ILLEGAL_ADDRESS`), and 18 fails `verify` with a
/// `Binary` dtype mismatch.
#[test]
#[ignore = "one manual config; see the docs"]
fn qwen_prefill_cfg_probe() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    let (_, kk, nn) = WORST;
    let naive = naive_gemm(dev, kk, nn);
    let recipe = naive.opt_tensorize();
    let mut k = naive.clone();
    // The search applies its epilogue after every step, so the state a plan
    // sees has its constants and ranges hoisted.
    k.default_epilogue();
    let cfg: u64 = std::env::var("ZYX_CFG").ok().and_then(|s| s.parse().ok()).unwrap_or(52);
    eprintln!("probe cfg {cfg} of {}", recipe.nconfigs());
    recipe.apply(&mut k, cfg);
    k.default_epilogue();
    k.verify();
    let (a32, sa, w16, out) = operands(dev, kk, nn)?;
    let (_winner, nanos) = BeamSearch::new().run([k], &[&a32, &sa, &w16, &out], &[], |_| {}, |_| 0)?;
    eprintln!("probe cfg {cfg}: survived, {nanos} ns");
    Ok(())
}

/// The standalone PQ2_0 dequantize for one projection's weight: the same stage
/// that faults inside the prefill graph (`qwen_prefill_graph_ms`) autotunes on
/// its own, so the two differ in the kernel the graph fuses, not in the tensor.
#[test]
#[ignore = "pairs with qwen_prefill_graph_ms; see its docs"]
fn qwen_prefill_probe() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    let (_, kk, nn) = PROJS[7];
    let w = Tensor::rand([nn * kk / BLOCK, ROW], DType::U8)?.to(dev)?;
    eprintln!("[probe] dequantize alone");
    {
        let tape = Tape::new([&w])?;
        let d = w.dequantize(QDType::PQ2_0, DType::F16)?;
        tape.realize([&d])?;
        let _: Vec<f32> = d.cast(DType::F32).try_into()?;
    }
    eprintln!("[probe] dequantize alone fine");
    Ok(())
}

// ===========================================================================
// 3. the `mmq` block for the slowest projection
// ===========================================================================

/// The slowest prefill projection at `M = 128` (`qwen_prefill_gemm_ms`).
const WORST: (&str, i64, i64) = PROJS[7];

/// Quantize the fp16 activations to `q8_1` exactly as llama.cpp's
/// `quantize_q8_1` does: one group of 32 values per lane, `d = amax/127`,
/// `q = amax == 0 ? 0 : roundf(x/d)` (ties away from zero), and the scale
/// stored as f16 (llama's `make_half2(d, sum)`; the `sum` half is only read by
/// the integer `dp4a` formats, never by `mmq`'s PQ2_0 fold).
fn quantize_q8_kernel(dev: Dev, kd: i64) -> Kernel {
    let mut k = Kernel::new(dev);
    let a = k.param(DType::F16);
    let q8 = k.param_mut(DType::I8);
    let sa = k.param_mut(DType::F16);
    let row = k.group_range(0, M);
    let g = k.local_range(0, u32::try_from(kgroup(kd)).expect("groups fit u32"));
    let kg = k.const_val(kd);
    let cg = k.const_val(QG);
    let gk = k.mul(g, cg);
    let base = k.mad(row, kg, gk);
    // The 32 values of the group are one lane: one load, one maximum, one
    // quantize, one store of the group's bytes.
    let l32 = k.unroll(1, 32);
    let vidx = k.add(base, l32);
    let raw = k.load(a, vidx);
    let vals = k.cast(raw, DType::F32);
    let av = k.abs(vals);
    let amax = k.reduce_max(av, l32);
    let hundred = k.const_val(127.0f32);
    let zero = k.const_val(0f32);
    let cond = k.equal(amax, zero);
    let scale = k.div(amax, hundred);
    // `make_half2(d, sum)`: the scale is the f16 of the f32 ratio.
    let scale16 = k.cast(scale, DType::F16);
    let ckg = k.const_val(kgroup(kd));
    let qidx = k.mad(row, ckg, g);
    k.store(sa, scale16, qidx);
    // `roundf(x/d)`: trunc(x + copysign(0.5, x)) — ties away from zero, unlike
    // the cast's own `cvt.rni`. The all-zero group quantizes to zero rather
    // than to a division by zero.
    let one_f = k.const_val(1.0f32);
    let divisor = k.select(cond, one_f, scale);
    let qf = k.div(vals, divisor);
    let bits = k.bitcast(qf, DType::U32);
    let sign_mask = k.const_val(0x8000_0000u32);
    let sign = k.bit_and(bits, sign_mask);
    let half_bits = k.const_val(0x3F00_0000u32);
    let half = k.bit_or(sign, half_bits);
    let adj = k.bitcast(half, DType::F32);
    let shifted = k.add(qf, adj);
    let rounded = k.trunc(shifted);
    let qz = k.select(cond, zero, rounded);
    let q = k.cast(qz, DType::I32);
    let qb = k.cast(q, DType::I8);
    k.store(q8, qb, vidx);
    k.verify();
    k
}

/// llama.cpp's Turing `mmq` block for PQ2_0 at this projection, written in zyx
/// custom-kernel IR: `out[M, N] = q8[M, K] @ w[N, K]^T`, the PQ2_0 weights
/// dequantized to int8 into shared memory (`code - 1`, the exact integer
/// payload, so the tensor core sums integers), the activations quantized to
/// `q8_1` in a pre-pass, and the per-32 activation and per-128 weight scales
/// folded into the f32 accumulator after each 32-element group.
///
/// The block is `mmq`'s: `I = 128` weight rows by `J = 128` activation rows,
/// 256 threads as eight warps, warp `warp/2` owning a 32-row weight band and
/// `warp%2` the 8-row half of each 16, a 128-element K chunk per tile stage, and
/// `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` (Turing's int8 tensor core,
/// which is what the SASS `IMMA.8816.S8.S8` above it is).
fn gemm_s8_kernel(dev: Dev, kd: i64, nd: i64) -> Kernel {
    // Weight rows per block (`I`) and activation rows per block (`J`).
    const IB: i64 = 128;
    const JB: i64 = 128;
    // `mmq`'s block config: 256 threads as 32x8.
    const NW: u32 = 8;
    // Words per row of the two shared tiles: the super-block's 32 payload
    // words, the four words of scales, then the 4-word (16-byte) tail that
    // keeps the rows of one `ldmatrix` off the same banks.
    const XS: i64 = 40;
    const YS: i64 = 36;

    let mut k = Kernel::new(dev);
    let a8 = k.param(DType::U32);
    // The q8_1 scales are the f16 `d` llama.cpp's quantizer stores.
    let sa = k.param(DType::F16);
    let w = k.param(DType::U16);
    let out = k.param_mut(DType::F32);

    let kblk = kd / BLOCK;
    let nb = k.group_range(0, nd / IB);
    let lane = k.local_range(0, 32);
    let warp = k.local_range(1, NW);
    let tid = k.mad(warp, 32i64, lane);
    let c1 = k.const_val(1i64);
    let c2 = k.const_val(2i64);
    let c3 = k.const_val(3i64);
    let c4 = k.const_val(4i64);
    let c8 = k.const_val(8i64);
    let c16 = k.const_val(16i64);
    let c32 = k.const_val(32i64);
    let c64 = k.const_val(64i64);
    let c256 = k.const_val(256i64);
    let cxs = k.const_val(XS);
    let cys = k.const_val(YS);
    let c17 = k.const_val(17i64);
    let cib = k.const_val(IB);
    let ckb = k.const_val(kblk);
    let ckg = k.const_val(kgroup(kd));
    let nstr = k.const_val(nd);
    let kw = k.const_val(kd / 4);
    let two = k.const_val(2u32);

    let xt = k.storage(DType::U32, MemScope::Local, IB * XS);
    let yt = k.storage(DType::U32, MemScope::Local, JB * YS);
    // `sum[64]`: two accumulator words per fragment pair, live across the K loop.
    let accf = k.storage(DType::F32, MemScope::Register, 64);

    // Warp split and lane geometry: `warp/2` is the 32-row weight band,
    // `warp%2` the 8-row activation band inside each 16, `lane/4` the fragment
    // row, `lane%4` the four-int8 group of a 16-wide K step.
    let warpl = k.bit_and(warp, c1);
    let warph = k.bit_shift_right(warp, c1);
    let i0 = k.mul(warph, c32);
    let jo = k.mul(warpl, c8);
    let gid = k.div(lane, c4);
    let lanecol = k.mod_(lane, c4);
    let tig2 = k.mul(lanecol, c2);
    let lanerow = k.mod_(lane, c16);
    let lanehalf = k.div(lane, c16);
    let zero_i = k.const_val(0i32);
    let zero2 = k.stack(&[zero_i, zero_i]);
    let zero_f = k.const_val(0f32);
    let szero = k.unroll(7, 64);
    k.store(accf, zero_f, szero);

    k.loop_over(kblk, |k, kb| {
        // ---- `ggml_cuda_mmq_load_tiles_pq2_0`: the weight tile ----
        {
            let it = k.unroll(0, 2);
            let j = k.unroll(1, 4);
            let lane4 = k.div(lane, c4);
            let row0 = k.mad(warp, c8, lane4);
            let step = k.mul(it, c64);
            let row = k.add(row0, step);
            let kq = k.mod_(lane, c4);
            let wrow = k.mad(nb, cib, row);
            let blk = k.mad(wrow, ckb, kb);
            let b17 = k.mul(blk, c17);
            let q0 = k.add(b17, c1);
            let qk = k.mul(kq, c4);
            let qbase = k.add(q0, qk);
            let qi = k.add(qbase, j);
            let qw = k.load(w, qi);
            // `const short * qxi` / `const int q = qxi[j]`: the half is read
            // signed (`cvt.s32.s16`), not zero-extended.
            let qs16 = k.bitcast(qw, DType::I16);
            let qi32 = k.cast(qs16, DType::I32);
            let q = k.bitcast(qi32, DType::U32);
            // `code -> code-1`: byte `i` of `0x020100FF` is `i-1`.
            let table = k.const_val(0x0201_00FFu32);
            let mask = k.const_val(0x7777u32);
            let qe_sel = k.bit_and(q, mask);
            let qe = k.prmt(table, table, qe_sel);
            let qsh = k.bit_shift_right(q, two);
            let qo_sel = k.bit_and(qsh, mask);
            let qo = k.prmt(table, table, qo_sel);
            let sel = k.const_val(0x5140u32);
            let qx = k.prmt(qe, qo, sel);
            let sel = k.const_val(0x7362u32);
            let qy = k.prmt(qe, qo, sel);
            let k8 = k.mul(kq, c8);
            let j2 = k.mul(j, c2);
            let col = k.add(k8, j2);
            let rowx = k.mul(row, cxs);
            let dst = k.add(rowx, col);
            k.store(xt, qx, dst);
            let dst1 = k.add(dst, c1);
            k.store(xt, qy, dst1);
        }
        // The block's f16 scale, one entry per row at word 32 of it.
        {
            let si = k.unroll(2, 16);
            let nw = k.const_val(i64::from(NW));
            let rows = k.mul(si, nw);
            let row = k.add(warp, rows);
            let wrow = k.mad(nb, cib, row);
            let blk = k.mad(wrow, ckb, kb);
            let b17 = k.mul(blk, c17);
            let d = k.load(w, b17);
            let d = k.bitcast(d, DType::F16);
            let d = k.cast(d, DType::F32);
            let d = k.bitcast(d, DType::U32);
            let rowx = k.mul(row, cxs);
            let at = k.add(rowx, c32);
            let at = k.add(at, lanecol);
            k.store(xt, d, at);
        }
        // ---- the y tile: `block_q8_1_mmq` is [four scales][32 words] ----
        {
            let w16 = k.unroll(3, 16);
            let m256 = k.mul(w16, c256);
            let off = k.add(tid, m256);
            let r = k.div(off, c32);
            let c = k.mod_(off, c32);
            let rw = k.mul(r, kw);
            let k32 = k.mul(kb, c32);
            let src = k.add(rw, k32);
            let src = k.add(src, c);
            let v = k.load(a8, src);
            let dst = k.mad(r, cys, c);
            let dst = k.add(dst, c4);
            k.store(yt, v, dst);
        }
        {
            let s2 = k.unroll(4, 2);
            let m256 = k.mul(s2, c256);
            let e = k.add(tid, m256);
            let r = k.div(e, c4);
            let g = k.mod_(e, c4);
            let k4 = k.mul(kb, c4);
            let gq = k.add(k4, g);
            let si = k.mad(r, ckg, gq);
            // The scale is the f16 llama.cpp's `quantize_q8_1` stored; the tile
            // holds the f32 the fold multiplies.
            let v = k.load(sa, si);
            let v = k.cast(v, DType::F32);
            let v = k.bitcast(v, DType::U32);
            let dst = k.mad(r, cys, g);
            k.store(yt, v, dst);
        }
        k.barrier();

        // ---- A, the mma and the fold ----
        // The K groups are a sequential chain (each group's fold reads the sums
        // the previous one wrote); the two weight sub-tiles are independent.
        for gv in 0..4i64 {
            let g = k.const_val(gv);
            for nv in 0..2i64 {
                let n = k.const_val(nv);
                let n16 = k.mul(n, c16);
                let row = k.add(i0, n16);
                let row = k.add(row, lanerow);
                let g8 = k.mul(g, c8);
                let addr = k.mad(row, cxs, g8);
                let side = k.mul(lanehalf, c4);
                let addr = k.add(addr, side);
                let frag = k.ldmatrix(
                    zyx::kernel::LdMatrixDims::m8n8,
                    zyx::kernel::LdMatrixNum::X4,
                    zyx::kernel::LdMatrixTrans::off,
                    zyx::kernel::LdMatrixType::b16,
                    xt,
                    addr,
                );
                let aw0 = k.index(frag, 0);
                let aw1 = k.index(frag, 1);
                let aw2 = k.index(frag, 2);
                let aw3 = k.index(frag, 3);
                // `dA[n][h][g]`: word `32+g` of rows `i0 + n*16 + h*8 + lane/4`.
                let crow = k.add(i0, n16);
                let crow = k.add(crow, gid);
                let crowx = k.mul(crow, cxs);
                let da0i = k.add(crowx, c32);
                let da0i = k.add(da0i, g);
                let da0w = k.load(xt, da0i);
                let da0 = k.bitcast(da0w, DType::F32);
                let drow8 = k.add(crow, c8);
                let drow8x = k.mul(drow8, cxs);
                let da1j = k.add(drow8x, c32);
                let da1j = k.add(da1j, g);
                let da1w = k.load(xt, da1j);
                let da1 = k.bitcast(da1w, DType::F32);

                let jt = k.unroll(9, 8);
                let j0 = k.mul(jt, c16);
                let jb = k.add(jo, j0);
                let rowb = k.add(jb, gid);
                let rowbx = k.mul(rowb, cys);
                let colb = k.add(rowbx, c4);
                let bcol = k.add(colb, g8);
                let b0i = k.add(bcol, lanecol);
                let b0 = k.load(yt, b0i);
                let bcol4 = k.add(bcol, c4);
                let b1i = k.add(bcol4, lanecol);
                let b1 = k.load(yt, b1i);
                // `dB[l]`: the activation scale of the two C columns.
                let rd0 = k.add(jb, tig2);
                let rd1 = k.add(rd0, c1);
                let d0i = k.mad(rd0, cys, g);
                let d1i = k.mad(rd1, cys, g);
                let db0w = k.load(yt, d0i);
                let db0 = k.bitcast(db0w, DType::F32);
                let db1w = k.load(yt, d1i);
                let db1 = k.bitcast(db1w, DType::F32);
                let c00 = k.wmma(
                    zyx::kernel::MMADims::m8n8k16,
                    zyx::kernel::MMALayout::row_col,
                    zyx::kernel::MMADType::s8_s8_s32_s32,
                    aw0,
                    b0,
                    zero2,
                );
                let clo = k.wmma(
                    zyx::kernel::MMADims::m8n8k16,
                    zyx::kernel::MMALayout::row_col,
                    zyx::kernel::MMADType::s8_s8_s32_s32,
                    aw2,
                    b1,
                    c00,
                );
                let c10 = k.wmma(
                    zyx::kernel::MMADims::m8n8k16,
                    zyx::kernel::MMALayout::row_col,
                    zyx::kernel::MMADType::s8_s8_s32_s32,
                    aw1,
                    b0,
                    zero2,
                );
                let chi = k.wmma(
                    zyx::kernel::MMADims::m8n8k16,
                    zyx::kernel::MMALayout::row_col,
                    zyx::kernel::MMADType::s8_s8_s32_s32,
                    aw3,
                    b1,
                    c10,
                );
                // `sum[(j0/8 + n)*4 + l] += C.x[l] * dA[n][l/2][g] * dB[l%2]`.
                let jt2 = k.mul(jt, c2);
                let jn = k.add(jt2, n);
                let base = k.mul(jn, c4);
                let s1 = k.add(base, c1);
                let s2 = k.add(base, c2);
                let s3 = k.add(base, c3);
                let ic0 = k.index(clo, 0);
                let ic1 = k.index(clo, 1);
                let ic2 = k.index(chi, 0);
                let ic3 = k.index(chi, 1);
                let v0 = k.cast(ic0, DType::F32);
                let v1 = k.cast(ic1, DType::F32);
                let v2 = k.cast(ic2, DType::F32);
                let v3 = k.cast(ic3, DType::F32);
                let k0 = k.mul(da0, db0);
                let k1 = k.mul(da0, db1);
                let k2 = k.mul(da1, db0);
                let k3 = k.mul(da1, db1);
                let p0 = k.load(accf, base);
                let p1 = k.load(accf, s1);
                let p2 = k.load(accf, s2);
                let p3 = k.load(accf, s3);
                let t0 = k.mad(v0, k0, p0);
                let t1 = k.mad(v1, k1, p1);
                let t2 = k.mad(v2, k2, p2);
                let t3 = k.mad(v3, k3, p3);
                k.store(accf, t0, base);
                k.store(accf, t1, s1);
                k.store(accf, t2, s2);
                k.store(accf, t3, s3);
            }
        }
        // The next super-block overwrites both tiles.
        k.barrier();
    });

    // ---- `ggml_cuda_mmq_write_back_mma`: C's two columns per lane ----
    {
        let s = k.unroll(10, 64);
        let jt = k.div(s, c8);
        let s4 = k.div(s, c4);
        let n = k.mod_(s4, c2);
        let l = k.mod_(s, c4);
        let half = k.mul(warpl, c8);
        let j0 = k.mul(jt, c16);
        let col0 = k.add(j0, half);
        let lmod = k.mod_(l, c2);
        let cpart = k.add(tig2, lmod);
        let col = k.add(col0, cpart);
        let l2 = k.div(l, c2);
        let n16 = k.mul(n, c16);
        let row = k.add(i0, n16);
        let l8 = k.mul(l2, c8);
        let row = k.add(row, l8);
        let row = k.add(row, gid);
        let nbib = k.mul(nb, cib);
        let wrow = k.add(nbib, row);
        let v = k.load(accf, s);
        let oi = k.mad(col, nstr, wrow);
        k.store(out, v, oi);
    }
    k.verify();
    k
}

/// The `mmq` block for the slowest projection, checked against a host GEMM on
/// the same int8 operands: the weight's PQ2_0 super-block dequantized to
/// `code - 1`, the activation's q8_1 payload dequantized per 32-group, as
/// `mmq` folds the two scales into the accumulator.
#[test]
fn qwen_prefill_mmq() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    let (name, kk, nn) = WORST;

    // ---- operands: the checkpoint's packed weight, fp16 activations ----
    let blocks = usize::try_from(nn * kk / BLOCK).expect("block count fits usize");
    let row = usize::try_from(ROW).expect("row fits usize");
    let mut raw = vec![0u8; blocks * row];
    let mut state = 0x2545_f491u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    for b in 0..blocks {
        let scale = 0.01f32 + (b % 97) as f32 * 0.001;
        let bits = f16::from_f32(scale).0;
        raw[b * 34] = (bits & 0xff) as u8;
        raw[b * 34 + 1] = (bits >> 8) as u8;
        for j in 0..32 {
            raw[b * 34 + 2 + j] = next();
        }
    }
    let ak = usize::try_from(M * kk).expect("activation fits usize");
    let a_f16: Vec<f16> = (0..ak).map(|i| f16::from_f32(((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5)).collect();
    let kg = usize::try_from(kgroup(kk)).expect("groups fit usize");
    let mut q8_host = vec![0i8; ak];
    // The scale as llama.cpp stores it: the f16 of `amax / 127`.
    let mut sa16_host = vec![f16(0); usize::try_from(M).unwrap() * kg];
    for m in 0..usize::try_from(M).unwrap() {
        for g in 0..kg {
            let base = m * usize::try_from(kk).unwrap() + g * 32;
            let mut amax = 0f32;
            for i in 0..32 {
                amax = amax.max(a_f16[base + i].to_f32().abs());
            }
            // `d = amax / 127.0f`; `make_half2(d, sum)` stores the f16.
            let d = amax / 127.0;
            sa16_host[m * kg + g] = f16::from_f32(d);
            for i in 0..32 {
                let v = a_f16[base + i].to_f32();
                // `q = amax == 0.0f ? 0 : roundf(xi / d)` — ties away from zero.
                let q = if amax == 0.0 { 0 } else { (v / d).round() as i32 };
                q8_host[base + i] = q as i8;
            }
        }
    }
    // The scale the projection multiplies is the stored f16 read back.
    let sa_host: Vec<f32> = sa16_host.iter().map(|s| s.to_f32()).collect();

    // ---- reference: the same quantized operands, scales folded in ----
    // Every 64th output column: 128 x 272 outputs over the whole K.
    const STRIDE: i64 = 64;
    let cols: Vec<i64> = (0..nn).step_by(usize::try_from(STRIDE).unwrap()).collect();
    let ncol = cols.len();
    let mut aref = vec![0f32; ak];
    for m in 0..usize::try_from(M).unwrap() {
        for kx in 0..usize::try_from(kk).unwrap() {
            let q = f32::from(q8_host[m * usize::try_from(kk).unwrap() + kx]);
            aref[m * usize::try_from(kk).unwrap() + kx] = q * sa_host[m * kg + kx / 32];
        }
    }
    let mut bref = vec![0f32; ncol * usize::try_from(kk).unwrap()];
    for (s, &n) in cols.iter().enumerate() {
        for kx in 0..kk {
            let blk = (n * kk + kx) / BLOCK;
            let off = (n * kk + kx) % BLOCK;
            let base = usize::try_from(blk * ROW).unwrap();
            let bits = u16::from(raw[base]) | (u16::from(raw[base + 1]) << 8);
            let sw = f16(bits).to_f32();
            let code = (raw[base + 2 + usize::try_from(off).unwrap() / 4] >> (2 * (off % 4) as u32)) & 3;
            bref[s * usize::try_from(kk).unwrap() + usize::try_from(kx).unwrap()] = (f32::from(code) - 1.0) * sw;
        }
    }
    let mut want = vec![0f32; usize::try_from(M).unwrap() * ncol];
    unsafe {
        matrixmultiply::sgemm(
            usize::try_from(M).unwrap(),
            usize::try_from(kk).unwrap(),
            ncol,
            1.0,
            aref.as_ptr(),
            kk as isize,
            1,
            bref.as_ptr(),
            1,
            kk as isize,
            0.0,
            want.as_mut_ptr(),
            ncol as isize,
            1,
        )
    };

    // ---- the q8_1 pre-pass ----
    let a: Tensor = Tensor::from_vec(a_f16.clone(), [M, kk])?.to(dev)?;
    let q8_kernel = quantize_q8_kernel(dev, kk).compile()?;
    let q8_out = q8_kernel.forward(&[&a], vec![[M, kk], [M, kgroup(kk)]])?;
    let (q8_t, sa_t) = (&q8_out[0], &q8_out[1]);
    let q8_got: Vec<i8> = q8_t.clone().try_into()?;
    let sa_got: Vec<f16> = sa_t.clone().try_into()?;
    for (i, (g, h)) in q8_got.iter().zip(q8_host.iter()).enumerate() {
        if g != h {
            return Err(ZyxError::kernel_error(format!("q8 activation {i}: {g} != {h}").into()));
        }
    }
    for (i, (g, h)) in sa_got.iter().zip(sa16_host.iter()).enumerate() {
        if g.0 != h.0 {
            return Err(ZyxError::kernel_error(format!("q8 scale {i}: {} != {}", g.to_f32(), h.to_f32()).into()));
        }
    }

    // ---- the kernel: llama.cpp's mmq block in zyx IR ----
    let q8_words: Vec<u32> =
        q8_host.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
    let w_words_host: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let a8_words: Tensor = Tensor::from_vec(q8_words, [M, kk / 4])?.to(dev)?;
    let w_words: Tensor = Tensor::from_vec(w_words_host, [nn * kk / BLOCK, ROW / 2])?.to(dev)?;
    let out_t: Tensor = Tensor::from_vec(vec![0f32; usize::try_from(M * nn).unwrap()], [M, nn])?.to(dev)?;

    let gemm = gemm_s8_kernel(dev, kk, nn);
    gemm.verify();
    let values = gemm.clone();
    let scratch = Tensor::zeros([M, nn], DType::F32).to(dev)?;
    let (_winner, nanos) = BeamSearch::new().run([gemm], &[&a8_words, &sa_t, &w_words, &scratch], &[], |_| {}, |_| 0)?;
    let gemm_nanos = nanos as f64;
    let compiled = values.compile()?;
    let out = compiled.forward(&[&a8_words, &sa_t, &w_words], vec![[M, nn]])?.pop().unwrap();

    let got: Vec<f32> = out.try_into()?;
    let mut worst = (0f32, 0i64, 0i64);
    for m in 0..usize::try_from(M).unwrap() {
        for (s, &n) in cols.iter().enumerate() {
            let g = got[m * usize::try_from(nn).unwrap() + usize::try_from(n).unwrap()];
            let w = want[m * ncol + s];
            let err = (g - w).abs();
            let tol = 2e-3f32.max(2e-3 * w.abs());
            if err > tol {
                return Err(ZyxError::kernel_error(
                    format!("qwen prefill mmq ({name}) disagrees at (m={m}, n={n}): {g} vs {w}, tol {tol}").into(),
                ));
            }
            if err > worst.0 {
                worst = (err, m as i64, n);
            }
        }
    }
    println!(
        "qwen prefill mmq ({name}) matches the int8 reference, max |err| = {:.3e} at (m={}, n={})",
        worst.0, worst.1, worst.2
    );
    let flop = 2.0 * (M * nn * kk) as f64;
    println!(
        "qwen prefill {name} [{M}x{kk}]x[{kk}x{nn}] int8 mmq: {:.4} ms, {:.2} TFLOP/s",
        gemm_nanos * 1e-6,
        flop / gemm_nanos * 1e-3
    );
    let _ = out_t;
    Ok(())
}

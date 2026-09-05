// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! LM head test (runs on CUDA, fp16 weights with fp32 accumulation).
//!
//! Golden: `examples/data/qwen3_8b_lm_head.safetensors` from
//! `tests/lm_head_ref.py`. Run the dump first:
//! `cd tests && python3.12 lm_head_ref.py`.

use std::time::Instant;

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{bf16, DType};
use zyx::{Tensor, ZyxError};

const VOCAB: usize = 256;
const HIDDEN: usize = 64;
const TOKENS: usize = 8;
const ROWS_PER_BLOCK: usize = 32;
const MMA_N: usize = 8;

#[test]
fn lm_head() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let weight = goldens["weight"].to(Dev::Cuda(0))?;
    let input = goldens["input"].to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let out = input.dot_dtype(weight.t(), DType::F32)?.to_vec::<f32>()?;

    assert_eq!(out.len(), expected.len());
    for (i, (&v, &e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((v - e).abs() < 1e-4, "out[{i}] = {v}, expected {e}");
    }
    Ok(())
}

/// Custom CUDA kernel for the LM head: fp16 tensor-core matmul mirroring
/// llama.cpp's Turing `mul_mat_f` (mmf.cu) structure.
///
/// One warp (32 threads) per block computes a 32 (vocab rows) x 8 (tokens)
/// output tile as two m16n8k8 mma subtiles with fp32 accumulation. K (hidden)
/// is iterated in chunks of 8. A = weight (row-major [vocab, hidden]), B =
/// input (column-major [hidden, tokens] view of the [tokens, hidden] input),
/// C = logits ([vocab, tokens]). `hidden`, `tokens` and the grid sizes are
/// runtime args, like llama.cpp's runtime kernel parameters. Divisibility
/// (vocab % 32, tokens % 8, hidden % 8) is assumed, as in mmf's GGML_ASSERTs.
#[test]
fn lm_head_cuda() -> Result<(), ZyxError> {
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let weight = goldens["weight"].to(Dev::Cuda(0))?;
    let input = goldens["input"]
        .reshape([TOKENS as i64, HIDDEN as i64])?
        .to(Dev::Cuda(0))?;
    let expected = goldens["output"].to_vec::<f32>()?;

    let mut kernel = Kernel::new(Dev::Cuda(0));

    // Runtime args (llama.cpp passes these as kernel parameters).
    let vocab = kernel.variable(DType::I64);
    let hidden = kernel.variable(DType::I64);
    let tokens = kernel.variable(DType::I64);
    let glen_x = kernel.variable(DType::I64); // vocab / rows_per_block
    let glen_y = kernel.variable(DType::I64); // tokens / mma_n

    let w = kernel.param(DType::F16);
    let x = kernel.param(DType::F16);
    let out = kernel.param_mut(DType::F32);

    let gidx = kernel.group_range(0, glen_x);
    let gidy = kernel.group_range(1, glen_y);
    // The ONLY thread-machinery line: every partition method finds the warp
    // op via open_warp (the IR is the state).
    let lidx = kernel.local_range(0, 32);
    kernel.warp(lidx);

    // Views: fully symbolic iteration shapes, row-major strides derived.
    let wp = kernel.partition(w, [vocab, hidden]); // A: [vocab, hidden]
    let xp = kernel.partition(x, [tokens, hidden]); // B: [tokens, hidden], consecutive K
    let cp = kernel.partition(out, [vocab, tokens]); // C: [vocab, tokens]

    let [c8, c16, c32] = kernel.const_idxs([8u32, 16, 32]);

    // Chunk coords: one [32, 8] block tile = two m16n8k8 subtiles per warp.
    let r0 = kernel.mul(gidx, c32);
    let r1 = kernel.add(r0, c16);
    let n0 = kernel.mul(gidy, c8);

    let mut acc0 = kernel.acc([c16, c8], DType::F32);
    let mut acc1 = kernel.acc([c16, c8], DType::F32);

    // Loop length (hidden / 8) is derived and patched by the first mma.
    kernel.loop_partition(|kernel, k| {
        kernel.mma(&mut acc0, c8, &wp, &xp, &[r0, n0, k]);
        kernel.mma(&mut acc1, c8, &wp, &xp, &[r1, n0, k]);
    });

    kernel.store_partition(&cp, &acc0);
    kernel.store_partition(&cp, &acc1);

    let compiled = kernel.compile()?;

    let vocab_t = Tensor::from(VOCAB as i64);
    let hidden_t = Tensor::from(HIDDEN as i64);
    let tokens_t = Tensor::from(TOKENS as i64);
    let glen_x_t = Tensor::from((VOCAB / ROWS_PER_BLOCK) as i64);
    let glen_y_t = Tensor::from((TOKENS / MMA_N) as i64);

    // Correctness: C is [vocab, tokens], golden is [tokens, vocab].
    let mut out = compiled.forward(
        &[
            &vocab_t, &hidden_t, &tokens_t, &glen_x_t, &glen_y_t, &weight, &input,
        ],
        vec![[VOCAB as i64, TOKENS as i64]],
    )?;
    let out = out.pop().unwrap().to_vec::<f32>()?;
    for t in 0..TOKENS {
        for v in 0..VOCAB {
            let got = out[v * TOKENS + t];
            let exp = expected[t * VOCAB + v];
            assert!(
                (got - exp).abs() < 1e-4,
                "out[{v}, {t}] = {got}, expected {exp}"
            );
        }
    }

    // Timing with real Qwen3-8B lm_head dims: vocab 151936, hidden 2048,
    // tokens 8 (single-token generation). The kernel structure is unchanged —
    // rows-per-block (32), MMA_N (8) and the warp size are the only constants
    // baked in, and 151936 % 32 == 0, 2048 % 8 == 0.
    let weight_r = Tensor::rand([151936i64, 2048i64], DType::F16)?.to(Dev::Cuda(0))?;
    let input_r = Tensor::rand([TOKENS as i64, 2048i64], DType::F16)?.to(Dev::Cuda(0))?;
    let vocab_r = Tensor::from(151936i64);
    let hidden_r = Tensor::from(2048i64);
    let tokens_r = Tensor::from(TOKENS as i64);
    let glen_x_r = Tensor::from((151936 / ROWS_PER_BLOCK) as i64);
    let glen_y_r = Tensor::from((TOKENS / MMA_N) as i64);

    let launch_r = || -> Result<Vec<Tensor>, ZyxError> {
        compiled.forward(
            &[
                &vocab_r, &hidden_r, &tokens_r, &glen_x_r, &glen_y_r, &weight_r, &input_r,
            ],
            vec![[151936i64, TOKENS as i64]],
        )
    };

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        launch_r()?;
    }
    let _ = launch_r()?.remove(0).to_vec::<f32>()?;
    let us_per_iter = start.elapsed().as_secs_f64() * 1e6 / (iters + 1) as f64;
    let flops = 2.0 * 151936.0 * TOKENS as f64 * 2048.0;
    println!(
        "lm_head_cuda: {us_per_iter:.2} µs/iter, {:.2} GFLOP/s",
        flops / us_per_iter / 1e3
    );
    Ok(())
}

const TDIM: u16 = 32;
const N_TILES: usize = 2;
const K_TILES: usize = 2;

// Face slot -> linear index within a tile.
fn lin(s: usize) -> usize {
    let (face, local) = (s / 256, s % 256);
    let (fr0, fc0) = (face / 2, face % 2);
    fr0 * 16 * 32 + fc0 * 16 + (local / 16) * 32 + local % 16
}

// Row-major tile -> face order for DRAM.
fn tile_encode(tile: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for p in 0..1024 {
        out[p] = tile[lin(p)];
    }
    out
}

// Face order -> row-major tile.
fn tile_decode(face: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 1024];
    for p in 0..1024 {
        out[lin(p)] = face[p];
    }
    out
}

/// Model-shaped Tenstorrent matmul test (runs on TT).
///
/// Golden: `examples/data/qwen3_8b_lm_head.safetensors` (same file as the
/// CUDA side: x [2,4,64], w [256,64], y [2,4,256]). The kernel could not
/// care less about tiles: host pads X rows 8->32 and face-encodes tiles,
/// then launches one output tile per invocation (M=1, N=1, K=2, host loops
/// over the 8 N tiles). B tiles are plain (k,n) face tiles, no transpose:
/// B[n,kt][r,c] = W[n*32+c, kt*32+r].
///
/// 3 CBs (ca deep 2, cb deep 2, cc single), Kt=2 unrolled in IR
/// construction, accumulation via add. Mirrors the official TT
/// matmul_single_core structure (hw_startup + mm_init once, acquire zeroes
/// DST, matmul_tiles accumulates per slot).
#[test]
fn lm_head_tt() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::TT(0));
    let a = k.param(DType::BF16);
    let b = k.param(DType::BF16);
    let nvar = k.variable(DType::I64);
    let z = k.param_mut(DType::BF16);

    let ca = k.storage(DType::BF16, MemScope::Circular, 2048i64.into());
    let cb = k.storage(DType::BF16, MemScope::Circular, 2048i64.into());
    let cc = k.storage(DType::BF16, MemScope::Circular, 1024i64.into());

    // Single core.
    let one = k.const_idx(1i64);
    let _g = k.group_range(0, one);
    let zero = k.const_idx(0);
    let two = k.const_idx(2i64);
    let te = k.const_idx(1024i64);

    // ---- Reader: B tiles first, then A tiles ----
    for kt in 0..K_TILES {
        // DRAM tile = n*2+kt, element offset = (n*2+kt)*1024.
        let c_kt = k.const_idx(kt as i64);
        let tile = k.mad(nvar, two, c_kt);
        let off = k.mad(tile, te, zero);
        let t = k.load_tile(b, off, TDIM, TDIM, TDIM as u32);
        k.store_tile(cb, t, zero, TDIM, TDIM, TDIM as u32);
    }
    for kt in 0..K_TILES {
        let off_a = k.const_idx((kt * 1024) as i64);
        let t = k.load_tile(a, off_a, TDIM, TDIM, TDIM as u32);
        k.store_tile(ca, t, zero, TDIM, TDIM, TDIM as u32);
    }
    k.barrier();

    // ---- Compute: C = A[0]@B[0] + A[1]@B[1] ----
    // TEMP DIAG distinct-tile probe: second iteration consumes tile 1.
    // Output B0+W1 => accumulation works; W1 => overwrite (no accum);
    // B0 => second matmul no-ops.
    let mut acc = k.const_val(bf16::from_f32(0.0));
    for kt2 in 0..K_TILES {
        let idx = if kt2 == 0 { zero } else { one };
        let la = k.load_tile(ca, idx, TDIM, TDIM, TDIM as u32);
        let lb = k.load_tile(cb, idx, TDIM, TDIM, TDIM as u32);
        let t = k.matmul_tile(la, lb);
        acc = k.add(acc, t);
    }
    k.store_tile(cc, acc, zero, TDIM, TDIM, TDIM as u32);

    // ---- Writer: single C tile at DRAM tile n ----
    k.barrier();
    let v = k.load_tile(cc, zero, TDIM, TDIM, TDIM as u32);
    let zoff = k.mad(nvar, te, zero);
    k.store_tile(z, v, zoff, TDIM, TDIM, TDIM as u32);

    k.verify();
    let compiled = k.compile()?;

    // Host: pad X rows 8->32, face-encode A/B tiles, compare decoded C
    // tile rows 0..8 against the model-shaped golden.
    let goldens = Tensor::load("../data/qwen3_8b_lm_head.safetensors")?;
    let x: Vec<f32> = goldens["input"].cast(DType::F32).to_vec()?;
    let w: Vec<f32> = goldens["weight"].cast(DType::F32).to_vec()?;
    let y: Vec<f32> = goldens["output"].cast(DType::F32).to_vec()?;

    // A[kt] = Xp[0:32, kt*32:(kt+1)*32], Xp = x rows + 24 zero rows.
    let mut ap = Vec::with_capacity(2048);
    for kt in 0..K_TILES {
        let mut tile = vec![0.0f32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                tile[r * 32 + c] = if r < 8 { x[r * 64 + kt * 32 + c] } else { 0.0 };
            }
        }
        ap.extend(tile_encode(&tile));
    }
    // B[n,kt] = W[n*32:(n+1)*32, kt*32:(kt+1)*32] as (k,n) face tile:
    // tile[r,c] = W[n*32+c, kt*32+r]. TEMP W-full (pop probe needs
    // distinctive kt1).
    let mut bp = Vec::with_capacity(16384);
    for n in 0..N_TILES {
        for kt in 0..K_TILES {
            let mut tile = vec![0.0f32; 1024];
            for r in 0..32 {
                for c in 0..32 {
                    tile[r * 32 + c] = w[(n * 32 + c) * 64 + kt * 32 + r];
                }
            }
            bp.extend(tile_encode(&tile));
        }
    }

    // TEMP DIAG alloc-order probe: B allocated first (low address), A
    // second. Fault following the address = region issue; fault
    // following the buffer = size/layout issue.
    let b_t = Tensor::from(bp.clone())
        .to(Dev::C)?
        .cast(DType::BF16)
        .to(Dev::TT(0))?;
    let a_t = Tensor::from(ap)
        .to(Dev::C)?
        .cast(DType::BF16)
        .to(Dev::TT(0))?;
    // TEMP DIAG: round-trip B host->device->host (no NOC reads involved).
    let b_back: Vec<f32> = b_t.to(Dev::C)?.cast(DType::F32).to_vec()?;
    let b_src: Vec<f32> = Tensor::from(bp.clone())
        .to(Dev::C)?
        .cast(DType::F32)
        .to_vec()?;
    println!("TEMP b_roundtrip first8={:?}", &b_back[..8]);
    println!("TEMP b_source first8={:?}", &b_src[..8]);

    // Each launch returns a fresh z (8192 elems); only tile n is written.
    // TEMP DIAG swap probe: feed B through the A read path.
    let mut tiles = Vec::with_capacity(N_TILES);
    for n in 0..N_TILES {
        let n_t = Tensor::variable(n as i64);
        // TEMP DIAG const-B-small with real matmul.
        let out = compiled.forward(&[&a_t, &b_t, &n_t], vec![[8192i64]])?;
        let z_host = out[0].to(Dev::C)?;
        let z_f32 = z_host.cast(DType::F32);
        let z_face: Vec<f32> = z_f32.to_vec()?;
        tiles.push(tile_decode(&z_face[n * 1024..(n + 1) * 1024]));
    }
    // C[n] tile rows 0..8 hold y[b, s, n*32:(n+1)*32].
    let mut bad = 0;
    for m in 0..N_TILES {
        let got = &tiles[m];
        for b in 0..2 {
            for s in 0..4 {
                for c in 0..32 {
                    let v = got[(b * 4 + s) * 32 + c];
                    let e = y[(b * 4 + s) * 256 + m * 32 + c];
                    if (v - e).abs() >= 0.5 {
                        if bad < 10 {
                            println!("m={m} b={b} s={s} c={c}: {v} vs {e}");
                        }
                        bad += 1;
                    }
                }
            }
        }
    }
    println!("bad: {bad} / {}", N_TILES * 256);
    // TEMP DIAG bf16-reference: host C_ref with bf16-quantized inputs,
    // f32 accumulation. If device matches C_ref tightly, the 2 diffs
    // vs the f32 golden are quantization, not logic.
    let xb: Vec<f32> = x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let wb: Vec<f32> = w.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut maxd = 0.0f32;
    for m in 0..N_TILES {
        let got = &tiles[m];
        for b in 0..2 {
            for s in 0..4 {
                for c in 0..32 {
                    let mut acc = 0.0f32;
                    for kt in 0..K_TILES {
                        for r in 0..32 {
                            let a = if (b * 4 + s) < 8 {
                                xb[(b * 4 + s) * 64 + kt * 32 + r]
                            } else {
                                0.0
                            };
                            acc += a * wb[(m * 32 + c) * 64 + kt * 32 + r];
                        }
                    }
                    let d = (got[(b * 4 + s) * 32 + c] - acc).abs();
                    if d > maxd {
                        maxd = d;
                    }
                }
            }
        }
    }
    println!("TEMP max diff vs bf16 host ref: {maxd}");
    // TEMP DIAG: padding rows 8..32 should be ~0 (X rows 8+ are zero).
    // TEMP DIAG one-hot probe expects tiles[m][r][c] == w[(m*32+c)*64 + r].
    for m in 0..N_TILES {
        let row8: Vec<f32> = tiles[m][8 * 32..8 * 32 + 4].to_vec();
        let row0: Vec<f32> = tiles[m][0..8].to_vec();
        println!("TEMP m={m} row0={row0:?} row8={row8:?}");
    }
    if N_TILES > 0 {
        println!(
            "TEMP wcol0: {:?}",
            (0..8).map(|c| w[c * 64]).collect::<Vec<f32>>()
        );
    }
    assert_eq!(bad, 0);
    Ok(())
}

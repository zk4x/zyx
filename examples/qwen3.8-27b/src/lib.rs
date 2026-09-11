// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Qwen3.8-27B inference example (UD-Q4_K_XL).
//!
//! Per-op verification against torch goldens: each op in `tests/` has a
//! `<op>.rs` test and a `<op>_ref.py` golden-dump script. Reference side
//! runs on CUDA, tiled kernels run on Tenstorrent.
//!
//! Shared custom-kernel builders live here so every test assembles the
//! same kernels. The rule for the kernel path: custom kernels only — no
//! tensor ops between them, layouts baked into the kernel index math.

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{f16, DType, Tensor, ZyxError};

// Qwen3.8-27B linear-attention geometry.
pub const S: i64 = 6; // demo seq len (single chunk)
pub const M_PAD: i64 = 16; // GEMM row multiple
pub const HIDDEN: i64 = 5120;
pub const KH: i64 = 16; // key heads
pub const VH: i64 = 48; // value heads
pub const KD: i64 = 128;
pub const VD: i64 = 128;
pub const KEY_DIM: i64 = 2048; // KH * KD
pub const VAL_DIM: i64 = 6144; // VH * VD
pub const CONV_DIM: i64 = 10240; // KEY_DIM * 2 + VAL_DIM
pub const CK: i64 = 4; // depthwise conv kernel
pub const CH: i64 = 64; // delta-rule chunk size
pub const DT_RANK: i64 = 48; // beta/decay rank (= VH)
pub const INTERMEDIATE: i64 = 17408; // FFN intermediate size
pub const HEADS: i64 = 24; // full-attn heads
pub const KV_HEADS: i64 = 4; // full-attn KV heads
pub const HEAD_DIM: i64 = 256; // full-attn head dim
pub const ROT_DIM: i64 = 64; // partial RoPE (HEAD_DIM / 4)
pub const FFN_DIM: i64 = INTERMEDIATE; // alias

/// Zero-pads rows: in [s, d] f32 -> out [m, d] f16.
pub fn pad_kernel(s: i64, m: i64, d: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let inp = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F16);
    let [cc, row] = kernel.group_ranges([d / 32, m]);
    let [lane] = kernel.local_ranges([32]);
    let c = kernel.mad(cc, 32i64, lane);
    let keep = kernel.cmplt(row, s);
    let safe_row = kernel.branchless_where(keep, row, 0i64);
    let safe_idx = kernel.mad(safe_row, d, c);
    let x_raw = kernel.load(inp, safe_idx);
    let x = kernel.cast(x_raw, DType::F16);
    let v = kernel.branchless_where(keep, x, f16::from_f32(0.0));
    let out_idx = kernel.mad(row, d, c);
    kernel.store(out, v, out_idx);
    kernel.default_epilogue();
    kernel
}

/// Tenstorrent port of `pad_kernel` (tile path, mask input).
///
/// F32 tilized data `[mp, dp]` + F32 tilized mask (1.0 on kept rows) ->
/// F16 tilized out, where `mp`/`dp` are `m`/`d` padded to multiples of
/// 32. A single group loops over all tiles (`loop_over`); movement is
/// bulk tile transfers; predication is masking (`mul_binary_tile`);
/// conversion is `typecast_tile`. No scalar data movement.
pub fn pad_kernel_tt(s: i64, m: i64, d: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(s <= m);
    let mp = (m + 31) / 32 * 32;
    let dp = (d + 31) / 32 * 32;
    let ntiles = mp / 32 * (dp / 32);
    let mut kernel = Kernel::new(Dev::TT(0));
    let data = kernel.param(DType::F32);
    let mask = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F16);

    let cdata = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cmask = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cout = kernel.storage(DType::F16, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ntiles = kernel.const_idx(ntiles);
    let tile_elems = kernel.const_idx(TILE_ELEMS);
    let zero = kernel.const_idx(0);

    // Reader: whole-tile DRAM -> CB transfers (tilized DRAM).
    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let td = kernel.load_tile(data, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cdata, td, zero, TDIM, TDIM, TDIM as u32);
        let tm = kernel.load_tile(mask, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cmask, tm, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    // Compute: apply mask, convert to F16.
    kernel.loop_over(ntiles, |kernel, _t| {
        let a = kernel.load_tile(cdata, zero, TDIM, TDIM, TDIM as u32);
        let b = kernel.load_tile(cmask, zero, TDIM, TDIM, TDIM as u32);
        let mm = kernel.mul(a, b);
        let h = kernel.cast(mm, DType::F16);
        kernel.store_tile(cout, h, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    // Writer: whole-tile CB -> DRAM transfer.
    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let v = kernel.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// Bring-up diagnostic: pure F32 tile movement, empty compute section.
/// Passes iff reader/writer/CB/DRAM paths are all correct; isolates
/// compute math/pack as the suspect when values come back wrong.
pub fn pad_move_tt(s: i64, m: i64, d: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(s <= m);
    let mp = (m + 31) / 32 * 32;
    let dp = (d + 31) / 32 * 32;
    let ntiles = mp / 32 * (dp / 32);
    let mut kernel = Kernel::new(Dev::TT(0));
    let data = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F32);

    let cdata = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ntiles = kernel.const_idx(ntiles);
    let tile_elems = kernel.const_idx(TILE_ELEMS);
    let zero = kernel.const_idx(0);

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let td = kernel.load_tile(data, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cdata, td, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let v = kernel.load_tile(cdata, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// Bring-up diagnostic: F32 copy in compute (copy_tile + pack, no cast,
/// no binary op). Passes iff copy/pack are correct; isolates the F32->F16
/// typecast as the suspect when values come back wrong.
pub fn pad_copy_tt(s: i64, m: i64, d: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(s <= m);
    let mp = (m + 31) / 32 * 32;
    let dp = (d + 31) / 32 * 32;
    let ntiles = mp / 32 * (dp / 32);
    let mut kernel = Kernel::new(Dev::TT(0));
    let data = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F32);

    let cdata = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cout = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ntiles = kernel.const_idx(ntiles);
    let tile_elems = kernel.const_idx(TILE_ELEMS);
    let zero = kernel.const_idx(0);

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let td = kernel.load_tile(data, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cdata, td, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, _t| {
        let a = kernel.load_tile(cdata, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cout, a, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let v = kernel.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// Bring-up diagnostic: single-F32-tile passthrough (copy + cast, no
/// binary op). Isolates reader/tilize/cast/pack/writer from binary-op
/// unpack configuration when values come back wrong.
pub fn pad_passthrough_tt(s: i64, m: i64, d: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(s <= m);
    let mp = (m + 31) / 32 * 32;
    let dp = (d + 31) / 32 * 32;
    let ntiles = mp / 32 * (dp / 32);
    let mut kernel = Kernel::new(Dev::TT(0));
    let data = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F16);

    let cdata = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cout = kernel.storage(DType::F16, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ntiles = kernel.const_idx(ntiles);
    let tile_elems = kernel.const_idx(TILE_ELEMS);
    let zero = kernel.const_idx(0);

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let td = kernel.load_tile(data, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cdata, td, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, _t| {
        let a = kernel.load_tile(cdata, zero, TDIM, TDIM, TDIM as u32);
        let h = kernel.cast(a, DType::F16);
        kernel.store_tile(cout, h, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let v = kernel.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// Tenstorrent standalone F32->F16 typecast (tilized): F32 tilized in ->
/// F16 tilized out, one tile per iteration, no compute math. Runs in
/// 16-bit DST (no F32 output), which is what typecast.h sanctions for the
/// F32->F16b SFPU op. Second stage of the split mixed-format kernels.
pub fn pad_cast_tt(s: i64, m: i64, d: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(s <= m);
    let mp = (m + 31) / 32 * 32;
    let dp = (d + 31) / 32 * 32;
    let ntiles = mp / 32 * (dp / 32);
    let mut kernel = Kernel::new(Dev::TT(0));
    let data = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F16);

    let cdata = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cout = kernel.storage(DType::F16, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ntiles = kernel.const_idx(ntiles);
    let tile_elems = kernel.const_idx(TILE_ELEMS);
    let zero = kernel.const_idx(0);

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let td = kernel.load_tile(data, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cdata, td, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, _t| {
        let a = kernel.load_tile(cdata, zero, TDIM, TDIM, TDIM as u32);
        let h = kernel.cast(a, DType::F16);
        kernel.store_tile(cout, h, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let v = kernel.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// Tenstorrent masked F32 tile multiply (first stage of the split pad):
/// F32 tilized data `[mp, dp]` + F32 tilized mask -> F32 tilized out.
/// Runs in 32-bit DST (F32 output). Packs F32; the F16 conversion is a
/// separate `pad_cast_tt` launch.
pub fn pad_mul_tt(s: i64, m: i64, d: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(s <= m);
    let mp = (m + 31) / 32 * 32;
    let dp = (d + 31) / 32 * 32;
    let ntiles = mp / 32 * (dp / 32);
    let mut kernel = Kernel::new(Dev::TT(0));
    let data = kernel.param(DType::F32);
    let mask = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F32);

    let cdata = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cmask = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);
    let cout = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ntiles = kernel.const_idx(ntiles);
    let tile_elems = kernel.const_idx(TILE_ELEMS);
    let zero = kernel.const_idx(0);

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let td = kernel.load_tile(data, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cdata, td, zero, TDIM, TDIM, TDIM as u32);
        let tm = kernel.load_tile(mask, tbase, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cmask, tm, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, _t| {
        let a = kernel.load_tile(cdata, zero, TDIM, TDIM, TDIM as u32);
        let b = kernel.load_tile(cmask, zero, TDIM, TDIM, TDIM as u32);
        let mm = kernel.mul(a, b);
        kernel.store_tile(cout, mm, zero, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    kernel.loop_over(ntiles, |kernel, t| {
        let tbase = kernel.mad(t, tile_elems, zero);
        let v = kernel.load_tile(cout, zero, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, tbase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// Tenstorrent GEMM (official `matmul_single_core` structure, single
/// core): out [R, N] F32 = A [R, K] F16 @ B [N, K]^T F16, F32 DST
/// accumulate.
///
/// Tilized DRAM on all sides; B stored KN-tilized (`[Kt, Nt]` tiles).
/// Outer loop over output tiles; inner loop over Kt streaming 1 A + 1 B
/// tile per iteration (slot 0, pops advance the FIFO). The running sum
/// is an explicit acc CB: the IR carries loop-carried values through
/// named storage (verify-clean, passes can reason about it), and
/// codegen folds the add onto DST accumulation, emitting the official
/// wait/matmul/pop loop. 4 params: a, b, zero tile, out.
pub fn gemm_tt(r: i64, k: i64, n: i64) -> Kernel {
    const TDIM: u16 = 32;
    const TILE_ELEMS: i64 = 1024;
    debug_assert!(r % 32 == 0 && k % 32 == 0 && n % 32 == 0);
    debug_assert!(r == 32, "gemm_tt: single tile row (Mt==1) for now");
    let kt = k / 32;
    let nt = n / 32;
    let mut kernel = Kernel::new(Dev::TT(0));
    let a = kernel.param(DType::F16);
    let b = kernel.param(DType::F16);
    let zero = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F32);

    let ca = kernel.storage(DType::F16, MemScope::Circular, TILE_ELEMS);
    let cb_ = kernel.storage(DType::F16, MemScope::Circular, TILE_ELEMS);
    let acc = kernel.storage(DType::F32, MemScope::Circular, 2 * TILE_ELEMS);
    let cout = kernel.storage(DType::F32, MemScope::Circular, TILE_ELEMS);

    let _g = kernel.group_range(0, 1);
    let ckt = kernel.const_idx(kt);
    let cnt = kernel.const_idx(nt);
    let c1024 = kernel.const_idx(TILE_ELEMS);
    let c0 = kernel.const_idx(0);
    let c_nt1024 = kernel.const_idx(nt * TILE_ELEMS);

    // Reader: per output tile, seed acc with zero, then stream Kt pairs.
    // NOTE: the zero seed store emits nothing (scratch suppression) and
    // exists only so the acc CB registers in cb_map; the DST sum is seeded
    // by the outer acquire, which zeroes DST for free.
    kernel.loop_over(cnt, |kernel, nti| {
        let z = kernel.load_tile(zero, c0, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(acc, z, c0, TDIM, TDIM, TDIM as u32);
        kernel.loop_over(ckt, |kernel, kti| {
            let abase = kernel.mad(kti, c1024, c0);
            let ta = kernel.load_tile(a, abase, TDIM, TDIM, TDIM as u32);
            kernel.store_tile(ca, ta, c0, TDIM, TDIM, TDIM as u32);
            let nti1024 = kernel.mad(nti, c1024, c0);
            let bbase = kernel.mad(kti, c_nt1024, nti1024);
            let tb = kernel.load_tile(b, bbase, TDIM, TDIM, TDIM as u32);
            kernel.store_tile(cb_, tb, c0, TDIM, TDIM, TDIM as u32);
        });
    });
    kernel.barrier();

    // Compute: per output tile, fold Kt products into acc, pack out.
    kernel.loop_over(cnt, |kernel, _nti| {
        kernel.loop_over(ckt, |kernel, _kti| {
            let va = kernel.load_tile(ca, c0, TDIM, TDIM, TDIM as u32);
            let vb = kernel.load_tile(cb_, c0, TDIM, TDIM, TDIM as u32);
            let vc = kernel.load_tile(acc, c0, TDIM, TDIM, TDIM as u32);
            let m = kernel.matmul_tile(va, vb);
            let s = kernel.add(vc, m);
            kernel.store_tile(acc, s, c0, TDIM, TDIM, TDIM as u32);
        });
        let f = kernel.load_tile(acc, c0, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(cout, f, c0, TDIM, TDIM, TDIM as u32);
    });
    kernel.barrier();

    // Writer: stream packed output tiles to DRAM.
    kernel.loop_over(cnt, |kernel, nti| {
        let obase = kernel.mad(nti, c1024, c0);
        let v = kernel.load_tile(cout, c0, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, obase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

/// GEMM with 16-row blocks, n=8 (lm_head pattern): out [R, N] =
/// A [R, K] @ B [N, K]^T. `R/K/N` are emitted as constants so `flop_mem_rw`
/// and the compiler see concrete trip counts; compile one instance per
/// projection shape (10 ms each).
pub fn gemm_kernel(r: i64, k: i64, n: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [a, b] = kernel.params([DType::F16; 2]);
    let out = kernel.param_mut(DType::F32);

    let glen_x = r / 16;
    let glen_y = n / 8;
    let [gidx, gidy] = kernel.group_ranges([glen_x, glen_y]);
    let lidx = kernel.local_range(0, 32);
    kernel.warp(lidx);

    let [rr, kk, nn] = kernel.const_idxs([r, k, n]);
    let ap = kernel.view_global_register(a, [rr, kk]);
    let bp = kernel.view_global_register(b, [nn, kk]);
    let cp = kernel.view_global_register(out, [rr, nn]);

    let [c_block, c8] = kernel.const_idxs([16, 8]);
    let r0 = kernel.mul(gidx, c_block);
    let n0 = kernel.mul(gidy, c8);
    let acc = kernel.acc([c_block, c8], DType::F32);
    kernel.loop_partition(|kernel, kk| {
        kernel.mma_at(&acc, &ap, &bp, [r0, n0, kk]);
    });
    kernel.store_partition(&cp, &acc, [r0, n0]);
    kernel.default_epilogue();
    kernel
}

/// Fused Q4_K × Q8_1 GEMM. A is Q4_K (R/16, K/256) blocks; B is Q8_1
/// (N/8, K/32) blocks. Uses `view_global_local` + `stage_global_local_fused`
/// to dequantize during the stage (like `lm_head_cuda_local`).
pub fn gemm_cuda_q4_k(r: i64, k: i64, n: i64) -> Kernel {
    assert!(r % 16 == 0 && n % 8 == 0 && k % 32 == 0);
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let a_qs = kernel.param(DType::U32);
    let a_scales = kernel.param(DType::F16);
    let a_mins = kernel.param(DType::F16);
    let b_qs = kernel.param(DType::U32);
    let b_scales = kernel.param(DType::F16);
    let b_sums = kernel.param(DType::F16);
    let out = kernel.param_mut(DType::F32);
    let [rr, _, nn] = kernel.const_idxs([r, k, n]);
    let c16 = kernel.const_idx(16);
    let c8 = kernel.const_idx(8);
    let c4 = kernel.const_idx(4);
    let c0 = kernel.const_idx(0);
    let glen_x = r / 16;
    let glen_y = n / 8;
    let kblocks = kernel.const_idx(k / 32);
    let [gidx, gidy] = kernel.group_ranges([glen_x, glen_y]);
    let lidx = kernel.local_range(0, 32);
    kernel.warp(lidx);
    let cp = kernel.view_global_register(out, [rr, nn]);
    let r0 = kernel.mul(gidx, c16);
    let n0 = kernel.mul(gidy, c8);
    let acc = kernel.acc([c16, c8], DType::F32);
    let a_rblocks = kernel.div(rr, c16);
    let b_nblocks = kernel.div(nn, c8);
    let a_shared = kernel.view_global_local(a_qs, [a_rblocks, kblocks, c4], [c16, c4, c4], c4, 1);
    let b_shared = kernel.view_global_local(b_qs, [b_nblocks, kblocks, c4], [c8, c4, c4], c4, 1);
    kernel.loop_over(kblocks, |kernel, kb| {
        let a_scale_idx = kernel.mad(gidx, kblocks, kb);
        let a_scale = kernel.load(a_scales, a_scale_idx);
        let a_min = kernel.load(a_mins, a_scale_idx);
        kernel.stage_global_local_fused(&a_shared, [gidx, kb, c0], |kernel, v| {
            kernel.dequant_q4_k(v, a_scale, a_min)
        });
        let b_scale_idx = kernel.mad(gidy, kblocks, kb);
        let b_scale = kernel.load(b_scales, b_scale_idx);
        let b_sum = kernel.load(b_sums, b_scale_idx);
        kernel.stage_global_local_fused(&b_shared, [gidy, kb, c0], |kernel, v| {
            kernel.dequant_q8_1(v, b_scale, b_sum)
        });
        kernel.barrier();
        let ap = kernel.view_local_register(&a_shared);
        let bp = kernel.view_local_register(&b_shared);
        kernel.mma(&acc, &ap, &bp);
        kernel.barrier();
    });
    kernel.store_partition(&cp, &acc, [r0, n0]);
    kernel.default_epilogue();
    kernel
}

/// Depthwise causal conv1d (kernel 4, left pad 3) + SiLU:
pub fn conv_silu_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [inp, convw] = kernel.params([DType::F32; 2]);
    let out = kernel.param_mut(DType::F32);
    let [cg, t] = kernel.group_ranges([CONV_DIM / 32, M_PAD]);
    let [lane] = kernel.local_ranges([32]);
    let c = kernel.mad(cg, 32i64, lane);

    let mut acc = kernel.const_val(0.0f32);
    for k in 0..CK {
        // Causal offset as an op (can go negative); clamp guards the load.
        let t_k = kernel.add(t, k);
        let src_t = kernel.sub(t_k, CK - 1);
        let ok = kernel.cmpge(src_t, 0i64);
        let src_c = kernel.branchless_where(ok, src_t, 0i64);
        let in_idx = kernel.mad(src_c, CONV_DIM, c);
        let x_raw = kernel.load(inp, in_idx);
        let zf = kernel.const_val(0.0f32);
        let x = kernel.branchless_where(ok, x_raw, zf);
        let w_idx = kernel.mad(c, CK, k);
        let w = kernel.load(convw, w_idx);
        acc = kernel.mad(w, x, acc);
    }
    // silu(x) = x / (1 + 2^(-x*log2e))
    let y = kernel.silu(acc);
    let out_idx = kernel.mad(t, CONV_DIM, c);
    kernel.store(out, y, out_idx);
    kernel.default_epilogue();
    kernel
}

/// Gated delta-rule core, token-recurrent (llama.cpp gated_delta_net):
/// mixed [M_PAD, CONV_DIM] f32 (conv+SiLU output), b/a [M_PAD, DT_RANK]
/// f32, exp_a_log/dt_bias [DT_RANK] f32 -> core [VH, S, VD] f32.
/// One block per (v-head, column-block); 4 warps own 4 state columns;
/// each lane holds 4 of its column's 128 rows. l2norm, q scale, k-head
/// expansion (h/3), sigmoid(beta), softplus/exp(g) all inline.
pub fn delta_core_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [mixed, bp, ap, ealog, dtb] = kernel.params([DType::F32; 5]);
    let out = kernel.param_mut(DType::F32);

    let [h, cblk] = kernel.group_ranges([VH, VD / 4]);
    let [lane, w4] = kernel.local_ranges([32, 4]);
    let cw = kernel.mul(cblk, 4i64);
    let col = kernel.add(cw, w4);
    let kh = kernel.div(h, VH / KH);

    let s = kernel.storage(DType::F32, MemScope::Register, 4i64);
    let zero = kernel.const_val(0.0f32);
    for r in 0..4i64 {
        kernel.store(s, zero, r);
    }
    let eps = kernel.const_val(1e-6f32);
    // llama.cpp delta-net-base: scale = 1/sqrt(S_k) on q only.
    let qscale = kernel.const_val(1.0f32 / (KD as f32).sqrt());

    kernel.loop_over(S, |kernel, t| {
        // Row bases: q/k rows are KD-contiguous in mixed, v rows VD.
        let t_q = kernel.mul(t, CONV_DIM);
        let t_h = kernel.mul(t, DT_RANK);
        let kh_kd = kernel.mul(kh, KD);
        let h_vd = kernel.mul(h, VD);
        let q_base = kernel.add(t_q, kh_kd);
        let k_off = kernel.add(KEY_DIM, kh_kd);
        let k_base = kernel.add(t_q, k_off);
        let v_off = kernel.add(2 * KEY_DIM, h_vd);
        let v_base = kernel.add(t_q, v_off);
        let g_idx = kernel.add(t_h, h);

        // l2norm scales (butterfly sumsq over the full KD row per warp).
        let mut qss = kernel.const_val(0.0f32);
        let mut kss = kernel.const_val(0.0f32);
        for r in 0..4i64 {
            let d = kernel.add(lane, 32 * r);
            let qi = kernel.add(q_base, d);
            let ki = kernel.add(k_base, d);
            let q_raw = kernel.load(mixed, qi);
            let k_raw = kernel.load(mixed, ki);
            qss = kernel.mad(q_raw, q_raw, qss);
            kss = kernel.mad(k_raw, k_raw, kss);
        }
        let qss = kernel.warp_reduce(qss);
        let kss = kernel.warp_reduce(kss);
        let qn_denom = kernel.add(qss, eps);
        let kn_denom = kernel.add(kss, eps);
        let qn_sqrt = kernel.sqrt(qn_denom);
        let kn_sqrt = kernel.sqrt(kn_denom);
        let qn_r = kernel.reciprocal(qn_sqrt);
        let kn_r = kernel.reciprocal(kn_sqrt);
        let qn = kernel.mul(qn_r, qscale);

        // beta = sigmoid(b), g = exp(-exp_a_log * softplus(a + dt_bias)).
        let b_raw = kernel.load(bp, g_idx);
        let beta = kernel.sigmoid(b_raw);
        let a_raw = kernel.load(ap, g_idx);
        let dt_raw = kernel.load(dtb, h);
        let adt = kernel.add(a_raw, dt_raw);
        let sp = kernel.softplus(adt, 20.0f32);
        let ea_raw = kernel.load(ealog, h);
        let g_arg = kernel.mul(ea_raw, sp);
        let g_neg = kernel.neg(g_arg);
        let gval = kernel.exp(g_neg);

        // kv = S^T k (k normalized on the fly).
        let mut partial = kernel.const_val(0.0f32);
        for r in 0..4i64 {
            let d = kernel.add(lane, 32 * r);
            let ki = kernel.add(k_base, d);
            let k_raw = kernel.load(mixed, ki);
            let kf = kernel.mul(k_raw, kn_r);
            let sr = kernel.load(s, r);
            partial = kernel.mad(sr, kf, partial);
        }
        let kv = kernel.warp_reduce(partial);
        let gkv = kernel.mul(gval, kv);
        let v_col = kernel.add(v_base, col);
        let vf = kernel.load(mixed, v_col);
        let vmg = kernel.sub(vf, gkv);
        let delta = kernel.mul(vmg, beta);

        // S = g*S + k*delta^T; out partial = S^T q (q scaled on the fly).
        let mut attn = kernel.const_val(0.0f32);
        for r in 0..4i64 {
            let d = kernel.add(lane, 32 * r);
            let ki = kernel.add(k_base, d);
            let k_raw = kernel.load(mixed, ki);
            let kf = kernel.mul(k_raw, kn_r);
            let sr = kernel.load(s, r);
            let gs = kernel.mul(gval, sr);
            let kdel = kernel.mul(kf, delta);
            let s_new = kernel.add(gs, kdel);
            kernel.store(s, s_new, r);
            let qi = kernel.add(q_base, d);
            let q_raw = kernel.load(mixed, qi);
            let qf = kernel.mul(q_raw, qn);
            attn = kernel.mad(s_new, qf, attn);
        }
        let attn = kernel.warp_reduce(attn);

        // All 32 lanes write the same per-warp column value.
        let ot = kernel.mul(t, VD);
        let oh = kernel.mul(h, S * VD);
        let o_base = kernel.add(oh, ot);
        let o_idx = kernel.add(o_base, col);
        kernel.store(out, attn, o_idx);
    });
    kernel.default_epilogue();
    kernel
}

/// Gated RMSNorm per (token, v-head) 128-row: core [VH, S, VD] f32,
/// z [M_PAD, VAL_DIM] f32, norm_w [VD] f32 -> normed [S, VAL_DIM] f32.
/// out row = core*rsqrt(mean+eps)*norm_w*silu(z).
pub fn rmsnorm_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [core, zp, nw] = kernel.params([DType::F32; 3]);
    let out = kernel.param_mut(DType::F32);
    let [t, h] = kernel.group_ranges([S, VH]);
    let [lane] = kernel.local_ranges([32]);
    let eps = kernel.const_val(1e-6f32);

    let tt = kernel.mul(t, VD);
    let hh = kernel.mul(h, S * VD);
    let c_base = kernel.add(hh, tt);
    let mut ss = kernel.const_val(0.0f32);
    for r in 0..4i64 {
        let d = kernel.add(lane, 32 * r);
        let ci = kernel.add(c_base, d);
        let c_raw = kernel.load(core, ci);
        ss = kernel.mad(c_raw, c_raw, ss);
    }
    let ss = kernel.warp_reduce(ss);
    let mean = kernel.div(ss, 128.0f32);
    let denom = kernel.add(mean, eps);
    let sq = kernel.sqrt(denom);
    let scale = kernel.reciprocal(sq);

    for r in 0..4i64 {
        let d = kernel.add(lane, 32 * r);
        let ci = kernel.add(c_base, d);
        let c_raw = kernel.load(core, ci);
        let n = kernel.mul(c_raw, scale);
        let w = kernel.load(nw, d);
        let nw_v = kernel.mul(n, w);
        let th = kernel.mul(t, VAL_DIM);
        let hd = kernel.mad(h, VD, d);
        let z_off = kernel.add(th, hd);
        let z_raw = kernel.load(zp, z_off);
        let gz = kernel.silu(z_raw);
        let y = kernel.mul(nw_v, gz);
        let o_off = kernel.add(th, hd);
        kernel.store(out, y, o_off);
    }
    kernel.default_epilogue();
    kernel
}

/// Element-wise residual add: out [m, d] = a [m, d] + b [m, d] (F32).
pub fn residual_add_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [a, b] = kernel.params([DType::F32; 2]);
    let out = kernel.param_mut(DType::F32);
    let m = M_PAD;
    let d = HIDDEN;
    let [cg, t] = kernel.group_ranges([d / 32, m]);
    let [lane] = kernel.local_ranges([32]);
    let c = kernel.mad(cg, 32i64, lane);
    let idx = kernel.mad(t, d, c);
    let av = kernel.load(a, idx);
    let bv = kernel.load(b, idx);
    let y = kernel.add(av, bv);
    kernel.store(out, y, idx);
    kernel.default_epilogue();
    kernel
}

/// Input RMSNorm: x [S, HIDDEN] F32, weight [HIDDEN] F32 -> out [S, HIDDEN] F32.
/// out = x * rsqrt(mean(x^2) + eps) * weight.
pub fn input_rmsnorm_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [x, w] = kernel.params([DType::F32; 2]);
    let out = kernel.param_mut(DType::F32);
    let [t, h] = kernel.group_ranges([S, HIDDEN / 32]);
    let [lane] = kernel.local_ranges([32]);
    let eps = kernel.const_val(1e-6f32);
    let hidden_c = kernel.const_idx(HIDDEN);
    let c = kernel.mad(h, 32i64, lane);
    let idx = kernel.mad(t, HIDDEN, c);
    let x_raw = kernel.load(x, idx);
    let mut ss = kernel.const_val(0.0f32);
    for r in 0..(HIDDEN / 32) {
        let d = kernel.add(lane, 32 * r);
        let idx2 = kernel.mad(t, HIDDEN, d);
        let x_raw2 = kernel.load(x, idx2);
        ss = kernel.mad(x_raw2, x_raw2, ss);
    }
    let ss = kernel.warp_reduce(ss);
    let hidden_f = kernel.const_val(HIDDEN as f32);
    let mean = kernel.div(ss, hidden_f);
    let denom = kernel.add(mean, eps);
    let sq = kernel.sqrt(denom);
    let scale = kernel.reciprocal(sq);
    let w_raw = kernel.load(w, c);
    let n = kernel.mul(x_raw, scale);
    let y = kernel.mul(n, w_raw);
    let _ = hidden_c;
    kernel.store(out, y, idx);
    kernel.default_epilogue();
    kernel
}

/// Per-head Q/K norm: q [H, S, D] F32, k [KV, S, D] F32 -> q_out, k_out.
/// out = x * rsqrt(mean(x^2) + eps) * weight (per-head RMSNorm with weight).
/// `heads` and `kv_heads` are passed to support both linear-attn (24/16) and
/// full-attn (24/4) cases. `head_dim` parameterizes the per-head dim.
pub fn qk_norm_kernel(heads: i64, kv_heads: i64, head_dim: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [q, k, qw, kw] = kernel.params([DType::F32; 4]);
    let q_out = kernel.param_mut(DType::F32);
    let k_out = kernel.param_mut(DType::F32);
    let [t, h] = kernel.group_ranges([S, heads]);
    let [lane] = kernel.local_ranges([32]);
    let eps = kernel.const_val(1e-6f32);
    let head_dim_c = kernel.const_idx(head_dim);
    let s_hd_c = kernel.const_idx(S * head_dim);
    // q: [H, S, head_dim] indexed as h*S*hd + t*hd + d
    let h_shd = kernel.mul(h, s_hd_c);
    let t_hd = kernel.mul(t, head_dim_c);
    let q_base = kernel.add(h_shd, t_hd);
    let k_head = kernel.div(h, heads / kv_heads);
    let kh_shd = kernel.mul(k_head, s_hd_c);
    let k_base = kernel.add(kh_shd, t_hd);
    // Q norm: ss over head_dim
    let mut ss_q = kernel.const_val(0.0f32);
    for r in 0..(head_dim / 32) {
        let d = kernel.add(lane, 32 * r);
        let qd = kernel.add(q_base, d);
        let q_v = kernel.load(q, qd);
        ss_q = kernel.mad(q_v, q_v, ss_q);
    }
    let head_dim_f = kernel.const_val(head_dim as f32);
    let ss_q = kernel.warp_reduce(ss_q);
    let mean_q = kernel.div(ss_q, head_dim_f);
    let denom_q = kernel.add(mean_q, eps);
    let sq_q = kernel.sqrt(denom_q);
    let scale_q = kernel.reciprocal(sq_q);
    // K norm: ss over head_dim
    let mut ss_k = kernel.const_val(0.0f32);
    for r in 0..(head_dim / 32) {
        let d = kernel.add(lane, 32 * r);
        let kd = kernel.add(k_base, d);
        let k_v = kernel.load(k, kd);
        ss_k = kernel.mad(k_v, k_v, ss_k);
    }
    let ss_k = kernel.warp_reduce(ss_k);
    let mean_k = kernel.div(ss_k, head_dim_f);
    let denom_k = kernel.add(mean_k, eps);
    let sq_k = kernel.sqrt(denom_k);
    let scale_k = kernel.reciprocal(sq_k);
    for r in 0..(head_dim / 32) {
        let d = kernel.add(lane, 32 * r);
        let qd = kernel.add(q_base, d);
        let q_v = kernel.load(q, qd);
        let qw_v = kernel.load(qw, d);
        let n_q = kernel.mul(q_v, scale_q);
        let n_q = kernel.mul(n_q, qw_v);
        kernel.store(q_out, n_q, qd);
        let kd = kernel.add(k_base, d);
        let k_v = kernel.load(k, kd);
        let kw_v = kernel.load(kw, d);
        let n_k = kernel.mul(k_v, scale_k);
        let n_k = kernel.mul(n_k, kw_v);
        kernel.store(k_out, n_k, kd);
    }
    let _ = head_dim_c;
    kernel.default_epilogue();
    kernel
}

/// Partial RoPE: x [H, S, D] (flattened H*S*D, H outermost) + cos/sin [S, ROT_DIM]
/// -> y [H, S, D]. Fuses the narrow/cat/rotate_half. `rot_dim` is the
/// number of leading dims that are rotated (partial factor).
pub fn rope_kernel(seq: i64, heads: i64, head_dim: i64, rot_dim: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [x, cos, sin] = kernel.params([DType::F32; 3]);
    let out = kernel.param_mut(DType::F32);
    //let m = heads * seq;
    //let hd_elems = heads * head_dim;
    let cb = head_dim / 32;
    let half = rot_dim / 2;
    let m = heads * seq;
    // grid: (cb, m) blocks; each block handles one 32-col slice of one row.
    let [gidx, gidy] = kernel.group_ranges([cb, m]);
    let [lane] = kernel.local_ranges([32]);
    kernel.warp(lane);
    // gidx = col_block, gidy = row
    let head_dim_c = kernel.const_idx(head_dim);
    let rot_dim_c = kernel.const_idx(rot_dim);
    let half_c = kernel.const_idx(half);
    let col_base = kernel.mul(gidx, 32);
    let col = kernel.add(col_base, lane);
    // s = row % seq (rows are flat H*S, seq-major).
    let s = kernel.mod_(gidy, seq);
    // x[row, col], out[row, col]
    let x_idx = kernel.mad(gidy, head_dim_c, col);
    let x_val = kernel.load(x, x_idx);
    let is_rot = kernel.cmplt(col, rot_dim_c);
    let safe_col = kernel.branchless_where(is_rot, col, 0i64);
    let cos_idx = kernel.mad(s, rot_dim_c, safe_col);
    let cos_val = kernel.load(cos, cos_idx);
    let sin_val = kernel.load(sin, cos_idx);
    // rotate_half on rot_dim: half = rot_dim/2
    let is_first = kernel.cmplt(col, half_c);
    let col_plus = kernel.add(col, half_c);
    let col_minus = kernel.sub(col, half_c);
    let rot_col = kernel.branchless_where(is_first, col_plus, col_minus);
    let safe_rot = kernel.branchless_where(is_rot, rot_col, 0i64);
    let x_rot_idx = kernel.mad(gidy, head_dim_c, safe_rot);
    let x_rot_raw = kernel.load(x, x_rot_idx);
    let neg = kernel.neg(x_rot_raw);
    let x_rot = kernel.branchless_where(is_first, neg, x_rot_raw);
    let y1 = kernel.mul(x_val, cos_val);
    let y2 = kernel.mul(x_rot, sin_val);
    let rot_y = kernel.add(y1, y2);
    let y = kernel.branchless_where(is_rot, rot_y, x_val);
    let out_idx = kernel.mad(gidy, head_dim_c, col);
    kernel.store(out, y, out_idx);
    let _ = m; // silence unused
    kernel.default_epilogue();
    kernel
}

/// Dense GQA attention core (fused repeat/gate): q [H,S,D], k/v [KV,S,D],
/// gate [S, H*D]. Causal j>i -> -inf inside. For demo S<=64 direct
/// warp-softmax. Fuses repeat_kv (h/kv) and gate sigmoid. D may be <32
/// (e.g. 8 in the attention golden), so dot is masked and chunked as
/// (D+31)/32.
pub fn attention_kernel(seq: i64, h: i64, kv: i64, d: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [q, k, v, gate] = kernel.params([DType::F32; 4]);
    let out = kernel.param_mut(DType::F32);
    let [head, s] = kernel.group_ranges([h, seq]);
    let [lane] = kernel.local_ranges([32]);
    let h_div_kv = h / kv;
    let kv_h = kernel.div(head, h_div_kv);
    let scale = kernel.const_val(1.0f32 / (d as f32).sqrt());
    let zero = kernel.const_val(0.0f32);
    let neg_inf = kernel.const_val(f32::NEG_INFINITY);
    let d_chunks = (d + 31) / 32;
    // init to large negative finite to avoid Const(-inf)-Const(-inf) folding to NaN
    let mut max_score = kernel.const_val(-1e30f32);
    for s2 in 0..seq {
        let s2_c = kernel.const_idx(s2 as u32);
        let s1 = kernel.add(s, 1i64);
        let is_causal = kernel.cmplt(s2_c, s1);
        let mut dot = kernel.const_val(0.0f32);
        for r in 0..d_chunks {
            let r32 = kernel.const_idx((r * 32) as u32);
            let dd = kernel.add(lane, r32);
            let valid = kernel.cmplt(dd, d);
            let hs = kernel.mul(head, seq);
            let hs_d = kernel.mul(hs, d);
            let s_d = kernel.mul(s, d);
            let q_idx0 = kernel.add(hs_d, s_d);
            let q_idx = kernel.add(q_idx0, dd);
            let kv_s = kernel.mul(kv_h, seq);
            let k_base = kernel.add(kv_s, s2_c);
            let k_idx0 = kernel.mul(k_base, d);
            let k_idx = kernel.add(k_idx0, dd);
            let q_raw = kernel.load(q, q_idx);
            let k_raw = kernel.load(k, k_idx);
            let qv = kernel.branchless_where(valid, q_raw, zero);
            let kvv = kernel.branchless_where(valid, k_raw, zero);
            dot = kernel.mad(qv, kvv, dot);
        }
        let dot_sum = kernel.warp_reduce(dot);
        let scaled = kernel.mul(dot_sum, scale);
        let masked = kernel.ternary_where(is_causal, scaled, neg_inf);
        let is_gt = kernel.cmpgt(masked, max_score);
        max_score = kernel.ternary_where(is_gt, masked, max_score);
    }
    let mut sum_exp = kernel.const_val(0.0f32);
    for s2 in 0..seq {
        let s2_c = kernel.const_idx(s2 as u32);
        let s1 = kernel.add(s, 1i64);
        let is_causal = kernel.cmplt(s2_c, s1);
        let mut dot = kernel.const_val(0.0f32);
        for r in 0..d_chunks {
            let r32 = kernel.const_idx((r * 32) as u32);
            let dd = kernel.add(lane, r32);
            let valid = kernel.cmplt(dd, d);
            let hs = kernel.mul(head, seq);
            let hs_d = kernel.mul(hs, d);
            let s_d = kernel.mul(s, d);
            let q_idx0 = kernel.add(hs_d, s_d);
            let q_idx = kernel.add(q_idx0, dd);
            let kv_s = kernel.mul(kv_h, seq);
            let k_base = kernel.add(kv_s, s2_c);
            let k_idx0 = kernel.mul(k_base, d);
            let k_idx = kernel.add(k_idx0, dd);
            let q_raw = kernel.load(q, q_idx);
            let k_raw = kernel.load(k, k_idx);
            let qv = kernel.branchless_where(valid, q_raw, zero);
            let kvv = kernel.branchless_where(valid, k_raw, zero);
            dot = kernel.mad(qv, kvv, dot);
        }
        let dot_sum = kernel.warp_reduce(dot);
        let scaled = kernel.mul(dot_sum, scale);
        let masked = kernel.ternary_where(is_causal, scaled, neg_inf);
        let sub = kernel.sub(masked, max_score);
        let exp_val = kernel.exp(sub);
        let exp_masked = kernel.ternary_where(is_causal, exp_val, zero);
        sum_exp = kernel.add(sum_exp, exp_masked);
    }
    let inv_sum = kernel.reciprocal(sum_exp);
    for r in 0..d_chunks {
        let r32 = kernel.const_idx((r * 32) as u32);
        let dd = kernel.add(lane, r32);
        let valid_out = kernel.cmplt(dd, d);
        let mut acc = kernel.const_val(0.0f32);
        for s2 in 0..seq {
            let s2_c = kernel.const_idx(s2 as u32);
            let s1 = kernel.add(s, 1i64);
            let is_causal = kernel.cmplt(s2_c, s1);
            let mut dot = kernel.const_val(0.0f32);
            for rr in 0..d_chunks {
                let rr32 = kernel.const_idx((rr * 32) as u32);
                let ddd = kernel.add(lane, rr32);
                let valid2 = kernel.cmplt(ddd, d);
                let hs = kernel.mul(head, seq);
                let hs_d = kernel.mul(hs, d);
                let s_d = kernel.mul(s, d);
                let q_idx0 = kernel.add(hs_d, s_d);
                let q_idx = kernel.add(q_idx0, ddd);
                let kv_s = kernel.mul(kv_h, seq);
                let k_base = kernel.add(kv_s, s2_c);
                let k_idx0 = kernel.mul(k_base, d);
                let k_idx = kernel.add(k_idx0, ddd);
                let q_raw = kernel.load(q, q_idx);
                let k_raw = kernel.load(k, k_idx);
                let qv = kernel.branchless_where(valid2, q_raw, zero);
                let kvv = kernel.branchless_where(valid2, k_raw, zero);
                dot = kernel.mad(qv, kvv, dot);
            }
            let dot_sum = kernel.warp_reduce(dot);
            let scaled = kernel.mul(dot_sum, scale);
            let masked = kernel.ternary_where(is_causal, scaled, neg_inf);
            let sub = kernel.sub(masked, max_score);
            let exp_val = kernel.exp(sub);
            let prob = kernel.mul(exp_val, inv_sum);
            let prob_masked = kernel.ternary_where(is_causal, prob, zero);
            let kv_s = kernel.mul(kv_h, seq);
            let v_base = kernel.add(kv_s, s2_c);
            let v_idx0 = kernel.mul(v_base, d);
            let v_idx = kernel.add(v_idx0, dd);
            let vv_raw = kernel.load(v, v_idx);
            let vv = kernel.branchless_where(valid_out, vv_raw, zero);
            acc = kernel.mad(prob_masked, vv, acc);
        }
        let hd = h * d;
        let gate_base = kernel.mul(s, hd);
        let head_d = kernel.mul(head, d);
        let gate_off = kernel.add(head_d, dd);
        let gate_idx = kernel.add(gate_base, gate_off);
        let g_raw = kernel.load(gate, gate_idx);
        let g = kernel.sigmoid(g_raw);
        let gated = kernel.mul(acc, g);
        let out_idx = kernel.add(gate_base, gate_off);
        kernel.if_(valid_out);
        kernel.store(out, gated, out_idx);
        kernel.end_if();
    }
    kernel.default_epilogue();
    kernel
}

/// SwiGLU fused elementwise: gate [M, INTER] f32, up [M, INTER] f32 -> mid [M, INTER] f32
/// mid = silu(gate) * up, silu(x)= x*sigmoid(x) via exp.
pub fn mlp_kernel(m: i64, inter: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [gate, up] = kernel.params([DType::F32; 2]);
    let out = kernel.param_mut(DType::F32);
    let [cg, t] = kernel.group_ranges([inter / 32, m]);
    let [lane] = kernel.local_ranges([32]);
    let c = kernel.mad(cg, 32i64, lane);
    let idx = kernel.mad(t, inter, c);
    let g = kernel.load(gate, idx);
    let u = kernel.load(up, idx);
    let sg = kernel.silu(g);
    let y = kernel.mul(sg, u);
    kernel.store(out, y, idx);
    kernel.default_epilogue();
    kernel
}

/// Embedding gather: weight [VOCAB, DIM] f32, ids [S] i64 -> out [S, DIM] f32
/// out[s,d] = weight[ ids[s], d ]
pub fn embed_kernel(_vocab: i64, dim: i64, seq: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let w = kernel.param(DType::F32);
    let ids = kernel.param(DType::I64);
    let out = kernel.param_mut(DType::F32);
    let [cg, s] = kernel.group_ranges([dim / 32, seq]);
    let [lane] = kernel.local_ranges([32]);
    let d = kernel.mad(cg, 32i64, lane);
    let id = kernel.load(ids, s);
    let row = kernel.mul(id, dim);
    let w_idx = kernel.add(row, d);
    let v = kernel.load(w, w_idx);
    let out_idx = kernel.mad(s, dim, d);
    kernel.store(out, v, out_idx);
    kernel.default_epilogue();
    kernel
}

/// Repack gguf Q4_K raw super-blocks into the device-ready dequant layout.
///
/// Offline prep: runs on any backend (CUDA for speed, C as fallback — device
/// follows the input tensor, single IR). Layout per llama.cpp `ggml-quants.c`
/// (`dequantize_row_q4_K`, `get_scale_min_k4`): each 144B super-block holds
/// d (F16), dmin (F16), scales[12] (6-bit scales+mins), qs[128] (4-bit quants,
/// byte l of group g: low nibble = weight 64g+l, high = weight 64g+32+l).
///
/// Input `raw` is [N, 144] U8 with N = rows*cols/256. Returns
/// (packed, scales, mins):
/// - packed: [rows*cols/4] U16, tile-ordered weights plane-interleaved:
///   u16[i] nibble p = tile-ordered weight p*(rows*cols/4)+i. The device
///   kernel extracts plane p straight into output tile p, no shuffle.
/// - scales/mins: [num_tiles, 32] BF16, tile-major; slot i holds row-in-tile
///   i's true sub-block scale (d*sc) / min (dmin*m). BF16 (not F16): TT-native
///   with F32 range, so a large d can never saturate the sidecar; the 8-bit
///   mantissa is far below the 4-bit quant noise. The device `bcast_cols`
///   op replicates each across its row.
/// # Errors
/// Returns [`ZyxError`] on shape mismatch or backend failure.
pub fn repack_q4k(
    raw: &Tensor,
    rows: i64,
    cols: i64,
) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
    debug_assert!(
        rows % 32 == 0,
        "repack_q4k needs rows % 32 == 0, got {rows}"
    );
    debug_assert!(
        cols % 32 == 0,
        "repack_q4k needs cols % 32 == 0, got {cols}"
    );
    debug_assert!(
        (rows * cols) % 256 == 0,
        "repack_q4k needs rows*cols % 256 == 0"
    );
    let n = rows * cols / 256;

    // d/dmin are F16 bit patterns: assemble U16 values in ops, reinterpret
    // via bitcast (any 16-bit pattern is a valid F16), value-cast to BF16.
    let raw3 = raw.split([4i64, 12i64, 128i64], 1)?;
    let (dd, s, qs) = (&raw3[0], &raw3[1], &raw3[2]);
    let lanes = dd.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let d_bits = (lanes[0].cast(DType::U16) + lanes[1].cast(DType::U16) * 256u16).reshape([n])?;
    let dmin_bits =
        (lanes[2].cast(DType::U16) + lanes[3].cast(DType::U16) * 256u16).reshape([n])?;
    // Bitcast U16 -> F16 is equal-width; every 16-bit pattern is a valid f16.
    let d = d_bits.bitcast(DType::F16)?.cast(DType::BF16);
    let dmin = dmin_bits.bitcast(DType::F16)?.cast(DType::BF16);

    // Nibbles in llama order: byte l of group g -> weights 64g+l (low),
    // 64g+32+l (high).
    let hi = qs >> 4u8;
    let lo = qs & 15u8;
    let lo4 = lo.reshape([n, 4, 32])?.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let hi4 = hi.reshape([n, 4, 32])?.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let mut segs = Vec::with_capacity(4);
    for g in 0..4 {
        let lg = lo4[g].reshape([n, 32])?;
        let hg = hi4[g].reshape([n, 32])?;
        segs.push(Tensor::cat([&lg, &hg], 1)?);
    }
    // [N, 256], group g at cols 64g..64g+64, reshapes straight to [rows, cols].
    let mat = Tensor::cat(&segs, 1)?.reshape([rows, cols])?;
    // Tile order first (device consumes tilized), then plane-split over the
    // tilized linear order so each plane IS an output tile's weights.
    let l = rows * cols;
    let flat = mat.cast(DType::U16).tilize()?.reshape([l])?;
    let l4 = l / 4;
    let planes = flat.split([l4, l4, l4, l4], 0)?;
    let packed = &planes[0] + (&planes[1] << 4u16) + (&planes[2] << 8u16) + (&planes[3] << 12u16);

    // 6-bit scales/mins per get_scale_min_k4, then true scale = d*sc.
    let slanes = s.split(
        [
            1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64,
        ],
        1,
    )?;
    let mut sc_parts = Vec::with_capacity(8);
    let mut m_parts = Vec::with_capacity(8);
    for j in 0..8usize {
        if j < 4 {
            sc_parts.push((&slanes[j] & 63u8).reshape([n, 1])?);
            m_parts.push((&slanes[j + 4] & 63u8).reshape([n, 1])?);
        } else {
            // get_scale_min_k4: sc high bits live in S[j-4], m high bits in S[j].
            let sc = (&slanes[j + 4] & 15u8).reshape([n, 1])?
                + (&slanes[j - 4] >> 6u8).reshape([n, 1])? * 16u8;
            let m = (&slanes[j + 4] >> 4u8).reshape([n, 1])?
                + (&slanes[j] >> 6u8).reshape([n, 1])? * 16u8;
            sc_parts.push(sc);
            m_parts.push(m);
        }
    }
    let sc = Tensor::cat(&sc_parts, 1)?.cast(DType::BF16);
    let mm = Tensor::cat(&m_parts, 1)?.cast(DType::BF16);
    // [N, 8] true scales/mins -> [rows, cols/32] sub-block grid -> tile-major.
    let c32 = cols / 32;
    let scales_grid = (d.reshape([n, 1])? * sc).reshape([rows, c32])?;
    let mins_grid = (dmin.reshape([n, 1])? * mm).reshape([rows, c32])?;
    let ntiles = rows / 32 * c32;
    let scales = scales_grid
        .reshape([rows / 32, 32, c32])?
        .permute([0, 2, 1])?
        .reshape([ntiles, 32])?;
    let mins = mins_grid
        .reshape([rows / 32, 32, c32])?
        .permute([0, 2, 1])?
        .reshape([ntiles, 32])?;
    Ok((packed, scales, mins))
}

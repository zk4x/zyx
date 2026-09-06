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
use zyx::{f16, DType};

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

/// Zero-pads rows: in [s, d] f32 -> out [m, d] f16.
pub fn pad_kernel(s: i64, m: i64, d: i64) -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let inp = kernel.param(DType::F32);
    let out = kernel.param_mut(DType::F16);
    let [cc, row] = kernel.group_ranges([d / 32, m]);
    let [lane] = kernel.local_ranges([32]);
    let c = kernel.mad(cc, 32i64, lane);
    let in_idx = kernel.mad(row, d, c);
    let x_raw = kernel.load(inp, in_idx);
    let x = kernel.cast(x_raw, DType::F16);
    let keep = kernel.cmplt(row, s);
    let v = kernel.branchless_where(keep, x, f16::from_f32(0.0));
    let out_idx = kernel.mad(row, d, c);
    kernel.store(out, v, out_idx);
    kernel
}

/// GEMM with 16-row blocks, n=8 (lm_head pattern): out [R, N] =
/// A [R, K] @ B [N, K]^T, R/K/N/glens as runtime variables.
/// One compiled instance serves every projection in the layer.
pub fn gemm_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [rows, kk, nn, glen_x, glen_y] = kernel.variables([DType::I64; 5]);
    let [a, b] = kernel.params([DType::F16; 2]);
    let out = kernel.param_mut(DType::F32);

    let [gidx, gidy] = kernel.group_ranges([glen_x, glen_y]);
    let lidx = kernel.local_range(0, 32);
    kernel.warp(lidx);

    let ap = kernel.view_global_register(a, [rows, kk]);
    let bp = kernel.view_global_register(b, [nn, kk]);
    let cp = kernel.view_global_register(out, [rows, nn]);

    let c_block = kernel.const_idx(16u32);
    let [c8] = kernel.const_idxs([8u32]);
    let r0 = kernel.mul(gidx, c_block);
    let n0 = kernel.mul(gidy, c8);
    let acc = kernel.acc([c_block, c8], DType::F32);
    kernel.loop_partition(|kernel, k| {
        kernel.mma_at(&acc, &ap, &bp, [r0, n0, k]);
    });
    kernel.store_partition(&cp, &acc, [r0, n0]);
    kernel
}

/// Depthwise causal conv1d (kernel 4, left pad 3) + SiLU:
/// in [M, C] f32 -> out [M, C] f32, convw [C, 4] f32.
pub fn conv_silu_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let [inp, convw] = kernel.params([DType::F32; 2]);
    let out = kernel.param_mut(DType::F32);
    let [cg, t] = kernel.group_ranges([CONV_DIM / 32, M_PAD]);
    let [lane] = kernel.local_ranges([32]);
    let c = kernel.mad(cg, 32i64, lane);

    let mut acc = kernel.const_val(0.0f32);
    for k in 0..CK {
        let src_t = kernel.add(t, k - (CK - 1));
        let ok = kernel.cmpge(src_t, 0i64);
        let in_idx = kernel.mad(src_t, CONV_DIM, c);
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
    let qscale = kernel.const_val(0.125f32);

    kernel.loop_over(S, |kernel, t| {
        // Row bases: q/k rows are KD-contiguous in mixed, v rows VD.
        let t_q = kernel.mul(t, CONV_DIM);
        let t_h = kernel.mul(t, M_PAD);
        let kh_kd = kernel.mul(kh, KD);
        let kh_kk = kernel.mul(kh, KEY_DIM);
        let h_vd = kernel.mul(h, VD);
        let q_base = kernel.add(t_q, kh_kd);
        let k_off = kernel.add(KEY_DIM, kh_kk);
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
    kernel
}

/// Gated RMSNorm per (token, v-head) 128-row: core [VH, S, VD] f32,
/// z [M_PAD, VAL_DIM] f32, norm_w [VD] f32 -> normed [S, VAL_DIM] f32.
/// out row = core*rsqrt(mean+eps)*norm_w*silu(z).
pub fn rmsnorm_kernel() -> Kernel {
    let mut kernel = Kernel::new(Dev::Cuda(0));
    let core = kernel.param(DType::F32);
    let zp = kernel.param(DType::F32);
    let nw = kernel.param(DType::F32);
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
    kernel
}

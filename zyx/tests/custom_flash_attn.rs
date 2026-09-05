// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Flash attention written with the zyx custom kernel API.
//!
//! Tiled causal flash attention, the variant used in modern models:
//! one workgroup computes a `BLOCK_M × D` output tile. K/V tiles are loaded
//! cooperatively into shared memory (`MemScope::Local`), a `barrier()` sits
//! between the load and compute phases, and each thread (`local_range`) owns
//! one query row. Scores are computed tile-by-tile with online softmax —
//! running max `m`, running sum `l`, and a `D`-wide accumulator rescaled by
//! `exp(m_old - m_new)` per tile — so the `S×S` score matrix is never
//! materialized.

use zyx::DType;
use zyx::ZyxError;
use zyx::kernel::{Dev, Kernel, MemScope, OpId};

const S: i64 = 128; // sequence length (one batch, one head)
const D: i64 = 64; // head dim
const BLOCK_M: i64 = 32; // query rows per workgroup
const BLOCK_N: i64 = 32; // keys per shared-memory tile

#[test]
fn flash_attention_compiles() -> Result<(), ZyxError> {
    let mut k = Kernel::new(Dev::Auto);

    // Dims / constants
    let d_c = k.const_idx(D);
    let bm_c = k.const_idx(BLOCK_M);
    let bn_c = k.const_idx(BLOCK_N);
    let nb_c = k.const_idx(S / BLOCK_N); // number of key tiles
    let idx0 = k.const_idx(0);
    let neg_inf = k.const_val(f32::NEG_INFINITY);
    let scale = k.const_val(1.0 / (D as f32).sqrt());

    // Params: Q, K, V inputs + Out output, all [S, D] row-major
    let q = k.param(DType::F32);
    let k_param = k.param(DType::F32);
    let v = k.param(DType::F32);
    let out = k.param_mut(DType::F32);

    // Shared memory tiles (SRAM): one K and one V tile per workgroup
    let k_tile = k.storage(DType::F32, MemScope::Local, BLOCK_N * D);
    let v_tile = k.storage(DType::F32, MemScope::Local, BLOCK_N * D);

    // Per-thread register state: q row, output accumulator (zeroed), running max m, running sum l (zeroed)
    let q_reg = k.storage(DType::F32, MemScope::Register, D);
    let acc_reg = k.zeros(DType::F32, D);
    let m_reg = k.storage(DType::F32, MemScope::Register, 1);
    let l_reg = k.zeros(DType::F32, 1);

    // Grid: one workgroup per query tile; thread = row within the tile
    let qb_len = k.const_idx(S / BLOCK_M);
    let q_block = k.group_range(0, qb_len);
    let local = k.local_range(0, BLOCK_M as u32);
    // Query row this thread owns: i = q_block * BLOCK_M + local
    let i = k.mad(q_block, bm_c, local);

    // Load the query row into registers
    k.loop_over(d_c, |k, d| {
        let q_idx = k.mad(i, d_c, d);
        let qv = k.load(q, q_idx);
        k.store(q_reg, qv, d);
    });

    // Online softmax state: m = -inf
    k.store(m_reg, neg_inf, idx0);

    // Stream over key tiles
    k.loop_over(nb_c, |k, nb| {
        // Cooperative load: thread `local` fetches K/V rows of the tile into SRAM
        k.copy_tile_local(k_param, k_tile, bn_c, nb, d_c, local);
        k.copy_tile_local(v, v_tile, bn_c, nb, d_c, local);
        k.barrier();

        // Each thread: online softmax over the BLOCK_N keys in SRAM
        k.loop_over(bn_c, |k, n| {
            // score = q · k[n], reading the tile from shared memory
            let mut s_reg = OpId::NULL;
            k.loop_over(d_c, |k, d| {
                let tile_idx = k.mad(n, d_c, d);
                s_reg = k.dot(DType::F32, d, k_tile, tile_idx, q_reg, d);
            });
            let mut s = k.load(s_reg, idx0);
            s = k.mul(s, scale);

            // Causal mask: key j must not attend to future positions (j > i)
            let j = k.add(nb, n); // global key index (nb counts tiles)
            let masked = k.cmplt(i, j);
            let s = k.ternary_where(masked, neg_inf, s);

            // Online softmax update
            let m = k.load(m_reg, idx0);
            let m_new = k.max(m, s);
            let s_m = k.sub(s, m_new);
            let p = k.exp(s_m); // exp(s - m_new), exp(-inf - m_new) = 0
            let m_m = k.sub(m, m_new);
            let alpha = k.exp(m_m); // rescale factor for acc and l
            let l = k.load(l_reg, idx0);
            let l_new = k.mad(alpha, l, p);
            k.store(l_reg, l_new, idx0);
            k.store(m_reg, m_new, idx0);

            // acc = acc * alpha + p * v[n], reading the tile from shared memory
            k.mad_tile_local(acc_reg, alpha, p, v_tile, n, d_c);
        });
        // Barrier before the next tile overwrites the shared-memory buffers
        k.barrier();
    });

    // Normalize by the softmax denominator and store the output row
    k.loop_over(d_c, |k, d| {
        let acc = k.load(acc_reg, d);
        let l = k.load(l_reg, idx0);
        let o = k.div(acc, l);
        let o_idx = k.mad(i, d_c, d);
        k.store(out, o, o_idx);
    });

    k.compile()?;

    Ok(())
}

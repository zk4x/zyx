// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! DRAFT: `Op::Asm`-based SFPU microcode dequant. Same data/geometry as
//! `dequant_q4k_tt` (plane-interleaved U16 pages + full-tile BF16
//! scales/mins), but the nibble extraction runs as hand-written SFPU
//! microcode emitted through `Kernel::asm` instead of the F32
//! divide/trunc carry arithmetic: per plane k, `nibble = (w >> 4k) & 0xF`
//! with native int32 SFPSHFT/SFPAND, SFPCAST to F32, SFPMAD with the
//! per-row scale/min. No carry CBs, no scratch CBs.
//!
//! The purpose of this test is to measure how much the TT codegen must
//! change to admit hand-written microcode. Microcode instruction sequence
//! mirrors the vendor quant LLKs (tt-metal ckernel_sfpu_quant.h /
//! ckernel_sfpu_shift.h): 4 faces x 8 iterations, SFPSTORE under
//! ADDR_MOD_2 walks dst_reg through a face, inc_dst_face_addr between
//! faces. UNVERIFIED ON DEVICE.

use zyx::kernel::{Dev, Kernel};
use zyx::DType;

const TDIM: u16 = 32;
const TILE_ELEMS: i64 = 1024;

fn dequant_q4k_tt_asm(ntiles: i64) -> Kernel {
    debug_assert!(ntiles % 4 == 0, "needs ntiles % 4 == 0, got {ntiles}");
    let pages = ntiles / 4;
    let mut kernel = Kernel::new(Dev::TT(0));
    let packed = kernel.param(DType::U16);
    let sc = kernel.param(DType::BF16);
    let mn = kernel.param(DType::BF16);
    let out = kernel.param_mut(DType::F32);

    // Packed page is popped once per plane (4 pops/page), so the CB
    // holds ntiles tiles, not 1 like the carry-based kernel.
    let cu16 = kernel.circular_storage(DType::U16, ntiles);
    let csc = kernel.circular_storage(DType::BF16, 1);
    let cmn = kernel.circular_storage(DType::BF16, 1);
    let cout = kernel.circular_storage(DType::F32, 1);

    let _g = kernel.group_range(0, 1);
    let cpages = kernel.const_idx(pages);
    let ctiles = kernel.const_idx(ntiles);
    let c4 = kernel.const_idx(4);
    let c1024 = kernel.const_idx(TILE_ELEMS);
    let c0 = kernel.const_idx(0);

    // Reader: page pushed once per plane (4x), scales/mins per tile.
    kernel.loop_over(cpages, |kernel, pi| {
        let ubase = kernel.mad(pi, c1024, c0);
        let u = kernel.load_tile(packed, ubase, TDIM, TDIM, TDIM as u32);
        for _ in 0..4 {
            kernel.store_tile(cu16, u, c0, TDIM, TDIM, TDIM as u32);
        }
        kernel.loop_over(c4, |kernel, ki| {
            let t = kernel.mad(ki, c1024, c0);
            let s = kernel.load_tile(sc, t, TDIM, TDIM, TDIM as u32);
            kernel.store_tile(csc, s, c0, TDIM, TDIM, TDIM as u32);
            let m = kernel.load_tile(mn, t, TDIM, TDIM, TDIM as u32);
            kernel.store_tile(cmn, m, c0, TDIM, TDIM, TDIM as u32);
        });
    });
    kernel.barrier();

    // Compute: per plane, one asm block does the whole dequant in place
    // on the packed tile's DST slot: LO16 load, shift, mask, cast,
    // MAD with scale/min (loaded from their DST slots), F32 store back.
    kernel.loop_over(cpages, |kernel, _pi| {
        for k in 0..4i64 {
            let w = kernel.load_tile(cu16, c0, TDIM, TDIM, TDIM as u32);
            let s = kernel.load_tile(csc, c0, TDIM, TDIM, TDIM as u32);
            let sf = kernel.cast(s, DType::F32);
            let m = kernel.load_tile(cmn, c0, TDIM, TDIM, TDIM as u32);
            let mf = kernel.cast(m, DType::F32);
            let tpl = format!(
                r#"math::clear_dst_reg_addr();
for (int face = 0; face < 4; face++) {{
    for (int d = 0; d < 8; d++) {{
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, {{0}} * 64);
        _sfpu_load_imm32_(p_sfpu::LREG2, {shift});
        TT_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        _sfpu_load_imm32_(p_sfpu::LREG3, 15);
        TT_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
        TT_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_INT32_TO_FP32_RNE);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_3, {{1}} * 64);
        TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::FP32, ADDR_MOD_3, {{2}} * 64);
        TT_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG4, 6);
        TT_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TT_SFPNOP;
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_2, {{0}} * 64);
    }}
    _llk_math_eltwise_sfpu_inc_dst_face_addr_();
}}"#,
                shift = 4 * k
            );
            let v = kernel.asm(&tpl, &[w, sf, mf]);
            kernel.store_tile(cout, v, c0, TDIM, TDIM, TDIM as u32);
        }
    });
    kernel.barrier();

    // Writer: stream dequantized tiles to DRAM.
    kernel.loop_over(ctiles, |kernel, ti| {
        let obase = kernel.mad(ti, c1024, c0);
        let v = kernel.load_tile(cout, c0, TDIM, TDIM, TDIM as u32);
        kernel.store_tile(out, v, obase, TDIM, TDIM, TDIM as u32);
    });

    kernel.verify();
    kernel
}

#[test]
fn dequant_q4k_tt_asm_compile() {
    let kk = dequant_q4k_tt_asm(4);
    let _k = kk.compile().expect("compile asm dequant kernel");
}

// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Q4_K repack building blocks (llama.cpp `block_q4_K` layout) plus the whole
//! pipeline on 4 super-blocks. Each part tested alone with exact asserts,
//! then the whole thing byte-exact. Runs on C; that is expected.

use zyx::{DType, Tensor, ZyxError};

// get_scale_min_k4 host reference (ggml-quants.c).
fn scale_min(j: usize, s: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        ((s[j + 4] & 15) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
    }
}

#[test]
fn q4k_nibble_split() -> Result<(), ZyxError> {
    // Byte l of group g: low nibble = weight 64g+l, high = weight 64g+32+l.
    let qs: Vec<u8> = (0..128).map(|i| (i * 13 + 7) as u8).collect();
    let q = Tensor::from_vec(qs.clone(), [4, 32])?;
    let hi = &q >> 4u8;
    let lo = &q & 15u8;
    let hv: Vec<u8> = hi.to_vec()?;
    let lv: Vec<u8> = lo.to_vec()?;
    for (i, &b) in qs.iter().enumerate() {
        assert_eq!(lv[i], b & 15, "byte {i} lo");
        assert_eq!(hv[i], b >> 4, "byte {i} hi");
    }
    Ok(())
}

#[test]
fn q4k_scale_unpack() -> Result<(), ZyxError> {
    let s: [u8; 12] = [31, 38, 45, 52, 59, 66, 73, 80, 87, 94, 101, 108];
    let t = Tensor::from_vec(s.to_vec(), [12i64])?;
    let lanes = t.split([1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64], 0)?;
    for j in 0..8usize {
        let (sc, m) = if j < 4 {
            ((&lanes[j] & 63u8).to_vec::<u8>()?[0], (&lanes[j + 4] & 63u8).to_vec::<u8>()?[0])
        } else {
            let sc = (&lanes[j + 4] & 15u8).to_vec::<u8>()?[0] | ((&lanes[j - 4] >> 6u8).to_vec::<u8>()?[0] << 4);
            let m = (&lanes[j + 4] >> 4u8).to_vec::<u8>()?[0] | ((&lanes[j] >> 6u8).to_vec::<u8>()?[0] << 4);
            (sc, m)
        };
        let (esc, em) = scale_min(j, &s);
        assert_eq!((sc, m), (esc, em), "sub-block {j}");
    }
    Ok(())
}

#[test]
fn q4k_plane_pack() -> Result<(), ZyxError> {
    // 4 planes of 256 nibbles -> 256 u16 and back, exact.
    let p: Vec<u16> = (0..1024).map(|i| (i * 5 + 3) as u16 % 16).collect();
    let t = Tensor::from_vec(p.clone(), [4, 256])?.cast(DType::U16);
    let planes = t.split([1i64, 1i64, 1i64, 1i64], 0)?;
    let p0 = planes[0].reshape([256])?;
    let p1 = planes[1].reshape([256])?;
    let p2 = planes[2].reshape([256])?;
    let p3 = planes[3].reshape([256])?;
    let packed = &p0 + (&p1 << 4u16) + (&p2 << 8u16) + (&p3 << 12u16);
    let pv: Vec<u16> = packed.to_vec()?;
    assert_eq!(pv.len(), 256);
    for i in 0..256 {
        assert_eq!(pv[i] & 15, p[i], "slot {i} plane 0");
        assert_eq!((pv[i] >> 4) & 15, p[256 + i], "slot {i} plane 1");
        assert_eq!((pv[i] >> 8) & 15, p[512 + i], "slot {i} plane 2");
        assert_eq!((pv[i] >> 12) & 15, p[768 + i], "slot {i} plane 3");
    }
    Ok(())
}

#[test]
fn q4k_dmin_bitcast() -> Result<(), ZyxError> {
    // Byte pairs -> U16 values -> F16 bits -> BF16 values, exact.
    let bytes: Vec<u8> = vec![0x00, 0x3c, 0x00, 0x38, 0x34, 0x56, 0x00, 0x00];
    let t = Tensor::from_vec(bytes, [4, 2])?;
    let lanes = t.split([1i64, 1i64], 1)?;
    let bits = (lanes[0].cast(DType::U16) + lanes[1].cast(DType::U16) * 256u16).reshape([4])?;
    let f = bits.bitcast(DType::F16)?.cast(DType::BF16);
    let fv: Vec<zyx::bf16> = f.to_vec()?;
    let expected = [1.0f32, 0.5, zyx::f16::from_bits(0x5634).to_f32(), 0.0];
    for (i, (&g, &e)) in fv.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_f32(), zyx::bf16::from_f32(e).to_f32(), "lane {i}");
    }
    Ok(())
}

#[test]
fn q4k_u8_to_bf16_cast() -> Result<(), ZyxError> {
    let v: Vec<u8> = (0..32).map(|i| (i * 7 + 3) as u8).collect();
    let t = Tensor::from_vec(v.clone(), [4, 8])?;
    let f: Vec<zyx::bf16> = t.cast(DType::BF16).to_vec()?;
    for (i, (&g, &e)) in f.iter().zip(v.iter()).enumerate() {
        assert_eq!(g.to_f32(), e as f32, "lane {i}");
    }
    Ok(())
}

#[test]
fn q4k_bf16_bcast_mul() -> Result<(), ZyxError> {
    let a: Vec<f32> = vec![1.0, 0.5, 2.0, 3.0];
    let b: Vec<f32> = (0..32).map(|i| i as f32 + 1.0).collect();
    let ta = Tensor::from_vec(a.clone(), [4, 1])?.cast(DType::BF16);
    let tb = Tensor::from_vec(b.clone(), [4, 8])?.cast(DType::BF16);
    let r: Vec<zyx::bf16> = (&ta * &tb).to_vec()?;
    assert_eq!(r.len(), 32);
    for i in 0..4 {
        for j in 0..8 {
            let exp = zyx::bf16::from_f32(a[i] * b[i * 8 + j]).to_f32();
            assert_eq!(r[i * 8 + j].to_f32(), exp, "cell {i},{j}");
        }
    }
    Ok(())
}

#[test]
fn q4k_repack_whole() -> Result<(), ZyxError> {
    // 4 super-blocks = 32x32 weights. Full pipeline inline, byte-exact.
    let (rows, cols) = (32i64, 32i64);
    let n = 4usize;
    let mut raw = vec![0u8; n * 144];
    for b in 0..n {
        for i in 0..144 {
            raw[b * 144 + i] = ((b * 144 + i) * 7 + 3) as u8;
        }
        raw[b * 144] = 0x00;
        raw[b * 144 + 1] = 0x3c;
        raw[b * 144 + 2] = 0x00;
        raw[b * 144 + 3] = 0x38;
    }
    let r = Tensor::from_vec(raw.clone(), [n as i64, 144])?;
    let parts = r.split([4i64, 12i64, 128i64], 1)?;
    let (dd, s, qs) = (&parts[0], &parts[1], &parts[2]);

    // Nibbles -> [N, 256] in llama order -> [rows, cols] -> tilized -> planes.
    let hi = qs >> 4u8;
    let lo = qs & 15u8;
    let lo4 = lo.reshape([n as i64, 4, 32])?.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let hi4 = hi.reshape([n as i64, 4, 32])?.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let mut segs = Vec::with_capacity(4);
    for g in 0..4 {
        let lg = lo4[g].reshape([n as i64, 32])?;
        let hg = hi4[g].reshape([n as i64, 32])?;
        segs.push(Tensor::cat([&lg, &hg], 1)?);
    }
    let mat = Tensor::cat(&segs, 1)?.reshape([rows, cols])?;
    let flat = mat.cast(DType::U16).tilize()?.reshape([rows * cols])?;
    let l4 = rows * cols / 4;
    let planes = flat.split([l4, l4, l4, l4], 0)?;
    let packed = &planes[0] + (&planes[1] << 4u16) + (&planes[2] << 8u16) + (&planes[3] << 12u16);
    let pv: Vec<u16> = packed.to_vec()?;
    assert_eq!(pv.len(), (rows * cols / 4) as usize);

    // Invert: planes -> tilized flat -> untilize -> row-major nibbles.
    let l = (rows * cols) as usize;
    let mut til = vec![0u16; l];
    for (i, &w) in pv.iter().enumerate() {
        til[i] = w & 15;
        til[l / 4 + i] = (w >> 4) & 15;
        til[l / 2 + i] = (w >> 8) & 15;
        til[3 * l / 4 + i] = (w >> 12) & 15;
    }
    let nib: Vec<u16> = Tensor::from_vec(til, [rows, cols])?.untilize(rows, cols)?.to_vec()?;
    for b in 0..n {
        for w in 0..256 {
            let g = w / 64;
            let p = w % 64;
            let (byte, high) = if p < 32 { (32 * g + p, false) } else { (32 * g + p - 32, true) };
            let got = raw[b * 144 + 16 + byte];
            let exp = if high { (got >> 4) & 15 } else { got & 15 };
            assert_eq!(nib[b * 256 + w], u16::from(exp), "block {b} weight {w}");
        }
    }

    // Scales: unpack -> BF16 -> recover -> forward-pack -> bytes.
    let slanes = s.split([1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64], 1)?;
    let lanes = dd.split([1i64, 1i64, 1i64, 1i64], 1)?;
    let d = (lanes[0].cast(DType::U16) + lanes[1].cast(DType::U16) * 256u16)
        .reshape([n as i64])?
        .bitcast(DType::F16)?
        .cast(DType::BF16);
    let dmin = (lanes[2].cast(DType::U16) + lanes[3].cast(DType::U16) * 256u16)
        .reshape([n as i64])?
        .bitcast(DType::F16)?
        .cast(DType::BF16);
    let nn = n as i64;
    let mut scp = Vec::with_capacity(8);
    let mut mp = Vec::with_capacity(8);
    for j in 0..8usize {
        if j < 4 {
            scp.push((&slanes[j] & 63u8).reshape([nn, 1])?);
            mp.push((&slanes[j + 4] & 63u8).reshape([nn, 1])?);
        } else {
            scp.push((&slanes[j + 4] & 15u8).reshape([nn, 1])? + (&slanes[j - 4] >> 6u8).reshape([nn, 1])? * 16u8);
            mp.push((&slanes[j + 4] >> 4u8).reshape([nn, 1])? + (&slanes[j] >> 6u8).reshape([nn, 1])? * 16u8);
        }
    }
    let sc_t = Tensor::cat(&scp, 1)?.cast(DType::BF16);
    let m_t = Tensor::cat(&mp, 1)?.cast(DType::BF16);
    // d = 1.0, dmin = 0.5: true scales invert exactly via round().
    let sv: Vec<zyx::bf16> = (d.reshape([nn, 1])? * sc_t).reshape([nn * 8])?.to_vec()?;
    let mv: Vec<zyx::bf16> = (dmin.reshape([nn, 1])? * m_t).reshape([nn * 8])?.to_vec()?;
    assert_eq!(sv.len(), n * 8);
    let mut rsc = [[0u8; 8]; 4];
    let mut rm = [[0u8; 8]; 4];
    for b in 0..n {
        for j in 0..8 {
            rsc[b][j] = sv[b * 8 + j].to_f32().round() as u8;
            rm[b][j] = (mv[b * 8 + j].to_f32() * 2.0).round() as u8;
        }
    }
    for b in 0..n {
        let mut sb = [0u8; 12];
        for j in 0..4 {
            sb[j] = rsc[b][j];
            sb[j + 4] = rm[b][j];
        }
        for j in 4..8 {
            sb[j + 4] = (rsc[b][j] & 15) | ((rm[b][j] & 15) << 4);
            sb[j - 4] |= (rsc[b][j] >> 4) << 6;
            sb[j] |= (rm[b][j] >> 4) << 6;
        }
        assert_eq!(&raw[b * 144 + 4..b * 144 + 16], &sb, "block {b} scales");
    }
    Ok(())
}

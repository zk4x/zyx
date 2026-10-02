// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! f8 scalar bit patterns and the C-backend F32->F8->F32 roundtrip.

use zyx::{DType, Tensor, f8e4m3, f8e5m2};

#[test]
fn f8_roundtrip() {
    // E4M3 known bit patterns (OCP: bias 7, NaN-only 0x7F/0xFF).
    assert_eq!(f8e4m3::from_bits(0x38).to_f32(), 1.0);
    assert_eq!(f8e4m3::from_bits(0x3c).to_f32(), 1.5);
    assert_eq!(f8e4m3::from_bits(0x7e).to_f32(), 448.0);
    assert_eq!(f8e4m3::from_bits(0xfe).to_f32(), -448.0);
    assert!(f8e4m3::from_bits(0x7f).is_nan());
    assert!(f8e4m3::from_bits(0xff).is_nan());
    assert_eq!(f8e4m3::from_f32(1.0).to_bits(), 0x38);
    assert_eq!(f8e4m3::from_f32(-1.5).to_bits(), 0xbc);
    // Overflow saturates, never inf.
    assert_eq!(f8e4m3::from_f32(1000.0).to_bits(), 0x7e);
    assert_eq!(f8e4m3::from_f32(f32::INFINITY).to_bits(), 0x7e);
    assert!(!f8e4m3::from_f32(1000.0).is_infinite());
    // E5M2 known bit patterns (bias 15, IEEE inf/NaN).
    assert_eq!(f8e5m2::from_bits(0x3c).to_f32(), 1.0);
    assert_eq!(f8e5m2::from_bits(0x7b).to_f32(), 57344.0);
    assert_eq!(f8e5m2::from_bits(0xfb).to_f32(), -57344.0);
    assert!(f8e5m2::from_bits(0x7c).is_infinite());
    assert!(f8e5m2::from_bits(0x7e).is_nan());
    assert_eq!(f8e5m2::from_f32(1.0).to_bits(), 0x3c);
    assert_eq!(f8e5m2::from_f32(f32::INFINITY).to_bits(), 0x7c);
    // Arithmetic round-trips through f32.
    assert_eq!((f8e4m3::from_f32(1.5) + f8e4m3::from_f32(2.25)).to_f32(), 3.75);
    assert_eq!((f8e5m2::from_f32(2.0) * f8e5m2::from_f32(3.0)).to_f32(), 6.0);
    assert_eq!(f8e4m3::from_f32(7.0).to_f32(), 7.0);
}

#[test]
fn f8_c_roundtrip() {
    // C backend F32->F8->F32 must match scalar::from_f32/to_f32 exactly.
    let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 3.0).collect();
    for dtype in [DType::F8E4M3, DType::F8E5M2] {
        let t = Tensor::from_vec(data.clone(), [16, 16]).unwrap().cast(dtype).cast(DType::F32);
        let back: Vec<f32> = t.to_vec().unwrap();
        assert_eq!(back.len(), 256);
        for (i, (&x, &v)) in data.iter().zip(back.iter()).enumerate() {
            let expected = match dtype {
                DType::F8E4M3 => f8e4m3::from_f32(x).to_f32(),
                _ => f8e5m2::from_f32(x).to_f32(),
            };
            assert_eq!(v, expected, "{dtype:?}[{i}] x={x}");
        }
    }
}

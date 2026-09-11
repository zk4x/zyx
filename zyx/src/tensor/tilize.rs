// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tilized layouts (host-side only).
//!
//! A tilized tensor stores each 32x32 tile contiguously, with the four
//! 16x16 faces inside each tile in face order (the same convention as the
//! `14_tenstorrent` golden kernel's `lin()` mapping). Outer dims are
//! zero-padded up to multiples of 32. This is a pure host-side byte
//! permutation. Tilize once at load time (weights) or once per forward
//! (inputs); on-device kernels chain tilized layouts with no conversion
//! between layers.

use crate::{Tensor, ZyxError, dtype::DType, scalar::Scalar, shape::Dim};

impl Tensor {
    /// Permutes the last two dims into Tenstorrent tilized face order
    /// (host-side). Outer dims are zero-padded up to multiples of 32,
    /// so the returned shape is `[*batch, ceil32(rows), ceil32(cols)]`.
    /// Dtype is preserved. Tilize weights once at load; tilize inputs
    /// once per forward; on-device kernels then chain with no conversion.
    /// Method form (chainable): `Tensor::from_vec(..)?.tilize()?`.
    ///
    /// # Errors
    ///
    /// Returns [`ZyxError`] if the tensor cannot be read to host or the
    /// result cannot be allocated.
    pub fn tilize(&self) -> Result<Tensor, ZyxError> {
        fn ceil32(d: i64) -> i64 {
            debug_assert!(d >= 0, "tilize needs a concrete non-negative dim, got {d}");
            (d + 31) / 32 * 32
        }
        fn permute<T: Scalar>(input: &[T], batches: usize, rows: i64, cols: i64, pr: i64, pc: i64) -> Vec<T> {
            let ntc = pc / 32;
            let mut out = vec![T::zero(); batches * (pr * pc) as usize];
            for b in 0..batches {
                for tr in 0..pr / 32 {
                    for tc in 0..ntc {
                        for f in 0..4 {
                            let fr = f / 2;
                            let fc = f % 2;
                            for l in 0..256 {
                                let lr = l / 16;
                                let lc = l % 16;
                                let r = tr * 32 + fr * 16 + lr;
                                let c = tc * 32 + fc * 16 + lc;
                                let p = ((tr * ntc + tc) * 4 + f) * 256 + l;
                                let v = if r < rows && c < cols {
                                    input[b * (rows * cols) as usize + (r * cols + c) as usize]
                                } else {
                                    T::zero()
                                };
                                out[b * (pr * pc) as usize + p as usize] = v;
                            }
                        }
                    }
                }
            }
            out
        }
        fn run<T: Scalar>(t: &Tensor, shape: &[i64]) -> Result<Tensor, ZyxError> {
            let rank = shape.len();
            debug_assert!(rank >= 2, "tilize needs rank >= 2, got {rank}");
            let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
            let batches: usize = shape[..rank - 2].iter().product::<i64>() as usize;
            let input: Vec<T> = t.to_vec()?;
            let (pr, pc) = (ceil32(rows), ceil32(cols));
            let out = permute(&input, batches, rows, cols, pr, pc);
            let mut new_shape: Vec<i64> = shape[..rank - 2].to_vec();
            new_shape.push(pr);
            new_shape.push(pc);
            Tensor::from_vec(out, new_shape)
        }
        let shape: Vec<i64> = self.shape().iter().map(|d| d.item::<Dim>() as i64).collect();
        match self.dtype() {
            DType::F32 => run::<f32>(self, &shape),
            DType::F16 => run::<crate::scalar::f16>(self, &shape),
            DType::BF16 => run::<crate::scalar::bf16>(self, &shape),
            DType::F64 => run::<f64>(self, &shape),
            DType::U8 => run::<u8>(self, &shape),
            DType::I8 => run::<i8>(self, &shape),
            DType::U16 => run::<u16>(self, &shape),
            DType::I16 => run::<i16>(self, &shape),
            DType::U32 => run::<u32>(self, &shape),
            DType::I32 => run::<i32>(self, &shape),
            DType::U64 => run::<u64>(self, &shape),
            DType::I64 => run::<i64>(self, &shape),
            DType::Bool => run::<bool>(self, &shape),
        }
    }

    /// Inverse of [`Tensor::tilize`] (host-side). `rows`/`cols` are the
    /// true (unpadded) outer dims; padding is stripped. Method form
    /// (chainable): `tilized.untilize(rows, cols)?`.
    ///
    /// # Errors
    ///
    /// Returns [`ZyxError`] if the tensor cannot be read to host or the
    /// result cannot be allocated.
    pub fn untilize(&self, rows: i64, cols: i64) -> Result<Tensor, ZyxError> {
        fn permute<T: Scalar>(input: &[T], batches: usize, pr: i64, pc: i64, rows: i64, cols: i64) -> Vec<T> {
            let ntc = pc / 32;
            let mut out = vec![T::zero(); batches * (rows * cols) as usize];
            for b in 0..batches {
                for tr in 0..pr / 32 {
                    for tc in 0..ntc {
                        for f in 0..4 {
                            let fr = f / 2;
                            let fc = f % 2;
                            for l in 0..256 {
                                let lr = l / 16;
                                let lc = l % 16;
                                let r = tr * 32 + fr * 16 + lr;
                                let c = tc * 32 + fc * 16 + lc;
                                let p = ((tr * ntc + tc) * 4 + f) * 256 + l;
                                if r < rows && c < cols {
                                    out[b * (rows * cols) as usize + (r * cols + c) as usize] =
                                        input[b * (pr * pc) as usize + p as usize];
                                }
                            }
                        }
                    }
                }
            }
            out
        }
        fn run<T: Scalar>(t: &Tensor, shape: &[i64], rows: i64, cols: i64) -> Result<Tensor, ZyxError> {
            let rank = shape.len();
            debug_assert!(rank >= 2, "untilize needs rank >= 2, got {rank}");
            debug_assert!(shape[rank - 2] % 32 == 0 && shape[rank - 1] % 32 == 0);
            let (pr, pc) = (shape[rank - 2], shape[rank - 1]);
            debug_assert!(rows <= pr && cols <= pc);
            let batches: usize = shape[..rank - 2].iter().product::<i64>() as usize;
            let input: Vec<T> = t.to_vec()?;
            let out = permute(&input, batches, pr, pc, rows, cols);
            let mut new_shape: Vec<i64> = shape[..rank - 2].to_vec();
            new_shape.push(rows);
            new_shape.push(cols);
            Tensor::from_vec(out, new_shape)
        }
        let shape: Vec<i64> = self.shape().iter().map(|d| d.item::<Dim>() as i64).collect();
        match self.dtype() {
            DType::F32 => run::<f32>(self, &shape, rows, cols),
            DType::F16 => run::<crate::scalar::f16>(self, &shape, rows, cols),
            DType::BF16 => run::<crate::scalar::bf16>(self, &shape, rows, cols),
            DType::F64 => run::<f64>(self, &shape, rows, cols),
            DType::U8 => run::<u8>(self, &shape, rows, cols),
            DType::I8 => run::<i8>(self, &shape, rows, cols),
            DType::U16 => run::<u16>(self, &shape, rows, cols),
            DType::I16 => run::<i16>(self, &shape, rows, cols),
            DType::U32 => run::<u32>(self, &shape, rows, cols),
            DType::I32 => run::<i32>(self, &shape, rows, cols),
            DType::U64 => run::<u64>(self, &shape, rows, cols),
            DType::I64 => run::<i64>(self, &shape, rows, cols),
            DType::Bool => run::<bool>(self, &shape, rows, cols),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilize_roundtrip_odd_shape() -> Result<(), ZyxError> {
        // Odd, non-tile-multiple shape exercises padding + face order.
        let rows = 37i64;
        let cols = 45i64;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.5 - 100.0).collect();
        let t = Tensor::from_vec(data.clone(), [rows, cols])?;
        let til = Tensor::tilize(&t)?;
        let shape: Vec<i64> = til.shape().iter().map(|d| d.item::<Dim>() as i64).collect();
        assert_eq!(shape, vec![64, 64]);
        // First tile, first face must hold rows 0..16, cols 0..16 in order.
        let flat: Vec<f32> = til.to_vec()?;
        for lr in 0..16 {
            for lc in 0..16 {
                assert_eq!(flat[(lr * 16 + lc) as usize], (lr * cols + lc) as f32 * 0.5 - 100.0);
            }
        }
        // Padded region reads back as zero.
        assert_eq!(flat[((63 * 64 + 63) % (64 * 64)) as usize], 0.0);
        let back = Tensor::untilize(&til, rows, cols)?;
        let shape: Vec<i64> = back.shape().iter().map(|d| d.item::<Dim>() as i64).collect();
        assert_eq!(shape, vec![rows, cols]);
        let rt: Vec<f32> = back.to_vec()?;
        assert_eq!(rt, data);
        Ok(())
    }

    #[test]
    fn tilize_matches_golden_face_order() -> Result<(), ZyxError> {
        // Conformance with the `14_tenstorrent` golden kernel's `lin()`
        // mapping: face slot -> linear index within a tile.
        let lin = |s: usize| {
            let (face, local) = (s / 256, s % 256);
            let (fr0, fc0) = (face / 2, face % 2);
            fr0 * 16 * 32 + fc0 * 16 + (local / 16) * 32 + local % 16
        };
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, [32, 32])?;
        let til = Tensor::tilize(&t)?;
        let flat: Vec<f32> = til.to_vec()?;
        for p in 0..1024 {
            assert_eq!(flat[p], lin(p) as f32, "face slot {p}");
        }
        Ok(())
    }

    #[test]
    fn tilize_batched_f16() -> Result<(), ZyxError> {
        let data: Vec<crate::scalar::f16> =
            (0..2 * 6 * 40).map(|i| crate::scalar::f16::from_f32(i as f32)).collect();
        let t = Tensor::from_vec(data.clone(), [2, 6, 40])?;
        let til = Tensor::tilize(&t)?;
        let shape: Vec<i64> = til.shape().iter().map(|d| d.item::<Dim>() as i64).collect();
        assert_eq!(shape, vec![2, 32, 64]);
        let back = Tensor::untilize(&til, 6, 40)?;
        let rt: Vec<crate::scalar::f16> = back.to_vec()?;
        assert_eq!(rt, data);
        Ok(())
    }
}

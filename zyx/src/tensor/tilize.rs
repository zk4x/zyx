// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tilized layouts as ordinary movement ops.
//!
//! A tilized tensor stores each 32x32 tile contiguously, with the four
//! 16x16 faces inside each tile in face order (the same convention as the
//! `14_tenstorrent` golden kernel's `lin()` mapping). Outer dims are
//! zero-padded up to multiples of 32.
//!
//! Tilize is exactly pad-to-32 + reshape + permute + reshape back to the
//! padded shape — no custom kernel, no host roundtrip, no layout tag. The
//! chain is lazy like any movement op: it fuses, cancels (`tilize` followed
//! by `untilize` simplifies through the existing algebraic rewrites), and is
//! visible to the egraph as dataflow. Tilize once at load time (weights) or
//! once per forward (inputs); on-device kernels chain tilized layouts with
//! no conversion between layers.
//!
//! Both ops need concrete dims (pad amounts and split sizes are constants)
//! and rank >= 2.

use crate::{
    RT, Tensor, ZyxError,
    dtype::{Constant, DType},
    kernel::IDX_T,
    shape::UAxis,
};

impl Tensor {
    /// Permutes the last two dims into Tenstorrent tilized face order,
    /// zero-padding them to multiples of 32; dtype is preserved.
    ///
    /// This is exactly `pad_zeros` + `reshape` + `permute` + `reshape` back
    /// to the padded shape — a lazy movement chain, not a host roundtrip.
    /// Dims stay symbolic throughout: pad amounts and split sizes are dim
    /// expressions, never realized values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::zeros([32, 48], DType::F32);
    /// let til = t.tilize()?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    /// Returns a shape error if rank < 2.
    pub fn tilize(&self) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let rank = shape.len();
        if rank < 2 {
            return Err(ZyxError::shape_error(format!("tilize needs rank >= 2, got {rank}").into()));
        }
        let rows = shape[rank - 2].cast(IDX_T);
        let cols = shape[rank - 1].cast(IDX_T);
        // Dims always resolve (Const/Variable leaves carry values); these
        // asserts document that pad inputs are valid by construction.
        for (name, d) in [("rows", &rows), ("cols", &cols)] {
            let v = {
                let rt = RT.lock();
                rt.resolve_symbolic(d.id).and_then(|c| match c.cast(DType::I64) {
                    Constant::I64(b) => Some(i64::from_le_bytes(b)),
                    _ => None,
                })
            };
            if let Some(v) = v {
                debug_assert!(v >= 0, "tilize: {name} must be non-negative, got {v}");
            }
        }
        // Padded dims, still symbolic: pr = ceil(rows / 32) * 32.
        let c32 = Tensor::from(32i64);
        let pr = &(&(&rows + 31i64) / &c32) * &c32;
        let pc = &(&(&cols + 31i64) / &c32) * &c32;
        let ntr = &pr / &c32;
        let ntc = &pc / &c32;
        // Zero-pad the last two dims up to multiples of 32 (right pad is
        // implicit in `len`, so no pad amount is ever computed). Valid by
        // construction: pr >= rows, pc >= cols.
        let zero = Tensor::from(0i64);
        let padded = self
            .pad_zeros_axis((rank - 2) as UAxis, zero.clone(), pr.clone())
            .unwrap()
            .pad_zeros_axis((rank - 1) as UAxis, zero, pc.clone())
            .unwrap();
        // Split each padded dim into tiles x faces x rows: [..., ntr, 2, 16, ntc, 2, 16].
        let mut split = shape[..rank - 2].to_vec();
        split.extend([
            ntr,
            Tensor::from(2i64),
            Tensor::from(16i64),
            ntc,
            Tensor::from(2i64),
            Tensor::from(16i64),
        ]);
        let split = padded.reshape(split).unwrap();
        // Bring tile/face axes out front: [..., ntr, ntc, 2, 2, 16, 16].
        let k = (rank - 2) as i32;
        let axes: Vec<i32> = (0..k).chain([k, k + 3, k + 1, k + 4, k + 2, k + 5]).collect();
        let permuted = split.permute(axes).unwrap();
        // Merge back to the padded shape; contents are now in face order.
        let mut merged = shape[..rank - 2].to_vec();
        merged.extend([pr, pc]);
        Ok(permuted.reshape(merged).unwrap())
    }

    /// Inverse of [`Tensor::tilize`]: undoes the face-order permutation,
    /// then narrows the last two dims to the true `rows` and `cols`,
    /// stripping the 32-aligned padding.
    ///
    /// This is exactly `reshape` + `permute` + `reshape` + `narrow` — the
    /// mirror of [`Tensor::tilize`], a lazy movement chain with symbolic
    /// dims throughout.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zyx::{Tensor, DType};
    /// let til = Tensor::zeros([64, 64], DType::F32);
    /// let back = til.untilize(32, 48)?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    /// Returns a shape error if rank < 2, the last two dims are not
    /// multiples of 32, or `rows`/`cols` are negative or exceed them.
    pub fn untilize(&self, rows: i64, cols: i64) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let rank = shape.len();
        if rank < 2 {
            return Err(ZyxError::shape_error(format!("untilize needs rank >= 2, got {rank}").into()));
        }
        let prt = shape[rank - 2].cast(IDX_T);
        let pct = shape[rank - 1].cast(IDX_T);
        // Dims always resolve; when they do, wrong user input is a real
        // error here (not a panic). Unresolvable dims skip this and defer
        // to realize time, where narrow fails loudly on the same violation.
        for (name, d, len) in [("rows", &prt, rows), ("cols", &pct, cols)] {
            let v = {
                let rt = RT.lock();
                rt.resolve_symbolic(d.id).and_then(|c| match c.cast(DType::I64) {
                    Constant::I64(b) => Some(i64::from_le_bytes(b)),
                    _ => None,
                })
            };
            if let Some(p) = v {
                if p % 32 != 0 {
                    return Err(ZyxError::shape_error(
                        format!("untilize needs the last two dims to be multiples of 32, got {p}").into(),
                    ));
                }
                if len < 0 || len > p {
                    return Err(ZyxError::shape_error(format!("untilize: {name}={len} out of range for padded dim {p}").into()));
                }
            }
        }
        let c32 = Tensor::from(32i64);
        let ntr = &prt / &c32;
        let ntc = &pct / &c32;
        // Split the face-ordered dims back out: [..., ntr, ntc, 2, 2, 16, 16].
        // Valid by construction once the preconditions above hold.
        let mut split = shape[..rank - 2].to_vec();
        split.extend([
            ntr,
            ntc,
            Tensor::from(2i64),
            Tensor::from(2i64),
            Tensor::from(16i64),
            Tensor::from(16i64),
        ]);
        let split = self.reshape(split).unwrap();
        // Inverse permutation: [..., ntr, 2, 16, ntc, 2, 16].
        let k = (rank - 2) as i32;
        let axes: Vec<i32> = (0..k).chain([k, k + 2, k + 4, k + 1, k + 3, k + 5]).collect();
        let permuted = split.permute(axes).unwrap();
        // Merge back to the padded shape, then strip the padding.
        let mut merged = shape[..rank - 2].to_vec();
        merged.extend([prt, pct]);
        let merged = permuted.reshape(merged).unwrap();
        Ok(merged.narrow((rank - 2) as i32, 0i64, rows).unwrap().narrow((rank - 1) as i32, 0i64, cols).unwrap())
    }
}

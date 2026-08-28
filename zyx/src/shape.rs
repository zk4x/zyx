// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Few traits that describe shapes, axes, padding, etc.

use core::fmt::Debug;

use crate::{error::ZyxError, tensor::Axis};

/// Type alias for dimension values (i64)
///
/// # Convention: `-1` means dynamic/symbolic
///
/// A `Dim` of `-1` does NOT mean an invalid dimension — it marks a **dynamic
/// (symbolic) dimension** whose length is only known at kernel launch time
/// (supplied as a `Param { kind: Variable }` scalar). Any nonnegative value
/// is a static, known length (emitted as a `Const`).
pub type Dim = i64;
/// Type alias for axis indices (usize)
pub type UAxis = usize;

pub fn into_axis(axis: Axis, rank: UAxis) -> Result<UAxis, ZyxError> {
    TryInto::<Axis>::try_into(rank).map_or_else(
        |_| Err(ZyxError::ShapeError(format!("Axis {axis} is out of range of rank {rank}").into())),
        |rank2| {
            TryInto::<UAxis>::try_into(axis + rank2).map_or_else(
                |_| Err(ZyxError::ShapeError(format!("Axis {axis} is out of range of rank {rank}").into())),
                |a| {
                    if a < 2 * rank {
                        Ok(a % rank)
                    } else {
                        Err(ZyxError::ShapeError(format!("Axis {axis} is out of range of rank {rank}").into()))
                    }
                },
            )
        },
    )
}

pub fn into_axes(axes: impl IntoIterator<Item = Axis>, rank: UAxis) -> Result<Vec<UAxis>, ZyxError> {
    let mut res = Vec::with_capacity(rank);
    let mut visited = std::collections::BTreeSet::new();
    for axis in axes {
        let a = into_axis(axis, rank)?;
        if visited.insert(a) {
            res.push(a);
        }
    }
    if res.is_empty() {
        return Ok((0..rank).collect());
    }
    Ok(res)
}

#[must_use]
pub fn permute<T: Clone>(shape: &[T], axes: &[UAxis]) -> Vec<T> {
    debug_assert_eq!(shape.len(), axes.len());
    axes.iter().map(|a| shape[*a].clone()).collect()
}

pub fn pad(shape: &mut [Dim], padding: &[(i64, i64)]) {
    let mut i = 0;
    for d in shape.iter_mut() {
        *d = Dim::try_from(i64::try_from(*d).unwrap() + padding[i].0 + padding[i].1).unwrap();
        i += 1;
        if i >= padding.len() {
            break;
        }
    }
}

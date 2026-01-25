//! Few traits that describe shapes, axes, padding, etc.

use core::fmt::Debug;

use crate::{error::ZyxError, tensor::Axis};

pub type Dim = usize;
pub type UAxis = usize;

/// `IntoShape` trait
pub trait IntoShape: Clone + Debug {
    /// Convert value into shape (iterator over dimensions)
    fn into_shape(self) -> impl Iterator<Item = Dim>;
    /// Get the rank of the shape
    fn rank(&self) -> UAxis;
}

impl IntoShape for Dim {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        [self].into_iter()
    }

    fn rank(&self) -> UAxis {
        1
    }
}

impl IntoShape for (Dim, Dim) {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        [self.0, self.1].into_iter()
    }

    fn rank(&self) -> UAxis {
        2
    }
}

impl IntoShape for (Dim, Dim, Dim) {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        [self.0, self.1, self.2].into_iter()
    }

    fn rank(&self) -> UAxis {
        3
    }
}

impl<const N: usize> IntoShape for [Dim; N] {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.into_iter()
    }

    fn rank(&self) -> UAxis {
        N as UAxis
    }
}

impl IntoShape for &[Dim] {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.iter().copied()
    }

    fn rank(&self) -> UAxis {
        self.len() as UAxis
    }
}

impl IntoShape for Vec<Dim> {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.into_iter()
    }

    fn rank(&self) -> UAxis {
        self.len() as UAxis
    }
}

impl IntoShape for &Vec<Dim> {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.iter().copied()
    }

    fn rank(&self) -> UAxis {
        self.len() as UAxis
    }
}

pub fn into_axis(axis: Axis, rank: UAxis) -> Result<UAxis, ZyxError> {
    TryInto::<Axis>::try_into(rank).map_or_else(
        |_| {
            Err(ZyxError::ShapeError(
                format!("Axis {axis} is out of range of rank {rank}").into(),
            ))
        },
        |rank2| {
            TryInto::<UAxis>::try_into(axis + rank2).map_or_else(
                |_| {
                    Err(ZyxError::ShapeError(
                        format!("Axis {axis} is out of range of rank {rank}").into(),
                    ))
                },
                |a| {
                    if a < 2 * rank {
                        Ok(a % rank)
                    } else {
                        Err(ZyxError::ShapeError(
                            format!("Axis {axis} is out of range of rank {rank}").into(),
                        ))
                    }
                },
            )
        },
    )
}

pub fn into_axes(axes: impl IntoIterator<Item = Axis>, rank: UAxis) -> Result<Vec<UAxis>, ZyxError> {
    let mut res = Vec::new();
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
    axes.iter().map(|a| shape[*a as usize].clone()).collect()
}

pub fn reduce(shape: &[Dim], axes: &[UAxis]) -> Vec<Dim> {
    let res: Vec<_> = shape
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, d)| if axes.contains(&(i as UAxis)) { None } else { Some(d) })
        .collect();
    if res.is_empty() { vec![1] } else { res }
}

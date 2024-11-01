//! Few traits that describe shapes, axes, padding, etc.

use crate::ZyxError;
use core::fmt::Debug;

pub(crate) type Dimension = usize;
pub(crate) type Axis = usize;

/// IntoShape trait
pub trait IntoShape: Clone + Debug {
    /// Convert value into shape (iterator over dimensions)
    fn into_shape(self) -> impl Iterator<Item = Dimension>;
    /// Get the rank of the shape
    fn rank(&self) -> usize;
}

impl IntoShape for Dimension {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        [self].into_iter()
    }

    fn rank(&self) -> usize {
        1
    }
}

impl IntoShape for (Dimension, Dimension) {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        [self.0, self.1].into_iter()
    }

    fn rank(&self) -> usize {
        2
    }
}

impl IntoShape for (Dimension, Dimension, Dimension) {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        [self.0, self.1, self.2].into_iter()
    }

    fn rank(&self) -> usize {
        3
    }
}

impl<const N: usize> IntoShape for [Dimension; N] {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        self.into_iter()
    }

    fn rank(&self) -> usize {
        N
    }
}

impl IntoShape for &[Dimension] {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        self.iter().copied()
    }

    fn rank(&self) -> usize {
        self.len()
    }
}

impl IntoShape for Vec<Dimension> {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        self.into_iter()
    }

    fn rank(&self) -> usize {
        self.len()
    }
}

impl IntoShape for &Vec<Dimension> {
    fn into_shape(self) -> impl Iterator<Item = Dimension> {
        self.iter().copied()
    }

    fn rank(&self) -> usize {
        self.len()
    }
}

pub(crate) fn into_axis(axis: isize, rank: usize) -> Result<usize, ZyxError> {
    if let Ok(rank2) = TryInto::<isize>::try_into(rank) {
        if let Ok(a) = TryInto::<usize>::try_into(axis + rank2) {
            if a < 2 * rank {
                Ok(a % rank)
            } else {
                Err(ZyxError::ShapeError(format!(
                    "Axis {axis} is out of range of rank {rank}"
                )))
            }
        } else {
            Err(ZyxError::ShapeError(format!(
                "Axis {axis} is out of range of rank {rank}"
            )))
        }
    } else {
        Err(ZyxError::ShapeError(format!(
            "Axis {axis} is out of range of rank {rank}"
        )))
    }
}

pub(crate) fn into_axes(
    axes: impl IntoIterator<Item = isize>,
    rank: usize,
) -> Result<Vec<usize>, ZyxError> {
    let mut res = Vec::new();
    let mut visited = std::collections::BTreeSet::new();
    for axis in axes.into_iter() {
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

pub(crate) fn permute(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    assert_eq!(shape.len(), axes.len());
    axes.iter().map(|a| shape[*a]).collect()
}

pub(crate) fn reduce(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let res: Vec<usize> = shape
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(i, d)| if axes.contains(&i) { None } else { Some(d) })
        .collect();
    if res.is_empty() {
        vec![1]
    } else {
        res
    }
}

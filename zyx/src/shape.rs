//! Few traits that describe shapes, axes, padding, etc.

use core::fmt::Debug;

use crate::error::ZyxError;

pub type Dim = usize;
pub type Axis = usize;

/// `IntoShape` trait
pub trait IntoShape: Clone + Debug {
    /// Convert value into shape (iterator over dimensions)
    fn into_shape(self) -> impl Iterator<Item = Dim>;
    /// Get the rank of the shape
    fn rank(&self) -> Axis;
}

impl IntoShape for Dim {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        [self].into_iter()
    }

    fn rank(&self) -> Axis {
        1
    }
}

impl IntoShape for (Dim, Dim) {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        [self.0, self.1].into_iter()
    }

    fn rank(&self) -> Axis {
        2
    }
}

impl IntoShape for (Dim, Dim, Dim) {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        [self.0, self.1, self.2].into_iter()
    }

    fn rank(&self) -> Axis {
        3
    }
}

impl<const N: usize> IntoShape for [Dim; N] {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.into_iter()
    }

    fn rank(&self) -> Axis {
        N
    }
}

impl IntoShape for &[Dim] {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.iter().copied()
    }

    fn rank(&self) -> Axis {
        self.len()
    }
}

impl IntoShape for Vec<Dim> {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.into_iter()
    }

    fn rank(&self) -> Axis {
        self.len()
    }
}

impl IntoShape for &Vec<Dim> {
    fn into_shape(self) -> impl Iterator<Item = Dim> {
        self.iter().copied()
    }

    fn rank(&self) -> Axis {
        self.len()
    }
}

pub fn into_axis(axis: isize, rank: Axis) -> Result<Axis, ZyxError> {
    TryInto::<isize>::try_into(rank).map_or_else(
        |_| {
            Err(ZyxError::ShapeError(format!(
                "Axis {axis} is out of range of rank {rank}"
            ).into()))
        },
        |rank2| {
            TryInto::<Axis>::try_into(axis + rank2).map_or_else(
                |_| {
                    Err(ZyxError::ShapeError(format!(
                        "Axis {axis} is out of range of rank {rank}"
                    ).into()))
                },
                |a| {
                    if a < 2 * rank {
                        Ok(a % rank)
                    } else {
                        Err(ZyxError::ShapeError(format!(
                            "Axis {axis} is out of range of rank {rank}"
                        ).into()))
                    }
                },
            )
        },
    )
}

pub fn into_axes(
    axes: impl IntoIterator<Item = isize>,
    rank: Axis,
) -> Result<Vec<Dim>, ZyxError> {
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

pub fn permute(shape: &[Dim], axes: &[Axis]) -> Vec<Dim> {
    debug_assert_eq!(shape.len(), axes.len());
    axes.iter().map(|a| shape[*a]).collect()
}

pub fn reduce(shape: &[Dim], axes: &[Axis]) -> Vec<Dim> {
    let res: Vec<_> = shape
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

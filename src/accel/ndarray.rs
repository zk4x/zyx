use crate::{ops::{self, IntoVec, FromVec}, shape::{IntoShape, Shape}};
use ndarray::{ArrayBase, Dim, IxDynImpl, RawData, DataOwned, Dimension, OwnedRepr};
use num_traits::identities::{Zero, One};

impl<T, D> ops::GetShape for ArrayBase<T, D>
where
    D: Dimension,
    T: RawData,
{
    fn shape(&self) -> Shape {
        ArrayBase::shape(self).shape()
    }
}

impl<S, T> ops::Zeros for ArrayBase<S, Dim<IxDynImpl>>
where
    T: Clone + Zero,
    S: DataOwned<Elem = T>,
{
    fn zeros(shape: impl IntoShape) -> Self {
        ArrayBase::zeros(shape.shape().to_vec())
    }
}

impl<S, T> ops::Zeros for ArrayBase<S, Dim<[usize; 1]>>
where
    T: Clone + Zero,
    S: DataOwned<Elem = T>,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 1]>>::zeros(shape[-1])
    }
}

impl<S, T> ops::Zeros for ArrayBase<S, Dim<[usize; 2]>>
where
    T: Clone + Zero,
    S: DataOwned<Elem = T>,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 2]>>::zeros((shape[-2], shape[-1]))
    }
}

impl<S, T> ops::Zeros for ArrayBase<S, Dim<[usize; 3]>>
where
    T: Clone + Zero,
    S: DataOwned<Elem = T>,
{
    fn zeros(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 3]>>::zeros([shape[-3], shape[-2], shape[-1]])
    }
}

impl<S, T> ops::Ones for ArrayBase<S, Dim<IxDynImpl>>
where
    T: Clone + One,
    S: DataOwned<Elem = T>,
{
    fn ones(shape: impl IntoShape) -> Self {
        ArrayBase::ones(shape.shape().to_vec())
    }
}

impl<S, T> ops::Ones for ArrayBase<S, Dim<[usize; 1]>>
where
    T: Clone + One,
    S: DataOwned<Elem = T>,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 1]>>::ones(shape[-1])
    }
}

impl<S, T> ops::Ones for ArrayBase<S, Dim<[usize; 2]>>
where
    T: Clone + One,
    S: DataOwned<Elem = T>,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 2]>>::ones((shape[-2], shape[-1]))
    }
}

impl<S, T> ops::Ones for ArrayBase<S, Dim<[usize; 3]>>
where
    T: Clone + One,
    S: DataOwned<Elem = T>,
{
    fn ones(shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 3]>>::ones([shape[-3], shape[-2], shape[-1]])
    }
}

impl<S, T> FromVec<T> for ArrayBase<S, Dim<[usize; 1]>>
where
    S: DataOwned<Elem = T>,
{
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 1]>>::from_shape_vec([shape[-1]], data).ok().unwrap()
    }
}

impl<S, T> FromVec<T> for ArrayBase<S, Dim<[usize; 2]>>
where
    S: DataOwned<Elem = T>,
{
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 2]>>::from_shape_vec([shape[-2], shape[-1]], data).ok().unwrap()
    }
}

impl<S, T> FromVec<T> for ArrayBase<S, Dim<[usize; 3]>>
where
    S: DataOwned<Elem = T>,
{
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        let shape = shape.shape();
        ArrayBase::<S, Dim<[usize; 3]>>::from_shape_vec([shape[-3], shape[-2], shape[-1]], data).ok().unwrap()
    }
}

impl<T, S> FromVec<T> for ArrayBase<S, Dim<IxDynImpl>>
where
    S: DataOwned<Elem = T>,
{
    fn from_vec(data: Vec<T>, shape: impl IntoShape) -> Self {
        ArrayBase::from_shape_vec(shape.shape().to_vec(), data).ok().unwrap()
    }
}

impl<T, D> ops::ReLU for ArrayBase<OwnedRepr<T>, D>
where
    D: Dimension,
    T: Clone + ops::ReLU<Output = T>,
{
    type Output = Self;
    fn relu(self) -> Self::Output {
        self.map(|x| x.clone().relu())
    }
}

impl<T, D> ops::DReLU for ArrayBase<OwnedRepr<T>, D>
where
    D: Dimension,
    T: Clone + ops::DReLU<Output = T>,
{
    type Output = Self;
    fn drelu(self) -> Self::Output {
        self.map(|x| x.clone().drelu())
    }
}

impl<T, D> ops::Exp for ArrayBase<OwnedRepr<T>, D>
where
    D: Dimension,
    T: Clone + ops::Exp<Output = T>,
{
    type Output = Self;
    fn exp(self) -> Self::Output {
        self.map(|x| x.clone().exp())
    }
}

impl<T, D> ops::Ln for ArrayBase<OwnedRepr<T>, D>
where
    D: Dimension,
    T: Clone + ops::Ln<Output = T>,
{
    type Output = Self;
    fn ln(self) -> Self::Output {
        self.map(|x| x.clone().ln())
    }
}

impl<T, D> ops::Tanh for ArrayBase<OwnedRepr<T>, D>
where
    D: Dimension,
    T: Clone + ops::Tanh<Output = T>,
{
    type Output = Self;
    fn tanh(self) -> Self::Output {
        self.map(|x| x.clone().tanh())
    }
}

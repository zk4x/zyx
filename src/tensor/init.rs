//! Methods for tensor initialization
//! 

use crate::{buffer::cpu, tensor::Tensor, ops, shape::Shape};
use std::rc::Rc;

pub trait DType {}

impl DType for f32 {}
impl DType for f64 {}
impl DType for u8 {}
impl DType for i8 {}
impl DType for i16 {}
impl DType for i32 {}
impl DType for i64 {}
impl DType for i128 {}
impl DType for bool {}

// #[cfg(Default)]
type Storage<T> = cpu::Buffer<T>;
// #[cfg(Opencl)]
//type Storage<T, const D3: usize, const D2: usize, const D1: usize, const D0: usize> = opencl::Buffer<T, D3, D2, D1, D0>;

impl<T> Tensor<Storage<T>>
where
    T: Default,
{
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> Tensor<Storage<T>>
{
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        debug_assert_eq!(data.len(), shape.numel());
        use ops::FromVec;
        Self {
            data: Rc::new(Storage::from_vec(data, shape)),
        }
    }
}

impl<T> Tensor<Storage<T>>
where
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    pub fn randn(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| rng.gen()).collect(), shape)
    }
}

impl<T> Tensor<Storage<T>>
where
    T: rand::distributions::uniform::SampleUniform,
{
    pub fn uniform(shape: &[usize], low: T, high: T) -> Self {
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        Self::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| dist.sample(&mut rng)).collect(), shape)
    }
}

impl<T> ops::Zeros for Tensor<Storage<T>>
where
    T: ops::Zeros + Clone,
{
    fn zeros(shape: &[usize]) -> Self {
        Self::from_vec(vec![T::zeros(&[]); shape.numel()], shape)
    }
}

impl<T> ops::Ones for Tensor<Storage<T>>
where
    T: ops::Ones + Clone,
{
    fn ones(shape: &[usize]) -> Self {
        Self::from_vec(vec![T::ones(&[]); shape.numel()], shape)
    }
}

impl<F, T> From<(F, &[usize])> for Tensor<Storage<T>>
where
    T: Clone,
    F: FnMut() -> T,
{
    fn from(mut f: (F, &[usize])) -> Self {
        Self::from_vec(std::iter::repeat(0).take(f.1.numel()).map(|_| f.0()).collect(), f.1)
    }
}

impl<T, const D0: usize> From<[T; D0]> for Tensor<Storage<T>>
where
    T: DType + Clone
{
    fn from(x: [T; D0]) -> Self {
        Self::from_vec(x.to_vec(), &[D0])
    }
}

impl<T, const D1: usize, const D0: usize> From<[[T; D0]; D1]> for Tensor<Storage<T>>
where
    T: DType + Clone
{
    fn from(x: [[T; D0]; D1]) -> Self {
        Self::from_vec(x.into_iter().flatten().collect(), &[D1, D0])
    }
}

impl<T, const D2: usize, const D1: usize, const D0: usize> From<[[[T; D0]; D1]; D2]> for Tensor<Storage<T>>
where
    T: DType + Clone
{
    fn from(x: [[[T; D0]; D1]; D2]) -> Self {
        Self::from_vec(x.into_iter().flatten().flatten().collect(), &[D2, D1, D0])
    }
}

impl<T, const D3: usize, const D2: usize, const D1: usize, const D0: usize> From<[[[[T; D0]; D1]; D2]; D3]> for Tensor<Storage<T>>
where
    T: DType + Clone
{
    fn from(x: [[[[T; D0]; D1]; D2]; D3]) -> Self {
        Self::from_vec(x.into_iter().flatten().flatten().flatten().collect(), &[D3, D2, D1, D0])
    }
}

impl<T, const D4: usize, const D3: usize, const D2: usize, const D1: usize, const D0: usize> From<[[[[[T; D0]; D1]; D2]; D3]; D4]> for Tensor<Storage<T>>
where
    T: DType + Clone
{
    fn from(x: [[[[[T; D0]; D1]; D2]; D3]; D4]) -> Self {
        Self::from_vec(x.into_iter().flatten().flatten().flatten().flatten().collect(), &[D4, D3, D2, D1, D0])
    }
}

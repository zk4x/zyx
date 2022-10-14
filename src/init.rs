use crate::{ops::FromVec, shape::IntoShape};

pub trait RandInit<T> {
    fn randn(shape: impl IntoShape) -> Self;
}

pub trait UniformInit<T> {
    fn uniform(shape: impl IntoShape, low: T, high: T) -> Self;
}

impl<S, T> RandInit<T> for S
where
    S: FromVec<T>,
    rand::distributions::Standard: rand::prelude::Distribution<T>,
{
    fn randn(shape: impl IntoShape) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let shape = shape.shape();
        Self::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| rng.gen()).collect(), shape)
    }
}

impl<S, T> UniformInit<T> for S
where
    S: FromVec<T>,
    T: rand::distributions::uniform::SampleUniform,
{
    fn uniform(shape: impl IntoShape, low: T, high: T) -> Self {
        use rand::distributions::Distribution;
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(low, high);
        let shape = shape.shape();
        Self::from_vec(std::iter::repeat(0).take(shape.numel()).map(|_| dist.sample(&mut rng)).collect(), shape)
    }
}

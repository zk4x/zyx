use super::Shape;

/// # BinOpShape
/// 
/// This trait is a way to encode at compile time the type of the shape resulting
/// from addition of two tensors with different shapes.
pub trait BinOpShape<RhsShape>
where
    RhsShape: Shape<D = usize>,
{
    /// Output [Shape] when adding two tensors with different [shapes](Shape) together
    type Output: Shape<D = usize>;
}

// This requires nightly
/*impl<const N2: usize, const N: usize> BinOpShape<D, [i32; N2]> for [usize; N] {
    type Output = [Self::D; { (N > N2) * N + (N2 > N) * N2 }];
}*/

impl BinOpShape<usize> for usize {
    type Output = usize;
}

impl BinOpShape<(usize, usize)> for (usize, usize) {
    type Output = (usize, usize);
}

impl BinOpShape<usize> for (usize, usize) {
    type Output = (usize, usize);
}

impl BinOpShape<(usize, usize)> for usize {
    type Output = (usize, usize);
}

impl BinOpShape<(usize, usize, usize)> for (usize, usize, usize) {
    type Output = (usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize)> for (usize, usize) {
    type Output = (usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize)> for usize {
    type Output = (usize, usize, usize);
}

impl BinOpShape<usize> for (usize, usize, usize) {
    type Output = (usize, usize, usize);
}

impl BinOpShape<(usize, usize)> for (usize, usize, usize) {
    type Output = (usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize, usize)> for (usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize)> for (usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize)> for (usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<usize> for (usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize, usize)> for (usize, usize, usize) {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize, usize)> for (usize, usize) {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize, usize)> for usize {
    type Output = (usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize, usize, usize)> for (usize, usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize, usize)> for (usize, usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize, usize)> for (usize, usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize, usize);
}

impl BinOpShape<(usize, usize)> for (usize, usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize, usize);
}

impl BinOpShape<usize> for (usize, usize, usize, usize, usize) {
    type Output = (usize, usize, usize, usize, usize);
}

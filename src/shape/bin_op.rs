use super::Shape;

pub trait BinOpShape<RhsShape>
where
    RhsShape: Shape<D = usize>,
{
    type Output: Shape<D = usize>;
}

impl BinOpShape<usize> for usize {
    type Output = usize;
}

// This requires nightly
/*impl<const N2: usize, const N: usize> BinOpShape<D, [i32; N2]> for [usize; N] {
    type Output = [Self::D; { (N > N2) * N + (N2 > N) * N2 }];
}*/

impl BinOpShape<(usize, usize)> for (usize, usize) {
    type Output = (usize, usize);
}

impl BinOpShape<usize> for (usize, usize) {
    type Output = (usize, usize);
}

impl BinOpShape<(usize, usize)> for usize {
    type Output = (usize, usize);
}

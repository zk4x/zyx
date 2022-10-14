mod into_vec;
mod into_shape;
mod relu;
mod exp;
mod ln;
mod tanh;
mod neg;
mod sum;
mod max;
mod min;
mod reshape;
mod expand;
mod permute;
mod add;
mod sub;
mod mul;
mod div;
mod pow;
mod matmul;
mod conv;

// We need custom implementation of replace_take() for RefCell that has F: FnOnce(T) -> T instead of F: FnOnce(&mut T) -> T
// This is due to the fact, that buffers implement operations on consumed values, not on references
pub(super) trait RefCellReplaceTake<T, F> {
    fn replace_take(&self, f: F);
}

// This requires T to be Default
// However we would like to make it so that it is not needed.
// There should be unsafe way to simply move the data from RefCell and leave
// it filled with zeros
impl<T, F> RefCellReplaceTake<T, F> for std::cell::RefCell<T>
where
    T: Default,
    F: FnOnce(T) -> T,
{
    fn replace_take(&self, f: F) {
        self.replace(f(self.take()));
    }
}

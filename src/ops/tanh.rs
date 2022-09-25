use super::Tanh;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Tanh for dtype {
    type Output = Self;
    fn tanh(self) -> Self::Output {
        self.tanh()
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Tanh for &dtype {
    type Output = dtype;
    fn tanh(self) -> Self::Output {
        (*self).tanh()
    }
}

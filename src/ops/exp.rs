use super::Exp;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Exp for dtype {
    type Output = Self;
    fn exp(self) -> Self::Output {
        self.exp()
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Exp for &dtype {
    type Output = dtype;
    fn exp(self) -> Self::Output {
        (*self).exp()
    }
}

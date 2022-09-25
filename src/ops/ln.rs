use super::Ln;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Ln for dtype {
    type Output = Self;
    fn ln(self) -> Self::Output {
        self.ln()
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl Ln for &dtype {
    type Output = dtype;
    fn ln(self) -> Self::Output {
        (*self).ln()
    }
}

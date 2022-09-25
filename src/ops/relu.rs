use super::ReLU;
use duplicate::duplicate_item;

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl ReLU for dtype {
    type Output = Self;
    fn relu(self) -> Self::Output {
        self.max(0.)
    }
}

#[duplicate_item(
    dtype;
    [f32];
    [f64];
)]

impl ReLU for &dtype {
    type Output = dtype;
    fn relu(self) -> Self::Output {
        self.max(0.)
    }
}


#[duplicate_item(
    dtypei;
    [i32];
    [i64];
)]

impl ReLU for dtypei {
    type Output = Self;
    fn relu(self) -> Self::Output {
        self.max(0)
    }
}
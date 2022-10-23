use super::ConvertFrom;

impl ConvertFrom<f64> for f32 {
    fn cfrom(x: f64) -> Self {
        x as f32
    }
}

impl ConvertFrom<i32> for f32 {
    fn cfrom(x: i32) -> Self {
        x as f32
    }
}

impl ConvertFrom<usize> for f32 {
    fn cfrom(x: usize) -> Self {
        x as f32
    }
}

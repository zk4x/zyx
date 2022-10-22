use super::ConvertFrom;

impl ConvertFrom<f64> for f32 {
    fn cfrom(x: f64) -> Self {
        x as f32
    }
}

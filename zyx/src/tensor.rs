use core::ops::{Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use crate::device::Device;
use crate::dtype::DType;
use crate::RT;
use crate::scalar::Scalar;
use crate::shape::{IntoAxes, IntoShape};
use alloc::vec::Vec;
use half::f16;

pub struct Tensor {
    id: u32,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        RT.lock().retain(self.id);
        Tensor {
            id: self.id
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        RT.lock().release(self.id);
    }
}

impl Tensor {
    pub(crate) fn new(id: usize) -> Tensor {
        Tensor {
            id: id as u32,
        }
    }
}

impl Tensor {
    /// Get default device used for new tensors.
    #[must_use]
    pub fn default_device() -> Device {
        RT.lock().default_device
    }

    /// Set default device used for new tensors.
    /// Returns true if the device initialized successfully.
    /// Returns false if the device failed to initialize.
    pub fn set_default_device(device: Device) -> bool {
        let mut g = RT.lock();
        g.default_device = device;
        g.initialize_device(device)
    }

    /// Tries to initialize all devices and set the first
    /// successfully initialized device as the default_device in this order:
    /// 1. CUDA
    /// 2. OpenCL
    /// 3. WGPU
    /// If they all fail to initialize, then default_device
    /// is set to CPU.
    pub fn set_default_device_best() {
        RT.lock().set_default_device_best();
    }

    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        RT.lock().shape(self.id)
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        todo!()
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        todo!()
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        RT.lock().dtype(self.id)
    }

    #[must_use]
    pub fn device(&self) -> Device {
        RT.lock().device(self.id)
    }

    #[must_use]
    pub fn to(self, device: Device) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn randn(shape: impl IntoShape, dtype: DType) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn uniform<T: Scalar>(shape: impl IntoShape, range: impl RangeBounds<T>) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn kaiming_uniform<T: Scalar>(shape: impl IntoShape, range: impl RangeBounds<T>) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn zeros(shape: impl IntoShape, dtype: DType) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn ones(shape: impl IntoShape, dtype: DType) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn eye(n: usize, dtype: DType) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn full(shape: impl IntoShape, value: impl Scalar) -> Tensor {
        todo!()
    }

    // unary
    #[must_use]
    pub fn abs(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn celu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn cos(&self) -> Tensor {
        RT.lock().cos(self.id)
    }

    #[must_use]
    pub fn dropout(&self, probability: impl Scalar) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn elu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn exp(&self) -> Tensor {
        RT.lock().exp(self.id)
    }

    #[must_use]
    pub fn gelu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn leaky_relu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn ln(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn mish(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn quick_gelu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn reciprocal(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn relu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn rsqrt(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn selu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn sigmoid(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn sin(&self) -> Tensor {
        RT.lock().sin(self.id)
    }

    #[must_use]
    pub fn softplus(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        RT.lock().sqrt(self.id)
    }

    #[must_use]
    pub fn swish(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn tan(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn tanh(&self) -> Tensor {
        RT.lock().tanh(self.id)
    }

    // movement
    #[must_use]
    pub fn reshape(&self, shape: impl IntoShape) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn transpose(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn get(&self, index: impl IntoIndex) -> Tensor {
        todo!()
    }
}

/*impl<T: Scalar> TryInto<T> for Tensor {
    type Error = ();
    fn try_into(self) -> Result<T, Self::Error> {
        todo!()
    }
}*/

impl<T: Scalar> TryInto<Vec<T>> for Tensor {
    type Error = ();
    fn try_into(self) -> Result<Vec<T>, Self::Error> {
        todo!()
    }
}

impl core::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
        //f.write_fmt(format_args!("Tensor {{ id = {:?} }}", self.id))
    }
}

impl core::fmt::Display for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO don't print the whole tensor if it is too big
        let precision = if let Some(precision) = f.precision() {
            precision
        } else {
            3
        };
        let res = match self.dtype() {
            DType::F16 => {
                if let Ok(data) = &self.to_vec::<f16>() {
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "f32 tensor failed to realize".into()
                }
            }
            DType::F32 => {
                if let Ok(data) = &self.to_vec::<f32>() {
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "f32 tensor failed to realize".into()
                }
            }
            DType::F64 => {
                if let Ok(data) = &self.to_vec::<f64>() {
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "f64 tensor failed to realize".into()
                }
            }
            DType::I32 => {
                if let Ok(data) = &self.to_vec::<i32>() {
                    tensor_to_string(data, &self.shape(), precision, f.width())
                } else {
                    "i32 tensor failed to realize".into()
                }
            }
            _ => todo!()
        };
        f.write_fmt(format_args!(
            "Tensor {:?} {}\n{res}",
            self.shape(),
            self.dtype()
        ))
    }
}

fn tensor_to_string<T: core::fmt::Display>(
    data: &[T],
    shape: &[usize],
    precision: usize,
    width: Option<usize>,
) -> alloc::string::String {
    use core::fmt::Write;
    let n = shape.numel();
    let ndim = shape.rank();
    let mut res = alloc::string::String::new();
    if data.is_empty() {
        return "[]".into();
    }
    // get maximal width of single value
    let mut w = 0;
    if let Some(width) = width {
        w = width;
    } else {
        for x in data {
            let l = alloc::format!("{x:>.precision$}").len();
            if l > w {
                w = l;
            }
        }
    }
    let d0 = shape[-1];
    for (i, x) in data.iter().enumerate() {
        {
            let mut var = 1;
            let mut r = ndim;
            while r > 0 {
                if i % (n / var) == 0 {
                    res += &(" ".repeat(ndim - r) + "[".repeat(r - 1).as_str());
                    break;
                }
                var *= shape[ndim - r];
                r -= 1;
            }
        }
        let _ = write!(res, "{x:>w$.precision$}");
        if (i + 1) % d0 != 0usize {
            res += "  ";
        }
        {
            let mut var = 1;
            let mut r = ndim;
            while r > 0 {
                if (i + 1) % (n / var) == 0 {
                    res += &"]".repeat(r - 1);
                    break;
                }
                var *= shape[ndim - r];
                r -= 1;
            }
        }
        if (i + 1) % d0 == 0usize && i != n - 1 {
            res += "\n";
        }
    }
    res
}

/// Into i64 range, used for indexing
pub trait IntoRange: Clone {
    /// Convert self to range i64, if it is scalar, it gets converted to x..x+1
    fn into_range(self) -> Range<i64>;
}

impl IntoRange for RangeFull {
    fn into_range(self) -> Range<i64> {
        0..i64::MAX
    }
}

impl IntoRange for RangeFrom<i64> {
    fn into_range(self) -> Range<i64> {
        self.start..i64::MAX
    }
}

impl IntoRange for RangeTo<i64> {
    fn into_range(self) -> Range<i64> {
        0..self.end
    }
}

impl IntoRange for RangeInclusive<i64> {
    fn into_range(self) -> Range<i64> {
        *self.start()..*self.end() + 1
    }
}

impl IntoRange for RangeToInclusive<i64> {
    fn into_range(self) -> Range<i64> {
        0..self.end + 1
    }
}

impl IntoRange for Range<i64> {
    fn into_range(self) -> Range<i64> {
        self
    }
}

impl IntoRange for i64 {
    fn into_range(self) -> Range<i64> {
        self..self + 1
    }
}

/// Implemented for objects that can be used to index tensors.
pub trait IntoIndex {
    /// Convert self to tensor index.
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>>;
}

impl<I: IntoRange> IntoIndex for &[I] {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        self.iter().cloned().map(IntoRange::into_range)
    }
}

impl<I0: IntoRange> IntoIndex for I0 {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [self.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange> IntoIndex for (I0, I1) {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [self.0.into_range(), self.1.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange> IntoIndex for (I0, I1, I2) {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
        ]
            .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange> IntoIndex for (I0, I1, I2, I3) {
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
        ]
            .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange> IntoIndex
for (I0, I1, I2, I3, I4)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
        ]
            .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange, I4: IntoRange, I5: IntoRange>
IntoIndex for (I0, I1, I2, I3, I4, I5)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
            self.5.into_range(),
        ]
            .into_iter()
    }
}

impl<
    I0: IntoRange,
    I1: IntoRange,
    I2: IntoRange,
    I3: IntoRange,
    I4: IntoRange,
    I5: IntoRange,
    I6: IntoRange,
> IntoIndex for (I0, I1, I2, I3, I4, I5, I6)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
            self.5.into_range(),
            self.6.into_range(),
        ]
            .into_iter()
    }
}

impl<
    I0: IntoRange,
    I1: IntoRange,
    I2: IntoRange,
    I3: IntoRange,
    I4: IntoRange,
    I5: IntoRange,
    I6: IntoRange,
    I7: IntoRange,
> IntoIndex for (I0, I1, I2, I3, I4, I5, I6, I7)
{
    fn into_index(self) -> impl IntoIterator<Item = Range<i64>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
            self.3.into_range(),
            self.4.into_range(),
            self.5.into_range(),
            self.6.into_range(),
            self.7.into_range(),
        ]
            .into_iter()
    }
}

/// A range of axes that can be used for flattening tensors.
pub trait FlattenAxes {
    /// Get flatten axes
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item = i64>;
}

impl FlattenAxes for RangeFrom<i64> {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item = i64> {
        debug_assert!(
            if self.start > 0 {
                (self.start as usize) < rank
            } else {
                ((-self.start) as usize) <= rank
            },
            "Cannot use {self:?} as flatten axes."
        );
        self.start..i64::MAX
    }
}

impl FlattenAxes for RangeTo<i64> {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item = i64> {
        debug_assert!(
            if self.end > 0 {
                (self.end as usize) < rank
            } else {
                ((-self.end) as usize) <= rank
            },
            "Cannot use {self:?} as flatten axes."
        );
        0..self.end
    }
}

impl FlattenAxes for RangeToInclusive<i64> {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item = i64> {
        debug_assert!(
            if self.end > 0 {
                (self.end as usize) < rank
            } else {
                ((-self.end) as usize) <= rank
            },
            "Cannot use {self:?} as flatten axes."
        );
        0..self.end + 1
    }
}

impl FlattenAxes for RangeFull {
    fn into_flatten_axes(self, rank: usize) -> impl IntoIterator<Item = i64> {
        0..rank as i64
    }
}

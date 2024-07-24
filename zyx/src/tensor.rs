use crate::device::Device;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::shape::{IntoAxes, IntoShape};
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::{
    Add, Div, Mul, Neg, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive, Sub,
};

use crate::runtime::ZyxError;
use crate::RT;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

#[cfg(feature = "rand")]
use rand::rngs::SmallRng;
#[cfg(feature = "rand")]
use rand::Rng;

pub(crate) type TensorId = usize;

pub struct Tensor {
    id: TensorId,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        RT.lock().retain(self.id);
        Tensor { id: self.id }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        //println!("dropping");
        RT.lock().release(self.id).unwrap();
    }
}

impl Tensor {
    /*#[cfg(feature = "debug1")]
    pub fn debug_graph() {
        RT.lock().debug_graph()
    }*/

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
        g.default_device_set_by_user = true;
        g.initialize_device(device)
    }

    #[must_use]
    pub fn backward<'a>(
        &self,
        sources: impl IntoIterator<Item = &'a Tensor>,
    ) -> Vec<Option<Tensor>> {
        let _ = sources;
        todo!()
    }

    /// Immediatelly evaluate passed tensors
    pub fn realize<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) {
        RT.lock()
            .realize(tensors.into_iter().map(|t| t.id).collect())
            .unwrap();
    }

    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        RT.lock().shape(self.id).to_vec()
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        RT.lock().dtype(self.id)
    }

    /// Returns on which devices is this tensor stored or where it will be stored once evaluated
    #[must_use]
    pub fn device(&self) -> Device {
        RT.lock().device(self.id)
    }

    #[must_use]
    pub fn to(self, device: Device) -> Tensor {
        let mut rt = RT.lock();
        let shape = rt.shape(self.id).to_vec();
        return Tensor {
            id: match self.dtype() {
                #[cfg(feature = "half")]
                DType::BF16 => {
                    let data = rt.load::<bf16>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                #[cfg(feature = "half")]
                DType::F16 => {
                    let data = rt.load::<f16>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::F32 => {
                    let data = rt.load::<f32>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::F64 => {
                    let data = rt.load::<f64>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                #[cfg(feature = "complex")]
                DType::CF32 => {
                    let data = rt.load::<Complex<f32>>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                #[cfg(feature = "complex")]
                DType::CF64 => {
                    let data = rt.load::<Complex<f64>>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::U8 => {
                    let data = rt.load::<u8>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::I8 => {
                    let data = rt.load::<i8>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::I16 => {
                    let data = rt.load::<i16>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::I32 => {
                    let data = rt.load::<i32>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
                DType::I64 => {
                    let data = rt.load::<i64>(self.id).unwrap();
                    rt.store(&data, device, shape)
                }
            }
            .unwrap(),
        };
    }

    // Initializers
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn randn(shape: impl IntoShape, dtype: DType) -> Tensor {
        use rand::distributions::Standard;
        use rand::SeedableRng;
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let shape: Vec<usize> = shape.into_shape().collect();
        let n = shape.iter().product();
        let default_device = rt.default_device;
        rt.rng.get_or_init(|| SmallRng::seed_from_u64(crate::SEED));
        let rng = rt.rng.get_mut().unwrap();
        let tensor_id = match dtype {
            #[cfg(feature = "half")]
            DType::BF16 => todo!(),
            #[cfg(feature = "half")]
            DType::F16 => todo!(),
            DType::F32 => {
                let data = &(0..n)
                    .map(move |_| rng.sample(Standard))
                    .collect::<Vec<f32>>();
                rt.store(data, default_device, shape).unwrap()
            }
            DType::F64 => {
                let data = &(0..n)
                    .map(move |_| rng.sample(Standard))
                    .collect::<Vec<f64>>();
                rt.store(data, default_device, shape).unwrap()
            }
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => todo!(),
            DType::I8 => todo!(),
            DType::I16 => todo!(),
            DType::I32 => todo!(),
            DType::I64 => todo!(),
        };
        return Tensor { id: tensor_id };
    }

    #[cfg(feature = "rand")]
    #[must_use]
    pub fn uniform<T: Scalar>(
        shape: impl IntoShape,
        range: impl core::ops::RangeBounds<T>,
    ) -> Tensor {
        let _ = shape;
        let _ = range;
        RT.lock().set_default_device_best();
        todo!()
    }

    #[cfg(feature = "rand")]
    #[must_use]
    pub fn kaiming_uniform<T: Scalar>(
        shape: impl IntoShape,
        range: impl core::ops::RangeBounds<T>,
    ) -> Tensor {
        let _ = shape;
        let _ = range;
        RT.lock().set_default_device_best();
        todo!()
    }

    #[must_use]
    pub fn zeros(shape: impl IntoShape, dtype: DType) -> Tensor {
        let n = shape.clone().into_shape().product();
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let tensor_id = match dtype {
            #[cfg(feature = "half")]
            DType::BF16 => {
                let data: Vec<_> = core::iter::repeat(bf16::ZERO).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            #[cfg(feature = "half")]
            DType::F16 => {
                let data: Vec<_> = core::iter::repeat(f16::ZERO).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::F32 => {
                let data: Vec<_> = core::iter::repeat(0f32).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::F64 => {
                let data: Vec<_> = core::iter::repeat(0f64).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let data: Vec<_> = core::iter::repeat(Complex::<f32>::ZERO).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let data: Vec<_> = core::iter::repeat(Complex::<f32>::ZERO).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::U8 => {
                let data: Vec<_> = core::iter::repeat(0u8).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I8 => {
                let data: Vec<_> = core::iter::repeat(0i8).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I16 => {
                let data: Vec<_> = core::iter::repeat(0i16).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I32 => {
                let data: Vec<_> = core::iter::repeat(0i32).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I64 => {
                let data: Vec<_> = core::iter::repeat(0i64).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
        };
        return Tensor { id: tensor_id };
    }

    #[must_use]
    pub fn ones(shape: impl IntoShape, dtype: DType) -> Tensor {
        let n = shape.clone().into_shape().product();
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let tensor_id = match dtype {
            #[cfg(feature = "half")]
            DType::BF16 => {
                let data: Vec<_> = core::iter::repeat(bf16::ONE).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            #[cfg(feature = "half")]
            DType::F16 => {
                let data: Vec<_> = core::iter::repeat(f16::ONE).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::F32 => {
                let data: Vec<_> = core::iter::repeat(1f32).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::F64 => {
                let data: Vec<_> = core::iter::repeat(1f64).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let data: Vec<_> = core::iter::repeat(Complex::<f32>::ONE).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let data: Vec<_> = core::iter::repeat(Complex::<f32>::ONE).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::U8 => {
                let data: Vec<_> = core::iter::repeat(1u8).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I8 => {
                let data: Vec<_> = core::iter::repeat(1i8).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I16 => {
                let data: Vec<_> = core::iter::repeat(1i16).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I32 => {
                let data: Vec<_> = core::iter::repeat(1i32).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
            DType::I64 => {
                let data: Vec<_> = core::iter::repeat(1i64).take(n).collect();
                rt.store(&data, default_device, shape).unwrap()
            }
        };
        return Tensor { id: tensor_id };
    }

    #[must_use]
    pub fn eye(n: usize, dtype: DType) -> Tensor {
        let _ = n;
        let _ = dtype;
        RT.lock().set_default_device_best();
        todo!()
    }

    #[must_use]
    pub fn full(shape: impl IntoShape, value: impl Scalar) -> Tensor {
        let _ = shape;
        let _ = value;
        RT.lock().set_default_device_best();
        todo!()
    }

    // unary
    #[must_use]
    pub fn abs(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor {
        return Tensor {
            id: RT.lock().cast(self.id, dtype),
        };
    }

    #[must_use]
    pub fn celu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn cos(&self) -> Tensor {
        Self {
            id: RT.lock().cos(self.id),
        }
    }

    #[must_use]
    pub fn dropout(&self, probability: impl Scalar) -> Tensor {
        let _ = probability;
        todo!()
    }

    #[must_use]
    pub fn elu(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn exp(&self) -> Tensor {
        return Tensor {
            id: RT.lock().exp(self.id),
        };
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
        return Tensor {
            id: RT.lock().ln(self.id),
        };
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
        return Tensor {
            id: RT.lock().reciprocal(self.id),
        };
    }

    #[must_use]
    pub fn relu(&self) -> Tensor {
        return Tensor {
            id: RT.lock().relu(self.id),
        };
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
        return Tensor {
            id: RT.lock().sin(self.id),
        };
    }

    #[must_use]
    pub fn softplus(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        return Tensor {
            id: RT.lock().sqrt(self.id),
        };
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
        return Tensor {
            id: RT.lock().tanh(self.id),
        };
    }

    // movement
    #[must_use]
    pub fn expand(&self, shape: impl IntoShape) -> Tensor {
        assert!(shape.rank() > 0);
        let mut sh = self.shape();
        let shape: Vec<usize> = shape.into_shape().collect();
        if shape.rank() > sh.rank() {
            let mut i = sh.len();
            for d in shape.iter().copied().rev() {
                if i == 0 {
                    // Adding dimensions to the front of the shape
                    sh.insert(i, 1);
                } else {
                    i -= 1;
                }
                if d != sh[i] {
                    assert_eq!(
                        sh[i],
                        1,
                        "Cannot expand {:?} into {:?}",
                        self.shape(),
                        shape
                    )
                }
            }
            let x = self.reshape(sh);
            return Tensor {
                id: RT.lock().expand(x.id, shape),
            };
        };
        return Tensor {
            id: RT.lock().expand(self.id, shape),
        };
    }

    #[must_use]
    pub fn permute(&self, axes: impl IntoAxes) -> Tensor {
        assert_eq!(
            self.shape().len(),
            axes.len(),
            "Axes has rank {}, but tensor has rank {}. It must be the same for permute.",
            axes.len(),
            self.rank()
        );
        return Tensor {
            id: RT.lock().permute(self.id, axes),
        };
    }

    #[must_use]
    pub fn pad(&self, padding: impl IntoPadding, value: impl Scalar) -> Tensor {
        let _ = padding;
        let _ = value;
        // TODO add nonzero padding
        //return Tensor { id: RT.lock().pad(self.id, padding) }
        todo!()
    }

    #[must_use]
    pub fn reshape(&self, shape: impl IntoShape) -> Tensor {
        debug_assert_eq!(
            shape.clone().into_shape().product::<usize>(),
            self.numel(),
            "Invalid reshape {:?} into {:?}",
            self.shape(),
            shape
        );
        Tensor {
            id: RT.lock().reshape(self.id, shape),
        }
    }

    #[must_use]
    pub fn transpose(&self) -> Tensor {
        todo!()
    }

    // reduce
    #[must_use]
    pub fn ln_softmax(&self, axes: impl IntoAxes) -> Tensor {
        let m = self - self.max(axes.clone());
        &m - m.exp().sum(axes).ln()
    }

    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor {
        let _ = axes;
        todo!()
    }

    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor {
        let _ = axes;
        todo!()
    }

    #[must_use]
    pub fn norm(&self, axes: impl IntoAxes, p: impl Scalar) -> Tensor {
        self.pow(p.clone()).sum(axes).pow(p.reciprocal())
    }

    #[must_use]
    pub fn product(&self, axes: impl IntoAxes) -> Tensor {
        self.ln().sum(axes).exp()
    }

    #[must_use]
    pub fn std(&self, axes: impl IntoAxes) -> Tensor {
        self.var(axes).sqrt()
    }

    // Removes kernel dimensions for now
    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor {
        #[cfg(debug_assertions)]
        {
            // We can add checks for axes being less than rank and axes not containing duplicates
            let mut unique = alloc::collections::BTreeSet::new();
            let rank = self.rank();
            let axes: Vec<usize> = axes.clone().into_axes(rank).collect();
            for a in &axes {
                debug_assert!(unique.insert(a), "Axes contain duplicates.");
                // This is checked by into_axes function
                //debug_assert!(a < rank, "Axes are too high");
            }
        }
        return Tensor {
            id: RT.lock().sum(self.id, axes),
        };
    }

    #[must_use]
    pub fn softmax(&self, axes: impl IntoAxes) -> Tensor {
        let e = (self - self.max(axes.clone())).exp();
        &e / e.sum(axes)
    }

    #[must_use]
    pub fn var(&self, axes: impl IntoAxes) -> Tensor {
        let _ = axes;
        todo!()
    }

    // index
    #[must_use]
    pub fn get(&self, index: impl IntoIndex) -> Tensor {
        let _ = index;
        todo!()
    }

    #[must_use]
    pub fn diagonal(&self) -> Tensor {
        let n = *self.shape().last().unwrap();
        self.flatten(..)
            .pad([(0, n as i64)], 0)
            .reshape([n, n + 1])
            .get((.., 0))
    }

    // binary
    #[must_use]
    pub fn cmplt(&self, rhs: impl Into<Tensor>) -> Tensor {
        let (x, y) = Tensor::broadcast(self, rhs);
        return Tensor {
            id: RT.lock().cmplt(x.id, y.id),
        };
    }

    #[must_use]
    pub fn dot(&self, rhs: impl Into<Tensor>) -> Tensor {
        let _ = rhs;
        todo!()
    }

    pub fn pow(&self, exponent: impl Into<Tensor>) -> Tensor {
        let (x, y) = Tensor::broadcast(self, exponent);
        return Tensor {
            id: RT.lock().pow(x.id, y.id),
        };
    }

    // ternary
    pub fn where_(&self, if_true: impl Into<Tensor>, if_false: impl Into<Tensor>) -> Tensor {
        let (x, y) = Tensor::broadcast(self, if_true);
        let (x, z) = Tensor::broadcast(x, if_false);
        let (y, z) = Tensor::broadcast(y, z);
        return Tensor {
            id: RT.lock().where_(x.id, y.id, z.id),
        };
    }

    // loss functions
    #[must_use]
    pub fn cross_entropy_loss(&self, target: impl Into<Tensor>, axes: impl IntoAxes) -> Tensor {
        self.ln_softmax(axes) * target
    }

    #[must_use]
    pub fn l1_loss(&self, target: impl Into<Tensor>) -> Tensor {
        (self - target).abs()
    }

    #[must_use]
    pub fn mse_loss(&self, target: impl Into<Tensor>) -> Tensor {
        (self - target).pow(2)
    }

    #[must_use]
    pub fn cosine_similarity(&self, rhs: impl Into<Tensor>, eps: impl Into<Tensor>) -> Tensor {
        let rhs: Tensor = rhs.into();
        let eps: Tensor = eps.into();
        let x = self.pow(2).sqrt() * rhs.pow(2).sqrt();
        self * rhs / x.cmplt(&eps).where_(eps, x)
    }

    // misc
    /// Flatten. Joins axes into one dimension,
    #[must_use]
    pub fn flatten(&self, axes: impl FlattenAxes) -> Tensor {
        let sh = self.shape();
        let n: usize = sh.iter().product();
        let rank = sh.len();
        let mut ld = 1;
        let mut first_dims = false;
        for a in axes.into_flatten_axes(rank) {
            let a = if a > 0 {
                a as usize
            } else {
                (a + rank as i64) as usize
            };
            if a == 0 {
                first_dims = true;
            }
            ld *= sh[a];
        }
        if first_dims {
            self.reshape([ld, n / ld])
        } else {
            self.reshape([n / ld, ld])
        }
    }

    #[must_use]
    pub fn cat<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: i64) -> Tensor {
        let _ = tensors;
        let _ = dim;
        todo!()
    }

    #[must_use]
    pub fn stack<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: i64) -> Tensor {
        let _ = tensors;
        let _ = dim;
        todo!()
    }

    #[must_use]
    pub fn split(&self, sizes: &[usize], dim: i64) -> Vec<Tensor> {
        let _ = sizes;
        let _ = dim;
        todo!()
    }

    #[must_use]
    pub fn pool(&self) -> Tensor {
        todo!()
    }

    #[must_use]
    pub fn conv(&self) -> Tensor {
        todo!()
    }
}

impl Tensor {
    /// Braodcasts to synchronize shapes and casts to synchronize dtypss
    /// This does both automatic expand AND automatic casting between dtypes.
    // TODO Both of these can be disable by changing a setting in the backend.
    #[must_use]
    fn broadcast(x: impl Into<Tensor>, y: impl Into<Tensor>) -> (Tensor, Tensor) {
        let mut x = x.into();
        let mut y = y.into();
        /*assert_eq!(
            graph.dtype(xid),
            graph.dtype(yid),
            "{op} parameters {xid} and {yid} have different dtypes: {} and {}",
            graph.dtype(xid),
            graph.dtype(yid)
        );*/
        // Now we just do implicit conversions. Not exactly rust style, but it's convenient.
        // We can later add option for backend to disable these implicit conversions.
        match (x.dtype(), y.dtype()) {
            (DType::F32, DType::I32) => y = y.cast(DType::F32),
            (DType::F32, DType::F64) => x = x.cast(DType::F64),
            (DType::I32, DType::F32) => x = x.cast(DType::F32),
            (DType::I32, DType::F64) => x = x.cast(DType::F64),
            (DType::F64, DType::F32) => y = y.cast(DType::F64),
            (DType::F64, DType::I32) => y = y.cast(DType::F64),
            _ => {}
        }
        let mut x_shape = x.shape();
        let mut y_shape = y.shape();

        for (x, y) in x_shape.iter().rev().zip(y_shape.iter().rev()) {
            if x != y {
                debug_assert!(
                    *x == 1 || *y == 1,
                    "Left and right tensor shapes can not be broadcasted: {x_shape:?} and {y_shape:?}"
                );
            }
        }

        let rx = x_shape.rank();
        let ry = y_shape.rank();
        match rx.cmp(&ry) {
            Ordering::Less => {
                x_shape = core::iter::repeat(1)
                    .take(ry - rx)
                    .chain(x_shape.into_iter())
                    .collect();
            }
            Ordering::Greater => {
                y_shape = core::iter::repeat(1)
                    .take(rx - ry)
                    .chain(y_shape.into_iter())
                    .collect();
            }
            Ordering::Equal => {}
        }
        let mut eshape = Vec::new();
        for (x, y) in x_shape.iter().zip(y_shape.iter()) {
            eshape.push(*x.max(y));
        }
        if x_shape != eshape {
            x = x.expand(&*eshape);
        }
        if y_shape != eshape {
            y = y.expand(eshape);
        }
        return (x, y);
    }
}

#[cfg(feature = "half")]
impl TryFrom<&Tensor> for bf16 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

#[cfg(feature = "half")]
impl TryFrom<&Tensor> for f16 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for f32 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for f64 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

#[cfg(feature = "complex")]
impl TryFrom<&Tensor> for Complex<f32> {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

#[cfg(feature = "complex")]
impl TryFrom<&Tensor> for Complex<f64> {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for u8 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for i8 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for i16 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for i32 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<&Tensor> for i64 {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl<T: Scalar> TryFrom<&Tensor> for Vec<T> {
    type Error = ZyxError;
    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        RT.lock().load(value.id)
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
            #[cfg(feature = "half")]
            DType::F16 => {
                let data: Result<Vec<f16>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => alloc::format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F32 => {
                let data: Result<Vec<f32>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => alloc::format!("f32 tensor failed to realize {e:?}"),
                }
            }
            DType::F64 => {
                let data: Result<Vec<f64>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => alloc::format!("f64 tensor failed to realize {e:?}"),
                }
            }
            DType::I32 => {
                let data: Result<Vec<i32>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => alloc::format!("i32 tensor failed to realize {e:?}"),
                }
            }
            _ => todo!(),
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
    let n: usize = shape.iter().product();
    let rank = shape.len();
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
    let d0 = shape[rank - 1];
    for (i, x) in data.iter().enumerate() {
        {
            let mut var = 1;
            let mut r = rank;
            while r > 0 {
                if i % (n / var) == 0 {
                    res += &(" ".repeat(rank - r) + "[".repeat(r - 1).as_str());
                    break;
                }
                var *= shape[rank - r];
                r -= 1;
            }
        }
        let _ = write!(res, "{x:>w$.precision$}");
        if (i + 1) % d0 != 0usize {
            res += "  ";
        }
        {
            let mut var = 1;
            let mut r = rank;
            while r > 0 {
                if (i + 1) % (n / var) == 0 {
                    res += &"]".repeat(r - 1);
                    break;
                }
                var *= shape[rank - r];
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

pub trait IntoPadding {
    fn into_padding(self) -> Vec<(i64, i64)>;
}

impl<const N: usize> IntoPadding for [(i64, i64); N] {
    fn into_padding(self) -> Vec<(i64, i64)> {
        self.into()
    }
}

impl From<&Tensor> for Tensor {
    fn from(value: &Tensor) -> Self {
        value.clone()
    }
}

impl<T: Scalar> From<T> for Tensor {
    fn from(value: T) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let id = rt.store(&[value], default_device, 1).unwrap();
        return Tensor { id };
    }
}

impl<T: Scalar> From<Vec<T>> for Tensor {
    fn from(data: Vec<T>) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let id = rt.store(&data, default_device, data.len()).unwrap();
        return Tensor { id };
    }
}

impl<T: Scalar> From<&[T]> for Tensor {
    fn from(data: &[T]) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let id = rt.store(data, default_device, data.len()).unwrap();
        return Tensor { id };
    }
}

impl<T: Scalar, const D0: usize> From<[T; D0]> for Tensor {
    fn from(data: [T; D0]) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let id = rt.store(&data, default_device, D0).unwrap();
        return Tensor { id };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> From<[[T; D1]; D0]> for Tensor {
    fn from(data: [[T; D1]; D0]) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let data = unsafe { core::slice::from_raw_parts(data[0].as_ptr(), D0 * D1) };
        let id = rt.store(data, default_device, [D0, D1]).unwrap();
        return Tensor { id };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> From<[[[T; D2]; D1]; D0]>
    for Tensor
{
    fn from(data: [[[T; D2]; D1]; D0]) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let data = unsafe { core::slice::from_raw_parts(data[0][0].as_ptr(), D0 * D1 * D2) };
        let id = rt.store(data, default_device, [D0, D1, D2]).unwrap();
        return Tensor { id };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    From<[[[[T; D3]; D2]; D1]; D0]> for Tensor
{
    fn from(data: [[[[T; D3]; D2]; D1]; D0]) -> Self {
        let mut rt = RT.lock();
        rt.set_default_device_best();
        let default_device = rt.default_device;
        let data =
            unsafe { core::slice::from_raw_parts(data[0][0][0].as_ptr(), D0 * D1 * D2 * D3) };
        let id = rt.store(data, default_device, [D0, D1, D2, D3]).unwrap();
        return Tensor { id };
    }
}

impl<IT: Into<Tensor>> Add<IT> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().add(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Add<IT> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().add(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Sub<IT> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().sub(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Sub<IT> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().sub(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Mul<IT> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().mul(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Mul<IT> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().mul(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Div<IT> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().div(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Div<IT> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs);
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().div(x.id, y.id),
        };
        return tensor;
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor {
            id: RT.lock().neg(self.id),
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor {
            id: RT.lock().neg(self.id),
        }
    }
}

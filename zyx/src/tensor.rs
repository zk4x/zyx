use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::shape::{to_axis, IntoAxes, IntoPadding, IntoShape};
use core::cmp::Ordering;
use core::ops::{
    Add, Div, Mul, Neg, Not, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive, Sub,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display};
use std::iter::repeat;

use crate::runtime::ZyxError;
use crate::RT;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

pub(crate) type TensorId = usize;

#[cfg_attr(feature = "py", pyo3::pyclass)]
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
        //std::println!("dropping");
        RT.lock().release(self.id).unwrap();
    }
}

impl Tensor {
    /// Shape of tensor
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        RT.lock().shape(self.id).to_vec()
    }

    /// Number of scalar elements stored in self
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Rank of self. Rank means number of dimensions/axes.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Datatype of self. See [DType](crate::DType) for available datatypes.
    #[must_use]
    pub fn dtype(&self) -> DType {
        RT.lock().dtype(self.id)
    }

    /// Is zyx in training mode?
    #[must_use]
    pub fn training() -> bool {
        RT.lock().training
    }

    /// Set training mode
    pub fn set_training(training: bool) {
        RT.lock().training = training;
    }

    /// Immediatelly evaluate passed tensors
    pub fn realize<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> Result<(), ZyxError> {
        RT.lock()
            .realize(tensors.into_iter().map(|t| t.id).collect())
    }

    /// Returns gradients of self derived w.r.t. sources
    #[must_use]
    pub fn backward<'a>(
        &self,
        sources: impl IntoIterator<Item = &'a Tensor>,
    ) -> Vec<Option<Tensor>> {
        let sources: Vec<TensorId> = sources.into_iter().map(|t| t.id).collect();
        let grads: BTreeMap<TensorId, TensorId> = RT
            .lock()
            .backward(self.id, sources.iter().copied().collect());
        sources
            .into_iter()
            .map(|x: TensorId| grads.get(&x).copied())
            .map(|id: Option<TensorId>| id.map(|id| Tensor { id }))
            .collect()
    }

    /// Detaches tensor from graph.
    /// This function returns a new tensor with the same data as the previous one,
    /// but drops it's backpropagation graph. This is usefull for recurrent networks:
    /// ```rust
    /// let mut x = Tensor::randn([8, 8]);
    /// let z = Tensor::randn([8]);
    /// for _ in 0..100 {
    ///     // Without detach the graph would grow bigger with every iteration
    ///     x = x.detach() + z;
    /// }
    /// ```
    pub fn detach(self) -> Tensor {
        // TODO remove realization from here
        let shape = self.shape();
        let dtype = self.dtype();
        match dtype {
            #[cfg(feature = "half")]
            DType::F16 => {
                let data: Vec<f16> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            #[cfg(feature = "half")]
            DType::BF16 => {
                let data: Vec<bf16> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::F32 => {
                let data: Vec<f32> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::F64 => {
                let data: Vec<f64> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let data: Vec<Complex<f32>> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let data: Vec<Complex<f64>> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::U8 => {
                let data: Vec<u8> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::I8 => {
                let data: Vec<i8> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::I16 => {
                let data: Vec<i16> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::I32 => {
                let data: Vec<i32> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::I64 => {
                let data: Vec<i64> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
            DType::Bool => {
                let data: Vec<bool> = (&self).try_into().unwrap();
                Tensor::from(data).reshape(shape)
            }
        }
    }

    /// Create debug guard at the beginning of the block to debug that block.
    /// Once the guard is dropped, debug gets reset to global state,
    /// the one set by ZYX_DEBUG env variable.
    /// ZYX_DEBUG is bitmask
    /// 0000 0001 DEBUG_DEV
    /// 0000 0010 DEBUG_PERF
    /// 0000 0100 DEBUG_SCHED
    /// 0000 1000 DEBUG_IR
    /// 0001 0000 DEBUG_ASM
    #[must_use]
    pub fn debug_guard(debug: u32) -> DebugGuard {
        let mut rt = RT.lock();
        let guard = DebugGuard {
            debug: rt.debug,
        };
        rt.debug = debug;
        guard
    }

    /// Write graph of operations between tensors as png image with given filename
    /// Expects dot program to be in the path. Otherwise create dot graph file
    /// without converting it to png.
    pub fn plot_graph<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, name: &str) {
        use std::format;
        let graph = RT
            .lock()
            .plot_dot_graph(&tensors.into_iter().map(|t| t.id).collect());
        std::fs::write(format!("{name}.dot"), graph).unwrap();
        let output = std::process::Command::new("dot")
            .arg("-Tpng")
            .arg(format!("{name}.dot"))
            .arg("-o")
            .arg(format!("{name}.png"))
            .output();
        if let Err(err) = output {
            std::println!("Graph png could not be created: {err}");
        } else {
            let _ = std::fs::remove_file(format!("{name}.dot"));
        }
    }

    #[cfg(feature = "rand")]
    pub fn manual_seed(seed: u64) {
        RT.lock().manual_seed(seed);
    }

    // Initializers
    /// Create tensor sampled from standard distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn randn(shape: impl IntoShape, dtype: DType) -> Tensor {
        // TODO just use threefry
        // This can be generated from uniform or just generate on cpu
        // and pass into device whole buffer
        match dtype {
            #[cfg(feature = "half")]
            DType::BF16 => todo!(),
            #[cfg(feature = "half")]
            DType::F16 => todo!(),
            DType::F32 => {
                Tensor::uniform(shape.clone(), -1f32..1f32) / Tensor::uniform(shape, -1f32..1f32)
            }
            DType::F64 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => {
                Tensor::uniform(shape.clone(), 0u8..255u8) / Tensor::uniform(shape, 0u8..255u8)
            }
            DType::I8 => todo!(),
            DType::I16 => todo!(),
            DType::I32 => todo!(),
            DType::I64 => todo!(),
            DType::Bool => todo!(),
        }
    }

    /// Create tensor sampled from uniform distribution
    /// Start of the range must be less than the end of the range.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn uniform<T: Scalar>(
        shape: impl IntoShape,
        range: impl core::ops::RangeBounds<T>,
    ) -> Tensor {
        use core::ops::Bound;
        let start = match range.start_bound() {
            Bound::Included(value) => *value,
            Bound::Excluded(value) => *value,
            Bound::Unbounded => T::min_value(),
        };
        let end = match range.end_bound() {
            Bound::Included(value) => *value,
            Bound::Excluded(value) => *value,
            Bound::Unbounded => T::max_value(),
        };
        Tensor {
            id: RT
                .lock()
                .uniform(shape.into_shape().collect(), start, end)
                .unwrap(),
        }
    }

    /// Create tensor sampled from kaiming uniform distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn kaiming_uniform<T: Scalar>(shape: impl IntoShape, a: T) -> Tensor {
        let n = T::from_i64(shape.clone().into_shape().skip(1).product::<usize>() as i64);
        // bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
        let one = T::one();
        let x = Scalar::add(one, Scalar::mul(a, a));
        let two = Scalar::add(one, one);
        let three = Scalar::add(two, one);
        let x = Scalar::div(two, x).sqrt();
        let bound = Scalar::mul(three.sqrt(), Scalar::div(x, n));
        return Tensor::uniform(shape, bound.neg()..bound);
    }

    /// Create tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: impl IntoShape, dtype: DType) -> Tensor {
        return Tensor {
            id: RT.lock().zeros(shape.into_shape().collect(), dtype),
        };
    }

    /// Create tensor filled with ones.
    #[must_use]
    pub fn ones(shape: impl IntoShape, dtype: DType) -> Tensor {
        return Tensor {
            id: RT.lock().ones(shape.into_shape().collect(), dtype),
        };
    }

    /// Create tensor filled with value.
    #[must_use]
    pub fn full(shape: impl IntoShape, value: impl Scalar) -> Tensor {
        return Tensor {
            id: RT.lock().full(shape.into_shape().collect(), value),
        };
    }

    /// Create square tensor with ones on the main diagonal and all other values set to zero.
    #[must_use]
    pub fn eye(n: usize, dtype: DType) -> Tensor {
        return Tensor::ones(vec![n, 1], dtype)
            .pad_zeros([(0, n as isize)])
            .reshape([n + 1, n])
            .get((..-1, ..));
    }

    // unary
    /// Computes the absolute value of each element in self.
    #[must_use]
    pub fn abs(&self) -> Tensor {
        self.relu() + (-self).relu()
    }

    /// Casts self to [dtype](crate::DType).
    #[must_use]
    pub fn cast(&self, dtype: DType) -> Tensor {
        return Tensor {
            id: RT.lock().cast(self.id, dtype),
        };
    }

    /// Applies element-wise, CELU(x)=max⁡(0,x)+min⁡(0,α∗(exp⁡(x/α)−1)).
    #[must_use]
    pub fn celu(&self, alpha: impl Scalar) -> Tensor {
        return self.relu() - (-((self / alpha).exp() - 1) * alpha).relu();
    }

    /// Returns a new tensor with the cosine of the elements of self.
    #[must_use]
    pub fn cos(&self) -> Tensor {
        Self {
            id: RT.lock().cos(self.id),
        }
    }

    #[must_use]
    pub fn cosh(&self) -> Tensor {
        // (e^x + e^-x) / 2
        let nx = self.neg();
        let enx = nx.exp();
        let ex = self.exp();
        (ex + enx)/2
    }

    /// During training, randomly zeroes some of the elements of the input tensor with probability.
    /// The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.
    /// Each channel will be zeroed out independently on every forward call.
    /// Furthermore, the outputs are scaled by a factor of 11−p1−p1​ during training.
    /// This means that during evaluation the module simply computes an identity function.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn dropout(&self, probability: impl Scalar) -> Tensor {
        Tensor::from(probability)
            .cmplt(Tensor::uniform(self.shape(), 0f32..1.0))
            .cast(self.dtype())
            * self
    }

    #[must_use]
    pub fn elu(&self, alpha: impl Scalar) -> Tensor {
        self.relu() - (Tensor::ones(1, self.dtype()) - self.exp()).relu() * alpha
    }

    #[must_use]
    pub fn exp2(&self) -> Tensor {
        return Tensor {
            id: RT.lock().exp2(self.id),
        };
    }

    #[must_use]
    pub fn exp(&self) -> Tensor {
        let c: Tensor = std::f64::consts::E.log2().try_into().unwrap();
        (self*c.cast(self.dtype())).exp2()
    }

    #[must_use]
    pub fn gelu(&self) -> Tensor {
        self * 0.5f32
            * (((self + self.pow(3f32) * 0.044_715f32) * (2f32 / core::f32::consts::PI).sqrt())
                .tanh()
                + 1f32)
    }

    #[must_use]
    pub fn leaky_relu(&self, neg_slope: impl Scalar) -> Tensor {
        self.relu() - (self * (-Tensor::from(neg_slope))).relu()
    }

    #[must_use]
    pub fn log2(&self) -> Tensor {
        return Tensor {
            id: RT.lock().log2(self.id),
        };
    }

    #[must_use]
    pub fn ln(&self) -> Tensor {
        let c: Tensor = (1f64/std::f64::consts::E.log2()).try_into().unwrap();
        self.log2()*c.cast(self.dtype())
    }

    #[must_use]
    pub fn inv(&self) -> Tensor {
        return Tensor {
            id: RT.lock().inv(self.id),
        };
    }

    #[must_use]
    pub fn mish(&self) -> Tensor {
        self * self.softplus(1, 20).tanh()
    }

    #[must_use]
    pub fn quick_gelu(&self) -> Tensor {
        self * (1.702f32 * self).sigmoid()
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
        self.reciprocal().sqrt()
    }

    #[must_use]
    pub fn selu(&self) -> Tensor {
        1.0507009873554804934193349852946f32
            * (self.relu()
                - (1.6732632423543772848170429916717f32
                    * (Tensor::ones(1, self.dtype()) - self.exp()))
                .relu())
    }

    #[must_use]
    pub fn sigmoid(&self) -> Tensor {
        let one = Tensor::ones(1, self.dtype());
        let exp_x = self.exp();
        return &exp_x / (&one + &exp_x);
    }

    #[must_use]
    pub fn sin(&self) -> Tensor {
        return Tensor {
            id: RT.lock().sin(self.id),
        };
    }

    #[must_use]
    pub fn sinh(&self) -> Tensor {
        // (e^x - e^-x) / 2
        let nx = self.neg();
        let enx = nx.exp();
        let ex = self.exp();
        (ex - enx)/2
    }

    #[must_use]
    pub fn softplus(&self, beta: impl Scalar, threshold: impl Scalar) -> Tensor {
        let x = self * beta;
        x.cmplt(threshold)
            .where_(((x).exp() + 1).ln() * beta.reciprocal(), x)
    }

    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        return Tensor {
            id: RT.lock().sqrt(self.id),
        };
    }

    #[must_use]
    pub fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }

    #[must_use]
    pub fn tan(&self) -> Tensor {
        self.sin() / self.cos()
    }

    #[must_use]
    pub fn tanh(&self) -> Tensor {
        let e2x = (self + self).exp();
        (&e2x + 1)/(e2x - 1)
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
        let rank = self.rank();
        let axes: Vec<usize> = axes.into_axes(rank).collect();
        assert_eq!(
            rank,
            axes.len(),
            "Axes has rank {}, but tensor has rank {}. It must be the same for permute.",
            axes.len(),
            rank
        );
        return Tensor {
            id: RT.lock().permute(self.id, axes),
        };
    }

    pub fn pad_zeros(&self, padding: impl IntoPadding) -> Tensor {
        return Tensor {
            id: RT.lock().pad_zeros(self.id, padding.into_padding()),
        };
    }

    /// Constant padding
    ///
    /// This can both add and remove values from tensor. Negative padding removes values, positive padding
    /// adds values.
    ///
    /// Pad last dimension by (1, 2)
    /// ```rust
    /// use zyx::Tensor;
    /// let x = Tensor::from([[2, 3],
    ///                       [4, 1]]);
    /// let z = x.pad([(1, 2)], 0);
    /// std::println!("{}", z);
    /// assert_eq!(z, [[0, 2, 3, 0, 0],
    ///                [0, 4, 1, 0, 0]]);
    /// ```
    /// Pad last dimension by (2, -1) and second last dimension by (1, 1)
    /// ```rust
    /// # use zyx::Tensor;
    /// # let x = Tensor::from([[2, 3],
    /// #                       [4, 1]]);
    /// let z = x.pad([(2, -1), (1, 1)], 0);
    /// println!("z: {z}");
    /// assert_eq!(z, [[0, 0, 0],
    ///                [0, 0, 2],
    ///                [0, 0, 4],
    ///                [0, 0, 0]]);
    /// ```
    ///
    /// # Panics
    /// T must be of the same dtype as Tensor's dtype, otherwise this function panics.
    #[must_use]
    pub fn pad(&self, padding: impl IntoPadding, value: impl Into<Tensor>) -> Tensor {
        let dtype = self.dtype();
        let value: Tensor = value.into();
        assert_eq!(
            value.dtype(),
            dtype,
            "Cannot pad tensor with dtype {} with value of dtype {}",
            dtype,
            value.dtype()
        );
        let padding = padding.into_padding();
        let sh = self.shape();
        assert!(
            padding.len() <= sh.rank()
                && padding
                    .iter()
                    .zip(sh.iter().rev())
                    .all(|((lp, rp), d)| if *lp < 0 {
                        ((-*lp) as usize) <= *d
                    } else {
                        true
                    } && if *rp < 0 {
                        ((-*rp) as usize) <= *d
                    } else {
                        true
                    }),
            "Cannot pad tensor with shape {sh:?} with padding {padding:?}"
        );
        let t0 = self.pad_zeros(padding.clone());
        if value.numel() == 1
            && match dtype {
                #[cfg(feature = "half")]
                DType::BF16 => {
                    let x: bf16 = value.clone().try_into().unwrap();
                    x == bf16::ZERO
                }
                #[cfg(feature = "half")]
                DType::F16 => {
                    let x: f16 = value.clone().try_into().unwrap();
                    x == f16::ZERO
                }
                DType::F32 => {
                    let x: f32 = value.clone().try_into().unwrap();
                    x == 0.
                }
                DType::F64 => {
                    let x: f64 = value.clone().try_into().unwrap();
                    x == 0.
                }
                #[cfg(feature = "complex")]
                DType::CF32 => {
                    let x: Complex<f32> = value.clone().try_into().unwrap();
                    x == Complex::new(0., 0.)
                }
                #[cfg(feature = "complex")]
                DType::CF64 => {
                    let x: Complex<f64> = value.clone().try_into().unwrap();
                    x == Complex::new(0., 0.)
                }
                DType::U8 => {
                    let x: u8 = value.clone().try_into().unwrap();
                    x == 0
                }
                DType::I8 => {
                    let x: i8 = value.clone().try_into().unwrap();
                    x == 0
                }
                DType::I16 => {
                    let x: i16 = value.clone().try_into().unwrap();
                    x == 0
                }
                DType::I32 => {
                    let x: i32 = value.clone().try_into().unwrap();
                    x == 0
                }
                DType::I64 => {
                    let x: i64 = value.clone().try_into().unwrap();
                    x == 0
                }
                DType::Bool => {
                    let x: bool = value.clone().try_into().unwrap();
                    x == false
                }
            }
        {
            t0
        } else {
            let ones = Tensor::ones(sh.clone(), dtype);
            let zeros = Tensor::zeros(sh, self.dtype());
            t0 + ones.pad_zeros(padding).where_(zeros, value)
        }
    }

    #[must_use]
    pub fn reshape(&self, shape: impl IntoShape) -> Tensor {
        let shape: Vec<usize> = shape.into_shape().collect();
        assert_eq!(
            shape.iter().product::<usize>(),
            self.numel(),
            "Invalid reshape {:?} into {:?}",
            self.shape(),
            shape
        );
        Tensor {
            id: RT.lock().reshape(self.id, shape),
        }
    }

    /// Transpose last two dimensions of this tensor.
    /// If self.rank() == 1, returns tensor with shape `[self.shape()[0], 1]` (column tensor)
    #[must_use]
    pub fn t(&self) -> Tensor {
        let mut rank = self.rank();
        let x = if rank == 1 {
            let n = self.numel();
            rank = 2;
            self.reshape([1, n])
        } else {
            self.clone()
        };
        let mut axes: Vec<isize> = (0..rank as isize).collect();
        axes.swap(rank - 1, rank - 2);
        x.permute(axes)
    }

    /// Transpose two arbitrary dimensions
    #[must_use]
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Tensor {
        let rank = self.rank();
        let mut axes: Vec<isize> = (0..rank as isize).collect();
        axes.swap(to_axis(dim0, rank), to_axis(dim1, rank));
        self.permute(axes)
    }

    // reduce
    #[must_use]
    pub fn ln_softmax(&self, axes: impl IntoAxes) -> Tensor {
        let m = self - self.max_kd(axes.clone());
        &m - m.exp().sum_kd(axes).ln()
    }

    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor {
        let rank = self.rank();
        let axes: Vec<usize> = axes.into_axes(rank).collect();
        {
            let mut unique = BTreeSet::new();
            for a in &axes {
                assert!(unique.insert(a), "Axes contain duplicates.");
            }
        }
        return Tensor {
            id: RT.lock().max_reduce(self.id, axes),
        };
    }

    #[must_use]
    pub fn max_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.max(axes.clone()).reshape(self.reduce_kd_shape(axes))
    }

    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor {
        let shape = self.shape();
        self.sum(axes.clone())
            / axes
                .into_axes(shape.rank())
                .map(|a| shape[a])
                .product::<usize>() as i64
    }

    #[must_use]
    pub fn mean_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.mean(axes.clone()).reshape(self.reduce_kd_shape(axes))
    }

    #[must_use]
    pub fn product(&self, axes: impl IntoAxes) -> Tensor {
        self.ln().sum(axes).exp()
    }

    #[must_use]
    pub fn std(&self, axes: impl IntoAxes) -> Tensor {
        self.var(axes).sqrt()
    }

    #[must_use]
    pub fn std_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.std(axes.clone()).reshape(self.reduce_kd_shape(axes))
    }

    /// Sum reduce. Removes tensor dimensions.
    /// Equivalent to pytorch sum(axes, keepdim=False)
    /// If you want to keep reduce dimensions, see [sum_kd](Tensor::sum_kd)
    /// Passing empty axes executes reduce across all dimensions and result will have shape [1]
    #[must_use]
    pub fn sum(&self, axes: impl IntoAxes) -> Tensor {
        let rank = self.rank();
        let axes: Vec<usize> = axes.into_axes(rank).collect();
        {
            // We can add checks for axes being less than rank and axes not containing duplicates
            let mut unique = BTreeSet::new();
            for a in &axes {
                assert!(unique.insert(a), "Axes contain duplicates.");
                // This is checked by into_axes function
                //assert!(a < rank, "Axes are too high");
            }
        }
        return Tensor {
            id: RT.lock().sum_reduce(self.id, axes),
        };
    }

    // Probably just have sum_kd, max_kd that keep tensor dimensions
    /// Like [sum](Tensor::sum) but keeps reduce dimensions, setting them to 1.
    /// Equivalent to pytorch sum(axes, keepdim=True)
    #[must_use]
    pub fn sum_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.sum(axes.clone()).reshape(self.reduce_kd_shape(axes))
    }

    #[must_use]
    pub fn cumsum(&self, axis: isize) -> Tensor {
        let _ = axis;
        //let axis = to_axis(axis, self.rank());
        //let pl_sz = (self.shape()[axis] - 1) as isize;
        //let axis = axis as isize;
        //return self.transpose(axis,-1).pad_zeros([(pl_sz, 0)]).pool(self.shape[axis]).sum(-1).transpose(axis,-1)
        todo!()
    }

    #[must_use]
    pub fn softmax(&self, axes: impl IntoAxes) -> Tensor {
        let e = (self - self.max_kd(axes.clone())).exp();
        &e / e.sum_kd(axes)
    }

    #[must_use]
    pub fn var(&self, axes: impl IntoAxes) -> Tensor {
        (self - self.mean(axes.clone())).pow(2).sum(axes)
    }

    #[must_use]
    pub fn var_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.var(axes.clone()).reshape(self.reduce_kd_shape(axes))
    }

    // index
    #[must_use]
    pub fn get(&self, index: impl IntoIndex) -> Tensor {
        let shape = self.shape();
        let padding: Vec<(isize, isize)> = index
            .into_index()
            .into_iter()
            .zip(shape.iter())
            .map(|(r, d)| {
                (
                    if r.start >= 0 {
                        -r.start
                    } else {
                        -r.start - *d as isize
                    },
                    if r.end == isize::MAX {
                        0
                    } else if r.end > 0 {
                        -(*d as isize - r.end)
                    } else {
                        r.end
                    },
                )
            })
            .collect();
        let n = shape.rank() - padding.len();
        let padding: Vec<(isize, isize)> = padding
            .into_iter()
            .chain(core::iter::repeat((0, 0)).take(n))
            .collect::<Vec<(isize, isize)>>()
            .into_iter()
            .rev()
            .collect();
        //std::println!("Get padding: {padding:?}");
        self.pad_zeros(padding)
    }

    #[must_use]
    pub fn diagonal(&self) -> Tensor {
        let n = *self.shape().last().unwrap();
        self.flatten(..)
            .pad_zeros([(0, n as isize)])
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
    pub fn maximum(&self, rhs: impl Into<Tensor>) -> Tensor {
        let (x, y) = Tensor::broadcast(self, rhs);
        return Tensor {
            id: RT.lock().maximum(x.id, y.id),
        };
    }

    #[must_use]
    pub fn dot(&self, rhs: impl Into<Tensor>) -> Tensor {
        let rhs = rhs.into();
        let org_y_shape = rhs.shape();
        let y = rhs.t();
        let xshape = self.shape();
        let yshape = y.shape();
        let xrank = xshape.rank();
        let yrank = yshape.rank();
        assert_eq!(
            xshape[xrank - 1],
            yshape[yrank - 1],
            //yshape[-(yrank.min(2) as i64)],
            "Cannot dot tensors with shapes {xshape:?} and {org_y_shape:?}",
        );
        let x_shape = xshape[..xrank - 1]
            .iter()
            .copied()
            .chain([1])
            .chain([xshape[xrank - 1]])
            .collect::<Vec<usize>>();
        let y_shape = yshape[0..yrank - 2]
            .iter()
            .copied()
            .chain([1])
            .chain(yshape[yrank - yrank.min(2)..yrank].iter().copied())
            .collect::<Vec<usize>>();
        //std::println!("{x_shape:?}");
        //std::println!("{y_shape:?}");
        (self.reshape(x_shape) * y.reshape(y_shape))
            .sum(-1)
            .reshape(
                xshape[0..xshape.len() - 1]
                    .iter()
                    .copied()
                    .chain([yshape[yshape.len() - 2]])
                    .collect::<Vec<usize>>(),
            )
    }

    #[must_use]
    pub fn pow(&self, exponent: impl Into<Tensor>) -> Tensor {
        let (x, y) = Tensor::broadcast(self, exponent);
        return Tensor {
            id: RT.lock().pow(x.id, y.id),
        };
    }

    /// Returns ones where self is true and zeros where it is false.
    #[must_use]
    pub fn nonzero(&self) -> Tensor {
        return Tensor {
            id: RT.lock().nonzero(self.id),
        };
    }

    // ternary
    #[must_use]
    pub fn where_(&self, if_true: impl Into<Tensor>, if_false: impl Into<Tensor>) -> Tensor {
        let (x, y) = Tensor::broadcast(self, if_true);
        let (x, z) = Tensor::broadcast(x, if_false);
        let (y, z) = Tensor::broadcast(y, z);
        let x_nonzero = x.nonzero();
        return &x_nonzero * y + !x_nonzero * z;
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
    pub fn cat<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: isize) -> Tensor {
        let tensors: Vec<&Tensor> = tensors.into_iter().collect();
        let shape = tensors[0].shape();
        let rank = shape.rank();
        let dim = if dim < 0 { dim + rank as isize } else { dim } as usize;
        // Dimension check
        for tensor in &tensors {
            for (i, (d1, d2)) in shape.iter().zip(tensor.shape().iter()).enumerate() {
                if i != dim {
                    assert_eq!(*d1, *d2, "Cannot concatenate these tensors.");
                }
            }
        }
        let mut offset = 0isize;
        let mut res = Tensor::zeros(tensors[0].shape(), tensors[0].dtype());
        for tensor in tensors {
            res = res
                + tensor.pad_zeros(
                    core::iter::repeat((0isize, 0isize))
                        .take(rank - dim - 1)
                        .chain([(offset, 0isize)]),
                );
            offset += tensor.shape()[dim] as isize;
        }
        res
    }

    #[must_use]
    pub fn stack<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: isize) -> Tensor {
        let _ = tensors;
        let _ = dim;
        todo!()
    }

    #[must_use]
    pub fn split(&self, sizes: &[usize], dim: isize) -> Vec<Tensor> {
        let _ = sizes;
        let _ = dim;
        todo!()
    }

    #[must_use]
    pub fn pool(
        &self,
        kernel_size: impl IntoShape,
        stride: impl IntoShape,
        dilation: impl IntoShape,
    ) -> Tensor {
        let k_: Vec<usize> = kernel_size.into_shape().collect();
        let stride: Vec<usize> = stride.into_shape().collect();
        let dilation: Vec<usize> = dilation.into_shape().collect();

        let shape = self.shape();
        let rank = shape.len();

        let s_: Vec<usize> = if stride.len() == 1 {
            repeat(stride[0]).take(k_.len()).collect()
        } else {
            stride
        };
        let d_: Vec<usize> = if dilation.len() == 1 {
            repeat(dilation[0]).take(k_.len()).collect()
        } else {
            dilation
        };
        // noop_ = [None] * len(self.shape[:-len(k_)])
        let noop_: Vec<Option<()>> = repeat(None).take(rank - k_.len()).collect();
        let i_ = &shape[rank - k_.len()..];
        let o_ = i_
            .iter()
            .cloned()
            .zip(d_.iter().cloned())
            .zip(k_.iter().cloned())
            .zip(s_.iter().cloned())
            .map(|(((i, d), k), s)| (i - d * (k - 1)).div_ceil(s));

        let repeats: Vec<usize> = repeat(1)
            .take(rank - k_.len())
            .chain(
                k_.iter()
                    .copied()
                    .zip(i_.iter().copied())
                    .zip(d_.iter().copied())
                    .map(|((k, i), d)| k * (i + d).div_ceil(i)),
            )
            .collect();
        let xup = self.repeat(repeats);
        // dilation
        let padding: Vec<(isize, isize)> = k_.iter().copied().zip(i_.iter().copied()).zip(d_.iter().copied()).map(|((k, i), d)| (0, -((k*(i+d)) as isize))).collect();
        let xup = xup.pad_zeros(padding);

        //tuple(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])

        //let xup = xup.shrink(tuple(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)]))
            //.reshape(noop_ + flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
        // stride
        //xup = xup.shrink(
        //tuple(noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_)))).reshape(noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
        //xup = xup.shrink(tuple(noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_)))).reshape(noop_ + flatten((k,o) for k,o in zip(k_, o_)))

        /*if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
        # repeats such that we don't need padding
        xup = self.repeat([1]*len(noop_) + [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)])
        # handle dilation
        xup = xup.shrink(tuple(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])).reshape(noop_ + flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
        # handle stride
        xup = xup.shrink(
            tuple(noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_)))).reshape(noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
        xup = xup.shrink(tuple(noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_)))).reshape(noop_ + flatten((k,o) for k,o in zip(k_, o_)))
        # permute to move reduce to the end
        return xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])*/

        //xup = self.pad(tuple(noop_ + [(0, max(0,o*s-i)) for i,o,s in zip(i_, o_, s_)])).shrink(tuple(noop_ + [(0,o*s) for o,s in zip(o_, s_)]))
        //xup = xup.reshape(noop_ + flatten(((o,s) for o,s in zip(o_, s_))))
        //xup = xup.shrink(tuple(noop_ + flatten(((0,o), (0,k)) for o,k in zip(o_, k_))))
        //return xup.permute(*range(len(noop_)), *[len(noop_)+i*2 for i in range(len(i_))], *[len(noop_)+i*2+1 for i in range(len(i_))])
        todo!()
    }

    #[must_use]
    pub fn repeat(&self, repeats: impl IntoShape) -> Tensor {
        let repeats: Vec<usize> = repeats.into_shape().collect();
        let shape = self.shape();
        let rank = shape.len();

        let base_shape: Vec<usize> = repeat(1)
            .take(repeats.len() - rank)
            .chain(shape.iter().copied())
            .collect();
        let new_shape: Vec<usize> = repeat(1)
            .take(repeats.len() - rank)
            .chain(shape.into_iter())
            .flat_map(|d| [1, d])
            .collect();
        let expand_shape: Vec<usize> = repeats
            .iter()
            .copied()
            .zip(base_shape.iter().copied())
            .flat_map(|(r, d)| [r, d])
            .collect();
        let final_shape: Vec<usize> = repeats
            .iter()
            .copied()
            .zip(base_shape.iter().copied())
            .map(|(r, d)| r * d)
            .collect();

        return self
            .reshape(new_shape)
            .expand(expand_shape)
            .reshape(final_shape);
    }

    #[must_use]
    pub fn conv(&self) -> Tensor {
        todo!()
    }
}

pub struct DebugGuard {
    debug: u32,
}

impl Drop for DebugGuard {
    fn drop(&mut self) {
        RT.lock().debug = self.debug;
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
                assert!(
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

    // Calculate shape for reduce which keeps reduced dims set to 1
    fn reduce_kd_shape(&self, axes: impl IntoAxes) -> Vec<usize> {
        let mut shape = self.shape();
        for a in axes.clone().into_axes(shape.len()) {
            shape[a] = 1;
        }
        shape
    }
}

#[cfg(feature = "half")]
impl TryFrom<Tensor> for bf16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

#[cfg(feature = "half")]
impl TryFrom<Tensor> for f16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for f32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for f64 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

#[cfg(feature = "complex")]
impl TryFrom<Tensor> for Complex<f32> {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

#[cfg(feature = "complex")]
impl TryFrom<Tensor> for Complex<f64> {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for u8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for i8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for i16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for i32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for i64 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        RT.lock()
            .load(value.id)?
            .first()
            .copied()
            .ok_or(ZyxError::EmptyTensor)
    }
}

impl TryFrom<Tensor> for bool {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
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

impl Debug for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
        //f.write_fmt(format_args!("Tensor {{ id = {:?} }}", self.id))
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO don't print the whole tensor if it is too big
        let precision = if let Some(precision) = f.precision() {
            precision
        } else {
            3
        };
        let res = match self.dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => {
                let data: Result<Vec<bf16>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            #[cfg(feature = "half")]
            DType::F16 => {
                let data: Result<Vec<f16>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F32 => {
                let data: Result<Vec<f32>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f32 tensor failed to realize {e:?}"),
                }
            }
            DType::F64 => {
                let data: Result<Vec<f64>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f64 tensor failed to realize {e:?}"),
                }
            }
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let data: Result<Vec<Complex<f32>>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f32 tensor failed to realize {e:?}"),
                }
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let data: Result<Vec<Complex<f64>>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f64 tensor failed to realize {e:?}"),
                }
            }
            DType::U8 => {
                let data: Result<Vec<u8>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I8 => {
                let data: Result<Vec<i8>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I16 => {
                let data: Result<Vec<i16>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I32 => {
                let data: Result<Vec<i32>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I64 => {
                let data: Result<Vec<i64>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::Bool => {
                let data: Result<Vec<bool>, _> = self.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
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
) -> String {
    use core::fmt::Write;
    let n: usize = shape.iter().product();
    let rank = shape.len();
    let mut res = String::new();
    if data.is_empty() {
        return "[]".into();
    }
    // get maximal width of single value
    let mut w = 0;
    if let Some(width) = width {
        w = width;
    } else {
        for x in data {
            let l = format!("{x:>.precision$}").len();
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
    fn into_range(self) -> Range<isize>;
}

impl IntoRange for RangeFull {
    fn into_range(self) -> Range<isize> {
        0..isize::MAX
    }
}

impl IntoRange for RangeFrom<isize> {
    fn into_range(self) -> Range<isize> {
        self.start..isize::MAX
    }
}

impl IntoRange for RangeTo<isize> {
    fn into_range(self) -> Range<isize> {
        0..self.end
    }
}

impl IntoRange for RangeInclusive<isize> {
    fn into_range(self) -> Range<isize> {
        *self.start()..*self.end() + 1
    }
}

impl IntoRange for RangeToInclusive<isize> {
    fn into_range(self) -> Range<isize> {
        0..self.end + 1
    }
}

impl IntoRange for Range<isize> {
    fn into_range(self) -> Range<isize> {
        self
    }
}

impl IntoRange for isize {
    fn into_range(self) -> Range<isize> {
        self..self + 1
    }
}

/// Implemented for objects that can be used to index tensors.
pub trait IntoIndex {
    /// Convert self to tensor index.
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>>;
}

impl<I: IntoRange> IntoIndex for &[I] {
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
        self.iter().cloned().map(IntoRange::into_range)
    }
}

impl<I0: IntoRange> IntoIndex for I0 {
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
        [self.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange> IntoIndex for (I0, I1) {
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
        [self.0.into_range(), self.1.into_range()].into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange> IntoIndex for (I0, I1, I2) {
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
        [
            self.0.into_range(),
            self.1.into_range(),
            self.2.into_range(),
        ]
        .into_iter()
    }
}

impl<I0: IntoRange, I1: IntoRange, I2: IntoRange, I3: IntoRange> IntoIndex for (I0, I1, I2, I3) {
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
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
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
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
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
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
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
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
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
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
        assert!(
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
        assert!(
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
        assert!(
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

impl From<&Tensor> for Tensor {
    fn from(value: &Tensor) -> Self {
        value.clone()
    }
}

impl<T: Scalar> From<T> for Tensor {
    fn from(value: T) -> Self {
        return Tensor {
            id: RT.lock().temp(vec![1], &[value]).unwrap(),
        };
    }
}

impl<T: Scalar> From<Vec<T>> for Tensor {
    fn from(data: Vec<T>) -> Self {
        return Tensor {
            id: RT.lock().temp(vec![data.len()], &data).unwrap(),
        };
    }
}

impl<T: Scalar> From<&[T]> for Tensor {
    fn from(data: &[T]) -> Self {
        let n = data.len();
        return Tensor {
            id: RT.lock().temp(vec![n], data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize> From<[T; D0]> for Tensor {
    fn from(data: [T; D0]) -> Self {
        return Tensor {
            id: RT.lock().temp(vec![D0], &data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> From<[[T; D1]; D0]> for Tensor {
    fn from(data: [[T; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0].as_ptr(), D0 * D1) };
        return Tensor {
            id: RT.lock().temp(vec![D0, D1], data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> From<[[[T; D2]; D1]; D0]>
    for Tensor
{
    fn from(data: [[[T; D2]; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0][0].as_ptr(), D0 * D1 * D2) };
        return Tensor {
            id: RT.lock().temp(vec![D0, D1, D2], data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    From<[[[[T; D3]; D2]; D1]; D0]> for Tensor
{
    fn from(data: [[[[T; D3]; D2]; D1]; D0]) -> Self {
        let data =
            unsafe { core::slice::from_raw_parts(data[0][0][0].as_ptr(), D0 * D1 * D2 * D3) };
        return Tensor {
            id: RT.lock().temp(vec![D0, D1, D2, D3], data).unwrap(),
        };
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

impl Not for Tensor {
    type Output = Tensor;
    fn not(self) -> Self::Output {
        Tensor {
            id: RT.lock().not(self.id),
        }
    }
}

impl Not for &Tensor {
    type Output = Tensor;
    fn not(self) -> Self::Output {
        Tensor {
            id: RT.lock().not(self.id),
        }
    }
}

macro_rules! impl_trait {
    ($trait:ident for $type:ty, $fn_name:ident) => {
        impl $trait<Tensor> for $type {
            type Output = Tensor;
            fn $fn_name(self, rhs: Tensor) -> Self::Output {
                rhs * self
            }
        }

        impl $trait<&Tensor> for $type {
            type Output = Tensor;
            fn $fn_name(self, rhs: &Tensor) -> Self::Output {
                rhs * self
            }
        }
    };
}

#[cfg(feature = "half")]
impl_trait!(Add for bf16, add);
#[cfg(feature = "half")]
impl_trait!(Add for f16, add);
impl_trait!(Add for f32, add);
impl_trait!(Add for f64, add);
#[cfg(feature = "complex")]
impl_trait!(Add for Complex<f32>, add);
#[cfg(feature = "complex")]
impl_trait!(Add for Complex<f64>, add);
impl_trait!(Add for u8, add);
impl_trait!(Add for i8, add);
impl_trait!(Add for i16, add);
impl_trait!(Add for i32, add);
impl_trait!(Add for i64, add);
impl_trait!(Add for bool, add);

#[cfg(feature = "half")]
impl_trait!(Sub for bf16, sub);
#[cfg(feature = "half")]
impl_trait!(Sub for f16, sub);
impl_trait!(Sub for f32, sub);
impl_trait!(Sub for f64, sub);
#[cfg(feature = "complex")]
impl_trait!(Sub for Complex<f32>, sub);
#[cfg(feature = "complex")]
impl_trait!(Sub for Complex<f64>, sub);
impl_trait!(Sub for u8, sub);
impl_trait!(Sub for i8, sub);
impl_trait!(Sub for i16, sub);
impl_trait!(Sub for i32, sub);
impl_trait!(Sub for i64, sub);
impl_trait!(Sub for bool, sub);

#[cfg(feature = "half")]
impl_trait!(Mul for bf16, mul);
#[cfg(feature = "half")]
impl_trait!(Mul for f16, mul);
impl_trait!(Mul for f32, mul);
impl_trait!(Mul for f64, mul);
#[cfg(feature = "complex")]
impl_trait!(Mul for Complex<f32>, mul);
#[cfg(feature = "complex")]
impl_trait!(Mul for Complex<f64>, mul);
impl_trait!(Mul for u8, mul);
impl_trait!(Mul for i8, mul);
impl_trait!(Mul for i16, mul);
impl_trait!(Mul for i32, mul);
impl_trait!(Mul for i64, mul);
impl_trait!(Mul for bool, mul);

#[cfg(feature = "half")]
impl_trait!(Div for bf16, div);
#[cfg(feature = "half")]
impl_trait!(Div for f16, div);
impl_trait!(Div for f32, div);
impl_trait!(Div for f64, div);
#[cfg(feature = "complex")]
impl_trait!(Div for Complex<f32>, div);
#[cfg(feature = "complex")]
impl_trait!(Div for Complex<f64>, div);
impl_trait!(Div for u8, div);
impl_trait!(Div for i8, div);
impl_trait!(Div for i16, div);
impl_trait!(Div for i32, div);
impl_trait!(Div for i64, div);
impl_trait!(Div for bool, div);

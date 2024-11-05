//! Tensor
//!
//! Tensors are at the core of all machine learning.

#![allow(clippy::fallible_impl_from)]

use crate::dtype::DType;
use crate::runtime::ZyxError;
use crate::scalar::{Float, Scalar};
use crate::shape::{into_axes, into_axis, IntoShape};
use core::cmp::Ordering;
use float8::F8E4M3 as f8;
use half::{bf16, f16};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display};
use std::iter::{once, repeat};
use std::ops::{
    Add, BitAnd, BitOr, BitXor, Bound, Div, Mul, Neg, Not, Range, RangeBounds, RangeFrom,
    RangeFull, RangeInclusive, RangeTo, RangeToInclusive, Sub,
};
use std::path::Path;

use crate::RT;

pub type TensorId = u32;

/// A tensor represents a multi-dimensional array of values. This is the primary data structure in the library.
/// The `Tensor` struct contains an internal identifier (`id`) that uniquely identifies each tensor.
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

    /// Datatype of self. See [`DType`](crate::DType) for available datatypes.
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
    /// # Errors
    /// Returns device error if the device fails to realize one or more tensors.
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
            .backward(self.id, &sources.iter().copied().collect());
        sources
            .into_iter()
            .map(|x: TensorId| grads.get(&x).copied())
            .map(|id: Option<TensorId>| id.map(|id| Tensor { id }))
            .collect()
    }

    /// Detaches tensor from graph.
    /// This function returns a new tensor with the same data as the previous one,
    /// but drops it's backpropagation graph. This is usefull for recurrent networks:
    /// ```rust no_run
    /// use zyx::{Tensor, DType};
    /// let mut x = Tensor::randn([8, 8], DType::F32)?;
    /// let z = Tensor::randn([8], DType::F32)?;
    /// for _ in 0..100 {
    ///     // Without detach the graph would grow bigger with every iteration
    ///     x = x.detach()? + &z;
    /// }
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    /// # Errors
    /// If function needs to realize tensor, it may return device error if the device
    /// fails to realize self.
    pub fn detach(self) -> Result<Tensor, ZyxError> {
        // TODO remove realization from here
        let shape = self.shape();
        match self.dtype() {
            DType::BF16 => {
                let data: Vec<bf16> = self.try_into()?;
                Tensor::from(data)
            }
            DType::F8 => {
                let data: Vec<f8> = self.try_into()?;
                Tensor::from(data)
            }
            DType::F16 => {
                let data: Vec<f16> = self.try_into()?;
                Tensor::from(data)
            }
            DType::F32 => {
                let data: Vec<f32> = self.try_into()?;
                Tensor::from(data)
            }
            DType::F64 => {
                let data: Vec<f64> = self.try_into()?;
                Tensor::from(data)
            }
            DType::U8 => {
                let data: Vec<u8> = self.try_into()?;
                Tensor::from(data)
            }
            DType::U32 => {
                let data: Vec<u32> = self.try_into()?;
                Tensor::from(data)
            }
            DType::I8 => {
                let data: Vec<i8> = self.try_into()?;
                Tensor::from(data)
            }
            DType::I16 => {
                let data: Vec<i16> = self.try_into()?;
                Tensor::from(data)
            }
            DType::I32 => {
                let data: Vec<i32> = self.try_into()?;
                Tensor::from(data)
            }
            DType::I64 => {
                let data: Vec<i64> = self.try_into()?;
                Tensor::from(data)
            }
            DType::Bool => {
                let data: Vec<bool> = self.try_into()?;
                Tensor::from(data)
            }
        }
        .reshape(shape)
    }

    /// Create debug guard at the beginning of the block to debug that block.
    /// Once the guard is dropped, debug gets reset to global state,
    /// the one set `by ZYX_DEBUG` env variable.
    /// For more look at `ENV_VARS.md`
    #[must_use]
    pub fn debug_guard(debug: u32) -> DebugGuard {
        let mut rt = RT.lock();
        let guard = DebugGuard { debug: rt.debug };
        rt.debug = debug;
        guard
    }

    /// Write graph of operations between tensors as png image with given filename
    /// Expects dot program to be in the path. Otherwise create dot graph file
    /// without converting it to png.
    /// # Errors
    /// Returns error if graph image failed to write to disk.
    pub fn plot_graph<'a>(
        tensors: impl IntoIterator<Item = &'a Tensor>,
        name: &str,
    ) -> Result<(), std::io::Error> {
        use std::format;
        let graph = RT
            .lock()
            .plot_dot_graph(&tensors.into_iter().map(|t| t.id).collect());
        std::fs::write(format!("{name}.dot"), graph)?;
        let output = std::process::Command::new("dot")
            .arg("-Tpng")
            .arg(format!("{name}.dot"))
            .arg("-o")
            .arg(format!("{name}.png"))
            .output();
        if let Err(err) = output {
            println!("Graph png could not be created: {err}");
        } else {
            let _ = std::fs::remove_file(format!("{name}.dot"));
        }
        Ok(())
    }

    /// Manually sets the seed for the random number generator.
    /// This function is only available if the `rand` feature is enabled.
    pub fn manual_seed(seed: u64) {
        RT.lock().manual_seed(seed);
    }

    /// Create random value in range 0f..1f with float dtype
    /// or 0..`{integer}::MAX` if it is integer
    /// # Errors
    /// Returns device error if the device fails to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc, reason = "all panics are checked ahead")]
    pub fn rand(shape: impl IntoShape, dtype: DType) -> Result<Tensor, ZyxError> {
        const SEED: u64 = 69420;
        use rand::distributions::Uniform;
        use rand::rngs::SmallRng;
        use rand::Rng;
        use rand::SeedableRng;
        let shape: Vec<usize> = shape.into_shape().collect();
        let n = shape.iter().product();
        if dtype.is_float() {
            // TODO later use threefry
            let mut rt = RT.lock();
            rt.rng.get_or_init(|| SmallRng::seed_from_u64(SEED));
            let Some(rng) = rt.rng.get_mut() else {
                panic!()
            };
            match dtype {
                DType::BF16 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f32> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    }
                    .cast(DType::BF16))
                }
                DType::F8 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f32> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    }
                    .cast(DType::F8))
                }
                DType::F16 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f32> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    }
                    .cast(DType::F16))
                }
                DType::F32 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f32> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::F64 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f64> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::U8
                | DType::U32
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::Bool => panic!(),
            }
        } else {
            let mut rt = RT.lock();
            rt.rng.get_or_init(|| SmallRng::seed_from_u64(SEED));
            let Some(rng) = rt.rng.get_mut() else {
                panic!()
            };
            match dtype {
                DType::U8 => {
                    let range = Uniform::new(0, u8::MAX);
                    let data: Vec<u8> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::U32 => {
                    let range = Uniform::new(0, u32::MAX);
                    let data: Vec<u32> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I8 => {
                    let range = Uniform::new(0, i8::MAX);
                    let data: Vec<i8> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I16 => {
                    let range = Uniform::new(0, i16::MAX);
                    let data: Vec<i16> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I32 => {
                    let range = Uniform::new(0, i32::MAX);
                    let data: Vec<i32> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I64 => {
                    let range = Uniform::new(0, i64::MAX);
                    let data: Vec<i64> = (0..n).map(|_| rng.sample(range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::Bool => Err(ZyxError::DTypeError(
                    "Uniform is not supported for bool".into(),
                )),
                DType::BF16 | DType::F8 | DType::F16 | DType::F32 | DType::F64 => panic!(),
            }
        }
        /*# threefry
        if (num := math.ceil(((num_ := prod(shape)) * dtype.itemsize) / 4)) == 0: return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)
        if not had_counter: Tensor._rng_counter.assign(Tensor._rng_counter + num)
        counts1 = (Tensor.arange(math.ceil(num / 2), device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._rng_counter.to(device))
        counts2 = counts1 + math.ceil(num / 2)*/

        /*# threefry random bits
        x = counts2.cast(dtypes.uint64) << 32 | counts1.cast(dtypes.uint64)
        x = F.Threefry.apply(*x._broadcasted(Tensor._seed))
        counts1, counts2 = (x & 0xffffffff).cast(dtypes.uint32), ((x >> 32) & 0xffffffff).cast(dtypes.uint32)
        bits = counts1.cat(counts2)[:num]

        # bitcast to uint with same number of bits
        _, nmant = dtypes.finfo(dtype)
        uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dtype.itemsize]
        bits = bits.bitcast(uint_dtype)
        # only randomize the mantissa bits and set the exponent to 1
        one = Tensor.ones_like(bits, device=bits.device, dtype=dtype).bitcast(uint_dtype)
        bits = bits.rshift((dtype.itemsize * 8) - nmant).bitwise_or(one)

        # bitcast back to the original dtype
        out = bits.bitcast(dtype)[:num_].sub(1).reshape(shape)
        out.requires_grad = kwargs.get("requires_grad")
        return out.contiguous()*/
    }

    // Initializers
    /// Create tensor sampled from standard distribution.
    /// # Errors
    /// Retuns device error if device fails to allocate memory for given tensor.
    pub fn randn(shape: impl IntoShape, dtype: DType) -> Result<Tensor, ZyxError> {
        // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        let shape: Vec<usize> = once(2).chain(shape.into_shape()).collect();
        let src = Tensor::rand(shape, dtype)?;
        let mut x = src.get(0)?;
        x = x.mul(Tensor::constant(2f32 * std::f32::consts::PI));
        //panic!();
        x = x.cos();
        let mut y = Tensor::constant(1f32) - src.get(1)?;
        //println!("{y} minus");
        y = y.ln().mul(Tensor::constant(-2f32)).sqrt();
        //println!("{y}");
        Ok(x.mul(y).cast(dtype))
    }

    /// Multinomial function
    /// # Errors
    /// Returns device error if the device fails to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc, reason = "TODO disallow panicking")]
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> Result<Tensor, ZyxError> {
        let sh = self.shape();
        let rank = sh.len();
        assert!(
            (1..=2).contains(&rank) && num_samples > 0,
            "rank={rank} must be 1 or 2"
        );
        assert!(
            replacement || num_samples == 1,
            "no replacement only supports num_samples = 1"
        );
        let weight = if rank == 1 {
            self.unsqueeze(0)?
        } else {
            self.clone()
        };
        let cw = weight.cumsum(1)?.float_cast();
        let cdf = &cw / cw.get((.., -1))?.unsqueeze(1)?;
        let cdf_sh = cdf.shape();
        let unif_samples = Tensor::rand([num_samples, cdf_sh[0], 1], DType::F32)?;
        let indices = unif_samples
            .expand([num_samples, cdf_sh[0], cdf_sh[1]])?
            .cmplt(cdf)?
            .not()
            .sum([2])?
            .permute([1, 0])?;
        Ok((if rank == 1 {
            indices.squeeze(0)?
        } else {
            indices
        })
        .cast(DType::I32))
    }

    /// Create tensor sampled from uniform distribution
    /// Start of the range must be less than the end of the range.
    /// # Errors
    /// Returns device error if the device fails to allocate memory for tensor.
    pub fn uniform<T: Scalar>(
        shape: impl IntoShape,
        range: impl core::ops::RangeBounds<T>,
    ) -> Result<Tensor, ZyxError> {
        use core::ops::Bound;
        let low = match range.start_bound() {
            Bound::Included(value) | Bound::Excluded(value) => *value,
            Bound::Unbounded => T::min_value(),
        };
        let high = match range.end_bound() {
            Bound::Included(value) | Bound::Excluded(value) => *value,
            Bound::Unbounded => T::max_value(),
        };
        Ok(Tensor::rand(shape, T::dtype())? * high.sub(low) + low)
    }

    /// Create tensor sampled from kaiming uniform distribution.
    /// # Errors
    /// Returns device error if the device fails to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc)]
    pub fn kaiming_uniform<T: Float>(shape: impl IntoShape, a: T) -> Result<Tensor, ZyxError> {
        let n = T::from_i64(
            shape
                .clone()
                .into_shape()
                .skip(1)
                .product::<usize>()
                .try_into()
                .unwrap(),
        );
        let one = T::one();
        let x = Scalar::add(one, Scalar::mul(a, a));
        let two = Scalar::add(one, one);
        let three = Scalar::add(two, one);
        let x = Scalar::div(two, x).sqrt();
        let bound = Scalar::mul(three.sqrt(), Scalar::div(x, n));
        Tensor::uniform(shape, bound.neg()..bound)
    }

    /// Create tensor sampled from glorot uniform distribution.
    /// # Errors
    /// Returns device error if the device fails to allocate memory for tensor.
    #[allow(clippy::cast_precision_loss)]
    pub fn glorot_uniform(shape: impl IntoShape, dtype: DType) -> Result<Tensor, ZyxError> {
        let shape: Vec<usize> = shape.into_shape().collect();
        let c = 6. / (shape[0] + shape.iter().skip(1).product::<usize>()) as f32;
        let mut x = Tensor::uniform(shape, -1f32..1f32)?;
        x = x * c.pow(0.5);
        Ok(x.cast(dtype))
    }

    /// Create tensor filled with zeros.
    #[must_use]
    pub fn zeros(shape: impl IntoShape, dtype: DType) -> Tensor {
        Tensor {
            id: RT.lock().zeros(shape.into_shape().collect(), dtype),
        }
    }

    /// Create tensor filled with ones.
    #[must_use]
    pub fn ones(shape: impl IntoShape, dtype: DType) -> Tensor {
        Tensor {
            id: RT.lock().ones(shape.into_shape().collect(), dtype),
        }
    }

    /// Create tensor filled with value.
    /// # Errors
    /// Returns device error if the device failed to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc)]
    pub fn full(shape: impl IntoShape, value: impl Scalar) -> Result<Tensor, ZyxError> {
        Ok(Tensor {
            id: RT.lock().full(shape.into_shape().collect(), value)?,
        })
    }

    /// Create square tensor with ones on the main diagonal and all other values set to zero.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn eye(n: usize, dtype: DType) -> Tensor {
        Tensor::ones(vec![n, 1], dtype)
            .pad_zeros([(0, isize::try_from(n).unwrap())])
            .unwrap()
            .reshape([n + 1, n])
            .unwrap()
            .get((..-1, ..))
            .unwrap()
    }

    /// Arange method, create range from start, stop, step
    /// # Errors
    /// Returns device error if the device failed to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc)]
    pub fn arange<T: Scalar>(start: T, stop: T, step: T) -> Result<Tensor, ZyxError> {
        // if (stop-start)/step <= 0: return Tensor([], dtype=dtype, **kwargs)
        // return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs)._cumsum() + (start - step)).cast(dtype)
        //println!("Arange {start:?}, {stop:?}, {step:?}");
        let n: i64 = stop.sub(start).div(step).cast();
        //println!("Shape {n}");
        let x = Tensor::full(usize::try_from(n).unwrap(), step)?;
        //println!("{x}");
        let x = x.cumsum(0)?;
        Ok(x + start - step)
    }

    /// Create constant that will be baked into compiled kernels.
    /// Using different value in graph in place of this constnat will force
    /// recompilation of one or more kernels.
    /// For performance reason use this if the value does not
    /// change during the run of the program or if there are only few repeating variations.
    #[must_use]
    pub fn constant(value: impl Scalar) -> Tensor {
        Tensor {
            id: RT.lock().constant(value),
        }
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

    /// Changes dtype of the tensor without mutating it.
    /// Currently this function will also realize the tensor (if it is not already realized)
    ///
    /// # Safety
    /// Not all bits of one type can be safely reinterpreted as bits of other type,
    /// therefore this function is marked as unsafe.
    ///
    /// # Errors
    /// Returns device error if the device failed to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc)]
    pub unsafe fn bitcast(&self, dtype: DType) -> Result<Tensor, ZyxError> {
        let id = RT.lock().bitcast(self.id, dtype)?;
        let x = Tensor { id };
        Ok(x)
    }

    /// Applies element-wise, CELU(x)=max⁡(0,x)+min⁡(0,α∗(exp⁡(x/α)−1)).
    #[must_use]
    pub fn celu(&self, alpha: impl Scalar) -> Tensor {
        self.relu() - (-((self / alpha).exp() - 1) * alpha).relu()
    }

    /// Returns a new tensor with the cosine of the elements of self.
    #[must_use]
    pub fn cos(&self) -> Tensor {
        let x = self.float_cast();
        let x = Tensor {
            id: RT.lock().cos(x.id),
        };
        x
    }

    /// `cosh(x) = (exp(x) + exp(-x)) / 2`.
    #[must_use]
    pub fn cosh(&self) -> Tensor {
        // (e^x + e^-x) / 2
        let nx = self.neg();
        let enx = nx.exp();
        let ex = self.exp();
        (ex + enx) / 2
    }

    /// Applies dropout to the tensor with a given probability.
    ///
    /// This function randomly sets elements of the input tensor to zero based on the provided probability.
    /// The output tensor has the same shape as the input tensor. Elements are preserved with probability `1 - probability`
    /// and set to zero with probability `probability`.
    ///
    /// # Errors
    /// Returns device error if the device failed to allocate memory for tensor.
    #[allow(clippy::missing_panics_doc)]
    pub fn dropout<P: Scalar + Float>(&self, probability: P) -> Result<Tensor, ZyxError> {
        // TODO fix this for training (dropout in training is just scaling)
        Ok(
            Tensor::from(probability).cmplt(Tensor::rand(self.shape(), P::dtype())?)?
                * self.clone(),
        )
    }

    /// Applies the Exponential Linear Unit function element-wise.
    ///
    /// The ELU function is defined as:
    /// ```text
    /// f(x) = x if x > 0
    ///       α(e^x - 1) otherwise
    /// ```
    /// where `α` is a given scaling factor. This function helps mitigate the "dying `ReLU`" problem.
    #[must_use]
    pub fn elu(&self, alpha: impl Scalar) -> Tensor {
        self.relu() - (Tensor::ones(1, self.dtype()) - self.exp()).relu() * alpha
    }

    /// Returns a new tensor with the exponential of 2 raised to the power of each element in self.
    #[must_use]
    pub fn exp2(&self) -> Tensor {
        let x = self.float_cast();
        let x = Tensor {
            id: RT.lock().exp2(x.id),
        };
        x
    }

    /// Computes the exponential of each element in the input tensor using base e.
    ///
    /// This function returns a new tensor that is computed by taking the exponential of each
    /// element in the input tensor. The output will have the same shape as the input tensor,
    /// and its elements will be calculated as `e^input_element`.
    ///
    /// @param self The input tensor.
    /// @return A new tensor with the same shape as the input, but with each element computed
    ///         as `e^input_element`.
    #[must_use]
    pub fn exp(&self) -> Tensor {
        let c: Tensor = Tensor::constant(std::f64::consts::E.log2());
        (self * c.cast(self.dtype())).exp2()
    }

    /// Returns a new tensor with the Gelu activation function applied to each element of self.
    ///
    /// The Gelu activation function is defined as:
    /// `gelu(x) = x * 0.5 * (1 + tanh(sqrt(2 / π) * (x + x^3 * 0.044715)))`.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn gelu(&self) -> Tensor {
        self * 0.5f32
            * (((self + self.pow(3f32).unwrap() * 0.044_715f32)
                * (2f32 / core::f32::consts::PI).sqrt())
            .tanh()
                + 1f32)
    }

    /// Applies the Leaky `ReLU` activation function element-wise.
    ///
    /// This function computes the Leaky `ReLU` of each element in the input tensor. If the element is greater than
    /// or equal to zero, it returns the element itself; otherwise, it returns `neg_slope * element`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    /// * `neg_slope`: The negative slope coefficient (`α` in the formula) for the Leaky `ReLU` function.
    ///
    /// **Returns:**
    ///
    /// A new tensor with the same shape as the input, but with each element computed as `max(0., x) + neg_slope * min(0., x)`.
    #[must_use]
    pub fn leaky_relu(&self, neg_slope: impl Scalar) -> Tensor {
        self.relu() - (self * (-Tensor::from(neg_slope))).relu()
    }

    /// Computes the base-2 logarithm of each element in the input tensor.
    ///
    /// This function returns a new tensor that is computed by taking the base-2 logarithm of each
    /// element in the input tensor. The output will have the same shape as the input tensor,
    /// and its elements will be calculated as `log2(input_element)`.
    ///
    /// @param self The input tensor.
    /// @return A new tensor with the same shape as the input, but with each element computed
    ///         as `log2(input_element)`.
    #[must_use]
    pub fn log2(&self) -> Tensor {
        let x = self.float_cast();
        return Tensor {
            id: RT.lock().log2(x.id),
        };
    }

    /// Computes the natural logarithm (ln) of each element in the input tensor.
    ///
    /// This function returns a new tensor that is computed by taking the natural logarithm of each
    /// element in the input tensor. The output will have the same shape as the input tensor,
    /// and its elements will be calculated as `ln(input_element)`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:**
    ///
    /// A new tensor with the same shape as the input, but with each element computed as `ln(input_element)`.
    #[must_use]
    pub fn ln(&self) -> Tensor {
        let x = self.float_cast();
        let c: Tensor = Tensor::constant(1f64 / std::f64::consts::E.log2());
        x.log2() * c.cast(x.dtype())
    }

    /// Computes the multiplicative inverse of each element in the input tensor.
    ///
    /// This function returns a new tensor with the same shape as the input, where each element is the multiplicative inverse (i.e., reciprocal) of the corresponding element in the input tensor.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is the multiplicative inverse (reciprocal) of the corresponding element in the input tensor.
    #[must_use]
    pub fn inv(&self) -> Tensor {
        return Tensor {
            id: RT.lock().inv(self.id),
        };
    }

    /// Computes the Mish activation function for each element in the input tensor.
    ///
    /// The Mish activation function is a continuous, non-monotonic function that behaves like `ReLU` for positive inputs and like sigmoid for negative inputs. It is defined as `x * tanh(softplus(x))`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `Mish(input_element)`.
    #[must_use]
    pub fn mish(&self) -> Tensor {
        self * self.softplus(1., 20.).tanh()
    }

    /// Computes the quick GELU activation function for each element in the input tensor.
    ///
    /// The `QuickGELU` activation function is an approximation of the Gaussian Error Linear Unit (GELU) function that uses a sigmoid function to compute the approximation. It is defined as `x * sigmoid(1.702 * x)`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `QuickGELU(input_element)`.
    #[must_use]
    pub fn quick_gelu(&self) -> Tensor {
        self * (1.702f32 * self).sigmoid()
    }

    /// Computes the multiplicative inverse of each element in the input tensor using a faster implementation.
    ///
    /// This function returns a new tensor with the same shape as the input, where each element is the multiplicative inverse (i.e., reciprocal) of the corresponding element in the input tensor. This implementation uses `1.0 / self` which is generally faster than calling the `inv()` method directly.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is the multiplicative inverse (reciprocal) of the corresponding element in the input tensor using a faster implementation.
    #[must_use]
    pub fn reciprocal(&self) -> Tensor {
        return Tensor {
            id: RT.lock().reciprocal(self.id),
        };
    }

    /// Applies the Rectified Linear Unit (`ReLU`) activation function to each element in the input tensor.
    ///
    /// The `ReLU` function returns `max(0, x)`, i.e., it replaces negative values with zero and leaves positive values unchanged. This makes it a popular choice for use in hidden layers of neural networks due to its simplicity and effectiveness.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `max(0, input_element)`.
    #[must_use]
    pub fn relu(&self) -> Tensor {
        return Tensor {
            id: RT.lock().relu(self.id),
        };
    }

    /// Computes the reciprocal square root of each element in the input tensor.
    ///
    /// This function returns a new tensor with the same shape as the input, where each element is the reciprocal square root (i.e., `1 / sqrt(x)`) of the corresponding element in the input tensor. This operation can be useful for scaling and stabilizing certain types of computations.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is the reciprocal square root (i.e., `1 / sqrt(x)`) of the corresponding element in the input tensor.
    #[must_use]
    pub fn rsqrt(&self) -> Tensor {
        self.reciprocal().sqrt()
    }

    /// Applies the Self-Normalized Linear Unit (Selu) activation function to each element in the input tensor.
    ///
    /// The Selu activation function is designed to maintain the mean and variance of the activations approximately constant when training deep neural networks with residual connections. It combines the benefits of both `ReLU` and sigmoid functions, making it a good choice for certain types of problems.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `Selu(input_element)`.
    #[must_use]
    pub fn selu(&self) -> Tensor {
        let dtype = self.dtype();
        (1.050_700_987_355_480_5f64
            * (self.relu()
                - (1.673_263_242_354_377_3f64 * (Tensor::ones(1, dtype) - self.exp())).relu()))
        .cast(dtype)
    }

    /// Applies the sigmoid activation function to each element in the input tensor.
    ///
    /// The sigmoid function returns `1 / (1 + exp(-x))`, i.e., it maps any real-valued input onto a value between 0 and 1. This function is commonly used for binary classification problems or as an activation function in neural networks.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `sigmoid(input_element)`.
    #[must_use]
    pub fn sigmoid(&self) -> Tensor {
        let one = Tensor::ones(1, self.dtype());
        let exp_x = self.exp();
        exp_x.clone() / (one + exp_x)
    }

    /// Applies the sine function to each element in the input tensor.
    ///
    /// This function returns a new tensor with the same shape as the input, where each element is the sine of the corresponding element in the input tensor. The sine function is useful for various mathematical and scientific computations involving angles or periodic phenomena.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is the sine of the corresponding element in the input tensor.
    #[must_use]
    pub fn sin(&self) -> Tensor {
        let x = self.float_cast();
        let x = Tensor {
            id: RT.lock().sin(x.id),
        };
        x
    }

    /// Applies the hyperbolic sine function to each element in the input tensor.
    ///
    /// The hyperbolic sine function returns `(e^x - e^-x) / 2`, i.e., it maps any real-valued input onto a value that grows exponentially. This function is useful for computations involving exponential growth or decay, such as in physics and engineering applications.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `sinh(input_element)`.
    #[must_use]
    pub fn sinh(&self) -> Tensor {
        // (e^x - e^-x) / 2
        let nx = self.neg();
        let enx = nx.exp();
        let ex = self.exp();
        (ex - enx) / 2
    }

    /// Applies the softplus function to each element in the input tensor with a given beta and threshold.
    ///
    /// The softplus function returns `log(exp(x) + 1)` for inputs greater than the threshold, and x otherwise. This function is useful for bounding outputs between zero and infinity when applying the `ReLU` function.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    /// * beta: A scalar multiplier applied to each element of the input tensor before comparison with the threshold.
    /// * threshold: The threshold value below which the input is returned unchanged, and above which the softplus function is applied.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is computed according to the softplus function with the given beta and threshold.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn softplus(&self, beta: impl Float, threshold: impl Float) -> Tensor {
        let x = self * beta;
        x.cmplt(threshold)
            .unwrap()
            .where_(((x).exp() + 1).ln() * beta.reciprocal(), x)
            .unwrap()
    }

    /// Applies the square root function to each element in the input tensor.
    ///
    /// This function returns a new tensor with the same shape as the input, where each element is the square root of the corresponding element in the input tensor. The square root function is useful for various mathematical computations involving squares or square roots.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is the square root of the corresponding element in the input tensor.
    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        let x = self.float_cast();
        let x = Tensor {
            id: RT.lock().sqrt(x.id),
        };
        x
    }

    /// Applies the Swish activation function to each element in the input tensor.
    ///
    /// The Swish function returns `x * sigmoid(x)`, where `sigmoid(x) = 1 / (1 + exp(-x))`. This function is useful for various deep learning applications, as it has been shown to improve convergence speed and generalization performance compared to other activation functions like `ReLU`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is computed according to the Swish function.
    #[must_use]
    pub fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }

    /// Applies the tangent function to each element in the input tensor.
    ///
    /// The tangent function returns the sine of the input divided by the cosine of the input. This function is useful for various mathematical computations involving angles and trigonometry.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is computed according to the tangent function.
    #[must_use]
    pub fn tan(&self) -> Tensor {
        self.sin() / self.cos()
    }

    /// Returns the hyperbolic tangent of each element in the tensor.
    ///
    /// The hyperbolic tangent is calculated as `(exp(2x) + 1) / (exp(2x) - 1)`, where `exp` is the exponential function and `x` is an element of the input tensor. This function applies the hyperbolic tangent element-wise to the input tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from(vec![0.5, 1.0]);
    /// assert_eq!(t.tanh(), [0.46211715738221946, 0.761594166564993]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor is empty.
    #[must_use]
    pub fn tanh(&self) -> Tensor {
        let x = (self.clone() + self.clone()).sigmoid();
        (x.clone() + x) - Tensor::constant(1).cast(self.dtype())
    }

    // movement
    /// Expands this tensor by adding singleton dimensions at the front until its rank matches that of the target shape.
    ///
    /// If the target shape has a higher rank than the current tensor, singleton dimensions are added to the front of the tensor's shape.
    /// If any dimension in the target shape does not match the corresponding dimension in the expanded tensor's shape,
    /// an assertion failure occurs unless the expanded dimension is 1 (in which case it is ignored).
    ///
    /// # Examples
    ///
    /// ```
    /// let t = zyx::Tensor::zeros([2, 3], zyx::DType::U8);
    /// assert_eq!(t.expand((4, 2, 3))?.shape(), &[4, 2, 3]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    /// # Errors
    /// Returns error if self cannot be expanded into shape.
    pub fn expand(&self, shape: impl IntoShape) -> Result<Tensor, ZyxError> {
        let sh = self.shape();
        let shape: Vec<usize> = shape.into_shape().collect();
        //println!("Expand to {shape:?}");
        if shape.len() < sh.len() {
            return Err(ZyxError::ShapeError(format!(
                "Cannot expand {:?} into {:?}",
                self.shape(),
                shape
            )));
        }
        Ok(Tensor {
            id: RT.lock().expand(self.id, shape),
        })
    }

    /// Permutes the axes of this tensor.
    ///
    /// This function rearranges the dimensions of the tensor according to the provided axes. The axes must be a permutation of the original axes, i.e., they must contain each index once and only once. If the axes have a different length than the rank of the tensor, a panic will occur with an appropriate error message.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::{Tensor, DType};
    /// let t = Tensor::rand([3, 4], DType::I64).unwrap();
    /// let p = [1, 0];
    /// let permuted_t = t.permute(p); // Results in a tensor with axes (4, 3)
    /// ```
    ///
    /// # Errors
    /// Returns error if self cannot be permute by axes.
    pub fn permute(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        let axes = into_axes(axes, rank)?;
        //println!("Axes: {axes:?}, rank {rank:?}");
        if rank != axes.len() {
            return Err(ZyxError::ShapeError(format!(
                "Axes has rank {}, but tensor has rank {}. It must be the same for permute.",
                axes.len(),
                rank
            )));
        }
        Ok(Tensor {
            id: RT.lock().permute(self.id, &axes),
        })
    }

    /// Creates a new tensor by padding zeros around this tensor based on the specified padding configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from([1, 2, 3]);
    /// let padded = t.pad_zeros([(1, 1)])?.reshape([5])?;
    /// assert_eq!(padded, [0, 1, 2, 3, 0]);
    ///
    /// let padded = t.pad_zeros([(1, 2)])?;
    /// assert_eq!(padded.shape(), &[6]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    /// Returns error if self cannot be padded by padding.
    #[allow(clippy::missing_panics_doc)]
    pub fn pad_zeros(
        &self,
        padding: impl IntoIterator<Item = (isize, isize)>,
    ) -> Result<Tensor, ZyxError> {
        let padding: Vec<(isize, isize)> = padding.into_iter().collect();
        for (i, &(l, r)) in padding.iter().enumerate() {
            let shape = self.shape();
            let rank = shape.len();
            let mut total = 0;
            if l < 0 {
                total -= l;
            }
            if r < 0 {
                total -= r;
            }
            if usize::try_from(total).unwrap() >= shape[rank - i - 1] {
                return Err(ZyxError::ShapeError(format!(
                    "Invalid padding {padding:?} on shape {shape:?}"
                )));
            }
        }
        Ok(Tensor {
            id: RT.lock().pad_zeros(self.id, padding),
        })
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
    /// let z = x.pad([(1, 2)], 0)?;
    /// assert_eq!(z, [[0, 2, 3, 0, 0],
    ///                [0, 4, 1, 0, 0]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    /// Pad last dimension by (2, -1) and second last dimension by (1, 1)
    /// ```rust
    /// # use zyx::Tensor;
    /// # let x = Tensor::from([[2, 3],
    /// #                       [4, 1]]);
    /// let z = x.pad([(2, -1), (1, 1)], 0)?;
    /// assert_eq!(z, [[0, 0, 0],
    ///                [0, 0, 2],
    ///                [0, 0, 4],
    ///                [0, 0, 0]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    /// Returns error if self cannot be padded by padding.
    #[allow(clippy::missing_panics_doc)]
    pub fn pad(
        &self,
        padding: impl IntoIterator<Item = (isize, isize)>,
        value: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let dtype = self.dtype();
        let value: Tensor = value.into();
        let padding: Vec<(isize, isize)> = padding.into_iter().collect();
        let sh = self.shape();
        if value.dtype() != dtype {
            return Err(ZyxError::DTypeError(format!(
                "Cannot pad tensor with dtype {} with value of dtype {}",
                dtype,
                value.dtype()
            )));
        }
        if !padding.len() <= sh.rank() && padding.iter().zip(sh.iter().rev()).all(|(&(lp, rp), &d)| if lp < 0 { usize::try_from(-lp).unwrap() <= d } else { true } && if rp < 0 { usize::try_from(-rp).unwrap() <= d } else { true }) {
            return Err(ZyxError::ShapeError(format!("Cannot pad tensor with shape {sh:?} with padding {padding:?}")));
        }
        let t0 = self.pad_zeros(padding.clone());
        let ones = Tensor::ones(sh.clone(), dtype);
        let zeros = Tensor::zeros(sh, self.dtype());
        Ok(t0? + ones.pad_zeros(padding)?.where_(zeros, value)?)
    }

    /// Narrow tensor along an axis, is essentially just padding
    /// ```
    /// # use zyx::Tensor;
    /// let x = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    /// assert_eq!(x.narrow(0, 0, 2)?, [[1, 2, 3], [4, 5, 6]]);
    /// assert_eq!(x.narrow(1, 1, 2)?, [[2, 3], [5, 6], [8, 9]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    /// # Errors
    /// Returns error if self cannot be narrowed.
    #[allow(clippy::missing_panics_doc)]
    pub fn narrow(&self, axis: isize, start: usize, length: usize) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let rank = shape.len();
        let axis = into_axis(axis, rank)?;
        let dim = isize::try_from(shape[axis]).unwrap();
        let padding: Vec<(isize, isize)> = once((
            -isize::try_from(start).unwrap(),
            -dim + isize::try_from(length).unwrap() + isize::try_from(start).unwrap(),
        ))
        .chain(core::iter::repeat((0, 0)).take(rank - axis - 1))
        .collect::<Vec<(isize, isize)>>()
        .into_iter()
        .rev()
        .collect();
        Ok(self.pad_zeros(padding).unwrap())
    }

    /// Applies a new shape to this tensor while preserving its total number of elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    /// let t = Tensor::from([1, 2, 3, 4]);
    /// assert_eq!(t.reshape((2, 2))?, [[1, 2], [3, 4]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    /// Returns error if self cannot be reshaped to shape.
    pub fn reshape(&self, shape: impl IntoShape) -> Result<Tensor, ZyxError> {
        let shape: Vec<usize> = shape.into_shape().collect();
        if shape.iter().product::<usize>() != self.numel() {
            return Err(ZyxError::ShapeError(format!(
                "Invalid reshape {:?} into {:?}",
                self.shape(),
                shape
            )));
        };
        Ok(Tensor {
            id: RT.lock().reshape(self.id, shape),
        })
    }

    /// An alias to reshape
    /// # Errors
    /// Returns error if self cannot be reshaped to shape.
    pub fn view(&self, shape: impl IntoShape) -> Result<Tensor, ZyxError> {
        self.reshape(shape)
    }

    /// Transpose last two dimensions of this tensor.
    /// If `self.rank() == 1`, returns tensor with shape `[self.shape()[0], 1]` (column tensor)
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn t(&self) -> Tensor {
        let mut rank = self.rank();
        let x = if rank == 1 {
            let n = self.numel();
            rank = 2;
            &self.reshape([1, n]).unwrap()
        } else {
            self
        };
        let mut axes: Vec<isize> = (0..isize::try_from(rank).unwrap()).collect();
        axes.swap(rank - 1, rank - 2);
        x.permute(axes).unwrap()
    }

    /// Transpose two arbitrary dimensions
    /// ```rust
    /// use zyx::Tensor;
    /// let t = Tensor::from([[[1, 2]], [[3, 4]]]);
    /// assert_eq!(t.transpose(0, -1)?, [[[1, 3]], [[2, 4]]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    /// Returns error if self cannot be transposed by dim0 and dim1.
    #[allow(clippy::missing_panics_doc)]
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        if (dim0 < 0 && usize::try_from(-dim0).unwrap() > rank)
            || usize::try_from(dim0).unwrap() >= rank
        {
            return Err(ZyxError::ShapeError(format!(
                "Cannot transpose dimensions {dim0} and {dim1}, {dim0} is greater than rank {rank}"
            )));
        }
        if (dim1 < 0 && usize::try_from(-dim1).unwrap() > rank)
            || usize::try_from(dim1).unwrap() >= rank
        {
            return Err(ZyxError::ShapeError(format!(
                "Cannot transpose dimensions {dim0} and {dim1}, {dim1} is greater than rank {rank}"
            )));
        }
        let mut axes: Vec<isize> = (0..isize::try_from(rank).unwrap()).collect();
        axes.swap(into_axis(dim0, rank)?, into_axis(dim1, rank)?);
        self.permute(axes)
    }

    // reduce
    /// Computes the natural logarithm of the softmax of the input tensor along the specified axes.
    ///
    /// This function first subtracts the maximum value along the given axes from the input tensor,
    /// then computes the exponential of the result, sums over the specified axes using `sum_kd`,
    /// and finally takes the natural logarithm of the sum before returning it.
    ///
    /// # Arguments
    ///
    /// * `self` - The input tensor to compute the softmax and natural logarithm of.
    /// * `axes` - A trait implementing `IntoAxes`, specifying along which axes the softmax should be computed.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    /// let x = Tensor::from([2f32, 3., 4.]);
    /// let y = x.ln_softmax([]);
    /// ```
    ///
    /// # Returns
    ///
    /// The resulting tensor after computing the natural logarithm of the softmax of `self`.
    ///
    /// # Errors
    ///
    /// Returns error if any of the specified axes are out-of-bounds for the input tensor.
    #[allow(clippy::missing_panics_doc)]
    pub fn ln_softmax(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        let m = self - self.max_kd(axes.clone())?;
        Ok(&m - m.exp().sum_kd(axes)?.ln())
    }

    /// Returns a new tensor containing the maximum value along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to compute the maximum. This can be any type that implements `IntoAxes`
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    /// let arr = Tensor::from([1, 2, 3, 4]);
    /// assert_eq!(arr.max([0])?, [4]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if the axes contain duplicates or are out of bounds.
    pub fn max(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let rank = self.rank();
        let axes = into_axes(axes, rank)?;
        let mut unique = BTreeSet::new();
        for a in &axes {
            if !unique.insert(a) {
                return Err(ZyxError::ShapeError("Axes contain duplicates.".into()));
            }
        }
        Ok(Tensor {
            id: RT.lock().max_reduce(self.id, axes),
        })
    }

    /// Returns the maximum value along the specified axes.
    ///
    /// This function computes the maximum value of each slice determined by the `axes`.
    /// It first calculates the maximum along the specified axes using the `max` method,
    /// and then reshapes the result to have the same number of dimensions as the input tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let a = Tensor::from([1, 2, 3, 4]);
    /// assert_eq!(a.max_kd([0])?, [4]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn max_kd(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        self.max(axes.clone())?.reshape(self.reduce_kd_shape(axes))
    }

    /// Calculates the mean of a tensor along specified axes.
    ///
    /// This function computes the sum of all elements in the tensor along the specified axes and then divides by the product of their sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::{Tensor, DType};
    ///
    /// let arr = Tensor::eye(3, DType::F32);
    /// assert_eq!(arr.mean([0])?, [0.333333f32, 0.333333, 0.333333]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    #[allow(clippy::missing_panics_doc)]
    pub fn mean(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        let shape = self.shape();
        Ok(self.sum(axes.clone())?
            / Tensor::constant(
                i64::try_from(
                    into_axes(axes, shape.rank())?
                        .into_iter()
                        .map(|a| shape[a])
                        .product::<usize>(),
                )
                .unwrap(),
            )
            .cast(self.dtype()))
    }

    /// Calculates the mean of this tensor along the specified axes and reshapes it using `reduce_kd_shape`.
    ///
    /// This function first calculates the mean of the input tensor along the specified axes using the `mean`
    /// method. It then reshapes the resulting tensor using `reduce_kd_shape` to match the output shape expected
    /// by the caller.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let a = Tensor::from([1, 2, 3, 4]);
    /// assert_eq!(a.mean_kd([])?, [2]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn mean_kd(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        self.mean(axes.clone())?.reshape(self.reduce_kd_shape(axes))
    }

    /// Calculates the product of elements along specified axes.
    ///
    /// This function first applies the natural logarithm element-wise (`ln()`), then sums along the specified axes,
    /// and finally exponentiates the result element-wise (`exp()`).
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(arr.product([1])?, [2., 12.]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn product(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        Ok(self.ln().sum(axes)?.exp())
    }

    /// Calculates the standard deviation of the input tensor along specified axes.
    ///
    /// This function calculates the standard deviation by first computing the mean along the specified axes,
    /// then subtracting that mean from each element, squaring the result, and finally taking the square root
    /// of the average of those squared differences. If no axes are provided, it computes the standard deviation
    /// over all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let a = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
    /// assert_eq!(a.std([0, 1], 1)?, 1.8708);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn std(
        &self,
        axes: impl IntoIterator<Item = isize>,
        correction: usize,
    ) -> Result<Tensor, ZyxError> {
        Ok(self.var(axes, correction)?.sqrt())
    }

    /// Creates a new tensor by applying standard deviation along specified axes.
    ///
    /// This function first computes the standard deviation of the input tensor along the specified axes,
    /// and then reshapes the result to match the shape of the original tensor after reduction along those axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::{Tensor, DType};
    ///
    /// let t = Tensor::rand([3, 4], DType::F32).unwrap();
    /// let std_kd = t.std_kd([0, 1], 1)?;
    /// assert_eq!(std_kd.shape(), [1, 1]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn std_kd(
        &self,
        axes: impl IntoIterator<Item = isize>,
        correction: usize,
    ) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        self.std(axes.clone(), correction)?
            .reshape(self.reduce_kd_shape(axes))
    }

    /// Sum reduce. Removes tensor dimensions.
    /// Equivalent to pytorch sum(axes, keepdim=False)
    /// If you want to keep reduce dimensions, see [`sum_kd`](Tensor::sum_kd)
    /// Passing empty axes executes reduce across all dimensions and result will have shape `[1]`
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn sum(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        // TODO handle axes out of range error
        let rank = self.rank();
        let axes = into_axes(axes, rank)?;
        Ok(Tensor {
            id: RT.lock().sum_reduce(self.id, axes),
        })
    }

    // Probably just have sum_kd, max_kd that keep tensor dimensions
    /// Like [sum](Tensor::sum) but keeps reduce dimensions, setting them to 1.
    /// Equivalent to pytorch sum(axes, keepdim=True)
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn sum_kd(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        self.sum(axes.clone())?.reshape(self.reduce_kd_shape(axes))
    }

    /// Comulative sum along axis.
    ///
    /// # Errors
    ///
    /// Returns error if axis is out of range.
    #[allow(clippy::missing_panics_doc)]
    pub fn cumsum(&self, axis: isize) -> Result<Tensor, ZyxError> {
        let axis = into_axis(axis, self.rank())?;
        //println!("Cumsum, shape: {:?}", self.shape());
        let pl_sz = isize::try_from(self.shape()[axis] - 1).unwrap();
        let k = self.shape()[axis];
        let axis = isize::try_from(axis).unwrap();
        let mut x = self.transpose(axis, -1)?;
        x = x.pad_zeros([(pl_sz, 0)])?;
        //println!("{x:?} padded");
        x = x.pool(k, 1, 1)?;
        //println!("{x:?} pooled");
        x = x.sum([-1])?;
        //println!("{x:?} summed");
        x = x.transpose(axis, -1)?;
        //println!("{x:?} transposed");
        Ok(x)
    }

    /// Calculates the softmax of this tensor along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `axes`: The axes along which to calculate the softmax.
    ///
    /// # Returns
    ///
    /// * A new tensor containing the result of the softmax operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from(vec![1.0, 2.0, 3.0]);
    /// let sm = t.softmax([])?;
    /// assert_eq!(sm, [0.0900305748, 0.2447281546, 0.6652412706]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    pub fn softmax(&self, axes: impl IntoIterator<Item = isize>) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        let e = (self - self.max_kd(axes.clone())?).exp();
        Ok(&e / e.sum_kd(axes)?)
    }

    /// Calculates the variance of this tensor along the specified axes.
    ///
    /// This function first computes the mean of the tensor along the provided axes,
    /// then subtracts this mean from each element in the tensor, squares the result,
    /// and finally means these squared differences along the same axes to obtain the variance.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to compute the mean and variance. This can be a single axis or a tuple of axes.
    ///
    /// # Returns
    ///
    /// * A new tensor containing the variance values computed for each axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::from([[1f32, 2.], [3., 4.]]);
    /// let var = arr.var([0], 0)?; // Compute variance along rows (axis=0)
    /// assert_eq!(var, [1f32, 1.]);
    ///
    /// let var = arr.var([1], 1)?; // Compute variance along columns (axis=1)
    /// assert_eq!(var, [0.5f32, 0.5]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    #[allow(clippy::missing_panics_doc)]
    pub fn var(
        &self,
        axes: impl IntoIterator<Item = isize>,
        correction: usize,
    ) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        let shape = self.shape();
        let x = self - self.mean_kd(axes.clone())?;
        let d = i64::try_from(
            into_axes(axes.clone(), shape.rank())?
                .into_iter()
                .map(|a| shape[a])
                .product::<usize>(),
        )
        .unwrap()
            - i64::try_from(correction).unwrap();
        Ok((x.clone() * x.clone()).sum(axes)? / Tensor::constant(d).cast(x.dtype()))
    }

    /// Calculates the variance along the specified axes.
    ///
    /// This function first calculates the mean along the specified axes using `var()`,
    /// then subtracts that mean from the original tensor, squares the result,
    /// and finally takes the mean of those squared values.
    ///
    /// # Arguments
    ///
    /// * `axes`: The axes to reduce over. If not provided, reduces over all axes.
    ///
    /// # Returns
    ///
    /// A new tensor containing the variance along the specified axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let a = Tensor::from([[2f64, 3., 4.], [5., 6., 7.]]);
    /// assert_eq!(a.var_kd([0], 0)?, [[2.25f64, 2.25, 2.25]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be reduced by axes.
    #[allow(clippy::missing_panics_doc)]
    pub fn var_kd(
        &self,
        axes: impl IntoIterator<Item = isize>,
        correction: usize,
    ) -> Result<Tensor, ZyxError> {
        let axes: Vec<_> = axes.into_iter().collect();
        self.var(axes.clone(), correction)?
            .reshape(self.reduce_kd_shape(axes))
    }

    // index
    /// Get function
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be indexed by index.
    #[allow(clippy::missing_panics_doc)]
    pub fn get(&self, index: impl IntoIndex) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let padding: Vec<(isize, isize)> = index
            .into_index()
            .into_iter()
            .zip(shape.iter())
            .map(|(r, &d)| {
                (
                    if r.start >= 0 {
                        -r.start
                    } else {
                        -r.start - isize::try_from(d).unwrap()
                    },
                    if r.end == isize::MAX {
                        0
                    } else if r.end > 0 {
                        -(isize::try_from(d).unwrap() - r.end)
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

    /// Returns a tensor containing only the diagonal elements of this tensor.
    ///
    /// The diagonal is obtained by flattening the input tensor, padding it with zeros to make its last dimension size equal
    /// to the number of rows or columns in the original tensor, reshaping it into a 2D matrix, and then extracting the diagonal.
    ///
    /// # Returns
    ///
    /// * A new tensor containing only the diagonal elements of this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape([3, 3])?;
    /// assert_eq!(arr.diagonal(), [1, 5, 9]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn diagonal(&self) -> Tensor {
        let n = *self
            .shape()
            .last()
            .expect("Shape in invalid state. Internal bug.");
        self.flatten(..)
            .unwrap()
            .pad_zeros([(0, isize::try_from(n).unwrap())])
            .unwrap()
            .reshape([n, n + 1])
            .unwrap()
            .get((.., 0))
            .unwrap()
            .flatten(..)
            .unwrap()
    }

    // binary
    /// Compares this tensor with another tensor element-wise.
    ///
    /// Returns a new tensor of boolean values indicating where `self` is less than `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let a = Tensor::from([1.0, 2.0, 3.0]);
    /// let b = Tensor::from([4.0, 5.0, 6.0]);
    /// assert_eq!(a.cmplt(b)?, [1., 1., 1.]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn cmplt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().cmplt(x.id, y.id);
        Ok(Tensor { id })
    }

    /// Compare greater than
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn cmpgt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().cmpgt(x.id, y.id);
        Ok(Tensor { id })
    }

    /// Elementwise maximum between two tensors.
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn maximum(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().maximum(x.id, y.id);
        Ok(Tensor { id })
    }

    /// Elementwise minimum between two tensors
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn minimum(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        Ok(-(-self).maximum(-rhs.into())?)
    }

    /// Matmul and dot
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn dot(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let rhs = rhs.into();
        let org_y_shape = rhs.shape();
        let y = rhs.t();
        let xshape = self.shape();
        let yshape = y.shape();
        //println!("xshape {xshape:?}, yshape {yshape:?}");
        let xrank = xshape.len();
        let yrank = yshape.len();
        if xshape[xrank - 1] != yshape[yrank - 1] {
            //yshape[-(yrank.min(2) as i64)],
            return Err(ZyxError::ShapeError(format!(
                "Cannot dot tensors with shapes {xshape:?} and {org_y_shape:?}"
            )));
        }
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
        (self.reshape(x_shape)? * y.reshape(y_shape)?)
            .sum([-1])?
            .reshape(
                xshape[0..xshape.len() - 1]
                    .iter()
                    .copied()
                    .chain([yshape[yshape.len() - 2]])
                    .collect::<Vec<usize>>(),
            )
    }

    /// Matmul is just alias to dot
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn matmul(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        self.dot(rhs)
    }

    /// Returns a new tensor where each element is the result of raising the corresponding element in `self` to the power of `exponent`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::from([1.0, 2.0]);
    /// assert_eq!(arr.pow(2.0)?, [1.0, 4.0]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Returns
    ///
    /// A new tensor where each element is the result of raising the corresponding element in `self` to the power of `exponent`.
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn pow(&self, exponent: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), exponent)?;
        let id = RT.lock().pow(x.id, y.id);
        Ok(Tensor { id })
    }

    /// Logical and
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn logical_and(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().and(x.id, y.id);
        Ok(Tensor { id })
    }

    /// Logical or
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn logical_or(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().or(x.id, y.id);
        Ok(Tensor { id })
    }

    /// Returns boolean mask with true where self == rhs
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn equal(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().not_eq(x.id, y.id);
        let x = Tensor { id };
        Ok(x.not())
    }

    /// Returns ones where self is different from zero and zeros otherwise.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn nonzero(&self) -> Tensor {
        !self.equal(Tensor::constant(0).cast(self.dtype())).unwrap()
    }

    // ternary
    /// Where operation. Replaces elementwise true values with `if_true` and false values with `if_false`.
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    #[allow(clippy::missing_panics_doc)]
    pub fn where_(
        &self,
        if_true: impl Into<Tensor>,
        if_false: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), if_true)?;
        let (x, z) = Tensor::broadcast(x, if_false)?;
        let (y, z) = Tensor::broadcast(y, z)?;
        let x_nonzero = x.nonzero();
        Ok(&x_nonzero * y + !x_nonzero * z)
    }

    // loss functions
    /// Calculates the cross-entropy loss for this tensor.
    ///
    /// This function takes a target tensor and axes as input. It first calculates the softmax of the input tensor along the specified axes,
    /// then multiplies the result by the logarithm of the target tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    /// let input = Tensor::from([0.5, 0.2, 0.3]);
    /// let target = Tensor::from([1., 0., 0.]);
    /// assert_eq!(input.cross_entropy_loss(target, [])?.mean([])?, -0.3133);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes or axes cannot reduce self.
    pub fn cross_entropy_loss(
        &self,
        target: impl Into<Tensor>,
        axes: impl IntoIterator<Item = isize>,
    ) -> Result<Tensor, ZyxError> {
        Ok(self.ln_softmax(axes)? * target)
    }

    /// Calculates the L1 loss between `self` and the target tensor.
    ///
    /// # Arguments
    ///
    /// * `target`: The target tensor to compare against. It will be converted into a `Tensor`.
    ///
    /// # Returns
    ///
    /// A new `Tensor` containing the absolute difference between `self` and the target tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let self_tensor = Tensor::from([1.0, 2.0, 3.0]);
    /// let target_tensor = Tensor::from([2.0, 3.0, 4.0]);
    ///
    /// assert_eq!(self_tensor.l1_loss(target_tensor), [1.0, 1.0, 1.0]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    #[must_use]
    pub fn l1_loss(&self, target: impl Into<Tensor>) -> Tensor {
        (self - target).abs()
    }

    /// Calculates the Mean Squared Error (MSE) loss.
    ///
    /// # Arguments
    ///
    /// * `target`: The target tensor to compare against the input tensor (`self`).
    ///
    /// # Returns
    ///
    /// * A new tensor containing the MSE loss values.
    ///
    /// # Example
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let input = Tensor::from([2.0, 3.0]);
    /// let target = Tensor::from([4.0, 5.0]);
    ///
    /// assert_eq!(input.mse_loss(target), [4.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    #[must_use]
    pub fn mse_loss(&self, target: impl Into<Tensor>) -> Tensor {
        let x = self - target;
        x.clone() * x
    }

    /// Calculates the cosine similarity between this tensor and another.
    ///
    /// # Arguments
    ///
    /// * `rhs`: The other tensor to compare against. It will be converted into a `Tensor`.
    /// * `eps`: A tolerance value for numerical stability, which will also be converted into a `Tensor`.
    ///
    /// # Returns
    ///
    /// A new `Tensor` containing the cosine similarity values.
    ///
    /// # Example
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let tensor1 = Tensor::from([1.0, 2.0, 3.0]);
    /// let tensor2 = Tensor::from([4.0, 5.0, 6.0]);
    /// let eps = Tensor::from([1e-9]);
    ///
    /// let similarity = tensor1.cosine_similarity(tensor2, eps);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn cosine_similarity(
        &self,
        rhs: impl Into<Tensor>,
        eps: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let rhs: Tensor = rhs.into();
        let eps: Tensor = eps.into();
        let x = self.pow(2)?.sqrt() * rhs.pow(2)?.sqrt();
        Ok(self * rhs / x.cmplt(eps.clone())?.where_(eps, x)?)
    }

    // misc
    /// Flatten. Joins axes into one dimension,
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be flattened by axes.
    pub fn flatten(&self, axes: impl RangeBounds<isize>) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        let rank = shape.len();
        let start_dim = into_axis(
            match axes.start_bound() {
                Bound::Included(dim) => *dim,
                Bound::Excluded(dim) => *dim + 1,
                Bound::Unbounded => 0,
            },
            rank,
        )?;
        let end_dim = into_axis(
            match axes.end_bound() {
                Bound::Included(dim) => *dim,
                Bound::Excluded(dim) => *dim - 1,
                Bound::Unbounded => -1,
            },
            rank,
        )? + 1;
        let dim = shape[start_dim..end_dim].iter().product();
        let new_shape: Vec<usize> = shape[..start_dim]
            .iter()
            .copied()
            .chain([dim])
            .chain(shape[end_dim..].iter().copied())
            .collect();
        self.reshape(new_shape)
    }

    /// Concatenates a list of tensors along a specified dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors`: An iterator of tensor references to concatenate.
    /// * `dim`: The dimension along which to concatenate. If negative, it is interpreted as counting from the end.
    ///
    /// # Returns
    ///
    /// A new tensor containing the concatenated input tensors.
    ///
    /// # Panics
    ///
    /// This function panics if any two tensors have different shapes except at the specified dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let a = Tensor::from([[1, 2], [3, 4]]);
    /// let b = Tensor::from([[5, 6], [7, 8]]);
    /// let c = Tensor::cat([&a, &b], 0)?;
    /// assert_eq!(c, [[1, 2], [3, 4], [5, 6], [7, 8]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if tensors cannot be concattenated along axis.
    pub fn cat<'a>(
        tensors: impl IntoIterator<Item = &'a Tensor>,
        axis: isize,
    ) -> Result<Tensor, ZyxError> {
        let tensors: Vec<&Tensor> = tensors.into_iter().collect();
        if tensors.len() < 2 {
            return Err(ZyxError::ShapeError(
                "Cat requires two or more tensors.".into(),
            ));
        }
        let shape = tensors[0].shape();
        let rank = shape.rank();
        let dim: usize = (if axis < 0 {
            axis + isize::try_from(rank).unwrap()
        } else {
            axis
        })
        .try_into()
        .unwrap();
        // Dimension check
        for tensor in &tensors {
            for (i, (d1, d2)) in shape.iter().zip(tensor.shape().iter()).enumerate() {
                if i != dim && *d1 != *d2 {
                    return Err(ZyxError::ShapeError(
                        "Cannot concatenate these tensors.".into(),
                    ));
                }
            }
        }
        let mut offset = 0isize;
        let mut offset2 = tensors
            .iter()
            .fold(0, |acc, t| acc + isize::try_from(t.shape()[dim]).unwrap());
        let mut shape = tensors[0].shape();
        shape[dim] = usize::try_from(offset2).unwrap();
        let mut res = None;
        for tensor in tensors {
            let d = isize::try_from(tensor.shape()[dim]).unwrap();
            offset2 -= d;
            let padding: Vec<(isize, isize)> = core::iter::repeat((0isize, 0isize))
                .take(rank - dim - 1)
                .chain([(offset, offset2)])
                .collect();
            let t = tensor.pad_zeros(padding)?;
            if let Some(r) = res {
                res = Some(r + t);
            } else {
                res = Some(t);
            }
            offset += d;
        }
        Ok(res.unwrap())
    }

    /// Squeeze
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be squeezed along axis.
    #[allow(clippy::missing_panics_doc)]
    pub fn squeeze(&self, axis: isize) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        if axis < 0 {
            let rank = shape.len();
            let dim = usize::try_from(-axis).unwrap();
            let dim = rank - dim + 1;
            if shape[dim] != 1 {
                return Ok(self.clone());
            }
            self.reshape(
                shape[..dim]
                    .iter()
                    .copied()
                    .chain(shape[dim + 1..].iter().copied())
                    .collect::<Vec<usize>>(),
            )
        } else {
            let dim = usize::try_from(axis).unwrap();
            if shape[dim] != 1 {
                return Ok(self.clone());
            }
            self.reshape(
                shape[..dim]
                    .iter()
                    .copied()
                    .chain(shape[dim + 1..].iter().copied())
                    .collect::<Vec<usize>>(),
            )
        }
    }

    /// Expands the dimensionality of a tensor by inserting singleton dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim`: The dimension to insert the singleton dimension at. If negative, it is counted from the end.
    ///
    /// # Returns
    ///
    /// A new tensor with expanded dimensionality.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::{Tensor, DType};
    ///
    /// let t = Tensor::zeros([2, 3], DType::I8);
    /// assert_eq!(t.unsqueeze(1)?.shape(), &[2, 1, 3]);
    /// assert_eq!(t.unsqueeze(-1)?.shape(), &[2, 3, 1]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be unsqueezed along axis.
    #[allow(clippy::missing_panics_doc)]
    pub fn unsqueeze(&self, dim: isize) -> Result<Tensor, ZyxError> {
        let shape = self.shape();
        if dim < 0 {
            let rank = shape.len();
            let dim = usize::try_from(-dim).unwrap();
            let dim = rank - dim + 1;
            self.reshape(
                shape[..dim]
                    .iter()
                    .copied()
                    .chain([1])
                    .chain(shape[dim..].iter().copied())
                    .collect::<Vec<usize>>(),
            )
        } else {
            let dim = usize::try_from(dim).unwrap();
            self.reshape(
                shape[..dim]
                    .iter()
                    .copied()
                    .chain([1])
                    .chain(shape[dim..].iter().copied())
                    .collect::<Vec<usize>>(),
            )
        }
    }

    /// Creates a new tensor by stacking the input tensors along the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors`: An iterator of tensor references to stack.
    /// * `dim`: The dimension along which to stack the tensors.
    ///
    /// # Returns
    ///
    /// A new tensor containing the stacked tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    /// let a = Tensor::from([[1, 2], [3, 4]]);
    /// let b = Tensor::from([[5, 6], [7, 8]]);
    /// assert_eq!(Tensor::stack([&a, &b], 0)?, [[[1, 2],
    ///                                           [3, 4]],
    ///                                          [[5, 6],
    ///                                           [7, 8]]]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have different shapes along the stacking dimension.
    ///
    /// # See also
    ///
    /// [`unsqueeze`](Tensor::unsqueeze), [`cat`](Tensor::cat)
    #[allow(clippy::missing_panics_doc)]
    pub fn stack<'a>(
        tensors: impl IntoIterator<Item = &'a Tensor>,
        dim: isize,
    ) -> Result<Tensor, ZyxError> {
        // TODO handle dim corretly
        let tensors: Vec<Tensor> = tensors
            .into_iter()
            .map(|t| t.unsqueeze(dim).unwrap())
            .collect();
        Tensor::cat(&tensors, dim)
    }

    /// Split tensor into multiple tensors at given dim/axis
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be split along axis.
    #[allow(clippy::missing_panics_doc)]
    pub fn split(&self, sizes: impl IntoShape, axis: isize) -> Result<Vec<Tensor>, ZyxError> {
        // assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
        // dim = self._resolve_dim(dim)
        // if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
        // assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
        // return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])
        let sizes: Vec<usize> = sizes.into_shape().collect();
        let shape = self.shape();
        let rank = shape.rank();
        let dim: usize = usize::try_from(if axis < 0 {
            axis + isize::try_from(rank).unwrap()
        } else {
            axis
        })
        .unwrap();
        if sizes.iter().sum::<usize>() != shape[dim] {
            return Err(ZyxError::ShapeError(format!(
                "Sizes must sum exactly to {}, but got {:?}, which sums to {}",
                shape[dim],
                sizes,
                sizes.iter().sum::<usize>()
            )));
        }

        let mut res = Vec::new();
        let mut acc_size = 0;
        for size in sizes {
            let size = isize::try_from(size).unwrap();
            let mut index = Vec::new();
            for &d in shape.iter().take(dim) {
                index.push(0..isize::try_from(d).unwrap());
            }
            index.push(acc_size..acc_size + size);
            //println!("Index {index:?}");
            res.push(self.get(index)?);
            acc_size += size;
        }
        Ok(res)
    }

    /// Masked fill
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be masked with mask.
    pub fn masked_fill(
        &self,
        mask: impl Into<Tensor>,
        value: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        mask.into().where_(value, self.clone())
    }

    /*#[must_use]
    fn tri(n: usize, dtype: DType) -> Tensor {
        // if r == 0 or c == 0 or diagonal >= c: return Tensor.zeros(r,c,**kwargs)
        // if r+diagonal <= 0: return Tensor.ones(r,c,**kwargs)
        // s = r+c-1
        // # build a (s, s) upper triangle
        // t = Tensor.ones(s,s,**kwargs).pad((None,(0,s))).flatten().shrink(((0,s*(2*s-1)),)).reshape(s,-1).shrink((None,(0,s)))
        // return t[:r,-diagonal:c-diagonal] if diagonal <= 0 else t[diagonal:r+diagonal,:c]
        Tensor::ones([n * n / 2], dtype).pad_zeros([(0, n * n / 2)])
    }*/

    // Returns upper triangular part of the input tensor, other elements are set to zero
    /*#[must_use]
    pub fn triu(&self, diagonal: isize) -> Tensor {
        todo!()
    }*/

    /// Pooling function with kernel size, stride and dilation
    ///
    /// # Errors
    ///
    /// Returns error if self cannot be pooled with stride and dilation.
    #[allow(clippy::missing_panics_doc)]
    pub fn pool(
        &self,
        kernel_size: impl IntoShape,
        stride: impl IntoShape,
        dilation: impl IntoShape,
    ) -> Result<Tensor, ZyxError> {
        // What a complex function ...
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
        let i_ = &shape[rank - k_.len()..];
        let o_: Vec<usize> = i_
            .iter()
            .copied()
            .zip(d_.iter().copied())
            .zip(k_.iter().copied())
            .zip(s_.iter().copied())
            .map(|(((i, d), k), s)| (i - d * (k - 1)).div_ceil(s))
            .collect();
        //println!("s_ {s_:?}, d_ {d_:?}, i_ {i_:?} o_ {o_:?}");
        let repeats: Vec<usize> = repeat(1)
            .take(rank - k_.len())
            .chain(
                k_.iter()
                    .copied()
                    .zip(i_.iter().copied())
                    .zip(d_.iter().copied())
                    .map(|((k, i), d)| (k * (i + d)).div_ceil(i)),
            )
            .collect();
        //println!("repeats {repeats:?}");
        let pad_b: Vec<Range<isize>> = shape[..rank - k_.len()]
            .iter()
            .map(|&d| 0..isize::try_from(d).unwrap())
            .collect();
        let sh_b: Vec<usize> = shape[..rank - k_.len()].into();
        let mut xup = self.repeat(repeats)?;

        // dilation
        //println!("{xup:?} before padding");
        let padding: Vec<Range<isize>> = pad_b
            .iter()
            .cloned()
            .chain(
                k_.iter()
                    .copied()
                    .zip(i_.iter().copied())
                    .zip(d_.iter().copied())
                    .map(|((k, i), d)| (0..isize::try_from(k * (i + d)).unwrap())),
            )
            .collect();
        //println!("Padding {padding:?}");
        xup = xup.get(padding)?;
        //println!("{xup} padded");
        let sh: Vec<usize> = sh_b
            .iter()
            .copied()
            .chain(
                k_.iter()
                    .copied()
                    .zip(i_.iter().copied())
                    .zip(d_.iter().copied())
                    .flat_map(|((k, i), d)| [k, i + d]),
            )
            .collect();
        //println!("Reshape {sh:?}");
        xup = xup.reshape(sh)?;

        // stride
        // padding = noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_))
        // xup = xup.shrink(padding)
        let padding: Vec<Range<isize>> = pad_b
            .iter()
            .cloned()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .zip(s_.iter().copied())
                    .flat_map(|((k, o), s)| {
                        [
                            (0..isize::try_from(k).unwrap()),
                            (0..isize::try_from(o * s).unwrap()),
                        ]
                    }),
            )
            .collect();
        xup = xup.get(padding)?;
        // sh = noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_))
        // xup = xup.reshape(sh)
        let sh: Vec<usize> = sh_b
            .iter()
            .copied()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .zip(s_.iter().copied())
                    .flat_map(|((k, o), s)| [k, o, s]),
            )
            .collect();
        xup = xup.reshape(sh)?;
        // padding = noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))
        // xup = xup.shrink(padding)
        let padding: Vec<Range<isize>> = pad_b
            .iter()
            .cloned()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .flat_map(|(k, o)| {
                        [
                            (0..isize::try_from(k).unwrap()),
                            (0..isize::try_from(o).unwrap()),
                            (0..1),
                        ]
                    }),
            )
            .collect();
        xup = xup.get(padding)?;
        // sh = noop_ + flatten((k,o) for k,o in zip(k_, o_))
        // xup = xup.reshape(sh)
        let sh: Vec<usize> = sh_b
            .iter()
            .copied()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .flat_map(Into::<[usize; 2]>::into),
            )
            .collect();
        xup = xup.reshape(sh)?;

        // xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
        let axes: Vec<isize> = (0..rank - k_.len())
            .chain((0..i_.len()).map(|i| rank - k_.len() + i * 2 + 1))
            .chain((0..i_.len()).map(|i| rank - k_.len() + i * 2))
            .map(|i| isize::try_from(i).unwrap())
            .collect();
        xup = xup.permute(axes)?;

        Ok(xup)
    }

    /// Creates a new tensor by repeating the input tensor along its dimensions.
    ///
    /// The `repeats` parameter specifies how many times to repeat each dimension of the tensor. If the length of `repeats`
    /// is less than the rank of the tensor, it will be padded with ones at the beginning.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::from(vec![1, 2, 3]);
    /// assert_eq!(arr.repeat([2])?, [1, 2, 3, 1, 2, 3]);
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Returns
    ///
    /// Returns a new tensor with the repeated values.
    ///
    /// # Errors
    ///
    /// Returns error if the input tensor has zero dimensions.
    #[allow(clippy::missing_panics_doc)]
    pub fn repeat(&self, repeats: impl IntoShape) -> Result<Tensor, ZyxError> {
        let repeats: Vec<usize> = repeats.into_shape().collect();
        let shape = self.shape();
        let rank = shape.len();
        if repeats.len() < rank {
            return Err(ZyxError::ShapeError(
                "Repeats must be greater or equal to rank of the tensor.".into(),
            ));
        }
        let base_shape: Vec<usize> = repeat(1)
            .take(repeats.len() - rank)
            .chain(shape.iter().copied())
            .collect();
        let new_shape: Vec<usize> = repeat(1)
            .take(repeats.len() - rank)
            .chain(shape)
            .flat_map(|d| [1, d])
            .collect();
        let expand_shape: Vec<usize> = repeats
            .iter()
            .copied()
            .zip(base_shape.iter().copied())
            .flat_map(Into::<[usize; 2]>::into)
            .collect();
        let final_shape: Vec<usize> = repeats
            .iter()
            .copied()
            .zip(base_shape.iter().copied())
            .map(|(r, d)| r * d)
            .collect();
        //println!("base_shape {base_shape:?} {new_shape:?} {expand_shape:?} {final_shape:?}");
        let mut x = self.reshape(new_shape).unwrap();
        x = x.expand(expand_shape).unwrap();
        x = x.reshape(final_shape).unwrap();
        Ok(x)
    }

    /// Rotary embeddings
    ///
    /// # Errors
    ///
    /// Returns error if shapes of tensors are not compatible.
    #[allow(clippy::missing_panics_doc)]
    pub fn rope(
        &self,
        sin_freqs: impl Into<Tensor>,
        cos_freqs: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let sh = self.shape();
        let sin_freqs = sin_freqs.into();
        let cos_freqs = cos_freqs.into();
        let sin_freqs = sin_freqs.squeeze(1).unwrap().squeeze(0).unwrap(); // [seq_len, dim]
        let cos_freqs = cos_freqs.squeeze(1).unwrap().squeeze(0).unwrap(); // [seq_len, dim]
        let d = isize::try_from(*sh.last().unwrap()).unwrap();
        let a = self.get((.., .., .., ..d / 2)).unwrap();
        let b = -self.get((.., .., .., d / 2..)).unwrap();
        let ro = a.clone() * cos_freqs.clone() - b.clone() * sin_freqs.clone();
        let co = a * sin_freqs + b * cos_freqs;
        Ok(Tensor::cat([&co, &ro], -1).unwrap())
    }

    /*#[must_use]
    pub fn conv(&self) -> Tensor {
        todo!()
    }*/

    // io
    /// Load module from path. This function will determine the filetype based on file extension.
    ///
    /// # Errors
    ///
    /// Errors if loading from disk failed or if loaded tensors could not be allocated to device.
    #[allow(clippy::missing_panics_doc)]
    pub fn load<Module: FromIterator<(String, Tensor)>>(
        path: impl AsRef<Path>,
    ) -> Result<Module, ZyxError> {
        let e = path
            .as_ref()
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap();
        match e {
            "safetensors" => Self::load_safetensors(path),
            #[cfg(feature = "gguf")]
            "gguf" => Self::load_gguf(path),
            _ => panic!("Unknown file extension. Zyx currently supports only safetensors format."),
        }
    }

    /// Load safetensors module from path
    fn load_safetensors<Module: FromIterator<(String, Tensor)>>(
        path: impl AsRef<Path>,
    ) -> Result<Module, ZyxError> {
        fn read_into_tensor<T: Scalar>(
            //f: &mut std::fs::File,
            mptr: &mut *const u8,
            shape: &[usize],
        ) -> Result<Tensor, ZyxError> {
            // TODO later switch to mmapped memory
            let dtype = T::dtype();
            let n: usize = shape.iter().product();
            let n_bytes = n * dtype.byte_size();
            let x = if cfg!(target_endian = "big") {
                let buf: &[u8] =
                    unsafe { std::slice::from_raw_parts(*mptr, n * dtype.byte_size()) };
                //let mut buf = Vec::with_capacity(n_bytes);
                //unsafe { buf.set_len(n_bytes) }
                //let mut buf: Vec<u8> = vec![u8::zero(); n_bytes];
                //f.read_exact(&mut buf)?;
                let vec: Vec<T> = buf
                    .chunks_exact(dtype.byte_size())
                    .map(Scalar::from_le_bytes)
                    .collect();
                Tensor::from(vec).reshape(shape)
            } else {
                //let mut buf: Vec<T> = Vec::with_capacity(n);
                //unsafe { buf.set_len(n) }
                //let mut buf: Vec<T> = vec![T::zero(); n];
                //f.read_exact(unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), n_bytes) })?;
                let buf: &[T] = unsafe { std::slice::from_raw_parts((*mptr).cast(), n) };
                Tensor::from(buf).reshape(shape)
            };
            //println!("Adding {} bytes", n*dtype.byte_size());
            *mptr = (*mptr).wrapping_add(n_bytes);
            x
        }
        use std::io::Read;
        RT.lock().initialize_devices()?;
        let debug_print: bool = RT.lock().debug_dev();
        let mut f = std::fs::File::open(path)?;
        //println!("File size is {} bytes", f.metadata()?.len());
        let mut header_len = [0u8; 8];
        f.read_exact(&mut header_len)?;
        let n = usize::try_from(u64::from_le_bytes(header_len)).map_err(|e| {
            ZyxError::ParseError(format!(
                "Failed to parse header len in safetensors file. {e}"
            ))
        })?;
        let mut header = vec![0u8; n];
        f.read_exact(&mut header)?;
        let header = core::str::from_utf8(&header)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        let mut text = String::with_capacity(10);
        let mut begin_str = false;
        let mut i = 0;
        let mut tensors = std::collections::HashMap::new();
        let mut dtype = DType::F32;
        let mut shape = vec![1];
        let mut label = String::new();
        let mut metadata = true;
        let progress_bar = if debug_print {
            println!("Loading tensors from safetensors file");
            let bar = indicatif::ProgressBar::new(
                header.chars().filter(|&c| c == '[').count() as u64 / 2,
            );
            bar.set_style(indicatif::ProgressStyle::with_template("[{elapsed_precise}/{duration_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap());
            Some(bar)
        } else {
            None
        };
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        let mut mptr = mmap.as_ptr();
        mptr = mptr.wrapping_add(8 + header.len());
        for x in header.chars() {
            // We skip metadata for now
            if metadata && text.starts_with("__metadata__") {
                if x == '}' {
                    text.clear();
                    begin_str = false;
                    metadata = false;
                }
                continue;
            }
            if ['"', '[', ']'].contains(&x) {
                if begin_str {
                    //std::println!("{text}");
                    if i % 7 == 0 {
                        #[allow(clippy::assigning_clones)]
                        {
                            label = text.clone();
                        }
                    } else if i % 7 == 2 {
                        dtype = DType::from_safetensors(&text)?;
                    } else if i % 7 == 4 {
                        shape = text
                            .split(',')
                            .map(|d| {
                                d.parse::<usize>().map_err(|err| {
                                    ZyxError::ParseError(format!(
                                        "Cannot parse safetensors shape: {err}"
                                    ))
                                })
                            })
                            .collect::<Result<_, ZyxError>>()?;
                    } else if i % 7 == 6 {
                        // TODO assert offsets
                        //println!("Offsets: {text}");
                        let offsets = text
                            .split(',')
                            .map(|offset| {
                                offset.parse::<usize>().map_err(|err| {
                                    ZyxError::ParseError(format!(
                                        "Could not parse safetensors offset: {err}"
                                    ))
                                })
                            })
                            .collect::<Result<Vec<usize>, ZyxError>>()?;
                        //println!("Offsets: {offsets:?}");
                        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
                        if offsets[1] - offsets[0] != bytes {
                            return Err(ZyxError::ParseError(
                                "Safetensors shapes and offsets are incorrect.".into(),
                            ));
                        }
                        if let Some(bar) = &progress_bar {
                            bar.inc(1);
                            bar.set_message(format!("{label}, {shape:?}, {dtype:?}"));
                        }
                        tensors.insert(
                            label.clone(),
                            match dtype {
                                DType::F8 => read_into_tensor::<f8>(&mut mptr, &shape)?,
                                DType::BF16 => read_into_tensor::<bf16>(&mut mptr, &shape)?,
                                DType::F16 => read_into_tensor::<f16>(&mut mptr, &shape)?,
                                DType::F32 => read_into_tensor::<f32>(&mut mptr, &shape)?,
                                DType::F64 => read_into_tensor::<f64>(&mut mptr, &shape)?,
                                DType::U8 => read_into_tensor::<u8>(&mut mptr, &shape)?,
                                DType::I8 => read_into_tensor::<i8>(&mut mptr, &shape)?,
                                DType::I16 => read_into_tensor::<i16>(&mut mptr, &shape)?,
                                DType::I32 => read_into_tensor::<i32>(&mut mptr, &shape)?,
                                DType::U32 => read_into_tensor::<u32>(&mut mptr, &shape)?,
                                DType::I64 => read_into_tensor::<i64>(&mut mptr, &shape)?,
                                DType::Bool => read_into_tensor::<bool>(&mut mptr, &shape)?,
                            },
                        );
                    }
                    i += 1;
                    text.clear();
                    begin_str = false;
                } else {
                    text.clear();
                    begin_str = true;
                }
            } else {
                text.push(x);
            }
        }
        Ok(Module::from_iter(tensors))
    }

    /// Load gguf model
    #[cfg(feature = "gguf")]
    fn load_gguf<M: FromIterator<(String, Tensor)>>(path: impl AsRef<Path>) -> Result<M, ZyxError> {
        let f = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(f);
        let f = gguf::GGUFFile::read(reader.buffer()).unwrap().unwrap();
        println!("{:?}, {:?}", f.header.version, f.header.tensor_count);
        //println!("{}", f.tensors);
        todo!()
    }

    /// All tensor elements as contiguous `le_bytes` vector in row major order
    ///
    /// # Errors
    ///
    /// Returns error if self failed to realize.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, ZyxError> {
        Ok(match self.dtype() {
            DType::BF16 => {
                let data: Vec<bf16> = self.clone().try_into()?;
                data.into_iter().flat_map(bf16::to_le_bytes).collect()
            }
            DType::F8 => {
                //let data: Vec<f8> = self.clone().try_into()?;
                //data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
                todo!()
            }
            DType::F16 => {
                let data: Vec<f16> = self.clone().try_into()?;
                data.into_iter().flat_map(f16::to_le_bytes).collect()
            }
            DType::F32 => {
                let data: Vec<f32> = self.clone().try_into()?;
                data.into_iter().flat_map(f32::to_le_bytes).collect()
            }
            DType::F64 => {
                let data: Vec<f64> = self.clone().try_into()?;
                data.into_iter().flat_map(f64::to_le_bytes).collect()
            }
            DType::U8 => {
                let data: Vec<u8> = self.clone().try_into()?;
                data.into_iter().flat_map(u8::to_le_bytes).collect()
            }
            DType::U32 => {
                let data: Vec<u32> = self.clone().try_into()?;
                data.into_iter().flat_map(u32::to_le_bytes).collect()
            }
            DType::I8 => {
                let data: Vec<i8> = self.clone().try_into()?;
                data.into_iter().flat_map(i8::to_le_bytes).collect()
            }
            DType::I16 => {
                let data: Vec<i16> = self.clone().try_into()?;
                data.into_iter().flat_map(i16::to_le_bytes).collect()
            }
            DType::I32 => {
                let data: Vec<i32> = self.clone().try_into()?;
                data.into_iter().flat_map(i32::to_le_bytes).collect()
            }
            DType::I64 => {
                let data: Vec<i64> = self.clone().try_into()?;
                data.into_iter().flat_map(i64::to_le_bytes).collect()
            }
            DType::Bool => {
                let data: Vec<bool> = self.clone().try_into()?;
                #[allow(clippy::transmute_undefined_repr)]
                unsafe {
                    std::mem::transmute::<Vec<bool>, Vec<u8>>(data)
                }
            }
        })
    }

    // Load tensor from `le_bytes` in row major order
    /*fn from_le_bytes(bytes: &[u8]) -> Result<Tensor, ZyxError> {
        let _ = bytes;
        todo!()
    }*/
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
    /// If self is not float, then cast it to float
    #[must_use]
    fn float_cast(&self) -> Tensor {
        let dtype = self.dtype();
        if !dtype.is_float() {
            return match dtype.byte_size() {
                1 => self.cast(DType::F8),
                2 => self.cast(DType::F16),
                4 => self.cast(DType::F32),
                8 => self.cast(DType::F64),
                _ => panic!(),
            };
        }
        self.clone()
    }

    /// Braodcasts to synchronize shapes and casts to synchronize dtypss
    /// This does both automatic expand AND automatic casting between dtypes.
    // TODO Both of these can be disable by changing a setting in the backend.
    fn broadcast(x: impl Into<Tensor>, y: impl Into<Tensor>) -> Result<(Tensor, Tensor), ZyxError> {
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
            (DType::I16 | DType::I8 | DType::U8 | DType::Bool | DType::F8, DType::BF16) => {
                x = x.cast(DType::BF16);
            }
            (DType::BF16, DType::I16 | DType::I8 | DType::U8 | DType::Bool | DType::F8) => {
                y = y.cast(DType::BF16);
            }
            (DType::I16 | DType::I8 | DType::U8 | DType::Bool, DType::F8) => x = x.cast(DType::F8),
            (DType::F8, DType::I16 | DType::I8 | DType::U8 | DType::Bool) => y = y.cast(DType::F8),
            (
                DType::F8 | DType::BF16 | DType::I16 | DType::I8 | DType::U8 | DType::Bool,
                DType::F16,
            ) => x = x.cast(DType::F16),
            (
                DType::F16,
                DType::F8 | DType::BF16 | DType::I16 | DType::I8 | DType::U8 | DType::Bool,
            ) => y = y.cast(DType::F16),
            (
                DType::F16
                | DType::F8
                | DType::BF16
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U32
                | DType::U8
                | DType::Bool,
                DType::F32,
            ) => x = x.cast(DType::F32),
            (
                DType::F32,
                DType::F16
                | DType::F8
                | DType::BF16
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U32
                | DType::U8
                | DType::Bool,
            ) => y = y.cast(DType::F32),
            (
                DType::F32
                | DType::F16
                | DType::F8
                | DType::BF16
                | DType::I64
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U32
                | DType::U8
                | DType::Bool,
                DType::F64,
            ) => x = x.cast(DType::F64),
            (
                DType::F64,
                DType::F32
                | DType::F16
                | DType::F8
                | DType::BF16
                | DType::I64
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U32
                | DType::U8
                | DType::Bool,
            ) => y = y.cast(DType::F64),
            (DType::BF16, DType::BF16)
            | (DType::F8, DType::F8)
            | (DType::F16, DType::F16)
            | (DType::F32, DType::F32)
            | (DType::F64, DType::F64)
            | (DType::U8, DType::U8)
            | (DType::U32, DType::U32)
            | (DType::I8, DType::I8)
            | (DType::I16, DType::I16)
            | (DType::I32, DType::I32)
            | (DType::I64, DType::I64)
            | (DType::Bool, DType::Bool) => {}
            (dt0, dt1) => {
                return Err(ZyxError::DTypeError(format!("Binary operands have dtypes {dt0} and {dt1}, which could not be implicitly casted. Please explicitly cast them to common dtype.")));
            }
        }
        let x_shape = x.shape();
        let y_shape = y.shape();

        for (&x, &y) in x_shape.iter().rev().zip(y_shape.iter().rev()) {
            if x != y && x != 1 && y != 1 {
                return Err(ZyxError::ShapeError(format!(
                    "Tensor shapes can not be broadcasted: {x_shape:?} and {y_shape:?}"
                )));
            }
        }

        let rx = x_shape.rank();
        let ry = y_shape.rank();
        let mut nx_shape = x_shape.clone();
        let mut ny_shape = y_shape.clone();
        match rx.cmp(&ry) {
            Ordering::Less => {
                nx_shape = core::iter::repeat(1)
                    .take(ry - rx)
                    .chain(nx_shape)
                    .collect();
            }
            Ordering::Greater => {
                ny_shape = core::iter::repeat(1)
                    .take(rx - ry)
                    .chain(ny_shape)
                    .collect();
            }
            Ordering::Equal => {}
        }
        let mut eshape = Vec::new();
        for (x, y) in nx_shape.iter().zip(ny_shape.iter()) {
            eshape.push(*x.max(y));
        }
        if x_shape != eshape {
            x = x.expand(&eshape)?;
        }
        //println!("Second broadcast operand {y}");
        //println!("{x_shape:?}, {y_shape:?}, {eshape:?}");
        //println!("After reshape second broadcast operand {y}");
        //Tensor::plot_graph([], "graph");
        if y_shape != eshape {
            y = y.expand(&eshape)?;
        }
        //println!("Second broadcast operand {y}");
        //println!("Broadcasted to {eshape:?}");
        //println!("y shape {:?}", y.shape());
        Ok((x, y))
    }

    // Calculate shape for reduce which keeps reduced dims set to 1
    fn reduce_kd_shape(&self, axes: impl IntoIterator<Item = isize>) -> Vec<usize> {
        let mut shape = self.shape();
        for a in into_axes(axes, shape.len()).unwrap() {
            shape[a] = 1;
        }
        shape
    }

    pub(super) const fn id(&self) -> TensorId {
        self.id
    }
}

impl TryFrom<Tensor> for bf16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [bf16::ZERO];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [f8::ZERO];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [f16::ZERO];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0.];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for f64 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0.];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for u8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for u32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i8 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i16 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i32 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for i64 {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl TryFrom<Tensor> for bool {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [false];
        RT.lock().load(value.id, &mut data)?;
        Ok(data[0])
    }
}

impl<T: Scalar> TryFrom<Tensor> for Vec<T> {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let numel = value.numel();
        let mut data = vec![T::zero(); numel];
        RT.lock().load(value.id, &mut data)?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize> TryFrom<Tensor> for [T; D0] {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [T::zero(); D0];
        RT.lock().load(value.id, &mut data)?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> TryFrom<Tensor> for [[T; D1]; D0] {
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[T::zero(); D1]; D0];
        RT.lock().load(value.id, data.as_flattened_mut())?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> TryFrom<Tensor>
    for [[[T; D2]; D1]; D0]
{
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[[T::zero(); D2]; D1]; D0];
        RT.lock()
            .load(value.id, data.as_flattened_mut().as_flattened_mut())?;
        Ok(data)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize> TryFrom<Tensor>
    for [[[[T; D3]; D2]; D1]; D0]
{
    type Error = ZyxError;
    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let mut data = [[[[T::zero(); D3]; D2]; D1]; D0];
        RT.lock().load(
            value.id,
            data.as_flattened_mut()
                .as_flattened_mut()
                .as_flattened_mut(),
        )?;
        Ok(data)
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
        let precision = f.precision().unwrap_or(3);
        let x = self.clone();
        let res = match self.dtype() {
            DType::BF16 => {
                let data: Result<Vec<bf16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F8 => {
                let data: Result<Vec<f8>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F16 => {
                let data: Result<Vec<f16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            DType::F32 => {
                let data: Result<Vec<f32>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f32 tensor failed to realize {e:?}"),
                }
            }
            DType::F64 => {
                let data: Result<Vec<f64>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f64 tensor failed to realize {e:?}"),
                }
            }
            DType::U8 => {
                let data: Result<Vec<u8>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("u8 tensor failed to realize {e:?}"),
                }
            }
            DType::U32 => {
                let data: Result<Vec<u32>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("u32 tensor failed to realize {e:?}"),
                }
            }
            DType::I8 => {
                let data: Result<Vec<i8>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I16 => {
                let data: Result<Vec<i16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I32 => {
                let data: Result<Vec<i32>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::I64 => {
                let data: Result<Vec<i64>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
                }
            }
            DType::Bool => {
                let data: Result<Vec<bool>, _> = x.try_into();
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

/// Into isize range, used for indexing
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
        #[allow(clippy::range_plus_one)]
        {
            *self.start()..*self.end() + 1
        }
    }
}

impl IntoRange for RangeToInclusive<isize> {
    fn into_range(self) -> Range<isize> {
        #[allow(clippy::range_plus_one)]
        {
            0..self.end + 1
        }
    }
}

impl IntoRange for Range<isize> {
    fn into_range(self) -> Range<isize> {
        self
    }
}

impl IntoRange for isize {
    fn into_range(self) -> Range<isize> {
        #[allow(clippy::range_plus_one)]
        {
            self..self + 1
        }
    }
}

/// Implemented for objects that can be used to index tensors.
pub trait IntoIndex {
    /// Convert self to tensor index.
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>>;
}

impl IntoIndex for Vec<Range<isize>> {
    fn into_index(self) -> impl IntoIterator<Item = Range<isize>> {
        self.into_iter()
    }
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

impl From<&Tensor> for Tensor {
    fn from(value: &Tensor) -> Self {
        value.clone()
    }
}

impl<T: Scalar> From<T> for Tensor {
    fn from(value: T) -> Self {
        Tensor {
            id: RT.lock().variable(vec![1], &[value]).unwrap(),
        }
    }
}

impl<T: Scalar> From<Vec<T>> for Tensor {
    fn from(data: Vec<T>) -> Self {
        Tensor {
            id: RT.lock().variable(vec![data.len()], &data).unwrap(),
        }
    }
}

impl<T: Scalar> From<&Vec<T>> for Tensor {
    fn from(data: &Vec<T>) -> Self {
        Tensor {
            id: RT.lock().variable(vec![data.len()], data).unwrap(),
        }
    }
}

impl<T: Scalar> From<&[T]> for Tensor {
    fn from(data: &[T]) -> Self {
        let n = data.len();
        Tensor {
            id: RT.lock().variable(vec![n], data).unwrap(),
        }
    }
}

impl<T: Scalar, const D0: usize> From<[T; D0]> for Tensor {
    fn from(data: [T; D0]) -> Self {
        Tensor {
            id: RT.lock().variable(vec![D0], &data).unwrap(),
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> From<[[T; D1]; D0]> for Tensor {
    fn from(data: [[T; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0].as_ptr(), D0 * D1) };
        Tensor {
            id: RT.lock().variable(vec![D0, D1], data).unwrap(),
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> From<[[[T; D2]; D1]; D0]>
    for Tensor
{
    fn from(data: [[[T; D2]; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0][0].as_ptr(), D0 * D1 * D2) };
        Tensor {
            id: RT.lock().variable(vec![D0, D1, D2], data).unwrap(),
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    From<[[[[T; D3]; D2]; D1]; D0]> for Tensor
{
    fn from(data: [[[[T; D3]; D2]; D1]; D0]) -> Self {
        let data =
            unsafe { core::slice::from_raw_parts(data[0][0][0].as_ptr(), D0 * D1 * D2 * D3) };
        Tensor {
            id: RT.lock().variable(vec![D0, D1, D2, D3], data).unwrap(),
        }
    }
}

impl PartialEq<f32> for Tensor {
    fn eq(&self, other: &f32) -> bool {
        self.clone()
            .try_into()
            .map_or(false, |data| Scalar::is_equal(data, *other))
    }
}

impl PartialEq<f64> for Tensor {
    fn eq(&self, other: &f64) -> bool {
        self.clone()
            .try_into()
            .map_or(false, |data| Scalar::is_equal(data, *other))
    }
}

impl PartialEq<i32> for Tensor {
    fn eq(&self, other: &i32) -> bool {
        self.clone()
            .try_into()
            .map_or(false, |data| Scalar::is_equal(data, *other))
    }
}

impl<T: Scalar, const D0: usize> PartialEq<[T; D0]> for Tensor {
    fn eq(&self, other: &[T; D0]) -> bool {
        if self.shape() != [D0] {
            return false;
        }
        if let Ok(data) = self.clone().try_into() {
            let data: [T; D0] = data;
            for (x, y) in data.into_iter().zip(other) {
                if !Scalar::is_equal(x, *y) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> PartialEq<[[T; D1]; D0]> for Tensor {
    fn eq(&self, other: &[[T; D1]; D0]) -> bool {
        if self.shape() != [D0, D1] {
            return false;
        }
        self.clone()
            .try_into()
            .map_or(false, |data: [[T; D1]; D0]| &data == other)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> PartialEq<[[[T; D2]; D1]; D0]>
    for Tensor
{
    fn eq(&self, other: &[[[T; D2]; D1]; D0]) -> bool {
        if self.shape() != [D0, D1, D2] {
            return false;
        }
        self.clone()
            .try_into()
            .map_or(false, |data: [[[T; D2]; D1]; D0]| &data == other)
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PartialEq<[[[[T; D3]; D2]; D1]; D0]> for Tensor
{
    fn eq(&self, other: &[[[[T; D3]; D2]; D1]; D0]) -> bool {
        if self.shape() != [D0, D1, D2, D3] {
            return false;
        }
        self.clone()
            .try_into()
            .map_or(false, |data: [[[[T; D3]; D2]; D1]; D0]| &data == other)
    }
}

impl<IT: Into<Tensor>> Add<IT> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().add(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Add<IT> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().add(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Sub<IT> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().sub(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Sub<IT> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().sub(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Mul<IT> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into();
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        //println!("Multiply by {y}");
        let tensor = Tensor {
            id: RT.lock().mul(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Mul<IT> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into();
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        // We have to do this using temporary variable,
        // otherwise rust drops tensor before dropping mutexguard,
        // causing deadlock. But with temporary variable
        // it works. Welcome to most beloved language of all time.
        let tensor = Tensor {
            id: RT.lock().mul(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Div<IT> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().div(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> Div<IT> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().div(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> BitOr<IT> for Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitor(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> BitOr<IT> for &Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitor(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> BitXor<IT> for Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitxor(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> BitXor<IT> for &Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitxor(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> BitAnd<IT> for Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitand(x.id, y.id),
        };
        tensor
    }
}

impl<IT: Into<Tensor>> BitAnd<IT> for &Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitand(x.id, y.id),
        };
        tensor
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
                rhs.$fn_name(self)
            }
        }

        impl $trait<&Tensor> for $type {
            type Output = Tensor;
            fn $fn_name(self, rhs: &Tensor) -> Self::Output {
                rhs.$fn_name(self)
            }
        }
    };
}

impl_trait!(Add for bf16, add);
impl_trait!(Add for f8, add);
impl_trait!(Add for f16, add);
impl_trait!(Add for f32, add);
impl_trait!(Add for f64, add);
impl_trait!(Add for u8, add);
impl_trait!(Add for u32, add);
impl_trait!(Add for i8, add);
impl_trait!(Add for i16, add);
impl_trait!(Add for i32, add);
impl_trait!(Add for i64, add);
impl_trait!(Add for bool, add);

impl_trait!(Sub for bf16, sub);
impl_trait!(Sub for f8, sub);
impl_trait!(Sub for f16, sub);
impl_trait!(Sub for f32, sub);
impl_trait!(Sub for f64, sub);
impl_trait!(Sub for u8, sub);
impl_trait!(Sub for u32, sub);
impl_trait!(Sub for i8, sub);
impl_trait!(Sub for i16, sub);
impl_trait!(Sub for i32, sub);
impl_trait!(Sub for i64, sub);
impl_trait!(Sub for bool, sub);

impl_trait!(Mul for bf16, mul);
impl_trait!(Mul for f8, mul);
impl_trait!(Mul for f16, mul);
impl_trait!(Mul for f32, mul);
impl_trait!(Mul for f64, mul);
impl_trait!(Mul for u8, mul);
impl_trait!(Mul for u32, mul);
impl_trait!(Mul for i8, mul);
impl_trait!(Mul for i16, mul);
impl_trait!(Mul for i32, mul);
impl_trait!(Mul for i64, mul);
impl_trait!(Mul for bool, mul);

impl_trait!(Div for bf16, div);
impl_trait!(Div for f8, div);
impl_trait!(Div for f16, div);
impl_trait!(Div for f32, div);
impl_trait!(Div for f64, div);
impl_trait!(Div for u8, div);
impl_trait!(Div for u32, div);
impl_trait!(Div for i8, div);
impl_trait!(Div for i16, div);
impl_trait!(Div for i32, div);
impl_trait!(Div for i64, div);
impl_trait!(Div for bool, div);

impl_trait!(BitXor for bf16, bitxor);
impl_trait!(BitXor for f8, bitxor);
impl_trait!(BitXor for f16, bitxor);
impl_trait!(BitXor for f32, bitxor);
impl_trait!(BitXor for f64, bitxor);
impl_trait!(BitXor for u8, bitxor);
impl_trait!(BitXor for u32, bitxor);
impl_trait!(BitXor for i8, bitxor);
impl_trait!(BitXor for i16, bitxor);
impl_trait!(BitXor for i32, bitxor);
impl_trait!(BitXor for i64, bitxor);
impl_trait!(BitXor for bool, bitxor);

impl_trait!(BitOr for bf16, bitor);
impl_trait!(BitOr for f8, bitor);
impl_trait!(BitOr for f16, bitor);
impl_trait!(BitOr for f32, bitor);
impl_trait!(BitOr for f64, bitor);
impl_trait!(BitOr for u8, bitor);
impl_trait!(BitOr for u32, bitor);
impl_trait!(BitOr for i8, bitor);
impl_trait!(BitOr for i16, bitor);
impl_trait!(BitOr for i32, bitor);
impl_trait!(BitOr for i64, bitor);
impl_trait!(BitOr for bool, bitor);

impl_trait!(BitAnd for bf16, bitand);
impl_trait!(BitAnd for f8, bitand);
impl_trait!(BitAnd for f16, bitand);
impl_trait!(BitAnd for f32, bitand);
impl_trait!(BitAnd for f64, bitand);
impl_trait!(BitAnd for u8, bitand);
impl_trait!(BitAnd for u32, bitand);
impl_trait!(BitAnd for i8, bitand);
impl_trait!(BitAnd for i16, bitand);
impl_trait!(BitAnd for i32, bitand);
impl_trait!(BitAnd for i64, bitand);
impl_trait!(BitAnd for bool, bitand);

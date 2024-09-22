//! Tensor
//!
//! Tensors are at the core of all machine learning.

use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::shape::{to_axis, IntoAxes, IntoPadding, IntoShape};
use core::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display};
use std::iter::repeat;
use std::ops::{
    Add, BitAnd, BitOr, BitXor, Bound, Div, Mul, Neg, Not, Range, RangeBounds, RangeFrom,
    RangeFull, RangeInclusive, RangeTo, RangeToInclusive, Sub,
};
use std::path::Path;

use crate::runtime::ZyxError;
use crate::RT;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

pub(crate) type TensorId = usize;

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
    #[must_use]
    pub fn detach(self) -> Result<Tensor, ZyxError> {
        // TODO remove realization from here
        let shape = self.shape();
        let dtype = self.dtype();
        match dtype {
            #[cfg(feature = "half")]
            DType::F16 => {
                let data: Vec<f16> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            #[cfg(feature = "half")]
            DType::BF16 => {
                let data: Vec<bf16> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::F32 => {
                let data: Vec<f32> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::F64 => {
                let data: Vec<f64> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let data: Vec<Complex<f32>> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let data: Vec<Complex<f64>> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::U8 => {
                let data: Vec<u8> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::I8 => {
                let data: Vec<i8> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::I16 => {
                let data: Vec<i16> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::I32 => {
                let data: Vec<i32> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::I64 => {
                let data: Vec<i64> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
            }
            DType::Bool => {
                let data: Vec<bool> = self.try_into()?;
                Ok(Tensor::from(data).reshape(shape))
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
    /// For more look at ENV_VARS.md
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
    #[cfg(feature = "rand")]
    pub fn manual_seed(seed: u64) {
        RT.lock().manual_seed(seed);
    }

    /// Create random value in range 0f..1f with float dtype
    /// or 0..int::MAX if it is integer
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn rand(shape: impl IntoShape, dtype: DType) -> Result<Tensor, ZyxError> {
        const SEED: u64 = 69420;
        use std::i32;

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
                DType::F32 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f32> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::F64 => {
                    let range = Uniform::new(0., 1.);
                    let data: Vec<f64> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                _ => panic!(),
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
                    let data: Vec<u8> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I8 => {
                    let range = Uniform::new(0, i8::MAX);
                    let data: Vec<i8> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I16 => {
                    let range = Uniform::new(0, i16::MAX);
                    let data: Vec<i16> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I32 => {
                    let range = Uniform::new(0, i32::MAX);
                    let data: Vec<i32> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                DType::I64 => {
                    let range = Uniform::new(0, i64::MAX);
                    let data: Vec<i64> = (0..n).map(|_| rng.sample(&range)).collect();
                    Ok(Tensor {
                        id: rt.variable(shape, &data)?,
                    })
                }
                _ => panic!(),
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
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn randn(shape: impl IntoShape, dtype: DType) -> Result<Tensor, ZyxError> {
        // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        // src = Tensor.rand((2, *argfix(*shape)), **{**kwargs, "dtype": dtypes.float32})
        // return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

        let shape: Vec<usize> = [2].into_iter().chain(shape.into_shape()).collect();
        let src = Tensor::rand(shape, dtype)?;
        let mut x = src.get(0);
        x = x.mul(Tensor::constant(2f32 * std::f32::consts::PI));
        //panic!();
        x = x.cos();
        let mut y = Tensor::constant(1f32) - src.get(1);
        //println!("{y} minus");
        y = y.ln().mul(Tensor::constant(-2f32)).sqrt();
        //println!("{y}");
        Ok(x.mul(y).cast(dtype))
    }

    /// Create tensor sampled from uniform distribution
    /// Start of the range must be less than the end of the range.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn uniform<T: Scalar>(
        shape: impl IntoShape,
        range: impl core::ops::RangeBounds<T>,
    ) -> Result<Tensor, ZyxError> {
        use core::ops::Bound;
        let low = match range.start_bound() {
            Bound::Included(value) => *value,
            Bound::Excluded(value) => *value,
            Bound::Unbounded => T::min_value(),
        };
        let high = match range.end_bound() {
            Bound::Included(value) => *value,
            Bound::Excluded(value) => *value,
            Bound::Unbounded => T::max_value(),
        };
        Ok(Tensor::rand(shape, T::dtype())? * high.sub(low) + low)
    }

    /// Create tensor sampled from kaiming uniform distribution.
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn kaiming_uniform<T: Scalar>(shape: impl IntoShape, a: T) -> Result<Tensor, ZyxError> {
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
    pub fn full(shape: impl IntoShape, value: impl Scalar) -> Result<Tensor, ZyxError> {
        return Ok(Tensor {
            id: RT.lock().full(shape.into_shape().collect(), value)?,
        });
    }

    /// Create square tensor with ones on the main diagonal and all other values set to zero.
    #[must_use]
    pub fn eye(n: usize, dtype: DType) -> Tensor {
        return Tensor::ones(vec![n, 1], dtype)
            .pad_zeros([(0, n as isize)])
            .reshape([n + 1, n])
            .get((..-1, ..));
    }

    /// Arange method, create range from start, stop, step
    #[must_use]
    pub fn arange<T: Scalar>(start: T, stop: T, step: T) -> Result<Tensor, ZyxError> {
        // if (stop-start)/step <= 0: return Tensor([], dtype=dtype, **kwargs)
        // return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs)._cumsum() + (start - step)).cast(dtype)
        let n: i64 = stop.sub(start).div(step).cast();
        let n = n as usize;
        //println!("Shape {n}");
        let m = start.sub(step);
        let x = Tensor::full(n, step)?;
        //println!("{x}");
        let x = x.cumsum(0);
        Ok(x + m)
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

    /// Applies element-wise, CELU(x)=max⁡(0,x)+min⁡(0,α∗(exp⁡(x/α)−1)).
    #[must_use]
    pub fn celu(&self, alpha: impl Scalar) -> Tensor {
        return self.relu() - (-((self / alpha).exp() - 1) * alpha).relu();
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
    #[cfg(feature = "rand")]
    #[must_use]
    pub fn dropout<P: Scalar>(&self, probability: P) -> Result<Tensor, ZyxError> {
        let dtype = P::dtype();
        assert!(
            dtype.is_float(),
            "Dropout only works on floating dtype probability."
        );
        // TODO fix this for training (dropout in training is just scaling)
        Ok(Tensor::from(probability).cmplt(Tensor::rand(self.shape(), dtype)?)? * self)
    }

    /// Applies the Exponential Linear Unit function element-wise.
    ///
    /// The ELU function is defined as:
    /// ```
    /// f(x) = x if x > 0
    ///       α(e^x - 1) otherwise
    /// ```
    /// where `α` is a given scaling factor. This function helps mitigate the "dying ReLU" problem.
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
    #[must_use]
    pub fn gelu(&self) -> Result<Tensor, ZyxError> {
        Ok(self * 0.5f32
            * (((self + self.pow(3f32)? * 0.044_715f32) * (2f32 / core::f32::consts::PI).sqrt())
                .tanh()
                + 1f32))
    }

    /// Applies the Leaky ReLU activation function element-wise.
    ///
    /// This function computes the Leaky ReLU of each element in the input tensor. If the element is greater than
    /// or equal to zero, it returns the element itself; otherwise, it returns `neg_slope * element`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    /// * neg_slope: The negative slope coefficient (`α` in the formula) for the Leaky ReLU function.
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
    /// The Mish activation function is a continuous, non-monotonic function that behaves like ReLU for positive inputs and like sigmoid for negative inputs. It is defined as `x * tanh(softplus(x))`.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `Mish(input_element)`.
    #[must_use]
    pub fn mish(&self) -> Result<Tensor, ZyxError> {
        Ok(self * self.softplus(1, 20)?.tanh())
    }

    /// Computes the quick GELU activation function for each element in the input tensor.
    ///
    /// The QuickGELU activation function is an approximation of the Gaussian Error Linear Unit (GELU) function that uses a sigmoid function to compute the approximation. It is defined as `x * sigmoid(1.702 * x)`.
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

    /// Applies the Rectified Linear Unit (ReLU) activation function to each element in the input tensor.
    ///
    /// The ReLU function returns `max(0, x)`, i.e., it replaces negative values with zero and leaves positive values unchanged. This makes it a popular choice for use in hidden layers of neural networks due to its simplicity and effectiveness.
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
    /// The Selu activation function is designed to maintain the mean and variance of the activations approximately constant when training deep neural networks with residual connections. It combines the benefits of both ReLU and sigmoid functions, making it a good choice for certain types of problems.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:** A new tensor with the same shape as the input, but with each element computed as `Selu(input_element)`.
    #[must_use]
    pub fn selu(&self) -> Tensor {
        1.0507009873554804934193349852946f32
            * (self.relu()
                - (1.6732632423543772848170429916717f32
                    * (Tensor::ones(1, self.dtype()) - self.exp()))
                .relu())
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
        return &exp_x / (&one + &exp_x);
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
    /// The softplus function returns `log(exp(x) + 1)` for inputs greater than the threshold, and x otherwise. This function is useful for bounding outputs between zero and infinity when applying the ReLU function.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    /// * beta: A scalar multiplier applied to each element of the input tensor before comparison with the threshold.
    /// * threshold: The threshold value below which the input is returned unchanged, and above which the softplus function is applied.
    ///
    /// **Returns:** A new tensor with the same shape as the input, where each element is computed according to the softplus function with the given beta and threshold.
    #[must_use]
    pub fn softplus(&self, beta: impl Scalar, threshold: impl Scalar) -> Result<Tensor, ZyxError> {
        let x = self * beta;
        x.cmplt(threshold)?.where_(((x).exp() + 1).ln() * beta.reciprocal(), x)
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
    /// The Swish function returns `x * sigmoid(x)`, where `sigmoid(x) = 1 / (1 + exp(-x))`. This function is useful for various deep learning applications, as it has been shown to improve convergence speed and generalization performance compared to other activation functions like ReLU.
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
        let x = (self + self).sigmoid();
        (&x + &x) - Tensor::constant(1).cast(self.dtype())
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
    /// let t = Tensor::zeros((2, 3));
    /// assert_eq!(t.expand((4, 2, 3)).shape(), &[4, 2, 3]);
    /// ```
    #[must_use]
    pub fn expand(&self, shape: impl IntoShape) -> Tensor {
        assert!(shape.rank() > 0);
        let mut sh = self.shape();
        let shape: Vec<usize> = shape.into_shape().collect();
        //println!("Expand to {shape:?}");
        assert!(shape.rank() >= sh.rank());
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
                    );
                }
            }
            let x = self.reshape(sh);
            let id = RT.lock().expand(x.id, shape);
            return Tensor { id };
        };
        return Tensor {
            id: RT.lock().expand(self.id, shape),
        };
    }

    /// Permutes the axes of this tensor.
    ///
    /// This function rearranges the dimensions of the tensor according to the provided axes. The axes must be a permutation of the original axes, i.e., they must contain each index once and only once. If the axes have a different length than the rank of the tensor, a panic will occur with an appropriate error message.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    /// let t = Tensor::rand((3, 4)).unwrap();
    /// let p = [1, 0];
    /// let permuted_t = t.permute(p); // Results in a tensor with axes (4, 3)
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the length of `axes` is not equal to the rank of this tensor.
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

    /// Creates a new tensor by padding zeros around this tensor based on the specified padding configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from([1, 2, 3]);
    /// let padded = t.pad_zeros(1).into_shape((5,))?;
    /// assert_eq!(padded, [0., 1., 2., 3., 0.]);
    ///
    /// let padded = t.pad_zeros([(1, 2)]);
    /// assert_eq!(padded.shape(), &[5]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the padding configuration is invalid.
    #[must_use]
    pub fn pad_zeros(&self, padding: impl IntoPadding) -> Tensor {
        // TODO asserts
        let padding = padding.into_padding();
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
            assert!((total as usize) < shape[rank-i-1]);
        }
        return Tensor {
            id: RT.lock().pad_zeros(self.id, padding),
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
    pub fn pad(
        &self,
        padding: impl IntoPadding,
        value: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
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
                    let x: bf16 = value.clone().try_into()?;
                    x == bf16::ZERO
                }
                #[cfg(feature = "half")]
                DType::F16 => {
                    let x: f16 = value.clone().try_into()?;
                    x == f16::ZERO
                }
                DType::F32 => {
                    let x: f32 = value.clone().try_into()?;
                    x == 0.
                }
                DType::F64 => {
                    let x: f64 = value.clone().try_into()?;
                    x == 0.
                }
                #[cfg(feature = "complex")]
                DType::CF32 => {
                    let x: Complex<f32> = value.clone().try_into()?;
                    x == Complex::new(0., 0.)
                }
                #[cfg(feature = "complex")]
                DType::CF64 => {
                    let x: Complex<f64> = value.clone().try_into()?;
                    x == Complex::new(0., 0.)
                }
                DType::U8 => {
                    let x: u8 = value.clone().try_into()?;
                    x == 0
                }
                DType::I8 => {
                    let x: i8 = value.clone().try_into()?;
                    x == 0
                }
                DType::I16 => {
                    let x: i16 = value.clone().try_into()?;
                    x == 0
                }
                DType::I32 => {
                    let x: i32 = value.clone().try_into()?;
                    x == 0
                }
                DType::I64 => {
                    let x: i64 = value.clone().try_into()?;
                    x == 0
                }
                DType::Bool => {
                    let x: bool = value.clone().try_into()?;
                    x == false
                }
            }
        {
            Ok(t0)
        } else {
            let ones = Tensor::ones(sh.clone(), dtype);
            let zeros = Tensor::zeros(sh, self.dtype());
            Ok(t0 + ones.pad_zeros(padding).where_(zeros, value)?)
        }
    }

    /// Applies a new shape to this tensor while preserving its total number of elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    /// let t = Tensor::from([1, 2, 3, 4]);
    /// assert_eq!(t.reshape((2, 2)), [[1, 2], [3, 4]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the product of the new shape is not equal to the number of elements in this tensor.
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

    /// An alias to reshape
    #[must_use]
    pub fn view(&self, shape: impl IntoShape) -> Tensor {
        let shape: Vec<usize> = shape.into_shape().collect();
        assert_eq!(
            shape.iter().product::<usize>(),
            self.numel(),
            "Invalid view of {:?} into {:?}",
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
    /// println!("{y}");
    /// ```
    ///
    /// # Returns
    ///
    /// The resulting tensor after computing the natural logarithm of the softmax of `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if any of the specified axes are out-of-bounds for the input tensor.
    pub fn ln_softmax(&self, axes: impl IntoAxes) -> Tensor {
        let m = self - self.max_kd(axes.clone());
        &m - m.exp().sum_kd(axes).ln()
    }

    /// Returns a new tensor containing the maximum value along the specified axes.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to compute the maximum. This can be any type that implements `IntoAxes`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor
    /// let arr = Tensor::from([1, 2, 3, 4]);
    /// assert_eq!(arr.max(0), [4]);
    /// assert_eq!(arr.max(1), [2, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the axes contain duplicates.
    #[must_use]
    pub fn max(&self, axes: impl IntoAxes) -> Tensor {
        let rank = self.rank();
        let axes: Vec<usize> = axes.into_axes(rank).collect();
        let mut unique = BTreeSet::new();
        for a in &axes {
            assert!(unique.insert(a), "Axes contain duplicates.");
        }
        Tensor {
            id: RT.lock().max_reduce(self.id, axes),
        }
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
    /// assert_eq!(a.max_kd(&[0]), &[[4]]);
    /// ```
    ///
    #[must_use]
    pub fn max_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.max(axes.clone()).reshape(self.reduce_kd_shape(axes))
    }

    /// Calculates the mean of a tensor along specified axes.
    ///
    /// This function computes the sum of all elements in the tensor along the specified axes and then divides by the product of their sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use zyx::Tensor;
    ///
    /// let arr = Tensor::eye(3, DType::F32);
    /// assert_eq!(arr.mean(0, &[1.0, 1.0, 1.0]));
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the tensor is empty.
    #[must_use]
    pub fn mean(&self, axes: impl IntoAxes) -> Tensor {
        let shape = self.shape();
        self.sum(axes.clone())
            / axes
                .into_axes(shape.rank())
                .map(|a| shape[a])
                .product::<usize>() as i64
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
    /// assert_eq!(a.mean_kd(0), [2.5]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the input tensor is empty.
    #[must_use]
    pub fn mean_kd(&self, axes: impl IntoAxes) -> Tensor {
        self.mean(axes.clone()).reshape(self.reduce_kd_shape(axes))
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
    /// assert_eq!(arr.product(1), [3., 8.]);
    /// ```
    #[must_use]
    pub fn product(&self, axes: impl IntoAxes) -> Tensor {
        self.ln().sum(axes).exp()
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
    /// assert_eq!(a.std(()), 1.5);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor is empty.
    ///
    #[must_use]
    pub fn std(&self, axes: impl IntoAxes) -> Result<Tensor, ZyxError> {
        Ok(self.var(axes)?.sqrt())
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
    /// let std_kd = t.std_kd([0, 1]);
    /// assert_eq!(std_kd.shape(), [1, 2]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the input tensor has no elements.
    #[must_use]
    pub fn std_kd(&self, axes: impl IntoAxes) -> Result<Tensor, ZyxError> {
        Ok(self.std(axes.clone())?.reshape(self.reduce_kd_shape(axes)))
    }

    /// Sum reduce. Removes tensor dimensions.
    /// Equivalent to pytorch sum(axes, keepdim=False)
    /// If you want to keep reduce dimensions, see [sum_kd](Tensor::sum_kd)
    /// Passing empty axes executes reduce across all dimensions and result will have shape `[1]`
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

    /// Comulative sum along axis.
    #[must_use]
    pub fn cumsum(&self, axis: isize) -> Tensor {
        let axis = to_axis(axis, self.rank());
        let pl_sz = (self.shape()[axis] - 1) as isize;
        let k = self.shape()[axis];
        let axis = axis as isize;
        let mut x = self.transpose(axis, -1);
        x = x.pad_zeros([(pl_sz, 0)]);
        //println!("{x:?} padded");
        x = x.pool(k, 1, 1);
        //println!("{x:?} pooled");
        x = x.sum(-1);
        //println!("{x:?} summed");
        x = x.transpose(axis, -1);
        //println!("{x:?} transposed");
        return x;
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
    /// let sm = t.softmax(0);
    /// assert_eq!(sm, [0.0900305748, 0.2447281546, 0.6652412706]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor is empty.
    #[must_use]
    pub fn softmax(&self, axes: impl IntoAxes) -> Tensor {
        let e = (self - self.max_kd(axes.clone())).exp();
        &e / e.sum_kd(axes)
    }

    /// Calculates the variance of this tensor along the specified axes.
    ///
    /// This function first computes the mean of the tensor along the provided axes,
    /// then subtracts this mean from each element in the tensor, squares the result,
    /// and finally sums these squared differences along the same axes to obtain the variance.
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
    /// let arr = Tensor::from([[1, 2], [3, 4]]);
    /// let var = arr.var(0); // Compute variance along rows (axis=0)
    /// assert_eq!(var, [[5.0, 2.5]]); // Expected output: [[5.0, 2.5]]
    ///
    /// let var = arr.var(1); // Compute variance along columns (axis=1)
    /// assert_eq!(var, [[2.5], [2.5]]); // Expected output: [[2.5], [2.5]]
    /// ```
    #[must_use]
    pub fn var(&self, axes: impl IntoAxes) -> Result<Tensor, ZyxError> {
        Ok((self - self.mean(axes.clone())).pow(2)?.sum(axes))
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
    /// let a = Tensor::from([[1., 2., 3.], [4., 5., 6.]]);
    /// assert_eq!(a.var_kd(0), 1.5);
    /// ```
    #[must_use]
    pub fn var_kd(&self, axes: impl IntoAxes) -> Result<Tensor, ZyxError> {
        Ok(self.var(axes.clone())?.reshape(self.reduce_kd_shape(axes)))
    }

    // index
    /// Get function
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
    /// let arr = Tensor::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape([3, 3]);
    /// assert_eq!(arr.diagonal(), [[1, 0, 0], [0, 5, 0], [0, 0, 9]]); // diagonal elements are [1, 5, 9]
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the input tensor has fewer than two dimensions.
    #[must_use]
    pub fn diagonal(&self) -> Tensor {
        let Some(&n) = self.shape().last() else {
            panic!("Shape in invalid state. Internal bug.")
        };
        self.flatten(..)
            .pad_zeros([(0, n as isize)])
            .reshape([n, n + 1])
            .get((.., 0))
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
    /// assert_eq!(a.cmplt(b), [1., 1., 1.]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function panics if the tensors have different shapes.
    #[must_use]
    pub fn cmplt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self, rhs)?;
        Ok(Tensor {
            id: RT.lock().cmplt(x.id, y.id),
        })
    }

    /// Elementwise maximum between two tensors.
    #[must_use]
    pub fn maximum(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self, rhs)?;
        Ok(Tensor {
            id: RT.lock().maximum(x.id, y.id),
        })
    }

    /// Matmul and dot
    #[must_use]
    pub fn dot(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
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
        Ok((self.reshape(x_shape) * y.reshape(y_shape))
            .sum(-1)
            .reshape(
                xshape[0..xshape.len() - 1]
                    .iter()
                    .copied()
                    .chain([yshape[yshape.len() - 2]])
                    .collect::<Vec<usize>>(),
            ))
    }

    /// Matmul is just alias to dot
    #[must_use]
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
    /// assert_eq!(arr.pow(2.0), [1.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the exponent tensor contains any invalid or non-finite values.
    ///
    /// # Returns
    ///
    /// A new tensor where each element is the result of raising the corresponding element in `self` to the power of `exponent`.
    #[must_use]
    pub fn pow(&self, exponent: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self, exponent)?;
        Ok(Tensor {
            id: RT.lock().pow(x.id, y.id),
        })
    }

    /// Returns ones where self is true and zeros where it is false.
    #[must_use]
    pub fn nonzero(&self) -> Tensor {
        Tensor {
            id: RT.lock().nonzero(self.id),
        }
    }

    // ternary
    /// Where operation. Replaces elementwise true values with if_true and false values with if_false.
    #[must_use]
    pub fn where_(&self, if_true: impl Into<Tensor>, if_false: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self, if_true)?;
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
    /// assert_eq!(input.cross_entropy_loss(target, ()), -0.69314718);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor and target tensor have different shapes.
    #[must_use]
    pub fn cross_entropy_loss(&self, target: impl Into<Tensor>, axes: impl IntoAxes) -> Tensor {
        self.ln_softmax(axes) * target
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
    /// let self_tensor = Tensor::from(&[1.0, 2.0, 3.0]);
    /// let target_tensor = Tensor::from(&[2.0, 3.0, 4.0]);
    ///
    /// assert_eq!(self_tensor.l1_loss(target_tensor), Tensor::from(&[1.0, 1.0, 1.0]));
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
    /// assert_eq!(input.mse_loss(target), Tensor::from([1.0, 1.0]));
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor and target tensor have different shapes.
    pub fn mse_loss(&self, target: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        (self - target).pow(2)
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
    /// # Panics
    ///
    /// This function panics if the input tensors have different shapes.
    #[must_use]
    pub fn cosine_similarity(&self, rhs: impl Into<Tensor>, eps: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let rhs: Tensor = rhs.into();
        let eps: Tensor = eps.into();
        let x = self.pow(2)?.sqrt() * rhs.pow(2)?.sqrt();
        Ok(self * rhs / x.cmplt(&eps)?.where_(eps, x)?)
    }

    // misc
    /// Flatten. Joins axes into one dimension,
    #[must_use]
    pub fn flatten(&self, axes: impl RangeBounds<isize>) -> Tensor {
        let shape = self.shape();
        let rank = shape.len();
        let start_dim = to_axis(
            match axes.start_bound() {
                Bound::Included(dim) => *dim,
                Bound::Excluded(dim) => *dim + 1,
                Bound::Unbounded => 0,
            },
            rank,
        );
        let end_dim = to_axis(
            match axes.end_bound() {
                Bound::Included(dim) => *dim,
                Bound::Excluded(dim) => *dim - 1,
                Bound::Unbounded => 0,
            },
            rank,
        );
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
    /// let c = Tensor::cat([&a, &b], 0);
    /// assert_eq!(c, [[1, 2], [3, 4], [5, 6], [7, 8]]);
    /// ```
    ///
    #[must_use]
    pub fn cat<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: isize) -> Tensor {
        let tensors: Vec<&Tensor> = tensors.into_iter().collect();
        assert!(tensors.len() > 1, "Cat requires two or more tensors.");
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
        let mut offset2 = tensors.iter().fold(0, |acc, t| acc + t.shape()[dim] as isize);
        let mut shape = tensors[0].shape();
        shape[dim] = offset2 as usize;
        let mut res = None;
        for tensor in tensors {
            let d = tensor.shape()[dim] as isize;
            offset2 -= d;
            let padding: Vec<(isize, isize)> = core::iter::repeat((0isize, 0isize))
                    .take(rank - dim - 1)
                    .chain([(offset, offset2)]).collect();
            let t = tensor.pad_zeros(padding);
            if let Some(r) = res {
                res = Some(r + t);
            } else {
                res = Some(t);
            }
            offset += d;
        }
        res.unwrap()
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
    /// assert_eq!(t.unsqueeze(1).shape(), &[2, 1, 3]);
    /// assert_eq!(t.unsqueeze(-1).shape(), &[2, 3, 1]);
    /// ```
    #[must_use]
    pub fn unsqueeze(&self, dim: isize) -> Tensor {
        let shape = self.shape();
        if dim < 0 {
            let rank = shape.len();
            let dim = (-dim) as usize;
            let dim = rank - dim + 1;
            return self.reshape(
                shape[..dim]
                    .iter()
                    .copied()
                    .chain([1])
                    .chain(shape[dim..].iter().copied())
                    .collect::<Vec<usize>>(),
            );
        } else {
            let dim = dim as usize;
            return self.reshape(
                shape[..dim]
                    .iter()
                    .copied()
                    .chain([1])
                    .chain(shape[dim..].iter().copied())
                    .collect::<Vec<usize>>(),
            );
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
    /// assert_eq!(Tensor::stack([&a, &b], 0), array![[[1, 2],
    ///                                                [3, 4]],
    ///                                               [[5, 6],
    ///                                                [7, 8]]]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the tensors have different shapes along the stacking dimension.
    ///
    /// # See also
    ///
    /// [`unsqueeze`](Tensor::unsqueeze), [`cat`](Tensor::cat)
    #[must_use]
    pub fn stack<'a>(tensors: impl IntoIterator<Item = &'a Tensor>, dim: isize) -> Tensor {
        let tensors: Vec<Tensor> = tensors.into_iter().map(|t| t.unsqueeze(dim)).collect();
        Tensor::cat(&tensors, dim)
    }

    /// Split tensor into multiple tensors at given dim/axis
    #[must_use]
    pub fn split(&self, sizes: impl IntoShape, dim: isize) -> Vec<Tensor> {
        // assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
        // dim = self._resolve_dim(dim)
        // if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
        // assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
        // return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])
        let sizes: Vec<usize> = sizes.into_shape().collect();
        let shape = self.shape();
        let rank = shape.rank();
        let dim: usize = if dim < 0 { dim + rank as isize } else { dim } as usize;
        assert_eq!(
            sizes.iter().sum::<usize>(),
            shape[dim],
            "Sizes must sum exactly to {}, but got {:?}, which sums to {}",
            shape[dim],
            sizes,
            sizes.iter().sum::<usize>()
        );

        let mut res = Vec::new();
        let mut acc_size = 0;
        for size in sizes {
            let size = size as isize;
            let mut index = Vec::new();
            for i in 0..dim {
                index.push(0..shape[i] as isize);
            }
            index.push(acc_size..acc_size + size);
            //println!("Index {index:?}");
            res.push(self.get(index));
            acc_size += size;
        }
        res
    }

    /// Masked fill
    #[must_use]
    pub fn masked_fill(&self, mask: impl Into<Tensor>, value: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        mask.into().where_(value, self)
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
    #[must_use]
    pub fn pool(
        &self,
        kernel_size: impl IntoShape,
        stride: impl IntoShape,
        dilation: impl IntoShape,
    ) -> Tensor {
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
            .cloned()
            .zip(d_.iter().cloned())
            .zip(k_.iter().cloned())
            .zip(s_.iter().cloned())
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
            .map(|&d| 0..d as isize)
            .collect();
        let sh_b: Vec<usize> = shape[..rank - k_.len()].into();
        let mut xup = self.repeat(repeats);

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
                    .map(|((k, i), d)| (0..(k * (i + d)) as isize)),
            )
            .collect();
        //println!("Padding {padding:?}");
        xup = xup.get(padding);
        //println!("{xup} padded");
        let sh: Vec<usize> = sh_b
            .iter()
            .copied()
            .chain(
                k_.iter()
                    .copied()
                    .zip(i_.iter().copied())
                    .zip(d_.iter().copied())
                    .map(|((k, i), d)| [k, i + d])
                    .flatten(),
            )
            .collect();
        //println!("Reshape {sh:?}");
        xup = xup.reshape(sh);

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
                    .map(|((k, o), s)| [(0..k as isize), (0..(o * s) as isize)])
                    .flatten(),
            )
            .collect();
        xup = xup.get(padding);
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
                    .map(|((k, o), s)| [k, o, s])
                    .flatten(),
            )
            .collect();
        xup = xup.reshape(sh);
        // padding = noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))
        // xup = xup.shrink(padding)
        let padding: Vec<Range<isize>> = pad_b
            .iter()
            .cloned()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .map(|(k, o)| [(0..k as isize), (0..o as isize), (0..1)])
                    .flatten(),
            )
            .collect();
        xup = xup.get(padding);
        // sh = noop_ + flatten((k,o) for k,o in zip(k_, o_))
        // xup = xup.reshape(sh)
        let sh: Vec<usize> = sh_b
            .iter()
            .copied()
            .chain(
                k_.iter()
                    .copied()
                    .zip(o_.iter().copied())
                    .map(|(k, o)| [k, o])
                    .flatten(),
            )
            .collect();
        xup = xup.reshape(sh);

        // xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
        let axes: Vec<isize> = (0..rank - k_.len())
            .chain((0..i_.len()).map(|i| rank - k_.len() + i * 2 + 1))
            .chain((0..i_.len()).map(|i| rank - k_.len() + i * 2))
            .map(|i| i as isize)
            .collect();
        xup = xup.permute(axes);

        xup
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
    /// assert_eq!(arr.repeat([2]), vec![1, 2, 3, 4, 5, 6]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor has zero dimensions.
    ///
    /// Returns a new tensor with the repeated values.
    #[must_use]
    pub fn repeat(&self, repeats: impl IntoShape) -> Tensor {
        let repeats: Vec<usize> = repeats.into_shape().collect();
        let shape = self.shape();
        let rank = shape.len();
        assert!(
            repeats.len() >= rank,
            "Repeats must be greater or equal to rank of the tensor."
        );

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

        //println!("base_shape {base_shape:?} {new_shape:?} {expand_shape:?} {final_shape:?}");

        let mut x = self.reshape(new_shape);
        x = x.expand(expand_shape);
        x = x.reshape(final_shape);
        return x;
    }

    /*#[must_use]
    pub fn conv(&self) -> Tensor {
        todo!()
    }*/

    // io
    /// Load module from path
    pub fn load<Module: FromIterator<Tensor>>(path: impl AsRef<Path>) -> Result<Module, ZyxError> {
        let debug_print: bool = RT.lock().debug_dev();
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
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
        let mut tensors = Vec::new();
        let mut dtype = DType::F32;
        let mut shape = vec![1];
        for x in header.chars() {
            if ['"', '[', ']'].contains(&x) {
                if begin_str {
                    //std::println!("{text}");
                    if i % 7 == 0 {
                        //params[i / 7].set_label(&text);
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
                        let mut buf = vec![0u8; bytes];
                        if debug_print {
                            print!("Loading tensor with shape {shape:?}, {dtype:?} ...");
                        }
                        f.read_exact(&mut buf)?;
                        if debug_print {
                            println!(" DONE");
                        }
                        tensors.push(match dtype {
                            DType::F32 => {
                                let vec: Vec<f32> = buf
                                    .chunks_exact(dtype.byte_size())
                                    .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                                    .collect();
                                Tensor::from(vec).reshape(&shape)
                            }
                            DType::F64 => {
                                let vec: Vec<f64> = buf
                                    .chunks_exact(dtype.byte_size())
                                    .map(|x| {
                                        f64::from_le_bytes([
                                            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                                        ])
                                    })
                                    .collect();
                                Tensor::from(vec).reshape(&shape)
                            }
                            DType::I32 => {
                                let vec: Vec<i32> = buf
                                    .chunks_exact(dtype.byte_size())
                                    .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                                    .collect();
                                Tensor::from(vec).reshape(&shape)
                            }
                            _ => todo!(),
                        });
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

    /// All tensor elements as contiguous le_bytes vector in row major order
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, ZyxError> {
        Ok(match self.dtype() {
            DType::F32 => {
                let data: Vec<f32> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::F64 => {
                let data: Vec<f64> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::U8 => {
                let data: Vec<u8> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::I8 => {
                let data: Vec<i8> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::I16 => {
                let data: Vec<i16> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::I32 => {
                let data: Vec<i32> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::I64 => {
                let data: Vec<i64> = self.clone().try_into()?;
                data.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            DType::Bool => {
                let data: Vec<bool> = self.clone().try_into()?;
                unsafe { std::mem::transmute(data) }
            }
        })
    }

    /// Load tensor from le_bytes in row major order
    pub fn from_le_bytes(&self, bytes: &[u8]) -> Result<(), ZyxError> {
        let _ = bytes;
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
    /// If self is not float, then cast it to float
    #[must_use]
    fn float_cast(&self) -> Tensor {
        let dtype = self.dtype();
        if !dtype.is_float() {
            return match dtype.byte_size() {
                #[cfg(feature = "half")]
                1 | 2 => self.cast(DType::F16),
                #[cfg(feature = "half")]
                4 => self.cast(DType::F32),
                #[cfg(not(feature = "half"))]
                1 | 2 | 4 => self.cast(DType::F32),
                8 => self.cast(DType::F64),
                _ => panic!(),
            };
        }
        self.clone()
    }

    /// Braodcasts to synchronize shapes and casts to synchronize dtypss
    /// This does both automatic expand AND automatic casting between dtypes.
    // TODO Both of these can be disable by changing a setting in the backend.
    #[must_use]
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
                return Err(ZyxError::BroadcastError(format!("Left and right tensor shapes can not be broadcasted: {x_shape:?} and {y_shape:?}")));
                //assert!( *x == 1 || *y == 1, "Left and right tensor shapes can not be broadcasted: {x_shape:?} and {y_shape:?}");
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
        x = x.reshape(&x_shape);
        if x_shape != eshape {
            x = x.expand(&eshape);
        }
        //println!("Second broadcast operand {y}");
        y = y.reshape(&y_shape);
        //println!("{x_shape:?}, {y_shape:?}, {eshape:?}");
        //println!("After reshape second broadcast operand {y}");
        //Tensor::plot_graph([], "graph");
        if y_shape != eshape {
            y = y.expand(&eshape);
        }
        //println!("Second broadcast operand {y}");
        //println!("Broadcasted to {eshape:?}");
        //println!("y shape {:?}", y.shape());
        return Ok((x, y));
    }

    // Calculate shape for reduce which keeps reduced dims set to 1
    fn reduce_kd_shape(&self, axes: impl IntoAxes) -> Vec<usize> {
        let mut shape = self.shape();
        for a in axes.clone().into_axes(shape.len()) {
            shape[a] = 1;
        }
        shape
    }

    pub(super) fn id(&self) -> TensorId {
        self.id
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
        let mut data = Vec::with_capacity(numel);
        unsafe { data.set_len(numel) };
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
        let precision = if let Some(precision) = f.precision() {
            precision
        } else {
            3
        };
        let x = self.clone();
        let res = match self.dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => {
                let data: Result<Vec<bf16>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f16 tensor failed to realize {e:?}"),
                }
            }
            #[cfg(feature = "half")]
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
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let data: Result<Vec<Complex<f32>>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f32 tensor failed to realize {e:?}"),
                }
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let data: Result<Vec<Complex<f64>>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("f64 tensor failed to realize {e:?}"),
                }
            }
            DType::U8 => {
                let data: Result<Vec<u8>, _> = x.try_into();
                match data {
                    Ok(data) => tensor_to_string(&data, &self.shape(), precision, f.width()),
                    Err(e) => format!("i32 tensor failed to realize {e:?}"),
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
        return Tensor {
            id: RT.lock().variable(vec![1], &[value]).unwrap(),
        };
    }
}

impl<T: Scalar> From<Vec<T>> for Tensor {
    fn from(data: Vec<T>) -> Self {
        return Tensor {
            id: RT.lock().variable(vec![data.len()], &data).unwrap(),
        };
    }
}

impl<T: Scalar> From<&Vec<T>> for Tensor {
    fn from(data: &Vec<T>) -> Self {
        return Tensor {
            id: RT.lock().variable(vec![data.len()], &data).unwrap(),
        };
    }
}

impl<T: Scalar> From<&[T]> for Tensor {
    fn from(data: &[T]) -> Self {
        let n = data.len();
        return Tensor {
            id: RT.lock().variable(vec![n], data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize> From<[T; D0]> for Tensor {
    fn from(data: [T; D0]) -> Self {
        return Tensor {
            id: RT.lock().variable(vec![D0], &data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> From<[[T; D1]; D0]> for Tensor {
    fn from(data: [[T; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0].as_ptr(), D0 * D1) };
        return Tensor {
            id: RT.lock().variable(vec![D0, D1], data).unwrap(),
        };
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> From<[[[T; D2]; D1]; D0]>
    for Tensor
{
    fn from(data: [[[T; D2]; D1]; D0]) -> Self {
        let data = unsafe { core::slice::from_raw_parts(data[0][0].as_ptr(), D0 * D1 * D2) };
        return Tensor {
            id: RT.lock().variable(vec![D0, D1, D2], data).unwrap(),
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
            id: RT.lock().variable(vec![D0, D1, D2, D3], data).unwrap(),
        };
    }
}

impl PartialEq<f32> for Tensor {
    fn eq(&self, other: &f32) -> bool {
        if let Ok(data) = self.clone().try_into() {
            let data: f32 = data;
            &data == other
        } else {
            false
        }
    }
}

impl PartialEq<i32> for Tensor {
    fn eq(&self, other: &i32) -> bool {
        if let Ok(data) = self.clone().try_into() {
            let data: i32 = data;
            &data == other
        } else {
            false
        }
    }
}

impl<T: Scalar, const D0: usize> PartialEq<[T; D0]> for Tensor {
    fn eq(&self, other: &[T; D0]) -> bool {
        if self.shape() != [D0] {
            return false
        }
        if let Ok(data) = self.clone().try_into() {
            let data: [T; D0] = data;
            &data == other
        } else {
            false
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize> PartialEq<[[T; D1]; D0]> for Tensor {
    fn eq(&self, other: &[[T; D1]; D0]) -> bool {
        if self.shape() != [D0, D1] {
            return false
        }
        if let Ok(data) = self.clone().try_into() {
            let data: [[T; D1]; D0] = data;
            &data == other
        } else {
            false
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize> PartialEq<[[[T; D2]; D1]; D0]>
    for Tensor
{
    fn eq(&self, other: &[[[T; D2]; D1]; D0]) -> bool {
        if self.shape() != [D0, D1, D2] {
            return false
        }
        if let Ok(data) = self.clone().try_into() {
            let data: [[[T; D2]; D1]; D0] = data;
            &data == other
        } else {
            false
        }
    }
}

impl<T: Scalar, const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PartialEq<[[[[T; D3]; D2]; D1]; D0]> for Tensor
{
    fn eq(&self, other: &[[[[T; D3]; D2]; D1]; D0]) -> bool {
        if self.shape() != [D0, D1, D2, D3] {
            return false
        }
        if let Ok(data) = self.clone().try_into() {
            let data: [[[[T; D3]; D2]; D1]; D0] = data;
            &data == other
        } else {
            false
        }
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
        return tensor;
    }
}

impl<IT: Into<Tensor>> Add<IT> for &Tensor {
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
        return tensor;
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
        return tensor;
    }
}

impl<IT: Into<Tensor>> Sub<IT> for &Tensor {
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
        return tensor;
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
        return tensor;
    }
}

impl<IT: Into<Tensor>> Mul<IT> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into();
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
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
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().div(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> Div<IT> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().div(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> BitOr<IT> for Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitor(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> BitOr<IT> for &Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitor(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> BitXor<IT> for Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitxor(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> BitXor<IT> for &Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitxor(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> BitAnd<IT> for Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitand(x.id, y.id),
        };
        return tensor;
    }
}

impl<IT: Into<Tensor>> BitAnd<IT> for &Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        let tensor = Tensor {
            id: RT.lock().bitand(x.id, y.id),
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

#[cfg(feature = "half")]
impl_trait!(BitXor for bf16, bitxor);
#[cfg(feature = "half")]
impl_trait!(BitXor for f16, bitxor);
impl_trait!(BitXor for f32, bitxor);
impl_trait!(BitXor for f64, bitxor);
#[cfg(feature = "complex")]
impl_trait!(BitXor for Complex<f32>, bitxor);
#[cfg(feature = "complex")]
impl_trait!(BitXor for Complex<f64>, bitxor);
impl_trait!(BitXor for u8, bitxor);
impl_trait!(BitXor for i8, bitxor);
impl_trait!(BitXor for i16, bitxor);
impl_trait!(BitXor for i32, bitxor);
impl_trait!(BitXor for i64, bitxor);
impl_trait!(BitXor for bool, bitxor);

#[cfg(feature = "half")]
impl_trait!(BitOr for bf16, bitor);
#[cfg(feature = "half")]
impl_trait!(BitOr for f16, bitor);
impl_trait!(BitOr for f32, bitor);
impl_trait!(BitOr for f64, bitor);
#[cfg(feature = "complex")]
impl_trait!(BitOr for Complex<f32>, bitor);
#[cfg(feature = "complex")]
impl_trait!(BitOr for Complex<f64>, bitor);
impl_trait!(BitOr for u8, bitor);
impl_trait!(BitOr for i8, bitor);
impl_trait!(BitOr for i16, bitor);
impl_trait!(BitOr for i32, bitor);
impl_trait!(BitOr for i64, bitor);
impl_trait!(BitOr for bool, bitor);

#[cfg(feature = "half")]
impl_trait!(BitAnd for bf16, bitand);
#[cfg(feature = "half")]
impl_trait!(BitAnd for f16, bitand);
impl_trait!(BitAnd for f32, bitand);
impl_trait!(BitAnd for f64, bitand);
#[cfg(feature = "complex")]
impl_trait!(BitAnd for Complex<f32>, bitand);
#[cfg(feature = "complex")]
impl_trait!(BitAnd for Complex<f64>, bitand);
impl_trait!(BitAnd for u8, bitand);
impl_trait!(BitAnd for i8, bitand);
impl_trait!(BitAnd for i16, bitand);
impl_trait!(BitAnd for i32, bitand);
impl_trait!(BitAnd for i64, bitand);
impl_trait!(BitAnd for bool, bitand);

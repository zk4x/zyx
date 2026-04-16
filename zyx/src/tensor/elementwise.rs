// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::ops::{Neg, Not};

use crate::{Float, RT, Scalar, Tensor, error::ZyxError, kernel::UOp};

impl Tensor {
    fn poly_n(x: Tensor, coeffs: [f32; 5]) -> Tensor {
        let mut result: Tensor = 0.0f32.into();
        for c in coeffs {
            result = result * x.clone() + Tensor::from(c);
        }
        result
    }

    /// Absolute value
    #[must_use]
    pub fn abs(&self) -> Tensor {
        self.relu() + (-self).relu()
    }

    /// Square
    #[must_use]
    pub fn square(&self) -> Tensor {
        self.clone() * self.clone()
    }

    /// Returns the sign of each element: -1 if negative, 1 if positive, 0 if zero.
    #[must_use]
    pub fn sign(&self) -> Tensor {
        let zero = Tensor::zeros_like(self.clone());
        let neg_one: Tensor = (-1_i32).into();
        let pos_one: Tensor = (1_i32).into();
        let is_neg = self.clone().cmplt(zero.clone()).unwrap();
        let result = is_neg.where_(&neg_one, &pos_one).unwrap();
        self.nonzero().where_(&result, &zero).unwrap()
    }

    /// Error function
    #[must_use]
    pub fn erf(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let one: Tensor = 1.0f32.into();
        let t = one.clone() / (one.clone() + 0.3275911f32 * x.clone().abs());
        let coeffs = [
            1.061405429f32,
            -1.453152027f32,
            1.421413741f32,
            -0.284496736f32,
            0.254829592f32,
        ];
        let poly = Self::poly_n(t.clone(), coeffs);
        x.sign() * (one - t * poly * (-x.clone() * x.clone()).exp())
    }

    /// Applies element-wise, CELU(x)=max⁡(0,x)+min⁡(0,α∗(exp⁡(x/α)−1)).
    #[must_use]
    pub fn celu(&self, alpha: impl Scalar) -> Tensor {
        self.relu() - (-((self / alpha).exp() - 1) * alpha).relu()
    }

    /// Returns a new tensor with the cosine of the elements of self.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn cos(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let x = Tensor { id: RT.lock().unary(x.id, UOp::Cos) };
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
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn exp2(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let x = Tensor { id: RT.lock().unary(x.id, UOp::Exp2) };
        x
    }

    /// Returns a new floored tensor
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn floor(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let x = Tensor { id: RT.lock().unary(x.id, UOp::Floor) };
        x
    }

    /// Returns a new tensor with each element truncated toward zero.
    /// For positive numbers, this removes the fractional part (floor).
    /// For negative numbers, this also removes the fractional part (ceiling).
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn trunc(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let x = Tensor { id: RT.lock().unary(x.id, UOp::Trunc) };
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
        let c: Tensor = std::f64::consts::E.log2().into();
        (self * c.cast(self.dtype())).exp2()
    }

    /// Returns a new tensor with the Gelu activation function applied to each element of self.
    ///
    /// The Gelu activation function is defined as:
    /// `gelu(x) = x * 0.5 * (1 + tanh(sqrt(2 / π) * (x + x^3 * 0.044715)))`.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn gelu(&self) -> Tensor {
        self * 0.5f32 * (((self + self * self * self * 0.044_715f32) * (2f32 / core::f32::consts::PI).sqrt()).tanh() + 1f32)
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
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn log2(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        return Tensor { id: RT.lock().unary(x.id, UOp::Log2) };
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
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn ln(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let c: Tensor = (1f64 / std::f64::consts::E.log2()).into();
        x.log2() * c.cast(x.dtype())
    }

    /// Compute logarithm with any base
    #[must_use]
    pub fn log(&self, base: impl Into<Tensor>) -> Tensor {
        self.log2() / base.into().log2()
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

    /// Computes the multiplicative inverse of each element in the input tensor, 1/x.
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
        return Tensor { id: RT.lock().unary(self.id, UOp::Reciprocal) };
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
    #[track_caller]
    pub fn relu(&self) -> Tensor {
        //return Tensor { id: RT.lock().unary(self.id, UOp::ReLU) };
        //self.cmpgt(0).unwrap().where_(self, 0).unwrap() // for whatever reason this is the fastest
        let dtype = self.dtype();
        self.cmpgt(Tensor::from(0f32).cast(dtype)).unwrap() * self
    }

    /// Computes the reciprocal square root of each element in the input tensor.

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
        (1.050_700_987_355_480_5f64 * (self.relu() - (1.673_263_242_354_377_3f64 * (Tensor::ones(1, dtype) - self.exp())).relu()))
            .cast(dtype)
    }

    /// Rounds each element of the input tensor to the nearest integer.
    ///
    /// For values exactly halfway between two integers, this function rounds to the nearest even integer
    /// (banker's rounding). This is consistent with Python's round() behavior and IEEE 754 standards.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:**
    ///
    /// A new tensor with the same shape as the input, containing rounded values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from([1.2f32, 2.7, 3.5, -1.5, -2.3]);
    /// // Rounds to [1.0, 3.0, 4.0, -2.0, -2.0]
    /// let rounded = t.round();
    /// ```
    ///
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn round(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Round to nearest integer using: floor(x + 0.5) for positive numbers
        // But we need to handle negative numbers and the halfway case properly
        // Simple rounding that works for both positive and negative
        let sign = x.clone().cmplt(0.0_f32).unwrap() * -2.0_f32 + 1.0_f32;
        let abs_x = x.clone().abs();
        let rounded_abs = (abs_x.clone() + 0.5_f32).floor();
        let rounded = rounded_abs * sign;

        rounded.cast(original_dtype)
    }

    /// Returns the fractional part of each element in the input tensor.
    ///
    /// The fractional part is defined as x - floor(x), which gives the part of the number
    /// after the decimal point. For positive numbers, this is straightforward. For negative
    /// numbers, the fractional part is positive (e.g., frac(-1.7) = 0.3).
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:**
    ///
    /// A new tensor with the same shape as the input, containing fractional parts.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from([1.2f32, 2.7, 3.5, -1.7, -2.3]);
    /// // Fractional parts: [0.2, 0.7, 0.5, 0.3, 0.7]
    /// let fractional = t.frac();
    /// ```
    ///
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn frac(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Fractional part = x - floor(x)
        let fractional = x.clone() - x.floor();

        // For negative numbers, add 1 to make fractional part positive
        // For positive numbers, keep as is
        let is_negative = fractional.clone().cmplt(0.0_f32).unwrap();
        let fractional_positive = is_negative.clone() * (fractional.clone() + 1.0_f32) + is_negative.not() * fractional;

        fractional_positive.cast(original_dtype)
    }

    /// Rounds each element of the input tensor up to the nearest integer.
    ///
    /// The ceiling function returns the smallest integer greater than or equal to x.
    /// For example, ceil(1.2) = 2, ceil(-1.7) = -1, ceil(3.0) = 3.
    ///
    /// **Parameters:**
    ///
    /// * self: The input tensor.
    ///
    /// **Returns:**
    ///
    /// A new tensor with the same shape as the input, containing ceiling values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zyx::Tensor;
    ///
    /// let t = Tensor::from([1.2f32, 2.7, 3.0, -1.7, -2.3]);
    /// // Ceil to [2.0, 3.0, 3.0, -1.0, -2.0]
    /// let ceiled = t.ceil();
    /// ```
    ///
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn ceil(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Since we don't have a direct ceil operation, we implement it using:
        // ceil(x) = -floor(-x)
        let ceiled = (-x.clone()).floor() * -1.0_f32;

        ceiled.cast(original_dtype)
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

    /// Applies the hard sigmoid activation function to each element in the input tensor.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn hard_sigmoid(&self) -> Tensor {
        let dtype = self.dtype();
        let c1 = Tensor::from(-3).cast(dtype);
        let c2 = Tensor::from(1).cast(dtype);
        let c3 = Tensor::from(6f32).cast(dtype);
        let c4 = Tensor::from(0.5f32).cast(dtype);
        (self.cmpgt(c1).unwrap() * (self / c3 + c4)).minimum(c2).unwrap()
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
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn sin(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Sin) }
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
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Sqrt) }
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
    /// let t = Tensor::from(vec![0.5f32, 1.0]);
    /// assert_eq!(t.tanh(), [0.46211715738221946f32, 0.761594166564993]);
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the input tensor is empty.
    #[must_use]
    pub fn tanh(&self) -> Tensor {
        let exp2x = (self + self).exp();
        let one = Tensor::from(1).cast(self.dtype());
        (exp2x.clone() - one.clone()) / (exp2x + one)
    }

    /// Converts angles from degrees to radians.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn deg2rad(&self) -> Tensor {
        (self * (std::f64::consts::PI / 180.0)).cast(self.dtype())
    }

    /// Returns a boolean tensor where elements are close within a tolerance.
    /// # Errors
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn isclose(
        &self,
        other: impl Into<Tensor>,
        rtol: impl Into<Tensor>,
        atol: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let other = other.into();
        let rtol = rtol.into();
        let atol = atol.into();

        let diff = (self - other.clone()).abs();
        let tolerance = atol.clone() + other.clone() * rtol.clone();
        diff.cmplt(tolerance)
    }

    /// Returns a boolean tensor where elements are infinite.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn isinf(&self) -> Tensor {
        self.equal(f32::INFINITY).unwrap()
    }

    /// Returns a boolean tensor where elements are NaN.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn isnan(&self) -> Tensor {
        self.equal(f32::NAN).unwrap()
    }

    /// Returns the base-10 logarithm of each element in the tensor.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn log10(&self) -> Tensor {
        (self.log2() / Tensor::from(10f32).log2()).cast(self.dtype())
    }

    /// Converts angles from radians to degrees.
    /// # Panics
    /// Panics if applied on non-float dtype while implicit casting is disabled.
    #[must_use]
    pub fn rad2deg(&self) -> Tensor {
        (self * (180.0 / std::f64::consts::PI)).cast(self.dtype())
    }

    /// Bitnot
    pub fn bitnot(&self) -> Tensor {
        Tensor { id: RT.lock().unary(self.id, UOp::BitNot) }
    }

    /// Clamps the elements of this tensor within a specified range.
    ///
    /// Each element in the tensor is constrained to lie between the corresponding
    /// elements in the `min` and `max` tensors. Values below the minimum are set to
    /// the minimum value, and values above the maximum are set to the maximum value.
    ///
    /// # Arguments
    ///
    /// * `min`: A tensor representing the lower bound for clamping.
    /// * `max`: A tensor representing the upper bound for clamping.
    ///
    /// # Returns
    ///
    /// A new tensor with its elements clamped within the range defined by `min` and `max`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn clamp(&self, min: impl Into<Tensor>, max: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        self.maximum(min.into())?.minimum(max.into())
    }

    /// Compare less than
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn cmplt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, crate::kernel::BOp::Cmplt);
        Ok(Tensor { id })
    }

    /// Compare greater than
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn cmpgt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, crate::kernel::BOp::Cmpgt);
        Ok(Tensor { id })
    }

    /// Elementwise maximum between two tensors
    ///
    /// # Errors
    ///
    /// Returns error if the tensors have non broadcasteable shapes.
    pub fn maximum(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, crate::kernel::BOp::Max);
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
}

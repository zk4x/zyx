// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use std::ops::{Neg, Not};

use crate::{DType, Float, RT, Scalar, Tensor, error::ZyxError, kernel::UOp};

impl Tensor {
    #[allow(clippy::needless_pass_by_value)]
    fn poly_n(x: Tensor, coeffs: [f32; 5]) -> Tensor {
        let mut result: Tensor = 0.0f32.into();
        for c in coeffs {
            result = result * x.clone() + Tensor::from(c);
        }
        result
    }

    /// Absolute value of each element.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-3.0f32, -1.0, 2.0]);
    /// let y = t.abs();
    /// ```
    #[must_use]
    pub fn abs(&self) -> Tensor {
        self.relu() + (-self).relu()
    }

    /// Element-wise square: `x * x`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, -2.0, 3.0]);
    /// let y = t.square();
    /// ```
    #[must_use]
    pub fn square(&self) -> Tensor {
        self.clone() * self.clone()
    }

    /// Element-wise sign: -1 for negatives, 1 for positives, 0 for zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.0, 5.0]);
    /// let y = t.sign();
    /// ```
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn sign(&self) -> Tensor {
        let zero = Tensor::zeros_like(self.clone());
        let neg_one = Tensor::from(-1i32);
        let pos_one = Tensor::from(1i32);
        let is_neg = self.clone().cmplt(zero.clone()).unwrap();
        let result = is_neg.where_(&neg_one, &pos_one).unwrap();
        self.nonzero().where_(&result, &zero).unwrap()
    }

    /// Error function `erf(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 0.5, 1.0]);
    /// let y = t.erf();
    /// ```
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn erf(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let t: Tensor = 1 / (1 + 0.327_591_1f32 * x.abs());
        let coeffs = [1.061_405_4f32, -1.453_152_1, 1.421_413_8, -0.284_496_74, 0.254_829_6];
        let poly = Self::poly_n(t.clone(), coeffs);
        x.sign() * (1 - t * poly * (-x.clone() * x).exp())
    }

    /// Inverse error function `erfinv(x)`, defined for `|x| < 1`
    /// via the Winitzki approximation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 0.5, -0.5]);
    /// let y = t.erfinv();
    /// ```
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn erfinv(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let a = Tensor::from(0.147f32);
        let four_over_pi = Tensor::from(4.0f32 / core::f32::consts::PI);
        let one = Tensor::from(1.0f32);
        let xsq = x.clone().square();
        let one_minus_xsq = one - xsq;
        let l = one_minus_xsq.ln();
        let a_val = a.clone();
        let big_a = four_over_pi + a_val.clone() * l.clone();
        let inner = big_a.clone().square() - (Tensor::from(4.0f32) * a_val.clone() * l.clone());
        let t = (inner.sqrt() - big_a) / (Tensor::from(2.0f32) * a_val);
        x.sign() * t.sqrt()
    }

    /// CELU activation: `max(0, x) + min(0, α*(exp(x/α) - 1))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.celu(1.0f32);
    /// ```
    #[must_use]
    pub fn celu(&self, alpha: impl Scalar) -> Tensor {
        self.relu() - (-((self / alpha).exp() - 1) * alpha).relu()
    }

    /// Element-wise cosine.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, core::f32::consts::PI / 2.0]);
    /// let y = t.cos();
    /// ```
    #[must_use]
    pub fn cos(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Cos) }
    }

    /// Element-wise hyperbolic cosine: `cosh(x) = (exp(x) + exp(-x)) / 2`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 1.0]);
    /// let y = t.cosh();
    /// ```
    #[must_use]
    pub fn cosh(&self) -> Tensor {
        // (e^x + e^-x) / 2
        let nx = self.neg();
        let enx = nx.exp();
        let ex = self.exp();
        (ex + enx) / 2
    }

    /// Exponential Linear Unit (ELU) activation: `x` for `x > 0`,
    /// `α*(exp(x) - 1)` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.elu(1.0f32);
    /// ```
    #[must_use]
    pub fn elu(&self, alpha: impl Scalar) -> Tensor {
        self.relu() - (self.exp().neg() + 1).relu() * alpha
    }

    /// Element-wise power of 2: `2^x`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 1.0, 2.0]);
    /// let y = t.exp2();
    /// ```
    #[must_use]
    pub fn exp2(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Exp2) }
    }

    /// Element-wise floor: greatest integer less than or equal to each element.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.2f32, -1.7, 3.0]);
    /// let y = t.floor();
    /// ```
    #[must_use]
    pub fn floor(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Floor) }
    }

    /// Element-wise truncation toward zero (drops the fractional part).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.7f32, -2.3, 3.0]);
    /// let y = t.trunc();
    /// ```
    #[must_use]
    pub fn trunc(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Trunc) }
    }

    /// Element-wise exponential: `e^x`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 1.0, -1.0]);
    /// let y = t.exp();
    /// ```
    #[must_use]
    pub fn exp(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Exp) }
    }

    /// Gaussian Error Linear Unit (Gelu) activation:
    /// `gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + x³ * 0.044715)))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.gelu();
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn gelu(&self) -> Tensor {
        self * 0.5f32 * (((self + self * self * self * 0.044_715f32) * (2f32 / core::f32::consts::PI).sqrt()).tanh() + 1f32)
    }

    /// Leaky ReLU activation: `max(0, x) + neg_slope * min(0, x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.leaky_relu(0.01f32);
    /// ```
    #[must_use]
    pub fn leaky_relu(&self, neg_slope: impl Scalar) -> Tensor {
        self.relu() - (self * (-Tensor::from(neg_slope))).relu()
    }

    /// Element-wise base-2 logarithm: `log2(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.0, 4.0]);
    /// let y = t.log2();
    /// ```
    #[must_use]
    pub fn log2(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        return Tensor { id: RT.lock().unary(x.id, UOp::Log2) };
    }

    /// Element-wise natural logarithm: `ln(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 2.718, 10.0]);
    /// let y = t.ln();
    /// ```
    #[must_use]
    pub fn ln(&self) -> Tensor {
        self.log2() * core::f64::consts::LN_2
    }

    /// Element-wise logarithm with an arbitrary base: `log_base(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 10.0, 100.0]);
    /// let y = t.log(Tensor::from(10.0f32));
    /// ```
    #[must_use]
    #[allow(clippy::suboptimal_flops)]
    pub fn log(&self, base: impl Into<Tensor>) -> Tensor {
        self.log2() / base.into().log2()
    }

    /// Mish activation: `x * tanh(softplus(x))`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.mish();
    /// ```
    #[must_use]
    pub fn mish(&self) -> Tensor {
        self * self.softplus(1., 20.).tanh()
    }

    /// QuickGELU activation: `x * sigmoid(1.702 * x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.quick_gelu();
    /// ```
    #[must_use]
    pub fn quick_gelu(&self) -> Tensor {
        self * (1.702f32 * self).sigmoid()
    }

    /// Element-wise reciprocal: `1 / x`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.5f32, 1.0, 2.0]);
    /// let y = t.reciprocal();
    /// ```
    #[must_use]
    pub fn reciprocal(&self) -> Tensor {
        return Tensor { id: RT.lock().unary(self.id, UOp::Reciprocal) };
    }

    /// Rectified Linear Unit (ReLU) activation: `max(0, x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.relu();
    /// ```
    #[must_use]
    #[track_caller]
    #[allow(clippy::missing_panics_doc)]
    pub fn relu(&self) -> Tensor {
        self.cmpgt(0f32).unwrap() * self
    }

    /// Element-wise reciprocal square root: `1 / sqrt(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 4.0, 9.0]);
    /// let y = t.rsqrt();
    /// ```
    #[must_use]
    pub fn rsqrt(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Rsqrt) }
    }

    /// Self-Normalized Linear Unit (SELU) activation.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.selu();
    /// ```
    #[must_use]
    pub fn selu(&self) -> Tensor {
        1.050_701_f32 * (self.relu() - (1.673_263_2_f32 * (self.exp().neg() + 1)).relu())
    }

    /// Rounds each element to the nearest integer (half to even).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.2f32, 2.7, 3.5, -1.5, -2.3]);
    /// let y = t.round();
    /// ```
    #[must_use]
    pub fn round(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Round half to even (banker's rounding), matching torch.round and numpy.
        let floor_x = x.floor();
        let frac = x - floor_x.clone();

        // Round up when the fraction is past the midpoint.
        let round_up = frac.cmpgt(0.5f32).unwrap().cast(DType::F32);
        // On a tie, round to the nearest even integer.
        let is_half = frac.equal(0.5f32).unwrap().cast(DType::F32);
        let floor_odd = (floor_x.clone() / 2.0f32).frac().ne(0.0f32).unwrap().cast(DType::F32);

        let add = round_up + is_half * floor_odd;
        let rounded = floor_x + add;

        rounded.cast(original_dtype)
    }

    /// Element-wise fractional part: `x - floor(x)`, always non-negative.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.2f32, 2.7, 3.5, -1.7, -2.3]);
    /// let y = t.frac();
    /// ```
    #[must_use]
    pub fn frac(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Fractional part = x - floor(x)
        let fractional = x.clone() - x.floor();

        // For negative numbers, add 1 to make fractional part positive
        // For positive numbers, keep as is
        let is_negative = fractional.cmplt(0).unwrap();
        let fractional_positive = is_negative.clone() * (fractional.clone() + 1) + is_negative.not() * fractional;
        fractional_positive.cast(original_dtype)
    }

    /// Element-wise ceiling: smallest integer greater than or equal to each
    /// element.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.2f32, 2.7, 3.0, -1.7, -2.3]);
    /// let y = t.ceil();
    /// ```
    #[must_use]
    pub fn ceil(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        let original_dtype = self.dtype();

        // Since we don't have a direct ceil operation, we implement it using:
        // ceil(x) = -floor(-x)
        let ceiled = (-x).floor() * -1;

        ceiled.cast(original_dtype)
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`, in [0, 1].
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.sigmoid();
    /// ```
    #[must_use]
    pub fn sigmoid(&self) -> Tensor {
        let exp_x = self.exp();
        exp_x.clone() / (exp_x + 1)
    }

    /// Element-wise hard sigmoid: `clamp(x/6 + 0.5, 0, 1)` for `x > -3`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-4.0f32, -3.0, 0.0, 3.0, 4.0]);
    /// let y = t.hard_sigmoid();
    /// ```
    #[must_use]
    pub fn hard_sigmoid(&self) -> Tensor {
        (self.cmpgt(-3).unwrap() * (self / 6 + 0.5)).minimum(1).unwrap()
    }

    /// Element-wise sine.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, core::f32::consts::PI / 2.0, core::f32::consts::PI]);
    /// let y = t.sin();
    /// ```
    #[must_use]
    pub fn sin(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Sin) }
    }

    /// Element-wise hyperbolic sine: `sinh(x) = (exp(x) - exp(-x)) / 2`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 1.0]);
    /// let y = t.sinh();
    /// ```
    #[must_use]
    pub fn sinh(&self) -> Tensor {
        // (e^x - e^-x) / 2
        let nx = self.neg();
        let enx = nx.exp();
        let ex = self.exp();
        (ex - enx) / 2
    }

    /// Softplus activation: `log(exp(beta * x) / beta)` for `beta * x > threshold`,
    /// `beta * x` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.softplus(1.0f32, 0.0);
    /// ```
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn softplus(&self, beta: impl Float, threshold: impl Float) -> Tensor {
        let x = self * beta;
        x.cmplt(threshold).unwrap().where_(((x).exp() + 1).ln() * beta.reciprocal(), x).unwrap()
    }

    /// Element-wise square root: `sqrt(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 4.0, 9.0]);
    /// let y = t.sqrt();
    /// ```
    #[must_use]
    pub fn sqrt(&self) -> Tensor {
        let x = self.float_cast().unwrap();
        Tensor { id: RT.lock().unary(x.id, UOp::Sqrt) }
    }

    /// Swish activation: `x * sigmoid(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([-2.0f32, -0.5, 0.5, 2.0]);
    /// let y = t.swish();
    /// ```
    #[must_use]
    pub fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }

    /// Element-wise tangent: `sin(x) / cos(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, core::f32::consts::PI / 4.0, core::f32::consts::PI]);
    /// let y = t.tan();
    /// ```
    #[must_use]
    pub fn tan(&self) -> Tensor {
        self.sin() / self.cos()
    }

    /// Element-wise hyperbolic tangent: `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from(vec![0.5f32, 1.0]);
    /// assert_eq!(t.tanh(), [0.46211715738221946f32, 0.761594166564993]);
    /// ```
    #[must_use]
    pub fn tanh(&self) -> Tensor {
        let exp2x = (self + self).exp();
        (exp2x.clone() - 1) / (exp2x + 1)
    }

    /// Element-wise conversion of angles from degrees to radians.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, 90.0, 180.0]);
    /// let y = t.deg2rad();
    /// ```
    #[must_use]
    pub fn deg2rad(&self) -> Tensor {
        self * (core::f32::consts::PI / 180.0)
    }

    /// Boolean tensor true where `self` and `other` are within
    /// `atol + rtol * |other|` of each other.
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::Tensor;
    /// let a = Tensor::from([0.1f32, 0.2, 0.3]);
    /// let b = Tensor::from([0.1f32, 0.200001, 0.4]);
    /// let y = a.isclose(b, Tensor::from(1e-5f32), Tensor::from(1e-8f32))?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors have non-broadcastable shapes.
    pub fn isclose(
        &self,
        other: impl Into<Tensor>,
        rtol: impl Into<Tensor>,
        atol: impl Into<Tensor>,
    ) -> Result<Tensor, ZyxError> {
        let other = other.into();
        let rtol = rtol.into();
        let atol = atol.into();

        let diff = (self - &other).abs();
        let tolerance = &atol + &other * &rtol;
        diff.cmplt(tolerance)
    }

    /// Boolean tensor true where elements are infinite.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, f32::INFINITY, f32::NEG_INFINITY]);
    /// let y = t.isinf();
    /// ```
    #[must_use]
    pub fn isinf(&self) -> Tensor {
        self.equal(f32::INFINITY).unwrap()
    }

    /// Boolean tensor true where elements are NaN.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, f32::NAN, 0.0]);
    /// let y = t.isnan();
    /// ```
    #[must_use]
    pub fn isnan(&self) -> Tensor {
        self.equal(f32::NAN).unwrap()
    }

    /// Element-wise base-10 logarithm: `log10(x)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([1.0f32, 10.0, 100.0]);
    /// let y = t.log10();
    /// ```
    #[must_use]
    pub fn log10(&self) -> Tensor {
        (self.log(10)).cast(self.dtype())
    }

    /// Element-wise conversion of angles from radians to degrees.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0.0f32, core::f32::consts::PI / 2.0, core::f32::consts::PI]);
    /// let y = t.rad2deg();
    /// ```
    #[must_use]
    pub fn rad2deg(&self) -> Tensor {
        self * (180.0 / core::f32::consts::PI)
    }

    /// Element-wise bitwise NOT.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use zyx::Tensor;
    /// let t = Tensor::from([0u8, 255, 170, 85]);
    /// let y = t.bitnot();
    /// ```
    #[must_use]
    pub fn bitnot(&self) -> Tensor {
        Tensor { id: RT.lock().unary(self.id, UOp::BitNot) }
    }

    /// Clamp elements to the range `[min, max]`, broadcastable.
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::{Tensor, DType};
    /// let x = Tensor::from([0.5f32, 2.0, 3.5]);
    /// let y = x.clamp(Tensor::from(1.0f32), Tensor::from(3.0f32))?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `min`/`max` are non-broadcastable with `self`.
    #[allow(clippy::missing_panics_doc)]
    pub fn clamp(&self, min: impl Into<Tensor>, max: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        self.maximum(min.into())?.minimum(max.into())
    }

    /// Element-wise `self < rhs` (broadcastable), as a bool tensor.
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::{Tensor, DType};
    /// let a = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = a.cmplt(Tensor::from([2.0f32, 2.0, 3.0]))?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors are non-broadcastable.
    pub fn cmplt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, crate::kernel::BOp::Cmplt)?;
        Ok(Tensor { id })
    }

    /// Element-wise `self > rhs` (broadcastable), as a bool tensor.
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::{Tensor, DType};
    /// let a = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = a.cmpgt(Tensor::from([2.0f32, 2.0, 3.0]))?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors are non-broadcastable.
    pub fn cmpgt(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, crate::kernel::BOp::Cmpgt)?;
        Ok(Tensor { id })
    }

    /// Element-wise `max(self, rhs)` (broadcastable).
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::{Tensor, DType};
    /// let a = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = a.maximum(Tensor::from([2.0f32, 1.0, 3.0]))?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors are non-broadcastable.
    pub fn maximum(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let (x, y) = Tensor::broadcast(self.clone(), rhs)?;
        let id = RT.lock().binary(x.id, y.id, crate::kernel::BOp::Max)?;
        Ok(Tensor { id })
    }

    /// Element-wise `min(self, rhs)` (broadcastable).
    ///
    /// # Example
    ///
    /// ```rust no_run
    /// # use zyx::{Tensor, DType};
    /// let a = Tensor::from([1.0f32, 2.0, 3.0]);
    /// let y = a.minimum(Tensor::from([2.0f32, 1.0, 3.0]))?;
    /// # Ok::<(), zyx::ZyxError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the tensors are non-broadcastable.
    pub fn minimum(&self, rhs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        Ok(-(-self).maximum(-rhs.into())?)
    }
}

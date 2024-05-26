use crate::{DType, Scalar};
use half::{bf16, f16};

impl Scalar for f32 {
    fn from_bf16(t: bf16) -> Self {
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        t
    }

    fn from_f64(t: f64) -> Self {
        t as f32
    }

    fn from_u8(t: u8) -> Self {
        todo!()
    }

    fn from_i8(t: i8) -> Self {
        todo!()
    }

    fn from_i16(t: i16) -> Self {
        todo!()
    }

    fn from_i32(t: i32) -> Self {
        t as f32
    }

    fn from_i74(t: i32) -> Self {
        todo!()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn byte_size() -> usize {
        4
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn into_f64(self) -> f64 {
        self as f64
    }

    fn into_i32(self) -> i32 {
        self as i32
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn reciprocal(self) -> Self {
        1.0 / self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        self.max(0.)
    }

    fn sin(self) -> Self {
        f32::sin(self)
    }

    fn cos(self) -> Self {
        f32::cos(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
    }

    fn sqrt(self) -> Self {
        // good enough (error of ~ 5%)
        if self >= 0. {
            Self::from_bits((self.to_bits() + 0x3f80_0000) >> 1)
        } else {
            Self::NAN
        }
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        powf(self, rhs)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32 as f32
    }

    fn max(self, rhs: Self) -> Self {
        f32::max(self, rhs)
    }

    fn max_value() -> Self {
        f32::MAX
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn epsilon() -> Self {
        0.00001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f32::INFINITY && rhs == -f32::INFINITY)
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }
}

const BP: [f32; 2] = [1.0, 1.5];
const DP_H: [f32; 2] = [0.0, 5.84960938e-01]; /* 0x3f15c000 */
const DP_L: [f32; 2] = [0.0, 1.56322085e-06]; /* 0x35d1cfdc */
const TWO24: f32 = 16777216.0; /* 0x4b800000 */
const HUGE: f32 = 1.0e30;
const TINY: f32 = 1.0e-30;
const L1: f32 = 6.0000002384e-01; /* 0x3f19999a */
const L2: f32 = 4.2857143283e-01; /* 0x3edb6db7 */
const L3: f32 = 3.3333334327e-01; /* 0x3eaaaaab */
const L4: f32 = 2.7272811532e-01; /* 0x3e8ba305 */
const L5: f32 = 2.3066075146e-01; /* 0x3e6c3255 */
const L6: f32 = 2.0697501302e-01; /* 0x3e53f142 */
const P1: f32 = 1.6666667163e-01; /* 0x3e2aaaab */
const P2: f32 = -2.7777778450e-03; /* 0xbb360b61 */
const P3: f32 = 6.6137559770e-05; /* 0x388ab355 */
const P4: f32 = -1.6533901999e-06; /* 0xb5ddea0e */
const P5: f32 = 4.1381369442e-08; /* 0x3331bb4c */
const LG2: f32 = 6.9314718246e-01; /* 0x3f317218 */
const LG2_H: f32 = 6.93145752e-01; /* 0x3f317200 */
const LG2_L: f32 = 1.42860654e-06; /* 0x35bfbe8c */
const OVT: f32 = 4.2995665694e-08; /* -(128-log2(ovfl+.5ulp)) */
const CP: f32 = 9.6179670095e-01; /* 0x3f76384f =2/(3ln2) */
const CP_H: f32 = 9.6191406250e-01; /* 0x3f764000 =12b cp */
const CP_L: f32 = -1.1736857402e-04; /* 0xb8f623c6 =tail of cp_h */
const IVLN2: f32 = 1.4426950216e+00;
const IVLN2_H: f32 = 1.4426879883e+00;
const IVLN2_L: f32 = 7.0526075433e-06;

macro_rules! i {
    ($array:expr, $index:expr) => {
        unsafe { *$array.get_unchecked($index) }
    };
    ($array:expr, $index:expr, = , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) = $rhs;
        }
    };
    ($array:expr, $index:expr, += , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) += $rhs;
        }
    };
    ($array:expr, $index:expr, -= , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) -= $rhs;
        }
    };
    ($array:expr, $index:expr, &= , $rhs:expr) => {
        unsafe {
            *$array.get_unchecked_mut($index) &= $rhs;
        }
    };
    ($array:expr, $index:expr, == , $rhs:expr) => {
        unsafe { *$array.get_unchecked_mut($index) == $rhs }
    };
}

pub fn fabsf(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fffffff)
}

pub fn scalbnf(mut x: f32, mut n: i32) -> f32 {
    let x1p127 = f32::from_bits(0x7f000000); // 0x1p127f === 2 ^ 127
    let x1p_126 = f32::from_bits(0x800000); // 0x1p-126f === 2 ^ -126
    let x1p24 = f32::from_bits(0x4b800000); // 0x1p24f === 2 ^ 24

    if n > 127 {
        x *= x1p127;
        n -= 127;
        if n > 127 {
            x *= x1p127;
            n -= 127;
            if n > 127 {
                n = 127;
            }
        }
    } else if n < -126 {
        x *= x1p_126 * x1p24;
        n += 126 - 24;
        if n < -126 {
            x *= x1p_126 * x1p24;
            n += 126 - 24;
            if n < -126 {
                n = -126;
            }
        }
    }
    x * f32::from_bits(((0x7f + n) as u32) << 23)
}

pub fn sqrtf(x: f32) -> f32 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f32.sqrt` native instruction, so we can leverage this for both code size
    // and speed.
    #[cfg(target_feature = "sse")]
    {
        // Note: This path is unlikely since LLVM will usually have already
        // optimized sqrt calls into hardware instructions if sse is available,
        // but if someone does end up here they'll apprected the speed increase.
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        unsafe {
            let m = _mm_set_ss(x);
            let m_sqrt = _mm_sqrt_ss(m);
            _mm_cvtss_f32(m_sqrt)
        }
    }
    #[cfg(not(target_feature = "sse"))]
    {
        const TINY: f32 = 1.0e-30;

        let mut z: f32;
        let sign: i32 = 0x80000000u32 as i32;
        let mut ix: i32;
        let mut s: i32;
        let mut q: i32;
        let mut m: i32;
        let mut t: i32;
        let mut i: i32;
        let mut r: u32;

        ix = x.to_bits() as i32;

        /* take care of Inf and NaN */
        if (ix as u32 & 0x7f800000) == 0x7f800000 {
            return x * x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
        }

        /* take care of zero */
        if ix <= 0 {
            if (ix & !sign) == 0 {
                return x; /* sqrt(+-0) = +-0 */
            }
            if ix < 0 {
                return (x - x) / (x - x); /* sqrt(-ve) = sNaN */
            }
        }

        /* normalize x */
        m = ix >> 23;
        if m == 0 {
            /* subnormal x */
            i = 0;
            while ix & 0x00800000 == 0 {
                ix <<= 1;
                i = i + 1;
            }
            m -= i - 1;
        }
        m -= 127; /* unbias exponent */
        ix = (ix & 0x007fffff) | 0x00800000;
        if m & 1 == 1 {
            /* odd m, double x to make it even */
            ix += ix;
        }
        m >>= 1; /* m = [m/2] */

        /* generate sqrt(x) bit by bit */
        ix += ix;
        q = 0;
        s = 0;
        r = 0x01000000; /* r = moving bit from right to left */

        while r != 0 {
            t = s + r as i32;
            if t <= ix {
                s = t + r as i32;
                ix -= t;
                q += r as i32;
            }
            ix += ix;
            r >>= 1;
        }

        /* use floating add to find out rounding direction */
        if ix != 0 {
            z = 1.0 - TINY; /* raise inexact flag */
            if z >= 1.0 {
                z = 1.0 + TINY;
                if z > 1.0 {
                    q += 2;
                } else {
                    q += q & 1;
                }
            }
        }

        ix = (q >> 1) + 0x3f000000;
        ix += m << 23;
        f32::from_bits(ix as u32)
    }
}

pub fn powf(x: f32, y: f32) -> f32 {
    let mut z: f32;
    let mut ax: f32;
    let z_h: f32;
    let z_l: f32;
    let mut p_h: f32;
    let mut p_l: f32;
    let y1: f32;
    let mut t1: f32;
    let t2: f32;
    let mut r: f32;
    let s: f32;
    let mut sn: f32;
    let mut t: f32;
    let mut u: f32;
    let mut v: f32;
    let mut w: f32;
    let i: i32;
    let mut j: i32;
    let mut k: i32;
    let mut yisint: i32;
    let mut n: i32;
    let hx: i32;
    let hy: i32;
    let mut ix: i32;
    let iy: i32;
    let mut is: i32;

    hx = x.to_bits() as i32;
    hy = y.to_bits() as i32;

    ix = hx & 0x7fffffff;
    iy = hy & 0x7fffffff;

    /* x**0 = 1, even if x is NaN */
    if iy == 0 {
        return 1.0;
    }

    /* 1**y = 1, even if y is NaN */
    if hx == 0x3f800000 {
        return 1.0;
    }

    /* NaN if either arg is NaN */
    if ix > 0x7f800000 || iy > 0x7f800000 {
        return x + y;
    }

    /* determine if y is an odd int when x < 0
     * yisint = 0       ... y is not an integer
     * yisint = 1       ... y is an odd int
     * yisint = 2       ... y is an even int
     */
    yisint = 0;
    if hx < 0 {
        if iy >= 0x4b800000 {
            yisint = 2; /* even integer y */
        } else if iy >= 0x3f800000 {
            k = (iy >> 23) - 0x7f; /* exponent */
            j = iy >> (23 - k);
            if (j << (23 - k)) == iy {
                yisint = 2 - (j & 1);
            }
        }
    }

    /* special value of y */
    if iy == 0x7f800000 {
        /* y is +-inf */
        if ix == 0x3f800000 {
            /* (-1)**+-inf is 1 */
            return 1.0;
        } else if ix > 0x3f800000 {
            /* (|x|>1)**+-inf = inf,0 */
            return if hy >= 0 { y } else { 0.0 };
        } else {
            /* (|x|<1)**+-inf = 0,inf */
            return if hy >= 0 { 0.0 } else { -y };
        }
    }
    if iy == 0x3f800000 {
        /* y is +-1 */
        return if hy >= 0 { x } else { 1.0 / x };
    }

    if hy == 0x40000000 {
        /* y is 2 */
        return x * x;
    }

    if hy == 0x3f000000
       /* y is  0.5 */
       && hx >= 0
    {
        /* x >= +0 */
        return sqrtf(x);
    }

    ax = fabsf(x);
    /* special value of x */
    if ix == 0x7f800000 || ix == 0 || ix == 0x3f800000 {
        /* x is +-0,+-inf,+-1 */
        z = ax;
        if hy < 0 {
            /* z = (1/|x|) */
            z = 1.0 / z;
        }

        if hx < 0 {
            if ((ix - 0x3f800000) | yisint) == 0 {
                z = (z - z) / (z - z); /* (-1)**non-int is NaN */
            } else if yisint == 1 {
                z = -z; /* (x<0)**odd = -(|x|**odd) */
            }
        }
        return z;
    }

    sn = 1.0; /* sign of result */
    if hx < 0 {
        if yisint == 0 {
            /* (x<0)**(non-int) is NaN */
            return (x - x) / (x - x);
        }

        if yisint == 1 {
            /* (x<0)**(odd int) */
            sn = -1.0;
        }
    }

    /* |y| is HUGE */
    if iy > 0x4d000000 {
        /* if |y| > 2**27 */
        /* over/underflow if x is not close to one */
        if ix < 0x3f7ffff8 {
            return if hy < 0 {
                sn * HUGE * HUGE
            } else {
                sn * TINY * TINY
            };
        }

        if ix > 0x3f800007 {
            return if hy > 0 {
                sn * HUGE * HUGE
            } else {
                sn * TINY * TINY
            };
        }

        /* now |1-x| is TINY <= 2**-20, suffice to compute
        log(x) by x-x^2/2+x^3/3-x^4/4 */
        t = ax - 1.; /* t has 20 trailing zeros */
        w = (t * t) * (0.5 - t * (0.333333333333 - t * 0.25));
        u = IVLN2_H * t; /* IVLN2_H has 16 sig. bits */
        v = t * IVLN2_L - w * IVLN2;
        t1 = u + v;
        is = t1.to_bits() as i32;
        t1 = f32::from_bits(is as u32 & 0xfffff000);
        t2 = v - (t1 - u);
    } else {
        let mut s2: f32;
        let mut s_h: f32;
        let s_l: f32;
        let mut t_h: f32;
        let mut t_l: f32;

        n = 0;
        /* take care subnormal number */
        if ix < 0x00800000 {
            ax *= TWO24;
            n -= 24;
            ix = ax.to_bits() as i32;
        }
        n += ((ix) >> 23) - 0x7f;
        j = ix & 0x007fffff;
        /* determine interval */
        ix = j | 0x3f800000; /* normalize ix */
        if j <= 0x1cc471 {
            /* |x|<sqrt(3/2) */
            k = 0;
        } else if j < 0x5db3d7 {
            /* |x|<sqrt(3)   */
            k = 1;
        } else {
            k = 0;
            n += 1;
            ix -= 0x00800000;
        }
        ax = f32::from_bits(ix as u32);

        /* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
        u = ax - i!(BP, k as usize); /* bp[0]=1.0, bp[1]=1.5 */
        v = 1.0 / (ax + i!(BP, k as usize));
        s = u * v;
        s_h = s;
        is = s_h.to_bits() as i32;
        s_h = f32::from_bits(is as u32 & 0xfffff000);
        /* t_h=ax+bp[k] High */
        is = (((ix as u32 >> 1) & 0xfffff000) | 0x20000000) as i32;
        t_h = f32::from_bits(is as u32 + 0x00400000 + ((k as u32) << 21));
        t_l = ax - (t_h - i!(BP, k as usize));
        s_l = v * ((u - s_h * t_h) - s_h * t_l);
        /* compute log(ax) */
        s2 = s * s;
        r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + s);
        s2 = s_h * s_h;
        t_h = 3.0 + s2 + r;
        is = t_h.to_bits() as i32;
        t_h = f32::from_bits(is as u32 & 0xfffff000);
        t_l = r - ((t_h - 3.0) - s2);
        /* u+v = s*(1+...) */
        u = s_h * t_h;
        v = s_l * t_h + t_l * s;
        /* 2/(3log2)*(s+...) */
        p_h = u + v;
        is = p_h.to_bits() as i32;
        p_h = f32::from_bits(is as u32 & 0xfffff000);
        p_l = v - (p_h - u);
        z_h = CP_H * p_h; /* cp_h+cp_l = 2/(3*log2) */
        z_l = CP_L * p_h + p_l * CP + i!(DP_L, k as usize);
        /* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
        t = n as f32;
        t1 = ((z_h + z_l) + i!(DP_H, k as usize)) + t;
        is = t1.to_bits() as i32;
        t1 = f32::from_bits(is as u32 & 0xfffff000);
        t2 = z_l - (((t1 - t) - i!(DP_H, k as usize)) - z_h);
    };

    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
    is = y.to_bits() as i32;
    y1 = f32::from_bits(is as u32 & 0xfffff000);
    p_l = (y - y1) * t1 + y * t2;
    p_h = y1 * t1;
    z = p_l + p_h;
    j = z.to_bits() as i32;
    if j > 0x43000000 {
        /* if z > 128 */
        return sn * HUGE * HUGE; /* overflow */
    } else if j == 0x43000000 {
        /* if z == 128 */
        if p_l + OVT > z - p_h {
            return sn * HUGE * HUGE; /* overflow */
        }
    } else if (j & 0x7fffffff) > 0x43160000 {
        /* z < -150 */
        // FIXME: check should be  (uint32_t)j > 0xc3160000
        return sn * TINY * TINY; /* underflow */
    } else if j as u32 == 0xc3160000
              /* z == -150 */
              && p_l <= z - p_h
    {
        return sn * TINY * TINY; /* underflow */
    }

    /*
     * compute 2**(p_h+p_l)
     */
    i = j & 0x7fffffff;
    k = (i >> 23) - 0x7f;
    n = 0;
    if i > 0x3f000000 {
        /* if |z| > 0.5, set n = [z+0.5] */
        n = j + (0x00800000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 23) - 0x7f; /* new k for n */
        t = f32::from_bits(n as u32 & !(0x007fffff >> k));
        n = ((n & 0x007fffff) | 0x00800000) >> (23 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    t = p_l + p_h;
    is = t.to_bits() as i32;
    t = f32::from_bits(is as u32 & 0xffff8000);
    u = t * LG2_H;
    v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    z = u + v;
    w = v - (z - u);
    t = z * z;
    t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    j = z.to_bits() as i32;
    j += n << 23;
    if (j >> 23) <= 0 {
        /* subnormal output */
        z = scalbnf(z, n);
    } else {
        z = f32::from_bits(j as u32);
    }
    sn * z
}

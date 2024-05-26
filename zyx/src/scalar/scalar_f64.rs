use crate::{DType, Scalar};
use half::{bf16, f16};

impl Scalar for f64 {
    fn from_bf16(t: bf16) -> Self {
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        t as f64
    }

    fn from_f64(t: f64) -> Self {
        t
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
        t as f64
    }

    fn from_i74(t: i32) -> Self {
        todo!()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn dtype() -> DType {
        DType::F64
    }

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn byte_size() -> usize {
        8
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> f64 {
        self
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
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
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
        f64::pow(self, rhs)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32 as f64
    }

    fn max(self, rhs: Self) -> Self {
        f64::max(self, rhs)
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn epsilon() -> Self {
        0.00001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f64::INFINITY && rhs == -f64::INFINITY)
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }
}

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

#[inline]
fn with_set_high_word(f: f64, hi: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0x00000000_ffffffff;
    tmp |= (hi as u64) << 32;
    f64::from_bits(tmp)
}

#[inline]
fn with_set_low_word(f: f64, lo: u32) -> f64 {
    let mut tmp = f.to_bits();
    tmp &= 0xffffffff_00000000;
    tmp |= lo as u64;
    f64::from_bits(tmp)
}

#[inline]
fn get_high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

pub fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0000000000000); // 0x1p1023 === 2 ^ 1023
    let x1p53 = f64::from_bits(0x4340000000000000); // 0x1p53 === 2 ^ 53
    let x1p_1022 = f64::from_bits(0x0010000000000000); // 0x1p-1022 === 2 ^ (-1022)

    let mut y = x;

    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        /* make sure final n < -53 to avoid double
        rounding in the subnormal range */
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

const BP: [f64; 2] = [1.0, 1.5];
const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01]; /* 0x3fe2b803_40000000 */
const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08]; /* 0x3E4CFDEB, 0x43CFD006 */
const TWO53: f64 = 9007199254740992.0; /* 0x43400000_00000000 */
const HUGE: f64 = 1.0e300;
const TINY: f64 = 1.0e-300;

// poly coefs for (3/2)*(log(x)-2s-2/3*s**3:
const L1: f64 = 5.99999999999994648725e-01; /* 0x3fe33333_33333303 */
const L2: f64 = 4.28571428578550184252e-01; /* 0x3fdb6db6_db6fabff */
const L3: f64 = 3.33333329818377432918e-01; /* 0x3fd55555_518f264d */
const L4: f64 = 2.72728123808534006489e-01; /* 0x3fd17460_a91d4101 */
const L5: f64 = 2.30660745775561754067e-01; /* 0x3fcd864a_93c9db65 */
const L6: f64 = 2.06975017800338417784e-01; /* 0x3fca7e28_4a454eef */
const P1: f64 = 1.66666666666666019037e-01; /* 0x3fc55555_5555553e */
const P2: f64 = -2.77777777770155933842e-03; /* 0xbf66c16c_16bebd93 */
const P3: f64 = 6.61375632143793436117e-05; /* 0x3f11566a_af25de2c */
const P4: f64 = -1.65339022054652515390e-06; /* 0xbebbbd41_c5d26bf1 */
const P5: f64 = 4.13813679705723846039e-08; /* 0x3e663769_72bea4d0 */
const LG2: f64 = 6.93147180559945286227e-01; /* 0x3fe62e42_fefa39ef */
const LG2_H: f64 = 6.93147182464599609375e-01; /* 0x3fe62e43_00000000 */
const LG2_L: f64 = -1.90465429995776804525e-09; /* 0xbe205c61_0ca86c39 */
const OVT: f64 = 8.0085662595372944372e-017; /* -(1024-log2(ovfl+.5ulp)) */
const CP: f64 = 9.61796693925975554329e-01; /* 0x3feec709_dc3a03fd =2/(3ln2) */
const CP_H: f64 = 9.61796700954437255859e-01; /* 0x3feec709_e0000000 =(float)cp */
const CP_L: f64 = -7.02846165095275826516e-09; /* 0xbe3e2fe0_145b01f5 =tail of cp_h*/
const IVLN2: f64 = 1.44269504088896338700e+00; /* 0x3ff71547_652b82fe =1/ln2 */
const IVLN2_H: f64 = 1.44269502162933349609e+00; /* 0x3ff71547_60000000 =24b 1/ln2*/
const IVLN2_L: f64 = 1.92596299112661746887e-08; /* 0x3e54ae0b_f85ddf44 =1/ln2 tail*/

pub fn fabs(x: f64) -> f64 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f64.abs` native instruction, so we can leverage this for both code size
    // and speed.
    f64::from_bits(x.to_bits() & (u64::MAX / 2))
}

pub fn sqrt(x: f64) -> f64 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f64.sqrt` native instruction, so we can leverage this for both code size
    // and speed.
    #[cfg(target_feature = "sse2")]
    {
        // Note: This path is unlikely since LLVM will usually have already
        // optimized sqrt calls into hardware instructions if sse2 is available,
        // but if someone does end up here they'll apprected the speed increase.
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        unsafe {
            let m = _mm_set_sd(x);
            let m_sqrt = _mm_sqrt_pd(m);
            _mm_cvtsd_f64(m_sqrt)
        }
    }
    #[cfg(not(target_feature = "sse2"))]
    {
        use core::num::Wrapping;

        const TINY: f64 = 1.0e-300;

        let mut z: f64;
        let sign: Wrapping<u32> = Wrapping(0x80000000);
        let mut ix0: i32;
        let mut s0: i32;
        let mut q: i32;
        let mut m: i32;
        let mut t: i32;
        let mut i: i32;
        let mut r: Wrapping<u32>;
        let mut t1: Wrapping<u32>;
        let mut s1: Wrapping<u32>;
        let mut ix1: Wrapping<u32>;
        let mut q1: Wrapping<u32>;

        ix0 = (x.to_bits() >> 32) as i32;
        ix1 = Wrapping(x.to_bits() as u32);

        /* take care of Inf and NaN */
        if (ix0 & 0x7ff00000) == 0x7ff00000 {
            return x * x + x; /* sqrt(NaN)=NaN, sqrt(+inf)=+inf, sqrt(-inf)=sNaN */
        }
        /* take care of zero */
        if ix0 <= 0 {
            if ((ix0 & !(sign.0 as i32)) | ix1.0 as i32) == 0 {
                return x; /* sqrt(+-0) = +-0 */
            }
            if ix0 < 0 {
                return (x - x) / (x - x); /* sqrt(-ve) = sNaN */
            }
        }
        /* normalize x */
        m = ix0 >> 20;
        if m == 0 {
            /* subnormal x */
            while ix0 == 0 {
                m -= 21;
                ix0 |= (ix1 >> 11).0 as i32;
                ix1 <<= 21;
            }
            i = 0;
            while (ix0 & 0x00100000) == 0 {
                i += 1;
                ix0 <<= 1;
            }
            m -= i - 1;
            ix0 |= (ix1 >> (32 - i) as usize).0 as i32;
            ix1 = ix1 << i as usize;
        }
        m -= 1023; /* unbias exponent */
        ix0 = (ix0 & 0x000fffff) | 0x00100000;
        if (m & 1) == 1 {
            /* odd m, double x to make it even */
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
        }
        m >>= 1; /* m = [m/2] */

        /* generate sqrt(x) bit by bit */
        ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
        ix1 += ix1;
        q = 0; /* [q,q1] = sqrt(x) */
        q1 = Wrapping(0);
        s0 = 0;
        s1 = Wrapping(0);
        r = Wrapping(0x00200000); /* r = moving bit from right to left */

        while r != Wrapping(0) {
            t = s0 + r.0 as i32;
            if t <= ix0 {
                s0 = t + r.0 as i32;
                ix0 -= t;
                q += r.0 as i32;
            }
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
            r >>= 1;
        }

        r = sign;
        while r != Wrapping(0) {
            t1 = s1 + r;
            t = s0;
            if t < ix0 || (t == ix0 && t1 <= ix1) {
                s1 = t1 + r;
                if (t1 & sign) == sign && (s1 & sign) == Wrapping(0) {
                    s0 += 1;
                }
                ix0 -= t;
                if ix1 < t1 {
                    ix0 -= 1;
                }
                ix1 -= t1;
                q1 += r;
            }
            ix0 += ix0 + ((ix1 & sign) >> 31).0 as i32;
            ix1 += ix1;
            r >>= 1;
        }

        /* use floating add to find out rounding direction */
        if (ix0 as u32 | ix1.0) != 0 {
            z = 1.0 - TINY; /* raise inexact flag */
            if z >= 1.0 {
                z = 1.0 + TINY;
                if q1.0 == 0xffffffff {
                    q1 = Wrapping(0);
                    q += 1;
                } else if z > 1.0 {
                    if q1.0 == 0xfffffffe {
                        q += 1;
                    }
                    q1 += Wrapping(2);
                } else {
                    q1 += q1 & Wrapping(1);
                }
            }
        }
        ix0 = (q >> 1) + 0x3fe00000;
        ix1 = q1 >> 1;
        if (q & 1) == 1 {
            ix1 |= sign;
        }
        ix0 += m << 20;
        f64::from_bits((ix0 as u64) << 32 | ix1.0 as u64)
    }
}

pub fn pow(x: f64, y: f64) -> f64 {
    let t1: f64;
    let t2: f64;

    let (hx, lx): (i32, u32) = ((x.to_bits() >> 32) as i32, x.to_bits() as u32);
    let (hy, ly): (i32, u32) = ((y.to_bits() >> 32) as i32, y.to_bits() as u32);

    let mut ix: i32 = (hx & 0x7fffffff) as i32;
    let iy: i32 = (hy & 0x7fffffff) as i32;

    /* x**0 = 1, even if x is NaN */
    if ((iy as u32) | ly) == 0 {
        return 1.0;
    }

    /* 1**y = 1, even if y is NaN */
    if hx == 0x3ff00000 && lx == 0 {
        return 1.0;
    }

    /* NaN if either arg is NaN */
    if ix > 0x7ff00000
        || (ix == 0x7ff00000 && lx != 0)
        || iy > 0x7ff00000
        || (iy == 0x7ff00000 && ly != 0)
    {
        return x + y;
    }

    /* determine if y is an odd int when x < 0
     * yisint = 0       ... y is not an integer
     * yisint = 1       ... y is an odd int
     * yisint = 2       ... y is an even int
     */
    let mut yisint: i32 = 0;
    let mut k: i32;
    let mut j: i32;
    if hx < 0 {
        if iy >= 0x43400000 {
            yisint = 2; /* even integer y */
        } else if iy >= 0x3ff00000 {
            k = (iy >> 20) - 0x3ff; /* exponent */

            if k > 20 {
                j = (ly >> (52 - k)) as i32;

                if (j << (52 - k)) == (ly as i32) {
                    yisint = 2 - (j & 1);
                }
            } else if ly == 0 {
                j = iy >> (20 - k);

                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    if ly == 0 {
        /* special value of y */
        if iy == 0x7ff00000 {
            /* y is +-inf */

            return if ((ix - 0x3ff00000) | (lx as i32)) == 0 {
                /* (-1)**+-inf is 1 */
                1.0
            } else if ix >= 0x3ff00000 {
                /* (|x|>1)**+-inf = inf,0 */
                if hy >= 0 {
                    y
                } else {
                    0.0
                }
            } else {
                /* (|x|<1)**+-inf = 0,inf */
                if hy >= 0 {
                    0.0
                } else {
                    -y
                }
            };
        }

        if iy == 0x3ff00000 {
            /* y is +-1 */
            return if hy >= 0 { x } else { 1.0 / x };
        }

        if hy == 0x40000000 {
            /* y is 2 */
            return x * x;
        }

        if hy == 0x3fe00000 {
            /* y is 0.5 */
            if hx >= 0 {
                /* x >= +0 */
                return sqrt(x);
            }
        }
    }

    let mut ax: f64 = fabs(x);
    if lx == 0 {
        /* special value of x */
        if ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000 {
            /* x is +-0,+-inf,+-1 */
            let mut z: f64 = ax;

            if hy < 0 {
                /* z = (1/|x|) */
                z = 1.0 / z;
            }

            if hx < 0 {
                if ((ix - 0x3ff00000) | yisint) == 0 {
                    z = (z - z) / (z - z); /* (-1)**non-int is NaN */
                } else if yisint == 1 {
                    z = -z; /* (x<0)**odd = -(|x|**odd) */
                }
            }

            return z;
        }
    }

    let mut s: f64 = 1.0; /* sign of result */
    if hx < 0 {
        if yisint == 0 {
            /* (x<0)**(non-int) is NaN */
            return (x - x) / (x - x);
        }

        if yisint == 1 {
            /* (x<0)**(odd int) */
            s = -1.0;
        }
    }

    /* |y| is HUGE */
    if iy > 0x41e00000 {
        /* if |y| > 2**31 */
        if iy > 0x43f00000 {
            /* if |y| > 2**64, must o/uflow */
            if ix <= 0x3fefffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }

            if ix >= 0x3ff00000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }

        /* over/underflow if x is not close to one */
        if ix < 0x3fefffff {
            return if hy < 0 {
                s * HUGE * HUGE
            } else {
                s * TINY * TINY
            };
        }
        if ix > 0x3ff00000 {
            return if hy > 0 {
                s * HUGE * HUGE
            } else {
                s * TINY * TINY
            };
        }

        /* now |1-x| is TINY <= 2**-20, suffice to compute
        log(x) by x-x^2/2+x^3/3-x^4/4 */
        let t: f64 = ax - 1.0; /* t has 20 trailing zeros */
        let w: f64 = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        let u: f64 = IVLN2_H * t; /* ivln2_h has 21 sig. bits */
        let v: f64 = t * IVLN2_L - w * IVLN2;
        t1 = with_set_low_word(u + v, 0);
        t2 = v - (t1 - u);
    } else {
        // double ss,s2,s_h,s_l,t_h,t_l;
        let mut n: i32 = 0;

        if ix < 0x00100000 {
            /* take care subnormal number */
            ax *= TWO53;
            n -= 53;
            ix = get_high_word(ax) as i32;
        }

        n += (ix >> 20) - 0x3ff;
        j = ix & 0x000fffff;

        /* determine interval */
        let k: i32;
        ix = j | 0x3ff00000; /* normalize ix */
        if j <= 0x3988E {
            /* |x|<sqrt(3/2) */
            k = 0;
        } else if j < 0xBB67A {
            /* |x|<sqrt(3)   */
            k = 1;
        } else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax = with_set_high_word(ax, ix as u32);

        /* compute ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
        let u: f64 = ax - i!(BP, k as usize); /* bp[0]=1.0, bp[1]=1.5 */
        let v: f64 = 1.0 / (ax + i!(BP, k as usize));
        let ss: f64 = u * v;
        let s_h = with_set_low_word(ss, 0);

        /* t_h=ax+bp[k] High */
        let t_h: f64 = with_set_high_word(
            0.0,
            ((ix as u32 >> 1) | 0x20000000) + 0x00080000 + ((k as u32) << 18),
        );
        let t_l: f64 = ax - (t_h - i!(BP, k as usize));
        let s_l: f64 = v * ((u - s_h * t_h) - s_h * t_l);

        /* compute log(ax) */
        let s2: f64 = ss * ss;
        let mut r: f64 = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        let s2: f64 = s_h * s_h;
        let t_h: f64 = with_set_low_word(3.0 + s2 + r, 0);
        let t_l: f64 = r - ((t_h - 3.0) - s2);

        /* u+v = ss*(1+...) */
        let u: f64 = s_h * t_h;
        let v: f64 = s_l * t_h + t_l * ss;

        /* 2/(3log2)*(ss+...) */
        let p_h: f64 = with_set_low_word(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h: f64 = CP_H * p_h; /* cp_h+cp_l = 2/(3*log2) */
        let z_l: f64 = CP_L * p_h + p_l * CP + i!(DP_L, k as usize);

        /* log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l */
        let t: f64 = n as f64;
        t1 = with_set_low_word(((z_h + z_l) + i!(DP_H, k as usize)) + t, 0);
        t2 = z_l - (((t1 - t) - i!(DP_H, k as usize)) - z_h);
    }

    /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
    let y1: f64 = with_set_low_word(y, 0);
    let p_l: f64 = (y - y1) * t1 + y * t2;
    let mut p_h: f64 = y1 * t1;
    let z: f64 = p_l + p_h;
    let mut j: i32 = (z.to_bits() >> 32) as i32;
    let i: i32 = z.to_bits() as i32;
    // let (j, i): (i32, i32) = ((z.to_bits() >> 32) as i32, z.to_bits() as i32);

    if j >= 0x40900000 {
        /* z >= 1024 */
        if (j - 0x40900000) | i != 0 {
            /* if z > 1024 */
            return s * HUGE * HUGE; /* overflow */
        }

        if p_l + OVT > z - p_h {
            return s * HUGE * HUGE; /* overflow */
        }
    } else if (j & 0x7fffffff) >= 0x4090cc00 {
        /* z <= -1075 */
        // FIXME: instead of abs(j) use unsigned j

        if (((j as u32) - 0xc090cc00) | (i as u32)) != 0 {
            /* z < -1075 */
            return s * TINY * TINY; /* underflow */
        }

        if p_l <= z - p_h {
            return s * TINY * TINY; /* underflow */
        }
    }

    /* compute 2**(p_h+p_l) */
    let i: i32 = j & (0x7fffffff as i32);
    k = (i >> 20) - 0x3ff;
    let mut n: i32 = 0;

    if i > 0x3fe00000 {
        /* if |z| > 0.5, set n = [z+0.5] */
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff; /* new k for n */
        let t: f64 = with_set_high_word(0.0, (n & !(0x000fffff >> k)) as u32);
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }

    let t: f64 = with_set_low_word(p_l + p_h, 0);
    let u: f64 = t * LG2_H;
    let v: f64 = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let mut z: f64 = u + v;
    let w: f64 = v - (z - u);
    let t: f64 = z * z;
    let t1: f64 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r: f64 = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    j = get_high_word(z) as i32;
    j += n << 20;

    if (j >> 20) <= 0 {
        /* subnormal output */
        z = scalbn(z, n);
    } else {
        z = with_set_high_word(z, j as u32);
    }

    s * z
}

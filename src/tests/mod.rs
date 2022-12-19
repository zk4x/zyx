// TODO: move all the tests into the respective modules.
// No reason to have them all in here like this.

type Buffer<T, Sh> = crate::accel::cpu::Buffer<T, Sh>;

fn cmp_vec(x: &[f32], y: &[f32]) {
    const PRECISION: i32 = 3;
    for (a, b) in x.iter().zip(y.iter()) {
        let a = (a * 10f32.powi(PRECISION)).round();
        let b = (b * 10f32.powi(PRECISION)).round();
        if a.is_nan() && b.is_nan() { continue }
        assert_eq!(a, b);
    }
}

fn cmp_vec_f64(x: &[f64], y: &[f64]) {
    const PRECISION: i32 = 3;
    for (a, b) in x.iter().zip(y.iter()) {
        let a = (a * 10f64.powi(PRECISION)).round();
        let b = (b * 10f64.powi(PRECISION)).round();
        if a.is_nan() && b.is_nan() { continue }
        assert_eq!(a, b);
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray() {
    extern crate alloc;
    use alloc::vec;
    use crate::prelude::*;
    use ndarray::Array;

    let _x = Array::<f32, _>::eye(4).with_grad();
    //let x = x.sum((1));
    //println!("{}", x);

    let _x = Buffer::<f32>::eye(4);
    //println!("{}", x);

    //panic!();
    use ndarray::array;

    let x = array![[2., 4., 3.], [4., 2., 5.]];
    let y = x.with_grad();
    let z = y.exp();
    z.backward();

    //println!("{}", y);
    //panic!();
}

mod nn {
    #[test]
    fn linear() {
        use crate::prelude::*;
        use super::Buffer;
        use crate::nn;
        let linear = nn::Linear::<Buffer<_, _>, Buffer<_, _>>::new::<f32>(3, 2);
        //let linear = nn::Linear::new::<f32>(3, 2);
        let x = Buffer::cfrom([[2., 3., 1.]]);
        let z = linear.forward(x);
        assert_eq!(z.shape(), (1, 2));
        //println!("{}", z);
        z.backward();
    }
}

mod tensor {
    mod init {
        use crate::ops::ConvertFrom;
        use super::super::Buffer;

        #[test]
        fn from() {
            let _ = Buffer::cfrom([[2, 3]]);
            let _ = Buffer::cfrom([[2, 3], [3, 4], [5, 3]]);
            let _ = Buffer::cfrom([[[2, 3]]]);
            let _ = Buffer::cfrom([[[[2, 3]], [[2, 3]]]]);

            #[cfg(feature = "ndarray")]
            {
                use ndarray::{ArrayBase, OwnedRepr, Ix2};
                let _ = ArrayBase::<OwnedRepr<i32>, Ix2>::cfrom([[2, 3]]);
            }
        }

        #[test]
        fn ones() {
            use crate::ops::Ones;
            let _ = Buffer::<i32, (usize, usize, usize, usize)>::ones((3, 4, 2, 5));
        }

        #[test]
        fn zeros() {
            use crate::ops::Zeros;
            let _ = Buffer::<i128, _>::zeros((2usize, 3, 4));
        }

        #[test]
        fn randn() {
            use crate::init::RandnInit;
            let _ = Buffer::<f32, _>::randn([3usize, 1, 4]);
        }

        #[test]
        fn uniform() {
            use crate::init::UniformInit;
            let _ = Buffer::uniform((2usize, 4, 1, 6, 3), 0., 1.);
        }
    }

    mod ops {
        use crate::{ops::{HasShape, FromVec, IntoVec, ConvertFrom, ConvertInto}, tensor::IntoVariable};
        use super::super::{cmp_vec, cmp_vec_f64, Buffer};
        extern crate alloc;
        use alloc::vec;
        use alloc::vec::Vec;

        #[test]
        fn convert_from() {
            // TODO finish all variations, including type and accelerator conversions
            use crate::ops::ConvertInto;
            let vec = vec![3f32, 1., 2., 4.];
            let x = Buffer::from_vec(&vec, (1usize, 4, 1));
            let y = Buffer::<f64, _>::cfrom(x.clone());
            cmp_vec_f64(&vec.clone().into_iter().map(|x| x as f64).collect::<Vec<f64>>(), &y.to_vec());
            let y: Buffer::<f64, _> = x.cinto();
            cmp_vec_f64(&vec.into_iter().map(|x| x as f64).collect::<Vec<f64>>(), &y.to_vec());
        }

        #[test]
        fn get_shape() {
            // TODO finish all variations
            let vec = vec![3., 1., 2., 4.];
            let x = Buffer::from_vec(&vec, (1usize, 4, 1));
            assert_eq!((1, 4, 1), x.shape());
        }

        #[test]
        fn relu() {
            // TODO: all tests should look like this, test Buffer, Variable and Tensor,
            // with binary operators, also test all 9 variations like Buffer + Variable and Variable + Buffer
            use crate::ops::ReLU;
            let vec = vec![3., 1., 2., 4.];
            // test Buffer
            let x = Buffer::from_vec(&vec, 4);
            let y = x.clone().relu();
            cmp_vec(&vec.iter().map(|x| if *x > 0. { *x } else { 0. }).collect::<Vec<f32>>(), &y.to_vec());
            // test Variable
            let x = x.with_grad();
            let y = x.relu();
            cmp_vec(&vec.iter().map(|x| x.max(0.)).collect::<Vec<f32>>(), &y.data().clone().to_vec());
            y.backward();
            cmp_vec(&x.grad().clone().to_vec(), &[1., 1., 1., 1.]);
            // test Tensor
            let y = x.relu().relu();
            cmp_vec(&vec.iter().map(|x| x.max(0.)).collect::<Vec<f32>>(), &y.data().clone().to_vec());
            y.backward();
            cmp_vec(&x.grad().clone().to_vec(), &[2., 2., 2., 2.]);
        }

        #[test]
        fn exp() {
            use crate::ops::Exp;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            // test Buffer
            let x = Buffer::from_vec(&vec, 9);
            let y = x.clone().exp();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &y.to_vec());
            // test Variable
            let x = x.with_grad();
            let y = x.exp();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &y.data().clone().to_vec());
            y.backward();
            cmp_vec(&x.grad().clone().to_vec(), &vec.iter().map(|x| x.exp()).collect::<Vec<f32>>());
            // test Tensor
            //let h = x.register_hook(|grad| println!("Reg grad: {}", grad));
            let y = x.exp().exp();
            cmp_vec(&vec.iter().map(|x| x.exp().exp()).collect::<Vec<f32>>(), &y.data().clone().to_vec());
            y.backward();
            cmp_vec(&x.grad().clone().to_vec(), &vec.iter().map(|x| x.exp() + (x.exp() + x).exp()).collect::<Vec<f32>>());
        }

        #[test]
        fn ln() {
            use crate::ops::Ln;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            // test Buffer
            let x = Buffer::from_vec(&vec, 9);
            let y = x.clone().ln();
            cmp_vec(&vec.iter().map(|x| x.ln()).collect::<Vec<f32>>(), &y.to_vec());
            // test Variable
            let x = x.with_grad();
            let y = x.ln();
            cmp_vec(&vec.iter().map(|x| x.ln()).collect::<Vec<f32>>(), &y.data().clone().to_vec());
            y.backward();
            cmp_vec(&x.grad().clone().to_vec(), &vec.iter().map(|x| 1./x).collect::<Vec<f32>>());
            // test Tensor
            let y = x.ln().ln();
            cmp_vec(&vec.iter().map(|x| x.ln().ln()).collect::<Vec<f32>>(), &y.data().clone().to_vec());
            y.backward();
            cmp_vec(&x.grad().clone().to_vec(), &vec.iter().map(|x| 1./x + 1./(x*x.ln())).collect::<Vec<f32>>());
        }

        #[test]
        fn tanh() {
            // TODO finish all variations
            use crate::ops::Tanh;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = x.tanh();
            assert_eq!(vec.iter().map(|x| x.tanh()).collect::<Vec<f32>>(), y.data().clone().to_vec());
            y.backward();
            assert_eq!(x.grad().clone().to_vec(), vec.iter().map(|x| 1. - x.tanh().powi(2)).collect::<Vec<f32>>());
        }

        #[test]
        fn neg() {
            // TODO finish all variations
            use crate::ops::IntoVec;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = -&x;
            assert_eq!(vec.iter().map(|x| -x).collect::<Vec<f32>>(), y.data().to_vec());
            y.backward();
            assert_eq!(x.grad().to_vec(), vec.iter().map(|_| -1.).collect::<Vec<f32>>());
        }

        #[test]
        fn sum() {
            // TODO finish all variations
            use crate::ops::Sum;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, (1usize, 3, 1, 3, 1)).with_grad();
            let y = x.sum((-1i32, -2));
            cmp_vec(&[6., 5., 12.], &y.to_vec());
        }

        #[test]
        fn max() {
            // TODO finish all variations
            /*use crate::ops::Max;
            let vec: Vec<f32> = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, (1usize, 3, 1, 3, 1));
            let y = x.max((-1i32, -2));
            let x = Buffer::from_vec(&vec, (1usize, 3, 1, 3, 1)).with_grad();
            let y = x.max(-1);
            cmp_vec(&[3., 4., 5.], &y.to_vec());*/
        }

        #[test]
        fn min() {
            // TODO finish all variations
            /*use crate::ops::Min;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, (1usize, 3, 1, 3, 1)).with_grad();
            let y = x.min((-1i32, -2));
            cmp_vec(&[1., 0., 3.], &y.to_vec());*/
        }

        #[test]
        fn reshape() {
            // TODO finish all variations
            use crate::ops::Reshape;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, (1usize, 1, 9));
            let y = x.reshape((3usize, 3));
            assert_eq!(vec, y.to_vec())
        }

        #[test]
        fn expand() {
            // TODO finish all variations
            use crate::ops::Expand;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, (1usize, 1, 9));
            let y = x.expand((3usize, 1, 9));
            assert_eq!(y.shape(), (3, 1, 9));
            //assert_eq!(vec, &y.to_vec().into_iter().repeat(3).collect::<Vec<f32>>())
        }

        #[test]
        fn permute() {
            // TODO finish all variations
            use crate::ops::Permute;
            let x = Buffer::cfrom([[2, 3, 1], [3, 4, 5]]);
            let _ = x.permute((-1, -2));

            let x = Buffer::cfrom([[[3, 2], [4, 1], [4, 2]], [[2, 3], [3, 4], [4, 1]]]);
            let y = x.clone().permute((2, 1, 0));
            assert_eq!(&y.to_vec(), &[3, 2, 4, 3, 4, 4, 2, 3, 1, 4, 2, 1]);
            assert_eq!(y.shape(), (2, 3, 2));

            let y = x.clone().permute((1, 0, 2));
            assert_eq!(&y.to_vec(), &[3, 2, 2, 3, 4, 1, 3, 4, 4, 2, 4, 1]);
            assert_eq!(y.shape(), (3, 2, 2));

            let y = x.clone().permute((0, 2, 1));
            assert_eq!(&y.to_vec(), &[3, 4, 4, 2, 1, 2, 2, 3, 4, 3, 4, 1]);
            assert_eq!(y.shape(), (2, 2, 3));

            let y = x.clone().permute((1, 2, 0));
            assert_eq!(&y.to_vec(), &[3, 2, 2, 3, 4, 3, 1, 4, 4, 4, 2, 1]);
            assert_eq!(y.shape(), (3, 2, 2));

            let y = x.clone().permute((2, 0, 1));
            assert_eq!(&y.to_vec(), &[3, 4, 4, 2, 3, 4, 2, 1, 2, 3, 4, 1]);
            assert_eq!(y.shape(), (2, 2, 3));

            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::from_vec(&vec, (9, 1)).with_grad();
            let y = x.permute((1, 0));
            assert_eq!(vec, y.to_vec());
            y.backward();
            assert_eq!(x.grad().to_vec(), vec.iter().map(|_| 1.).collect::<Vec<f32>>());
        }

        #[test]
        fn add() {
            use crate::ops::Exp;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let vec2 = vec![4., 2., 2., 4., 1., -2., 4., 3., 7.];

            // test Buffer
            let x = Buffer::from_vec(&vec, 9);
            let y = Buffer::from_vec(&vec2, 9);
            let z = x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());

            // test Variable
            let x = Buffer::from_vec(&vec, 9);
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = Buffer::from_vec(&vec2, 9);
            let z = &x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = &x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            // test Tensor
            let x = Buffer::from_vec(&vec, 9);
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = y.exp() + x.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = Buffer::from_vec(&vec2, 9);
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = &x + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = x.exp() + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(&vec, 9).with_grad();
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());
        }

        #[test]
        fn add_scalar() {
            // TODO
            let x = Buffer::<f32, _>::cfrom([[2., 3., 1.], [3., 4., 5.]]);
            let _y = x.clone()/2f32;
            let _y = x.clone()/2f64;
            let _y = x.clone()/2i8;
            let _y = x.clone()/2i16;
            let _y = x.clone()/2i32;
            let _y = x.clone()/2i64;
            let _y = x.clone()/2i128;
            let _y = x.clone()/2isize;
            let _y = x.clone()/2u8;
            let _y = x.clone()/2u16;
            let _y = x.clone()/2u32;
            let _y = x.clone()/2u64;
            let _y = x.clone()/2u128;
            let _y = x.clone()/2usize;
            let _x: Buffer<i32, _> = x.cinto();
            //println!("{}", x);
            //panic!();
        }

        #[test]
        fn sub() {
            // TODO
        }

        #[test]
        fn mul() {
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let vec2 = vec![4., 2., 2., 4., 1., -2., 4., 3., 7.];

            // test Buffer
            let x = Buffer::from_vec(&vec, 9);
            let y = Buffer::from_vec(&vec2, 9);
            let z = x * y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x * *y).collect::<Vec<f32>>(), &z.to_vec());

            // test Variable
            let x = Buffer::from_vec(&vec, 9);
            let y = Buffer::from_vec(&vec2, 9).with_grad();
            let z = x * &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x * *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            //println!("{}", y);
            cmp_vec(&vec, &y.grad().to_vec());

            /*let x = Buffer::from_vec(vec.clone(), &[9]).with_grad();
            let y = Buffer::from_vec(vec2.clone(), &[9]);
            let z = &x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x = Buffer::from_vec(vec.clone(), &[9]).with_grad();
            let y = Buffer::from_vec(vec2.clone(), &[9]).with_grad();
            let z = &x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            // test Tensor
            let x = Buffer::from_vec(vec.clone(), &[9]);
            let y = Buffer::from_vec(vec2.clone(), &[9]).with_grad();
            let z = y.exp() + x.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(vec.clone(), &[9]).with_grad();
            let y = Buffer::from_vec(vec2.clone(), &[9]);
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x = Buffer::from_vec(vec.clone(), &[9]).with_grad();
            let y = Buffer::from_vec(vec2.clone(), &[9]).with_grad();
            let z = &x + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(vec.clone(), &[9]).with_grad();
            let y = Buffer::from_vec(vec2.clone(), &[9]).with_grad();
            let z = x.exp() + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::from_vec(vec.clone(), &[9]).with_grad();
            let y = Buffer::from_vec(vec2.clone(), &[9]).with_grad();
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());*/
        }

        #[test]
        fn div() {
            // TODO
        }

        #[test]
        fn pow() {
            // TODO
        }

        #[test]
        fn pow_scalar() {
            // TODO
            use crate::ops::Pow;
            let x = Buffer::<f32, _>::cfrom([[2., 3., 1.], [3., 4., 5.]]);
            let _y = x.clone().pow(2i32);
            let _x: Buffer<i32, _> = x.cinto();
            //println!("{}", x);
            //panic!();
        }

        #[test]
        fn matmul() {
            // TODO finish all variations
            use crate::ops::{MatMul, ConvertFrom};
            let x = Buffer::cfrom([[2f32, 3., 4.]]);
            let y = Buffer::cfrom([[2., 3.], [3., 4.], [5., 3.]]);
            let z = x.clone().matmul(y.clone());
            assert_eq!(z.to_vec(), [33., 30.]);

            let x = x;
            let y = y.with_grad();
            let z = x.clone().matmul(&y);
            assert_eq!(z.data().to_vec(), [33., 30.]);
            z.backward();
            assert_eq!(y.grad().to_vec(), [2., 2., 3., 3., 4., 4.]);

            let x = x.with_grad();
            let y = Buffer::cfrom([[2., 3.], [3., 4.], [5., 3.]]);
            let z = x.matmul(y.clone());
            assert_eq!(z.data().to_vec(), [33., 30.]);
            z.backward();
            assert_eq!(x.grad().to_vec(), [5., 7., 8.].to_vec());

            let x = Buffer::<f32, _>::cfrom([[2., 3., 4.]]).with_grad();
            let y = Buffer::cfrom([[2., 3.], [3., 4.], [5., 3.]]).with_grad();
            let z = x.matmul(&y);
            assert_eq!(z.data().to_vec(), [33., 30.]);
            z.backward();
            assert_eq!(x.grad().to_vec(), [5., 7., 8.].to_vec());
            assert_eq!(y.grad().to_vec(), [2., 2., 3., 3., 4., 4.].to_vec());
        }

        #[test]
        fn conv() {
            use crate::ops::Conv;
            let x = Buffer::cfrom([[2, 3, 4, 1], [4, 2, 1, 3]]);
            let y = Buffer::cfrom([[2, 3, 2], [3, 4, 1]]);
            let _ = x.clone().conv(y.clone(), (1usize, 2));
            //println!("{}", x);
            //println!("{}", y);
            //println!("{}", z);
            //panic!()
        }
    }
}

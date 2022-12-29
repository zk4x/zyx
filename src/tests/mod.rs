// TODO: move all the tests into the respective modules.
// No reason to have them all in here like this.

use crate::device::cpu::Buffer;

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

mod tensor {
    use super::Buffer;

    mod init {
        use crate::shape::{Sh3, Sh4, Sh5};
        use super::Buffer;

        #[test]
        fn cfrom() {
            use crate::prelude::*;
            use crate::device::cpu::Device;

            let device = Device::default();
            let _ = device.buffer([[2, 3]]);
            let _ = device.buffer([[2, 3], [3, 4], [5, 3]]);
            let _ = device.buffer([[[2, 3]]]);
            let _ = device.buffer([[[[2, 3]], [[2, 3]]]]);

            #[cfg(feature = "ndarray")]
            {
                use ndarray::{ArrayBase, OwnedRepr, Ix2};
                let _ = ArrayBase::<OwnedRepr<i32>, Ix2>::cfrom([[2, 3]]);
            }
        }

        #[test]
        fn ones() {
            use crate::prelude::*;
            use crate::device::cpu::Device;
            let device = Device::default();
            let _: Buffer<'_, Sh4<3, 4, 2, 5>> = device.ones();
        }

        #[test]
        fn zeros() {
            use crate::prelude::*;
            use crate::device::cpu::Device;
            let device = Device::default();
            let _: Buffer<'_, Sh3<2, 3, 4>> = device.zeros();
        }

        #[test]
        fn randn() {
            use crate::prelude::*;
            use crate::device::cpu::Device;
            let device = Device::default();
            let _: Buffer<'_, Sh3<3, 1, 4>> = device.randn();
        }

        #[test]
        fn uniform() {
            use crate::prelude::*;
            use crate::device::cpu::Device;
            let device = Device::default();
            let _: Buffer<'_, Sh5<2, 4, 1, 6, 3>> = device.uniform(0., 1.);
        }
    }

    mod ops {
        //use crate::prelude::*;
        use super::super::{cmp_vec, cmp_vec_f64, Buffer};
        extern crate alloc;
        use alloc::vec;
        use alloc::vec::Vec;
        use crate::shape::{Sh1, Sh2, Sh3, Ax3};

        #[test]
        fn convert_from() {
            use crate::prelude::*;
            use crate::device::cpu::Device;
            let device = Device::default();
            // TODO finish all variations, including type and device conversions
            use crate::ops::ConvertInto;
            let vec = vec![3f32, 1., 2., 4.];
            let x: Buffer<'_, Sh3<1, 4, 1>> = device.slice(&vec);
            let y = Buffer::<'_, _, f64>::cfrom(x.clone());
            cmp_vec_f64(&vec.clone().into_iter().map(|x| x as f64).collect::<Vec<f64>>(), &y.to_vec());
            let y: Buffer::<'_, _, f64> = x.cinto();
            cmp_vec_f64(&vec.into_iter().map(|x| x as f64).collect::<Vec<f64>>(), &y.to_vec());
        }

        #[test]
        fn get_shape() {
            // TODO finish all variations
            use crate::prelude::*;
            use crate::device::cpu::Device;
            let device = Device::default();
            let vec = vec![3., 1., 2., 4.];
            let x: Buffer<'_, Sh3<1, 4, 1>> = device.slice(&vec);
            assert_eq!([1, 4, 1], x.shape());
        }

        #[test]
        fn relu() {
            // TODO: all tests should look like this, test Buffer, Variable and Tensor,
            // with binary operators, also test all 9 variations like Buffer + Variable and Variable + Buffer
            //use crate::prelude::*;
            use crate::device::BufferFromSlice;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            let device = Device::default();
            use crate::ops::ReLU;
            let vec = vec![3., 1., 2., 4.];
            // test Buffer
            let x: Buffer<'_, Sh1<4>> = device.slice(&vec);
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
            use crate::device::BufferFromSlice;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            let device = Device::default();
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            // test Buffer
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
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
            use crate::device::BufferFromSlice;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            let device = Device::default();
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            // test Buffer
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
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
            use crate::device::BufferFromSlice;
            use crate::tensor::Variable;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            let device = Device::default();
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y = x.tanh();
            assert_eq!(vec.iter().map(|x| x.tanh()).collect::<Vec<f32>>(), y.data().clone().to_vec());
            y.backward();
            assert_eq!(x.grad().clone().to_vec(), vec.iter().map(|x| 1. - x.tanh().powi(2)).collect::<Vec<f32>>());
        }

        #[test]
        fn neg() {
            // TODO finish all variations
            use crate::device::BufferFromSlice;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            use crate::tensor::Variable;
            let device = Device::default();
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y = -&x;
            assert_eq!(vec.iter().map(|x| -x).collect::<Vec<f32>>(), y.data().to_vec());
            y.backward();
            assert_eq!(x.grad().to_vec(), vec.iter().map(|_| -1.).collect::<Vec<f32>>());
        }

        #[test]
        fn max() {
            // TODO finish all variations
            /*use crate::ops::Maximizable;
            let vec: Vec<f32> = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::slice(&vec, (1usize, 3, 1, 3, 1));
            let y = x.max((-1i32, -2));
            let x = Buffer::slice(&vec, (1usize, 3, 1, 3, 1)).with_grad();
            let y = x.max(-1);
            cmp_vec(&[3., 4., 5.], &y.to_vec());*/
        }

        #[test]
        fn min() {
            // TODO finish all variations
            /*use crate::ops::Min;
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Buffer::slice(&vec, (1usize, 3, 1, 3, 1)).with_grad();
            let y = x.min((-1i32, -2));
            cmp_vec(&[1., 0., 3.], &y.to_vec());*/
        }

        #[test]
        fn reshape() {
            // TODO finish all variations
            use crate::prelude::*;
            use crate::device::cpu::{Device, Buffer};
            let device = Device::default();
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x: Buffer<'_, Sh3<1, 9, 1>> = device.slice(&vec);
            let y = x.reshape::<Sh2<3, 3>>();
            assert_eq!(vec, y.to_vec())
        }

        #[test]
        fn expand() {
            // TODO finish all variations
            use crate::prelude::*;
            use crate::shape::Ax1;
            use crate::device::cpu::Device;
            let device = Device::default();
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x: Buffer<'_, Sh3<1, 1, 9>> = device.slice(&vec);
            let y = x.expand::<Sh3<3, 1, 9>, Ax1<0>>();
            assert_eq!(y.shape(), [3, 1, 9]);
            //assert_eq!(vec, &y.to_vec().into_iter().repeat(3).collect::<Vec<f32>>())
        }

        #[test]
        fn permute() {
            // TODO finish all variations
            use crate::ops::Permute;
            use crate::device::{BufferFromSlice, ShapedBufferInit};
            use crate::ops::{IntoVec, IntoVariable, HasShape};
            use crate::tensor::Variable;
            use crate::device::cpu::Device;
            let device = Device::default();
            //let x = Buffer::cfrom([[2, 3, 1], [3, 4, 5]]);
            //let _ = x.permute::<Ax2<-1, -2>>();

            let x = device.buffer([[[3, 2], [4, 1], [4, 2]], [[2, 3], [3, 4], [4, 1]]]);
            let y = x.clone().permute::<Ax3<2, 1, 0>>();
            assert_eq!(&y.to_vec(), &[3, 2, 4, 3, 4, 4, 2, 3, 1, 4, 2, 1]);
            assert_eq!(y.shape(), [2, 3, 2]);

            let y = x.clone().permute::<Ax3<1, 0, 2>>();
            assert_eq!(&y.to_vec(), &[3, 2, 2, 3, 4, 1, 3, 4, 4, 2, 4, 1]);
            assert_eq!(y.shape(), [3, 2, 2]);

            let y = x.clone().permute::<Ax3<0, 2, 1>>();
            assert_eq!(&y.to_vec(), &[3, 4, 4, 2, 1, 2, 2, 3, 4, 3, 4, 1]);
            assert_eq!(y.shape(), [2, 2, 3]);

            let y = x.clone().permute::<Ax3<1, 2, 0>>();
            assert_eq!(&y.to_vec(), &[3, 2, 2, 3, 4, 3, 1, 4, 4, 4, 2, 1]);
            assert_eq!(y.shape(), [3, 2, 2]);

            let y = x.clone().permute::<Ax3<2, 0, 1>>();
            assert_eq!(&y.to_vec(), &[3, 4, 4, 2, 3, 4, 2, 1, 2, 3, 4, 1]);
            assert_eq!(y.shape(), [2, 2, 3]);

            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x: Variable<Buffer<'_, Sh2<9, 1>>> = device.slice(&vec).with_grad();
            //let y = (&x).permute::<Ax2<1, 0>>();
            use crate::ops::Transpose;
            let y = x.transpose();
            assert_eq!(vec, y.to_vec());
            y.backward();
            assert_eq!(x.grad().to_vec(), vec.iter().map(|_| 1.).collect::<Vec<f32>>());
        }

        #[test]
        fn add() {
            use crate::ops::Exp;
            use crate::device::BufferFromSlice;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            use crate::tensor::Variable;

            let device = Device::default();

            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let vec2 = vec![4., 2., 2., 4., 1., -2., 4., 3., 7.];

            // test Buffer
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
            let y: Buffer<'_, Sh1<9>> = device.slice(&vec2);
            let z = x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());

            // test Variable
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y: Buffer<'_, Sh1<9>> = device.slice(&vec2);
            let z = &x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = &x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            // test Tensor
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = y.exp() + x.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y: Buffer<'_, Sh1<9>> = device.slice(&vec2);
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = &x + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = x.exp() + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec).with_grad();
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());
        }

        #[test]
        fn add_scalar() {
            // TODO
            use crate::ops::ConvertInto;
            use crate::device::ShapedBufferInit;
            use crate::device::cpu::Device;
            let device = Device::default();

            let x = device.buffer([[2., 3., 1.], [3., 4., 5.]]);
            /*let _y = x.clone()/2f32;
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
            let _y = x.clone()/2usize;*/
            let _x: Buffer<'_, _, i32> = x.cinto();
            //println!("{}", x);
            //panic!();
        }

        #[test]
        fn sub() {
            // TODO
        }

        #[test]
        fn mul() {
            use crate::device::BufferFromSlice;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;
            use crate::tensor::Variable;
            let device = Device::default();

            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let vec2 = vec![4., 2., 2., 4., 1., -2., 4., 3., 7.];

            // test Buffer
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
            let y: Buffer<'_, Sh1<9>> = device.slice(&vec2);
            let z = x * y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x * *y).collect::<Vec<f32>>(), &z.to_vec());

            // test Variable
            let x: Buffer<'_, Sh1<9>> = device.slice(&vec);
            let y: Variable<Buffer<'_, Sh1<9>>> = device.slice(&vec2).with_grad();
            let z = x * &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x * *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            //println!("{}", y);
            cmp_vec(&vec, &y.grad().to_vec());

            /*let x = Buffer::slice(vec.clone(), &[9]).with_grad();
            let y = Buffer::slice(vec2.clone(), &[9]);
            let z = &x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x = Buffer::slice(vec.clone(), &[9]).with_grad();
            let y = Buffer::slice(vec2.clone(), &[9]).with_grad();
            let z = &x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            // test Tensor
            let x = Buffer::slice(vec.clone(), &[9]);
            let y = Buffer::slice(vec2.clone(), &[9]).with_grad();
            let z = y.exp() + x.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::slice(vec.clone(), &[9]).with_grad();
            let y = Buffer::slice(vec2.clone(), &[9]);
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());

            let x = Buffer::slice(vec.clone(), &[9]).with_grad();
            let y = Buffer::slice(vec2.clone(), &[9]).with_grad();
            let z = &x + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::slice(vec.clone(), &[9]).with_grad();
            let y = Buffer::slice(vec2.clone(), &[9]).with_grad();
            let z = x.exp() + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().to_vec());

            let x = Buffer::slice(vec.clone(), &[9]).with_grad();
            let y = Buffer::slice(vec2.clone(), &[9]).with_grad();
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
            use crate::device::ShapedBufferInit;
            use crate::device::cpu::Device;
            let device = Device::default();

            use crate::ops::Pow;
            let x = device.buffer([[2., 3., 1.], [3., 4., 5.]]);
            let _y = x.clone().pow(2);
            use crate::ops::ConvertInto;
            let _x: Buffer<'_, _, i32> = x.cinto();
            //println!("{}", x);
            //panic!();
        }

        #[test]
        fn matmul() {
            // TODO finish all variations
            use crate::device::ShapedBufferInit;
            use crate::ops::{IntoVec, IntoVariable};
            use crate::device::cpu::Device;

            use crate::ops::MatMul;

            let device = Device::default();
            
            let x = device.buffer([[2f32, 3., 4.]]);
            let y = device.buffer([[2., 3.], [3., 4.], [5., 3.]]);
            let z = x.clone().matmul(y.clone());
            assert_eq!(z.to_vec(), [33., 30.]);

            let x = x;
            let y = y.with_grad();
            let z = x.clone().matmul(&y);
            assert_eq!(z.data().to_vec(), [33., 30.]);
            z.backward();
            assert_eq!(y.grad().to_vec(), [2., 2., 3., 3., 4., 4.]);

            let x = x.with_grad();
            let y = device.buffer([[2., 3.], [3., 4.], [5., 3.]]);
            let z = x.matmul(y.clone());
            assert_eq!(z.data().to_vec(), [33., 30.]);
            z.backward();
            assert_eq!(x.grad().to_vec(), [5., 7., 8.].to_vec());

            let x = device.buffer([[2f32, 3., 4.]]).with_grad();
            let y = device.buffer([[2., 3.], [3., 4.], [5., 3.]]).with_grad();
            let z = x.matmul(&y);
            assert_eq!(z.data().to_vec(), [33., 30.]);
            z.backward();
            assert_eq!(x.grad().to_vec(), [5., 7., 8.].to_vec());
            assert_eq!(y.grad().to_vec(), [2., 2., 3., 3., 4., 4.].to_vec());
        }

        /*#[test]
        fn conv() {
            //use crate::ops::Conv;
            let x = Buffer::cfrom([[2, 3, 4, 1], [4, 2, 1, 3]]);
            let y = Buffer::cfrom([[2, 3, 2], [3, 4, 1]]);
            //let _ = x.clone().conv(y.clone(), (1usize, 2));
            //println!("{}", x);
            //println!("{}", y);
            //println!("{}", z);
            //panic!()
        }*/
    }
}

mod nn {
    #[test]
    fn linear() {
        use crate::prelude::*;
        use crate::device::cpu::{Device, Buffer};
        use crate::shape::Sh2;
        use crate::nn;

        let device = Device::default();

        let mut linear = nn::Linear::<3, 2>::new(&device);
        let x = device.buffer([[2., 3., 1.]]);
        let z = linear.forward(x);
        assert_eq!(z.shape(), [1, 2]);
        //println!("{}", z);
        z.backward();
        let params = <nn::Linear<'_, 3, 2, _, _> as nn::module::Module<'_, Buffer<'_, Sh2<1, 3>, _>>>::parameters(&mut linear);
        std::println!("{}\n{}", params.0, params.1);
        panic!();
    }
}

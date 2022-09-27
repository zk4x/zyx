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

mod nn {
    #[test]
    fn linear() {
        use crate::prelude::*;
        use crate::tensor::Tensor;
        use crate::nn;
        use crate::buffer::cpu;
        let linear = nn::Linear::new::<f32>(3, 2);
        let x: Tensor<cpu::Buffer<f64>> = Tensor::from([[2., 3., 1.]]);
        let z = linear.forward(x);
        assert_eq!(z.shape(), vec![1, 2]);
        //println!("{}", z);
        z.backward();
    }
}

mod tensor {
    mod init {
        #[test]
        fn from() {
            use crate::tensor::Tensor;
            let _ = Tensor::from([[2, 3]]);
            let _ = Tensor::from([[2, 3], [3, 4], [5, 3]]);
            let _ = Tensor::from([[[2, 3]]]);
            let _ = Tensor::from([[[[2, 3]], [[2, 3]]]]);
        }

        #[test]
        fn ones() {
            use crate::{tensor::Tensor, ops::Ones, buffer::cpu};
            let _ = Tensor::<cpu::Buffer<f32>>::ones(&[3, 4, 2, 5]);
        }

        #[test]
        fn zeros() {
            use crate::{tensor::Tensor, ops::Zeros, buffer::cpu};
            let _ = Tensor::<cpu::Buffer<f32>>::zeros(&[2, 3, 4]);
        }

        #[test]
        fn randn() {
            use crate::{buffer::cpu, tensor::Tensor};
            let _ = Tensor::<cpu::Buffer<f32>>::randn(&[3, 1, 4]);
        }

        #[test]
        fn uniform() {
            use crate::{buffer::cpu, tensor::Tensor};
            let _ = Tensor::<cpu::Buffer<f32>>::uniform(&[2, 4, 1, 6, 3], 0., 1.);
        }
    }

    mod ops {
        use super::super::{cmp_vec, cmp_vec_f64};
        #[test]
        fn convert_from() {
            use crate::{tensor::Tensor, buffer::cpu, ops::ToVec, ops::ConvertFrom};
            let vec = vec![3f32, 1., 2., 4.];
            let x = Tensor::from_vec(vec.clone(), &[1, 4, 1]);
            let y = Tensor::<cpu::Buffer<f64>>::convert_from(x);
            cmp_vec_f64(&vec.into_iter().map(|x| x as f64).collect::<Vec<f64>>(), &y.data().to_vec());
        }

        #[test]
        fn get_shape() {
            use crate::{tensor::Tensor, ops::GetShape};
            let vec = vec![3f32, 1., 2., 4.];
            let x = Tensor::from_vec(vec.clone(), &[1, 4, 1]);
            assert_eq!(vec![1, 4, 1], x.shape());
        }

        #[test]
        fn relu() {
            // TODO: all tests should look like this, test Tensor, TensorGrad and TensorFunc,
            // with binary operators, also test all 9 variations like Tensor + TensorGrad and TensorGrad + Tensor
            use crate::{tensor::Tensor, ops::ToVec, ops::ReLU};
            let vec = vec![3., 1., 2., 4.];
            // test Tensor
            let x = Tensor::from_vec(vec.clone(), &[4]);
            let y = x.clone().relu();
            cmp_vec(&vec.iter().map(|x| if *x > 0. { *x } else { 0. }).collect::<Vec<f32>>(), &y.data().to_vec());
            // test TensorGrad
            let x = x.with_grad();
            let y = x.relu();
            cmp_vec(&vec.iter().map(|x| x.max(0.)).collect::<Vec<f32>>(), &y.data().to_vec());
            y.backward();
            cmp_vec(&x.grad().borrow().to_vec(), &[1., 1., 1., 1.]);
            // test TensorFunc
            let y = x.relu().relu();
            cmp_vec(&vec.iter().map(|x| x.max(0.)).collect::<Vec<f32>>(), &y.data().to_vec());
            y.backward();
            cmp_vec(&x.grad().borrow().to_vec(), &[2., 2., 2., 2.]);
        }

        #[test]
        fn exp() {
            use crate::{tensor::Tensor, ops::ToVec, ops::Exp};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            // test Tensor
            let x = Tensor::from_vec(vec.clone(), &[9]);
            let y = x.clone().exp();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &y.data().to_vec());
            // test TensorGrad
            let x = x.with_grad();
            let y = x.exp();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &y.data().to_vec());
            y.backward();
            cmp_vec(&x.grad().borrow().to_vec(), &vec.iter().map(|x| x.exp()).collect::<Vec<f32>>());
            // test TensorFunc
            let y = x.exp().exp();
            cmp_vec(&vec.iter().map(|x| x.exp().exp()).collect::<Vec<f32>>(), &y.data().to_vec());
            y.backward();
            cmp_vec(&x.grad().borrow().to_vec(), &vec.iter().map(|x| x.exp() + (x.exp() + x).exp()).collect::<Vec<f32>>());
        }

        #[test]
        fn ln() {
            use crate::{tensor::Tensor, ops::ToVec, ops::Ln};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            // test Tensor
            let x = Tensor::from_vec(vec.clone(), &[9]);
            let y = x.clone().ln();
            cmp_vec(&vec.iter().map(|x| x.ln()).collect::<Vec<f32>>(), &y.data().to_vec());
            // test TensorGrad
            let x = x.with_grad();
            let y = x.ln();
            cmp_vec(&vec.iter().map(|x| x.ln()).collect::<Vec<f32>>(), &y.data().to_vec());
            y.backward();
            cmp_vec(&x.grad().borrow().to_vec(), &vec.iter().map(|x| 1./x).collect::<Vec<f32>>());
            // test TensorFunc
            let y = x.ln().ln();
            cmp_vec(&vec.iter().map(|x| x.ln().ln()).collect::<Vec<f32>>(), &y.data().to_vec());
            y.backward();
            cmp_vec(&x.grad().borrow().to_vec(), &vec.iter().map(|x| 1./x + 1./(x*x.ln())).collect::<Vec<f32>>());
        }

        #[test]
        fn tanh() {
            use crate::{tensor::Tensor, buffer::cpu, ops::ToVec, ops::Tanh};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Tensor::<cpu::Buffer<f32>>::from_vec(vec.clone(), &[9]).with_grad();
            let y = x.tanh();
            assert_eq!(vec.iter().map(|x| x.tanh()).collect::<Vec<f32>>(), y.data().to_vec());
            y.backward();
            assert_eq!(x.grad().borrow().to_vec(), vec.iter().map(|x| 1. - x.tanh().powi(2)).collect::<Vec<f32>>());
        }

        #[test]
        fn neg() {
            use crate::{tensor::Tensor, ops::ToVec};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = -&x;
            assert_eq!(vec.iter().map(|x| -x).collect::<Vec<f32>>(), y.data().to_vec());
            y.backward();
            assert_eq!(x.grad().borrow().to_vec(), vec.iter().map(|_| -1.).collect::<Vec<f32>>());
        }

        #[test]
        fn sum() {
            use crate::{tensor::Tensor, ops::ToVec, ops::Sum};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Tensor::from_vec(vec.clone(), &[1, 3, 1, 3, 1]).with_grad();
            let y = x.sum(&[-1, -2]);
            cmp_vec(&[6., 5., 12.], &y.to_vec());
        }

        #[test]
        fn max() {
            // TODO
        }

        #[test]
        fn min() {
            // TODO
        }

        #[test]
        fn reshape() {
            use crate::{tensor::Tensor, ops::ToVec, ops::Reshape};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Tensor::from_vec(vec.clone(), &[1, 1, 9]);
            let y = x.reshape(&[3, 1, 3]);
            assert_eq!(vec, y.to_vec())
        }

        #[test]
        fn expand() {
            // TODO
        }

        #[test]
        fn permute() {
            use crate::{tensor::Tensor, ops::ToVec, ops::Permute};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let x = Tensor::from_vec(vec.clone(), &[9, 1]).with_grad();
            let y = x.permute(&[1, 0]);
            assert_eq!(vec, y.to_vec());
            y.backward();
            assert_eq!(x.grad().borrow().to_vec(), vec.iter().map(|_| 1.).collect::<Vec<f32>>());
        }

        #[test]
        fn add() {
            use crate::{tensor::Tensor, ops::ToVec, ops::Exp};
            let vec = vec![3., 1., 2., 4., 1., 0., 4., 3., 5.];
            let vec2 = vec![4., 2., 2., 4., 1., -2., 4., 3., 7.];
            // test Tensor
            let x = Tensor::from_vec(vec.clone(), &[9]);
            let y = Tensor::from_vec(vec2.clone(), &[9]);
            let z = x.clone() + y.clone();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            // test TensorGrad
            let x = Tensor::from_vec(vec.clone(), &[9]);
            let y = Tensor::from_vec(vec2.clone(), &[9]).with_grad();
            let z = x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().borrow().to_vec());

            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = Tensor::from_vec(vec2.clone(), &[9]);
            let z = &x + y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().borrow().to_vec());

            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = Tensor::from_vec(vec2.clone(), &[9]).with_grad();
            let z = &x + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| *x + *y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().borrow().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().borrow().to_vec());

            // test TensorFunc
            let x = Tensor::from_vec(vec.clone(), &[9]);
            let y = Tensor::from_vec(vec2.clone(), &[9]).with_grad();
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().borrow().to_vec());

            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = Tensor::from_vec(vec2.clone(), &[9]);
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().borrow().to_vec());

            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = Tensor::from_vec(vec2.clone(), &[9]).with_grad();
            let z = &x + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|_| 1.).collect::<Vec<f32>>(), &x.grad().borrow().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().borrow().to_vec());

            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = Tensor::from_vec(vec2.clone(), &[9]).with_grad();
            let z = x.exp() + &y;
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().borrow().to_vec());
            cmp_vec(&vec2.iter().map(|_| 1.).collect::<Vec<f32>>(), &y.grad().borrow().to_vec());

            let x = Tensor::from_vec(vec.clone(), &[9]).with_grad();
            let y = Tensor::from_vec(vec2.clone(), &[9]).with_grad();
            let z = x.exp() + y.exp();
            cmp_vec(&vec.iter().zip(vec2.iter()).map(|(x, y)| x.exp() + y.exp()).collect::<Vec<f32>>(), &z.to_vec());
            z.backward();
            cmp_vec(&vec.iter().map(|x| x.exp()).collect::<Vec<f32>>(), &x.grad().borrow().to_vec());
            cmp_vec(&vec2.iter().map(|y| y.exp()).collect::<Vec<f32>>(), &y.grad().borrow().to_vec());
        }

        #[test]
        fn sub() {
            use crate::prelude::*;
            use crate::tensor::Tensor;
            let x = Tensor::from_vec(vec![3, 5, 2], &[3]).with_grad();
            let z = &mut x.relu();
            //z.backward();
            println!("{}", z);
            //panic!();
        }

        #[test]
        fn div() {
        }

        #[test]
        fn mul() {
        }

        #[test]
        fn pow() {
        }

        #[test]
        fn matmul() {
            use crate::{tensor::Tensor, ops::ToVec, ops::MatMul};
            let x = Tensor::from([[2, 3, 4]]);
            let y = Tensor::from([[2, 3], [3, 4], [5, 3]]);
            let z = x.clone().matmul(y.clone());
            assert_eq!(z.data().to_vec(), [33, 30]);

            let x = x;
            let y = y.with_grad();
            let z = x.clone().matmul(&y);
            assert_eq!(z.data().to_vec(), [33, 30]);
            z.backward();
            assert_eq!(y.grad().borrow().to_vec(), [2, 2, 3, 3, 4, 4].to_vec());

            let x = x.with_grad();
            let y = Tensor::from([[2, 3], [3, 4], [5, 3]]);
            let z = x.matmul(y.clone());
            assert_eq!(z.data().to_vec(), [33, 30]);
            z.backward();
            assert_eq!(x.grad().borrow().to_vec(), [5, 7, 8].to_vec());

            let x = Tensor::from([[2, 3, 4]]).with_grad();
            let y = Tensor::from([[2, 3], [3, 4], [5, 3]]).with_grad();
            let z = x.matmul(&y);
            assert_eq!(z.data().to_vec(), [33, 30]);
            z.backward();
            assert_eq!(x.grad().borrow().to_vec(), [5, 7, 8].to_vec());
            assert_eq!(y.grad().borrow().to_vec(), [2, 2, 3, 3, 4, 4].to_vec());
        }

        #[test]
        fn conv() {
            //todo!()
        }
    }
}

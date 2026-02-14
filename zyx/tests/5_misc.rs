use half::f16;
use zyx::{DType, Scalar, Tensor, ZyxError};

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];

    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        )
    };

    c
}

#[test]
fn test_max_pool() -> Result<(), ZyxError> {
    // Create a 4x4 tensor
    let input = Tensor::from([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);

    // Perform max pooling with 2x2 kernel and stride 2x2
    let output = input.max_pool(
        [2, 2],           // kernel_size
        [2, 2],           // stride
        [1, 1],           // dilation
        [(0, 0), (0, 0)], // padding (no padding)
        false,            // ceil_mode
        false,            // return_indices
    )?;

    // Verify the output shape and values
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output, [[6.0, 8.0], [14.0, 16.0]]);

    Ok(())
}

#[test]
fn memory1() {
    let x = Tensor::from([[2, 3], [4, 5]]);
    //println!("{x}");
    assert_eq!(x, [[2, 3], [4, 5]]);
}

#[test]
fn memory2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    assert_eq!(x, [[2, 4, 3], [1, 5, 1]]);
    Ok(())
}

#[test]
fn complex_binary() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]).cast(DType::F32);
    let y = Tensor::from([[2, 4, 3], [1, 5, 7]]).cast(DType::F32);
    let z = x.sqrt() + y.exp2();
    Tensor::realize([&z])?;
    Ok(())
}

#[test]
fn tri1() -> Result<(), ZyxError> {
    let x = Tensor::tri(3, 5, 2, DType::I32);
    assert_eq!(x, [[0i32, 0, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]]);
    Ok(())
}

#[test]
fn fuse_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[2f32, 4., 3.], [1., 5., 1.]]);
    let z = x.exp2() + x;
    assert_eq!(z, [[6f32, 20., 11.], [3., 37., 3.]]);
    Ok(())
}

#[test]
fn fuse_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2f32, 4., 3.], [1., 5., 1.]]);
    let z = x.expand([2, 2, 3])? + x;
    assert_eq!(z, [[[4f32, 8., 6.], [2., 10., 2.]], [[4., 8., 6.], [2., 10., 2.]]]);
    Ok(())
}

#[test]
fn fuse_3() -> Result<(), ZyxError> {
    let x = Tensor::from([[2f32, 4., 3.], [1., 5., 1.]]);
    let z = x.sum([0])?.expand([2, 3])? + x;
    assert_eq!(z, [[5f32, 13., 7.], [4., 14., 5.]]);
    Ok(())
}

#[test]
fn fuse_4() -> Result<(), ZyxError> {
    let x = Tensor::from([[2f32, 4., 3.], [1., 5., 1.]]);
    let y = Tensor::from([[2f32, 4., 3.], [1., 5., 3.]]).exp2();
    let z1 = x + &y;
    let z2 = y.exp2();
    Tensor::realize([&z1, &z2])?;
    Ok(())
}

#[test]
fn fuse_5() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.t();
    let mut y = x.log2();
    x = x.exp2();
    x = x.reshape([2, 3])?;
    y = y.t();
    Tensor::realize([&x, &y])?;
    Ok(())
}

#[test]
fn fuse_6() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    let y = x.log2();
    let x = x.exp2();
    Tensor::realize([&x, &y])?;
    assert_eq!(x, [512f32, 128.]);
    assert_eq!(y, [3.16993f32, 2.807355]);
    Ok(())
}

#[test]
fn matmul_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let z = x.dot(y)?;
    assert_eq!(z, [[31, 15], [22, 10]]);
    Ok(())
}

#[test]
fn matmul_1() -> Result<(), ZyxError> {
    for m in (56..576).step_by(259) {
        for k in (12..890).step_by(231) {
            for n in (5..97).step_by(71) {
                let x_data: Vec<Vec<i32>> = (0..m).map(|i| (0..k).map(|j| i as i32 + j as i32).collect()).collect();
                let y_data: Vec<Vec<i32>> = (0..k).map(|i| (0..n).map(|j| i as i32 - j as i32).collect()).collect();

                let x = Tensor::from(x_data.clone());
                let y = Tensor::from(y_data.clone());

                let z = x.dot(y)?;

                // Reference matmul (CPU, naive)
                let mut expected = vec![vec![0i32; n]; m];
                for i in 0..m {
                    for kk in 0..k {
                        for j in 0..n {
                            expected[i][j] += x_data[i][kk] * y_data[kk][j];
                        }
                    }
                }

                if z != expected {
                    panic!();
                }
            }
        }
    }
    Ok(())
}

#[test]
fn batched_matmul() -> Result<(), ZyxError> {
    for b in (19..25).step_by(4) {
        for m in (32..67).step_by(31) {
            for k in (16..256).step_by(173) {
                for n in (8..128).step_by(59) {
                    // x: [B, M, K]
                    let x_data: Vec<Vec<Vec<i32>>> = (0..b)
                        .map(|bb| (0..m).map(|i| (0..k).map(|j| bb as i32 + i as i32 + j as i32).collect()).collect())
                        .collect();

                    // y: [B, K, N]
                    let y_data: Vec<Vec<Vec<i32>>> = (0..b)
                        .map(|bb| (0..k).map(|i| (0..n).map(|j| bb as i32 + i as i32 - j as i32).collect()).collect())
                        .collect();

                    let x = Tensor::from(x_data.clone());
                    let y = Tensor::from(y_data.clone());

                    let z = x.dot(y)?;

                    // Reference batched matmul (CPU, naive)
                    let mut expected = vec![vec![vec![0i32; n]; m]; b];
                    for bb in 0..b {
                        for i in 0..m {
                            for kk in 0..k {
                                for j in 0..n {
                                    expected[bb][i][j] += x_data[bb][i][kk] * y_data[bb][kk][j];
                                }
                            }
                        }
                    }

                    let expected_shape = vec![b, m, n];
                    assert_eq!(
                        z.shape(),
                        expected_shape,
                        "Shape mismatch: expected {:?}, got {:?}",
                        expected_shape,
                        z.shape()
                    );

                    // ---- Dtype check ----
                    assert_eq!(
                        z.dtype(),
                        DType::I32,
                        "Dtype mismatch: expected I32, got {:?}",
                        z.dtype()
                    );

                    if z != expected {
                        //println!("{z}");
                        panic!("Batched matmul mismatch for b={}, m={}, k={}, n={}", b, m, k, n);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn boolean_buffer() -> Result<(), ZyxError> {
    let x = Tensor::from([true, true, false, true]);
    assert_eq!(x, [true, true, false, true]);
    Ok(())
}

#[test]
fn mix_expand_reduce() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    println!("{:?}", x.shape());
    x = x.expand([2, 2])?;
    assert_eq!(x, [[9i32, 7], [9, 7]]);
    Ok(())
}

#[test]
fn mix_pad_reduce() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    x = x.pad_zeros([(0, 1)])?;
    assert_eq!(x, [9i32, 7, 0]);
    Ok(())
}

#[test]
fn mix_permute_pad() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.pad_zeros([(1, 0)])?.t();
    assert_eq!(x, [[0i32, 0], [2, 1], [4, 5], [3, 1]]);
    Ok(())
}

#[test]
fn mix_expand_reshape_reduce() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    let y = x.expand([2, 2])?;
    x = x.reshape([2, 1])?.expand([2, 2])?;
    Tensor::realize([&x, &y])?;
    println!("{y}");
    println!("{x}");
    assert_eq!(y, [[9i32, 7], [9, 7]]);
    assert_eq!(x, [[9i32, 9], [7, 7]]);
    Ok(())
}

#[test]
fn mix_pad_reshape_expand() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 4, 3, 3, 4], [1, 2, 1, 5, 1]]);
    x = x.pad_zeros([(1, 0), (2, 1)])?;
    x = x.reshape([2, 1, 3, 5])?;
    x = x.expand([2, 2, 3, 5])?;
    assert_eq!(
        x,
        [
            [
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 2, 4]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 2, 4]]
            ],
            [
                [[3, 3, 4, 0, 1], [2, 1, 5, 1, 0], [0, 0, 0, 0, 0]],
                [[3, 3, 4, 0, 1], [2, 1, 5, 1, 0], [0, 0, 0, 0, 0]]
            ]
        ]
    );
    Ok(())
}

#[test]
fn mix_reshape1() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[[[2i32], [4]], [[3], [1]], [[5], [1]]]]);
    //println!("x shape {:?}", x.shape());
    x = x.permute([0, 2, 1, 3])?;
    assert_eq!(x.shape(), [1, 2, 3, 1]);
    x = x.reshape([1, 2, 1, 3, 1]).unwrap();
    Tensor::realize([&x])?;
    assert_eq!(x.shape(), [1, 2, 1, 3, 1]);
    assert_eq!(x, [[[[[2i32], [3], [5]]], [[[4], [1], [1]]]]]);
    Ok(())
}

#[test]
fn pool() -> Result<(), ZyxError> {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3))?;
    //x = x.repeat([2, 2]);
    //println!("{x}");
    //x = x.reshape([12, 3]);
    //println!("{x}");
    x = x.pool([2, 2], 1, 1)?;
    assert_eq!(
        x,
        [
            [[[0, 1], [3, 4]], [[1, 2], [4, 5]]],
            [[[3, 4], [6, 7]], [[4, 5], [7, 8]]]
        ]
    );
    //println!("{x}");
    Ok(())
}

#[test]
fn cumsum() -> Result<(), ZyxError> {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3))?;
    x = x.cumsum(1)?;
    assert_eq!(x, [[0, 1, 3], [3, 7, 12], [6, 13, 21]]);
    Ok(())
}

#[test]
fn arange_3() -> Result<(), ZyxError> {
    let x = Tensor::arange(0, 10, 2)?;
    //println!("{x}");
    assert_eq!(x, [0, 2, 4, 6, 8]);
    Ok(())
}

/*#[test]
fn randn() {
    use zyx::DType;
    let x = Tensor::randn([10, 10], DType::F32).unwrap();
    //Tensor::plot_graph([], "graph0");
    //Tensor::realize([&x]).unwrap();
    println!("{x}");
    assert_eq!(x.isnan().sum(), 0);
}*/

#[test]
fn const_() -> Result<(), ZyxError> {
    let x = Tensor::from([[3f32, 4., 2.], [4., 3., 2.]]);
    //.get(1);
    let y = 1f32 + x;
    //println!("{y}'");
    //Tensor::plot_graph([], "graph0");
    //let c: Tensor = Tensor::constant(1f64 / std::f64::consts::E.log2());
    //y = y.log2() * c.cast(y.dtype());
    assert_eq!(y, [[4f32, 5., 3.], [5., 4., 3.]]);
    //y = y.ln();
    //println!("{y}'");
    Ok(())
}

#[test]
fn graph_shapes() -> Result<(), ZyxError> {
    let x: Tensor = 2.into();
    let y = x.expand([1, 1])?;
    println!("{y}");
    Ok(())
}

/*#[test]
fn uni_matmul() -> Result<(), ZyxError> {
    //use zyx::DType;
    //let x = Tensor::rand([5, 5], DType::F32) * 2f32 + 3f32;
    //let y = Tensor::rand([5, 5], DType::F32) * 3f32 + 4f32;
    //let x = Tensor::uniform([5, 5], -1f32..2f32)?;
    //let y = Tensor::uniform([5, 5], -1f32..5f32)?;
    //let z = x.dot(y)?;
    //println!("{z}");
    Ok(())
}*/

#[test]
fn cat() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let b = Tensor::from([[5, 6], [7, 8]]);
    let c = Tensor::cat([&a, &b], 0)?;
    assert_eq!(c, [[1, 2], [3, 4], [5, 6], [7, 8]]);
    let c = Tensor::cat([&a, &b], 1)?;
    assert_eq!(c, [[1, 2, 5, 6], [3, 4, 7, 8]]);
    Ok(())
}

#[test]
fn pad_zeros() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 3], [4, 5]]);
    //let x = x.pad_zeros([(0, 1)]);
    let x = x.pad_zeros([(4, 3), (1, 2)])?;
    //Tensor::plot_dot_graph([], "graph0");
    assert_eq!(
        x,
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 3, 0, 0, 0],
            [0, 0, 0, 0, 4, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    );
    Ok(())
}

#[test]
fn one_hot() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 4]);
    let y = x.one_hot(4);
    assert_eq!(y, [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]);
    Ok(())
}

#[test]
fn ones() {
    let x = Tensor::ones([2, 3], DType::I32);
    assert_eq!(x, [[1i32, 1, 1], [1, 1, 1]]);
}

#[test]
fn graph_node_reuse() {
    let x = Tensor::from([4, 2, 3]);
    let y = Tensor::from([4, 2, 3]);
    let a = x + y;
    assert_eq!(a, [8, 4, 6]);
    drop(a);
    let x = Tensor::from([4, 2, 3]);
    let y = Tensor::from([4, 2, 3]);
    let b = x + y;
    assert_eq!(b, [8, 4, 6]);
}

#[test]
fn get1() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    assert_eq!(x.slice((.., 2..3)).unwrap(), [[1], [4]]);
}

#[test]
fn split1() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    let tensors = x.split([2, 1], 1).unwrap();
    //Tensor::realize(&tensors).unwrap();
    assert_eq!(tensors[0], [[2, 3], [2, 1]]);
    assert_eq!(tensors[1], [[1], [4]]);
    //for t in tensors { println!("{t}"); }
}

#[test]
fn split2() -> Result<(), ZyxError> {
    let a = Tensor::arange(0, 10, 1)?.reshape([5, 2])?;
    let x = a.split([2, 2, 1], 0)?;
    assert_eq!(x[0], [[0, 1], [2, 3]]);
    assert_eq!(x[1], [[4, 5], [6, 7]]);
    assert_eq!(x[2], [[8, 9]]);
    let x = a.split([1, 4], 0)?;
    assert_eq!(x[0], [[0, 1]]);
    assert_eq!(x[1], [[2, 3], [4, 5], [6, 7], [8, 9]]);
    Ok(())
}

#[test]
fn matmul_disk() -> Result<(), ZyxError> {
    //let mut xy: Vec<Tensor> = Tensor::load("xy.safetensors").unwrap();
    //let y = xy.pop().unwrap();
    //let x = xy.pop().unwrap();
    let mut xyz: std::collections::HashMap<String, Tensor> = Tensor::load("./tests/xyz2.safetensors")?;
    let z = xyz.remove("z").unwrap();
    let y = xyz.remove("y").unwrap();
    let x = xyz.remove("x").unwrap();
    println!("{:?}", x.shape());
    println!("{:?}", y.shape());
    let dataz: Vec<i64> = z.try_into()?;
    let zz = x.matmul(y)?;
    let datazz: Vec<i64> = zz.try_into()?;
    for (i, (x, y)) in dataz.iter().zip(datazz).enumerate() {
        //println!("{x}, {y}");
        assert!(x.is_equal(y), "{x} != {y} at index {i}");
    }
    //println!("{z}");
    Ok(())
}

/*#[test]
fn save() -> Result<(), ZyxError> {
    //use zyx::TensorSave;
    //let x = Tensor::from([2f32, 4., 3.]);
    //[&x].save("../x.safetensors")?;
    //let x: HashMap<String, Tensor> = Tensor::load("../x.safetensors")?;
    //let x: Vec<i64> = x["x"].clone().try_into()?;
    //println!("{:?}", x);
    Ok(())
}*/

#[test]
fn softmax_1() -> Result<(), ZyxError> {
    let x = Tensor::from([2f32, 4., 3.]);
    //let y = x.softmax([]);
    //println!("{y:?}");
    //let y = x.sum_kd([])?;
    //println!("{y}");
    //let e = (&x - y).exp();
    //println!("{e}");
    //let y = &e / e.sum_kd([])?;
    //println!("{e:?}");
    //panic!();
    //Tensor::plot_graph([], "graph");
    //println!("{y:.20}");
    //assert_eq!(y, [0.09003056585788726807, 0.66524088382720947266, 0.24472846090793609619]);
    let y = x.softmax([])?;
    //println!("{y}");
    assert_eq!(
        y,
        [
            0.09003056585788726807f32,
            0.66524088382720947266,
            0.24472846090793609619,
        ]
    );
    //Tensor::plot_graph([], "graph").unwrap();
    Ok(())
}

/*#[test]
fn var1() -> Result<(), ZyxError> {
    let x = Tensor::randn(1024, DType::F32)?;
    let y = x.var([-1], 1)?;
    Tensor::realize([&y])?;
    Ok(())
}*/

#[test]
fn dot_pad() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    let y = Tensor::from([[2, 3], [1, 2], [4, 1]]);
    x = x.dot(y)?.pad_zeros([(2, 1)])?;
    assert_eq!(x, [[0, 0, 11, 13, 0], [0, 0, 12, 15, 0]]);
    Ok(())
}

#[test]
#[should_panic]
fn t3() {
    let x = Tensor::randn([1024, 1024], DType::F32).unwrap().expand([1024, 1024, 1024, 1024, 1024, 1024]).unwrap();
    Tensor::realize([&x]).unwrap();
}

#[test]
fn layer_norm() -> Result<(), ZyxError> {
    let weight = Some(Tensor::from([4f32, 5., 1., 2.]));
    let d_dims = weight.as_ref().unwrap().rank();
    let bias: Option<Tensor> = None;
    let eps = 0.00001f32;

    let x = Tensor::from([[3, 5, 2, 1], [6, 1, 4, 2]]).cast(DType::F32);

    let axes = -(d_dims as i32)..=-1;
    let eps = Tensor::from(eps).cast(x.dtype());
    let a = &x - x.mean_keepdim(axes.clone())?;
    //println!("{a}");
    let b = (x.var_keepdim(axes)? + eps).sqrt();
    let mut x = a / b;
    if let Some(w) = &weight {
        x = x * w;
    }
    if let Some(b) = &bias {
        x = x + b;
    }
    assert_eq!(
        x,
        [
            [0.585539f32, 6.587314, -0.439154, -2.049387],
            [4.960858, -5.073606, 0.338240, -1.127468]
        ]
    );
    Ok(())
}

#[test]
fn multiple_stores() -> Result<(), ZyxError> {
    let x = Tensor::from([[3f32, 4., 2.], [5., 4., 1.]]);
    let y = x.ln();
    let z = y.tanh();
    Tensor::realize([&y, &z])?;
    println!("{z:.14}");
    assert_eq!(
        z,
        [
            [0.8000000119f32, 0.8823529482, 0.6000000238],
            [0.9230769277, 0.8823529482, 0.0000000000]
        ]
    );
    Ok(())
}

#[test]
fn dot2() -> Result<(), ZyxError> {
    let n = 512;
    let mut x = Tensor::randn([n, n], DType::F32)?;
    let y = Tensor::randn([n, n], DType::F32)?;
    for _ in 0..5 {
        x = x.dot(&y)?;
        Tensor::realize([&x])?;
    }
    Tensor::realize([&x])?;
    //println!("{x}");
    Ok(())
}

#[test]
fn repeat1() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    x = x.repeat([2, 3, 1])?;
    println!("{x}");
    assert_eq!(
        x,
        [
            [[2, 3, 1], [2, 4, 1], [2, 3, 1], [2, 4, 1], [2, 3, 1], [2, 4, 1]],
            [[2, 3, 1], [2, 4, 1], [2, 3, 1], [2, 4, 1], [2, 3, 1], [2, 4, 1]]
        ]
    );
    Ok(())
}

#[test]
fn mix_2() {
    let x = Tensor::from([[2f32, 3.], [4., 5.]]);
    let y = x.t();
    let z = x.exp().cast(DType::I32);
    Tensor::realize([&y, &z]).unwrap();
    assert_eq!(z, [[7i32, 20], [54, 148]]);
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn rand_get() -> Result<(), ZyxError> {
    Tensor::manual_seed(69420);
    let x = Tensor::rand([3, 12], DType::U8)?;
    let x = x.slice((.., 8..=-2))?;
    assert_eq!(x, [[41u8, 171, 236], [212, 222, 77], [16, 125, 60]]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn gather_test() -> Result<(), ZyxError> {
    // Hardcoded 3x5 tensor
    let x = Tensor::from([[10u16, 20, 30, 40, 50], [11, 21, 31, 41, 51], [12, 22, 32, 42, 52]]);

    // Indices to gather along axis 1
    let indices = Tensor::from([
        [0u16, 2, 4], // from row 0 take columns 0,2,4
        [1, 3, 0],    // from row 1 take columns 1,3,0
        [4, 1, 2],    // from row 2 take columns 4,1,2
    ]);

    // Perform gather along axis 1
    let gathered = x.gather(1, &indices)?;

    // Expected output
    let expected = [[10u16, 30, 50], [21, 41, 11], [52, 22, 32]];

    assert_eq!(gathered, expected);

    Ok(())
}

#[test]
fn eye1() {
    let x = Tensor::eye(8, DType::I32);
    assert_eq!(
        x,
        [
            [1i32, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    );
}

#[test]
fn bench_mm1() -> Result<(), ZyxError> {
    const N: usize = 1024;
    let dtype = zyx::DType::F16;
    let x = Tensor::rand([N, N], dtype)?;
    let y = Tensor::rand([N, N], dtype)?;

    let x_data: Vec<f16> = x.clone().try_into()?;
    let y_data: Vec<f16> = y.clone().try_into()?;

    let z = x.matmul(&y)?;

    let z_data: Vec<f16> = z.try_into()?;
    /*let expected = matmul(&x_data, &y_data, N, N, N);
    for (x, y) in z_data.into_iter().zip(expected) {
        if !x.is_equal(y) {
            panic!("Wrong matmul");
        }
    }*/

    Ok(())
}

#[test]
fn double_vec() -> Result<(), ZyxError> {
    let x = Tensor::from(vec![vec![4, 1, 2], vec![4, 6, 2]]);
    assert_eq!(x.shape(), [2, 3]);
    Ok(())
}

#[test]
fn binary_y_depends_on_x() -> Result<(), ZyxError> {
    let z = {
        let x = Tensor::from([[2, 4, 1], [3, 2, 4]]).cast(DType::F32);

        let x = x
            .exp2()
            .log2()
            .exp2()
            .log2()
            .exp2()
            .log2()
            .exp2()
            .log2()
            .exp2()
            .log2()
            .exp2()
            .log2()
            .exp2()
            .log2()
            .exp2()
            .log2();

        let y = x.permute([1, 0]).unwrap();

        let z = x.reshape(6).unwrap() + y.reshape(6).unwrap() + x.reshape(6).unwrap();
        z.exp2().log2()
    };
    assert_eq!(z, [6f32, 11., 6., 8., 5., 12.]);
    //Tensor::plot_graph([], "graph").unwrap();
    //println!("{z}");
    Ok(())
}

#[test]
fn dot5() {
    let x = Tensor::from([[2, 3, 1], [3, 4, 1]]);
    let y = Tensor::from([[2, 3], [2, 1], [4, 1]]);
    let x = x.dot(y).unwrap();
    //let x = x.reshape([2, 1, 3]) * y.t().reshape([1, 2, 3]);
    //let x = x.sum(2);
    assert_eq!(x, [[14, 10], [18, 14]]);
}

/*#[test]
fn t1() {
    use crate::DType;
    let x = Tensor::from([0f32, 5., 1.]);
    let y = Tensor::rand([3, 5], DType::F32);
    let a = x.dot(y);
    let x = Tensor::from([0f32, 5., 1.]);
    let y = Tensor::rand([3, 5], DType::F32);
    let b = x.dot(y);
    println!("{a}, {b}");
}*/

#[test]
fn conv1() -> Result<(), ZyxError> {
    let t = Tensor::arange(0f32, 9., 1.)?.reshape([1, 1, 3, 3])?;
    let w = Tensor::ones([1, 1, 2, 2], DType::F32);
    let x = t.conv(&w, None, 1, 1, 1, 0)?;

    println!("{x}");

    Ok(())
}

#[test]
fn graph_tensor_ordering() -> Result<(), ZyxError> {
    let z2 = {
        let x = Tensor::from([3f32, 4., 2.]); // 0
        let z1 = x.exp2() + x.log2(); // 3
        z1.exp2() // 4
    };
    println!("{z2}");
    let z3 = {
        z2.exp2() * z2 // 6
    };
    println!("{z3}");

    Ok(())
}

#[test]
fn rope_3() -> Result<(), ZyxError> {
    let z = {
        let xs = Tensor::from([[1f32, 4., 2., 4., 4., 3.], [4., 2., 4., 4., 3., 4.]]).reshape([1, 1, 2, 6])?;
        let sin = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
        let cos = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
        let [d] = xs.rdims()?;
        let sin_freqs = sin.squeeze([0, 1]);
        let cos_freqs = cos.squeeze([0, 1]);
        let a = xs.slice((.., .., .., ..d / 2)).unwrap();
        //assert_eq!(a, [[[[1f32, 4., 2.], [4., 2., 4.]]]]);
        let b = -xs.slice((.., .., .., d / 2..)).unwrap();
        //assert_eq!(b, [[[[-4f32, -4., -3.], [-4., -3., -4.]]]]);
        let ro = a.clone() * cos_freqs.clone() - b.clone() * sin_freqs.clone();
        assert_eq!(ro, [[[[5f32, 32., 10.], [32., 20., 24.]]]]);
        let co = a * sin_freqs + b * cos_freqs;
        //assert_eq!(co, [[[[-3f32, 0., -2.], [0., -4., 0.]]]]);
        Tensor::cat([&co, &ro], -1).unwrap()
    };
    assert_eq!(
        z.cast(DType::I32),
        [[[[-3i32, 0, -2, 5, 32, 10], [0, -4, 0, 32, 20, 24]]]]
    );
    Ok(())
}

#[test]
fn rope_4() -> Result<(), ZyxError> {
    let xs = Tensor::from([1f32, 4., 2., 4., 4., 3., 4., 2., 4., 4., 3., 4.]).reshape([1, 1, 2, 6])?;
    let sin = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
    let cos = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
    let z = xs.rope(&cos, &sin)?.cast(DType::I32);
    assert_eq!(z, [[[[-3i32, 0, -2, 5, 32, 10], [0, -4, 0, 32, 20, 24]]]]);
    Ok(())
}

#[test]
fn complex_movement_reduce() -> Result<(), ZyxError> {
    let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]]).expand([2, 3, 2])?.exp().ln().reshape([2, 3, 2, 1])?;
    let y = Tensor::from([[2f32, 3., 1.], [4., 3., 2.]]).reshape([2, 3, 1, 1])?.expand([2, 3, 2, 1])?;
    let z = (&x + &y).expand([2, 3, 2, 2])?.sum([3, 0])?;
    let z = z.exp().ln().permute([1, 0])?.sum([0])?;
    assert_eq!(z, [52f32, 52., 40.]);
    Ok(())
}

#[test]
fn mean1() -> Result<(), ZyxError> {
    let x = Tensor::from([[1i32, 2, 3], [4, 5, 6]]);
    let mean = x.sum([1])? * 0.3333333333333f32;
    //assert_eq!(mean, [2f32, 5.]);
    let y = x - mean.reshape([2, 1])?;
    //panic!("{y}");
    assert_eq!(y, [[-1f32, 0., 1.], [-1., 0., 1.]]);
    Ok(())
}

#[test]
fn var1() -> Result<(), ZyxError> {
    let x = Tensor::from([[1f32, 2., 3.], [4., 5., 6.]]);
    let [n] = x.dims()?;
    let mean = x.mean_keepdim([0])?;
    let x = x - mean;
    let squared = &x * &x;
    let summed = squared.sum([0])?;
    let y = summed / n as u32;
    assert_eq!(y, [2.25f32, 2.25, 2.25]);
    Ok(())
}

#[test]
fn mean2() -> Result<(), ZyxError> {
    let x = Tensor::from([[1i32, 2, 3], [4, 5, 6]]);
    let mean = x.mean_keepdim([1])?;
    let y = x - mean;
    assert_eq!(y, [[-1i32, 0, 1], [-1, 0, 1]]);
    Ok(())
}

#[test]
fn var2() -> Result<(), ZyxError> {
    let x = Tensor::from([[1f32, 2., 3.], [4., 5., 6.]]);
    let [_, n] = x.dims()?;
    let mean = x.mean_keepdim([1])?;
    let x = x - mean;
    let squared = &x * &x;
    let summed = squared.sum([1])?;
    let y = summed / n as u32;
    assert_eq!(y, [0.666666f32, 0.666666]);
    Ok(())
}

#[test]
fn var3() -> Result<(), ZyxError> {
    let x = Tensor::from([[1f32, 2., 3.], [4., 5., 6.]]);
    let y = x.var_correction([1], 0)?;
    assert_eq!(y, [0.666666f32, 0.666666]);
    Ok(())
}

#[test]
fn var4() -> Result<(), ZyxError> {
    let x = Tensor::from([[1f32, 2., 3.], [4., 5., 6.]]);
    let y = x.var_correction([0], 0)?;
    assert_eq!(y, [2.25f32, 2.25, 2.25]);
    let y = x.var_correction([1], 0)?;
    assert_eq!(y, [0.666666f32, 0.666666]);
    Ok(())
}

#[test]
fn softmax_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2f32, 4., 3.], [4., 2., 3.]]);
    let y = x.softmax([])?;
    assert_eq!(
        y,
        [
            [0.0450152867f32, 0.3326204717, 0.1223642379],
            [0.3326204717, 0.0450152867, 0.1223642379]
        ]
    );
    let y = x.softmax([0])?;
    assert_eq!(
        y,
        [[0.1192029193f32, 0.8807970285, 0.5], [0.8807970285, 0.1192029193, 0.5]]
    );
    let y = x.softmax([1])?;
    assert_eq!(
        y,
        [
            [0.0900305659f32, 0.6652408838, 0.2447284609],
            [0.6652408838, 0.0900305659, 0.2447284609]
        ]
    );
    Ok(())
}

#[test]
fn complex_causal_self_attention() -> Result<(), ZyxError> {
    let y = {
        let dtype = DType::F32;
        let n_embd = 4;
        let n_head = 4;
        let c_attn_weight = Tensor::from([
            [3, 1, 2, 3, 1, 2, 5, 4, 2, 3, 1, 3],
            [1, 1, 2, 3, 1, 2, 5, 4, 2, 3, 1, 3],
            [3, 1, 5, 3, 1, 2, 5, 4, 2, 3, 1, 3],
            [3, 1, 2, 3, 1, 2, 5, 8, 2, 3, 1, 3],
        ])
        .t()
        .cast(dtype);
        //let c_proj_weight = Tensor::from([[5, 4, 2, 1], [9, 1, 5, 2], [7, 5, 6, 2], [6, 2, 7, 1]]).cast(dtype);

        let x = Tensor::from([[[1, 0, 4, 2], [2, 5, 0, 1], [0, 8, 1, 0], [5, 1, 0, 0]]]).cast(dtype);

        let [b, t, c] = x.shape()[..] else {
            return Err(ZyxError::ShapeError("x must have exactly 3 dims, b, t, c".into()));
        };
        let mut splits = x.dot(c_attn_weight.t())?.split([n_embd, n_embd, n_embd], 2)?;
        let mut v = splits.pop().unwrap();
        let mut k = splits.pop().unwrap();
        let mut q = splits.pop().unwrap();

        k = k.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
        q = q.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
        v = v.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;

        let mut att = q.dot(k.t())? * (1f32 / (*k.shape().last().unwrap() as f32).sqrt());

        /*assert_eq!(
            att,
            [[
                [
                    [147f32, 168., 189., 126.],
                    [98., 112., 126., 84.],
                    [77., 88., 99., 66.],
                    [112., 128., 144., 96.]
                ],
                [
                    [98., 112., 126., 84.],
                    [112., 128., 144., 96.],
                    [126., 144., 162., 108.],
                    [84., 96., 108., 72.]
                ],
                [
                    [910., 1040., 1170., 780.],
                    [560., 640., 720., 480.],
                    [735., 840., 945., 630.],
                    [420., 480., 540., 360.]
                ],
                [
                    [756., 756., 756., 504.],
                    [864., 864., 864., 576.],
                    [972., 972., 972., 648.],
                    [648., 648., 648., 432.]
                ]
            ]]
        );*/

        att = att.softmax([-1])?;
        let mut y = att.dot(v)?;

        /*assert_eq!(
            y,
            [[
                [[18f32], [18.], [18.], [18.]],
                [[27.], [27.], [27.], [27.]],
                [[9.], [9.], [9.], [9.]],
                [[24.], [24.], [24.], [24.]]
            ]]
        );*/

        y = y.transpose(1, 2)?.reshape([b, t, c])?;
        y
    };
    //y = y.dot(c_proj_weight.t())?;

    assert_eq!(
        y,
        [[
            [18f32, 27., 9., 24.],
            [18., 27., 9., 24.],
            [18., 27., 9., 24.],
            [18., 27., 9., 24.]
        ]]
    );

    Ok(())
}

#[test]
fn dot6() -> Result<(), ZyxError> {
    let mut x = Tensor::from([2i32, 3, 1]);
    let w = Tensor::from([[2i32, 3, 2], [2, 1, 1], [4, 1, 4]]);
    for _ in 0..10 {
        x = x.matmul(&w)?;
    }
    assert_eq!(x, [492004322i32, 323660910, 445342573]);
    Ok(())
}

#[test]
fn dot4() -> Result<(), ZyxError> {
    let mut x = Tensor::from([2i32, 3, 1]);
    let w = Tensor::from([[2i32, 3, 2], [2, 1, 1], [4, 1, 4]]);
    let b = Tensor::from([2i32, 3, 5]);
    for _ in 0..10 {
        x = x.dot(&w)? + &b;
        //Tensor::realize([&x]).unwrap();
    }
    assert_eq!(x, [671627020i32, 441824135, 607929878]);
    Ok(())
}

#[test]
fn cross_entropy() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 3, 4], [5, 6, 7]]).cast(DType::F32);
    let target = Tensor::from([[0, 1, 0], [0, 0, 1]]).cast(DType::F32);
    let m = &x - x.max_keepdim([1])?;
    //println!("{}", m);
    //Tensor::realize([&m])?;
    let neg_log2_softmax = m.exp().sum_keepdim([1])?.ln() - m;
    //println!("{}", neg_log2_softmax);
    //panic!();
    let ce = neg_log2_softmax * target;
    //println!("{ce:.6}");
    assert_eq!(ce, [[0.000000f32, 1.407606, 0.000000], [0.000000, 0.000000, 0.407606]]);
    Ok(())
}

#[test]
fn test_padding_on_elementwise_kernel() {
    let t = Tensor::from([2, 3, 4]);
    let padded = t.pad([(1, 1)], 0).unwrap();
    let result = padded + 1;
    assert_eq!(result.shape(), &[5]);
    assert_eq!(result.slice(1).unwrap(), 3);
}

#[test]
fn test_expand_on_elementwise_kernel() {
    let t = Tensor::from([2, 3, 4]);
    let expanded = t.expand([3, 3]).unwrap();
    let result = expanded + 1.0;
    assert_eq!(result.shape(), &[3, 3]);
    assert_eq!(result.slice((1, 1)).unwrap(), 4.0);
}

#[test]
fn test_reshape_on_elementwise_kernel() {
    let t = Tensor::from([2, 3, 4]);
    let reshaped = t.reshape([3, 1]).unwrap();
    let result = reshaped * 2.0;
    assert_eq!(result.shape(), &[3, 1]);
    assert_eq!(result.slice((2, 0)).unwrap(), 8.0);
}

#[test]
fn test_permute_on_elementwise_kernel() {
    let t = Tensor::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
    let permuted = t.permute([2, 0, 1]).unwrap();
    let result = permuted + 1.0;
    assert_eq!(result.shape(), &[2, 2, 2]);
    let value: f64 = result.slice((1, 0, 1)).unwrap().item();
    assert_eq!(value, 5.0);
}

#[test]
fn test_padding_on_reduce_kernel() {
    let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    let padded = t.pad([(1, 1), (0, 0)], 0.0).unwrap();
    let reduced = padded.sum([0]).unwrap();
    assert_eq!(reduced.shape(), &[4]);
    assert_eq!(reduced.slice(0).unwrap(), 0.0);
    assert_eq!(reduced.slice(1).unwrap(), 4.0);
    assert_eq!(reduced.slice(2).unwrap(), 6.0);
    assert_eq!(reduced.slice(3).unwrap(), 0.0);
}

#[test]
fn test_expand_on_reduce_kernel() {
    let t = Tensor::from([[1.0], [2.0], [3.0]]);
    let expanded = t.expand([3, 2]).unwrap();
    let reduced = expanded.mean([1]).unwrap();
    assert_eq!(reduced.shape(), &[3]);
    assert_eq!(reduced.slice(1).unwrap(), 2.0);
}

#[test]
fn test_reshape_on_reduce_kernel() {
    let t = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
    let reshaped = t.reshape([4]).unwrap();
    let reduced = reshaped.sum([0]).unwrap();
    assert_eq!(reduced.shape(), &[1]);
    assert_eq!(reduced.item::<f64>(), 10.0);
}

#[test]
fn test_permute_on_reduce_kernel() {
    let t = Tensor::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
    let permuted = t.permute([1, 2, 0]).unwrap();
    let reduced = permuted.sum([2]).unwrap();
    assert_eq!(reduced.shape(), &[2, 2]);
    assert_eq!(reduced.slice((0, 0)).unwrap(), 6.0);
}

#[test]
fn arange_1() -> Result<(), ZyxError> {
    let x = Tensor::arange(0, 784 * 7, 1)?.cast(DType::F32).exp2().sin();
    //x = x.sum(0)
    Tensor::realize([&x]).unwrap();
    Ok(())
}

#[test]
fn arange_2() {
    let x = Tensor::arange(0, 2, 1).unwrap().exp2().sin();
    //x = x.sum(0)
    Tensor::realize([&x]).unwrap();
}

#[test]
fn rope_2() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5, 6, 7, 8]).reshape([1, 2, 4])?.cast(zyx::DType::F32);
    let base = 10000f32;

    let [_batch_size, seq_len, embed_dim] = x.dims()?;

    assert_eq!(embed_dim % 2, 0, "Embedding dimension should be even for RoPE.");

    // Generate the position indices
    let position = Tensor::arange(0., seq_len as f32, 1.)?.unsqueeze(1)?; // Shape: (seq_len, 1)

    // Create a tensor of frequencies for each dimension
    let mut freqs = Tensor::arange(0., embed_dim as f32 / 2., 1.)?; // Shape: (embed_dim // 2)
    freqs = Tensor::from(base).pow(freqs * (2.0 / embed_dim as f32))?; // Apply scaling for frequency
    //println!("freqs={freqs}");

    // Create the positional encoding matrix (sinusoidal)
    let pos_enc = position * freqs; // Shape: (seq_len, embed_dim // 2)
    //println!("{pos_enc}");

    // Apply sin and cos to each dimension
    let sin_enc = pos_enc.sin(); // Shape: (seq_len, embed_dim // 2)
    let cos_enc = pos_enc.cos(); // Shape: (seq_len, embed_dim // 2)
    //Tensor::realize([&sin_enc, &cos_enc])?;
    //println!("{sin_enc}\n{cos_enc}");

    //drop(pos_enc);
    //Tensor::realize([&sin_enc, &cos_enc])?;
    //println!("{sin_enc}\n{cos_enc}");
    //panic!();

    // Combine sin and cos to create the final embedding
    // The idea is to apply sin/cos to even and odd dimensions
    let x = x.rope(sin_enc, cos_enc)?;
    //drop(pos_enc);

    assert_eq!(
        x.squeeze(0..),
        [
            [-3.0f32, -4.0, 1.0, 2.0],
            [0.42523819, -9.93675995, 8.591808319, 1.12284958]
        ]
    );

    Ok(())
}

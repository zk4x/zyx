use std::collections::HashMap;

use zyx::{DType, Scalar, Tensor, ZyxError};

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
fn matmul_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let z = x.dot(y)?;
    assert_eq!(z, [[31, 15], [22, 10]]);
    Ok(())
}

#[test]
fn boolean_buffer() -> Result<(), ZyxError> {
    let x = Tensor::from([true, true, false, true]);
    assert_eq!(x, [true, true, false, true]);
    Ok(())
}

#[test]
fn sum1() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    assert_eq!(x.sum([0])?, [3, 9, 4]);
    assert_eq!(x.sum([1])?, [9, 7]);
    assert_eq!(x.sum([])?, 16);
    Ok(())
}

#[test]
fn sum2() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 4]);
    assert_eq!(x.sum([])?, 6);
    Ok(())
}

#[test]
fn sum3() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    x = x.sum([])?;
    debug_assert_eq!(x, [13i32]);
    Ok(())
}

#[test]
fn pad_reduce() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    x = x.pad_zeros([(0, 1)])?;
    assert_eq!(x, [9i32, 7, 0]);
    Ok(())
}

#[test]
fn permute_pad() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2i32, 4, 3], [1, 5, 1]]);
    x = x.pad_zeros([(1, 0)])?.t();
    assert_eq!(x, [[0i32, 0], [2, 1], [4, 5], [3, 1]]);
    Ok(())
}

#[test]
fn expand_reduce() -> Result<(), ZyxError> {
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
fn rope1() -> Result<(), ZyxError> {
    let xs =
        Tensor::from([1f32, 4., 2., 4., 4., 3., 4., 2., 4., 4., 3., 4.]).reshape([1, 1, 2, 6])?;
    let sin = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
    let cos = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
    let z = xs.rope(&cos, &sin)?;
    assert_eq!(
        z,
        [[[[-3f32, 0., -2., 5., 32., 10.], [0., -4., 0., 32., 20., 24.]]]]
    );
    Ok(())
}

#[test]
fn rope2() -> Result<(), ZyxError> {
    let z = {
        let xs = Tensor::from([[1f32, 4., 2., 4., 4., 3.], [4., 2., 4., 4., 3., 4.]])
            .reshape([1, 1, 2, 6])?;
        let sin = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
        let cos = Tensor::from([1f32, 4., 2., 4., 4., 3.]).reshape([2, 3])?;
        let sh = xs.shape();
        let sin_freqs = sin.squeeze(1).unwrap().squeeze(0).unwrap();
        let cos_freqs = cos.squeeze(1).unwrap().squeeze(0).unwrap();
        let d = isize::try_from(*sh.last().unwrap()).unwrap();
        let a = xs.get((.., .., .., ..d / 2)).unwrap();
        //assert_eq!(a, [[[[1f32, 4., 2.], [4., 2., 4.]]]]);
        let b = -xs.get((.., .., .., d / 2..)).unwrap();
        //assert_eq!(b, [[[[-4f32, -4., -3.], [-4., -3., -4.]]]]);
        let ro = a.clone() * cos_freqs.clone() - b.clone() * sin_freqs.clone();
        //assert_eq!(ro, [[[[5f32, 32., 10.], [32., 20., 24.]]]]);
        let co = a * sin_freqs + b * cos_freqs;
        //assert_eq!(co, [[[[-3f32, 0., -2.], [0., -4., 0.]]]]);
        Tensor::cat([&co, &ro], -1).unwrap()
    };
    assert_eq!(
        z,
        [[[[-3f32, 0., -2., 5., 32., 10.], [0., -4., 0., 32., 20., 24.]]]]
    );
    Ok(())
}

#[test]
fn pad_reshape_expand() -> Result<(), ZyxError> {
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
fn reshape1() -> Result<(), ZyxError> {
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
fn arange() -> Result<(), ZyxError> {
    let x = Tensor::arange(0, 10, 2)?;
    //println!("{x}");
    assert_eq!(x, [0, 2, 4, 6, 8]);
    Ok(())
}

/*#[test]
fn rand() {
    use zyx::DType;
    let x = Tensor::randn([10, 10], DType::F32).unwrap();
    //Tensor::plot_graph([], "graph0");
    //Tensor::realize([&x]).unwrap();
    println!("{x}");
}*/

#[test]
fn const_() -> Result<(), ZyxError> {
    let x = Tensor::from([[3f32, 4., 2.], [4., 3., 2.]]);
    //.get(1);
    let y = Tensor::constant(1) + x;
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
    let x = Tensor::constant(2);
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
fn pad1() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let c = a.pad_zeros([(0, 0), (0, 2)])?;
    assert_eq!(c, [[1, 2], [3, 4], [0, 0], [0, 0]]);
    Ok(())
}

#[test]
fn pad2() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]).reshape([1, 1, 2, 2])?;
    let b = Tensor::from([[5, 6], [7, 8]]).reshape([1, 1, 1, 4])?;
    let c = a.pad_zeros([(0, 2), (0, 2)])? + b;
    assert_eq!(
        c,
        [[[[6, 8, 7, 8], [8, 10, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]]]
    );
    Ok(())
}

#[test]
fn pad3() -> Result<(), ZyxError> {
    let x = Tensor::from([1, 2, 3, 4, 5, 6, 7, 8]).reshape([2, 4])?;
    let a = x.pad_zeros([(-2, 0)])?;
    let b = -x.pad_zeros([(0, -2)])?;
    let sin = Tensor::from([[2, 3], [3, 1]]);
    let cos = Tensor::from([[2, 3], [3, 1]]);
    let z = &a * &sin - &b * &cos;
    let z2 = a * sin + b * cos;
    let z = z.pad_zeros([(0, 2)])? + z2.pad_zeros([(2, 0)])?;
    assert_eq!(z, [[8, 18, 4, 6], [36, 14, 6, 2]]);
    Ok(())
}

#[test]
fn expand1() -> Result<(), ZyxError> {
    let a = Tensor::from([[1, 2], [3, 4]]).reshape([1, 1, 1, 4])?;
    let b = Tensor::from([[5, 6], [7, 8]]).reshape([1, 1, 4, 1])?;
    let c = a + b;
    assert_eq!(
        c,
        [[[[6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12]]]]
    );
    Ok(())
}

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
fn matmul_1024() -> Result<(), ZyxError> {
    //let mut xy: Vec<Tensor> = Tensor::load("xy.safetensors").unwrap();
    //let y = xy.pop().unwrap();
    //let x = xy.pop().unwrap();
    let mut xyz: HashMap<String, Tensor> = Tensor::load("../xyz2.safetensors")?;
    let z = xyz.remove("z").unwrap();
    let y = xyz.remove("y").unwrap();
    let x = xyz.remove("x").unwrap();
    //println!("{:?}", x.shape());
    //println!("{:?}", y.shape());
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

#[test]
fn save() -> Result<(), ZyxError> {
    //use zyx::TensorSave;
    //let x = Tensor::from([2f32, 4., 3.]);
    //[&x].save("../x.safetensors")?;
    //let x: HashMap<String, Tensor> = Tensor::load("../x.safetensors")?;
    //let x: Vec<i64> = x["x"].clone().try_into()?;
    //println!("{:?}", x);
    Ok(())
}

#[test]
fn softmax1() -> Result<(), ZyxError> {
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
    let y_data: Vec<f32> = y.try_into()?;
    for (x, y) in y_data.into_iter().zip([
        0.09003056585788726807,
        0.66524088382720947266,
        0.24472846090793609619,
    ]) {
        //assert!((x - y).abs() < 0.00001);
        assert!(x.is_equal(y));
    }
    //Tensor::plot_graph([], "graph").unwrap();
    Ok(())
}

#[test]
fn softmax2() -> Result<(), ZyxError> {
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
        [
            [0.1192029193f32, 0.8807970285, 0.5],
            [0.8807970285, 0.1192029193, 0.5]
        ]
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

/*#[test]
fn var1() -> Result<(), ZyxError> {
    let x = Tensor::randn(1024, DType::F32)?;
    let y = x.var([-1], 1)?;
    Tensor::realize([&y])?;
    Ok(())
}*/

#[test]
fn var() -> Result<(), ZyxError> {
    let x = Tensor::from([[1f32, 2., 3.], [4., 5., 6.]]);
    let y = x.var([0], 0)?;
    assert_eq!(y, [2.25f32, 2.25, 2.25]);
    let y = x.var([1], 0)?;
    assert_eq!(y, [0.666666f32, 0.666666]);
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
fn get() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    assert_eq!(x.get((.., 2..3)).unwrap(), [[1], [4]]);
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
fn complex_movement_reduce() -> Result<(), ZyxError> {
    let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]])
        .expand([2, 3, 2])?
        .exp()
        .ln()
        .reshape([2, 3, 2, 1])?;
    let y =
        Tensor::from([[2f32, 3., 1.], [4., 3., 2.]]).reshape([2, 3, 1, 1])?.expand([2, 3, 2, 1])?;
    let z = (&x + &y).expand([2, 3, 2, 2])?.sum([3, 0])?;
    let z = z.exp().ln().permute([1, 0])?.sum([0])?;
    assert_eq!(z, [52f32, 52., 40.]);
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn fp16() -> Result<(), ZyxError> {
    let x = Tensor::from([0., 1., 2.]).cast(DType::F16);
    let x = x.exp2();
    println!("{x}");
    Ok(())
}

#[test]
fn causal_self_attention() -> Result<(), ZyxError> {
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
        return Err(ZyxError::ShapeError(
            "x must have exactly 3 dims, b, t, c".into(),
        ));
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
    let x = Tensor::randn([1024, 1024], DType::F32).unwrap().expand([1024, 1024, 1024]).unwrap();
    Tensor::realize([&x]).unwrap();
}

#[test]
fn t_15() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    for _ in 0..10 {
        x = &x + &x;
        //println!("{x}");
        //Tensor::plot_graph([], &format!("graph{i}"));
        //Tensor::realize([&x]).unwrap();
    }
    //println!("{x}");
    assert_eq!(x, [[2048, 3072, 1024], [2048, 4096, 1024]]);
}

#[test]
fn layer_norm() -> Result<(), ZyxError> {
    let weight = Some(Tensor::from([4, 5, 1, 2]));
    let d_dims = weight.as_ref().unwrap().rank();
    let bias: Option<Tensor> = None;
    let eps = 0.00001;

    let x = Tensor::from([[3, 5, 2, 1], [6, 1, 4, 2]]).cast(DType::F32);

    let axes = -(d_dims as isize)..=-1;
    let eps = Tensor::constant(eps).cast(x.dtype());
    let a = &x - x.mean_kd(axes.clone())?;
    //println!("{a}");
    let b = (x.var_kd(axes, 1)? + eps).sqrt();
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
    let x = Tensor::from([[3, 4, 2], [5, 4, 1]]);
    let y = x.exp();
    let z = y.tanh();
    Tensor::realize([&y, &z])?;
    //println!("{z:.6}");
    assert_eq!(z, [[1f32, 1., 0.999329], [1., 1., 0.964028]]);
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
            [
                [2, 3, 1],
                [2, 4, 1],
                [2, 3, 1],
                [2, 4, 1],
                [2, 3, 1],
                [2, 4, 1]
            ],
            [
                [2, 3, 1],
                [2, 4, 1],
                [2, 3, 1],
                [2, 4, 1],
                [2, 3, 1],
                [2, 4, 1]
            ]
        ]
    );
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
    println!("{x}");
    assert_eq!(x, [671627020i32, 441824135, 607929878]);
    Ok(())
}

#[test]
fn exp2() {
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
    let x = x.get((.., 8..=-2))?;
    assert_eq!(x, [[41u8, 171, 236], [212, 222, 77], [16, 125, 60]]);
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

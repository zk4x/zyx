use std::collections::HashMap;
use zyx::{DType, Scalar, Tensor, ZyxError};

#[test]
fn matmul_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let z = x.dot(y)?;
    assert_eq!(z, [[31, 15], [22, 10]]);
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
fn pad_reduce() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    x = x.pad_zeros([(0, 1)])?;
    assert_eq!(x, [9, 7, 0]);
    Ok(())
}

#[test]
fn permute_pad() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.pad_zeros([(1, 0)])?.t();
    assert_eq!(x, [[0, 0], [2, 1], [4, 5], [3, 1]]);
    Ok(())
}

#[test]
fn expand_reduce() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.sum([1])?;
    let y = x.expand([2, 2])?;
    x = x.reshape([2, 1])?.expand([2, 2])?;
    Tensor::realize([&x, &y])?;
    assert_eq!(y, [[9, 7], [9, 7]]);
    assert_eq!(x, [[9, 9], [7, 7]]);
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
    println!("{z}");
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
fn softmax() -> Result<(), ZyxError> {
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
fn split() {
    let x = Tensor::from([[2, 3, 1], [2, 1, 4]]);
    let tensors = x.split([2, 1], 1).unwrap();
    //Tensor::realize(&tensors).unwrap();
    assert_eq!(tensors[0], [[2, 3], [2, 1]]);
    assert_eq!(tensors[1], [[1], [4]]);
    //for t in tensors { println!("{t}"); }
}

#[test]
fn complex_movement_reduce() -> Result<(), ZyxError> {
    let x = Tensor::from([[[2f32, 3.]], [[4., 5.]]])
        .expand([2, 3, 2])?
        .exp()
        .ln()
        .reshape([2, 3, 2, 1])?;
    let y = Tensor::from([[2f32, 3., 1.], [4., 3., 2.]])
        .reshape([2, 3, 1, 1])?
        .expand([2, 3, 2, 1])?;
    let z = (&x + &y).expand([2, 3, 2, 2])?.sum([3, 0])?;
    let z = z.exp().ln().permute([1, 0])?.sum([0])?;
    assert_eq!(z, [52f32, 52., 40.]);
    Ok(())
}

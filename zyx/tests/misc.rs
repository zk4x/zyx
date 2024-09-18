use zyx::Tensor;

#[test]
fn matmul() {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let z = x.dot(y);
    assert_eq!(z, [[31, 15], [22, 10]]);
}

#[test]
fn pad_reduce() {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.sum(1);
    x = x.pad_zeros([(0, 1)]);
    assert_eq!(x, [9, 7, 0]);
}

#[test]
fn permute_pad() {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.pad_zeros([(1, 0)]).t();
    assert_eq!(x, [[0, 0], [2, 1], [4, 5], [3, 1]]);
}

#[test]
fn expand_reduce() {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.sum(1);
    let y = x.expand([2, 2]);
    x = x.reshape([2, 1]).expand([2, 2]);
    Tensor::realize([&x, &y]).unwrap();
    assert_eq!(y, [[9, 7], [9, 7]]);
    assert_eq!(x, [[9, 9], [7, 7]]);
}

#[test]
fn pad_reshape_expand() {
    let mut x = Tensor::from([[2, 4, 3, 3, 4], [1, 2, 1, 5, 1]]);
    x = x.pad_zeros([(1, 0), (2, 1)]);
    x = x.reshape([2, 1, 3, 5]);
    x = x.expand([2, 2, 3, 5]);
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
}

#[test]
fn pool() {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3));
    //x = x.repeat([2, 2]);
    //println!("{x}");
    //x = x.reshape([12, 3]);
    //println!("{x}");
    x = x.pool([2, 2], 1, 1);
    assert_eq!(
        x,
        [
            [[[0, 1], [3, 4]], [[1, 2], [4, 5]]],
            [[[3, 4], [6, 7]], [[4, 5], [7, 8]]]
        ]
    );
    //println!("{x}");
}

#[test]
fn cumsum() {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3));
    x = x.cumsum(1);
    assert_eq!(x, [[0, 1, 3], [3, 7, 12], [6, 13, 21]]);
}

#[test]
fn arange() {
    let x = Tensor::arange(0, 10, 2).unwrap();
    //println!("{x}");
    assert_eq!(x, [0, 2, 4, 6, 8]);
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
fn const_() {
    let x = Tensor::from([[3f32, 4., 2.], [4., 3., 2.]]);
    let mut y = Tensor::constant(1) + x; //.get(1);
    println!("{y}'");
    //Tensor::plot_graph([], "graph0");
    //let c: Tensor = Tensor::constant(1f64 / std::f64::consts::E.log2());
    //y = y.log2() * c.cast(y.dtype());
    y = y.ln();
    println!("{y}'");
}

#[test]
fn graph_shapes() {
    let x = Tensor::constant(2);
    let y = x.expand([1, 1]);
    println!("{y}");
}

#[test]
fn uni_matmul() {
    //use zyx::DType;
    //let x = Tensor::rand([5, 5], DType::F32) * 2f32 + 3f32;
    //let y = Tensor::rand([5, 5], DType::F32) * 3f32 + 4f32;
    let x = Tensor::uniform([5, 5], -1f32..2f32).unwrap();
    let y = Tensor::uniform([5, 5], -1f32..5f32).unwrap();
    let z = x.dot(y);
    println!("{z}");
}

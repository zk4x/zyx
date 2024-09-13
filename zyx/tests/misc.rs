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
    println!("{x}");
}

#[test]
fn pool() {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3));
    //x = x.repeat([2, 2]);
    //println!("{x}");
    //x = x.reshape([12, 3]);
    //println!("{x}");
    x = x.pool([2, 2], 1, 1);
    assert_eq!(x, [[[[0, 1], [3, 4]], [[1, 2], [4, 5]]], [[[3, 4], [6, 7]], [[4, 5], [7, 8]]]]);
    println!("{x}");
}

#[test]
fn cumsum() {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3));
    x = x.cumsum(1);
    println!("{x}");
}
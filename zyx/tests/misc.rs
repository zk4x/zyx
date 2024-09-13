use zyx::Tensor;

#[test]
fn matmul() {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let z = x.dot(y);
    assert_eq!(z, [[31, 15], [22, 10]]);
}

#[test]
fn pad_after_reduce() {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.sum(1);
    x = x.pad_zeros([(0, 1)]);
    assert_eq!(x, [9, 7, 0]);
}

#[test]
fn permute_after_pad() {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.pad_zeros([(1, 0)]).t();
    assert_eq!(x, [[0, 0], [2, 1], [4, 5], [3, 1]]);
}

#[test]
fn expand_after_reduce() {
    let mut x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    x = x.sum(1);
    let y = x.expand([2, 2]);
    x = x.reshape([2, 1]).expand([2, 2]);
    Tensor::realize([&x, &y]).unwrap();
    assert_eq!(y, [[9, 7], [9, 7]]);
    assert_eq!(x, [[9, 9], [7, 7]]);
}

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

#[test]
fn pool() {
    let mut x = Tensor::from((0..9).collect::<Vec<i32>>()).reshape((3, 3));
    //x = x.repeat([2, 2]);
    //println!("{x}");
    //x = x.reshape([12, 3]);
    //println!("{x}");
    x = x.pool([2, 2], 1, 1);
    println!("{x}");
}

use zyx::Tensor;

#[test]
fn matmul() {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    println!("{x}");
    println!("{y}");
    let z = x.dot(y);
    println!("{z}");
    panic!();
}
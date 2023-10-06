use zyx::prelude::*;

#[test]
fn dropout() {
    let ctx = Context::new();
    let x = ctx.randn_i32((2, 8));
    let mut z = x.dropout(0.9);
    z.realize().unwrap();
    std::println!("{z}");
}

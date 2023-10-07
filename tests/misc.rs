use zyx::prelude::*;

#[test]
fn backpropagation() -> Result<(), OutOfMemoryError> {
    let ctx = Context::new();
    let mut x = ctx.tensor([[3, 4, 2], [4, 2, 3]]);
    let z = (&x + &x).relu();
    for _ in 0..3 {
        z.backward(&mut x);
        std::println!();
        for n in ctx.debug_nodes() {
            std::println!("{n}");
        }
        use std::io::Write;
        std::fs::File::create("graph.dot")
            .unwrap()
            .write_all(ctx.dot_graph().as_bytes())
            .unwrap();
        x.realize_grad()?;
    }
    Ok(())
}

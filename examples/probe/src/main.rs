// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
use zyx::Tensor;
fn main() -> Result<(), zyx::ZyxError> {
    let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
    let b = a + 1.0;
    println!("{}", b.item::<f32>() * 4.0);
    Ok(())
}

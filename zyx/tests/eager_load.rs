// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{Scalar, Tensor, ZyxError};

#[test]
fn load_survives_drop() -> Result<(), ZyxError> {
    let x = Tensor::from([2.0f32, 3.0]);
    let y = x.sin();
    drop(x);
    let ydata: Vec<f32> = y.try_into()?;
    assert!(ydata[0].is_equal(2f32.sin()));
    assert!(ydata[1].is_equal(3f32.sin()));
    Ok(())
}

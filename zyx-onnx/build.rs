// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["src/onnx.proto3"], &["src/"])?;
    Ok(())
}

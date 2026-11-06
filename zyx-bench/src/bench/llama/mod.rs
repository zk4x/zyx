// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! llama-3.2-3B block kernels, one file per kernel.

pub mod attention;
pub mod embedding;
pub mod kv_assign;
pub mod matmul;
pub mod rmsnorm;
pub mod rope;
pub mod shapes;
pub mod softmax;
pub mod swiglu;

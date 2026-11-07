// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! One module per benchmark; each exposes `bench()` so it can be run alone.

pub mod activation;
pub mod embedding;
pub mod gelu;
pub mod large_matmul;
pub mod layer_norm_like;
pub mod llama;
pub mod ln_softmax;
pub mod matmul;
pub mod mma_nt;
pub mod reduce;
pub mod reduce_axis;
pub mod reduce_multi_axis;
pub mod silu_like;
pub mod softmax;

use zyx::{DType, Dev, Tensor, ZyxError};

/// The device the benchmarks run on: the first non-host device `Dev::all()`
/// reports, so a CUDA box benches CUDA and a CPU-only box falls back to C.
pub(crate) fn bench_dev() -> Dev {
    Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)).unwrap_or(Dev::C)
}

/// Execute `t` and drop the result: eager ops stay lazy until a value is read,
/// and reading is what compiles, autotunes and measures the kernel.
pub(crate) fn run(t: Tensor) -> Result<(), ZyxError> {
    // Drop the producer handle before forcing: a live handle means the runtime
    // must materialize that tensor, which would leave a second store of the
    // (already fused) value in the kernel. Reading does not need it.
    eprintln!("[mark] run: cast");
    let out = t.cast(DType::F32);
    drop(t);
    eprintln!("[mark] run: read");
    let v: Vec<f32> = out.try_into()?;
    eprintln!("[mark] run: done len={}", v.len());
    std::hint::black_box(v.len());
    Ok(())
}

/// Force `t` to execute — eager kernels are lazy until a value is read, and
/// dropping a result does not launch anything.
pub(crate) fn force(t: Tensor) -> Result<(), ZyxError> {
    let _ = t.item::<f32>();
    Ok(())
}

// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::collections::HashMap;
use zyx::{DType, Tensor, ZyxError};

fn main() -> Result<(), ZyxError> {
    let data: HashMap<String, Tensor> = Tensor::load("data/test_data.safetensors")?;
    let py_results: HashMap<String, Tensor> = Tensor::load("data/py_results.safetensors")?;
    let x = data["x"].clone();
    let w1 = data["w1"].clone();
    let w2 = data["w2"].clone();
    let w3 = data["w3"].clone();

    let t = 5;
    let tau: f32 = 100.0;
    let threshold: f32 = 1.0;
    let alpha: f32 = (-1.0f32 / tau).exp();
    let oma = 1.0 - alpha;

    println!("alpha = {alpha}");
    println!("oma = {oma}");
    println!("x shape: {:?}", x.shape());
    println!("w1 shape: {:?}", w1.shape());

    let b1 = Tensor::zeros([256], DType::F32);
    let b2 = Tensor::zeros([128], DType::F32);
    let b3 = Tensor::zeros([10], DType::F32);

    let alpha_t: Tensor = alpha.into();
    let oma_t: Tensor = oma.into();
    let th_t: Tensor = threshold.into();

    let b = x.shape()[0] as usize;
    let mut v1 = Tensor::zeros([b, 256], DType::F32);
    let mut v2 = Tensor::zeros([b, 128], DType::F32);
    let mut sum_out = Tensor::zeros([b, 10], DType::F32);

    Tensor::realize_all()?;

    for t_idx in 0..t {
        let xw1 = x.matmul(&w1)?;
        let xw1_b = &xw1 + &b1;
        let v1_pre = &alpha_t * &v1 + &oma_t * &xw1_b;
        let spike1 = v1_pre.cmpgt(&th_t)?.cast(DType::F32);
        v1 = &v1_pre - &spike1 * &th_t;

        let v2_pre = &alpha_t * &v2 + &oma_t * (&spike1.matmul(&w2)? + &b2);
        let spike2 = v2_pre.cmpgt(&th_t)?.cast(DType::F32);
        v2 = &v2_pre - &spike2 * &th_t;

        sum_out = &sum_out + &spike2.matmul(&w3)? + &b3;

        Tensor::realize_all()?;

        if t_idx == 0 {
            println!("\nRust step 0:");
            println!("  x @ W1 mean: {:.6}", xw1.mean_all().item::<f32>());
            println!("  v1_pre mean: {:.6}", v1_pre.mean_all().item::<f32>());
            println!("  spike1 frac active: {:.6}", spike1.mean_all().item::<f32>());

            let xw1_data: Vec<f32> = Vec::try_from(xw1.clone())?;
            let py_xw1: Vec<f32> = Vec::try_from(py_results["xw1_0"].clone())?;
            let mut max_diff = 0.0f32;
            for i in 0..xw1_data.len() {
                let diff = (xw1_data[i] - py_xw1[i]).abs();
                if diff > max_diff { max_diff = diff; }
            }
            println!("  x @ W1 max diff vs Python: {:.10}", max_diff);

            let v1_pre_data: Vec<f32> = Vec::try_from(v1_pre.clone())?;
            let py_v1_pre: Vec<f32> = Vec::try_from(py_results["v1_pre_0"].clone())?;
            let mut max_diff = 0.0f32;
            for i in 0..v1_pre_data.len() {
                let diff = (v1_pre_data[i] - py_v1_pre[i]).abs();
                if diff > max_diff { max_diff = diff; }
            }
            println!("  v1_pre max diff vs Python: {:.10}", max_diff);
        }

        if t_idx == t - 1 {
            println!("\nRust step {}:", t - 1);
            println!("  v1_pre mean: {:.6}", v1_pre.mean_all().item::<f32>());
            println!("  spike1 frac active: {:.6}", spike1.mean_all().item::<f32>());

            let v1_pre_data: Vec<f32> = Vec::try_from(v1_pre.clone())?;
            let py_v1_pre: Vec<f32> = Vec::try_from(py_results["v1_pre_last"].clone())?;
            let mut max_diff = 0.0f32;
            for i in 0..v1_pre_data.len() {
                let diff = (v1_pre_data[i] - py_v1_pre[i]).abs();
                if diff > max_diff { max_diff = diff; }
            }
            println!("  v1_pre max diff vs Python: {:.10}", max_diff);
        }
    }

    let inv_t: Tensor = (1.0f32 / t as f32).into();
    let output = &sum_out * &inv_t;
    Tensor::realize_all()?;

    println!("\nRust output mean: {:.6}", output.mean_all().item::<f32>());
    let output_data: Vec<f32> = Vec::try_from(output.clone())?;
    println!("Rust output[0]: {:?}", &output_data[0..10]);

    let py_output: Vec<f32> = Vec::try_from(py_results["output"].clone())?;
    let mut max_diff = 0.0f32;
    for i in 0..output_data.len() {
        let diff = (output_data[i] - py_output[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("Output max diff vs Python: {:.10}", max_diff);

    Ok(())
}

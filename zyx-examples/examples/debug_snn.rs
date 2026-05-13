// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::collections::HashMap;
use zyx::{DType, Tensor, ZyxError};

fn main() -> Result<(), ZyxError> {
    let dataset: HashMap<String, Tensor> = Tensor::load("data/mnist_dataset.safetensors")?;
    let train_x = dataset["train_x"].clone().reshape([60000, 784])?;
    let train_y = dataset["train_y"].clone();

    Tensor::manual_seed(0);

    let batch_size: u64 = 128;
    let n_train = 60000;
    let indices = Tensor::randint::<i64>(batch_size, 0, n_train as i64)?;
    let x = train_x.index_select(0, &indices)?;
    let y = train_y.index_select(0, &indices)?;

    let t = 30;
    let tau: f32 = 100.0;
    let threshold: f32 = 1.0;
    let alpha: f32 = (-1.0f32 / tau).exp();
    let oma = 1.0 - alpha;

    println!("alpha = {alpha}");
    println!("oma = {oma}");

    let w1 = Tensor::kaiming_uniform([784, 256], 0.0f32)?;
    let b1 = Tensor::zeros([256], DType::F32);
    let w2 = Tensor::kaiming_uniform([256, 128], 0.0f32)?;
    let b2 = Tensor::zeros([128], DType::F32);
    let w3 = Tensor::kaiming_uniform([128, 10], 0.0f32)?;
    let b3 = Tensor::zeros([10], DType::F32);

    let one: Tensor = 1.0f32.into();
    let neg_one: Tensor = (-1.0f32).into();
    let sigma: Tensor = 0.5f32.into();
    let th_t: Tensor = threshold.into();
    let alpha_t: Tensor = alpha.into();
    let oma_t: Tensor = oma.into();
    let t_f32: Tensor = (t as f32).into();
    let inv_t = &one / &t_f32;

    let h1 = 256;
    let h2 = 128;
    let n_out = 10;
    let b = x.shape()[0] as usize;

    let mut v1 = Tensor::zeros([b, h1], DType::F32);
    let mut v2 = Tensor::zeros([b, h2], DType::F32);
    let mut sum_out = Tensor::zeros([b, n_out], DType::F32);

    Tensor::realize_all()?;

    let mut cache: Vec<(Tensor, Tensor, Tensor, Tensor)> = Vec::with_capacity(t);

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

        cache.push((spike1.clone(), spike2.clone(), v1_pre.clone(), v2_pre.clone()));

        Tensor::realize_all()?;

        if t_idx == 0 {
            println!("\n=== Step {t_idx} ===");
            println!("x @ W1 shape: {:?}", xw1.shape());
            println!("x @ W1 mean: {:.6}", xw1.mean_all().item::<f32>());
            println!("v1_pre mean: {:.6}", v1_pre.mean_all().item::<f32>());
            println!("spike1 frac active: {:.6}", spike1.mean_all().item::<f32>());
            println!("spike2 frac active: {:.6}", spike2.mean_all().item::<f32>());
            let xw1_data: Vec<f32> = Vec::try_from(xw1.clone())?;
            let v1_pre_data: Vec<f32> = Vec::try_from(v1_pre.clone())?;
            let spike1_data: Vec<f32> = Vec::try_from(spike1.clone())?;
            println!("x @ W1[0, :5] = {:?}", &xw1_data[0..5]);
            println!("v1_pre[0, :5] = {:?}", &v1_pre_data[0..5]);
            println!("spike1[0, :5] = {:?}", &spike1_data[0..5]);
        }

        if t_idx == 2 {
            let v1_pre_data: Vec<f32> = Vec::try_from(v1_pre.clone())?;
            let spike1_data: Vec<f32> = Vec::try_from(spike1.clone())?;
            println!("\n=== Step {t_idx} ===");
            println!("v1_pre mean: {:.6}", v1_pre.mean_all().item::<f32>());
            println!("spike1 frac active: {:.6}", spike1.mean_all().item::<f32>());
            println!("v1_pre[0, :5] = {:?}", &v1_pre_data[0..5]);
            println!("spike1[0, :5] = {:?}", &spike1_data[0..5]);
        }
    }

    let output = &sum_out * &inv_t;
    Tensor::realize_all()?;

    println!("\nFinal output mean: {:.6}", output.mean_all().item::<f32>());
    let output_data: Vec<f32> = Vec::try_from(output.clone())?;
    println!("Final output[0] = {:?}", &output_data[0..10]);

    let loss = output.cross_entropy(y.one_hot(10), [-1])?.mean_all();
    println!("Loss: {:.6}", loss.item::<f32>());

    // Backward
    let bs: Tensor = (1.0f32 / b as f32).into();
    let d_output = (&output.softmax([-1])? - &y.one_hot(10).cast(DType::F32)) * &bs;
    let d_sum_out = &d_output * &inv_t;
    println!("\nd_output mean: {:.6}", d_output.mean_all().item::<f32>());
    println!("d_sum_out mean: {:.6}", d_sum_out.mean_all().item::<f32>());

    let mut dv1 = Tensor::zeros([b, h1], DType::F32);
    let mut dv2 = Tensor::zeros([b, h2], DType::F32);
    let mut dw1_acc = Tensor::zeros([784, 256], DType::F32);

    for t_idx in (0..t).rev() {
        let (spike1, spike2, v1_pre, v2_pre) = &cache[t_idx];

        let d_spike2 = d_sum_out.matmul(&w3.t())?;
        let diff2 = v2_pre - &th_t;
        let surr2 = &sigma * (&neg_one * &sigma * &diff2.abs()).exp();
        let dv2_pre = &d_spike2 * &surr2 + &dv2 * (&one - &th_t * &surr2);
        let d_pre2 = &oma_t * &dv2_pre;
        let d_spike1 = d_pre2.matmul(&w2.t())?;
        dv2 = &dv2_pre * &alpha_t;

        let diff1 = v1_pre - &th_t;
        let surr1 = &sigma * (&neg_one * &sigma * &diff1.abs()).exp();
        let dv1_pre = &d_spike1 * &surr1 + &dv1 * (&one - &th_t * &surr1);
        let d_pre1 = &oma_t * &dv1_pre;
        dw1_acc = &dw1_acc + &x.t().matmul(&d_pre1)?;
        dv1 = &dv1_pre * &alpha_t;

        if t_idx == 29 {
            println!("\n=== Backward step {t_idx} ===");
            println!("d_spike2 mean: {:.6}", d_spike2.mean_all().item::<f32>());
            println!("d_spike1 mean: {:.6}", d_spike1.mean_all().item::<f32>());
            println!("dv1_pre mean: {:.6}", dv1_pre.mean_all().item::<f32>());
            println!("d_pre1 mean: {:.6}", d_pre1.mean_all().item::<f32>());
            println!("dw1_acc mean: {:.6}", dw1_acc.mean_all().item::<f32>());
        }
    }

    Tensor::realize_all()?;
    let dw1_data: Vec<f32> = Vec::try_from(dw1_acc.clone())?;
    println!("\ndw1_acc[0, :5] = {:?}", &dw1_data[0..5]);

    Ok(())
}

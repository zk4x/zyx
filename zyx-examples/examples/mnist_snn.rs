// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::collections::HashMap;
use std::time::Instant;

use zyx::{DType, ReduceOp, Tensor, ZyxError};
use zyx_optim::Adam;

struct Snn {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,

    t: usize,

    _one: Tensor,
    _neg_one: Tensor,
    _sigma: Tensor,
    _th_t: Tensor,
    _alpha_t: Tensor,
    _oma_t: Tensor,
    _inv_t: Tensor,
}

impl Snn {
    fn new(t: usize, tau: f32, threshold: f32) -> Result<Self, ZyxError> {
        let alpha: f32 = (-1.0f32 / tau).exp();
        let oma = 1.0 - alpha;

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
        let t_t: Tensor = (t as f32).into();
        let inv_t = &one / &t_t;

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            t,
            _one: one,
            _neg_one: neg_one,
            _sigma: sigma,
            _th_t: th_t,
            _alpha_t: alpha_t,
            _oma_t: oma_t,
            _inv_t: inv_t,
        })
    }

    fn forward_store(&self, x: &Tensor) -> Result<(Tensor, SpikeCache), ZyxError> {
        let b = x.shape()[0] as usize;
        let h1 = 256;
        let h2 = 128;
        let n_out = 10;

        let alpha_t = &self._alpha_t;
        let oma_t = &self._oma_t;
        let th_t = &self._th_t;

        let mut v1 = Tensor::zeros([b, h1], DType::F32);
        let mut v2 = Tensor::zeros([b, h2], DType::F32);
        let mut sum_out = Tensor::zeros([b, n_out], DType::F32);

        let mut cache: SpikeCache = Vec::with_capacity(self.t);

        for _ in 0..self.t {
            let v1_pre = alpha_t * &v1 + oma_t * (x.matmul(&self.w1)? + &self.b1);
            let spike1 = v1_pre.cmpgt(th_t)?.cast(DType::F32);
            v1 = &v1_pre - &spike1 * th_t;

            let v2_pre = alpha_t * &v2 + oma_t * (spike1.matmul(&self.w2)? + &self.b2);
            let spike2 = v2_pre.cmpgt(th_t)?.cast(DType::F32);
            v2 = &v2_pre - &spike2 * th_t;

            sum_out = &sum_out + spike2.matmul(&self.w3)? + &self.b3;

            cache.push((spike1, spike2, v1_pre, v2_pre));
        }

        let output = &sum_out * &self._inv_t;
        Ok((output, cache))
    }

    fn backward(
        &self,
        x: &Tensor,
        target: &Tensor,
        output: &Tensor,
        stored: &SpikeCache,
    ) -> Result<Vec<Tensor>, ZyxError> {
        let b = x.shape()[0] as f32;
        let t = self.t;

        let bs: Tensor = (1.0f32 / b).into();
        let d_output = (output.softmax([-1])? - &target.one_hot(10).cast(DType::F32)) * &bs;
        let d_sum_out = &d_output * &self._inv_t;

        let mut dw1_acc = Tensor::zeros([784, 256], DType::F32);
        let mut db1_acc = Tensor::zeros([256], DType::F32);
        let mut dw2_acc = Tensor::zeros([256, 128], DType::F32);
        let mut db2_acc = Tensor::zeros([128], DType::F32);
        let mut dw3_acc = Tensor::zeros([128, 10], DType::F32);
        let mut db3_acc = Tensor::zeros([10], DType::F32);

        let mut dv1 = Tensor::zeros([b as usize, 256], DType::F32);
        let mut dv2 = Tensor::zeros([b as usize, 128], DType::F32);

        for t_idx in (0..t).rev() {
            let (spike1, spike2, v1_pre, v2_pre) = &stored[t_idx];

            let dw3_t = spike2.t().matmul(&d_sum_out)?;
            let db3_t = d_sum_out.sum([0])?;
            let d_spike2 = d_sum_out.matmul(&self.w3.t())?;

            dw3_acc = &dw3_acc + dw3_t;
            db3_acc = &db3_acc + db3_t;

            let diff2 = v2_pre - &self._th_t;
            let surr2 = &self._sigma * (&self._neg_one * &self._sigma * &diff2.abs()).exp();
            let dv2_pre = &d_spike2 * &surr2 + &dv2 * (&self._one - &self._th_t * &surr2);

            let d_pre2 = &self._oma_t * &dv2_pre;
            let dw2_t = spike1.t().matmul(&d_pre2)?;
            let db2_t = d_pre2.sum([0])?;
            dw2_acc = &dw2_acc + dw2_t;
            db2_acc = &db2_acc + db2_t;
            let d_spike1 = d_pre2.matmul(&self.w2.t())?;

            dv2 = &dv2_pre * &self._alpha_t;

            let diff1 = v1_pre - &self._th_t;
            let surr1 = &self._sigma * (&self._neg_one * &self._sigma * &diff1.abs()).exp();
            let dv1_pre = &d_spike1 * &surr1 + &dv1 * (&self._one - &self._th_t * &surr1);

            let d_pre1 = &self._oma_t * &dv1_pre;
            let dw1_t = x.t().matmul(&d_pre1)?;
            let db1_t = d_pre1.sum([0])?;
            dw1_acc = &dw1_acc + dw1_t;
            db1_acc = &db1_acc + db1_t;

            dv1 = &dv1_pre * &self._alpha_t;
        }

        Ok(vec![dw1_acc, db1_acc, dw2_acc, db2_acc, dw3_acc, db3_acc])
    }

    fn params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.w3,
            &mut self.b3,
        ]
    }
}

type SpikeCache = Vec<(Tensor, Tensor, Tensor, Tensor)>;

fn evaluate(model: &Snn, test_x: &Tensor, test_y: &Tensor) -> Result<f32, ZyxError> {
    let (logits, _) = model.forward_store(test_x)?;
    let pred = logits.argmax_axis(-1)?;
    let correct_f32 = pred.equal(test_y)?.cast(DType::F32);
    let correct = correct_f32.sum_all().item::<f32>();
    let total = test_y.shape()[0];
    let accuracy = 100.0 * correct / total as f32;

    println!("\nOverall Test Accuracy: {accuracy:.2}% ({correct:.0}/{total})");

    println!("\nPer-class Accuracy:");
    for i in 0..10 {
        let i_t: Tensor = (i as i64).into();
        let is_class_i = test_y.equal(&i_t)?.cast(DType::F32);
        let class_total = is_class_i.sum_all().item::<f32>();
        if class_total > 0.0 {
            let is_correct = pred.equal(test_y)?.cast(DType::F32);
            let class_correct = (&is_correct * &is_class_i).sum_all().item::<f32>();
            println!(
                "  Digit {i}: {:.2}% ({class_correct:.0}/{class_total:.0})",
                100.0 * class_correct / class_total
            );
        }
    }

    Ok(accuracy)
}

fn train() -> Result<(), ZyxError> {
    println!("Loading MNIST...");
    let dataset: HashMap<String, Tensor> = Tensor::load("data/mnist_dataset.safetensors")?;

    let train_x = dataset["train_x"].clone().reshape([60000, 784])?;
    let train_y = dataset["train_y"].clone();
    let test_x = dataset["test_x"].clone().reshape([10000, 784])?;
    let test_y = dataset["test_y"].clone();

    Tensor::manual_seed(0);
    let mut model = Snn::new(30, 100.0, 1.0)?;
    let mut optimizer = Adam {
        learning_rate: 3e-3,
        ..Default::default()
    };

    let n_train = 60000;
    let batch_size: u64 = 128;
    let total_steps: usize = 110;
    let warmup_steps: usize = 10;

    println!("\nTraining SNN for {total_steps} steps (T=30 time steps)");
    println!("Architecture: 784 -> 256 -> 128 -> 10\n");

    Tensor::realize_all()?;

    for step in 0..total_steps {
        let indices = Tensor::randint::<i64>(batch_size, 0, n_train as i64)?;

        let x = train_x.index_select(0, &indices)?;
        let y = train_y.index_select(0, &indices)?;

        let t0 = Instant::now();

        let (output, stored) = model.forward_store(&x)?;
        Tensor::realize_all()?;
        let loss = output.cross_entropy(y, ReduceOp::Mean)?;
        let pred = output.argmax_axis(-1)?;
        let correct_t = pred.equal(&y)?.cast(DType::F32).sum_all();
        let loss_val = loss.item::<f32>();
        let correct_val = correct_t.item::<f32>();

        let grads = model.backward(&x, &y, &output, &stored)?;
        optimizer.update(model.params_mut(), grads.into_iter().map(Some));

        Tensor::realize_all()?;

        let t1 = Instant::now();
        let elapsed_ms = (t1 - t0).as_secs_f64() * 1000.0;

        if step >= warmup_steps {
            if (step - warmup_steps) % 10 == 0 || step == total_steps - 1 {
                println!(
                    "Step {:4} | Loss: {:.4} | Acc: {:.1}% | Time: {:.2} ms",
                    step + 1,
                    loss_val,
                    100.0 * correct_val / batch_size as f32,
                    elapsed_ms
                );
            }
        }
    }

    println!("\nEvaluating...");
    let accuracy = evaluate(&model, &test_x, &test_y)?;
    println!("\nFinal accuracy: {accuracy:.2}%");

    Ok(())
}

fn main() -> Result<(), ZyxError> {
    train()
}

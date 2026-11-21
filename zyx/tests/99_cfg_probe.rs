// TEMP PROBE - remove after the qwen ffn_gate config is chosen

use std::time::Instant;

use zyx::kernel::{Dev, Kernel, MemScope};
use zyx::{DType, Tensor, ZyxError};

const M: i64 = 128;
const N: i64 = 17408;
const K: i64 = 5120;
const BLOCK: i64 = 128;
const ROW: i64 = 34;
const KGROUP: i64 = K / 32;
const QG: i64 = 32;

fn naive(dev: Dev) -> Kernel {
    let mut k = Kernel::new(dev);
    let a8 = k.param(DType::U32);
    let sa = k.param(DType::F32);
    let w = k.param(DType::U16);
    let c = k.param_mut(DType::F32);
    let m = k.group_range(0, M);
    let n = k.group_range(1, N);
    let acc = k.storage(DType::F32, MemScope::Register, 1);
    let zero = k.const_val(0.0f32);
    k.store(acc, zero, 0i64);
    k.loop_over(K, |k, k_loop| {
        let elem = k.mad(n, K, k_loop);
        let blk = k.div(elem, BLOCK);
        let off = k.mod_(elem, BLOCK);
        let c17 = k.const_val(17i64);
        let wrow = k.mul(blk, c17);
        let two = k.const_val(2i64);
        let eight_i = k.const_val(8i64);
        let off_word = k.div(off, eight_i);
        let word_in_row = k.add(1i64, off_word);
        let word_idx = k.add(wrow, word_in_row);
        let word = k.load(w, word_idx);
        let off_mod = k.mod_(off, eight_i);
        let shift = k.mul(off_mod, two);
        let shift = k.cast(shift, DType::U16);
        let shifted = k.bit_shift_right(word, shift);
        let mask = k.const_val(3u16);
        let code = k.bit_and(shifted, mask);
        let bits = k.load(w, wrow);
        let scale_f16 = k.bitcast(bits, DType::F16);
        let scale = k.cast(scale_f16, DType::F32);
        let code_f32 = k.cast(code, DType::F32);
        let centered = k.sub(code_f32, 1.0f32);
        let val = k.mul(centered, scale);
        let val = k.cast(val, DType::F16);
        let act_idx = k.mad(m, K, k_loop);
        let four = k.const_val(4i64);
        let word_i = k.div(act_idx, four);
        let aword = k.load(a8, word_i);
        let byte_off = k.mod_(act_idx, four);
        let bo32 = k.cast(byte_off, DType::U32);
        let eight_u = k.const_val(8u32);
        let sh = k.mul(bo32, eight_u);
        let bword = k.bit_shift_right(aword, sh);
        let bmask = k.const_val(0xffu32);
        let byte_bits = k.bit_and(bword, bmask);
        let byte_u = k.cast(byte_bits, DType::U8);
        let byte = k.bitcast(byte_u, DType::I8);
        let byte_f = k.cast(byte, DType::F32);
        let group = k.div(k_loop, QG);
        let kgroup = k.const_val(KGROUP);
        let sa_idx = k.mad(m, kgroup, group);
        let sa_v = k.load(sa, sa_idx);
        let act = k.mul(byte_f, sa_v);
        let act = k.cast(act, DType::F16);
        let prod = k.mul(val, act);
        let prod = k.cast(prod, DType::F32);
        let prev = k.load(acc, 0i64);
        let sum = k.add(prod, prev);
        k.store(acc, sum, 0i64);
    });
    let res = k.load(acc, 0i64);
    let c_idx = k.mad(m, N, n);
    k.store(c, res, c_idx);
    k.verify();
    k
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[test]
fn cfg_probe() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    let base = naive(dev);
    let recipe = base.opt_tensorize();
    println!("nconfigs = {}", recipe.nconfigs());
    let from = env("CFG_FROM", 0);
    let to = env("CFG_TO", 0).min(recipe.nconfigs());
    if to == 0 {
        return Ok(());
    }
    let a32 = Tensor::zeros([M * K / 4], DType::U32).to(dev)?;
    let sa = Tensor::zeros([M * KGROUP], DType::F32).to(dev)?;
    let w = Tensor::zeros([N * K / BLOCK * 17], DType::U16).to(dev)?;
    let out = Tensor::zeros([M, N], DType::F32).to(dev)?;
    for cfg in from..to {
        let mut k = base.clone();
        let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recipe.apply(&mut k, cfg);
            k.default_epilogue();
        }));
        if applied.is_err() {
            println!("cfg {cfg}: apply panicked");
            continue;
        }
        let verified = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| k.verify()));
        if verified.is_err() {
            println!("cfg {cfg}: verify failed");
            continue;
        }
        let compiled = match k.compile() {
            Ok(c) => c,
            Err(e) => {
                println!("cfg {cfg}: compile failed: {e}");
                continue;
            }
        };
        let mut ok = true;
        for _ in 0..3 {
            if compiled.forward(&[&a32, &sa, &w], vec![[M, N]]).is_err() {
                ok = false;
                break;
            }
        }
        if !ok {
            println!("cfg {cfg}: launch failed (stopping: the context is poisoned)");
            return Ok(());
        }
        let t0 = Instant::now();
        let iters = 5;
        for _ in 0..iters {
            if compiled.forward(&[&a32, &sa, &w], vec![[M, N]]).is_err() {
                println!("cfg {cfg}: launch failed (stopping: the context is poisoned)");
                return Ok(());
            }
        }
        let _ = out.item::<f32>();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let flop = 2.0 * (M * N * K) as f64;
        println!("cfg {cfg}: {ms:.4} ms, {:.2} TFLOP/s", flop / (ms * 1e-3) * 1e-12);
    }
    Ok(())
}

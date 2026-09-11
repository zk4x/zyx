// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Repack one gguf Q4_K tensor into the device-ready dequant layout.
//! Calls `repack_q4k` directly: rows/cols pass straight through from CLI,
//! nothing baked in. Placement is automatic (fastest available backend).
//!
//! Usage:
//!   repack_gguf <gguf> <tensor> <rows> <cols> <out.safetensors>
//! Example:
//!   repack_gguf examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf blk.1.attn_qkv.weight 10240 5120 \
//!     examples/data/qwen3_attn_qkv_q4k_repacked.safetensors
//!
//! rows/cols are the storage-order matrix dims (gguf dim0 contiguous):
//! rows*cols must equal 256 * num_blocks, both multiples of 32.
//! Output file holds three tensors: packed (U16), scales (BF16), mins (BF16).

use qwen3_8_27b::repack_q4k;
use std::collections::HashMap;
use zyx::{Module, Tensor, ZyxError};

fn main() -> Result<(), ZyxError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: repack_gguf <gguf> <tensor> <rows> <cols> <out.safetensors>");
        std::process::exit(2);
    }
    let (gguf, name, out) = (args[1].clone(), args[2].clone(), args[5].clone());
    let rows: i64 = args[3].parse().expect("rows must be an integer");
    let cols: i64 = args[4].parse().expect("cols must be an integer");

    let t0 = std::time::Instant::now();
    let (_meta, tensors) = Tensor::load_gguf(&gguf)?;
    let raw = &tensors[&name];
    let (packed, scales, mins) = repack_q4k(raw, rows, cols)?;
    let mut out_map = HashMap::new();
    out_map.insert("packed".to_string(), packed);
    out_map.insert("scales".to_string(), scales);
    out_map.insert("mins".to_string(), mins);
    out_map.save(&out)?;
    eprintln!("repacked {name} [{rows}x{cols}] -> {out} in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}

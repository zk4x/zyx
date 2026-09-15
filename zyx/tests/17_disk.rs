// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tests for the disk loaders: gguf, safetensors and numpy.
//! Fixtures are generated into the system temp dir.

use std::{collections::HashMap, fs, io::Write, path::PathBuf};
use zyx::{Module, Tensor, ZyxError};

fn temp_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("zyx_17_disk_{}_{}.tmp", name, std::process::id()));
    path
}

/// Minimal GGUF v3 writer: one Uint32 metadata kv (`general.alignment` = 64,
/// exercising the data-section alignment logic) and one F32 tensor.
fn write_gguf(path: &PathBuf, dims: &[i64], data: &[f32]) {
    let mut f = fs::File::create(path).unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    buf.extend_from_slice(&1u64.to_le_bytes()); // metadata kv count
    let key = b"general.alignment";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // value type: Uint32
    buf.extend_from_slice(&64u32.to_le_bytes());
    let name = b"test.tensor";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&0u32.to_le_bytes()); // dtype: F32
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor data offset
    // Pad the header to the declared 64-byte alignment.
    while buf.len() % 64 != 0 {
        buf.push(0);
    }
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf).unwrap();
}

#[test]
fn gguf_load() -> Result<(), ZyxError> {
    let path = temp_path("gguf");
    write_gguf(&path, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let (_metadata, tensors) = Tensor::load_gguf(&path)?;
    let x = &tensors["test.tensor"];
    assert_eq!(x.shape(), [2, 3]);
    assert_eq!(x.to_vec::<f32>()?, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn safetensors_roundtrip() -> Result<(), ZyxError> {
    let path = temp_path("safetensors");
    let a = Tensor::from([1.0f32, 2.0, 3.0, -4.5]).reshape([2, 2])?;
    let b = Tensor::from([10i32, -20, 30]);
    let module: HashMap<String, Tensor> = [("a", a), ("b", b)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    module.save(&path)?;

    let loaded = Tensor::load_safetensors(&path)?;
    assert_eq!(loaded.len(), 2);
    let a = &loaded["a"];
    assert_eq!(a.shape(), [2, 2]);
    assert_eq!(a.to_vec::<f32>()?, [1.0, 2.0, 3.0, -4.5]);
    let b = &loaded["b"];
    assert_eq!(b.to_vec::<i32>()?, [10, -20, 30]);

    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn numpy_roundtrip() -> Result<(), ZyxError> {
    let path = temp_path("npy");
    let module = Tensor::from([1.5f32, -2.5, 3.25, 4.0]).reshape([2, 2])?;
    module.save_numpy(&path)?;

    let x = Tensor::load_numpy(&path)?;
    assert_eq!(x.shape(), [2, 2]);
    assert_eq!(x.to_vec::<f32>()?, [1.5, -2.5, 3.25, 4.0]);

    fs::remove_file(path)?;
    Ok(())
}

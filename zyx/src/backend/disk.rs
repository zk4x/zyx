// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

#[cfg(windows)]
use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::{
    fs::File,
    path::{Path, PathBuf},
};

use super::PoolBufferId;
use crate::{
    error::{BackendError, ErrorStatus},
    shape::Dim,
    slab::Slab,
};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug)]
pub struct DiskMemoryPool {
    free_bytes: Dim,
    buffers: Slab<PoolBufferId, DiskBuffer>,
}

/// Constructs the global disk pool. Infallible — `Disk` is always available.
pub(super) fn ensure_pool() -> DiskMemoryPool {
    if super::debug_backends() {
        println!("[disk] initialized");
    }
    DiskMemoryPool { free_bytes: 0, buffers: Slab::new() }
}

/// Process-wide global disk pool. Owned here — `mod.rs` only holds the `Pool::Disk` handle.
static DISK_POOL: OnceLock<Arc<Mutex<DiskMemoryPool>>> = OnceLock::new();

pub(super) fn pool() -> Arc<Mutex<DiskMemoryPool>> {
    DISK_POOL.get_or_init(|| Arc::new(Mutex::new(ensure_pool()))).clone()
}

#[derive(Debug)]
struct DiskBuffer {
    bytes: Dim,
    path: PathBuf,
    offset_bytes: u64,
}

impl DiskMemoryPool {
    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn buffer_from_path(&mut self, bytes: Dim, path: &Path, offset_bytes: u64) -> PoolBufferId {
        self.buffers.push(DiskBuffer { bytes, path: path.into(), offset_bytes })
    }

    /// Return the size in bytes of a disk-backed buffer.
    pub fn buffer_bytes(&self, src: PoolBufferId) -> Dim {
        self.buffers[src].bytes
    }

    /// Read a slice of the file backing the buffer into host memory.
    /// Synchronous — the disk pool never defers work.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8]) -> Result<(), BackendError> {
        let buffer = &self.buffers[src];
        let f = File::open(&buffer.path).unwrap();
        #[cfg(unix)]
        let result = f.read_exact_at(dst, buffer.offset_bytes);
        #[cfg(windows)]
        let result = (|| -> io::Result<()> {
            let mut off = buffer.offset_bytes;
            let mut remaining = dst;
            while !remaining.is_empty() {
                let n = f.seek_read(remaining, off)?;
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "failed to fill whole buffer"));
                }
                remaining = &mut remaining[n..];
                off += n as i64;
            }
            Ok(())
        })();
        result.map_err(|err| BackendError { status: ErrorStatus::MemoryCopyP2H, context: format!("{err}").into() })
    }
}

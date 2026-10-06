// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

#[cfg(windows)]
use std::io;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::{
    fs::File,
    hash::BuildHasherDefault,
    path::{Path, PathBuf},
};

use super::ChunkId;
#[cfg(windows)]
use crate::error::ErrorStatus;
use crate::{Map, Set, error::BackendError, shape::Dim, slab::Slab};
use std::sync::Mutex;

// ── Raw mmap bindings (unix) ──────────────────────────────────────────────────
// Hand-declared: two symbols, stable ABI forever, no dependency.
#[cfg(unix)]
mod mmap {
    pub const PROT_READ: i32 = 1;
    pub const MAP_SHARED: i32 = 1;
    pub const MAP_FAILED: *mut core::ffi::c_void = !0 as *mut core::ffi::c_void;

    unsafe extern "C" {
        pub fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut core::ffi::c_void;
        pub fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
    }
}

// ── Global state ──────────────────────────────────────────────────────────────

static DISK_POOL: Mutex<DiskMemoryPool> = Mutex::new(DiskMemoryPool {
    free_bytes: 0,
    buffers: Slab::new(),
    free_set: Set::with_hasher(BuildHasherDefault::new()),
    #[cfg(unix)]
    mappings: Map::with_hasher(BuildHasherDefault::new()),
});

#[derive(Debug)]
pub struct DiskMemoryPool {
    free_bytes: Dim,
    buffers: Slab<ChunkId, DiskBuffer>,
    free_set: Set<ChunkId>,
    /// Whole-file mappings by path (unix): one mapping serves every chunk
    /// of the file — per-chunk mapping is impossible, data offsets are not
    /// page-aligned. Refcounted by open buffers, unmapped at zero.
    #[cfg(unix)]
    mappings: Map<PathBuf, DiskMapping>,
}

/// A whole-file read-only mapping. Raw pointer only — every access runs
/// under the pool lock; unmap happens only on last release, and every
/// mapping consumer today is blocking, so nothing references it past that.
#[cfg(unix)]
#[derive(Debug)]
struct DiskMapping {
    base: *mut u8,
    len: usize,
    refs: usize,
}

// SAFETY: every access runs under the pool `Mutex`; the pointer never
// escapes the lock except as a borrowed read/DMA source whose lifetime is
// bounded by the mapping entry itself.
#[cfg(unix)]
unsafe impl Send for DiskMapping {}

pub(super) fn pool() -> &'static Mutex<DiskMemoryPool> {
    &DISK_POOL
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

    /// Return the size in bytes of a disk-backed buffer.
    pub fn buffer_bytes(&self, src: ChunkId) -> Dim {
        self.buffers[src].bytes
    }

    pub fn buffer_from_path(&mut self, bytes: Dim, path: &Path, offset_bytes: u64) -> ChunkId {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let entry = self.mappings.entry(path.into()).or_insert_with(|| {
                let file = File::open(path).expect("disk: cannot open mapped file");
                let len = file.metadata().expect("disk: cannot stat mapped file").len() as usize;
                // Empty files have nothing to map: a null base with len 0.
                // Reads of 0 bytes still work; anything else trips the
                // extent asserts below.
                let base = if len == 0 {
                    core::ptr::null_mut()
                } else {
                    let base =
                        unsafe { mmap::mmap(core::ptr::null_mut(), len, mmap::PROT_READ, mmap::MAP_SHARED, file.as_raw_fd(), 0) };
                    assert!(base != mmap::MAP_FAILED, "disk: mmap failed");
                    base.cast()
                };
                // The mapping holds its own pages: the fd can go.
                drop(file);
                DiskMapping { base, len, refs: 0 }
            });
            entry.refs += 1;
        }
        self.buffers.push(DiskBuffer { bytes, path: path.into(), offset_bytes })
    }

    /// Put a buffer id into the free list. Frees nothing yet: the slab
    /// entry stays so ChunkIds never alias a different buffer. On unix the
    /// file mapping refcount drops; the last release unmaps. Safe: every
    /// mapping consumer today is blocking, so nothing references the
    /// mapping past the last release — async DMA later must gate unmap on
    /// completion instead.
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.buffers.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown disk buffer {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of disk buffer {buffer_id:?}");
        self.free_set.insert(buffer_id);
        #[cfg(unix)]
        {
            let path = self.buffers[buffer_id].path.clone();
            let unmap = match self.mappings.get_mut(&path) {
                Some(mapping) => {
                    mapping.refs -= 1;
                    mapping.refs == 0
                }
                None => {
                    debug_assert!(false, "disk release of unmapped path {path:?}");
                    false
                }
            };
            if unmap {
                let mapping = self.mappings.remove(&path).expect("disk: just-checked mapping is missing");
                if mapping.len > 0 {
                    assert!(unsafe { mmap::munmap(mapping.base.cast(), mapping.len) } == 0, "disk: munmap failed");
                }
            }
        }
    }

    /// Read the buffer's extent into host memory: a memcpy from the file
    /// mapping, straight into the caller's slice — no intermediate buffer.
    /// Synchronous — the disk pool never defers work.
    /// Windows has no mapping: falls back to positional reads.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let buffer = &self.buffers[src];
        debug_assert!(
            dst.len() as Dim <= buffer.bytes,
            "disk read of {} bytes exceeds buffer extent of {} bytes",
            dst.len(),
            buffer.bytes
        );
        #[cfg(unix)]
        {
            let mapping = self.mappings.get(&buffer.path).expect("disk: read from unmapped file");
            debug_assert!(
                buffer.offset_bytes as usize + dst.len() <= mapping.len,
                "disk read runs past the end of the mapped file"
            );
            unsafe {
                core::ptr::copy_nonoverlapping(mapping.base.add(buffer.offset_bytes as usize), dst.as_mut_ptr(), dst.len());
            }
            return Ok(());
        }
        #[cfg(windows)]
        {
            let f = File::open(&buffer.path).unwrap();
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

    /// Raw pointer to the mapped chunk extent, for direct device DMA with
    /// no staging buffer and no read call. Valid while the chunk is
    /// unreleased; unmap happens only on last release (see [`DiskMemoryPool::release`]).
    #[cfg(unix)]
    pub fn mapped_ptr(&self, src: ChunkId) -> (*const u8, Dim) {
        let buffer = &self.buffers[src];
        let mapping = self.mappings.get(&buffer.path).expect("disk: DMA from unmapped file");
        debug_assert!(
            buffer.offset_bytes as usize + buffer.bytes as usize <= mapping.len,
            "disk chunk runs past the end of the mapped file"
        );
        unsafe { (mapping.base.add(buffer.offset_bytes as usize).cast_const(), buffer.bytes) }
    }
}
